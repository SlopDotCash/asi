"""Multi-seed experiment runner for publication-quality analysis.

Provides infrastructure for running experiments across multiple seeds
with optional parallelization and aggregation of results.

Two conventions matter for downstream analysis:

* :func:`run_multi_seed_experiment` runs every config on the *same* seed
  list, so per-seed values are aligned across configs.  Keep it that way:
  the significance tests in :mod:`alberta_framework.utils.statistics`
  default to paired tests, which require seed-matched samples.
* "Final value" summaries are the mean over the last 100 steps (or the
  whole trace when shorter), not the last step — see
  :func:`aggregate_metrics`.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any, NamedTuple, NoReturn, cast

import jax.random as jr
import numpy as np
from numpy.typing import NDArray

from alberta_framework.core.learners import (
    LinearLearner,
    metrics_to_dicts,
    run_learning_loop,
)
from alberta_framework.core.types import LearnerState
from alberta_framework.streams.base import ScanStream
from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.utils.statistics import common_final_window

_NUMPY_COORDINATE_TYPES = frozenset(
    np.dtype(dtype_code).type
    for dtype_code in (
        "?",
        "b",
        "B",
        "h",
        "H",
        "i",
        "I",
        "l",
        "L",
        "q",
        "Q",
        "e",
        "f",
        "d",
        "g",
        "F",
        "D",
        "G",
        "S",
        "U",
    )
)


def _type_identity_in(
    value_type: type[object],
    candidates: Iterable[type[object]],
) -> bool:
    """Match an untrusted runtime type without metaclass equality or hashing."""
    return any(value_type is candidate for candidate in candidates)


class _CanonicalFractionCoordinate(tuple[int, int]):
    """An intrinsically immutable rational key with Fraction hash semantics."""

    __slots__ = ()

    def __new__(
        cls,
        numerator: int,
        denominator: int,
    ) -> "_CanonicalFractionCoordinate":
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise ValueError("canonical Fraction coordinates require builtin integer components")
        normalized = Fraction(numerator, denominator)
        return tuple.__new__(cls, (normalized.numerator, normalized.denominator))

    @property
    def numerator(self) -> int:
        return self[0]

    @property
    def denominator(self) -> int:
        return self[1]

    def _as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def __hash__(self) -> int:
        return hash(self._as_fraction())

    def __eq__(self, other: object) -> bool:
        other_type = type(other)
        if other_type is _CanonicalFractionCoordinate:
            coordinate = cast(_CanonicalFractionCoordinate, other)
            return (
                self.numerator == coordinate.numerator
                and self.denominator == coordinate.denominator
            )
        if _type_identity_in(other_type, (bool, int, float, complex, Decimal, Fraction)):
            return bool(self._as_fraction() == other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return repr(self._as_fraction())

    def __str__(self) -> str:
        return str(self._as_fraction())

    def __getnewargs__(self) -> tuple[int, int]:
        return (self.numerator, self.denominator)


_PYTHON_NUMERIC_COORDINATE_TYPES = (
    bool,
    int,
    float,
    complex,
    Decimal,
    Fraction,
    _CanonicalFractionCoordinate,
)
_STRING_COORDINATE_TYPES = (str, np.str_)
_BYTES_COORDINATE_TYPES = (bytes, np.bytes_)
_MAX_HYPERPARAMETER_COORDINATE_NESTING = 32


class ExperimentConfig(NamedTuple):
    """Configuration for a single experiment.

    Attributes:
        name: Human-readable name for this configuration
        learner_factory: Callable that returns a fresh learner instance
        stream_factory: Callable that returns a fresh stream instance
        num_steps: Number of learning steps to run
    """

    name: str
    learner_factory: Callable[[], LinearLearner]
    stream_factory: Callable[[], ScanStream[Any]]
    num_steps: int


class SingleRunResult(NamedTuple):
    """Result from a single experiment run.

    Attributes:
        config_name: Name of the configuration that was run
        seed: Random seed used for this run
        metrics_history: List of metric dictionaries from each step
        final_state: Final learner state after training
    """

    config_name: str
    seed: int
    metrics_history: list[dict[str, float]]
    final_state: LearnerState


class MetricSummary(NamedTuple):
    """Summary statistics for a single metric.

    Attributes:
        mean: Mean across seeds
        std: Sample standard deviation across seeds (zero for one seed)
        min: Minimum value across seeds
        max: Maximum value across seeds
        n_seeds: Number of seeds
        values: Raw values per seed
    """

    mean: float
    std: float
    min: float
    max: float
    n_seeds: int
    values: NDArray[np.float64]


class AggregatedResults(NamedTuple):
    """Aggregated results across multiple seeds.

    Attributes:
        config_name: Name of the configuration
        seeds: List of seeds used
        metric_arrays: Dict mapping metric name to (n_seeds, n_steps) array
        summary: Dict mapping metric name to MetricSummary (final values)
    """

    config_name: str
    seeds: list[int]
    metric_arrays: dict[str, NDArray[np.float64]]
    summary: dict[str, MetricSummary]


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
) -> SingleRunResult:
    """Run a single experiment with a given seed.

    Args:
        config: Experiment configuration
        seed: Random seed for the stream

    Returns:
        SingleRunResult with metrics and final state
    """
    learner = config.learner_factory()
    stream = config.stream_factory()
    key = jr.key(seed)

    result = run_learning_loop(learner, stream, config.num_steps, key)
    final_state, metrics = cast(tuple[LearnerState, Any], result)
    normalized = learner.normalizer is not None
    metrics_history = metrics_to_dicts(metrics, normalized=normalized)

    return SingleRunResult(
        config_name=config.name,
        seed=seed,
        metrics_history=metrics_history,
        final_state=final_state,
    )


def _require_finite_metric_array(arr: NDArray[np.float64], metric: str) -> None:
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"metric {metric!r} contains non-finite samples")


def aggregate_metrics(results: list[SingleRunResult]) -> AggregatedResults:
    """Aggregate results from multiple seeds into summary statistics.

    Each metric's per-seed "final value" is the mean over the last 100 steps
    (the whole trace when shorter).  The 100-step window is a fixed
    convention here, chosen to estimate settled performance less noisily
    than the last step; use :func:`get_final_performance` when a different
    window is needed.

    Args:
        results: List of SingleRunResult from multiple seeds

    Returns:
        AggregatedResults with aggregated metrics

    Raises:
        ValueError: If ``results`` is empty, mixes configuration names or seed
            identities, drifts in metric keys, or any metric sample is
            non-finite.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    config_names = sorted({r.config_name for r in results})
    if len(config_names) != 1:
        raise ValueError(
            f"aggregate_metrics requires runs from one configuration; got {config_names}"
        )
    config_name = config_names[0]
    seeds = [r.seed for r in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"aggregate_metrics requires unique seed identities; got {seeds}")

    # Get all metric keys from first result
    if any(
        not r.metrics_history
        or any(set(metrics) != set(results[0].metrics_history[0]) for metrics in r.metrics_history)
        for r in results
    ):
        raise ValueError("all runs must contain the same metric keys at every step")
    metric_keys = list(results[0].metrics_history[0].keys())

    # Build metric arrays: (n_seeds, n_steps)
    metric_arrays: dict[str, NDArray[np.float64]] = {}
    for key in metric_keys:
        arrays = []
        for r in results:
            values = np.array([m[key] for m in r.metrics_history])
            arrays.append(values)
        metric_arrays[key] = np.stack(arrays)
        _require_finite_metric_array(metric_arrays[key], key)

    # Compute summary statistics for final values (mean of last 100 steps)
    summary: dict[str, MetricSummary] = {}
    n_seeds = len(results)
    for key in metric_keys:
        # Use mean of last 100 steps as the final value
        window = min(100, metric_arrays[key].shape[1])
        final_values = np.mean(metric_arrays[key][:, -window:], axis=1)
        summary[key] = MetricSummary(
            mean=float(np.mean(final_values)),
            std=float(np.std(final_values, ddof=1)) if n_seeds > 1 else 0.0,
            min=float(np.min(final_values)),
            max=float(np.max(final_values)),
            n_seeds=n_seeds,
            values=final_values,
        )

    return AggregatedResults(
        config_name=config_name,
        seeds=seeds,
        metric_arrays=metric_arrays,
        summary=summary,
    )


def run_multi_seed_experiment(
    configs: Sequence[ExperimentConfig],
    seeds: int | Sequence[int] = 30,
    parallel: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> dict[str, AggregatedResults]:
    """Run experiments across multiple seeds with optional parallelization.

    Args:
        configs: List of experiment configurations to run. Names must be unique.
        seeds: Number of seeds (generates 0..n-1) or explicit list of seeds.
            Explicit seeds must be unique.
        parallel: Whether to use parallel execution (requires joblib)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        Dictionary mapping config name to AggregatedResults

    Raises:
        ValueError: If two or more configurations have the same name, if the
            seed count is not a positive built-in integer, or if the explicit
            seed list is empty, non-canonical, or contains duplicates
    """
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for config in configs:
        if config.name in seen_names:
            duplicate_names.add(config.name)
        else:
            seen_names.add(config.name)

    if duplicate_names:
        formatted_names = ", ".join(repr(name) for name in sorted(duplicate_names))
        raise ValueError(
            f"Experiment configuration names must be unique; duplicates: {formatted_names}"
        )

    # Convert seeds to list. Bool is a subclass of int, so isinstance(seeds, int)
    # would treat True as a one-seed experiment and False as an empty run.
    if type(seeds) is int:
        if seeds < 1:
            raise ValueError("seeds count must be a positive built-in integer")
        seed_list = list(range(seeds))
    else:
        if isinstance(seeds, (str, bytes, bytearray)):
            raise ValueError(
                "seeds must be a positive built-in integer count or a sequence "
                "of unique built-in integer seeds"
            )
        try:
            raw_seeds = tuple(seeds)
        except TypeError as exc:
            raise ValueError(
                "seeds must be a positive built-in integer count or a sequence "
                "of unique built-in integer seeds"
            ) from exc
        if not raw_seeds:
            raise ValueError(
                "seeds must be a non-empty sequence of unique built-in integer seeds"
            )
        seed_list = [
            require_jax_seed(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(raw_seeds)
        ]

    seen_seeds: set[int] = set()
    duplicate_seeds: set[int] = set()
    for seed in seed_list:
        if seed in seen_seeds:
            duplicate_seeds.add(seed)
        else:
            seen_seeds.add(seed)

    if duplicate_seeds:
        formatted_seeds = ", ".join(str(seed) for seed in sorted(duplicate_seeds))
        raise ValueError(f"Experiment seeds must be unique; duplicates: {formatted_seeds}")

    # Build list of (config, seed) pairs
    tasks: list[tuple[ExperimentConfig, int]] = []
    for config in configs:
        for seed in seed_list:
            tasks.append((config, seed))

    # Run experiments
    if parallel:
        try:
            from joblib import Parallel, delayed

            if show_progress:
                try:
                    from tqdm import tqdm

                    results_list: list[SingleRunResult] = Parallel(n_jobs=n_jobs)(
                        delayed(run_single_experiment)(config, seed)
                        for config, seed in tqdm(tasks, desc="Running experiments")
                    )
                except ImportError:
                    results_list = Parallel(n_jobs=n_jobs)(
                        delayed(run_single_experiment)(config, seed) for config, seed in tasks
                    )
            else:
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_experiment)(config, seed) for config, seed in tasks
                )
        except ImportError:
            # Fallback to sequential if joblib not available
            results_list = _run_sequential(tasks, show_progress)
    else:
        results_list = _run_sequential(tasks, show_progress)

    # Group results by config name
    grouped: dict[str, list[SingleRunResult]] = {}
    for result in results_list:
        if result.config_name not in grouped:
            grouped[result.config_name] = []
        grouped[result.config_name].append(result)

    # Aggregate each config
    aggregated: dict[str, AggregatedResults] = {}
    for config_name, group_results in grouped.items():
        aggregated[config_name] = aggregate_metrics(group_results)

    return aggregated


def _run_sequential(
    tasks: list[tuple[ExperimentConfig, int]],
    show_progress: bool,
) -> list[SingleRunResult]:
    """Run experiments sequentially."""
    if show_progress:
        try:
            from tqdm import tqdm

            return [run_single_experiment(config, seed) for config, seed in tqdm(tasks)]
        except ImportError:
            pass
    return [run_single_experiment(config, seed) for config, seed in tasks]


def get_metric_timeseries(
    results: AggregatedResults,
    metric: str = "squared_error",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Get the mean and one-sample-standard-deviation band for a metric.

    Args:
        results: Aggregated results
        metric: Name of the metric

    Returns:
        Tuple of ``(mean, mean - sample_std, mean + sample_std)`` arrays. A
        one-seed aggregate has zero spread and therefore a point band.
    """
    arr = results.metric_arrays[metric]
    _require_finite_metric_array(arr, metric)
    mean = np.mean(arr, axis=0)
    std = np.zeros_like(mean) if arr.shape[0] == 1 else np.std(arr, axis=0, ddof=1)
    return mean, mean - std, mean + std


def get_final_performance(
    results: dict[str, AggregatedResults],
    metric: str = "squared_error",
    window: int = 100,
) -> dict[str, tuple[float, float]]:
    """Get final performance (mean, sample std) for each config.

    Args:
        results: Dictionary of aggregated results
        metric: Metric to evaluate
        window: Number of final steps to average. Must be positive. A window
            longer than the trace uses the whole trace, matching
            :func:`aggregate_metrics`.

    Returns:
        Dictionary mapping config name to ``(mean, sample_std)``. A one-seed
        aggregate reports zero spread.

        Raises:
        ValueError: If ``window`` is not positive, a metric array has no
            time steps, any metric sample is non-finite, or ``window`` exceeds
            the shortest trace while trace lengths differ between methods.
            ``window=0`` is not "the last step": ``arr[:, -0:]`` is the full
            trace, so the helper refuses rather than silently reporting a
            full-horizon mean.
    """
    if type(window) is not int:
        raise ValueError("window must be a positive built-in integer")
    if window <= 0:
        raise ValueError(f"window must be positive (got {window})")

    metric_arrays: dict[str, NDArray[np.float64]] = {}
    for name, agg in results.items():
        arr = agg.metric_arrays[metric]
        if arr.shape[1] == 0:
            raise ValueError(
                f"AggregatedResults {name!r} must contain at least one metric step "
                f"for {metric!r}"
            )
        _require_finite_metric_array(arr, metric)
        metric_arrays[name] = arr
    if not metric_arrays:
        return {}
    final_window = common_final_window(
        {name: arr.shape[1] for name, arr in metric_arrays.items()}, window, metric
    )

    performance: dict[str, tuple[float, float]] = {}
    for name, arr in metric_arrays.items():
        final_means = np.mean(arr[:, -final_window:], axis=1)
        std = float(np.std(final_means, ddof=1)) if len(final_means) > 1 else 0.0
        performance[name] = (float(np.mean(final_means)), std)
    return performance


def extract_hyperparameter_results(
    results: dict[str, AggregatedResults],
    metric: str = "squared_error",
    param_extractor: Callable[[str], Any] | None = None,
) -> dict[Any, tuple[float, float]]:
    """Extract results indexed by hyperparameter value.

    Useful for creating hyperparameter sensitivity plots.

    Args:
        results: Dictionary of aggregated results
        metric: Metric to evaluate
        param_extractor: Function to extract a canonical immutable coordinate
            from a config name. Accepted scalar coordinates are exact ``None``,
            ``bool``, ``int``, finite ``float`` or ``complex``, finite ``Decimal``,
            ``Fraction``, ``str``, ``bytes``, and the corresponding supported exact
            NumPy scalar types. Fraction leaves are normalized into private immutable
            rational keys with the same Python-numeric equality, hash, representation,
            and lookup behavior; their input object identity is not retained. Exact
            tuples and frozensets may compose scalar coordinates to at most 32 nested
            container levels. Across configurations, Python numeric types form one
            coherent family, while a NumPy numeric scalar is compatible only with the
            same exact NumPy scalar type. Coordinate pairs are certified recursively;
            incompatible container or scalar families fail closed before dictionary
            equality is invoked. Enums, UUIDs, datetimes, named-tuple subclasses, and
            other user-defined or noncanonical immutable keys are not accepted.

    Returns:
        Dictionary mapping each canonical parameter value to its (mean, std) tuple.
        Non-Fraction coordinates retain their original key objects; Fraction leaves
        use immutable, numerically equivalent snapshots.

    Raises:
        ValueError: If ``param_extractor`` returns a coordinate outside the
            canonical immutable-key contract, exceeds the nesting limit, mixes
            mutually incompatible coordinate families, or maps more than one
            configuration to the same value; a sensitivity curve built from a
            silently truncated, insertion-order-dependent subset is not a measurement.
    """
    performance = get_final_performance(results, metric)

    if param_extractor is None:
        return {k: v for k, v in performance.items()}

    coordinate_entries: list[tuple[str, object, int]] = []
    for name in performance:
        coordinate = _require_hyperparameter_coordinate(param_extractor(name), name=name)
        coordinate_hash = _require_coordinate_hash(coordinate, name=name)
        coordinate_entries.append((name, coordinate, coordinate_hash))

    for left_index, (left_name, left, left_hash) in enumerate(coordinate_entries):
        for right_name, right, right_hash in coordinate_entries[left_index + 1 :]:
            if not _coordinate_pair_is_compatible(left, right):
                raise ValueError(
                    "param_extractor must return mutually compatible canonical coordinate "
                    f"families; configurations {left_name!r} and {right_name!r} do not"
                )
            if _coordinate_values_equal(left, right) and left_hash != right_hash:
                raise ValueError(
                    "param_extractor returned equal coordinates with different hashes for "
                    f"configurations {left_name!r} and {right_name!r}"
                )

    coordinate_groups: list[tuple[object, list[str]]] = []
    for name, coordinate, _ in coordinate_entries:
        for representative, names in coordinate_groups:
            if _coordinate_values_equal(representative, coordinate):
                names.append(name)
                break
        else:
            coordinate_groups.append((coordinate, [name]))

    collisions = [(value, names) for value, names in coordinate_groups if len(names) > 1]
    if collisions:
        described = "; ".join(f"{value!r} <- {names}" for value, names in collisions)
        raise ValueError(
            f"param_extractor maps several configurations to one value: {described}"
        )
    try:
        return {value: performance[names[0]] for value, names in coordinate_groups}
    except Exception as exc:
        raise ValueError(
            "param_extractor coordinates could not maintain the canonical dictionary-key contract"
        ) from exc


def _require_hyperparameter_coordinate(
    value: object,
    *,
    name: str,
    nesting_depth: int = 0,
) -> Any:
    """Require one coordinate whose finiteness and hash semantics are intrinsic.

    Accepted coordinates are exact immutable Python scalar types, ``Decimal``,
    ``Fraction``, supported exact NumPy scalar types, and exact ``tuple`` or
    ``frozenset`` compositions of those types. User-defined subclasses and keys
    are deliberately excluded with identity-only type matching because calling
    their metaclass, conversion, equality, or hash hooks cannot establish a durable
    dictionary-key contract.
    """

    def reject() -> NoReturn:
        raise ValueError(
            "param_extractor returned a noncanonical coordinate for "
            f"configuration {name!r}; coordinates must use canonical immutable "
            "scalar types or exact tuple/frozenset compositions, and floating "
            "or complex coordinates must be finite"
        )

    value_type = type(value)
    if _type_identity_in(value_type, (tuple, frozenset)):
        if nesting_depth >= _MAX_HYPERPARAMETER_COORDINATE_NESTING:
            raise ValueError(
                "param_extractor returned an over-nested coordinate for "
                f"configuration {name!r}; coordinates support at most "
                f"{_MAX_HYPERPARAMETER_COORDINATE_NESTING} nested "
                "tuple/frozenset levels"
            )
        components = cast(tuple[object, ...] | frozenset[object], value)
        canonical_components = tuple(
            _require_hyperparameter_coordinate(
                component,
                name=name,
                nesting_depth=nesting_depth + 1,
            )
            for component in components
        )
        if all(
            canonical is original
            for canonical, original in zip(canonical_components, components, strict=True)
        ):
            return value
        if value_type is tuple:
            return canonical_components
        canonical_set = frozenset(canonical_components)
        if len(canonical_set) != len(components):
            reject()
        return canonical_set

    if value_type is Fraction:
        fraction = cast(Fraction, value)
        try:
            numerator = fraction.numerator
            denominator = fraction.denominator
        except AttributeError:
            reject()
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
        ):
            reject()
        return _CanonicalFractionCoordinate(numerator, denominator)

    if value is None or _type_identity_in(value_type, (bool, int, str, bytes)):
        return value

    if value_type is float:
        if not math.isfinite(cast(float, value)):
            reject()
        return value

    if value_type is complex:
        number = cast(complex, value)
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            reject()
        return value

    if value_type is Decimal:
        if not cast(Decimal, value).is_finite():
            reject()
        return value

    if _type_identity_in(value_type, _NUMPY_COORDINATE_TYPES):
        if np.dtype(value_type).kind in ("f", "c") and not bool(
            np.isfinite(cast(Any, value))
        ):
            reject()
        return value

    reject()


def _require_coordinate_hash(value: object, *, name: str) -> int:
    """Hash a validated coordinate without allowing raw runtime errors to escape."""
    try:
        return hash(value)
    except Exception as exc:
        raise ValueError(
            "param_extractor returned a coordinate that cannot be hashed for "
            f"configuration {name!r}"
        ) from exc


def _is_python_numeric_coordinate_type(value_type: type[object]) -> bool:
    return _type_identity_in(value_type, _PYTHON_NUMERIC_COORDINATE_TYPES)


def _is_numpy_numeric_coordinate_type(value_type: type[object]) -> bool:
    return _type_identity_in(value_type, _NUMPY_COORDINATE_TYPES) and np.dtype(
        value_type
    ).kind in (
        "b",
        "i",
        "u",
        "f",
        "c",
    )


def _coordinate_pair_is_compatible(left: object, right: object) -> bool:
    """Certify that equality between two validated coordinates cannot narrow."""
    left_type = type(left)
    right_type = type(right)

    if left_type is tuple and right_type is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return all(
            _coordinate_pair_is_compatible(left_item, right_item)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=False)
        )

    if left_type is frozenset and right_type is frozenset:
        left_set = cast(frozenset[object], left)
        right_set = cast(frozenset[object], right)
        return all(
            _coordinate_pair_is_compatible(left_item, right_item)
            for left_item in left_set
            for right_item in right_set
        )

    if _type_identity_in(left_type, (tuple, frozenset)) or _type_identity_in(
        right_type, (tuple, frozenset)
    ):
        return False

    if left_type is right_type:
        return True
    if _is_python_numeric_coordinate_type(left_type) and _is_python_numeric_coordinate_type(
        right_type
    ):
        return True
    if _is_numpy_numeric_coordinate_type(left_type) and _is_numpy_numeric_coordinate_type(
        right_type
    ):
        return False
    if (
        _is_python_numeric_coordinate_type(left_type)
        and _is_numpy_numeric_coordinate_type(right_type)
    ) or (
        _is_numpy_numeric_coordinate_type(left_type)
        and _is_python_numeric_coordinate_type(right_type)
    ):
        return False
    return True


def _coordinate_values_equal(left: object, right: object) -> bool:
    """Compare a pair after :func:`_coordinate_pair_is_compatible` accepts it."""
    left_type = type(left)
    right_type = type(right)

    if left_type is tuple and right_type is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return len(left_tuple) == len(right_tuple) and all(
            _coordinate_values_equal(left_item, right_item)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=True)
        )

    if left_type is frozenset and right_type is frozenset:
        left_items = list(cast(frozenset[object], left))
        unmatched_right = list(cast(frozenset[object], right))
        if len(left_items) != len(unmatched_right):
            return False
        for left_item in left_items:
            for right_index, right_item in enumerate(unmatched_right):
                if _coordinate_values_equal(left_item, right_item):
                    unmatched_right.pop(right_index)
                    break
            else:
                return False
        return True

    if _type_identity_in(left_type, _STRING_COORDINATE_TYPES) and _type_identity_in(
        right_type, _STRING_COORDINATE_TYPES
    ):
        return bool(left == right)
    if _type_identity_in(left_type, _BYTES_COORDINATE_TYPES) and _type_identity_in(
        right_type, _BYTES_COORDINATE_TYPES
    ):
        return bool(left == right)
    if left_type is right_type or (
        _is_python_numeric_coordinate_type(left_type)
        and _is_python_numeric_coordinate_type(right_type)
    ):
        return bool(left == right)
    return False
