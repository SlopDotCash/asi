"""Contract tests for validating public multi-seed experiment inputs."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any, Never

import numpy as np
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.streams.base import ScanStream
from alberta_framework.streams.synthetic import RandomWalkStream
from alberta_framework.utils.experiments import (
    AggregatedResults,
    ExperimentConfig,
    SingleRunResult,
    aggregate_metrics,
    extract_hyperparameter_results,
    get_final_performance,
    get_metric_timeseries,
    run_multi_seed_experiment,
)

pytestmark = pytest.mark.unit


def _fail_if_called() -> Never:
    raise AssertionError("experiment factory must not be called")


def _stream_factory() -> ScanStream[Any]:
    return RandomWalkStream(feature_dim=2)


def _config(
    name: str,
    *,
    learner_factory: Callable[[], LinearLearner] = LinearLearner,
    stream_factory: Callable[[], ScanStream[Any]] = _stream_factory,
    num_steps: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        learner_factory=learner_factory,
        stream_factory=stream_factory,
        num_steps=num_steps,
    )


def test_duplicate_names_reject_before_distinct_factories_execute() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=1,
        ),
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment(configs, seeds=[0, 1], parallel=False, show_progress=False)


def test_repeated_config_object_rejects_before_factory_executes() -> None:
    config = _config(
        "baseline",
        learner_factory=_fail_if_called,
        stream_factory=_fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment([config, config], seeds=[0], parallel=False, show_progress=False)


def test_multiple_duplicate_names_are_reported_deterministically() -> None:
    configs = [
        _config(
            name,
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
        for name in ("zeta", "alpha", "zeta", "beta", "alpha", "beta", "zeta")
    ]

    with pytest.raises(ValueError) as exc_info:
        run_multi_seed_experiment(configs, seeds=[0], parallel=False, show_progress=False)

    assert str(exc_info.value) == (
        "Experiment configuration names must be unique; duplicates: 'alpha', 'beta', 'zeta'"
    )


def test_duplicate_seeds_reject_before_factories_execute() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
    ]

    with pytest.raises(
        ValueError,
        match=r"^Experiment seeds must be unique; duplicates: 0$",
    ):
        run_multi_seed_experiment(configs, seeds=[0, 0, 1], parallel=False, show_progress=False)


def test_multiple_duplicate_seeds_are_reported_deterministically() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        run_multi_seed_experiment(
            configs, seeds=[7, 3, 7, 0, 3, 7], parallel=False, show_progress=False
        )

    assert str(exc_info.value) == "Experiment seeds must be unique; duplicates: 3, 7"


def test_unique_names_preserve_config_and_seed_order() -> None:
    results = run_multi_seed_experiment(
        [_config("second"), _config("first")],
        seeds=[7, 3],
        parallel=False,
        show_progress=False,
    )

    assert list(results) == ["second", "first"]
    assert results["second"].seeds == [7, 3]
    assert results["first"].seeds == [7, 3]
    assert all(
        summary.n_seeds == 2
        for result in results.values()
        for summary in result.summary.values()
    )
    for result in results.values():
        for summary in result.summary.values():
            assert summary.std == pytest.approx(float(np.std(summary.values, ddof=1)))


def test_seed_axis_surfaces_use_sample_standard_deviation() -> None:
    values = np.asarray([0.10, 0.12, 0.30], dtype=np.float64)
    aggregate = AggregatedResults(
        config_name="candidate",
        seeds=[0, 1, 2],
        metric_arrays={"squared_error": values[:, None]},
        summary={},
    )
    expected_std = float(np.std(values, ddof=1))

    mean, lower, upper = get_metric_timeseries(aggregate)
    performance = get_final_performance({"candidate": aggregate}, window=1)

    assert mean == pytest.approx([float(np.mean(values))])
    assert lower == pytest.approx(mean - expected_std)
    assert upper == pytest.approx(mean + expected_std)
    assert performance["candidate"] == pytest.approx((float(np.mean(values)), expected_std))


def test_single_seed_surfaces_report_zero_spread() -> None:
    trajectory = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    aggregate = AggregatedResults(
        config_name="single",
        seeds=[7],
        metric_arrays={"squared_error": trajectory},
        summary={},
    )

    mean, lower, upper = get_metric_timeseries(aggregate)
    performance = get_final_performance({"single": aggregate}, window=2)

    assert mean == pytest.approx(trajectory[0])
    assert lower == pytest.approx(mean)
    assert upper == pytest.approx(mean)
    assert performance["single"] == pytest.approx((2.5, 0.0))


def test_empty_config_sequence_still_returns_empty_results() -> None:
    assert (
        run_multi_seed_experiment([], seeds=[0], parallel=False, show_progress=False) == {}
    )


def _two_seed_trace() -> AggregatedResults:
    return AggregatedResults(
        config_name="candidate",
        seeds=[0, 1],
        metric_arrays={
            "squared_error": np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
        },
        summary={},
    )


@pytest.mark.parametrize("window", [0, -1, -5])
def test_get_final_performance_rejects_non_positive_window(window: int) -> None:
    """window<=0 is undefined: window=0 slices the whole trace, negatives drop a prefix."""
    with pytest.raises(ValueError, match=rf"^window must be positive \(got {window}\)$"):
        get_final_performance({"candidate": _two_seed_trace()}, window=window)


def test_get_final_performance_rejects_empty_time_axis() -> None:
    empty = AggregatedResults(
        config_name="empty",
        seeds=[0, 1],
        metric_arrays={"squared_error": np.zeros((2, 0), dtype=np.float64)},
        summary={},
    )
    with pytest.raises(
        ValueError,
        match=r"^AggregatedResults 'empty' must contain at least one metric step "
        r"for 'squared_error'$",
    ):
        get_final_performance({"empty": empty}, window=1)


def test_get_final_performance_rejects_unequal_final_windows() -> None:
    """Two methods must not be averaged over different numbers of final steps."""
    short = AggregatedResults(
        config_name="short",
        seeds=[0, 1],
        metric_arrays={"squared_error": np.full((2, 3), 2.0, dtype=np.float64)},
        summary={},
    )
    long = AggregatedResults(
        config_name="long",
        seeds=[0, 1],
        metric_arrays={"squared_error": np.full((2, 8), 2.0, dtype=np.float64)},
        summary={},
    )
    with pytest.raises(
        ValueError,
        match=r"^window=5 exceeds the shortest 'squared_error' trace and the traces differ "
        r"in length \(long: 8 steps, short: 3 steps\); every method must average the same "
        r"number of final steps$",
    ):
        get_final_performance({"short": short, "long": long}, window=5)


def test_get_final_performance_accepts_unequal_trace_lengths_when_window_fits() -> None:
    short = AggregatedResults(
        config_name="short",
        seeds=[0, 1],
        metric_arrays={"squared_error": np.asarray([[9.0, 1.0, 1.0], [9.0, 3.0, 3.0]])},
        summary={},
    )
    long = AggregatedResults(
        config_name="long",
        seeds=[0, 1],
        metric_arrays={
            "squared_error": np.asarray([[9.0] * 6 + [1.0, 1.0], [9.0] * 6 + [3.0, 3.0]])
        },
        summary={},
    )
    performance = get_final_performance({"short": short, "long": long}, window=2)
    assert performance["short"] == performance["long"] == (2.0, pytest.approx(np.sqrt(2.0)))


def test_get_final_performance_window_longer_than_trace_uses_full_trace() -> None:
    """The documented min(window, n_steps) convention is unchanged for window > 0."""
    result = get_final_performance({"candidate": _two_seed_trace()}, window=100)
    expected_values = np.asarray([2.0, 20.0], dtype=np.float64)
    assert result["candidate"][0] == pytest.approx(float(np.mean(expected_values)))
    assert result["candidate"][1] == pytest.approx(float(np.std(expected_values, ddof=1)))


def test_get_final_performance_positive_window_is_a_suffix() -> None:
    result = get_final_performance({"candidate": _two_seed_trace()}, window=2)
    expected_values = np.asarray([2.5, 25.0], dtype=np.float64)
    assert result["candidate"][0] == pytest.approx(float(np.mean(expected_values)))
    assert result["candidate"][1] == pytest.approx(float(np.std(expected_values, ddof=1)))


def _single_run(seed: int, values: list[float]) -> SingleRunResult:
    learner = LinearLearner(optimizer=LMS(step_size=0.05))
    return SingleRunResult(
        config_name="candidate",
        seed=seed,
        metrics_history=[{"squared_error": value} for value in values],
        final_state=learner.init(2),
    )


def test_aggregate_metrics_rejects_nonfinite_samples() -> None:
    """A NaN seed mean would be published as the method's final performance."""
    with pytest.raises(ValueError, match="non-finite samples"):
        aggregate_metrics(
            [
                _single_run(0, [1.0, 2.0]),
                _single_run(1, [3.0, float("nan")]),
            ]
        )


def test_aggregate_metrics_rejects_metric_schema_drift_in_later_seed() -> None:
    """A metric appearing in only one seed must not be silently discarded."""
    with pytest.raises(ValueError, match="same metric keys"):
        aggregate_metrics(
            [
                _single_run(0, [1.0, 2.0]),
                SingleRunResult(
                    config_name="candidate",
                    seed=1,
                    metrics_history=[
                        {"squared_error": 3.0, "accuracy": 0.4},
                        {"squared_error": 4.0, "accuracy": 0.8},
                    ],
                    final_state=LinearLearner().init(2),
                ),
            ]
        )


def test_aggregate_metrics_rejects_runs_from_different_configs() -> None:
    """Two arms must not be averaged into one AggregatedResults under the first arm's name."""
    treatment = _single_run(0, [9.0, 9.0])._replace(config_name="treatment")
    with pytest.raises(
        ValueError,
        match=r"^aggregate_metrics requires runs from one configuration; "
        r"got \['candidate', 'treatment'\]$",
    ):
        aggregate_metrics([_single_run(0, [1.0, 1.0]), treatment])


def test_aggregate_metrics_rejects_duplicate_seed_identities() -> None:
    with pytest.raises(
        ValueError,
        match=r"^aggregate_metrics requires unique seed identities; got \[0, 1, 0\]$",
    ):
        aggregate_metrics(
            [_single_run(0, [1.0, 1.0]), _single_run(1, [2.0, 2.0]), _single_run(0, [3.0, 3.0])]
        )


def _flat_trace(name: str, value: float) -> AggregatedResults:
    return AggregatedResults(
        config_name=name,
        seeds=[0, 1],
        metric_arrays={"squared_error": np.full((2, 3), value, dtype=np.float64)},
        summary={},
    )


def test_extract_hyperparameter_results_rejects_colliding_parameter_values() -> None:
    """A non-injective extractor must not silently keep the dict-last configuration."""
    results = {
        "lr_0.01_decay_0.0": _flat_trace("lr_0.01_decay_0.0", 1.0),
        "lr_0.01_decay_0.1": _flat_trace("lr_0.01_decay_0.1", 9.0),
        "lr_0.10_decay_0.0": _flat_trace("lr_0.10_decay_0.0", 3.0),
    }
    with pytest.raises(
        ValueError,
        match=r"^param_extractor maps several configurations to one value: "
        r"0\.01 <- \['lr_0\.01_decay_0\.0', 'lr_0\.01_decay_0\.1'\]$",
    ):
        extract_hyperparameter_results(
            results, param_extractor=lambda name: float(name.split("_")[1])
        )


def test_extract_hyperparameter_results_keeps_injective_extractors() -> None:
    results = {
        "lr_0.01": _flat_trace("lr_0.01", 1.0),
        "lr_0.10": _flat_trace("lr_0.10", 3.0),
    }
    extracted = extract_hyperparameter_results(
        results, param_extractor=lambda name: float(name.split("_")[1])
    )
    assert extracted == {0.01: (1.0, 0.0), 0.1: (3.0, 0.0)}


@pytest.mark.parametrize(
    "coordinate",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        complex(float("nan"), 0.0),
        Decimal("NaN"),
        Decimal("Infinity"),
        np.longdouble("nan"),
        ("nested", float("nan")),
        frozenset(("nested", float("inf"))),
    ],
)
def test_extract_hyperparameter_results_rejects_nonfinite_coordinates(
    coordinate: object,
) -> None:
    results = {"lr_0.01": _flat_trace("lr_0.01", 1.0)}
    with pytest.raises(
        ValueError,
        match=r"^param_extractor returned a noncanonical coordinate for "
        r"configuration 'lr_0\.01'",
    ):
        extract_hyperparameter_results(results, param_extractor=lambda _: coordinate)


@pytest.mark.parametrize("coordinate", [[], {}, np.asarray([0.01])])
def test_extract_hyperparameter_results_rejects_unhashable_coordinates(
    coordinate: object,
) -> None:
    results = {"lr_0.01": _flat_trace("lr_0.01", 1.0)}
    with pytest.raises(
        ValueError,
        match=r"^param_extractor returned a noncanonical coordinate for "
        r"configuration 'lr_0\.01'",
    ):
        extract_hyperparameter_results(results, param_extractor=lambda _: coordinate)


def test_extract_hyperparameter_results_keeps_canonical_categorical_coordinates() -> None:
    results = {
        "sgd": _flat_trace("sgd", 1.0),
        "adam": _flat_trace("adam", 3.0),
    }
    extracted = extract_hyperparameter_results(
        results, param_extractor=lambda name: ("optimizer", name)
    )
    assert extracted == {
        ("optimizer", "sgd"): (1.0, 0.0),
        ("optimizer", "adam"): (3.0, 0.0),
    }


@pytest.mark.parametrize(
    "coordinate",
    [
        None,
        True,
        -7,
        0.125,
        1.0 + 2.0j,
        "adam",
        b"adam",
        Decimal("0.125"),
        Fraction(1, 8),
        np.bool_(True),
        np.int64(7),
        np.float32(0.125),
        np.complex64(1.0 + 2.0j),
        np.str_("adam"),
        np.bytes_("adam"),
        ("optimizer", np.float32(0.125)),
        frozenset(("optimizer", np.int16(7))),
    ],
)
def test_extract_hyperparameter_results_keeps_canonical_coordinate_families(
    coordinate: object,
) -> None:
    extracted = extract_hyperparameter_results(
        {"candidate": _flat_trace("candidate", 1.0)},
        param_extractor=lambda _: coordinate,
    )

    assert len(extracted) == 1
    returned_coordinate = next(iter(extracted))
    if type(coordinate) is Fraction:
        assert returned_coordinate is not coordinate
        assert returned_coordinate == coordinate
        assert hash(returned_coordinate) == hash(coordinate)
        assert repr(returned_coordinate) == repr(coordinate)
    else:
        assert returned_coordinate is coordinate


class _NonfiniteFloatThatConvertsFinite(float):
    def __new__(cls) -> _NonfiniteFloatThatConvertsFinite:
        return super().__new__(cls, float("inf"))

    def __complex__(self) -> complex:
        return 0j


def _platform_longdouble_1e4000() -> np.longdouble:
    """Parse the cross-platform boundary without leaking an expected warning."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.longdouble("1e4000")


@pytest.mark.parametrize(
    "coordinate",
    [
        np.longdouble("inf"),
        np.longdouble("nan"),
        # Overflows to inf at parse wherever longdouble is float64 (aarch64),
        # and stays a finite extended value where it is wider (x86-64): the
        # validator must classify by the platform's actual value, not the
        # literal, so this case is finite-or-rejected but never a crash.
        _platform_longdouble_1e4000(),
    ],
)
def test_extract_hyperparameter_results_classifies_longdouble_by_platform_value(
    coordinate: object,
) -> None:
    trace = {"candidate": _flat_trace("candidate", 1.0)}
    if bool(np.isfinite(coordinate)):
        extracted = extract_hyperparameter_results(trace, param_extractor=lambda _: coordinate)
        assert next(iter(extracted)) is coordinate
    else:
        with pytest.raises(ValueError, match=r"noncanonical coordinate"):
            extract_hyperparameter_results(trace, param_extractor=lambda _: coordinate)


def test_extract_hyperparameter_results_rejects_spoofed_numeric_subclasses() -> None:
    coordinate = _NonfiniteFloatThatConvertsFinite()
    assert coordinate == float("inf")

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: coordinate,
        )


@pytest.mark.parametrize(
    "coordinate",
    [
        10**1000,
        Fraction(10**1000, 3),
        Decimal("1e4000"),
        # The widest finite value the platform's longdouble can hold. On x86-64
        # this is an 80-bit extended value far outside float64 range; on
        # aarch64 (macOS arm64) longdouble *is* float64, so a literal such as
        # ``np.longdouble("1e4000")`` overflows to inf at parse and can never
        # be a "finite" fixture there.
        np.finfo(np.longdouble).max,
    ],
)
def test_extract_hyperparameter_results_keeps_wide_finite_numeric_coordinates(
    coordinate: object,
) -> None:
    extracted = extract_hyperparameter_results(
        {"candidate": _flat_trace("candidate", 1.0)},
        param_extractor=lambda _: coordinate,
    )

    assert len(extracted) == 1
    returned_coordinate = next(iter(extracted))
    if type(coordinate) is Fraction:
        assert returned_coordinate is not coordinate
        assert returned_coordinate == coordinate
        assert hash(returned_coordinate) == hash(coordinate)
    else:
        assert returned_coordinate is coordinate


def _nested_fraction_coordinate(value: object, nesting: str) -> object:
    if nesting == "scalar":
        return value
    if nesting == "tuple":
        return ("fraction", value)
    assert nesting == "frozenset"
    return frozenset(("fraction", value))


@pytest.mark.parametrize("nesting", ["scalar", "tuple", "frozenset"])
@pytest.mark.parametrize("poisoned_first", [True, False], ids=("poisoned-first", "poisoned-last"))
def test_extract_hyperparameter_results_rejects_hidden_fraction_integer_hooks(
    nesting: str,
    poisoned_first: bool,
) -> None:
    calls: list[str] = []
    hooks_armed = False

    def record_hook(name: str) -> None:
        calls.append(name)
        if hooks_armed:
            raise RuntimeError(f"unsafe Fraction component hook: {name}")

    class HookedInt(int):
        def __hash__(self) -> int:
            record_hook("hash")
            return int.__hash__(self)

        def __eq__(self, other: object) -> bool:
            record_hook("eq")
            result = int.__eq__(self, other)
            return False if result is NotImplemented else result

        def __int__(self) -> int:
            record_hook("int")
            return int.__int__(self)

        def __index__(self) -> int:
            record_hook("index")
            return int.__index__(self)

    class Carrier(Fraction):
        @property
        def numerator(self) -> int:
            return HookedInt(1)

        @property
        def denominator(self) -> int:
            return HookedInt(2)

    poisoned = Fraction(Carrier(1, 2))
    assert type(poisoned) is Fraction
    assert type(poisoned.numerator) is HookedInt
    poisoned_coordinate = _nested_fraction_coordinate(poisoned, nesting)
    safe_coordinate = _nested_fraction_coordinate(Fraction(3, 4), nesting)
    calls.clear()
    hooks_armed = True
    coordinates = (
        (poisoned_coordinate, safe_coordinate)
        if poisoned_first
        else (safe_coordinate, poisoned_coordinate)
    )
    coordinate_iterator = iter(coordinates)
    results = {
        "candidate_0": _flat_trace("candidate_0", 1.0),
        "candidate_1": _flat_trace("candidate_1", 2.0),
    }

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            results,
            param_extractor=lambda _: next(coordinate_iterator),
        )

    assert calls == []


@pytest.mark.parametrize("nesting", ["scalar", "tuple", "frozenset"])
@pytest.mark.parametrize("reverse", [False, True], ids=("forward", "reverse"))
def test_extract_hyperparameter_results_snapshots_fraction_before_private_mutation(
    nesting: str,
    reverse: bool,
) -> None:
    fractions = [Fraction(1, 3), Fraction(2, 3)]
    if reverse:
        fractions.reverse()
    expected_coordinates = tuple(
        _nested_fraction_coordinate(Fraction(value.numerator, value.denominator), nesting)
        for value in fractions
    )
    calls = 0

    def extract_and_mutate(_: str) -> object:
        nonlocal calls
        if calls == 0:
            calls += 1
            return _nested_fraction_coordinate(fractions[0], nesting)
        object.__setattr__(fractions[0], "_numerator", fractions[1].numerator)
        object.__setattr__(fractions[0], "_denominator", fractions[1].denominator)
        calls += 1
        return _nested_fraction_coordinate(fractions[1], nesting)

    extracted = extract_hyperparameter_results(
        {
            "candidate_0": _flat_trace("candidate_0", 1.0),
            "candidate_1": _flat_trace("candidate_1", 2.0),
        },
        param_extractor=extract_and_mutate,
    )

    assert tuple(extracted) == expected_coordinates
    assert extracted[expected_coordinates[0]] == (1.0, 0.0)
    assert extracted[expected_coordinates[1]] == (2.0, 0.0)

    returned_coordinate = next(iter(extracted))
    if nesting == "scalar":
        returned_fraction = returned_coordinate
    elif nesting == "tuple":
        returned_fraction = returned_coordinate[1]
    else:
        returned_fraction = next(
            component for component in returned_coordinate if component != "fraction"
        )
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(returned_fraction, "_numerator", 99)

    object.__setattr__(fractions[0], "_numerator", 7)
    object.__setattr__(fractions[1], "_numerator", 11)
    assert extracted[expected_coordinates[0]] == (1.0, 0.0)
    assert extracted[expected_coordinates[1]] == (2.0, 0.0)


def test_extract_hyperparameter_results_rejects_delayed_hash_metaclass_spoof() -> None:
    metaclass_calls: list[str] = []

    class NumpyIntegerSpoof(type):
        def __hash__(cls) -> int:
            metaclass_calls.append("hash")
            return hash(np.int64)

        def __eq__(cls, other: object) -> bool:
            metaclass_calls.append("eq")
            return other is np.int64

    class DelayedHash(metaclass=NumpyIntegerSpoof):
        def __init__(self) -> None:
            self.hash_calls = 0

        def __hash__(self) -> int:
            self.hash_calls += 1
            return 100 + self.hash_calls

        def __eq__(self, other: object) -> bool:
            return self is other

    coordinate = DelayedHash()

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: coordinate,
        )

    assert metaclass_calls == []
    assert coordinate.hash_calls == 0


@pytest.mark.parametrize("nesting", ["scalar", "tuple", "frozenset"])
def test_extract_hyperparameter_results_rejects_raising_metaclass_without_hooks(
    nesting: str,
) -> None:
    metaclass_calls: list[str] = []

    class RaisingMetaclass(type):
        def __hash__(cls) -> int:
            metaclass_calls.append("hash")
            raise RuntimeError("metaclass hash must not run")

        def __eq__(cls, other: object) -> bool:
            metaclass_calls.append("eq")
            raise RuntimeError("metaclass equality must not run")

    class Coordinate(metaclass=RaisingMetaclass):
        pass

    coordinate = _nested_fraction_coordinate(Coordinate(), nesting)
    metaclass_calls.clear()

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: coordinate,
        )

    assert metaclass_calls == []


def test_canonical_fraction_comparison_does_not_run_hostile_metaclass_equality() -> None:
    metaclass_calls: list[str] = []

    class RaisingMetaclass(type):
        def __eq__(cls, other: object) -> bool:
            metaclass_calls.append("eq")
            raise RuntimeError("metaclass equality must not run")

    class Coordinate(metaclass=RaisingMetaclass):
        pass

    extracted = extract_hyperparameter_results(
        {"candidate": _flat_trace("candidate", 1.0)},
        param_extractor=lambda _: Fraction(1, 2),
    )
    canonical_fraction = next(iter(extracted))
    hostile = Coordinate()

    assert (canonical_fraction == hostile) is False
    assert (hostile == canonical_fraction) is False
    assert (canonical_fraction != hostile) is True
    assert (hostile != canonical_fraction) is True
    assert metaclass_calls == []


@pytest.mark.parametrize("nesting", ["scalar", "tuple", "frozenset"])
@pytest.mark.parametrize("component", ["numerator", "denominator"])
def test_extract_hyperparameter_results_normalizes_missing_fraction_slots(
    nesting: str,
    component: str,
) -> None:
    fraction = Fraction(1, 2)
    coordinate = _nested_fraction_coordinate(fraction, nesting)
    object.__delattr__(fraction, f"_{component}")

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: coordinate,
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        (Decimal(0), np.int64(0)),
        (np.int64(0), Decimal(0)),
        (5e-324, np.float16(0.0)),
        (np.float16(0.0), 5e-324),
    ],
)
def test_extract_hyperparameter_results_rejects_incompatible_numeric_families(
    coordinates: tuple[object, object],
) -> None:
    """Cross-family aliases must fail before dict equality or hashing can narrow."""
    coordinate_iterator = iter(coordinates)
    results = {
        "candidate_0": _flat_trace("candidate_0", 1.0),
        "candidate_1": _flat_trace("candidate_1", 2.0),
    }

    with pytest.raises(ValueError, match=r"mutually compatible canonical coordinate families"):
        extract_hyperparameter_results(
            results,
            param_extractor=lambda _: next(coordinate_iterator),
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        (("value", Decimal(0)), ("value", np.int64(0))),
        (
            frozenset(("value", Decimal(0))),
            frozenset(("value", np.int64(0))),
        ),
    ],
)
def test_extract_hyperparameter_results_rejects_nested_incompatible_families(
    coordinates: tuple[object, object],
) -> None:
    coordinate_iterator = iter(coordinates)
    results = {
        "candidate_0": _flat_trace("candidate_0", 1.0),
        "candidate_1": _flat_trace("candidate_1", 2.0),
    }

    with pytest.raises(ValueError, match=r"mutually compatible canonical coordinate families"):
        extract_hyperparameter_results(
            results,
            param_extractor=lambda _: next(coordinate_iterator),
        )


def test_extract_hyperparameter_results_bounds_coordinate_nesting() -> None:
    within_limit: object = 0
    beyond_limit: object = 0
    for _ in range(32):
        within_limit = (within_limit,)
    for _ in range(33):
        beyond_limit = (beyond_limit,)

    extracted = extract_hyperparameter_results(
        {"candidate": _flat_trace("candidate", 1.0)},
        param_extractor=lambda _: within_limit,
    )
    assert next(iter(extracted)) is within_limit

    with pytest.raises(ValueError, match=r"at most 32 nested tuple/frozenset levels"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: beyond_limit,
        )


def test_extract_hyperparameter_results_keeps_coherent_python_numeric_families() -> None:
    coordinates = (1, 2.0, Decimal(3), Fraction(4, 1), complex(5, 0))
    coordinate_iterator = iter(coordinates)
    results = {
        f"candidate_{index}": _flat_trace(f"candidate_{index}", float(index))
        for index in range(len(coordinates))
    }

    extracted = extract_hyperparameter_results(
        results,
        param_extractor=lambda _: next(coordinate_iterator),
    )

    assert tuple(extracted) == coordinates
    assert all(
        actual is not expected if type(expected) is Fraction else actual is expected
        for actual, expected in zip(extracted, coordinates, strict=True)
    )


@pytest.mark.parametrize(
    "coordinates",
    [
        (Fraction(1, 2), 0.5),
        (0.5, Fraction(1, 2)),
        (("value", Fraction(1, 2)), ("value", Decimal("0.5"))),
        (("value", Decimal("0.5")), ("value", Fraction(1, 2))),
        (
            frozenset(("value", Fraction(1, 2))),
            frozenset(("value", Decimal("0.5"))),
        ),
        (
            frozenset(("value", Decimal("0.5"))),
            frozenset(("value", Fraction(1, 2))),
        ),
    ],
)
def test_extract_hyperparameter_results_preserves_fraction_numeric_collisions(
    coordinates: tuple[object, object],
) -> None:
    coordinate_iterator = iter(coordinates)
    results = {
        "candidate_0": _flat_trace("candidate_0", 1.0),
        "candidate_1": _flat_trace("candidate_1", 2.0),
    }

    with pytest.raises(ValueError, match=r"maps several configurations to one value"):
        extract_hyperparameter_results(
            results,
            param_extractor=lambda _: next(coordinate_iterator),
        )


class _DelayedHashDrift:
    def __init__(self) -> None:
        self.hash_calls = 0

    def __hash__(self) -> int:
        self.hash_calls += 1
        return 7 if self.hash_calls <= 2 else self.hash_calls

    def __eq__(self, other: object) -> bool:
        return self is other


def test_extract_hyperparameter_results_rejects_user_defined_coordinate_keys() -> None:
    coordinate = _DelayedHashDrift()

    with pytest.raises(ValueError, match=r"noncanonical coordinate"):
        extract_hyperparameter_results(
            {"candidate": _flat_trace("candidate", 1.0)},
            param_extractor=lambda _: coordinate,
        )

    assert coordinate.hash_calls == 0


def test_get_metric_timeseries_rejects_nonfinite_samples() -> None:
    poisoned = _two_seed_trace()
    poisoned.metric_arrays["squared_error"][0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite samples"):
        get_metric_timeseries(poisoned)


def test_get_final_performance_rejects_nonfinite_samples() -> None:
    poisoned = _two_seed_trace()
    poisoned.metric_arrays["squared_error"][1, -1] = np.nan
    with pytest.raises(ValueError, match="non-finite samples"):
        get_final_performance({"candidate": poisoned}, window=1)
