"""Metrics and analysis utilities for continual learning experiments.

Provides functions for computing tracking error, learning curves,
and other metrics useful for evaluating continual learners. Nested
boolean-trace walks stop at ``_BOOLEAN_TRACE_MAX_DEPTH`` so a hostile nest
raises ``ValueError`` instead of ``RecursionError``. Host list, tuple, and
object-array width stops at ``_BOOLEAN_TRACE_MAX_NODES`` before the walk.

The task-matrix metrics in this module use a common convention: rows are
evaluation checkpoints and columns are tasks or regimes. ``first_exposure[j]``
is the row immediately after task ``j`` was first learned. Values before that
row may be finite (for forward-transfer probes) or ``NaN`` (not evaluated).

References:
    Lopez-Paz & Ranzato (2017). "Gradient Episodic Memory for Continual
        Learning." (backward/forward transfer)
    De Lange, van de Ven, & Tuytelaars (2023). "Continual Evaluation for
        Lifelong Learning: Identifying the Stability Gap."
"""

import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

_INT32_MAX: int = 2**31 - 1
# Bound recursive validation before numeric traces are materialized.
_BOOLEAN_TRACE_MAX_DEPTH: int = 32
_BOOLEAN_TRACE_MAX_NODES: int = 1_000_000
# Dense typed arrays do not consume the Python traversal budget, but they still
# feed allocation-heavy metric kernels.  ``compute_running_mean`` retains up to
# twelve float64/int64 vectors while constructing its result, so this separate
# ceiling keeps that complete envelope below 256 MiB without conflating array
# compatibility with recursive host-node admission.
_NUMERIC_TRACE_MAX_VALUES: int = 2_000_000

_ACTUAL_INT_TYPES: tuple[type, ...] = (int, *(np.dtype(code).type for code in "bBhHiIlLqQpP"))
_ACTUAL_FLOAT_TYPES = (
    float,
    Fraction,
    *(np.dtype(code).type for code in ("e", "f", "d", "g")),
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES + _ACTUAL_FLOAT_TYPES


def _type_is_one_of(actual_type: type[object], allowed: tuple[object, ...]) -> bool:
    """Use identity dispatch so hostile metaclass equality/hash hooks never run."""
    return any(actual_type is candidate for candidate in allowed)


def _is_bool(value: object) -> bool:
    actual_type = type(value)
    return actual_type is bool or actual_type is np.bool_


def _is_index_int(value: object) -> bool:
    return _type_is_one_of(type(value), _ACTUAL_INT_TYPES)


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_positive_builtin_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive built-in integer")
    number = operator.index(cast(SupportsIndex, value))
    if not 1 <= number <= _INT32_MAX:
        raise ValueError(f"{name} must be a positive built-in integer <= {_INT32_MAX}")
    return number


def _require_finite_real(name: str, value: object) -> float:
    if _is_bool(value):
        raise ValueError(f"{name} must be a finite real number")
    if not _type_is_one_of(type(value), _ALLOWED_REAL_TYPES):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(cast(float, value))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _reject_boolean_numeric_trace(
    values: object,
    *,
    name: str,
    depth: int = 0,
    _remaining_nodes: list[int] | None = None,
    _remaining_dense_values: list[int] | None = None,
) -> None:
    if _remaining_nodes is None:
        _remaining_nodes = [_BOOLEAN_TRACE_MAX_NODES]
    if _remaining_dense_values is None:
        _remaining_dense_values = [_NUMERIC_TRACE_MAX_VALUES]
    _remaining_nodes[0] -= 1
    if _remaining_nodes[0] < 0:
        raise ValueError(f"{name} exceeds the boolean-trace value limit")
    if depth > _BOOLEAN_TRACE_MAX_DEPTH:
        raise ValueError(f"{name} exceeds the boolean-trace depth limit")
    if _is_bool(values):
        raise ValueError(f"{name} must not be a boolean")
    actual_type = type(values)
    if _type_is_one_of(actual_type, _ALLOWED_REAL_TYPES):
        return
    if actual_type is list or actual_type is tuple:
        sequence = cast(list[object] | tuple[object, ...], values)
        if len(sequence) > _remaining_nodes[0]:
            raise ValueError(f"{name} exceeds the boolean-trace value limit")
        for index, item in enumerate(sequence):
            _reject_boolean_numeric_trace(
                item,
                name=f"{name}[{index}]",
                depth=depth + 1,
                _remaining_nodes=_remaining_nodes,
                _remaining_dense_values=_remaining_dense_values,
            )
        return
    if actual_type is np.ndarray:
        array = cast(NDArray[np.generic], values)
        if array.dtype.kind == "b":
            raise ValueError(f"{name} must not be a boolean array")
        if array.dtype.kind == "O":
            if array.size > _remaining_nodes[0]:
                raise ValueError(f"{name} exceeds the boolean-trace value limit")
            for index, item in enumerate(array.flat):
                _reject_boolean_numeric_trace(
                    item,
                    name=f"{name}[{index}]",
                    depth=depth + 1,
                    _remaining_nodes=_remaining_nodes,
                    _remaining_dense_values=_remaining_dense_values,
                )
        elif array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        else:
            if array.size > _remaining_dense_values[0]:
                raise ValueError(f"{name} exceeds the dense numeric value limit")
            _remaining_dense_values[0] -= int(array.size)
        return
    raise ValueError(f"{name} must contain exact real numeric values")


def _numeric_array(values: object, *, name: str) -> NDArray[np.float64]:
    _reject_boolean_numeric_trace(values, name=name)
    try:
        return np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain float64-compatible real values") from exc


def _record_vector(
    values: object,
    *,
    name: str,
    allow_nan: bool,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact NumPy array")
    array = np.array(_numeric_array(values, name=name), dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise ValueError(f"{name} contains unsupported non-finite values")
    if nonnegative and np.any(array[np.isfinite(array)] < 0.0):
        raise ValueError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _scale_safe_mean(values: NDArray[np.float64], *, name: str) -> float:
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    result = float(np.mean(values / scale) * scale)
    if not math.isfinite(result):
        raise ValueError(f"{name} mean must remain finite")
    return result


def _require_index_vector(values: object, *, name: str) -> NDArray[np.int64]:
    if _is_bool(values):
        raise ValueError(f"{name} must be integer indices, not a boolean")
    if type(values) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], values)
        if len(sequence) > _BOOLEAN_TRACE_MAX_NODES:
            raise ValueError(f"{name} exceeds the boolean-trace value limit")
        for index, item in enumerate(sequence):
            if not _is_index_int(item):
                raise ValueError(f"{name}[{index}] must be an integer index, not a boolean")
        return np.asarray(values, dtype=np.int64)
    elif type(values) is np.ndarray:
        array = values
        if array.size > _NUMERIC_TRACE_MAX_VALUES:
            raise ValueError(f"{name} exceeds the dense numeric value limit")
        if array.size == 0:
            return np.asarray(array, dtype=np.int64)
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer indices, not coerced values")
        if array.dtype.kind == "u" and bool(np.any(array > np.iinfo(np.int64).max)):
            raise ValueError(f"{name} contains an index outside the int64 domain")
        return np.asarray(array, dtype=np.int64)
    else:
        raise ValueError(f"{name} must be an exact list, tuple, or NumPy array")


@dataclass(frozen=True)
class StabilityGap:
    """Deficit below a reference performance during online learning."""

    mean: float
    maximum: float
    per_step: NDArray[np.float64]

    def __post_init__(self) -> None:
        mean = _require_finite_real("mean", self.mean)
        maximum = _require_finite_real("maximum", self.maximum)
        per_step = _record_vector(
            self.per_step, name="per_step", allow_nan=True, nonnegative=True
        )
        finite = per_step[np.isfinite(per_step)]
        if finite.size == 0:
            raise ValueError("per_step must contain a finite gap")
        if mean < 0.0 or maximum < 0.0:
            raise ValueError("stability gap summaries must be non-negative")
        if mean != _scale_safe_mean(finite, name="per_step") or maximum != float(
            np.max(finite)
        ):
            raise ValueError("stability gap summaries must match per_step")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "per_step", per_step)


@dataclass(frozen=True)
class ContinualLearningSummary:
    """Summary of a task-performance matrix and an online learning trace.

    Positive backward transfer means retained tasks improved after their first
    exposure. Forgetting and stability gaps are non-negative, where zero is
    best. ``final_performance`` and ``prequential_performance`` remain in the
    caller's original units.
    """

    final_performance: float
    prequential_performance: float
    mean_forgetting: float
    max_forgetting: float
    backward_transfer: float
    stability_gap_mean: float
    stability_gap_max: float
    per_task_final_performance: NDArray[np.float64]
    per_task_forgetting: NDArray[np.float64]
    per_task_backward_transfer: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "final_performance",
            "prequential_performance",
            "mean_forgetting",
            "max_forgetting",
            "backward_transfer",
            "stability_gap_mean",
            "stability_gap_max",
        ):
            object.__setattr__(self, name, _require_finite_real(name, getattr(self, name)))
        final = _record_vector(
            self.per_task_final_performance,
            name="per_task_final_performance",
            allow_nan=False,
        )
        forgetting = _record_vector(
            self.per_task_forgetting,
            name="per_task_forgetting",
            allow_nan=False,
            nonnegative=True,
        )
        transfer = _record_vector(
            self.per_task_backward_transfer,
            name="per_task_backward_transfer",
            allow_nan=False,
        )
        if not (final.shape == forgetting.shape == transfer.shape):
            raise ValueError("per-task summary arrays must have matching shapes")
        expected = {
            "final_performance": _scale_safe_mean(final, name="per_task_final_performance"),
            "mean_forgetting": _scale_safe_mean(forgetting, name="per_task_forgetting"),
            "max_forgetting": float(np.max(forgetting)),
            "backward_transfer": _scale_safe_mean(
                transfer, name="per_task_backward_transfer"
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must match its per-task record")
        if self.stability_gap_mean < 0.0 or self.stability_gap_max < self.stability_gap_mean:
            raise ValueError("stability gap summaries must be ordered and non-negative")
        object.__setattr__(self, "per_task_final_performance", final)
        object.__setattr__(self, "per_task_forgetting", forgetting)
        object.__setattr__(self, "per_task_backward_transfer", transfer)


def _performance_matrix(
    performance_matrix: NDArray[np.floating] | list[list[float]],
) -> NDArray[np.float64]:
    matrix = _numeric_array(performance_matrix, name="performance_matrix")
    if matrix.ndim != 2:
        raise ValueError("performance_matrix must have shape (checkpoints, tasks)")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("performance_matrix must be non-empty")
    return matrix


def _first_exposure_rows(
    first_exposure: NDArray[np.integer] | list[int],
    *,
    n_checkpoints: int,
    n_tasks: int,
) -> NDArray[np.int64]:
    rows = _require_index_vector(first_exposure, name="first_exposure")
    if rows.shape != (n_tasks,):
        raise ValueError("first_exposure must contain one row index per task")
    if np.any(rows < 0) or np.any(rows >= n_checkpoints):
        raise ValueError("first_exposure rows must index performance_matrix")
    return rows


def _task_trajectories(
    performance_matrix: NDArray[np.floating] | list[list[float]],
    first_exposure: NDArray[np.integer] | list[int],
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[NDArray[np.float64]]]:
    matrix = _performance_matrix(performance_matrix)
    rows = _first_exposure_rows(
        first_exposure,
        n_checkpoints=matrix.shape[0],
        n_tasks=matrix.shape[1],
    )
    trajectories: list[NDArray[np.float64]] = []
    for task, first_row in enumerate(rows):
        values = matrix[first_row:, task]
        if np.any(np.isinf(values)):
            raise ValueError(f"task {task} has an infinite evaluation at or after first exposure")
        if not np.isfinite(values[0]):
            raise ValueError(
                f"task {task} has no finite evaluation at first-post-exposure checkpoint"
            )
        if not np.isfinite(values[-1]):
            raise ValueError(f"task {task} has no finite final evaluation")
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"task {task} has no finite evaluation at or after first exposure")
        trajectories.append(finite_values)
    return matrix, rows, trajectories


def compute_per_task_forgetting(
    performance_matrix: NDArray[np.floating] | list[list[float]],
    first_exposure: NDArray[np.integer] | list[int],
    *,
    higher_is_better: bool = True,
) -> NDArray[np.float64]:
    """Return peak-to-final degradation for each previously learned task.

    This is an evaluation-only measure: callers should obtain the matrix from
    held-out probes that do not update the learner. For losses, set
    ``higher_is_better=False``.
    """

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    _, _, trajectories = _task_trajectories(performance_matrix, first_exposure)
    forgetting = []
    for values in trajectories:
        final = values[-1]
        best = np.max(values) if higher_is_better else np.min(values)
        degradation = best - final if higher_is_better else final - best
        forgetting.append(max(0.0, float(degradation)))
    return np.asarray(forgetting, dtype=np.float64)


def compute_backward_transfer(
    performance_matrix: NDArray[np.floating] | list[list[float]],
    first_exposure: NDArray[np.integer] | list[int],
    *,
    higher_is_better: bool = True,
) -> NDArray[np.float64]:
    """Return final minus first-post-exposure performance for each task.

    This is the BWT measure of Lopez-Paz & Ranzato (2017, GEM), generalized
    from their final-row convention to arbitrary checkpoint matrices.

    The sign is normalized so positive values always mean improvement. Unlike
    peak-to-final forgetting, backward transfer can be positive when later
    experience improves an earlier task.
    """

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    _, _, trajectories = _task_trajectories(performance_matrix, first_exposure)
    transfer = []
    for values in trajectories:
        first = values[0]
        final = values[-1]
        change = final - first if higher_is_better else first - final
        transfer.append(float(change))
    return np.asarray(transfer, dtype=np.float64)


def compute_forward_transfer(
    performance_matrix: NDArray[np.floating] | list[list[float]],
    first_exposure: NDArray[np.integer] | list[int],
    baseline_performance: NDArray[np.floating] | list[float],
    *,
    higher_is_better: bool = True,
) -> NDArray[np.float64]:
    """Return pre-exposure performance relative to a task baseline.

    This is the FWT measure of Lopez-Paz & Ranzato (2017, GEM): performance
    on a task probed *before* it is ever trained on, minus a task-specific
    baseline (in GEM, the untrained-network performance).

    Task 0, and any task without a finite pre-exposure probe, receives ``NaN``.
    The sign is normalized so positive values always mean beneficial transfer.
    """

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    matrix = _performance_matrix(performance_matrix)
    rows = _first_exposure_rows(
        first_exposure,
        n_checkpoints=matrix.shape[0],
        n_tasks=matrix.shape[1],
    )
    baseline = _numeric_array(baseline_performance, name="baseline_performance")
    if baseline.shape != (matrix.shape[1],):
        raise ValueError("baseline_performance must contain one value per task")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_performance must contain only finite values")

    transfer = np.full((matrix.shape[1],), np.nan, dtype=np.float64)
    for task, first_row in enumerate(rows):
        if first_row == 0:
            continue
        prior = matrix[:first_row, task]
        finite_prior = prior[np.isfinite(prior)]
        if finite_prior.size == 0:
            continue
        pre_exposure = finite_prior[-1]
        transfer[task] = (
            pre_exposure - baseline[task] if higher_is_better else baseline[task] - pre_exposure
        )
    return transfer


def compute_prequential_performance(
    online_performance: NDArray[np.floating] | list[float],
) -> float:
    """Return mean predict-before-update performance over an online trace."""

    values = _numeric_array(online_performance, name="online_performance")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("online_performance must be a non-empty one-dimensional trace")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("online_performance must contain at least one finite value")
    return _scale_safe_mean(finite, name="online_performance")


def compute_stability_gap(
    online_performance: NDArray[np.floating] | list[float],
    reference_performance: NDArray[np.floating] | list[float] | float,
    *,
    higher_is_better: bool = True,
) -> StabilityGap:
    """Measure transient online degradation relative to a reference trace.

    The stability gap (De Lange et al. 2023) is the dip that end-of-task
    evaluation cannot see: a learner may recover to its reference level by the
    next checkpoint while still having spent many online steps below it.

    The reference may be a scalar target or one value per online step. Gaps are
    clipped at zero, so exceeding the reference is not treated as instability.
    """

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    _reject_boolean_numeric_trace(online_performance, name="online_performance")
    _reject_boolean_numeric_trace(reference_performance, name="reference_performance")
    if not isinstance(reference_performance, (list, tuple, np.ndarray)):
        _require_finite_real("reference_performance", reference_performance)
    values = _numeric_array(online_performance, name="online_performance")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("online_performance must be a non-empty one-dimensional trace")
    reference = _numeric_array(reference_performance, name="reference_performance")
    try:
        broadcast_reference = np.broadcast_to(reference, values.shape)
    except ValueError as exc:
        raise ValueError(
            "reference_performance must be scalar or match online_performance"
        ) from exc

    valid = np.isfinite(values) & np.isfinite(broadcast_reference)
    if not np.any(valid):
        raise ValueError("stability-gap inputs must contain a finite aligned value")
    raw_gap = broadcast_reference - values if higher_is_better else values - broadcast_reference
    per_step = np.where(valid, np.maximum(raw_gap, 0.0), np.nan)
    finite_gaps = per_step[np.isfinite(per_step)]
    return StabilityGap(
        mean=_scale_safe_mean(finite_gaps, name="stability gaps"),
        maximum=float(np.max(finite_gaps)),
        per_step=np.asarray(per_step, dtype=np.float64),
    )


def compute_recovery_lengths(
    online_performance: NDArray[np.floating] | list[float],
    change_points: NDArray[np.integer] | list[int],
    threshold: float,
    *,
    window_size: int = 1,
    higher_is_better: bool = True,
) -> NDArray[np.int64]:
    """Return steps from each change point to sustained threshold recovery.

    A recovery occurs only after the first full window whose every finite value
    meets the threshold.  The returned length therefore counts observations
    through the end of that qualifying window; an immediately successful
    window has length ``window_size``, not zero. ``-1`` means the trace never
    recovered before the next change point (or the end of the stream).
    """

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    values = _numeric_array(online_performance, name="online_performance")
    points = _require_index_vector(change_points, name="change_points")
    window_size = _require_positive_builtin_int("window_size", window_size)
    threshold_value = _require_finite_real("threshold", threshold)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("online_performance must be a non-empty one-dimensional trace")
    if points.ndim != 1 or points.size == 0:
        raise ValueError("change_points must be a non-empty one-dimensional array")
    if np.any(points < 0) or np.any(points >= values.size):
        raise ValueError("change_points must index online_performance")
    if np.any(np.diff(points) <= 0):
        raise ValueError("change_points must be strictly increasing")

    recoveries = np.full(points.shape, -1, dtype=np.int64)
    for point_index, start in enumerate(points):
        stop = points[point_index + 1] if point_index + 1 < points.size else values.size
        for window_start in range(int(start), int(stop) - window_size + 1):
            window = values[window_start : window_start + window_size]
            meets = np.all(np.isfinite(window)) and (
                np.all(window >= threshold_value)
                if higher_is_better
                else np.all(window <= threshold_value)
            )
            if meets:
                recoveries[point_index] = (
                    window_start + window_size - int(start)
                )
                break
    return recoveries


def summarize_continual_learning(
    performance_matrix: NDArray[np.floating] | list[list[float]],
    first_exposure: NDArray[np.integer] | list[int],
    online_performance: NDArray[np.floating] | list[float],
    reference_performance: NDArray[np.floating] | list[float] | float,
    *,
    higher_is_better: bool = True,
) -> ContinualLearningSummary:
    """Compute retention, transfer, prequential, and stability metrics."""

    higher_is_better = _require_exact_bool("higher_is_better", higher_is_better)
    matrix, _, trajectories = _task_trajectories(
        performance_matrix,
        first_exposure,
    )
    final = np.asarray([values[-1] for values in trajectories], dtype=np.float64)
    forgetting = compute_per_task_forgetting(
        matrix,
        first_exposure,
        higher_is_better=higher_is_better,
    )
    backward_transfer = compute_backward_transfer(
        matrix,
        first_exposure,
        higher_is_better=higher_is_better,
    )
    stability = compute_stability_gap(
        online_performance,
        reference_performance,
        higher_is_better=higher_is_better,
    )
    return ContinualLearningSummary(
        final_performance=_scale_safe_mean(final, name="per-task final performance"),
        prequential_performance=compute_prequential_performance(online_performance),
        mean_forgetting=_scale_safe_mean(forgetting, name="per-task forgetting"),
        max_forgetting=float(np.max(forgetting)),
        backward_transfer=_scale_safe_mean(
            backward_transfer, name="per-task backward transfer"
        ),
        stability_gap_mean=stability.mean,
        stability_gap_max=stability.maximum,
        per_task_final_performance=final,
        per_task_forgetting=forgetting,
        per_task_backward_transfer=backward_transfer,
    )


def _metric_history_values(
    metrics_history: object,
    key: str,
    *,
    name: str,
    _remaining_items: list[int] | None = None,
) -> NDArray[np.float64]:
    if type(key) is not str:
        raise ValueError(f"{name} metric key must be an exact string")
    if type(metrics_history) is not list:
        raise ValueError(f"{name} must be an exact list")
    history = cast(list[object], metrics_history)
    remaining_items = (
        [_BOOLEAN_TRACE_MAX_NODES] if _remaining_items is None else _remaining_items
    )
    if len(history) > remaining_items[0]:
        raise ValueError(f"{name} exceeds the boolean-trace value limit")
    remaining_items[0] -= len(history)
    values: list[float] = []
    for index, record in enumerate(history):
        if type(record) is not dict:
            raise ValueError(f"{name}[{index}] must be an exact dictionary")
        if len(record) > 4_096:
            raise ValueError(f"{name}[{index}] exceeds the metric-record key limit")
        if len(record) > remaining_items[0]:
            raise ValueError(f"{name} exceeds the aggregate metric-record item limit")
        remaining_items[0] -= len(record)
        if any(type(record_key) is not str for record_key in record):
            raise ValueError(f"{name}[{index}] keys must be exact strings")
        if key not in record:
            raise ValueError(f"{name}[{index}] is missing metric {key}")
        values.append(_require_finite_real(f"{name}[{index}].{key}", record[key]))
    return np.asarray(values, dtype=np.float64)


def compute_cumulative_error(
    metrics_history: list[dict[str, float]],
    error_key: str = "squared_error",
) -> NDArray[np.float64]:
    """Compute cumulative error over time.

    Cumulative error is the online-learning (regret-style) view of a run: its
    slope at time ``t`` is the instantaneous error, so a persistent slope
    difference between learners is sustained tracking advantage, while a
    one-time vertical offset is a transient.

    Args:
        metrics_history: List of metric dictionaries from learning loop
        error_key: Key to extract error values

    Returns:
        Array of cumulative errors at each time step
    """
    _require_exact_str("error_key", error_key)
    errors = _metric_history_values(metrics_history, error_key, name="metrics_history")
    cumulative = np.cumsum(errors, dtype=np.float64)
    if np.any(~np.isfinite(cumulative)):
        raise ValueError("cumulative error must remain finite")
    return cumulative


def compute_running_mean(
    values: NDArray[np.float64] | list[float],
    window_size: int = 100,
) -> NDArray[np.float64]:
    """Compute the trailing (causal) running mean of values.

    The output has the same length as the input. Position ``i`` holds the
    mean of ``values[i - window_size + 1 : i + 1]`` -- the window ending at
    and including step ``i`` -- so no returned value is ever computed from
    observations that had not yet occurred at that step. The first
    ``window_size - 1`` entries therefore have no complete trailing window
    and are ``NaN`` (not yet computable; see the module docstring's
    NaN-for-"not evaluated" convention), rather than being backdated with a
    later window's mean. If the input is shorter than ``window_size``, no
    window is ever complete and every returned position is ``NaN``.

    Args:
        values: Array of values
        window_size: Size of the moving average window

    Returns:
        Array of running mean values (same length as input), with ``NaN``
        wherever a complete trailing window is not yet available.
    """
    window_size = _require_positive_builtin_int("window_size", window_size)

    values_arr = _numeric_array(values, name="values")
    if values_arr.ndim != 1:
        raise ValueError("values must be a one-dimensional trace")
    if np.any(np.isinf(values_arr)):
        raise ValueError("values must not contain infinity")
    if len(values_arr) < window_size:
        return np.full(values_arr.shape, np.nan, dtype=np.float64)

    finite_values = values_arr[np.isfinite(values_arr)]
    scale = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if scale == 0.0:
        scale = 1.0
    normalized = np.where(np.isfinite(values_arr), values_arr / scale, 0.0)
    cumsum = np.cumsum(np.insert(normalized, 0, 0.0), dtype=np.float64)
    valid = np.isfinite(values_arr).astype(np.int64)
    valid_cumsum = np.cumsum(np.insert(valid, 0, 0), dtype=np.int64)
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    valid_counts = valid_cumsum[window_size:] - valid_cumsum[:-window_size]
    running_mean = (window_sums / window_size) * scale
    running_mean = np.where(valid_counts == window_size, running_mean, np.nan)
    if np.any(np.isinf(running_mean)):
        raise ValueError("running mean must remain finite")

    # The first `window_size - 1` steps have no complete trailing window yet;
    # mark them NaN instead of backdating the first complete window's mean.
    if len(running_mean) > 0:
        padding = np.full(window_size - 1, np.nan)
        return np.concatenate([padding, running_mean])
    return np.full(values_arr.shape, np.nan, dtype=np.float64)


def compute_tracking_error(
    metrics_history: list[dict[str, float]],
    window_size: int = 100,
) -> NDArray[np.float64]:
    """Compute tracking error (trailing running mean of squared error).

    This is the key metric for evaluating continual learners:
    how well can the learner track the non-stationary target? See
    ``compute_running_mean`` for the causal windowing contract: the first
    ``window_size - 1`` entries are ``NaN`` (no complete trailing window
    yet), not a value borrowed from later time steps.

    Args:
        metrics_history: List of metric dictionaries from learning loop
        window_size: Size of the moving average window

    Returns:
        Array of tracking errors at each time step
    """
    window_size = _require_positive_builtin_int("window_size", window_size)
    errors = _metric_history_values(metrics_history, "squared_error", name="metrics_history")
    return compute_running_mean(errors, window_size)


def extract_metric(
    metrics_history: list[dict[str, float]],
    key: str,
) -> NDArray[np.float64]:
    """Extract a single metric from the history.

    Args:
        metrics_history: List of metric dictionaries
        key: Key to extract

    Returns:
        Array of values for that metric
    """
    _require_exact_str("key", key)
    return _metric_history_values(metrics_history, key, name="metrics_history")


def compare_learners(
    results: dict[str, list[dict[str, float]]],
    metric: str = "squared_error",
) -> dict[str, dict[str, float]]:
    """Compare multiple learners on a given metric.

    ``final_100_mean`` averages the last 100 steps (the whole trace when it is
    shorter), estimating settled performance less noisily than the last value.
    ``std`` is the population standard deviation over the complete recorded
    time-step trajectory; it is not a spread estimate across independent seeds.

    Args:
        results: Dictionary mapping learner name to metrics history
        metric: Metric to compare

    Returns:
        Dictionary with summary statistics for each learner
    """
    _require_exact_str("metric", metric)
    if type(results) is not dict:
        raise ValueError("results must be an exact dictionary")
    if not results:
        raise ValueError("results must contain at least one learner")
    if len(results) > _BOOLEAN_TRACE_MAX_NODES:
        raise ValueError("results exceeds the learner traversal limit")
    summary = {}
    remaining_items = [_BOOLEAN_TRACE_MAX_NODES - len(results)]
    for name, metrics_history in results.items():
        _require_exact_str("learner name", name)
        values = _metric_history_values(
            metrics_history,
            metric,
            name="metrics_history",
            _remaining_items=remaining_items,
        )
        if values.size == 0:
            raise ValueError(f"learner {name} must contain at least one metric record")
        scale = float(np.max(np.abs(values)))
        normalized = values if scale == 0.0 else values / scale
        mean = _scale_safe_mean(values, name=f"learner {name}")
        std = float(np.std(normalized) * (1.0 if scale == 0.0 else scale))
        cumulative = float(np.sum(values, dtype=np.float64))
        tail = values[-100:] if len(values) >= 100 else values
        final_100_mean = _scale_safe_mean(tail, name=f"learner {name} tail")
        if not math.isfinite(std) or not math.isfinite(cumulative):
            raise ValueError(f"learner {name} summaries must remain finite")
        summary[name] = {
            "mean": mean,
            "std": std,
            "cumulative": cumulative,
            "final_100_mean": final_100_mean,
        }
    return summary
