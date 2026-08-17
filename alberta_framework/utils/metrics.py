"""Metrics and analysis utilities for continual learning experiments.

Provides functions for computing tracking error, learning curves,
and other metrics useful for evaluating continual learners.

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
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray


def _is_bool(value: object) -> bool:
    return type(value) is bool or type(value) is np.bool_


def _is_index_int(value: object) -> bool:
    return (type(value) is int or isinstance(value, np.integer)) and not _is_bool(value)


def _require_positive_builtin_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _require_finite_real(name: str, value: object) -> float:
    if _is_bool(value) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _reject_boolean_numeric_trace(values: object, *, name: str) -> None:
    if _is_bool(values):
        raise ValueError(f"{name} must not be a boolean")
    if isinstance(values, (list, tuple)):
        for index, item in enumerate(values):
            if _is_bool(item):
                raise ValueError(f"{name}[{index}] must not be a boolean")
            if isinstance(item, (list, tuple)):
                for inner_index, inner in enumerate(item):
                    if _is_bool(inner):
                        raise ValueError(f"{name}[{index}][{inner_index}] must not be a boolean")
    elif isinstance(values, np.ndarray) and values.dtype == np.bool_:
        raise ValueError(f"{name} must not be a boolean array")


def _require_index_vector(values: object, *, name: str) -> NDArray[np.int64]:
    if _is_bool(values):
        raise ValueError(f"{name} must be integer indices, not a boolean")
    if isinstance(values, (list, tuple)):
        for index, item in enumerate(values):
            if not _is_index_int(item):
                raise ValueError(f"{name}[{index}] must be an integer index, not a boolean")
    else:
        array = np.asarray(values)
        if array.dtype == np.bool_ or array.dtype.kind == "b":
            raise ValueError(f"{name} must be integer indices, not booleans")
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class StabilityGap:
    """Deficit below a reference performance during online learning."""

    mean: float
    maximum: float
    per_step: NDArray[np.float64]


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


def _performance_matrix(
    performance_matrix: NDArray[np.floating] | list[list[float]],
) -> NDArray[np.float64]:
    _reject_boolean_numeric_trace(performance_matrix, name="performance_matrix")
    matrix = np.asarray(performance_matrix, dtype=np.float64)
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

    matrix = _performance_matrix(performance_matrix)
    rows = _first_exposure_rows(
        first_exposure,
        n_checkpoints=matrix.shape[0],
        n_tasks=matrix.shape[1],
    )
    baseline = np.asarray(baseline_performance, dtype=np.float64)
    if baseline.shape != (matrix.shape[1],):
        raise ValueError("baseline_performance must contain one value per task")

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

    _reject_boolean_numeric_trace(online_performance, name="online_performance")
    values = np.asarray(online_performance, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("online_performance must be a non-empty one-dimensional trace")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("online_performance must contain at least one finite value")
    return float(np.mean(finite))


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

    _reject_boolean_numeric_trace(online_performance, name="online_performance")
    _reject_boolean_numeric_trace(reference_performance, name="reference_performance")
    if not isinstance(reference_performance, (list, tuple, np.ndarray)):
        _require_finite_real("reference_performance", reference_performance)
    values = np.asarray(online_performance, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("online_performance must be a non-empty one-dimensional trace")
    reference = np.asarray(reference_performance, dtype=np.float64)
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
    return StabilityGap(
        mean=float(np.nanmean(per_step)),
        maximum=float(np.nanmax(per_step)),
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

    _reject_boolean_numeric_trace(online_performance, name="online_performance")
    values = np.asarray(online_performance, dtype=np.float64)
    points = _require_index_vector(change_points, name="change_points")
    window_size = _require_positive_builtin_int("window_size", window_size)
    _require_finite_real("threshold", threshold)
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
                np.all(window >= threshold) if higher_is_better else np.all(window <= threshold)
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
        final_performance=float(np.mean(final)),
        prequential_performance=compute_prequential_performance(online_performance),
        mean_forgetting=float(np.mean(forgetting)),
        max_forgetting=float(np.max(forgetting)),
        backward_transfer=float(np.mean(backward_transfer)),
        stability_gap_mean=stability.mean,
        stability_gap_max=stability.maximum,
        per_task_final_performance=final,
        per_task_forgetting=forgetting,
        per_task_backward_transfer=backward_transfer,
    )


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
    errors = np.array([m[error_key] for m in metrics_history])
    return np.cumsum(errors)


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
    if type(window_size) is not int:
        raise ValueError("window_size must be a built-in int")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) < window_size:
        return np.full(values_arr.shape, np.nan, dtype=np.float64)

    cumsum = np.cumsum(np.insert(values_arr, 0, 0))
    running_mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

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
    errors = np.array([m["squared_error"] for m in metrics_history])
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
    return np.array([m[key] for m in metrics_history])


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
    summary = {}
    for name, metrics_history in results.items():
        values = extract_metric(metrics_history, metric)
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "cumulative": float(np.sum(values)),
            "final_100_mean": (
                float(np.mean(values[-100:])) if len(values) >= 100 else float(np.mean(values))
            ),
        }
    return summary
