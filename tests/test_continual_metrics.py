"""Tests for continual-learning retention, transfer, and recovery metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alberta_framework.utils.metrics import (
    ContinualLearningSummary,
    StabilityGap,
    compare_learners,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_per_task_forgetting,
    compute_prequential_performance,
    compute_recovery_lengths,
    compute_running_mean,
    compute_stability_gap,
    compute_tracking_error,
    summarize_continual_learning,
)


def test_compare_learners_uses_population_spread_over_recorded_steps() -> None:
    summary = compare_learners(
        {
            "learner": [
                {"squared_error": 0.0},
                {"squared_error": 2.0},
            ]
        }
    )

    assert summary["learner"]["std"] == pytest.approx(1.0)


def test_accuracy_matrix_forgetting_and_backward_transfer() -> None:
    performance = np.array(
        [
            [0.80, 0.55, np.nan],
            [0.60, 0.70, 0.45],
            [0.75, 0.80, 0.90],
        ]
    )
    first_exposure = [0, 1, 2]

    forgetting = compute_per_task_forgetting(performance, first_exposure)
    transfer = compute_backward_transfer(performance, first_exposure)
    forward = compute_forward_transfer(
        performance,
        first_exposure,
        baseline_performance=[0.50, 0.50, 0.50],
    )

    np.testing.assert_allclose(forgetting, [0.05, 0.0, 0.0])
    np.testing.assert_allclose(transfer, [-0.05, 0.10, 0.0])
    assert np.isnan(forward[0])
    np.testing.assert_allclose(forward[1:], [0.05, -0.05])


def test_loss_matrix_normalizes_metric_directions() -> None:
    losses = np.array(
        [
            [0.20, np.nan],
            [0.50, 0.40],
            [0.25, 0.30],
        ]
    )

    forgetting = compute_per_task_forgetting(
        losses,
        [0, 1],
        higher_is_better=False,
    )
    transfer = compute_backward_transfer(
        losses,
        [0, 1],
        higher_is_better=False,
    )

    np.testing.assert_allclose(forgetting, [0.05, 0.0])
    np.testing.assert_allclose(transfer, [-0.05, 0.10])


def test_stability_gap_and_prequential_performance_ignore_nan_probes() -> None:
    online = np.array([0.9, 0.4, np.nan, 0.8, 1.0])
    gap = compute_stability_gap(online, 0.8)

    assert compute_prequential_performance(online) == pytest.approx(0.775)
    assert gap.mean == pytest.approx(0.1)
    assert gap.maximum == pytest.approx(0.4)
    np.testing.assert_allclose(
        gap.per_step,
        [0.0, 0.4, np.nan, 0.0, 0.0],
        equal_nan=True,
    )


def test_recovery_lengths_are_bounded_by_next_change() -> None:
    online = [0.9, 0.2, 0.4, 0.8, 0.9, 0.1, 0.7, 0.85, 0.86]

    recovery = compute_recovery_lengths(
        online,
        change_points=[1, 5],
        threshold=0.8,
        window_size=2,
    )

    np.testing.assert_array_equal(recovery, [4, 4])


def test_recovery_counts_the_full_sustained_window() -> None:
    recovery = compute_recovery_lengths(
        [0.9, 0.9, 0.9],
        change_points=[0],
        threshold=0.8,
        window_size=3,
    )

    np.testing.assert_array_equal(recovery, [3])


def test_recovery_reports_minus_one_when_threshold_is_not_reached() -> None:
    recovery = compute_recovery_lengths(
        [0.2, 0.3, 0.4],
        change_points=[0],
        threshold=0.8,
        window_size=2,
    )

    np.testing.assert_array_equal(recovery, [-1])


def test_summary_preserves_direct_evidence_arrays() -> None:
    performance = np.array(
        [
            [0.80, np.nan],
            [0.60, 0.70],
            [0.75, 0.80],
        ]
    )
    online = [0.8, 0.6, 0.9]

    summary = summarize_continual_learning(
        performance,
        first_exposure=[0, 1],
        online_performance=online,
        reference_performance=0.8,
    )

    assert summary.final_performance == pytest.approx(0.775)
    assert summary.prequential_performance == pytest.approx(0.7666666667)
    assert summary.mean_forgetting == pytest.approx(0.025)
    assert summary.max_forgetting == pytest.approx(0.05)
    assert summary.backward_transfer == pytest.approx(0.025)
    assert summary.stability_gap_mean == pytest.approx(0.0666666667)
    assert summary.stability_gap_max == pytest.approx(0.2)
    np.testing.assert_allclose(summary.per_task_final_performance, [0.75, 0.80])


@pytest.mark.parametrize(
    ("matrix", "first_exposure", "message"),
    [
        ([0.1, 0.2], [0], "shape"),
        ([[0.1, 0.2]], [0], "one row index per task"),
        ([[0.1, 0.2]], [0, 1], "must index"),
        ([[0.1, np.nan]], [0, 0], "no finite evaluation"),
    ],
)
def test_task_matrix_validation(
    matrix: list[float] | list[list[float]],
    first_exposure: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_per_task_forgetting(matrix, first_exposure)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[np.nan], [0.6]], "first-post-exposure"),
        ([[0.8], [np.nan]], "final evaluation"),
    ],
)
def test_task_metrics_reject_missing_boundary_evaluations(
    matrix: list[list[float]],
    message: str,
) -> None:
    """Do not silently backfill required first/final checkpoints."""

    with pytest.raises(ValueError, match=message):
        compute_per_task_forgetting(matrix, [0])


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_task_metrics_reject_infinite_post_exposure_evaluations(value: float) -> None:
    """Do not treat a divergent metric as an unevaluated probe gap."""

    with pytest.raises(ValueError, match="infinite evaluation"):
        compute_per_task_forgetting([[0.8], [value], [0.6]], [0])


def test_running_mean_does_not_backdate_a_future_informed_value() -> None:
    """The leading positions must not be filled with a later window's mean.

    Before this fix, ``compute_running_mean`` padded the first
    ``window_size - 1`` entries with the mean of the *first complete*
    trailing window -- a value that depends on observations from steps that
    had not yet occurred at those earlier positions. #175/#176 documented
    and worked around exactly this defect at one call site
    (``plot_learning_curves``) without fixing the underlying function, so
    every other caller of the public, top-level-exported
    ``compute_running_mean``/``compute_tracking_error`` still received the
    corrupted, future-informed trace directly.
    """

    result = compute_running_mean([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)

    assert result.shape == (6,)
    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:], [1.0, 2.0, 3.0, 4.0])


def test_running_mean_exact_window_length_has_one_valid_entry() -> None:
    result = compute_running_mean([1.0, 2.0, 3.0], window_size=3)

    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:], [2.0])


def test_running_mean_shorter_than_window_has_no_computable_values() -> None:
    result = compute_running_mean([2, 4], window_size=3)

    assert result.shape == (2,)
    assert result.dtype == np.float64
    assert np.all(np.isnan(result))


def test_tracking_error_inherits_the_causal_running_mean_fix() -> None:
    history = [{"squared_error": float(v)} for v in range(6)]

    result = compute_tracking_error(history, window_size=3)

    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:], [1.0, 2.0, 3.0, 4.0])


def test_tracking_error_does_not_erase_small_windows_after_large_error() -> None:
    """A departed large error must not cancel later finite window sums."""
    history = [
        {"squared_error": value}
        for value in (1e16, 1.0, 1.0, 1.0)
    ]

    result = compute_tracking_error(history, window_size=2)

    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:], [5e15, 1.0, 1.0])


def test_tracking_error_shorter_than_window_has_no_computable_values() -> None:
    result = compute_tracking_error(
        [{"squared_error": 2.0}, {"squared_error": 4.0}],
        window_size=3,
    )

    assert result.shape == (2,)
    assert np.all(np.isnan(result))


@pytest.mark.parametrize("window_size", [True, False, np.int64(3), 3.0, 0, -1])
def test_running_mean_rejects_invalid_window_size(window_size: object) -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0, 2.0, 3.0], window_size=window_size)  # type: ignore[arg-type]


def test_first_exposure_true_is_not_checkpoint_row_one() -> None:
    """Boolean True is a subclass of int, so asarray(..., int64) stored row 1.

    On a one-task matrix that hides 0.05 forgetting when the legal first
    exposure is row 0. The exposure index is the task-identity axis for
    forgetting and backward transfer.
    """

    performance = np.array([[0.80], [0.60], [0.75]])

    with pytest.raises(ValueError, match="first_exposure"):
        compute_per_task_forgetting(performance, [True])

    np.testing.assert_allclose(compute_per_task_forgetting(performance, [0]), [0.05])
    np.testing.assert_allclose(compute_per_task_forgetting(performance, [1]), [0.0])


@pytest.mark.parametrize("window_size", [True, False, np.int64(2), 2.0, 0, -1])
def test_recovery_rejects_non_canonical_window_size(window_size: object) -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_recovery_lengths(
            [0.9, 0.2, 0.8, 0.9],
            change_points=[1],
            threshold=0.8,
            window_size=window_size,  # type: ignore[arg-type]
        )


def test_recovery_rejects_boolean_change_point() -> None:
    """change_points=[True] used to start at index 1 instead of failing."""

    online = [0.1, 0.9, 0.9]
    with pytest.raises(ValueError, match="change_points"):
        compute_recovery_lengths(online, change_points=[True], threshold=0.8, window_size=1)

    np.testing.assert_array_equal(
        compute_recovery_lengths(online, change_points=[0], threshold=0.8, window_size=1),
        [2],
    )
    np.testing.assert_array_equal(
        compute_recovery_lengths(online, change_points=[1], threshold=0.8, window_size=1),
        [1],
    )


@pytest.mark.parametrize(
    "change_points",
    [
        np.array([1.0]),
        np.array([1.5]),
        np.array([np.nan]),
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
    ],
)
def test_recovery_rejects_coerced_numpy_change_points(change_points: object) -> None:
    with pytest.raises(ValueError, match="change_points"):
        compute_recovery_lengths(
            [0.1, 0.9, 0.9],
            change_points=change_points,  # type: ignore[arg-type]
            threshold=0.8,
            window_size=1,
        )


def test_nested_numpy_boolean_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="online_performance"):
        compute_prequential_performance([np.array(True), np.array(False)])


@pytest.mark.parametrize("threshold", [True, False, float("nan"), float("inf"), "0.8"])
def test_recovery_rejects_boolean_or_nonfinite_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_recovery_lengths(
            [0.0, 1.0, 1.0],
            change_points=[0],
            threshold=threshold,  # type: ignore[arg-type]
            window_size=1,
        )


def test_stability_and_prequential_reject_boolean_identities() -> None:
    with pytest.raises(ValueError, match="reference_performance"):
        compute_stability_gap([0.0, 1.0, 0.5], True)
    with pytest.raises(ValueError, match="online_performance"):
        compute_prequential_performance([True, False])

    gap = compute_stability_gap([0.0, 1.0, 0.5], 1.0)
    np.testing.assert_allclose(gap.per_step, [1.0, 0.0, 0.5])
    assert compute_prequential_performance([0.0, 1.0]) == pytest.approx(0.5)


def test_stability_gap_rejects_leftover_identities() -> None:
    """Public gap records must not keep leftover bool/NaN identities."""

    with pytest.raises(ValueError, match="mean"):
        StabilityGap(mean=True, maximum=0.0, per_step=np.array([0.0]))
    with pytest.raises(ValueError, match="mean"):
        StabilityGap(mean=float("nan"), maximum=0.0, per_step=np.array([0.0]))
    with pytest.raises(ValueError, match="maximum"):
        StabilityGap(mean=0.0, maximum=float("inf"), per_step=np.array([0.0]))

    legal = StabilityGap(mean=0.1, maximum=0.2, per_step=np.array([0.0, 0.2]))
    dumped = json.dumps({"mean": legal.mean, "maximum": legal.maximum}, allow_nan=False)
    assert '"mean": 0.1' in dumped
    assert '"mean": true' not in dumped


def _legal_continual_summary(**overrides: object) -> ContinualLearningSummary:
    payload: dict[str, object] = {
        "final_performance": 0.8,
        "prequential_performance": 0.7,
        "mean_forgetting": 0.1,
        "max_forgetting": 0.1,
        "backward_transfer": 0.0,
        "stability_gap_mean": 0.05,
        "stability_gap_max": 0.1,
        "per_task_final_performance": np.array([0.8]),
        "per_task_forgetting": np.array([0.1]),
        "per_task_backward_transfer": np.array([0.0]),
    }
    payload.update(overrides)
    return ContinualLearningSummary(**payload)  # type: ignore[arg-type]


def test_continual_learning_summary_rejects_leftover_identities() -> None:
    with pytest.raises(ValueError, match="final_performance"):
        _legal_continual_summary(final_performance=True)
    with pytest.raises(ValueError, match="mean_forgetting"):
        _legal_continual_summary(mean_forgetting=float("nan"))
    with pytest.raises(ValueError, match="stability_gap_max"):
        _legal_continual_summary(stability_gap_max=float("inf"))

    legal = _legal_continual_summary()
    dumped = json.dumps(
        {"final_performance": legal.final_performance},
        allow_nan=False,
    )
    assert '"final_performance": 0.8' in dumped
    assert '"final_performance": true' not in dumped


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_prequential_performance_rejects_infinite_trace(value: float) -> None:
    with pytest.raises(ValueError, match="online_performance must not contain infinity"):
        compute_prequential_performance([0.5, value, 0.7])


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_stability_gap_rejects_infinite_online_trace(value: float) -> None:
    with pytest.raises(ValueError, match="online_performance must not contain infinity"):
        compute_stability_gap([0.5, value, 0.7], 0.8)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_stability_gap_rejects_infinite_reference_trace(value: float) -> None:
    with pytest.raises(ValueError, match="reference_performance must not contain infinity"):
        compute_stability_gap([0.5, 0.6, 0.7], [0.8, value, 0.8])


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_recovery_lengths_reject_infinite_online_trace(value: float) -> None:
    with pytest.raises(ValueError, match="online_performance must not contain infinity"):
        compute_recovery_lengths([0.1, value, 0.9], change_points=[0], threshold=0.8, window_size=1)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_forward_transfer_rejects_infinite_pre_exposure(value: float) -> None:
    with pytest.raises(ValueError, match="infinite evaluation"):
        compute_forward_transfer(
            [[value, np.nan], [0.5, 0.6]],
            first_exposure=[1, 1],
            baseline_performance=[0.5, 0.5],
        )


def test_tracking_error_preserves_small_windows_after_long_high_error_phase() -> None:
    history = [{"squared_error": value} for value in ([1e15] * 1000 + [1.0] * 4)]
    result = compute_tracking_error(history, window_size=2)
    np.testing.assert_array_equal(result[-3:], [1.0, 1.0, 1.0])
