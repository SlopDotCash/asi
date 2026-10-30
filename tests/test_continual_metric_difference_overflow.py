"""Finite input traces cannot publish infinite derived metric differences."""

import warnings

import numpy as np
import pytest

from alberta_framework.utils.metrics import (
    compute_backward_transfer,
    compute_forward_transfer,
    compute_per_task_forgetting,
    compute_stability_gap,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("higher_is_better", [False, True])
@pytest.mark.parametrize("metric", ["forgetting", "backward", "forward"])
def test_unrepresentable_task_metric_is_rejected(
    metric: str, higher_is_better: bool
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="must remain finite"):
            if metric == "forgetting":
                compute_per_task_forgetting(
                    [[1e308], [-1e308]] if higher_is_better else [[-1e308], [1e308]],
                    [0], higher_is_better=higher_is_better,
                )
            elif metric == "backward":
                compute_backward_transfer(
                    [[-1e308], [1e308]], [0], higher_is_better=higher_is_better
                )
            else:
                compute_forward_transfer(
                    [[1e308], [1e308]], [1], [-1e308], higher_is_better=higher_is_better
                )


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_beneficial_extreme_performance_has_zero_stability_gap(higher_is_better: bool) -> None:
    sign = 1.0 if higher_is_better else -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gap = compute_stability_gap([sign * 1e308], [-sign * 1e308],
                                    higher_is_better=higher_is_better)
    assert gap.mean == gap.maximum == 0.0
    np.testing.assert_array_equal(gap.per_step, [0.0])


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_unrepresentable_stability_gap_is_rejected(higher_is_better: bool) -> None:
    sign = 1.0 if higher_is_better else -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="must remain finite"):
            compute_stability_gap([-sign * 1e308], [sign * 1e308],
                                  higher_is_better=higher_is_better)


def test_forward_transfer_preserves_missing_checkpoint_nan() -> None:
    result = compute_forward_transfer([[np.nan], [0.5]], [1], [0.1])
    assert np.isnan(result[0])
