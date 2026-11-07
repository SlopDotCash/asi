"""Integer counters keep their paired differences in rank-test reports."""

import warnings

import numpy as np
import pytest
from scipy import stats

from alberta_framework.utils.statistics import (
    cohens_d,
    mann_whitney_comparison,
    wilcoxon_comparison,
)


@pytest.mark.parametrize("dtype,offset", [(np.int64, 2**60), (np.uint64, 2**63)])
def test_wilcoxon_integer_shift_retains_significance_and_effect(dtype, offset) -> None:
    baseline = np.full(8, offset, dtype=dtype)
    shifted = np.array([offset + k for k in range(1, 9)], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = wilcoxon_comparison(shifted, baseline)
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(2.0 / 2**8)
    assert result.significant
    assert result.effect_size == pytest.approx(4.5 / np.sqrt(3.0))
    assert cohens_d(shifted, baseline) == pytest.approx(result.effect_size)
    assert cohens_d(shifted[::-1], baseline) == result.effect_size


def test_wilcoxon_mixed_integer_pairs_are_not_rounded_to_identical() -> None:
    baseline = np.full(8, 2**63 - 1, dtype=np.int64)
    shifted = np.array([2**63 + k for k in range(1, 9)], dtype=np.uint64)
    result = wilcoxon_comparison(shifted, baseline)
    assert result.p_value == pytest.approx(2.0 / 2**8)
    assert result.significant


def test_wilcoxon_integer_rank_ties_and_zero_pairs_match_small_control() -> None:
    differences = [0, 1, 1, -2, 3, 0, -3, 4]
    offset = 2**60
    shifted = np.array([offset + k for k in differences], dtype=np.int64)
    baseline = np.full(len(differences), offset, dtype=np.int64)
    result = wilcoxon_comparison(shifted, baseline)
    control = stats.wilcoxon(np.array(differences, dtype=float))
    assert result.statistic == control.statistic
    assert result.p_value == pytest.approx(control.pvalue)


def test_wilcoxon_integer_identity_and_float_controls_stay_unchanged() -> None:
    with pytest.raises(ValueError, match="identical samples"):
        wilcoxon_comparison(np.array([2**60] * 8), np.array([2**60] * 8))
    floats = np.arange(1.0, 9.0)
    result = wilcoxon_comparison(floats, np.zeros(8))
    assert result.p_value == pytest.approx(2.0 / 2**8)
    assert result.effect_size == pytest.approx(4.5 / np.sqrt(3.0))


def test_mann_whitney_mixed_integer_order_is_preserved() -> None:
    high = np.array([2**63 + k for k in range(1, 9)], dtype=np.uint64)
    low = np.full(8, 2**63 - 1, dtype=np.int64)
    control = mann_whitney_comparison(np.arange(1, 9), np.zeros(8, dtype=np.int64))
    result = mann_whitney_comparison(high, low)
    assert result.statistic == control.statistic == 64.0
    assert result.p_value == pytest.approx(control.p_value)
    assert result.significant
    assert result.effect_size == 1.0
    tied = mann_whitney_comparison(
        np.full(8, 2**63 - 1, dtype=np.uint64), low
    )
    assert tied.p_value == 1.0
    assert tied.effect_size == 0.0


def test_integer_pooled_effect_preserves_constant_group_separation() -> None:
    baseline = np.full(4, 2**60, dtype=np.int64)
    shifted = np.full(4, 2**60 + 1, dtype=np.int64)
    assert cohens_d(shifted, baseline) == np.inf
    assert cohens_d(baseline, shifted) == -np.inf
    assert cohens_d(baseline, baseline) == 0.0
