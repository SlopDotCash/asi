"""Paired inference must not wrap integer differences or square in float32."""

import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from alberta_framework.utils.statistics import (
    _paired_cohens_d,
    cohens_d,
    ttest_comparison,
    wilcoxon_comparison,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([100, 110, 120], dtype=np.int8),
         np.array([-100, -110, -125], dtype=np.int8)),
        (np.array([1, 2, 3], dtype=np.uint8),
         np.array([100, 105, 120], dtype=np.uint8)),
        (np.array([2, 4, 6], dtype=np.float32) * np.float32(1e20),
         np.array([1, 2, 3], dtype=np.float32) * np.float32(1e20)),
    ],
    ids=["signed-counts", "unsigned-counts", "large-float32"],
)
def test_paired_ttest_matches_float64_samples(a: np.ndarray, b: np.ndarray) -> None:
    expected = stats.ttest_rel(a.astype(np.float64), b.astype(np.float64))
    differences = a.astype(np.float64) - b.astype(np.float64)
    effect = np.mean(differences) / np.std(differences, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = ttest_comparison(a, b)
    assert actual.statistic == pytest.approx(expected.statistic)
    assert actual.p_value == expected.pvalue
    assert actual.effect_size == pytest.approx(effect)
    assert actual.significant == (expected.pvalue < actual.alpha)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_paired_ttest_retains_small_differences_at_large_integer_offsets(dtype: type) -> None:
    offset = 2**60
    a = np.array([offset + 1, offset + 4, offset + 7], dtype=dtype)
    b = np.array([offset, offset + 2, offset + 4], dtype=dtype)
    expected = stats.ttest_rel([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    actual = ttest_comparison(a, b)
    assert actual.statistic == pytest.approx(expected.statistic)
    assert actual.p_value == expected.pvalue
    assert actual.effect_size == pytest.approx(2.0)


def test_independent_ttest_preserves_finite_float32_variance() -> None:
    a = np.array([10, 11, 12, 13, 14], dtype=np.float32) * np.float32(1e20)
    b = np.array([0, 1, 2, 3, 4], dtype=np.float32) * np.float32(1e20)
    expected = stats.ttest_ind(a.astype(np.float64), b.astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = ttest_comparison(a, b, paired=False)
    assert actual.statistic == pytest.approx(expected.statistic)
    assert actual.p_value == expected.pvalue
    assert actual.significant
    assert actual.effect_size == pytest.approx(10.0 / np.sqrt(2.5), rel=1e-6)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integer_differences_retain_sub_float64_variation(dtype: type) -> None:
    if dtype is np.int64:
        a = np.array([2**62, 2**62 + 5, 2**62 + 9], dtype=dtype)
        b = np.array([-2**62, -2**62, -2**62 + 1], dtype=dtype)
    else:
        a = np.array([2**63, 2**63 + 5, 2**63 + 8], dtype=dtype)
        b = np.zeros(3, dtype=dtype)
    with localcontext() as context:
        context.prec = 80
        differences = [Decimal(int(x) - int(y)) for x, y in zip(a, b, strict=True)]
        mean = sum(differences) / Decimal(3)
        variance = sum((value - mean) ** 2 for value in differences) / Decimal(2)
        expected_effect = float(mean / variance.sqrt())
        expected_statistic = float(mean / (variance / Decimal(3)).sqrt())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = ttest_comparison(a, b)
    assert np.isfinite(actual.statistic)
    assert np.isfinite(actual.effect_size)
    assert actual.statistic == pytest.approx(expected_statistic, rel=1e-14)
    assert actual.effect_size == pytest.approx(expected_effect, rel=1e-14)
    assert actual.p_value == pytest.approx(
        2.0 * stats.t.sf(abs(expected_statistic), 2), rel=1e-14, abs=0.0
    )
    assert actual.significant


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_independent_ttest_retains_small_differences_at_large_integer_offsets(
    dtype: type,
) -> None:
    offset = 2**60
    a = np.array([offset + 10, offset + 11, offset + 12, offset + 13], dtype=dtype)
    b = np.array([offset, offset + 1, offset + 2, offset + 3], dtype=dtype)
    expected = stats.ttest_ind([10.0, 11.0, 12.0, 13.0], [0.0, 1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = ttest_comparison(a, b, paired=False)
    assert actual.statistic == pytest.approx(float(expected.statistic))
    assert actual.p_value == pytest.approx(float(expected.pvalue))
    assert actual.effect_size == pytest.approx(cohens_d(a, b))
    assert actual.significant


def test_independent_ttest_retains_sub_float64_variation_across_opposite_int64_extremes() -> None:
    a = np.array([2**62, 2**62 + 2, 2**62 + 4], dtype=np.int64)
    b = np.array([-(2**62), -(2**62) + 2, -(2**62) + 4], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = ttest_comparison(a, b, paired=False)
    assert np.isfinite(actual.statistic)
    assert np.isfinite(actual.effect_size)
    assert actual.effect_size == pytest.approx(cohens_d(a, b))
    assert actual.significant


def test_wilcoxon_comparison_promotes_float32_and_scales_overflowing_float64_pairs() -> None:
    a_f32 = np.array([2.0e38, 1.5e38, 1.0e38, 0.5e38, -0.2e38, 0.8e38], dtype=np.float32)
    b_f32 = np.array([-2.0e38, -1.5e38, -1.0e38, 0.0, 0.1e38, -0.4e38], dtype=np.float32)
    expected_f32 = wilcoxon_comparison(a_f32.astype(np.float64), b_f32.astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_f32 = wilcoxon_comparison(a_f32, b_f32)
    assert actual_f32.statistic == expected_f32.statistic
    assert actual_f32.p_value == pytest.approx(expected_f32.p_value)

    base_a = np.array([1.5, 1.2, 0.9, -0.4, 0.7, 1.1], dtype=np.float64)
    base_b = np.array([-1.5, -1.2, -0.9, 0.2, -0.5, -0.8], dtype=np.float64)
    expected_f64 = wilcoxon_comparison(base_a, base_b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_f64 = wilcoxon_comparison(base_a * 1.0e308, base_b * 1.0e308)
    assert actual_f64.statistic == expected_f64.statistic
    assert actual_f64.p_value == pytest.approx(expected_f64.p_value)


def test_wilcoxon_comparison_preserves_tiny_differences_alongside_overflow() -> None:
    # Single extreme overflowing pair plus 9 tiny positive pairs (all positive)
    a_pos = np.array([1.0e308] + [1.0e-20 * k for k in range(1, 10)], dtype=np.float64)
    b_pos = np.array([-1.0e308] + [0.0] * 9, dtype=np.float64)
    ref_pos = wilcoxon_comparison(
        np.array([10.0] + [float(k) for k in range(1, 10)]),
        np.zeros(10),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_pos = wilcoxon_comparison(a_pos, b_pos)
    assert actual_pos.statistic == ref_pos.statistic
    assert actual_pos.p_value == pytest.approx(ref_pos.p_value)
    assert actual_pos.significant is True

    # Mixed signs with overflowing pair and tiny nonzero differences
    a_mix = np.array(
        [1.0e308] + [(-1.0) ** k * 1.0e-20 * k for k in range(1, 10)],
        dtype=np.float64,
    )
    b_mix = np.array([-1.0e308] + [0.0] * 9, dtype=np.float64)
    ref_mix = wilcoxon_comparison(
        np.array([10.0] + [(-1.0) ** k * float(k) for k in range(1, 10)]),
        np.zeros(10),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_mix = wilcoxon_comparison(a_mix, b_mix)
    assert actual_mix.statistic == ref_mix.statistic
    assert actual_mix.p_value == pytest.approx(ref_mix.p_value)


def test_paired_cohens_d_is_scale_invariant_across_extreme_magnitudes() -> None:
    a = np.array([2.0, 4.0, 7.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    expected = _paired_cohens_d(a, b)
    for scale in (1.0e200, 1.0e-200, 1.0e307):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _paired_cohens_d(a * scale, b * scale) == pytest.approx(expected)

