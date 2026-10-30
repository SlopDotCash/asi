"""Paired inference must not wrap integer differences or square in float32."""

import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from alberta_framework.utils.statistics import ttest_comparison

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
