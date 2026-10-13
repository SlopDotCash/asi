"""Paired inference must not wrap integer differences or square in float32."""

import warnings

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
