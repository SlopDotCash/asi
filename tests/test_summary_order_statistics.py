"""Unrelated outliers must not erase finite central order statistics."""

import statistics
import warnings

import numpy as np
import pytest

from alberta_framework.utils.statistics import compute_statistics

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_summary_preserves_tiny_median_beside_large_outlier(sign: float) -> None:
    values = [sign * 1e-200, sign * 1e-200, sign * 1e308]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = compute_statistics(values)
    assert summary.median == statistics.median(values)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_summary_preserves_tiny_interquartile_range(sign: float) -> None:
    values = np.concatenate((np.linspace(1e-200, 9e-200, 99), [1e308])) * sign
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = compute_statistics(values)
    assert summary.median != 0.0
    assert summary.median == pytest.approx(median, rel=1e-14, abs=0.0)
    assert summary.iqr > 0.0
    assert summary.iqr == pytest.approx(q75 - q25, rel=1e-14, abs=0.0)


def test_summary_large_constant_quantiles_stay_finite() -> None:
    maximum = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = compute_statistics([maximum] * 4)
    assert summary.median == maximum
    assert summary.iqr == 0.0
