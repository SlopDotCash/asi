"""Accepted confidence scalars must reach NumPy/SciPy in one host domain."""

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.utils.statistics import (
    bootstrap_ci,
    compute_statistics,
    compute_timeseries_statistics,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("confidence", [Fraction(19, 20), np.float32(0.95), np.float64(0.95)])
def test_summary_confidence_matches_same_host_float(confidence: object) -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    actual = compute_statistics(values, confidence_level=confidence)  # type: ignore[arg-type]
    expected = compute_statistics(values, confidence_level=float(confidence))  # type: ignore[arg-type]
    assert actual == expected


@pytest.mark.parametrize("confidence", [Fraction(19, 20), np.float32(0.95)])
def test_timeseries_confidence_matches_same_host_float(confidence: object) -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    actual = compute_timeseries_statistics(values, confidence_level=confidence)  # type: ignore[arg-type]
    expected = compute_timeseries_statistics(values, confidence_level=float(confidence))  # type: ignore[arg-type]
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("confidence", [Fraction(19, 20), np.float32(0.95)])
def test_bootstrap_confidence_matches_same_host_float(confidence: object) -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    actual = bootstrap_ci(values, confidence_level=confidence, n_bootstrap=20)  # type: ignore[arg-type]
    expected = bootstrap_ci(values, confidence_level=float(confidence), n_bootstrap=20)  # type: ignore[arg-type]
    assert actual == expected
