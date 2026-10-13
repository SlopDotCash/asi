"""Effect sizes must not change when metric units change."""

import warnings

import numpy as np
import pytest

from alberta_framework.utils.statistics import cohens_d, wilcoxon_comparison

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scale", [1.0, 1.0e200, 1.0e-200, 1.0e307])
def test_cohens_d_preserves_known_effect_across_metric_units(scale: float) -> None:
    # Means differ by one and both sample variances are four: d = 1 / 2.
    a = np.asarray([2.0, 4.0, 6.0]) * scale
    b = np.asarray([1.0, 3.0, 5.0]) * scale
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cohens_d(a, b) == pytest.approx(0.5)
        assert cohens_d(b, a) == pytest.approx(-0.5)


@pytest.mark.parametrize("scale", [1.0e200, 1.0e-200])
def test_cohens_d_preserves_singleton_group_effect(scale: float) -> None:
    # Only the two-element group contributes variance: pooled SD = sqrt(0.5).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cohens_d([scale], [2.0 * scale, 3.0 * scale]) == pytest.approx(
            -1.5 / np.sqrt(0.5)
        )


def test_wilcoxon_effect_size_matches_rescaled_metric() -> None:
    a = np.asarray([2.0, 4.0, 6.0, 9.0])
    b = np.asarray([1.0, 3.0, 5.0, 7.0])
    expected = wilcoxon_comparison(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = wilcoxon_comparison(a * 1.0e200, b * 1.0e200)
    assert actual.statistic == expected.statistic
    assert actual.p_value == expected.p_value
    assert actual.significant == expected.significant
    assert actual.effect_size == pytest.approx(expected.effect_size)


def test_cohens_d_retains_spread_of_smaller_group() -> None:
    # A constant large group has no spread; the other group's SD is one.
    # The pooled SD is sqrt(1/2), so the result still fits float64.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cohens_d([1.0e308] * 3, [1.0, 2.0, 3.0]) == pytest.approx(
            1.0e308 / np.sqrt(0.5)
        )
