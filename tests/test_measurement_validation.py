"""Unit coverage for alberta_framework.evaluation._measurement_validation.

Tests the exact measurement-record gates: host-type enforcement, finiteness,
non-negativity, and confidence-interval bounds.
"""

import pytest

from alberta_framework.evaluation._measurement_validation import (
    finite_real,
    nonnegative_finite_real,
    real_number,
    validate_interval_bounds,
)


def test_real_number_accepts_int_float() -> None:
    assert real_number("x", 1) == 1.0
    assert real_number("x", 1.5) == 1.5
    import math

    assert math.isnan(real_number("x", float("nan")))  # non-finite allowed


def test_real_number_rejects_facade_types() -> None:
    with pytest.raises(ValueError, match="real number"):
        real_number("x", "1")
    with pytest.raises(ValueError, match="real number"):
        real_number("x", True)
    with pytest.raises(ValueError, match="real number"):
        real_number("x", [1.0])
    with pytest.raises(ValueError, match="real number"):
        real_number("x", None)


def test_finite_real() -> None:
    assert finite_real("x", 2) == 2.0
    assert finite_real("x", -0.5) == -0.5
    with pytest.raises(ValueError, match="finite"):
        finite_real("x", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        finite_real("x", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        finite_real("x", "1.5")


def test_nonnegative_finite_real() -> None:
    assert nonnegative_finite_real("x", 0) == 0.0
    assert nonnegative_finite_real("x", 3.5) == 3.5
    with pytest.raises(ValueError, match="non-negative"):
        nonnegative_finite_real("x", -1)
    with pytest.raises(ValueError, match="finite"):
        nonnegative_finite_real("x", float("inf"))


def test_validate_interval_bounds() -> None:
    validate_interval_bounds(lower=0.1, upper=0.9, confidence_level=0.95)
    validate_interval_bounds(lower=0.5, upper=0.5, confidence_level=0.5)  # equal ok
    with pytest.raises(ValueError, match="lower must not exceed"):
        validate_interval_bounds(lower=0.9, upper=0.1, confidence_level=0.95)
    with pytest.raises(ValueError, match="confidence_level"):
        validate_interval_bounds(lower=0.1, upper=0.9, confidence_level=1.0)
    with pytest.raises(ValueError, match="confidence_level"):
        validate_interval_bounds(lower=0.1, upper=0.9, confidence_level=0.0)
