"""Unit coverage for alberta_framework.core._float32_scalars.

Tests the fail-closed float32-consumed scalar validation: domain policy
checks (positive, lower/upper bounds, inclusivity), exact vs narrowed
value enforcement, and the stored-value semantics.
"""

from fractions import Fraction

import pytest

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)


def test_basic_finite() -> None:
    assert validated_float32_scalar("x", 1.5) == 1.5
    assert validated_float32_scalar("x", 3) == 3.0


def test_positive_domain() -> None:
    assert validated_float32_scalar("x", 0.1, positive=True) == 0.1
    with pytest.raises(ValueError, match="positive"):
        validated_float32_scalar("x", 0, positive=True)
    with pytest.raises(ValueError, match="positive"):
        validated_float32_scalar("x", -1.0, positive=True)


def test_lower_bound() -> None:
    assert validated_float32_scalar("x", 1.0, lower=0.5) == 1.0
    with pytest.raises(ValueError, match=">="):
        validated_float32_scalar("x", 0.1, lower=0.5)


def test_upper_bound_inclusive() -> None:
    assert validated_float32_scalar("x", 1.0, upper=1.0) == 1.0
    with pytest.raises(ValueError, match="\\["):
        validated_float32_scalar("x", 1.1, upper=1.0)


def test_upper_bound_exclusive() -> None:
    with pytest.raises(ValueError, match="\\["):
        validated_float32_scalar("x", 1.0, upper=1.0, upper_inclusive=False)
    assert validated_float32_scalar("x", 0.9, upper=1.0, upper_inclusive=False) == 0.9


def test_rejects_bool() -> None:
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar("x", True)


def test_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validated_float32_scalar("x", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        validated_float32_scalar("x", float("nan"))


def test_rejects_narrowed_to_inf() -> None:
    # 1e40 is finite in float64 but narrows to float32 inf → must fail.
    with pytest.raises(ValueError, match="finite once narrowed"):
        validated_float32_scalar("x", 1e40)


def test_with_ratio_returns_exact() -> None:
    stored, num, den = validated_float32_scalar_with_ratio("x", 0.1)
    assert stored == 0.1
    assert num > 0
    assert den > 0


def test_stored_value_semantics() -> None:
    # ints are stored as the narrowed float32 value.
    stored = validated_float32_scalar("x", 3)
    assert stored == 3.0
    # floats are stored as-is.
    stored_f = validated_float32_scalar("x", 0.1)
    assert stored_f == 0.1


def test_fraction_value() -> None:
    assert validated_float32_scalar("x", Fraction(1, 2)) == 0.5
