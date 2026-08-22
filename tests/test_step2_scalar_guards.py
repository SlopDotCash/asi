"""Supplementary coverage for steps/step2.py scalar guards.

Covers the previously untested step2 variant of finite_real_and_float32,
which additionally accepts Fraction subclasses via MRO lineage and reports
a distinct message for finite-but-unnarrowable real values.
"""

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.steps.step2 import (
    canonical_float32_storage,
    finite_real_and_float32,
)


def test_accepts_int() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", 5)
    assert real == 5
    assert narrowed == 5.0


def test_accepts_float() -> None:
    real, _, _, narrowed = finite_real_and_float32("x", 0.75)
    assert real == 0.75
    assert narrowed == 0.75


def test_accepts_fraction() -> None:
    # Fraction is not in _ALLOWED_REAL_TYPES but has real MRO lineage.
    real, num, den, narrowed = finite_real_and_float32("x", Fraction(1, 3))
    assert real == Fraction(1, 3)
    assert num == 1
    assert den == 3
    assert narrowed == pytest.approx(1 / 3)


def test_rejects_bool() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", True)


def test_rejects_string() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", "abc")


def test_rejects_inf() -> None:
    with pytest.raises(ValueError, match="finite float32"):
        finite_real_and_float32("x", float("inf"))


def test_rejects_object() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", object())


def test_canonical_storage_roundtrip() -> None:
    assert canonical_float32_storage(1.0, 1.0) == 1.0
    assert canonical_float32_storage(np.float32(2.5), 2.5) == 2.5
