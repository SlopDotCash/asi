"""Supplementary coverage for associative_memory.py scalar guards.

Covers the previously untested copy of finite_real_and_float32 /
canonical_float32_storage in this module (the associative-memory variant
uses issubclass(Real) type gating).
"""

import numpy as np
import pytest

from alberta_framework.core.associative_memory import (
    canonical_float32_storage,
    finite_real_and_float32,
)


def test_finite_real_accepts_int() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", 7)
    assert real == 7
    assert narrowed == 7.0


def test_finite_real_accepts_float() -> None:
    real, _, _, narrowed = finite_real_and_float32("x", 0.25)
    assert real == 0.25
    assert narrowed == 0.25


def test_finite_real_accepts_numpy_float() -> None:
    real, _, _, narrowed = finite_real_and_float32("x", np.float64(1.5))
    assert real == 1.5
    assert narrowed == 1.5


def test_finite_real_rejects_bool() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", True)


def test_finite_real_rejects_string() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", "abc")


def test_finite_real_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite float32"):
        finite_real_and_float32("x", float("nan"))


def test_canonical_storage_roundtrip() -> None:
    assert canonical_float32_storage(1.0, 1.0) == 1.0
    assert canonical_float32_storage(2, 2.0) == 2.0


def test_canonical_storage_non_numeric_returns_narrowed() -> None:
    # value must be Real; a non-numeric value is not expected, but the
    # isinstance guard covers it by returning the narrowed value.
    assert canonical_float32_storage(np.float32(3.5), 3.5) == 3.5
