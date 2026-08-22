"""Supplementary coverage for temporal_context.py scalar guards.

Covers previously untested helpers: finite_real_and_float32 (type gating,
finite-narrowing validation) and canonical_float32_storage (round-trip
stability for numeric vs non-numeric inputs).
"""

import numpy as np
import pytest

from alberta_framework.core.temporal_context import (
    canonical_float32_storage,
    finite_real_and_float32,
)


def test_finite_real_accepts_int() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", 3)
    assert real == 3
    assert narrowed == 3.0


def test_finite_real_accepts_float() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", 0.5)
    assert real == 0.5
    assert narrowed == 0.5


def test_finite_real_accepts_numpy_scalar() -> None:
    real, _, _, narrowed = finite_real_and_float32("x", np.float32(1.5))
    assert real == 1.5
    assert narrowed == 1.5


def test_finite_real_rejects_string() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", "not-a-number")


def test_finite_real_rejects_bool() -> None:
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", True)


def test_finite_real_rejects_inf() -> None:
    with pytest.raises(ValueError, match="finite float32"):
        finite_real_and_float32("x", float("inf"))


def test_canonical_storage_roundtrip() -> None:
    assert canonical_float32_storage(0.5, 0.5) == 0.5
    assert canonical_float32_storage(3, 3.0) == 3.0


def test_canonical_storage_non_numeric_returns_narrowed() -> None:
    assert canonical_float32_storage(object(), 2.5) == 2.5


def test_canonical_storage_rejects_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_float32_storage(float("inf"), 1.0)
