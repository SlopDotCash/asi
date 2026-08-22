"""Supplementary coverage for feature_discovery and normalizers helpers.

Covers previously untested helpers: validated_float32_scalar (float32 sink
with nonzero-collapse rejection) and measure_normalizer_state_nbytes (leaf
byte accounting across concrete normalizer states).
"""

import jax.numpy as jnp
import pytest

from alberta_framework.core.feature_discovery import validated_float32_scalar
from alberta_framework.core.normalizers import measure_normalizer_state_nbytes


def test_validated_float32_scalar_plain() -> None:
    assert validated_float32_scalar("x", 0.5) == 0.5
    assert validated_float32_scalar("x", 3) == 3.0


def test_validated_float32_scalar_domain() -> None:
    assert validated_float32_scalar("x", 0.25, lower=0.0, upper=1.0) == 0.25
    with pytest.raises(ValueError):
        validated_float32_scalar("x", 2.0, upper=1.0)


def test_validated_float32_scalar_nonzero_collapse() -> None:
    # A value that narrows to float32 zero but is not exactly zero must be
    # rejected (numerator != 0 but stored == 0.0).
    with pytest.raises(ValueError, match="remain nonzero"):
        validated_float32_scalar("tiny", 1e-50)


def test_validated_float32_scalar_exact_zero_ok() -> None:
    assert validated_float32_scalar("zero", 0.0) == 0.0


def test_measure_normalizer_nbytes_flat() -> None:
    # A normalizer wrapping one float32 array of shape (4,).
    state = {"counts": jnp.zeros(4, dtype=jnp.float32)}
    assert measure_normalizer_state_nbytes(state) == 16


def test_measure_normalizer_nbytes_nested() -> None:
    # Note: JAX float64 is disabled in this environment, so float64 arrays
    # narrow to float32 (2x2 float32 = 16 bytes).
    state = {"stats": {"mean": jnp.zeros((2, 2), dtype=jnp.float64), "n": 5}}
    assert measure_normalizer_state_nbytes(state) == 2 * 2 * 4


def test_measure_normalizer_nbytes_non_array_ignored() -> None:
    state = {"config": {"name": "x"}, "arr": jnp.zeros(1, dtype=jnp.float32)}
    assert measure_normalizer_state_nbytes(state) == 4
