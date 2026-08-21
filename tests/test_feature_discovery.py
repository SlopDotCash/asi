"""Unit coverage for alberta_framework.streams.feature_discovery.

Tests the trusted-array and Threefry-key validation gates plus the
resource math helpers.
"""

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.streams.feature_discovery import (
    _require_array,
    _require_key,
)


def test_require_array_accepts() -> None:
    arr = jnp.zeros((3, 4), dtype=jnp.float32)
    assert _require_array(arr, name="x", shape=(3, 4), dtype=jnp.float32) is arr


def test_require_array_rejects_type() -> None:
    with pytest.raises(TypeError, match="trusted JAX array"):
        _require_array([[1.0]], name="x", shape=(1, 1), dtype=jnp.float32)


def test_require_array_rejects_shape() -> None:
    arr = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match="shape"):
        _require_array(arr, name="x", shape=(4, 3), dtype=jnp.float32)


def test_require_array_rejects_dtype() -> None:
    arr = jnp.zeros((3, 4), dtype=jnp.int32)
    with pytest.raises(ValueError, match="dtype"):
        _require_array(arr, name="x", shape=(3, 4), dtype=jnp.float32)


def test_require_key_accepts() -> None:
    key = jax.random.key(0)
    assert _require_key(key, name="k") is key


def test_require_key_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="Threefry"):
        _require_key(123, name="k")


def test_require_key_rejects_wrong_shape() -> None:
    # A non-scalar array of uint32 is not a key.
    arr = jnp.zeros((2,), dtype=jnp.uint32)
    with pytest.raises(TypeError, match="Threefry"):
        _require_key(arr, name="k")
