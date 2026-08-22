"""Reject oversized checkpoint empty-array shape ranks before walk hang."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.checkpoints import (
    _MAX_ARRAY_RANK,
    _restore_empty_arrays,
)


def test_checkpoint_empty_array_rank_cap_constant() -> None:
    assert _MAX_ARRAY_RANK == 32


def test_checkpoint_empty_array_accepts_max_rank() -> None:
    template = {"x": jnp.empty((0,) * _MAX_ARRAY_RANK, dtype=jnp.float32)}
    restored = {"x": jnp.zeros((1,), dtype=jnp.float32)}
    manifest = [
        {"shape": [0] * _MAX_ARRAY_RANK, "dtype": str(template["x"].dtype)},
    ]
    loaded = _restore_empty_arrays(template, restored, manifest)
    assert loaded["x"].shape == (0,) * _MAX_ARRAY_RANK
    assert loaded["x"].dtype == jnp.float32


def test_checkpoint_empty_array_rejects_oversized_rank_before_dimension_walk() -> None:
    template = {"x": jnp.empty((0,), dtype=jnp.float32)}
    restored = {"x": jnp.zeros((1,), dtype=jnp.float32)}
    manifest = [{"shape": [0] * (_MAX_ARRAY_RANK + 1), "dtype": "float32"}]
    with pytest.raises(ValueError, match="shape rank"):
        _restore_empty_arrays(template, restored, manifest)
