"""Complete update working-set preflight for the RLS reward model."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from alberta_framework.core.reward_model import RLSRewardModel, RLSRewardModelConfig

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 20_000


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_rls_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    dim = _WORKING_SET_OVERFLOW
    one_bank_bytes = 4 * (dim * dim)
    persistent_bytes = 4 * (dim * dim + dim + 2)
    update_bytes = 4 * (3 * dim * dim + 5 * dim + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    config = RLSRewardModelConfig(feature_dim=dim)
    assert config.feature_dim == dim
    with pytest.raises(ValueError, match="update working set byte count"):
        RLSRewardModel(config).init()


def test_rls_persistent_byte_bound_still_fires_first() -> None:
    scalar_limit = _INT32_MAX // 4
    last_legal = (math.isqrt(1 + 4 * (scalar_limit - 2)) - 1) // 2
    RLSRewardModelConfig(feature_dim=last_legal)
    with pytest.raises(ValueError, match="state bytes"):
        RLSRewardModelConfig(feature_dim=last_legal + 1)


def test_legal_rls_update_identity_is_unchanged() -> None:
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=4, forgetting=1.0, ridge=1.0))
    state = model.init()
    assert state.weights.shape == (4,)
    assert state.covariance.shape == (4, 4)
    assert 4 * (4 * 4 + 4 + 2) <= _INT32_MAX
    result = model.update(
        state,
        jnp.ones(4, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert result.state.weights.shape == (4,)
    assert result.state.covariance.shape == (4, 4)
    assert result.gain.shape == (4,)
