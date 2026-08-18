"""Complete update working-set preflight for the sparse FTL world model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 10_000


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_ftl_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=1,
        action_dim=1,
        projection_dim=_WORKING_SET_OVERFLOW,
        bins=2,
    )
    feature_dim = config.feature_dim
    one_bank_bytes = 4 * (feature_dim * feature_dim)
    persistent_bytes = SparseFTLWorldModel(config).state_nbytes
    update_bytes = 4 * (2 * feature_dim * feature_dim + 5 * feature_dim + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        SparseFTLWorldModel(config).init(jr.key(0))


def test_ftl_last_legal_persistent_config_still_constructs() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=1,
        action_dim=1,
        projection_dim=11_584,
        bins=2,
    )
    model = SparseFTLWorldModel(config)
    assert model.state_nbytes == 2_147_302_920
    with pytest.raises(ValueError, match="derived state_nbytes"):
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=11_585,
            bins=2,
        )


def test_legal_ftl_update_identity_is_unchanged() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=4,
            bins=2,
        )
    )
    state = model.init(jr.key(0))
    assert state.gram.shape == (8, 8)
    assert 4 * (8 * 8 + 5 * 8 + 8) <= _INT32_MAX
    result = model.update(
        state,
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    assert result.state.gram.shape == (8, 8)
    assert result.state.weights.shape == (8, 1)
    assert result.prediction.next_observation.shape == (1,)
