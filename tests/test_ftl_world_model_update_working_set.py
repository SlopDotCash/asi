"""Complete update working-set preflight for the sparse FTL world model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
    _preflight_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 10_000


def _working_scalars(config: SparseFTLWorldModelConfig) -> int:
    feature_dim = config.feature_dim
    active_dim = config.active_feature_count
    return (
        config.projection_dim * config.input_dim
        + 2 * feature_dim * feature_dim
        + 4 * feature_dim * config.observation_dim
        + feature_dim
        + 4 * active_dim * active_dim
        + 8 * active_dim * config.observation_dim
        + active_dim * feature_dim
        + 2 * active_dim
        + config.input_dim
        + 8 * config.projection_dim
        + 6 * config.observation_dim
        + config.action_dim
        + 8
    )


def _preflight(config: SparseFTLWorldModelConfig) -> None:
    _preflight_update_working_set(
        projection_dim=config.projection_dim,
        input_dim=config.input_dim,
        feature_dim=config.feature_dim,
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
    )


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
        projection_dim=4_500,
        bins=2,
    )
    feature_dim = config.feature_dim
    one_bank_bytes = 4 * (feature_dim * feature_dim)
    persistent_bytes = SparseFTLWorldModel(config).state_nbytes
    contributor_formula = (
        config.projection_dim * config.input_dim
        + 2 * feature_dim * feature_dim
        + 4 * feature_dim * config.observation_dim
        + feature_dim
        + 2 * config.active_feature_count**2
        + 4 * config.active_feature_count * config.observation_dim
        + 2 * config.active_feature_count
        + 6 * config.observation_dim
        + config.action_dim
        + 8
    )
    update_bytes = 4 * _working_scalars(config)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert 4 * contributor_formula <= _INT32_MAX
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
    assert 4 * _working_scalars(model.config) <= _INT32_MAX
    result = model.update(
        state,
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    assert result.state.gram.shape == (8, 8)
    assert result.state.weights.shape == (8, 1)
    assert result.prediction.next_observation.shape == (1,)


def test_ftl_cross_and_weight_copies_are_scaled_by_observation_dim() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=100_000,
        action_dim=1,
        projection_dim=1_000,
        bins=2,
    )
    model = SparseFTLWorldModel(config)
    contributor_formula = 2 * config.feature_dim**2 + 5 * config.feature_dim + 8
    assert model.state_nbytes <= _INT32_MAX
    assert 4 * contributor_formula <= _INT32_MAX
    assert 4 * _working_scalars(config) > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        model.init(jr.key(1))


def test_ftl_exact_last_legal_update_width_and_first_overflow() -> None:
    def config(projection_dim: int) -> SparseFTLWorldModelConfig:
        return SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=projection_dim,
            bins=2,
        )

    low, high = 1, 11_584
    while low < high:
        middle = (low + high + 1) // 2
        if 4 * _working_scalars(config(middle)) <= _INT32_MAX:
            low = middle
        else:
            high = middle - 1
    last_legal = config(low)
    first_overflowing = config(low + 1)
    assert 4 * _working_scalars(last_legal) <= _INT32_MAX
    assert 4 * _working_scalars(first_overflowing) > _INT32_MAX
    _preflight(last_legal)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight(first_overflowing)
