"""#1383-complete update working-set preflight for the latent world model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.latent_world_model import (
    LatentWorldModel,
    LatentWorldModelConfig,
    _latent_direct_state_scalars,
    _latent_update_working_set_bytes,
    _preflight_latent_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_LATENT_DIM = 10_000
_LAST_FIT_LATENT_DIM = 9_454
_FIRST_OVERFLOW_LATENT_DIM = 9_455
_LINEAR_KWARGS = {
    "observation_dim": 1,
    "n_actions": 2,
    "hidden_sizes": (),
    "include_action_interactions": False,
}


def _linear_persist_bytes(latent_dim: int) -> int:
    return 4 * _latent_direct_state_scalars(
        observation_dim=1,
        n_actions=2,
        latent_dim=latent_dim,
        hidden_sizes=(),
        include_action_interactions=False,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_latent_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _linear_persist_bytes(_OVERFLOW_LATENT_DIM)
    working_set_bytes = _latent_update_working_set_bytes(
        latent_dim=_OVERFLOW_LATENT_DIM,
        **_LINEAR_KWARGS,
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 800_560_076
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_LATENT_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        LatentWorldModelConfig(latent_dim=_OVERFLOW_LATENT_DIM, **_LINEAR_KWARGS)


def test_latent_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for latent_dim in range(_LAST_FIT_LATENT_DIM, _FIRST_OVERFLOW_LATENT_DIM + 2):
        persist_bytes = _linear_persist_bytes(latent_dim)
        working_set_bytes = _latent_update_working_set_bytes(
            latent_dim=latent_dim,
            **_LINEAR_KWARGS,
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * latent_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = latent_dim
        elif first_overflow is None:
            first_overflow = latent_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_LATENT_DIM
    config = LatentWorldModelConfig(latent_dim=last_fit, **_LINEAR_KWARGS)
    assert config.latent_dim == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        LatentWorldModelConfig(latent_dim=first_overflow, **_LINEAR_KWARGS)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_latent_update_working_set(
            latent_dim=_OVERFLOW_LATENT_DIM,
            **_LINEAR_KWARGS,
        )


def test_persist_bound_still_fires_before_working_set() -> None:
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        LatentWorldModelConfig(
            observation_dim=1,
            n_actions=1,
            latent_dim=8,
            hidden_sizes=(16_384, 16_384),
        )


def test_legal_small_latent_world_model_still_constructs() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
    )
    persist_bytes = 4 * _latent_direct_state_scalars(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
        include_action_interactions=False,
    )
    assert persist_bytes == 444
    model = LatentWorldModel(config)
    state = model.init(jr.key(0))
    model.update(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.99, dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
