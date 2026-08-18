"""#1383-complete update working-set preflight for actor-critic."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    _actor_critic_persistent_bytes,
    _actor_critic_update_working_set_bytes,
    _continuous_actor_critic_persistent_bytes,
    _continuous_actor_critic_update_working_set_bytes,
    _preflight_actor_critic_update_working_set,
    _preflight_continuous_actor_critic_update_working_set,
    _require_continuous_state_resources,
    _require_discrete_state_resources,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_FEATURE_DIM = 40_000_000
_LAST_FIT_FEATURE_DIM = 35_791_392
_FIRST_OVERFLOW_FEATURE_DIM = 35_791_393
_CONTINUOUS_LAST_FIT_FEATURE_DIM = 35_791_391
_CONTINUOUS_FIRST_OVERFLOW_FEATURE_DIM = 35_791_392


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _actor_critic_persistent_bytes(1, _OVERFLOW_FEATURE_DIM)
    working_set_bytes = _actor_critic_update_working_set_bytes(1, _OVERFLOW_FEATURE_DIM)
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 800_000_032
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
    with pytest.raises(ValueError, match="update working set byte count"):
        agent.init(_OVERFLOW_FEATURE_DIM, jr.key(0))


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_LAST_FIT_FEATURE_DIM, _FIRST_OVERFLOW_FEATURE_DIM + 2):
        persist_bytes = _actor_critic_persistent_bytes(1, feature_dim)
        working_set_bytes = _actor_critic_update_working_set_bytes(1, feature_dim)
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_FEATURE_DIM
    _preflight_actor_critic_update_working_set(1, last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_actor_critic_update_working_set(1, first_overflow)
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
    with pytest.raises(ValueError, match="update working set byte count"):
        agent.init(first_overflow, jr.key(0))


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_actor_critic_update_working_set(1, _OVERFLOW_FEATURE_DIM)


def test_persist_bound_still_fires_before_working_set() -> None:
    _require_discrete_state_resources(1, 107_374_180)
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
    with pytest.raises(ValueError, match="state exceeds"):
        agent.init(107_374_181, jr.key(0))


def test_legal_small_actor_critic_still_updates() -> None:
    persist_bytes = _actor_critic_persistent_bytes(1, 4)
    assert persist_bytes == 112
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
    state = agent.init(4, jr.key(0))
    observation = jnp.zeros((4,), dtype=jnp.float32)
    state, _, _ = agent.start(state, observation)
    agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        observation,
        jnp.asarray(False),
    )


def test_continuous_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _continuous_actor_critic_persistent_bytes(1, _OVERFLOW_FEATURE_DIM)
    working_set_bytes = _continuous_actor_critic_update_working_set_bytes(
        1, _OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 800_000_040
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    with pytest.raises(ValueError, match="update working set byte count"):
        agent.init(_OVERFLOW_FEATURE_DIM, jr.key(0))


def test_continuous_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(
        _CONTINUOUS_LAST_FIT_FEATURE_DIM, _CONTINUOUS_FIRST_OVERFLOW_FEATURE_DIM + 2
    ):
        persist_bytes = _continuous_actor_critic_persistent_bytes(1, feature_dim)
        working_set_bytes = _continuous_actor_critic_update_working_set_bytes(
            1, feature_dim
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _CONTINUOUS_LAST_FIT_FEATURE_DIM
    _preflight_continuous_actor_critic_update_working_set(1, last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_continuous_actor_critic_update_working_set(1, first_overflow)
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    with pytest.raises(ValueError, match="update working set byte count"):
        agent.init(first_overflow, jr.key(0))


def test_continuous_persist_bound_still_fires_before_working_set() -> None:
    _require_continuous_state_resources(1, 107_374_180)
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    with pytest.raises(ValueError, match="state exceeds"):
        agent.init(107_374_181, jr.key(0))


def test_legal_small_continuous_actor_critic_still_updates() -> None:
    persist_bytes = _continuous_actor_critic_persistent_bytes(1, 5)
    assert persist_bytes == 140
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    state = agent.init(5, jr.key(0))
    observation = jnp.zeros((5,), dtype=jnp.float32)
    state, _, _, _ = agent.start(state, observation)
    agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        observation,
        jnp.asarray(False),
    )
