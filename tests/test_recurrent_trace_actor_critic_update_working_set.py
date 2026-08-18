"""Complete update working-set preflight for recurrent-trace actor-critic."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 30_000_000


def _small_config() -> RecurrentTraceActorCriticConfig:
    return RecurrentTraceActorCriticConfig(
        n_actions=3,
        hidden_size=3,
        encoder_width=2,
        output_width=4,
        sparsity=0.0,
        r_min=0.1,
        r_max=0.9,
        normalize_observations=False,
        normalize_rewards=False,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_rtac_one_state_and_persistent_fit_while_update_working_set_does_not() -> None:
    config = _small_config()
    budget = config.state_resource_budget(_WORKING_SET_OVERFLOW)
    persist = budget["state_nbytes"]
    assert persist <= _INT32_MAX
    assert 2 * persist > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        RecurrentTraceActorCriticAgent(config).init(_WORKING_SET_OVERFLOW, jr.key(0))


def test_rtac_persistent_derived_bound_still_fires_first() -> None:
    with pytest.raises(ValueError, match="derived"):
        RecurrentTraceActorCriticAgent(_small_config()).init(2**31 - 1, jr.key(0))


def test_legal_rtac_init_and_update_identity_is_unchanged() -> None:
    agent = RecurrentTraceActorCriticAgent(_small_config())
    state = agent.init(2, jr.key(2))
    budget = agent.config.state_resource_budget(2)
    assert budget["state_nbytes"] <= _INT32_MAX
    assert 2 * budget["state_nbytes"] <= _INT32_MAX
    state, _, _ = agent.start(state, jnp.asarray((0.4, -0.2), dtype=jnp.float32))
    result = agent.update(
        state,
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.6), dtype=jnp.float32),
    )
    assert result.state.last_observation.shape == (2,)
    assert result.action.shape == ()
