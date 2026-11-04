"""A rejected recurrent-trace actor-critic update reports the prior action.

When an update is rejected, the returned state rolls back to the incoming
state, so the agent keeps its prior action. The returned ``action`` must
mirror that state instead of reporting action 0.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
)


def _agent() -> RecurrentTraceActorCriticAgent:
    config = RecurrentTraceActorCriticConfig(
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
    return RecurrentTraceActorCriticAgent(config)


def test_rejected_update_reports_prior_last_action() -> None:
    agent = _agent()
    state = agent.init(2, jr.key(0))
    state, _, _ = agent.start(state, jnp.asarray((0.25, -0.5), dtype=jnp.float32))
    # An out-of-range previous discount is only shape/dtype-checked at the
    # eager boundary, but it gates candidate_applied inside the update, so
    # this update is rejected.
    state = state.replace(
        last_action=jnp.asarray(1, dtype=jnp.int32),
        previous_discount=jnp.asarray(1.5, dtype=jnp.float32),
    )
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.4), dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    assert int(result.state.last_action) == 1
    assert int(result.action) == 1
