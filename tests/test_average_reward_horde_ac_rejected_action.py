"""A rejected average-reward Horde actor-critic update must report the
prior action, not action 0.

On rejection the returned state rolls back to the incoming state, so its
``last_action`` is the prior action. Reporting ``0`` instead makes the
result disagree with the state: a caller that trusts ``result.action``
dispatches action 0 while the agent still holds the prior action.
"""

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.average_reward import (
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
)


def test_rejected_update_reports_prior_action() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(n_actions=2, hidden_sizes=(4,))
    )
    state, start_action = agent.start(
        agent.init(2, jr.key(1)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    assert int(start_action) == 1

    result = agent.update(
        state,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    assert int(result.state.last_action) == 1
    assert int(result.action) == int(result.state.last_action)
