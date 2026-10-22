"""Compiled temperature-scaling regressions for the average-reward Horde actor.

Under ``jax.jit`` on XLA:CPU, ``softmax(x / T)`` with a non-power-of-two scale
returns NaN for finite inputs once ``|x / T|`` reaches roughly ``2**31``: XLA
recomputes the product inside both softmax fusions and the contracted FMA
leaves a rounding residual at the argmax.  Issue #2886 fixed this for the
actor-critic, behavior-model, and Horde actor agents; the average-reward Horde
actor still divided logits by the temperature directly.  A validated finite
temperature then produced a NaN policy, and the update finiteness gates
rejected every later transition, so the agent never took its first step.

These tests pin the stabilized construction: finite inputs keep a finite
policy, accepted finite updates, and a finite committed state.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
    AverageRewardHordeActorCriticState,
)

_OBSERVATION = jnp.asarray([1.0, 1.0], dtype=jnp.float32)


def _agent(temperature: float) -> AverageRewardHordeActorCriticAgent:
    return AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(n_actions=2, temperature=temperature)
    )


def _constant_logits_state(
    agent: AverageRewardHordeActorCriticAgent,
    logits: tuple[float, float],
) -> AverageRewardHordeActorCriticState:
    """Force the actor head to report one constant logit vector."""
    state = agent.init(2, jr.key(0))
    return state.replace(  # type: ignore[attr-defined,no-any-return]
        actor_weights=jnp.zeros_like(state.actor_weights),
        actor_bias=jnp.asarray(logits, dtype=jnp.float32),
    )


def test_low_temperature_policy_stays_finite() -> None:
    """A validated finite temperature must not emit a NaN policy."""
    agent = _agent(temperature=1e-10)
    state = _constant_logits_state(agent, (20.0, -20.0))

    probabilities = np.asarray(agent.policy(state, _OBSERVATION))
    np.testing.assert_array_equal(np.isfinite(probabilities), True)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=1e-5)
    # logits / 1e-10 is a large finite number: the argmax must survive.
    np.testing.assert_array_equal(probabilities.argmax(), 0)

    started, _action = agent.start(state, _OBSERVATION)
    np.testing.assert_array_equal(
        np.isfinite(np.asarray(started.last_policy_sample.target_policy)), True
    )


def test_low_temperature_updates_keep_applying() -> None:
    """Finite transitions must be accepted instead of rejected forever."""
    agent = _agent(temperature=1e-10)
    state = _constant_logits_state(agent, (20.0, -20.0))
    started, _action = agent.start(state, _OBSERVATION)

    first = agent.update(started, jnp.asarray(0.5), _OBSERVATION)
    assert bool(first.update_applied)
    np.testing.assert_array_equal(np.isfinite(np.asarray(first.policy)), True)
    np.testing.assert_allclose(float(jnp.sum(first.policy)), 1.0, rtol=1e-5)
    assert int(first.state.step_count) == 1

    second = agent.update(first.state, jnp.asarray(-0.5), _OBSERVATION)
    assert bool(second.update_applied)
    np.testing.assert_array_equal(np.isfinite(np.asarray(second.policy)), True)
    assert int(second.state.step_count) == 2


@pytest.mark.parametrize("temperature", [1.0, 0.25])
def test_moderate_temperature_matches_the_historical_expression(
    temperature: float,
) -> None:
    """Ordinary policies are unchanged up to float32 rounding."""
    agent = _agent(temperature=temperature)
    state = _constant_logits_state(agent, (0.75, -1.25))

    probabilities = agent.policy(state, _OBSERVATION)
    expected = jax.nn.softmax(
        jnp.asarray([0.75, -1.25], dtype=jnp.float32) / jnp.float32(temperature)
    )
    np.testing.assert_allclose(
        np.asarray(probabilities), np.asarray(expected), rtol=1e-5
    )
