"""Array runners must report the action taken when an update is rejected.

A rejected learning update rolls the agent state back, but the action for
that row was still selected (sampled on-policy) or supplied (fixed log
actions). Reporting action 0 for the rejected row falsifies the recorded
trajectory: action 0 was never taken, and a caller replaying the recorded
actions dispatches the wrong action. Only an invalid action itself keeps
the 0 fallback.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    run_horde_actor_critic_from_arrays,
    run_nonlinear_horde_actor_critic_from_arrays,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec


def test_discrete_runner_reports_fixed_action_on_rejected_update() -> None:
    """A valid fixed action survives a rejected update in the recorded actions."""
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=3, gamma=0.9))
    state = agent.init(feature_dim=2, key=jr.key(0))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.array([1.0, jnp.inf], dtype=jnp.float32)
    terminated = jnp.array([False, False])
    actions = jnp.array([1, 2], dtype=jnp.int32)

    result = run_actor_critic_from_arrays(
        agent, state, observations, rewards, terminated, next_observations, actions=actions
    )

    assert bool(result.updates_applied[0])
    assert not bool(result.updates_applied[1])
    chex.assert_trees_all_equal(result.actions, jnp.array([1, 2], dtype=jnp.int32))


def test_discrete_runner_reports_zero_for_invalid_fixed_action() -> None:
    """An invalid fixed action still reports 0 on its rejected row.

    Concrete out-of-range actions raise eagerly, so this exercises the
    traced (JIT) path, where invalid values become a validity mask.
    """
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=3, gamma=0.9))
    state = agent.init(feature_dim=2, key=jr.key(0))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 1.0], dtype=jnp.float32)
    terminated = jnp.array([False, False])
    actions = jnp.array([1, 5], dtype=jnp.int32)

    compiled = jax.jit(
        lambda initial_state, fixed_actions: run_actor_critic_from_arrays(
            agent,
            initial_state,
            observations,
            rewards,
            terminated,
            next_observations,
            actions=fixed_actions,
        )
    )
    result = compiled(state, actions)

    assert bool(result.updates_applied[0])
    assert not bool(result.updates_applied[1])
    chex.assert_trees_all_equal(result.actions, jnp.array([1, 0], dtype=jnp.int32))


def test_continuous_runner_reports_fixed_action_on_rejected_update() -> None:
    """A valid fixed continuous action survives a rejected update."""
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    state = agent.init(feature_dim=2, key=jr.key(7))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.array([1.0, jnp.inf], dtype=jnp.float32)
    terminated = jnp.array([False, False])
    actions = jnp.array([[0.5, -0.5], [0.25, 0.75]], dtype=jnp.float32)

    result = run_continuous_actor_critic_from_arrays(
        agent, state, observations, rewards, terminated, next_observations, actions=actions
    )

    assert bool(result.updates_applied[0])
    assert not bool(result.updates_applied[1])
    chex.assert_trees_all_close(result.actions[1], jnp.array([0.25, 0.75], dtype=jnp.float32))


def _make_horde_agent() -> HordeActorCriticAgent:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.8,
            cumulant_index=-1,
        )
    ]
    critic = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )
    return HordeActorCriticAgent(
        HordeActorCriticConfig(
            n_actions=2,
            actor_step_size=0.05,
            actor_lamda=0.7,
        ),
        critic=critic,
    )


def test_horde_runner_reports_fixed_action_on_rejected_update() -> None:
    """A valid fixed action survives a rejected Horde actor-critic update."""
    agent = _make_horde_agent()
    state = agent.init(feature_dim=4, key=jr.key(0))
    observations = jnp.zeros((2, 4), dtype=jnp.float32)
    next_observations = jnp.zeros((2, 4), dtype=jnp.float32)
    rewards = jnp.array([1.0, jnp.inf], dtype=jnp.float32)
    actions = jnp.array([1, 1], dtype=jnp.int32)

    result = run_horde_actor_critic_from_arrays(
        agent, state, observations, rewards, next_observations, actions=actions
    )

    assert bool(result.updates_applied[0])
    assert not bool(result.updates_applied[1])
    chex.assert_trees_all_equal(result.actions, jnp.array([1, 1], dtype=jnp.int32))


def _make_nonlinear_agent() -> NonlinearHordeActorCriticAgent:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=0.99,
            lamda=0.0,
            cumulant_index=0,
        )
    ]
    critic = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(32,),
        step_size=0.03,
    )
    return NonlinearHordeActorCriticAgent(
        NonlinearHordeActorCriticConfig(
            n_actions=3,
            hidden_sizes=(),
            temperature=0.5,
            actor_lamda=0.9,
        ),
        critic,
    )


def test_nonlinear_horde_runner_reports_sampled_action_on_rejected_update() -> None:
    """A sampled action survives a rejected nonlinear Horde update."""
    agent = _make_nonlinear_agent()
    state = agent.init(feature_dim=8, key=jr.key(0)).replace(
        actor_head_w=jnp.zeros((3, 8), dtype=jnp.float32),
        actor_head_b=jnp.asarray([0.0, 10.0, 0.0], dtype=jnp.float32),
    )
    state, _, _ = agent.start(state, jnp.zeros(8))
    observations = jnp.zeros((2, 8), dtype=jnp.float32)
    next_observations = jnp.zeros((2, 8), dtype=jnp.float32)
    rewards = jnp.array([1.0, jnp.inf], dtype=jnp.float32)

    result = run_nonlinear_horde_actor_critic_from_arrays(
        agent, state, observations, rewards, next_observations
    )

    assert bool(result.updates_applied[0])
    assert not bool(result.updates_applied[1])
    assert int(result.actions[1]) == 1
