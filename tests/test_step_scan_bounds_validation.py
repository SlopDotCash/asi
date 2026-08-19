"""Tests for scan sequence bounds and array validation across Steps 6, 8, 10, 12, Stacked Horde,
Independent Demon Horde, Actor-Critic, and UPGD.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.independent_demon_horde import (
    IndependentDemonHorde,
    run_independent_horde_learning_loop,
    run_independent_horde_learning_loop_batched,
)
from alberta_framework.core.options import SubtaskSpec
from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    StackedLinearHorde,
    run_stacked_horde_scan,
)
from alberta_framework.core.upgd import (
    UPGDLearner,
    run_upgd_arrays,
    run_upgd_loop,
)
from alberta_framework.steps.step6 import (
    Step6DifferentialSARSAConfig,
    init_step6_state,
    make_step6_differential_sarsa_agent,
    run_step6_scan,
)
from alberta_framework.steps.step8 import (
    Step8WorldModelConfig,
    init_step8_state,
    make_step8_world_model,
    run_step8_scan,
)
from alberta_framework.steps.step10 import (
    Step10STOMPConfig,
    init_step10_state,
    make_step10_stomp_agent,
    run_step10_scan,
)
from alberta_framework.steps.step12 import (
    Step12IAConfig,
    init_step12_state,
    make_step12_ia_agent,
    run_step12_scan,
)
from alberta_framework.streams.synthetic import RandomWalkStream

# ---------------------------------------------------------------------------
# Step 6 Scan Tests
# ---------------------------------------------------------------------------


def test_step6_scan_validation() -> None:
    cfg = Step6DifferentialSARSAConfig(n_actions=2)
    agent = make_step6_differential_sarsa_agent(cfg)
    key = jr.key(0)
    state = init_step6_state(agent, feature_dim=4, key=key, initial_features=jnp.zeros((4,)))

    rewards = jnp.zeros((5,), dtype=jnp.float32)
    next_features = jnp.zeros((5, 4), dtype=jnp.float32)

    # Valid run
    res = run_step6_scan(agent, state, rewards, next_features)
    assert res.q_values.shape == (5, 2)
    assert res.td_errors.shape == (5,)

    # Invalid agent / state
    with pytest.raises(TypeError, match="agent must be an exact"):
        run_step6_scan(None, state, rewards, next_features)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_step6_scan(agent, None, rewards, next_features)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="rewards must be a trusted array"):
        run_step6_scan(agent, state, [0.0] * 5, next_features)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rewards must contain between 1"):
        run_step6_scan(agent, state, jnp.zeros((0,)), jnp.zeros((0, 4)))
    with pytest.raises(ValueError, match="next_features must have shape"):
        run_step6_scan(agent, state, rewards, jnp.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Step 8 Scan Tests
# ---------------------------------------------------------------------------


def test_step8_scan_validation() -> None:
    cfg = Step8WorldModelConfig(observation_dim=3, n_actions=2)
    model = make_step8_world_model(cfg)
    key = jr.key(0)
    state = init_step8_state(model, key=key)

    obs = jnp.zeros((5, 3), dtype=jnp.float32)
    actions = jnp.zeros((5,), dtype=jnp.float32)
    rewards = jnp.zeros((5,), dtype=jnp.float32)
    next_obs = jnp.zeros((5, 3), dtype=jnp.float32)

    # Valid run
    res = run_step8_scan(model, state, obs, actions, rewards, next_obs)
    assert res.reward_predictions.shape == (5,)
    assert res.next_observation_predictions.shape == (5, 3)

    # Invalid model / state
    with pytest.raises(TypeError, match="model must be an exact"):
        run_step8_scan(None, state, obs, actions, rewards, next_obs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_step8_scan(model, None, obs, actions, rewards, next_obs)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="observations must be a trusted array"):
        run_step8_scan(model, state, [[0.0, 0.0, 0.0]], actions, rewards, next_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observations must contain between 1"):
        run_step8_scan(
            model, state, jnp.zeros((0, 3)), jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0, 3))
        )
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_step8_scan(model, state, obs, actions, rewards, jnp.zeros((5, 4)))


# ---------------------------------------------------------------------------
# Step 10 Scan Tests
# ---------------------------------------------------------------------------


def test_step10_scan_validation() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=1.0)
    cfg = Step10STOMPConfig(observation_dim=3, n_primitive_actions=2, subtask_specs=(spec,))
    agent = make_step10_stomp_agent(cfg)
    key = jr.key(0)
    state = init_step10_state(agent, key=key, initial_observation=jnp.zeros((3,)))

    rewards = jnp.zeros((5,), dtype=jnp.float32)
    next_obs = jnp.zeros((5, 3), dtype=jnp.float32)

    # Valid run
    res = run_step10_scan(agent, state, rewards, next_obs)
    assert res.primitive_actions.shape == (5,)

    # Invalid agent / state
    with pytest.raises(TypeError, match="agent must be an exact"):
        run_step10_scan(None, state, rewards, next_obs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_step10_scan(agent, None, rewards, next_obs)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="rewards must be a trusted array"):
        run_step10_scan(agent, state, [0.0] * 5, next_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rewards must contain between 1"):
        run_step10_scan(agent, state, jnp.zeros((0,)), jnp.zeros((0, 3)))
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_step10_scan(agent, state, rewards, jnp.zeros((5, 4)))


# ---------------------------------------------------------------------------
# Step 12 Scan Tests
# ---------------------------------------------------------------------------


def test_step12_scan_validation() -> None:
    spec = SubtaskSpec(feature_index=0, threshold=1.0)
    cfg = Step12IAConfig(
        observation_dim=3,
        n_primitive_actions=2,
        n_demons=2,
        subtask_specs=(spec,),
    )
    agent = make_step12_ia_agent(cfg)
    key = jr.key(0)
    state = init_step12_state(agent, key=key, initial_observation=jnp.zeros((3,)))

    partner_obs = jnp.zeros((5, 3), dtype=jnp.float32)
    partner_rewards = jnp.zeros((5,), dtype=jnp.float32)
    partner_next_obs = jnp.zeros((5, 3), dtype=jnp.float32)

    # Valid run
    res = run_step12_scan(agent, state, partner_obs, partner_rewards, partner_next_obs)
    assert res.predictions.shape == (5, 2)
    assert res.recommendations.shape == (5,)

    # Invalid agent / state
    with pytest.raises(TypeError, match="agent must be an exact"):
        run_step12_scan(None, state, partner_obs, partner_rewards, partner_next_obs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_step12_scan(agent, None, partner_obs, partner_rewards, partner_next_obs)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="partner_rewards must be a trusted array"):
        run_step12_scan(agent, state, partner_obs, [0.0] * 5, partner_next_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="partner_rewards must contain between 1"):
        run_step12_scan(
            agent, state, partner_obs, jnp.zeros((0,)), partner_next_obs
        )
    with pytest.raises(ValueError, match="partner_next_obs must have shape"):
        run_step12_scan(agent, state, partner_obs, partner_rewards, jnp.zeros((5, 4)))


# ---------------------------------------------------------------------------
# Stacked Linear Horde Scan Tests
# ---------------------------------------------------------------------------


def test_stacked_horde_scan_validation() -> None:
    config = StackedHordeConfig(
        n_demons=2,
        feature_dim=3,
        gammas=(0.9, 0.95),
        lamdas=(0.8, 0.8),
        cumulant_indices=(0, 1),
    )
    horde = StackedLinearHorde(config)
    state = horde.init()

    features = jnp.zeros((6, 3), dtype=jnp.float32)
    cumulant_sources = jnp.zeros((6, 2), dtype=jnp.float32)

    # Valid run
    final_state, td_errors = run_stacked_horde_scan(horde, state, features, cumulant_sources)
    assert td_errors.shape == (5, 2)

    # Invalid horde / state
    with pytest.raises(TypeError, match="horde must be an exact"):
        run_stacked_horde_scan(None, state, features, cumulant_sources)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_stacked_horde_scan(horde, None, features, cumulant_sources)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="features must be a trusted array"):
        run_stacked_horde_scan(horde, state, [[0.0, 0.0, 0.0]] * 6, cumulant_sources)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="features must contain between 2"):
        run_stacked_horde_scan(horde, state, jnp.zeros((1, 3)), jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="features must have shape"):
        run_stacked_horde_scan(horde, state, jnp.zeros((6, 4)), cumulant_sources)


# ---------------------------------------------------------------------------
# Independent Demon Horde Scan Tests
# ---------------------------------------------------------------------------


def test_independent_demon_horde_scan_validation() -> None:
    specs = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"d{i}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=i,
        )
        for i in range(2)
    ]
    horde = IndependentDemonHorde(horde_spec=create_horde_spec(specs), hidden_sizes=(8,))
    key = jr.key(0)
    state = horde.init(feature_dim=3, key=key)

    obs = jnp.zeros((5, 3), dtype=jnp.float32)
    cums = jnp.zeros((5, 2), dtype=jnp.float32)
    next_obs = jnp.zeros((5, 3), dtype=jnp.float32)

    # Valid run
    res = run_independent_horde_learning_loop(horde, state, obs, cums, next_obs)
    assert res.td_errors.shape == (5, 2)

    # Valid batched run
    keys = jr.split(key, 3)
    res_b = run_independent_horde_learning_loop_batched(horde, obs, cums, next_obs, keys)
    assert res_b.td_errors.shape == (3, 5, 2)

    # Invalid horde / state
    with pytest.raises(TypeError, match="horde must be an exact"):
        run_independent_horde_learning_loop(None, state, obs, cums, next_obs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_independent_horde_learning_loop(horde, None, obs, cums, next_obs)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="observations must be a trusted array"):
        run_independent_horde_learning_loop(horde, state, [[0.0, 0.0, 0.0]], cums, next_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observations must have shape"):
        run_independent_horde_learning_loop(horde, state, jnp.zeros((0, 3)), cums, next_obs)


# ---------------------------------------------------------------------------
# Actor-Critic Scan Tests
# ---------------------------------------------------------------------------


def test_actor_critic_scan_validation() -> None:
    # Discrete
    disc_cfg = ActorCriticConfig(n_actions=2)
    disc_agent = ActorCriticAgent(disc_cfg)
    key = jr.key(0)
    disc_state = disc_agent.init(feature_dim=3, key=key)

    obs = jnp.zeros((5, 3), dtype=jnp.float32)
    rewards = jnp.zeros((5,), dtype=jnp.float32)
    next_obs = jnp.zeros((5, 3), dtype=jnp.float32)
    discounts = jnp.ones((5,), dtype=jnp.float32)

    res_d = run_actor_critic_from_arrays(
        disc_agent,
        disc_state,
        obs,
        rewards,
        terminated=None,
        next_observations=next_obs,
        discounts=discounts,
    )
    assert res_d.actions.shape == (5,)

    with pytest.raises(TypeError, match="agent must be an exact ActorCriticAgent"):
        run_actor_critic_from_arrays(
            None,  # type: ignore[arg-type]
            disc_state,
            obs,
            rewards,
            terminated=None,
            next_observations=next_obs,
            discounts=discounts,
        )
    with pytest.raises(TypeError, match="state must be an exact ActorCriticState"):
        run_actor_critic_from_arrays(
            disc_agent,
            None,  # type: ignore[arg-type]
            obs,
            rewards,
            terminated=None,
            next_observations=next_obs,
            discounts=discounts,
        )

    # Continuous
    cont_cfg = ContinuousActorCriticConfig(action_dim=2)
    cont_agent = ContinuousActorCriticAgent(cont_cfg)
    cont_state = cont_agent.init(feature_dim=3, key=key)

    res_c = run_continuous_actor_critic_from_arrays(
        cont_agent,
        cont_state,
        obs,
        rewards,
        terminated=None,
        next_observations=next_obs,
        discounts=discounts,
    )
    assert res_c.actions.shape == (5, 2)

    with pytest.raises(TypeError, match="agent must be an exact ContinuousActorCriticAgent"):
        run_continuous_actor_critic_from_arrays(
            None,  # type: ignore[arg-type]
            cont_state,
            obs,
            rewards,
            terminated=None,
            next_observations=next_obs,
            discounts=discounts,
        )
    with pytest.raises(TypeError, match="state must be an exact ContinuousActorCriticState"):
        run_continuous_actor_critic_from_arrays(
            cont_agent,
            None,  # type: ignore[arg-type]
            obs,
            rewards,
            terminated=None,
            next_observations=next_obs,
            discounts=discounts,
        )


# ---------------------------------------------------------------------------
# UPGD Scan Tests
# ---------------------------------------------------------------------------


def test_upgd_scan_validation() -> None:
    learner = UPGDLearner(n_heads=2, hidden_sizes=(16,))
    key = jr.key(0)
    state = learner.init(3, key)

    obs = jnp.zeros((5, 3), dtype=jnp.float32)
    targets = jnp.zeros((5, 2), dtype=jnp.float32)

    # Valid array run
    res_arr = run_upgd_arrays(learner, state, obs, targets)
    assert res_arr.metrics.shape == (5, 4)

    # Valid stream loop run
    stream = RandomWalkStream(feature_dim=3)
    res_loop = run_upgd_loop(learner, stream, num_steps=5, key=key, learner_state=state)
    assert res_loop.metrics.shape == (5, 4)

    # Invalid learner / state
    with pytest.raises(TypeError, match="learner must be an exact UPGDLearner"):
        run_upgd_arrays(None, state, obs, targets)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact UPGDState"):
        run_upgd_arrays(learner, None, obs, targets)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="observations must be a trusted array"):
        run_upgd_arrays(learner, state, [[0.0, 0.0, 0.0]], targets)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observations must have shape"):
        run_upgd_arrays(learner, state, jnp.zeros((5, 2)), targets)
    with pytest.raises(ValueError, match="targets must have shape"):
        run_upgd_arrays(learner, state, obs, jnp.zeros((5, 3)))
