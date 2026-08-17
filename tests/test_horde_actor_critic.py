"""Tests for Horde-backed actor-critic integration."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import HordeActorCriticAgent as TopLevelHordeActorCriticAgent
from alberta_framework.core import HordeActorCriticAgent as CoreHordeActorCriticAgent
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
    QHordeActorCriticState,
    run_horde_actor_critic_from_arrays,
)
from alberta_framework.core.optimizers import Autostep, ObGD, ObGDBounding
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    create_horde_spec,
)


def _assert_state_unchanged(actual, expected) -> None:
    """Compare actor states while handling typed PRNG keys explicitly."""
    chex.assert_trees_all_equal(
        jr.key_data(actual.rng_key),
        jr.key_data(expected.rng_key),
    )
    chex.assert_trees_all_close(
        actual.replace(rng_key=jr.key_data(actual.rng_key)),
        expected.replace(rng_key=jr.key_data(expected.rng_key)),
    )


def _make_agent(n_demons: int = 1) -> HordeActorCriticAgent:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.8,
            cumulant_index=-1,
        )
    ]
    for idx in range(1, n_demons):
        demons.append(
            GVFSpec(  # type: ignore[call-arg]
                name=f"aux_{idx}",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
        )
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


def _make_qhorde_agent(n_actions: int = 2, n_aux: int = 0) -> QHordeActorCriticAgent:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"q_{idx}",
            demon_type=DemonType.CONTROL,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=-1,
        )
        for idx in range(n_actions)
    ]
    demons.extend(
        GVFSpec(  # type: ignore[call-arg]
            name=f"aux_{idx}",
            demon_type=DemonType.PREDICTION,
            gamma=0.5,
            lamda=0.0,
            cumulant_index=0,
        )
        for idx in range(n_aux)
    )
    critic = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )
    return QHordeActorCriticAgent(
        QHordeActorCriticConfig(
            n_actions=n_actions,
            gamma=0.9,
            actor_step_size=0.05,
            actor_lamda=0.7,
        ),
        critic=critic,
    )


def test_horde_actor_critic_value_head_updates_actor_and_critic() -> None:
    agent = _make_agent()
    state = agent.init(feature_dim=2, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 1
    chex.assert_trees_all_close(result.td_error, result.critic_result.td_errors[0])
    assert not jnp.allclose(result.state.actor_weights, state.actor_weights)
    assert not jnp.allclose(
        agent.critic.predict(result.state.critic_state, state.last_observation)[0],
        agent.critic.predict(state.critic_state, state.last_observation)[0],
    )
    chex.assert_tree_all_finite(
        (
            result.state.actor_weights,
            result.state.actor_bias,
            result.value,
            result.next_value,
            result.td_error,
        )
    )


def test_horde_actor_critic_auxiliary_prediction_demon_updates() -> None:
    agent = _make_agent(n_demons=2)
    state = agent.init(feature_dim=2, key=jr.key(1)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([0.5, -1.0], dtype=jnp.float32),
        last_action=jnp.array(1, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        reward=jnp.array(0.25, dtype=jnp.float32),
        observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        auxiliary_cumulants=jnp.array([2.0], dtype=jnp.float32),
    )

    chex.assert_shape(result.critic_result.td_errors, (2,))
    chex.assert_trees_all_close(
        result.critic_result.td_targets[1],
        jnp.array(2.0, dtype=jnp.float32),
    )
    assert not jnp.allclose(
        agent.critic.predict(result.state.critic_state, state.last_observation)[1],
        agent.critic.predict(state.critic_state, state.last_observation)[1],
    )


def test_horde_actor_critic_config_roundtrip_and_exports() -> None:
    base_agent = _make_agent(n_demons=2)
    agent = HordeActorCriticAgent(
        HordeActorCriticConfig.from_config(
            {
                **base_agent.config.to_config(),
                "actor_td_error_clip": 0.75,
            }
        ),
        base_agent.critic,
        actor_bounder=ObGDBounding(kappa=1.5),
    )

    reconstructed = HordeActorCriticAgent.from_config(agent.to_config())

    assert reconstructed.config == agent.config
    assert reconstructed.config.actor_td_error_clip == 0.75
    assert reconstructed.critic.n_demons == 2
    assert isinstance(reconstructed.actor_bounder, ObGDBounding)
    assert TopLevelHordeActorCriticAgent is HordeActorCriticAgent
    assert CoreHordeActorCriticAgent is HordeActorCriticAgent


def test_horde_actor_critic_update_is_jittable() -> None:
    agent = _make_agent(n_demons=2)
    state = agent.init(feature_dim=2, key=jr.key(2)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    update = jax.jit(agent.update)
    result = update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array([0.5], dtype=jnp.float32),
    )

    chex.assert_shape(result.policy, (2,))
    chex.assert_shape(result.critic_result.td_errors, (2,))
    assert int(result.state.step_count) == 1


def test_run_horde_actor_critic_from_arrays_scan() -> None:
    agent = _make_agent(n_demons=2)
    state = agent.init(feature_dim=2, key=jr.key(3))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    aux = jnp.array([[0.5], [1.0], [-0.5]], dtype=jnp.float32)
    actions = jnp.array([0, 1, 0], dtype=jnp.int32)

    result = run_horde_actor_critic_from_arrays(
        agent,
        state,
        observations,
        rewards,
        next_observations,
        actions=actions,
        auxiliary_cumulants=aux,
    )

    chex.assert_shape(result.actions, (3,))
    chex.assert_shape(result.policies, (3, 2))
    chex.assert_shape(result.values, (3,))
    chex.assert_shape(result.td_errors, (3,))
    chex.assert_shape(result.critic_td_errors, (3, 2))
    assert int(result.state.step_count) == 3
    chex.assert_tree_all_finite((result.policies, result.values, result.td_errors))


def test_horde_actor_critic_explicit_discount_controls_value_target() -> None:
    agent = _make_agent(n_demons=1)
    state = agent.init(feature_dim=2, key=jr.key(4)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    # Make the next-state value non-zero so the explicit discount changes the
    # Horde target.
    head_weights = state.critic_state.head_params.weights
    critic_state = state.critic_state.replace(  # type: ignore[attr-defined]
        head_params=state.critic_state.head_params.replace(  # type: ignore[attr-defined]
            weights=(head_weights[0].at[0, 1].set(2.0), *head_weights[1:])
        )
    )
    state = state.replace(critic_state=critic_state)  # type: ignore[attr-defined]

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        discount=jnp.array(0.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.critic_result.td_targets[0],
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(result.state.actor_trace_weights, jnp.zeros((2, 2)))


def test_horde_actor_critic_actor_bounder_hook_runs() -> None:
    base_agent = _make_agent(n_demons=1)
    agent = HordeActorCriticAgent(
        base_agent.config,
        base_agent.critic,
        actor_bounder=ObGDBounding(kappa=10.0),
    )
    state = agent.init(feature_dim=2, key=jr.key(5)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([10.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        reward=jnp.array(10.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert float(result.bound_metric) < 1.0
    chex.assert_tree_all_finite((result.state.actor_weights, result.bound_metric))


def test_horde_actor_critic_infinite_reward_with_obgd_does_not_poison_actor() -> None:
    """Inf TD error zeros the ObGD step, then td_error*step is 0*inf=NaN."""
    base_agent = _make_agent()
    agent = HordeActorCriticAgent(
        base_agent.config,
        base_agent.critic,
        actor_bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=2, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)

    poisoned = agent.update(
        state,
        reward=jnp.array(jnp.inf, dtype=jnp.float32),
        observation=next_obs,
    )

    assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_weights)))
    assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_bias)))
    chex.assert_trees_all_close(poisoned.state.actor_weights, state.actor_weights)
    _assert_state_unchanged(poisoned.state, state)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_tree_all_finite(poisoned.policy)
    chex.assert_tree_all_finite(poisoned.critic_result.predictions)

    recovered = agent.update(
        poisoned.state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(jnp.all(jnp.isfinite(recovered.state.actor_weights)))
    assert bool(jnp.all(jnp.isfinite(recovered.state.actor_bias)))
    assert bool(recovered.update_applied)


def test_qhorde_actor_critic_updates_only_taken_q_head_and_actor() -> None:
    agent = _make_qhorde_agent(n_actions=2)
    state = agent.init(feature_dim=2, key=jr.key(10)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(1, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(0.0, dtype=jnp.float32),
    )

    assert isinstance(result.state, QHordeActorCriticState)
    assert int(result.state.step_count) == 1
    chex.assert_trees_all_close(result.critic_result.td_targets[1], result.target)
    assert jnp.isnan(result.critic_result.td_targets[0])
    assert not jnp.allclose(result.state.actor_weights, state.actor_weights)
    chex.assert_tree_all_finite(
        (result.policy, result.q_values, result.next_q_values, result.td_error)
    )


def test_qhorde_infinite_reward_with_obgd_does_not_poison_actor() -> None:
    """Inf TD error zeros the ObGD step, then actor_scale*step is 0*inf=NaN."""
    base_agent = _make_qhorde_agent(n_actions=2)
    agent = QHordeActorCriticAgent(
        base_agent.config,
        base_agent.critic,
        actor_bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=2, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)

    poisoned = agent.update(
        state,
        reward=jnp.array(jnp.inf, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(0.0, dtype=jnp.float32),
    )

    assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_weights)))
    chex.assert_trees_all_close(poisoned.state.actor_weights, state.actor_weights)
    _assert_state_unchanged(poisoned.state, state)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_tree_all_finite(poisoned.policy)
    chex.assert_tree_all_finite(poisoned.critic_result.predictions)

    recovered = agent.update(
        poisoned.state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        terminated=jnp.array(0.0, dtype=jnp.float32),
    )
    assert bool(jnp.all(jnp.isfinite(recovered.state.actor_weights)))
    assert bool(recovered.update_applied)


def test_qhorde_terminal_does_not_multiply_inf_next_value() -> None:
    """gamma=0 * inf Q(s', a') is 0*inf = NaN and would freeze a terminal update."""
    agent = _make_qhorde_agent(n_actions=2)
    huge = jnp.float32(1e38)
    state = agent.init(feature_dim=2, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    poisoned_w = jnp.asarray([[huge, 0.0]], dtype=jnp.float32)
    head_params = state.critic_state.head_params.replace(weights=(poisoned_w, poisoned_w))
    state = state.replace(  # type: ignore[attr-defined]
        critic_state=state.critic_state.replace(head_params=head_params)
    )
    next_obs = jnp.array([huge, 0.0], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * (huge * huge)
    assert not bool(jnp.isfinite(raw))

    result = agent.update(
        state,
        reward=jnp.array(3.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(1.0, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.next_q_values, jnp.zeros((2,), jnp.float32))
    chex.assert_tree_all_finite(result.state.replace(rng_key=jr.key_data(result.state.rng_key)))
    chex.assert_tree_all_finite(
        (result.policy, result.q_values, result.next_q_values, result.td_error)
    )
    assert bool(jnp.all(jnp.isfinite(result.state.actor_weights)))


def test_horde_zero_discount_neutralizes_inf_next_value_diagnostic() -> None:
    agent = _make_agent()
    huge = jnp.float32(1e38)
    state = agent.init(feature_dim=2, key=jr.key(19)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    poisoned_w = jnp.asarray([[huge, 0.0]], dtype=jnp.float32)
    head_params = state.critic_state.head_params.replace(weights=(poisoned_w,))
    state = state.replace(  # type: ignore[attr-defined]
        critic_state=state.critic_state.replace(head_params=head_params)
    )

    result = agent.update(
        state,
        reward=jnp.array(3.0, dtype=jnp.float32),
        observation=jnp.array([huge, 0.0], dtype=jnp.float32),
        discount=jnp.array(0.0, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.next_value, jnp.array(0.0, jnp.float32))
    chex.assert_tree_all_finite(result.state.replace(rng_key=jr.key_data(result.state.rng_key)))
    chex.assert_tree_all_finite((result.policy, result.value, result.next_value, result.td_error))


def test_qhorde_actor_critic_auxiliary_prediction_and_terminal_trace_reset() -> None:
    agent = _make_qhorde_agent(n_actions=2, n_aux=1)
    state = agent.init(feature_dim=2, key=jr.key(11)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(1.0, dtype=jnp.float32),
        prediction_cumulants=jnp.array([0.25], dtype=jnp.float32),
    )

    chex.assert_shape(result.critic_result.td_errors, (3,))
    chex.assert_trees_all_close(
        result.critic_result.td_targets[2],
        jnp.array(0.25, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(result.state.actor_trace_weights, jnp.zeros((2, 2)))


def test_qhorde_actor_critic_config_roundtrip_and_exports() -> None:
    base_agent = _make_qhorde_agent(n_actions=2, n_aux=1)
    agent = QHordeActorCriticAgent(
        QHordeActorCriticConfig.from_config(
            {
                **base_agent.config.to_config(),
                "actor_td_error_clip": 0.5,
            }
        ),
        base_agent.critic,
        actor_bounder=ObGDBounding(kappa=1.25),
    )

    restored = QHordeActorCriticAgent.from_config(agent.to_config())

    assert restored.config == agent.config
    assert restored.critic.n_demons == 3
    assert isinstance(restored.actor_bounder, ObGDBounding)
    from alberta_framework import QHordeActorCriticAgent as TopLevelQHordeAC
    from alberta_framework.core import QHordeActorCriticAgent as CoreQHordeAC

    assert TopLevelQHordeAC is QHordeActorCriticAgent
    assert CoreQHordeAC is QHordeActorCriticAgent


def test_qhorde_actor_critic_requires_zero_gamma_for_action_heads() -> None:
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(  # type: ignore[call-arg]
                    name="q",
                    demon_type=DemonType.CONTROL,
                    gamma=0.9,
                    lamda=0.0,
                    cumulant_index=-1,
                )
            ]
        ),
        hidden_sizes=(),
    )
    with pytest.raises(ValueError, match="already-bootstrapped"):
        QHordeActorCriticAgent(
            QHordeActorCriticConfig(n_actions=1),
            critic,
        )


def test_qhorde_actor_critic_update_is_jittable() -> None:
    agent = _make_qhorde_agent(n_actions=2, n_aux=1)
    state = agent.init(feature_dim=2, key=jr.key(12)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    update = jax.jit(agent.update)
    result = update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array([0.5], dtype=jnp.float32),
    )

    chex.assert_shape(result.policy, (2,))
    assert int(result.state.step_count) == 1


def test_qhorde_actor_critic_sampled_target_uses_returned_action() -> None:
    base_agent = _make_qhorde_agent(n_actions=2)
    agent = QHordeActorCriticAgent(
        QHordeActorCriticConfig.from_config(
            {
                **base_agent.config.to_config(),
                "critic_target": "sampled_sarsa",
            }
        ),
        base_agent.critic,
    )
    state = agent.init(feature_dim=2, key=jr.key(13)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    head_weights = state.critic_state.head_params.weights
    critic_state = state.critic_state.replace(  # type: ignore[attr-defined]
        head_params=state.critic_state.head_params.replace(  # type: ignore[attr-defined]
            weights=(head_weights[0].at[0, 1].set(2.0), *head_weights[1:])
        )
    )
    state = state.replace(critic_state=critic_state)  # type: ignore[attr-defined]

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(0.0, dtype=jnp.float32),
    )

    expected = 1.0 + 0.9 * result.next_q_values[result.action]
    chex.assert_trees_all_close(result.target, expected)


def test_qhorde_actor_critic_expected_advantage_actor_update() -> None:
    base_agent = _make_qhorde_agent(n_actions=2)
    agent = QHordeActorCriticAgent(
        QHordeActorCriticConfig.from_config(
            {
                **base_agent.config.to_config(),
                "actor_update": "expected_advantage",
            }
        ),
        base_agent.critic,
    )
    state = agent.init(feature_dim=2, key=jr.key(14)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    head_weights = state.critic_state.head_params.weights
    critic_state = state.critic_state.replace(  # type: ignore[attr-defined]
        head_params=state.critic_state.head_params.replace(  # type: ignore[attr-defined]
            weights=(
                head_weights[0],
                head_weights[1].at[0, 0].set(2.0),
                *head_weights[2:],
            )
        )
    )
    state = state.replace(critic_state=critic_state)  # type: ignore[attr-defined]

    result = agent.update(
        state,
        reward=jnp.array(0.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(0.0, dtype=jnp.float32),
    )

    assert result.state.actor_weights[1, 0] > state.actor_weights[1, 0]
    assert result.state.actor_weights[0, 0] < state.actor_weights[0, 0]


# ===========================================================================
# NonlinearHordeActorCriticAgent tests
# ===========================================================================


from alberta_framework.core.horde_actor_critic import (  # noqa: E402
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearHordeActorCriticState,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    run_nonlinear_horde_actor_critic_from_arrays,
)

OBS_DIM = 8
N_ACTIONS = 3


def _make_nlhac_agent(
    hidden_sizes: tuple[int, ...] = (32,),
    n_aux: int = 0,
) -> NonlinearHordeActorCriticAgent:
    demons: list[GVFSpec] = [
        GVFSpec(  # type: ignore[call-arg]
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=0.99,
            lamda=0.0,
            cumulant_index=0,
        )
    ]
    for i in range(n_aux):
        demons.append(
            GVFSpec(  # type: ignore[call-arg]
                name=f"aux_{i}",
                demon_type=DemonType.PREDICTION,
                gamma=float(0.5 + i * 0.1),
                lamda=0.0,
                cumulant_index=0,
            )
        )
    critic = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(32,),
        step_size=0.03,
    )
    cfg = NonlinearHordeActorCriticConfig(
        n_actions=N_ACTIONS,
        hidden_sizes=hidden_sizes,
        temperature=0.5,
        actor_lamda=0.9,
    )
    return NonlinearHordeActorCriticAgent(cfg, critic)


def _init_nlhac(
    agent: NonlinearHordeActorCriticAgent,
) -> NonlinearHordeActorCriticState:
    state = agent.init(feature_dim=OBS_DIM, key=jr.key(0))
    state, _, _ = agent.start(state, jnp.zeros(OBS_DIM))
    return state


def test_nonlinear_horde_zero_discount_neutralizes_inf_next_value() -> None:
    spec = create_horde_spec(
        [
            GVFSpec(  # type: ignore[call-arg]
                name="value",
                demon_type=DemonType.PREDICTION,
                gamma=0.9,
                lamda=0.0,
                cumulant_index=-1,
            )
        ]
    )
    critic = HordeLearner(
        spec,
        hidden_sizes=(),
        step_size=0.1,
        use_layer_norm=False,
    )
    agent = NonlinearHordeActorCriticAgent(
        NonlinearHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(),
            actor_sparsity=0.0,
        ),
        critic,
    )
    huge = jnp.float32(1e38)
    state = agent.init(2, jr.key(20))
    state, _, _ = agent.start(state, jnp.array([0.0, 1.0], jnp.float32))
    poisoned_w = jnp.asarray([[huge, 0.0]], dtype=jnp.float32)
    head_params = state.critic_state.head_params.replace(weights=(poisoned_w,))
    state = state.replace(  # type: ignore[attr-defined]
        critic_state=state.critic_state.replace(head_params=head_params)
    )

    result = agent.update(
        state,
        jnp.array(3.0, jnp.float32),
        jnp.array([huge, 0.0], jnp.float32),
        discount=jnp.array(0.0, jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.next_value, jnp.array(0.0, jnp.float32))
    chex.assert_tree_all_finite(result.state.replace(rng_key=jr.key_data(result.state.rng_key)))
    chex.assert_tree_all_finite((result.policy, result.value, result.next_value, result.td_error))


class TestNonlinearHordeActorCriticConfig:
    def _simple_critic(self) -> HordeLearner:
        spec = create_horde_spec(
            [
                GVFSpec(  # type: ignore[call-arg]
                    name="v",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.0,
                    cumulant_index=0,
                )
            ]
        )
        return HordeLearner(spec, hidden_sizes=(16,))

    def test_n_actions_positive(self) -> None:
        with pytest.raises(ValueError, match="n_actions"):
            agent = NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(n_actions=0, hidden_sizes=(16,)),
                self._simple_critic(),
            )
            del agent

    def test_temperature_positive(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            critic = self._simple_critic()
            critic = HordeLearner(
                create_horde_spec(
                    [
                        GVFSpec(  # type: ignore[call-arg]
                            name="v",
                            demon_type=DemonType.PREDICTION,
                            gamma=0.9,
                            lamda=0.0,
                            cumulant_index=0,
                        )
                    ]
                ),
                hidden_sizes=(16,),
            )
            NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(16,), temperature=0.0),
                critic,
            )

    def test_actor_gradient_clip_norm_positive(self) -> None:
        with pytest.raises(ValueError, match="actor_gradient_clip_norm"):
            NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(
                    n_actions=2,
                    hidden_sizes=(16,),
                    actor_gradient_clip_norm=0.0,
                ),
                self._simple_critic(),
            )

    def test_actor_epsilon_bounds(self) -> None:
        with pytest.raises(ValueError, match="actor_epsilon"):
            NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(
                    n_actions=2,
                    hidden_sizes=(16,),
                    actor_epsilon=1.0,
                ),
                self._simple_critic(),
            )

    def test_actor_td_error_normalizer_decay_bounds(self) -> None:
        with pytest.raises(ValueError, match="actor_td_error_normalizer_decay"):
            NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(
                    n_actions=2,
                    hidden_sizes=(16,),
                    actor_td_error_normalizer_decay=1.0,
                ),
                self._simple_critic(),
            )

    def test_actor_optimizer_must_support_mlp(self) -> None:
        # An optimizer without the shape-generic MLP hooks (init_for_shape /
        # update_from_gradient) must be rejected at construction time rather
        # than raising NotImplementedError deep inside a jitted update.
        with pytest.raises(ValueError, match="does not support the MLP"):
            NonlinearHordeActorCriticAgent(
                NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(16,)),
                self._simple_critic(),
                actor_optimizer=ObGD(),  # type: ignore[arg-type]
            )

    def test_config_roundtrip(self) -> None:
        cfg = NonlinearHordeActorCriticConfig(
            n_actions=4,
            hidden_sizes=(64, 32),
            temperature=0.3,
            actor_gradient_clip_norm=0.25,
            actor_epsilon=0.05,
            actor_td_error_normalizer_decay=0.99,
        )
        restored = NonlinearHordeActorCriticConfig.from_config(cfg.to_config())
        assert restored.n_actions == 4
        assert restored.hidden_sizes == (64, 32)
        assert restored.temperature == pytest.approx(0.3)
        assert restored.actor_gradient_clip_norm == pytest.approx(0.25)
        assert restored.actor_epsilon == pytest.approx(0.05)
        assert restored.actor_td_error_normalizer_decay == pytest.approx(0.99)


class TestNonlinearHordeActorCriticInit:
    def test_actor_head_shape(self) -> None:
        agent = _make_nlhac_agent(hidden_sizes=(32,))
        state = agent.init(OBS_DIM, jr.key(0))
        chex.assert_shape(state.actor_head_w, (N_ACTIONS, 32))
        chex.assert_shape(state.actor_head_b, (N_ACTIONS,))

    def test_actor_trunk_shape(self) -> None:
        agent = _make_nlhac_agent(hidden_sizes=(64, 32))
        state = agent.init(OBS_DIM, jr.key(0))
        assert len(state.actor_trunk.weights) == 2
        chex.assert_shape(state.actor_trunk.weights[0], (64, OBS_DIM))
        chex.assert_shape(state.actor_trunk.weights[1], (32, 64))

    def test_traces_zero_at_init(self) -> None:
        agent = _make_nlhac_agent(hidden_sizes=(32,))
        state = agent.init(OBS_DIM, jr.key(0))
        chex.assert_trees_all_close(state.actor_head_trace_w, jnp.zeros((N_ACTIONS, 32)))

    def test_linear_actor_no_trunk(self) -> None:
        agent = _make_nlhac_agent(hidden_sizes=())
        state = agent.init(OBS_DIM, jr.key(0))
        assert len(state.actor_trunk.weights) == 0
        chex.assert_shape(state.actor_head_w, (N_ACTIONS, OBS_DIM))


class TestNonlinearHordeActorCriticUpdate:
    def test_returns_result(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert isinstance(result.state, NonlinearHordeActorCriticState)

    def test_step_count_increments(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        result = agent.update(state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert int(result.state.step_count) == 1

    def test_td_error_finite(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert jnp.isfinite(result.td_error)

    def test_infinite_reward_with_obgd_does_not_poison_actor(self) -> None:
        """Inf TD error zeros the ObGD step, then td_error*step is 0*inf=NaN."""
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.0,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(),
            step_size=0.03,
        )
        agent = NonlinearHordeActorCriticAgent(
            NonlinearHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
            ),
            critic,
            actor_optimizer=Autostep(initial_step_size=0.5),
            actor_bounder=ObGDBounding(kappa=2.0),
        )
        observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
        next_obs = jnp.array([0.5, 1.0], dtype=jnp.float32)
        state = agent.init(2, jr.key(27))
        state, _, _ = agent.start(state, observation)

        poisoned = agent.update(state, jnp.array(jnp.inf, dtype=jnp.float32), next_obs)
        assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_head_w)))
        chex.assert_trees_all_close(poisoned.state.actor_head_w, state.actor_head_w)
        _assert_state_unchanged(poisoned.state, state)
        assert not bool(poisoned.update_applied)
        assert float(poisoned.td_error) == 0.0
        chex.assert_tree_all_finite(poisoned.policy)
        chex.assert_tree_all_finite(poisoned.critic_result.predictions)

        recovered = agent.update(poisoned.state, jnp.array(1.0, dtype=jnp.float32), observation)
        assert bool(jnp.all(jnp.isfinite(recovered.state.actor_head_w)))
        assert bool(recovered.update_applied)

    def test_actor_td_error_normalizer_updates(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.99,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(32,),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(
            n_actions=N_ACTIONS,
            hidden_sizes=(32,),
            actor_td_error_normalizer_decay=0.9,
        )
        agent = NonlinearHordeActorCriticAgent(cfg, critic)
        state = agent.init(OBS_DIM, jr.key(3))
        obs = jr.normal(jr.key(4), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        result = agent.update(state, jnp.array(1.0), obs)
        assert float(result.state.actor_td_error_normalizer) > 0.0

    def test_zero_td_error_normalizer_decay_does_not_multiply_inf_ema(self) -> None:
        """decay=0 times leftover inf actor TD-error EMA is NaN."""
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.99,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(8,),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(
            n_actions=N_ACTIONS,
            hidden_sizes=(8,),
            actor_td_error_normalizer_decay=0.0,
        )
        agent = NonlinearHordeActorCriticAgent(cfg, critic)
        state = agent.init(OBS_DIM, jr.key(5))
        obs = jr.normal(jr.key(6), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        state = state.replace(  # type: ignore[attr-defined]
            actor_td_error_normalizer=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = agent.update(state, jnp.array(1.0, dtype=jnp.float32), obs)
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.state.actor_td_error_normalizer)

    def test_policy_sums_to_one(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        result = agent.update(state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        chex.assert_trees_all_close(jnp.sum(result.policy), jnp.array(1.0), atol=1e-5)

    def test_actor_weights_update(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.99,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(32,),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(n_actions=N_ACTIONS, hidden_sizes=(32,))
        agent = NonlinearHordeActorCriticAgent(
            cfg, critic, actor_optimizer=Autostep(initial_step_size=1.0)
        )
        state = agent.init(OBS_DIM, jr.key(0))
        obs = jr.normal(jr.key(42), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        before = state.actor_head_w.copy()
        result = agent.update(state, jnp.array(1.0), obs)
        after = result.state.actor_head_w
        assert not jnp.allclose(before, after, atol=1e-6)

    def test_actor_gradient_clip_limits_new_trace_norm(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.99,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(32,),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(
            n_actions=N_ACTIONS,
            hidden_sizes=(32,),
            actor_gradient_clip_norm=0.05,
        )
        agent = NonlinearHordeActorCriticAgent(
            cfg, critic, actor_optimizer=Autostep(initial_step_size=0.01)
        )
        state = agent.init(OBS_DIM, jr.key(9))
        obs = 100.0 * jr.normal(jr.key(10), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        result = agent.update(state, jnp.array(1.0), obs)
        trace_norm = jnp.sqrt(
            jnp.sum(jnp.square(result.state.actor_head_trace_w))
            + jnp.sum(jnp.square(result.state.actor_head_trace_b))
            + sum(jnp.sum(jnp.square(trace)) for trace in result.state.actor_trunk_traces)
        )
        assert float(trace_norm) <= 0.0501

    def test_obgd_bounds_raw_autostep_step_before_td_error_application(self) -> None:
        """Large TD errors must enter the ObGD denominator exactly once."""
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.0,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(),
            actor_sparsity=0.0,
        )
        optimizer = Autostep(initial_step_size=0.5)
        unbounded = NonlinearHordeActorCriticAgent(
            cfg,
            critic,
            actor_optimizer=optimizer,
        )
        kappa = 0.5
        bounded = NonlinearHordeActorCriticAgent(
            cfg,
            critic,
            actor_optimizer=optimizer,
            actor_bounder=ObGDBounding(kappa=kappa),
        )
        observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
        initial = unbounded.init(2, jr.key(27))
        initial, _, _ = unbounded.start(initial, observation)
        bounded_initial = bounded.init(2, jr.key(27))
        bounded_initial, _, _ = bounded.start(bounded_initial, observation)

        unbounded_result = unbounded.update(
            initial,
            jnp.asarray(10.0, dtype=jnp.float32),
            observation,
        )
        bounded_result = bounded.update(
            bounded_initial,
            jnp.asarray(10.0, dtype=jnp.float32),
            observation,
        )
        td_error = float(unbounded_result.td_error)
        assert abs(td_error) > 1.0

        full_steps = (
            unbounded_result.state.actor_head_w - initial.actor_head_w,
            unbounded_result.state.actor_head_b - initial.actor_head_b,
        )
        raw_l1 = sum(float(jnp.sum(jnp.abs(step / td_error))) for step in full_steps)
        expected_scale = 1.0 / max(kappa * max(abs(td_error), 1.0) * raw_l1, 1.0)
        assert float(bounded_result.bound_metric) == pytest.approx(
            expected_scale,
            rel=1e-5,
        )
        chex.assert_trees_all_close(
            bounded_result.state.actor_head_w - bounded_initial.actor_head_w,
            expected_scale * full_steps[0],
            rtol=1e-5,
            atol=1e-6,
        )
        chex.assert_trees_all_close(
            bounded_result.state.actor_head_b - bounded_initial.actor_head_b,
            expected_scale * full_steps[1],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_trunk_weights_update(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.99,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(32,),
            step_size=0.03,
        )
        cfg = NonlinearHordeActorCriticConfig(n_actions=N_ACTIONS, hidden_sizes=(32,))
        agent = NonlinearHordeActorCriticAgent(
            cfg, critic, actor_optimizer=Autostep(initial_step_size=1.0)
        )
        state = agent.init(OBS_DIM, jr.key(7))
        obs = jr.normal(jr.key(7), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        before = state.actor_trunk.weights[0].copy()
        result = agent.update(state, jnp.array(1.0), obs)
        after = result.state.actor_trunk.weights[0]
        assert not jnp.allclose(before, after, atol=1e-6)

    def test_jittable(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        f = jax.jit(agent.update)
        result = f(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert jnp.isfinite(result.td_error)


class TestNonlinearHordeActorCriticScan:
    def test_scan_shapes(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        n_steps = 15
        obs = jnp.zeros((n_steps, OBS_DIM))
        rews = jnp.ones(n_steps)
        result = run_nonlinear_horde_actor_critic_from_arrays(agent, state, obs, rews, obs)
        chex.assert_shape(result.actions, (n_steps,))
        chex.assert_shape(result.values, (n_steps,))
        chex.assert_shape(result.td_errors, (n_steps,))
        chex.assert_shape(result.policies, (n_steps, N_ACTIONS))

    def test_scan_td_errors_finite(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        n_steps = 20
        obs = jr.normal(jr.key(5), (n_steps, OBS_DIM))
        rews = jr.normal(jr.key(6), (n_steps,))
        result = run_nonlinear_horde_actor_critic_from_arrays(agent, state, obs, rews, obs)
        chex.assert_tree_all_finite(result.td_errors)

    def test_scan_step_count_final(self) -> None:
        agent = _make_nlhac_agent()
        state = _init_nlhac(agent)
        n_steps = 10
        obs = jnp.zeros((n_steps, OBS_DIM))
        result = run_nonlinear_horde_actor_critic_from_arrays(
            agent, state, obs, jnp.zeros(n_steps), obs
        )
        assert int(result.state.step_count) == n_steps

    def test_200_step_fineness(self) -> None:
        agent = _make_nlhac_agent(hidden_sizes=(32,))
        state = _init_nlhac(agent)
        n_steps = 200
        obs = jr.normal(jr.key(99), (n_steps, OBS_DIM))
        rews = jr.normal(jr.key(100), (n_steps,))
        result = run_nonlinear_horde_actor_critic_from_arrays(agent, state, obs, rews, obs)
        chex.assert_tree_all_finite(result.td_errors)
        assert int(result.state.step_count) == n_steps

    def test_auxiliary_demons_work(self) -> None:
        agent = _make_nlhac_agent(n_aux=2)
        state = _init_nlhac(agent)
        n_steps = 10
        obs = jnp.zeros((n_steps, OBS_DIM))
        aux = jnp.ones((n_steps, 2))
        result = run_nonlinear_horde_actor_critic_from_arrays(
            agent, state, obs, jnp.ones(n_steps), obs, auxiliary_cumulants=aux
        )
        chex.assert_shape(result.critic_td_errors, (n_steps, 3))


class TestNonlinearHordeActorCriticExport:
    def test_exported_from_core(self) -> None:
        from alberta_framework.core import NonlinearHordeActorCriticAgent as Cls

        assert Cls is NonlinearHordeActorCriticAgent

    def test_to_config_roundtrip(self) -> None:
        agent = _make_nlhac_agent()
        cfg = agent.to_config()
        restored = NonlinearHordeActorCriticAgent.from_config(cfg)
        assert restored.config.n_actions == agent.config.n_actions
        assert restored.config.hidden_sizes == agent.config.hidden_sizes


class TestNonlinearQHordeActorCritic:
    def _agent(self) -> NonlinearQHordeActorCriticAgent:
        demons = [
            GVFSpec(  # type: ignore[call-arg]
                name=f"q_{action}",
                demon_type=DemonType.CONTROL,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
            for action in range(N_ACTIONS)
        ]
        critic = HordeLearner(
            create_horde_spec(demons),
            hidden_sizes=(16,),
            step_size=0.03,
        )
        cfg = NonlinearQHordeActorCriticConfig(
            n_actions=N_ACTIONS,
            hidden_sizes=(16,),
            actor_td_error_clip=1.0,
            actor_gradient_clip_norm=1.0,
        )
        return NonlinearQHordeActorCriticAgent(
            cfg,
            critic,
            actor_optimizer=Autostep(initial_step_size=0.01),
        )

    def test_actor_optimizer_must_support_mlp(self) -> None:
        demons = [
            GVFSpec(  # type: ignore[call-arg]
                name=f"q_{action}",
                demon_type=DemonType.CONTROL,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
            for action in range(N_ACTIONS)
        ]
        critic = HordeLearner(
            create_horde_spec(demons),
            hidden_sizes=(16,),
            step_size=0.03,
        )
        with pytest.raises(ValueError, match="does not support the MLP"):
            NonlinearQHordeActorCriticAgent(
                NonlinearQHordeActorCriticConfig(n_actions=N_ACTIONS, hidden_sizes=(16,)),
                critic,
                actor_optimizer=ObGD(),  # type: ignore[arg-type]
            )

    def test_update_returns_finite_q_values(self) -> None:
        agent = self._agent()
        state = agent.init(OBS_DIM, jr.key(11))
        obs = jr.normal(jr.key(12), (OBS_DIM,))
        state, _, _ = agent.start(state, obs)
        result = agent.update(
            state,
            jnp.array(1.0),
            jr.normal(jr.key(13), (OBS_DIM,)),
            jnp.array(0.0),
        )
        chex.assert_shape(result.q_values, (N_ACTIONS,))
        chex.assert_tree_all_finite(result.q_values)
        assert int(result.state.step_count) == 1

    def test_terminal_neutralizes_inf_next_q_diagnostic(self) -> None:
        demons = [
            GVFSpec(  # type: ignore[call-arg]
                name=f"q_{action}",
                demon_type=DemonType.CONTROL,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
            for action in range(2)
        ]
        critic = HordeLearner(
            create_horde_spec(demons),
            hidden_sizes=(),
            step_size=0.1,
            use_layer_norm=False,
        )
        agent = NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
            ),
            critic,
        )
        huge = jnp.float32(1e38)
        state = agent.init(2, jr.key(21))
        state, _, _ = agent.start(state, jnp.array([0.0, 1.0], jnp.float32))
        poisoned_w = jnp.asarray([[huge, 0.0]], dtype=jnp.float32)
        head_params = state.critic_state.head_params.replace(weights=(poisoned_w, poisoned_w))
        state = state.replace(  # type: ignore[attr-defined]
            critic_state=state.critic_state.replace(head_params=head_params)
        )

        result = agent.update(
            state,
            jnp.array(3.0, jnp.float32),
            jnp.array([huge, 0.0], jnp.float32),
            jnp.array(1.0, jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_trees_all_equal(result.next_q_values, jnp.zeros((2,), jnp.float32))
        chex.assert_tree_all_finite(result.state.replace(rng_key=jr.key_data(result.state.rng_key)))
        chex.assert_tree_all_finite(
            (result.policy, result.q_values, result.next_q_values, result.td_error)
        )

    def test_infinite_reward_with_obgd_does_not_poison_actor(self) -> None:
        """Inf TD error zeros the ObGD step, then actor_signal*step is 0*inf=NaN."""
        demons = [
            GVFSpec(  # type: ignore[call-arg]
                name=f"q_{action}",
                demon_type=DemonType.CONTROL,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
            for action in range(2)
        ]
        critic = HordeLearner(
            create_horde_spec(demons),
            hidden_sizes=(),
            step_size=0.03,
        )
        agent = NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
            ),
            critic,
            actor_optimizer=Autostep(initial_step_size=0.5),
            actor_bounder=ObGDBounding(kappa=2.0),
        )
        observation = jnp.array([1.0, 0.0], dtype=jnp.float32)
        next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        state = agent.init(2, jr.key(18))
        state, _, _ = agent.start(state, observation)

        poisoned = agent.update(
            state,
            jnp.array(jnp.inf, dtype=jnp.float32),
            next_obs,
            jnp.array(0.0, dtype=jnp.float32),
        )
        assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_head_w)))
        chex.assert_trees_all_close(poisoned.state.actor_head_w, state.actor_head_w)
        _assert_state_unchanged(poisoned.state, state)
        assert not bool(poisoned.update_applied)
        assert float(poisoned.td_error) == 0.0
        chex.assert_tree_all_finite(poisoned.policy)
        chex.assert_tree_all_finite(poisoned.critic_result.predictions)

        recovered = agent.update(
            poisoned.state,
            jnp.array(1.0, dtype=jnp.float32),
            observation,
            jnp.array(0.0, dtype=jnp.float32),
        )
        assert bool(jnp.all(jnp.isfinite(recovered.state.actor_head_w)))
        assert bool(recovered.update_applied)

    def test_requires_control_heads(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="v",
                        demon_type=DemonType.PREDICTION,
                        gamma=0.9,
                        lamda=0.0,
                        cumulant_index=0,
                    )
                ]
            ),
            hidden_sizes=(16,),
        )
        with pytest.raises(ValueError, match="control demon"):
            NonlinearQHordeActorCriticAgent(
                NonlinearQHordeActorCriticConfig(n_actions=1),
                critic,
            )

    def test_requires_zero_gamma_for_prebootstrapped_action_heads(self) -> None:
        critic = HordeLearner(
            create_horde_spec(
                [
                    GVFSpec(  # type: ignore[call-arg]
                        name="q",
                        demon_type=DemonType.CONTROL,
                        gamma=0.9,
                        lamda=0.0,
                        cumulant_index=-1,
                    )
                ]
            ),
            hidden_sizes=(16,),
        )
        with pytest.raises(ValueError, match="already-bootstrapped"):
            NonlinearQHordeActorCriticAgent(
                NonlinearQHordeActorCriticConfig(n_actions=1),
                critic,
            )

    def test_config_roundtrip(self) -> None:
        cfg = NonlinearQHordeActorCriticConfig(
            n_actions=4,
            hidden_sizes=(32, 16),
            actor_gradient_clip_norm=0.25,
            critic_target="sampled_sarsa",
            actor_update="expected_advantage",
        )
        restored = NonlinearQHordeActorCriticConfig.from_config(cfg.to_config())
        assert restored.n_actions == 4
        assert restored.hidden_sizes == (32, 16)
        assert restored.actor_gradient_clip_norm == pytest.approx(0.25)
        assert restored.critic_target == "sampled_sarsa"
        assert restored.actor_update == "expected_advantage"

    def test_expected_advantage_actor_update_moves_toward_better_action(self) -> None:
        demons = [
            GVFSpec(  # type: ignore[call-arg]
                name=f"q_{action}",
                demon_type=DemonType.CONTROL,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=-1,
            )
            for action in range(2)
        ]
        critic = HordeLearner(
            create_horde_spec(demons),
            hidden_sizes=(),
            step_size=0.03,
        )
        cfg = NonlinearQHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(),
            actor_sparsity=0.0,
            actor_update="expected_advantage",
        )
        agent = NonlinearQHordeActorCriticAgent(
            cfg,
            critic,
            actor_optimizer=Autostep(initial_step_size=0.1),
        )
        obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
        state = agent.init(2, jr.key(18)).replace(  # type: ignore[attr-defined]
            last_observation=obs,
            last_action=jnp.array(0, dtype=jnp.int32),
        )
        head_weights = state.critic_state.head_params.weights
        critic_state = state.critic_state.replace(  # type: ignore[attr-defined]
            head_params=state.critic_state.head_params.replace(  # type: ignore[attr-defined]
                weights=(
                    head_weights[0],
                    head_weights[1].at[0, 0].set(2.0),
                )
            )
        )
        state = state.replace(critic_state=critic_state)  # type: ignore[attr-defined]

        before = agent.policy(state, obs)
        result = agent.update(
            state,
            jnp.array(0.0, dtype=jnp.float32),
            obs,
            jnp.array(0.0, dtype=jnp.float32),
        )
        after = agent.policy(result.state, obs)

        assert after[1] > before[1]
        assert after[0] < before[0]


def test_horde_actor_critic_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        HordeActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        HordeActorCriticConfig(n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_head_index"):
        HordeActorCriticConfig(n_actions=2, value_head_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actor_step_size"):
        HordeActorCriticConfig(n_actions=2, actor_step_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_actions"):
        QHordeActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamma"):
        QHordeActorCriticConfig(n_actions=2, gamma=True)  # type: ignore[arg-type]


def test_horde_actor_critic_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = HordeActorCriticConfig(n_actions=np.int32(3), value_head_index=np.uint16(0))
    assert type(cfg.n_actions) is int
    assert type(cfg.value_head_index) is int
    assert cfg.n_actions == 3
    assert cfg.value_head_index == 0

    q_cfg = QHordeActorCriticConfig(n_actions=np.int64(4))
    assert type(q_cfg.n_actions) is int
    assert q_cfg.n_actions == 4


def test_nonlinear_horde_actor_critic_configs_reject_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        NonlinearHordeActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes"):
        NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=[32])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes"):
        NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actor_lamda"):
        NonlinearHordeActorCriticConfig(n_actions=2, actor_lamda=1.0 + 1e-6)

    with pytest.raises(ValueError, match="n_actions"):
        NonlinearQHordeActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes"):
        NonlinearQHordeActorCriticConfig(n_actions=2, hidden_sizes=(np.int32(0),))


def test_nonlinear_horde_actor_critic_configs_accept_and_canonicalizes_numpy_integers() -> None:
    cfg = NonlinearHordeActorCriticConfig(
        n_actions=np.int32(2),
        value_head_index=np.uint16(0),
        hidden_sizes=(np.int32(32), np.int64(16)),
    )
    assert type(cfg.n_actions) is int
    assert type(cfg.value_head_index) is int
    assert type(cfg.hidden_sizes[0]) is int
    assert type(cfg.hidden_sizes[1]) is int
    assert cfg.hidden_sizes == (32, 16)

    q_cfg = NonlinearQHordeActorCriticConfig(
        n_actions=np.int32(3),
        hidden_sizes=(np.int64(64),),
    )
    assert type(q_cfg.n_actions) is int
    assert type(q_cfg.hidden_sizes[0]) is int
    assert q_cfg.hidden_sizes == (64,)


@pytest.mark.parametrize(
    "config",
    [
        HordeActorCriticConfig(n_actions=2),
        QHordeActorCriticConfig(n_actions=2),
        NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=()),
        NonlinearQHordeActorCriticConfig(n_actions=2, hidden_sizes=()),
    ],
)
def test_actor_configs_require_exact_complete_serialized_dicts(config: object) -> None:
    payload = config.to_config()  # type: ignore[union-attr]
    config_type = type(config)
    assert config_type.from_config(payload) == config

    class HostileDict(dict[str, object]):
        def __iter__(self):
            raise AssertionError("hostile iteration must not run")

        def __repr__(self) -> str:
            raise AssertionError("hostile repr must not run")

    with pytest.raises(ValueError, match="exact built-in dict"):
        config_type.from_config(HostileDict(payload))
    with pytest.raises(ValueError, match="serialized schema"):
        first_key = next(iter(payload))
        config_type.from_config(
            {key: value for key, value in payload.items() if key != first_key}
        )
    with pytest.raises(ValueError, match="serialized schema"):
        config_type.from_config({**payload, "unknown": 1})


@pytest.mark.parametrize(
    "config_type",
    [NonlinearHordeActorCriticConfig, NonlinearQHordeActorCriticConfig],
)
def test_nonlinear_actor_configs_require_exact_bool_and_json_list(config_type: type) -> None:
    for value in (0, 1, np.bool_(True), object()):
        with pytest.raises(ValueError, match="use_layer_norm"):
            config_type(n_actions=2, hidden_sizes=(), use_layer_norm=value)
    payload = config_type(n_actions=2, hidden_sizes=()).to_config()
    with pytest.raises(ValueError, match="serialized hidden_sizes"):
        config_type.from_config({**payload, "hidden_sizes": ()})


@pytest.mark.parametrize("config_type", [QHordeActorCriticConfig, NonlinearQHordeActorCriticConfig])
def test_q_actor_configs_reject_enum_spoofs_without_hooks(config_type: type) -> None:
    class EqualSpoof:
        def __hash__(self) -> int:
            raise AssertionError("hash must not run")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("equality must not run")

        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    for field in ("critic_target", "actor_update"):
        with pytest.raises(ValueError, match=field):
            config_type(n_actions=2, hidden_sizes=(), **{field: EqualSpoof()}) if (
                config_type is NonlinearQHordeActorCriticConfig
            ) else config_type(n_actions=2, **{field: EqualSpoof()})


def test_actor_resource_budgets_preflight_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear = HordeActorCriticConfig(n_actions=2)
    assert linear.actor_resource_budget(3) == {
        "parameter_scalars": 8,
        "float32_state_scalars": 19,
        "state_scalars": 23,
        "state_nbytes": 92,
    }
    nonlinear = NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(3,))
    budget = nonlinear.actor_resource_budget(4)
    assert budget["parameter_scalars"] == 23
    assert budget["optimizer_tensor_count"] == 4
    assert budget["float32_state_scalars"] == 128
    assert budget["state_nbytes"] == 528

    with pytest.raises(ValueError, match="derived actor"):
        HordeActorCriticConfig(n_actions=2**31 - 1)
    with pytest.raises(ValueError, match="hidden_product"):
        NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(50_000, 50_000))

    agent = _make_agent()
    monkeypatch.setattr(
        "alberta_framework.core.horde_actor_critic.jnp.zeros",
        lambda *args, **kwargs: pytest.fail("allocation must not run"),
    )
    for feature_dim in (True, 2**31 - 1):
        with pytest.raises(ValueError):
            agent.init(feature_dim, jr.key(0))  # type: ignore[arg-type]


def test_run_horde_actor_critic_from_arrays_rejects_non_integer_actions() -> None:
    from alberta_framework.core.horde_actor_critic import run_horde_actor_critic_from_arrays

    agent = _make_agent()
    state = agent.init(feature_dim=4, key=jr.key(0))
    obs = jnp.zeros((2, 4), dtype=jnp.float32)
    rewards = jnp.zeros(2, dtype=jnp.float32)
    next_obs = jnp.zeros((2, 4), dtype=jnp.float32)

    # Fractional float actions rejected
    with pytest.raises(ValueError, match="actions must have an integer dtype"):
        run_horde_actor_critic_from_arrays(
            agent, state, obs, rewards, next_obs, actions=jnp.array([1.75, 0.25], dtype=jnp.float32)
        )

    # Boolean actions rejected
    with pytest.raises(ValueError, match="actions must have an integer dtype"):
        run_horde_actor_critic_from_arrays(
            agent, state, obs, rewards, next_obs, actions=jnp.array([True, False])
        )

    # Valid integer actions accepted
    result = run_horde_actor_critic_from_arrays(
        agent, state, obs, rewards, next_obs, actions=jnp.array([1, 0], dtype=jnp.int32)
    )
    assert result.actions.shape == (2,)


def test_horde_array_action_contract_is_jittable_and_invalid_values_are_atomic() -> None:
    agent = _make_agent()
    state = agent.init(feature_dim=4, key=jr.key(0))
    obs = jnp.zeros((1, 4), dtype=jnp.float32)
    rewards = jnp.ones(1, dtype=jnp.float32)

    compiled = jax.jit(
        lambda initial_state, fixed_actions: run_horde_actor_critic_from_arrays(
            agent,
            initial_state,
            obs,
            rewards,
            obs,
            actions=fixed_actions,
        )
    )
    valid = compiled(state, jnp.asarray([1], dtype=jnp.int32))
    assert bool(valid.updates_applied[0])

    invalid = compiled(state, jnp.asarray([2], dtype=jnp.int32))
    _assert_state_unchanged(invalid.state, state)
    assert not bool(jnp.any(invalid.updates_applied))
    assert not bool(jnp.any(invalid.actions))
    assert not bool(jnp.any(invalid.policies))
    assert not bool(jnp.any(invalid.values))
    assert not bool(jnp.any(invalid.td_errors))
    assert not bool(jnp.any(invalid.critic_td_errors))

    for bad_actions in (
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([True], dtype=jnp.bool_),
        jnp.asarray([[0]], dtype=jnp.int32),
    ):
        with pytest.raises(ValueError):
            compiled(state, bad_actions)
