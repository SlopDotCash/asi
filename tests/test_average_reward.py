"""Tests for average-reward Step 5/6 primitives."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import (
    DifferentialSARSAAgent as TopLevelDifferentialSARSAAgent,
)
from alberta_framework.core import DifferentialTDLearner as CoreDifferentialTDLearner
from alberta_framework.core.average_reward import (
    _AVERAGE_REWARD_SEQUENCE_MAX_STEPS,
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
    AverageRewardHordeLearner,
    DifferentialGTDConfig,
    DifferentialGTDLearner,
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialTDConfig,
    DifferentialTDLearner,
    _require_avg_reward_matching_length,
    _require_avg_reward_sequence_length,
    run_average_reward_horde_actor_critic_from_arrays,
    run_average_reward_horde_from_arrays,
    run_differential_gtd_from_arrays,
    run_differential_sarsa_from_arrays,
    run_differential_td_from_arrays,
)


def test_differential_td_config_and_top_level_exports() -> None:
    config = DifferentialTDConfig(
        step_size=0.1,
        average_reward_step_size=0.02,
        trace_decay=0.5,
    )
    learner = DifferentialTDLearner.from_config(DifferentialTDLearner(config).to_config())

    assert learner.config == config
    assert CoreDifferentialTDLearner is DifferentialTDLearner
    assert TopLevelDifferentialSARSAAgent is DifferentialSARSAAgent


def test_differential_td_error_matches_average_reward_target() -> None:
    learner = DifferentialTDLearner(DifferentialTDConfig(step_size=0.0))
    state = learner.init(2).replace(  # type: ignore[attr-defined]
        weights=jnp.array([1.0, -1.0], dtype=jnp.float32),
        bias=jnp.array(0.5, dtype=jnp.float32),
        average_reward=jnp.array(0.25, dtype=jnp.float32),
    )
    obs = jnp.array([2.0, 1.0], dtype=jnp.float32)
    next_obs = jnp.array([0.0, 3.0], dtype=jnp.float32)

    td_error = learner.td_error(
        state,
        obs,
        jnp.array(1.25, dtype=jnp.float32),
        next_obs,
    )

    chex.assert_trees_all_close(td_error, jnp.array(-3.0, dtype=jnp.float32))


def test_differential_td_update_moves_average_reward_and_is_jittable() -> None:
    learner = DifferentialTDLearner(
        DifferentialTDConfig(
            step_size=0.1,
            average_reward_step_size=0.2,
            trace_decay=0.0,
        )
    )
    state = learner.init(1)
    update = jax.jit(learner.update)
    result = update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.average_reward,
        jnp.array(0.2, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(result)


def test_differential_td_infinite_reward_does_not_poison_weights() -> None:
    """Inf reward is 0*inf = NaN on a silent feature and inf rbar.

    Differential SARSA already refuses non-finite rewards. Hold the
    previous finite state so a later finite reward can still learn.
    """
    learner = DifferentialTDLearner(
        DifferentialTDConfig(step_size=0.1, average_reward_step_size=0.2, trace_decay=0.0)
    )
    state = learner.init(2)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)

    poisoned = learner.update(state, obs, jnp.array(jnp.inf, dtype=jnp.float32), nxt)
    chex.assert_trees_all_close(poisoned.state.weights, state.weights)
    chex.assert_trees_all_close(poisoned.state.average_reward, state.average_reward)
    assert int(poisoned.state.step_count) == int(state.step_count)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_trees_all_close(poisoned.metrics, jnp.zeros_like(poisoned.metrics))

    recovered = learner.update(poisoned.state, obs, jnp.array(1.0, dtype=jnp.float32), nxt)
    chex.assert_tree_all_finite(recovered.state.weights)
    chex.assert_tree_all_finite(recovered.state.average_reward)
    assert int(recovered.state.step_count) == int(state.step_count) + 1
    assert bool(recovered.update_applied)


def test_differential_gtd_infinite_reward_does_not_poison_weights() -> None:
    """Same 0*inf hole on the GTD primary/secondary products."""
    learner = DifferentialGTDLearner(
        DifferentialGTDConfig(value_step_size=0.1, secondary_step_size=0.01)
    )
    state = learner.init(2)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    rho = jnp.array(1.0, dtype=jnp.float32)

    poisoned = learner.update(state, obs, jnp.array(jnp.inf, dtype=jnp.float32), nxt, rho)
    chex.assert_trees_all_close(poisoned.state.weights, state.weights)
    chex.assert_trees_all_close(poisoned.state.secondary_weights, state.secondary_weights)
    chex.assert_trees_all_close(poisoned.state.average_reward, state.average_reward)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_trees_all_close(poisoned.metrics, jnp.zeros_like(poisoned.metrics))

    recovered = learner.update(poisoned.state, obs, jnp.array(1.0, dtype=jnp.float32), nxt, rho)
    chex.assert_tree_all_finite(recovered.state.weights)
    chex.assert_tree_all_finite(recovered.state.secondary_weights)
    assert bool(recovered.update_applied)


def test_average_reward_horde_infinite_cumulant_does_not_poison_rbar() -> None:
    """Inf is not NaN, so it used to keep a demon 'active'.

    MultiHead then refused the whole trunk, while rbar still added the
    inf TD error. Treat non-finite cumulants as inactive and only move
    rbar when the nested learner actually commits.
    """
    learner = AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(4,),
        sparsity=0.0,
        use_layer_norm=False,
        average_reward_step_size=0.01,
    )
    state = learner.init(3, jr.key(0))
    obs = jnp.ones(3, dtype=jnp.float32)
    nxt = jnp.ones(3, dtype=jnp.float32)

    poisoned = learner.update(state, obs, jnp.array([jnp.inf, 1.0], dtype=jnp.float32), nxt)
    chex.assert_trees_all_close(poisoned.average_rewards[0], state.average_rewards[0])
    assert bool(jnp.isfinite(poisoned.average_rewards[1]))
    assert int(poisoned.state.step_count) == 1
    assert bool(poisoned.update_applied)
    chex.assert_trees_all_equal(
        poisoned.head_updates_applied,
        jnp.array([False, True]),
    )
    assert float(poisoned.td_errors[0]) == 0.0

    recovered = learner.update(poisoned.state, obs, jnp.array([0.5, 1.0], dtype=jnp.float32), nxt)
    chex.assert_tree_all_finite(recovered.average_rewards)
    assert int(recovered.state.step_count) == 2
    assert bool(recovered.update_applied)
    assert bool(jnp.all(recovered.head_updates_applied))


def test_average_reward_horde_rejects_scalar_cumulant() -> None:
    """A scalar cumulant must not broadcast to every demon head."""
    learner = AverageRewardHordeLearner(
        n_demons=3,
        hidden_sizes=(4,),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(3, jr.key(11))
    obs = jnp.ones(3, dtype=jnp.float32)
    nxt = jnp.ones(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="cumulants must have shape"):
        learner.update(state, obs, jnp.array(1.0, dtype=jnp.float32), nxt)

    with pytest.raises(ValueError, match="cumulants must have shape"):
        learner.update(state, obs, jnp.array([1.0], dtype=jnp.float32), nxt)


def test_differential_td_scan_shapes_and_finite_metrics() -> None:
    learner = DifferentialTDLearner(DifferentialTDConfig(trace_decay=0.2))
    state = learner.init(2)
    observations = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    next_observations = jnp.array(
        [[0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32)

    result = run_differential_td_from_arrays(
        learner,
        state,
        observations,
        rewards,
        next_observations,
    )

    chex.assert_shape(result.predictions, (3,))
    chex.assert_shape(result.td_errors, (3,))
    chex.assert_shape(result.average_rewards, (3,))
    chex.assert_shape(result.metrics, (3, 4))
    assert int(result.state.step_count) == 3
    chex.assert_tree_all_finite(
        (result.predictions, result.td_errors, result.average_rewards, result.metrics)
    )


def test_differential_gtd_config_roundtrip_and_ratio_clipping() -> None:
    config = DifferentialGTDConfig(
        value_step_size=0.1,
        secondary_step_size=0.05,
        average_reward_step_size=0.02,
        trace_decay=0.3,
        ratio_clip=1.5,
    )
    learner = DifferentialGTDLearner.from_config(DifferentialGTDLearner(config).to_config())
    state = learner.init(1)

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(3.0, dtype=jnp.float32),
    )

    assert learner.config == config
    chex.assert_trees_all_close(result.rho_clipped, jnp.array(1.5, dtype=jnp.float32))
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(result)


def test_differential_gtd_scan_learns_average_reward_cycle() -> None:
    learner = DifferentialGTDLearner(
        DifferentialGTDConfig(
            value_step_size=0.05,
            secondary_step_size=0.01,
            average_reward_step_size=0.01,
            trace_decay=0.0,
            ratio_clip=2.0,
        )
    )
    rewards_by_state = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    steps = 20_000
    states = jnp.arange(steps, dtype=jnp.int32) % 3
    next_states = (states + 1) % 3
    observations = jnp.eye(3, dtype=jnp.float32)[states]
    next_observations = jnp.eye(3, dtype=jnp.float32)[next_states]
    rewards = rewards_by_state[states]
    rhos = jnp.ones((steps,), dtype=jnp.float32)
    state = learner.init(3)

    result = run_differential_gtd_from_arrays(
        learner,
        state,
        observations,
        rewards,
        next_observations,
        rhos,
    )

    predictions = learner.predict(result.state, jnp.eye(3, dtype=jnp.float32))
    centered_predictions = predictions - jnp.mean(predictions)
    true_values = jnp.array([-2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=jnp.float32)
    chex.assert_trees_all_close(
        result.state.average_reward,
        jnp.array(1.0, dtype=jnp.float32),
        atol=2e-2,
    )
    chex.assert_trees_all_close(centered_predictions, true_values, atol=5e-2)
    assert float(jnp.mean(result.td_errors[-1000:] ** 2)) <= 2e-3
    chex.assert_tree_all_finite(result)


def test_average_reward_horde_shared_trunk_scan_learns_reward_rates() -> None:
    learner = AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(8,),
        step_size=0.02,
        average_reward_step_size=0.01,
        sparsity=0.0,
        use_layer_norm=False,
    )
    restored = AverageRewardHordeLearner.from_config(learner.to_config())
    assert restored.n_demons == 2

    steps = 20_000
    states = jnp.arange(steps, dtype=jnp.int32) % 3
    next_states = (states + 1) % 3
    observations = jnp.eye(3, dtype=jnp.float32)[states]
    next_observations = jnp.eye(3, dtype=jnp.float32)[next_states]
    cumulants = jnp.stack(
        [
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)[states],
            jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)[states],
        ],
        axis=1,
    )
    state = learner.init(3, jr.key(0))

    result = run_average_reward_horde_from_arrays(
        learner,
        state,
        observations,
        cumulants,
        next_observations,
    )

    chex.assert_trees_all_close(
        result.state.average_rewards,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        atol=3e-2,
    )
    assert float(jnp.mean(result.td_errors[-1000:] ** 2)) <= 5e-3
    chex.assert_tree_all_finite(result)


@pytest.mark.parametrize(
    ("observations_shape", "cumulants_shape", "next_observations_shape", "match"),
    [
        ((5, 3), (), (5, 3), "cumulants"),
        ((5, 3), (5,), (5, 3), "cumulants"),
        ((5, 3), (5, 1), (5, 3), "cumulants"),
        ((5, 3), (4, 2), (5, 3), "cumulants"),
        ((5,), (5, 2), (5, 3), "observations"),
        ((5, 3), (5, 2), (5,), "next_observations"),
        ((5, 3), (5, 2), (5, 4), "matching shapes"),
    ],
)
def test_average_reward_horde_scan_rejects_misaligned_arrays(
    observations_shape,
    cumulants_shape,
    next_observations_shape,
    match,
) -> None:
    learner = AverageRewardHordeLearner(n_demons=2, hidden_sizes=())
    state = learner.init(3, jr.key(2))
    with pytest.raises(ValueError, match=match):
        run_average_reward_horde_from_arrays(
            learner,
            state,
            jnp.zeros(observations_shape, dtype=jnp.float32),
            jnp.zeros(cumulants_shape, dtype=jnp.float32),
            jnp.zeros(next_observations_shape, dtype=jnp.float32),
        )


def test_average_reward_horde_actor_critic_single_update_is_finite() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(4,),
            critic_step_size=0.02,
            average_reward_step_size=0.01,
        )
    )
    restored = AverageRewardHordeActorCriticAgent.from_config(agent.to_config())
    assert restored.config == agent.config
    assert type(restored.actor_optimizer) is type(agent.actor_optimizer)
    state = agent.init(2, jr.key(0))
    state, action = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))

    result = agent.update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert int(action) in (0, 1)
    assert int(result.action) in (0, 1)
    assert int(result.state.step_count) == 1
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(
        (
            result.policy,
            result.target_policy,
            result.behavior_action_probability,
            result.target_action_probability,
            result.actor_score_scale,
            result.td_error,
            result.average_reward,
            result.critic_prediction,
            result.state.actor_weights,
            result.state.actor_bias,
            result.state.critic_state.average_rewards,
        )
    )


def test_average_reward_horde_actor_critic_rejects_nonfinite_reward() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(n_actions=2, hidden_sizes=(4,))
    )
    state, _ = agent.start(
        agent.init(2, jr.key(41)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    result = agent.update(
        state,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(jr.key_data(result.state.rng_key), jr.key_data(state.rng_key))
    chex.assert_trees_all_close(
        result.state.replace(rng_key=jr.key_data(result.state.rng_key)),
        state.replace(rng_key=jr.key_data(state.rng_key)),
    )
    chex.assert_trees_all_close(result.policy, jnp.zeros_like(result.policy))
    chex.assert_trees_all_close(result.target_policy, jnp.zeros_like(result.target_policy))
    assert float(result.td_error) == 0.0


def test_average_reward_actor_critic_behavior_policy_is_exact_epsilon_mixture() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=3,
        hidden_sizes=(4,),
        epsilon=0.25,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    state = agent.init(2, jr.key(0)).replace(
        actor_bias=jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)
    )
    observation = jnp.array([0.5, -0.25], dtype=jnp.float32)

    target = agent.policy(state, observation)
    behavior = agent.behavior_policy(state, observation)
    expected = 0.75 * target + 0.25 / 3.0
    chex.assert_trees_all_close(behavior, expected)
    assert float(jnp.sum(behavior)) == pytest.approx(1.0)

    sample, _ = agent.sample_policy(state, observation)
    chex.assert_trees_all_close(sample.target_policy, target)
    chex.assert_trees_all_close(sample.behavior_policy, behavior)
    chex.assert_trees_all_close(
        sample.target_probability,
        target[sample.action],
    )
    chex.assert_trees_all_close(
        sample.behavior_probability,
        behavior[sample.action],
    )
    chex.assert_trees_all_close(
        sample.target_log_probability,
        jnp.log(sample.target_probability),
    )
    chex.assert_trees_all_close(
        sample.behavior_log_probability,
        jnp.log(sample.behavior_probability),
    )


def test_average_reward_actor_sampling_preserves_reported_rare_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(4,),
            epsilon=0.0,
        )
    )
    state = agent.init(2, jr.key(101)).replace(
        actor_weights=jnp.zeros((2, 4), dtype=jnp.float32),
        actor_bias=jnp.asarray((0.0, -20.0), dtype=jnp.float32),
    )
    observation = jnp.zeros((2,), dtype=jnp.float32)
    observed_logits: list[jax.Array] = []
    observed_modes: list[object] = []

    def fake_categorical(
        _key: jax.Array,
        logits: jax.Array,
        **kwargs: object,
    ) -> jax.Array:
        observed_logits.append(logits)
        observed_modes.append(kwargs.get("mode"))
        return jnp.asarray(1, dtype=jnp.int32)

    monkeypatch.setattr(jr, "categorical", fake_categorical)
    with jax.disable_jit():
        started, action = agent.start(state, observation)
        result = agent.update(
            started,
            jnp.asarray(1.0, dtype=jnp.float32),
            observation,
        )
    sample = started.last_policy_sample

    assert int(action) == 1
    assert observed_modes == ["high", "high"]
    chex.assert_trees_all_close(
        observed_logits[0],
        jnp.log(sample.behavior_policy),
    )
    assert float(sample.behavior_policy[1]) < 1e-8
    assert bool(result.update_applied)
    assert float(result.actor_score_scale) == pytest.approx(1.0)


@pytest.mark.parametrize("epsilon", [0.0, 0.3, 1.0])
def test_average_reward_actor_critic_score_matches_mixture_derivative(
    epsilon: float,
) -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon=epsilon,
        temperature=0.7,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    initial = agent.init(2, jr.key(4)).replace(actor_bias=jnp.array([1.0, -1.0], dtype=jnp.float32))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    state, _ = agent.start(initial, observation)
    stored = state.last_policy_sample

    result = agent.update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([-0.25, 0.75], dtype=jnp.float32),
    )
    expected_scale = (1.0 - epsilon) * stored.target_probability / stored.behavior_probability
    chex.assert_trees_all_close(result.actor_score_scale, expected_scale)

    # Finite-difference d log(mu_a) / d target-logit_a agrees with the
    # analytical mixture score's selected-action component.
    action = int(stored.action)
    bias = initial.actor_bias

    def selected_log_behavior(selected_bias):
        logits = bias.at[action].set(selected_bias)
        target = jax.nn.softmax(logits / config.temperature)
        behavior = (1.0 - epsilon) * target + epsilon / config.n_actions
        return jnp.log(behavior[action])

    finite_difference = jax.grad(selected_log_behavior)(bias[action])
    expected_component = expected_scale * (1.0 - stored.target_policy[action]) / config.temperature
    chex.assert_trees_all_close(
        finite_difference,
        expected_component,
        atol=1e-6,
    )


def test_epsilon_one_freezes_actor_but_not_critic() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.02,
        average_reward_step_size=0.01,
        epsilon=1.0,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    initial = agent.init(2, jr.key(9)).replace(
        actor_weights=jnp.array(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.2, 0.5, -0.4, 0.1],
            ],
            dtype=jnp.float32,
        ),
        actor_bias=jnp.array([0.7, -0.2], dtype=jnp.float32),
    )
    state, _ = agent.start(initial, jnp.array([1.0, 0.0], dtype=jnp.float32))
    result = agent.update(
        state,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.state.actor_weights, state.actor_weights)
    chex.assert_trees_all_equal(result.state.actor_bias, state.actor_bias)
    chex.assert_trees_all_equal(result.state.actor_opt_w, state.actor_opt_w)
    chex.assert_trees_all_equal(result.state.actor_opt_b, state.actor_opt_b)
    chex.assert_trees_all_close(
        result.policy,
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    )
    assert float(result.actor_score_scale) == pytest.approx(0.0)
    assert not jnp.allclose(
        result.state.critic_state.average_rewards,
        state.critic_state.average_rewards,
    )


def test_update_samples_successor_from_committed_parameters_across_execution_paths() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon=0.2,
        actor_update_clip=1.0,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    state, _ = agent.start(
        agent.init(2, jr.key(12)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    next_observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    stale_sample, _ = agent.sample_policy(state, next_observation)
    result = agent.update(state, 10.0, next_observation)
    with jax.disable_jit():
        eager = agent.update(state, 10.0, next_observation)

    # The actor did change, so this distinguishes the causal boundary from the
    # historical pre-update successor-sampling behavior.
    assert not jnp.array_equal(result.target_policy, stale_sample.target_policy)
    for candidate in (result, eager):
        current_target = agent.policy(candidate.state, next_observation)
        current_behavior = agent.behavior_policy(candidate.state, next_observation)
        expected_sample, expected_key = agent.sample_policy(
            candidate.state.replace(rng_key=state.rng_key),
            next_observation,
        )
        chex.assert_trees_all_equal(candidate.target_policy, current_target)
        chex.assert_trees_all_equal(candidate.policy, current_behavior)
        chex.assert_trees_all_equal(candidate.state.last_policy_sample, expected_sample)
        chex.assert_trees_all_equal(candidate.state.rng_key, expected_key)
        chex.assert_trees_all_equal(
            candidate.state.last_policy_sample.target_policy,
            current_target,
        )
        chex.assert_trees_all_equal(
            candidate.state.last_policy_sample.behavior_policy,
            current_behavior,
        )
        chex.assert_trees_all_equal(
            candidate.target_action_probability,
            current_target[candidate.action],
        )
        chex.assert_trees_all_equal(
            candidate.behavior_action_probability,
            current_behavior[candidate.action],
        )

    scanned = run_average_reward_horde_actor_critic_from_arrays(
        agent,
        state,
        jnp.asarray([10.0], dtype=jnp.float32),
        next_observation[None, :],
    )
    scanned_target = agent.policy(scanned.state, next_observation)
    scanned_behavior = agent.behavior_policy(scanned.state, next_observation)
    chex.assert_trees_all_equal(scanned.target_policies[0], scanned_target)
    chex.assert_trees_all_equal(scanned.policies[0], scanned_behavior)
    chex.assert_trees_all_equal(scanned.state.last_policy_sample.target_policy, scanned_target)
    chex.assert_trees_all_equal(scanned.state.last_policy_sample.behavior_policy, scanned_behavior)
    chex.assert_trees_all_equal(scanned.actions[0], scanned.state.last_action)


def test_average_reward_actor_critic_scan_logs_action_probabilities() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=3,
            hidden_sizes=(4,),
            epsilon=0.2,
        )
    )
    state, _ = agent.start(
        agent.init(2, jr.key(17)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    result = run_average_reward_horde_actor_critic_from_arrays(
        agent,
        state,
        jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32),
        jnp.array(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        ),
    )

    row = jnp.arange(result.actions.shape[0])
    assert bool(jnp.all(result.updates_applied))
    chex.assert_trees_all_close(
        result.behavior_action_probabilities,
        result.policies[row, result.actions],
    )
    chex.assert_trees_all_close(
        result.target_action_probabilities,
        result.target_policies[row, result.actions],
    )
    chex.assert_trees_all_close(
        jnp.sum(result.policies, axis=1),
        jnp.ones(3),
    )
    chex.assert_tree_all_finite(
        (
            result.actions,
            result.policies,
            result.target_policies,
            result.behavior_action_probabilities,
            result.target_action_probabilities,
            result.actor_score_scales,
            result.td_errors,
            result.average_rewards,
        )
    )


def test_differential_sarsa_config_roundtrip_and_exact_td_error() -> None:
    config = DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon_start=0.0,
    )
    agent = DifferentialSARSAAgent.from_config(DifferentialSARSAAgent(config).to_config())
    state = agent.init(2, jr.key(0)).replace(  # type: ignore[attr-defined]
        q_weights=jnp.array([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32),
        q_bias=jnp.array([0.5, -0.5], dtype=jnp.float32),
        average_reward=jnp.array(0.25, dtype=jnp.float32),
        last_observation=jnp.array([2.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    next_obs = jnp.array([1.0, 3.0], dtype=jnp.float32)

    result = agent.update(
        state,
        jnp.array(2.0, dtype=jnp.float32),
        next_obs,
        next_action=jnp.array(1, dtype=jnp.int32),
    )

    assert agent.config == config
    chex.assert_trees_all_close(result.td_error, jnp.array(4.75, dtype=jnp.float32))
    chex.assert_trees_all_close(result.average_reward, state.average_reward)


def test_differential_sarsa_start_with_action_rejects_fractional_action() -> None:
    agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=3))
    state = agent.init(2, jr.key(0))

    with pytest.raises(TypeError, match="action must have dtype int32"):
        agent.start_with_action(
            state,
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            jnp.array(1.75, dtype=jnp.float32),
        )


def test_differential_sarsa_update_and_scan_are_finite() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=3,
            q_step_size=0.05,
            average_reward_step_size=0.01,
            trace_decay=0.2,
            epsilon_start=0.2,
        )
    )
    state = agent.init(2, jr.key(1))
    state, _ = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
    rewards = jnp.array([1.0, 0.0, 0.5, -0.25], dtype=jnp.float32)
    next_observations = jnp.array(
        [[0.0, 1.0], [1.0, 1.0], [0.5, -0.5], [1.0, 0.0]],
        dtype=jnp.float32,
    )

    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        rewards,
        next_observations,
    )

    chex.assert_shape(result.q_values, (4, 3))
    chex.assert_shape(result.td_errors, (4,))
    chex.assert_shape(result.average_rewards, (4,))
    chex.assert_shape(result.actions, (4,))
    assert int(result.state.step_count) == 4
    chex.assert_tree_all_finite((result.q_values, result.td_errors, result.average_rewards))
    assert bool(jnp.all(result.actions >= 0))
    assert bool(jnp.all(result.actions < 3))


def test_differential_sarsa_learns_better_action_on_continuing_bandit() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.04,
            average_reward_step_size=0.01,
            trace_decay=0.0,
            epsilon_start=0.1,
            epsilon_end=0.02,
            epsilon_decay_steps=200,
        )
    )
    obs = jnp.array([1.0], dtype=jnp.float32)
    state = agent.init(1, jr.key(42))
    state, _ = agent.start(state, obs)

    for _ in range(800):
        reward = jnp.asarray(state.last_action == 1, dtype=jnp.float32)
        result = agent.update(state, reward, obs)
        state = result.state

    q_values = agent.q_values(state, obs)
    assert float(q_values[1]) > float(q_values[0]) + 0.25
    assert float(state.average_reward) > 0.75


def test_average_reward_horde_actor_critic_config_scalar_validation() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        AverageRewardHordeActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        AverageRewardHordeActorCriticConfig(n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes"):
        AverageRewardHordeActorCriticConfig(n_actions=2, hidden_sizes=(True,))  # type: ignore[arg-type]

    cfg = AverageRewardHordeActorCriticConfig(
        n_actions=np.int32(4),
        hidden_sizes=(np.int32(32), np.int64(16)),
    )
    assert type(cfg.n_actions) is int
    assert type(cfg.hidden_sizes[0]) is int
    assert type(cfg.hidden_sizes[1]) is int
    assert cfg.n_actions == 4
    assert cfg.hidden_sizes == (32, 16)


def test_differential_sarsa_config_scalar_validation() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        DifferentialSARSAConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        DifferentialSARSAConfig(n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon_decay_steps"):
        DifferentialSARSAConfig(n_actions=2, epsilon_decay_steps=True)  # type: ignore[arg-type]

    cfg = DifferentialSARSAConfig(
        n_actions=np.int32(3),
        epsilon_decay_steps=np.int64(100),
    )
    assert type(cfg.n_actions) is int
    assert type(cfg.epsilon_decay_steps) is int
    assert cfg.n_actions == 3
    assert cfg.epsilon_decay_steps == 100


_NUMPY_INTEGER_TYPES = tuple(dict.fromkeys(np.dtype(code).type for code in "bhilqBHILQpP"))


@pytest.mark.parametrize("integer_type", _NUMPY_INTEGER_TYPES)
def test_average_reward_configs_canonicalize_all_numpy_integer_families(
    integer_type: type[np.integer],
) -> None:
    actor = AverageRewardHordeActorCriticConfig(
        n_actions=integer_type(2),
        hidden_sizes=(integer_type(3),),
    )
    sarsa = DifferentialSARSAConfig(
        n_actions=integer_type(2),
        epsilon_decay_steps=integer_type(3),
    )

    assert type(actor.n_actions) is int
    assert type(actor.hidden_sizes[0]) is int
    assert type(sarsa.n_actions) is int
    assert type(sarsa.epsilon_decay_steps) is int


def test_average_reward_configs_reject_hostile_integer_and_container_types() -> None:
    class HostileInt(int):
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    class TupleSubclass(tuple):
        pass

    with pytest.raises(ValueError, match="n_actions"):
        AverageRewardHordeActorCriticConfig(n_actions=HostileInt(2))
    with pytest.raises(ValueError, match="hidden_sizes"):
        AverageRewardHordeActorCriticConfig(n_actions=2, hidden_sizes=TupleSubclass((3,)))
    payload = AverageRewardHordeActorCriticConfig(n_actions=2).to_config()
    payload["hidden_sizes"] = "16"
    with pytest.raises(ValueError, match="hidden_sizes"):
        AverageRewardHordeActorCriticConfig.from_config(payload)


def test_average_reward_decoders_reject_schema_and_container_ambiguity() -> None:
    class DictSubclass(dict[str, object]):
        pass

    actor_config = AverageRewardHordeActorCriticConfig(n_actions=2).to_config()
    with pytest.raises(ValueError, match="actual dict"):
        AverageRewardHordeActorCriticConfig.from_config(DictSubclass(actor_config))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields"):
        AverageRewardHordeActorCriticConfig.from_config({**actor_config, "extra": 1})
    with pytest.raises(ValueError, match="type"):
        AverageRewardHordeActorCriticConfig.from_config({**actor_config, "type": "wrong"})
    with pytest.raises(ValueError, match="serialized hidden_sizes"):
        AverageRewardHordeActorCriticConfig.from_config(
            {**actor_config, "hidden_sizes": tuple(actor_config["hidden_sizes"])}
        )

    actor_agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(n_actions=2)
    ).to_config()
    with pytest.raises(ValueError, match="actual dict"):
        AverageRewardHordeActorCriticAgent.from_config(DictSubclass(actor_agent))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nested configs"):
        AverageRewardHordeActorCriticAgent.from_config(
            {**actor_agent, "config": DictSubclass(actor_agent["config"])}  # type: ignore[arg-type]
        )

    sarsa_config = DifferentialSARSAConfig(n_actions=2).to_config()
    with pytest.raises(ValueError, match="actual dict"):
        DifferentialSARSAConfig.from_config(DictSubclass(sarsa_config))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields"):
        DifferentialSARSAConfig.from_config({**sarsa_config, "extra": 1})
    with pytest.raises(ValueError, match="type"):
        DifferentialSARSAConfig.from_config({**sarsa_config, "type": "wrong"})

    sarsa_agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=2)).to_config()
    with pytest.raises(ValueError, match="actual dict"):
        DifferentialSARSAAgent.from_config(DictSubclass(sarsa_agent))
    with pytest.raises(ValueError, match="nested config"):
        DifferentialSARSAAgent.from_config(
            {**sarsa_agent, "config": DictSubclass(sarsa_agent["config"])}  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("config_type", "field", "value"),
    (
        (AverageRewardHordeActorCriticConfig, "temperature", 0.0),
        (AverageRewardHordeActorCriticConfig, "epsilon", 1.0 + 1.0e-10),
        (AverageRewardHordeActorCriticConfig, "critic_step_size", 1.0e100),
        (DifferentialSARSAConfig, "q_step_size", -1.0),
        (DifferentialSARSAConfig, "trace_decay", 1.0 + 1.0e-10),
        (DifferentialSARSAConfig, "epsilon_start", 1.0e100),
        (DifferentialSARSAConfig, "use_bias", np.bool_(True)),
        (DifferentialTDConfig, "step_size", float("nan")),
        (DifferentialTDConfig, "step_size", float("inf")),
        (DifferentialTDConfig, "average_reward_step_size", float("nan")),
        (DifferentialTDConfig, "trace_decay", float("nan")),
        (DifferentialGTDConfig, "value_step_size", float("nan")),
        (DifferentialGTDConfig, "secondary_step_size", float("inf")),
        (DifferentialGTDConfig, "average_reward_step_size", float("-inf")),
        (DifferentialGTDConfig, "trace_decay", float("nan")),
        (DifferentialGTDConfig, "ratio_clip", float("nan")),
        (DifferentialGTDConfig, "ratio_clip", 0.0),
    ),
)
def test_average_reward_configs_reject_invalid_float32_sink_values(
    config_type: type,
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {field: value}
    if config_type in (AverageRewardHordeActorCriticConfig, DifferentialSARSAConfig):
        kwargs["n_actions"] = 2
    with pytest.raises(ValueError, match=field):
        config_type(**kwargs)


def test_average_reward_horde_rejects_nonfinite_reward_rate_scalars() -> None:
    with pytest.raises(ValueError, match="average_reward_step_size"):
        AverageRewardHordeLearner(n_demons=2, average_reward_step_size=float("nan"))
    with pytest.raises(ValueError, match="average_reward_step_size"):
        AverageRewardHordeLearner(n_demons=2, average_reward_step_size=float("inf"))
    with pytest.raises(ValueError, match="trace_decay"):
        AverageRewardHordeLearner(n_demons=2, trace_decay=float("nan"))
    AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(4,),
        average_reward_step_size=0.0,
        trace_decay=0.0,
    )


@pytest.mark.parametrize(
    ("config_type", "field"),
    [
        (DifferentialTDConfig, "step_size"),
        (DifferentialTDConfig, "average_reward_step_size"),
        (DifferentialTDConfig, "trace_decay"),
        (DifferentialGTDConfig, "value_step_size"),
        (DifferentialGTDConfig, "secondary_step_size"),
        (DifferentialGTDConfig, "average_reward_step_size"),
        (DifferentialGTDConfig, "trace_decay"),
    ],
)
def test_differential_configs_reject_nonzero_float32_underflow(
    config_type: type[DifferentialTDConfig] | type[DifferentialGTDConfig],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=f"{field} must remain nonzero"):
        config_type(**{field: 2.0**-150})
    with pytest.raises(ValueError, match=f"{field} must remain nonzero"):
        config_type(**{field: 5e-324})


def test_nonnegative_average_reward_float32_sinks_preserve_zero_and_minsubnormal() -> None:
    smallest_float32 = 2.0**-149
    td = DifferentialTDConfig(
        step_size=0.0,
        average_reward_step_size=smallest_float32,
        trace_decay=smallest_float32,
    )
    gtd = DifferentialGTDConfig(
        value_step_size=smallest_float32,
        secondary_step_size=0.0,
        average_reward_step_size=smallest_float32,
        trace_decay=0.0,
    )
    horde = AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(4,),
        average_reward_step_size=smallest_float32,
        trace_decay=0.0,
    )

    assert td.step_size == 0.0
    assert td.average_reward_step_size == smallest_float32
    assert td.trace_decay == smallest_float32
    assert gtd.value_step_size == smallest_float32
    assert gtd.secondary_step_size == 0.0
    assert horde.to_config()["average_reward_step_size"] == smallest_float32


def test_average_reward_horde_rejects_nonzero_float32_underflow() -> None:
    with pytest.raises(ValueError, match="average_reward_step_size must remain nonzero"):
        AverageRewardHordeLearner(n_demons=2, average_reward_step_size=2.0**-150)
    with pytest.raises(ValueError, match="trace_decay must remain nonzero"):
        AverageRewardHordeLearner(n_demons=2, trace_decay=5e-324)


def test_average_reward_configs_canonicalize_float32_sink_values() -> None:
    actor = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        critic_step_size=np.float64(0.2),
    )
    sarsa = DifferentialSARSAConfig(
        n_actions=2,
        epsilon_start=np.float64(0.2),
    )
    td = DifferentialTDConfig(step_size=np.float64(0.2))
    gtd = DifferentialGTDConfig(value_step_size=np.float64(0.2))

    assert type(actor.critic_step_size) is float
    assert actor.critic_step_size == float(np.float32(0.2))
    assert type(sarsa.epsilon_start) is float
    assert sarsa.epsilon_start == float(np.float32(0.2))
    assert type(td.step_size) is float
    assert td.step_size == float(np.float32(0.2))
    assert type(gtd.value_step_size) is float
    assert gtd.value_step_size == float(np.float32(0.2))


def test_average_reward_actor_preflights_state_before_allocation() -> None:
    # For observation_dim=1 and hidden_sizes=(1,), the aggregate owns 10
    # scalars per action plus 32 fixed scalars. The latter includes all four
    # scalar LMS states for the critic's trunk/head weight and bias arrays.
    last_legal_n_actions = (2**29 - 1 - 32) // 10
    AverageRewardHordeActorCriticConfig(
        n_actions=last_legal_n_actions,
        hidden_sizes=(1,),
    )
    with pytest.raises(ValueError, match="state bytes"):
        AverageRewardHordeActorCriticConfig(
            n_actions=last_legal_n_actions + 1,
            hidden_sizes=(1,),
        )


def test_differential_sarsa_preflights_state_before_allocation() -> None:
    agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=1))
    last_legal_feature_dim = (2**29 - 1 - 10) // 3

    with pytest.raises(ValueError, match="state bytes"):
        agent.init(last_legal_feature_dim + 1, jr.key(0))
    with pytest.raises(ValueError, match="feature_dim"):
        agent.init(True, jr.key(0))  # type: ignore[arg-type]


def test_differential_learners_integer_validation() -> None:
    gtd = DifferentialGTDLearner(DifferentialGTDConfig())
    td = DifferentialTDLearner(DifferentialTDConfig())
    horde = AverageRewardHordeLearner(n_demons=2, hidden_sizes=(16,))

    with pytest.raises(ValueError, match="feature_dim"):
        gtd.init(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        gtd.init(feature_dim=0)
    with pytest.raises(ValueError, match="feature_dim"):
        gtd.init(feature_dim=4.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="feature_dim"):
        td.init(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        td.init(feature_dim=0)

    with pytest.raises(ValueError, match="feature_dim"):
        horde.init(feature_dim=True, key=jr.key(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        horde.init(feature_dim=0, key=jr.key(0))

    s_gtd = gtd.init(feature_dim=np.int32(4))
    s_td = td.init(feature_dim=np.int64(4))
    s_horde = horde.init(feature_dim=np.int32(4), key=jr.key(0))

    assert s_gtd.weights.shape == (4,)
    assert s_td.weights.shape == (4,)
    assert s_horde.average_rewards.shape == (2,)


def test_differential_initializers_preflight_state_before_allocation() -> None:
    gtd = DifferentialGTDLearner(DifferentialGTDConfig())
    td = DifferentialTDLearner(DifferentialTDConfig())
    max_float32_scalars = (2**31 - 1) // 4
    last_legal_td_dim = (max_float32_scalars - 4) // 2
    last_legal_gtd_dim = (max_float32_scalars - 5) // 3

    with pytest.raises(ValueError, match="differential TD state bytes"):
        td.init(feature_dim=last_legal_td_dim + 1)
    with pytest.raises(ValueError, match="differential GTD state bytes"):
        gtd.init(feature_dim=last_legal_gtd_dim + 1)


@pytest.mark.parametrize("learner_type", [DifferentialTDLearner, DifferentialGTDLearner])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_differential_initializers_reject_nonfinite_average_reward(
    learner_type, value
) -> None:
    learner = learner_type()
    with pytest.raises(ValueError, match="average_reward"):
        learner.init(feature_dim=1, average_reward=value)


@pytest.mark.parametrize("learner_type", [DifferentialTDLearner, DifferentialGTDLearner])
@pytest.mark.parametrize("value", [2.0**-150, -(2.0**-150), 5e-324, -5e-324])
def test_differential_initializers_reject_average_reward_underflow(
    learner_type, value
) -> None:
    learner = learner_type()
    with pytest.raises(ValueError, match="average_reward must remain nonzero"):
        learner.init(feature_dim=1, average_reward=value)


@pytest.mark.parametrize("learner_type", [DifferentialTDLearner, DifferentialGTDLearner])
@pytest.mark.parametrize("value", [0.0, 2.0**-149, -(2.0**-149)])
def test_differential_initializers_preserve_zero_and_float32_minsubnormal(
    learner_type, value
) -> None:
    state = learner_type().init(feature_dim=1, average_reward=value)
    assert float(state.average_reward) == value


def test_differential_td_and_gtd_configs_reject_invalid_scalars() -> None:
    # DifferentialTDConfig
    with pytest.raises(ValueError, match="step_size"):
        DifferentialTDConfig(step_size=float("nan"))
    with pytest.raises(ValueError, match="step_size"):
        DifferentialTDConfig(step_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="average_reward_step_size"):
        DifferentialTDConfig(average_reward_step_size=float("nan"))
    with pytest.raises(ValueError, match="trace_decay"):
        DifferentialTDConfig(trace_decay=float("nan"))

    # DifferentialGTDConfig
    with pytest.raises(ValueError, match="value_step_size"):
        DifferentialGTDConfig(value_step_size=float("nan"))
    with pytest.raises(ValueError, match="secondary_step_size"):
        DifferentialGTDConfig(secondary_step_size=float("nan"))
    with pytest.raises(ValueError, match="ratio_clip"):
        DifferentialGTDConfig(ratio_clip=float("nan"))
    with pytest.raises(ValueError, match="ratio_clip"):
        DifferentialGTDConfig(ratio_clip=0.0)

    # AverageRewardHordeLearner
    with pytest.raises(ValueError, match="average_reward_step_size"):
        AverageRewardHordeLearner(1, average_reward_step_size=float("nan"))
    with pytest.raises(ValueError, match="step_size"):
        AverageRewardHordeLearner(1, step_size=float("nan"))
    with pytest.raises(ValueError, match="trace_decay"):
        AverageRewardHordeLearner(1, trace_decay=float("nan"))
    with pytest.raises(ValueError, match="sparsity"):
        AverageRewardHordeLearner(1, sparsity=float("nan"))
    with pytest.raises(ValueError, match="use_layer_norm"):
        AverageRewardHordeLearner(1, use_layer_norm=1)  # type: ignore[arg-type]


# =============================================================================
# Scan sequence-length ceiling (hang guard)
# =============================================================================
#
# ``run_differential_td_from_arrays``, ``run_differential_gtd_from_arrays``,
# ``run_average_reward_horde_from_arrays``,
# ``run_average_reward_horde_actor_critic_from_arrays``, and
# ``run_differential_sarsa_from_arrays`` hand their step arrays straight to
# ``jax.lax.scan`` with no bound on the leading (step) axis. A hostile or
# mistaken caller supplying a huge array forces JAX to materialize per-step
# outputs at that length, hanging the process well before any step executes
# -- the same hang class already fixed for other scan-driven array loops in
# ``core`` and ``utils``.


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.average_reward.jax.lax.scan", spy)
    return seen


class TestAverageRewardSequenceCeiling:
    def test_documented_protocol_ceiling(self) -> None:
        assert _AVERAGE_REWARD_SEQUENCE_MAX_STEPS == 50_000

    def test_last_fit_length_is_accepted(self) -> None:
        vector = jnp.zeros((_AVERAGE_REWARD_SEQUENCE_MAX_STEPS,))
        assert (
            _require_avg_reward_sequence_length("rewards", vector)
            == _AVERAGE_REWARD_SEQUENCE_MAX_STEPS
        )

    def test_first_overflow_length_is_rejected(self) -> None:
        vector = jnp.zeros((_AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1,))
        with pytest.raises(
            ValueError, match=r"rewards length must be an integer in \[1, 50000\]"
        ):
            _require_avg_reward_sequence_length("rewards", vector)

    def test_empty_length_is_rejected(self) -> None:
        vector = jnp.zeros((0,))
        with pytest.raises(ValueError, match=r"rewards length must be an integer in"):
            _require_avg_reward_sequence_length("rewards", vector)

    def test_scalar_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="leading step axis"):
            _require_avg_reward_sequence_length("rewards", jnp.array(1.0))

    def test_non_array_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_avg_reward_sequence_length("rewards", [1.0, 2.0, 3.0])

        class _HostileArrayLike:
            shape = (3,)
            ndim = 1

        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_avg_reward_sequence_length("rewards", _HostileArrayLike())

    def test_mismatched_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same leading length"):
            _require_avg_reward_matching_length("rewards", jnp.zeros((3,)), expected=4)

    def test_matching_length_non_array_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a JAX array"):
            _require_avg_reward_matching_length("rewards", [1.0, 2.0], expected=2)

    def test_run_differential_td_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner = DifferentialTDLearner(DifferentialTDConfig())
        state = learner.init(2)
        n = _AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1
        observations = jnp.ones((n, 2), dtype=jnp.float32)
        rewards = jnp.ones((n,), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations length must be an integer in"):
            run_differential_td_from_arrays(
                learner, state, observations, rewards, next_observations
            )
        assert seen == []

    def test_run_differential_td_from_arrays_rejects_mismatched_length_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner = DifferentialTDLearner(DifferentialTDConfig())
        state = learner.init(2)
        observations = jnp.ones((10, 2), dtype=jnp.float32)
        rewards = jnp.ones((5,), dtype=jnp.float32)
        next_observations = jnp.ones((10, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="same leading length"):
            run_differential_td_from_arrays(
                learner, state, observations, rewards, next_observations
            )
        assert seen == []

    def test_run_differential_gtd_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner = DifferentialGTDLearner(DifferentialGTDConfig())
        state = learner.init(2)
        n = _AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1
        observations = jnp.ones((n, 2), dtype=jnp.float32)
        rewards = jnp.ones((n,), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        rhos = jnp.ones((n,), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations length must be an integer in"):
            run_differential_gtd_from_arrays(
                learner, state, observations, rewards, next_observations, rhos
            )
        assert seen == []

    def test_run_average_reward_horde_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        learner = AverageRewardHordeLearner(n_demons=1, hidden_sizes=())
        state = learner.init(2, jr.key(0))
        n = _AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1
        observations = jnp.ones((n, 2), dtype=jnp.float32)
        cumulants = jnp.ones((n, 1), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations length must be an integer in"):
            run_average_reward_horde_from_arrays(
                learner, state, observations, cumulants, next_observations
            )
        assert seen == []

    def test_run_average_reward_horde_actor_critic_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        agent = AverageRewardHordeActorCriticAgent(
            AverageRewardHordeActorCriticConfig(n_actions=2, hidden_sizes=())
        )
        state = agent.init(2, jr.key(0))
        state, _ = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
        n = _AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1
        rewards = jnp.ones((n,), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="rewards length must be an integer in"):
            run_average_reward_horde_actor_critic_from_arrays(
                agent, state, rewards, next_observations
            )
        assert seen == []

    def test_run_differential_sarsa_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=2))
        state = agent.init(2, jr.key(0))
        state, _ = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
        n = _AVERAGE_REWARD_SEQUENCE_MAX_STEPS + 1
        rewards = jnp.ones((n,), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="rewards length must be an integer in"):
            run_differential_sarsa_from_arrays(agent, state, rewards, next_observations)
        assert seen == []

    def test_run_differential_sarsa_from_arrays_rejects_mismatched_discounts_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=2))
        state = agent.init(2, jr.key(0))
        state, _ = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
        rewards = jnp.ones((5,), dtype=jnp.float32)
        next_observations = jnp.ones((5, 2), dtype=jnp.float32)
        discounts = jnp.ones((3,), dtype=jnp.float32)
        with pytest.raises(ValueError, match="same leading length"):
            run_differential_sarsa_from_arrays(
                agent, state, rewards, next_observations, discounts
            )
        assert seen == []

    def test_origin_hang_class_is_rejected_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A far larger sequence length -- the actual hang class -- is also
        rejected before ``jax.lax.scan`` is ever called."""
        seen = _spy_scan(monkeypatch)
        learner = DifferentialTDLearner(DifferentialTDConfig())
        state = learner.init(2)
        n = 2_000_000
        observations = jnp.ones((n, 2), dtype=jnp.float32)
        rewards = jnp.ones((n,), dtype=jnp.float32)
        next_observations = jnp.ones((n, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations length must be an integer in"):
            run_differential_td_from_arrays(
                learner, state, observations, rewards, next_observations
            )
        assert seen == []
