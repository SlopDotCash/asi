"""Exact RNG-threading contracts for Step 9 guarded dreaming.

Every positive-budget dream owns four disjoint branches of the carried control
key: the reserved next master, candidate search, behavior rollout, and control
rollout.  The reserved branch is persisted whether the dream commits or rolls
back; a zero budget leaves the post-real-update key untouched.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.behavior_model import BehaviorModel, BehaviorModelConfig
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    Step9DreamingState,
    _update_control_with_linear_rng,
    init_step9_state,
    make_step9_components,
    run_step9_scan,
    step9_update,
)

OBSERVATION_DIM = 2
N_ACTIONS = 7


def _config(
    planning_budget: int,
    *,
    accepted: bool = True,
    rollout_horizon: int = 1,
    candidate_count: int = 1,
) -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=N_ACTIONS,
            epsilon_start=1.0,
            epsilon_end=1.0,
        ),
        observation_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=planning_budget,
        dream_rollout_horizon=rollout_horizon,
        dream_candidate_count=candidate_count,
        dreaming_warmup_steps=0 if accepted else 100,
        dreaming_max_model_error=1e30,
    )


def _init(
    config: Step9DreamingConfig,
    *,
    seed: int,
) -> tuple[object, object, object, Step9DreamingState]:
    agent, model, buffer = make_step9_components(config)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(seed),
        initial_observation=jnp.array([0.25, -0.5], dtype=jnp.float32),
    )
    return agent, model, buffer, state


def _advance_master(key: jax.Array, budget: int) -> jax.Array:
    for _ in range(budget):
        key = jr.split(key, 4)[0]
    return key


def _assert_key_equal(actual: jax.Array, expected: jax.Array) -> None:
    chex.assert_trees_all_equal(jr.key_data(actual), jr.key_data(expected))


def _assert_pairwise_distinct(keys: tuple[jax.Array, ...]) -> None:
    words = tuple(jr.key_data(key) for key in keys)
    for index, left in enumerate(words):
        for right in words[index + 1 :]:
            assert not bool(jnp.array_equal(left, right))


def _real_behavior_state(
    config: Step9DreamingConfig,
    state: Step9DreamingState,
) -> tuple[BehaviorModel, object]:
    behavior_model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=config.n_actions,
            step_size=config.behavior_model_step_size,
        )
    )
    behavior_after_real = behavior_model.update(
        state.behavior_model_state,
        state.control_state.last_observation,
        state.control_state.last_action,
    ).state
    return behavior_model, behavior_after_real


def test_zero_budget_is_bit_identical_to_the_real_update_only() -> None:
    """Budget zero performs no planning split and preserves exact real outputs."""
    config = _config(0)
    agent, model, buffer, state = _init(config, seed=3)
    reward = jnp.array(0.75, dtype=jnp.float32)
    next_observation = jnp.array([-0.2, 0.4], dtype=jnp.float32)

    expected_model = model.update(
        state.world_model_state,
        state.control_state.last_observation,
        state.control_state.last_action,
        reward,
        jnp.asarray(config.model_gamma, dtype=jnp.float32),
        next_observation,
    )
    expected_control = agent.update(
        state.control_state,
        reward,
        next_observation,
        discount=jnp.asarray(config.model_gamma, dtype=jnp.float32),
    )
    _, expected_behavior = _real_behavior_state(config, state)
    expected_state = Step9DreamingState(
        control_state=expected_control.state,
        world_model_state=expected_model.state,
        behavior_model_state=expected_behavior,
        buffer_state=buffer.add(state.buffer_state, next_observation),
        step_count=state.step_count + 1,
    )

    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        reward,
        next_observation,
    )

    chex.assert_trees_all_equal(result.state, expected_state)
    chex.assert_trees_all_equal(result.real_control_result, expected_control)
    chex.assert_trees_all_equal(result.real_model_result, expected_model)
    chex.assert_shape(result.dream_td_errors, (0,))
    chex.assert_shape(result.dream_accepted, (0,))


def test_zero_budget_rejected_real_update_preserves_the_original_key() -> None:
    """Even rejection stays on the unchanged real-only path at budget zero."""
    config = _config(0)
    agent, model, buffer, state = _init(config, seed=5)
    saturated_control = state.control_state.replace(
        step_count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        step_words=jnp.full(
            (2,),
            jnp.iinfo(jnp.uint32).max,
            dtype=jnp.uint32,
        ),
    )
    state = state.replace(control_state=saturated_control)
    reward = jnp.array(0.25, dtype=jnp.float32)
    next_observation = jnp.array([0.2, -0.3], dtype=jnp.float32)
    expected = agent.update(
        saturated_control,
        reward,
        next_observation,
        discount=jnp.asarray(config.model_gamma, dtype=jnp.float32),
    )
    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        reward,
        next_observation,
    )

    assert not bool(expected.update_applied)
    chex.assert_trees_all_equal(result.real_control_result, expected)
    chex.assert_trees_all_equal(result.state.control_state, expected.state)


def test_dream_roots_and_rollout_descendants_are_pairwise_distinct() -> None:
    """The reserved, candidate, behavior, and control key families do not alias."""
    master = jr.key(41)
    reserved, candidate, behavior, control = jr.split(master, 4)
    candidate_0, candidate_1 = jr.split(candidate, 2)
    anchor_0, candidate_sample_0 = jr.split(candidate_0)
    anchor_1, candidate_sample_1 = jr.split(candidate_1)
    behavior_next, behavior_sample = jr.split(behavior)
    control_next, explore, tie_break, random_action = jr.split(control, 4)

    _assert_pairwise_distinct(
        (
            reserved,
            anchor_0,
            candidate_sample_0,
            anchor_1,
            candidate_sample_1,
            behavior_next,
            behavior_sample,
            control_next,
            explore,
            tie_break,
            random_action,
        )
    )


@pytest.mark.parametrize("planning_budget", [1, 2])
def test_accepted_dreams_persist_the_reserved_future_master(
    planning_budget: int,
) -> None:
    config = _config(planning_budget, accepted=True, rollout_horizon=2)
    agent, model, buffer, state = _init(config, seed=7)
    reward = jnp.array(0.5, dtype=jnp.float32)
    next_observation = jnp.array([0.1, 0.2], dtype=jnp.float32)
    expected_real = agent.update(
        state.control_state,
        reward,
        next_observation,
        discount=jnp.asarray(config.model_gamma, dtype=jnp.float32),
    )
    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        reward,
        next_observation,
    )

    assert bool(jnp.all(result.dream_accepted))
    chex.assert_trees_all_equal(result.real_control_result, expected_real)
    master = result.real_control_result.state.rng_key
    _assert_key_equal(
        result.state.control_state.rng_key,
        _advance_master(master, planning_budget),
    )


@pytest.mark.parametrize("planning_budget", [1, 2])
def test_gate_rejected_dreams_still_persist_the_reserved_future_master(
    planning_budget: int,
) -> None:
    config = _config(planning_budget, accepted=False, rollout_horizon=2)
    agent, model, buffer, state = _init(config, seed=11)
    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )

    assert not bool(jnp.any(result.dream_accepted))
    master = result.real_control_result.state.rng_key
    expected_control = result.real_control_result.state.replace(
        rng_key=_advance_master(master, planning_budget)
    )
    chex.assert_trees_all_equal(result.state.control_state, expected_control)
    _, expected_behavior = _real_behavior_state(config, state)
    chex.assert_trees_all_equal(result.state.behavior_model_state, expected_behavior)


def test_accepted_dream_uses_dedicated_candidate_behavior_and_control_roots() -> None:
    """A one-step dream exactly follows each branch of the four-way split."""
    config = _config(1, accepted=True)
    agent, model, buffer, state = _init(config, seed=0)
    learner_state = state.world_model_state.learner_state
    head_params = learner_state.head_params
    state = state.replace(
        control_state=state.control_state.replace(
            q_bias=jnp.arange(N_ACTIONS, dtype=jnp.float32)
        ),
        world_model_state=state.world_model_state.replace(
            learner_state=learner_state.replace(
                head_params=head_params.replace(
                    biases=(
                        *head_params.biases[:-1],
                        jnp.array([10.0], dtype=jnp.float32),
                    )
                )
            )
        ),
    )
    reward = jnp.array(0.25, dtype=jnp.float32)
    next_observation = jnp.array([0.3, -0.1], dtype=jnp.float32)
    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        reward,
        next_observation,
    )
    assert bool(result.dream_accepted[0])

    control_after_real = result.real_control_result.state
    model_after_real = result.real_model_result.state
    behavior_model, behavior_after_real = _real_behavior_state(config, state)
    buffer_after_real = buffer.add(state.buffer_state, next_observation)
    reserved, candidate_root, behavior_root, control_root = jr.split(
        control_after_real.rng_key, 4
    )

    candidate_key = jr.split(candidate_root, 1)[0]
    anchor_key, sample_key = jr.split(candidate_key)
    anchor_observation, _ = buffer.sample(buffer_after_real, anchor_key)
    candidate = behavior_model.sample_action(
        behavior_after_real.replace(rng_key=sample_key),
        anchor_observation,
    )
    prediction = model.predict(
        model_after_real,
        anchor_observation,
        candidate.action,
    )

    dedicated_input = control_after_real.replace(
        last_observation=anchor_observation,
        last_action=candidate.action,
        rng_key=control_root,
        q_trace_weights=jnp.zeros_like(control_after_real.q_trace_weights),
        q_trace_bias=jnp.zeros_like(control_after_real.q_trace_bias),
    )
    expected_dream = agent.update(
        dedicated_input,
        prediction.reward,
        prediction.next_observation,
        discount=prediction.discount,
    )
    expected_control = expected_dream.state.replace(
        average_reward=dedicated_input.average_reward,
        last_observation=control_after_real.last_observation,
        last_action=control_after_real.last_action,
        epsilon=control_after_real.epsilon,
        step_count=control_after_real.step_count,
        step_words=control_after_real.step_words,
        q_trace_weights=control_after_real.q_trace_weights,
        q_trace_bias=control_after_real.q_trace_bias,
        rng_key=reserved,
    )
    expected_behavior = behavior_model.sample_action(
        behavior_after_real.replace(rng_key=behavior_root),
        prediction.next_observation,
    ).state

    # Validate that this fixture observes the former parent-key alias instead
    # of passing merely because two random actions happened to match.
    aliased_input = dedicated_input.replace(rng_key=control_after_real.rng_key)
    aliased_dream = agent.update(
        aliased_input,
        prediction.reward,
        prediction.next_observation,
        discount=prediction.discount,
    )
    assert int(expected_dream.action) != int(aliased_dream.action)
    assert not bool(jnp.array_equal(expected_dream.td_error, aliased_dream.td_error))

    chex.assert_trees_all_equal(result.dream_td_errors[0], expected_dream.td_error)
    chex.assert_trees_all_equal(result.state.control_state, expected_control)
    chex.assert_trees_all_equal(result.state.behavior_model_state, expected_behavior)


def test_rejected_control_update_advances_rng_and_repaired_state_recovers() -> None:
    """Transactional control rejection cannot freeze the dream master stream."""
    config = _config(1, accepted=True)
    agent, model, buffer, state = _init(config, seed=31)
    valid_weights = state.control_state.q_weights
    corrupt = state.replace(
        control_state=state.control_state.replace(
            q_weights=jnp.full_like(valid_weights, jnp.nan)
        )
    )
    real_parent = corrupt.control_state.rng_key
    rejected = step9_update(
        config,
        agent,
        model,
        buffer,
        corrupt,
        jnp.array(0.25, dtype=jnp.float32),
        jnp.array([0.2, -0.3], dtype=jnp.float32),
    )

    assert not bool(rejected.real_control_result.update_applied)
    assert not bool(rejected.dream_accepted[0])
    real_reserved, real_explore, real_tie_break, real_random_action = jr.split(
        real_parent, 4
    )
    dream_reserved, candidate, behavior, control = jr.split(real_reserved, 4)
    _assert_key_equal(rejected.real_control_result.state.rng_key, real_reserved)
    _assert_key_equal(rejected.state.control_state.rng_key, dream_reserved)
    _assert_pairwise_distinct(
        (
            real_explore,
            real_tie_break,
            real_random_action,
            dream_reserved,
            candidate,
            behavior,
            control,
        )
    )

    repaired = rejected.state.replace(
        control_state=rejected.state.control_state.replace(q_weights=valid_weights)
    )
    recovered = step9_update(
        config,
        agent,
        model,
        buffer,
        repaired,
        jnp.array(0.125, dtype=jnp.float32),
        jnp.array([-0.1, 0.4], dtype=jnp.float32),
    )
    assert bool(recovered.real_control_result.update_applied)
    assert bool(recovered.dream_accepted[0])
    chex.assert_tree_all_finite(recovered.state.control_state.q_weights)
    chex.assert_tree_all_finite(recovered.state.control_state.q_bias)
    chex.assert_tree_all_finite(recovered.state.world_model_state.model_error_ema)


def test_rejected_rollout_updates_advance_the_local_control_stream() -> None:
    """Each rejected rollout update receives a new local control key."""
    config = _config(1)
    agent, _model, _buffer, state = _init(config, seed=37)
    corrupt_control = state.control_state.replace(
        q_weights=jnp.full_like(state.control_state.q_weights, jnp.nan)
    )
    reward = jnp.array(0.0, dtype=jnp.float32)
    next_observation = jnp.array([0.1, -0.1], dtype=jnp.float32)
    discount = jnp.array(0.9, dtype=jnp.float32)

    first = _update_control_with_linear_rng(
        agent,
        corrupt_control,
        reward,
        next_observation,
        discount,
    )
    second = _update_control_with_linear_rng(
        agent,
        first.state,
        reward,
        next_observation,
        discount,
    )

    assert not bool(first.update_applied)
    assert not bool(second.update_applied)
    first_expected = jr.split(corrupt_control.rng_key, 4)[0]
    second_expected = jr.split(first_expected, 4)[0]
    _assert_key_equal(first.state.rng_key, first_expected)
    _assert_key_equal(second.state.rng_key, second_expected)


def test_invalid_next_observation_rejects_before_building_a_control_candidate() -> None:
    """The ``-1`` selection sentinel must never index the SARSA target."""
    config = _config(1)
    agent, _model, _buffer, state = _init(config, seed=43)
    control = state.control_state
    result = _update_control_with_linear_rng(
        agent,
        control,
        jnp.array(0.25, dtype=jnp.float32),
        jnp.full((OBSERVATION_DIM,), jnp.nan, dtype=jnp.float32),
        jnp.array(0.9, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    assert not bool(result.input_valid)
    assert bool(result.candidate_state_finite)
    expected_key = jr.split(control.rng_key, 4)[0]
    chex.assert_trees_all_equal(
        result.state,
        control.replace(rng_key=expected_key),
    )


def test_eager_jit_and_scan_reproduce_the_same_rng_trajectory() -> None:
    config = _config(2, accepted=True, rollout_horizon=3, candidate_count=2)
    agent, model, buffer, state = _init(config, seed=47)
    reward = jnp.array(0.375, dtype=jnp.float32)
    next_observation = jnp.array([0.4, -0.2], dtype=jnp.float32)

    with jax.disable_jit():
        eager = step9_update(
            config, agent, model, buffer, state, reward, next_observation
        )
    compiled = step9_update(
        config, agent, model, buffer, state, reward, next_observation
    )
    repeated = step9_update(
        config, agent, model, buffer, state, reward, next_observation
    )
    _assert_key_equal(
        eager.state.control_state.rng_key,
        compiled.state.control_state.rng_key,
    )
    _assert_key_equal(
        eager.state.behavior_model_state.rng_key,
        compiled.state.behavior_model_state.rng_key,
    )
    chex.assert_trees_all_equal(eager.dream_td_errors, compiled.dream_td_errors)
    chex.assert_trees_all_equal(eager.dream_accepted, compiled.dream_accepted)
    chex.assert_trees_all_equal(
        eager.real_control_result.action,
        compiled.real_control_result.action,
    )
    chex.assert_trees_all_equal(compiled, repeated)

    rewards = jnp.array([0.25, -0.5, 0.125], dtype=jnp.float32)
    observations = jnp.array(
        [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.25]],
        dtype=jnp.float32,
    )
    first_scan = run_step9_scan(
        config, agent, model, buffer, state, rewards, observations
    )
    second_scan = run_step9_scan(
        config, agent, model, buffer, state, rewards, observations
    )
    with jax.disable_jit():
        eager_scan = run_step9_scan(
            config, agent, model, buffer, state, rewards, observations
        )
    chex.assert_trees_all_equal(first_scan, second_scan)
    _assert_key_equal(
        eager_scan.state.control_state.rng_key,
        first_scan.state.control_state.rng_key,
    )
    _assert_key_equal(
        eager_scan.state.behavior_model_state.rng_key,
        first_scan.state.behavior_model_state.rng_key,
    )
    chex.assert_trees_all_equal(
        eager_scan.dream_accepted,
        first_scan.dream_accepted,
    )
