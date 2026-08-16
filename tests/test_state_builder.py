"""Contract, learning, resource, and checkpoint tests for state builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.state_builder import (
    STATE_BUILDER_CHECKPOINT_SCHEMA,
    FixedTraceStateBuilder,
    FixedTraceStateBuilderConfig,
    IdentityStateBuilder,
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    StateBuilder,
    StateBuilderBudget,
    StateBuilderConfig,
    StateBuilderLearningProposal,
    load_state_builder_checkpoint,
    replace_state_builder_learning_proposal_update,
    save_state_builder_checkpoint,
    state_builder_config_from_config,
    state_builder_from_config,
)


def test_state_builder_public_exports_resolve_to_core_implementation() -> None:
    assert alberta.StateBuilder is core.StateBuilder
    assert alberta.OnlineGatedStateBuilder is core.OnlineGatedStateBuilder
    assert alberta.state_builder_from_config is core.state_builder_from_config
    assert alberta.save_state_builder_checkpoint is core.save_state_builder_checkpoint
    assert alberta.StateBuilderLearningProposal is StateBuilderLearningProposal
    assert (
        alberta.replace_state_builder_learning_proposal_update
        is replace_state_builder_learning_proposal_update
    )


def _state_scalar_count(state: object) -> int:
    return sum(
        int(np.prod(np.asarray(leaf).shape, dtype=np.int64))
        for leaf in jax.tree_util.tree_leaves(state)
    )


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=3)),
        FixedTraceStateBuilder(FixedTraceStateBuilderConfig(observation_dim=3, n_actions=2)),
        OnlineGatedStateBuilder(
            OnlineGatedStateBuilderConfig(
                observation_dim=3,
                n_actions=2,
                hidden_dim=4,
            )
        ),
    ],
)
def test_builders_satisfy_runtime_contract_and_exact_state_budget(
    builder: StateBuilder[object],
) -> None:
    assert isinstance(builder, StateBuilder)
    state = builder.init(jr.key(0))
    budget = builder.resource_budget()

    assert budget.output_scalars == builder.feature_dim()
    assert budget.state_scalars == _state_scalar_count(state)
    assert budget.state_bytes == 4 * budget.state_scalars
    assert budget.trainable_scalars <= budget.state_scalars


def test_identity_is_observation_only_and_encode_is_pure() -> None:
    builder = IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2))
    state = builder.init(jr.key(0))
    observation = jnp.asarray([1.0, -2.0])

    state, features = builder.start(state, observation)
    encoded = builder.encode(state, observation)
    learned_state, diagnostics = builder.learn(state, jnp.ones(2))

    chex.assert_trees_all_close(features, observation)
    chex.assert_trees_all_close(encoded, observation)
    chex.assert_trees_all_equal(learned_state, state)
    assert int(state.step_count) == 1
    assert float(diagnostics.parameter_update_norm) == 0.0
    assert bool(diagnostics.valid)
    assert not bool(diagnostics.rejected)


def test_identity_episode_reset_is_a_noop_and_start_remains_monotonic() -> None:
    builder = IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2))
    state, _ = builder.start(builder.init(jr.key(0)), jnp.asarray([1.0, -2.0]))

    reset_state = builder.reset_episode(state)
    restarted_state, features = builder.start(reset_state, jnp.asarray([3.0, 4.0]))

    chex.assert_trees_all_equal(reset_state, state)
    chex.assert_trees_all_close(features, jnp.asarray([3.0, 4.0]))
    assert int(reset_state.step_count) == 1
    assert int(restarted_state.step_count) == 2


def test_fixed_trace_state_is_post_update_and_encode_does_not_advance() -> None:
    builder = FixedTraceStateBuilder(
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            outcome_decay_rates=(),
        )
    )
    state = builder.init(jr.key(0))
    state, first = builder.start(state, jnp.asarray([2.0]))
    encoded = builder.encode(state, jnp.asarray([2.0]))
    next_state, second = builder.update(
        state,
        jnp.asarray([0.0]),
        -1,
        0.0,
        1.0,
    )

    # Representation is [raw observation, post-update trace].
    chex.assert_trees_all_close(first, jnp.asarray([2.0, 1.0]))
    chex.assert_trees_all_close(encoded, first)
    chex.assert_trees_all_close(second, jnp.asarray([0.0, 0.5]))
    assert int(state.step_count) == 1
    assert int(next_state.step_count) == 2


def test_fixed_trace_episode_reset_clears_memory_but_preserves_event_count() -> None:
    builder = FixedTraceStateBuilder(
        FixedTraceStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            observation_decay_rates=(0.5, 0.9),
            action_decay_rates=(0.75,),
            outcome_decay_rates=(0.25,),
        )
    )
    state, _ = builder.start(
        builder.init(jr.key(0)),
        jnp.asarray([1.0, -2.0]),
        last_action=1,
        last_reward=3.0,
        last_discount=0.0,
    )
    state, _ = builder.update(state, jnp.asarray([0.5, 0.25]), 0, -1.0, 0.5)

    reset_state = builder.reset_episode(state)

    chex.assert_trees_all_equal(
        reset_state.observation_traces,
        jnp.zeros_like(reset_state.observation_traces),
    )
    chex.assert_trees_all_equal(
        reset_state.action_traces,
        jnp.zeros_like(reset_state.action_traces),
    )
    chex.assert_trees_all_equal(
        reset_state.reward_traces,
        jnp.zeros_like(reset_state.reward_traces),
    )
    chex.assert_trees_all_equal(reset_state.last_gate, jnp.ones(3))
    assert int(reset_state.step_count) == 2

    restarted_state, _ = builder.start(reset_state, jnp.asarray([-0.5, 0.75]))
    assert int(restarted_state.step_count) == 3
    assert bool(jnp.any(restarted_state.observation_traces != 0.0))


def test_online_gated_builder_updates_recurrent_parameters_from_delayed_gradient() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            hidden_dim=3,
            step_size=0.05,
        )
    )
    state = builder.init(jr.key(7))
    initial_parameters = state.parameters

    state, _ = builder.start(state, jnp.asarray([1.0, 1.0]))
    for _ in range(4):
        state, features = builder.update(
            state,
            jnp.asarray([0.0, 0.0]),
            -1,
            0.0,
            1.0,
        )
    gradient = jnp.concatenate([jnp.zeros(2), jnp.ones(3)])
    learned_state, diagnostics = builder.learn(state, gradient)

    assert features.shape == (5,)
    assert float(jnp.linalg.norm(learned_state.parameters - initial_parameters)) > 0.0
    assert float(diagnostics.gradient_norm) > 0.0
    assert float(diagnostics.parameter_update_norm) > 0.0
    assert int(learned_state.update_count) == 1
    assert bool(diagnostics.valid)
    assert not bool(diagnostics.rejected)
    chex.assert_tree_all_finite(learned_state)


def test_online_gated_episode_reset_preserves_learning_and_lifetime_counters() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.05,
        )
    )
    state, _ = builder.start(
        builder.init(jr.key(12)),
        jnp.asarray([1.0, -0.5]),
        last_action=1,
        last_reward=0.25,
        last_discount=0.9,
    )
    state, _ = builder.learn(state, jnp.ones(builder.feature_dim(), dtype=jnp.float32))
    assert bool(jnp.any(state.hidden != 0.0))
    assert bool(jnp.any(state.parameter_sensitivity != 0.0))

    reset_state = builder.reset_episode(state)

    chex.assert_trees_all_equal(reset_state.parameters, state.parameters)
    chex.assert_trees_all_equal(reset_state.hidden, jnp.zeros_like(state.hidden))
    chex.assert_trees_all_equal(
        reset_state.parameter_sensitivity,
        jnp.zeros_like(state.parameter_sensitivity),
    )
    chex.assert_trees_all_equal(reset_state.last_gradient_norm, state.last_gradient_norm)
    assert int(reset_state.step_count) == 1
    assert int(reset_state.update_count) == 1

    restarted_state, _ = builder.start(reset_state, jnp.asarray([-0.25, 0.75]))
    assert int(restarted_state.step_count) == 2
    assert int(restarted_state.update_count) == 1
    assert bool(jnp.any(restarted_state.parameter_sensitivity != 0.0))


def _online_learning_source() -> tuple[
    OnlineGatedStateBuilder,
    object,
    jax.Array,
]:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.025,
            gradient_clip=2.0,
        )
    )
    state, _ = builder.start(
        builder.init(jr.key(101)),
        jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        last_action=1,
        last_reward=0.2,
        last_discount=0.9,
    )
    gradient = jnp.asarray([0.1, -0.2, 0.4, -0.3, 0.5], dtype=jnp.float32)
    return builder, state, gradient


def test_online_learning_proposal_is_pure_source_bound_and_uses_current_sensitivity() -> None:
    builder, source, gradient = _online_learning_source()
    source_before = jax.tree_util.tree_map(lambda value: value.copy(), source)

    proposal = builder.propose_learning_update(source, gradient)

    chex.assert_trees_all_equal(source, source_before)
    chex.assert_trees_all_equal(proposal.source_parameters, source.parameters)
    chex.assert_trees_all_equal(proposal.source_update_count, source.update_count)
    expected_raw_gradient = source.parameter_sensitivity.T @ gradient[-builder.config.hidden_dim :]
    chex.assert_trees_all_close(proposal.raw_parameter_gradient, expected_raw_gradient)
    chex.assert_trees_all_close(
        proposal.candidate_parameter_update,
        -builder.config.step_size * proposal.clipped_parameter_gradient,
    )
    assert bool(proposal.valid)
    assert not bool(proposal.rejected)
    assert not bool(proposal.fixed_noop)
    assert not bool(proposal.candidate_update_transformed)
    assert bool(proposal.candidate_update_approved)


def test_online_proposal_from_source_commits_into_advanced_destination_causally() -> None:
    builder, source, gradient = _online_learning_source()
    proposal = builder.propose_learning_update(source, gradient)
    destination, _ = builder.update(
        source,
        jnp.asarray([-0.4, 0.75], dtype=jnp.float32),
        0,
        -0.1,
        0.8,
    )

    eager_state, eager_diagnostics = builder.commit_learning_update(destination, proposal)
    compiled_state, compiled_diagnostics = jax.jit(builder.commit_learning_update)(
        destination,
        proposal,
    )

    chex.assert_trees_all_equal(compiled_state, eager_state)
    chex.assert_trees_all_close(compiled_diagnostics, eager_diagnostics)
    chex.assert_trees_all_close(
        eager_state.parameters,
        destination.parameters + proposal.candidate_parameter_update,
    )
    chex.assert_trees_all_equal(eager_state.hidden, destination.hidden)
    chex.assert_trees_all_equal(
        eager_state.parameter_sensitivity,
        destination.parameter_sensitivity,
    )
    chex.assert_trees_all_equal(eager_state.step_count, destination.step_count)
    assert int(eager_state.update_count) == int(destination.update_count) + 1
    chex.assert_trees_all_equal(eager_state.last_gradient_norm, proposal.gradient_norm)
    assert bool(eager_diagnostics.source_matches)
    assert bool(eager_diagnostics.applied)
    assert bool(eager_diagnostics.valid)


def test_online_commit_rejects_stale_wrong_builder_and_reused_proposals_exactly() -> None:
    builder, source, gradient = _online_learning_source()
    proposal = builder.propose_learning_update(source, gradient)
    first_state, _ = builder.commit_learning_update(source, proposal)
    reused_state, reused_diagnostics = builder.commit_learning_update(first_state, proposal)
    chex.assert_trees_all_equal(reused_state, first_state)
    assert not bool(reused_diagnostics.source_matches)
    assert bool(reused_diagnostics.rejected)

    stale = source.replace(parameters=source.parameters.at[0].add(jnp.float32(1.0e-3)))
    stale_state, stale_diagnostics = jax.jit(builder.commit_learning_update)(stale, proposal)
    chex.assert_trees_all_equal(stale_state, stale)
    assert not bool(stale_diagnostics.source_matches)
    assert not bool(stale_diagnostics.applied)

    wrong_builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.05,
            gradient_clip=2.0,
        )
    )
    wrong_state, wrong_diagnostics = wrong_builder.commit_learning_update(source, proposal)
    chex.assert_trees_all_equal(wrong_state, source)
    assert not bool(wrong_diagnostics.source_matches)
    assert bool(wrong_diagnostics.rejected)


def test_learning_update_transform_requires_explicit_scalar_bool_approval() -> None:
    builder, source, gradient = _online_learning_source()
    proposal = builder.propose_learning_update(source, gradient)
    zero_update = jnp.zeros_like(proposal.candidate_parameter_update)

    vetoed = replace_state_builder_learning_proposal_update(
        proposal,
        zero_update,
        jnp.asarray(False),
    )
    eager_state, eager_diagnostics = builder.commit_learning_update(source, vetoed)
    compiled_vetoed = jax.jit(replace_state_builder_learning_proposal_update)(
        proposal,
        zero_update,
        jnp.asarray(False),
    )
    compiled_state, compiled_diagnostics = jax.jit(builder.commit_learning_update)(
        source,
        compiled_vetoed,
    )
    chex.assert_trees_all_equal(eager_state, source)
    chex.assert_trees_all_equal(compiled_state, source)
    chex.assert_trees_all_equal(compiled_vetoed, vetoed)
    chex.assert_trees_all_equal(compiled_diagnostics, eager_diagnostics)
    assert bool(vetoed.candidate_update_transformed)
    assert not bool(vetoed.candidate_update_approved)
    assert not bool(vetoed.valid)
    assert bool(eager_diagnostics.rejected)

    approved = replace_state_builder_learning_proposal_update(
        proposal,
        zero_update,
        jnp.asarray(True),
    )
    approved_state, approved_diagnostics = builder.commit_learning_update(source, approved)
    chex.assert_trees_all_equal(approved_state.parameters, source.parameters)
    assert int(approved_state.update_count) == int(source.update_count) + 1
    assert bool(approved_diagnostics.applied)

    with pytest.raises(TypeError, match="approved must be an array"):
        replace_state_builder_learning_proposal_update(proposal, zero_update, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="approved must have shape"):
        replace_state_builder_learning_proposal_update(
            proposal,
            zero_update,
            jnp.asarray([True]),
        )
    with pytest.raises(TypeError, match="approved must have dtype bool"):
        jax.jit(replace_state_builder_learning_proposal_update)(
            proposal,
            zero_update,
            jnp.asarray(1, dtype=jnp.int32),
        )


def test_online_learn_equals_propose_then_same_state_commit() -> None:
    builder, source, gradient = _online_learning_source()
    proposal = builder.propose_learning_update(source, gradient)
    committed_state, committed_diagnostics = builder.commit_learning_update(source, proposal)
    learned_state, learned_diagnostics = builder.learn(source, gradient)

    chex.assert_trees_all_equal(learned_state, committed_state)
    chex.assert_trees_all_equal(learned_diagnostics, committed_diagnostics)


def test_online_proposal_capacity_max_minus_one_and_exhaustion_are_fail_closed() -> None:
    builder, source, gradient = _online_learning_source()
    almost_exhausted = source.replace(update_count=jnp.asarray(2**31 - 2, dtype=jnp.int32))
    proposal = builder.propose_learning_update(almost_exhausted, gradient)
    final_state, final_diagnostics = builder.commit_learning_update(
        almost_exhausted,
        proposal,
    )
    assert bool(proposal.capacity_available)
    assert bool(final_diagnostics.applied)
    assert int(final_state.update_count) == 2**31 - 1

    exhausted_proposal = builder.propose_learning_update(final_state, gradient)
    exhausted_state, exhausted_diagnostics = jax.jit(builder.commit_learning_update)(
        final_state,
        exhausted_proposal,
    )
    chex.assert_trees_all_equal(exhausted_state, final_state)
    assert not bool(exhausted_proposal.capacity_available)
    assert not bool(exhausted_diagnostics.capacity_available)
    assert bool(exhausted_diagnostics.rejected)


def test_online_commit_rejects_corrupt_proposal_and_static_contract_errors() -> None:
    builder, source, gradient = _online_learning_source()
    proposal = builder.propose_learning_update(source, gradient)
    corrupt_proposals = (
        proposal.replace(gradient_norm=jnp.asarray(0.0, dtype=jnp.float32)),
        proposal.replace(
            candidate_parameter_update=proposal.candidate_parameter_update.at[0].set(jnp.nan)
        ),
        proposal.replace(candidate_parameters_valid=jnp.asarray(False)),
    )
    for corrupt in corrupt_proposals:
        eager_state, eager_diagnostics = builder.commit_learning_update(source, corrupt)
        compiled_state, compiled_diagnostics = jax.jit(builder.commit_learning_update)(
            source,
            corrupt,
        )
        chex.assert_trees_all_equal(eager_state, source)
        chex.assert_trees_all_equal(compiled_state, source)
        chex.assert_trees_all_equal(compiled_diagnostics, eager_diagnostics)
        assert not bool(eager_diagnostics.proposal_valid)
        assert bool(eager_diagnostics.rejected)
        chex.assert_tree_all_finite(eager_diagnostics)

    wrong_shape = proposal.replace(
        source_parameters=jnp.zeros(
            (builder.config.parameter_count() + 1,),
            dtype=jnp.float32,
        )
    )
    with pytest.raises(ValueError, match="proposal.source_parameters must have shape"):
        builder.commit_learning_update(source, wrong_shape)
    wrong_dtype = proposal.replace(
        candidate_parameter_update=proposal.candidate_parameter_update.astype(jnp.float16)
    )
    with pytest.raises(TypeError, match="proposal.candidate_parameter_update must have dtype"):
        jax.jit(builder.commit_learning_update)(source, wrong_dtype)


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2)),
        FixedTraceStateBuilder(FixedTraceStateBuilderConfig(observation_dim=2, n_actions=2)),
    ],
    ids=("identity", "fixed-trace"),
)
def test_fixed_builders_propose_and_commit_honest_noops(
    builder: StateBuilder[object],
) -> None:
    source = builder.init(jr.key(0))
    gradient = jnp.ones((builder.feature_dim(),), dtype=jnp.float32)
    proposal = builder.propose_learning_update(source, gradient)
    destination, _ = builder.start(
        source,
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        last_action=1,
    )
    committed, diagnostics = builder.commit_learning_update(destination, proposal)

    chex.assert_trees_all_equal(committed, destination)
    assert proposal.source_parameters.shape == (0,)
    assert proposal.candidate_parameter_update.shape == (0,)
    assert bool(proposal.fixed_noop)
    assert bool(proposal.valid)
    assert bool(diagnostics.fixed_noop)
    assert not bool(diagnostics.applied)
    assert bool(diagnostics.valid)

    invalid = builder.propose_learning_update(
        source,
        gradient.at[0].set(jnp.nan),
    )
    rejected_state, rejected_diagnostics = builder.commit_learning_update(
        destination,
        invalid,
    )
    chex.assert_trees_all_equal(rejected_state, destination)
    assert not bool(invalid.valid)
    assert bool(rejected_diagnostics.rejected)


def test_event_uses_current_observation_and_preceding_transition_values() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=3,
            hidden_dim=2,
        )
    )
    event = builder._event(  # noqa: SLF001
        jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        1,
        -0.5,
        0.0,
    )
    chex.assert_trees_all_close(
        event,
        jnp.asarray(
            [0.25, -0.75, 0.0, 1.0, 0.0, -0.5, 0.0],
            dtype=jnp.float32,
        ),
    )

    state = builder.init(jr.key(1))
    started, _ = builder.start(
        state,
        jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        last_action=1,
        last_reward=-0.5,
        last_discount=0.0,
    )
    expected_hidden = builder._transition(  # noqa: SLF001
        state.parameters,
        state.hidden,
        event,
    )
    chex.assert_trees_all_close(started.hidden, expected_hidden)


def test_online_recurrent_sensitivity_and_learn_match_central_finite_difference() -> None:
    config = OnlineGatedStateBuilderConfig(
        observation_dim=2,
        n_actions=2,
        hidden_dim=2,
        step_size=0.03,
        gradient_clip=1.0e6,
        initialization_scale=0.15,
    )
    builder = OnlineGatedStateBuilder(config)
    initial_state = builder.init(jr.key(13))
    observations = jnp.asarray(
        [[0.4, -0.7], [1.1, 0.2], [-0.3, 0.8]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.2, -0.5, 0.7], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.4, 1.0], dtype=jnp.float32)

    state = initial_state
    for index in range(observations.shape[0]):
        state, _ = builder.update(
            state,
            observations[index],
            actions[index],
            rewards[index],
            discounts[index],
        )

    def unrolled_hidden(parameters: jax.Array) -> jax.Array:
        hidden = initial_state.hidden
        for index in range(observations.shape[0]):
            event = builder._event(  # noqa: SLF001
                observations[index],
                actions[index],
                rewards[index],
                discounts[index],
            )
            hidden = builder._transition(  # noqa: SLF001
                parameters,
                hidden,
                event,
            )
        return hidden

    epsilon = jnp.asarray(1.0e-3, dtype=jnp.float32)
    basis = jnp.eye(config.parameter_count(), dtype=jnp.float32)
    finite_difference_sensitivity = jax.vmap(
        lambda direction: (
            (
                unrolled_hidden(initial_state.parameters + epsilon * direction)
                - unrolled_hidden(initial_state.parameters - epsilon * direction)
            )
            / (2.0 * epsilon)
        )
    )(basis).T

    chex.assert_trees_all_close(
        state.parameter_sensitivity,
        finite_difference_sensitivity,
        atol=3.0e-5,
        rtol=3.0e-4,
    )

    representation_gradient = jnp.asarray(
        [0.6, -0.2, 0.7, -1.1],
        dtype=jnp.float32,
    )

    def scalar_loss(parameters: jax.Array) -> jax.Array:
        representation = jnp.concatenate([observations[-1], unrolled_hidden(parameters)])
        return representation_gradient @ representation

    finite_difference_gradient = jax.vmap(
        lambda direction: (
            (
                scalar_loss(initial_state.parameters + epsilon * direction)
                - scalar_loss(initial_state.parameters - epsilon * direction)
            )
            / (2.0 * epsilon)
        )
    )(basis)
    learned_state, diagnostics = builder.learn(state, representation_gradient)
    implemented_gradient = (state.parameters - learned_state.parameters) / config.step_size

    chex.assert_trees_all_close(
        implemented_gradient,
        finite_difference_gradient,
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    chex.assert_trees_all_close(
        diagnostics.gradient_norm,
        jnp.linalg.norm(finite_difference_gradient),
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    chex.assert_trees_all_close(
        diagnostics.clipped_gradient_norm,
        diagnostics.gradient_norm,
        atol=1.0e-6,
    )


def test_online_gated_learn_rejects_static_shape_and_dtype_errors() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=2)
    )
    state = builder.init(jr.key(2))

    with pytest.raises(ValueError, match="representation_gradient must have shape"):
        builder.learn(state, jnp.ones((builder.feature_dim() + 1,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="representation_gradient must have dtype float32"):
        builder.learn(state, jnp.ones((builder.feature_dim(),), dtype=jnp.float16))
    with pytest.raises(TypeError, match="representation_gradient must have dtype float32"):
        builder.learn(state, np.ones((builder.feature_dim(),), dtype=np.float64))
    with pytest.raises(TypeError, match="static shape and dtype metadata"):
        builder.learn(state, [1.0] * builder.feature_dim())  # type: ignore[arg-type]

    corrupt_shape = state.replace(parameters=jnp.zeros((state.parameters.size + 1,)))
    with pytest.raises(ValueError, match="state.parameters must have shape"):
        builder.learn(corrupt_shape, jnp.ones(builder.feature_dim(), dtype=jnp.float32))

    corrupt_dtype = state.replace(hidden=state.hidden.astype(jnp.float16))
    with pytest.raises(TypeError, match="state.hidden must have dtype float32"):
        jax.jit(builder.learn)(
            corrupt_dtype,
            jnp.ones(builder.feature_dim(), dtype=jnp.float32),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_online_gated_learn_nonfinite_gradient_is_atomic_noop(invalid: float) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=2)
    )
    state, _ = builder.start(builder.init(jr.key(6)), jnp.asarray([0.5, -0.25]))
    gradient = jnp.ones(builder.feature_dim(), dtype=jnp.float32).at[-1].set(invalid)

    eager_state, eager_diagnostics = builder.learn(state, gradient)
    compiled_state, compiled_diagnostics = jax.jit(builder.learn)(state, gradient)

    chex.assert_trees_all_equal(eager_state, state)
    chex.assert_trees_all_equal(compiled_state, state)
    chex.assert_trees_all_equal(compiled_diagnostics, eager_diagnostics)
    assert not bool(eager_diagnostics.valid)
    assert bool(eager_diagnostics.rejected)
    chex.assert_tree_all_finite(eager_diagnostics)


def test_online_gated_learn_corrupt_dynamic_state_is_atomic_noop() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=2)
    )
    state, _ = builder.start(builder.init(jr.key(8)), jnp.asarray([0.5, -0.25]))
    gradient = jnp.ones(builder.feature_dim(), dtype=jnp.float32)
    corrupt_states = (
        state.replace(parameters=state.parameters.at[0].set(jnp.nan)),
        state.replace(hidden=state.hidden.at[0].set(jnp.inf)),
        state.replace(parameter_sensitivity=state.parameter_sensitivity.at[0, 0].set(jnp.nan)),
        state.replace(last_gradient_norm=jnp.asarray(jnp.inf, dtype=jnp.float32)),
        state.replace(step_count=jnp.asarray(-1, dtype=jnp.int32)),
        state.replace(update_count=jnp.asarray(-1, dtype=jnp.int32)),
    )

    compiled_learn = jax.jit(builder.learn)
    for corrupt_state in corrupt_states:
        eager_state, eager_diagnostics = builder.learn(corrupt_state, gradient)
        compiled_state, compiled_diagnostics = compiled_learn(corrupt_state, gradient)
        chex.assert_trees_all_equal(eager_state, corrupt_state)
        chex.assert_trees_all_equal(compiled_state, corrupt_state)
        chex.assert_trees_all_equal(compiled_diagnostics, eager_diagnostics)
        assert not bool(eager_diagnostics.valid)
        assert bool(eager_diagnostics.rejected)
        chex.assert_tree_all_finite(eager_diagnostics)


def test_online_gated_learn_scale_safe_clips_extreme_finite_gradient() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=1,
            hidden_dim=1,
            step_size=0.1,
            gradient_clip=10.0,
        )
    )
    state = builder.init(jr.key(11))
    state = state.replace(parameter_sensitivity=jnp.ones_like(state.parameter_sensitivity))
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    gradient = jnp.asarray([0.0, maximum], dtype=jnp.float32)

    eager_state, eager_diagnostics = builder.learn(state, gradient)
    compiled_state, compiled_diagnostics = jax.jit(builder.learn)(state, gradient)

    chex.assert_trees_all_close(compiled_state, eager_state)
    chex.assert_trees_all_close(compiled_diagnostics, eager_diagnostics)
    assert bool(eager_diagnostics.valid)
    assert not bool(eager_diagnostics.rejected)
    assert int(eager_state.update_count) == 1
    assert float(eager_diagnostics.gradient_norm) == float(maximum)
    chex.assert_trees_all_close(eager_diagnostics.clipped_gradient_norm, 10.0, atol=1e-5)
    chex.assert_trees_all_close(eager_diagnostics.parameter_update_norm, 1.0, atol=1e-5)
    chex.assert_tree_all_finite(eager_state)
    chex.assert_tree_all_finite(eager_diagnostics)


def test_online_gated_learn_rejects_overflowing_finite_candidate_atomically() -> None:
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=1,
            hidden_dim=1,
            step_size=1.0,
            gradient_clip=float(np.finfo(np.float32).max),
        )
    )
    state = builder.init(jr.key(15)).replace(
        parameters=jnp.full((builder.config.parameter_count(),), maximum),
        parameter_sensitivity=-jnp.ones((1, builder.config.parameter_count()), dtype=jnp.float32),
    )
    gradient = jnp.asarray([0.0, maximum], dtype=jnp.float32)

    eager_state, eager_diagnostics = builder.learn(state, gradient)
    compiled_state, compiled_diagnostics = jax.jit(builder.learn)(state, gradient)

    chex.assert_trees_all_equal(eager_state, state)
    chex.assert_trees_all_equal(compiled_state, state)
    chex.assert_trees_all_equal(compiled_diagnostics, eager_diagnostics)
    assert not bool(eager_diagnostics.valid)
    assert bool(eager_diagnostics.rejected)
    chex.assert_tree_all_finite(eager_diagnostics)


def test_online_gated_learn_rejects_finite_matmul_overflow_atomically() -> None:
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=1, hidden_dim=1)
    )
    state = builder.init(jr.key(19)).replace(
        parameter_sensitivity=jnp.full(
            (1, builder.config.parameter_count()),
            maximum,
            dtype=jnp.float32,
        )
    )
    gradient = jnp.asarray([0.0, maximum], dtype=jnp.float32)

    learned_state, diagnostics = builder.learn(state, gradient)

    chex.assert_trees_all_equal(learned_state, state)
    assert not bool(diagnostics.valid)
    assert bool(diagnostics.rejected)
    chex.assert_tree_all_finite(diagnostics)


def test_online_gated_python_loop_matches_jitted_scan() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.02,
            gradient_clip=2.0,
        )
    )
    observations = jnp.asarray(
        [[0.2, -0.4], [0.8, 0.1], [-0.5, 0.7], [0.3, 0.9]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.1, -0.2, 0.4, 0.0], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.9, 0.5, 1.0], dtype=jnp.float32)
    gradients = jnp.asarray(
        [
            [0.1, -0.2, 0.3, 0.1, -0.4],
            [-0.3, 0.2, 0.1, -0.2, 0.5],
            [0.0, 0.1, -0.4, 0.3, 0.2],
            [0.2, 0.0, 0.5, -0.1, -0.3],
        ],
        dtype=jnp.float32,
    )
    initial_state = builder.init(jr.key(21))

    loop_state = initial_state
    loop_features = []
    loop_update_norms = []
    for index in range(observations.shape[0]):
        loop_state, features = builder.update(
            loop_state,
            observations[index],
            actions[index],
            rewards[index],
            discounts[index],
        )
        loop_state, diagnostics = builder.learn(loop_state, gradients[index])
        loop_features.append(features)
        loop_update_norms.append(diagnostics.parameter_update_norm)

    def run_scan(initial: object) -> tuple[object, tuple[jax.Array, jax.Array]]:
        def step(
            state: object,
            inputs: tuple[jax.Array, ...],
        ) -> tuple[object, tuple[jax.Array, jax.Array]]:
            observation, action, reward, discount, gradient = inputs
            next_state, features = builder.update(
                state,
                observation,
                action,
                reward,
                discount,
            )
            next_state, diagnostics = builder.learn(next_state, gradient)
            return next_state, (features, diagnostics.parameter_update_norm)

        return jax.lax.scan(
            step,
            initial,
            (observations, actions, rewards, discounts, gradients),
        )

    scan_state, (scan_features, scan_update_norms) = jax.jit(run_scan)(initial_state)

    chex.assert_trees_all_close(scan_state, loop_state, atol=1.0e-6)
    chex.assert_trees_all_close(
        scan_features,
        jnp.stack(loop_features),
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(
        scan_update_norms,
        jnp.stack(loop_update_norms),
        atol=1.0e-6,
    )


def test_online_gated_encode_is_pure_and_does_not_apply_recurrence_twice() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=3)
    )
    state, features = builder.start(
        builder.init(jr.key(3)),
        jnp.asarray([0.25, -0.75]),
    )
    state_before = jax.tree_util.tree_map(lambda value: value.copy(), state)

    encoded_once = builder.encode(state, jnp.asarray([0.25, -0.75]))
    encoded_twice = builder.encode(state, jnp.asarray([0.25, -0.75]))

    chex.assert_trees_all_close(encoded_once, features)
    chex.assert_trees_all_close(encoded_twice, features)
    chex.assert_trees_all_equal(state, state_before)


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2)),
        FixedTraceStateBuilder(
            FixedTraceStateBuilderConfig(
                observation_dim=2,
                n_actions=3,
                observation_decay_rates=(0.25, 0.9),
            )
        ),
        OnlineGatedStateBuilder(
            OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=3,
                hidden_dim=5,
            )
        ),
    ],
)
def test_config_factory_round_trip(builder: StateBuilder[object]) -> None:
    parsed: StateBuilderConfig = state_builder_config_from_config(builder.to_config())
    restored = state_builder_from_config(builder.to_config())
    assert parsed.to_config() == builder.to_config()
    assert restored.to_config() == builder.to_config()
    assert restored.observation_dim() == parsed.observation_dim
    assert restored.feature_dim() == builder.feature_dim()
    assert restored.resource_budget() == builder.resource_budget()


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2)),
        FixedTraceStateBuilder(
            FixedTraceStateBuilderConfig(
                observation_dim=2,
                n_actions=2,
                observation_decay_rates=(0.5, 0.9),
            )
        ),
    ],
    ids=("identity", "fixed-trace"),
)
def test_fixed_builder_checkpoints_restore_config_and_state(
    tmp_path: Path,
    builder: StateBuilder[object],
) -> None:
    state, _ = builder.start(
        builder.init(jr.key(4)),
        jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        last_action=1,
        last_reward=0.75,
        last_discount=0.0,
    )
    checkpoint_path = tmp_path / type(builder).__name__
    save_state_builder_checkpoint(builder, state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)

    assert restored_builder.to_config() == builder.to_config()
    assert restored_builder.resource_budget() == builder.resource_budget()
    chex.assert_trees_all_close(restored_state, state)


def test_online_gated_state_checkpoint_restores_config_parameters_and_sensitivity(
    tmp_path: Path,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.02,
        )
    )
    state, _ = builder.start(builder.init(jr.key(9)), jnp.asarray([1.0, 0.0]))
    state, _ = builder.update(state, jnp.asarray([0.0, 1.0]), 1, 0.5, 0.9)
    state, _ = builder.learn(state, jnp.ones(builder.feature_dim()))
    checkpoint_path = tmp_path / "state_builder"

    save_state_builder_checkpoint(builder, state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)

    assert restored_builder.to_config() == builder.to_config()
    assert restored_builder.resource_budget() == builder.resource_budget()
    chex.assert_trees_all_close(restored_state, state)


def test_state_builder_checkpoint_config_digest_rejects_behavior_tampering(
    tmp_path: Path,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            hidden_dim=3,
            step_size=0.02,
        )
    )
    state = builder.init(jr.key(27))
    original_path = tmp_path / "original"
    save_state_builder_checkpoint(builder, state, original_path)
    metadata = load_checkpoint_metadata(original_path)
    assert metadata["schema"] == STATE_BUILDER_CHECKPOINT_SCHEMA
    assert isinstance(metadata["config_sha256"], str)

    forged_config = dict(metadata["builder_config"])
    forged_config["step_size"] = 0.5
    forged_metadata = dict(metadata)
    forged_metadata["builder_config"] = forged_config
    forged_path = tmp_path / "forged"
    save_checkpoint(state, forged_path, metadata=forged_metadata)

    with pytest.raises(ValueError, match="config digest does not match"):
        load_state_builder_checkpoint(forged_path)

    identity = IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2))
    identity_state = identity.init(jr.key(0))
    identity_path = tmp_path / "identity"
    save_state_builder_checkpoint(identity, identity_state, identity_path)
    identity_metadata = load_checkpoint_metadata(identity_path)
    noncanonical_config = dict(identity_metadata["builder_config"])
    noncanonical_config["ignored_field"] = "must-not-be-silently-dropped"
    noncanonical_metadata = dict(identity_metadata)
    noncanonical_metadata["builder_config"] = noncanonical_config
    noncanonical_metadata["config_sha256"] = hashlib.sha256(
        json.dumps(
            noncanonical_config,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    noncanonical_path = tmp_path / "noncanonical"
    save_checkpoint(identity_state, noncanonical_path, metadata=noncanonical_metadata)

    with pytest.raises(ValueError, match="config is not canonical"):
        load_state_builder_checkpoint(noncanonical_path)


def test_state_builder_v1_checkpoint_without_config_digest_fails_closed(
    tmp_path: Path,
) -> None:
    builder = IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2))
    state = builder.init(jr.key(0))
    legacy_path = tmp_path / "legacy"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.state_builder.v1",
            "builder_config": builder.to_config(),
            "resource_budget": builder.resource_budget().to_config(),
        },
    )

    with pytest.raises(ValueError, match="v2 checkpoint"):
        load_state_builder_checkpoint(legacy_path)


def test_online_gated_checkpoint_resume_matches_uninterrupted_learning(
    tmp_path: Path,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.015,
            gradient_clip=1.5,
        )
    )
    observations = jnp.asarray(
        [
            [0.3, -0.1],
            [0.7, 0.4],
            [-0.2, 0.9],
            [0.5, -0.8],
            [0.1, 0.6],
            [-0.4, -0.3],
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 0, 1, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.2, 0.0, -0.3, 0.6, -0.1, 0.4], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.8, 0.4, 1.0, 0.9, 0.0], dtype=jnp.float32)
    gradients = jnp.reshape(
        jnp.linspace(-0.5, 0.7, observations.shape[0] * builder.feature_dim()),
        (observations.shape[0], builder.feature_dim()),
    )

    def advance(
        active_builder: StateBuilder[object],
        state: object,
        start: int,
        stop: int,
    ) -> tuple[object, jax.Array]:
        emitted = []
        for index in range(start, stop):
            state, features = active_builder.update(
                state,
                observations[index],
                actions[index],
                rewards[index],
                discounts[index],
            )
            state, _ = active_builder.learn(state, gradients[index])
            emitted.append(features)
        return state, jnp.stack(emitted)

    initial_state = builder.init(jr.key(31))
    uninterrupted_state, uninterrupted_features = advance(
        builder,
        initial_state,
        0,
        observations.shape[0],
    )
    prefix_state, prefix_features = advance(builder, initial_state, 0, 3)
    checkpoint_path = tmp_path / "resume_state_builder"
    save_state_builder_checkpoint(builder, prefix_state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)
    resumed_state, suffix_features = advance(
        restored_builder,
        restored_state,
        3,
        observations.shape[0],
    )

    chex.assert_trees_all_close(resumed_state, uninterrupted_state, atol=1.0e-7)
    chex.assert_trees_all_close(
        jnp.concatenate([prefix_features, suffix_features], axis=0),
        uninterrupted_features,
        atol=1.0e-7,
    )


def test_invalid_builder_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        IdentityStateBuilderConfig(observation_dim=0)
    with pytest.raises(ValueError, match="n_actions"):
        FixedTraceStateBuilderConfig(observation_dim=1, n_actions=-1)
    with pytest.raises(ValueError, match="step_size"):
        OnlineGatedStateBuilderConfig(observation_dim=1, step_size=0.0)
    with pytest.raises(ValueError, match="unknown state builder"):
        state_builder_from_config({"type": "not-a-builder"})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_builder_hyperparameters_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="observation_decay_rates"):
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(invalid,),
        )
    with pytest.raises(ValueError, match="step_size"):
        OnlineGatedStateBuilderConfig(observation_dim=1, step_size=invalid)
    with pytest.raises(ValueError, match="gradient_clip"):
        OnlineGatedStateBuilderConfig(observation_dim=1, gradient_clip=invalid)
    with pytest.raises(ValueError, match="initial_gate_bias"):
        OnlineGatedStateBuilderConfig(observation_dim=1, initial_gate_bias=invalid)
    with pytest.raises(ValueError, match="initialization_scale"):
        OnlineGatedStateBuilderConfig(observation_dim=1, initialization_scale=invalid)


def test_builder_configs_reject_booleans_and_non_integers() -> None:
    # Booleans
    with pytest.raises(ValueError, match="observation_dim"):
        IdentityStateBuilderConfig(observation_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observation_dim"):
        FixedTraceStateBuilderConfig(observation_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        FixedTraceStateBuilderConfig(observation_dim=4, n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_raw_observation"):
        FixedTraceStateBuilderConfig(observation_dim=4, include_raw_observation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_dim"):
        OnlineGatedStateBuilderConfig(observation_dim=4, hidden_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_raw_observation"):
        OnlineGatedStateBuilderConfig(observation_dim=4, include_raw_observation=0)  # type: ignore[arg-type]

    # Non-integer floats
    with pytest.raises(ValueError, match="observation_dim"):
        IdentityStateBuilderConfig(observation_dim=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        FixedTraceStateBuilderConfig(observation_dim=4, n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_dim"):
        OnlineGatedStateBuilderConfig(observation_dim=4, hidden_dim=8.5)  # type: ignore[arg-type]

    # Non-tuple decay rates
    with pytest.raises(ValueError, match="observation_decay_rates"):
        FixedTraceStateBuilderConfig(observation_dim=4, observation_decay_rates=[0.5, 0.9])  # type: ignore[arg-type]

    # Float32 boundary decay rate (1.0 - 1e-10 rounds to 1.0 in float32)
    with pytest.raises(ValueError, match="observation_decay_rates"):
        FixedTraceStateBuilderConfig(observation_dim=4, observation_decay_rates=(1.0 - 1e-10,))


def test_builder_configs_accept_and_canonicalize_numpy_integers() -> None:
    cfg_id = IdentityStateBuilderConfig(observation_dim=np.int32(4))
    assert type(cfg_id.observation_dim) is int
    assert cfg_id.observation_dim == 4

    cfg_ft = FixedTraceStateBuilderConfig(
        observation_dim=np.int64(4),
        n_actions=np.uint16(2),
    )
    assert type(cfg_ft.observation_dim) is int
    assert type(cfg_ft.n_actions) is int
    assert cfg_ft.observation_dim == 4
    assert cfg_ft.n_actions == 2

    cfg_og = OnlineGatedStateBuilderConfig(
        observation_dim=np.int32(4),
        n_actions=np.int16(2),
        hidden_dim=np.uint8(8),
    )
    assert type(cfg_og.observation_dim) is int
    assert type(cfg_og.n_actions) is int
    assert type(cfg_og.hidden_dim) is int
    assert cfg_og.observation_dim == 4
    assert cfg_og.n_actions == 2
    assert cfg_og.hidden_dim == 8


def test_state_builder_budget_validation() -> None:
    budget = StateBuilderBudget(
        output_scalars=np.int32(10),
        trainable_scalars=np.int64(20),
        state_scalars=np.uint16(30),
        state_bytes=np.int32(120),
    )
    assert type(budget.output_scalars) is int
    assert type(budget.trainable_scalars) is int
    assert type(budget.state_scalars) is int
    assert type(budget.state_bytes) is int

    large_host_budget = StateBuilderBudget(
        output_scalars=1,
        trainable_scalars=0,
        state_scalars=2**31,
        state_bytes=2**33,
    )
    assert large_host_budget.state_scalars == 2**31
    assert large_host_budget.state_bytes == 2**33

    with pytest.raises(ValueError, match="output_scalars"):
        StateBuilderBudget(
            output_scalars=-1,
            trainable_scalars=0,
            state_scalars=0,
            state_bytes=0,
        )
    with pytest.raises(ValueError, match="state_bytes"):
        StateBuilderBudget(
            output_scalars=1,
            trainable_scalars=0,
            state_scalars=0,
            state_bytes=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: IdentityStateBuilderConfig(observation_dim=value),
        lambda value: FixedTraceStateBuilderConfig(observation_dim=value),
        lambda value: OnlineGatedStateBuilderConfig(observation_dim=value),
    ],
)
@pytest.mark.parametrize("value", [2**31, np.uint64(2**63), -1])
def test_builder_dimensions_reject_values_outside_signed_int32(factory, value) -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        factory(value)


def test_fixed_trace_from_config_does_not_coerce_invalid_scalars() -> None:
    base = FixedTraceStateBuilderConfig(observation_dim=4).to_config()
    for field, value in (
        ("observation_dim", True),
        ("n_actions", False),
        ("include_raw_observation", 1),
        ("include_raw_observation", "false"),
    ):
        payload = {**base, field: value}
        with pytest.raises(ValueError, match=field):
            FixedTraceStateBuilderConfig.from_config(payload)


def test_builder_configs_reject_class_spoofed_scalars() -> None:
    class Spoof:
        @property
        def __class__(self):
            return int

        def __index__(self):
            return 4

    with pytest.raises(ValueError, match="observation_dim"):
        IdentityStateBuilderConfig(observation_dim=Spoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_size"):
        OnlineGatedStateBuilderConfig(  # type: ignore[arg-type]
            observation_dim=4,
            step_size=Spoof(),
        )


@pytest.mark.parametrize("field", ["step_size", "gradient_clip", "initialization_scale"])
@pytest.mark.parametrize("value", [True, 1e100, 1e-100])
def test_online_builder_positive_scalars_are_valid_at_float32_sink(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        OnlineGatedStateBuilderConfig(observation_dim=4, **{field: value})


@pytest.mark.parametrize("value", [True, 1e100])
def test_online_builder_gate_bias_is_finite_at_float32_sink(value) -> None:
    with pytest.raises(ValueError, match="initial_gate_bias"):
        OnlineGatedStateBuilderConfig(observation_dim=4, initial_gate_bias=value)
