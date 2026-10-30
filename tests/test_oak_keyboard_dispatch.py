# mypy: disable-error-code="attr-defined,no-any-return,operator"
"""Ownership and safety contracts for dispatchable option-keyboard actions."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import (
    DISPATCH_OWNER_BASE_PRIMITIVE,
    DISPATCH_OWNER_OPTION,
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
    replace_dispatched_primitive_action,
)

OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)
NEXT_OBS = jnp.array([0.5, 0.0], dtype=jnp.float32)


def _config() -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(
            SubtaskSpec(feature_index=0, threshold=99.0, max_option_steps=8),
            SubtaskSpec(feature_index=1, threshold=99.0, max_option_steps=8),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=0.25,
        base_avg_reward_step_size=0.0,
        option_step_size=0.25,
        option_avg_reward_step_size=0.0,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _primitive_owned_state(agent: STOMPAgent) -> STOMPState:
    state = agent.init(jr.key(7))
    return state.replace(
        base_last_obs=OBS,
        base_last_action=jnp.int32(0),
        last_primitive_action=jnp.int32(0),
        executing_option=jnp.int32(-1),
    )


def _option_owned_state(agent: STOMPAgent) -> STOMPState:
    state = agent.init(jr.key(11))
    return state.replace(
        base_last_obs=OBS,
        base_last_action=jnp.int32(agent.config.n_primitive_actions),
        last_primitive_action=jnp.int32(0),
        executing_option=jnp.int32(0),
        option_start_obs=OBS,
        option_last_intra_action=jnp.int32(0),
        option_cumreward=jnp.float32(0.0),
        option_env_cumreward=jnp.float32(0.0),
        option_baseline_mass=jnp.float32(0.0),
        option_discount=jnp.float32(1.0),
        option_steps=jnp.int32(0),
    )


def test_primitive_owner_override_moves_next_base_credit_to_effective_action() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)

    replaced = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    assert int(replaced.decision.owner) == DISPATCH_OWNER_BASE_PRIMITIVE
    assert int(replaced.decision.counterfactual_action) == 0
    assert int(replaced.decision.effective_action) == 1
    assert bool(replaced.decision.applied)
    assert int(replaced.state.last_primitive_action) == 1
    assert int(replaced.state.base_last_action) == 1
    assert int(replaced.state.option_last_intra_action) == int(state.option_last_intra_action)

    before = replaced.state.base_learner_state.head_params
    updated = agent.update(
        replaced.state,
        jnp.float32(1.0),
        NEXT_OBS,
        jnp.float32(1.0),
        enable_planning=False,
    )
    after = updated.state.base_learner_state.head_params
    chex.assert_trees_all_equal(before.weights[0], after.weights[0])
    assert not bool(jnp.array_equal(before.weights[1], after.weights[1]))


def test_replacement_accepts_valid_fixed_hidden_trunk_learner_contract() -> None:
    config = dataclasses.replace(_config(), base_hidden_sizes=(3, 2))
    agent = STOMPAgent(config)
    state = _primitive_owned_state(agent)
    result = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(result.decision.state_static_contract_valid)
    assert bool(result.decision.state_valid)
    assert int(result.decision.effective_action) == 1


def test_option_owner_override_moves_next_intra_option_credit_to_effective_action() -> None:
    agent = STOMPAgent(_config())
    state = _option_owned_state(agent)

    replaced = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    assert int(replaced.decision.owner) == DISPATCH_OWNER_OPTION
    assert int(replaced.state.last_primitive_action) == 1
    assert int(replaced.state.option_last_intra_action) == 1
    assert int(replaced.state.base_last_action) == agent.config.n_primitive_actions

    before = replaced.state.option_policies.q_weights
    updated = agent.update(
        replaced.state,
        jnp.float32(0.0),
        NEXT_OBS,
        jnp.float32(1.0),
        enable_planning=False,
    )
    after = updated.state.option_policies.q_weights
    chex.assert_trees_all_equal(before[0, 0], after[0, 0])
    assert not bool(jnp.array_equal(before[0, 1], after[0, 1]))
    chex.assert_trees_all_equal(before[1], after[1])


def test_unsafe_proposal_uses_safe_base_fallback_as_exact_state_noop() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)
    result = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        safety_action_mask=jnp.array([True, False], dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    assert int(result.decision.counterfactual_action) == 0
    assert int(result.decision.effective_action) == 0
    assert bool(result.decision.used_safe_base_fallback)
    assert not bool(result.decision.applied)
    assert not bool(result.decision.failed_closed)


def test_unsafe_base_fails_closed_as_exact_state_noop_even_when_proposal_is_safe() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)
    result = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        jnp.array([False, True], dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    assert int(result.decision.effective_action) == -1
    assert bool(result.decision.failed_closed)
    assert not bool(result.decision.applied)


def test_stale_observation_and_invalid_action_fail_closed_as_exact_noops() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)
    mask = jnp.ones((2,), dtype=jnp.bool_)

    stale = replace_dispatched_primitive_action(
        state,
        jnp.nextafter(OBS, jnp.full_like(OBS, jnp.inf)),
        jnp.int32(1),
        mask,
    )
    invalid_action = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(2),
        mask,
    )

    for result in (stale, invalid_action):
        chex.assert_trees_all_equal(result.state, state)
        assert int(result.decision.effective_action) == -1
        assert bool(result.decision.failed_closed)
        assert not bool(result.decision.applied)
    assert not bool(stale.decision.observation_matches)
    assert not bool(invalid_action.decision.proposed_action_valid)


def test_static_input_contract_mismatches_fail_closed_as_exact_noops() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)
    cases = (
        replace_dispatched_primitive_action(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.int32(1),
        ),
        replace_dispatched_primitive_action(
            state,
            OBS,
            jnp.float32(1.0),
        ),
        replace_dispatched_primitive_action(
            state,
            OBS,
            jnp.int32(1),
            safety_action_mask=jnp.ones((2,), dtype=jnp.int32),
        ),
    )
    for result in cases:
        chex.assert_trees_all_equal(result.state, state)
        assert int(result.decision.effective_action) == -1
        assert bool(result.decision.failed_closed)
    assert not bool(cases[0].decision.observation_static_contract_valid)
    assert not bool(cases[1].decision.proposed_action_static_contract_valid)
    assert not bool(cases[2].decision.safety_action_mask_static_contract_valid)


def test_inconsistent_owner_state_fails_closed_as_exact_noop() -> None:
    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent).replace(last_primitive_action=jnp.int32(1))
    result = replace_dispatched_primitive_action(
        state,
        OBS,
        jnp.int32(1),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.decision.state_valid)
    assert int(result.decision.effective_action) == -1


def test_corrupt_stomp_float_counter_and_key_states_each_fail_closed_without_mutation() -> None:
    agent = STOMPAgent(_config())
    valid = _primitive_owned_state(agent)
    learner = valid.base_learner_state
    nan_head_weights = (
        learner.head_params.weights[0].at[0, 0].set(jnp.nan),
        *learner.head_params.weights[1:],
    )
    nan_learner = learner.replace(
        head_params=learner.head_params.replace(weights=nan_head_weights)
    )
    malformed_head_weights = (
        jnp.zeros((1, 3), dtype=jnp.float32),
        *learner.head_params.weights[1:],
    )
    malformed_learner = learner.replace(
        head_params=learner.head_params.replace(weights=malformed_head_weights)
    )
    nan_models = valid.option_models.replace(
        next_state_weights=valid.option_models.next_state_weights.at[0, 0, 0].set(
            jnp.nan
        )
    )
    corrupt_states = (
        valid.replace(base_learner_state=nan_learner),
        valid.replace(option_models=nan_models),
        valid.replace(base_learner_state=malformed_learner),
        valid.replace(step_count=jnp.int32(-1)),
        valid.replace(rng_key=jnp.array([0, 1], dtype=jnp.uint32)),
    )

    for corrupt in corrupt_states:
        result = replace_dispatched_primitive_action(
            corrupt,
            OBS,
            jnp.int32(1),
            safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(result.state, corrupt)
        assert not bool(result.decision.state_valid)
        assert int(result.decision.effective_action) == -1
        assert bool(result.decision.failed_closed)
    assert not bool(
        replace_dispatched_primitive_action(
            corrupt_states[0], OBS, jnp.int32(1)
        ).decision.state_values_finite
    )
    assert not bool(
        replace_dispatched_primitive_action(
            corrupt_states[1], OBS, jnp.int32(1)
        ).decision.state_values_finite
    )
    assert not bool(
        replace_dispatched_primitive_action(
            corrupt_states[2], OBS, jnp.int32(1)
        ).decision.state_static_contract_valid
    )
    assert not bool(
        replace_dispatched_primitive_action(
            corrupt_states[3], OBS, jnp.int32(1)
        ).decision.state_counters_valid
    )
    assert not bool(
        replace_dispatched_primitive_action(
            corrupt_states[4], OBS, jnp.int32(1)
        ).decision.rng_key_valid
    )


def test_replacement_has_eager_jit_and_scan_parity_without_rng_consumption() -> None:
    agent = STOMPAgent(_config())
    state = cast(
        STOMPState,
        jax.tree_util.tree_map(jnp.asarray, _primitive_owned_state(agent)),
    )
    mask = jnp.ones((2,), dtype=jnp.bool_)
    eager = replace_dispatched_primitive_action(state, OBS, jnp.int32(1), mask)
    compiled = jax.jit(replace_dispatched_primitive_action)(
        state, OBS, jnp.int32(1), mask
    )
    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(eager.state.rng_key, state.rng_key)

    proposals = jnp.array([1, 0, 1], dtype=jnp.int32)

    def body(
        carry: STOMPState, proposal: jax.Array
    ) -> tuple[STOMPState, jax.Array]:
        result = replace_dispatched_primitive_action(carry, OBS, proposal, mask)
        return result.state, result.decision.effective_action

    scan_state, scan_actions = jax.lax.scan(body, state, proposals)
    loop_state = state
    loop_actions = []
    for proposal in proposals:
        loop_result = replace_dispatched_primitive_action(loop_state, OBS, proposal, mask)
        loop_state = loop_result.state
        loop_actions.append(loop_result.decision.effective_action)
    chex.assert_trees_all_equal(scan_state, loop_state)
    chex.assert_trees_all_equal(scan_actions, jnp.stack(loop_actions))
    chex.assert_trees_all_equal(scan_state.rng_key, state.rng_key)


def test_oak_keyboard_dispatch_uses_exact_current_q_and_commits_owner() -> None:
    agent = OaKAgent(OaKConfig(stomp=_config()))
    outer_state = agent.init(jr.key(19))
    stomp_state = _option_owned_state(agent.stomp_agent)
    q_weights = stomp_state.option_policies.q_weights
    q_weights = q_weights.at[0].set(
        jnp.array([[0.0, 0.0], [4.0, 0.0]], dtype=jnp.float32)
    )
    q_weights = q_weights.at[1].set(
        jnp.array([[0.0, 0.0], [-1.0, 0.0]], dtype=jnp.float32)
    )
    state = outer_state.replace(
        stomp_state=stomp_state.replace(
            option_policies=stomp_state.option_policies.replace(q_weights=q_weights)
        )
    )
    chord = jnp.array([0.75, 0.25], dtype=jnp.float32)

    result = agent.dispatch_keyboard_policy(
        state,
        OBS,
        chord,
        jnp.ones((2,), dtype=jnp.bool_),
    )

    expected_q = agent.keyboard_q_values(state, OBS, chord)
    chex.assert_trees_all_equal(result.decision.proposal.q_values, expected_q)
    assert int(result.decision.proposal.action) == 1
    assert int(result.decision.replacement.owner) == DISPATCH_OWNER_OPTION
    assert int(result.state.stomp_state.last_primitive_action) == 1
    assert int(result.state.stomp_state.option_last_intra_action) == 1
    chex.assert_trees_all_equal(result.state.stomp_state.rng_key, state.stomp_state.rng_key)
    chex.assert_trees_all_equal(result.state.execution_counts, state.execution_counts)
    chex.assert_trees_all_equal(result.state.utility_ema, state.utility_ema)


def test_oak_keyboard_dispatch_preserves_large_finite_chord_scale() -> None:
    agent = OaKAgent(OaKConfig(stomp=_config()))
    outer_state = agent.init(jr.key(19))
    stomp_state = _option_owned_state(agent.stomp_agent)
    q_weights = stomp_state.option_policies.q_weights
    q_weights = q_weights.at[0].set(
        jnp.array([[0.0, 0.0], [4.0, 0.0]], dtype=jnp.float32)
    )
    q_weights = q_weights.at[1].set(
        jnp.array([[0.0, 0.0], [-1.0, 0.0]], dtype=jnp.float32)
    )
    state = outer_state.replace(
        stomp_state=stomp_state.replace(
            option_policies=stomp_state.option_policies.replace(q_weights=q_weights)
        )
    )
    unit_chord = jnp.ones((2,), dtype=jnp.float32)
    large_chord = jnp.full((2,), jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    assert bool(jnp.all(jnp.isfinite(large_chord)))
    assert not bool(jnp.isfinite(jnp.sum(jnp.abs(large_chord))))

    expected_q = agent.keyboard_q_values(state, OBS, unit_chord)
    eager = agent.dispatch_keyboard_policy(
        state,
        OBS,
        large_chord,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    compiled = jax.jit(agent.dispatch_keyboard_policy)(
        state,
        OBS,
        large_chord,
        jnp.ones((2,), dtype=jnp.bool_),
    )

    for result in (eager, compiled):
        assert bool(result.decision.proposal.available)
        chex.assert_trees_all_equal(result.decision.proposal.q_values, expected_q)
        assert int(result.decision.proposal.action) == 1
        assert int(result.decision.replacement.effective_action) == 1


def test_oak_keyboard_dispatch_rejects_nonfinite_or_zero_chord_without_rng_change() -> None:
    agent = OaKAgent(OaKConfig(stomp=_config()))
    outer_state = agent.init(jr.key(23))
    state = outer_state.replace(stomp_state=_primitive_owned_state(agent.stomp_agent))
    mask = jnp.ones((2,), dtype=jnp.bool_)

    for chord in (
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
    ):
        result = agent.dispatch_keyboard_policy(state, OBS, chord, mask)
        chex.assert_trees_all_equal(result.state, state)
        assert not bool(result.decision.proposal.available)
        assert int(result.decision.replacement.effective_action) == -1
        chex.assert_trees_all_equal(
            result.state.stomp_state.rng_key, state.stomp_state.rng_key
        )


def test_oak_keyboard_static_input_mismatches_fail_closed_without_exception() -> None:
    agent = OaKAgent(OaKConfig(stomp=_config()))
    outer_state = agent.init(jr.key(27))
    state = outer_state.replace(stomp_state=_primitive_owned_state(agent.stomp_agent))
    cases = (
        agent.dispatch_keyboard_policy(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((2,), dtype=jnp.float32),
        ),
        agent.dispatch_keyboard_policy(
            state,
            OBS,
            jnp.ones((3,), dtype=jnp.float32),
        ),
    )
    for result in cases:
        chex.assert_trees_all_equal(result.state, state)
        assert not bool(result.decision.proposal.available)
        assert int(result.decision.replacement.effective_action) == -1
    assert not bool(cases[0].decision.proposal.observation_static_contract_valid)
    assert not bool(cases[1].decision.proposal.keyboard_vector_static_contract_valid)


def test_oak_keyboard_dispatch_rejects_corrupt_outer_utility_and_counters() -> None:
    agent = OaKAgent(OaKConfig(stomp=_config()))
    outer_state = agent.init(jr.key(29))
    valid = outer_state.replace(stomp_state=_primitive_owned_state(agent.stomp_agent))
    corrupt_states = (
        valid.replace(utility_ema=valid.utility_ema.at[0].set(jnp.nan)),
        valid.replace(execution_counts=valid.execution_counts.at[0].set(-1)),
        valid.replace(step_count=jnp.int32(1)),
    )

    for corrupt in corrupt_states:
        result = agent.dispatch_keyboard_policy(
            corrupt,
            OBS,
            jnp.ones((2,), dtype=jnp.float32),
            safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(result.state, corrupt)
        assert not bool(result.decision.proposal.outer_state_valid)
        assert not bool(result.decision.proposal.state_valid)
        assert not bool(result.decision.proposal.available)
        assert int(result.decision.replacement.effective_action) == -1


def test_new_dispatch_surface_is_public_without_changing_state_tree_shape() -> None:
    import alberta_framework as package
    import alberta_framework.core as core_package
    import alberta_framework.core.oak as oak_module
    import alberta_framework.core.options as options_module

    assert "OaKKeyboardDispatchDecision" in oak_module.__all__
    assert "OaKKeyboardDispatchResult" in oak_module.__all__
    assert "OaKKeyboardPolicyProposal" in oak_module.__all__
    assert "replace_dispatched_primitive_action" in options_module.__all__
    assert "DispatchedPrimitiveActionDecision" in options_module.__all__
    assert "DispatchedPrimitiveActionReplacementResult" in options_module.__all__
    for name in (
        "DISPATCH_OWNER_BASE_PRIMITIVE",
        "DISPATCH_OWNER_INVALID",
        "DISPATCH_OWNER_OPTION",
        "DispatchedPrimitiveActionDecision",
        "DispatchedPrimitiveActionReplacementResult",
        "replace_dispatched_primitive_action",
    ):
        assert getattr(core_package, name) is getattr(options_module, name)
        assert getattr(package, name) is getattr(options_module, name)
    for name in (
        "OaKKeyboardDispatchDecision",
        "OaKKeyboardDispatchResult",
        "OaKKeyboardPolicyProposal",
    ):
        assert getattr(core_package, name) is getattr(oak_module, name)
        assert getattr(package, name) is getattr(oak_module, name)

    agent = STOMPAgent(_config())
    state = _primitive_owned_state(agent)
    result = replace_dispatched_primitive_action(state, OBS, jnp.int32(1))
    assert jax.tree_util.tree_structure(result.state) == jax.tree_util.tree_structure(state)
