# mypy: disable-error-code="attr-defined,call-arg,operator"
"""Causal representation learning and candidate-update audit integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.delight import GradientJoyConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeGradientJoyEvidence,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    replace_state_builder_learning_proposal_update,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = [pytest.mark.unit, pytest.mark.slow]

RAW_DIM = 1
FEATURE_DIM = 2
N_ACTIONS = 2
PARAMETER_COUNT = 12

# Gradients of three fixed synthetic linear probe objectives. They are chosen
# independently of the world-model candidate and merely exercise the mechanism
# contract; they are not performance or scientific evidence.
OBJECTIVE_PROBE_GRADIENT = jnp.ones((PARAMETER_COUNT,), dtype=jnp.float32)
RETENTION_PROBE_GRADIENT = jnp.linspace(
    0.5,
    1.5,
    PARAMETER_COUNT,
    dtype=jnp.float32,
)
SAFETY_PROBE_GRADIENT = jnp.tile(
    jnp.asarray([1.25, 0.75], dtype=jnp.float32),
    PARAMETER_COUNT // 2,
)


def _builder_config() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=1,
        include_raw_observation=True,
        step_size=0.5,
        gradient_clip=10.0,
    )


def _ensemble_config(*, max_input_magnitude: float = 100.0) -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.95,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.8,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=FEATURE_DIM + 2,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            fast_loss_decay=0.5,
            slow_loss_decay=0.9,
            max_input_magnitude=max_input_magnitude,
            max_predicted_variance=max_input_magnitude**2,
            max_observed_loss=max_input_magnitude**2,
        ),
        ensemble_size=2,
        bootstrap_probability=0.8,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
    )


def _agent(
    *,
    joy: bool,
    learn: bool = True,
    max_input_magnitude: float = 100.0,
) -> PrototypeAgent:
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=FEATURE_DIM,
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            state_builder=_builder_config(),
            world_model_ensemble=_ensemble_config(max_input_magnitude=max_input_magnitude),
            learn_state_builder_from_world_model=learn,
            gradient_joy=(
                GradientJoyConfig(
                    candidate_semantics="update",
                    max_update_norm=10.0,
                    alignment_temperature=1.0,
                    norm_temperature=1.0,
                    diagnostics_epsilon=1.0e-12,
                )
                if joy
                else None
            ),
        )
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: float,
) -> PrototypeTransition:
    next_array = jnp.asarray([next_observation], dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.4 + 0.1 * next_observation, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=next_array,
        next_decision_observation=next_array,
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _candidate_for_transition(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    transition: PrototypeTransition,
) -> tuple[Any, Any, Any]:
    builder = agent.state_builder
    ensemble = agent.world_model_ensemble
    assert builder is not None
    assert ensemble is not None
    destination, bootstrap = builder.update(
        state.state_builder_state,
        transition.next_observation,
        transition.action,
        transition.reward,
        transition.discount,
    )
    ensemble_result = ensemble.update(
        state.world_model_state,
        state.current_representation,
        transition.action,
        transition.reward,
        transition.discount,
        bootstrap,
    )
    proposal = builder.propose_learning_update(
        state.state_builder_state,
        ensemble_result.representation_gradient,
    )
    return destination, ensemble_result, proposal


def _complete_sidecar(
    state: PrototypeAgentState,
) -> PrototypeGradientJoyEvidence:
    available = jnp.asarray(True, dtype=jnp.bool_)
    return PrototypeGradientJoyEvidence(
        decision_id=state.current_decision_id,
        objective_probe_gradient=OBJECTIVE_PROBE_GRADIENT,
        retention_probe_gradient=RETENTION_PROBE_GRADIENT,
        safety_cost_gradient=SAFETY_PROBE_GRADIENT,
        objective_probe_available=available,
        retention_probe_available=available,
        safety_probe_available=available,
        probe_independence_attested=available,
        advantage=jnp.asarray(1.0, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_available=available,
        action_surprisal_available=available,
        safety_cost_available=available,
    )


def test_learning_and_joy_configuration_is_strict_and_round_trips() -> None:
    agent = _agent(joy=True)
    assert PrototypeAgent.from_config(agent.to_config()).to_config() == agent.to_config()

    with pytest.raises(ValueError, match="world_model_ensemble"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            state_builder=_builder_config(),
            learn_state_builder_from_world_model=True,
        )
    with pytest.raises(ValueError, match="OnlineGated"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            state_builder=IdentityStateBuilderConfig(observation_dim=FEATURE_DIM),
            world_model_ensemble=_ensemble_config(),
            learn_state_builder_from_world_model=True,
        )
    with pytest.raises(ValueError, match="candidate_semantics='update'"):
        PrototypeAgentConfig(
            oak=agent.config.oak,
            state_builder=_builder_config(),
            world_model_ensemble=_ensemble_config(),
            learn_state_builder_from_world_model=True,
            gradient_joy=GradientJoyConfig(candidate_semantics="gradient"),
        )
    payload = agent.to_config()
    payload["learn_state_builder_from_world_model"] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        PrototypeAgent.from_config(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("update_count", jnp.asarray(-1, dtype=jnp.int32)),
        ("last_gradient_norm", jnp.asarray(-1.0, dtype=jnp.float32)),
    ),
)
def test_corrupt_online_builder_state_cannot_be_checkpointed(
    tmp_path: Path,
    field: str,
    value: jax.Array,
) -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(31)), jnp.asarray([0.2], dtype=jnp.float32))
    corrupt_builder = state.state_builder_state.replace(**{field: value})
    corrupt = state.replace(state_builder_state=corrupt_builder)

    assert not bool(agent.state_builder.state_valid(corrupt_builder))  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(agent, corrupt, tmp_path / field)


def test_ungated_builder_update_matches_source_proposal_destination_commit() -> None:
    agent = _agent(joy=False)
    state = agent.start(agent.init(jr.key(1)), jnp.asarray([0.2], dtype=jnp.float32))
    transition = _transition(state, -0.7)
    destination, ensemble_result, proposal = _candidate_for_transition(
        agent,
        state,
        transition,
    )
    builder = agent.state_builder
    assert builder is not None
    filtered = replace_state_builder_learning_proposal_update(
        proposal,
        proposal.candidate_parameter_update,
        ensemble_result.representation_gradient_valid,
    )
    expected_state, expected_diagnostics = builder.commit_learning_update(
        destination,
        filtered,
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.state_builder_learning_diagnostics.applied)
    _assert_tree_equal(result.state.state_builder_state, expected_state)
    chex.assert_trees_all_close(
        result.state_builder_learning_diagnostics,
        expected_diagnostics,
    )
    chex.assert_trees_all_equal(
        result.state.state_builder_state.hidden,
        destination.hidden,
    )
    chex.assert_trees_all_equal(
        result.state.state_builder_state.parameter_sensitivity,
        destination.parameter_sensitivity,
    )
    assert result.gradient_joy_application is None


def test_rejected_ensemble_vetoes_parameters_but_preserves_recurrent_advance() -> None:
    agent = _agent(joy=False, max_input_magnitude=0.5)
    state = agent.start(agent.init(jr.key(2)), jnp.asarray([0.1], dtype=jnp.float32))
    transition = _transition(state, 0.2)
    destination, _, _ = _candidate_for_transition(agent, state, transition)

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.world_model_ensemble_diagnostics.rejected)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    chex.assert_trees_all_equal(
        result.state.state_builder_state.parameters,
        state.state_builder_state.parameters,
    )
    chex.assert_trees_all_equal(
        result.state.state_builder_state.hidden,
        destination.hidden,
    )
    assert int(result.state.state_builder_state.update_count) == 0


def test_terminal_commit_preserves_reset_destination_recurrence_and_cache() -> None:
    agent = _agent(joy=False)
    state = agent.start(agent.init(jr.key(22)), jnp.asarray([0.15], dtype=jnp.float32))
    final_observation = jnp.asarray([0.8], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.6], dtype=jnp.float32)
    transition = PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.7, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True),
        truncated=jnp.asarray(False),
        next_observation=final_observation,
        next_decision_observation=reset_observation,
    )
    builder = agent.state_builder
    ensemble = agent.world_model_ensemble
    assert builder is not None
    assert ensemble is not None
    bootstrap_state, bootstrap = builder.update(
        state.state_builder_state,
        final_observation,
        transition.action,
        transition.reward,
        transition.discount,
    )
    reset_state = builder.reset_episode(bootstrap_state)
    destination, decision_representation = builder.start(
        reset_state,
        reset_observation,
    )
    ensemble_result = ensemble.update(
        state.world_model_state,
        state.current_representation,
        transition.action,
        transition.reward,
        transition.discount,
        bootstrap,
    )
    proposal = builder.propose_learning_update(
        state.state_builder_state,
        ensemble_result.representation_gradient,
    )
    filtered = replace_state_builder_learning_proposal_update(
        proposal,
        proposal.candidate_parameter_update,
        ensemble_result.representation_gradient_valid,
    )
    expected, _ = builder.commit_learning_update(destination, filtered)

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.state_builder_learning_diagnostics.applied)
    _assert_tree_equal(result.state.state_builder_state, expected)
    chex.assert_trees_all_equal(
        result.state.state_builder_state.hidden,
        destination.hidden,
    )
    chex.assert_trees_all_equal(
        result.state.state_builder_state.parameter_sensitivity,
        destination.parameter_sensitivity,
    )
    chex.assert_trees_all_equal(
        result.state.current_representation,
        decision_representation,
    )
    chex.assert_trees_all_equal(
        builder.encode(result.state.state_builder_state, reset_observation),
        decision_representation,
    )


def test_missing_joy_evidence_answers_no_without_blocking_real_learning() -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(3)), jnp.asarray([0.1], dtype=jnp.float32))
    result = agent.update_transition(state, _transition(state, 0.6))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.world_model_ensemble_diagnostics.applied)
    assert result.gradient_joy_application is not None
    assert not bool(result.gradient_joy_evidence_supplied)
    assert not bool(result.sparks_joy)
    assert not bool(result.joyful_gradient_applied)
    assert not bool(result.gradient_joy_application.assessment.sparks_joy)
    assert not bool(result.gradient_joy_application.applied)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.step_count) == int(state.step_count) + 1
    assert int(result.state.world_model_state.event_count) == (
        int(state.world_model_state.event_count) + 1
    )


def test_builder_learning_counter_saturates_without_wrapping_control() -> None:
    agent = _agent(joy=False)
    state = agent.start(agent.init(jr.key(32)), jnp.asarray([0.2], dtype=jnp.float32))
    maximum = 2**31 - 1
    near_capacity_builder = state.state_builder_state.replace(
        update_count=jnp.asarray(maximum - 1, dtype=jnp.int32)
    )
    state = state.replace(state_builder_state=near_capacity_builder)

    final_update = agent.update_transition(state, _transition(state, -0.7))

    assert bool(final_update.transition_diagnostics.valid)
    assert bool(final_update.state_builder_learning_diagnostics.applied)
    assert int(final_update.state.state_builder_state.update_count) == maximum

    exhausted_state = final_update.state
    exhausted = agent.update_transition(
        exhausted_state,
        _transition(exhausted_state, 0.6),
    )
    assert bool(exhausted.transition_diagnostics.valid)
    assert not bool(exhausted.state_builder_learning_diagnostics.applied)
    assert bool(exhausted.state_builder_learning_diagnostics.rejected)
    assert int(exhausted.state.state_builder_state.update_count) == maximum
    assert int(exhausted.state.step_count) == int(exhausted_state.step_count) + 1


def _warm_signals(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    for observation in (-0.4, 0.7, -0.8):
        result = agent.update_transition(state, _transition(state, observation))
        assert bool(result.transition_diagnostics.valid)
        assert not bool(result.state_builder_learning_diagnostics.applied)
        state = result.state
    return state


def test_complete_matching_evidence_sparks_joy_and_applies_eager_jit_checkpoint(
    tmp_path: Path,
) -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(4)), jnp.asarray([0.25], dtype=jnp.float32))
    state = _warm_signals(agent, state)
    transition = _transition(state, 0.9)
    _, ensemble_result, proposal = _candidate_for_transition(agent, state, transition)
    assert bool(ensemble_result.signals.availability.epistemic)
    assert bool(ensemble_result.signals.availability.aleatoric)
    assert bool(ensemble_result.signals.availability.learning_progress)
    assert bool(ensemble_result.signals.availability.change_probability)
    assert bool(ensemble_result.representation_gradient_valid)
    assert bool(jnp.any(proposal.candidate_parameter_update != 0.0))
    sidecar = _complete_sidecar(state)

    eager = agent.update_transition(state, transition, sidecar)
    compiled_update = jax.jit(agent.update_transition)
    compiled = compiled_update(state, transition, sidecar)

    assert eager.gradient_joy_application is not None
    assert bool(eager.gradient_joy_evidence_supplied)
    assert bool(eager.gradient_joy_decision_id_matches)
    assert bool(eager.sparks_joy)
    assert bool(eager.joyful_gradient_applied)
    assert bool(eager.candidate_update_audit_passed)
    assert bool(eager.audited_candidate_update_applied)
    assert bool(eager.gradient_joy_application.assessment.sparks_joy)
    assert bool(eager.gradient_joy_application.effective_assessment.sparks_joy)
    assert bool(eager.gradient_joy_application.applied)
    assert bool(eager.state_builder_learning_diagnostics.applied)
    chex.assert_trees_all_equal(
        eager.state.state_builder_state.parameters,
        eager.gradient_joy_application.parameters,
    )
    assert not bool(
        jnp.array_equal(
            eager.state.state_builder_state.parameters,
            state.state_builder_state.parameters,
        )
    )
    _assert_tree_equal(eager, compiled)

    invalid_transition = transition.replace(
        action=(transition.action + jnp.asarray(1, dtype=jnp.int32)) % N_ACTIONS
    )
    invalid_eager = agent.update_transition(state, invalid_transition, sidecar)
    invalid_compiled = compiled_update(state, invalid_transition, sidecar)
    assert not bool(invalid_eager.transition_diagnostics.valid)
    assert not bool(invalid_eager.sparks_joy)
    assert not bool(invalid_eager.joyful_gradient_applied)
    _assert_tree_equal(invalid_eager.state, state)
    _assert_tree_equal(invalid_eager, invalid_compiled)

    nonfinite_sidecar = sidecar.replace(
        objective_probe_gradient=sidecar.objective_probe_gradient.at[0].set(jnp.nan)
    )
    nonfinite_eager = agent.update_transition(state, transition, nonfinite_sidecar)
    nonfinite_compiled = compiled_update(state, transition, nonfinite_sidecar)
    assert bool(nonfinite_eager.transition_diagnostics.valid)
    assert bool(nonfinite_eager.gradient_joy_evidence_supplied)
    assert bool(nonfinite_eager.gradient_joy_decision_id_matches)
    assert not bool(nonfinite_eager.sparks_joy)
    assert not bool(nonfinite_eager.joyful_gradient_applied)
    chex.assert_trees_all_equal(
        nonfinite_eager.state.state_builder_state.parameters,
        state.state_builder_state.parameters,
    )
    _assert_tree_equal(nonfinite_eager, nonfinite_compiled)

    with pytest.raises(ValueError, match="dtype float32"):
        agent.update_transition(
            state,
            transition,
            sidecar.replace(
                objective_probe_gradient=jnp.ones(
                    (PARAMETER_COUNT,),
                    dtype=jnp.int32,
                )
            ),
        )
    with pytest.raises(ValueError, match="shape"):
        agent.update_transition(
            state,
            transition,
            sidecar.replace(
                objective_probe_gradient=jnp.ones(
                    (PARAMETER_COUNT - 1,),
                    dtype=jnp.float32,
                )
            ),
        )

    checkpoint = tmp_path / "prototype_joy"
    save_prototype_checkpoint(agent, state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    resumed = restored_agent.update_transition(restored_state, transition, sidecar)
    _assert_tree_equal(eager, resumed)


@pytest.mark.parametrize(
    ("advantage", "action_surprisal"),
    (
        # The actor chose its action with probability 1: -log(1) == -0.0.
        (0.0, -0.0),
        # A zero advantage that carries the negative sign.
        (-0.0, 0.5),
    ),
)
def test_zero_advantage_sparks_joy_whichever_zero_sign_it_carries(
    advantage: float,
    action_surprisal: float,
) -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(4)), jnp.asarray([0.25], dtype=jnp.float32))
    state = _warm_signals(agent, state)
    transition = _transition(state, 0.9)
    positive_zero = _complete_sidecar(state).replace(
        advantage=jnp.asarray(0.0, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
    )
    signed_zero = _complete_sidecar(state).replace(
        advantage=jnp.asarray(advantage, dtype=jnp.float32),
        action_surprisal=jnp.asarray(action_surprisal, dtype=jnp.float32),
    )

    reference = agent.update_transition(state, transition, positive_zero)
    assert bool(reference.sparks_joy)
    assert bool(reference.joyful_gradient_applied)

    for result in (
        agent.update_transition(state, transition, signed_zero),
        jax.jit(agent.update_transition)(state, transition, signed_zero),
    ):
        assert result.gradient_joy_application is not None
        assessment = result.gradient_joy_application.assessment
        assert bool(assessment.channel_availability.delight)
        assert bool(assessment.diagnostics.learning_value_complete)
        assert bool(result.sparks_joy)
        assert bool(result.joyful_gradient_applied)
        assert bool(result.state_builder_learning_diagnostics.applied)
        chex.assert_trees_all_close(
            result.state.state_builder_state.parameters,
            reference.state.state_builder_state.parameters,
        )


def test_candidate_sparks_joy_when_builder_capacity_vetoes_the_commit() -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(41)), jnp.asarray([0.25], dtype=jnp.float32))
    state = _warm_signals(agent, state)
    maximum = 2**31 - 1
    saturated_builder = state.state_builder_state.replace(
        update_count=jnp.asarray(maximum, dtype=jnp.int32)
    )
    state = state.replace(state_builder_state=saturated_builder)
    transition = _transition(state, 0.9)
    sidecar = _complete_sidecar(state)

    eager = agent.update_transition(state, transition, sidecar)
    compiled = jax.jit(agent.update_transition)(state, transition, sidecar)

    assert bool(eager.transition_diagnostics.valid)
    assert eager.gradient_joy_application is not None
    assert bool(eager.gradient_joy_application.assessment.sparks_joy)
    assert bool(eager.gradient_joy_application.effective_assessment.sparks_joy)
    assert bool(eager.gradient_joy_application.applied)
    assert bool(eager.sparks_joy)
    assert not bool(eager.state_builder_learning_diagnostics.applied)
    assert bool(eager.state_builder_learning_diagnostics.rejected)
    assert not bool(eager.joyful_gradient_applied)
    chex.assert_trees_all_equal(
        eager.state.state_builder_state.parameters,
        state.state_builder_state.parameters,
    )
    assert int(eager.state.state_builder_state.update_count) == maximum
    _assert_tree_equal(eager, compiled)


def test_stale_joy_sidecar_vetoes_only_the_builder_update() -> None:
    agent = _agent(joy=True)
    state = agent.start(agent.init(jr.key(5)), jnp.asarray([0.25], dtype=jnp.float32))
    state = _warm_signals(agent, state)
    transition = _transition(state, -0.9)
    _, _, proposal = _candidate_for_transition(agent, state, transition)
    sidecar = _complete_sidecar(state)
    sidecar = sidecar.replace(
        decision_id=sidecar.decision_id.at[3].add(jnp.asarray(1, dtype=jnp.uint32))
    )

    result = agent.update_transition(state, transition, sidecar)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.gradient_joy_evidence_supplied)
    assert not bool(result.gradient_joy_decision_id_matches)
    assert not bool(result.sparks_joy)
    assert not bool(result.joyful_gradient_applied)
    assert result.gradient_joy_application is not None
    assert not bool(result.gradient_joy_application.assessment.sparks_joy)
    assert not bool(result.gradient_joy_application.applied)
    assert not bool(result.state_builder_learning_diagnostics.applied)
    chex.assert_trees_all_equal(
        result.state.state_builder_state.parameters,
        state.state_builder_state.parameters,
    )
    assert int(result.state.step_count) == int(state.step_count) + 1


def test_joy_scan_matches_loop_and_reports_the_storage_implication() -> None:
    agent = _agent(joy=True)
    initial = agent.start(
        agent.init(jr.key(4)),
        jnp.asarray([0.25], dtype=jnp.float32),
    )
    initial = _warm_signals(agent, initial)

    first_transition = _transition(initial, 0.9)
    first_sidecar = _complete_sidecar(initial)
    first = agent.update_transition(initial, first_transition, first_sidecar)
    second_transition = _transition(first.state, -0.5)
    second_sidecar = _complete_sidecar(first.state)
    second = agent.update_transition(first.state, second_transition, second_sidecar)

    transitions = jax.tree.map(
        lambda first_leaf, second_leaf: jnp.stack((first_leaf, second_leaf)),
        first_transition,
        second_transition,
    )
    sidecars = jax.tree.map(
        lambda first_leaf, second_leaf: jnp.stack((first_leaf, second_leaf)),
        first_sidecar,
        second_sidecar,
    )
    scanned = agent.scan_transitions(initial, transitions, sidecars)
    compiled = jax.jit(agent.scan_transitions)(initial, transitions, sidecars)

    _assert_tree_equal(scanned, compiled)
    _assert_tree_equal(scanned.state, second.state)
    chex.assert_trees_all_equal(
        scanned.state_builder_learning_applied,
        jnp.stack(
            (
                first.state_builder_learning_diagnostics.applied,
                second.state_builder_learning_diagnostics.applied,
            )
        ),
    )
    chex.assert_trees_all_equal(
        scanned.gradient_sparks_joy,
        jnp.stack((first.sparks_joy, second.sparks_joy)),
    )
    chex.assert_trees_all_equal(
        scanned.joyful_gradient_applied,
        jnp.stack(
            (
                first.joyful_gradient_applied,
                second.joyful_gradient_applied,
            )
        ),
    )
    assert bool(jnp.any(scanned.state_builder_learning_applied))
    implication = (~scanned.state_builder_learning_applied) | (
        scanned.gradient_sparks_joy & scanned.joyful_gradient_applied
    )
    assert bool(jnp.all(implication))
