# mypy: disable-error-code="attr-defined,call-arg"
"""L0 contracts for support-aware option-model search control."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.core as alberta_core
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.option_search_control import (
    OPTION_SEARCH_CONTROL_CONFIG_SCHEMA,
    OptionSearchControl,
    OptionSearchControlConfig,
)
from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)

pytestmark = pytest.mark.unit

ANCHOR = jnp.array([1.0, 0.0], dtype=jnp.float32)


def _stomp_config(
    *,
    n_options: int = 2,
    base_step_size: float = 0.25,
    hidden_sizes: tuple[int, ...] = (),
    legacy_backups: int = 0,
) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=tuple(
            SubtaskSpec(
                feature_index=index % 2,
                threshold=1.0e6,
                max_option_steps=8,
            )
            for index in range(n_options)
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=base_step_size,
        base_avg_reward_step_size=0.0,
        base_hidden_sizes=hidden_sizes,
        option_planning_backups_per_step=legacy_backups,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _supported_state(
    agent: STOMPAgent,
    *,
    targets: tuple[float, ...] = (1.0, 4.0),
) -> STOMPState:
    state = agent.start(agent.init(jr.key(7)), ANCHOR)
    n_options = agent.config.n_options
    assert len(targets) == n_options
    learner = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=tuple(
                jnp.zeros_like(weight) for weight in state.base_learner_state.head_params.weights
            ),
            biases=tuple(
                jnp.zeros_like(bias) for bias in state.base_learner_state.head_params.biases
            ),
        )
    )
    models = state.option_models.replace(
        cumreward_ema=jnp.arange(n_options, dtype=jnp.float32) + 100.0,
        env_return_ema=jnp.asarray(targets, dtype=jnp.float32),
        duration_ema=jnp.ones((n_options,), dtype=jnp.float32),
        baseline_mass_ema=jnp.ones((n_options,), dtype=jnp.float32),
        discount_ema=jnp.zeros((n_options,), dtype=jnp.float32),
        next_state_weights=jnp.zeros((n_options, 2, 2), dtype=jnp.float32),
        n_completions=jnp.ones((n_options,), dtype=jnp.int32),
    )
    return cast(
        STOMPState,
        state.replace(
            base_learner_state=learner,
            option_models=models,
            base_average_reward=jnp.array(0.0, dtype=jnp.float32),
        ),
    )


def test_config_is_strict_and_resource_budget_is_explicit() -> None:
    config = OptionSearchControlConfig(backup_budget=3, min_model_completions=2)
    payload = config.to_config()

    assert payload["schema"] == OPTION_SEARCH_CONTROL_CONFIG_SCHEMA
    assert payload["scientific_promotion_allowed"] is False
    assert OptionSearchControlConfig.from_config(payload) == config
    with pytest.raises(ValueError, match="fields"):
        OptionSearchControlConfig.from_config({**payload, "extra": 1})
    for invalid in (True, 0, -1, 1.5, 4_097):
        with pytest.raises(ValueError, match="backup_budget"):
            OptionSearchControlConfig(backup_budget=invalid)  # type: ignore[arg-type]

    agent = STOMPAgent(_stomp_config())
    budget = OptionSearchControl(agent, config).resource_budget
    assert budget.persistent_state_bytes == 0
    assert budget.rng_draws_per_call == 0
    assert budget.max_candidate_evaluations_per_call == 6
    assert budget.max_base_learner_updates_per_call == 3
    assert budget.max_model_matrix_vector_products_per_call == 6
    assert budget.max_base_value_forward_calls_per_call == 12
    assert budget.max_base_value_backward_calls_per_call == 3
    assert budget.stomp_self_audits_per_call == 1
    assert budget.max_diagnostic_payload_bytes_per_call == 191

    wide_agent = STOMPAgent(_stomp_config(n_options=65))
    with pytest.raises(ValueError, match="diagnostic slot ceiling"):
        OptionSearchControl(
            wide_agent,
            OptionSearchControlConfig(backup_budget=4_096),
        )

    empty_agent = STOMPAgent(_stomp_config(n_options=0))
    with pytest.raises(ValueError, match="at least one option"):
        OptionSearchControl(empty_agent)


def test_option_search_control_is_exported_from_public_namespaces() -> None:
    assert alberta_framework.OptionSearchControl is OptionSearchControl
    assert alberta_core.OptionSearchControl is OptionSearchControl
    assert "OptionSearchControl" in alberta_framework.__all__
    assert "OptionSearchControl" in alberta_core.__all__


def test_differential_semi_mdp_target_uses_all_declared_terms_exactly() -> None:
    agent = STOMPAgent(_stomp_config(n_options=1))
    state = _supported_state(agent, targets=(2.5,))
    head_weights = (
        jnp.array([[0.0, 3.0]], dtype=jnp.float32),
        jnp.array([[0.0, 1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0]], dtype=jnp.float32),
    )
    learner = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=head_weights,
        )
    )
    models = state.option_models.replace(
        env_return_ema=jnp.array([2.5], dtype=jnp.float32),
        baseline_mass_ema=jnp.array([1.25], dtype=jnp.float32),
        discount_ema=jnp.array([0.5], dtype=jnp.float32),
        next_state_weights=jnp.array(
            [[[-1.0, 0.0], [1.0, 0.0]]],
            dtype=jnp.float32,
        ),
    )
    state = state.replace(
        base_learner_state=learner,
        option_models=models,
        base_average_reward=jnp.array(0.4, dtype=jnp.float32),
    )

    diagnostics = OptionSearchControl(agent).apply(state, ANCHOR).diagnostics

    # predicted_next=[0, 1], max_a Q=3; target=2.5-0.4*1.25+0.5*3=3.5.
    assert float(diagnostics.candidate_targets[0, 0]) == pytest.approx(3.5)
    # The current option value is 0.5, so |Bellman residual| is 3.0.
    assert float(diagnostics.candidate_bellman_residuals[0, 0]) == pytest.approx(3.0)
    assert float(diagnostics.candidate_priorities[0, 0]) == pytest.approx(3.0)


def test_zero_discount_does_not_multiply_overflowed_next_values() -> None:
    """discount_ema=0 must skip max Q(s') so an overflowed next-state model still ranks."""
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent, targets=(1.0, 4.0))
    huge = jnp.finfo(jnp.float32).max
    ones = tuple(jnp.ones_like(weight) for weight in state.base_learner_state.head_params.weights)
    learner = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(weights=ones)
    )
    state = state.replace(
        base_learner_state=learner,
        option_models=state.option_models.replace(
            next_state_weights=jnp.full((2, 2, 2), huge, dtype=jnp.float32),
        ),
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    diagnostics = OptionSearchControl(agent).apply(state, ANCHOR).diagnostics
    np.testing.assert_array_equal(
        np.asarray(diagnostics.candidate_valid[0]),
        np.array([True, True]),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.candidate_targets[0]),
        np.array([1.0, 4.0], dtype=np.float32),
    )
    assert int(diagnostics.selected_option_indices[0]) == 1


def test_equal_residual_priority_uses_stable_lowest_option_index() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent, targets=(1.0, -1.0))

    diagnostics = OptionSearchControl(agent).apply(state, ANCHOR).diagnostics

    np.testing.assert_array_equal(
        np.asarray(diagnostics.candidate_priorities[0]),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert int(diagnostics.selected_option_indices[0]) == 0


def test_partial_completion_support_excludes_the_unsupported_candidate() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent, targets=(1.0, 100.0))
    state = state.replace(
        option_models=state.option_models.replace(
            n_completions=jnp.array([2, 1], dtype=jnp.int32),
        )
    )
    diagnostics = (
        OptionSearchControl(
            agent,
            OptionSearchControlConfig(min_model_completions=2),
        )
        .apply(state, ANCHOR)
        .diagnostics
    )

    np.testing.assert_array_equal(
        np.asarray(diagnostics.completion_supported[0]),
        np.array([True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.candidate_valid[0]),
        np.array([True, False]),
    )
    assert int(diagnostics.selected_option_indices[0]) == 0


def test_highest_supported_bellman_residual_is_applied_to_its_option_head() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    controller = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=1),
    )
    before = state.base_learner_state

    result = controller.apply(state, ANCHOR)

    diagnostics = result.diagnostics
    assert int(diagnostics.selected_option_indices[0]) == 1
    assert int(diagnostics.selected_extended_action_indices[0]) == 3
    np.testing.assert_allclose(
        np.asarray(diagnostics.candidate_targets[0]),
        np.array([1.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.candidate_bellman_residuals[0]),
        np.array([1.0, 4.0], dtype=np.float32),
    )
    assert bool(diagnostics.applied[0])
    assert int(diagnostics.applied_count) == 1
    for head in (0, 1, 2):
        chex.assert_trees_all_equal(
            result.state.base_learner_state.head_params.weights[head],
            before.head_params.weights[head],
        )
    assert not bool(
        jnp.array_equal(
            result.state.base_learner_state.head_params.weights[3],
            before.head_params.weights[3],
        )
    )


def test_residuals_are_recomputed_after_each_backup() -> None:
    agent = STOMPAgent(_stomp_config(base_step_size=0.25))
    state = _supported_state(agent, targets=(1.0, 0.9))
    controller = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=2),
    )

    result = controller.apply(state, ANCHOR)

    np.testing.assert_array_equal(
        np.asarray(result.diagnostics.selected_option_indices),
        np.array([0, 1], dtype=np.int32),
    )
    assert bool(jnp.all(result.diagnostics.applied))
    assert float(result.diagnostics.candidate_priorities[1, 0]) < float(
        result.diagnostics.candidate_priorities[0, 0]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("n_completions", jnp.array([0, 0], dtype=jnp.int32)),
        ("discount_ema", jnp.array([1.1, 1.1], dtype=jnp.float32)),
        ("baseline_mass_ema", jnp.array([0.0, 0.0], dtype=jnp.float32)),
        ("env_return_ema", jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32)),
    ],
)
def test_unsupported_or_invalid_models_are_exact_planner_noops(
    field: str,
    replacement: jax.Array,
) -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    state = state.replace(option_models=state.option_models.replace(**{field: replacement}))
    result = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=2),
    ).apply(state, ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    assert not bool(jnp.any(result.diagnostics.applied))
    assert int(result.diagnostics.applied_count) == 0
    np.testing.assert_array_equal(
        np.asarray(result.diagnostics.selected_option_indices),
        np.array([-1, -1], dtype=np.int32),
    )


def test_malformed_base_state_contracts_fail_closed_without_indexing() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    learner = state.base_learner_state
    shortened_heads = learner.replace(
        head_params=learner.head_params.replace(
            weights=learner.head_params.weights[:-1],
            biases=learner.head_params.biases[:-1],
        )
    )
    malformed_states = (
        state.replace(base_average_reward=jnp.zeros((1,), dtype=jnp.float32)),
        state.replace(base_average_reward=jnp.asarray(0, dtype=jnp.int32)),
        state.replace(base_average_reward=0.0),
        state.replace(step_count=0),
        state.replace(base_learner_state=shortened_heads),
    )
    controller = OptionSearchControl(agent)

    for malformed in malformed_states:
        result = controller.apply(malformed, ANCHOR)
        chex.assert_trees_all_equal(result.state, malformed)
        assert not bool(result.diagnostics.base_state_static_contract_valid)
        assert not bool(result.diagnostics.planner_inputs_valid)
        assert int(result.diagnostics.applied_count) == 0


def test_malformed_nested_model_and_outer_leaves_fail_closed() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    malformed_states = (
        state.replace(option_models=state.option_models.replace(cumreward_ema=[100.0, 101.0])),
        state.replace(base_last_obs=[1.0, 0.0]),
        state.replace(base_last_action=0),
        state.replace(
            option_policies=state.option_policies.replace(
                q_weights=state.option_policies.q_weights.tolist()
            )
        ),
    )
    controller = OptionSearchControl(agent)

    for malformed in malformed_states:
        result = controller.apply(malformed, ANCHOR)
        assert result.state is malformed
        assert not bool(result.diagnostics.planner_inputs_valid)
        assert int(result.diagnostics.applied_count) == 0


def test_nonfinite_base_state_is_an_atomic_noop() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    learner = state.base_learner_state
    weights = list(learner.head_params.weights)
    weights[0] = weights[0].at[0, 0].set(jnp.nan)
    state = state.replace(
        base_learner_state=learner.replace(
            head_params=learner.head_params.replace(weights=tuple(weights))
        )
    )

    result = OptionSearchControl(agent).apply(state, ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    assert bool(result.diagnostics.base_state_static_contract_valid)
    assert not bool(result.diagnostics.base_state_values_finite)
    assert not bool(result.diagnostics.planner_inputs_valid)
    assert int(result.diagnostics.applied_count) == 0


@pytest.mark.parametrize("counter_owner", ["base_learner", "outer"])
def test_negative_counters_are_atomic_noops(counter_owner: str) -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    if counter_owner == "base_learner":
        state = state.replace(
            base_learner_state=state.base_learner_state.replace(
                step_count=jnp.asarray(-1, dtype=jnp.int32)
            )
        )
    else:
        state = state.replace(step_count=jnp.asarray(-1, dtype=jnp.int32))

    result = OptionSearchControl(agent).apply(state, ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.diagnostics.state_counters_valid)
    assert not bool(result.diagnostics.planner_inputs_valid)
    assert int(result.diagnostics.applied_count) == 0


def test_outer_rng_or_action_ownership_corruption_is_an_atomic_noop() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    corrupted_states = (
        state.replace(rng_key=jr.key_data(state.rng_key)),
        state.replace(executing_option=jnp.asarray(99, dtype=jnp.int32)),
    )
    controller = OptionSearchControl(agent)

    for corrupted in corrupted_states:
        result = controller.apply(corrupted, ANCHOR)
        chex.assert_trees_all_equal(result.state, corrupted)
        assert not bool(result.diagnostics.stomp_state_valid)
        assert not bool(result.diagnostics.planner_inputs_valid)
        assert int(result.diagnostics.applied_count) == 0


def test_pseudo_return_and_raw_duration_do_not_enter_task_bellman_priority() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    controller = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=1),
    )
    baseline = controller.apply(state, ANCHOR)
    mutated = state.replace(
        option_models=state.option_models.replace(
            cumreward_ema=jnp.array([-1.0e6, 1.0e6], dtype=jnp.float32),
            duration_ema=jnp.array([999.0, 0.01], dtype=jnp.float32),
        )
    )
    counterfactual = controller.apply(mutated, ANCHOR)

    chex.assert_trees_all_equal(
        baseline.diagnostics.candidate_targets,
        counterfactual.diagnostics.candidate_targets,
    )
    chex.assert_trees_all_equal(
        baseline.diagnostics.selected_option_indices,
        counterfactual.diagnostics.selected_option_indices,
    )
    chex.assert_trees_all_equal(
        baseline.state.base_learner_state,
        counterfactual.state.base_learner_state,
    )


def test_planning_preserves_real_traces_normalizer_caches_rng_and_option_state() -> None:
    agent = STOMPAgent(_stomp_config(hidden_sizes=(3,)))
    state = _supported_state(agent)
    learner = state.base_learner_state.replace(
        trunk_traces=tuple(
            jnp.full_like(trace, 0.25) for trace in state.base_learner_state.trunk_traces
        ),
        head_traces=tuple(
            (jnp.full_like(weight_trace, 0.5), jnp.full_like(bias_trace, -0.5))
            for weight_trace, bias_trace in state.base_learner_state.head_traces
        ),
    )
    state = state.replace(base_learner_state=learner)
    result = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=2),
    ).apply(state, ANCHOR)

    chex.assert_trees_all_equal(
        result.state.base_learner_state.trunk_traces,
        state.base_learner_state.trunk_traces,
    )
    chex.assert_trees_all_equal(
        result.state.base_learner_state.head_traces,
        state.base_learner_state.head_traces,
    )
    chex.assert_trees_all_equal(
        result.state.base_learner_state.normalizer_state,
        state.base_learner_state.normalizer_state,
    )
    chex.assert_trees_all_equal(result.state.rng_key, state.rng_key)
    chex.assert_trees_all_equal(result.state.option_models, state.option_models)
    chex.assert_trees_all_equal(result.state.option_policies, state.option_policies)
    chex.assert_trees_all_equal(result.state.base_average_reward, state.base_average_reward)
    chex.assert_trees_all_equal(result.state.base_last_obs, state.base_last_obs)
    chex.assert_trees_all_equal(result.state.base_last_action, state.base_last_action)
    chex.assert_trees_all_equal(result.state.last_primitive_action, state.last_primitive_action)
    chex.assert_trees_all_equal(result.state.executing_option, state.executing_option)
    assert bool(jnp.all(result.diagnostics.trace_isolation_preserved))


def test_anchor_contract_and_eager_jit_scan_parity() -> None:
    agent = STOMPAgent(_stomp_config())
    state = _supported_state(agent)
    controller = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=1),
    )
    stale_shape = controller.apply(
        state,
        jnp.ones((3,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(stale_shape.state, state)
    assert not bool(stale_shape.diagnostics.decision_observation_static_contract_valid)

    stale_value = controller.apply(
        state,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(stale_value.state, state)
    assert not bool(stale_value.diagnostics.decision_observation_matches_state)
    assert not bool(stale_value.diagnostics.planner_inputs_valid)

    eager = controller.apply(state, ANCHOR)
    compiled = jax.jit(controller.apply)(state, ANCHOR)
    chex.assert_trees_all_equal(eager, compiled)

    anchors = jnp.stack((ANCHOR, ANCHOR))

    def body(
        carry: STOMPState,
        anchor: jax.Array,
    ) -> tuple[STOMPState, jax.Array]:
        result = controller.apply(carry, anchor)
        return result.state, result.diagnostics.selected_option_indices

    scan_state, scan_indices = jax.lax.scan(body, state, anchors)
    loop_state = state
    loop_indices = []
    for anchor in anchors:
        result = controller.apply(loop_state, anchor)
        loop_state = result.state
        loop_indices.append(result.diagnostics.selected_option_indices)
    chex.assert_trees_all_equal(scan_state, loop_state)
    chex.assert_trees_all_equal(scan_indices, jnp.stack(loop_indices))


def test_prototype_config_rejects_double_option_planning_and_roundtrips() -> None:
    option_search = OptionSearchControlConfig(backup_budget=2)
    config = PrototypeAgentConfig(
        oak=OaKConfig(stomp=_stomp_config()),
        option_search_control=option_search,
    )
    restored = PrototypeAgentConfig.from_config(config.to_config())
    assert restored.option_search_control == option_search

    with pytest.raises(ValueError, match="option_planning_backups_per_step"):
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=_stomp_config(legacy_backups=1)),
            option_search_control=option_search,
        )


def test_prototype_checkpoint_preserves_option_search_config(tmp_path: Path) -> None:
    config = PrototypeAgentConfig(
        oak=OaKConfig(stomp=_stomp_config()),
        option_search_control=OptionSearchControlConfig(
            backup_budget=2,
            min_model_completions=3,
        ),
    )
    agent = PrototypeAgent(config)
    state = agent.start(agent.init(jr.key(19)), ANCHOR)
    checkpoint_path = tmp_path / "option-search-control"

    save_prototype_checkpoint(agent, state, checkpoint_path)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint_path)

    assert restored_agent.config.option_search_control == config.option_search_control
    assert (
        restored_agent.option_search_control_resource_budget
        == agent.option_search_control_resource_budget
    )
    chex.assert_trees_all_equal(restored_state, state)


@pytest.mark.slow
def test_prototype_applies_search_at_next_decision_observation() -> None:
    option_search = OptionSearchControlConfig(backup_budget=1)
    config = PrototypeAgentConfig(
        oak=OaKConfig(stomp=_stomp_config()),
        option_search_control=option_search,
    )
    search_agent = PrototypeAgent(config)
    baseline_agent = PrototypeAgent(PrototypeAgentConfig(oak=config.oak))
    state = search_agent.start(search_agent.init(jr.key(21)), ANCHOR)
    supported_stomp = _supported_state(search_agent.oak_agent.stomp_agent)
    state = state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=state.oak_state.stomp_state.replace(
                option_models=supported_stomp.option_models,
                base_average_reward=jnp.array(0.0, dtype=jnp.float32),
            )
        )
    )
    next_decision = jnp.array([0.0, 1.0], dtype=jnp.float32)
    transition = PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.array(0.0, dtype=jnp.float32),
        discount=jnp.array(1.0, dtype=jnp.float32),
        terminated=jnp.array(False),
        truncated=jnp.array(False),
        next_observation=next_decision,
        next_decision_observation=next_decision,
    )

    searched = search_agent.update_transition(state, transition)
    baseline = baseline_agent.update_transition(state, transition)

    diagnostics = searched.option_search_control_diagnostics
    assert diagnostics is not None
    chex.assert_trees_all_equal(
        diagnostics.decision_observation,
        next_decision,
    )
    assert int(diagnostics.applied_count) == 1
    assert not bool(diagnostics.cached_decision_action_refreshed)
    assert bool(diagnostics.value_effect_deferred_to_next_extended_action_selection)
    searched_q = search_agent.oak_agent.base_q_values(
        searched.state.oak_state,
        next_decision,
    )
    baseline_q = baseline_agent.oak_agent.base_q_values(
        baseline.state.oak_state,
        next_decision,
    )
    assert not bool(jnp.array_equal(searched_q, baseline_q))
    chex.assert_trees_all_equal(searched.action, baseline.action)
    chex.assert_trees_all_equal(
        searched.state.oak_state.stomp_state.base_last_action,
        baseline.state.oak_state.stomp_state.base_last_action,
    )
    chex.assert_trees_all_equal(
        searched.state.oak_state.stomp_state.last_primitive_action,
        baseline.state.oak_state.stomp_state.last_primitive_action,
    )
    chex.assert_trees_all_equal(
        searched.state.oak_state.stomp_state.executing_option,
        baseline.state.oak_state.stomp_state.executing_option,
    )
    chex.assert_trees_all_equal(
        searched.state.oak_state.stomp_state.rng_key,
        baseline.state.oak_state.stomp_state.rng_key,
    )


def test_rejected_prototype_transition_returns_fixed_neutral_search_diagnostics() -> None:
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=_stomp_config()),
            option_search_control=OptionSearchControlConfig(backup_budget=2),
        )
    )
    state = agent.start(agent.init(jr.key(33)), ANCHOR)
    rejected = agent.update_transition(
        state,
        PrototypeTransition(
            observation=state.current_raw_observation,
            action=state.current_action,
            decision_id=state.current_decision_id,
            reward=jnp.array(jnp.nan, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            next_observation=ANCHOR,
            next_decision_observation=ANCHOR,
        ),
    )

    diagnostics = rejected.option_search_control_diagnostics
    assert diagnostics is not None
    assert not bool(jnp.any(diagnostics.applied))
    assert int(diagnostics.applied_count) == 0
    chex.assert_trees_all_equal(rejected.state, state)


def test_option_search_control_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="backup_budget"):
        OptionSearchControlConfig(backup_budget=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backup_budget"):
        OptionSearchControlConfig(backup_budget=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_model_completions"):
        OptionSearchControlConfig(min_model_completions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_model_completions"):
        OptionSearchControlConfig(min_model_completions=1.5)  # type: ignore[arg-type]


def test_option_search_control_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = OptionSearchControlConfig(
        backup_budget=np.int32(4),
        min_model_completions=np.int64(2),
    )
    assert type(cfg.backup_budget) is int
    assert type(cfg.min_model_completions) is int
    assert cfg.backup_budget == 4
    assert cfg.min_model_completions == 2


def test_option_search_control_config_rejects_noncanonical_serialized_containers() -> None:
    class DictSubclass(dict[str, object]):
        pass

    payload = OptionSearchControlConfig().to_config()
    with pytest.raises(ValueError, match="actual dict"):
        OptionSearchControlConfig.from_config(DictSubclass(payload))
