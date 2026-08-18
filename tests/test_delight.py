"""Candidate-update audit tests and separate paper-defined delight tests."""

from __future__ import annotations

import json
import math

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import alberta_framework as alberta
import alberta_framework.core as core
import alberta_framework.core.delight as delight_module
from alberta_framework.core.delight import (
    DelightfulPolicyGradientConfig,
    GradientJoyApplicationResult,
    GradientJoyAssessment,
    GradientJoyConfig,
    GradientJoyEvidence,
    LearningValue,
    LearningValueAvailability,
    apply_gradient_joy_update,
    assess_gradient_joy,
    discrete_delightful_policy_gradient,
    stratify_delight_outcomes,
)

pytestmark = pytest.mark.unit


def test_delight_public_exports_resolve_to_core_implementation() -> None:
    for name in delight_module.__all__:
        implementation = getattr(delight_module, name)
        assert name in core.__all__
        assert name in alberta.__all__
        assert getattr(core, name) is implementation
        assert getattr(alberta, name) is implementation


def test_delight_config_roundtrip_and_fail_closed_guards() -> None:
    config = DelightfulPolicyGradientConfig(
        mode="delightful_pg",
        temperature=0.75,
        diagnostics_epsilon=1.0e-7,
    )

    payload = config.to_config()
    restored = DelightfulPolicyGradientConfig.from_config(payload)

    assert restored == config
    json.dumps(payload)
    with pytest.raises(ValueError, match="mode"):
        DelightfulPolicyGradientConfig(mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=float("nan"))
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=float("inf"))
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=1.0e-100)
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=1.0e100)
    with pytest.raises(ValueError, match="temperature"):
        DelightfulPolicyGradientConfig(temperature=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actor_trace_lambda=0"):
        DelightfulPolicyGradientConfig(actor_trace_lambda=0.5)
    with pytest.raises(ValueError, match="actor_trace_lambda"):
        DelightfulPolicyGradientConfig(actor_trace_lambda=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diagnostics_epsilon"):
        DelightfulPolicyGradientConfig(diagnostics_epsilon=0.0)
    with pytest.raises(ValueError, match="diagnostics_epsilon"):
        DelightfulPolicyGradientConfig(diagnostics_epsilon=float("nan"))
    with pytest.raises(ValueError, match="diagnostics_epsilon"):
        DelightfulPolicyGradientConfig(diagnostics_epsilon=1.0e-100)
    with pytest.raises(ValueError, match="diagnostics_epsilon"):
        DelightfulPolicyGradientConfig(diagnostics_epsilon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Kondo compute gating is unavailable"):
        DelightfulPolicyGradientConfig(kondo_enabled=True)
    with pytest.raises(ValueError, match="kondo_enabled must be an actual bool"):
        DelightfulPolicyGradientConfig(kondo_enabled=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kondo_enabled must be an actual bool"):
        DelightfulPolicyGradientConfig(kondo_enabled=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kondo_enabled must be an actual bool"):
        DelightfulPolicyGradientConfig(kondo_enabled=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kondo_enabled must be an actual bool"):
        DelightfulPolicyGradientConfig(kondo_enabled="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kondo_enabled must be an actual bool"):
        DelightfulPolicyGradientConfig.from_config(
            {
                "type": "DelightfulPolicyGradientConfig",
                "kondo_enabled": 0,
            }
        )


def test_learning_value_is_a_jax_pytree_without_an_implicit_sum() -> None:
    expected_channels = (
        "advantage",
        "action_surprisal",
        "delight",
        "epistemic_surprise",
        "aleatoric_uncertainty",
        "learning_progress",
        "change_probability",
        "safety_cost",
    )
    values = jnp.array([0.25, -0.5], dtype=jnp.float32)
    learning_value = LearningValue(
        advantage=values,
        action_surprisal=values + 1.0,
        delight=values + 2.0,
        epistemic_surprise=values + 3.0,
        aleatoric_uncertainty=values + 4.0,
        learning_progress=values + 5.0,
        change_probability=jax.nn.sigmoid(values),
        safety_cost=jnp.abs(values),
    )

    selected_consumer = jax.jit(lambda item: item.epistemic_surprise - item.aleatoric_uncertainty)
    selected = selected_consumer(learning_value)

    chex.assert_trees_all_close(selected, -jnp.ones_like(values))
    assert tuple(LearningValue.__dataclass_fields__) == expected_channels
    assert tuple(LearningValueAvailability.__dataclass_fields__) == expected_channels
    assert len(jax.tree_util.tree_leaves(learning_value)) == 8
    assert not hasattr(learning_value, "total")
    assert not hasattr(learning_value, "score")


def _scalar_learning_value(**overrides) -> LearningValue:
    values = {
        "advantage": jnp.array(1.0, dtype=jnp.float32),
        "action_surprisal": jnp.array(0.5, dtype=jnp.float32),
        "delight": jnp.array(0.5, dtype=jnp.float32),
        "epistemic_surprise": jnp.array(0.2, dtype=jnp.float32),
        "aleatoric_uncertainty": jnp.array(0.1, dtype=jnp.float32),
        "learning_progress": jnp.array(0.3, dtype=jnp.float32),
        "change_probability": jnp.array(0.4, dtype=jnp.float32),
        "safety_cost": jnp.array(0.0, dtype=jnp.float32),
    }
    values.update(overrides)
    return LearningValue(**values)


def _gradient_joy_evidence(
    *,
    objective_probe_gradient=None,
    retention_probe_gradient=None,
    safety_cost_gradient=None,
    probe_independence_attested=True,
    learning_value=None,
    learning_value_availability=None,
) -> GradientJoyEvidence:
    probe = {"weights": jnp.array([1.0, 0.0], dtype=jnp.float32)}
    all_channels_available = LearningValueAvailability(
        advantage=jnp.array(True),
        action_surprisal=jnp.array(True),
        delight=jnp.array(True),
        epistemic_surprise=jnp.array(True),
        aleatoric_uncertainty=jnp.array(True),
        learning_progress=jnp.array(True),
        change_probability=jnp.array(True),
        safety_cost=jnp.array(True),
    )
    return GradientJoyEvidence(
        objective_probe_gradient=(
            probe if objective_probe_gradient is None else objective_probe_gradient
        ),
        retention_probe_gradient=(
            probe if retention_probe_gradient is None else retention_probe_gradient
        ),
        safety_cost_gradient=(probe if safety_cost_gradient is None else safety_cost_gradient),
        objective_probe_available=jnp.array(True),
        retention_probe_available=jnp.array(True),
        safety_probe_available=jnp.array(True),
        probe_independence_attested=jnp.array(probe_independence_attested),
        learning_value=(_scalar_learning_value() if learning_value is None else learning_value),
        learning_value_availability=(
            all_channels_available
            if learning_value_availability is None
            else learning_value_availability
        ),
    )


def test_gradient_joy_config_roundtrip_and_fail_closed_contract() -> None:
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=2.0,
        alignment_temperature=0.5,
        norm_temperature=0.5,
    )

    assert GradientJoyConfig.from_config(config.to_config()) == config
    json.dumps(config.to_config())
    with pytest.raises(ValueError, match="candidate_semantics"):
        GradientJoyConfig(candidate_semantics="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_update_norm"):
        GradientJoyConfig(max_update_norm=0.0)
    with pytest.raises(ValueError, match="min_objective_decrease"):
        GradientJoyConfig(min_objective_decrease=-1.0)
    with pytest.raises(ValueError, match="alignment"):
        GradientJoyConfig(min_safety_descent_alignment=1.1)
    with pytest.raises(ValueError, match="max_update_norm"):
        GradientJoyConfig(
            max_update_norm=1.0e-8,
            diagnostics_epsilon=1.0e-8,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gradient_step_size", 1.0e-100),
        ("max_update_norm", 1.0e-100),
        ("alignment_temperature", 1.0e-100),
        ("norm_temperature", 1.0e-100),
        ("diagnostics_epsilon", 1.0e-100),
        ("min_objective_decrease", 1.0e-100),
        ("max_retention_loss_increase", 1.0e100),
        ("max_update_norm", 1.0e100),
        ("max_safety_cost_increase", 1.0e100),
        ("min_objective_descent_alignment", 1.0e-100),
        ("min_retention_descent_alignment", 1.0e-100),
        ("min_safety_descent_alignment", 1.0e-100),
    ),
)
def test_gradient_joy_config_rejects_values_that_do_not_survive_float32(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        GradientJoyConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "gradient_step_size",
        "max_update_norm",
        "min_objective_decrease",
        "max_retention_loss_increase",
        "max_safety_cost_increase",
        "min_objective_descent_alignment",
        "min_retention_descent_alignment",
        "min_safety_descent_alignment",
        "alignment_temperature",
        "norm_temperature",
        "diagnostics_epsilon",
    ),
)
def test_gradient_joy_config_rejects_boolean_numeric_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GradientJoyConfig(**{field: True})


def test_gradient_joy_hand_calculation_and_eight_channel_separation() -> None:
    """A jointly improving bounded update gets the weakest named factor."""
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=2.0,
        alignment_temperature=0.5,
        norm_temperature=0.5,
    )
    candidate_update = {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)}

    result = assess_gradient_joy(
        candidate_update,
        _gradient_joy_evidence(),
        config,
    )

    expected_factor = jax.nn.sigmoid(jnp.array(2.0, dtype=jnp.float32))
    assert bool(result.accepted)
    assert bool(result.sparks_joy)
    assert result.sparks_joy is result.accepted
    assert result.candidate_update_audit_passed is result.accepted
    chex.assert_trees_all_close(result.weight, expected_factor)
    chex.assert_trees_all_close(
        result.weighted_update["weights"],
        expected_factor * candidate_update["weights"],
    )
    chex.assert_trees_all_close(
        result.diagnostics.update_norm,
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.predicted_objective_decrease,
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.predicted_retention_loss_change,
        jnp.array(-1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.predicted_safety_cost_change,
        jnp.array(-1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        (
            result.diagnostics.objective_factor,
            result.diagnostics.retention_factor,
            result.diagnostics.safety_factor,
            result.diagnostics.trust_factor,
        ),
        (expected_factor,) * 4,
    )
    assert len(jax.tree_util.tree_leaves(result.learning_value)) == 8
    assert all(bool(flag) for flag in jax.tree_util.tree_leaves(result.channel_availability))
    assert not hasattr(result.learning_value, "total")
    assert not hasattr(result.learning_value, "score")

    changed_channels = _scalar_learning_value(
        advantage=jnp.array(-1000.0, dtype=jnp.float32),
        action_surprisal=jnp.array(1000.0, dtype=jnp.float32),
        delight=jnp.array(-1.0e6, dtype=jnp.float32),
        epistemic_surprise=jnp.array(100.0, dtype=jnp.float32),
        aleatoric_uncertainty=jnp.array(50.0, dtype=jnp.float32),
        learning_progress=jnp.array(-100.0, dtype=jnp.float32),
        change_probability=jnp.array(1.0, dtype=jnp.float32),
        safety_cost=jnp.array(1000.0, dtype=jnp.float32),
    )
    changed_result = assess_gradient_joy(
        candidate_update,
        _gradient_joy_evidence(learning_value=changed_channels),
        config,
    )
    chex.assert_trees_all_close(changed_result.weight, result.weight)


def test_gradient_joy_application_jits_over_nested_pytrees_and_preserves_dtypes() -> None:
    parameters = {
        "encoder": {
            "kernel": jnp.array([1.0, -2.0], dtype=jnp.float32),
        },
        "head": (jnp.array([0.5], dtype=jnp.float16),),
    }
    candidate = {
        "encoder": {
            "kernel": jnp.array([-0.2, -0.1], dtype=jnp.float32),
        },
        "head": (jnp.array([-0.05], dtype=jnp.float32),),
    }
    probe = jax.tree_util.tree_map(jnp.ones_like, candidate)
    evidence = _gradient_joy_evidence(
        objective_probe_gradient=probe,
        retention_probe_gradient=probe,
        safety_cost_gradient=probe,
    )
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=1.0,
    )
    compiled = jax.jit(
        lambda params, update: apply_gradient_joy_update(
            params,
            update,
            evidence,
            config,
        )
    )

    result = compiled(parameters, candidate)
    assessment = result.assessment
    expected = jax.tree_util.tree_map(
        lambda parameter, update: parameter
        + jnp.asarray(update, dtype=parameter.dtype),
        parameters,
        assessment.weighted_update,
    )

    assert isinstance(result, GradientJoyApplicationResult)
    assert bool(assessment.accepted)
    assert bool(result.effective_assessment.accepted)
    assert bool(result.applied)
    assert bool(result.parameters_finite)
    assert bool(result.cast_update_finite)
    assert bool(result.proposed_parameters_finite)
    assert bool(result.proposed_change_nonzero)
    chex.assert_trees_all_equal_structs(result.parameters, parameters)
    chex.assert_trees_all_close(result.parameters, expected)
    assert [leaf.dtype for leaf in jax.tree_util.tree_leaves(result.parameters)] == [
        leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)
    ]


def test_gradient_joy_application_rejection_is_an_atomic_noop_under_jit() -> None:
    parameters = {
        "body": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "head": (jnp.array([-3.0], dtype=jnp.float16),),
    }
    candidate = jax.tree_util.tree_map(
        lambda parameter: -jnp.ones_like(parameter, dtype=jnp.float32) * 0.1,
        parameters,
    )
    improving_probe = jax.tree_util.tree_map(jnp.ones_like, candidate)
    harmful_safety_probe = jax.tree_util.tree_map(
        lambda leaf: -jnp.ones_like(leaf),
        candidate,
    )
    evidence = _gradient_joy_evidence(
        objective_probe_gradient=improving_probe,
        retention_probe_gradient=improving_probe,
        safety_cost_gradient=harmful_safety_probe,
    )
    compiled = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            evidence,
            GradientJoyConfig(candidate_semantics="update", max_update_norm=1.0),
        )
    )

    result = compiled(parameters)

    assert not bool(result.assessment.accepted)
    assert not bool(result.assessment.sparks_joy)
    assert not bool(result.effective_assessment.accepted)
    assert not bool(result.applied)
    assert bool(result.parameters_finite)
    assert bool(result.cast_update_finite)
    assert bool(result.proposed_parameters_finite)
    assert not bool(result.proposed_change_nonzero)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_application_rejects_broadcasting_and_invalid_parameter_trees() -> None:
    candidate = {"weights": jnp.array([-0.5, 0.0], dtype=jnp.float32)}
    evidence = _gradient_joy_evidence()
    config = GradientJoyConfig(candidate_semantics="update", max_update_norm=1.0)
    compiled = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            evidence,
            config,
        )
    )

    with pytest.raises(ValueError, match="shapes must match exactly"):
        compiled({"weights": jnp.ones((2, 1), dtype=jnp.float32)})
    with pytest.raises(ValueError, match="structures must match exactly"):
        compiled({"other": jnp.ones(2, dtype=jnp.float32)})
    with pytest.raises(ValueError, match="real floating dtype"):
        compiled({"weights": jnp.ones(2, dtype=jnp.int32)})
    with pytest.raises(ValueError, match="non-empty PyTree"):
        compiled({})


def test_gradient_joy_application_fails_closed_atomically_on_unsafe_values() -> None:
    cast_parameters = {
        "weights": jnp.zeros(2, dtype=jnp.float16),
    }
    cast_candidate = {
        "weights": jnp.array([-100000.0, 0.0], dtype=jnp.float32),
    }
    cast_result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            cast_candidate,
            _gradient_joy_evidence(),
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=200000.0,
                norm_temperature=1000.0,
            ),
        )
    )(cast_parameters)

    assert bool(cast_result.assessment.accepted)
    assert not bool(cast_result.effective_assessment.accepted)
    assert not bool(cast_result.applied)
    assert bool(cast_result.parameters_finite)
    assert not bool(cast_result.cast_update_finite)
    assert not bool(cast_result.proposed_parameters_finite)
    assert bool(cast_result.proposed_change_nonzero)
    chex.assert_trees_all_equal(cast_result.parameters, cast_parameters)

    overflow_parameters = {
        "overflow": jnp.array([65504.0], dtype=jnp.float16),
        "safe": jnp.array([1.0], dtype=jnp.float32),
    }
    overflow_candidate = {
        "overflow": jnp.array([65504.0], dtype=jnp.float32),
        "safe": jnp.array([-0.25], dtype=jnp.float32),
    }
    overflow_probe = {
        "overflow": jnp.array([-1.0], dtype=jnp.float32),
        "safe": jnp.array([1.0], dtype=jnp.float32),
    }
    overflow_evidence = _gradient_joy_evidence(
        objective_probe_gradient=overflow_probe,
        retention_probe_gradient=overflow_probe,
        safety_cost_gradient=overflow_probe,
    )
    overflow_apply = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            overflow_candidate,
            overflow_evidence,
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=70000.0,
                norm_temperature=1000.0,
            ),
        )
    )

    overflow_result = overflow_apply(overflow_parameters)

    assert bool(overflow_result.assessment.accepted)
    assert not bool(overflow_result.effective_assessment.accepted)
    assert float(overflow_result.assessment.weighted_update["safe"][0]) != 0.0
    assert not bool(overflow_result.applied)
    assert bool(overflow_result.parameters_finite)
    assert bool(overflow_result.cast_update_finite)
    assert not bool(overflow_result.proposed_parameters_finite)
    assert bool(overflow_result.proposed_change_nonzero)
    chex.assert_trees_all_equal(overflow_result.parameters, overflow_parameters)

    nonfinite_parameters = {
        "bad": jnp.array([jnp.nan], dtype=jnp.float32),
        "safe": jnp.array([1.0], dtype=jnp.float32),
    }
    finite_candidate = {
        "bad": jnp.array([-0.1], dtype=jnp.float32),
        "safe": jnp.array([-0.1], dtype=jnp.float32),
    }
    finite_probe = jax.tree_util.tree_map(jnp.ones_like, finite_candidate)
    nonfinite_result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            finite_candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=finite_probe,
                retention_probe_gradient=finite_probe,
                safety_cost_gradient=finite_probe,
            ),
            GradientJoyConfig(candidate_semantics="update", max_update_norm=1.0),
        )
    )(nonfinite_parameters)

    assert bool(nonfinite_result.assessment.accepted)
    assert not bool(nonfinite_result.effective_assessment.accepted)
    assert not bool(nonfinite_result.applied)
    assert not bool(nonfinite_result.parameters_finite)
    assert bool(nonfinite_result.cast_update_finite)
    assert not bool(nonfinite_result.proposed_parameters_finite)
    assert bool(nonfinite_result.proposed_change_nonzero)
    assert bool(jnp.isnan(nonfinite_result.parameters["bad"][0]))
    chex.assert_trees_all_equal(
        nonfinite_result.parameters["safe"],
        nonfinite_parameters["safe"],
    )

    finite_parameters = {
        "weights": jnp.array([1.0, 2.0], dtype=jnp.float32),
    }
    nonfinite_candidate = {
        "weights": jnp.array([jnp.nan, -0.1], dtype=jnp.float32),
    }
    rejected_result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            nonfinite_candidate,
            _gradient_joy_evidence(),
            GradientJoyConfig(candidate_semantics="update", max_update_norm=1.0),
        )
    )(finite_parameters)

    assert not bool(rejected_result.assessment.accepted)
    assert not bool(rejected_result.effective_assessment.accepted)
    assert not bool(rejected_result.assessment.diagnostics.candidate_finite)
    assert not bool(rejected_result.applied)
    assert bool(rejected_result.parameters_finite)
    assert bool(rejected_result.cast_update_finite)
    assert bool(rejected_result.proposed_parameters_finite)
    assert not bool(rejected_result.proposed_change_nonzero)
    chex.assert_tree_all_finite(rejected_result.assessment.weighted_update)
    chex.assert_trees_all_equal(rejected_result.parameters, finite_parameters)


def test_gradient_joy_application_reports_update_lost_to_parameter_precision() -> None:
    parameters = {
        "weights": jnp.array([1.0], dtype=jnp.float16),
    }
    candidate = {
        "weights": jnp.array([-1.0e-4], dtype=jnp.float32),
    }
    probe = jax.tree_util.tree_map(jnp.ones_like, candidate)
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=probe,
                retention_probe_gradient=probe,
                safety_cost_gradient=probe,
            ),
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=1.0e-3,
            ),
        )
    )(parameters)

    assert bool(result.assessment.accepted)
    assert not bool(result.effective_assessment.accepted)
    assert not bool(result.applied)
    assert bool(result.parameters_finite)
    assert bool(result.cast_update_finite)
    assert bool(result.proposed_parameters_finite)
    assert not bool(result.proposed_change_nonzero)
    assert float(result.assessment.weighted_update["weights"][0]) != 0.0
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_application_vetoes_quantized_delta_outside_trust_bound() -> None:
    """A finite stored delta must satisfy the same norm bound as its candidate."""
    parameters = {
        "weights": jnp.array([1.0], dtype=jnp.float16),
    }
    candidate = {
        "weights": jnp.array([-2.5e-4], dtype=jnp.float32),
    }
    probe = {
        "weights": jnp.ones(1, dtype=jnp.float32),
    }
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=2.6e-4,
        alignment_temperature=1.0e-2,
        norm_temperature=1.0e-6,
    )
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=probe,
                retention_probe_gradient=probe,
                safety_cost_gradient=probe,
            ),
            config,
        )
    )(parameters)

    assert bool(result.assessment.accepted)
    assert bool(result.assessment.diagnostics.within_trust_region)
    assert float(result.assessment.diagnostics.update_norm) < config.max_update_norm
    assert not bool(result.effective_assessment.accepted)
    assert not bool(result.effective_assessment.diagnostics.within_trust_region)
    chex.assert_trees_all_close(
        result.effective_assessment.candidate_update["weights"],
        jnp.array([-4.8828125e-4], dtype=jnp.float32),
    )
    assert (
        float(result.effective_assessment.diagnostics.update_norm)
        > config.max_update_norm
    )
    assert bool(result.proposed_change_nonzero)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_application_promotes_stored_endpoints_before_delta_audit() -> None:
    """Float16 subtraction rounding cannot hide an out-of-bound stored delta."""
    parameters = {
        "weights": jnp.array([10.7421875], dtype=jnp.float16),
    }
    candidate = {
        "weights": jnp.array([16.36760711669922], dtype=jnp.float32),
    }
    descending_probe = {
        "weights": jnp.array([-1.0], dtype=jnp.float32),
    }
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=16.378,
        alignment_temperature=1.0e-4,
        norm_temperature=1.0e-6,
    )
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=descending_probe,
                retention_probe_gradient=descending_probe,
                safety_cost_gradient=descending_probe,
            ),
            config,
        )
    )(parameters)

    assert bool(result.assessment.accepted)
    assert float(result.assessment.weight) == 1.0
    chex.assert_trees_all_close(
        result.effective_assessment.candidate_update["weights"],
        jnp.array([16.3828125], dtype=jnp.float32),
    )
    assert (
        float(result.effective_assessment.diagnostics.update_norm)
        > config.max_update_norm
    )
    assert not bool(result.effective_assessment.diagnostics.within_trust_region)
    assert not bool(result.effective_assessment.accepted)
    assert bool(result.proposed_change_nonzero)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_certifies_norm_boundary_in_eager_jit_and_application() -> None:
    """A rounded-down norm point cannot cross an exact trust boundary."""
    candidate = {
        "first": jnp.array([-6.4952946559060365e-06], dtype=jnp.float32),
        "second": (jnp.array([2.201160168624483e-05], dtype=jnp.float32),),
    }
    probe = jax.tree_util.tree_map(lambda leaf: -leaf, candidate)
    max_update_norm = float(jnp.float32(2.294993282703217e-05))
    exact_norm = math.sqrt(
        sum(float(value) ** 2 for leaf in jax.tree_util.tree_leaves(candidate) for value in leaf)
    )
    assert exact_norm > max_update_norm
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=max_update_norm,
        alignment_temperature=1.0e-8,
        norm_temperature=1.2e-38,
    )
    evidence = _gradient_joy_evidence(
        objective_probe_gradient=probe,
        retention_probe_gradient=probe,
        safety_cost_gradient=probe,
    )

    for result in (
        assess_gradient_joy(candidate, evidence, config),
        jax.jit(lambda update: assess_gradient_joy(update, evidence, config))(candidate),
    ):
        assert float(result.diagnostics.update_norm) < max_update_norm
        assert float(result.diagnostics.update_norm_upper_bound) > max_update_norm
        assert float(result.diagnostics.update_norm_lower_bound) <= exact_norm
        assert exact_norm <= float(result.diagnostics.update_norm_upper_bound)
        assert bool(result.diagnostics.update_norm_resolved)
        assert not bool(result.diagnostics.within_trust_region)
        assert not bool(result.accepted)
        chex.assert_trees_all_equal(
            result.weighted_update,
            jax.tree_util.tree_map(jnp.zeros_like, candidate),
        )

    parameters = jax.tree_util.tree_map(jnp.zeros_like, candidate)

    def apply(params, update):
        return apply_gradient_joy_update(params, update, evidence, config)

    for result in (
        apply(parameters, candidate),
        jax.jit(apply)(parameters, candidate),
    ):
        assert not bool(result.assessment.diagnostics.within_trust_region)
        assert not bool(result.assessment.accepted)
        assert not bool(result.effective_assessment.accepted)
        assert not bool(result.applied)
        chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_certifies_alignment_boundary_in_eager_jit_and_application() -> None:
    """A rounded-up point cosine cannot cross an exact alignment threshold."""
    candidate = jnp.array(
        [-0.3096868097782135, 0.3673213720321655],
        dtype=jnp.float32,
    )
    probe = jnp.array(
        [-1.7595300674438477, -2.5661325454711914],
        dtype=jnp.float32,
    )
    threshold = float(jnp.float32(0.26603594422340393))
    exact_dot = sum(float(left) * float(right) for left, right in zip(probe, candidate))
    exact_alignment = -exact_dot / math.sqrt(
        sum(float(value) ** 2 for value in probe)
        * sum(float(value) ** 2 for value in candidate)
    )
    assert exact_alignment < threshold
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=1.0,
        min_objective_descent_alignment=threshold,
        min_retention_descent_alignment=threshold,
        min_safety_descent_alignment=threshold,
    )

    def evidence_for(protected_probe: Array) -> GradientJoyEvidence:
        return _gradient_joy_evidence(
            objective_probe_gradient=protected_probe,
            retention_probe_gradient=protected_probe,
            safety_cost_gradient=protected_probe,
        )

    evidence = evidence_for(probe)
    for result in (
        assess_gradient_joy(candidate, evidence, config),
        jax.jit(
            lambda update, protected_probe: assess_gradient_joy(
                update,
                evidence_for(protected_probe),
                config,
            )
        )(candidate, probe),
    ):
        assert float(result.diagnostics.objective_descent_alignment) == threshold
        assert (
            float(result.diagnostics.objective_descent_alignment_lower_bound)
            < threshold
        )
        assert exact_alignment >= float(
            result.diagnostics.objective_descent_alignment_lower_bound
        )
        assert not bool(result.diagnostics.objective_improves)
        assert not bool(result.accepted)

    parameters = jnp.zeros_like(candidate)

    def apply(params: Array, update: Array, protected_probe: Array):
        return apply_gradient_joy_update(
            params,
            update,
            evidence_for(protected_probe),
            config,
        )

    for result in (
        apply(parameters, candidate, probe),
        jax.jit(apply)(parameters, candidate, probe),
    ):
        assert not bool(result.assessment.accepted)
        assert not bool(result.effective_assessment.accepted)
        assert not bool(result.applied)
        chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_rejects_underflowed_harmful_protected_probe_dot() -> None:
    """A normal-valued safety probe cannot underflow into a passing zero."""
    parameters = {"weights": jnp.array([0.0], dtype=jnp.float32)}
    candidate = {"weights": jnp.array([-8.0e-8], dtype=jnp.float32)}
    objective_probe = {"weights": jnp.array([1.0e8], dtype=jnp.float32)}
    safety_probe = {"weights": jnp.array([-1.0e-31], dtype=jnp.float32)}
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=objective_probe,
                retention_probe_gradient=objective_probe,
                safety_cost_gradient=safety_probe,
            ),
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=1.0e-6,
            ),
        )
    )(parameters)

    assert bool(result.assessment.diagnostics.candidate_finite)
    assert bool(result.assessment.diagnostics.safety_probe_finite)
    assert float(result.assessment.diagnostics.safety_probe_norm) > 0.0
    assert float(result.assessment.diagnostics.safety_descent_alignment) < 0.0
    assert not bool(result.assessment.diagnostics.safety_preserved)
    assert not bool(result.assessment.accepted)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


@pytest.mark.parametrize(
    ("probe_kind", "verdict_field", "resolved_field"),
    (
        ("objective", "objective_improves", "objective_dot_resolved"),
        ("retention", "retention_preserved", "retention_dot_resolved"),
        ("safety", "safety_preserved", "safety_dot_resolved"),
    ),
)
def test_gradient_joy_rejects_cancellation_sensitive_dot(
    probe_kind: str,
    verdict_field: str,
    resolved_field: str,
) -> None:
    """Cancellation cannot turn a numerically uncertain dot into a verdict."""
    candidate = {"weights": jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32)}
    descending_probe = {
        "weights": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    }
    cancellation_probe = {
        "weights": jnp.array([-1.0e20, 1.0e20, -1.0], dtype=jnp.float32),
    }
    probe_arguments = {
        "objective_probe_gradient": descending_probe,
        "retention_probe_gradient": descending_probe,
        "safety_cost_gradient": descending_probe,
    }
    probe_arguments[
        {
            "objective": "objective_probe_gradient",
            "retention": "retention_probe_gradient",
            "safety": "safety_cost_gradient",
        }[probe_kind]
    ] = cancellation_probe

    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(**probe_arguments),
        GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
    )

    if probe_kind == "objective":
        assert float(result.diagnostics.predicted_objective_decrease) < 0.0
    elif probe_kind == "retention":
        assert float(result.diagnostics.predicted_retention_loss_change) > 0.0
    else:
        assert float(result.diagnostics.predicted_safety_cost_change) > 0.0
    assert not bool(getattr(result.diagnostics, resolved_field))
    assert not bool(result.diagnostics.derived_numerics_valid)
    assert not bool(getattr(result.diagnostics, verdict_field))
    assert not bool(result.accepted)
    chex.assert_trees_all_equal(
        result.weighted_update,
        {"weights": jnp.zeros(3, dtype=jnp.float32)},
    )


def test_gradient_joy_application_vetoes_dot_sign_disagreement_atomically() -> None:
    """The stored-delta boundary inherits the cancellation fail-closed gate."""
    parameters = {"weights": jnp.zeros(3, dtype=jnp.float32)}
    candidate = {"weights": jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32)}
    cancellation_probe = {
        "weights": jnp.array([-1.0e20, 1.0e20, -1.0], dtype=jnp.float32),
    }
    descending_probe = {
        "weights": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    }
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=cancellation_probe,
                retention_probe_gradient=descending_probe,
                safety_cost_gradient=descending_probe,
            ),
            GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
        )
    )(parameters)

    assert float(result.assessment.diagnostics.predicted_objective_decrease) < 0.0
    assert not bool(result.assessment.diagnostics.derived_numerics_valid)
    assert not bool(result.assessment.accepted)
    assert not bool(result.effective_assessment.accepted)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_rejects_same_sign_cancellation_in_eager_and_jit() -> None:
    """A roundoff interval must catch two reductions agreeing in the wrong sign."""
    candidate = jnp.ones(6, dtype=jnp.float32)
    descending_probe = -jnp.ones(6, dtype=jnp.float32)
    cancellation_probe = jnp.array(
        [2.0, 1.0e10, -1.0e10, -1.0e10, 1.0e10, -1.0],
        dtype=jnp.float32,
    )
    assert sum(float(value) for value in cancellation_probe) == 1.0
    config = GradientJoyConfig(candidate_semantics="update", max_update_norm=3.0)

    def evaluate(
        update: Array,
        objective: Array,
        safety: Array,
    ) -> GradientJoyAssessment:
        return assess_gradient_joy(
            update,
            _gradient_joy_evidence(
                objective_probe_gradient=objective,
                retention_probe_gradient=objective,
                safety_cost_gradient=safety,
            ),
            config,
        )

    for result in (
        evaluate(candidate, descending_probe, cancellation_probe),
        jax.jit(evaluate)(candidate, descending_probe, cancellation_probe),
    ):
        assert bool(result.diagnostics.objective_dot_resolved)
        assert not bool(result.diagnostics.safety_dot_resolved)
        assert float(result.diagnostics.safety_dot_error_bound) > 0.0
        assert not bool(result.diagnostics.derived_numerics_valid)
        assert not bool(result.diagnostics.safety_preserved)
        assert not bool(result.accepted)
        chex.assert_trees_all_equal(
            result.weighted_update,
            jnp.zeros(6, dtype=jnp.float32),
        )


def test_gradient_joy_application_vetoes_same_sign_cancellation_eager_and_jit() -> None:
    """Unresolved cancellation is an atomic no-op at the stored-delta boundary."""
    parameters = jnp.zeros(6, dtype=jnp.float32)
    candidate = jnp.ones(6, dtype=jnp.float32)
    descending_probe = -jnp.ones(6, dtype=jnp.float32)
    cancellation_probe = jnp.array(
        [2.0, 1.0e10, -1.0e10, -1.0e10, 1.0e10, -1.0],
        dtype=jnp.float32,
    )
    config = GradientJoyConfig(candidate_semantics="update", max_update_norm=3.0)

    def apply(
        params: Array,
        update: Array,
        objective: Array,
        safety: Array,
    ) -> GradientJoyApplicationResult:
        return apply_gradient_joy_update(
            params,
            update,
            _gradient_joy_evidence(
                objective_probe_gradient=objective,
                retention_probe_gradient=objective,
                safety_cost_gradient=safety,
            ),
            config,
        )

    for result in (
        apply(parameters, candidate, descending_probe, cancellation_probe),
        jax.jit(apply)(parameters, candidate, descending_probe, cancellation_probe),
    ):
        assert not bool(result.assessment.diagnostics.safety_dot_resolved)
        assert not bool(result.assessment.accepted)
        assert not bool(result.effective_assessment.accepted)
        assert not bool(result.applied)
        chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_rejects_overflowed_derived_protected_probe_dot() -> None:
    """Finite leaves whose derived dot overflows must fail the evidence gate."""
    parameters = {"weights": jnp.array([0.0], dtype=jnp.float32)}
    candidate = {"weights": jnp.array([-1.0e10], dtype=jnp.float32)}
    objective_probe = {"weights": jnp.array([1.0], dtype=jnp.float32)}
    safety_probe = {"weights": jnp.array([-1.0e30], dtype=jnp.float32)}
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=objective_probe,
                retention_probe_gradient=objective_probe,
                safety_cost_gradient=safety_probe,
            ),
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=2.0e10,
                norm_temperature=1.0e9,
            ),
        )
    )(parameters)

    assert bool(result.assessment.diagnostics.candidate_finite)
    assert bool(result.assessment.diagnostics.safety_probe_finite)
    assert not bool(result.assessment.diagnostics.derived_numerics_valid)
    assert not bool(result.assessment.diagnostics.evidence_complete)
    assert not bool(result.assessment.accepted)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_application_vetoes_quantization_that_flips_probe_verdicts() -> None:
    """Partially rounded updates cannot discard benefit while retaining harm."""
    parameters = {
        "weights": jnp.array([1000.0, 0.0], dtype=jnp.float16),
    }
    candidate = {
        "weights": jnp.array([-1.0e-3, 1.0e-4], dtype=jnp.float32),
    }
    objective_probe = {
        "weights": jnp.array([1.0, 0.0], dtype=jnp.float32),
    }
    safety_probe = {
        "weights": jnp.array([1.0, 5.0], dtype=jnp.float32),
    }
    result = jax.jit(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient=objective_probe,
                retention_probe_gradient=objective_probe,
                safety_cost_gradient=safety_probe,
            ),
            GradientJoyConfig(
                candidate_semantics="update",
                max_update_norm=1.0e-2,
            ),
        )
    )(parameters)

    assert bool(result.assessment.accepted)
    assert float(result.assessment.diagnostics.predicted_tentative_objective_decrease) > 0.0
    assert float(result.assessment.diagnostics.predicted_tentative_safety_cost_change) < 0.0
    assert not bool(result.effective_assessment.accepted)
    chex.assert_trees_all_close(
        result.effective_assessment.candidate_update["weights"],
        jnp.array([0.0, 5.2273273e-5], dtype=jnp.float32),
    )
    assert (
        float(result.effective_assessment.diagnostics.predicted_objective_decrease)
        == 0.0
    )
    assert float(result.effective_assessment.diagnostics.predicted_safety_cost_change) > 0.0
    assert not bool(result.effective_assessment.diagnostics.objective_improves)
    assert not bool(result.effective_assessment.diagnostics.safety_preserved)
    assert bool(result.proposed_change_nonzero)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_application_has_identity_parameter_gradient_only() -> None:
    parameters = jnp.array([1.0, -2.0], dtype=jnp.float32)
    candidate = jnp.array([-0.2, -0.1], dtype=jnp.float32)
    probe = jnp.ones_like(candidate)
    evidence = _gradient_joy_evidence(
        objective_probe_gradient=probe,
        retention_probe_gradient=probe,
        safety_cost_gradient=probe,
    )
    config = GradientJoyConfig(candidate_semantics="update", max_update_norm=1.0)

    parameter_jacobian = jax.jacrev(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            evidence,
            config,
        ).parameters
    )(parameters)
    candidate_jacobian = jax.jacrev(
        lambda update: apply_gradient_joy_update(
            parameters,
            update,
            evidence,
            config,
        ).parameters
    )(candidate)
    effective_candidate_jacobian = jax.jacrev(
        lambda update: apply_gradient_joy_update(
            parameters,
            update,
            evidence,
            config,
        ).effective_assessment.candidate_update
    )(candidate)
    effective_parameter_jacobian = jax.jacrev(
        lambda params: apply_gradient_joy_update(
            params,
            candidate,
            evidence,
            config,
        ).effective_assessment.candidate_update
    )(parameters)

    chex.assert_trees_all_close(
        parameter_jacobian,
        jnp.eye(2, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        candidate_jacobian,
        jnp.zeros((2, 2), dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        effective_candidate_jacobian,
        jnp.zeros((2, 2), dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        effective_parameter_jacobian,
        jnp.zeros((2, 2), dtype=jnp.float32),
    )


def test_gradient_joy_alignment_is_scale_invariant_above_the_norm_floor() -> None:
    candidate = {"weights": jnp.array([-1.0e-6], dtype=jnp.float32)}
    tiny_aligned_probe = {"weights": jnp.array([1.0e-12], dtype=jnp.float32)}
    evidence = _gradient_joy_evidence(
        objective_probe_gradient=tiny_aligned_probe,
        retention_probe_gradient=tiny_aligned_probe,
        safety_cost_gradient=tiny_aligned_probe,
    )
    result = assess_gradient_joy(
        candidate,
        evidence,
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=1.0e-3,
            min_objective_descent_alignment=0.9,
            min_retention_descent_alignment=0.9,
            min_safety_descent_alignment=0.9,
            alignment_temperature=0.1,
            norm_temperature=1.0e-4,
            diagnostics_epsilon=1.0e-8,
        ),
    )

    assert bool(result.accepted)
    chex.assert_trees_all_close(
        result.diagnostics.objective_descent_alignment,
        jnp.array(1.0, dtype=jnp.float32),
        atol=1.0e-5,
    )


@pytest.mark.parametrize("seed", (2, 11))
def test_gradient_joy_fails_closed_when_endpoint_alignment_is_not_certified(
    seed: int,
) -> None:
    """A point cosine near one cannot outrun its certified lower endpoint."""
    probe_values = jax.random.normal(
        jax.random.PRNGKey(seed),
        (1003,),
        dtype=jnp.float32,
    )
    candidate = {"weights": -0.01 * probe_values}
    probe = {"weights": probe_values}
    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(
            objective_probe_gradient=probe,
            retention_probe_gradient=probe,
            safety_cost_gradient=probe,
        ),
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=1.0,
            min_objective_descent_alignment=1.0,
            min_retention_descent_alignment=1.0,
            min_safety_descent_alignment=1.0,
        ),
    )

    assert not bool(result.accepted)
    for alignment, lower_bound in (
        (
            result.diagnostics.objective_descent_alignment,
            result.diagnostics.objective_descent_alignment_lower_bound,
        ),
        (
            result.diagnostics.retention_descent_alignment,
            result.diagnostics.retention_descent_alignment_lower_bound,
        ),
        (
            result.diagnostics.safety_descent_alignment,
            result.diagnostics.safety_descent_alignment_lower_bound,
        ),
    ):
        assert -1.0 <= float(alignment) <= 1.0
        assert float(alignment) >= 1.0 - 4.0 * float(jnp.finfo(jnp.float32).eps)
        assert float(lower_bound) < 1.0 - 4.0 * float(jnp.finfo(jnp.float32).eps)


def test_gradient_joy_accepts_exact_one_dimensional_endpoint_alignment() -> None:
    """An exactly certifiable one-dimensional cosine can meet endpoint one."""
    candidate = {"weights": jnp.array([-0.25], dtype=jnp.float32)}
    probe = {"weights": jnp.array([3.0], dtype=jnp.float32)}
    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(
            objective_probe_gradient=probe,
            retention_probe_gradient=probe,
            safety_cost_gradient=probe,
        ),
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=1.0,
            min_objective_descent_alignment=1.0,
            min_retention_descent_alignment=1.0,
            min_safety_descent_alignment=1.0,
        ),
    )

    assert bool(result.accepted)
    assert float(result.diagnostics.objective_descent_alignment) == 1.0
    assert float(
        result.diagnostics.objective_descent_alignment_lower_bound
    ) >= 1.0 - 4.0 * float(jnp.finfo(jnp.float32).eps)


def test_gradient_joy_rejects_tentative_update_below_update_norm_floor() -> None:
    candidate = {"weights": jnp.array([-1.5e-8], dtype=jnp.float32)}
    probe = {"weights": jnp.array([1.0e8], dtype=jnp.float32)}
    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(
            objective_probe_gradient=probe,
            retention_probe_gradient=probe,
            safety_cost_gradient=probe,
        ),
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=1.5e-8,
            diagnostics_epsilon=1.0e-8,
        ),
    )

    assert bool(result.diagnostics.nonzero_update)
    assert not bool(result.diagnostics.tentative_nonzero_update)
    assert float(result.diagnostics.tentative_update_norm) < 1.0e-8
    assert not bool(result.accepted)
    chex.assert_trees_all_close(
        result.weighted_update["weights"],
        jnp.zeros(1, dtype=jnp.float32),
    )


def test_gradient_joy_audits_elementwise_rounded_tentative_update() -> None:
    """Scalar scaling cannot stand in for the actually rounded update tree."""
    candidate = jnp.array([-2.836829920794233e-32], dtype=jnp.float32)
    probe = jnp.array([0.8063097596168518], dtype=jnp.float32)
    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(
            objective_probe_gradient=probe,
            retention_probe_gradient=probe,
            safety_cost_gradient=probe,
        ),
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=1.0e-30,
            diagnostics_epsilon=1.2e-38,
            min_objective_descent_alignment=1.0,
            min_retention_descent_alignment=1.0,
            min_safety_descent_alignment=1.0,
            alignment_temperature=0.1,
            norm_temperature=1.0e-30,
        ),
    )

    assert float(result.diagnostics.tentative_weight) > 0.0
    assert not bool(result.diagnostics.tentative_objective_dot_resolved)
    assert not bool(result.diagnostics.derived_numerics_valid)
    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.weighted_update, jnp.zeros_like(candidate))


def test_gradient_joy_tentative_rounding_boundary_is_atomic_eager_and_jit() -> None:
    """A resolved candidate cannot lend its certificate to rounded scaling."""
    candidate = jnp.array(
        [-0.53493332862854, 0.9215275645256042],
        dtype=jnp.float32,
    )
    probe = jnp.array(
        [-1.6531749943149495e-32, -1.3062328325207423e-32],
        dtype=jnp.float32,
    )
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=1.0877355337142944,
        min_objective_decrease=1.7734800820290934e-33,
        alignment_temperature=0.001,
        norm_temperature=0.1,
    )

    def evidence_for(protected_probe: Array) -> GradientJoyEvidence:
        return _gradient_joy_evidence(
            objective_probe_gradient=protected_probe,
            retention_probe_gradient=protected_probe,
            safety_cost_gradient=protected_probe,
        )

    def assess(update: Array, protected_probe: Array) -> GradientJoyAssessment:
        return assess_gradient_joy(update, evidence_for(protected_probe), config)

    for result in (
        assess(candidate, probe),
        jax.jit(assess)(candidate, probe),
    ):
        assert float(result.diagnostics.tentative_weight) == pytest.approx(
            0.5552690029144287
        )
        assert bool(result.diagnostics.objective_dot_resolved)
        assert not bool(result.diagnostics.tentative_objective_dot_resolved)
        assert float(result.diagnostics.tentative_objective_dot_error_bound) == 0.0
        assert not bool(result.diagnostics.derived_numerics_valid)
        assert not bool(result.diagnostics.objective_improves)
        assert not bool(result.accepted)
        chex.assert_trees_all_equal(result.weighted_update, jnp.zeros_like(candidate))

    parameters = jnp.zeros_like(candidate)

    def apply(params: Array, update: Array, protected_probe: Array):
        return apply_gradient_joy_update(
            params,
            update,
            evidence_for(protected_probe),
            config,
        )

    for result in (
        apply(parameters, candidate, probe),
        jax.jit(apply)(parameters, candidate, probe),
    ):
        assert not bool(result.assessment.accepted)
        assert not bool(result.effective_assessment.accepted)
        assert not bool(result.applied)
        chex.assert_trees_all_equal(result.parameters, parameters)


def test_gradient_joy_jits_converts_gradient_and_blocks_meta_gradients() -> None:
    config = GradientJoyConfig(
        candidate_semantics="gradient",
        gradient_step_size=0.25,
        max_update_norm=1.0,
        alignment_temperature=0.5,
        norm_temperature=0.5,
    )
    evidence = _gradient_joy_evidence()
    candidate_gradient = {"weights": jnp.array([2.0, 0.0], dtype=jnp.float32)}
    compiled = jax.jit(lambda candidate: assess_gradient_joy(candidate, evidence, config))

    result = compiled(candidate_gradient)
    weight_gradient = jax.grad(
        lambda candidate: (
            assess_gradient_joy(
                {"weights": candidate},
                evidence,
                config,
            ).weight
        )
    )(candidate_gradient["weights"])
    weighted_update_jacobian = jax.jacrev(
        lambda candidate: assess_gradient_joy(
            {"weights": candidate},
            evidence,
            config,
        ).weighted_update["weights"]
    )(candidate_gradient["weights"])
    candidate_update_jacobian = jax.jacrev(
        lambda candidate: assess_gradient_joy(
            {"weights": candidate},
            evidence,
            config,
        ).candidate_update["weights"]
    )(candidate_gradient["weights"])
    diagnostic_gradient = jax.grad(
        lambda candidate: assess_gradient_joy(
            {"weights": candidate},
            evidence,
            config,
        ).diagnostics.predicted_objective_decrease
    )(candidate_gradient["weights"])
    probe_meta_gradient = jax.grad(
        lambda probe: (
            assess_gradient_joy(
                candidate_gradient,
                evidence.replace(
                    objective_probe_gradient={"weights": probe},
                ),
                config,
            ).weight
        )
    )(evidence.objective_probe_gradient["weights"])
    channel_meta_gradient = jax.grad(
        lambda advantage: (
            assess_gradient_joy(
                candidate_gradient,
                evidence.replace(
                    learning_value=evidence.learning_value.replace(
                        advantage=advantage,
                    )
                ),
                config,
            ).weight
        )
    )(evidence.learning_value.advantage)

    assert bool(result.accepted)
    chex.assert_trees_all_close(
        result.candidate_update["weights"],
        jnp.array([-0.5, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        weight_gradient,
        jnp.zeros_like(candidate_gradient["weights"]),
    )
    chex.assert_trees_all_close(
        weighted_update_jacobian,
        jnp.zeros((2, 2), dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        candidate_update_jacobian,
        jnp.zeros((2, 2), dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        diagnostic_gradient,
        jnp.zeros_like(candidate_gradient["weights"]),
    )
    chex.assert_trees_all_close(
        probe_meta_gradient,
        jnp.zeros_like(evidence.objective_probe_gradient["weights"]),
    )
    chex.assert_trees_all_close(
        channel_meta_gradient,
        jnp.array(0.0, dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result)


def test_gradient_joy_safety_conflict_is_a_named_hard_veto() -> None:
    result = assess_gradient_joy(
        {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)},
        _gradient_joy_evidence(
            safety_cost_gradient={"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)}
        ),
        GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
    )

    assert not bool(result.accepted)
    assert not bool(result.diagnostics.safety_preserved)
    chex.assert_trees_all_close(
        result.diagnostics.predicted_safety_cost_change,
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.weight,
        jnp.array(0.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.weighted_update["weights"],
        jnp.zeros(2, dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    ("config_overrides", "objective_probe_scale", "raw_field", "tentative_field"),
    (
        (
            {"min_objective_decrease": 0.1},
            1.0,
            "predicted_objective_decrease",
            "predicted_tentative_objective_decrease",
        ),
        (
            {"max_retention_loss_increase": -0.1},
            10.0,
            "predicted_retention_loss_change",
            "predicted_tentative_retention_loss_change",
        ),
        (
            {"max_safety_cost_increase": -0.1},
            10.0,
            "predicted_safety_cost_change",
            "predicted_tentative_safety_cost_change",
        ),
    ),
)
def test_gradient_joy_hard_magnitude_gates_certify_the_weighted_update(
    config_overrides: dict[str, float],
    objective_probe_scale: float,
    raw_field: str,
    tentative_field: str,
) -> None:
    """Soft scaling must not invalidate a hard improvement requirement."""
    candidate = {"weights": jnp.array([-0.11, 0.0], dtype=jnp.float32)}
    evidence = _gradient_joy_evidence(
        objective_probe_gradient={
            "weights": jnp.array([objective_probe_scale, 0.0], dtype=jnp.float32)
        },
    )
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=0.2,
        norm_temperature=0.1,
        **config_overrides,
    )

    result = assess_gradient_joy(candidate, evidence, config)

    assert float(result.diagnostics.tentative_weight) < 1.0
    if raw_field == "predicted_objective_decrease":
        assert float(getattr(result.diagnostics, raw_field)) > 0.1
        assert float(getattr(result.diagnostics, tentative_field)) < 0.1
    else:
        assert float(getattr(result.diagnostics, raw_field)) < -0.1
        assert float(getattr(result.diagnostics, tentative_field)) > -0.1
    assert not bool(result.accepted)
    chex.assert_trees_all_close(result.weight, jnp.array(0.0, dtype=jnp.float32))
    chex.assert_trees_all_close(
        result.weighted_update["weights"],
        jnp.zeros(2, dtype=jnp.float32),
    )


@pytest.mark.parametrize("protected_probe", ("retention", "safety"))
def test_gradient_joy_positive_harm_tolerance_checks_raw_and_weighted_update(
    protected_probe: str,
) -> None:
    candidate = {"weights": jnp.array([-0.11, 0.0], dtype=jnp.float32)}
    harmful_probe = {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)}
    evidence_kwargs = {
        "objective_probe_gradient": {
            "weights": jnp.array([10.0, 0.0], dtype=jnp.float32)
        },
        "retention_probe_gradient": {
            "weights": jnp.array([1.0, 0.0], dtype=jnp.float32)
        },
        "safety_cost_gradient": {
            "weights": jnp.array([1.0, 0.0], dtype=jnp.float32)
        },
    }
    evidence_kwargs[f"{protected_probe}_probe_gradient"] = harmful_probe
    if protected_probe == "safety":
        evidence_kwargs["safety_cost_gradient"] = evidence_kwargs.pop(
            "safety_probe_gradient"
        )
    config_kwargs = {f"max_{protected_probe}_loss_increase": 0.1}
    if protected_probe == "safety":
        config_kwargs = {"max_safety_cost_increase": 0.1}

    result = assess_gradient_joy(
        candidate,
        _gradient_joy_evidence(**evidence_kwargs),
        GradientJoyConfig(
            candidate_semantics="update",
            max_update_norm=0.2,
            norm_temperature=0.1,
            min_retention_descent_alignment=-1.0,
            min_safety_descent_alignment=-1.0,
            **config_kwargs,
        ),
    )

    raw_change = getattr(
        result.diagnostics,
        f"predicted_{protected_probe}_"
        + ("cost_change" if protected_probe == "safety" else "loss_change"),
    )
    tentative_change = getattr(
        result.diagnostics,
        f"predicted_tentative_{protected_probe}_"
        + ("cost_change" if protected_probe == "safety" else "loss_change"),
    )
    assert float(raw_change) > 0.1
    assert float(tentative_change) < 0.1
    assert not bool(result.accepted)


def test_gradient_joy_fails_closed_on_unavailable_or_nonfinite_evidence() -> None:
    config = GradientJoyConfig(
        candidate_semantics="update",
        max_update_norm=2.0,
    )
    candidate = {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)}
    unavailable = _gradient_joy_evidence().replace(
        objective_probe_gradient=None,
    )
    explicitly_unavailable = _gradient_joy_evidence().replace(
        retention_probe_available=jnp.array(False),
    )
    unattested = _gradient_joy_evidence(
        probe_independence_attested=False,
    )
    invalid_channel = _gradient_joy_evidence(
        learning_value=_scalar_learning_value(
            epistemic_surprise=jnp.array(jnp.nan, dtype=jnp.float32),
        )
    )
    finite_but_unavailable = _gradient_joy_evidence(
        learning_value_availability=LearningValueAvailability(
            advantage=jnp.array(True),
            action_surprisal=jnp.array(True),
            delight=jnp.array(True),
            epistemic_surprise=jnp.array(False),
            aleatoric_uncertainty=jnp.array(True),
            learning_progress=jnp.array(True),
            change_probability=jnp.array(True),
            safety_cost=jnp.array(True),
        )
    )

    unavailable_result = assess_gradient_joy(
        candidate,
        unavailable,
        config,
    )
    unattested_result = assess_gradient_joy(
        candidate,
        unattested,
        config,
    )
    explicitly_unavailable_result = assess_gradient_joy(
        candidate,
        explicitly_unavailable,
        config,
    )
    invalid_channel_result = assess_gradient_joy(
        candidate,
        invalid_channel,
        config,
    )
    unavailable_channel_result = assess_gradient_joy(
        candidate,
        finite_but_unavailable,
        config,
    )
    invalid_candidate_result = assess_gradient_joy(
        {"weights": jnp.array([jnp.nan, 0.0], dtype=jnp.float32)},
        _gradient_joy_evidence(),
        config,
    )

    assert not bool(unavailable_result.accepted)
    assert not bool(unavailable_result.diagnostics.objective_probe_available)
    assert not bool(unattested_result.accepted)
    assert not bool(unattested_result.diagnostics.probe_independence_attested)
    assert not bool(explicitly_unavailable_result.accepted)
    assert not bool(explicitly_unavailable_result.diagnostics.retention_probe_available)
    assert not bool(invalid_channel_result.accepted)
    assert not bool(invalid_channel_result.channel_availability.epistemic_surprise)
    assert not bool(invalid_channel_result.diagnostics.learning_value_complete)
    assert not bool(unavailable_channel_result.accepted)
    assert not bool(unavailable_channel_result.channel_availability.epistemic_surprise)
    assert not bool(unavailable_channel_result.diagnostics.learning_value_complete)
    assert not bool(invalid_candidate_result.accepted)
    assert not bool(invalid_candidate_result.diagnostics.candidate_finite)
    for result in (
        unavailable_result,
        unattested_result,
        explicitly_unavailable_result,
        invalid_channel_result,
        unavailable_channel_result,
        invalid_candidate_result,
    ):
        chex.assert_trees_all_close(
            result.weight,
            jnp.array(0.0, dtype=jnp.float32),
        )
        chex.assert_trees_all_close(
            result.weighted_update["weights"],
            jnp.zeros(2, dtype=jnp.float32),
        )


def test_gradient_joy_requires_exact_paper_dg_delight_identity() -> None:
    """A historical DG field mismatch cannot count as complete joy evidence."""
    mismatched = _scalar_learning_value(
        delight=jnp.nextafter(
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
    )

    result = assess_gradient_joy(
        {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)},
        _gradient_joy_evidence(learning_value=mismatched),
        GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
    )
    compiled = jax.jit(
        lambda value: assess_gradient_joy(
            {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)},
            _gradient_joy_evidence(learning_value=value),
            GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
        )
    )(mismatched)

    for assessment in (result, compiled):
        assert not bool(assessment.channel_availability.delight)
        assert not bool(assessment.diagnostics.learning_value_complete)
        assert not bool(assessment.sparks_joy)
        chex.assert_trees_all_close(
            assessment.weight,
            jnp.asarray(0.0, dtype=jnp.float32),
        )


@pytest.mark.parametrize(
    ("probe_field", "finite_diagnostic"),
    (
        ("objective_probe_gradient", "objective_probe_finite"),
        ("retention_probe_gradient", "retention_probe_finite"),
        ("safety_cost_gradient", "safety_probe_finite"),
    ),
)
@pytest.mark.parametrize("bad_value", (jnp.nan, jnp.inf, -jnp.inf))
def test_gradient_joy_nonfinite_probe_channels_fail_closed(
    probe_field: str,
    finite_diagnostic: str,
    bad_value: Array,
) -> None:
    evidence = _gradient_joy_evidence(
        **{
            probe_field: {
                "weights": jnp.array([bad_value, 0.0], dtype=jnp.float32),
            }
        }
    )
    result = assess_gradient_joy(
        {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)},
        evidence,
        GradientJoyConfig(candidate_semantics="update", max_update_norm=2.0),
    )

    assert not bool(result.accepted)
    assert not bool(getattr(result.diagnostics, finite_diagnostic))
    chex.assert_trees_all_close(result.weight, jnp.array(0.0, dtype=jnp.float32))
    chex.assert_trees_all_close(
        result.weighted_update["weights"],
        jnp.zeros(2, dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result)


def test_gradient_joy_rejects_ambiguous_shapes_and_non_scalar_channels() -> None:
    candidate = {"weights": jnp.array([-1.0, 0.0], dtype=jnp.float32)}
    with pytest.raises(ValueError, match="at least one floating value"):
        assess_gradient_joy(
            {"weights": jnp.empty((0,), dtype=jnp.float32)},
            _gradient_joy_evidence(),
        )
    with pytest.raises(ValueError, match="floating dtypes"):
        assess_gradient_joy(
            {"weights": jnp.array([-1.0 + 1.0j], dtype=jnp.complex64)},
            _gradient_joy_evidence(),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="leaf shapes"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                objective_probe_gradient={"weights": jnp.ones(3, dtype=jnp.float32)}
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="real-valued"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value=_scalar_learning_value(
                    learning_progress=jnp.array(1.0 + 1.0j, dtype=jnp.complex64),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="floating dtype"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value=_scalar_learning_value(
                    advantage=jnp.array(True),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="floating dtype"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value=_scalar_learning_value(
                    advantage=jnp.array(1, dtype=jnp.int32),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="must be scalar"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value=_scalar_learning_value(
                    learning_progress=jnp.ones(2, dtype=jnp.float32),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="must be scalar"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value=_scalar_learning_value(
                    learning_progress=jnp.array([0.3], dtype=jnp.float32),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="must be scalar"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence().replace(
                probe_independence_attested=jnp.array([True]),
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence().replace(
                objective_probe_available=jnp.array(jnp.nan),
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        assess_gradient_joy(
            candidate,
            _gradient_joy_evidence(
                learning_value_availability=LearningValueAvailability(
                    advantage=jnp.array(jnp.nan),
                    action_surprisal=jnp.array(True),
                    delight=jnp.array(True),
                    epistemic_surprise=jnp.array(True),
                    aleatoric_uncertainty=jnp.array(True),
                    learning_progress=jnp.array(True),
                    change_probability=jnp.array(True),
                    safety_cost=jnp.array(True),
                )
            ),
            GradientJoyConfig(candidate_semantics="update"),
        )


def test_paper_specific_dg_delight_equations_match_definition() -> None:
    log_probabilities = jnp.log(jnp.array([0.5, 0.1, 0.8], dtype=jnp.float32))
    advantages = jnp.array([2.0, -3.0, 0.5], dtype=jnp.float32)
    temperature = 0.7
    result = discrete_delightful_policy_gradient(
        log_probabilities,
        advantages,
        DelightfulPolicyGradientConfig(
            mode="delightful_pg",
            temperature=temperature,
        ),
    )

    expected_surprisal = -log_probabilities
    expected_delight = advantages * expected_surprisal
    expected_weights = jax.nn.sigmoid(expected_delight / temperature)
    expected_coefficients = expected_weights * advantages
    expected_loss = -jnp.mean(expected_coefficients * log_probabilities)
    expected_ordinary_loss = -jnp.mean(advantages * log_probabilities)

    chex.assert_trees_all_close(result.action_surprisal, expected_surprisal)
    chex.assert_trees_all_close(result.delight, expected_delight)
    chex.assert_trees_all_close(result.sample_weights, expected_weights)
    chex.assert_trees_all_close(result.actor_coefficients, expected_coefficients)
    chex.assert_trees_all_close(result.actor_loss, expected_loss)
    chex.assert_trees_all_close(result.ordinary_actor_loss, expected_ordinary_loss)


def test_paper_specific_dg_loss_jits_and_stops_gate_and_advantage_gradients() -> None:
    config = DelightfulPolicyGradientConfig(
        mode="delightful_pg",
        temperature=0.5,
    )
    log_probabilities = jnp.log(jnp.array([0.7, 0.2, 0.4, 0.9], dtype=jnp.float32))
    advantages = jnp.array([1.5, -0.75, 0.25, -2.0], dtype=jnp.float32)
    compiled = jax.jit(
        lambda log_prob, advantage: discrete_delightful_policy_gradient(
            log_prob,
            advantage,
            config,
        )
    )

    result = compiled(log_probabilities, advantages)
    log_probability_gradient = jax.grad(
        lambda log_prob: discrete_delightful_policy_gradient(
            log_prob,
            advantages,
            config,
        ).actor_loss.sum()
    )(log_probabilities)
    advantage_gradient = jax.grad(
        lambda advantage: (
            discrete_delightful_policy_gradient(
                log_probabilities,
                advantage,
                config,
            ).actor_loss
        )
    )(advantages)

    expected_log_probability_gradient = -result.sample_weights * advantages / advantages.size
    chex.assert_trees_all_close(
        log_probability_gradient,
        expected_log_probability_gradient,
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(advantage_gradient, jnp.zeros_like(advantages))
    chex.assert_tree_all_finite(result)


def test_ordinary_mode_is_the_matched_ungated_policy_gradient() -> None:
    log_probabilities = jnp.log(jnp.array([0.25, 0.75], dtype=jnp.float32))
    advantages = jnp.array([1.0, -2.0], dtype=jnp.float32)
    result = discrete_delightful_policy_gradient(
        log_probabilities,
        advantages,
        DelightfulPolicyGradientConfig(mode="ordinary_pg"),
    )

    expected_loss = -jnp.mean(advantages * log_probabilities)
    chex.assert_trees_all_close(result.sample_weights, jnp.ones_like(advantages))
    chex.assert_trees_all_close(result.actor_coefficients, advantages)
    chex.assert_trees_all_close(result.actor_loss, expected_loss)
    chex.assert_trees_all_close(result.ordinary_actor_loss, expected_loss)
    chex.assert_trees_all_close(
        result.diagnostics.effective_sample_size,
        jnp.array(2.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.effective_sample_fraction,
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.positive_advantage_gate_rate,
        jnp.array(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.diagnostics.negative_advantage_gate_rate,
        jnp.array(1.0, dtype=jnp.float32),
    )


def test_effective_sample_size_and_signed_gate_rates_are_auditable() -> None:
    log_probabilities = jnp.log(jnp.array([0.5, 0.25, 0.1, 0.8], dtype=jnp.float32))
    advantages = jnp.array([2.0, 1.0, -1.0, -3.0], dtype=jnp.float32)
    result = discrete_delightful_policy_gradient(
        log_probabilities,
        advantages,
        DelightfulPolicyGradientConfig(temperature=1.25),
    )
    weights = result.sample_weights
    expected_ess = jnp.sum(weights) ** 2 / jnp.sum(weights**2)

    chex.assert_trees_all_close(
        result.diagnostics.effective_sample_size,
        expected_ess,
    )
    chex.assert_trees_all_close(
        result.diagnostics.effective_sample_fraction,
        expected_ess / weights.size,
    )
    chex.assert_trees_all_close(
        result.diagnostics.positive_advantage_gate_rate,
        jnp.mean(weights[:2]),
    )
    chex.assert_trees_all_close(
        result.diagnostics.negative_advantage_gate_rate,
        jnp.mean(weights[2:]),
    )
    assert float(result.diagnostics.positive_advantage_gate_rate) > 0.5
    assert float(result.diagnostics.negative_advantage_gate_rate) < 0.5


def test_heteroskedastic_gambling_strata_expose_lucky_rare_action_pathology() -> None:
    """A negative-mean rare gamble flips positive after paper-specific DG weighting.

    This deterministic finite-population diagnostic is intentionally a
    pathology probe, not benchmark evidence. The common action has low-variance
    ±0.2 outcomes. The rare action has one lucky +10 outcome and nineteen -1
    failures, so its ordinary mean advantage is negative despite much higher
    variance. Its large action surprisal strongly retains the lucky success and
    suppresses the failures.
    """
    common_advantages = jnp.concatenate(
        (
            jnp.full((10,), 0.2, dtype=jnp.float32),
            jnp.full((10,), -0.2, dtype=jnp.float32),
        )
    )
    rare_advantages = jnp.concatenate(
        (
            jnp.array([10.0], dtype=jnp.float32),
            jnp.full((19,), -1.0, dtype=jnp.float32),
        )
    )
    advantages = jnp.concatenate((common_advantages, rare_advantages))
    log_probabilities = jnp.concatenate(
        (
            jnp.full((20,), jnp.log(0.95), dtype=jnp.float32),
            jnp.full((20,), jnp.log(0.05), dtype=jnp.float32),
        )
    )
    rare_mask = jnp.concatenate(
        (
            jnp.zeros((20,), dtype=jnp.bool_),
            jnp.ones((20,), dtype=jnp.bool_),
        )
    )
    result = discrete_delightful_policy_gradient(
        log_probabilities,
        advantages,
        DelightfulPolicyGradientConfig(temperature=1.0),
    )
    stratification = jax.jit(stratify_delight_outcomes)(result, rare_mask)

    assert int(stratification.common_success.count) == 10
    assert int(stratification.common_failure.count) == 10
    assert int(stratification.rare_success.count) == 1
    assert int(stratification.rare_failure.count) == 19
    assert float(stratification.rare_advantage_variance) > 100.0 * float(
        stratification.common_advantage_variance
    )
    assert float(stratification.rare_advantage_mean) < 0.0
    assert float(stratification.rare_gated_advantage_mean) > 0.0
    assert float(stratification.rare_success.mean_gate_rate) > float(
        stratification.common_success.mean_gate_rate
    )
    assert float(stratification.rare_failure.mean_gate_rate) < float(
        stratification.common_failure.mean_gate_rate
    )
    chex.assert_tree_all_finite(stratification)


def test_paper_specific_dg_rejects_shape_mismatch_and_empty_batches() -> None:
    with pytest.raises(ValueError, match="same shape"):
        discrete_delightful_policy_gradient(
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="at least one"):
        discrete_delightful_policy_gradient(
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        )
    result = discrete_delightful_policy_gradient(
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="rare_mask"):
        stratify_delight_outcomes(result, jnp.zeros((3,), dtype=jnp.bool_))
