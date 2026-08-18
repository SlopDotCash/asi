"""Complete IAConditionResult identity contract: leftover, types, and arrays."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.continual_ia import (
    CONDITION_NAMES,
    ConditionTiming,
    ControllerBudget,
    IAConditionResult,
)


class StringSubclass(str):
    """Leftover string identity that must not cross the result boundary."""


def _budget() -> ControllerBudget:
    return ControllerBudget(
        state_scalars=1,
        state_bytes=8,
        observation_scalars=2,
        action_scalars_per_step=1,
        interaction_steps=1,
        ia_attached=True,
    )


def _timing() -> ConditionTiming:
    return ConditionTiming(wall_seconds=0.1, mean_step_latency_ms=0.1)


def _legal(**overrides: object) -> IAConditionResult:
    steps = 2
    payload: dict[str, object] = {
        "seed": 30,
        "condition": CONDITION_NAMES[0],
        "rewards": np.zeros(steps, dtype=np.float64),
        "executed_actions": np.zeros(steps, dtype=np.int64),
        "credited_actions": np.zeros(steps, dtype=np.int64),
        "recommendations": np.zeros(steps, dtype=np.int64),
        "partner_proposals": np.zeros(steps, dtype=np.int64),
        "accepted_recommendations": np.zeros(steps, dtype=np.bool_),
        "mean_reward": 0.0,
        "phase_mean_rewards": np.zeros(1, dtype=np.float64),
        "recovery_lengths": np.zeros(1, dtype=np.int64),
        "nominal_recommendation_decisions": 0,
        "nominal_accepted_recommendations": 0,
        "executed_accepted_recommendations": 0,
        "action_changing_interventions": 0,
        "changed_action_intervention_rate": 0.0,
        "executed_action_credit_mismatches": 0,
        "controller_budget": _budget(),
        "timing": _timing(),
    }
    payload.update(overrides)
    return IAConditionResult(**payload)  # type: ignore[arg-type]


def test_ia_condition_result_accepts_canonical_identity() -> None:
    result = _legal()
    assert result.seed == 30
    assert result.condition == "partner_alone"
    assert result.changed_action_intervention_rate == 0.0


def test_ia_condition_result_rejects_leftover_integer_identities() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        _legal(seed=True)
    with pytest.raises(ValueError, match="action_changing_interventions must be an integer"):
        _legal(action_changing_interventions=True)
    with pytest.raises(ValueError, match="executed_action_credit_mismatches must be an integer"):
        _legal(executed_action_credit_mismatches=True)


def test_ia_condition_result_rejects_leftover_float_identities() -> None:
    with pytest.raises(ValueError, match="mean_reward must be a finite real number"):
        _legal(mean_reward=True)
    with pytest.raises(ValueError, match="changed_action_intervention_rate must be a finite"):
        _legal(changed_action_intervention_rate=True)
    with pytest.raises(ValueError, match="changed_action_intervention_rate must lie in"):
        _legal(changed_action_intervention_rate=1.5)


def test_ia_condition_result_rejects_leftover_condition_identities() -> None:
    with pytest.raises(ValueError, match="condition must be a known IA condition name"):
        _legal(condition=True)
    with pytest.raises(ValueError, match="condition must be a known IA condition name"):
        _legal(condition=StringSubclass("partner_alone"))
    with pytest.raises(ValueError, match="condition must be a known IA condition name"):
        _legal(condition="not_a_condition")


def test_ia_condition_result_rejects_invalid_arrays_and_hosts() -> None:
    with pytest.raises(ValueError, match="rewards must be an exact numpy.ndarray"):
        _legal(rewards=[0.0, 0.0])
    with pytest.raises(ValueError, match="executed_actions must have dtype"):
        _legal(executed_actions=np.zeros(2, dtype=np.int32))
    with pytest.raises(ValueError, match="accepted_recommendations must have length"):
        _legal(accepted_recommendations=np.zeros(1, dtype=np.bool_))
    with pytest.raises(ValueError, match="controller_budget must be a ControllerBudget"):
        _legal(controller_budget=None)
    with pytest.raises(ValueError, match="timing must be a ConditionTiming"):
        _legal(timing=None)
