"""Complete IAConditionResult identity contract: leftover, types, and arrays."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.continual_ia import (
    CONDITION_NAMES,
    ConditionTiming,
    ContinualIAConfig,
    ControllerBudget,
    IAConditionResult,
    aggregate_ia_evidence,
)


class StringSubclass(str):
    """Leftover string identity that must not cross the result boundary."""


def _budget() -> ControllerBudget:
    return ControllerBudget(
        state_scalars=1,
        state_bytes=8,
        observation_scalars=2,
        action_scalars_per_step=1,
        interaction_steps=2,
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
        "recommendations": np.full(steps, -1, dtype=np.int64),
        "partner_proposals": np.full(steps, -1, dtype=np.int64),
        "accepted_recommendations": np.zeros(steps, dtype=np.bool_),
        "mean_reward": 0.0,
        "phase_mean_rewards": np.zeros(1, dtype=np.float64),
        "recovery_lengths": np.zeros(0, dtype=np.int64),
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


def test_ia_condition_result_cross_binds_primitive_summaries() -> None:
    with pytest.raises(ValueError, match="mean_reward does not match"):
        _legal(rewards=np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="mismatch count does not match"):
        _legal(executed_action_credit_mismatches=1)
    with pytest.raises(ValueError, match="rate does not match"):
        _legal(changed_action_intervention_rate=0.5)
    with pytest.raises(ValueError, match="interaction_steps must match"):
        _legal(controller_budget=ControllerBudget(
            state_scalars=1,
            state_bytes=8,
            observation_scalars=2,
            action_scalars_per_step=1,
            interaction_steps=1,
            ia_attached=True,
        ))


def test_ia_condition_result_rejects_hostile_array_subclasses_without_hooks() -> None:
    class HostileArray(np.ndarray):
        calls = 0

        def __array_function__(self, *args: object, **kwargs: object) -> object:
            type(self).calls += 1
            raise AssertionError("hostile array hook")

    hostile = np.zeros(2, dtype=np.float64).view(HostileArray)
    HostileArray.calls = 0
    with pytest.raises(ValueError, match="rewards must be an exact numpy.ndarray"):
        _legal(rewards=hostile)
    assert HostileArray.calls == 0


def test_ia_condition_result_owns_read_only_array_snapshots() -> None:
    rewards = np.zeros(2, dtype=np.float64)
    result = _legal(rewards=rewards)

    rewards[0] = 1.0

    np.testing.assert_array_equal(result.rewards, np.zeros(2, dtype=np.float64))
    assert not result.rewards.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.rewards[0] = 1.0


def test_ia_condition_result_preserves_executed_and_credited_array_roles() -> None:
    executed = np.asarray([0, 1], dtype=np.int64)
    credited = np.asarray([1, 1], dtype=np.int64)
    result = _legal(
        executed_actions=executed,
        credited_actions=credited,
        executed_action_credit_mismatches=1,
    )
    np.testing.assert_array_equal(result.executed_actions, executed)
    np.testing.assert_array_equal(result.credited_actions, credited)


def test_ia_condition_result_rejects_nested_subclasses() -> None:
    class BudgetSubclass(ControllerBudget):
        pass

    class TimingSubclass(ConditionTiming):
        pass

    with pytest.raises(ValueError, match="controller_budget must be"):
        _legal(controller_budget=BudgetSubclass(**vars(_budget())))
    with pytest.raises(ValueError, match="timing must be"):
        _legal(timing=TimingSubclass(**vars(_timing())))


def test_ia_aggregation_revalidates_records_at_consumption() -> None:
    result = _legal()
    object.__setattr__(result, "mean_reward", 1.0)

    with pytest.raises(ValueError, match="mean_reward does not match"):
        aggregate_ia_evidence(
            [result],
            config=ContinualIAConfig(num_steps=2, phase_length=2, recovery_window=1),
        )


def test_ia_aggregation_revalidates_config_at_consumption() -> None:
    config = ContinualIAConfig(num_steps=2, phase_length=2, recovery_window=1)
    object.__setattr__(config, "phase_length", 0)

    with pytest.raises(ValueError, match="phase_length must be an integer"):
        aggregate_ia_evidence([_legal()], config=config)


def test_ia_aggregation_binds_recommendation_protocol() -> None:
    recommendation = {
        "condition": "observe_only",
        "recommendations": np.zeros(2, dtype=np.int64),
        "partner_proposals": np.zeros(2, dtype=np.int64),
        "nominal_recommendation_decisions": 2,
    }
    config = ContinualIAConfig(num_steps=2, phase_length=2, recovery_window=1)
    invalid_actions = _legal(
        **recommendation,
        executed_actions=np.ones(2, dtype=np.int64),
        credited_actions=np.ones(2, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="executed actions violate"):
        aggregate_ia_evidence([invalid_actions], config=config)
    invalid_count = _legal(**(recommendation | {"nominal_recommendation_decisions": 1}))
    with pytest.raises(ValueError, match="decisions must equal"):
        aggregate_ia_evidence([invalid_count], config=config)


def test_ia_aggregation_binds_live_component_budget() -> None:
    results = []
    for condition in CONDITION_NAMES:
        overrides: dict[str, object] = {"condition": condition}
        if condition in {"observe_only", "recommendation_p05", "accept_always"}:
            accepted = np.array([False, condition == "accept_always"], dtype=np.bool_)
            overrides.update(
                recommendations=np.zeros(2, dtype=np.int64),
                partner_proposals=np.zeros(2, dtype=np.int64),
                accepted_recommendations=accepted,
                nominal_recommendation_decisions=2,
                nominal_accepted_recommendations=2 if condition == "accept_always" else 0,
                executed_accepted_recommendations=int(np.count_nonzero(accepted)),
            )
        results.append(_legal(**overrides))
    with pytest.raises(ValueError, match="live configured components"):
        aggregate_ia_evidence(
            results,
            config=ContinualIAConfig(num_steps=2, phase_length=2, recovery_window=1),
        )


def test_ia_aggregation_fails_closed_on_a_schedule_with_no_phase_switch() -> None:
    """One phase has no switch to recover from; the aggregate must raise, not report NaN."""
    from alberta_framework.evaluation import continual_ia

    config = ContinualIAConfig(num_steps=2, phase_length=2, recovery_window=1)
    budgets = continual_ia._condition_budgets(
        config, continual_ia._make_partner(config), continual_ia._make_ia(config)
    )
    results = []
    for condition in CONDITION_NAMES:
        overrides: dict[str, object] = {
            "condition": condition,
            "controller_budget": budgets[condition],
        }
        if condition in {"observe_only", "recommendation_p05", "accept_always"}:
            accepted = np.array([False, condition == "accept_always"], dtype=np.bool_)
            overrides.update(
                recommendations=np.zeros(2, dtype=np.int64),
                partner_proposals=np.zeros(2, dtype=np.int64),
                accepted_recommendations=accepted,
                nominal_recommendation_decisions=2,
                nominal_accepted_recommendations=2 if condition == "accept_always" else 0,
                executed_accepted_recommendations=int(np.count_nonzero(accepted)),
            )
        results.append(_legal(**overrides))
    with pytest.raises(ValueError, match="recovery evidence is empty"):
        aggregate_ia_evidence(results, config=config)


def test_ia_result_preflights_retained_bytes_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alberta_framework.evaluation import continual_ia

    monkeypatch.setattr(continual_ia, "_MAX_CONDITION_RESULT_BYTES", 0)
    monkeypatch.setattr(
        continual_ia.np,
        "array",
        lambda *_args, **_kwargs: pytest.fail("snapshot allocation must not start"),
    )
    with pytest.raises(ValueError, match="bounded record budget"):
        _legal()


def test_ia_aggregation_rejects_hostile_outer_sequence_without_hooks() -> None:
    class HostileResults(list[IAConditionResult]):
        calls = 0

        def __len__(self) -> int:
            self.calls += 1
            raise AssertionError("must not size")

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    hostile = HostileResults()
    with pytest.raises(ValueError, match="exact list or tuple"):
        aggregate_ia_evidence(hostile, config=ContinualIAConfig())
    assert hostile.calls == 0
