"""Hostile string gates for interaction feature discovery candidate params."""

from __future__ import annotations

import pytest

from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_candidate_strategy_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("random")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="candidate_strategy"):
        FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            candidate_strategy=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0
    # benign still works
    learner = FixedBudgetInteractionLearner(
        n_features=2, n_tasks=1, candidate_strategy="random"
    )
    assert learner is not None


def test_utility_aggregation_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("mean")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="utility_aggregation"):
        FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            utility_aggregation=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0
    learner = FixedBudgetInteractionLearner(
        n_features=2, n_tasks=1, utility_aggregation="mean"
    )
    assert learner is not None


def test_utility_task_balancing_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("none")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="utility_task_balancing"):
        FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            utility_task_balancing=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0
    learner = FixedBudgetInteractionLearner(
        n_features=2, n_tasks=1, utility_task_balancing="none"
    )
    assert learner is not None


def test_relevance_probe_mode_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("conditional_v1")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="relevance_probe_mode"):
        FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            relevance_probe_mode=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0
    learner = FixedBudgetInteractionLearner(
        n_features=2, n_tasks=1, relevance_probe_mode="conditional_v1"
    )
    assert learner is not None


def test_benign_all_params_pass() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        candidate_strategy="all_pairs",
        utility_aggregation="max",
        utility_task_balancing="active",
        relevance_probe_mode="target_only_v1",
    )
    assert learner is not None


