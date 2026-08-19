"""Hostile string gates for compositional feature learner modes."""

from __future__ import annotations

import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner

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


def test_promotion_output_mode_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("scaled_candidate")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="promotion_output_mode"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            promotion_output_mode=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0


def test_generation_strategy_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("utility")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="generation_strategy"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            generation_strategy=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0


def test_future_utility_trace_mode_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("contribution")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="future_utility_trace_mode"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            future_utility_trace_mode=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0


def test_future_utility_normalization_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("none")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="future_utility_normalization"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            future_utility_normalization=hostile,  # type: ignore[arg-type]
        )
    assert _HostileStr.calls == 0


def test_benign_modes_pass() -> None:
    learner = CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        promotion_output_mode="blend",
        generation_strategy="uniform",
        future_utility_trace_mode="marginal",
        future_utility_normalization="age",
    )
    assert learner is not None
