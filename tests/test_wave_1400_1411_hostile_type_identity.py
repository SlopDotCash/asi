"""Regression checks for exact-type gates audited after PRs 1400--1411."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from alberta_framework.core.behavior_model import selected_action_probabilities
from alberta_framework.core.ftl_world_model import SparseFTLWorldModelConfig
from alberta_framework.core.history_features import (
    HistoryFeatureExtractor,
    HistoryFeatureState,
)
from alberta_framework.core.model_replay_rehearsal import _require_int
from alberta_framework.core.off_policy_horde import _require_float32, _require_int32
from alberta_framework.evaluation._measurement_validation import real_number
from alberta_framework.evaluation.continual_ia import IAAcceptanceEvidence
from alberta_framework.evaluation.continual_multiagent import AcceptanceEvidence
from alberta_framework.streams.pavlovian import PavlovianPhase
from alberta_framework.utils.timing import format_duration


class _HostileType(type):
    calls = 0

    def __hash__(cls) -> int:  # pragma: no cover - must not execute
        type(cls).calls += 1
        raise AssertionError("metaclass hash hook must not run")

    def __eq__(cls, other: object) -> bool:  # pragma: no cover - must not execute
        type(cls).calls += 1
        raise AssertionError("metaclass equality hook must not run")


class _HostileValue(metaclass=_HostileType):
    @property
    def __class__(self) -> type:  # pragma: no cover - must not execute
        type(type(self)).calls += 1
        raise AssertionError("instance class hook must not run")

    def __array__(self, *args: object, **kwargs: object) -> Any:  # pragma: no cover
        type(type(self)).calls += 1
        raise AssertionError("array conversion hook must not run")

    def __float__(self) -> float:  # pragma: no cover - must not execute
        type(type(self)).calls += 1
        raise AssertionError("float conversion hook must not run")

    def __index__(self) -> int:  # pragma: no cover - must not execute
        type(type(self)).calls += 1
        raise AssertionError("index conversion hook must not run")


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _HostileType.calls = 0


def test_changed_scalar_gates_do_not_dispatch_hostile_metaclass_hooks() -> None:
    hostile = _HostileValue()
    checks = (
        lambda: HistoryFeatureExtractor(raw_dim=hostile),
        lambda: SparseFTLWorldModelConfig(observation_dim=hostile, action_dim=1),
        lambda: _require_int("value", hostile, minimum=0),
        lambda: _require_int32("value", hostile, minimum=0),
        lambda: _require_float32("value", hostile),
        lambda: PavlovianPhase("phase", 1, hostile, (0,)),
        lambda: format_duration(hostile),
        lambda: real_number("value", hostile),
    )
    for check in checks:
        with pytest.raises((TypeError, ValueError)):
            check()
    assert _HostileType.calls == 0


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_acceptance_evidence_cannot_claim_success(actual: float) -> None:
    with pytest.raises(ValueError, match="passed check must have a finite actual"):
        AcceptanceEvidence("check", True, actual, ">=", 0.0, "detail")
    with pytest.raises(ValueError, match="passed check must have a finite actual"):
        IAAcceptanceEvidence("check", "primary", True, actual, ">=", 0.0, "detail")

    assert not AcceptanceEvidence("check", False, actual, ">=", 0.0, "detail").passed
    assert not IAAcceptanceEvidence(
        "check", "primary", False, actual, ">=", 0.0, "detail"
    ).passed


def test_behavior_action_gate_does_not_hash_an_untrusted_runtime_type() -> None:
    with pytest.raises(TypeError, match="trusted array"):
        selected_action_probabilities(
            jnp.asarray([0.25, 0.75], dtype=jnp.float32),
            _HostileValue(),
        )
    assert _HostileType.calls == 0


def test_history_state_gate_precedes_hostile_array_metadata() -> None:
    extractor = HistoryFeatureExtractor(raw_dim=1)
    state = HistoryFeatureState(traces=_HostileValue())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state.traces must be a JAX array"):
        extractor.step(state, jnp.ones((1,), dtype=jnp.float32))
    assert _HostileType.calls == 0
