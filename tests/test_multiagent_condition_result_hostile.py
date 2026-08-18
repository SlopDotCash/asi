"""Complete ConditionResult identity contract: leftover, types, and arrays."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.continual_multiagent import (
    ConditionResult,
    ControllerBudget,
    TimingMetrics,
)
from alberta_framework.utils.metrics import ContinualLearningSummary


class StringSubclass(str):
    """Leftover string identity that must not cross the result boundary."""


def _summary() -> ContinualLearningSummary:
    return ContinualLearningSummary(
        final_performance=0.0,
        prequential_performance=0.0,
        mean_forgetting=0.0,
        max_forgetting=0.0,
        backward_transfer=0.0,
        stability_gap_mean=0.0,
        stability_gap_max=0.0,
        per_task_final_performance=np.zeros(2, dtype=np.float64),
        per_task_forgetting=np.zeros(2, dtype=np.float64),
        per_task_backward_transfer=np.zeros(2, dtype=np.float64),
    )


def _legal(**overrides: object) -> ConditionResult:
    payload: dict[str, object] = {
        "seed": 30,
        "condition": "frozen",
        "learning_mask": (False, False),
        "online_rewards": np.zeros(4, dtype=np.float64),
        "phase_mean_rewards": np.zeros(3, dtype=np.float64),
        "performance_matrix": np.zeros((3, 2), dtype=np.float64),
        "summary": _summary(),
        "recovery_lengths": np.asarray([-1, -1], dtype=np.int64),
        "recurrence_recovery_steps": -1,
        "interference_forgetting": 0.0,
        "controller_budget": ControllerBudget(
            state_scalars=1, state_bytes=8, action_scalars_per_step=1
        ),
        "timing": TimingMetrics(
            wall_seconds=0.1,
            mean_step_latency_ms=0.1,
            mean_update_latency_ms=0.1,
            p95_update_latency_ms=0.1,
        ),
    }
    payload.update(overrides)
    return ConditionResult(**payload)  # type: ignore[arg-type]


def test_condition_result_accepts_canonical_identity() -> None:
    result = _legal()
    assert result.seed == 30
    assert result.condition == "frozen"
    assert result.learning_mask == (False, False)
    assert result.recurrence_recovery_steps == -1


def test_condition_result_rejects_leftover_integer_and_float_identities() -> None:
    with pytest.raises(ValueError, match="seeds must lie in"):
        _legal(seed=True)
    with pytest.raises(ValueError, match="recurrence_recovery_steps must be an integer"):
        _legal(recurrence_recovery_steps=True)
    with pytest.raises(ValueError, match="interference_forgetting must be a finite"):
        _legal(interference_forgetting=True)


def test_condition_result_rejects_leftover_condition_and_mask_identities() -> None:
    with pytest.raises(ValueError, match="condition must be a known multiagent condition name"):
        _legal(condition=True)
    with pytest.raises(ValueError, match="condition must be a known multiagent condition name"):
        _legal(condition=StringSubclass("frozen"))
    with pytest.raises(ValueError, match="learning_mask must be a pair of booleans"):
        _legal(learning_mask=(1, 0))
    with pytest.raises(ValueError, match="learning_mask must be a pair of booleans"):
        _legal(learning_mask=[False, False])


def test_condition_result_rejects_invalid_arrays_and_hosts() -> None:
    with pytest.raises(ValueError, match="online_rewards must be an exact numpy.ndarray"):
        _legal(online_rewards=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="performance_matrix must be 2-dimensional"):
        _legal(performance_matrix=np.zeros(3, dtype=np.float64))
    with pytest.raises(ValueError, match="summary must be a ContinualLearningSummary"):
        _legal(summary=None)
    with pytest.raises(ValueError, match="controller_budget must be a ControllerBudget"):
        _legal(controller_budget=None)
    with pytest.raises(ValueError, match="timing must be a TimingMetrics"):
        _legal(timing=None)
