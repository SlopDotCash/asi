"""FTL MPC enumeration rejects oversized horizons before 4**H hang."""

from __future__ import annotations

import time

import numpy as np
import pytest

from alberta_framework.benchmarks.ftl_online_agent_development import (
    MAX_PLANNING_HORIZON,
    _mpc_action,
)

pytestmark = pytest.mark.unit


def _identity_predict(observation: np.ndarray, action: int) -> np.ndarray:
    return observation


def test_mpc_rejects_unbounded_horizon_before_product_hang() -> None:
    observation = np.zeros(2, dtype=np.float32)
    goal = np.ones(2, dtype=np.float32)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="planning_horizon"):
        _mpc_action(observation, goal, MAX_PLANNING_HORIZON + 6, _identity_predict)
    assert time.perf_counter() - started < 0.25


def test_mpc_enumerates_the_public_horizon_bound() -> None:
    observation = np.zeros(2, dtype=np.float32)
    goal = np.ones(2, dtype=np.float32)
    action, queries, candidates = _mpc_action(
        observation, goal, MAX_PLANNING_HORIZON, _identity_predict
    )
    assert action == 0
    assert candidates == 4**MAX_PLANNING_HORIZON
    assert queries == MAX_PLANNING_HORIZON * candidates
