"""Unit coverage for continual-multiagent evaluator policy probes."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.continual_multiagent import (
    ContinualMultiAgentConfig,
    _initial_controller,
    _probe_policy,
)
from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentWorld

pytestmark = pytest.mark.unit


def test_tied_probe_values_select_a_supported_controller_action() -> None:
    config = ContinualMultiAgentConfig(
        phase_steps=2,
        nuisance_dim=0,
        probe_horizon=1,
        probe_tail_steps=1,
        recovery_window=1,
    )
    world = RecurringTwoAgentWorld(
        context_length=config.phase_steps,
        nuisance_dim=config.nuisance_dim,
        damping=0.0,
        acceleration=0.25,
        max_speed=0.25,
    )
    observed_actions: list[np.ndarray] = []

    def recording_step(state, actions):  # type: ignore[no-untyped-def]
        observed_actions.append(np.asarray(actions))
        return world.step(state, actions)

    _probe_policy(
        world,
        recording_step,
        _initial_controller(seed=30),
        context=0,
        config=config,
    )

    np.testing.assert_array_equal(
        np.stack(observed_actions),
        -np.ones((config.probe_horizon, 2), dtype=np.float32),
    )
