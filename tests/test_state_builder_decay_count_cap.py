"""Reject oversized fixed-trace decay lists before per-rate validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.state_builder import (
    _MAX_STATE_BUILDER_DECAY_RATES,
    FixedTraceStateBuilderConfig,
)


def test_state_builder_decay_cap_constant() -> None:
    assert _MAX_STATE_BUILDER_DECAY_RATES == 4096


def test_fixed_trace_accepts_max_observation_decay_count() -> None:
    FixedTraceStateBuilderConfig(
        observation_dim=1,
        observation_decay_rates=(0.5,) * _MAX_STATE_BUILDER_DECAY_RATES,
        action_decay_rates=(),
        outcome_decay_rates=(),
    )


def test_fixed_trace_rejects_oversized_observation_decay_count() -> None:
    with pytest.raises(ValueError, match="observation_decay_rates length"):
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(0.5,) * (_MAX_STATE_BUILDER_DECAY_RATES + 1),
            action_decay_rates=(),
            outcome_decay_rates=(),
        )


def test_fixed_trace_from_config_rejects_oversized_decay_list() -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    payload["action_decay_rates"] = [0.5] * (_MAX_STATE_BUILDER_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="decay-rate length"):
        FixedTraceStateBuilderConfig.from_config(payload)
