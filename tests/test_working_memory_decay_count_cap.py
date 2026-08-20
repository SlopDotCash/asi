"""Reject oversized working-memory decay lists before per-rate validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.working_memory import (
    _MAX_WORKING_MEMORY_DECAY_RATES,
    WorkingMemoryConfig,
)


def test_working_memory_decay_cap_constant() -> None:
    assert _MAX_WORKING_MEMORY_DECAY_RATES == 4096


def test_working_memory_accepts_max_observation_decay_count() -> None:
    WorkingMemoryConfig(
        observation_dim=1,
        observation_decay_rates=(0.5,) * _MAX_WORKING_MEMORY_DECAY_RATES,
        action_decay_rates=(),
        reward_decay_rates=(),
    )


def test_working_memory_rejects_oversized_observation_decay_count() -> None:
    with pytest.raises(ValueError, match="observation_decay_rates length"):
        WorkingMemoryConfig(
            observation_dim=1,
            observation_decay_rates=(0.5,) * (_MAX_WORKING_MEMORY_DECAY_RATES + 1),
            action_decay_rates=(),
            reward_decay_rates=(),
        )


def test_working_memory_from_config_rejects_oversized_decay_list() -> None:
    payload = WorkingMemoryConfig(observation_dim=1).to_config()
    payload["action_decay_rates"] = [0.5] * (_MAX_WORKING_MEMORY_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="action_decay_rates length"):
        WorkingMemoryConfig.from_config(payload)
