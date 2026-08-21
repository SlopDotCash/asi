"""Regression coverage for #2220: WorkingMemoryConfig decay-rate tuples must
have a bounded cardinality before per-item validation walks.

Oversized or hostile sequences previously forced unbounded Python iteration
in _validate_decay_rates and were not capped in from_config.
"""

import pytest

from alberta_framework.core.working_memory import (
    _MAX_WORKING_MEMORY_DECAY_RATES,
    WorkingMemoryConfig,
)


def test_oversized_decay_rates_rejected() -> None:
    with pytest.raises(ValueError, match="at most"):
        WorkingMemoryConfig(
            observation_dim=4,
            observation_decay_rates=(0.5,) * (_MAX_WORKING_MEMORY_DECAY_RATES + 1),
        )


def test_oversized_from_config_rejected() -> None:
    payload = {
        "type": "WorkingMemoryConfig",
        "observation_dim": 4,
        "action_dim": 0,
        "reward_dim": 1,
        "observation_decay_rates": [0.5] * (_MAX_WORKING_MEMORY_DECAY_RATES + 1),
        "action_decay_rates": [0.5, 0.9],
        "reward_decay_rates": [0.5, 0.9],
        "include_current_observation": True,
        "include_current_action": True,
        "include_current_reward": True,
        "include_traces": True,
        "include_innovations": True,
        "gated_update": False,
        "gate_threshold": 0.5,
        "gate_temperature": 0.1,
    }
    with pytest.raises(ValueError, match="at most"):
        WorkingMemoryConfig.from_config(payload)


def test_boundary_exact_max_allowed() -> None:
    cfg = WorkingMemoryConfig(
        observation_dim=4,
        observation_decay_rates=(0.5,) * _MAX_WORKING_MEMORY_DECAY_RATES,
    )
    assert len(cfg.observation_decay_rates) == _MAX_WORKING_MEMORY_DECAY_RATES


def test_normal_config_unchanged() -> None:
    cfg = WorkingMemoryConfig(observation_dim=4)
    assert cfg.observation_decay_rates == (0.5, 0.9, 0.99)
