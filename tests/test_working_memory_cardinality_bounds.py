"""Cardinality bounds for working-memory decay rates (#2220).

Oversized decay-rate tuples (> 4096) must raise ValueError at config
validation before per-rate iteration, and from_config must reject
oversized or hostile list subclasses before element copy.
"""

from __future__ import annotations

import pytest

from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    _MAX_WORKING_MEMORY_DECAY_RATES,
    _validate_decay_rates,
)


class _HostileList(list):
    calls = 0

    def __len__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile __len__ must not be called")

    def __iter__(self):  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile __iter__ must not be called")


def test_max_constant_is_4096() -> None:
    assert _MAX_WORKING_MEMORY_DECAY_RATES == 4096


def test_oversized_tuple_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"observation_decay_rates must contain at most 4096 decay rates",
    ):
        _validate_decay_rates(
            "observation_decay_rates", tuple([0.5] * (4096 + 1))
        )


def test_boundary_accepted() -> None:
    rates = _validate_decay_rates(
        "observation_decay_rates", tuple([0.5] * 4096)
    )
    assert len(rates) == 4096


def test_from_config_rejects_oversized_serialized_list() -> None:
    with pytest.raises(
        ValueError,
        match=r"serialized reward_decay_rates must contain at most 4096 decay rates",
    ):
        WorkingMemoryConfig.from_config(
            {
                "type": "WorkingMemoryConfig",
                "observation_decay_rates": [0.5, 0.9, 0.99],
                "action_decay_rates": [0.5, 0.9],
                "reward_decay_rates": [0.5] * (4096 + 1),
                "observation_dim": 4,
                "action_dim": 2,
                "reward_dim": 1,
            }
        )


def test_from_config_rejects_hostile_list_subclass() -> None:
    hostile = _HostileList([0.5, 0.9])
    with pytest.raises(ValueError, match="must be an actual list or tuple"):
        WorkingMemoryConfig.from_config(
            {
                "type": "WorkingMemoryConfig",
                "observation_decay_rates": hostile,
                "action_decay_rates": [0.5, 0.9],
                "reward_decay_rates": [0.5, 0.9],
                "observation_dim": 4,
                "action_dim": 2,
                "reward_dim": 1,
            }
        )
