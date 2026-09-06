"""Cardinality preflights for working-memory and fixed-trace decay rates.

These guard the same class of unbounded/hostile serialized-sequence input that
``HistoryFeatureExtractor`` (``_MAX_HISTORY_CONFIGURATION_ITEMS``) and
``HordeSpec`` (``_MAX_HORDE_DEMONS``) already cap: ``WorkingMemoryConfig`` and
``FixedTraceStateBuilderConfig`` must reject oversized decay-rate sequences and
list subclasses *before* iterating their elements.
"""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.core.state_builder import (
    _MAX_FIXED_TRACE_DECAY_RATES,
    FixedTraceStateBuilderConfig,
)
from alberta_framework.core.working_memory import (
    _MAX_WORKING_MEMORY_CONFIGURATION_ITEMS,
    _MAX_WORKING_MEMORY_DECAY_RATES,
    WorkingMemoryConfig,
)


class _HostileList(list[object]):
    """A list subclass whose length/iteration hooks must never be reached."""

    calls = 0

    def __len__(self) -> int:  # pragma: no cover - must not execute
        type(self).calls += 1
        raise AssertionError("list length hook executed")

    def __iter__(self) -> Any:  # pragma: no cover - must not execute
        type(self).calls += 1
        raise AssertionError("list iteration hook executed")


class _HostileFloat:
    """A decay-rate element whose numeric coercion must never be reached."""

    calls = 0

    def __float__(self) -> float:  # pragma: no cover - must not execute
        type(self).calls += 1
        raise AssertionError("oversized decay sequence walked an element")


def test_constants_equal_4096() -> None:
    assert _MAX_WORKING_MEMORY_CONFIGURATION_ITEMS == 4096
    assert _MAX_WORKING_MEMORY_DECAY_RATES == 4096
    assert _MAX_FIXED_TRACE_DECAY_RATES == 4096


def test_working_memory_rejects_oversized_decay_tuple_before_per_rate_walk() -> None:
    _HostileFloat.calls = 0
    hostile: Any = tuple(_HostileFloat() for _ in range(1))
    oversized: Any = hostile * (_MAX_WORKING_MEMORY_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig(observation_dim=1, observation_decay_rates=oversized)
    assert _HostileFloat.calls == 0


def test_working_memory_boundary_decay_count_is_accepted() -> None:
    rates = tuple(0.5 for _ in range(_MAX_WORKING_MEMORY_DECAY_RATES))
    config = WorkingMemoryConfig(observation_dim=1, observation_decay_rates=rates)
    assert len(config.observation_decay_rates) == _MAX_WORKING_MEMORY_DECAY_RATES


def test_working_memory_from_config_rejects_oversized_serialized_list() -> None:
    payload = WorkingMemoryConfig(observation_dim=1).to_config()
    payload["observation_decay_rates"] = [0.5] * (_MAX_WORKING_MEMORY_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig.from_config(payload)


def test_working_memory_from_config_rejects_oversized_serialized_tuple() -> None:
    payload = WorkingMemoryConfig(observation_dim=1).to_config()
    payload["reward_decay_rates"] = tuple(0.5 for _ in range(_MAX_WORKING_MEMORY_DECAY_RATES + 1))
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig.from_config(payload)


def test_working_memory_from_config_boundary_serialized_list_is_accepted() -> None:
    payload = WorkingMemoryConfig(observation_dim=1).to_config()
    payload["observation_decay_rates"] = [0.5] * _MAX_WORKING_MEMORY_DECAY_RATES
    config = WorkingMemoryConfig.from_config(payload)
    assert len(config.observation_decay_rates) == _MAX_WORKING_MEMORY_DECAY_RATES


def test_working_memory_from_config_rejects_list_subclass_before_hooks() -> None:
    payload = WorkingMemoryConfig(observation_dim=1).to_config()
    payload["action_decay_rates"] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual list or tuple"):
        WorkingMemoryConfig.from_config(payload)
    assert _HostileList.calls == 0


def test_fixed_trace_rejects_oversized_decay_tuple_before_per_rate_walk() -> None:
    _HostileFloat.calls = 0
    oversized: Any = tuple(_HostileFloat() for _ in range(1)) * (
        _MAX_FIXED_TRACE_DECAY_RATES + 1
    )
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig(observation_dim=1, observation_decay_rates=oversized)
    assert _HostileFloat.calls == 0


def test_fixed_trace_boundary_decay_count_is_accepted() -> None:
    rates = tuple(0.5 for _ in range(_MAX_FIXED_TRACE_DECAY_RATES))
    config = FixedTraceStateBuilderConfig(observation_dim=1, observation_decay_rates=rates)
    assert len(config.observation_decay_rates) == _MAX_FIXED_TRACE_DECAY_RATES


def test_fixed_trace_from_config_rejects_oversized_serialized_list() -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    payload["observation_decay_rates"] = [0.5] * (_MAX_FIXED_TRACE_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig.from_config(payload)


def test_fixed_trace_from_config_boundary_serialized_list_is_accepted() -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    payload["observation_decay_rates"] = [0.5] * _MAX_FIXED_TRACE_DECAY_RATES
    config = FixedTraceStateBuilderConfig.from_config(payload)
    assert len(config.observation_decay_rates) == _MAX_FIXED_TRACE_DECAY_RATES


def test_fixed_trace_from_config_rejects_list_subclass_before_hooks() -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    payload["action_decay_rates"] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(payload)
    assert _HostileList.calls == 0
