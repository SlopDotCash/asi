"""Cardinality preflights for working-memory and fixed-trace decay rates."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.core.state_builder import (
    _MAX_FIXED_TRACE_DECAY_RATES,
    FixedTraceStateBuilderConfig,
)
from alberta_framework.core.working_memory import (
    _MAX_WORKING_MEMORY_DECAY_RATES,
    WorkingMemoryConfig,
)


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("list length hook executed")


def test_documented_protocol_ceilings() -> None:
    assert _MAX_WORKING_MEMORY_DECAY_RATES == 4096
    assert _MAX_FIXED_TRACE_DECAY_RATES == 4096


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "reward_decay_rates",
    ],
)
def test_working_memory_last_fit_decay_count_is_accepted(field: str) -> None:
    kwargs: dict[str, Any] = {
        "observation_dim": 1,
        "action_dim": 0,
        "reward_dim": 0,
        "observation_decay_rates": (),
        "action_decay_rates": (),
        "reward_decay_rates": (),
        "include_current_action": False,
        "include_current_reward": False,
        "include_traces": True,
        field: (0.5,) * _MAX_WORKING_MEMORY_DECAY_RATES,
    }
    if field == "action_decay_rates":
        kwargs["action_dim"] = 1
    elif field == "reward_decay_rates":
        kwargs["reward_dim"] = 1

    cfg = WorkingMemoryConfig(**kwargs)
    assert len(getattr(cfg, field)) == _MAX_WORKING_MEMORY_DECAY_RATES


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "reward_decay_rates",
    ],
)
def test_working_memory_rejects_oversized_decay_rates_before_per_rate_walk(field: str) -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized decay tuple walked an element")

    hostile: Any = HostileFloat()
    kwargs: dict[str, Any] = {
        "observation_dim": 1,
        "action_dim": 1,
        "reward_dim": 1,
        field: (hostile,) * (_MAX_WORKING_MEMORY_DECAY_RATES + 1),
    }
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig(**kwargs)
    assert calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "reward_decay_rates",
    ],
)
def test_working_memory_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str,
) -> None:
    config = WorkingMemoryConfig(observation_dim=1, action_dim=1, reward_dim=1).to_config()
    config[field] = [0.5] * (_MAX_WORKING_MEMORY_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "reward_decay_rates",
    ],
)
def test_working_memory_from_config_rejects_list_subclasses_before_length_hooks(
    field: str,
) -> None:
    config = WorkingMemoryConfig(observation_dim=1, action_dim=1, reward_dim=1).to_config()
    config[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual list or tuple"):
        WorkingMemoryConfig.from_config(config)
    assert _HostileList.calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "outcome_decay_rates",
    ],
)
def test_fixed_trace_last_fit_decay_count_is_accepted(field: str) -> None:
    kwargs: dict[str, Any] = {
        "observation_dim": 1,
        "n_actions": 1 if field == "action_decay_rates" else 0,
        "observation_decay_rates": (),
        "action_decay_rates": (),
        "outcome_decay_rates": (),
        "include_raw_observation": False,
        field: (0.5,) * _MAX_FIXED_TRACE_DECAY_RATES,
    }
    cfg = FixedTraceStateBuilderConfig(**kwargs)
    assert len(getattr(cfg, field)) == _MAX_FIXED_TRACE_DECAY_RATES


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "outcome_decay_rates",
    ],
)
def test_fixed_trace_rejects_oversized_decay_rates_before_per_rate_walk(field: str) -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized decay tuple walked an element")

    hostile: Any = HostileFloat()
    kwargs: dict[str, Any] = {
        "observation_dim": 1,
        "n_actions": 1,
        field: (hostile,) * (_MAX_FIXED_TRACE_DECAY_RATES + 1),
    }
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig(**kwargs)
    assert calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "outcome_decay_rates",
    ],
)
def test_fixed_trace_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str,
) -> None:
    config = FixedTraceStateBuilderConfig(observation_dim=1, n_actions=1).to_config()
    config[field] = [0.5] * (_MAX_FIXED_TRACE_DECAY_RATES + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    [
        "observation_decay_rates",
        "action_decay_rates",
        "outcome_decay_rates",
    ],
)
def test_fixed_trace_from_config_rejects_list_subclasses_before_length_hooks(
    field: str,
) -> None:
    config = FixedTraceStateBuilderConfig(observation_dim=1, n_actions=1).to_config()
    config[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="decay rates must be lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(config)
    assert _HostileList.calls == 0
