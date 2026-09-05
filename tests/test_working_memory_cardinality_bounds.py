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

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("list iterator hook executed")


def test_last_fit_working_memory_decay_count_is_accepted() -> None:
    rates = (0.5,) * _MAX_WORKING_MEMORY_DECAY_RATES
    cfg = WorkingMemoryConfig(
        observation_dim=1,
        action_dim=0,
        reward_dim=0,
        observation_decay_rates=rates,
        action_decay_rates=(),
        reward_decay_rates=(),
        include_current_action=False,
        include_current_reward=False,
        include_traces=False,
    )
    assert len(cfg.observation_decay_rates) == _MAX_WORKING_MEMORY_DECAY_RATES


def test_last_fit_fixed_trace_decay_count_is_accepted() -> None:
    rates = (0.5,) * _MAX_FIXED_TRACE_DECAY_RATES
    cfg = FixedTraceStateBuilderConfig(
        observation_dim=1,
        n_actions=0,
        observation_decay_rates=rates,
        action_decay_rates=(),
        outcome_decay_rates=(),
        include_raw_observation=True,
    )
    assert len(cfg.observation_decay_rates) == _MAX_FIXED_TRACE_DECAY_RATES


def test_rejects_oversized_working_memory_decay_rates_before_per_rate_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized decay tuple walked an element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig(
            observation_dim=1,
            observation_decay_rates=(hostile,) * 4097,  # type: ignore[arg-type]
        )
    assert calls == 0


def test_rejects_oversized_fixed_trace_decay_rates_before_per_rate_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized decay tuple walked an element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(hostile,) * 4097,  # type: ignore[arg-type]
        )
    assert calls == 0


@pytest.mark.parametrize(
    "field",
    ["observation_decay_rates", "action_decay_rates", "reward_decay_rates"],
)
def test_working_memory_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str,
) -> None:
    config = WorkingMemoryConfig(observation_dim=1).to_config()
    config[field] = [0.5] * 4097
    with pytest.raises(ValueError, match="at most 4096"):
        WorkingMemoryConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["observation_decay_rates", "action_decay_rates", "outcome_decay_rates"],
)
def test_fixed_trace_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str,
) -> None:
    config = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    config[field] = [0.5] * 4097
    with pytest.raises(ValueError, match="at most 4096"):
        FixedTraceStateBuilderConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["observation_decay_rates", "action_decay_rates", "reward_decay_rates"],
)
def test_working_memory_from_config_rejects_list_subclasses_before_length_hooks(
    field: str,
) -> None:
    config = WorkingMemoryConfig(observation_dim=1).to_config()
    config[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual list or tuple"):
        WorkingMemoryConfig.from_config(config)
    assert _HostileList.calls == 0


@pytest.mark.parametrize(
    "field",
    ["observation_decay_rates", "action_decay_rates", "outcome_decay_rates"],
)
def test_fixed_trace_from_config_rejects_list_subclasses_before_length_hooks(
    field: str,
) -> None:
    config = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    config[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(config)
    assert _HostileList.calls == 0
