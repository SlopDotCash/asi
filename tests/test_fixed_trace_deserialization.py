"""Fixed trace decoding rejects sequence hooks without imposing arbitrary caps."""

from typing import Any

import pytest

from alberta_framework.core.state_builder import FixedTraceStateBuilderConfig
from alberta_framework.core.working_memory import WorkingMemoryConfig


class HostileList(list[float]):
    def __iter__(self) -> Any:
        raise AssertionError("sequence iteration hook executed")

    def __len__(self) -> int:
        raise AssertionError("sequence length hook executed")


class HostileTuple(tuple[float, ...]):
    def __iter__(self) -> Any:
        raise AssertionError("sequence iteration hook executed")

    def __len__(self) -> int:
        raise AssertionError("sequence length hook executed")


@pytest.mark.parametrize(
    "field", ["observation_decay_rates", "action_decay_rates", "outcome_decay_rates"]
)
@pytest.mark.parametrize("sequence_type", [HostileList, HostileTuple])
def test_fixed_trace_decode_rejects_sequence_subclasses(field: str, sequence_type: Any) -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=1).to_config()
    payload[field] = sequence_type([0.5])
    with pytest.raises(ValueError, match="lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(payload)


@pytest.mark.parametrize("config_type", [WorkingMemoryConfig, FixedTraceStateBuilderConfig])
@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_budget_valid_decay_sequence_roundtrip(config_type: Any, sequence_type: Any) -> None:
    rates = (0.5,) * 4097
    config = config_type(observation_dim=1, observation_decay_rates=rates)
    payload = config.to_config()
    payload["observation_decay_rates"] = sequence_type(rates)
    assert config_type.from_config(payload) == config
