"""Reject oversized Horde hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    _MAX_HORDE_HIDDEN_LAYERS,
    PrototypeAgentConfig,
)


def _oak() -> OaKConfig:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=4,
        n_primitive_actions=2,
    )
    return OaKConfig(stomp=stomp)


def _cfg(**overrides: object) -> PrototypeAgentConfig:
    base: dict[str, object] = {"oak": _oak()}
    base.update(overrides)
    return PrototypeAgentConfig(**base)  # type: ignore[arg-type]


def test_horde_hidden_layer_cap_constant() -> None:
    assert _MAX_HORDE_HIDDEN_LAYERS == 4096


def test_prototype_accepts_max_horde_hidden_layer_count() -> None:
    _cfg(horde_hidden_sizes=(1,) * _MAX_HORDE_HIDDEN_LAYERS)


def test_prototype_rejects_oversized_horde_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="horde_hidden_sizes length"):
        _cfg(horde_hidden_sizes=(1,) * (_MAX_HORDE_HIDDEN_LAYERS + 1))


def test_prototype_from_config_rejects_oversized_horde_hidden_list() -> None:
    payload = _cfg(horde_hidden_sizes=(32, 16)).to_config()
    payload["horde_hidden_sizes"] = [1] * (_MAX_HORDE_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="horde_hidden_sizes length"):
        PrototypeAgentConfig.from_config(payload)
