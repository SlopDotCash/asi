"""Hostile-identity tests for RTU config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
    _copy_config_mapping,
)


class _HostileMapping(dict):
    calls=0
    def __iter__(self):  # type: ignore[override]
        type(self).calls+=1
        raise AssertionError("hostile __iter__")
    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls+=1
        raise AssertionError("hostile getitem")
    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls+=1
        raise AssertionError("hostile get")
    def keys(self):  # type: ignore[override]
        type(self).calls+=1
        raise AssertionError("hostile keys")
    def items(self):  # type: ignore[override]
        type(self).calls+=1
        raise AssertionError("hostile items")

def test_copy_rejects_hostile() -> None:
    h=_HostileMapping({"a":1})
    _HostileMapping.calls=0
    with pytest.raises(ValueError):
        _copy_config_mapping("config", h)  # type: ignore[arg-type]
    assert _HostileMapping.calls==0

def test_from_config_rejects_hostile() -> None:
    h=_HostileMapping({"type":"RecurrentTraceActorCriticConfig"})
    _HostileMapping.calls=0
    with pytest.raises(ValueError):
        RecurrentTraceActorCriticConfig.from_config(h)  # type: ignore[arg-type]
    assert _HostileMapping.calls==0
