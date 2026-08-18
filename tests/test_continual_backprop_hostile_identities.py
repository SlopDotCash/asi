"""Hostile-identity tests for continual backprop config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.continual_backprop import ContinualBackpropConfig, _copy_mapping


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __contains__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __contains__")

    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile keys")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")


def test_copy_mapping_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _copy_mapping(hostile, name="ContinualBackpropConfig")  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_continual_backprop_from_config_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"decay_rate": 0.9})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        ContinualBackpropConfig.from_config(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
