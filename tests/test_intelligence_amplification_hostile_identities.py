"""Hostile-identity tests for intelligence amplification legacy consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.intelligence_amplification import _host_field_mapping


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


def test_host_field_mapping_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"weights": 1, "step_count": 2})
    _HostileMapping.calls = 0
    with pytest.raises(TypeError):
        _host_field_mapping(hostile, name="exo-cerebellum")  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_host_field_mapping_rejects_hostile_empty_without_dispatch() -> None:
    hostile = _HostileMapping({})
    _HostileMapping.calls = 0
    with pytest.raises(TypeError):
        _host_field_mapping(hostile, name="ia")  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
