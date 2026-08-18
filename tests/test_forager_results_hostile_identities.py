"""Hostile-identity tests for forager results consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_results import _json_mapping_copy


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

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")


def test_json_mapping_copy_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _json_mapping_copy(hostile, name="test")  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_json_mapping_copy_rejects_hostile_nested_without_dispatch() -> None:
    hostile = _HostileMapping({"semantic": {"a": 1}, "implementation": {"b": 2}})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _json_mapping_copy(hostile, name="environment_provenance")  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
