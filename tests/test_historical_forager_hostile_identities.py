"""Hostile-identity tests for historical forager consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerArtifactError,
    HistoricalForagerContractError,
    _json_mapping_copy,
    _require_exact_keys,
)


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


def test_json_mapping_copy_rejects_hostile_without_iter() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(HistoricalForagerContractError):
        _json_mapping_copy(hostile, name="test")
    assert _HostileMapping.calls == 0


def test_require_exact_keys_rejects_hostile_without_set() -> None:
    hostile = _HostileMapping({"a": 1, "b": 2})
    _HostileMapping.calls = 0
    with pytest.raises(HistoricalForagerArtifactError):
        _require_exact_keys(hostile, {"a", "b"}, name="test")
    assert _HostileMapping.calls == 0
    # Also test wrong type (list) hostile not needed, but check hostile dict subclass
    _HostileMapping.calls = 0
    with pytest.raises(HistoricalForagerArtifactError):
        _require_exact_keys(hostile, {"x"}, name="test")
    assert _HostileMapping.calls == 0


def test_validate_adapter_rejects_hostile() -> None:
    from alberta_framework.benchmarks.historical_forager import _validate_adapter_manifest

    hostile = _HostileMapping({"mode": "test"})
    _HostileMapping.calls = 0
    with pytest.raises(HistoricalForagerArtifactError):
        _validate_adapter_manifest(hostile)
    assert _HostileMapping.calls == 0
