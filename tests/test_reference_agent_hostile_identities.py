"""Hostile-identity tests for reference agent consumer."""

from __future__ import annotations

import pytest

from alberta_framework.reference_agent import (
    _canonical_json_bytes,
    _validate_json_value,
    canonical_config_sha256,
)


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")


def test_canonical_json_bytes_rejects_hostile_mapping() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _canonical_json_bytes(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        canonical_config_sha256(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_validate_json_value_rejects_hostile_mapping() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _validate_json_value(hostile, path="config")
    assert _HostileMapping.calls == 0


def test_validate_json_value_rejects_hostile_list() -> None:
    hostile = _HostileList([1, 2])
    _HostileList.calls = 0
    with pytest.raises(ValueError):
        _validate_json_value(hostile, path="config")
    assert _HostileList.calls == 0
