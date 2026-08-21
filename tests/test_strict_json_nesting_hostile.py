"""Hostile string gate for strict json nesting before iter."""

from __future__ import annotations

import pytest

from alberta_framework._strict_json import _scan_json_nesting

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile iter must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_scan_rejects_hostile_before_iter() -> None:
    hostile = _HostileStr('{"a": 1}')
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="JSON text must be an exact string"):
        _scan_json_nesting(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_scan_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="JSON text must be an exact string"):
        _scan_json_nesting(123)  # type: ignore[arg-type]


def test_scan_benign_valid() -> None:
    _scan_json_nesting('{"a": 1}')
    _scan_json_nesting('[]')


def test_scan_benign_deep_rejects() -> None:
    deep = "[" * 65 + "]" * 65
    with pytest.raises(ValueError, match="nesting-depth limit"):
        _scan_json_nesting(deep)
