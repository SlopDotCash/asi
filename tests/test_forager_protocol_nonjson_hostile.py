"""Hostile int gate for forager matched protocol non-JSON value."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    _validate_json_complexity,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq")


def test_non_json_rejects_hostile_int_before_type_name() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    payload = {"a": hostile}
    with pytest.raises(Exception):
        _validate_json_complexity(payload)
    assert _HostileInt.calls == 0


def test_benign_int_passes() -> None:
    payload = {"a": 1}
    _validate_json_complexity(payload)


def test_benign_str_passes() -> None:
    payload = {"a": "hi"}
    _validate_json_complexity(payload)
