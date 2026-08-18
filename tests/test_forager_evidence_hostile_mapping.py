"""Hostile-identity tests for forager matched evidence mapping consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_evidence import (
    _thaw_json,
    parse_matched_score_evidence,
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

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")


def test_thaw_json_rejects_hostile_mapping_without_items_dispatch() -> None:
    hostile = _HostileMapping({"a": {"b": 1}})
    _HostileMapping.calls = 0
    result = _thaw_json(hostile)
    # _thaw_json now requires exact dict, so hostile mapping is returned as-is without iterating
    # It should not have called hostile items/iter
    assert _HostileMapping.calls == 0
    # Result should be the hostile object itself (since not dict, returned unchanged)
    assert result is hostile


def test_parse_score_evidence_rejects_hostile_mapping_without_dispatch() -> None:
    hostile = _HostileMapping({"schema_version": "test"})
    _HostileMapping.calls = 0
    with pytest.raises(TypeError):
        parse_matched_score_evidence(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_thaw_json_exact_dict_still_works() -> None:
    # Ensure normal dict still thaws correctly
    assert _thaw_json({"a": 1}) == {"a": 1}
    assert _thaw_json({"a": {"b": [1, 2]}}) == {"a": {"b": [1, 2]}}
