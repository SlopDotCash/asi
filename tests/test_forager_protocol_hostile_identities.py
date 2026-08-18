"""Hostile-identity tests for forager matched protocol consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocolError,
    _require_array,
    _require_object,
    _validate_json_complexity,
    parse_forager_matched_protocol,
)


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")

    def __getitem__(self, index):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")


def test_require_object_rejects_hostile_mapping_without_iter_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        _require_object(hostile, "test")
    assert _HostileMapping.calls == 0


def test_require_array_rejects_hostile_list_without_iter_dispatch() -> None:
    hostile = _HostileList([1, 2, 3])
    _HostileList.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        _require_array(hostile, "test")
    assert _HostileList.calls == 0


def test_validate_json_complexity_rejects_hostile_mapping_without_values_dispatch() -> None:
    hostile = _HostileMapping({"key": "value"})
    _HostileMapping.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        _validate_json_complexity(hostile)
    assert _HostileMapping.calls == 0
    _HostileList.calls = 0
    hostile_list = _HostileList([1, 2])
    with pytest.raises(ForagerMatchedProtocolError):
        _validate_json_complexity(hostile_list)
    assert _HostileList.calls == 0


def test_parse_rejects_hostile_top_level_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "alberta.forager_matched_protocol.v1",
            "stage": "open_tuning",
            "task": {},
            "horizon": 1,
            "tuning_seeds": [0],
            "evaluation_seeds": [1],
            "active_seeds": [0],
            "candidates": [],
            "selection_plan": {},
            "selection_outcome": {},
            "analysis_plan": {},
            "evaluation_panel": {},
            "primary_hypothesis": {},
            "secondary_hypotheses": [],
            "multiplicity_policy": {},
            "privileged_context": {},
            "historical_orientation": {},
            "runtime": {},
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        parse_forager_matched_protocol(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
