"""Hostile-identity tests for forager matched evaluation campaign consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_evaluation_campaign import (
    ForagerMatchedEvaluationCampaignError,
    _plain_json,
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


def test_plain_json_rejects_hostile_mapping_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ForagerMatchedEvaluationCampaignError):
        _plain_json(hostile)
    assert _HostileMapping.calls == 0


def test_plain_json_rejects_hostile_list_without_dispatch() -> None:
    hostile = _HostileList([1, 2, 3])
    _HostileList.calls = 0
    with pytest.raises(ForagerMatchedEvaluationCampaignError):
        _plain_json(hostile)
    assert _HostileList.calls == 0
    hostile_nested = {"key": _HostileList([1])}
    _HostileList.calls = 0
    with pytest.raises(ForagerMatchedEvaluationCampaignError):
        _plain_json(hostile_nested)
    assert _HostileList.calls == 0


def test_decode_schedule_rejects_hostile_mapping() -> None:
    hostile = _HostileMapping({"schema_version": "x"})
    _HostileMapping.calls = 0
    from alberta_framework.benchmarks.forager_matched_evaluation_campaign import _decode_schedule

    _HostileMapping.calls = 0
    with pytest.raises((ForagerMatchedEvaluationCampaignError, TypeError)):
        _decode_schedule(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
