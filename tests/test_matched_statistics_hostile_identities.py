"""Hostile-identity tests for forager matched statistics consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_statistics import (
    MatchedStatisticsError,
    _canonical_json_bytes,
    canonical_payload_sha256,
)


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

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


def test_canonical_json_bytes_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(MatchedStatisticsError):
        _canonical_json_bytes(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_canonical_payload_sha256_rejects_hostile() -> None:
    hostile = _HostileMapping({"a": 1, "b": 2})
    _HostileMapping.calls = 0
    with pytest.raises(MatchedStatisticsError):
        canonical_payload_sha256(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
