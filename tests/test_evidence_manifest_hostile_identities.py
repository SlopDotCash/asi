"""Hostile-identity tests for evidence manifest consumer."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    _artifact_digest,
    evidence_manifest_exit_code,
    evidence_manifest_json,
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

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


def test_exit_code_rejects_hostile_without_get_dispatch() -> None:
    hostile = _HostileMapping({"overall_status": "accepted"})
    _HostileMapping.calls = 0
    result = evidence_manifest_exit_code(hostile)  # type: ignore[arg-type]
    assert result == 2
    assert _HostileMapping.calls == 0


def test_manifest_json_rejects_hostile_without_iter() -> None:
    hostile = _HostileMapping({"overall_status": "accepted"})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        evidence_manifest_json(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_artifact_digest_rejects_hostile_without_get() -> None:
    hostile = _HostileMapping(
        {"scientific_digest": {"sha256": "a" * 64}, "content_digest": {"sha256": "b" * 64}}
    )
    _HostileMapping.calls = 0
    result = _artifact_digest(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileMapping.calls == 0
    # Hostile inner record
    inner_hostile = _HostileMapping({"sha256": "a" * 64})
    parent = {"scientific_digest": inner_hostile}
    _HostileMapping.calls = 0
    result = _artifact_digest(parent)  # type: ignore[arg-type]
    assert result is None
    assert _HostileMapping.calls == 0
