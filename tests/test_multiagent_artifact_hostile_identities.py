"""Hostile-identity tests for continual multiagent artifact consumer."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    _finite_number,
    _required_mapping,
    canonical_content_bytes,
    validate_evidence_artifact,
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

    def __contains__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __contains__")


class _HostileInt(int):
    calls = 0

    def __float__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


def test_required_mapping_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    parent: dict[str, object] = {"key": hostile}
    _HostileMapping.calls = 0
    result = _required_mapping(parent, "key", [])
    assert result is None
    assert _HostileMapping.calls == 0


def test_finite_number_rejects_hostile_int() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert _finite_number(hostile) is None
    assert _HostileInt.calls == 0


def test_canonical_content_bytes_rejects_hostile() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        canonical_content_bytes(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_validate_rejects_hostile_without_iter() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "alberta.continual_multiagent.v1",
            "content": {},
            "content_digest": {},
            "operational_diagnostics": {},
        }
    )
    _HostileMapping.calls = 0
    result = validate_evidence_artifact(hostile)  # type: ignore[arg-type]
    assert not result.valid
    assert _HostileMapping.calls == 0
