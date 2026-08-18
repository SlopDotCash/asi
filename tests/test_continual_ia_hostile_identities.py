"""Hostile-identity tests for continual IA artifact consumer."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import (
    _number,
    canonical_content_bytes,
    validate_ia_evidence_artifact,
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


def test_canonical_content_bytes_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        canonical_content_bytes(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_validate_ia_evidence_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"schema_version": "v1"})
    _HostileMapping.calls = 0
    result = validate_ia_evidence_artifact(hostile)  # type: ignore[arg-type]
    assert not result.valid
    assert _HostileMapping.calls == 0


def test_number_rejects_hostile_int_without_float() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    result = _number(hostile)
    assert result is None
    assert _HostileInt.calls == 0
