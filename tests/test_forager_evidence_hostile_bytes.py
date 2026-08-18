"""Hostile identities for forager matched evidence bytes/str."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_evidence import (
    ForagerMatchedEvidenceError,
    decode_strict_json,
)

pytestmark = pytest.mark.unit


class _HostileBytes(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bytes decode")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile encode")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")


def test_decode_rejects_hostile_bytes_subclass() -> None:
    hostile = _HostileBytes(b'{"a": 1}')
    _HostileBytes.calls = 0
    with pytest.raises(TypeError, match="bytes or str"):
        decode_strict_json(hostile)  # type: ignore[arg-type]
    assert _HostileBytes.calls == 0


def test_decode_rejects_hostile_str_subclass() -> None:
    hostile = _HostileStr('{"a": 1}')
    _HostileStr.calls = 0
    with pytest.raises(TypeError, match="bytes or str"):
        decode_strict_json(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_evidence_bytes_str_gate_rejects_hostile() -> None:
    # Direct gate mirror
    hostile_s = _HostileStr('{"x": 1}')
    _HostileStr.calls = 0
    assert (type(hostile_s) in (bytes, str)) is False
    assert _HostileStr.calls == 0
    hostile_b = _HostileBytes(b'{"x": 1}')
    _HostileBytes.calls = 0
    assert (type(hostile_b) in (bytes, str)) is False
    assert _HostileBytes.calls == 0


def test_score_evidence_rejects_hostile_without_encode() -> None:
    from alberta_framework.benchmarks.forager_matched_evidence import (
        parse_matched_score_evidence,
    )

    hostile = _HostileStr('{"schema_version": "x"}')
    _HostileStr.calls = 0
    with pytest.raises((TypeError, ForagerMatchedEvidenceError)):
        parse_matched_score_evidence(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
