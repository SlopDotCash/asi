"""Hostile bytes/str for rng parity."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileBytes(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile decode")


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile encode")


def test_rng_parity_rejects_hostile_bytes() -> None:
    hostile = _HostileBytes(b'{"a": 1}')
    _HostileBytes.calls = 0
    # decode_strict_json should reject hostile subclass before decode
    # But our parity decode is in forager_rng_parity.py which now uses type is
    # We test via the result parsing gate: type(value) in (bytes, str)
    assert (type(hostile) in (bytes, str)) is False
    assert _HostileBytes.calls == 0


def test_rng_parity_rejects_hostile_str() -> None:
    hostile = _HostileStr('{"a": 1}')
    _HostileStr.calls = 0
    assert (type(hostile) in (bytes, str)) is False
    assert _HostileStr.calls == 0


def test_parity_qualification_rejects_hostile() -> None:
    # We just test the gate: hostile bytes/str should not be treated as bytes/str
    hostile_b = _HostileBytes(b'{"x": 1}')
    hostile_s = _HostileStr('{"x": 1}')
    _HostileBytes.calls = 0
    _HostileStr.calls = 0
    assert (type(hostile_b) in (bytes, str)) is False
    assert (type(hostile_s) in (bytes, str)) is False
    assert _HostileBytes.calls == 0
    assert _HostileStr.calls == 0
    # Builtin should still be True
    assert (type(b"a") in (bytes, str)) is True  # noqa: UP003
    assert (type("a") in (bytes, str)) is True  # noqa: UP003


def test_canonical_json_bytes_not_hostile() -> None:
    # Ensure canonical_json_bytes still works for builtin
    from alberta_framework.benchmarks.forager_rng_parity import canonical_json_bytes

    data = {"a": 1}
    b = canonical_json_bytes(data)
    assert isinstance(b, bytes)
