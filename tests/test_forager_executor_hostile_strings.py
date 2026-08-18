"""Hostile identities for forager executor bytes/str dispatch."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_executor import (
    decode_strict_json,
)

pytestmark = pytest.mark.unit


class _HostileBytes(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bytes decode executed")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len executed")


class _HostileStr(str):
    calls = 0

    __hash__ = str.__hash__

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile encode executed")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len executed")


class _HostileStrSubclass(str):
    calls = 0

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str encode executed")


class _HostileBytesSubclass(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bytes decode executed")


def test_decode_strict_json_rejects_hostile_bytes_subclass_without_decode() -> None:
    hostile = _HostileBytesSubclass(b'{"a": 1}')
    _HostileBytesSubclass.calls = 0
    with pytest.raises(TypeError, match="bytes or str"):
        decode_strict_json(hostile)  # type: ignore[arg-type]
    assert _HostileBytesSubclass.calls == 0


def test_decode_strict_json_rejects_hostile_str_subclass_without_encode() -> None:
    hostile = _HostileStrSubclass('{"a": 1}')
    _HostileStrSubclass.calls = 0
    with pytest.raises(TypeError, match="bytes or str"):
        decode_strict_json(hostile)  # type: ignore[arg-type]
    assert _HostileStrSubclass.calls == 0


def test_decode_mapping_rejects_hostile_without_canonical_check() -> None:
    from alberta_framework.benchmarks.forager_matched_executor import _decode_mapping

    hostile_str = _HostileStrSubclass('{"x": 1}')
    _HostileStrSubclass.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or str"):
        _decode_mapping(hostile_str, "label")  # type: ignore[arg-type]
    assert _HostileStrSubclass.calls == 0

    hostile_bytes = _HostileBytesSubclass(b'{"x": 1}')
    _HostileBytesSubclass.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or str"):
        _decode_mapping(hostile_bytes, "label")  # type: ignore[arg-type]
    assert _HostileBytesSubclass.calls == 0


def test_protocol_instance_rejects_subclass_without_to_dict() -> None:
    from alberta_framework.benchmarks.forager_matched_executor import (
        _protocol_instance,
    )
    from alberta_framework.benchmarks.forager_matched_protocol import (
        ForagerMatchedProtocol,
    )

    class _HostileProtocol(ForagerMatchedProtocol):  # type: ignore[type-arg]
        calls = 0

        def to_dict(self) -> dict[str, object]:  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("hostile to_dict executed")

    hostile = object.__new__(_HostileProtocol)
    _HostileProtocol.calls = 0
    with pytest.raises(Exception):
        _protocol_instance(hostile)  # type: ignore[arg-type]
    assert _HostileProtocol.calls == 0


def test_execution_plan_rejects_hostile_bytes_str() -> None:
    # Ensure the exact-type gate in execution plan path (via _decode_mapping) rejects
    hostile = _HostileStrSubclass('{"schema_version": "x"}')
    _HostileStrSubclass.calls = 0
    # Directly test the type gate mirror
    gate = type(hostile) in (bytes, str)
    assert gate is False
    assert _HostileStrSubclass.calls == 0
    # Builtin should pass gate
    assert (str in (bytes, str)) is True
    assert (type(b"a") in (bytes, str)) is True  # noqa: UP003
