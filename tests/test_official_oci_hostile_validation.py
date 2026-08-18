"""Hostile validation for official OCI JSON."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.official_foragax_oci import (
    OfficialForagaxOciError,
    _require_exact_str,
    _strict_json_bytes,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(OfficialForagaxOciError, match="exact string") as exc:
        _require_exact_str("label", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_strict_json_rejects_duplicate_sanitized() -> None:
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(OfficialForagaxOciError, match="repeats object key") as exc:
        _strict_json_bytes(raw, label="test")
    assert "!r" not in str(exc.value)
    assert "a" not in str(exc.value) or str(exc.value) == "test repeats object key"


def test_strict_json_rejects_constant_sanitized() -> None:
    raw = b'{"a": NaN}'
    with pytest.raises(OfficialForagaxOciError, match="non-finite constant") as exc:
        _strict_json_bytes(raw, label="test")
    assert "!r" not in str(exc.value)
    assert "NaN" not in str(exc.value)


def test_strict_json_rejects_nonfinite_float_sanitized() -> None:
    raw = b'{"a": 1e999}'
    with pytest.raises(OfficialForagaxOciError, match="non-finite") as exc:
        _strict_json_bytes(raw, label="test")
    assert "!r" not in str(exc.value)


def test_label_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("test")
    _EvilStr.calls = 0
    with pytest.raises(OfficialForagaxOciError, match="exact string"):
        _strict_json_bytes(b'{"a":1}', label=evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/benchmarks/official_foragax_oci.py"
    ).read_text()
    assert "!r" not in text


def test_valid_json_passes() -> None:
    raw = b'{"a": 1, "b": 2}'
    data = _strict_json_bytes(raw, label="test")
    assert data == {"a": 1, "b": 2}
