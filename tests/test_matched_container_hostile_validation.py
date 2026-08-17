"""Hostile validation for matched container scorer."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks._forager_matched_container import (
    ContainerError,
    _require_exact_str,
    _strict_json,
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
    with pytest.raises(ContainerError, match="exact string") as exc:
        _require_exact_str("value", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ContainerError, match="exact string"):
        _require_exact_str("value", _StringSubclass("v"))  # type: ignore[arg-type]


def test_strict_json_rejects_nonfinite_constant_sanitized() -> None:
    raw = b'{"a": NaN}'
    with pytest.raises(ContainerError, match="non-finite number") as exc:
        _strict_json(raw)
    assert "!r" not in str(exc.value)
    assert "NaN" not in str(exc.value)


def test_strict_json_rejects_nonfinite_float_sanitized() -> None:
    raw = b'{"a": 1e999}'
    with pytest.raises(ContainerError, match="non-finite number") as exc:
        _strict_json(raw)
    assert "!r" not in str(exc.value)


def test_strict_json_rejects_duplicate_keys() -> None:
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(ContainerError, match="duplicate keys"):
        _strict_json(raw)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/benchmarks/_forager_matched_container.py"
    ).read_text()
    assert "!r" not in text


def test_valid_json_passes() -> None:
    raw = b'{"a": 1, "b": 2}'
    data = _strict_json(raw)
    assert data == {"a": 1, "b": 2}
