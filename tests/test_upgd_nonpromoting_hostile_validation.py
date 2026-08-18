"""Hostile validation for UPGD nonpromoting JSON."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import (
    _decode_strict_json_object,
    _require_exact_str,
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
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_decode_rejects_duplicate_sanitized() -> None:
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(ValueError, match="duplicate JSON key") as exc:
        _decode_strict_json_object(raw)
    assert "!r" not in str(exc.value)
    assert str(exc.value) == "duplicate JSON key"


def test_decode_rejects_nonfinite_constant_sanitized() -> None:
    raw = b'{"a": NaN}'
    with pytest.raises(ValueError, match="non-finite JSON constant") as exc:
        _decode_strict_json_object(raw)
    assert "!r" not in str(exc.value)
    assert "NaN" not in str(exc.value)


def test_decode_rejects_nonfinite_number_sanitized() -> None:
    raw = b'{"a": 1e999}'
    with pytest.raises(ValueError, match="non-finite JSON number") as exc:
        _decode_strict_json_object(raw)
    assert "!r" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/evaluation/upgd_ipmnist_nonpromoting.py"
    ).read_text()
    assert "!r" not in text


def test_valid_json_passes() -> None:
    raw = b'{"a": 1, "b": 2}'
    data = _decode_strict_json_object(raw)
    assert data == {"a": 1, "b": 2}
