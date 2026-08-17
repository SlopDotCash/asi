"""Hostile validation for forager matched protocol trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    _duplicate_free_object,
    _parse_json_float,
    _reject_nonfinite,
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
    with pytest.raises(Exception, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_duplicate_free_rejects_evil() -> None:
    evil = _EvilStr("dup")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _duplicate_free_object([("dup", 1), (evil, 2)])
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_duplicate_free_sanitized() -> None:
    with pytest.raises(Exception, match="duplicate JSON object key") as exc:
        _duplicate_free_object([("dup", 1), ("dup", 2)])
    assert "!r" not in str(exc.value)
    assert "'dup'" in str(exc.value)


def test_reject_nonfinite_rejects_evil() -> None:
    evil = _EvilStr("Infinity")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _reject_nonfinite(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_reject_nonfinite_sanitized() -> None:
    with pytest.raises(Exception, match="non-finite JSON number") as exc:
        _reject_nonfinite("Infinity")
    assert "!r" not in str(exc.value)
    assert "Infinity" in str(exc.value)


def test_parse_json_float_rejects_evil() -> None:
    evil = _EvilStr("1.0")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _parse_json_float(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_parse_json_float_invalid_sanitized() -> None:
    with pytest.raises(Exception, match="invalid JSON number") as exc:
        _parse_json_float("not_a_number")
    assert "!r" not in str(exc.value)
    assert "not_a_number" in str(exc.value)


def test_parse_json_float_nonfinite_sanitized() -> None:
    with pytest.raises(Exception, match="non-finite JSON number") as exc:
        _parse_json_float("Infinity")
    assert "!r" not in str(exc.value)


def test_parse_json_float_valid() -> None:
    assert _parse_json_float("1.5") == 1.5


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/forager_matched_protocol.py").read_text()
    assert "!r" not in text
