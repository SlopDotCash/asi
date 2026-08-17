"""Hostile validation for forager matched seal trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_seal import (
    _parse_finite_json_float,
    _reject_duplicate_keys,
    _reject_nonfinite,
    _require_exact_str,
    _write_exclusive_at,
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


def test_reject_duplicate_keys_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("dup")
    _EvilStr.calls = 0
    # Need duplicate: first insert ok, second triggers error with evil key
    with pytest.raises(Exception, match="exact string") as exc:
        _reject_duplicate_keys([("dup", 1), (evil, 2)])
    # The duplicate check uses key in result, but evil == "dup" so duplicate
    # But our key is EvilStr("dup") which equals "dup" string, so result already has "dup"
    # However the error path will validate host_key before formatting
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    # exact string error does not leak value, just confirms rejection


def test_reject_duplicate_keys_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string|duplicate"):
        _reject_duplicate_keys([("a", 1), (_StringSubclass("a"), 2)])


def test_reject_duplicate_keys_valid() -> None:
    # valid duplicate with plain str should be sanitized with single quotes
    with pytest.raises(Exception, match="duplicate JSON key") as exc:
        _reject_duplicate_keys([("dup", 1), ("dup", 2)])
    assert "!r" not in str(exc.value)
    assert "'dup'" in str(exc.value)


def test_reject_nonfinite_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("Infinity")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _reject_nonfinite(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_reject_nonfinite_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _reject_nonfinite(_StringSubclass("Infinity"))


def test_reject_nonfinite_sanitized() -> None:
    with pytest.raises(Exception, match="non-finite JSON number") as exc:
        _reject_nonfinite("Infinity")
    assert "!r" not in str(exc.value)
    assert "Infinity" in str(exc.value)
    assert "'" in str(exc.value)


def test_parse_finite_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("Infinity")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string|non-finite") as exc:
        _parse_finite_json_float(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_parse_finite_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _parse_finite_json_float(_StringSubclass("1.0"))


def test_parse_finite_nonfinite_sanitized() -> None:
    with pytest.raises(Exception, match="non-finite JSON number") as exc:
        _parse_finite_json_float("Infinity")
    assert "!r" not in str(exc.value)
    assert "Infinity" in str(exc.value)


def test_parse_finite_valid() -> None:
    assert _parse_finite_json_float("1.5") == 1.5


def test_write_exclusive_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("bad.json")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _write_exclusive_at(None, evil, b"data")
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_write_exclusive_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _write_exclusive_at(None, _StringSubclass("bad.json"), b"data")


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/forager_matched_seal.py").read_text()
    assert "!r" not in text
