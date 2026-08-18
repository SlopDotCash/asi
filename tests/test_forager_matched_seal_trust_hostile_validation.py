"""Trust-boundary validation for forager_matched_seal sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_seal import (
    ForagerMatchedSealError,
    _parse_finite_json_float,
    _reject_duplicate_keys,
    _reject_nonfinite,
    _require_exact_str,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _require_exact_str("value", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("val")
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string") as exc2:
        _require_exact_str("value", evil2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_duplicate_key_sanitized() -> None:
    with pytest.raises(ForagerMatchedSealError, match="duplicate JSON key") as exc:
        _reject_duplicate_keys([("evil_key", 1), ("evil_key", 2)])
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_key'" in msg


def test_duplicate_key_hostile_blocked_before_hash() -> None:
    evil = _EvilStr("evil")
    # Must be blocked before __hash__ is invoked (host gate before membership test)
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string") as exc:
        _reject_duplicate_keys([(evil, 1)])  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    # Subclass also rejected
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _reject_duplicate_keys([(_StringSubclass("evil"), 1)])  # type: ignore[arg-type]


def test_nonfinite_sanitized() -> None:
    with pytest.raises(ForagerMatchedSealError, match="non-finite JSON number") as exc:
        _reject_nonfinite("NaN")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'NaN'" in msg
    with pytest.raises(ForagerMatchedSealError, match="non-finite JSON number") as exc2:
        _parse_finite_json_float("Infinity")
    msg2 = str(exc2.value)
    assert "!r" not in msg2
    assert "'Infinity'" in msg2


def test_nonfinite_hostile() -> None:
    evil = _EvilStr("Infinity")
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string") as exc:
        _reject_nonfinite(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("NaN")
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _parse_finite_json_float(evil2)  # type: ignore[arg-type]
    with pytest.raises(ForagerMatchedSealError, match="must be an exact string"):
        _parse_finite_json_float(_StringSubclass("Infinity"))  # type: ignore[arg-type]


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_matched_seal.py")
    text = p.read_text(encoding="utf-8")
    assert "duplicate JSON key {key!r}" not in text
    assert "non-finite JSON number {value!r}" not in text
    assert "cannot stage seal artifact {name!r}" not in text
    assert "cannot completely stage artifact {name!r}" not in text
    assert "staged artifact {name!r}" not in text
    assert "duplicate JSON key '{host_key}'" in text
    assert "non-finite JSON number '{host_value}'" in text
    assert "cannot stage seal artifact '{host_name}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert _reject_duplicate_keys([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    assert _parse_finite_json_float("1.5") == 1.5
    # _reject_nonfinite always raises
    with pytest.raises(ForagerMatchedSealError):
        _reject_nonfinite("x")
