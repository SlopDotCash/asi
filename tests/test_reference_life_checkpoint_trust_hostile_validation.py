"""Trust-boundary validation for reference_life_checkpoint sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.reference_life_checkpoint import (
    _reject_duplicate_keys,
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
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("relative", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_duplicate_key_sanitized() -> None:
    with pytest.raises(ValueError, match="duplicate key") as exc:
        _reject_duplicate_keys([("evil_key", 1), ("evil_key", 2)])
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_key'" in msg


def test_duplicate_key_hostile_blocked_before_hash() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _reject_duplicate_keys([(evil, 1)])  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    with pytest.raises(ValueError, match="must be an exact string"):
        _reject_duplicate_keys([(_StringSubclass("evil"), 1)])  # type: ignore[arg-type]


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/reference_life_checkpoint.py")
    text = p.read_text(encoding="utf-8")
    assert "duplicate key {key!r}" not in text
    assert "symlink {relative!r}" not in text
    assert "non-regular entry {relative!r}" not in text
    assert "symlink {child_relative!r}" not in text
    assert "non-regular entry {child_relative!r}" not in text
    assert "TransactionPhase.ARMED.value!r" not in text
    assert "duplicate key '{host_key}'" in text
    assert "symlink '{host_relative}'" in text
    assert "symlink '{host_child_relative}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert _reject_duplicate_keys([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
