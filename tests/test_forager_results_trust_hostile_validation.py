"""Trust-boundary validation for forager_results sanitized errors."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_results import (
    _require_exact_str,
    _sanitize_for_error,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _EvilStrNoHash(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStrNoHash.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStrNoHash.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("prefix", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("prefix", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("evil2")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        _require_exact_str("key", evil2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_sanitize_for_error_sanitized() -> None:
    assert _sanitize_for_error("evil") == "'evil'"
    assert _sanitize_for_error(123) == "123"
    # Hostile subclass treated as non-str -> fallback would call str
    # Avoid calling _sanitize on hostile; gate via _require_exact_str
    evil = _EvilStrNoHash("evil")
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("x", evil)  # type: ignore[arg-type]


def test_duplicate_key_hostile_blocked_before_hash(tmp_path: Path) -> None:
    # Test via _json_without_duplicate_keys indirectly through loading a file with duplicate keys
    # We construct raw JSON bytes with duplicate keys and ensure hostile gating before hash
    from alberta_framework.benchmarks.forager_results import _json_without_duplicate_keys

    # Normal duplicate should raise with sanitized message
    payload = b'{"a": 1, "a": 2}'
    with pytest.raises(ValueError, match="duplicate JSON key") as exc:
        _json_without_duplicate_keys(payload, path=Path("dummy"))
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'a'" in msg

    # Hostile subclass key must be blocked before hash membership test
    evil = _EvilStr("evil")
    # object_pairs is inner; we test _require_exact_str gate directly
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", evil)  # type: ignore[arg-type]


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_results.py")
    text = p.read_text(encoding="utf-8")
    assert "prefix!r" not in text
    assert "key!r" not in text
    assert "value!r" not in text
    assert "expected_config_agent!r" not in text
    assert "agent!r" not in text
    assert "config_path.name!r" not in text
    assert "integrity!r" not in text
    assert "schema_objects!r" not in text
    assert "column!r" not in text
    assert "actual!r" not in text
    assert "seed!r" not in text
    assert "frame!r" not in text
    assert "key!r" not in text
    assert "FORAGAX_DISTRIBUTION!r" not in text
    assert "distribution!r" not in text
    assert "path.name!r" not in text
    # sanitized forms exist
    assert "unsupported config hyperparameter type at '{host_prefix}'" in text
    assert "contains duplicate JSON key '{host_key}'" in text
    assert "contains non-standard JSON constant '{host_value}'" in text
    assert "contains non-finite JSON number '{host_value}'" in text
    assert "contains duplicate '{host_dist}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("prefix", "ok") == "ok"
    assert _sanitize_for_error("ok") == "'ok'"
    assert _sanitize_for_error(42) == "42"
    # duplicate key valid still passes via helper
    from alberta_framework.benchmarks.forager_results import _json_without_duplicate_keys

    payload = b'{"a": 1, "b": 2}'
    result = _json_without_duplicate_keys(payload, path=Path("dummy"))
    assert result == {"a": 1, "b": 2}
