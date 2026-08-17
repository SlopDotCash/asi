"""Hostile validation for canonical UPGD identity."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.core.canonical_upgd import _require_exact_identity


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


def test_rejects_evil_value_without_hooks() -> None:
    evil = _EvilStr("expected")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_identity(evil, "path", "expected")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_rejects_evil_path_without_hooks() -> None:
    evil = _EvilStr("path")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_identity("value", evil, "expected")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_rejects_subclass_value() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_identity(_StringSubclass("expected"), "path", "expected")  # type: ignore[arg-type]


def test_mismatch_sanitized_without_repr() -> None:
    with pytest.raises(ValueError, match="must equal the canonical identity") as exc:
        _require_exact_identity("wrong", "path", "expected")
    assert "!r" not in str(exc.value)
    assert "expected" in str(exc.value)
    assert "wrong" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/core/canonical_upgd.py").read_text()
    assert "!r" not in text


def test_valid_passes() -> None:
    assert _require_exact_identity("expected", "path", "expected") == "expected"
