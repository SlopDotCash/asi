"""Hostile validation for forager RNG parity."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_rng_parity import (
    ForagerRngParityError,
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
    with pytest.raises(ForagerRngParityError, match="exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_duplicate_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("a")
    _EvilStr.calls = 0
    with pytest.raises(ForagerRngParityError, match="exact string"):
        _duplicate_free_object([(evil, 1)])  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_duplicate_sanitized_without_repr() -> None:
    with pytest.raises(ForagerRngParityError, match="duplicate JSON object key") as exc:
        _duplicate_free_object([("a", 1), ("a", 2)])
    assert "!r" not in str(exc.value)
    assert str(exc.value) == "duplicate JSON object key"


def test_reject_nonfinite_sanitized() -> None:
    evil = _EvilStr("1e999")
    _EvilStr.calls = 0
    with pytest.raises(ForagerRngParityError, match="exact string"):
        _reject_nonfinite(evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0

    with pytest.raises(ForagerRngParityError, match="non-finite JSON number") as exc:
        _reject_nonfinite("NaN")
    assert "!r" not in str(exc.value)
    assert "NaN" not in str(exc.value)


def test_parse_float_rejects_evil() -> None:
    evil = _EvilStr("1e999")
    _EvilStr.calls = 0
    with pytest.raises(ForagerRngParityError, match="exact string"):
        _parse_json_float(evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_parse_float_sanitized() -> None:
    with pytest.raises(ForagerRngParityError, match="non-finite JSON number") as exc:
        _parse_json_float("1e999")
    assert "!r" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/benchmarks/forager_rng_parity.py"
    ).read_text()
    assert "!r" not in text


def test_valid_duplicate_passes() -> None:
    assert _duplicate_free_object([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    assert _parse_json_float("1.23") == 1.23
