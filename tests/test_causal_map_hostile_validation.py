"""Hostile validation for causal map forager."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.causal_map_forager import _require_exact_str


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
        _require_exact_str("impl", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("impl", _StringSubclass("v"))  # type: ignore[arg-type]


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/benchmarks/causal_map_forager.py"
    ).read_text()
    assert "!r" not in text


def test_valid_exact_str_passes() -> None:
    assert _require_exact_str("impl", "threefry2x32") == "threefry2x32"


def test_kind_validation_sanitized() -> None:
    # Simulate the kind error path: host_kind validation before RuntimeError
    evil = _EvilStr("unknown_kind")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("kind", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
