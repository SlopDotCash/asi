"""Hostile validation for rule discovery int gates."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.rule_discovery import _require_search_int


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _EvilStr(str):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")


def test_require_search_int_rejects_hostile_without_hooks() -> None:
    evil = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", evil, minimum=1)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "!r" not in str(exc.value)
    assert "HostileInt" not in str(exc.value)


def test_require_search_int_rejects_string_subclass_name() -> None:
    evil = _EvilStr("my_param")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_search_int(evil, 1, minimum=0)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_require_search_int_rejects_below_minimum_sanitized() -> None:
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", 0, minimum=1)
    assert "!r" not in str(exc.value)
    assert "my_param" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/rule_discovery.py").read_text()
    assert "!r" not in text


def test_valid_int_passes() -> None:
    assert _require_search_int("my_param", 5, minimum=1) == 5
