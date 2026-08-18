"""Hostile validation for timing repr totalization."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.utils.timing import Timer


class _EvilStr(str):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


def test_timing_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/utils/timing.py").read_text()
    assert "!r" not in text


def test_timer_repr_sanitized_without_repr() -> None:
    t = Timer(name="good", verbose=False)
    t.duration = 1.23
    r = repr(t)
    assert "good" in r
    assert "!r" not in r
    t2 = Timer(name="good", verbose=False)
    r2 = repr(t2)
    assert "good" in r2


def test_timer_rejects_evil_name_before_repr() -> None:
    evil = _EvilStr("op")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        Timer(name=evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
