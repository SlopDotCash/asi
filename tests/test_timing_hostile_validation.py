"""Hostile-safe validation for timing utilities."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.utils.timing import Timer, format_duration


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")

    def __float__(self) -> float:
        type(self).calls += 1
        raise RuntimeError("float hook")


class _StringSubclass(str):
    pass


def test_format_rejects_bool() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(True)
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(np.bool_(True))


def test_format_rejects_hostile_float_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_format_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(_StringSubclass("1.0"))


def test_format_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(float("nan"))
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(float("inf"))


def test_format_rejects_negative() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        format_duration(-1.0)


def test_format_valid_cases() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(90.5) == "1m 30.50s"
    assert format_duration(3665) == "1h 1m 5.00s"
    assert format_duration(Fraction(1, 2)) == "0.50s"
    assert format_duration(np.float64(1.0)) == "1.00s"


def test_timer_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        Timer(name=_StringSubclass("op"))


def test_timer_does_not_invoke_hostile_name_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

        def __str__(self) -> str:  # pragma: no cover
            raise RuntimeError("str hook")

    with pytest.raises(ValueError, match="exact string"):
        Timer(name=EvilStr("op"))


def test_timer_rejects_verbose_not_bool() -> None:
    with pytest.raises(ValueError, match="built-in bool"):
        Timer(name="op", verbose=1)
    with pytest.raises(ValueError, match="built-in bool"):
        Timer(name="op", verbose=_StringSubclass("true"))


def test_timer_repr_does_not_invoke_hostile_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    evil = EvilStr("op")
    # Timer should reject EvilStr at construction, so repr not needed
    with pytest.raises(ValueError, match="exact string"):
        Timer(name=evil)
    # Valid timer repr should not use !r
    t = Timer(name="good")
    t.duration = 1.23
    r = repr(t)
    assert "good" in r
    assert "duration" in r


def test_timer_valid() -> None:
    t = Timer(name="ok", verbose=False)
    assert t.name == "ok"
    assert t.verbose is False
    with t:
        pass
    assert t.duration >= 0


def test_format_rejects_hostile_int() -> None:
    class HostileInt(int):
        def __repr__(self) -> str:  # pragma: no cover
            raise AssertionError("repr hook")

    with pytest.raises(ValueError, match="finite real number"):
        format_duration(HostileInt(1))
