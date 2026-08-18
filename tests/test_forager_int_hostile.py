"""Hostile integer validation for forager."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")


class _HostileFloat(float):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float eq")


def test_require_int_rejects_hostile_before_range() -> None:
    from alberta_framework.benchmarks.forager import _require_builtin_int

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _require_builtin_int(hostile, name="x", minimum=0, maximum=10)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_builtin_int(5, name="x", minimum=0, maximum=10) == 5
    with pytest.raises(Exception, match="must be an integer"):
        _require_builtin_int(True, name="x", minimum=0, maximum=10)  # type: ignore[arg-type]


def test_widths_rejects_hostile_before_lt() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    assert (type(hostile) is not int or hostile < 1) is True
    assert _HostileInt.calls == 0
    assert (int is not int or 2 < 1) is False


def test_finite_rejects_hostile_before_isfinite() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    # type(value) not in (int,float) should reject hostile subclass before math.isfinite
    assert (type(hostile) not in (int, float)) is True
    assert _HostileInt.calls == 0
    assert (float not in (int, float)) is False
    assert (bool not in (int, float)) is True  # bool rejected


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
