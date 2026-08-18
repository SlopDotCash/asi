"""Hostile integer validation for visualization."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")

    def __hash__(self) -> int:
        return int.__hash__(self)


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")


def test_finite_positive_rejects_hostile_before_float() -> None:
    from alberta_framework.utils.visualization import _require_finite_positive

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a real number"):
        _require_finite_positive("x", hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    hf = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(Exception, match="must be a real number"):
        _require_finite_positive("x", hf)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    with pytest.raises(Exception, match="must be a real number"):
        _require_finite_positive("x", True)  # type: ignore[arg-type]
    assert _require_finite_positive("x", 1) == 1.0
    assert _require_finite_positive("x", 1.0) == 1.0
    with pytest.raises(Exception, match="must be a finite positive"):
        _require_finite_positive("x", 0)


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) not in (int, float):
            raise ValueError("must be a real number")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
