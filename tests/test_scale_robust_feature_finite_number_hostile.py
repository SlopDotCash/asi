"""Hostile int/float gate for scale_robust_feature _finite_number/_strict_int."""

from __future__ import annotations

import math

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _finite_number,
    _strict_int,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")


def test_finite_number_rejects_hostile_int_before_float() -> None:
    hostile = _HostileInt(42)
    _HostileInt.calls = 0
    result = _finite_number(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileInt.calls == 0


def test_finite_number_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(3.14)
    _HostileFloat.calls = 0
    result = _finite_number(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileFloat.calls == 0


def test_finite_number_rejects_bool() -> None:
    assert _finite_number(True) is None
    assert _finite_number(False) is None


def test_finite_number_benign_still_works() -> None:
    assert _finite_number(1) == 1.0
    assert _finite_number(1.5) == 1.5
    assert _finite_number(0) == 0.0
    assert _finite_number("bad") is None  # type: ignore[arg-type]
    assert _finite_number(None) is None  # type: ignore[arg-type]
    assert _finite_number(math.inf) is None
    assert _finite_number(-math.inf) is None
    assert _finite_number(math.nan) is None


def test_strict_int_rejects_hostile_int_before_return() -> None:
    hostile = _HostileInt(7)
    _HostileInt.calls = 0
    result = _strict_int(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileInt.calls == 0


def test_strict_int_rejects_bool() -> None:
    assert _strict_int(True) is None
    assert _strict_int(False) is None


def test_strict_int_benign_still_works() -> None:
    assert _strict_int(7) == 7
    assert _strict_int(0) == 0
    assert _strict_int(-3) == -3
    assert _strict_int("bad") is None  # type: ignore[arg-type]
    assert _strict_int(1.5) is None  # type: ignore[arg-type]
    assert _strict_int(None) is None  # type: ignore[arg-type]
