"""Hostile int/float gate for recurring_feature _finite_number before float."""

from __future__ import annotations

import math

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import _finite_number

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
