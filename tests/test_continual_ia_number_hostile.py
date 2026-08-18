"""Hostile int gate for continual_ia _number before float."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import _number

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

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_number_rejects_hostile_int_before_float() -> None:
    hostile = _HostileInt(42)
    _HostileInt.calls = 0
    result = _number(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileInt.calls == 0


def test_number_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(3.14)
    _HostileFloat.calls = 0
    result = _number(hostile)  # type: ignore[arg-type]
    assert result is None
    assert _HostileFloat.calls == 0


def test_number_rejects_bool() -> None:
    assert _number(True) is None
    assert _number(False) is None
    assert _HostileInt.calls == 0


def test_number_benign_still_works() -> None:
    assert _number(1) == 1.0
    assert _number(1.5) == 1.5
    assert _number(0) == 0.0
    assert _number("bad") is None  # type: ignore[arg-type]
    assert _number(None) is None  # type: ignore[arg-type]
    import math

    assert _number(float("inf")) is None  # type: ignore[arg-type]
    assert _number(math.nan) is None  # type: ignore[arg-type]


def test_number_non_finite_rejected() -> None:
    import math

    assert _number(math.inf) is None
    assert _number(-math.inf) is None
