"""Hostile int/float identity gates for the remaining artifact numeric helpers.

Mirrors tests/test_continual_ia_number_hostile.py, which covers the already
hardened continual_ia _number. The gates here still used isinstance before
#1944; a hostile subclass passes the gate and its overridden dunder then runs
during trusted numeric conversion.
"""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import _integer
from alberta_framework.evaluation.continual_multiagent_artifact import (
    _finite_number as _ma_finite_number,
)
from alberta_framework.evaluation.ftl_decision_artifact import _finite_number as _ftl_finite_number
from alberta_framework.evaluation.recurring_feature_artifact import (
    _finite_number as _rf_finite_number,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _finite_number as _srf_finite_number,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import _strict_int

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


def _reset() -> None:
    _HostileInt.calls = 0
    _HostileFloat.calls = 0


@pytest.mark.parametrize(
    "gate",
    [
        _ma_finite_number,
        _ftl_finite_number,
        _rf_finite_number,
        _srf_finite_number,
    ],
)
def test_finite_number_rejects_hostile_int_before_float(gate) -> None:
    _reset()
    result = gate(_HostileInt(42))  # type: ignore[arg-type]
    assert result is None
    assert _HostileInt.calls == 0


@pytest.mark.parametrize(
    "gate",
    [
        _ma_finite_number,
        _ftl_finite_number,
        _rf_finite_number,
        _srf_finite_number,
    ],
)
def test_finite_number_rejects_hostile_float_before_float(gate) -> None:
    _reset()
    result = gate(_HostileFloat(3.14))  # type: ignore[arg-type]
    assert result is None
    assert _HostileFloat.calls == 0


@pytest.mark.parametrize(
    "gate",
    [
        _ma_finite_number,
        _ftl_finite_number,
        _rf_finite_number,
        _srf_finite_number,
    ],
)
def test_finite_number_rejects_bool_and_accepts_benign(gate) -> None:
    _reset()
    assert gate(True) is None
    assert gate(False) is None
    assert gate(1) == 1.0
    assert gate(1.5) == 1.5
    assert gate("bad") is None  # type: ignore[arg-type]
    assert gate(None) is None  # type: ignore[arg-type]
    import math

    assert gate(math.inf) is None
    assert gate(math.nan) is None


@pytest.mark.parametrize(
    "gate",
    [
        lambda v: _strict_int(v),
        lambda v: _integer(v, location="x", errors=[]),
    ],
)
def test_integer_gate_rejects_hostile_int(gate) -> None:
    _reset()
    result = gate(_HostileInt(42))  # type: ignore[arg-type]
    assert result is None
    assert _HostileInt.calls == 0


@pytest.mark.parametrize(
    "gate",
    [
        lambda v: _strict_int(v),
        lambda v: _integer(v, location="x", errors=[]),
    ],
)
def test_integer_gate_rejects_bool_and_accepts_benign(gate) -> None:
    _reset()
    assert gate(True) is None
    assert gate(False) is None
    assert gate(0) == 0
    assert gate(42) == 42
    assert gate(1.5) is None  # type: ignore[arg-type]
    assert gate("bad") is None  # type: ignore[arg-type]
    assert gate(None) is None  # type: ignore[arg-type]
