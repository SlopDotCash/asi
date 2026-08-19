"""Hostile int/float identity gate for the reference-life RiverSwim checkpoint
oracle before float() conversion.

reference_life.validate_checkpoint_state reads oracle_average_reward from the
environment manifest with an isinstance gate before float(oracle_value); a
hostile subclass passes the gate and its overridden __float__ runs during
checkpoint validation.
"""

from __future__ import annotations

import math

import pytest

from alberta_framework.reference_life import (
    DecisionOwnershipError,
    _validate_riverswim_oracle,
)

pytestmark = pytest.mark.unit


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _reset() -> None:
    _HostileFloat.calls = 0
    _HostileInt.calls = 0


def test_oracle_rejects_hostile_float_before_float() -> None:
    _reset()
    with pytest.raises(DecisionOwnershipError, match="RiverSwim oracle"):
        _validate_riverswim_oracle(_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_oracle_rejects_hostile_int_before_float() -> None:
    _reset()
    with pytest.raises(DecisionOwnershipError, match="RiverSwim oracle"):
        _validate_riverswim_oracle(_HostileInt(1))
    assert _HostileInt.calls == 0


def test_oracle_rejects_non_finite_and_non_numeric() -> None:
    _reset()
    for bad in (True, False, None, "1.0", math.inf, math.nan):
        with pytest.raises(DecisionOwnershipError, match="RiverSwim oracle"):
            _validate_riverswim_oracle(bad)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_oracle_accepts_benign_numeric() -> None:
    _reset()
    assert _validate_riverswim_oracle(1) == 1.0
    assert _validate_riverswim_oracle(0.5) == 0.5
    assert _HostileFloat.calls == 0
