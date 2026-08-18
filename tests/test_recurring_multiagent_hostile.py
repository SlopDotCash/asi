"""Hostile int/float gate for RecurringTwoAgentWorld before range/float."""

from __future__ import annotations

import pytest

from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentWorld

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile le must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_recurring_world_rejects_hostile_int_before_range() -> None:
    hostile = _HostileInt(64)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="context_length must be positive"):
        RecurringTwoAgentWorld(context_length=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    hostile2 = _HostileInt(4)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="nuisance_dim must be non-negative"):
        RecurringTwoAgentWorld(nuisance_dim=hostile2)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    with pytest.raises(ValueError, match="context_length must be positive"):
        RecurringTwoAgentWorld(context_length=True)  # type: ignore[arg-type]


def test_recurring_world_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="nuisance_scale must be a finite real number"):
        RecurringTwoAgentWorld(nuisance_scale=hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0

    hostile_int = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="nuisance_scale must be a finite real number"):
        RecurringTwoAgentWorld(nuisance_scale=hostile_int)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_recurring_world_benign_still_works() -> None:
    world = RecurringTwoAgentWorld()
    assert world is not None
    world2 = RecurringTwoAgentWorld(context_length=32, nuisance_dim=2, nuisance_scale=0.5)
    assert world2 is not None

    with pytest.raises(ValueError, match="must be finite"):
        RecurringTwoAgentWorld(nuisance_scale=float("inf"))  # type: ignore[arg-type]


def test_recurring_world_hostile_not_in_repr() -> None:
    hostile = _HostileInt(64)
    _HostileInt.calls = 0
    try:
        RecurringTwoAgentWorld(context_length=hostile)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "_HostileInt" not in str(exc)
        assert _HostileInt.calls == 0
    else:
        raise AssertionError("should have raised")
