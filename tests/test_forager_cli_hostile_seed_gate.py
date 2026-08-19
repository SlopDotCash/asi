"""Hostile int identity gates for the forager CLI evaluation-seed interval
before comparison.

_protocol_evaluation_seeds gates evaluation_seed_start / evaluation_seeds with
isinstance before the trusted `start < 0` / `count < 1` comparisons, so a
hostile int subclass passes the gate and its overridden __lt__ runs during
validation.
"""

from __future__ import annotations

import pytest

from alberta_framework.forager_cli import _protocol_evaluation_seeds

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile le must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __add__(self, other: object) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile add must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _Protocol:
    def __init__(self, start: object, count: object) -> None:
        self.evaluation_seed_start = start
        self.evaluation_seeds = count


def test_seed_interval_rejects_hostile_start_before_lt() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(_HostileInt(100), 3))
    assert _HostileInt.calls == 0


def test_seed_interval_rejects_hostile_count_before_lt() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(100, _HostileInt(3)))
    assert _HostileInt.calls == 0


def test_seed_interval_rejects_bool_and_accepts_benign() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(True, 3))
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(100, False))
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(-1, 3))
    with pytest.raises(ValueError, match="invalid evaluation seed interval"):
        _protocol_evaluation_seeds(_Protocol(100, 0))
    assert _protocol_evaluation_seeds(_Protocol(100, 3)) == (100, 101, 102)
    assert _HostileInt.calls == 0
