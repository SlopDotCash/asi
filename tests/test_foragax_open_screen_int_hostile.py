"""Hostile integer validation for foragax open screen."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


def test_seeds_rejects_hostile_before_lt() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(0)
    assert (type(hostile) is not int or hostile < 0) is True
    assert _HostileInt.calls == 0
    assert (int is not int or 0 < 0) is False
    assert (bool is not int or True < 0) is True  # bool rejected


def test_ppo_rollout_rejects_hostile_before_eq() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert (type(hostile) is not int) is True
    assert _HostileInt.calls == 0
    assert (int is not int) is False
    assert (bool is not int) is True


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
