"""Hostile string gate for cora_development _run_arm before membership."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.cora_development import _run_arm

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_run_arm_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("replay_q")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        _run_arm(0, 1, 1, hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_run_arm_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _run_arm(0, 1, 1, 123)  # type: ignore[arg-type]


def test_run_arm_benign_passes() -> None:
    res = _run_arm(15810, 1, 4, "replay_q")
    assert res.arm_id == "replay_q"
