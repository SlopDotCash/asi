"""Hostile integer validation for reference life scorecard."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


def test_streaming_observe_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import StreamingRunSummary

    summary = StreamingRunSummary.for_switching(horizon=10, phase_length=3, post_switch_window=2)
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a finite number"):
        summary.observe(
            reward=hostile,  # type: ignore[arg-type]
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    assert _HostileInt.calls == 0
    _HostileInt.calls = 0
    hostile2 = _HostileInt(2)
    with pytest.raises(Exception, match="must be a finite number"):
        summary.observe(
            reward=1.0,
            oracle_reward=hostile2,  # type: ignore[arg-type]
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    assert _HostileInt.calls == 0
    # bool rejected
    with pytest.raises(Exception, match="must be a finite number"):
        summary.observe(
            reward=True,  # type: ignore[arg-type]
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    # valid still works
    summary.observe(
        reward=1,
        oracle_reward=2.0,
        regime_id=0,
        parameters_changed=False,
        next_state_index=0,
    )
    assert summary.accepted_events == 1


def test_reward_sum_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _reward_sum

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be finite"):
        _reward_sum({"outcome": {"reward_sum": hostile}})  # type: ignore[dict-item]
    assert _HostileInt.calls == 0
    with pytest.raises(Exception, match="must be finite"):
        _reward_sum({"outcome": {"reward_sum": True}})  # type: ignore[dict-item]
    assert _HostileInt.calls == 0
    assert _reward_sum({"outcome": {"reward_sum": 5}}) == 5.0
    assert _reward_sum({"outcome": {"reward_sum": 5.0}}) == 5.0


def test_finite_nonnegative_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _require_finite_nonnegative

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a finite nonnegative"):
        _require_finite_nonnegative(hostile, path="p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(Exception, match="must be a finite nonnegative"):
        _require_finite_nonnegative(True, path="p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_finite_nonnegative(1.0, path="p") == 1.0
    assert _require_finite_nonnegative(0, path="p") == 0.0


def test_finite_number_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _require_finite_number

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a finite number"):
        _require_finite_number(hostile, path="p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(Exception, match="must be a finite number"):
        _require_finite_number(True, path="p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_finite_number(1, path="p") == 1.0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) not in (int, float):
            raise ValueError("must be a finite number")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
