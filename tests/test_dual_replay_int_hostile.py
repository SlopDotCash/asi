"""Hostile integer validation for dual replay."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")

    def __gt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile gt")

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")

    def __hash__(self) -> int:
        return int.__hash__(self)


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")


def test_reservoir_capacity_rejects_hostile_before_lt() -> None:
    import jax.numpy as jnp
    import jax.random as jr

    from alberta_framework.core.dual_replay import reservoir_selection

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    key = jr.PRNGKey(0)
    with pytest.raises(Exception, match="positive integer"):
        reservoir_selection(key, jnp.asarray(1, dtype=jnp.int32), hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # bool rejected
    with pytest.raises(Exception, match="positive integer"):
        reservoir_selection(key, jnp.asarray(1, dtype=jnp.int32), True)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_checkpoint_array_real_rejects_hostile_before_float() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must contain only JSON real numbers"):
        DualReplayMemory._checkpoint_array([[hostile]], name="x", shape=(1, 1), dtype=np.float32)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # also hostile float subclass
    hf = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(Exception, match="must contain only JSON real numbers"):
        DualReplayMemory._checkpoint_array([[hf]], name="x", shape=(1, 1), dtype=np.float32)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    # valid
    arr = DualReplayMemory._checkpoint_array([[1.0]], name="x", shape=(1, 1), dtype=np.float32)
    assert arr.shape == (1, 1)


def test_checkpoint_array_int_rejects_hostile_before_int_check() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must contain only JSON integers"):
        DualReplayMemory._checkpoint_array([[hostile]], name="x", shape=(1, 1), dtype=np.int32)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(Exception, match="must contain only JSON integers"):
        DualReplayMemory._checkpoint_array([[True]], name="x", shape=(1, 1), dtype=np.int32)  # type: ignore[arg-type]
    arr = DualReplayMemory._checkpoint_array([[1]], name="x", shape=(1, 1), dtype=np.int32)
    assert arr.shape == (1, 1)


def test_checkpoint_counter_rejects_hostile_before_range() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a JSON integer"):
        DualReplayMemory._checkpoint_counter(hostile, name="c")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(Exception, match="must be a JSON integer"):
        DualReplayMemory._checkpoint_counter(True, name="c")  # type: ignore[arg-type]
    assert DualReplayMemory._checkpoint_counter(1, name="c").shape == ()


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be a JSON integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
