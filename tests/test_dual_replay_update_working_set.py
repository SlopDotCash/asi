"""#1383-complete record/sample working-set preflight for dual replay."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree_util

from alberta_framework.core.dual_replay import (
    DualReplayConfig,
    DualReplayMemory,
    _allocation_sizes,
    _dual_replay_update_working_set_bytes,
    _preflight_dual_replay_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_CAPACITY = 10_000_000
_LAST_FIT_CAPACITY = 9_061_110
_FIRST_OVERFLOW_CAPACITY = 9_061_111
_UNIT_KWARGS = {
    "observation_dim": 1,
    "action_dim": 1,
    "batch_size": 2,
}


def _unit_config(total_capacity: int) -> DualReplayConfig:
    return DualReplayConfig(
        total_capacity=total_capacity,
        short_term_capacity=1,
        observation_dim=1,
        action_dim=1,
        short_term_sample_size=1,
        long_term_sample_size=1,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_dual_replay_persist_matches_jit_and_width_still_fits() -> None:
    config = DualReplayConfig(
        total_capacity=7,
        short_term_capacity=3,
        observation_dim=2,
        action_dim=2,
        short_term_sample_size=2,
        long_term_sample_size=2,
    )
    memory = DualReplayMemory(config)
    state = memory.init(jr.key(3))
    actual = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in tree_util.tree_leaves(state)
    )
    slot_bytes, persist_bytes = _allocation_sizes(config)
    assert actual == persist_bytes == memory.persistent_bytes == 697
    assert slot_bytes == memory.slot_bytes == 91
    overflow_persist = _OVERFLOW_CAPACITY * 79 + 60
    overflow_sample_extras = 2 * (79 + 13) + 28
    working_set_bytes = _dual_replay_update_working_set_bytes(
        total_capacity=_OVERFLOW_CAPACITY,
        **_UNIT_KWARGS,
    )
    assert overflow_persist <= _INT32_MAX
    assert overflow_persist + overflow_sample_extras <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_config(_OVERFLOW_CAPACITY)


def test_dual_replay_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for total_capacity in range(_LAST_FIT_CAPACITY, _FIRST_OVERFLOW_CAPACITY + 2):
        persist_bytes = total_capacity * 79 + 60
        working_set_bytes = _dual_replay_update_working_set_bytes(
            total_capacity=total_capacity,
            **_UNIT_KWARGS,
        )
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + (2 * (79 + 13) + 28) <= _INT32_MAX
        assert 4 * 1 <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = total_capacity
        elif first_overflow is None:
            first_overflow = total_capacity
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_CAPACITY
    constructed = _unit_config(last_fit)
    assert constructed.total_capacity == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_config(first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_dual_replay_update_working_set(
            total_capacity=_OVERFLOW_CAPACITY,
            **_UNIT_KWARGS,
        )


def test_legal_small_dual_replay_still_constructs() -> None:
    config = DualReplayConfig(
        total_capacity=6,
        short_term_capacity=3,
        observation_dim=2,
        action_dim=2,
        short_term_sample_size=2,
        long_term_sample_size=2,
    )
    DualReplayMemory(config).init(jr.key(1))
