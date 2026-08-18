"""#1383-complete update working-set preflight for STOMP."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    SubtaskSpec,
    _preflight_stomp_update_working_set,
    _stomp_direct_state_bytes,
    _stomp_update_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_OBS = 20_000
_LAST_FIT_OBS = 13_373
_FIRST_OVERFLOW_OBS = 13_374
_UNIT = {
    "n_primitive_actions": 1,
    "base_hidden_sizes": (),
}


def _unit_stomp(observation_dim: int) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=observation_dim,
        **_UNIT,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def _unit_persist_bytes(observation_dim: int) -> int:
    return 4 * (observation_dim * observation_dim + 8 * observation_dim + 37)


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _unit_persist_bytes(_OVERFLOW_OBS)
    working_set_bytes = 3 * persist_bytes + 64
    extras_bytes = 64
    assert persist_bytes == 1_600_640_148
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_OBS <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_stomp(_OVERFLOW_OBS)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for observation_dim in range(_LAST_FIT_OBS, _FIRST_OVERFLOW_OBS + 2):
        persist_bytes = _unit_persist_bytes(observation_dim)
        working_set_bytes = 3 * persist_bytes + 64
        extras_bytes = 64
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * observation_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = observation_dim
        elif first_overflow is None:
            first_overflow = observation_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_OBS
    last_stomp = _unit_stomp(last_fit)
    assert last_stomp.observation_dim == last_fit
    assert _stomp_direct_state_bytes(last_stomp) == _unit_persist_bytes(last_fit)
    assert _stomp_update_working_set_bytes(last_stomp) <= _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_stomp(first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    last_fit = _unit_stomp(_LAST_FIT_OBS)
    object.__setattr__(last_fit, "observation_dim", _OVERFLOW_OBS)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_stomp_update_working_set(last_fit)


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (2**29 - 1 - 22) // 4
    with pytest.raises(ValueError, match="direct array bytes"):
        STOMPConfig(observation_dim=last_legal + 1, n_primitive_actions=1)


def test_legal_small_stomp_still_constructs() -> None:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=4,
        n_primitive_actions=2,
        base_hidden_sizes=(),
    )
    persist_bytes = _stomp_direct_state_bytes(stomp)
    assert persist_bytes == 420
    agent = STOMPAgent(stomp)
    state = agent.init(jr.key(0))
    agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
