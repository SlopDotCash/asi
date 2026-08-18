"""#1383-complete update working-set preflight for the OaK host."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    _oak_direct_state_bytes,
    _oak_update_working_set_bytes,
    _preflight_oak_update_working_set,
)
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_OBS = 20_000
_LAST_FIT_OBS = 13_373
_FIRST_OVERFLOW_OBS = 13_374
_UNIT_STOMP = {
    "n_primitive_actions": 1,
    "base_hidden_sizes": (),
}


def _unit_stomp(observation_dim: int) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=observation_dim,
        **_UNIT_STOMP,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    stomp = _unit_stomp(_OVERFLOW_OBS)
    persist_bytes = _oak_direct_state_bytes(stomp)
    working_set_bytes = _oak_update_working_set_bytes(stomp)
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 1_600_640_172
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_OBS <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        OaKConfig(stomp=stomp)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for observation_dim in range(_LAST_FIT_OBS, _FIRST_OVERFLOW_OBS + 2):
        stomp = _unit_stomp(observation_dim)
        persist_bytes = _oak_direct_state_bytes(stomp)
        working_set_bytes = _oak_update_working_set_bytes(stomp)
        extras_bytes = working_set_bytes - 3 * persist_bytes
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
    config = OaKConfig(stomp=_unit_stomp(last_fit))
    assert config.observation_dim == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        OaKConfig(stomp=_unit_stomp(first_overflow))


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_oak_update_working_set(_unit_stomp(_OVERFLOW_OBS))


def test_persist_bound_still_fires_before_working_set() -> None:
    stomp_only_limit = (2**29 - 1 - 22) // 4
    stomp = STOMPConfig(observation_dim=stomp_only_limit, n_primitive_actions=1)
    with pytest.raises(ValueError, match="OaK direct array bytes"):
        OaKConfig(stomp=stomp)


def test_legal_small_oak_still_constructs() -> None:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=4,
        n_primitive_actions=2,
        base_hidden_sizes=(),
    )
    persist_bytes = _oak_direct_state_bytes(stomp)
    assert persist_bytes == 444
    agent = OaKAgent(OaKConfig(stomp=stomp))
    state = agent.init(jr.key(0))
    agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
