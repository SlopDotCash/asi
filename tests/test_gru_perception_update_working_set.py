"""#1383-complete update working-set preflight for GRU perception."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.prototype_agent import (
    GRUPerceptionConfig,
    _gru_perception_persistent_bytes,
    _gru_perception_update_working_set_bytes,
    _gru_step,
    _init_gru_state,
    _preflight_gru_perception_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_OBSERVATION_DIM = 60_000_000
_LAST_FIT_OBSERVATION_DIM = 53_687_088
_FIRST_OVERFLOW_OBSERVATION_DIM = 53_687_089
_UNIT_HIDDEN_DIM = 1


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _gru_perception_persistent_bytes(
        _OVERFLOW_OBSERVATION_DIM, _UNIT_HIDDEN_DIM
    )
    working_set_bytes = _gru_perception_update_working_set_bytes(
        _OVERFLOW_OBSERVATION_DIM, _UNIT_HIDDEN_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_028
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_OBSERVATION_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        GRUPerceptionConfig(
            observation_dim=_OVERFLOW_OBSERVATION_DIM,
            hidden_dim=_UNIT_HIDDEN_DIM,
        )


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for observation_dim in range(
        _LAST_FIT_OBSERVATION_DIM, _FIRST_OVERFLOW_OBSERVATION_DIM + 2
    ):
        persist_bytes = _gru_perception_persistent_bytes(
            observation_dim, _UNIT_HIDDEN_DIM
        )
        working_set_bytes = _gru_perception_update_working_set_bytes(
            observation_dim, _UNIT_HIDDEN_DIM
        )
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
    assert last_fit == _LAST_FIT_OBSERVATION_DIM
    cfg = GRUPerceptionConfig(
        observation_dim=last_fit, hidden_dim=_UNIT_HIDDEN_DIM
    )
    assert cfg.observation_dim == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        GRUPerceptionConfig(
            observation_dim=first_overflow, hidden_dim=_UNIT_HIDDEN_DIM
        )
    _preflight_gru_perception_update_working_set(last_fit, _UNIT_HIDDEN_DIM)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_gru_perception_update_working_set(
            first_overflow, _UNIT_HIDDEN_DIM
        )


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_gru_perception_update_working_set(
            _OVERFLOW_OBSERVATION_DIM, _UNIT_HIDDEN_DIM
        )


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (_INT32_MAX - 28) // 12
    with pytest.raises(ValueError, match="GRUPerception state byte count"):
        GRUPerceptionConfig(
            observation_dim=last_legal + 1, hidden_dim=_UNIT_HIDDEN_DIM
        )
    with pytest.raises(ValueError, match="fit signed int32"):
        GRUPerceptionConfig(observation_dim=1_000_000, hidden_dim=1_000_000)


def test_legal_small_gru_perception_still_steps() -> None:
    persist_bytes = _gru_perception_persistent_bytes(4, 1)
    assert persist_bytes == 76
    cfg = GRUPerceptionConfig(observation_dim=4, hidden_dim=1)
    state = _init_gru_state(cfg, jr.key(0))
    _gru_step(state, jnp.zeros((4,), dtype=jnp.float32))
