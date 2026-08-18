"""#1383-complete update working-set preflight for differential TD."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.average_reward import (
    DifferentialTDLearner,
    _differential_td_persistent_bytes,
    _differential_td_update_working_set_bytes,
    _preflight_differential_td_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_FEATURE_DIM = 90_000_000
_LAST_FIT_FEATURE_DIM = 89_478_481
_FIRST_OVERFLOW_FEATURE_DIM = 89_478_482


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _differential_td_persistent_bytes(_OVERFLOW_FEATURE_DIM)
    working_set_bytes = _differential_td_update_working_set_bytes(_OVERFLOW_FEATURE_DIM)
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_016
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        DifferentialTDLearner().init(_OVERFLOW_FEATURE_DIM)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_LAST_FIT_FEATURE_DIM, _FIRST_OVERFLOW_FEATURE_DIM + 2):
        persist_bytes = _differential_td_persistent_bytes(feature_dim)
        working_set_bytes = _differential_td_update_working_set_bytes(feature_dim)
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_FEATURE_DIM
    _preflight_differential_td_update_working_set(last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_differential_td_update_working_set(first_overflow)
    with pytest.raises(ValueError, match="update working set byte count"):
        DifferentialTDLearner().init(first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_differential_td_update_working_set(_OVERFLOW_FEATURE_DIM)


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = ((2**31 - 1) // 4 - 4) // 2
    with pytest.raises(ValueError, match="differential TD state bytes"):
        DifferentialTDLearner().init(last_legal + 1)


def test_legal_small_differential_td_still_updates() -> None:
    persist_bytes = _differential_td_persistent_bytes(5)
    assert persist_bytes == 56
    learner = DifferentialTDLearner()
    state = learner.init(5)
    observation = jnp.zeros((5,), dtype=jnp.float32)
    learner.update(
        state,
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        observation,
    )
