"""#1383-complete update working-set preflight for stacked Horde."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    StackedLinearHorde,
    _preflight_stacked_horde_update_working_set,
    _stacked_horde_persistent_bytes,
    _stacked_horde_update_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_FEATURE_DIM = 90_000_000
_LAST_FIT_FEATURE_DIM = 89_478_484
_FIRST_OVERFLOW_FEATURE_DIM = 89_478_485


def _unit_config(feature_dim: int) -> StackedHordeConfig:
    return StackedHordeConfig(
        n_demons=1,
        feature_dim=feature_dim,
        gammas=(0.9,),
        lamdas=(0.8,),
        cumulant_indices=(0,),
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _stacked_horde_persistent_bytes(1, _OVERFLOW_FEATURE_DIM)
    working_set_bytes = _stacked_horde_update_working_set_bytes(1, _OVERFLOW_FEATURE_DIM)
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_004
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_config(_OVERFLOW_FEATURE_DIM)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_LAST_FIT_FEATURE_DIM, _FIRST_OVERFLOW_FEATURE_DIM + 2):
        persist_bytes = _stacked_horde_persistent_bytes(1, feature_dim)
        working_set_bytes = _stacked_horde_update_working_set_bytes(1, feature_dim)
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
    cfg = _unit_config(last_fit)
    assert cfg.feature_dim == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        _unit_config(first_overflow)
    _preflight_stacked_horde_update_working_set(1, last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_stacked_horde_update_working_set(1, first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_stacked_horde_update_working_set(1, _OVERFLOW_FEATURE_DIM)


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (_INT32_MAX - 26) // 8
    with pytest.raises(ValueError, match="aggregate bytes"):
        _unit_config(last_legal + 1)


def test_legal_small_stacked_horde_still_updates() -> None:
    persist_bytes = _stacked_horde_persistent_bytes(1, 4)
    assert persist_bytes == 36
    horde = StackedLinearHorde(_unit_config(4))
    state = horde.init()
    horde.update(
        state,
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )
