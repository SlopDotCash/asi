"""#1383-complete update working-set preflight for the multi-head MLP learner."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    _direct_state_bytes,
    _multi_head_update_working_set_bytes,
    _preflight_multi_head_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_HIDDEN = 30_000_000
_LAST_FIT_HIDDEN = 25_565_280
_FIRST_OVERFLOW_HIDDEN = 25_565_281
_OVERFLOW_FEATURE_DIM = 90_000_000
_LAST_FIT_FEATURE_DIM = 89_478_480
_FIRST_OVERFLOW_FEATURE_DIM = 89_478_481


def _unit_persist_bytes(hidden: int) -> int:
    return _direct_state_bytes(1, (hidden,), 1)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _unit_persist_bytes(_OVERFLOW_HIDDEN)
    working_set_bytes = _multi_head_update_working_set_bytes(
        1, (_OVERFLOW_HIDDEN,), 1
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 840_000_020
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_HIDDEN <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        MultiHeadMLPLearner(n_heads=1, hidden_sizes=(_OVERFLOW_HIDDEN,))


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for hidden in range(_LAST_FIT_HIDDEN, _FIRST_OVERFLOW_HIDDEN + 2):
        persist_bytes = _unit_persist_bytes(hidden)
        working_set_bytes = _multi_head_update_working_set_bytes(1, (hidden,), 1)
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * hidden <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = hidden
        elif first_overflow is None:
            first_overflow = hidden
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_HIDDEN
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(last_fit,))
    assert learner._hidden_sizes == (last_fit,)
    with pytest.raises(ValueError, match="update working set byte count"):
        MultiHeadMLPLearner(n_heads=1, hidden_sizes=(first_overflow,))


def test_init_rejects_linear_feature_dim_working_set_before_allocation() -> None:
    persist_bytes = _direct_state_bytes(1, (), _OVERFLOW_FEATURE_DIM)
    working_set_bytes = _multi_head_update_working_set_bytes(
        1, (), _OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_020
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=())
    with pytest.raises(ValueError, match="update working set byte count"):
        learner.init(_OVERFLOW_FEATURE_DIM, jr.key(0))


def test_init_last_fit_and_first_overflow_feature_dims_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(
        _LAST_FIT_FEATURE_DIM, _FIRST_OVERFLOW_FEATURE_DIM + 2
    ):
        persist_bytes = _direct_state_bytes(1, (), feature_dim)
        working_set_bytes = _multi_head_update_working_set_bytes(
            1, (), feature_dim
        )
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
    _preflight_multi_head_update_working_set(1, (), last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_multi_head_update_working_set(1, (), first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_multi_head_update_working_set(1, (_OVERFLOW_HIDDEN,), 1)


def test_persist_bound_still_fires_before_working_set() -> None:
    with pytest.raises(ValueError, match="direct_state_bytes"):
        MultiHeadMLPLearner(n_heads=134_217_728, hidden_sizes=())


def test_legal_small_multi_head_learner_still_constructs() -> None:
    persist_bytes = _direct_state_bytes(2, (4,), 3)
    assert persist_bytes == 236
    learner = MultiHeadMLPLearner(
        n_heads=2, hidden_sizes=(4,), sparsity=0.0, use_layer_norm=False
    )
    state = learner.init(feature_dim=3, key=jr.key(0))
    learner.update(
        state,
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
