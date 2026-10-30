"""#1383-complete update working-set preflight for stacked Horde."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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
_LAST_FIT_FEATURE_DIM = 89_478_483
_FIRST_OVERFLOW_FEATURE_DIM = 89_478_484


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


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_stacked_horde_recovers_finite_td_cancellation(sign: float) -> None:
    horde = StackedLinearHorde(StackedHordeConfig(
        n_demons=3, feature_dim=1, gammas=(1.0,) * 3, lamdas=(0.0,) * 3,
        cumulant_indices=(0, 1, 2), step_size=0.1,
    ))
    state = horde.init().replace(
        weights=jnp.asarray([[sign * 2e38], [1e-20], [0.0]], dtype=jnp.float32),
    )
    features = jnp.ones(1, dtype=jnp.float32)
    cumulants = jnp.asarray([sign * 2e38, 1e-20, 1.0], dtype=jnp.float32)
    expected = np.asarray(state.weights, np.float64) + float(np.float32(0.1)) * (
        np.asarray(cumulants, np.float64)[:, None]
    )
    with jax.disable_jit():
        eager = horde.update(state, features, features, cumulants)
    compiled = jax.jit(horde.update)(state, features, features, cumulants)
    for result in (eager, compiled):
        np.testing.assert_array_equal(result.head_updates_applied, [True, True, True])
        np.testing.assert_allclose(result.state.weights, expected, rtol=2e-6, atol=0.0)
        np.testing.assert_array_equal(result.td_errors, cumulants)
        np.testing.assert_array_equal(result.predictions, state.weights[:, 0])
        np.testing.assert_array_equal(result.state.traces, np.ones((3, 1)))
        assert bool(result.update_applied) and int(result.state.step_count) == 1


@pytest.mark.parametrize("next_feature, alpha", [(0.0, 0.1)])
def test_stacked_horde_true_td_or_weight_overflow_still_rejects(next_feature, alpha) -> None:
    horde = StackedLinearHorde(StackedHordeConfig(
        n_demons=1, feature_dim=1, gammas=(1.0,), lamdas=(0.0,),
        cumulant_indices=(0,), step_size=alpha,
    ))
    weight = -2e38 if next_feature == 0.0 else 2e38
    state = horde.init().replace(weights=jnp.asarray([[weight]], dtype=jnp.float32))
    features = jnp.ones(1, dtype=jnp.float32)
    cumulants = jnp.asarray([2e38], dtype=jnp.float32)
    with jax.disable_jit():
        eager = horde.update(state, features, features * next_feature, cumulants)
    compiled = jax.jit(horde.update)(state, features, features * next_feature, cumulants)
    for result in (eager, compiled):
        assert not bool(result.update_applied) and not bool(result.head_updates_applied[0])
        for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(result.state)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(result.td_errors, [0.0])
