"""#1383-complete update working-set preflight for interaction features."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    _interaction_persistent_bytes,
    _interaction_update_working_set_bytes,
    _preflight_interaction_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_FEATURES = 20_000_000
_LAST_FIT_FEATURES = 11_243_368
_FIRST_OVERFLOW_FEATURES = 11_243_369
_UNIT = {"n_tasks": 1, "candidate_count": 0, "scale_robust": False}


def _unit_persist_bytes(n_features: int) -> int:
    return _interaction_persistent_bytes(n_features=n_features, **_UNIT)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _unit_persist_bytes(_OVERFLOW_FEATURES)
    working_set_bytes = _interaction_update_working_set_bytes(
        n_features=_OVERFLOW_FEATURES,
        **_UNIT,
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 900_000_044
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURES <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        FixedBudgetInteractionLearner(n_features=_OVERFLOW_FEATURES, **_UNIT)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for n_features in range(_LAST_FIT_FEATURES, _FIRST_OVERFLOW_FEATURES + 2):
        persist_bytes = _unit_persist_bytes(n_features)
        working_set_bytes = _interaction_update_working_set_bytes(
            n_features=n_features,
            **_UNIT,
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * n_features <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = n_features
        elif first_overflow is None:
            first_overflow = n_features
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_FEATURES
    learner = FixedBudgetInteractionLearner(n_features=last_fit, **_UNIT)
    assert learner.n_features == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        FixedBudgetInteractionLearner(n_features=first_overflow, **_UNIT)
    _preflight_interaction_update_working_set(n_features=last_fit, **_UNIT)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_interaction_update_working_set(n_features=first_overflow, **_UNIT)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_interaction_update_working_set(
            n_features=_OVERFLOW_FEATURES,
            **_UNIT,
        )


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (_INT32_MAX - 44) // 45
    with pytest.raises(ValueError, match="state byte count"):
        FixedBudgetInteractionLearner(n_features=last_legal + 1, **_UNIT)


def test_legal_small_interaction_learner_still_updates() -> None:
    persist_bytes = _unit_persist_bytes(4)
    assert persist_bytes == 224
    learner = FixedBudgetInteractionLearner(n_features=4, **_UNIT)
    state = learner.init(2, jr.key(0))
    learner.update(
        state,
        jnp.asarray((0.5, -0.25), dtype=jnp.float32),
        jnp.asarray((0.75,), dtype=jnp.float32),
    )
