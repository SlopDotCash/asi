"""Saturating lifetime clocks keep learner update identity at int32 exhaustion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.learners import (
    LinearLearner,
    MLPLearner,
    TDLinearLearner,
    TrueOnlineTDLearner,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _wrap_clock() -> jnp.ndarray:
    return jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)


def _committed_update_identity(step_count: jnp.ndarray) -> tuple[bool, bool]:
    """Ensemble/gate contracts that bind the committed learner clock.

    ``WorldModelEnsemble._member_state_valid`` requires ``step_count >= 0``
    and ``learner_state.step_count ==`` the saturated member clock.
    ``recurring_feature_gate`` publishes ``steps_seen=int(step_count)``.
    """
    ensemble_clock = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    non_negative = bool(step_count >= 0)
    matches_saturated_member = bool(step_count == ensemble_clock)
    return non_negative, matches_saturated_member


def _assert_exhausted(result: Any, planted: Any) -> None:
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, planted)
    chex.assert_trees_all_equal(result.prediction, jnp.zeros_like(result.prediction))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_int32_wrap_forges_a_different_update_identity() -> None:
    wrap = _wrap_clock()
    assert int(wrap) == _INT32_MIN
    wrap_nonneg, wrap_matches = _committed_update_identity(wrap)
    sat_nonneg, sat_matches = _committed_update_identity(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    assert sat_nonneg and sat_matches
    assert not wrap_nonneg
    assert not wrap_matches
    assert int(wrap) != _INT32_MAX


def test_linear_learner_clock_saturates_and_keeps_update_identity() -> None:
    learner = LinearLearner()
    predecessor = learner.init(2).replace(
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    result = learner.update(
        predecessor,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches
    exhausted = learner.update(
        result.state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    _assert_exhausted(exhausted, result.state)


def test_mlp_learner_clock_saturates_and_keeps_update_identity() -> None:
    learner = MLPLearner(hidden_sizes=(2,), use_layer_norm=False, sparsity=0.0)
    predecessor = learner.init(2, jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    result = learner.update(
        predecessor,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches
    exhausted = learner.update(
        result.state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    _assert_exhausted(exhausted, result.state)


def test_td_learner_clock_saturates_and_keeps_update_identity() -> None:
    learner = TDLinearLearner()
    predecessor = learner.init(2).replace(
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    result = learner.update(
        predecessor,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches
    exhausted = learner.update(
        result.state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    _assert_exhausted(exhausted, result.state)


def test_true_online_td_clock_saturates_and_keeps_update_identity() -> None:
    learner = TrueOnlineTDLearner()
    predecessor = learner.init(2).replace(
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    result = learner.update(
        predecessor,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches
    exhausted = learner.update(
        result.state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    _assert_exhausted(exhausted, result.state)


def test_linear_learner_early_lifetime_still_increments() -> None:
    learner = LinearLearner()
    state = learner.init(2)
    result = learner.update(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1


def test_exhausted_clock_is_atomic_under_jit_and_scan() -> None:
    learner = LinearLearner()
    initial = learner.init(2)
    planted = initial.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        # JIT carries Python scalar leaves as float32; plant the same exact
        # representation so this test isolates update atomicity.
        birth_timestamp=float(jnp.asarray(initial.birth_timestamp, dtype=jnp.float32)),
    )
    observation = jnp.ones(2, dtype=jnp.float32)
    target = jnp.asarray(1.0, dtype=jnp.float32)

    jitted = jax.jit(lambda state: learner.update(state, observation, target))(planted)
    _assert_exhausted(jitted, planted)

    def body(state: Any, _: Any) -> tuple[Any, Any]:
        update = learner.update(state, observation, target)
        return update.state, update.update_applied

    final, applied = jax.jit(lambda state: jax.lax.scan(body, state, None, length=3))(planted)
    chex.assert_trees_all_equal(final, planted)
    chex.assert_trees_all_equal(applied, jnp.zeros(3, dtype=jnp.bool_))


@pytest.mark.parametrize(
    "bad_clock",
    [
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(_INT32_MIN, dtype=jnp.int32),
    ],
)
def test_negative_clock_is_an_atomic_neutral_noop(bad_clock: jnp.ndarray) -> None:
    learner = LinearLearner()
    planted = learner.init(2).replace(step_count=bad_clock)
    result = learner.update(
        planted,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    _assert_exhausted(result, planted)


@pytest.mark.parametrize(
    "replace_clock",
    [
        lambda: jnp.asarray(0, dtype=jnp.int16),
        lambda: jnp.asarray([0], dtype=jnp.int32),
    ],
)
def test_clock_schema_fails_before_update(replace_clock: Callable[[], jnp.ndarray]) -> None:
    learner = LinearLearner()
    planted = learner.init(2).replace(step_count=replace_clock())
    with pytest.raises((TypeError, ValueError), match="step_count"):
        learner.update(
            planted,
            jnp.ones(2, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
