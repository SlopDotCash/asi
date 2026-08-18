"""Saturating lifetime clocks keep learner update identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

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
    planted = learner.init(2).replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    result = learner.update(
        planted,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches
    wrap_nonneg, wrap_matches = _committed_update_identity(_wrap_clock())
    assert not wrap_nonneg
    assert not wrap_matches


def test_mlp_learner_clock_saturates_and_keeps_update_identity() -> None:
    learner = MLPLearner(hidden_sizes=(2,), use_layer_norm=False, sparsity=0.0)
    planted = learner.init(2, jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    result = learner.update(
        planted,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches


def test_td_learner_clock_saturates_and_keeps_update_identity() -> None:
    learner = TDLinearLearner()
    planted = learner.init(2).replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    result = learner.update(
        planted,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches


def test_true_online_td_clock_saturates_and_keeps_update_identity() -> None:
    learner = TrueOnlineTDLearner()
    planted = learner.init(2).replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    result = learner.update(
        planted,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) * 0.5,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    nonneg, matches = _committed_update_identity(result.state.step_count)
    assert nonneg and matches


def test_linear_learner_early_lifetime_still_increments() -> None:
    learner = LinearLearner()
    state = learner.init(2)
    result = learner.update(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
