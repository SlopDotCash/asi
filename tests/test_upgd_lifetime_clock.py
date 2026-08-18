"""Saturating lifetime clocks keep the UPGD learner update identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.upgd import UPGDLearner

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _make_learner() -> UPGDLearner:
    return UPGDLearner(
        n_heads=2,
        hidden_sizes=(4,),
        sparsity=0.0,
        perturbation_sigma=0.0,
    )


def test_upgd_clock_wraps_without_saturation_at_int32_max() -> None:
    """The old bare increment wraps INT32_MAX + 1 to INT32_MIN."""

    wrap = jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    assert int(wrap) == _INT32_MIN


def test_upgd_clock_saturates_and_keeps_update_identity() -> None:
    """A planted INT32_MAX clock saturates instead of wrapping negative."""

    learner = _make_learner()
    predecessor = learner.init(feature_dim=4, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    result = learner.update(
        predecessor,
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.step_count) >= 0

    exhausted = learner.update(
        result.state,
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert int(exhausted.state.step_count) == _INT32_MAX
    assert int(exhausted.state.step_count) >= 0
    assert int(exhausted.state.step_count) == int(result.state.step_count)


def test_upgd_clock_at_int32_max_does_not_wrap_negative() -> None:
    """One more update from the saturated clock stays at INT32_MAX, never INT32_MIN."""

    learner = _make_learner()
    state = learner.init(feature_dim=4, key=jr.key(0)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    result = learner.update(
        state,
        jnp.ones(4, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.step_count) != _INT32_MIN
