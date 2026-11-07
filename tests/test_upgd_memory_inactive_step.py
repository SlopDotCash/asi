"""Tests for a fully inactive UPGDMemory update step.

NaN marks an inactive head. A step with no active head carries no learning
signal, so it must leave the wrapper state untouched. A step with at least
one active head still learns on that head.
"""

import jax
import jax.numpy as jnp

from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
)


def _learner() -> UPGDMemoryLearner:
    return UPGDMemoryLearner(
        UPGDMemoryConfig(feature_dim=2, n_heads=2, hidden_sizes=(4,), slots_per_class=2)
    )


_OBS = jnp.array([0.5, -0.5], dtype=jnp.float32)


def _states_equal(a: object, b: object) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: bool(jnp.all(x == y)), a, b)
        )
    )


def test_all_nan_target_is_a_noop() -> None:
    """An all-NaN target must not apply and must not change the state."""
    learner = _learner()
    good = learner.update(
        learner.init(), _OBS, jnp.array([1.0, 0.0], dtype=jnp.float32)
    )
    assert bool(good.update_applied)
    nan_step = learner.update(
        good.state, _OBS, jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32)
    )
    assert not bool(nan_step.update_applied)
    assert _states_equal(nan_step.state, good.state)


def test_all_nan_step_does_not_shift_later_predictions() -> None:
    """An inactive step must not change what a later probe predicts."""
    learner = _learner()
    good = learner.update(
        learner.init(), _OBS, jnp.array([1.0, 0.0], dtype=jnp.float32)
    )
    probe_target = jnp.array([0.2, 0.8], dtype=jnp.float32)
    control = learner.update(good.state, _OBS, probe_target)
    nan_step = learner.update(
        good.state, _OBS, jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32)
    )
    treated = learner.update(nan_step.state, _OBS, probe_target)
    assert bool(jnp.all(control.predictions == treated.predictions))


def test_partial_nan_target_still_learns_on_active_head() -> None:
    """A partially NaN target still updates through its active head."""
    learner = _learner()
    good = learner.update(
        learner.init(), _OBS, jnp.array([1.0, 0.0], dtype=jnp.float32)
    )
    partial = learner.update(
        good.state, _OBS, jnp.array([jnp.nan, 0.0], dtype=jnp.float32)
    )
    assert bool(partial.update_applied)
    assert not _states_equal(partial.state, good.state)


def test_inf_target_is_rejected_control() -> None:
    """An inf target stays rejected, as before this change."""
    learner = _learner()
    good = learner.update(
        learner.init(), _OBS, jnp.array([1.0, 0.0], dtype=jnp.float32)
    )
    inf_step = learner.update(
        good.state, _OBS, jnp.array([jnp.inf, 0.0], dtype=jnp.float32)
    )
    assert not bool(inf_step.update_applied)
    assert _states_equal(inf_step.state, good.state)
