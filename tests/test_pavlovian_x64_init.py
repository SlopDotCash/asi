"""Regression tests: ClassicalConditioningStream.init under JAX x64 mode.

``step`` checks the state contract first and requires int32 for every
counter; it writes all of them back with ``.astype(jnp.int32)``. ``init``
drew the first ITI with ``jr.randint`` and no dtype, which is int64 under
x64, so the first ``step`` on a fresh state raised ``TypeError``.
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.learners import LinearLearner, run_learning_loop
from alberta_framework.core.optimizers import LMS
from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    acquisition_scenario,
    blocking_scenario,
    extinction_scenario,
    partial_reinforcement_scenario,
    reacquisition_scenario,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "factory",
    [
        acquisition_scenario,
        extinction_scenario,
        blocking_scenario,
        partial_reinforcement_scenario,
        reacquisition_scenario,
    ],
)
def test_init_counters_are_int32_and_step_runs_with_x64_enabled(
    factory: Callable[[], ClassicalConditioningStream],
) -> None:
    with jax.enable_x64(True):
        stream = factory()
        state = stream.init(jr.key(0))
        for leaf in (
            state.cs_active_steps_remaining,
            state.us_pending_steps_remaining,
            state.phase_idx,
            state.step_in_phase,
            state.n_distractor_active,
            state.iti_steps_remaining,
        ):
            assert leaf.dtype == jnp.int32
        timestep, next_state = stream.step(state, jnp.array(0, dtype=jnp.int32))
    assert timestep.observation.shape == (stream.feature_dim,)
    assert next_state.iti_steps_remaining.dtype == jnp.int32


def test_init_state_is_x64_invariant() -> None:
    stream = acquisition_scenario()
    state32 = stream.init(jr.key(7))
    with jax.enable_x64(True):
        state64 = stream.init(jr.key(7))
    chex.assert_trees_all_equal(state32, state64)


def test_run_learning_loop_runs_on_a_pavlovian_stream_with_x64_enabled() -> None:
    """The public learning loop reaches the init/step contract under x64."""
    with jax.enable_x64(True):
        stream = acquisition_scenario(n_steps=200)
        learner = LinearLearner(optimizer=LMS(step_size=0.1))
        metrics = run_learning_loop(learner, stream, num_steps=200, key=jr.key(3))[1]
    assert metrics.shape[0] == 200
    assert bool(jnp.all(jnp.isfinite(metrics)))
