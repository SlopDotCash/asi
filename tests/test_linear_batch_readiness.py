"""Device completion is part of the batched linear learner's uptime contract."""

import chex
import jax
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import (
    LMS,
    LinearLearner,
    RandomWalkStream,
    StepSizeTrackingConfig,
    run_learning_loop,
    run_learning_loop_batched,
)


@pytest.mark.parametrize("tracking", [False, True])
def test_batched_linear_returns_completed_device_results(tracking: bool) -> None:
    """Use real array work, not callbacks that implicitly synchronize CPU dispatch."""
    learner = LinearLearner(optimizer=LMS(step_size=0.0001))
    stream = RandomWalkStream(feature_dim=1000)
    keys = jr.split(jr.key(712831, impl="threefry2x32"), 2)
    result = run_learning_loop_batched(
        learner,
        stream,
        num_steps=10_000,
        keys=keys,
        step_size_tracking=StepSizeTrackingConfig(interval=1000) if tracking else None,
    )
    # Do not convert/read a result leaf before probing readiness: that would
    # synchronize it and hide the missing batched completion barrier.
    assert result.metrics.is_ready()
    assert result.states.weights.is_ready()
    if result.step_size_history is not None:
        assert result.step_size_history.step_sizes.is_ready()
    np.testing.assert_array_equal(result.states.step_count, [10_000, 10_000])


def test_batched_completion_preserves_sequential_learning_results() -> None:
    learner = LinearLearner(optimizer=LMS(step_size=0.0001))
    stream = RandomWalkStream(feature_dim=3)
    keys = jr.split(jr.key(712832, impl="threefry2x32"), 2)
    batched = run_learning_loop_batched(learner, stream, num_steps=8, keys=keys)
    sequential = [run_learning_loop(learner, stream, num_steps=8, key=key) for key in keys]
    chex.assert_trees_all_close(
        batched.metrics, np.stack([result[1] for result in sequential]), rtol=1e-5
    )
    chex.assert_trees_all_close(
        batched.states.weights,
        np.stack([result[0].weights for result in sequential]),
        rtol=1e-5,
    )
    jax.block_until_ready(batched)
