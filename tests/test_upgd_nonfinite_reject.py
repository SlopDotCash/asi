"""UPGDLearner must reject non-finite inputs instead of poisoning its state."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.upgd import UPGDLearner


def _learner() -> UPGDLearner:
    return UPGDLearner(
        n_heads=2,
        hidden_sizes=(4,),
        step_size=0.01,
        perturbation_sigma=0.0,
        unit_replacement_rate=0.0,
    )


def _setup() -> tuple[UPGDLearner, object, jnp.ndarray, jnp.ndarray]:
    learner = _learner()
    state = learner.init(3, jr.key(0))
    observation = jnp.array([0.5, -0.2, 0.3], dtype=jnp.float32)
    target = jnp.array([1.0, 0.0], dtype=jnp.float32)
    return learner, state, observation, target


def test_inf_target_is_rejected_and_next_finite_step_recovers() -> None:
    learner, state, observation, target = _setup()
    baseline = learner.update(state, observation, target)

    rejected = learner.update(
        state, observation, jnp.array([jnp.inf, 0.0], dtype=jnp.float32)
    )
    assert int(rejected.state.step_count) == 0
    chex.assert_trees_all_close(
        rejected.state.trunk_params.weights, state.trunk_params.weights
    )
    chex.assert_trees_all_close(rejected.metrics, jnp.zeros(4, dtype=jnp.float32))

    recovered = learner.update(rejected.state, observation, target)
    chex.assert_trees_all_close(
        recovered.state.trunk_params.weights, baseline.state.trunk_params.weights
    )
    chex.assert_trees_all_close(recovered.metrics, baseline.metrics)


def test_nonfinite_observation_is_rejected() -> None:
    learner, state, _observation, target = _setup()
    for bad_observation in (
        jnp.array([jnp.inf, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([jnp.nan, 0.0, 0.0], dtype=jnp.float32),
    ):
        rejected = learner.update(state, bad_observation, target)
        assert int(rejected.state.step_count) == 0
        chex.assert_trees_all_close(
            rejected.state.trunk_params.weights, state.trunk_params.weights
        )
        chex.assert_trees_all_close(rejected.metrics, jnp.zeros(4, dtype=jnp.float32))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(rejected.state.loss_fast_ema))))


def test_finite_step_is_unchanged() -> None:
    learner, state, observation, target = _setup()
    result = learner.update(state, observation, target)
    assert int(result.state.step_count) == 1
    assert float(result.metrics[0]) == pytest.approx(1.4945053, rel=1e-5)
    chex.assert_tree_all_finite(result.state.trunk_params.weights)
    chex.assert_tree_all_finite(result.metrics)
