from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.bimu import (
    BIMU_PROTOCOL,
    bimu_update,
    late_window_mean,
    posterior_probability,
)


def test_bimu_update_matches_equations_six_and_seven() -> None:
    state = jnp.array([0.0, 1.0], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([2.0, -0.5], dtype=jnp.float32)
    updated = bimu_update(state, gradient, prior, memory_window=10, alpha_max=1.0)
    reciprocal = (
        1.0 / jnp.cosh(state) ** 2
        + 2.0 * jnp.tanh(state) * gradient
        + 1.0
        + 2.0 * jnp.abs(gradient)
    )
    eta = 1.0 / reciprocal
    expected = state - eta * (gradient + (state - prior) / (10 * jnp.cosh(state) ** 2))
    np.testing.assert_allclose(updated, expected, rtol=1e-6)


def test_mechanism_off_removes_forgetting_term() -> None:
    state = jnp.array([0.5], dtype=jnp.float32)
    result = bimu_update(
        state, jnp.zeros(1), jnp.zeros(1), memory_window=None, alpha_max=0.5
    )
    np.testing.assert_array_equal(result, state)


def test_posterior_probability_and_late_metric_are_distinct() -> None:
    np.testing.assert_allclose(posterior_probability(jnp.array([0.0])), [0.5])
    assert late_window_mean([0.1, 0.2, 0.8, 0.9], window=2) == pytest.approx(0.85)
    assert BIMU_PROTOCOL["primary_metric"] == "mean_test_accuracy_over_last_5_tasks"
    assert BIMU_PROTOCOL["whole_stream_online_accuracy_is_separate"] is True


def test_protocol_is_binary_bayesian_and_nonpromoting() -> None:
    assert BIMU_PROTOCOL["paper_revision"] == "arXiv:2605.30198v1"
    assert BIMU_PROTOCOL["weight_domain"] == (-1, 1)
    assert BIMU_PROTOCOL["development_only"] is True
    assert BIMU_PROTOCOL["scientific_promotion_allowed"] is False
