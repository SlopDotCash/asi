"""Regression tests for finite forgetting and reciprocal_eta in BiMU updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.bimu import (
    BiMUConfig,
    _apply_gradient,
    _make_state,
    bimu_update,
    bimu_update_transaction,
)


@pytest.mark.unit
def test_bimu_update_transaction_saturated_opposing_gradient_preserves_eta() -> None:
    """Opposing gradient on a saturated state must not absorb 1 / alpha_max into zero."""
    state = jnp.array([20.0, -20.0], dtype=jnp.float32)
    gradient = jnp.array([-1e10, 1e10], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)

    updated, valid = bimu_update_transaction(
        state,
        gradient,
        prior,
        memory_window=700,
        alpha_max=0.0023,
        likelihood_multiplier=161.3,
        kl_multiplier=3.76,
        gradient_scale=4.9,
    )
    jit_updated, jit_valid = jax.jit(
        lambda s, g, p: bimu_update_transaction(
            s,
            g,
            p,
            memory_window=700,
            alpha_max=0.0023,
            likelihood_multiplier=161.3,
            kl_multiplier=3.76,
            gradient_scale=4.9,
        )
    )(state, gradient, prior)

    assert bool(valid)
    assert bool(jit_valid)
    assert bool(jnp.all(jnp.isfinite(updated)))
    np.testing.assert_allclose(np.asarray(updated), np.asarray(jit_updated), rtol=0.0, atol=0.0)

    # At |state| = 20, tanh(state) = ±1 and uncertainty = 0, so the two gradient terms in
    # reciprocal_eta cancel identically and reciprocal_eta = 1 / alpha_max.
    expected_step = (4.9 * 161.3 * np.array([-1e10, 1e10], dtype=np.float64)) / (1.0 / 0.0023)
    expected = np.array([20.0, -20.0], dtype=np.float64) - expected_step
    np.testing.assert_allclose(np.asarray(updated, dtype=np.float64), expected, rtol=1e-6)


@pytest.mark.unit
def test_bimu_update_transaction_near_max_opposing_gradient_cancels_without_nan() -> None:
    """Opposing 2*tanh(s)*g and 2*|g| near float32 max must cancel instead of -inf + inf."""
    state = jnp.array([20.0, -20.0], dtype=jnp.float32)
    gradient = jnp.array([-2e38, 2e38], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)

    updated, valid = bimu_update_transaction(
        state,
        gradient,
        prior,
        memory_window=700,
        alpha_max=1e-38,
        likelihood_multiplier=1.0,
        kl_multiplier=1.0,
        gradient_scale=1.0,
    )

    assert bool(valid)
    assert bool(jnp.all(jnp.isfinite(updated)))
    np.testing.assert_allclose(
        np.asarray(updated),
        np.array([22.0, -22.0], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_bimu_update_transaction_zero_uncertainty_ignores_overflowing_prior_gap() -> None:
    """Zero uncertainty on saturated states must yield zero forgetting on large prior gaps."""
    state = jnp.array([2e38, -2e38], dtype=jnp.float32)
    prior = jnp.array([-2e38, 2e38], dtype=jnp.float32)
    gradient = jnp.zeros(2, dtype=jnp.float32)

    updated, valid = bimu_update_transaction(
        state,
        gradient,
        prior,
        memory_window=700,
        alpha_max=0.0023,
    )
    wrapped = bimu_update(
        state,
        gradient,
        prior,
        memory_window=700,
        alpha_max=0.0023,
    )

    assert bool(valid)
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert jnp.array_equal(updated, state)
    assert jnp.array_equal(wrapped, state)


@pytest.mark.unit
def test_bimu_update_transaction_forgetting_divides_by_memory_window_before_overflow() -> None:
    """Finite forgetting must divide uncertainty by memory_window before overflow."""
    state = jnp.array([0.0, 1.0], dtype=jnp.float32)
    prior = jnp.array([-1e38, 2e38], dtype=jnp.float32)
    gradient = jnp.zeros(2, dtype=jnp.float32)

    updated, valid = bimu_update_transaction(
        state,
        gradient,
        prior,
        memory_window=700,
        alpha_max=0.0023,
        likelihood_multiplier=1.0,
        kl_multiplier=10.0,
        gradient_scale=1.0,
    )

    assert bool(valid)
    assert bool(jnp.all(jnp.isfinite(updated)))

    state64 = np.asarray(state, dtype=np.float64)
    prior64 = np.asarray(prior, dtype=np.float64)
    uncertainty64 = 10.0 * (1.0 - np.tanh(state64) ** 2)
    forgetting64 = (state64 - prior64) * uncertainty64 / 700.0
    reciprocal_eta64 = uncertainty64 + 1.0 / 0.0023
    expected = state64 - forgetting64 / reciprocal_eta64
    np.testing.assert_allclose(np.asarray(updated, dtype=np.float64), expected, rtol=1e-6)


@pytest.mark.unit
def test_apply_gradient_commits_saturated_opposing_gradient_update() -> None:
    """_apply_gradient must return finite updated state instead of raising ValueError."""
    config = BiMUConfig(
        input_dim=2,
        hidden_units=2,
        n_classes=2,
        memory_window=700,
        alpha_max=0.0023,
        likelihood_multiplier=161.3,
        kl_multiplier=3.76,
        gradient_scale=4.9,
    )
    state = _make_state(
        input_hidden=jnp.full((2, 2), 20.0, dtype=jnp.float32),
        hidden_output=jnp.full((2, 2), -20.0, dtype=jnp.float32),
    )
    grads = _make_state(
        input_hidden=jnp.full((2, 2), -1e10, dtype=jnp.float32),
        hidden_output=jnp.full((2, 2), 1e10, dtype=jnp.float32),
    )

    updated_state = _apply_gradient(state, grads, config)

    assert bool(jnp.all(jnp.isfinite(updated_state.input_hidden)))
    assert bool(jnp.all(jnp.isfinite(updated_state.hidden_output)))
    assert bool(jnp.all(updated_state.input_hidden > 20.0))
    assert bool(jnp.all(updated_state.hidden_output < -20.0))

