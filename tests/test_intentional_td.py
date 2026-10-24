"""Intentional TD's measured output change and trace recurrence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.intentional_td import (
    IntentionalTDConfig,
    init_intentional_td,
    intentional_td_update,
)


def test_linear_prediction_moves_by_intended_fraction_of_frozen_td_error():
    # Same value prediction problem at three feature scales, including bias.
    for scale in (0.1, 1.0, 10.0):
        features = jnp.array([2.0, -1.0, 1.0]) * scale
        weights = jnp.zeros(3)
        config = IntentionalTDConfig(eta=0.2, lamda=0.0, use_rmsprop=False)
        changed, _ = intentional_td_update(
            weights, init_intentional_td(weights), features, jnp.array(3.0), config
        )
        np.testing.assert_allclose(changed @ features, 0.6, rtol=2e-6)


def test_trace_normalization_matches_independent_recurrence_and_resets_at_terminal():
    config = IntentionalTDConfig(eta=0.1, gamma=0.9, lamda=0.8, beta2=0.95)
    weights = jnp.zeros(2)
    state = init_intentional_td(weights)
    expected = np.zeros(2)
    trace = np.zeros(2)
    second = np.zeros(2)
    sigma = 0.0
    clip_ema = 0.0
    step = jax.jit(lambda w, s, g, d, done: intentional_td_update(w, s, g, d, config, done))
    for t, (gradient, error, terminal) in enumerate(
        [([1.0, -0.5], 2.0, False), ([0.2, 1.0], -0.3, True), ([1.0, 0.0], 0.7, False)],
        start=1,
    ):
        g = np.asarray(gradient)
        second = 0.95 * second + 0.05 * g**2
        divisor = np.sqrt(second / (1 - 0.95**t)) + 1e-8
        trace = 0.72 * trace + g
        sigma += (1 - 0.72) * (np.sum(g**2 / divisor) - sigma)
        normalizer = np.sqrt(sigma / (1 - 0.72**t) * np.sum(trace**2 / divisor))
        clip_ema = 0.9998 * clip_ema + 0.0002 * error**2
        cap = 20 * np.sqrt(clip_ema / (1 - 0.9998**t))
        expected += 0.1 / max(normalizer, 1e-8) * np.clip(error, -cap, cap) * trace / divisor
        weights, state = step(weights, state, jnp.array(gradient), jnp.array(error), terminal)
        np.testing.assert_allclose(weights, expected, rtol=2e-5, atol=2e-6)
        if terminal:
            trace.fill(0)
        np.testing.assert_allclose(state.trace, trace, rtol=2e-6, atol=1e-7)


def test_fixed_step_control_is_exact_sgd_with_mechanism_off():
    weights = jnp.array([0.2, -0.7])
    gradient = jnp.array([0.25, 0.75])
    error = jnp.array(0.8)
    config = IntentionalTDConfig(eta=0.125, lamda=0.0, enabled=False)
    changed, _ = intentional_td_update(
        weights, init_intentional_td(weights), gradient, error, config
    )
    np.testing.assert_array_equal(changed, weights + (0.125 * error) * gradient)


@pytest.mark.parametrize("compiled", [False, True])
def test_rms_normalization_preserves_finite_wide_gradient_update(compiled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([2.0e19, -2.0e19], dtype=jnp.float32)
    config = IntentionalTDConfig(eta=0.1, lamda=0.0)

    def update(w, state):
        return intentional_td_update(w, state, gradient, jnp.array(1.0), config)

    step = jax.jit(update) if compiled else update
    changed, state = step(weights, init_intentional_td(weights))

    np.testing.assert_allclose(changed, [2.5e-21, -2.5e-21], rtol=2e-5)
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(state))


@pytest.mark.parametrize("compiled", [False, True])
def test_adaptive_clip_preserves_finite_wide_td_error(compiled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([1.0, -1.0], dtype=jnp.float32)
    td_error = jnp.array(2.0e19, dtype=jnp.float32)
    config = IntentionalTDConfig(lamda=0.0)

    def update(w, state):
        return intentional_td_update(w, state, gradient, td_error, config)

    step = jax.jit(update) if compiled else update
    changed, state = step(weights, init_intentional_td(weights))

    expected_clip_second = (1.0 - config.beta_clip) * float(td_error) ** 2
    np.testing.assert_allclose(changed, [5.0e18, -5.0e18], rtol=2e-5)
    np.testing.assert_allclose(state.clip_squared_error, expected_clip_second, rtol=2e-5)
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(state))


@pytest.mark.parametrize("compiled", [False, True])
def test_adaptive_clip_preserves_finite_wide_cap(compiled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([1.0, -1.0], dtype=jnp.float32)
    td_error = jnp.array(5.8e20, dtype=jnp.float32)
    config = IntentionalTDConfig(lamda=0.0)
    state = init_intentional_td(weights).replace(step=jnp.array(999, dtype=jnp.int32))

    def update(w, optimizer_state):
        return intentional_td_update(w, optimizer_state, gradient, td_error, config)

    step = jax.jit(update) if compiled else update
    changed, state = step(weights, state)

    count = 1000.0
    clip_second = (1.0 - config.beta_clip) * float(td_error) ** 2
    clip_bias = 1.0 - config.beta_clip**count
    expected_cap = config.clip_mult * np.sqrt(clip_second / clip_bias)
    np.testing.assert_allclose(changed, [0.25 * expected_cap, -0.25 * expected_cap], rtol=2e-4)
    np.testing.assert_allclose(state.clip_squared_error, clip_second, rtol=2e-5)
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(state))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("error", [-10.0, 10.0])
def test_binding_clip_limits_prediction_change(compiled, adaptive, error):
    # With one unit feature and no trace/RMS, the observed prediction change
    # is eta * clipped_error. Choose caps that bind in both clipping modes.
    config = IntentionalTDConfig(
        eta=0.2, lamda=0.0, use_rmsprop=False,
        beta_clip=0.5, clip_mult=0.1, use_adaptive_clip=adaptive,
    )
    weights = jnp.zeros(1)
    state = init_intentional_td(weights)

    def update(w, s, d):
        return intentional_td_update(w, s, jnp.ones(1), d, config)

    step = jax.jit(update) if compiled else update
    # The second error also binds, and distinguishes the running adaptive
    # cap (sqrt(300) / 10) from a cap based only on the current error (2).
    for index, value in enumerate((error, 2 * error)):
        cap = (1.0 if index == 0 else np.sqrt(300.0) / 10) if adaptive else 1.0
        changed, state = step(weights, state, jnp.array(value))
        np.testing.assert_allclose(
            changed - weights, [0.2 * np.copysign(cap, value)], rtol=2e-6
        )
        weights = changed
