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


def test_config_accepts_trace_decay_that_rounds_to_one_in_float32():
    # A near-one gamma * lamda that rounds to one in float32 is accepted:
    # the bias corrections run on the float64 decay via expm1, so the first
    # update stays finite. An exact product of one is still rejected.
    gamma = 0.99999999
    assert gamma < 1.0
    assert float(jnp.asarray(gamma, dtype=jnp.float32)) == 1.0

    config = IntentionalTDConfig(gamma=gamma, lamda=1.0)
    weights = jnp.zeros(2, dtype=jnp.float32)
    changed, state = intentional_td_update(
        weights,
        init_intentional_td(weights),
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        config,
    )
    assert bool(jnp.all(jnp.isfinite(changed)))
    assert bool(jnp.isfinite(state.sigma))
    assert bool(jnp.all(changed != weights))

    with pytest.raises(ValueError, match="less than 1"):
        IntentionalTDConfig(gamma=1.0, lamda=1.0)


@pytest.mark.parametrize("field", ["beta2", "beta_clip"])
def test_config_accepts_ema_decay_that_rounds_to_one_in_float32(field):
    decay = 0.99999999
    assert decay < 1.0
    assert float(jnp.asarray(decay, dtype=jnp.float32)) == 1.0

    config = IntentionalTDConfig(**{field: decay})
    weights = jnp.zeros(2, dtype=jnp.float32)
    changed, state = intentional_td_update(
        weights,
        init_intentional_td(weights),
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        config,
    )
    assert bool(jnp.all(jnp.isfinite(changed)))
    assert bool(jnp.all(jnp.isfinite(state.squared_gradient)))
    assert bool(jnp.all(changed != weights))

    with pytest.raises(ValueError, match="EMA decays"):
        IntentionalTDConfig(**{field: 1.0})


@pytest.mark.parametrize("field", ["eta", "clip_mult"])
@pytest.mark.parametrize("value", [1.0e-50, 1.0e40])
def test_config_rejects_positive_scale_outside_float32(field, value):
    assert value > 0.0
    with np.errstate(over="ignore", under="ignore"):
        narrowed = np.float32(value)
    assert not bool(np.isfinite(narrowed)) or float(narrowed) == 0.0

    with pytest.raises(ValueError, match=field):
        IntentionalTDConfig(**{field: value})


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
@pytest.mark.parametrize("enabled", [False, True])
def test_lifetime_step_counter_saturates_without_stalling_learning(compiled, enabled):
    """A long-lived control must keep updating after int32 telemetry saturates."""
    weights = jnp.array([0.5, -0.25], dtype=jnp.float32)
    gradient = jnp.array([0.1, 0.2], dtype=jnp.float32)
    config = IntentionalTDConfig(
        eta=0.1,
        gamma=0.9,
        lamda=0.8,
        enabled=enabled,
    )
    maximum = jnp.iinfo(jnp.int32).max
    state = init_intentional_td(weights).replace(
        trace=jnp.array([0.2, -0.1], dtype=jnp.float32),
        squared_gradient=jnp.array([0.3, 0.4], dtype=jnp.float32),
        sigma=jnp.array(0.5, dtype=jnp.float32),
        clip_squared_error=jnp.array(0.25, dtype=jnp.float32),
        step=jnp.array(maximum, dtype=jnp.int32),
    )

    def update(w, s, g, error):
        return intentional_td_update(w, s, g, error, config)

    apply_update = jax.jit(update) if compiled else update
    changed, next_state = apply_update(
        weights,
        state,
        gradient,
        jnp.array(0.4, dtype=jnp.float32),
    )

    assert int(next_state.step) == maximum
    assert bool(jnp.all(jnp.isfinite(changed)))
    assert bool(jnp.all(changed != weights))
    assert bool(jnp.isfinite(next_state.sigma))
    assert float(next_state.sigma) < 1.0


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


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("field", ["beta2", "beta_clip", "lamda"])
# 1 - 2**-24 is the largest float32 below one: the config refuses decays that
# round to one in float32, so this is the nearest decay it accepts.
@pytest.mark.parametrize("decay", [0.0, 1.0 - 2.0**-24])
def test_bias_correction_preserves_finite_first_update(compiled, field, decay):
    # For a unit gradient and error, every first-step bias correction
    # cancels its EMA coefficient, even for the nearest float32 decay below one.
    settings = {field: decay}
    if field == "lamda":
        settings["gamma"] = 1.0
    config = IntentionalTDConfig(eta=0.2, clip_mult=0.1, **settings)
    weights = jnp.zeros(1, dtype=jnp.float32)
    state = init_intentional_td(weights)

    def update(w, s):
        return intentional_td_update(w, s, jnp.ones_like(w), jnp.array(1.0), config)

    changed, updated = (jax.jit(update) if compiled else update)(weights, state)
    np.testing.assert_allclose(changed, [0.02], rtol=2e-6)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(updated))

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
def test_near_one_trace_decay_survives_cancelling_gradient(compiled):
    decay = 1.0 - 2.0**-27
    config = IntentionalTDConfig(gamma=decay, lamda=1.0, enabled=False)
    weights = jnp.zeros(1, dtype=jnp.float32)
    old_trace = jnp.full_like(weights, 2.0**20)
    state = init_intentional_td(weights).replace(trace=old_trace)

    def update(current):
        return intentional_td_update(weights, current, -old_trace, jnp.float32(1.0), config)

    _, next_state = (jax.jit(update) if compiled else update)(state)
    np.testing.assert_array_equal(next_state.trace, jnp.asarray([-2.0**-7]))



@pytest.mark.parametrize("compiled", [False, True])
def test_wide_gradient_without_rmsprop_preserves_finite_sigma_and_factored_update(compiled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([5.8e20, -5.8e20], dtype=jnp.float32)
    config = IntentionalTDConfig(gamma=0.9998, lamda=1.0, use_rmsprop=False)
    state = init_intentional_td(weights).replace(step=jnp.array(999, dtype=jnp.int32))

    def update(w, optimizer_state):
        return intentional_td_update(w, optimizer_state, gradient, jnp.float32(1.0), config)

    changed, next_state = (jax.jit(update) if compiled else update)(weights, state)
    count = 1000.0
    decay = config.gamma * config.lamda
    expected_sigma = (1.0 - decay) * 2.0 * (5.8e20**2)
    bias = 1.0 - decay**count
    cap = config.clip_mult * np.sqrt((1.0 - config.beta_clip) / (1.0 - config.beta_clip**count))
    normalizer = np.sqrt((expected_sigma / bias) * (2.0 * (5.8e20**2)))
    expected_step = (config.eta * min(1.0, cap) / normalizer) * np.array([5.8e20, -5.8e20])
    np.testing.assert_allclose(changed, expected_step, rtol=2e-4)
    np.testing.assert_allclose(next_state.sigma, expected_sigma, rtol=2e-5)
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(next_state))


@pytest.mark.parametrize("compiled", [False, True])
def test_sigma_bias_correction_preserves_finite_sqrt_when_quotient_overflows(compiled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([1.0, -1.0], dtype=jnp.float32)
    config = IntentionalTDConfig(gamma=0.999, lamda=1.0, use_rmsprop=True)
    state = init_intentional_td(weights).replace(
        sigma=jnp.array(1.0e37, dtype=jnp.float32),
        squared_gradient=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )

    def update(w, optimizer_state):
        return intentional_td_update(w, optimizer_state, gradient, jnp.float32(1.0), config)

    changed, next_state = (jax.jit(update) if compiled else update)(weights, state)
    assert bool(jnp.all(jnp.isfinite(changed)))
    assert bool(jnp.all(changed != weights))
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(next_state))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_sparse_and_small_traces_avoid_overflow_and_zero_times_inf_nan(compiled, enabled):
    weights = jnp.zeros(2, dtype=jnp.float32)
    config = IntentionalTDConfig(
        eta=100.0,
        gamma=0.9,
        lamda=1.0,
        enabled=enabled,
        use_rmsprop=False,
        use_adaptive_clip=False,
    )
    if enabled:
        state = init_intentional_td(weights).replace(
            trace=jnp.array([1.0e-3, 0.0], dtype=jnp.float32),
            sigma=jnp.array(1.0, dtype=jnp.float32),
        )
        gradient = jnp.zeros(2, dtype=jnp.float32)
        td_error = jnp.float32(1.0)
    else:
        state = init_intentional_td(weights)
        gradient = jnp.array([1.0e-3, 0.0], dtype=jnp.float32)
        td_error = jnp.float32(1.0e38)

    def update(w, optimizer_state):
        return intentional_td_update(w, optimizer_state, gradient, td_error, config)

    changed, next_state = (jax.jit(update) if compiled else update)(weights, state)
    assert bool(jnp.all(jnp.isfinite(changed)))
    assert float(changed[0]) > 0.0
    assert float(changed[1]) == 0.0
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(next_state))
