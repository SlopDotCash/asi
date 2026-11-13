"""Regression tests for finite component and pre-activation sums in CompositionalStream."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.streams.out_of_class import (
    CompositionalStream,
    OutOfClassPolynomialStream,
)


def test_compositional_cancelling_component_sum_stays_finite_and_negative() -> None:
    """A finite outer component sum whose partial total overflows keeps the real total."""
    stream = CompositionalStream(
        feature_dim=1,
        n_tasks=1,
        inner_hidden=1,
        outer_components=5,
        n_contexts=1,
        context_length=1,
        noise_std=0.0,
    )
    state = stream.init(jr.key(0))
    amplitudes = jnp.asarray(
        [2.0e38, 2.0e38, -2.0e38, -2.0e38, -1.0e38],
        dtype=jnp.float32,
    )
    state = state.replace(  # type: ignore[attr-defined]
        inner_w=jnp.zeros_like(state.inner_w),
        inner_b=jnp.zeros_like(state.inner_b),
        outer_w=jnp.zeros_like(state.outer_w),
        outer_b=jnp.full_like(state.outer_b, 100.0),
        amplitudes=amplitudes.reshape(1, 1, 5),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(jnp.isfinite(jnp.sum(amplitudes)))
    expected = np.float32(sum(np.float64(value) for value in amplitudes))

    timestep, _new_state = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(jnp.all(jnp.isfinite(timestep.observation)))
    assert jnp.array_equal(
        timestep.target,
        jnp.asarray([expected], dtype=jnp.float32),
    )
    assert float(timestep.target[0]) < 0.0

    ordinary = jnp.asarray([1.0, -2.0, 0.5, 0.25, -0.125], dtype=jnp.float32)
    ordinary_step, _ordinary_state = stream.step(
        state.replace(
            amplitudes=ordinary.reshape(1, 1, 5),
        ),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert jnp.array_equal(ordinary_step.target, jnp.atleast_1d(jnp.sum(ordinary)))


def test_compositional_cancelling_outer_preactivation_preserves_negative_sign() -> None:
    """Cancelling outer-layer weights and bias must not overflow before tanh."""
    stream = CompositionalStream(
        feature_dim=1,
        n_tasks=1,
        inner_hidden=5,
        outer_components=1,
        n_contexts=1,
        context_length=1,
        noise_std=0.0,
    )
    state = stream.init(jr.key(1))
    outer_weights = jnp.asarray(
        [2.0e38, 2.0e38, -2.0e38, -2.0e38, -1.0e38],
        dtype=jnp.float32,
    )
    state = state.replace(  # type: ignore[attr-defined]
        inner_w=jnp.zeros_like(state.inner_w),
        inner_b=jnp.full_like(state.inner_b, 100.0),
        outer_w=outer_weights.reshape(1, 1, 1, 5),
        outer_b=jnp.zeros_like(state.outer_b),
        amplitudes=jnp.ones_like(state.amplitudes),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(jnp.isfinite(jnp.sum(outer_weights)))

    timestep, _ = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    assert jnp.array_equal(timestep.target, jnp.asarray([-1.0], dtype=jnp.float32))

    # Also check cancellation between outer_w @ inner and outer_b.
    bias_cancelling_weights = jnp.asarray(
        [2.0e38, 2.0e38, -2.0e38, 0.0, 0.0],
        dtype=jnp.float32,
    )
    state_with_bias = state.replace(
        outer_w=bias_cancelling_weights.reshape(1, 1, 1, 5),
        outer_b=jnp.full_like(state.outer_b, -3.0e38),
    )
    timestep_bias, _ = stream.step(state_with_bias, jnp.asarray(0, dtype=jnp.int32))
    assert jnp.array_equal(timestep_bias.target, jnp.asarray([-1.0], dtype=jnp.float32))


def test_compositional_cancelling_inner_preactivation_preserves_negative_sign() -> None:
    """Cancelling inner-layer products and bias must not overflow before tanh."""
    stream = CompositionalStream(
        feature_dim=5,
        n_tasks=1,
        inner_hidden=1,
        outer_components=1,
        n_contexts=1,
        context_length=1,
        feature_std=20.0,
        noise_std=0.0,
    )
    state = stream.init(jr.key(2))
    _key, k_x, _k_noise = jr.split(state.key, 3)
    x = stream._feature_std * jr.normal(k_x, (5,), dtype=jnp.float32)  # noqa: SLF001
    desired_products = jnp.asarray(
        [2.0e38, 2.0e38, -2.0e38, -2.0e38, -1.0e38],
        dtype=jnp.float32,
    )
    inner_weights = (desired_products / x).reshape(1, 1, 1, 1, 5)
    assert bool(jnp.all(jnp.isfinite(inner_weights)))
    assert not bool(jnp.isfinite(jnp.einsum("tohf,f->toh", inner_weights[0], x)))
    state = state.replace(  # type: ignore[attr-defined]
        inner_w=inner_weights,
        inner_b=jnp.zeros_like(state.inner_b),
        outer_w=jnp.full_like(state.outer_w, 100.0),
        outer_b=jnp.zeros_like(state.outer_b),
        amplitudes=jnp.ones_like(state.amplitudes),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )

    timestep, _ = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    assert jnp.array_equal(timestep.target, jnp.asarray([-1.0], dtype=jnp.float32))


def test_compositional_ordinary_step_is_bitwise_identical() -> None:
    """Ordinary finite compositional steps preserve exact direct float32 arithmetic."""
    stream = CompositionalStream(
        feature_dim=4,
        n_tasks=3,
        inner_hidden=3,
        outer_components=4,
        n_contexts=2,
        context_length=5,
        noise_std=0.05,
    )
    state = stream.init(jr.key(7))
    _key, k_x, k_noise = jr.split(state.key, 3)
    x = stream._feature_std * jr.normal(k_x, (4,), dtype=jnp.float32)  # noqa: SLF001
    inner_pre = jnp.einsum("tohf,f->toh", state.inner_w[0], x) + state.inner_b[0]
    inner = jnp.tanh(inner_pre)
    outer_pre = jnp.sum(state.outer_w[0] * inner, axis=-1) + state.outer_b[0]
    outer = jnp.tanh(outer_pre)
    noise = stream._noise_std * jr.normal(k_noise, (3,), dtype=jnp.float32)  # noqa: SLF001
    expected_target = jnp.sum(state.amplitudes[0] * outer, axis=-1) + noise

    timestep, _ = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    assert jnp.array_equal(timestep.observation, x)
    assert jnp.array_equal(timestep.target, expected_target)


def test_polynomial_cancelling_triple_and_linear_sums_stay_finite() -> None:
    """Cancelling polynomial triple and linear terms must not overflow to inf/NaN."""
    stream = OutOfClassPolynomialStream(
        feature_dim=4,
        n_tasks=1,
        n_contexts=1,
        context_length=1,
        active_triples_per_context=4,
        feature_std=20.0,
        linear_scale=0.0,
        noise_std=0.0,
    )
    state = stream.init(jr.key(3))
    _key, k_x, _k_noise = jr.split(state.key, 3)
    x = stream._feature_std * jr.normal(k_x, (4,), dtype=jnp.float32)  # noqa: SLF001
    triples = (
        x[state.triples_left]
        * x[state.triples_middle]
        * x[state.triples_right]
    )
    desired_terms = jnp.asarray(
        [2.0e38, 2.0e38, -2.0e38, -3.0e38],
        dtype=jnp.float32,
    )
    weights = (desired_terms / triples).reshape(1, 1, 4)
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert not bool(jnp.isfinite(weights[0] @ triples))

    planted = state.replace(  # type: ignore[attr-defined]
        context_weights=weights,
        linear_weights=jnp.zeros_like(state.linear_weights),
    )
    timestep, _ = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert bool(jnp.all(jnp.isfinite(timestep.target)))
    assert float(timestep.target[0]) < 0.0

    # Ordinary finite step stays bitwise identical to direct dot products.
    ordinary_ts, _ = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    expected_ordinary = state.context_weights[0] @ triples + state.linear_weights @ x
    assert jnp.array_equal(ordinary_ts.target, expected_ordinary)


def test_polynomial_overflowing_raw_triple_with_zero_and_small_weights_stays_finite() -> None:
    """Overflowing raw x_i*x_j*x_l must not turn zero weights into NaN or small weights into inf."""
    stream_hi = OutOfClassPolynomialStream(
        feature_dim=4,
        n_tasks=1,
        n_contexts=1,
        context_length=1,
        active_triples_per_context=1,
        feature_std=1.0,
        linear_scale=0.0,
        noise_std=0.0,
    )
    object.__setattr__(stream_hi, "_feature_std", float(np.float32(1.0e14)))
    state_hi = stream_hi.init(jr.key(0))
    _key, k_x, _k_noise = jr.split(state_hi.key, 3)
    x_hi = stream_hi._feature_std * jr.normal(k_x, (4,), dtype=jnp.float32)  # noqa: SLF001
    raw_triples = (
        x_hi[state_hi.triples_left]
        * x_hi[state_hi.triples_middle]
        * x_hi[state_hi.triples_right]
    )
    assert bool(jnp.any(~jnp.isfinite(raw_triples)))
    weights = jnp.asarray([[[1.0e-20, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
    planted_hi = state_hi.replace(  # type: ignore[attr-defined]
        context_weights=weights,
        linear_weights=jnp.zeros_like(state_hi.linear_weights),
    )
    ts_hi, _ = stream_hi.step(planted_hi, jnp.asarray(0, dtype=jnp.int32))
    assert bool(jnp.all(jnp.isfinite(ts_hi.target)))
