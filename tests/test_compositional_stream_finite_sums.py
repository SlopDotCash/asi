"""Regression tests for finite component and pre-activation sums in CompositionalStream."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.streams.out_of_class import CompositionalStream


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
