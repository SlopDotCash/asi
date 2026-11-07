"""Regression tests for finite observation noise in ClassicalConditioningStream."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
    PavlovianState,
)

pytestmark = pytest.mark.unit

_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _make_phase(n_steps: int = 64) -> PavlovianPhase:
    return PavlovianPhase(  # type: ignore[call-arg]
        name="acquisition",
        n_steps=n_steps,
        cs_us_contingency=1.0,
        cs_active=(0,),
    )


def test_pavlovian_float32_max_noise_std_emits_finite_observations() -> None:
    """Constructor-admitted float32 max noise_std must not emit +-Inf observations."""
    stream = ClassicalConditioningStream(
        phases=(_make_phase(),),
        n_cs=2,
        n_distractors=2,
        noise_std=_FLOAT32_MAX,
    )
    state = stream.init(jr.key(0))

    def step_fn(
        carry: PavlovianState, idx: jax.Array
    ) -> tuple[PavlovianState, tuple[jax.Array, jax.Array]]:
        ts, next_state = stream.step(carry, idx)
        return next_state, (ts.observation, ts.target)

    _, (obs, tgt) = jax.jit(lambda s: jax.lax.scan(step_fn, s, jnp.arange(32)))(state)
    assert obs.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert bool(jnp.all(jnp.isfinite(tgt)))
    assert bool(jnp.any(obs == jnp.float32(_FLOAT32_MAX)))
    assert bool(jnp.any(obs == jnp.float32(-_FLOAT32_MAX)))


def test_pavlovian_near_max_noise_std_saturates_overflowing_draws_only() -> None:
    """Near-max noise_std saturates overflowing normal draws while keeping in-range draws exact."""
    noise_std = float(np.float32(3.0e38))
    stream = ClassicalConditioningStream(
        phases=(_make_phase(),),
        n_cs=8,
        n_distractors=8,
        iti_min=10,
        iti_max=10,
        distractor_prob=0.0,
        noise_std=noise_std,
    )
    state = stream.init(jr.key(7))
    _, _, _, _, _, _, k_noise = jr.split(state.key, 7)
    raw_normal = jr.normal(k_noise, (stream.feature_dim,), dtype=jnp.float32)
    raw_noise = jnp.float32(noise_std) * raw_normal
    finite_mask = jnp.isfinite(raw_noise)
    assert bool(jnp.any(finite_mask))
    assert bool(jnp.any(~finite_mask))

    ts, _ = stream.step(state, jnp.array(0, dtype=jnp.int32))
    assert bool(jnp.all(jnp.isfinite(ts.observation)))
    assert bool(jnp.array_equal(ts.observation[finite_mask], raw_noise[finite_mask]))
    expected_saturated = jnp.where(
        raw_normal[~finite_mask] > 0.0,
        jnp.float32(_FLOAT32_MAX),
        jnp.float32(-_FLOAT32_MAX),
    )
    assert bool(jnp.array_equal(ts.observation[~finite_mask], expected_saturated))


def test_pavlovian_ordinary_noise_std_preserves_bitwise_observations() -> None:
    """Ordinary finite noise_std preserves exact float32 observation bits."""
    stream = ClassicalConditioningStream(
        phases=(_make_phase(),),
        n_cs=2,
        n_distractors=2,
        iti_min=2,
        iti_max=4,
        noise_std=0.05,
        distractor_prob=0.25,
    )
    state = stream.init(jr.key(19))
    for step_idx in range(16):
        _, _, _, _, _, _, k_noise = jr.split(state.key, 7)
        ts, next_state = stream.step(state, jnp.array(step_idx, dtype=jnp.int32))
        cs_ind = (next_state.cs_active_steps_remaining > 0).astype(jnp.float32)
        dist_ind = (next_state.n_distractor_active > 0).astype(jnp.float32)
        clean_obs = jnp.concatenate([cs_ind, dist_ind])
        noise = jnp.float32(stream._noise_std) * jr.normal(  # noqa: SLF001
            k_noise, (stream.feature_dim,), dtype=jnp.float32
        )
        direct_obs = (clean_obs + noise).astype(jnp.float32)
        assert bool(jnp.array_equal(ts.observation, direct_obs))
        state = next_state
