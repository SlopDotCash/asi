"""Saturating lifetime clocks keep feature-discovery context identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.feature_discovery import (
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryStream,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _wrap_slot() -> int:
    return int((jnp.asarray(_INT32_MIN, dtype=jnp.int32) // 1) % 2)


def _sat_slot() -> int:
    return int((jnp.asarray(_INT32_MAX, dtype=jnp.int32) // 1) % 2)


def test_nonlinear_feature_discovery_wrap_would_change_context_slot() -> None:
    stream = NonlinearFeatureDiscoveryStream(
        feature_dim=4,
        n_tasks=2,
        n_latents=4,
        n_contexts=2,
        context_length=1,
        active_latents_per_context=1,
        feature_std=1.0,
        noise_std=0.0,
    )
    planted = stream.init(jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    sat_slot = int((advanced.step_count // 1) % 2)
    assert sat_slot == _sat_slot() == 1
    assert _wrap_slot() == 0
    assert sat_slot != _wrap_slot()


def test_interaction_feature_discovery_wrap_would_change_context_slot() -> None:
    stream = InteractionFeatureDiscoveryStream(
        feature_dim=4,
        n_tasks=2,
        n_contexts=2,
        context_length=1,
        active_pairs_per_context=1,
        feature_std=1.0,
        noise_std=0.0,
    )
    planted = stream.init(jr.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    sat_slot = int((advanced.step_count // 1) % 2)
    assert sat_slot == _sat_slot() == 1
    assert sat_slot != _wrap_slot()
