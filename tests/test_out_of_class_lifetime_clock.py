"""Saturating lifetime clocks keep out-of-class context identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.out_of_class import OutOfClassPolynomialStream

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_polynomial_stream_wrap_would_change_context_slot() -> None:
    """INT32_MAX+1 wrap selects a different context than saturate."""

    stream = OutOfClassPolynomialStream(
        feature_dim=4,
        n_tasks=1,
        n_contexts=2,
        context_length=1,
        active_triples_per_context=1,
        noise_std=0.0,
        feature_std=1.0,
        linear_scale=0.0,
    )
    planted = stream.init(jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    wrap_slot = int((jnp.asarray(_INT32_MIN, dtype=jnp.int32) // 1) % 2)
    sat_slot = int((advanced.step_count // 1) % 2)
    assert sat_slot != wrap_slot
    assert sat_slot == 1
    assert wrap_slot == 0
