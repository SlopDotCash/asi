"""Protocol ceilings for pipeline array/smoke scan lengths."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.pipeline import (
    _PIPELINE_SCAN_MAX,
    make_alberta_pipeline,
    run_pipeline_smoke,
)


def test_documented_protocol_ceiling() -> None:
    assert _PIPELINE_SCAN_MAX == 10_000


def test_smoke_rejects_oversized_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_pipeline_smoke(steps=10_001)


def test_run_arrays_rejects_oversized_leading_axis() -> None:
    import jax.random as jr

    pipeline = make_alberta_pipeline()
    obs_dim = pipeline.config.features.observation_dim
    state = pipeline.init(jr.key(0), jnp.zeros(obs_dim, dtype=jnp.float32))
    n = _PIPELINE_SCAN_MAX + 1
    demons = pipeline.config.horde.n_demons
    with pytest.raises(ValueError, match="observations must contain between"):
        pipeline.run_arrays(
            state,
            jnp.zeros((n, obs_dim), dtype=jnp.float32),
            jnp.zeros((n,), dtype=jnp.float32),
            jnp.zeros((n,), dtype=jnp.float32),
            jnp.zeros((n, demons), dtype=jnp.float32),
        )
