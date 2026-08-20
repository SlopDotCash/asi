"""Protocol ceilings for Forager JAX chunk scans.

Public last-fit is jax_chunk_size=10_000 (and tests use 4096). Origin accepted
INT32-legal chunks and scanned jnp.arange(chunk) — hang, not leftover INT32 math.
"""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import ForagerBenchmarkConfig


def test_rejects_oversized_jax_chunks() -> None:
    with pytest.raises(ValueError, match="jax_chunk_size"):
        ForagerBenchmarkConfig(steps=8, jax_chunk_size=10_001)
