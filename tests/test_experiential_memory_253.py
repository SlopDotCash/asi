"""Regression test for #253: weight normalization must not floor with 1e-12.

When `score_sum` is a tiny positive float (e.g. exp(-36) ~ 2e-16 from a
single accepted neighbor at a sub-floor similarity), the previous
`jnp.maximum(score_sum, 1e-12)` floor produced weights ~2e-4 instead of
1.0. The fix uses `jnp.where(score_sum > 0.0, score_sum, 1.0)` so the
real score_sum is used whenever it's positive, falling back to 1.0 only
when it's exactly zero.
"""

import jax.numpy as jnp
import numpy as np

from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
)


def _config(**overrides):
    values = {
        "capacity": 2,
        "key_dim": 2,
        "observation_dim": 2,
        "action_dim": 2,
        "outcome_dim": 1,
        "top_k": 1,
        "min_similarity": 0.0,
        "min_effective_reliability": 1e-20,
        "distance_scale": 1.0,
        "staleness_scale": 1.0,
        "max_uncertainty": 1.0,
        "max_safety_cost": 1.0,
        "max_age": 100,
        "utility_decay": 1.0,
        "eviction_utility_weight": 1.0,
        "eviction_recency_weight": 1.0,
        "recency_scale": 10.0,
    }
    values.update(overrides)
    return ExperientialMemoryConfig(**values)


def test_neighbor_weights_use_real_score_sum_for_tiny_positive_sum():
    """Regression test for #253: weight normalization must not floor with 1e-12.

    When `score_sum` is a tiny positive float (e.g. exp(-36) ~ 2e-16 from a
    single accepted neighbor at a sub-floor similarity), the previous
    `jnp.maximum(score_sum, 1e-12)` floor produced weights ~2e-4 instead of
    1.0. The fix uses `jnp.where(score_sum > 0.0, score_sum, 1.0)` so the
    real score_sum is used whenever it's positive, falling back to 1.0 only
    when it's exactly zero.
    """
    # Configure for a single accepted neighbor with tiny similarity
    memory = ExperientialMemory(
        _config(
            capacity=2,
            key_dim=2,
            observation_dim= 2,
            action_dim=2,
            outcome_dim=1,
            top_k=1,
            min_similarity=0.0,
            min_effective_reliability=1e-20,
            distance_scale=1.0,
            staleness_scale=1.0,
            max_uncertainty=1.0,
            max_safety_cost=1.0,
            max_age=100,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=1.0,
            recency_scale=10.0,
        )
    )

    # Write exemplar at key (0, 0) with reliability 1.0
    state = memory.init()
    entry = ExperientialMemoryEntry(
        observation=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        key=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        action=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        outcome=jnp.asarray([1.0], dtype=jnp.float32),
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True),
        representation_version=jnp.asarray(1, dtype=jnp.int32),
        valid=jnp.asarray(True),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(1, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
    )
    state = memory.write(state, entry)
    assert bool(state.wrote)

    # Query with key at distance ~6 (squared distance 36) -> similarity exp(-36) ~ 2.3e-16
    query_key = jnp.asarray([6.0, 0.0], dtype=jnp.float32)
    retrieval = memory.query(
        state.state,
        query_key,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )

    # The score_sum will be ~2.3e-16 (tiny but positive)
    # With the fix, weight should be 1.0 (not floored to 1e-12)
    # So the returned action should match the stored action [1.0, 0.0]
    assert bool(retrieval.accepted)
    assert float(retrieval.neighbor_weights[0]) > 0.99  # should be ~1.0, not ~2e-4
    np.testing.assert_allclose(retrieval.action, jnp.asarray([1.0, 0.0]), atol=1e-6)


if __name__ == "__main__":
    test_neighbor_weights_use_real_score_sum_for_tiny_positive_sum()
    print("Test passed!")
