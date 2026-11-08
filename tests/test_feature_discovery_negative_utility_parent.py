"""Negative-utility parent selection must not collapse to one slot.

``FixedBudgetFeatureLearner._generate_one`` samples a mutate parent with
``jr.categorical`` over ``log(utilities + 1e-3)``.  Utilities are a
non-negative EMA on the public update path, but a restored or hand-built
state can carry negative utilities; ``log`` of a negative argument is NaN,
and categorical sampling over NaN logits silently returns slot 0 for every
key, so every parent-mutation regenerates from the same slot instead of
sampling by utility.  The sibling compositional learner already guards the
identical computation with ``log(maximum(utility_scores, 1e-6))``.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.feature_discovery import FixedBudgetFeatureLearner


@pytest.mark.unit
def test_negative_utilities_do_not_collapse_mutate_parent_to_slot_zero() -> None:
    """Mutate-parent sampling stays distributed with negative utilities."""
    learner = FixedBudgetFeatureLearner(
        n_features=4,
        n_tasks=2,
        generator_mix=(0.0, 1.0, 0.0),
        candidate_count=0,
        replacement_interval=1,
        min_feature_age=0,
    )
    state = learner.init(3, jr.key(0))
    negative = state.replace(
        utilities=jnp.array([-0.5, -2.0, -0.01, 0.0], dtype=jnp.float32)
    )
    observation = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    parents: set[int] = set()
    for seed in range(16):
        _, _, parent_a, _, _ = learner._generate_one(
            jr.key(1000 + seed),
            observation,
            negative.feature_weights,
            negative.feature_biases,
            negative.utilities,
        )
        parents.add(int(parent_a))
    # Before the fix every key returns parent 0 (NaN logits); after the fix
    # the negative utilities behave like zero utility and sampling is uniform,
    # so 16 draws land on more than one parent with overwhelming probability.
    assert len(parents) > 1, f"mutate parent collapsed to a single slot: {parents}"


@pytest.mark.unit
def test_negative_utilities_keep_update_finite_and_applied() -> None:
    """A replacement update from negative utilities stays finite."""
    learner = FixedBudgetFeatureLearner(
        n_features=4,
        n_tasks=2,
        generator_mix=(0.0, 1.0, 0.0),
        candidate_count=0,
        replacement_interval=1,
        min_feature_age=0,
    )
    state = learner.init(3, jr.key(0))
    negative = state.replace(
        utilities=jnp.array([-0.5, -2.0, -0.01, 0.0], dtype=jnp.float32)
    )
    result = learner.update(
        negative,
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.array([1.0, -1.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert bool(jnp.all(jnp.isfinite(result.state.feature_weights)))
    assert bool(jnp.all(jnp.isfinite(result.state.utilities)))
