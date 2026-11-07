"""Off-policy TD updates must reject corrupted lifetime counters atomically."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.off_policy_td import (
    ETDLinearLearner,
    GradientTDLinearLearner,
    OffPolicyTDLinearLearner,
)


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize(
    "learner",
    [
        OffPolicyTDLinearLearner(step_size=0.1),
        ETDLinearLearner(step_size=0.1),
        GradientTDLinearLearner(step_size=0.1),
    ],
)
def test_negative_step_count_rejects_update_atomically(
    learner: OffPolicyTDLinearLearner | ETDLinearLearner | GradientTDLinearLearner,
    disable_jit: bool,
) -> None:
    """A corrupted lifetime must not be rewritten into apparently valid state."""
    state = learner.init(1).replace(step_count=jnp.asarray(-1, dtype=jnp.int32))

    with jax.disable_jit(disable_jit):
        result = learner.update(
            state,
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    for name in ("prediction", "td_error", "metrics"):
        assert bool(jnp.all(jnp.asarray(getattr(result, name)) == 0))
    for name in ("rho_clipped", "follow_on_trace", "emphasis"):
        if hasattr(result, name):
            assert bool(jnp.all(jnp.asarray(getattr(result, name)) == 0))
