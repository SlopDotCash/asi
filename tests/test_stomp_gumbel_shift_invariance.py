"""Regression coverage for STOMP's modeled Gumbel behavior policy."""

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.options import (
    _epsilon_greedy_action_probabilities,
    _select_action_epsilon_greedy_from_q,
    _select_action_epsilon_greedy_from_q_masked,
)


def test_gumbel_selector_is_invariant_to_finite_additive_shifts() -> None:
    base = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    shifted = jnp.asarray([16.0, 16.0], dtype=jnp.float32)

    for seed in range(64):
        key = jax.random.key(seed)
        base_action, _ = _select_action_epsilon_greedy_from_q(base, key, 0.0, 2)
        shifted_action, _ = _select_action_epsilon_greedy_from_q(shifted, key, 0.0, 2)
        assert int(base_action) == int(shifted_action)


def test_masked_gumbel_selector_centers_only_eligible_actions() -> None:
    mask = jnp.asarray([False, True, True])
    base = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
    shifted = jnp.asarray([1.0e8, 16.0, 16.0], dtype=jnp.float32)

    for seed in range(64):
        key = jax.random.key(seed)
        base_action, _ = _select_action_epsilon_greedy_from_q_masked(base, key, 0.0, mask)
        shifted_action, _ = _select_action_epsilon_greedy_from_q_masked(
            shifted, key, 0.0, mask
        )
        assert int(base_action) == int(shifted_action)


def test_modeled_policy_is_shift_invariant_and_finite_at_float32_scale() -> None:
    epsilon = jnp.asarray(0.2, dtype=jnp.float32)
    base = _epsilon_greedy_action_probabilities(
        jnp.asarray([0.0, 5.0e-7], dtype=jnp.float32), epsilon
    )
    shifted = _epsilon_greedy_action_probabilities(
        jnp.asarray([1.0e8, 1.0e8 + 5.0e-7], dtype=jnp.float32), epsilon
    )
    large_tie = _epsilon_greedy_action_probabilities(
        jnp.asarray([1.0e38, 1.0e38], dtype=jnp.float32), epsilon
    )

    np.testing.assert_array_equal(np.asarray(shifted), np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(np.asarray(base), [0.40203255, 0.5979675], atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(large_tie), [0.5, 0.5], atol=0.0)
    assert bool(jnp.all(jnp.isfinite(large_tie)))
