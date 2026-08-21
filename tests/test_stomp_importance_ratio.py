"""Importance-ratio parity with the Gumbel intra-option behavior policy.

STOMP's intra-option selectors break ties with
``jnp.argmax(q + _GUMBEL_TIEBREAK_TEMPERATURE * gumbel(...))``.  By the
Gumbel-max trick, the greedy component of that policy is
``softmax(q / _GUMBEL_TIEBREAK_TEMPERATURE)`` — not a uniform split over
near-tied Q-values.  The importance-ratio helper must model the exact same
distribution, otherwise ``_update_intra_option_policy`` scales traces and
updates with probabilities from a policy that never generated the action.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import (
    _GUMBEL_TIEBREAK_TEMPERATURE,
    _clipped_epsilon_greedy_importance_ratio,
    _epsilon_greedy_action_probabilities,
    _select_action_epsilon_greedy_from_q,
)


def test_near_tie_probabilities_follow_gumbel_softmax() -> None:
    """Near-tied Q-values give the softmax split, not a uniform split."""
    q = jnp.array([0.0, 5.0e-7], dtype=jnp.float32)
    target = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.0, dtype=jnp.float32))
    # Uniform tie handling (the old behavior) would return [0.5, 0.5]; the
    # Gumbel-max policy at temperature 1e-6 returns softmax([0, 0.5]).
    np.testing.assert_allclose(
        np.asarray(target), np.array([0.37754068, 0.62245935]), atol=1e-5
    )
    assert not np.allclose(np.asarray(target), np.array([0.5, 0.5]), atol=1e-3)


def test_near_tie_importance_ratio_matches_behavior_policy() -> None:
    """The clipped ratio uses target/behavior of the true Gumbel policy."""
    q_weights = jnp.eye(2, dtype=jnp.float32)
    observation = jnp.array([0.0, 5.0e-7], dtype=jnp.float32)
    ratios = []
    for action in (0, 1):
        ratio = _clipped_epsilon_greedy_importance_ratio(
            q_weights,
            observation,
            jnp.asarray(action, dtype=jnp.int32),
            behavior_epsilon=0.2,
            target_epsilon=0.0,
            clip=1.0e6,  # large clip so it never binds
        )
        ratios.append(float(ratio))
    np.testing.assert_allclose(
        np.array(ratios), np.array([0.9390799, 1.0409585]), atol=1e-5
    )
    # The buggy uniform helper returned exactly [1.0, 1.0] here.
    assert not np.allclose(np.array(ratios), np.array([1.0, 1.0]), atol=1e-3)


def test_exact_ties_still_split_uniformly() -> None:
    """Exactly-tied Q-values retain the uniform greedy split and unit ratio."""
    q = jnp.array([1.0, 1.0], dtype=jnp.float32)
    probs = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.0, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(probs), np.array([0.5, 0.5]), atol=1e-6)
    q_weights = jnp.eye(2, dtype=jnp.float32)
    observation = jnp.array([1.0, 1.0], dtype=jnp.float32)
    for action in (0, 1):
        ratio = _clipped_epsilon_greedy_importance_ratio(
            q_weights,
            observation,
            jnp.asarray(action, dtype=jnp.int32),
            behavior_epsilon=0.2,
            target_epsilon=0.0,
            clip=1.0e6,
        )
        np.testing.assert_allclose(float(ratio), 1.0, atol=1e-6)


def test_well_separated_q_collapses_to_argmax() -> None:
    """Well-separated Q-values give near one-hot greedy mass on the argmax."""
    q = jnp.array([0.0, 3.0, 1.0], dtype=jnp.float32)
    greedy_probs = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.0, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(greedy_probs), np.array([0.0, 1.0, 0.0]), atol=1e-6)
    # Parity with the selector: the exploration-free selector always picks argmax.
    for seed in range(16):
        action, _ = _select_action_epsilon_greedy_from_q(
            q, jr.key(seed), 0.0, q.shape[0]
        )
        assert int(action) == 1


@pytest.mark.parametrize("temp", [_GUMBEL_TIEBREAK_TEMPERATURE])
def test_temperature_constant_is_shared_with_selectors(temp: float) -> None:
    """The module constant equals the historical inline tie-break scale."""
    assert temp == 1.0e-6
