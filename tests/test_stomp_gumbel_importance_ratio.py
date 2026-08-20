"""Regression tests: STOMP importance ratios must match the Gumbel behavior policy.

Action selection draws ``argmax(q + _GUMBEL_TIE_BREAK_SCALE * Gumbel(0, 1))``,
whose greedy component is ``softmax(q / _GUMBEL_TIE_BREAK_SCALE)``.  The
importance-ratio helper previously treated any values within ``1e-6`` of the
maximum as a hard uniform tie, which disagreed with the policy that actually
generated the action for near-tied Q-values.  These tests pin the exact
Gumbel-max probabilities while preserving exact-tie and well-separated limits.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.options import (
    _GUMBEL_TIE_BREAK_SCALE,
    _clipped_epsilon_greedy_importance_ratio,
    _epsilon_greedy_action_probabilities,
)


def _reference_probabilities(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Independent NumPy reference for softmax-based epsilon-greedy probabilities."""
    q = np.asarray(q_values, dtype=np.float64)
    scaled = q / _GUMBEL_TIE_BREAK_SCALE
    scaled = scaled - scaled.max()
    greedy = np.exp(scaled) / np.exp(scaled).sum()
    n = q.shape[0]
    return epsilon / n + (1.0 - epsilon) * greedy


@pytest.mark.unit
def test_near_tie_probabilities_match_gumbel_max_distribution() -> None:
    """Near-tied Q-values follow softmax(q / scale), not a hard uniform split."""
    q_values = jnp.asarray([0.0, 5.0e-7], dtype=jnp.float32)

    greedy = _epsilon_greedy_action_probabilities(q_values, jnp.asarray(0.0))

    # The old hard-tie helper returned [0.5, 0.5] here; the correct Gumbel-max
    # greedy distribution is markedly asymmetric.
    np.testing.assert_allclose(
        np.asarray(greedy), [0.37754068, 0.62245935], rtol=0.0, atol=1e-6
    )


@pytest.mark.unit
def test_near_tie_importance_ratio_matches_issue_example() -> None:
    """The #2136 example: ratios must be [~0.939, ~1.041], not [1.0, 1.0]."""
    # q_weights @ observation = [0, 5e-7] with a single scalar feature.
    q_weights = jnp.asarray([[0.0], [5.0e-7]], dtype=jnp.float32)
    observation = jnp.asarray([1.0], dtype=jnp.float32)

    ratios = [
        float(
            _clipped_epsilon_greedy_importance_ratio(
                q_weights,
                observation,
                jnp.asarray(action, dtype=jnp.int32),
                behavior_epsilon=0.2,
                target_epsilon=0.0,
                clip=10.0,
            )
        )
        for action in (0, 1)
    ]

    np.testing.assert_allclose(ratios, [0.9390799, 1.0409585], rtol=0.0, atol=1e-6)
    # The regression being fixed: these must not collapse to the uniform-tie ratio.
    assert not np.allclose(ratios, [1.0, 1.0], atol=1e-3)


@pytest.mark.unit
def test_exact_ties_remain_uniform() -> None:
    """Exactly equal Q-values still split greedy mass uniformly."""
    q_values = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)

    greedy = _epsilon_greedy_action_probabilities(q_values, jnp.asarray(0.0))

    np.testing.assert_allclose(np.asarray(greedy), [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


@pytest.mark.unit
def test_well_separated_values_are_hard_greedy() -> None:
    """Q-values separated far beyond the tie-break scale act like a hard argmax."""
    q_values = jnp.asarray([0.0, 5.0], dtype=jnp.float32)

    greedy = _epsilon_greedy_action_probabilities(q_values, jnp.asarray(0.0))

    np.testing.assert_allclose(np.asarray(greedy), [0.0, 1.0], atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.35, 1.0])
def test_probabilities_are_normalised_and_match_reference(epsilon: float) -> None:
    """Probabilities sum to one and match an independent NumPy reference."""
    q_np = np.asarray([0.0, 2.0e-7, -3.0e-7, 1.0], dtype=np.float32)
    q_values = jnp.asarray(q_np, dtype=jnp.float32)

    probs = np.asarray(
        _epsilon_greedy_action_probabilities(q_values, jnp.asarray(epsilon))
    )

    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(
        probs, _reference_probabilities(q_np, epsilon), rtol=0.0, atol=1e-6
    )


@pytest.mark.unit
def test_importance_ratio_respects_clip() -> None:
    """A tight clip bounds the returned ratio."""
    q_weights = jnp.asarray([[0.0], [5.0e-7]], dtype=jnp.float32)
    observation = jnp.asarray([1.0], dtype=jnp.float32)

    ratio = float(
        _clipped_epsilon_greedy_importance_ratio(
            q_weights,
            observation,
            jnp.asarray(1, dtype=jnp.int32),
            behavior_epsilon=0.2,
            target_epsilon=0.0,
            clip=1.02,
        )
    )

    assert ratio == pytest.approx(1.02, abs=1e-6)
