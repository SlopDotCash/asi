"""Supplementary coverage for options.py pseudo-reward logic.

Covers previously untested helpers: compute_pseudo_reward (scaled feature
value) and check_option_terminated (goal-reached OR max-steps-exceeded).
"""

import jax.numpy as jnp

from alberta_framework.core.options import (
    STOMPSpecArrays,
    check_option_terminated,
    compute_pseudo_reward,
)


def _specs(n_options: int = 2) -> STOMPSpecArrays:
    return STOMPSpecArrays(
        feature_indices=jnp.array([0, 2], dtype=jnp.int32),
        thresholds=jnp.array([1.0, 5.0], dtype=jnp.float32),
        pseudo_reward_scales=jnp.array([2.0, 3.0], dtype=jnp.float32),
        max_option_steps=jnp.array([10, 20], dtype=jnp.int32),
    )


def test_compute_pseudo_reward_scaled_feature() -> None:
    spec = _specs()
    obs = jnp.array([1.0, 100.0, 2.0])
    # Option 0: scale 2.0 * obs[feature 0] = 2.0.
    assert compute_pseudo_reward(spec, jnp.array(0), obs) == 2.0
    # Option 1: scale 3.0 * obs[feature 2] = 6.0.
    assert compute_pseudo_reward(spec, jnp.array(1), obs) == 6.0


def test_compute_pseudo_reward_negative_obs() -> None:
    spec = _specs()
    obs = jnp.array([-1.0, 0.0, 0.5])
    assert compute_pseudo_reward(spec, jnp.array(0), obs) == -2.0


def test_check_option_terminated_goal_reached() -> None:
    spec = _specs()
    obs = jnp.array([1.0, 0.0, 0.0])  # option 0 reward = 2.0 >= 1.0 threshold
    assert bool(check_option_terminated(spec, jnp.array(0), obs, jnp.array(0)))


def test_check_option_terminated_max_steps() -> None:
    spec = _specs()
    obs = jnp.array([0.0, 0.0, 0.0])  # reward 0 < 1.0
    # steps 10 >= max 10 → terminated by max-steps.
    assert bool(check_option_terminated(spec, jnp.array(0), obs, jnp.array(10)))


def test_check_option_terminated_neither() -> None:
    spec = _specs()
    obs = jnp.array([0.0, 0.0, 0.0])
    assert not bool(check_option_terminated(spec, jnp.array(0), obs, jnp.array(5)))


def test_check_option_terminated_threshold_not_met() -> None:
    spec = _specs()
    # Option 1 threshold 5.0; reward 6.0 >= 5.0 → terminated.
    obs = jnp.array([0.0, 0.0, 2.0])
    assert bool(check_option_terminated(spec, jnp.array(1), obs, jnp.array(0)))
