# mypy: disable-error-code="call-arg"
"""Saturating lifetime clock keeps DreamRolloutState.step_count non-negative."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.dreaming import (
    DreamBehaviorModelPrediction,
    DreamRolloutState,
    DreamWorldModelPrediction,
    dream_one_step,
    init_dream_rollout_state,
)

_INT32_MAX = 2**31 - 1


@dataclasses.dataclass(frozen=True)
class MockWorldModel:
    def predict(
        self, state: Any, observation: Array, action: Array, key: Array
    ) -> DreamWorldModelPrediction:
        return DreamWorldModelPrediction(
            next_observation=observation + 0.1,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            confidence=jnp.asarray(1.0, dtype=jnp.float32),
            model_error=jnp.asarray(0.0, dtype=jnp.float32),
        )


@dataclasses.dataclass(frozen=True)
class MockBehaviorModel:
    def sample_action(
        self, state: Any, observation: Array, key: Array
    ) -> DreamBehaviorModelPrediction:
        return DreamBehaviorModelPrediction(
            action=jnp.asarray(1, dtype=jnp.int32),
            action_probability=jnp.asarray(1.0, dtype=jnp.float32),
            log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        )


def test_dream_one_step_step_count_saturates_at_int32_max() -> None:
    world = MockWorldModel()
    behavior = MockBehaviorModel()
    initial = init_dream_rollout_state(
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jr.key(13),
    )
    near_max = DreamRolloutState(
        observation=initial.observation,
        rng_key=initial.rng_key,
        active=initial.active,
        cumulative_confidence=initial.cumulative_confidence,
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )

    next_state, _ = dream_one_step(
        world_model=world,
        world_state=None,
        behavior_model=behavior,
        behavior_state=None,
        rollout_state=near_max,
    )
    assert int(next_state.step_count) == _INT32_MAX
    assert int(next_state.step_count) >= 0


def test_dream_one_step_step_count_increments_below_max() -> None:
    world = MockWorldModel()
    behavior = MockBehaviorModel()
    initial = init_dream_rollout_state(
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jr.key(13),
    )

    next_state, _ = dream_one_step(
        world_model=world,
        world_state=None,
        behavior_model=behavior,
        behavior_state=None,
        rollout_state=initial,
    )
    assert int(next_state.step_count) == 1
