"""Tests for the online one-step world model."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.world_model import (
    OneStepWorldModel,
    WorldModelConfig,
    run_world_model_learning_loop,
)


@pytest.mark.parametrize(
    "malformed_observation",
    [
        pytest.param(jnp.zeros((2, 1), dtype=jnp.float32), id="column"),
        pytest.param(jnp.zeros((1, 2), dtype=jnp.float32), id="row"),
        pytest.param(jnp.zeros((1,), dtype=jnp.float32), id="short"),
    ],
)
def test_world_model_rejects_malformed_observation_vectors(
    malformed_observation: jax.Array,
) -> None:
    """Wrong-rank vectors must not be silently flattened through jitted APIs."""
    model = OneStepWorldModel(
        WorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=())
    )
    state = model.init(jr.key(10))
    observation = jnp.zeros((2,), dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)
    reward = jnp.array(0.0, dtype=jnp.float32)

    calls = (
        lambda: model.input_features(malformed_observation, action),
        lambda: model.targets(malformed_observation, reward, observation),
        lambda: model.targets(observation, reward, malformed_observation),
        lambda: model.predict(state, malformed_observation, action),
        lambda: model.update(state, malformed_observation, action, reward, observation),
        lambda: model.update(state, observation, action, reward, malformed_observation),
    )
    for call in calls:
        with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
            call()


def test_world_model_rejects_malformed_continuous_actions() -> None:
    model = OneStepWorldModel(
        WorldModelConfig(
            observation_dim=2,
            n_actions=None,
            action_dim=2,
            hidden_sizes=(),
        )
    )
    state = model.init(jr.key(11))
    observation = jnp.zeros((2,), dtype=jnp.float32)
    malformed_action = jnp.zeros((1, 2), dtype=jnp.float32)

    calls = (
        lambda: model.encode_action(malformed_action),
        lambda: model.input_features(observation, malformed_action),
        lambda: model.predict(state, observation, malformed_action),
        lambda: model.update(
            state,
            observation,
            malformed_action,
            jnp.array(0.0, dtype=jnp.float32),
            observation,
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match=r"action must have shape \(2,\)"):
            call()


def test_world_model_update_is_finite_and_shape_stable() -> None:
    cfg = WorldModelConfig(
        observation_dim=3,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = OneStepWorldModel(cfg)
    state = model.init(jr.key(0))

    obs = jnp.array([0.2, -0.1, 0.5], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)
    reward = jnp.array(0.75, dtype=jnp.float32)
    next_obs = jnp.array([0.3, -0.2, 0.9], dtype=jnp.float32)

    result = model.update(state, obs, action, reward, next_obs)

    chex.assert_shape(result.prediction.next_observation, (3,))
    chex.assert_shape(result.prediction.raw_predictions, (4,))
    chex.assert_shape(result.errors, (4,))
    chex.assert_shape(result.per_head_metrics, (4, 3))
    chex.assert_tree_all_finite(result.prediction.raw_predictions)
    chex.assert_tree_all_finite(result.reward_error)
    assert int(result.state.step_count) == 1


def test_world_model_config_roundtrip_preserves_action_encoding() -> None:
    cfg = WorldModelConfig(
        observation_dim=5,
        n_actions=4,
        hidden_sizes=(8, 4),
        step_size=0.02,
        predict_delta=True,
    )
    model = OneStepWorldModel(cfg)
    restored = OneStepWorldModel.from_config(model.to_config())

    assert restored.config == cfg
    chex.assert_trees_all_close(
        restored.encode_action(jnp.array(2)),
        jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32),
    )


def test_world_model_nan_targets_mask_missing_heads() -> None:
    cfg = WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.1,
        sparsity=0.0,
    )
    model = OneStepWorldModel(cfg)
    state = model.init(jr.key(1))

    result = model.update(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(jnp.nan, dtype=jnp.float32),
        jnp.array([0.5, jnp.nan], dtype=jnp.float32),
    )

    assert bool(jnp.isnan(result.per_head_metrics[0, 0]))
    assert bool(jnp.isfinite(result.per_head_metrics[1, 0]))
    assert bool(jnp.isnan(result.per_head_metrics[2, 0]))


def test_world_model_scan_is_jit_compatible() -> None:
    cfg = WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = OneStepWorldModel(cfg)
    state = model.init(jr.key(2))
    observations = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    actions = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    rewards = actions.astype(jnp.float32)
    next_observations = observations + rewards[:, None]

    result = jax.jit(
        lambda s: run_world_model_learning_loop(
            model,
            s,
            observations,
            actions,
            rewards,
            next_observations,
        )
    )(state)

    chex.assert_shape(result.reward_predictions, (4,))
    chex.assert_shape(result.next_observation_predictions, (4, 2))
    chex.assert_tree_all_finite(result.reward_predictions)
    chex.assert_tree_all_finite(result.next_observation_predictions)


def test_world_model_learns_action_conditional_deterministic_transition() -> None:
    cfg = WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.08,
        sparsity=0.0,
    )
    model = OneStepWorldModel(cfg)
    state = model.init(jr.key(3))

    base = jnp.tile(
        jnp.array(
            [[0.0, 0.0], [1.0, -1.0], [-0.5, 0.5], [0.25, -0.75]],
            dtype=jnp.float32,
        ),
        (160, 1),
    )
    actions = jnp.arange(base.shape[0], dtype=jnp.int32) % 2
    action_f = actions.astype(jnp.float32)
    rewards = 0.25 + 0.5 * action_f + 0.1 * base[:, 0]
    next_observations = jnp.stack(
        [
            base[:, 0] + action_f,
            base[:, 1] - 0.5 * action_f,
        ],
        axis=1,
    )

    result = run_world_model_learning_loop(
        model,
        state,
        base,
        actions,
        rewards,
        next_observations,
    )
    result.reward_errors.block_until_ready()
    first_mse = jnp.nanmean(result.per_head_metrics[:32, :, 0])
    last_mse = jnp.nanmean(result.per_head_metrics[-32:, :, 0])

    pred_a0 = model.predict(
        result.state,
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    pred_a1 = model.predict(
        result.state,
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
    )

    assert float(last_mse) < float(first_mse)
    assert float(pred_a1.reward - pred_a0.reward) > 0.25
    assert float(pred_a1.next_observation[0] - pred_a0.next_observation[0]) > 0.5
