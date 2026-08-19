"""Tests for scan sequence bounds and array validation across FTL, Latent WM, CBP, OVD, Step 4."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.continual_backprop import (
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    run_cbp_learning_loop,
)
from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
    run_sparse_ftl_world_model,
)
from alberta_framework.core.latent_world_model import (
    LatentWorldModel,
    LatentWorldModelConfig,
    run_latent_world_model_learning_loop,
)
from alberta_framework.core.learners import (
    LinearLearner,
    MLPLearner,
    TDLinearLearner,
    TrueOnlineTDLearner,
    run_learning_loop,
    run_learning_loop_batched,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_td_learning_loop,
    run_true_online_td_loop,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.option_value_duration import (
    OptionValueDurationLearner,
    run_option_value_duration_from_arrays,
)
from alberta_framework.core.types import TDTimeStep
from alberta_framework.steps.step4 import (
    init_step4_state,
    make_step4_sarsa_agent,
    run_step4_scan,
)
from alberta_framework.streams.synthetic import RandomWalkStream


class SimpleTDStream:
    def __init__(self, feature_dim: int = 4, gamma: float = 0.99) -> None:
        self.feature_dim = feature_dim
        self._gamma = gamma

    def init(self, key: Array) -> dict[str, Any]:
        return {"key": key, "step": 0}

    def step(self, state: dict[str, Any], idx: Array) -> tuple[TDTimeStep, dict[str, Any]]:
        key = state["key"]
        key, obs_key, next_key, reward_key = jr.split(key, 4)
        observation = jr.normal(obs_key, (self.feature_dim,), dtype=jnp.float32)
        next_observation = jr.normal(next_key, (self.feature_dim,), dtype=jnp.float32)
        reward = jr.normal(reward_key, (), dtype=jnp.float32)
        gamma = jnp.array(self._gamma, dtype=jnp.float32)
        timestep = TDTimeStep(  # type: ignore[call-arg]
            observation=observation,
            reward=reward,
            next_observation=next_observation,
            gamma=gamma,
        )
        return timestep, {"key": key, "step": state["step"] + 1}


# ---------------------------------------------------------------------------
# Sparse FTL World Model Scan Tests
# ---------------------------------------------------------------------------


def test_sparse_ftl_world_model_scan_validation() -> None:
    config = SparseFTLWorldModelConfig(observation_dim=3, action_dim=1, projection_dim=4, bins=4)
    model = SparseFTLWorldModel(config)
    key = jr.key(0)
    state = model.init(key)

    obs = jnp.zeros((5, 3), dtype=jnp.float32)
    next_obs = jnp.zeros((5, 3), dtype=jnp.float32)
    actions_1d = jnp.zeros((5,), dtype=jnp.float32)
    actions_2d = jnp.zeros((5, 1), dtype=jnp.float32)

    # Valid run with 1D action
    res1 = run_sparse_ftl_world_model(model, state, obs, actions_1d, next_obs)
    assert res1.predicted_next_observations.shape == (5, 3)

    # Valid run with 2D action
    res2 = run_sparse_ftl_world_model(model, state, obs, actions_2d, next_obs)
    assert res2.predicted_next_observations.shape == (5, 3)

    # Invalid model / state
    with pytest.raises(TypeError, match="model must be an exact"):
        run_sparse_ftl_world_model(None, state, obs, actions_1d, next_obs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_sparse_ftl_world_model(model, None, obs, actions_1d, next_obs)  # type: ignore[arg-type]

    # Untrusted array / invalid shape
    with pytest.raises(TypeError, match="must be a trusted array"):
        run_sparse_ftl_world_model(model, state, [[0.0, 0.0, 0.0]], actions_1d, next_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observations must have shape"):
        run_sparse_ftl_world_model(model, state, jnp.zeros((5, 2)), actions_1d, next_obs)
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_sparse_ftl_world_model(model, state, obs, actions_1d, jnp.zeros((5, 4)))
    with pytest.raises(ValueError, match="actions must have shape"):
        run_sparse_ftl_world_model(model, state, obs, jnp.zeros((5, 2)), next_obs)
    with pytest.raises(ValueError, match="scan sequence length"):
        run_sparse_ftl_world_model(
            model, state, jnp.zeros((0, 3)), jnp.zeros((0,)), jnp.zeros((0, 3))
        )


# ---------------------------------------------------------------------------
# Latent World Model Scan Tests
# ---------------------------------------------------------------------------


def test_latent_world_model_scan_validation() -> None:
    # Discrete actions
    config_discrete = LatentWorldModelConfig(
        observation_dim=4,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(8,),
    )
    model_d = LatentWorldModel(config_discrete)
    state_d = model_d.init(jr.key(1))

    obs = jnp.zeros((6, 4), dtype=jnp.float32)
    next_obs = jnp.zeros((6, 4), dtype=jnp.float32)
    actions_d = jnp.zeros((6,), dtype=jnp.int32)
    rewards = jnp.zeros((6,), dtype=jnp.float32)
    discounts = jnp.ones((6,), dtype=jnp.float32)

    # Valid run
    res_d = run_latent_world_model_learning_loop(
        model_d, state_d, obs, actions_d, rewards, next_obs, discounts
    )
    assert res_d.latent_predictions.shape == (6, 4)

    # Invalid types
    with pytest.raises(TypeError, match="model must be an exact"):
        run_latent_world_model_learning_loop(
            None,
            state_d,
            obs,
            actions_d,
            rewards,
            next_obs,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="state must be an exact"):
        run_latent_world_model_learning_loop(
            model_d,
            None,
            obs,
            actions_d,
            rewards,
            next_obs,  # type: ignore[arg-type]
        )

    # Shape errors
    with pytest.raises(ValueError, match="observations must have shape"):
        run_latent_world_model_learning_loop(
            model_d, state_d, jnp.zeros((6, 5)), actions_d, rewards, next_obs
        )
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_latent_world_model_learning_loop(
            model_d, state_d, obs, actions_d, rewards, jnp.zeros((6, 3))
        )
    with pytest.raises(ValueError, match="actions must have shape"):
        run_latent_world_model_learning_loop(
            model_d, state_d, obs, jnp.zeros((6, 2)), rewards, next_obs
        )
    with pytest.raises(ValueError, match="rewards must have shape"):
        run_latent_world_model_learning_loop(
            model_d, state_d, obs, actions_d, jnp.zeros((5,)), next_obs
        )
    with pytest.raises(ValueError, match="discounts must have shape"):
        run_latent_world_model_learning_loop(
            model_d, state_d, obs, actions_d, rewards, next_obs, discounts=jnp.zeros((5,))
        )
    with pytest.raises(ValueError, match="scan sequence length"):
        run_latent_world_model_learning_loop(
            model_d,
            state_d,
            jnp.zeros((0, 4)),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,)),
            jnp.zeros((0, 4)),
        )


# ---------------------------------------------------------------------------
# Continual Backprop Scan Tests
# ---------------------------------------------------------------------------


def test_cbp_scan_validation() -> None:
    learner = CBPMultiHeadMLPLearner(
        n_heads=2,
        hidden_sizes=(8,),
        cbp_config=ContinualBackpropConfig(),
    )
    state = learner.init(feature_dim=4, key=jr.key(2))

    obs = jnp.zeros((5, 4), dtype=jnp.float32)
    targets = jnp.zeros((5, 2), dtype=jnp.float32)

    # Valid run
    res = run_cbp_learning_loop(learner, state, obs, targets)
    assert res.per_head_metrics.shape == (5, 2, 3)

    # Invalid types
    with pytest.raises(TypeError, match="learner must be an exact"):
        run_cbp_learning_loop(None, state, obs, targets)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_cbp_learning_loop(learner, None, obs, targets)  # type: ignore[arg-type]

    # Shape errors
    with pytest.raises(ValueError, match="observations must have shape"):
        run_cbp_learning_loop(learner, state, jnp.zeros((5,)), targets)
    with pytest.raises(ValueError, match="targets must have shape"):
        run_cbp_learning_loop(learner, state, obs, jnp.zeros((5, 3)))
    with pytest.raises(ValueError, match="scan sequence length"):
        run_cbp_learning_loop(learner, state, jnp.zeros((0, 4)), jnp.zeros((0, 2)))


# ---------------------------------------------------------------------------
# Option Value Duration Scan Tests
# ---------------------------------------------------------------------------


def test_option_value_duration_scan_validation() -> None:
    learner = OptionValueDurationLearner(n_options=2)
    state = learner.init(feature_dim=4)

    obs = jnp.zeros((4, 4), dtype=jnp.float32)
    options = jnp.zeros((4,), dtype=jnp.int32)
    rewards = jnp.zeros((4,), dtype=jnp.float32)
    next_obs = jnp.zeros((4, 4), dtype=jnp.float32)
    discounts = jnp.ones((4,), dtype=jnp.float32)

    # Valid run
    res = run_option_value_duration_from_arrays(
        learner, state, obs, options, rewards, next_obs, discounts
    )
    assert res.predictions.shape == (4, 2)

    # Invalid sequence length
    with pytest.raises(ValueError, match="scan sequence length"):
        run_option_value_duration_from_arrays(
            learner,
            state,
            jnp.zeros((0, 4), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0, 4), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Step 4 SARSA Scan Tests
# ---------------------------------------------------------------------------


def test_step4_scan_validation() -> None:
    agent = make_step4_sarsa_agent()
    state = init_step4_state(
        agent,
        feature_dim=4,
        key=jr.key(3),
        initial_features=jnp.zeros((4,), dtype=jnp.float32),
    )

    next_features = jnp.zeros((5, 4), dtype=jnp.float32)
    rewards = jnp.zeros((5,), dtype=jnp.float32)
    terminated = jnp.zeros((5,), dtype=jnp.float32)

    # Valid run
    res = run_step4_scan(agent, state, next_features, rewards, terminated)
    assert res.q_values.shape[0] == 5

    # Invalid types
    with pytest.raises(TypeError, match="agent must be an exact"):
        run_step4_scan(None, state, next_features, rewards, terminated)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be an exact"):
        run_step4_scan(agent, None, next_features, rewards, terminated)  # type: ignore[arg-type]

    # Shape errors
    with pytest.raises(ValueError, match="next_features must have shape"):
        run_step4_scan(agent, state, jnp.zeros((5,)), rewards, terminated)
    with pytest.raises(ValueError, match="rewards must have shape"):
        run_step4_scan(agent, state, next_features, jnp.zeros((4,)), terminated)
    with pytest.raises(ValueError, match="terminated must have shape"):
        run_step4_scan(agent, state, next_features, rewards, jnp.zeros((4,)))
    with pytest.raises(ValueError, match="positive"):
        run_step4_scan(
            agent,
            state,
            jnp.zeros((0, 4), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Core Learners num_steps Tests
# ---------------------------------------------------------------------------


def test_core_learners_num_steps_validation() -> None:
    stream = RandomWalkStream(feature_dim=4)
    td_stream = SimpleTDStream(feature_dim=4)

    # LinearLearner
    linear = LinearLearner(optimizer=LMS(step_size=0.01))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_learning_loop(linear, stream, num_steps=0, key=jr.key(0))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_learning_loop(linear, stream, num_steps=cast(int, True), key=jr.key(0))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_learning_loop_batched(linear, stream, num_steps=-1, keys=jr.split(jr.key(0), 2))

    # MLPLearner
    mlp = MLPLearner(hidden_sizes=(8,))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_mlp_learning_loop(mlp, stream, num_steps=0, key=jr.key(0))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_mlp_learning_loop_batched(mlp, stream, num_steps=-5, keys=jr.split(jr.key(0), 2))

    # TDLinearLearner
    td = TDLinearLearner()
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_td_learning_loop(td, td_stream, num_steps=0, key=jr.key(0))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_td_learning_loop(td, td_stream, num_steps=cast(int, False), key=jr.key(0))

    # TrueOnlineTDLearner
    true_td = TrueOnlineTDLearner()
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_true_online_td_loop(true_td, td_stream, num_steps=0, key=jr.key(0))
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        run_true_online_td_loop(true_td, td_stream, num_steps=-1, key=jr.key(0))
