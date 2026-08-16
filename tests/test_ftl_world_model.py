"""Tests for the sparse lifetime-statistics FTL world model."""

from __future__ import annotations

import json
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
    run_sparse_ftl_world_model,
)


def _state_nbytes(state: object) -> int:
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(state))


def _shared_dynamics_stream(
    num_steps: int,
    low: float,
    high: float,
    phase: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    time = jnp.arange(num_steps, dtype=jnp.float32)
    observation = (low + (high - low) * (0.5 + 0.5 * jnp.sin(0.173 * time + phase)))[:, None]
    action = (0.8 * jnp.cos(0.119 * time + 1.7 * phase))[:, None]
    delta = 0.25 * observation + 0.55 * action + 0.15 * jnp.sin(1.7 * observation)
    return observation, action, observation + delta


def _shared_dynamics_mse(
    model: SparseFTLWorldModel,
    state: object,
    low: float,
    high: float,
) -> float:
    observation = jnp.linspace(low, high, 161, dtype=jnp.float32)[:, None]
    action = (0.8 * jnp.sin(jnp.linspace(-3.0, 3.0, 161, dtype=jnp.float32)))[:, None]
    target = 0.25 * observation + 0.55 * action + 0.15 * jnp.sin(1.7 * observation)
    prediction = jax.vmap(lambda obs, act: model.predict(state, obs, act).delta)(
        observation,
        action,
    )
    return float(jnp.mean((prediction - target) ** 2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", 0),
        ("action_dim", 0),
        ("projection_dim", 0),
        ("bins", 1),
        ("ridge", 0.0),
        ("ridge", float("nan")),
        ("statistics_decay", 0.0),
        ("statistics_decay", 1.01),
        ("prediction_clip", 0.0),
        ("prediction_clip", float("nan")),
        ("error_decay", 1.0),
    ],
)
def test_config_rejects_invalid_parameters(field: str, value: int | float) -> None:
    kwargs: dict[str, int | float] = {
        "observation_dim": 2,
        "action_dim": 1,
        "projection_dim": 4,
        "bins": 5,
        "ridge": 0.01,
        "statistics_decay": 1.0,
        "prediction_clip": 10.0,
        "error_decay": 0.99,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ridge", "prediction_clip"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("0.01", id="string"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(object(), id="object"),
    ],
)
def test_config_rejects_non_real_or_boolean_execution_scalars(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=rf"{field} must be a finite positive normal float32"):
        SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ridge", "prediction_clip"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            float(np.nextafter(np.float32(0.0), np.float32(1.0))),
            id="float32-subnormal",
        ),
        pytest.param(np.nextafter(0.0, 1.0), id="underflows-to-zero"),
        pytest.param(float(np.finfo(np.float32).max) * 2.0, id="overflows-to-infinity"),
    ],
)
def test_config_rejects_values_outside_normal_float32_execution_domain(
    field: str,
    value: float,
) -> None:
    assert np.isfinite(value) and value > 0.0
    kwargs: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=rf"{field} must be a finite positive normal float32"):
        SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ridge", "prediction_clip"])
def test_config_rounds_exact_normal_boundary_before_validation(field: str) -> None:
    minimum_normal = Fraction(1, 2**126)
    normal_midpoint = minimum_normal - Fraction(1, 2**150)
    below_midpoint = normal_midpoint - Fraction(1, 2**210)
    kwargs: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 1,
        field: below_midpoint,
    }

    with pytest.raises(ValueError, match=rf"{field} must be a finite positive normal float32"):
        SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]

    for accepted in (normal_midpoint, normal_midpoint + Fraction(1, 2**210)):
        kwargs[field] = accepted
        config = SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]
        assert getattr(config, field) == float(np.finfo(np.float32).tiny)


@pytest.mark.parametrize("field", ["ridge", "prediction_clip"])
def test_config_rounds_exact_overflow_midpoint_before_validation(field: str) -> None:
    float32_max = (2**24 - 1) * 2**104
    overflow_midpoint = Fraction(float32_max + 2**103)
    kwargs: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 1,
        field: overflow_midpoint - 1,
    }

    config = SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]
    assert getattr(config, field) == float(np.finfo(np.float32).max)

    for rejected in (overflow_midpoint, overflow_midpoint + 1):
        kwargs[field] = rejected
        with pytest.raises(
            ValueError,
            match=rf"{field} must be a finite positive normal float32",
        ):
            SparseFTLWorldModelConfig(**kwargs)  # type: ignore[arg-type]


def test_config_canonicalizes_numpy_scalars_for_strict_json_roundtrip() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=2,
        action_dim=1,
        ridge=np.float32(0.125),  # type: ignore[arg-type]
        prediction_clip=np.int64(4),  # type: ignore[arg-type]
    )

    assert type(config.ridge) is float
    assert type(config.prediction_clip) is float
    encoded = json.dumps(config.to_config(), allow_nan=False)
    restored = SparseFTLWorldModelConfig.from_config(json.loads(encoded))
    assert restored == config
    assert type(restored.ridge) is float
    assert type(restored.prediction_clip) is float


def test_config_preserves_existing_ordinary_serialization_payloads() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=2,
        action_dim=1,
        ridge=0.01,
        prediction_clip=10.1,
    )

    assert config.ridge == 0.01
    assert config.prediction_clip == 10.1
    assert config.to_config()["ridge"] == 0.01
    assert config.to_config()["prediction_clip"] == 10.1


def test_underflowing_ridge_is_rejected_before_update_poisoning() -> None:
    ridge = float(np.nextafter(np.float32(0.0), np.float32(1.0)))

    with pytest.raises(ValueError, match="ridge must be a finite positive normal float32"):
        config = SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=2,
            bins=3,
            ridge=ridge,
        )
        model = SparseFTLWorldModel(config)
        state = model.init(jr.key(100))
        result = model.update(
            state,
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert not bool(jnp.all(jnp.isfinite(result.state.weights)))


def test_overflowing_prediction_clip_is_rejected_before_prediction_poisoning() -> None:
    prediction_clip = float(np.finfo(np.float32).max) * 2.0

    with pytest.raises(
        ValueError,
        match="prediction_clip must be a finite positive normal float32",
    ):
        config = SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=2,
            bins=3,
            prediction_clip=prediction_clip,
        )
        model = SparseFTLWorldModel(config)
        state = model.init(jr.key(101)).replace(  # type: ignore[attr-defined]
            weights=jnp.full(
                (config.feature_dim, config.observation_dim),
                jnp.finfo(jnp.float32).max,
                dtype=jnp.float32,
            )
        )
        prediction = model.predict(
            state,
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
        )
        assert not bool(jnp.all(jnp.isfinite(prediction.delta)))


def test_soft_bin_features_are_bounded_local_and_boundary_safe() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=1,
        action_dim=1,
        projection_dim=4,
        bins=7,
    )
    model = SparseFTLWorldModel(config)
    state = model.init(jr.key(0))

    for value in (0.0, 1.0e6, -1.0e6):
        features = model.sparse_features(
            state,
            jnp.array([value], dtype=jnp.float32),
            jnp.array([value], dtype=jnp.float32),
        )
        paired_indices = features.indices.reshape((config.projection_dim, 2))
        paired_values = features.values.reshape((config.projection_dim, 2))
        expected_offsets = jnp.arange(config.projection_dim, dtype=jnp.int32) * config.bins

        chex.assert_shape(features.indices, (config.active_feature_count,))
        chex.assert_shape(features.dense, (config.feature_dim,))
        chex.assert_trees_all_equal(paired_indices[:, 1], paired_indices[:, 0] + 1)
        chex.assert_trees_all_equal(
            paired_indices[:, 0] // config.bins,
            jnp.arange(config.projection_dim, dtype=jnp.int32),
        )
        chex.assert_trees_all_close(
            paired_values.sum(axis=1),
            jnp.ones((config.projection_dim,), dtype=jnp.float32),
        )
        chex.assert_trees_all_close(
            features.dense,
            jnp.zeros((config.feature_dim,), dtype=jnp.float32)
            .at[features.indices]
            .add(features.values),
        )
        assert bool(jnp.all(features.indices >= expected_offsets.repeat(2)))
        assert bool(
            jnp.all(
                features.indices
                < (expected_offsets + jnp.asarray(config.bins, dtype=jnp.int32)).repeat(2)
            )
        )
        assert bool(jnp.all(features.values >= 0.0))
        assert bool(jnp.all(features.values <= 1.0))
        assert float(features.dense.sum()) == pytest.approx(config.projection_dim)


def test_update_reports_prediction_made_before_observing_target() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=6,
            bins=5,
            ridge=0.01,
        )
    )
    state = model.init(jr.key(1))
    observation = jnp.array([0.25], dtype=jnp.float32)
    action = jnp.array([-0.5], dtype=jnp.float32)
    next_observation = jnp.array([1.25], dtype=jnp.float32)
    before = model.predict(state, observation, action)

    result = model.update(state, observation, action, next_observation)
    after = model.predict(result.state, observation, action)

    chex.assert_trees_all_close(result.prediction.delta, before.delta)
    chex.assert_trees_all_close(result.error, jnp.array([1.0], dtype=jnp.float32))
    assert float(result.squared_error) == pytest.approx(1.0)
    assert float(after.delta[0]) > float(result.prediction.delta[0]) + 0.5
    assert int(result.state.step_count) == 1


def test_update_matches_active_block_ridge_solution_after_statistics_update() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=2,
        action_dim=1,
        projection_dim=4,
        bins=5,
        ridge=0.03,
        statistics_decay=0.97,
    )
    model = SparseFTLWorldModel(config)
    state = model.init(jr.key(2))
    observations = jnp.array(
        [[-0.7, 0.2], [0.1, -0.4], [0.8, 0.5], [-0.2, 0.9]],
        dtype=jnp.float32,
    )
    actions = jnp.array([[-0.3], [0.6], [-0.8], [0.2]], dtype=jnp.float32)
    next_observations = observations + jnp.array(
        [[0.2, -0.1], [-0.3, 0.4], [0.5, 0.1], [-0.2, -0.3]],
        dtype=jnp.float32,
    )
    state = run_sparse_ftl_world_model(
        model,
        state,
        observations,
        actions,
        next_observations,
    ).state

    observation = jnp.array([0.37, -0.61], dtype=jnp.float32)
    action = jnp.array([0.44], dtype=jnp.float32)
    next_observation = jnp.array([0.12, -0.19], dtype=jnp.float32)
    prior_prediction = model.predict(state, observation, action)
    indices = prior_prediction.features.indices
    values = prior_prediction.features.values
    target_delta = next_observation - observation
    expected_gram = config.statistics_decay * state.gram
    expected_cross = config.statistics_decay * state.cross
    expected_gram = expected_gram.at[indices[:, None], indices[None, :]].add(
        values[:, None] * values[None, :]
    )
    expected_cross = expected_cross.at[indices].add(values[:, None] * target_delta[None, :])
    active_gram = expected_gram[indices[:, None], indices[None, :]]
    inactive_contribution = (
        expected_gram[indices] @ state.weights - active_gram @ state.weights[indices]
    )
    expected_active_weights = jnp.linalg.solve(
        active_gram + config.ridge * jnp.eye(config.active_feature_count, dtype=jnp.float32),
        expected_cross[indices] - inactive_contribution,
    )

    result = model.update(state, observation, action, next_observation)

    chex.assert_trees_all_close(result.state.gram, expected_gram)
    chex.assert_trees_all_close(result.state.cross, expected_cross)
    chex.assert_trees_all_close(
        result.state.weights[indices],
        expected_active_weights,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_model_learns_deterministic_action_conditioned_dynamics() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=12,
            bins=7,
            ridge=0.01,
        )
    )
    state = model.init(jr.key(3))
    time = jnp.arange(200, dtype=jnp.float32)
    observations = jnp.sin(0.173 * time)[:, None]
    actions = jnp.cos(0.119 * time)[:, None]
    next_observations = observations + (0.3 * observations + 0.7 * actions + 0.2)

    result = run_sparse_ftl_world_model(
        model,
        state,
        observations,
        actions,
        next_observations,
    )
    test_observations = jnp.linspace(-1.0, 1.0, 201, dtype=jnp.float32)[:, None]
    test_actions = (jnp.cos(jnp.linspace(-3.0, 3.0, 201, dtype=jnp.float32)))[:, None]
    targets = 0.3 * test_observations + 0.7 * test_actions + 0.2
    predictions = jax.vmap(lambda obs, act: model.predict(result.state, obs, act).delta)(
        test_observations, test_actions
    )
    test_mse = jnp.mean((predictions - targets) ** 2)

    assert float(test_mse) < 5.0e-5
    assert float(jnp.mean(result.squared_errors[-32:])) < float(
        jnp.mean(result.squared_errors[:32])
    )


def test_lifetime_statistics_retain_unified_dynamics_across_a_b_a_visitation() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=12,
            bins=7,
            ridge=0.01,
            statistics_decay=1.0,
        )
    )
    state = model.init(jr.key(0))
    task_a = _shared_dynamics_stream(300, -1.0, -0.15, 0.2)
    task_b = _shared_dynamics_stream(300, 0.15, 1.0, 1.1)
    task_a_return = _shared_dynamics_stream(80, -1.0, -0.15, 2.3)

    state = run_sparse_ftl_world_model(model, state, *task_a).state
    task_a_mse_before_b = _shared_dynamics_mse(model, state, -1.0, -0.15)
    state = run_sparse_ftl_world_model(model, state, *task_b).state
    task_a_mse_after_b = _shared_dynamics_mse(model, state, -1.0, -0.15)
    task_b_mse = _shared_dynamics_mse(model, state, 0.15, 1.0)
    state = run_sparse_ftl_world_model(model, state, *task_a_return).state
    task_a_mse_after_return = _shared_dynamics_mse(model, state, -1.0, -0.15)

    assert task_a_mse_before_b < 2.0e-5
    assert task_a_mse_after_b < 1.0e-4
    assert task_b_mse < 1.0e-4
    assert task_a_mse_after_return < task_a_mse_after_b


def test_state_storage_is_fixed_over_stream_length() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=2,
        action_dim=1,
        projection_dim=5,
        bins=6,
    )
    model = SparseFTLWorldModel(config)
    initial_state = model.init(jr.key(4))
    observations = jnp.zeros((64, 2), dtype=jnp.float32)
    actions = jnp.ones((64, 1), dtype=jnp.float32)
    next_observations = observations + 0.25
    final_state = run_sparse_ftl_world_model(
        model,
        initial_state,
        observations,
        actions,
        next_observations,
    ).state

    assert jax.tree.map(lambda x: x.shape, initial_state) == jax.tree.map(
        lambda x: x.shape,
        final_state,
    )
    assert jax.tree.map(lambda x: x.dtype, initial_state) == jax.tree.map(
        lambda x: x.dtype,
        final_state,
    )
    assert _state_nbytes(initial_state) == model.state_nbytes
    assert _state_nbytes(final_state) == model.state_nbytes


def test_step_count_saturates_instead_of_overflowing() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=2,
            bins=3,
        )
    )
    state = model.init(jr.key(6)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        prediction_error_ema=jnp.array(0.25, dtype=jnp.float32),
    )

    result = model.update(
        state,
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == jnp.iinfo(jnp.int32).max
    assert float(result.state.prediction_error_ema) != pytest.approx(1.0)


def test_statistics_decay_trades_lifetime_retention_for_dynamics_adaptation() -> None:
    num_steps = 180
    time = jnp.arange(num_steps, dtype=jnp.float32)
    observations = jnp.sin(0.113 * time)[:, None]
    actions = jnp.cos(0.071 * time)[:, None]
    task_a_next = observations + (0.4 * observations + 0.2 * actions + 0.1)
    task_b_next = observations + (-0.5 * observations + 0.7 * actions - 0.2)
    test_observations = jnp.linspace(-1.0, 1.0, 201, dtype=jnp.float32)[:, None]
    test_actions = (0.8 * jnp.sin(jnp.linspace(-3.0, 3.0, 201, dtype=jnp.float32)))[:, None]
    task_a_targets = 0.4 * test_observations + 0.2 * test_actions + 0.1
    task_b_targets = -0.5 * test_observations + 0.7 * test_actions - 0.2

    def post_switch_mse(decay: float) -> tuple[float, float]:
        model = SparseFTLWorldModel(
            SparseFTLWorldModelConfig(
                observation_dim=1,
                action_dim=1,
                projection_dim=8,
                bins=7,
                ridge=0.01,
                statistics_decay=decay,
            )
        )
        state = model.init(jr.key(0))
        state = run_sparse_ftl_world_model(
            model,
            state,
            observations,
            actions,
            task_a_next,
        ).state
        state = run_sparse_ftl_world_model(
            model,
            state,
            observations,
            actions,
            task_b_next,
        ).state
        predictions = jax.vmap(lambda obs, act: model.predict(state, obs, act).delta)(
            test_observations, test_actions
        )
        return (
            float(jnp.mean((predictions - task_a_targets) ** 2)),
            float(jnp.mean((predictions - task_b_targets) ** 2)),
        )

    lifetime_task_a_mse, lifetime_task_b_mse = post_switch_mse(1.0)
    decayed_task_a_mse, decayed_task_b_mse = post_switch_mse(0.99)

    assert decayed_task_b_mse < 0.2 * lifetime_task_b_mse
    assert lifetime_task_a_mse < 0.5 * decayed_task_a_mse


def test_scan_is_jittable_and_configuration_roundtrips() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=2,
        action_dim=1,
        projection_dim=5,
        bins=4,
        ridge=0.02,
        statistics_decay=0.995,
        prediction_clip=3.0,
        error_decay=0.9,
    )
    restored_config = SparseFTLWorldModelConfig.from_config(config.to_config())
    model = SparseFTLWorldModel.from_config(SparseFTLWorldModel(restored_config).to_config())
    state = model.init(jr.key(5))
    observations = jnp.array(
        [[0.0, 0.0], [0.1, -0.2], [0.2, 0.3], [-0.4, 0.1]],
        dtype=jnp.float32,
    )
    actions = jnp.array([[0.0], [0.5], [-0.5], [0.25]], dtype=jnp.float32)
    next_observations = observations + jnp.array(
        [[0.1, -0.1], [0.2, 0.0], [-0.1, 0.3], [0.0, -0.2]],
        dtype=jnp.float32,
    )

    result = jax.jit(
        lambda current_state: run_sparse_ftl_world_model(
            model,
            current_state,
            observations,
            actions,
            next_observations,
        )
    )(state)

    assert model.config == config
    json.dumps(model.to_config())
    chex.assert_shape(result.predicted_next_observations, (4, 2))
    chex.assert_shape(result.errors, (4, 2))
    chex.assert_shape(result.squared_errors, (4,))
    chex.assert_tree_all_finite(result)
    assert int(result.state.step_count) == 4


def test_sparse_ftl_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        SparseFTLWorldModelConfig(observation_dim=True, action_dim=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action_dim"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projection_dim"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, projection_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bins"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, bins=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="observation_dim"):
        SparseFTLWorldModelConfig(observation_dim=2.5, action_dim=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action_dim"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projection_dim"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, projection_dim=32.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bins"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, bins=7.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="statistics_decay"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, statistics_decay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_decay"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, error_decay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_decay"):
        SparseFTLWorldModelConfig(observation_dim=1, action_dim=1, error_decay=1.0 - 1e-10)


def test_sparse_ftl_config_accepts_and_canonicalizes_numpy_integers() -> None:
    config = SparseFTLWorldModelConfig(
        observation_dim=np.int32(4),
        action_dim=np.int64(2),
        projection_dim=np.uint16(16),
        bins=np.int8(6),
    )
    assert type(config.observation_dim) is int
    assert type(config.action_dim) is int
    assert type(config.projection_dim) is int
    assert type(config.bins) is int
    assert config.observation_dim == 4
    assert config.action_dim == 2
    assert config.projection_dim == 16
    assert config.bins == 6


def test_sparse_ftl_config_rejects_derived_dimensions_outside_int32() -> None:
    int32_max = 2**31 - 1
    with pytest.raises(ValueError, match="derived input_dim"):
        SparseFTLWorldModelConfig(observation_dim=int32_max, action_dim=1)
    with pytest.raises(ValueError, match="derived feature_dim"):
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=2**30,
            bins=2,
        )
    with pytest.raises(ValueError, match="derived feature_dim"):
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=46_341,
            bins=46_341,
        )


def test_sparse_ftl_config_accepts_derived_int32_boundary() -> None:
    int32_max = 2**31 - 1
    widest_input = SparseFTLWorldModelConfig(
        observation_dim=int32_max - 1,
        action_dim=1,
        projection_dim=1,
        bins=int32_max,
    )
    widest_active_block = SparseFTLWorldModelConfig(
        observation_dim=1,
        action_dim=1,
        projection_dim=int32_max // 2,
        bins=2,
    )
    assert widest_input.input_dim == int32_max
    assert widest_input.feature_dim == int32_max
    assert widest_input.active_feature_count == 2
    assert widest_active_block.feature_dim == int32_max - 1
    assert widest_active_block.active_feature_count == int32_max - 1


def test_zero_error_decay_replaces_infinite_prior_ema_without_nan() -> None:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=2,
            bins=3,
            error_decay=0.0,
        )
    )
    state = model.init(jr.key(8)).replace(  # type: ignore[attr-defined]
        step_count=jnp.array(1, dtype=jnp.int32),
        prediction_error_ema=jnp.array(jnp.inf, dtype=jnp.float32),
    )

    result = model.update(
        state,
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )

    assert jnp.isfinite(result.squared_error)
    assert result.state.prediction_error_ema == result.squared_error
