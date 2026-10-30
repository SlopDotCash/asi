"""Bounded native qualification before external world-model adapters run."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.benchmarks.world_model_qualification import (
    WORLD_MODEL_SMOKE_SCHEMA,
    WorldModelSmokeResult,
    _trace,
    run_native_world_model_smoke,
)
from alberta_framework.core.latent_world_model import LatentWorldModel, LatentWorldModelConfig

pytestmark = pytest.mark.unit


def test_native_world_model_smoke_is_deterministic_matched_and_nonpromoting() -> None:
    first = run_native_world_model_smoke(seed=17, steps=4)
    second = run_native_world_model_smoke(seed=17, steps=4)

    assert first == second
    assert first.schema == WORLD_MODEL_SMOKE_SCHEMA
    assert first.actions == (0, 1, 0, 1)
    assert first.development_only is True
    assert first.scientific_promotion_allowed is False
    assert {arm.metric_space for arm in first.arms} == {
        "observation_mse",
        "latent_prediction_mse",
        "observation_delta_mse",
    }
    for arm in first.arms:
        assert arm.environment_steps == arm.model_updates == arm.model_queries == 4
        assert arm.persistent_bytes > 0
        assert len(arm.prequential_losses) == 4


def test_native_world_model_smoke_retains_explicit_mechanism_off_arm() -> None:
    result = run_native_world_model_smoke(seed=3, steps=4)
    enabled = result.arms[1]
    disabled = result.arms[2]

    assert enabled.arm_id == "latent_action_interactions"
    assert disabled.arm_id == "latent_no_interactions"
    assert enabled.persistent_bytes > disabled.persistent_bytes
    assert enabled.prequential_losses != disabled.prequential_losses


def test_native_world_model_smoke_latent_arms_record_latent_prediction_mse() -> None:
    result = run_native_world_model_smoke(seed=0, steps=4)
    actions, transitions = _trace(4)
    latent_config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=2,
        hidden_sizes=(),
        sparsity=0.0,
        use_layer_norm=False,
        include_action_interactions=True,
    )
    latent_key = jr.split(jr.key(0), 3)[1]
    for arm, config in (
        (result.arms[1], latent_config),
        (result.arms[2], dataclasses.replace(latent_config, include_action_interactions=False)),
    ):
        assert arm.metric_space == "latent_prediction_mse"
        model = LatentWorldModel(config)
        state = model.init(latent_key)
        expected = []
        for action, (observation, next_observation) in zip(actions, transitions, strict=True):
            action_array = jnp.asarray(action, dtype=jnp.int32)
            predicted = model.predict(state, observation, action_array).next_latent
            target = model.encode(state, next_observation)
            expected.append(float(jnp.mean((predicted - target) ** 2)))
            state = model.update(
                state,
                observation,
                action_array,
                jnp.asarray(float(next_observation[0]), dtype=jnp.float32),
                jnp.asarray(0.99, dtype=jnp.float32),
                next_observation,
            ).state

        assert arm.prequential_losses == pytest.approx(expected, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize(("seed", "steps"), [(True, 4), (0, True), (-1, 4), (0, 65)])
def test_native_world_model_smoke_rejects_hostile_or_unbounded_inputs(
    seed: object, steps: object
) -> None:
    with pytest.raises(ValueError, match="exact integer|must lie"):
        run_native_world_model_smoke(seed=seed, steps=steps)


def test_world_model_smoke_result_rejects_promotion_or_arm_aliases() -> None:
    result = run_native_world_model_smoke(seed=0, steps=2)
    with pytest.raises(ValueError, match="may not allow"):
        dataclasses.replace(result, scientific_promotion_allowed=True)
    with pytest.raises(ValueError, match="exact WorldModelSmokeArm"):
        WorldModelSmokeResult(
            schema=result.schema,
            seed=result.seed,
            steps=result.steps,
            actions=result.actions,
            arms=tuple(object() for _ in range(4)),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("prequential_losses", (-1.0, -1.0), "finite nonnegative"),
        ("persistent_bytes", 0, "must lie"),
    ],
)
def test_world_model_smoke_result_revalidates_nested_arms(
    field: str, value: object, match: str
) -> None:
    result = run_native_world_model_smoke(seed=0, steps=2)
    object.__setattr__(result.arms[0], field, value)

    with pytest.raises(ValueError, match=match):
        dataclasses.replace(result)


def test_world_model_smoke_result_binds_arm_metrics_and_step_count() -> None:
    result = run_native_world_model_smoke(seed=0, steps=2)
    wrong_metric = dataclasses.replace(result.arms[0], metric_space="latent_prediction_mse")
    with pytest.raises(ValueError, match="metric spaces"):
        dataclasses.replace(result, arms=(wrong_metric, *result.arms[1:]))

    with pytest.raises(ValueError, match="step count"):
        dataclasses.replace(result, steps=1, actions=(0,))
