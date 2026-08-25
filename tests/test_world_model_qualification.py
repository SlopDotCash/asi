"""Bounded native qualification before external world-model adapters run."""

from __future__ import annotations

import dataclasses

import pytest

from alberta_framework.benchmarks.world_model_qualification import (
    WORLD_MODEL_SMOKE_SCHEMA,
    WorldModelSmokeResult,
    run_native_world_model_smoke,
)

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
