from __future__ import annotations

import dataclasses

import jax
import pytest

from alberta_framework.benchmarks.dreamer_continual_development import (
    ARM_IDS,
    DREAMERV3_CODE,
    FROZEN_SEEDS,
    run_development_lane,
    validate_result,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_end_to_end_lane_exercises_world_model_replay_imagination_and_control() -> None:
    result = run_development_lane(
        seed=FROZEN_SEEDS[0], steps_per_task=2, replay_capacity=4, imaginations_per_step=2
    )
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    assert DREAMERV3_CODE.endswith("e3f02248693a79dc8b0ebd62c93683888ddaccfe")
    imagined, off, oracle = result.arms
    assert imagined.receipt.environment_steps == 6
    assert imagined.receipt.world_model_updates == off.receipt.world_model_updates == 6
    assert imagined.receipt.replay_inserts == off.receipt.replay_inserts == 6
    assert imagined.receipt.imagination_proposals == imagined.receipt.replay_samples == 12
    assert off.receipt.imagination_proposals == oracle.receipt.imagination_proposals == 0
    assert oracle.candidate_eligible is False
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained and not result.dreamer_parity_claimed


def test_jit_and_eager_paths_match_metrics_and_counters_except_timing() -> None:
    compiled = run_development_lane(
        seed=FROZEN_SEEDS[1], steps_per_task=1, replay_capacity=2, imaginations_per_step=1
    )
    with jax.disable_jit():
        eager = run_development_lane(
            seed=FROZEN_SEEDS[1], steps_per_task=1, replay_capacity=2, imaginations_per_step=1
        )
    for left, right in zip(compiled.arms, eager.arms, strict=True):
        assert left.task_returns == right.task_returns
        assert left.final_action_values == right.final_action_values
        assert dataclasses.replace(left.receipt, elapsed_ns=0) == dataclasses.replace(
            right.receipt, elapsed_ns=0
        )


def test_mechanism_off_is_causal_and_has_no_imagined_updates() -> None:
    result = run_development_lane(
        seed=FROZEN_SEEDS[2], steps_per_task=3, replay_capacity=3, imaginations_per_step=2
    )
    imagined, off, _ = result.arms
    assert imagined.receipt.imagined_value_updates > 0
    assert off.receipt.imagined_value_updates == 0
    assert imagined.final_action_values != off.final_action_values


@pytest.mark.parametrize("seed", [True, -1, 0, FROZEN_SEEDS[-1] + 1])
def test_runner_rejects_hostile_and_unfrozen_seeds(seed: object) -> None:
    with pytest.raises(ValueError):
        run_development_lane(seed=seed, steps_per_task=1)


def test_validator_rejects_promotion_and_receipt_forgery() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=1)
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))
    forged_receipt = dataclasses.replace(result.arms[0].receipt, world_model_queries=1)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=forged_receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="query receipt"):
        validate_result(forged)
    with pytest.raises(ValueError, match="decoding is intentionally unavailable"):
        validate_result({"schema": result.schema})

    forged_persistent = dataclasses.replace(result.arms[0].receipt, persistent_bytes=1)
    forged = dataclasses.replace(
        result,
        arms=(
            dataclasses.replace(result.arms[0], receipt=forged_persistent),
            *result.arms[1:],
        ),
    )
    with pytest.raises(ValueError, match="persistent-byte"):
        validate_result(forged)

    forged_identity = dataclasses.replace(result.identity, lane_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        validate_result(dataclasses.replace(result, identity=forged_identity))
