from __future__ import annotations

import dataclasses
import json

import jax
import pytest

from alberta_framework.benchmarks.cora_development import (
    ARM_IDS,
    CORA_CATALOG,
    FROZEN_SEEDS,
    OFFICIAL_CODE,
    catalog_payload,
    main,
    run_cora_development,
    validate_result,
)

pytestmark = pytest.mark.integration


def test_end_to_end_slice_has_metrics_information_contract_and_exact_receipts() -> None:
    result = run_cora_development(seed=FROZEN_SEEDS[0], steps_per_task=2, replay_capacity=3)
    assert result.paper_revision.startswith("Powers et al.")
    assert result.official_code_revision == OFFICIAL_CODE
    assert len(result.runner_source_sha256) == 64
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    assert result.task_boundaries_available_to_runner
    assert not result.task_ids_available_to_candidate
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained and not result.cora_parity_claimed
    for arm in result.arms:
        assert len(arm.evaluation_matrix) == 7
        assert arm.receipt.training_environment_steps == 12
        assert arm.receipt.evaluation_environment_steps == 21
        assert arm.receipt.model_queries == 33
    replay, off, task_control, random = result.arms
    assert replay.receipt.agent_updates == off.receipt.agent_updates == 23
    assert replay.receipt.replay_samples == 11
    assert off.receipt.replay_samples == random.receipt.replay_samples == 0
    assert random.receipt.persistent_bytes == 8
    assert task_control.candidate_eligible is False


def test_candidate_evaluation_tie_break_does_not_receive_task_identity() -> None:
    result = run_cora_development(
        seed=FROZEN_SEEDS[0], steps_per_task=1, replay_capacity=2
    )

    for arm in result.arms[:2]:
        assert arm.evaluation_matrix[0] in ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0))


def test_jit_and_eager_update_paths_match_except_timing() -> None:
    compiled = run_cora_development(seed=FROZEN_SEEDS[1], steps_per_task=1)
    with jax.disable_jit():
        eager = run_cora_development(seed=FROZEN_SEEDS[1], steps_per_task=1)
    for left, right in zip(compiled.arms, eager.arms, strict=True):
        assert left.evaluation_matrix == right.evaluation_matrix
        assert left.training_return == right.training_return
        assert dataclasses.replace(left.receipt, elapsed_ns=0) == dataclasses.replace(
            right.receipt, elapsed_ns=0
        )


def test_mechanism_off_matches_update_budget_but_never_samples_replay() -> None:
    result = run_cora_development(seed=FROZEN_SEEDS[2], steps_per_task=3, replay_capacity=2)
    replay, off, _, _ = result.arms
    assert replay.receipt.agent_updates == off.receipt.agent_updates
    assert replay.receipt.replay_samples > 0
    assert off.receipt.replay_samples == 0


def test_validator_rejects_promotion_metric_and_resource_forgery() -> None:
    result = run_cora_development(seed=FROZEN_SEEDS[0], steps_per_task=1)
    with pytest.raises(ValueError, match="nonpromotion"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))
    with pytest.raises(ValueError, match="source identity"):
        validate_result(dataclasses.replace(result, runner_source_sha256="0" * 64))
    forged_metric = dataclasses.replace(result.arms[0], continual_evaluation=0.123)
    with pytest.raises(ValueError, match="metric recomputation"):
        validate_result(dataclasses.replace(result, arms=(forged_metric, *result.arms[1:])))
    forged_return = dataclasses.replace(result.arms[0], training_return=0.0)
    with pytest.raises(ValueError, match="deterministic replay"):
        validate_result(dataclasses.replace(result, arms=(forged_return, *result.arms[1:])))
    receipt = dataclasses.replace(result.arms[0].receipt, model_queries=1)
    forged_resource = dataclasses.replace(result.arms[0], receipt=receipt)
    with pytest.raises(ValueError, match="resource receipt"):
        validate_result(dataclasses.replace(result, arms=(forged_resource, *result.arms[1:])))
    with pytest.raises(ValueError, match="exact CORADevelopmentResult"):
        validate_result(dataclasses.asdict(result))


@pytest.mark.parametrize("seed", [True, -1, 0, FROZEN_SEEDS[-1] + 1])
def test_runner_rejects_hostile_or_unfrozen_seeds(seed: object) -> None:
    with pytest.raises(ValueError):
        run_cora_development(seed=seed, steps_per_task=1)


def test_catalog_is_pinned_blocked_and_cli_is_metadata_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert len(CORA_CATALOG) == 4
    assert all(not family.ready for family in CORA_CATALOG)
    assert OFFICIAL_CODE.endswith("f2754bb282757829765beb4703f24b87efa13ff9")
    assert main(["--catalog"]) == 0
    assert json.loads(capsys.readouterr().out) == catalog_payload()
