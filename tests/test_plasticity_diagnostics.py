"""End-to-end and hostile contracts for loss-of-plasticity diagnostics."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks.plasticity_diagnostics import (
    ARM_IDS,
    FROZEN_SEEDS,
    INPUT_DIM,
    OFFICIAL_CODE_COMMIT,
    PAPER_REVISION,
    PROFILES,
    costly_lane_gates,
    main,
    require_costly_lane,
    run_diagnostic,
    validate_result,
)

pytestmark = pytest.mark.integration


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = 12
    values = np.arange(rows * INPUT_DIM, dtype=np.float32).reshape(rows, INPUT_DIM)
    images = (values % 256) / 255.0
    labels = np.asarray([index % 10 for index in range(rows)], dtype=np.int32)
    return images, labels


def test_hidden_network_lane_runs_end_to_end_and_mechanism_off_is_exact() -> None:
    images, labels = _fixture()
    result = run_diagnostic(images, labels, seed=FROZEN_SEEDS[0])
    assert result.paper_revision == PAPER_REVISION
    assert result.official_code_commit == OFFICIAL_CODE_COMMIT
    assert result.task_protocol == "cumulative-input-permutation"
    assert result.labels_permuted is False
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    control, mechanism_off, cbp = result.arms
    assert control.task_accuracy == mechanism_off.task_accuracy
    assert control.task_loss == mechanism_off.task_loss
    assert control.final_state_sha256 == mechanism_off.final_state_sha256
    assert control.receipt.replacements == mechanism_off.receipt.replacements == 0
    assert cbp.receipt.replacements > 0
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained
    assert not result.task_boundary_available_to_learner
    assert not result.task_id_available_to_learner


def test_jit_and_eager_paths_match_except_timing() -> None:
    images, labels = _fixture()
    compiled = run_diagnostic(images, labels, seed=FROZEN_SEEDS[1])
    with jax.disable_jit():
        eager = run_diagnostic(images, labels, seed=FROZEN_SEEDS[1])
    for left, right in zip(compiled.arms, eager.arms, strict=True):
        assert left.task_accuracy == right.task_accuracy
        np.testing.assert_allclose(left.task_loss, right.task_loss, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(left.dead_unit_fraction, right.dead_unit_fraction)
        np.testing.assert_allclose(left.effective_rank, right.effective_rank, rtol=1e-6)
        assert left.receipt.replacements == right.receipt.replacements
        assert dataclasses.replace(left.receipt, elapsed_ns=0) == dataclasses.replace(
            right.receipt, elapsed_ns=0
        )


def test_exact_resource_receipts_and_validator_reject_forgery() -> None:
    result = run_diagnostic(*_fixture(), seed=FROZEN_SEEDS[2])
    for arm in result.arms:
        assert arm.receipt.data_steps == 8
        assert arm.receipt.data_bytes_read == 8 * (INPUT_DIM * 4 + 4)
        assert arm.receipt.training_model_queries == 16
        assert arm.receipt.diagnostic_model_queries == 8
        assert arm.receipt.model_queries == 24
        assert arm.receipt.parameter_updates == 8
        assert arm.receipt.logical_forward_macs == 24 * (INPUT_DIM * 8 + 8 * 8 + 8 * 10)
        assert arm.receipt.logical_gradient_macs == 16 * (INPUT_DIM * 8 + 8 * 8 + 8 * 10)
        assert arm.receipt.persistent_bytes == 25_904
        assert arm.receipt.timing_telemetry_only
    forged_receipt = dataclasses.replace(result.arms[0].receipt, model_queries=8)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=forged_receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="resource receipt"):
        validate_result(forged)
    with pytest.raises(ValueError, match="exact DiagnosticResult"):
        validate_result(dataclasses.asdict(result))


def test_profiles_are_immutable_and_result_binds_the_complete_profile() -> None:
    assert isinstance(PROFILES, Mapping)
    with pytest.raises(TypeError):
        PROFILES["contract-smoke"] = dataclasses.replace(  # type: ignore[index]
            PROFILES["contract-smoke"], n_tasks=3
        )
    result = run_diagnostic(*_fixture(), seed=FROZEN_SEEDS[0])
    assert result.profile == PROFILES[result.profile_id]
    with pytest.raises(ValueError, match="profile payload"):
        validate_result(
            dataclasses.replace(
                result,
                profile=dataclasses.replace(result.profile, maturity_threshold=3),
            )
        )


def test_validator_rejects_forged_runtime_identity() -> None:
    result = run_diagnostic(*_fixture(), seed=FROZEN_SEEDS[0])
    with pytest.raises(ValueError, match="runtime identity drift"):
        validate_result(dataclasses.replace(result, runtime_identity=("forged",) * 4))


def test_random_label_and_privileged_task_claims_fail_closed() -> None:
    result = run_diagnostic(*_fixture(), seed=FROZEN_SEEDS[3])
    with pytest.raises(ValueError, match="random-label"):
        validate_result(dataclasses.replace(result, labels_permuted=True))
    with pytest.raises(ValueError, match="information"):
        validate_result(dataclasses.replace(result, task_id_available_to_learner=True))
    with pytest.raises(ValueError, match="information"):
        validate_result(
            dataclasses.replace(result, task_id_available_to_learner=0)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="nonpromotion"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))


@pytest.mark.parametrize(
    ("images_mutation", "labels_mutation"),
    [
        (lambda value: value.astype(np.float64), lambda value: value),
        (lambda value: value, lambda value: value.astype(np.int64)),
        (lambda value: value[:, :-1], lambda value: value),
        (lambda value: value, lambda value: np.asarray([True] * len(value))),
    ],
)
def test_hostile_dataset_types_and_shapes_are_rejected(
    images_mutation: object, labels_mutation: object
) -> None:
    images, labels = _fixture()
    assert callable(images_mutation) and callable(labels_mutation)
    with pytest.raises(ValueError):
        run_diagnostic(
            images_mutation(images), labels_mutation(labels), seed=FROZEN_SEEDS[0]
        )


def test_costly_imagenet_and_rl_lanes_are_unconditionally_gated() -> None:
    gates = costly_lane_gates()
    assert gates["execution_authorized"] is False
    assert gates["imagenet"]["qualified"] is False
    assert "100M" in gates["reinforcement_learning"]["minimum_known_cost"]
    for lane in ("imagenet", "reinforcement_learning"):
        with pytest.raises(RuntimeError, match="not qualified or authorized"):
            require_costly_lane(lane)


def test_catalog_cli_does_not_load_or_execute_data(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("--catalog",)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"].startswith("asi.loss_of_plasticity")
    assert payload["costly_lane_gates"]["execution_authorized"] is False


def test_cli_runs_only_caller_supplied_bounded_npz(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images, labels = _fixture()
    dataset = tmp_path / "mnist.npz"
    np.savez(dataset, images=images, labels=labels)
    assert main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0]))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_protocol"] == "cumulative-input-permutation"
    assert payload["labels_permuted"] is False
    assert payload["scientific_promotion_allowed"] is False
