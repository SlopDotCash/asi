"""End-to-end and hostile contracts for the bounded NaP comparator."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks.nap_ipmnist import (
    ARM_IDS,
    DEPENDENCY_COMMIT,
    FROZEN_SEEDS,
    PAPER_IDENTITY,
    PAPER_REVISION,
    PLASTICINE_COMMIT,
    NaPCatalogEntry,
    NaPResult,
    main,
    qualification_gates,
    run_comparator,
    validate_result,
)
from alberta_framework.benchmarks.plasticity_diagnostics import INPUT_DIM, PROFILES

pytestmark = pytest.mark.integration


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = 12
    values = np.arange(rows * INPUT_DIM, dtype=np.float32).reshape(rows, INPUT_DIM)
    images = (values % 256) / 255.0
    labels = np.asarray([index % 10 for index in range(rows)], dtype=np.int32)
    return images, labels


def _result(seed: int = FROZEN_SEEDS[0]) -> NaPResult:
    return run_comparator(*_fixture(), seed=seed)


def test_catalog_pins_final_paper_and_secondary_code_without_mislabeling() -> None:
    catalog = NaPCatalogEntry()
    catalog.validate()
    assert catalog.paper_revision == PAPER_REVISION
    assert catalog.final_paper_identity == PAPER_IDENTITY
    assert catalog.official_nap_code_available is False
    assert catalog.disclosed_baseline_revision is None
    assert catalog.secondary_implementation_commit == PLASTICINE_COMMIT
    assert catalog.secondary_implementation_official is False
    assert catalog.asi_dependency_commit == DEPENDENCY_COMMIT


def test_real_hidden_lane_runs_all_causal_arms_and_exact_off_reduction() -> None:
    result = _result()
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    current, off, normalization, projection, nap = result.arms
    assert current.task_accuracy == off.task_accuracy
    assert current.task_loss == off.task_loss
    assert current.dead_unit_fraction == off.dead_unit_fraction
    assert current.effective_rank == off.effective_rank
    assert current.final_state_sha256 == off.final_state_sha256
    assert dataclasses.replace(current.receipt, elapsed_ns=0) == dataclasses.replace(
        off.receipt, elapsed_ns=0
    )
    assert normalization.final_state_sha256 != off.final_state_sha256
    assert projection.final_state_sha256 != off.final_state_sha256
    assert nap.final_state_sha256 != off.final_state_sha256


def test_projection_arms_restore_each_hidden_weight_radius() -> None:
    result = _result(FROZEN_SEEDS[1])
    for arm in result.arms:
        if arm.projection_enabled:
            np.testing.assert_allclose(
                arm.final_hidden_norms,
                arm.initial_hidden_norms,
                rtol=2e-6,
                atol=2e-6,
            )
            assert arm.receipt.projection_events == 8
            assert arm.receipt.projected_tensor_queries == 16
            assert arm.receipt.projection_target_persistent_bytes == 8
        else:
            assert arm.receipt.projection_events == 0
            assert arm.receipt.projected_tensor_queries == 0
            assert arm.receipt.projection_target_persistent_bytes == 0


def test_exact_matched_and_logical_compute_receipts() -> None:
    result = _result(FROZEN_SEEDS[2])
    common = {
        (arm.receipt.data_steps, arm.receipt.observations, arm.receipt.data_bytes_read)
        for arm in result.arms
    }
    assert common == {(8, 8, 8 * (INPUT_DIM * 4 + 4))}
    assert {arm.receipt.training_model_queries for arm in result.arms} == {16}
    assert {arm.receipt.diagnostic_model_queries for arm in result.arms} == {8}
    assert {arm.receipt.model_queries for arm in result.arms} == {24}
    assert {arm.receipt.parameter_updates for arm in result.arms} == {8}
    assert len({arm.receipt.logical_forward_macs for arm in result.arms}) == 1
    assert len({arm.receipt.logical_gradient_macs for arm in result.arms}) == 1
    for arm in result.arms:
        assert arm.receipt.state_persistent_bytes > 0
        assert arm.receipt.logical_auxiliary_scalar_ops >= 0
        assert arm.receipt.elapsed_ns >= 0
        assert arm.receipt.timing_telemetry_only


def test_jit_and_eager_paths_match_except_timing_and_roundoff() -> None:
    compiled = _result(FROZEN_SEEDS[3])
    with jax.disable_jit():
        eager = _result(FROZEN_SEEDS[3])
    for left, right in zip(compiled.arms, eager.arms, strict=True):
        assert left.task_accuracy == right.task_accuracy
        np.testing.assert_allclose(left.task_loss, right.task_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(left.dead_unit_fraction, right.dead_unit_fraction)
        np.testing.assert_allclose(left.effective_rank, right.effective_rank, rtol=1e-5)
        np.testing.assert_allclose(left.final_hidden_norms, right.final_hidden_norms, rtol=2e-6)
        assert dataclasses.replace(left.receipt, elapsed_ns=0) == dataclasses.replace(
            right.receipt, elapsed_ns=0
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda result: dataclasses.replace(result, scientific_promotion_allowed=True),
            "promotion",
        ),
        (lambda result: dataclasses.replace(result, paper_parity_claimed=True), "promotion"),
        (
            lambda result: dataclasses.replace(result, task_ids_visible_to_learner=True),
            "allowed-information",
        ),
        (
            lambda result: dataclasses.replace(result, dependency_source_sha256="0" * 64),
            "dependency source",
        ),
        (
            lambda result: dataclasses.replace(
                result, nap_project_dependency_source_sha256="0" * 64
            ),
            "nap_project dependency source",
        ),
        (
            lambda result: dataclasses.replace(
                result,
                catalog=dataclasses.replace(
                    result.catalog, official_nap_code_available=True
                ),
            ),
            "official NaP code",
        ),
    ),
)
def test_validator_rejects_provenance_information_and_promotion_forgery(
    mutation: object, match: str
) -> None:
    assert callable(mutation)
    with pytest.raises(ValueError, match=match):
        validate_result(mutation(_result()))


def test_validator_rejects_hostile_receipt_and_arm_types() -> None:
    result = _result()
    arm = result.arms[0]
    with pytest.raises(ValueError, match="exact integer"):
        dataclasses.replace(arm.receipt, data_steps=True)
    with pytest.raises(ValueError, match="exact NaPResult"):
        validate_result(dataclasses.asdict(result))
    with pytest.raises(ValueError, match="exact bool"):
        dataclasses.replace(arm, normalization_enabled=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("images", "labels"),
    (
        (lambda value: value.astype(np.float64), lambda value: value),
        (lambda value: value, lambda value: value.astype(np.int64)),
        (lambda value: value[:, :-1], lambda value: value),
        (lambda value: value, lambda value: np.asarray([True] * len(value))),
    ),
)
def test_dataset_boundaries_fail_closed(images: object, labels: object) -> None:
    source_images, source_labels = _fixture()
    assert callable(images) and callable(labels)
    with pytest.raises(ValueError):
        run_comparator(
            images(source_images), labels(source_labels), seed=FROZEN_SEEDS[0]
        )


def test_schedule_is_frozen_and_seed_specific() -> None:
    first = _result(FROZEN_SEEDS[0])
    replay = _result(FROZEN_SEEDS[0])
    other = _result(FROZEN_SEEDS[1])
    assert first.dataset_sha256 == replay.dataset_sha256 == other.dataset_sha256
    assert first.schedule_sha256 == replay.schedule_sha256
    assert first.schedule_sha256 != other.schedule_sha256
    assert not first.labels_permuted
    assert not first.task_boundaries_visible_to_learner
    assert not first.task_ids_visible_to_learner


def test_result_binds_the_complete_immutable_profile() -> None:
    result = _result()
    assert result.profile == PROFILES[result.profile_id]
    with pytest.raises(ValueError, match="profile payload"):
        validate_result(
            dataclasses.replace(
                result,
                profile=dataclasses.replace(result.profile, learning_rate=0.004),
            )
        )


def test_validator_rejects_forged_runtime_identity() -> None:
    with pytest.raises(ValueError, match="runtime identity drift"):
        validate_result(dataclasses.replace(_result(), runtime_identity=("forged",) * 4))


def test_catalog_and_execution_cli_are_bounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(("--catalog",)) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["catalog"]["official_nap_code_available"] is False
    assert catalog["qualification_gates"]["execution_authorized"] is False
    images, labels = _fixture()
    dataset = tmp_path / "mnist.npz"
    np.savez(dataset, images=images, labels=labels)
    assert main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0]))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"].startswith("asi.nap_ipmnist")
    assert payload["scientific_promotion_allowed"] is False
    assert payload["paper_parity_claimed"] is False


def test_costly_paper_lanes_are_unconditionally_closed() -> None:
    gates = qualification_gates()
    assert gates["execution_authorized"] is False
    assert gates["paper_parity_allowed"] is False
    assert gates["official_nap_code_available"] is False
    random_label_cifar = gates["random_label_cifar"]
    sequential_ale = gates["sequential_ale"]
    assert isinstance(random_label_cifar, Mapping)
    assert isinstance(sequential_ale, Mapping)
    assert random_label_cifar["qualified"] is False
    assert sequential_ale["qualified"] is False


def _npy_header_bytes(shape: tuple[int, ...], dtype: object) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


def test_cli_rejects_oversize_npy_header_before_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    dataset = tmp_path / "oversize.npz"
    with zipfile.ZipFile(dataset, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images.npy", _npy_header_bytes((80_000, INPUT_DIM), np.float32))
        archive.writestr("labels.npy", _npy_header_bytes((80_000,), np.int32))

    def _forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.load must not run after an oversize npy header")

    monkeypatch.setattr(np, "load", _forbidden_load)
    with pytest.raises(ValueError, match="unbounded"):
        main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0])))


def test_cli_rejects_compressed_oversize_members_before_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "compressed-oversize.npz"
    images = np.zeros((80_000, INPUT_DIM), dtype=np.float32)
    labels = np.zeros((80_000,), dtype=np.int32)
    np.savez_compressed(dataset, images=images, labels=labels)
    del images, labels

    def _forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.load must not run after an oversize compressed NPZ")

    monkeypatch.setattr(np, "load", _forbidden_load)
    with pytest.raises(ValueError, match="unbounded"):
        main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0])))

