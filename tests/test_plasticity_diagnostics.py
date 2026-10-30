"""End-to-end and hostile contracts for loss-of-plasticity diagnostics."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks import plasticity_diagnostics
from alberta_framework.benchmarks.plasticity_diagnostics import (
    ARM_IDS,
    FROZEN_SEEDS,
    INPUT_DIM,
    OFFICIAL_CODE_COMMIT,
    PAPER_REVISION,
    PROFILES,
    SCHEMA,
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
    assert control.receipt.replacements_by_layer == (0, 0, 0)
    assert mechanism_off.receipt.replacements_by_layer == (0, 0, 0)
    assert cbp.receipt.replacements > 0
    assert all(count > 0 for count in cbp.receipt.replacements_by_layer)
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained
    assert not result.task_boundary_available_to_learner
    assert not result.task_id_available_to_learner


def test_bounded_lane_preserves_the_paper_three_hidden_layer_topology() -> None:
    profile = PROFILES["contract-smoke"]
    state = plasticity_diagnostics._init_diagnostic_state(
        jax.random.key(0), profile.hidden_width
    )
    logits, hidden1, hidden2, hidden3 = plasticity_diagnostics._forward_diagnostic(
        state, np.zeros((2, INPUT_DIM), dtype=np.float32)
    )
    assert SCHEMA == "asi.loss_of_plasticity_mnist_development.v2"
    assert state.w1.shape == (INPUT_DIM, profile.hidden_width)
    assert state.w2.shape == state.w3.shape == (profile.hidden_width, profile.hidden_width)
    assert state.w4.shape == (profile.hidden_width, 10)
    assert hidden1.shape == hidden2.shape == hidden3.shape == (2, profile.hidden_width)
    assert logits.shape == (2, 10)


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
        per_forward_macs = INPUT_DIM * 8 + 8 * 8 + 8 * 8 + 8 * 10
        assert arm.receipt.logical_forward_macs == 24 * per_forward_macs
        assert arm.receipt.logical_gradient_macs == 16 * per_forward_macs
        assert arm.receipt.persistent_bytes == 26_260
        assert len(arm.receipt.replacements_by_layer) == 3
        assert sum(arm.receipt.replacements_by_layer) == arm.receipt.replacements
        assert arm.receipt.timing_telemetry_only
    forged_receipt = dataclasses.replace(result.arms[0].receipt, model_queries=8)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=forged_receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="resource receipt"):
        validate_result(forged)
    with pytest.raises(ValueError, match="equal total"):
        dataclasses.replace(result.arms[2].receipt, replacements_by_layer=(0, 0, 0))
    # One layer claims more replacements than the run has steps: the total (11) still
    # sits inside the three-layer bound, so only the per-layer bound can reject it.
    over_layer = dataclasses.replace(
        result.arms[2].receipt, replacements=11, replacements_by_layer=(9, 1, 1)
    )
    with pytest.raises(ValueError, match="resource receipt"):
        validate_result(
            dataclasses.replace(
                result,
                arms=(*result.arms[:2], dataclasses.replace(result.arms[2], receipt=over_layer)),
            )
        )
    # A contract-smoke candidate whose replacements all land in one layer never
    # exercised the other two replacement paths.
    one_layer = dataclasses.replace(
        result.arms[2].receipt, replacements=1, replacements_by_layer=(1, 0, 0)
    )
    with pytest.raises(ValueError, match="every CBP replacement path"):
        validate_result(
            dataclasses.replace(
                result,
                arms=(*result.arms[:2], dataclasses.replace(result.arms[2], receipt=one_layer)),
            )
        )
    with pytest.raises(ValueError, match="exact DiagnosticResult"):
        validate_result(dataclasses.asdict(result))


def test_dead_unit_fraction_is_normalized_over_three_hidden_layers() -> None:
    profile = PROFILES["contract-smoke"]
    width = profile.hidden_width
    inputs = np.ones((2, INPUT_DIM), dtype=np.float32)
    state = plasticity_diagnostics._init_diagnostic_state(jax.random.key(0), width)
    all_dead = jax.tree.map(lambda leaf: leaf * 0, state)
    fraction, last_hidden = plasticity_diagnostics._dead_unit_fraction(all_dead, inputs, width)
    assert fraction == 1.0
    assert last_hidden.shape == (2, width)
    # Layer one alive through a positive bias; layers two and three dead.
    first_alive = all_dead._replace(b1=all_dead.b1 + 1.0)
    fraction, _ = plasticity_diagnostics._dead_unit_fraction(first_alive, inputs, width)
    assert fraction == 2.0 / 3.0


def test_effective_rank_survives_float32_spectral_overflow() -> None:
    """A large finite float32 spectrum must not collapse the rank to 1."""
    features = np.diag(np.array([2e38, 2e38], dtype=np.float32))
    observed = plasticity_diagnostics._effective_rank(features)
    assert abs(observed - 2.0) <= 1e-3


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


def test_cli_binds_preflight_to_materialized_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images, labels = _fixture()
    dataset = tmp_path / "mnist.npz"
    admitted = tmp_path / "admitted.npz"
    replacement = tmp_path / "replacement.npz"
    np.savez(dataset, images=images, labels=labels)
    np.savez(replacement, images=images.astype(np.float64), labels=labels)
    admitted_inode = dataset.stat().st_ino
    original_preflight = plasticity_diagnostics._preflight_dataset_npz
    original_load = np.load
    materialized_inodes: list[int] = []

    def _swap_path_after_preflight(source: IO[bytes]) -> None:
        original_preflight(source)
        dataset.rename(admitted)
        dataset.symlink_to(replacement.name)

    def _recording_load(source: Any, *args: Any, **kwargs: Any) -> Any:
        metadata = (
            os.stat(source)
            if isinstance(source, (str, os.PathLike))
            else os.fstat(source.fileno())
        )
        materialized_inodes.append(metadata.st_ino)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(
        plasticity_diagnostics, "_preflight_dataset_npz", _swap_path_after_preflight
    )
    monkeypatch.setattr(np, "load", _recording_load)
    with pytest.raises(ValueError, match="bounded regular file"):
        main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0])))
    assert materialized_inodes == [admitted_inode]


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


def test_module_dash_m_entry_emits_the_catalog() -> None:
    """`python -m` must run the same CLI, not exit 0 as a silent no-op."""

    import subprocess
    import sys

    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "alberta_framework.benchmarks.plasticity_diagnostics",
            "--catalog",
        ),
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == plasticity_diagnostics.SCHEMA
    assert payload["costly_lane_gates"]["execution_authorized"] is False


@pytest.mark.parametrize("without_posix_flags", [False, True])
def test_cli_accepts_bounded_hardlinked_dataset_on_portable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, without_posix_flags: bool
) -> None:
    images, labels = _fixture()
    original = tmp_path / "original.npz"
    dataset = tmp_path / "linked.npz"
    np.savez(original, images=images, labels=labels)
    os.link(original, dataset)
    observed = []

    def inspect_arrays(actual_images, actual_labels, **kwargs):
        np.testing.assert_array_equal(actual_images, images)
        np.testing.assert_array_equal(actual_labels, labels)
        observed.append(True)
        return None

    monkeypatch.setattr(plasticity_diagnostics, "run_diagnostic", inspect_arrays)
    monkeypatch.setattr(plasticity_diagnostics, "_json_result", lambda result: "{}")
    if without_posix_flags:
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
        monkeypatch.delattr(os, "O_CLOEXEC", raising=False)
    assert main(("--dataset", str(dataset), "--seed", str(FROZEN_SEEDS[0]))) == 0
    assert observed == [True]
