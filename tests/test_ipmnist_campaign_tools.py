"""Contracts for maintained campaign tools that supersede output-local scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest

import alberta_framework.benchmarks.ipmnist_campaign_tools as campaign_tools_module
from alberta_framework.benchmarks.ipmnist_campaign_tools import (
    CONFIRM_ALIGNMENT_ATOL,
    across_seed_spread,
    build_ceiling_summary,
    build_frontier,
    validate_confirm_alignment,
)
from alberta_framework.benchmarks.ipmnist_ceiling import (
    _atomic_publish,
    _publish_run,
    run_arm_per_step,
    run_batch_reference,
)
from alberta_framework.benchmarks.rule_discovery_summary import (
    CHAMPION,
    SCREEN_ARMS,
    build_legacy_rule_discovery_summary,
    build_rule_discovery_summary,
)

pytestmark = pytest.mark.unit
_REPO_ROOT = Path(__file__).resolve().parents[1]


class _HostileSeed(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("seed conversion hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("seed representation hook executed")


class _StringSubclass(str):
    pass


def _shard(path: Path, *, seed: int, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": seed, "per_task_accuracy": [accuracy, accuracy]}),
        encoding="utf-8",
    )


def _ceiling_run(
    root: Path,
    *,
    prefix: str,
    seed: int,
    mean_accuracy: float,
    tasks: int = 1,
) -> None:
    tag = prefix
    family, spec_name = prefix.split("_", 1)
    permutation_mode = {"stationary": "identity", "carried": "same", "full": "protocol"}[
        family
    ]
    correct = int(round(mean_accuracy * 5000))
    row = np.concatenate(
        (np.ones(correct, dtype=np.uint8), np.zeros(5000 - correct, dtype=np.uint8))
    )
    per_step = np.tile(row, (tasks, 1))
    per_task = per_step.astype(np.float64).mean(axis=1).tolist()
    (root / f"{prefix}_seed{seed}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "tag": tag,
                "spec_name": spec_name,
                "perm_mode": permutation_mode,
                "n_tasks": tasks,
                "task_length": 5000,
                "mean_accuracy": float(np.mean(per_task)),
                "per_task_accuracy": per_task,
            }
        ),
        encoding="utf-8",
    )
    np.save(
        root / f"{tag}_seed{seed}_per_step.npy",
        per_step,
        allow_pickle=False,
    )


@pytest.mark.parametrize(
    "seed",
    [True, False, _HostileSeed(1), -1, 2**32],
    ids=("true", "false", "hostile-int", "negative", "above-uint32"),
)
def test_ceiling_runner_rejects_noncanonical_seed_before_work(seed: object) -> None:
    with pytest.raises(ValueError, match="built-in integer.*uint32"):
        run_arm_per_step("unused", seed, 1, "identity")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seed",
    [True, False, _HostileSeed(1), -1, 2**32],
    ids=("true", "false", "hostile-int", "negative", "above-uint32"),
)
def test_batch_reference_rejects_noncanonical_seed_before_publication(
    tmp_path: Path, seed: object
) -> None:
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="built-in integer.*uint32"):
        run_batch_reference(output_dir, seed=seed)  # type: ignore[arg-type]
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("epochs", "batch_size"),
    [
        (True, 128),
        (0, 128),
        (31, 128),
        (1, True),
        (1, 31),
        (1, 60_001),
        (30, 32),
    ],
)
def test_batch_reference_preflights_exact_bounded_work_before_io(
    tmp_path: Path, epochs: object, batch_size: object
) -> None:
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        run_batch_reference(
            output_dir,
            seed=0,
            epochs=epochs,  # type: ignore[arg-type]
            batch_size=batch_size,  # type: ignore[arg-type]
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("base", "arms", "threshold", "created_unix"),
    [
        ("../base", ("candidate",), 0.0, 0.0),
        ("base", (), 0.0, 0.0),
        ("base", ("candidate", "candidate"), 0.0, 0.0),
        ("base", ("base",), 0.0, 0.0),
        ("base", (_StringSubclass("candidate"),), 0.0, 0.0),
        ("base", ("../candidate",), 0.0, 0.0),
        ("base", ("candidate",), True, 0.0),
        ("base", ("candidate",), float("nan"), 0.0),
        ("base", ("candidate",), 0.0, -1.0),
        ("base", ("candidate",), 0.0, float("inf")),
    ],
)
def test_frontier_rejects_hostile_identifiers_and_nonfinite_controls(
    tmp_path: Path,
    base: object,
    arms: object,
    threshold: object,
    created_unix: object,
) -> None:
    with pytest.raises(ValueError):
        build_frontier(
            tmp_path / "screen",
            tmp_path / "confirm",
            base=base,  # type: ignore[arg-type]
            arms=arms,  # type: ignore[arg-type]
            threshold=threshold,  # type: ignore[arg-type]
            created_unix=created_unix,  # type: ignore[arg-type]
        )


def test_frontier_requires_distinct_resolved_directories(tmp_path: Path) -> None:
    directory = tmp_path / "campaign"
    with pytest.raises(ValueError, match="distinct paths"):
        build_frontier(directory, directory, base="base", arms=("candidate",))


def test_frontier_seed_filename_must_bind_payload_seed(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    _shard(screen / "base_seed0.json", seed=1, accuracy=0.8)
    with pytest.raises(ValueError, match="filename disagrees"):
        build_frontier(screen, confirm, base="base", arms=("candidate",))


def test_campaign_cli_uses_strict_json_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        campaign_tools_module,
        "build_frontier",
        lambda *args, **kwargs: {"nonfinite": float("nan")},
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        campaign_tools_module.main(
            [
                "frontier",
                "--screen-dir",
                str(tmp_path / "screen"),
                "--confirm-dir",
                str(tmp_path / "confirm"),
            ]
        )


def test_across_seed_spread_uses_sample_estimator() -> None:
    values = [0.7814, 0.7842, 0.7870]
    assert across_seed_spread(values) == pytest.approx(np.std(values, ddof=1))
    assert across_seed_spread(values) > float(np.std(values))
    assert across_seed_spread([]) == 0.0
    assert across_seed_spread([0.5]) == 0.0


def test_frontier_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    screen.mkdir(parents=True, exist_ok=True)
    confirm.mkdir(parents=True, exist_ok=True)
    (screen / "base_seed0.json").write_text(
        '{"seed":0,"per_task_accuracy":[0.1,0.1],"per_task_accuracy":[0.9,0.9]}',
        encoding="utf-8",
    )
    _shard(screen / "candidate_seed0.json", seed=0, accuracy=0.81)
    _shard(confirm / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(confirm / "candidate_seed0.json", seed=0, accuracy=0.82)

    with pytest.raises(ValueError, match="duplicate"):
        build_frontier(
            screen,
            confirm,
            base="base",
            arms=["candidate"],
            created_unix=0.0,
        )


def test_frontier_requires_and_uses_exact_paired_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    for seed in (0, 1):
        _shard(screen / f"base_seed{seed}.json", seed=seed, accuracy=0.80)
        _shard(screen / f"candidate_seed{seed}.json", seed=seed, accuracy=0.81)
        _shard(confirm / f"base_seed{seed}.json", seed=seed, accuracy=0.80)
        _shard(confirm / f"candidate_seed{seed}.json", seed=seed, accuracy=0.82)
    frontier = build_frontier(
        screen,
        confirm,
        base="base",
        arms=["candidate"],
        created_unix=0.0,
    )
    row = frontier["results"][0]

    assert row["n_confirm_seeds"] == 2
    assert row["confirm_mean"] == pytest.approx(0.82)
    assert row["confirm_paired_delta_vs_base"] == pytest.approx(0.02)

    assert frontier["schema"] == "asi.ipmnist.frontier.v2"
    assert len(frontier["provenance"]["inputs"]) == 8
    assert len(frontier["provenance"]["sources"]["campaign_tools"]["sha256"]) == 64
    assert "ipmnist_provenance" in frontier["provenance"]["sources"]
    assert {
        item["path"] for item in frontier["provenance"]["environment_specifications"]
    } == {"pyproject.toml", "uv.lock"}


def test_frontier_rejects_mismatched_confirm_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    _shard(screen / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(screen / "candidate_seed0.json", seed=0, accuracy=0.81)
    _shard(confirm / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(confirm / "candidate_seed7.json", seed=7, accuracy=0.99)

    with pytest.raises(ValueError, match="confirm seed sets differ"):
        build_frontier(
            screen,
            confirm,
            base="base",
            arms=["candidate"],
            created_unix=0.0,
        )


def test_frontier_rejects_empty_or_no_overlap_screen_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    _shard(screen / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(screen / "candidate_seed7.json", seed=7, accuracy=0.81)

    with pytest.raises(ValueError, match="screen seed sets differ"):
        build_frontier(screen, confirm, base="base", arms=["candidate"])

    with pytest.raises(ValueError, match="has no seeds"):
        build_frontier(tmp_path / "empty", confirm, base="base", arms=["candidate"])


def test_current_frontier_legacy_numeric_payload_reconstructs() -> None:
    campaign = _REPO_ROOT / "outputs" / "ipmnist_screening"
    expected = json.loads((campaign / "frontier_results.json").read_text())
    actual = build_frontier(
        campaign / "shards",
        campaign / "confirm_full",
        created_unix=float(expected["created_unix"]),
    )
    actual.pop("schema")
    actual.pop("provenance")

    assert actual == expected


def test_ceiling_summary_records_sample_spread_from_explicit_paths(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling"
    confirm = tmp_path / "confirm"
    ceiling.mkdir()
    confirm.mkdir()
    for seed, accuracy in enumerate((0.78, 0.82)):
        _ceiling_run(
            ceiling,
            prefix="stationary_sigma0_ndecay099",
            seed=seed,
            mean_accuracy=accuracy,
        )

    summary = build_ceiling_summary(ceiling, confirm)
    result = summary["stationary_sigma0_ndecay099"]

    assert result["sample_standard_deviation"] == pytest.approx(
        np.std([0.78, 0.82], ddof=1)
    )
    assert summary["schema"] == "asi.ipmnist.ceiling_summary.v2"
    assert len(summary["provenance"]["inputs"]) == 4
    assert summary["provenance"]["runtime"]["dependencies"]["numpy"] is not None


def test_current_stored_ceiling_reconstructs_with_rounded_confirmation() -> None:
    campaign = _REPO_ROOT / "outputs" / "ipmnist_screening"

    summary = build_ceiling_summary(campaign / "ceiling", campaign / "confirm_full")
    alignment = summary["error_budget"]["confirm_alignment"]

    assert alignment["relative_tolerance"] == 0.0
    assert alignment["absolute_tolerance"] == CONFIRM_ALIGNMENT_ATOL
    assert alignment["max_abs_delta"] == pytest.approx(6.000000007944095e-08)


def test_ceiling_confirm_alignment_rejects_delta_above_frozen_tolerance(
    tmp_path: Path,
) -> None:
    confirm = tmp_path / "confirm"
    _shard(confirm / "sigma0_ndecay099_seed0.json", seed=0, accuracy=0.8)
    run = {"seed": 0, "per_task_accuracy": [0.8, 0.8 + 2 * CONFIRM_ALIGNMENT_ATOL]}

    with pytest.raises(ValueError, match="exceeds atol"):
        validate_confirm_alignment([run], confirm)


def test_ceiling_publication_refuses_to_replace_existing_bytes(tmp_path: Path) -> None:
    per_step = np.ones((1, 5000), dtype=np.uint8)
    per_task = np.asarray([1.0], dtype=np.float64)
    artifact_path = _publish_run(
        tmp_path,
        tag="stationary_sigma0_ndecay099",
        spec_name="sigma0_ndecay099",
        seed=0,
        permutation_mode="identity",
        n_tasks=1,
        per_step=per_step,
        per_task=per_task,
        wall_seconds=1.0,
        provenance={"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
    )
    before = artifact_path.read_bytes()
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert np.array_equal(archive["per_step"], per_step)
    assert metadata["schema"] == "asi.ipmnist_ceiling.run.v2"
    assert metadata["provenance"]["schema"] == (
        "asi.ipmnist.ceiling_run_provenance.v1"
    )
    reconstructed = build_ceiling_summary(tmp_path, tmp_path / "confirm")
    assert reconstructed["stationary_sigma0_ndecay099"]["avg_online_mean"] == 1.0

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publish_run(
            tmp_path,
            tag="stationary_sigma0_ndecay099",
            spec_name="sigma0_ndecay099",
            seed=0,
            permutation_mode="identity",
            n_tasks=1,
            per_step=per_step,
            per_task=per_task,
            wall_seconds=1.0,
            provenance={"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
        )

    assert artifact_path.read_bytes() == before


@pytest.mark.parametrize(
    "override",
    [
        {"tag": "../escape"},
        {"tag": _StringSubclass("stationary_sigma0_ndecay099")},
        {"spec_name": "other"},
        {"permutation_mode": "protocol"},
        {"n_tasks": 2},
        {"wall_seconds": float("nan")},
        {"provenance": {"schema": "wrong"}},
    ],
)
def test_ceiling_publisher_rejects_hostile_identity_and_scalar_contracts(
    tmp_path: Path, override: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        "tag": "stationary_sigma0_ndecay099",
        "spec_name": "sigma0_ndecay099",
        "seed": 0,
        "permutation_mode": "identity",
        "n_tasks": 1,
        "per_step": np.ones((1, 5000), dtype=np.uint8),
        "per_task": np.ones(1, dtype=np.float64),
        "wall_seconds": 1.0,
        "provenance": {"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        _publish_run(tmp_path / "out", **arguments)  # type: ignore[arg-type]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("per_step", "per_task"),
    [
        (np.ones((1, 5000), dtype=np.float32), np.ones(1, dtype=np.float64)),
        (np.full((1, 5000), 2, dtype=np.uint8), np.ones(1, dtype=np.float64)),
        (np.ones((1, 4999), dtype=np.uint8), np.ones(1, dtype=np.float64)),
        (np.ones((1, 5000), dtype=np.uint8), np.asarray([np.nan])),
        (np.ones((1, 5000), dtype=np.uint8), np.asarray([0.5])),
    ],
)
def test_ceiling_publisher_rejects_invalid_accuracy_arrays(
    tmp_path: Path, per_step: np.ndarray, per_task: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _publish_run(
            tmp_path / "out",
            tag="stationary_sigma0_ndecay099",
            spec_name="sigma0_ndecay099",
            seed=0,
            permutation_mode="identity",
            n_tasks=1,
            per_step=per_step,
            per_task=per_task,
            wall_seconds=1.0,
            provenance={"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
        )
    assert not (tmp_path / "out").exists()


def _write_legacy_ceiling_run(
    root: Path,
    *,
    filename: str,
    seed: int = 0,
    task_length: int = 5000,
    n_tasks: int = 1,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tag = "stationary_sigma0_ndecay099"
    payload = {
        "tag": tag,
        "spec_name": "sigma0_ndecay099",
        "seed": seed,
        "perm_mode": "identity",
        "n_tasks": n_tasks,
        "task_length": task_length,
        "per_task_accuracy": [1.0] * n_tasks,
        "mean_accuracy": 1.0,
    }
    (root / filename).write_text(json.dumps(payload), encoding="utf-8")
    np.save(
        root / f"{tag}_seed{seed}_per_step.npy",
        np.ones((n_tasks, task_length), dtype=np.uint8),
        allow_pickle=False,
    )


def test_ceiling_analyzer_rejects_short_task_dimension(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling"
    _write_legacy_ceiling_run(
        ceiling,
        filename="stationary_sigma0_ndecay099_seed0.json",
        task_length=1,
    )
    with pytest.raises(ValueError, match="task_length must equal 5000"):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


def test_ceiling_analyzer_rejects_duplicate_legacy_seed(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling"
    _write_legacy_ceiling_run(
        ceiling, filename="stationary_sigma0_ndecay099_seed0.json"
    )
    _write_legacy_ceiling_run(
        ceiling, filename="stationary_sigma0_ndecay099_seed0_duplicate.json"
    )
    with pytest.raises(ValueError, match="duplicate ceiling seed"):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


@pytest.mark.parametrize(
    "override",
    [
        {"development_only": False},
        {"scientific_promotion_allowed": True},
        {"evidence_class": "scientific_evidence"},
        {"wall_clock_seconds": float("nan")},
        {"provenance": {"schema": "wrong"}},
    ],
)
def test_ceiling_analyzer_rejects_tampered_maintained_metadata(
    tmp_path: Path, override: dict[str, object]
) -> None:
    ceiling = tmp_path / "ceiling"
    ceiling.mkdir()
    payload: dict[str, object] = {
        "schema": "asi.ipmnist_ceiling.run.v2",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "tag": "stationary_sigma0_ndecay099",
        "spec_name": "sigma0_ndecay099",
        "seed": 0,
        "perm_mode": "identity",
        "n_tasks": 1,
        "task_length": 5000,
        "per_task_accuracy": [1.0],
        "mean_accuracy": 1.0,
        "wall_clock_seconds": 1.0,
        "provenance": {"schema": "asi.ipmnist.ceiling_run_provenance.v1"},
    }
    payload.update(override)
    np.savez_compressed(
        ceiling / "stationary_sigma0_ndecay099_seed0.npz",
        metadata=np.asarray(json.dumps(payload)),
        per_step=np.ones((1, 5000), dtype=np.uint8),
    )
    with pytest.raises(ValueError):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


def test_confirm_alignment_requires_matching_canonical_reference_seed(
    tmp_path: Path,
) -> None:
    confirm = tmp_path / "confirm"
    _shard(confirm / "sigma0_ndecay099_seed0.json", seed=1, accuracy=0.8)
    with pytest.raises(ValueError, match="does not match ceiling seed"):
        validate_confirm_alignment(
            [{"seed": 0, "per_task_accuracy": [0.8, 0.8]}], confirm
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 1, "test_accuracy_best": 0.9},
        {"seed": -1, "test_accuracy_best": 0.9},
        {"seed": 0, "test_accuracy_best": float("nan")},
        {"seed": 0, "test_accuracy_best": 1.1},
    ],
)
def test_ceiling_analyzer_rejects_invalid_batch_summary(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    ceiling = tmp_path / "ceiling"
    ceiling.mkdir()
    (ceiling / "batch_reference_seed0.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        build_ceiling_summary(ceiling, tmp_path / "confirm")


def test_atomic_publication_cleans_up_writer_failure(tmp_path: Path) -> None:
    destination = tmp_path / "result.bin"

    def fail_after_write(stream: BinaryIO) -> None:
        stream.write(b"partial")
        raise RuntimeError("injected writer fault")

    with pytest.raises(RuntimeError, match="injected writer fault"):
        _atomic_publish(destination, fail_after_write)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_cleans_up_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.bin"

    def fail_link(source: object, target: object) -> None:
        del source, target
        raise OSError("injected link fault")

    monkeypatch.setattr("alberta_framework.benchmarks.ipmnist_ceiling.os.link", fail_link)
    with pytest.raises(OSError, match="injected link fault"):
        _atomic_publish(destination, lambda stream: stream.write(b"complete"))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_rolls_back_directory_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.bin"
    real_fsync = os.fsync
    calls = 0

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory sync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("alberta_framework.benchmarks.ipmnist_ceiling.os.fsync", fail_second_fsync)
    with pytest.raises(OSError, match="injected directory sync fault"):
        _atomic_publish(destination, lambda stream: stream.write(b"complete"))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_rule_discovery_summary_uses_explicit_directories(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    for name in SCREEN_ARMS:
        for seed in (0, 1, 2):
            accuracy = 0.8 if name == CHAMPION else 0.79
            _shard(screen / f"{name}_seed{seed}.json", seed=seed, accuracy=accuracy)

    summary = build_rule_discovery_summary(screen, confirm)

    assert summary["schema"] == "asi.rule_discovery.real_screen.v2"
    assert summary["legacy_compatibility"]["schema"] == (
        "alberta.rule_discovery.real_screen.v1"
    )
    assert summary["screen_60_task"][CHAMPION]["mean"] == pytest.approx(0.8)
    assert summary["paired_vs_champion_60_task"]["disc_r1"]["mean"] == pytest.approx(
        -0.01
    )
    assert summary["provenance"]["schema"] == "asi.ipmnist.analysis_provenance.v1"
    assert len(summary["provenance"]["inputs"]) == 24
    assert "rule_discovery" in summary["provenance"]["sources"]
    assert "ipmnist_provenance" in summary["provenance"]["sources"]


@pytest.mark.parametrize("seeds", [(), (0, 0), (True,), (-1,), (2**32,)])
def test_rule_summary_rejects_noncanonical_or_duplicate_seeds_before_io(
    tmp_path: Path, seeds: object
) -> None:
    with pytest.raises(ValueError):
        build_legacy_rule_discovery_summary(
            tmp_path / "screen",
            tmp_path / "confirm",
            seeds=seeds,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("accuracy", [[], [float("nan")], [-0.1], [1.1]])
def test_rule_summary_rejects_invalid_accuracy_payloads(
    tmp_path: Path, accuracy: list[float]
) -> None:
    screen = tmp_path / "screen"
    path = screen / f"{SCREEN_ARMS[0]}_seed0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": 0, "per_task_accuracy": accuracy}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        build_legacy_rule_discovery_summary(
            screen, tmp_path / "confirm", seeds=(0,)
        )


def test_rule_summary_requires_payload_seed_to_match_requested_seed(
    tmp_path: Path,
) -> None:
    screen = tmp_path / "screen"
    _shard(screen / f"{SCREEN_ARMS[0]}_seed0.json", seed=1, accuracy=0.8)
    with pytest.raises(ValueError, match="does not match requested seed"):
        build_legacy_rule_discovery_summary(
            screen, tmp_path / "confirm", seeds=(0,)
        )


def test_rule_summary_rejects_partial_confirmation_seed_set(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    for name in SCREEN_ARMS:
        for seed in (0, 1):
            _shard(screen / f"{name}_seed{seed}.json", seed=seed, accuracy=0.8)
    _shard(
        confirm / "disc_r1_pscale_norms_seed0.json", seed=0, accuracy=0.8
    )
    with pytest.raises(ValueError, match="confirmation seeds are incomplete"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0, 1))


def test_current_rule_discovery_legacy_payload_reconstructs_exactly() -> None:
    campaign = _REPO_ROOT / "outputs" / "ipmnist_screening"
    expected = json.loads(
        (_REPO_ROOT / "outputs" / "rule_discovery" / "real_screen_v1.json").read_text()
    )

    actual = build_legacy_rule_discovery_summary(
        campaign / "shards", campaign / "confirm_full"
    )

    assert actual == expected
