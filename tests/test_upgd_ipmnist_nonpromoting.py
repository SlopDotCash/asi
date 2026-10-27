"""Synthetic tests for the structurally nonpromoting UPGD-IPMNIST validator."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_ipmnist import (
    NONPROMOTING_POLICY,
    PARTIAL_SCHEMA,
    PROTOCOL_DEVIATIONS,
    IPMNISTConfig,
    build_artifact,
    build_legacy_v1_artifact,
    merge_legacy_v1_partial_results,
    merge_partial_results,
)
from alberta_framework.evaluation import upgd_ipmnist_nonpromoting as nonpromoting
from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import (
    EXPECTED_CONFIG,
    EXPECTED_HYPERPARAMETERS,
    EXPECTED_NOTE,
    UPGD_IPMNIST_PARTIAL_SCHEMA,
    validate_upgd_ipmnist_artifact,
    validate_upgd_ipmnist_partials,
    validate_upgd_ipmnist_reconstructed_provenance,
    validate_upgd_ipmnist_v2_artifact,
    validate_upgd_ipmnist_v2_partials,
)

_ROOT = Path(__file__).resolve().parents[1]
_IMMUTABLE_V1_ARTIFACT = _ROOT / "outputs/upgd_ipmnist/results.v1.json"
_IMMUTABLE_V1_RECEIPT = _ROOT / "outputs/upgd_ipmnist/nonpromoting_receipt.v1.json"
_IMMUTABLE_V1_RECEIPT_SHA256 = (
    "c32595829f93ac86b96c6eefc722291bf365dbf724982a70f207d193bbcfc26e"
)
_CURRENT_RECEIPT = _ROOT / "outputs/upgd_ipmnist/nonpromoting_receipt.v2.json"
_CURRENT_RECEIPT_SHA256 = (
    "0c36f97c60cf5d10ef5478d83f1de274920335ac592d7d6a16b1374da0c44083"
)


def _partial_payload(learner: str, seed: int) -> dict[str, object]:
    task_length = EXPECTED_CONFIG["task_length"]
    accuracy = [
        float(np.float32((3300 + ((seed + task) % 101)) / task_length))
        for task in range(EXPECTED_CONFIG["n_tasks"])
    ]
    return {
        "schema": UPGD_IPMNIST_PARTIAL_SCHEMA,
        "learner": learner,
        "hyperparameters": dict(EXPECTED_HYPERPARAMETERS[learner]),
        "seeds": [seed],
        "config": dict(EXPECTED_CONFIG),
        "per_task_accuracy": [accuracy],
        "per_task_loss": [[1.0 + 0.001 * task for task in range(200)]],
        "per_task_plasticity": [[0.4 + 0.0001 * task for task in range(200)]],
        "wall_clock_seconds": 1000.0 + seed,
    }


def _write_shards(tmp_path: Path, seeds: tuple[int, ...] = (0, 1)) -> list[Path]:
    paths: list[Path] = []
    for learner in EXPECTED_HYPERPARAMETERS:
        for seed in seeds:
            path = tmp_path / f"{learner}_seed{seed}.json"
            path.write_text(
                json.dumps(_partial_payload(learner, seed), sort_keys=True),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def _write_artifact(tmp_path: Path, paths: list[Path]) -> Path:
    results = merge_legacy_v1_partial_results(paths)
    artifact = build_legacy_v1_artifact(
        results,
        IPMNISTConfig(),
        tmp_path / "openml-cache",
        notes=(EXPECTED_NOTE,),
    )
    # This fixture exercises the immutable historical v1 contract.  The
    # builder records the live runner, while the validator correctly requires
    # the exact audited environment stored in the pinned artifact.
    historical_artifact = json.loads(_IMMUTABLE_V1_ARTIFACT.read_text(encoding="utf-8"))
    artifact["environment"] = historical_artifact["environment"]
    path = tmp_path / "results.v1.json"
    path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    return path


def _v2_partial_payload(learner: str, seed: int) -> dict[str, object]:
    legacy = _partial_payload(learner, seed)
    return {
        "schema": PARTIAL_SCHEMA,
        "schema_version": 2,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "learner": learner,
        "hyperparameters": legacy["hyperparameters"],
        "seed_id": seed,
        "seed_count": 1,
        "config": legacy["config"],
        "matches_selected_publication_configuration": True,
        "selected_publication_configuration_match_scope": (
            "network_task_shape_and_horizon_only"
        ),
        "deviations": [dict(deviation) for deviation in PROTOCOL_DEVIATIONS],
        "per_task_accuracy": legacy["per_task_accuracy"],
        "per_task_loss": legacy["per_task_loss"],
        "per_task_plasticity": legacy["per_task_plasticity"],
        "wall_clock_seconds": legacy["wall_clock_seconds"],
    }


def _write_v2_shards(tmp_path: Path, seeds: tuple[int, ...] = (0, 1)) -> list[Path]:
    paths: list[Path] = []
    for learner in EXPECTED_HYPERPARAMETERS:
        for seed in seeds:
            path = tmp_path / f"{learner}_seed{seed}.v2.json"
            path.write_text(
                json.dumps(_v2_partial_payload(learner, seed), sort_keys=True),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def _write_v2_artifact(tmp_path: Path, paths: list[Path]) -> Path:
    results = merge_partial_results(paths)
    artifact = build_artifact(
        results,
        IPMNISTConfig(),
        tmp_path / "openml-cache",
        notes=("synthetic v2 structural fixture",),
        partial_paths=paths,
    )
    path = tmp_path / "results.schema-v2.json"
    path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    return path


@pytest.mark.unit
def test_complete_artifact_recomputes_but_can_never_promote(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    artifact = _write_artifact(tmp_path, paths)

    validation = validate_upgd_ipmnist_artifact(
        artifact,
        paths,
        expected_seeds=(0, 1),
    )

    assert validation.valid, validation.errors
    assert validation.development_only
    assert not validation.scientific_promotion_allowed
    assert validation.artifact_sha256 is not None
    assert len(validation.partial_sha256) == 4
    assert validation.observed_seed_pairs == (
        ("adamw", 0),
        ("adamw", 1),
        ("upgd_w", 0),
        ("upgd_w", 1),
    )


@pytest.mark.unit
def test_duplicate_expected_seeds_cannot_bypass_artifact_validation(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, seeds=(0,))
    artifact = tmp_path / "empty-artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    validation = validate_upgd_ipmnist_artifact(
        artifact,
        paths,
        expected_seeds=(0, 0),
    )

    assert not validation.valid
    assert any("expected_seeds must contain unique seeds" in error for error in validation.errors)


@pytest.mark.unit
@pytest.mark.parametrize(
    "expected_seeds",
    [(), (True,), (np.int64(0),), (0.0,), (-1,), (2**32,)],
)
def test_invalid_expected_seed_schedules_fail_closed(
    tmp_path: Path, expected_seeds: tuple[object, ...]
) -> None:
    validation = validate_upgd_ipmnist_partials(
        [],
        expected_seeds=expected_seeds,  # type: ignore[arg-type]
    )

    assert not validation.valid
    assert any("expected_seeds" in error for error in validation.errors)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("extra_field", "fields do not match"),
        ("wrong_hyperparameter", "hyperparameters.step_size"),
        ("impossible_accuracy", "1/5000 count lattice"),
        ("negative_wall_clock", "wall_clock_seconds"),
    ],
)
def test_partial_tampering_fails_closed(tmp_path: Path, tamper: str, expected_error: str) -> None:
    paths = _write_shards(tmp_path, seeds=(0,))
    target = tmp_path / "upgd_w_seed0.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if tamper == "extra_field":
        payload["unregistered"] = True
    elif tamper == "wrong_hyperparameter":
        payload["hyperparameters"]["step_size"] = 0.02
    elif tamper == "impossible_accuracy":
        payload["per_task_accuracy"][0][0] = 0.12345
    elif tamper == "negative_wall_clock":
        payload["wall_clock_seconds"] = -1.0
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(tamper)
    target.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_upgd_ipmnist_partials(paths, expected_seeds=(0,))

    assert not validation.valid
    assert any(expected_error in error for error in validation.errors)
    assert not validation.scientific_promotion_allowed


@pytest.mark.unit
def test_partial_rejects_negative_zero_frozen_hyperparameter(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, seeds=(0,))
    target = tmp_path / "adamw_seed0.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["hyperparameters"]["beta1"] = -0.0
    target.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_upgd_ipmnist_partials(paths, expected_seeds=(0,))

    assert not validation.valid
    assert any("hyperparameters.beta1" in error for error in validation.errors)
    assert not validation.scientific_promotion_allowed


@pytest.mark.unit
def test_strict_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "upgd_w_seed0.json"
    duplicate.write_text(
        '{"schema":"upgd_ipmnist.partial.v1","schema":"forged"}',
        encoding="utf-8",
    )
    duplicate_validation = validate_upgd_ipmnist_partials([duplicate], expected_seeds=(0,))
    assert not duplicate_validation.valid
    assert any("duplicate JSON key" in error for error in duplicate_validation.errors)

    nonfinite = tmp_path / "adamw_seed0.json"
    nonfinite.write_text('{"schema": NaN}', encoding="utf-8")
    nonfinite_validation = validate_upgd_ipmnist_partials([nonfinite], expected_seeds=(0,))
    assert not nonfinite_validation.valid
    assert any("non-finite JSON constant" in error for error in nonfinite_validation.errors)


@pytest.mark.unit
def test_artifact_aggregate_tampering_is_recomputed_from_primitives(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    artifact_path = _write_artifact(tmp_path, paths)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["learners"]["upgd_w"]["average_online_accuracy_mean"] += 0.01
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    validation = validate_upgd_ipmnist_artifact(
        artifact_path,
        paths,
        expected_seeds=(0, 1),
    )

    assert not validation.valid
    assert any("summaries do not recompute" in error for error in validation.errors)


@pytest.mark.unit
def test_filename_and_coverage_are_bound_fail_closed(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, seeds=(0,))
    original = tmp_path / "upgd_w_seed0.json"
    renamed = tmp_path / "unbound.json"
    original.rename(renamed)
    paths[paths.index(original)] = renamed

    validation = validate_upgd_ipmnist_partials(paths, expected_seeds=(0, 1))

    assert not validation.valid
    assert any("filename must bind payload identity" in error for error in validation.errors)
    assert any("shard coverage mismatch" in error for error in validation.errors)


@pytest.mark.unit
def test_expected_file_count_with_duplicate_identities_fails_without_raising(
    tmp_path: Path,
) -> None:
    partial_paths: list[Path] = []
    for directory_name in ("copy-a", "copy-b"):
        directory = tmp_path / directory_name
        directory.mkdir()
        for seed in (0, 1):
            path = directory / f"upgd_w_seed{seed}.json"
            path.write_text(
                json.dumps(_partial_payload("upgd_w", seed), sort_keys=True),
                encoding="utf-8",
            )
            partial_paths.append(path)
    artifact = tmp_path / "results.v1.json"
    artifact.write_text("{}", encoding="utf-8")

    validation = validate_upgd_ipmnist_artifact(
        artifact,
        partial_paths,
        expected_seeds=(0, 1),
    )

    assert not validation.valid
    assert any("duplicate learner/seed identities" in error for error in validation.errors)
    assert any("shard coverage mismatch" in error for error in validation.errors)
    assert not validation.scientific_promotion_allowed


@pytest.mark.unit
def test_posthoc_provenance_check_fails_when_source_and_cache_are_absent(
    tmp_path: Path,
) -> None:
    validation = validate_upgd_ipmnist_reconstructed_provenance(
        tmp_path / "empty-root",
        tmp_path / "empty-cache",
    )

    assert not validation.valid
    assert any("cannot read reconstructed source" in error for error in validation.errors)
    assert any("cannot read cached MNIST archive" in error for error in validation.errors)
    assert not validation.scientific_promotion_allowed


@pytest.mark.unit
def test_v2_artifact_recomputes_and_remains_permanently_nonpromoting(tmp_path: Path) -> None:
    paths = _write_v2_shards(tmp_path)
    artifact = _write_v2_artifact(tmp_path, paths)

    partial_validation = validate_upgd_ipmnist_v2_partials(paths)
    artifact_validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)

    assert partial_validation.valid, partial_validation.errors
    assert artifact_validation.valid, artifact_validation.errors
    assert artifact_validation.development_only
    assert not artifact_validation.scientific_promotion_allowed
    assert artifact_validation.observed_seed_pairs == (
        ("adamw", 0),
        ("adamw", 1),
        ("upgd_w", 0),
        ("upgd_w", 1),
    )


@pytest.mark.unit
def test_v2_validator_recursively_rejects_legacy_protocol_marker(tmp_path: Path) -> None:
    paths = _write_v2_shards(tmp_path, seeds=(0,))
    artifact = _write_v2_artifact(tmp_path, paths)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["protocol"]["nested"] = {"is_protocol_exact": True}
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    artifact_validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)
    assert not artifact_validation.valid
    assert any(
        "forbidden legacy is_protocol_exact" in error
        for error in artifact_validation.errors
    )

    partial_payload = json.loads(paths[0].read_text(encoding="utf-8"))
    partial_payload["evidence_policy"]["is_protocol_exact"] = True
    paths[0].write_text(json.dumps(partial_payload), encoding="utf-8")
    partial_validation = validate_upgd_ipmnist_v2_partials(paths)
    assert not partial_validation.valid
    assert any("legacy is_protocol_exact" in error for error in partial_validation.errors)


@pytest.mark.unit
def test_v2_artifact_manifest_binds_exact_shard_bytes(tmp_path: Path) -> None:
    paths = _write_v2_shards(tmp_path, seeds=(0,))
    artifact = _write_v2_artifact(tmp_path, paths)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["partial_manifest"][0]["sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)

    assert not validation.valid
    assert any("does not bind exact shard bytes" in error for error in validation.errors)


@pytest.mark.unit
def test_v2_expected_manifest_does_not_mix_digest_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second-read identity swap cannot bind snapshot B to digest A.

    The public validator already one-reads shards before reconstructing the
    expected manifest, so a first-read file replacement would be consumed by
    that earlier guard. The remaining hole is the loop's second
    ``_strict_json_object(path)`` after ``path.read_bytes()``; swapping that
    helper is the same split-read as replacing the path between the two calls.
    """
    paths = _write_v2_shards(tmp_path, seeds=(0,))
    artifact = _write_v2_artifact(tmp_path, paths)
    target = next(path for path in paths if path.name.startswith("upgd_w_"))
    swapped_identity = _v2_partial_payload("adamw", 1)
    original = nonpromoting._strict_json_object

    def split_read_identity(path: Path) -> dict[str, object]:
        if Path(path).resolve() == target.resolve():
            return dict(swapped_identity)
        return original(path)

    monkeypatch.setattr(nonpromoting, "_strict_json_object", split_read_identity)

    validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)

    assert validation.valid, validation.errors
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    target_entry = next(
        entry for entry in payload["partial_manifest"] if entry["path"] == target.as_posix()
    )
    raw = target.read_bytes()
    assert target_entry["learner"] == "upgd_w"
    assert target_entry["seed_id"] == 0
    assert target_entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert target_entry["size_bytes"] == len(raw)
    assert (target_entry["learner"], target_entry["seed_id"]) != ("adamw", 1)


@pytest.mark.unit
def test_v2_expected_manifest_stable_shards_match_file_digest(tmp_path: Path) -> None:
    paths = _write_v2_shards(tmp_path, seeds=(0, 1))
    artifact = _write_v2_artifact(tmp_path, paths)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    identities = [(entry["learner"], entry["seed_id"]) for entry in payload["partial_manifest"]]
    assert identities == [("adamw", 0), ("adamw", 1), ("upgd_w", 0), ("upgd_w", 1)]
    by_path = {entry["path"]: entry for entry in payload["partial_manifest"]}
    for path in paths:
        raw = path.read_bytes()
        entry = by_path[path.as_posix()]
        assert entry["size_bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()

    validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)
    assert validation.valid, validation.errors


@pytest.mark.unit
@pytest.mark.parametrize("schema_version", [2.0, "2"])
def test_v2_artifact_rejects_non_int_schema_version(
    tmp_path: Path, schema_version: object
) -> None:
    paths = _write_v2_shards(tmp_path, seeds=(0,))
    artifact = _write_v2_artifact(tmp_path, paths)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)

    assert not validation.valid
    assert any("schema_version" in error for error in validation.errors)


@pytest.mark.unit
@pytest.mark.parametrize(
    "policy_updates",
    [
        {
            "development_only": 1,
            "scientific_promotion_allowed": 0,
            "execution_attestation": 0,
        },
        {
            "development_only": 1.0,
            "scientific_promotion_allowed": 0.0,
            "execution_attestation": 0.0,
        },
    ],
)
def test_v2_artifact_rejects_numeric_policy_aliases(
    tmp_path: Path, policy_updates: dict[str, object]
) -> None:
    paths = _write_v2_shards(tmp_path, seeds=(0,))
    artifact = _write_v2_artifact(tmp_path, paths)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    policy = dict(payload["evidence_policy"])
    policy.update(policy_updates)
    payload["evidence_policy"] = policy
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    validation = validate_upgd_ipmnist_v2_artifact(artifact, paths)

    assert not validation.valid
    assert any("evidence policy" in error for error in validation.errors)


@pytest.mark.unit
def test_v2_validator_rejects_v1_schema_even_when_filename_says_v2(tmp_path: Path) -> None:
    v1_paths = _write_shards(tmp_path, seeds=(0,))
    misleading_path = tmp_path / "results.reconciled_nonpromoting.v2.json"
    v1_artifact = _write_artifact(tmp_path, v1_paths)
    misleading_path.write_bytes(v1_artifact.read_bytes())

    validation = validate_upgd_ipmnist_v2_artifact(misleading_path, v1_paths)

    assert not validation.valid
    assert any("not a v2 partial" in error for error in validation.errors)
    assert any("forbidden legacy is_protocol_exact" in error for error in validation.errors)


@pytest.mark.unit
def test_nonpromoting_receipt_binds_stored_shards_and_summary() -> None:
    raw = _CURRENT_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _CURRENT_RECEIPT_SHA256
    receipt = json.loads(raw)

    assert receipt["schema_version"] == (
        "alberta.upgd_ipmnist_nonpromoting_receipt.v2"
    )
    assert receipt["receipt_role"] == "versioned_provenance_successor"
    predecessor = receipt["predecessor_receipt"]
    assert _ROOT / predecessor["path"] == _IMMUTABLE_V1_RECEIPT
    predecessor_raw = _IMMUTABLE_V1_RECEIPT.read_bytes()
    assert len(predecessor_raw) == predecessor["size_bytes"] == 13290
    assert hashlib.sha256(predecessor_raw).hexdigest() == predecessor["sha256"]
    assert predecessor["sha256"] == _IMMUTABLE_V1_RECEIPT_SHA256
    assert predecessor["preserved_byte_for_byte"] is True
    predecessor_payload = json.loads(predecessor_raw)
    predecessor_runbook = predecessor_payload["post_hoc_reconstructed_provenance"][
        "runbook"
    ]
    predecessor_runbook_raw = (_ROOT / predecessor_runbook["path"]).read_bytes()
    assert hashlib.sha256(predecessor_runbook_raw).hexdigest() == (
        predecessor_runbook["sha256"]
    )
    assert receipt["status"] == "complete_structural_validation"
    assert receipt["development_only"] is True
    assert receipt["scientific_promotion_allowed"] is False
    assert receipt["protocol_scope"]["complete_published_protocol_exact"] is False
    legacy_marker = receipt["protocol_scope"]["legacy_artifact_protocol_marker"]
    assert legacy_marker == {
        "field": "protocol.is_protocol_exact",
        "recorded_value": True,
        "interpretation": "legacy selected-configuration shape marker only",
        "does_not_override_complete_published_protocol_exact": True,
    }
    assert receipt["protocol_scope"]["recorded_seed_count"] == 10
    assert receipt["protocol_scope"]["published_seed_count"] == 20

    canonical_binding = receipt["canonical_reconciled_artifact"]
    canonical_path = _ROOT / canonical_binding["path"]
    canonical_raw = canonical_path.read_bytes()
    assert len(canonical_raw) == canonical_binding["size_bytes"]
    assert hashlib.sha256(canonical_raw).hexdigest() == canonical_binding["sha256"]

    prior_binding = receipt["prior_reconciliation"]
    prior_raw = (_ROOT / prior_binding["path"]).read_bytes()
    assert len(prior_raw) == prior_binding["size_bytes"]
    assert hashlib.sha256(prior_raw).hexdigest() == prior_binding["sha256"]

    finalizer_binding = receipt["preserved_finalizer_artifact"]
    finalizer_raw = (_ROOT / finalizer_binding["path"]).read_bytes()
    assert len(finalizer_raw) == finalizer_binding["size_bytes"]
    assert hashlib.sha256(finalizer_raw).hexdigest() == finalizer_binding["sha256"]
    assert finalizer_binding["strict_validator_valid"] is False
    assert finalizer_binding["preserved_without_overwrite"] is True

    learner_averages: dict[str, list[float]] = {
        "upgd_w": [],
        "adamw": [],
    }
    assert len(receipt["partial_shards"]) == 20
    for binding in receipt["partial_shards"]:
        path = _ROOT / binding["path"]
        shard_raw = path.read_bytes()
        assert len(shard_raw) == binding["size_bytes"]
        assert hashlib.sha256(shard_raw).hexdigest() == binding["sha256"]
        shard = json.loads(shard_raw)
        assert shard["learner"] == binding["learner"]
        assert shard["seeds"] == [binding["seed"]]
        accuracy = np.asarray(shard["per_task_accuracy"], dtype=np.float64)
        learner_averages[binding["learner"]].append(float(accuracy[0].mean()))

    provenance = receipt["post_hoc_reconstructed_provenance"]
    assert provenance["execution_attestation"] is False
    assert provenance["full_import_closure_snapshotted"] is False

    runbook_binding = provenance["runbook"]
    runbook_raw = (_ROOT / runbook_binding["path"]).read_bytes()
    assert hashlib.sha256(runbook_raw).hexdigest() == runbook_binding["sha256"]

    validator_binding = receipt["strict_validation"]
    validator_raw = (_ROOT / validator_binding["validator_path"]).read_bytes()
    assert len(validator_raw) == validator_binding["validator_size_bytes"]
    assert hashlib.sha256(validator_raw).hexdigest() == validator_binding["validator_sha256"]

    bundle_binding = provenance["reconstructed_source_bundle"]
    bundle_root = _ROOT / bundle_binding["path"]
    manifest_raw = (_ROOT / bundle_binding["manifest_path"]).read_bytes()
    assert len(manifest_raw) == bundle_binding["manifest_size_bytes"]
    assert hashlib.sha256(manifest_raw).hexdigest() == bundle_binding["manifest_sha256"]
    source_manifest = json.loads(manifest_raw)
    assert source_manifest["execution_attestation"] is False
    assert source_manifest["full_import_closure_snapshotted"] is False
    assert len(source_manifest["files"]) == bundle_binding["numeric_file_count"]
    assert sum(binding["size_bytes"] for binding in source_manifest["files"]) == (
        bundle_binding["numeric_file_size_bytes"]
    )
    for binding in source_manifest["files"]:
        source_raw = (bundle_root / binding["path"]).read_bytes()
        assert len(source_raw) == binding["size_bytes"]
        assert hashlib.sha256(source_raw).hexdigest() == binding["sha256"]

    runner_raw = (bundle_root / "alberta_framework/benchmarks/upgd_ipmnist.py").read_bytes()
    assert hashlib.sha256(runner_raw).hexdigest() == provenance["merge_runner_sha256"]
    active_runner_raw = (_ROOT / "alberta_framework/benchmarks/upgd_ipmnist.py").read_bytes()
    assert hashlib.sha256(active_runner_raw).hexdigest() != provenance["merge_runner_sha256"]
    numeric_source_map = provenance["numeric_source_sha256"]
    canonical_source_map = json.dumps(
        numeric_source_map,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical_source_map).hexdigest() == (
        provenance["numeric_source_hash_map_sha256"]
    )
    for path_string, expected_sha256 in numeric_source_map.items():
        source_raw = (bundle_root / path_string).read_bytes()
        assert hashlib.sha256(source_raw).hexdigest() == expected_sha256

    log_binding = provenance["operational_log_archive"]
    log_path = _ROOT / log_binding["path"]
    log_raw = log_path.read_bytes()
    assert len(log_raw) == log_binding["size_bytes"]
    assert hashlib.sha256(log_raw).hexdigest() == log_binding["sha256"]
    with tarfile.open(log_path, mode="r:") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    assert len(names) == log_binding["file_count"] == 23
    assert len(names) == len(set(names))

    upgd = np.asarray(learner_averages["upgd_w"], dtype=np.float64)
    adamw = np.asarray(learner_averages["adamw"], dtype=np.float64)
    paired = upgd - adamw
    summary = receipt["paired_development_summary"]
    np.testing.assert_array_equal(
        paired,
        np.asarray(
            summary["per_seed_average_online_accuracy_differences"],
            dtype=np.float64,
        ),
    )
    assert float(paired.mean()) == summary["mean_difference"]
    assert float(paired.std(ddof=1)) == summary["sample_standard_deviation"]
    assert int(np.count_nonzero(paired > 0.0)) == summary["wins"] == 10
    assert summary["inferential_statistics_admissible"] is False

    artifact = json.loads(canonical_raw)
    assert artifact["protocol"]["is_protocol_exact"] is legacy_marker["recorded_value"]
    for learner in ("upgd_w", "adamw"):
        artifact_summary = artifact["learners"][learner]
        receipt_summary = receipt["learner_summaries"][learner]
        assert artifact_summary["n_seeds"] == receipt_summary["n"]
        for field in (
            "average_online_accuracy_mean",
            "average_online_accuracy_stderr",
            "first_quarter_accuracy_mean",
            "last_quarter_accuracy_mean",
            "accuracy_drift_last_minus_first",
            "average_plasticity_mean",
        ):
            assert artifact_summary[field] == receipt_summary[field]
