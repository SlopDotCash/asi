"""Contracts for the bounded CLEAR qualification lane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from alberta_framework.benchmarks.clear_qualification import (
    AVALANCHE_COMMIT,
    BUCKETS,
    CURATION_COMMIT,
    DEV_SEEDS,
    OFFICIAL_SITE_COMMIT,
    OFFICIAL_SITE_TREE,
    PROSPECTIVE_ASSET_SCHEMA,
    PROVIDER_REVISION,
    REFERENCE_COMMIT,
    SCHEMA,
    ArchiveIdentity,
    ClearDatasetReceipt,
    ClearQualificationError,
    _decode,
    _metric_values,
    execution_config,
    load_dataset_manifest,
    main,
    prospective_clear100_asset_plan,
    qualification_plan,
    validate_result,
    verify_dataset_manifest,
)

pytestmark = pytest.mark.unit


def _result_plan() -> dict[str, object]:
    archive = ArchiveIdentity("fixture", "fixture.zip", 2, "a" * 64)
    identity = {
        "dataset": "clear100",
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": tuple(range(2005, 2015)),
        "samples_per_bucket": (1,) * 10,
        "archives": [
            {"role": archive.role, "path": archive.path, "size_bytes": 2, "sha256": "a" * 64}
        ],
    }
    dataset_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return dict(
        qualification_plan(
            ClearDatasetReceipt(
                archives=(archive,),
                samples_per_bucket=(1,) * 10,
                archive_bytes=2,
                sample_count=10,
                dataset_sha256=dataset_sha,
            )
        )
    )


def _plan_sha256(plan: object) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _manifest(root: Path) -> bytes:
    archive = root / "clear100-local.zip"
    archive.write_bytes(b"small CLEAR fixture")
    payload = {
        "schema_version": SCHEMA,
        "dataset": "clear100",
        "protocol": "streaming-near-future",
        "buckets": list(BUCKETS),
        "years": list(range(2005, 2015)),
        "samples_per_bucket": [index + 1 for index in range(10)],
        "archives": [
            {
                "role": "locally-acquired-clear100",
                "path": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ],
        "provider_archive_checksums_published": False,
    }
    return json.dumps(payload).encode()


def test_manifest_verification_and_plan_are_exact_and_nonpromoting(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_manifest(tmp_path), root=tmp_path)
    plan = qualification_plan(receipt)
    assert receipt.sample_count == 55
    assert plan["promotion_authorized"] is False
    assert plan["execution_authorized"] is False
    assert plan["negative_retention_required"] is True
    assert plan["control_config"] == plan["mechanism_off_config"]
    assert plan["source_revisions"] == {
        "curation": CURATION_COMMIT,
        "reference_runner": REFERENCE_COMMIT,
        "avalanche": AVALANCHE_COMMIT,
    }
    assert plan["axes"] == [
        {"seed": seed, "arm": arm}
        for seed in DEV_SEEDS
        for arm in ("control", "mechanism-off")
    ]
    resources = plan["resource_budget_per_axis"]
    assert isinstance(resources, dict)
    assert resources["training_observations"] == 5_500
    assert resources["data_samples_read"] == 6_050
    assert resources["optimizer_updates"] == 1_000
    assert resources["model_queries"] == 550
    assert resources["environment_steps"] == 0
    assert resources["timing"] == "telemetry-only"


def test_cli_verifies_local_data_and_emits_only_a_nonexecuting_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_manifest(tmp_path))
    assert main((str(manifest), "--dataset-root", str(tmp_path))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "development-only-permanently-nonpromoting"
    assert payload["execution_authorized"] is False
    assert payload["promotion_authorized"] is False


def test_prospective_asset_plan_binds_provider_assets_and_every_open_gate() -> None:
    plan = prospective_clear100_asset_plan()
    assert plan["schema_version"] == PROSPECTIVE_ASSET_SCHEMA
    assert plan["classification"] == "prospective-source-and-asset-freeze-only"
    assert plan["source_revisions"] == {
        "curation": CURATION_COMMIT,
        "reference_runner": REFERENCE_COMMIT,
        "avalanche": AVALANCHE_COMMIT,
        "official_site_commit": OFFICIAL_SITE_COMMIT,
        "official_site_tree": OFFICIAL_SITE_TREE,
    }
    provider = plan["provider"]
    assert isinstance(provider, dict)
    assert provider["revision"] == PROVIDER_REVISION
    assert provider["public_at_observation"] is True
    assert provider["gated_at_observation"] is False
    assert provider["license_metadata_git_oid"] == "b187bb7e7d837a367ccd0862441947ad412c77f7"
    assets = plan["assets"]
    assert isinstance(assets, list)
    assert assets == [
        {
            "role": "provider-labeled-clear100-train-images-only",
            "path": "clear100-train-image-only.zip",
            "size_bytes": 3_289_951_359,
            "lfs_sha256": "0376b952674e6ef55c3923ee4ce61e5b299fea4e29bbc4780530636e8988fd72",
            "pointer_git_oid": "e441ed603eef45715947fe06567206bd90b26cf9",
            "xet_hash": "f09eba2c90ad5295187b77a4af788629a908f2ea70ae2a33886ed55b9abecfb5",
        },
        {
            "role": "provider-labeled-clear100-test",
            "path": "clear100-test.zip",
            "size_bytes": 1_640_361_665,
            "lfs_sha256": "c939753be4e62dc7732347e5e636ea599022c82f45443ea9e7166167e467abd0",
            "pointer_git_oid": "3a57c37f5b8beaf478b5e9a00fd38ed2454f5d6c",
            "xet_hash": "a5fdd88c13d87116b6f1c10b41407bd68ca459ea67fb7b7df351fd07fec2ae86",
        },
    ]
    assert plan["provider_archive_bytes"] == 4_930_313_024
    assert sum(asset["size_bytes"] for asset in assets) == plan["provider_archive_bytes"]
    claims = plan["claims"]
    assert isinstance(claims, dict)
    assert claims and all(value is False for value in claims.values())
    blockers = plan["blockers"]
    assert isinstance(blockers, list)
    assert len(blockers) == 11
    assert any("rights" in blocker and "takedown" in blocker for blocker in blockers)
    assert any("split semantics" in blocker for blocker in blockers)
    assert any("execution authorization" in blocker for blocker in blockers)


def test_prospective_asset_cli_is_consumed_without_local_manifest(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alberta_framework.benchmarks.clear_qualification.load_dataset_manifest",
        lambda _path: pytest.fail("prospective asset plan touched a local manifest"),
    )
    assert main(("--prospective-assets",)) == 0
    assert json.loads(capsys.readouterr().out) == prospective_clear100_asset_plan()


def test_prospective_asset_cli_rejects_local_verification_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(("--prospective-assets", str(tmp_path / "manifest.json")))
    with pytest.raises(SystemExit):
        main(("--prospective-assets", "--dataset-root", str(tmp_path)))


def test_manifest_path_read_is_metadata_gated_bounded_and_does_not_use_read_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_manifest(tmp_path))
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("unbounded read_bytes"))
    assert main((str(manifest), "--dataset-root", str(tmp_path))) == 0
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False


def test_manifest_path_rejects_oversize_before_open_symlink_and_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * ((1 << 20) + 1))
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("file was opened"))
    with pytest.raises(ClearQualificationError, match="byte limit"):
        load_dataset_manifest(oversized)
    monkeypatch.undo()

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "manifest-link.json"
    link.symlink_to(target)
    with pytest.raises(ClearQualificationError, match="non-symlink"):
        load_dataset_manifest(link)

    fifo = tmp_path / "manifest.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ClearQualificationError, match="regular"):
        load_dataset_manifest(fifo)


def test_manifest_path_accepts_exact_byte_cap_without_overread(tmp_path: Path) -> None:
    manifest = tmp_path / "exact-limit.json"
    payload = b"{}" + b" " * ((1 << 20) - 2)
    manifest.write_bytes(payload)
    assert load_dataset_manifest(manifest) == payload


def test_candidate_is_not_silently_present_in_mechanism_off() -> None:
    assert execution_config(mechanism_enabled=False) == execution_config(
        mechanism_enabled=False
    )
    assert execution_config(mechanism_enabled=True) != execution_config(
        mechanism_enabled=False
    )


def test_metric_reduction_matches_official_matrix_definitions() -> None:
    matrix = [[float((row * 10 + column) / 100) for column in range(10)] for row in range(10)]
    metrics = _metric_values(matrix)
    assert metrics["in_domain"] == pytest.approx(sum(matrix[i][i] for i in range(10)) / 10)
    assert metrics["next_domain"] == pytest.approx(
        sum(matrix[i][i + 1] for i in range(9)) / 9
    )
    assert metrics["accuracy"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i + 1)) / 55
    )
    assert metrics["forward_transfer"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i + 1, 10)) / 45
    )
    assert metrics["backward_transfer"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i)) / 45
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(protocol="iid"), "streaming"),
        (lambda value: value.update(buckets=list(range(10))), "temporal"),
        (lambda value: value.update(provider_archive_checksums_published=True), "invented"),
        (lambda value: value.update(samples_per_bucket=[1] * 9), "every labeled"),
    ],
)
def test_manifest_rejects_protocol_and_shape_drift(
    tmp_path: Path, mutation: object, match: str
) -> None:
    payload = json.loads(_manifest(tmp_path))
    assert callable(mutation)
    mutation(payload)
    with pytest.raises(ClearQualificationError, match=match):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_manifest_rejects_hash_size_path_and_symlink_attacks(tmp_path: Path) -> None:
    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["sha256"] = "0" * 64
    with pytest.raises(ClearQualificationError, match="SHA-256 does not match"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["size_bytes"] += 1
    with pytest.raises(ClearQualificationError, match="size"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["path"] = "../escape.zip"
    with pytest.raises(ClearQualificationError, match="canonical and relative"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    target = tmp_path / "target.zip"
    target.write_bytes(b"target")
    link = tmp_path / "link.zip"
    link.symlink_to(target)
    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0] = {
        "role": "archive",
        "path": link.name,
        "size_bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ClearQualificationError, match="regular file"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_manifest_rejects_duplicate_paths_and_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(_manifest(tmp_path))
    payload["archives"].append({**payload["archives"][0], "role": "duplicate"})
    with pytest.raises(ClearQualificationError, match="paths must be unique"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)
    payload = json.loads(_manifest(tmp_path))
    payload["authority"] = True
    with pytest.raises(ClearQualificationError, match="fields"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_result_validator_enforces_receipts_nonpromotion_and_negative_retention() -> None:
    plan = _result_plan()
    plan_sha = _plan_sha256(plan)
    budget = plan["resource_budget_per_axis"]
    assert isinstance(budget, dict)
    matrix = [[0.5 for _ in range(10)] for _ in range(10)]
    result = {
        "schema_version": SCHEMA,
        "plan_sha256": plan_sha,
        "status": "negative-development",
        "promotion_authorized": False,
        "negative_retained": True,
        "accuracy_matrix": matrix,
        "metrics": {
            "accuracy": 0.5,
            "in_domain": 0.5,
            "next_domain": 0.5,
            "forward_transfer": 0.5,
            "backward_transfer": 0.5,
        },
        "resource_receipts": {
            "persistent_bytes": 1,
            "archive_bytes": 2,
            "training_observations": budget["training_observations"],
            "data_samples_read": budget["data_samples_read"],
            "optimizer_updates": budget["optimizer_updates"],
            "model_queries": budget["model_queries"],
            "environment_steps": 0,
            "wall_seconds_telemetry": 6,
        },
    }
    assert validate_result(json.dumps(result).encode(), expected_plan=plan) == result
    mismatched = json.loads(json.dumps(result))
    mismatched["resource_receipts"]["model_queries"] += 1
    with pytest.raises(ClearQualificationError, match="model_queries"):
        validate_result(json.dumps(mismatched).encode(), expected_plan=plan)
    hostile_metric = {**result, "metrics": {**result["metrics"], "accuracy": True}}
    with pytest.raises(ClearQualificationError, match="finite exact floats"):
        validate_result(json.dumps(hostile_metric).encode(), expected_plan=plan)
    for field, value, match in (
        ("promotion_authorized", True, "nonpromotion"),
        ("negative_retained", False, "negative retention"),
        ("status", "promoted", "status"),
        ("plan_sha256", "b" * 64, "provenance"),
    ):
        hostile = {**result, field: value}
        with pytest.raises(ClearQualificationError, match=match):
            validate_result(json.dumps(hostile).encode(), expected_plan=plan)

    forged_plan = json.loads(json.dumps(plan))
    forged_plan["resource_budget_per_axis"]["training_observations"] = 0
    with pytest.raises(ClearQualificationError, match="reviewed protocol"):
        validate_result(json.dumps(result).encode(), expected_plan=forged_plan)


def test_result_rejects_scalar_alias_and_unbounded_payload() -> None:
    plan = _result_plan()
    result = {
        "schema_version": SCHEMA,
        "plan_sha256": _plan_sha256(plan),
        "status": "completed-development",
        "promotion_authorized": False,
        "negative_retained": True,
        "accuracy_matrix": [[0.5 for _ in range(10)] for _ in range(10)],
        "metrics": {
            "accuracy": 0.5,
            "in_domain": 0.5,
            "next_domain": 0.5,
            "forward_transfer": 0.5,
            "backward_transfer": 0.5,
        },
        "resource_receipts": {
            "persistent_bytes": True,
            "archive_bytes": 0,
            "training_observations": 0,
            "data_samples_read": 0,
            "optimizer_updates": 0,
            "model_queries": 0,
            "environment_steps": 0,
            "wall_seconds_telemetry": 0,
        },
    }
    with pytest.raises(ClearQualificationError, match="exact integer"):
        validate_result(json.dumps(result).encode(), expected_plan=plan)
    with pytest.raises(ClearQualificationError, match="byte limit"):
        validate_result(b" " * ((1 << 20) + 1), expected_plan=plan)


def _nested_object_bytes(depth: int) -> bytes:
    return (b'{"a":' * depth) + b"0" + (b"}" * depth)


def test_decode_rejects_deep_object_nest_without_recursion_error() -> None:
    raw = _nested_object_bytes(10_000)
    assert len(raw) < (1 << 20)
    with pytest.raises(ClearQualificationError, match="nesting-depth|recursion"):
        _decode(raw, limit=1 << 20, label="dataset manifest")


def test_verify_dataset_manifest_rejects_deep_object_nest(tmp_path: Path) -> None:
    raw = _nested_object_bytes(10_000)
    with pytest.raises(ClearQualificationError, match="nesting-depth|recursion"):
        verify_dataset_manifest(raw, root=tmp_path)


def test_decode_rejects_one_past_strict_json_depth() -> None:
    raw = _nested_object_bytes(65)
    with pytest.raises(ClearQualificationError, match="nesting-depth|recursion"):
        _decode(raw, limit=1 << 20, label="dataset manifest")


def test_decode_accepts_shallow_object() -> None:
    assert _decode(b'{"ok": true}', limit=1 << 20, label="dataset manifest") == {"ok": True}


def test_decode_still_rejects_invalid_json() -> None:
    with pytest.raises(ClearQualificationError, match="not valid JSON|JSON"):
        _decode(b"{", limit=1 << 20, label="dataset manifest")

