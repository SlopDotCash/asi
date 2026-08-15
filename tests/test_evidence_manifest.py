"""Contract tests for the fail-closed evidence-claim manifest."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation import evidence_manifest_cli
from alberta_framework.evaluation.evidence_manifest import (
    CLAIM_CONTRACT_VERSION,
    DIRTY_STATE_POLICY_VERSION,
    MANIFEST_SCHEMA_VERSION,
    EvidenceClass,
    EvidenceLevel,
    EvidenceSpec,
    ValidationResult,
    build_evidence_manifest,
    evidence_manifest_exit_code,
    evidence_manifest_json,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Validation:
    valid: bool
    accepted: bool
    errors: tuple[str, ...] = ()


def _spec(
    name: str,
    *,
    accepted: bool,
    valid: bool = True,
    evidence_class: EvidenceClass = "scientific",
    evidence_level: EvidenceLevel = "L2",
    promotes_scientific_claim: bool = True,
) -> EvidenceSpec:
    def _load(path: Path) -> dict[str, object]:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        return cast(dict[str, object], parsed)

    def _validate(artifact: Mapping[str, object]) -> ValidationResult:
        assert isinstance(artifact, dict)
        return _Validation(valid=valid, accepted=accepted)

    protocol: dict[str, object] = {
        "protocol_version": "test.protocol.v1",
        "supported_claim": f"test-only scope for {name}",
        "seed_roles": {
            "development": [0, 1],
            "promoted_held_out_evidence": [30, 31],
        },
        "limitations": ["test fixture only", "not a general scientific result"],
    }
    return EvidenceSpec(
        name=name,
        claim_scope=f"test-only scope for {name}",
        evidence_class=evidence_class,
        evidence_level=evidence_level,
        promotes_scientific_claim=promotes_scientific_claim,
        relative_path=Path(f"{name}.json"),
        expected_schema="test.schema.v1",
        command_argv=("python", "-m", "test_generator", "--output", f"{name}.json"),
        protocol=protocol,
        configuration={"steps": 10, "learning_rate": 0.1},
        seeds=cast(Mapping[str, object], protocol["seed_roles"]),
        thresholds={"minimum_effect": 0.25},
        limitations=("test fixture only", "not a general scientific result"),
        source_paths=(Path(f"{name}.py"),),
        required_environment_fields=("python", "platform", "packages"),
        loader=_load,
        validator=_validate,
    )


def _write_source(root: Path, spec: EvidenceSpec) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in spec.source_paths:
        raw = f"# pinned source for {spec.name}\n".encode()
        (root / relative_path).write_bytes(raw)
        hashes[relative_path.as_posix()] = hashlib.sha256(raw).hexdigest()
    return hashes


def _write_artifact(
    root: Path,
    spec: EvidenceSpec,
    *,
    schema: str | None = None,
    configuration: Mapping[str, object] | None = None,
    include_environment: bool = True,
) -> bytes:
    source_hashes = _write_source(root, spec)
    scientific: dict[str, object] = {
        "protocol": spec.protocol,
        "configuration": (spec.configuration if configuration is None else configuration),
        "thresholds": spec.thresholds,
        "source_provenance": {
            "repository_subtree": "test",
            "source_sha256": source_hashes,
        },
    }
    artifact: dict[str, object] = {
        "schema_version": spec.expected_schema if schema is None else schema,
        "scientific_payload": scientific,
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": hashlib.sha256(
                json.dumps(
                    scientific,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "operational_metadata": {},
    }
    if include_environment:
        cast(dict[str, object], artifact["operational_metadata"])["runtime"] = {
            "python": {"version": "test"},
            "platform": {"system": "test"},
            "packages": {"test": "1"},
        }
    raw = (json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / spec.relative_path).write_bytes(raw)
    return raw


def _claims(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    claims = cast(list[dict[str, object]], manifest["claims"])
    return {cast(str, claim["claim_id"]): claim for claim in claims}


@pytest.mark.unit
def test_manifest_distinguishes_acceptance_rejection_and_missing(
    tmp_path: Path,
) -> None:
    accepted_spec = _spec("accepted", accepted=True)
    rejected_spec = _spec("rejected", accepted=False)
    missing_spec = _spec("missing", accepted=True)
    accepted_raw = _write_artifact(tmp_path, accepted_spec)
    _write_artifact(tmp_path, rejected_spec)
    _write_source(tmp_path, missing_spec)

    manifest = build_evidence_manifest(
        tmp_path,
        specs=(accepted_spec, rejected_spec, missing_spec),
    )

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["claim_contract_version"] == CLAIM_CONTRACT_VERSION
    assert manifest["overall_status"] == "not-run"
    assert manifest["all_required_present"] is False
    assert manifest["all_required_valid"] is False
    assert manifest["all_required_accepted"] is False
    records = _claims(manifest)
    assert records["accepted"]["status"] == "accepted"
    assert records["rejected"]["status"] == "valid-rejection"
    assert records["missing"]["status"] == "not-run"
    artifact = cast(list[dict[str, object]], records["accepted"]["artifacts"])[0]
    assert artifact["bytes_sha256"] == hashlib.sha256(accepted_raw).hexdigest()
    assert artifact["scientific_content_sha256"] is not None
    assert evidence_manifest_exit_code(manifest) == 1

    required_claim_fields = {
        "status",
        "scope",
        "level",
        "command",
        "artifacts",
        "protocol",
        "configuration",
        "seeds",
        "thresholds",
        "environment_provenance",
        "source_provenance",
        "dirty_state_policy",
        "limitations",
        "validation_timestamp_utc",
    }
    for record in records.values():
        assert required_claim_fields <= record.keys()
        command = cast(dict[str, object], record["command"])
        assert command["shell"] is False
        assert cast(list[str], command["argv"])[0] == "python"
        timestamp = datetime.fromisoformat(cast(str, record["validation_timestamp_utc"]))
        assert timestamp.tzinfo is not None
        assert record["validation_timestamp_utc"] == manifest["validation_timestamp_utc"]
        dirty_policy = cast(dict[str, object], record["dirty_state_policy"])
        assert dirty_policy["policy_version"] == DIRTY_STATE_POLICY_VERSION
        assert dirty_policy["clean_worktree_required"] is False
        assert dirty_policy["registered_source_hash_match_required"] is True


@pytest.mark.unit
def test_invalid_validator_schema_or_contract_fails_with_exit_two(
    tmp_path: Path,
) -> None:
    invalid_spec = _spec("invalid", accepted=True, valid=False)
    wrong_schema_spec = _spec("wrong_schema", accepted=True)
    drifted_config_spec = _spec("drifted_config", accepted=True)
    no_environment_spec = _spec("no_environment", accepted=True)
    _write_artifact(tmp_path, invalid_spec)
    _write_artifact(tmp_path, wrong_schema_spec, schema="wrong.schema.v1")
    _write_artifact(
        tmp_path,
        drifted_config_spec,
        configuration={"steps": 11, "learning_rate": 0.1},
    )
    _write_artifact(
        tmp_path,
        no_environment_spec,
        include_environment=False,
    )

    manifest = build_evidence_manifest(
        tmp_path,
        specs=(
            invalid_spec,
            wrong_schema_spec,
            drifted_config_spec,
            no_environment_spec,
        ),
    )

    assert manifest["overall_status"] == "invalid"
    assert manifest["all_required_present"] is True
    assert manifest["all_required_valid"] is False
    assert evidence_manifest_exit_code(manifest) == 2
    records = _claims(manifest)
    assert "registered frozen configuration" in " ".join(
        cast(list[str], records["drifted_config"]["errors"])
    )
    assert "environment provenance" in " ".join(
        cast(list[str], records["no_environment"]["errors"])
    )


@pytest.mark.unit
def test_all_accepted_is_strict_json_but_not_a_completion_claim(
    tmp_path: Path,
) -> None:
    one = _spec("one", accepted=True)
    two = _spec("two", accepted=True)
    _write_artifact(tmp_path, one)
    _write_artifact(tmp_path, two)
    manifest = build_evidence_manifest(tmp_path, specs=(one, two))

    assert manifest["overall_status"] == "accepted"
    assert manifest["all_required_accepted"] is True
    assert manifest["supported_scientific_claim_count"] == 2
    assert evidence_manifest_exit_code(manifest) == 0
    serialized = evidence_manifest_json(manifest)
    assert json.loads(serialized) == manifest
    assert "not a scientific artifact" in str(manifest["interpretation"])
    assert "completion certificate" in str(manifest["interpretation"])


@pytest.mark.unit
def test_unit_and_smoke_evidence_are_structurally_nonpromoting(
    tmp_path: Path,
) -> None:
    smoke = _spec(
        "smoke",
        accepted=True,
        evidence_class="smoke",
        evidence_level="L1",
        promotes_scientific_claim=False,
    )
    _write_artifact(tmp_path, smoke)
    manifest = build_evidence_manifest(tmp_path, specs=(smoke,))
    record = _claims(manifest)["smoke"]

    assert record["status"] == "verified-nonpromoting"
    assert record["validator_accepted"] is True
    assert record["accepted"] is False
    assert record["scientific_claim_supported"] is False
    assert manifest["overall_status"] == "nonpromoting"
    assert manifest["required_scientific_claim_count"] == 0
    assert manifest["supported_scientific_claim_count"] == 0
    assert evidence_manifest_exit_code(manifest) == 1

    illegal_promotion = replace(smoke, promotes_scientific_claim=True)
    with pytest.raises(ValueError, match="only scientific L2/L3"):
        build_evidence_manifest(tmp_path, specs=(illegal_promotion,))

    illegal_level = replace(
        smoke,
        evidence_class="unit",
        evidence_level="L2",
    )
    with pytest.raises(ValueError, match="unit evidence cannot use L2"):
        build_evidence_manifest(tmp_path, specs=(illegal_level,))


@pytest.mark.unit
def test_registry_rejects_empty_duplicate_or_escaping_specs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_evidence_manifest(tmp_path, specs=())

    duplicated = _spec("duplicate", accepted=True)
    with pytest.raises(ValueError, match="names must be unique"):
        build_evidence_manifest(tmp_path, specs=(duplicated, duplicated))

    escaping_artifact = replace(
        duplicated,
        name="escape_artifact",
        relative_path=Path("../outside.json"),
    )
    with pytest.raises(ValueError, match="remain under root"):
        build_evidence_manifest(tmp_path, specs=(escaping_artifact,))

    escaping_source = replace(
        duplicated,
        name="escape_source",
        source_paths=(Path("../outside.py"),),
    )
    with pytest.raises(ValueError, match="source provenance"):
        build_evidence_manifest(tmp_path, specs=(escaping_source,))


@pytest.mark.integration
def test_cli_emits_manifest_and_propagates_nonzero_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "overall_status": "not-run",
    }
    monkeypatch.setattr(
        evidence_manifest_cli,
        "build_evidence_manifest",
        lambda root: fake_manifest,
    )

    assert evidence_manifest_cli.main(["--root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out) == fake_manifest


@pytest.mark.unit
def test_cli_refuses_pinned_output_even_when_missing_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned = tmp_path / "outputs" / "evidence_manifest.json"
    assert not pinned.exists()
    monkeypatch.setattr(evidence_manifest_cli, "DEFAULT_OUTPUT", pinned)

    def forbidden_build(root: Path) -> dict[str, object]:
        raise AssertionError(f"validation must not run for refused output {root}")

    monkeypatch.setattr(evidence_manifest_cli, "build_evidence_manifest", forbidden_build)

    assert evidence_manifest_cli.main(["--root", str(tmp_path), "--output", str(pinned)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pinned canonical artifact path" in captured.err
    assert not pinned.exists()


@pytest.mark.unit
def test_cli_refuses_alternate_roots_canonical_output_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned = tmp_path / "outputs" / "evidence_manifest.json"
    assert not pinned.exists()

    def forbidden_build(root: Path) -> dict[str, object]:
        raise AssertionError(f"validation must not run for refused output {root}")

    monkeypatch.setattr(evidence_manifest_cli, "build_evidence_manifest", forbidden_build)

    assert evidence_manifest_cli.main(["--root", str(tmp_path), "--output", str(pinned)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pinned canonical artifact path" in captured.err
    assert not pinned.exists()


@pytest.mark.unit
def test_cli_refuses_existing_output_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing.json"
    sentinel = b"existing manifest must survive"
    output.write_bytes(sentinel)

    def forbidden_build(root: Path) -> dict[str, object]:
        raise AssertionError(f"validation must not run for refused output {root}")

    monkeypatch.setattr(evidence_manifest_cli, "build_evidence_manifest", forbidden_build)

    assert evidence_manifest_cli.main(["--root", str(tmp_path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "existing output path" in captured.err
    assert output.read_bytes() == sentinel


@pytest.mark.unit
def test_cli_exclusive_create_preserves_output_created_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "racing.json"
    sentinel = b"concurrent writer wins"
    fake_manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "overall_status": "not-run",
    }

    def racing_build(root: Path) -> dict[str, object]:
        assert root == tmp_path
        output.write_bytes(sentinel)
        return fake_manifest

    monkeypatch.setattr(evidence_manifest_cli, "build_evidence_manifest", racing_build)

    assert evidence_manifest_cli.main(["--root", str(tmp_path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "File exists" in captured.err
    assert output.read_bytes() == sentinel


@pytest.mark.unit
def test_cli_writes_manifest_to_new_output_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "new" / "manifest.json"
    fake_manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "overall_status": "not-run",
    }
    monkeypatch.setattr(
        evidence_manifest_cli,
        "build_evidence_manifest",
        lambda root: fake_manifest,
    )

    assert evidence_manifest_cli.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == fake_manifest
    assert json.loads(output.read_text(encoding="utf-8")) == fake_manifest


@pytest.mark.unit
def test_evidence_status_entrypoint_is_packaged() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]

    assert (
        scripts["alberta-evidence-status"]
        == "alberta_framework.evaluation.evidence_manifest_cli:main"
    )
