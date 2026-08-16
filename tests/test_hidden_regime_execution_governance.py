"""Fail-closed contracts for the managed hidden-regime execution boundary.

:mod:`alberta_framework.evaluation.hidden_regime_execution_governance` gates
who may execute the exact calibration worlds.  Vocabulary used throughout
this suite:

* **genesis** — the deterministic, zero-entry initial ledger tree binding
  the source-archive, runtime, and readiness-receipt digests;
* **ledger** — the on-disk record of execution attempts (started, completed,
  replay, and finalization records), only ever published via atomic
  whole-tree or single-record installs;
* **authorization** — a process-sealed capability bound to one readiness
  receipt, genesis, and case, consumed before the first learner transition,
  so each calibration case executes at most once (a crash permits an exact
  replay only);
* **zip provenance capability** — attestation that the worker executed from
  the content-addressed source ZIP with bytecode/``__pycache__`` isolation.

The tests drive real (tiny) hidden-regime learner executions through the
boundary and then attack it: forked children inheriting process-local
capabilities, fresh processes replaying seals, forged issuers, ledger
tampering and symlinks, fault injection at every atomic-install checkpoint,
and concurrent initializers/executors must fail closed with exactly one
winner.  The ``development`` marker keeps this suite in the same lane as the
calibration machinery it governs: passing here is never promoted evidence.
"""

from __future__ import annotations

import base64
import dataclasses
import errno
import hashlib
import json
import os
import pickle
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.hidden_regime_execution_governance as governance
import alberta_framework.evaluation.hidden_regime_factorial_calibration as calibration
import alberta_framework.evaluation.hidden_regime_trace_audit as trace_audit
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
    PROCESS_LOCAL_AUTHORIZATION_SCOPE,
    CalibrationExecutionAuthorization,
    CalibrationZipProvenanceCapability,
    HiddenRegimeCaseConsumedError,
    HiddenRegimeExecutionGovernanceError,
    HiddenRegimeProtectedExecutionError,
    begin_managed_hidden_regime_execution,
    build_calibration_execution_genesis,
    calibration_execution_configuration_sha256,
    calibration_execution_genesis_receipt_binding,
    calibration_execution_primitive_trace_sha256,
    calibration_execution_resource_sha256,
    calibration_execution_summary_sha256,
    canonical_json_bytes,
    canonical_sha256,
    classify_hidden_regime_world,
    complete_managed_hidden_regime_execution,
    initialize_calibration_execution_ledger,
    issue_calibration_execution_authorization,
    require_valid_calibration_execution_genesis,
    require_valid_calibration_execution_inventory,
    require_valid_calibration_execution_started_record,
    snapshot_calibration_execution_inventory,
    validate_completed_calibration_ledger_snapshot,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CONSUMED_CALIBRATION_NAMESPACE,
    N_MATCHED_CASES,
    SEED_SNAPSHOT_SHA256,
    build_hidden_regime_factorial_calibration_design,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    SELECTIVE_FULL,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimeSeedPair,
    run_hidden_regime_condition,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    EVIDENCE_BOUNDARY,
    HiddenRegimeTraceAuditReport,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    HiddenRegimeWorldConfig,
)

pytestmark = pytest.mark.development

_SOURCE_ARCHIVE_DIGEST = "1" * 64
_SOURCE_MANIFEST_DIGEST = "2" * 64
_RUNTIME_DIGEST = "3" * 64
_READINESS_DIGEST = "4" * 64


@dataclasses.dataclass(frozen=True)
class _TinySummary:
    num_steps: int


@dataclasses.dataclass(frozen=True)
class _TinyResource:
    state_bytes: int = 552


@dataclasses.dataclass(frozen=True)
class _TinyTrace:
    step: np.ndarray
    reward: np.ndarray


@dataclasses.dataclass(frozen=True)
class _TinyResult:
    condition: str
    seed_pair: HiddenRegimeSeedPair
    config: HiddenRegimeDevelopmentConfig
    summary: _TinySummary
    resource: _TinyResource
    trace: _TinyTrace
    final_state: tuple[np.ndarray, ...] = dataclasses.field(
        default_factory=lambda: (np.asarray([1, 2], dtype=np.int32),)
    )


def _genesis() -> dict[str, object]:
    return build_calibration_execution_genesis(
        source_archive_sha256=_SOURCE_ARCHIVE_DIGEST,
        source_manifest_sha256=_SOURCE_MANIFEST_DIGEST,
        runtime_identity_sha256=_RUNTIME_DIGEST,
    )


def _case_inputs(case_index: int = 0) -> tuple[str, HiddenRegimeSeedPair, Any]:
    case = build_hidden_regime_factorial_calibration_design().cases[case_index]
    seed_pair = HiddenRegimeSeedPair(
        namespace=CONSUMED_CALIBRATION_NAMESPACE,
        index=case.seed_index,
        world_seed=case.world_seed,
        learner_seed=case.learner_seed,
    )
    return case.condition, seed_pair, governance._expected_case_config(case)


def _fake_readiness(genesis: dict[str, object]) -> SimpleNamespace:
    body = {
        "authorization": {
            "ready_for_calibration": True,
            "protected_candidate_execution_permitted": False,
        },
        "source_snapshot": {"archive": {"sha256": _SOURCE_ARCHIVE_DIGEST}},
        governance.READINESS_EXECUTION_GOVERNANCE_FIELD: (
            calibration_execution_genesis_receipt_binding(genesis)
        ),
    }
    return SimpleNamespace(
        payload={"body": body, "receipt_sha256": _READINESS_DIGEST},
        receipt_sha256=_READINESS_DIGEST,
        source_archive_sha256=_SOURCE_ARCHIVE_DIGEST,
        source_manifest_sha256=_SOURCE_MANIFEST_DIGEST,
        runtime_identity_sha256=_RUNTIME_DIGEST,
    )


def _initialize(tmp_path: Path) -> tuple[Path, dict[str, object], SimpleNamespace]:
    genesis = _genesis()
    publication_root = tmp_path / "ledgers"
    publication_root.mkdir(parents=True)
    ledger = initialize_calibration_execution_ledger(
        publication_root,
        genesis,
        authorize_initialization=True,
    )
    return ledger.directory, genesis, _fake_readiness(genesis)


def _issue(
    monkeypatch: pytest.MonkeyPatch,
    ledger: Path,
    readiness: SimpleNamespace,
    *,
    case_index: int = 0,
    case_request_binding_sha256: str = "5" * 64,
    attempt_request_payload_sha256: str | None = None,
    allow_exact_replay: bool = False,
) -> tuple[CalibrationExecutionAuthorization, str, HiddenRegimeSeedPair, Any]:
    monkeypatch.setattr(governance, "_validate_readiness_bundle", lambda bundle, archive: bundle)
    condition, seed_pair, config = _case_inputs(case_index)
    capability = _fake_provenance(readiness)
    authorization = issue_calibration_execution_authorization(
        ledger_directory=ledger,
        readiness_bundle=readiness,
        readiness_source_archive=b"test-bound-archive",
        zip_provenance_capability=capability,
        case_index=case_index,
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        case_request_binding_sha256=case_request_binding_sha256,
        attempt_request_payload_sha256=(
            case_request_binding_sha256
            if attempt_request_payload_sha256 is None
            else attempt_request_payload_sha256
        ),
        explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        allow_exact_replay=allow_exact_replay,
    )
    return authorization, condition, seed_pair, config


def _fake_provenance(readiness: SimpleNamespace) -> CalibrationZipProvenanceCapability:
    binding = governance._zip_provenance_binding(
        readiness_receipt_sha256=readiness.receipt_sha256,
        source_archive_sha256=readiness.source_archive_sha256,
        source_manifest_sha256=readiness.source_manifest_sha256,
        runtime_identity_sha256=readiness.runtime_identity_sha256,
    )
    payload = governance._payload_with_digest(
        {
            "schema": governance.CALIBRATION_ZIP_PROVENANCE_SCHEMA,
            "binding": binding,
            "environment": {
                "source_archive_locator": governance.ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
                "canonical_runtime_search_paths_sha256": "b" * 64,
                "project_modules_sha256": "a" * 64,
            },
            "zip_provenance_policy": governance.ZIP_PROVENANCE_POLICY,
        },
        "zip_provenance_attestation_sha256",
    )
    nonce = os.urandom(32).hex()
    capability = CalibrationZipProvenanceCapability(
        payload=payload,
        seal=governance._seal(
            "calibration-zip-provenance-capability-v1",
            {"nonce": nonce, **payload},
        ),
        nonce=nonce,
    )
    governance._ZIP_PROVENANCE_CAPABILITIES[nonce] = capability
    return capability


def test_zip_provenance_attestation_and_aggregate_binding_ignore_staging_locator(
    tmp_path: Path,
) -> None:
    source_archive_sha256 = "1" * 64
    stdlib_search_paths = ["/bound-runtime/stdlib", "/bound-runtime/lib-dynload"]
    dependency_search_paths = ["/bound-runtime/purelib", "/bound-runtime/platlib"]
    module_rows: list[object] = [
        {
            "module": "alberta_framework.evaluation.hidden_regime_factorial_calibration",
            "archive_member": (
                "alberta_framework/evaluation/hidden_regime_factorial_calibration.py"
            ),
        }
    ]

    def environment_for(source_archive_path: Path) -> dict[str, object]:
        return governance._canonical_zip_provenance_environment(
            source_archive_path=source_archive_path,
            expected_source_archive_sha256=source_archive_sha256,
            expected_prefix=Path("/bound-runtime"),
            expected_exec_prefix=Path("/bound-runtime"),
            expected_dependency_paths=dependency_search_paths,
            stdlib_search_paths=stdlib_search_paths,
            exact_runtime_search_paths=[
                source_archive_path.absolute().as_posix(),
                *stdlib_search_paths,
                *dependency_search_paths,
            ],
            runtime_path_policy_sha256="2" * 64,
            module_rows=module_rows,
        )

    first_path = tmp_path / "staging-first-random-name" / "source.zip"
    second_path = tmp_path / "staging-second-random-name" / "source.zip"
    first_environment = environment_for(first_path)
    second_environment = environment_for(second_path)
    assert first_environment == second_environment
    serialized_environment = governance.canonical_json_bytes(first_environment)
    assert tmp_path.as_posix().encode() not in serialized_environment
    assert b"source_archive_path" not in serialized_environment

    binding = governance._zip_provenance_binding(
        readiness_receipt_sha256="3" * 64,
        source_archive_sha256=source_archive_sha256,
        source_manifest_sha256="4" * 64,
        runtime_identity_sha256="5" * 64,
    )
    first_attestation = governance._zip_provenance_attestation_payload(
        binding=binding,
        environment=first_environment,
    )
    second_attestation = governance._zip_provenance_attestation_payload(
        binding=binding,
        environment=second_environment,
    )
    assert first_attestation == second_attestation
    assert (
        first_attestation["zip_provenance_attestation_sha256"]
        == second_attestation["zip_provenance_attestation_sha256"]
    )
    assert governance.canonical_sha256(first_attestation) == governance.canonical_sha256(
        second_attestation
    )


def _tiny_audit(summary_digest: str) -> tuple[HiddenRegimeTraceAuditReport, dict[str, object]]:
    report = HiddenRegimeTraceAuditReport(
        valid=True,
        expected_steps=16_528,
        rows_checked=16_528,
        helper_transitions_checked=16_528,
        beneficiary_transitions_checked=16_528,
        world_transitions_checked=16_528,
        mismatches=(),
    )
    compact = {
        "trace_audit_report_sha256": canonical_sha256(
            governance._result_payload(report.to_dict(), "trace audit report")
        ),
        "valid": True,
        "expected_steps": 16_528,
        "rows_checked": 16_528,
        "helper_transitions_checked": 16_528,
        "beneficiary_transitions_checked": 16_528,
        "world_transitions_checked": 16_528,
        "commit_lineages_checked": 0,
        "recurrence_records_checked": 0,
        "retention_aggregate_fields_checked": 0,
        "summary_fields_checked": 0,
        "resource_fields_checked": 0,
        "mismatch_count": 0,
        "mismatches_sha256": canonical_sha256([]),
        "accepted_float32_contraction_count": 0,
        "accepted_float32_contractions_sha256": canonical_sha256([]),
        "unobserved_transition_fields": [],
        "evidence_boundary_sha256": hashlib.sha256(EVIDENCE_BOUNDARY.encode("utf-8")).hexdigest(),
        "lineage_oracle_valid": True,
        "lineage_oracle_mismatches_sha256": canonical_sha256([]),
        "audited_summary_sha256": summary_digest,
    }
    return report, compact


def _tiny_final_shard(
    *,
    snapshot: dict[str, object],
    genesis: dict[str, object],
    case_index: int,
    request_digest: str,
    config: HiddenRegimeDevelopmentConfig,
    summary: _TinySummary,
    resource: _TinyResource,
    trace: _TinyTrace,
) -> tuple[dict[str, object], HiddenRegimeTraceAuditReport]:
    case = build_hidden_regime_factorial_calibration_design().cases[case_index]
    started = next(
        item
        for item in cast(list[dict[str, object]], snapshot["started_records"])
        if item["case_index"] == case_index
    )
    completed = next(
        item
        for item in cast(list[dict[str, object]], snapshot["completed_records"])
        if item["case_index"] == case_index
    )
    attempt = governance.calibration_case_attempt_binding(snapshot, case_index)
    summary_payload = dataclasses.asdict(summary)
    resource_payload = dataclasses.asdict(resource)
    summary_digest = calibration_execution_summary_sha256(summary)
    resource_digest = calibration_execution_resource_sha256(resource)
    trace_digest = calibration_execution_primitive_trace_sha256(trace)
    audit_report, compact_audit = _tiny_audit(summary_digest)
    shard: dict[str, object] = {
        "schema": governance.CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "thresholds_frozen": False,
        "promotion_artifact": False,
        "case": case.to_payload(),
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "case_request_binding_sha256": request_digest,
        "readiness_binding": {
            "readiness_receipt_sha256": _READINESS_DIGEST,
            "source_archive_sha256": _SOURCE_ARCHIVE_DIGEST,
            "source_manifest_sha256": _SOURCE_MANIFEST_DIGEST,
            "runtime_identity_sha256": _RUNTIME_DIGEST,
            governance.READINESS_EXECUTION_GOVERNANCE_FIELD: (
                calibration_execution_genesis_receipt_binding(genesis)
            ),
        },
        "executed_steps": summary.num_steps,
        "managed_execution_attempt_count": attempt["managed_execution_attempt_count"],
        "unique_completed_outcome_count": 1,
        "configuration": governance._exact_json_value(config.to_dict(), label="configuration"),
        "configuration_sha256": calibration_execution_configuration_sha256(config),
        "summary": summary_payload,
        "summary_sha256": summary_digest,
        "resource": resource_payload,
        "resource_sha256": resource_digest,
        "primitive_trace": {"sha256": trace_digest, "persisted": False},
        "execution_record_binding": {
            "case_index": case_index,
            "genesis_sha256": snapshot["genesis_sha256"],
            "started_record_sha256": started["started_record_sha256"],
            "completed_record_sha256": completed["completed_record_sha256"],
            "summary_sha256": completed["summary_sha256"],
            "resource_sha256": completed["resource_sha256"],
            "primitive_trace_sha256": completed["primitive_trace_sha256"],
            "final_state_sha256": completed["final_state_sha256"],
            "outcome_sha256": completed["outcome_sha256"],
            "managed_execution_attempt_count": attempt["managed_execution_attempt_count"],
            "attempt_records_sha256": attempt["attempt_records_sha256"],
            "zip_provenance_binding_sha256": started["zip_provenance_binding_sha256"],
            "zip_provenance_attestation_sha256": started["zip_provenance_attestation_sha256"],
        },
        "audit": compact_audit,
    }
    shard["payload_sha256"] = canonical_sha256(shard)
    return shard, audit_report


def _patch_tiny_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    audit_report: HiddenRegimeTraceAuditReport,
) -> list[object]:
    audited: list[object] = []

    def audit(run: object) -> HiddenRegimeTraceAuditReport:
        audited.append(run)
        return audit_report

    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload: dict(payload),
    )
    monkeypatch.setattr(trace_audit, "audit_hidden_regime_run_result", audit)
    return audited


def test_pristine_genesis_and_inventory_are_deterministic_and_zero_entry(
    tmp_path: Path,
) -> None:
    assert governance.CALIBRATION_EXECUTION_GENESIS_SCHEMA.endswith(".v3")
    assert governance.CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA.endswith(".v3")
    assert governance.CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA.endswith(".v3")
    assert governance.CALIBRATION_EXECUTION_STARTED_SCHEMA.endswith(".v3")
    assert governance.CALIBRATION_EXECUTION_COMPLETED_SCHEMA.endswith(".v4")
    assert governance.CALIBRATION_EXECUTION_FINALIZED_SCHEMA.endswith(".v4")
    assert governance.CALIBRATION_EXECUTION_INVENTORY_SCHEMA.endswith(".v4")
    first = _genesis()
    second = _genesis()
    assert first == second
    assert require_valid_calibration_execution_genesis(first) == first
    assert first["genesis_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "genesis_sha256"}
    )
    ledger, _, _ = _initialize(tmp_path)
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["expected_case_count"] == N_MATCHED_CASES
    assert snapshot["started_case_indices"] == []
    assert snapshot["completed_case_indices"] == []
    assert snapshot["learner_interrupted_case_indices"] == []
    assert snapshot["post_audit_unfinalized_case_indices"] == []
    assert snapshot["finalized_case_indices"] == []
    assert snapshot["protected_started_record_count"] == 0
    assert snapshot["protected_completed_record_count"] == 0
    assert snapshot["pristine"] is True
    assert require_valid_calibration_execution_inventory(snapshot, ledger) == snapshot
    assert (ledger / "genesis.json").stat().st_mode & 0o777 == 0o444
    assert len(tuple((ledger / "cases").iterdir())) == N_MATCHED_CASES

    with pytest.raises(FileExistsError, match="overwrite"):
        initialize_calibration_execution_ledger(
            tmp_path / "ledgers",
            first,
            authorize_initialization=True,
        )
    changed = dict(first)
    changed["source_manifest_sha256"] = "9" * 64
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="digest"):
        require_valid_calibration_execution_genesis(changed)


@pytest.mark.parametrize(
    ("fault_stage", "published"),
    (
        ("staging_directory_created", False),
        ("cases_directory_created", False),
        ("case_directories_created", False),
        ("genesis_installed", False),
        ("staging_tree_synced", False),
        ("final_directory_published", True),
        ("publication_root_synced", True),
    ),
)
def test_atomic_whole_ledger_publication_survives_every_fault_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
    published: bool,
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    genesis = _genesis()
    digest = cast(str, genesis["genesis_sha256"])
    final_directory = root / digest
    stage_directory = root / f".{digest}{governance._LEDGER_STAGE_SUFFIX}"

    def inject_fault(stage: str, genesis_sha256: str) -> None:
        assert genesis_sha256 == digest
        if stage == fault_stage:
            raise RuntimeError(f"synthetic ledger crash at {stage}")

    monkeypatch.setattr(governance, "_ledger_initialization_checkpoint", inject_fault)
    with pytest.raises(RuntimeError, match="synthetic ledger crash"):
        initialize_calibration_execution_ledger(
            root,
            genesis,
            authorize_initialization=True,
        )
    assert final_directory.exists() is published
    assert not stage_directory.exists()

    monkeypatch.setattr(
        governance,
        "_ledger_initialization_checkpoint",
        lambda stage, genesis_sha256: None,
    )
    if published:
        snapshot = snapshot_calibration_execution_inventory(final_directory)
        assert snapshot["pristine"] is True
        assert snapshot["genesis_sha256"] == digest
        with pytest.raises(FileExistsError, match="overwrite"):
            initialize_calibration_execution_ledger(
                root,
                genesis,
                authorize_initialization=True,
            )
    else:
        recovered = initialize_calibration_execution_ledger(
            root,
            genesis,
            authorize_initialization=True,
        )
        assert recovered.directory == final_directory
        assert snapshot_calibration_execution_inventory(final_directory)["pristine"] is True


def test_atomic_whole_ledger_publication_recovers_an_exact_stale_partial_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    genesis = _genesis()
    digest = cast(str, genesis["genesis_sha256"])
    stage = root / f".{digest}{governance._LEDGER_STAGE_SUFFIX}"
    cases = stage / "cases"
    stage.mkdir(mode=0o700)
    cases.mkdir(mode=0o700)
    for case_index in range(17):
        (cases / f"case-{case_index:03d}").mkdir(mode=0o700)

    published = initialize_calibration_execution_ledger(
        root,
        genesis,
        authorize_initialization=True,
    )
    assert not stage.exists()
    assert published.directory == root / digest
    assert snapshot_calibration_execution_inventory(published.directory)["pristine"] is True


def test_atomic_whole_ledger_publication_rejects_an_unknown_staged_member(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    genesis = _genesis()
    digest = cast(str, genesis["genesis_sha256"])
    stage = root / f".{digest}{governance._LEDGER_STAGE_SUFFIX}"
    stage.mkdir(mode=0o700)
    unknown = stage / "untrusted"
    unknown.write_bytes(b"must-not-be-deleted")

    with pytest.raises(
        HiddenRegimeExecutionGovernanceError,
        match="unknown member",
    ):
        initialize_calibration_execution_ledger(
            root,
            genesis,
            authorize_initialization=True,
        )
    assert unknown.read_bytes() == b"must-not-be-deleted"
    assert not (root / digest).exists()


def test_atomic_whole_ledger_publication_rename_failure_cleans_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    genesis = _genesis()
    digest = cast(str, genesis["genesis_sha256"])
    real_rename = governance._rename_no_replace
    monkeypatch.setattr(
        governance,
        "_rename_no_replace",
        lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "synthetic rename failure")),
    )
    with pytest.raises(OSError, match="synthetic rename failure"):
        initialize_calibration_execution_ledger(
            root,
            genesis,
            authorize_initialization=True,
        )
    assert not (root / digest).exists()
    assert not (root / f".{digest}{governance._LEDGER_STAGE_SUFFIX}").exists()

    monkeypatch.setattr(governance, "_rename_no_replace", real_rename)
    published = initialize_calibration_execution_ledger(
        root,
        genesis,
        authorize_initialization=True,
    )
    assert snapshot_calibration_execution_inventory(published.directory)["pristine"] is True


def test_concurrent_whole_ledger_initializers_publish_exactly_one_complete_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    genesis = _genesis()

    def initialize() -> str:
        try:
            initialize_calibration_execution_ledger(
                root,
                genesis,
                authorize_initialization=True,
            )
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: initialize(), range(2)))
    assert sorted(outcomes) == ["exists", "published"]
    digest = cast(str, genesis["genesis_sha256"])
    assert snapshot_calibration_execution_inventory(root / digest)["pristine"] is True
    assert not (root / f".{digest}{governance._LEDGER_STAGE_SUFFIX}").exists()


@pytest.mark.parametrize(
    "record_name",
    (
        governance._STARTED_FILE,
        "replay-000001.json",
        governance._COMPLETED_FILE,
        governance._FINALIZED_FILE,
    ),
)
@pytest.mark.parametrize(
    ("fault_stage", "installed"),
    tuple((stage, False) for stage in governance._ATOMIC_INSTALL_PRECOMMIT_STAGES)
    + tuple((stage, True) for stage in governance._ATOMIC_INSTALL_POSTCOMMIT_STAGES),
)
def test_atomic_record_install_survives_every_fault_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
    fault_stage: str,
    installed: bool,
) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    raw = canonical_json_bytes({"record_name": record_name, "fault_stage": fault_stage})

    def inject_fault(stage: str, name: str) -> None:
        if stage == fault_stage and name == record_name:
            raise RuntimeError(f"synthetic crash at {stage}")

    monkeypatch.setattr(governance, "_atomic_install_checkpoint", inject_fault)
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(RuntimeError, match="synthetic crash"):
            governance._write_new_immutable(directory_fd, record_name, raw)
    finally:
        os.close(directory_fd)

    target = directory / record_name
    assert target.exists() is installed
    assert tuple(directory.iterdir()) == ((target,) if installed else ())
    if installed:
        metadata = target.lstat()
        assert target.read_bytes() == raw
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1


def test_atomic_record_install_partial_write_and_syscall_failures_never_expose_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    raw = canonical_json_bytes({"record": "started", "padding": "x" * 4096})
    real_write = os.write
    write_calls = 0

    def partial_then_fail(descriptor: int, value: object) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            view = memoryview(cast(Any, value))
            return real_write(descriptor, view[: len(view) // 2])
        raise OSError(errno.EIO, "synthetic interrupted write")

    monkeypatch.setattr(os, "write", partial_then_fail)
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError, match="synthetic interrupted write"):
            governance._write_new_immutable(directory_fd, governance._STARTED_FILE, raw)
    finally:
        os.close(directory_fd)
    assert tuple(directory.iterdir()) == ()

    monkeypatch.setattr(os, "write", real_write)
    monkeypatch.setattr(
        governance,
        "_link_anonymous_file_no_replace",
        lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "synthetic link failure")),
    )
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError, match="synthetic link failure"):
            governance._write_new_immutable(directory_fd, governance._STARTED_FILE, raw)
    finally:
        os.close(directory_fd)
    assert tuple(directory.iterdir()) == ()


def test_atomic_record_install_directory_sync_failure_leaves_complete_recoverable_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    raw = canonical_json_bytes({"record": "started"})
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    real_fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if descriptor == directory_fd:
            raise OSError(errno.EIO, "synthetic directory sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    try:
        with pytest.raises(OSError, match="synthetic directory sync failure"):
            governance._write_new_immutable(directory_fd, governance._STARTED_FILE, raw)
    finally:
        os.close(directory_fd)
    target = directory / governance._STARTED_FILE
    metadata = target.lstat()
    assert target.read_bytes() == raw
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1


def test_atomic_record_install_never_replaces_an_existing_target(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    target = directory / governance._STARTED_FILE
    original = b"original-complete-record"
    target.write_bytes(original)
    target.chmod(0o444)
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(FileExistsError):
            governance._write_new_immutable(
                directory_fd,
                governance._STARTED_FILE,
                b"replacement-record",
            )
    finally:
        os.close(directory_fd)
    assert target.read_bytes() == original
    assert target.stat().st_nlink == 1


@pytest.mark.parametrize("record_kind", ("started", "replay", "completed", "finalized"))
def test_concurrent_snapshot_never_observes_an_in_progress_record_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    target_name: str
    operation: Any
    if record_kind == "started":
        target_name = governance._STARTED_FILE
        operation = partial(
            begin_managed_hidden_regime_execution,
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=authorization,
        )
    else:
        ticket = begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=authorization,
        )
        assert ticket is not None
        if record_kind == "replay":
            replay, _, _, _ = _issue(
                monkeypatch,
                ledger,
                readiness,
                allow_exact_replay=True,
            )
            target_name = "replay-000001.json"
            operation = partial(
                begin_managed_hidden_regime_execution,
                condition=condition,
                seed_pair=seed_pair,
                config=config,
                authorization=replay,
            )
        else:
            if record_kind == "completed":
                target_name = governance._COMPLETED_FILE
                operation = partial(complete_managed_hidden_regime_execution, ticket, result)
            else:
                assert complete_managed_hidden_regime_execution(ticket, result) is not None
                snapshot = snapshot_calibration_execution_inventory(ledger)
                shard, audit_report = _tiny_final_shard(
                    snapshot=snapshot,
                    genesis=genesis,
                    case_index=0,
                    request_digest="5" * 64,
                    config=config,
                    summary=result.summary,
                    resource=result.resource,
                    trace=result.trace,
                )
                _patch_tiny_finalizer(monkeypatch, audit_report)
                target_name = governance._FINALIZED_FILE
                operation = partial(
                    governance.finalize_calibration_case_shard,
                    authorization,
                    ledger_directory=ledger,
                    shard_payload=shard,
                    run_result=result,
                )

    staged = threading.Event()
    release = threading.Event()

    def pause_before_install(stage: str, name: str) -> None:
        if stage == "stage_data_synced" and name == target_name:
            staged.set()
            assert release.wait(timeout=10)

    monkeypatch.setattr(governance, "_atomic_install_checkpoint", pause_before_install)
    outcomes: list[object] = []

    def invoke() -> None:
        try:
            outcomes.append(operation())
        except BaseException as error:  # pragma: no cover - asserted below
            outcomes.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert staged.wait(timeout=10)
    target = ledger / "cases" / "case-000" / target_name
    assert not target.exists()
    during = snapshot_calibration_execution_inventory(ledger)
    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(outcomes) == 1 and not isinstance(outcomes[0], BaseException)
    after = snapshot_calibration_execution_inventory(ledger)

    if record_kind == "started":
        assert during["started_case_indices"] == []
        assert after["started_case_indices"] == [0]
    elif record_kind == "replay":
        assert governance.calibration_case_attempt_binding(during, 0)[
            "managed_execution_attempt_count"
        ] == 1
        assert governance.calibration_case_attempt_binding(after, 0)[
            "managed_execution_attempt_count"
        ] == 2
    elif record_kind == "completed":
        assert during["completed_case_indices"] == []
        assert after["completed_case_indices"] == [0]
    else:
        assert during["finalized_case_indices"] == []
        assert after["finalized_case_indices"] == [0]


def test_frozen_case_configuration_digest_uses_exact_float_hex_payload() -> None:
    _, _, config = _case_inputs(0)
    encoded = governance._exact_json_value(config.to_dict(), label="configuration")
    exact_digest = calibration_execution_configuration_sha256(config)
    assert exact_digest == canonical_sha256(encoded)
    assert exact_digest != canonical_sha256(config.to_dict())


def test_exact_calibration_requires_sealed_authorization_and_consumes_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    assert authorization.payload["authorization_scope"] == PROCESS_LOCAL_AUTHORIZATION_SCOPE
    assert not hasattr(authorization, "to_json")

    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="sealed"):
        begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=None,
        )

    tampered_payload = dict(authorization.payload)
    tampered_payload["execution_mode"] = "exact_replay_after_interruption"
    tampered = dataclasses.replace(authorization, payload=tampered_payload)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="seal"):
        begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=tampered,
        )

    class _StopBeforeScanError(RuntimeError):
        pass

    def observe_consumption_before_scan(*_args: object, **_kwargs: object) -> object:
        snapshot = snapshot_calibration_execution_inventory(ledger)
        assert snapshot["started_case_indices"] == [0]
        assert snapshot["completed_case_indices"] == []
        raise _StopBeforeScanError

    import alberta_framework.evaluation.hidden_regime_signaling_development as development

    monkeypatch.setattr(development, "_scan_runner", observe_consumption_before_scan)
    with pytest.raises(_StopBeforeScanError):
        run_hidden_regime_condition(
            condition,  # type: ignore[arg-type]
            seed_pair=seed_pair,
            config=config,
            execution_authorization=authorization,
        )
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["started_case_indices"] == [0]
    assert snapshot["learner_interrupted_case_indices"] == [0]


def test_authorization_seal_is_rejected_by_a_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, _, _, _ = _issue(monkeypatch, ledger, readiness)
    encoded = base64.b64encode(canonical_json_bytes(authorization.payload)).decode("ascii")
    script = """
import base64
import hmac
import json
import sys
from alberta_framework.evaluation.hidden_regime_execution_governance import _seal

payload = json.loads(base64.b64decode(sys.argv[1], validate=True))
valid = hmac.compare_digest(
    sys.argv[2],
    _seal("calibration-execution-authorization-v3", payload),
)
raise SystemExit(0 if valid else 23)
"""
    completed = subprocess.run(
        (sys.executable, "-c", script, encoded, authorization.seal),
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 23


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_forked_child_rejects_every_inherited_process_local_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    completed = complete_managed_hidden_regime_execution(ticket, result)
    assert completed is not None
    case_directory = ledger / "cases" / "case-000"
    started = governance.require_valid_calibration_execution_started_record(
        ticket.started_record
    )
    attempt_rows = governance._case_attempt_rows(case_directory, started)
    components = governance._validated_result_component_digests(
        result,
        cast(dict[str, object], started["case_binding"]),
    )

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions occur in the parent
        os.close(read_fd)
        checks = {
            "provenance": lambda: governance._require_zip_provenance_capability(
                authorization.zip_provenance_capability
            ),
            "authorization": lambda: governance._require_authorization(
                authorization,
                condition=condition,
                seed_pair=seed_pair,
                config=config,
            ),
            "ticket": lambda: complete_managed_hidden_regime_execution(ticket, result),
            "completion": lambda: governance._require_completed_run_capability(
                result,
                ledger_directory=ledger,
                binding=cast(dict[str, object], started["case_binding"]),
                completed=completed,
                attempt_rows=attempt_rows,
                components=components,
            ),
        }
        outcomes: dict[str, str] = {}
        for label, check in checks.items():
            try:
                check()
            except HiddenRegimeExecutionGovernanceError:
                outcomes[label] = "rejected"
            except BaseException as exc:
                outcomes[label] = f"unexpected:{type(exc).__name__}"
            else:
                outcomes[label] = "accepted"
        raw = canonical_json_bytes(outcomes)
        try:
            os.write(write_fd, raw)
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    waited_pid, wait_status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid and os.waitstatus_to_exitcode(wait_status) == 0
    outcomes = json.loads(b"".join(chunks).decode("ascii"))
    assert outcomes == {
        "authorization": "rejected",
        "completion": "rejected",
        "provenance": "rejected",
        "ticket": "rejected",
    }


def test_direct_checkout_issuer_and_forged_zip_capabilities_fail_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    monkeypatch.setattr(governance, "_validate_readiness_bundle", lambda bundle, archive: bundle)
    archive = b"old-valid-readiness-archive"
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    archive_path.chmod(0o444)
    drifted_bundle = SimpleNamespace(
        payload={
            "body": {
                "runtime_identity": {
                    "python": {
                        "prefix": sys.prefix,
                        "exec_prefix": sys.exec_prefix,
                        "purelib": tmp_path.as_posix(),
                        "platlib": tmp_path.as_posix(),
                        "no_site_stdlib_search_paths": [],
                    }
                },
                "worker_execution": {
                    "no_site_flag": "-S",
                    "runtime_path_policy": "test raw dependency paths",
                },
            }
        },
        receipt_sha256=_READINESS_DIGEST,
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),
        source_manifest_sha256=_SOURCE_MANIFEST_DIGEST,
        runtime_identity_sha256=_RUNTIME_DIGEST,
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="working directory"):
        governance.attest_calibration_zip_provenance(
            readiness_bundle=drifted_bundle,
            readiness_source_archive=archive,
            source_archive_path=archive_path,
        )

    condition, seed_pair, config = _case_inputs(0)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="ZIP provenance capability"):
        issue_calibration_execution_authorization(
            ledger_directory=ledger,
            readiness_bundle=readiness,
            readiness_source_archive=b"test-bound-archive",
            zip_provenance_capability=None,
            case_index=0,
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            case_request_binding_sha256="5" * 64,
            attempt_request_payload_sha256="5" * 64,
            explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        )
    assert snapshot_calibration_execution_inventory(ledger)["pristine"] is True

    capability = _fake_provenance(readiness)
    forged = dataclasses.replace(capability)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="forged, copied"):
        issue_calibration_execution_authorization(
            ledger_directory=ledger,
            readiness_bundle=readiness,
            readiness_source_archive=b"test-bound-archive",
            zip_provenance_capability=forged,
            case_index=0,
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            case_request_binding_sha256="5" * 64,
            attempt_request_payload_sha256="5" * 64,
            explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        )
    with pytest.raises(TypeError, match="nonserializable"):
        pickle.dumps(capability)


def test_zip_provenance_requires_command_line_bytecode_and_pycache_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = b"archive-before-module-validation"
    archive_path = tmp_path / "readiness" / "source.zip"
    archive_path.parent.mkdir()
    archive_path.write_bytes(archive)
    archive_path.chmod(0o444)
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    monkeypatch.setattr(sys, "path", [archive_path.as_posix()])
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="bytecode writes"):
        governance._verify_exact_zip_worker_environment(
            source_archive_path=archive_path,
            source_archive=archive,
            expected_source_archive_sha256=hashlib.sha256(archive).hexdigest(),
            runtime_python_binding={
                "prefix": sys.prefix,
                "exec_prefix": sys.exec_prefix,
                "purelib": tmp_path.as_posix(),
                "platlib": tmp_path.as_posix(),
                "no_site_stdlib_search_paths": [],
            },
            runtime_path_policy="test raw dependency paths",
        )

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="pycache prefix is missing"):
        governance._verify_exact_zip_worker_environment(
            source_archive_path=archive_path,
            source_archive=archive,
            expected_source_archive_sha256=hashlib.sha256(archive).hexdigest(),
            runtime_python_binding={
                "prefix": sys.prefix,
                "exec_prefix": sys.exec_prefix,
                "purelib": tmp_path.as_posix(),
                "platlib": tmp_path.as_posix(),
                "no_site_stdlib_search_paths": [],
            },
            runtime_path_policy="test raw dependency paths",
        )


def test_replay_attempts_and_post_audit_finalization_are_immutable_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    first_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=first,
    )
    assert first_ticket is not None
    interrupted = snapshot_calibration_execution_inventory(ledger)
    assert interrupted["learner_interrupted_case_indices"] == [0]
    first_attempt = governance.calibration_case_attempt_binding(interrupted, 0)
    assert first_attempt["managed_execution_attempt_count"] == 1

    replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        allow_exact_replay=True,
    )
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None
    assert replay_ticket.attempt_index == 1
    replay_inventory = snapshot_calibration_execution_inventory(ledger)
    replay_attempts = governance.calibration_case_attempt_binding(replay_inventory, 0)
    assert replay_attempts["managed_execution_attempt_count"] == 2
    assert len(replay_attempts["attempts"]) == 2  # type: ignore[arg-type]

    summary = _TinySummary(config.num_steps)
    resource = _TinyResource()
    trace = _TinyTrace(
        step=np.asarray([0, 1], dtype=np.int32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    result = _TinyResult(condition, seed_pair, config, summary, resource, trace)
    assert complete_managed_hidden_regime_execution(replay_ticket, result) is not None
    completed = snapshot_calibration_execution_inventory(ledger)
    assert completed["post_audit_unfinalized_case_indices"] == [0]
    shard, audit_report = _tiny_final_shard(
        snapshot=completed,
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=summary,
        resource=resource,
        trace=trace,
    )
    audited = _patch_tiny_finalizer(monkeypatch, audit_report)
    finalized = governance.finalize_calibration_case_shard(
        replay,
        ledger_directory=ledger,
        shard_payload=shard,
        run_result=result,
    )
    assert audited == [result]
    assert finalized["managed_execution_attempt_count"] == 2
    assert governance.load_finalized_calibration_case_shard(ledger, 0) == shard
    final_inventory = snapshot_calibration_execution_inventory(ledger)
    assert final_inventory["finalized_case_indices"] == [0]
    assert final_inventory["post_audit_unfinalized_case_indices"] == []
    finalized_path = ledger / "cases" / "case-000" / "finalized.json"
    assert finalized_path.stat().st_mode & 0o777 == 0o444

    stored_report_tamper = deepcopy(finalized)
    stored_report = cast(dict[str, object], stored_report_tamper["trace_audit_report"])
    stored_report["accepted_float32_contractions"] = ["forged contraction"]
    stored_report_tamper.pop("finalized_record_sha256")
    stored_report_tamper["finalized_record_sha256"] = canonical_sha256(stored_report_tamper)
    case_directory = ledger / "cases" / "case-000"
    started_record = json.loads((case_directory / "started.json").read_text(encoding="ascii"))
    completed_record = json.loads((case_directory / "completed.json").read_text(encoding="ascii"))
    attempt_rows = governance._case_attempt_rows(case_directory, started_record)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="full trace-audit report"):
        governance.require_valid_calibration_execution_finalized_record(
            stored_report_tamper,
            expected_started=started_record,
            expected_completed=completed_record,
            expected_attempt_rows=attempt_rows,
        )

    input_binding_tamper = deepcopy(finalized)
    input_binding = cast(
        dict[str, object],
        input_binding_tamper["trace_audit_input_binding"],
    )
    input_binding["final_state_sha256"] = "0" * 64
    input_binding.pop("trace_audit_input_binding_sha256")
    input_binding["trace_audit_input_binding_sha256"] = canonical_sha256(input_binding)
    input_binding_tamper["trace_audit_input_binding_sha256"] = input_binding[
        "trace_audit_input_binding_sha256"
    ]
    input_binding_tamper.pop("finalized_record_sha256")
    input_binding_tamper["finalized_record_sha256"] = canonical_sha256(input_binding_tamper)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="input binding differs"):
        governance.require_valid_calibration_execution_finalized_record(
            input_binding_tamper,
            expected_started=started_record,
            expected_completed=completed_record,
            expected_attempt_rows=attempt_rows,
        )

    assert (
        governance.finalize_calibration_case_shard(
            replay,
            ledger_directory=ledger,
            shard_payload=shard,
            run_result=result,
        )
        == finalized
    )
    changed = deepcopy(shard)
    changed["unexpected_tamper"] = True
    changed.pop("payload_sha256")
    changed["payload_sha256"] = canonical_sha256(changed)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="replay differs"):
        governance.finalize_calibration_case_shard(
            replay,
            ledger_directory=ledger,
            shard_payload=changed,
            run_result=result,
        )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="cannot be replayed"):
        _issue(monkeypatch, ledger, readiness, allow_exact_replay=True)

    (ledger / "cases" / "case-001" / "unknown.json").write_text("{}", encoding="ascii")
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="unknown member"):
        snapshot_calibration_execution_inventory(ledger)


def test_finalizer_rejects_compact_audit_not_equal_to_full_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None
    summary = _TinySummary(config.num_steps)
    resource = _TinyResource()
    trace = _TinyTrace(
        step=np.asarray([0, 1], dtype=np.int32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    result = _TinyResult(condition, seed_pair, config, summary, resource, trace)
    assert complete_managed_hidden_regime_execution(ticket, result) is not None
    shard, audit_report = _tiny_final_shard(
        snapshot=snapshot_calibration_execution_inventory(ledger),
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=summary,
        resource=resource,
        trace=trace,
    )
    audit = shard["audit"]
    assert isinstance(audit, dict)
    audit["accepted_float32_contraction_count"] = 1
    shard.pop("payload_sha256")
    shard["payload_sha256"] = canonical_sha256(shard)
    audited = _patch_tiny_finalizer(monkeypatch, audit_report)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="contraction count"):
        governance.finalize_calibration_case_shard(
            authorization,
            ledger_directory=ledger,
            shard_payload=shard,
            run_result=result,
        )
    assert audited == [result]
    inventory = snapshot_calibration_execution_inventory(ledger)
    assert inventory["post_audit_unfinalized_case_indices"] == [0]
    assert inventory["finalized_case_indices"] == []


def test_finalizer_requires_the_exact_completed_run_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    assert complete_managed_hidden_regime_execution(ticket, result) is not None
    shard, audit_report = _tiny_final_shard(
        snapshot=snapshot_calibration_execution_inventory(ledger),
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=result.summary,
        resource=result.resource,
        trace=result.trace,
    )
    audited = _patch_tiny_finalizer(monkeypatch, audit_report)
    equivalent_but_uncompleted_object = dataclasses.replace(result)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="exact process-local"):
        governance.finalize_calibration_case_shard(
            authorization,
            ledger_directory=ledger,
            shard_payload=shard,
            run_result=equivalent_but_uncompleted_object,
        )
    assert audited == []
    assert not (ledger / "cases/case-000/finalized.json").exists()


def test_finalizer_rejects_run_mutation_during_trace_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    assert complete_managed_hidden_regime_execution(ticket, result) is not None
    shard, audit_report = _tiny_final_shard(
        snapshot=snapshot_calibration_execution_inventory(ledger),
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=result.summary,
        resource=result.resource,
        trace=result.trace,
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload: dict(payload),
    )
    audit_started = threading.Event()
    release_audit = threading.Event()

    def paused_audit(run: object) -> HiddenRegimeTraceAuditReport:
        assert run is result
        audit_started.set()
        assert release_audit.wait(timeout=10)
        return audit_report

    monkeypatch.setattr(trace_audit, "audit_hidden_regime_run_result", paused_audit)
    finalizer_result: list[object] = []

    def finalize() -> None:
        try:
            finalizer_result.append(
                governance.finalize_calibration_case_shard(
                    authorization,
                    ledger_directory=ledger,
                    shard_payload=shard,
                    run_result=result,
                )
            )
        except BaseException as exc:
            finalizer_result.append(exc)

    finalizer_thread = threading.Thread(target=finalize)
    finalizer_thread.start()
    assert audit_started.wait(timeout=10)
    result.trace.reward[0] = np.float32(1.0)
    result.final_state[0][0] = np.int32(99)
    release_audit.set()
    finalizer_thread.join(timeout=10)
    assert not finalizer_thread.is_alive()
    assert len(finalizer_result) == 1
    failure = finalizer_result[0]
    assert isinstance(failure, HiddenRegimeExecutionGovernanceError)
    assert "changed during trace audit" in str(failure)
    assert not (ledger / "cases/case-000/finalized.json").exists()


def test_latest_replay_must_complete_before_prior_outcome_can_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    first_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=first,
    )
    assert first_ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    assert complete_managed_hidden_regime_execution(first_ticket, result) is not None
    replay, _, _, _ = _issue(monkeypatch, ledger, readiness, allow_exact_replay=True)
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None
    snapshot = snapshot_calibration_execution_inventory(ledger)
    shard, audit_report = _tiny_final_shard(
        snapshot=snapshot,
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=result.summary,
        resource=result.resource,
        trace=result.trace,
    )
    audited = _patch_tiny_finalizer(monkeypatch, audit_report)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="capability is stale"):
        governance.finalize_calibration_case_shard(
            replay,
            ledger_directory=ledger,
            shard_payload=shard,
            run_result=result,
        )
    assert audited == []
    assert not (ledger / "cases/case-000/finalized.json").exists()


def test_stale_ticket_cannot_complete_after_a_new_replay_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _genesis_payload, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    first_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=first,
    )
    assert first_ticket is not None
    replay, _, _, _ = _issue(monkeypatch, ledger, readiness, allow_exact_replay=True)
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="not the latest"):
        complete_managed_hidden_regime_execution(first_ticket, result)
    assert not (ledger / "cases/case-000/completed.json").exists()
    assert complete_managed_hidden_regime_execution(replay_ticket, result) is not None


def test_finalization_serializes_against_a_preissued_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=first,
    )
    assert ticket is not None
    result = _TinyResult(
        condition,
        seed_pair,
        config,
        _TinySummary(config.num_steps),
        _TinyResource(),
        _TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    assert complete_managed_hidden_regime_execution(ticket, result) is not None
    replay, _, _, _ = _issue(monkeypatch, ledger, readiness, allow_exact_replay=True)
    shard, audit_report = _tiny_final_shard(
        snapshot=snapshot_calibration_execution_inventory(ledger),
        genesis=genesis,
        case_index=0,
        request_digest="5" * 64,
        config=config,
        summary=result.summary,
        resource=result.resource,
        trace=result.trace,
    )
    _patch_tiny_finalizer(monkeypatch, audit_report)

    finalizer_paused = threading.Event()
    release_finalizer = threading.Event()
    replay_requested_lock = threading.Event()
    real_write = governance._write_new_immutable
    real_critical_section = governance._case_mutation_critical_section

    def pausing_write(directory_fd: int, name: str, raw: bytes) -> None:
        if name == governance._FINALIZED_FILE:
            finalizer_paused.set()
            assert release_finalizer.wait(timeout=10)
        real_write(directory_fd, name, raw)

    @contextmanager
    def observed_critical_section(case_directory: Path, case_index: int) -> Any:
        if threading.current_thread().name == "replay-thread":
            replay_requested_lock.set()
        with real_critical_section(case_directory, case_index) as descriptor:
            yield descriptor

    monkeypatch.setattr(governance, "_write_new_immutable", pausing_write)
    monkeypatch.setattr(
        governance,
        "_case_mutation_critical_section",
        observed_critical_section,
    )
    finalizer_result: list[object] = []
    replay_result: list[object] = []

    def finalize() -> None:
        try:
            finalizer_result.append(
                governance.finalize_calibration_case_shard(
                    first,
                    ledger_directory=ledger,
                    shard_payload=shard,
                    run_result=result,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            finalizer_result.append(exc)

    def begin_replay() -> None:
        try:
            replay_result.append(
                begin_managed_hidden_regime_execution(
                    condition=condition,
                    seed_pair=seed_pair,
                    config=config,
                    authorization=replay,
                )
            )
        except BaseException as exc:
            replay_result.append(exc)

    finalizer_thread = threading.Thread(target=finalize, name="finalizer-thread")
    replay_thread = threading.Thread(target=begin_replay, name="replay-thread")
    finalizer_thread.start()
    assert finalizer_paused.wait(timeout=10)
    replay_thread.start()
    assert replay_requested_lock.wait(timeout=10)
    assert replay_thread.is_alive()
    release_finalizer.set()
    finalizer_thread.join(timeout=10)
    replay_thread.join(timeout=10)
    assert not finalizer_thread.is_alive()
    assert not replay_thread.is_alive()
    assert len(finalizer_result) == 1 and isinstance(finalizer_result[0], dict)
    assert len(replay_result) == 1
    assert isinstance(replay_result[0], HiddenRegimeExecutionGovernanceError)
    assert "cannot be replayed" in str(replay_result[0])
    assert not (ledger / "cases/case-000/replay-000001.json").exists()


def test_protected_manifests_and_tail_extensions_fail_closed() -> None:
    _, seed_pair, ordinary_config = _case_inputs(0)
    for manifest in HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values():
        worlds = [manifest.to_world_config(repeat_schedule=False)]
        worlds.extend(
            HiddenRegimeWorldConfig(
                segment_lengths=(
                    *manifest.segment_lengths[:-1],
                    manifest.segment_lengths[-1] + extension,
                ),
                segment_regimes=manifest.segment_regimes,
                regime_permutations=manifest.regime_permutations,
                repeat_schedule=False,
            )
            for extension in range(1, 16)
        )
        for world in worlds:
            classification = classify_hidden_regime_world(world)
            assert classification.sensitivity == "protected"
            config = dataclasses.replace(ordinary_config, world=world)
            with pytest.raises(HiddenRegimeProtectedExecutionError, match="no learner-execution"):
                begin_managed_hidden_regime_execution(
                    condition=SELECTIVE_FULL,
                    seed_pair=seed_pair,
                    config=config,
                    authorization=None,
                )

    for manifest in HIDDEN_REGIME_CALIBRATION_MANIFESTS.values():
        classification = classify_hidden_regime_world(
            manifest.to_world_config(repeat_schedule=False)
        )
        assert classification.sensitivity == "calibration"
        assert classification.manifest_name == manifest.name

    changed_world = dataclasses.replace(
        ordinary_config.world,
        segment_lengths=(
            ordinary_config.world.segment_lengths[0] + 1,
            *ordinary_config.world.segment_lengths[1:],
        ),
    )
    assert classify_hidden_regime_world(changed_world).sensitivity == "ordinary"


def test_concurrent_first_execution_has_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    second, _, _, _ = _issue(monkeypatch, ledger, readiness)

    def begin(authorization: CalibrationExecutionAuthorization) -> str:
        try:
            ticket = begin_managed_hidden_regime_execution(
                condition=condition,
                seed_pair=seed_pair,
                config=config,
                authorization=authorization,
            )
        except HiddenRegimeCaseConsumedError:
            return "consumed"
        assert ticket is not None
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(begin, (first, second)))
    assert sorted(outcomes) == ["consumed", "won"]
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["started_record_count"] == 1
    assert snapshot["completed_record_count"] == 0


def test_crash_consumption_allows_only_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None

    with pytest.raises(HiddenRegimeCaseConsumedError, match="exact replay"):
        _issue(monkeypatch, ledger, readiness)
    different_pair = dataclasses.replace(seed_pair, learner_seed=seed_pair.learner_seed ^ 1)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="seed pair"):
        issue_calibration_execution_authorization(
            ledger_directory=ledger,
            readiness_bundle=readiness,
            readiness_source_archive=b"test-bound-archive",
            zip_provenance_capability=_fake_provenance(readiness),
            case_index=0,
            condition=condition,
            seed_pair=different_pair,
            config=config,
            case_request_binding_sha256="5" * 64,
            attempt_request_payload_sha256="5" * 64,
            explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
            allow_exact_replay=True,
        )

    replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        allow_exact_replay=True,
    )
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None
    assert replay_ticket.execution_mode == "exact_replay_after_interruption"

    result = _TinyResult(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        summary=_TinySummary(config.num_steps),
        resource=_TinyResource(),
        trace=_TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    completed = complete_managed_hidden_regime_execution(replay_ticket, result)
    assert completed is not None
    assert completed["execution_state"] == "learner_execution_completed"
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["completed_case_indices"] == [0]
    assert snapshot["learner_interrupted_case_indices"] == []
    assert snapshot["post_audit_unfinalized_case_indices"] == [0]

    completed_replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        allow_exact_replay=True,
    )
    completed_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=completed_replay,
    )
    assert completed_ticket is not None
    assert completed_ticket.execution_mode == "exact_replay_after_completion"
    assert complete_managed_hidden_regime_execution(completed_ticket, result) == completed

    changed_result = dataclasses.replace(
        result,
        trace=_TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="outcome differs"):
        complete_managed_hidden_regime_execution(completed_ticket, changed_result)


def test_runner_request_consent_can_change_on_exact_crash_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay consent is attempt-local and cannot change the scientific case identity."""

    ledger, genesis, readiness = _initialize(tmp_path)
    case = calibration._design().cases[0]
    readiness_binding = {
        "readiness_receipt_sha256": _READINESS_DIGEST,
        "source_archive_sha256": _SOURCE_ARCHIVE_DIGEST,
        "source_manifest_sha256": _SOURCE_MANIFEST_DIGEST,
        "runtime_identity_sha256": _RUNTIME_DIGEST,
        "dependency_locks": [],
        "scipy_version": calibration.scipy_version,
        governance.READINESS_EXECUTION_GOVERNANCE_FIELD: (
            calibration_execution_genesis_receipt_binding(genesis)
        ),
    }
    common = {
        "case_index": 0,
        "case_binding": case.to_payload(),
        "readiness_binding": readiness_binding,
        "managed_ledger_genesis_sha256": genesis["genesis_sha256"],
        "explicit_acknowledgement": calibration.EXECUTION_ACKNOWLEDGEMENT,
    }
    first_request = calibration.CalibrationCaseRequest(
        **common,
        allow_exact_replay=False,
    )
    replay_request = calibration.CalibrationCaseRequest(
        **common,
        allow_exact_replay=True,
    )
    case_request_binding = calibration.calibration_case_request_binding_sha256(first_request)
    assert case_request_binding == calibration.calibration_case_request_binding_sha256(
        replay_request
    )
    first_request_digest = cast(str, first_request.to_payload()["payload_sha256"])
    replay_request_digest = cast(str, replay_request.to_payload()["payload_sha256"])
    assert first_request_digest != replay_request_digest

    first, condition, seed_pair, config = _issue(
        monkeypatch,
        ledger,
        readiness,
        case_request_binding_sha256=case_request_binding,
        attempt_request_payload_sha256=first_request_digest,
        allow_exact_replay=False,
    )
    first_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=first,
    )
    assert first_ticket is not None
    # Simulate interruption after immutable consumption but before completion/finalization.
    replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        case_request_binding_sha256=case_request_binding,
        attempt_request_payload_sha256=replay_request_digest,
        allow_exact_replay=True,
    )
    for field, replacement in (
        ("attempt_request_payload_sha256", first_request_digest),
        ("exact_replay_consent", False),
    ):
        tampered_payload = dict(replay.payload)
        tampered_payload[field] = replacement
        with pytest.raises(HiddenRegimeExecutionGovernanceError, match="seal"):
            begin_managed_hidden_regime_execution(
                condition=condition,
                seed_pair=seed_pair,
                config=config,
                authorization=dataclasses.replace(replay, payload=tampered_payload),
            )
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None and replay_ticket.attempt_index == 1

    inventory = snapshot_calibration_execution_inventory(ledger)
    attempts = governance.calibration_case_attempt_binding(inventory, 0)
    rows = cast(list[dict[str, object]], attempts["attempts"])
    assert [row["attempt_request_payload_sha256"] for row in rows] == [
        first_request_digest,
        replay_request_digest,
    ]
    assert [row["exact_replay_consent"] for row in rows] == [False, True]
    started = cast(list[dict[str, object]], inventory["started_records"])[0]
    assert started["case_request_binding_sha256"] == case_request_binding

    summary = _TinySummary(config.num_steps)
    resource = _TinyResource()
    trace = _TinyTrace(
        step=np.asarray([0, 1], dtype=np.int32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    result = _TinyResult(condition, seed_pair, config, summary, resource, trace)
    assert complete_managed_hidden_regime_execution(replay_ticket, result) is not None
    completed_inventory = snapshot_calibration_execution_inventory(ledger)
    shard, audit_report = _tiny_final_shard(
        snapshot=completed_inventory,
        genesis=genesis,
        case_index=0,
        request_digest=case_request_binding,
        config=config,
        summary=summary,
        resource=resource,
        trace=trace,
    )
    _patch_tiny_finalizer(monkeypatch, audit_report)
    governance.finalize_calibration_case_shard(
        replay,
        ledger_directory=ledger,
        shard_payload=shard,
        run_result=result,
    )
    assert governance.load_finalized_calibration_case_shard(ledger, 0) == shard


def test_completed_snapshot_joins_request_summary_resource_and_trace_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    governance_binding = calibration_execution_genesis_receipt_binding(genesis)
    readiness_binding = {
        "readiness_receipt_sha256": _READINESS_DIGEST,
        "source_archive_sha256": _SOURCE_ARCHIVE_DIGEST,
        "source_manifest_sha256": _SOURCE_MANIFEST_DIGEST,
        "runtime_identity_sha256": _RUNTIME_DIGEST,
        governance.READINESS_EXECUTION_GOVERNANCE_FIELD: governance_binding,
    }
    case_sources: dict[
        int,
        tuple[
            str,
            HiddenRegimeDevelopmentConfig,
            str,
            CalibrationExecutionAuthorization,
            _TinyResult,
        ],
    ] = {}
    trace = _TinyTrace(
        step=np.asarray([0, 1], dtype=np.int32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    summary = _TinySummary(16_528)
    resource = _TinyResource()
    for case_index in range(N_MATCHED_CASES):
        request_digest = canonical_sha256({"case_index": case_index, "request": "test"})
        authorization, condition, seed_pair, config = _issue(
            monkeypatch,
            ledger,
            readiness,
            case_index=case_index,
            case_request_binding_sha256=request_digest,
        )
        ticket = begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=authorization,
        )
        assert ticket is not None
        result = _TinyResult(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            summary=summary,
            resource=resource,
            trace=trace,
        )
        assert complete_managed_hidden_regime_execution(ticket, result) is not None
        case_sources[case_index] = (request_digest, config, condition, authorization, result)

    snapshot = snapshot_calibration_execution_inventory(ledger)
    completed_by_case = {
        item["case_index"]: item
        for item in cast(list[dict[str, object]], snapshot["completed_records"])
    }
    started_by_case = {
        item["case_index"]: item
        for item in cast(list[dict[str, object]], snapshot["started_records"])
    }
    design = build_hidden_regime_factorial_calibration_design()
    summary_payload = dataclasses.asdict(summary)
    resource_payload = dataclasses.asdict(resource)
    summary_digest = calibration_execution_summary_sha256(summary)
    resource_digest = calibration_execution_resource_sha256(resource)
    trace_digest = calibration_execution_primitive_trace_sha256(trace)
    audit_report = HiddenRegimeTraceAuditReport(
        valid=True,
        expected_steps=16_528,
        rows_checked=16_528,
        helper_transitions_checked=16_528,
        beneficiary_transitions_checked=16_528,
        world_transitions_checked=16_528,
        mismatches=(),
    )
    audit_report_digest = canonical_sha256(
        governance._result_payload(audit_report.to_dict(), "trace audit report")
    )
    audited = _patch_tiny_finalizer(monkeypatch, audit_report)
    shards: dict[int, dict[str, object]] = {}
    for case_index, case in enumerate(design.cases):
        request_digest, config, condition, authorization, result = case_sources[case_index]
        assert condition == case.condition
        started = started_by_case[case_index]
        completed = completed_by_case[case_index]
        attempt_binding = governance.calibration_case_attempt_binding(snapshot, case_index)
        shards[case_index] = {
            "schema": governance.CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "case": case.to_payload(),
            "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
            "case_request_binding_sha256": request_digest,
            "configuration": governance._exact_json_value(
                config.to_dict(),
                label="configuration",
            ),
            "configuration_sha256": calibration_execution_configuration_sha256(config),
            "readiness_binding": readiness_binding,
            "executed_steps": summary.num_steps,
            "managed_execution_attempt_count": 1,
            "unique_completed_outcome_count": 1,
            "summary": summary_payload,
            "summary_sha256": summary_digest,
            "resource": resource_payload,
            "resource_sha256": resource_digest,
            "primitive_trace": {"sha256": trace_digest, "persisted": False},
            "execution_record_binding": {
                "case_index": case_index,
                "genesis_sha256": snapshot["genesis_sha256"],
                "started_record_sha256": started["started_record_sha256"],
                "completed_record_sha256": completed["completed_record_sha256"],
                "summary_sha256": completed["summary_sha256"],
                "resource_sha256": completed["resource_sha256"],
                "primitive_trace_sha256": completed["primitive_trace_sha256"],
                "final_state_sha256": completed["final_state_sha256"],
                "outcome_sha256": completed["outcome_sha256"],
                "managed_execution_attempt_count": 1,
                "attempt_records_sha256": attempt_binding["attempt_records_sha256"],
                "zip_provenance_binding_sha256": started["zip_provenance_binding_sha256"],
                "zip_provenance_attestation_sha256": started["zip_provenance_attestation_sha256"],
            },
            "audit": {
                "trace_audit_report_sha256": audit_report_digest,
                "valid": True,
                "expected_steps": 16_528,
                "rows_checked": 16_528,
                "helper_transitions_checked": 16_528,
                "beneficiary_transitions_checked": 16_528,
                "world_transitions_checked": 16_528,
                "commit_lineages_checked": 0,
                "recurrence_records_checked": 0,
                "retention_aggregate_fields_checked": 0,
                "summary_fields_checked": 0,
                "resource_fields_checked": 0,
                "mismatch_count": 0,
                "mismatches_sha256": canonical_sha256([]),
                "accepted_float32_contraction_count": 0,
                "accepted_float32_contractions_sha256": canonical_sha256([]),
                "unobserved_transition_fields": [],
                "evidence_boundary_sha256": hashlib.sha256(
                    EVIDENCE_BOUNDARY.encode("utf-8")
                ).hexdigest(),
                "lineage_oracle_valid": True,
                "lineage_oracle_mismatches_sha256": canonical_sha256([]),
                "audited_summary_sha256": summary_digest,
            },
        }
        shards[case_index]["payload_sha256"] = canonical_sha256(shards[case_index])
        governance.finalize_calibration_case_shard(
            authorization,
            ledger_directory=ledger,
            shard_payload=shards[case_index],
            run_result=result,
        )
    assert audited == [case_sources[index][4] for index in range(N_MATCHED_CASES)]
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert validate_completed_calibration_ledger_snapshot(snapshot, shards) == snapshot

    summary_tamper = deepcopy(shards)
    summary_tamper[0]["summary"] = {"num_steps": 16_527}
    summary_tamper[0]["summary_sha256"] = canonical_sha256(summary_tamper[0]["summary"])
    summary_body = dict(summary_tamper[0])
    summary_body.pop("payload_sha256")
    summary_tamper[0]["payload_sha256"] = canonical_sha256(summary_body)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="immutable completion"):
        validate_completed_calibration_ledger_snapshot(snapshot, summary_tamper)

    resource_tamper = deepcopy(shards)
    resource_tamper[0]["resource"] = {"state_bytes": 551}
    resource_tamper[0]["resource_sha256"] = canonical_sha256(resource_tamper[0]["resource"])
    resource_body = dict(resource_tamper[0])
    resource_body.pop("payload_sha256")
    resource_tamper[0]["payload_sha256"] = canonical_sha256(resource_body)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="immutable completion"):
        validate_completed_calibration_ledger_snapshot(snapshot, resource_tamper)

    audit_fields = tuple(cast(dict[str, object], shards[0]["audit"]))
    for audit_field in audit_fields:
        audit_tamper = deepcopy(shards)
        audit_payload = cast(dict[str, object], audit_tamper[0]["audit"])
        original = audit_payload[audit_field]
        if type(original) is bool:
            audit_payload[audit_field] = not original
        elif type(original) is int:
            audit_payload[audit_field] = original + 1
        elif type(original) is str:
            audit_payload[audit_field] = ("0" if original[0] != "0" else "1") + original[1:]
        elif type(original) is list:
            audit_payload[audit_field] = ["tampered"]
        else:
            raise AssertionError(f"unhandled compact audit leaf: {audit_field}")
        audit_tamper[0].pop("payload_sha256")
        audit_tamper[0]["payload_sha256"] = canonical_sha256(audit_tamper[0])
        with pytest.raises(
            HiddenRegimeExecutionGovernanceError,
            match="finalization differs from exact final shard payload",
        ):
            validate_completed_calibration_ledger_snapshot(snapshot, audit_tamper)


def test_ledger_tamper_symlink_and_started_record_mutation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "publication-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="symlink"):
        initialize_calibration_execution_ledger(
            symlink,
            _genesis(),
            authorize_initialization=True,
        )

    real_ancestor = tmp_path / "real-ancestor"
    nested_root = real_ancestor / "nested" / "publication"
    nested_root.mkdir(parents=True)
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="traverses a symlink"):
        initialize_calibration_execution_ledger(
            linked_ancestor / "nested" / "publication",
            _genesis(),
            authorize_initialization=True,
        )

    ledger, _, readiness = _initialize(tmp_path / "separate")
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    assert (
        begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=authorization,
        )
        is not None
    )
    started_path = ledger / "cases" / "case-000" / "started.json"
    started = json.loads(started_path.read_text(encoding="ascii"))
    started["execution_state"] = "unconsumed"
    started["started_record_sha256"] = canonical_sha256(
        {key: value for key, value in started.items() if key != "started_record_sha256"}
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="state"):
        require_valid_calibration_execution_started_record(started)

    os.chmod(started_path, 0o644)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="mode"):
        snapshot_calibration_execution_inventory(ledger)
