"""Fail-closed tests for hidden-regime calibration execution and aggregation."""

from __future__ import annotations

import base64
import copy
import fcntl
import math
import os
import stat
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from scipy.stats import t as student_t

import alberta_framework.evaluation.hidden_regime_factorial_calibration as calibration
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    CALIBRATION_EXECUTION_STARTED_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    EstimandContract,
    MetricContract,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
)

pytestmark = pytest.mark.unit


def _record(
    identity: tuple[int, int, int, int],
    *,
    qualified: bool = True,
    selected: bool = True,
    errors: int = 0,
) -> dict[str, object]:
    segment, regime, occurrence, raw_occurrence = identity
    selected = selected and qualified
    probe = {
        "acquisition_qualified": True,
        "synchronized_generation_survives": True,
        "entry_composed_greedy_accuracy": 1.0,
    }
    dormant = {
        "composed_greedy_accuracy": 1.0,
        "zero_helper_accuracy": 1.0 / 3.0,
        "zero_beneficiary_accuracy": 1.0 / 3.0,
        "role_swapped_accuracy": 1.0 / 3.0,
    }
    return {
        "segment_index": segment,
        "regime_id": regime,
        "occurrence_index": occurrence,
        "raw_segment_occurrence_index": raw_occurrence,
        "lineage_retention_applicable": qualified,
        "acquisition_coverage_failure": not qualified,
        "first_world_window_complete": True,
        "first_world_window_errors": errors,
        "first_world_window_length": 16,
        "latest_prior_qualified_survived": True if qualified else None,
        "any_prior_qualified_survived": True if qualified else None,
        "selected_lineage_available": selected,
        "selected_lineage_joint_bit_exact_preserved": True if selected else None,
        "selected_exact_generation_relock_observed": True if selected else None,
        "selected_durable_retrieval_before_scratch": True if selected else None,
        "selected_lineage_entry_composed_greedy_accuracy": 1.0 if selected else None,
        "selected_lineage_entry_minus_commit_accuracy": 0.0 if selected else None,
        "latest_qualified_acquisition_comparison_available": qualified,
        "recurrence_minus_latest_qualified_acquisition_error_rate_delta": (
            errors / 16.0 if qualified else None
        ),
        "prior_same_regime_lineages": [probe] if qualified else [],
        "eligible_dormant_generations": [dormant],
        "best_dormant_composed_greedy_accuracy": 1.0,
        "best_dormant_zero_helper_accuracy": 1.0 / 3.0,
        "best_dormant_zero_beneficiary_accuracy": 1.0 / 3.0,
        "best_dormant_role_swapped_accuracy": 1.0 / 3.0,
        "selected_lineage_entry_activity_status": "dormant" if selected else None,
    }


def _summary(records: list[dict[str, object]], *, reward: float = 0.75) -> dict[str, object]:
    return {"mean_prequential_reward": reward, "recurrence_retention": records}


def _metric(metric_id: str, *, orientation: str = "higher") -> MetricContract:
    return MetricContract(
        metric_id=metric_id,
        role="primary",
        orientation=cast(Any, orientation),
        gate_mode="level_and_contrast",
        source_fields=("summary.test",),
        aggregation="test",
        eligibility="test",
        missingness="test",
        null_value_decimal="0",
    )


def _estimand(metric_id: str) -> EstimandContract:
    return EstimandContract(
        estimand_id="test_pair",
        role="primary",
        formula="test",
        condition_terms=(("selective_full", 1), ("writable_evidence", -1)),
        population_rule="test",
        metrics=(metric_id,),
    )


def _compact_shard(records: list[dict[str, object]], *, reward: float = 0.75) -> dict[str, object]:
    return {"summary": _summary(records, reward=reward)}


def _readiness_certification_binding() -> dict[str, object]:
    certification_id = calibration.READINESS_EQUIVALENCE_CERTIFICATION_ID
    return {
        "readiness_receipt_sha256": "a" * 64,
        "certification_ids": [certification_id],
        "certification_specifications_sha256": calibration.canonical_sha256(
            [{"certification_id": certification_id}]
        ),
        "certification_records_sha256": calibration.canonical_sha256(
            [{"certification_id": certification_id, "status": "passed"}]
        ),
        "all_required_certifications_passed": True,
    }


def _semantic_audit_shards() -> dict[int, dict[str, object]]:
    design = build_hidden_regime_factorial_calibration_design()
    shards: dict[int, dict[str, object]] = {}
    for case in design.cases:
        helper_learning = case.condition != "helper_frozen"
        beneficiary_learning = case.condition != "beneficiary_frozen"
        immutability_applicable = case.condition not in {
            "writable_evidence",
            "writable_lru",
        }
        summary: dict[str, object] = {
            "helper_value_write_count": int(helper_learning),
            "beneficiary_value_write_count": int(beneficiary_learning),
            "helper_effective_learning_update_count": int(helper_learning),
            "beneficiary_effective_learning_update_count": int(beneficiary_learning),
            "both_roles_learned": helper_learning and beneficiary_learning,
            "helper_commit_count": int(helper_learning),
            "beneficiary_commit_count": int(beneficiary_learning),
            "helper_replacement_count": int(helper_learning),
            "beneficiary_replacement_count": int(beneficiary_learning),
            "c_old_to_c_new_replacement_count": 1,
            "c_old_to_c_new_target_slots": [2],
            "c_old_to_c_new_generation_pairs": [[2, 3]],
            "c_old_to_c_new_exactly_one_target": True,
            "d_short_checked": True,
            "d_short_non_displacement": True,
            "selective_immutability_applicable": immutability_applicable,
            "helper_selective_mutation_violations": 0,
            "beneficiary_selective_mutation_violations": 0,
            "selective_durable_bit_immutable_until_atomic_replacement": (
                immutability_applicable
            ),
            "lifecycle_synchronized_every_step": True,
        }
        summary_encoded = cast(
            dict[str, object],
            calibration._encode_exact(summary),
        )
        resource: dict[str, object] = {
            "initial_state_scalars": 138,
            "final_state_scalars": 138,
            "initial_state_bytes": 552,
            "final_state_bytes": 552,
            "expected_state_bytes": 552,
            "resource_constant": True,
            "resource_matched": True,
        }
        audit: dict[str, object] = {
            "valid": True,
            "expected_steps": calibration.EXPECTED_STEPS,
            "rows_checked": calibration.EXPECTED_STEPS,
            "helper_transitions_checked": calibration.EXPECTED_STEPS,
            "beneficiary_transitions_checked": calibration.EXPECTED_STEPS,
            "world_transitions_checked": calibration.EXPECTED_STEPS,
            "mismatch_count": 0,
            "mismatches_sha256": calibration.canonical_sha256([]),
            "unobserved_transition_fields": [],
            "lineage_oracle_valid": True,
            "lineage_oracle_mismatches_sha256": calibration.canonical_sha256([]),
            "audited_summary_sha256": calibration.canonical_sha256(summary_encoded),
        }
        case_digest = calibration.canonical_sha256(case.to_payload())
        trace_digest = calibration.canonical_sha256(
            {"case_index": case.case_index, "kind": "trace"}
        )
        shards[case.case_index] = {
            "case": case.to_payload(),
            "payload_sha256": calibration.canonical_sha256(
                {"case_index": case.case_index, "kind": "shard"}
            ),
            "case_request_binding_sha256": case_digest,
            "configuration_sha256": calibration.canonical_sha256(
                {"case_index": case.case_index, "kind": "configuration"}
            ),
            "summary": summary_encoded,
            "summary_sha256": calibration.canonical_sha256(summary_encoded),
            "resource": resource,
            "resource_sha256": calibration.canonical_sha256(resource),
            "audit": audit,
            "primitive_trace": {
                "schema": calibration.HIDDEN_REGIME_TRACE_SCHEMA,
                "sha256": trace_digest,
                "rows": calibration.EXPECTED_STEPS,
                "persisted": False,
                "discard_required_after_audit": True,
            },
        }
    return shards


def _replace_semantic_summary(
    shard: dict[str, object],
    **updates: object,
) -> dict[str, object]:
    replacement = copy.deepcopy(shard)
    summary = cast(dict[str, object], calibration._decode_exact(replacement["summary"]))
    summary.update(updates)
    encoded = cast(dict[str, object], calibration._encode_exact(summary))
    replacement["summary"] = encoded
    replacement["summary_sha256"] = calibration.canonical_sha256(encoded)
    audit = cast(dict[str, object], replacement["audit"])
    audit["audited_summary_sha256"] = calibration.canonical_sha256(encoded)
    replacement["payload_sha256"] = calibration.canonical_sha256(
        {
            "case_index": cast(dict[str, object], replacement["case"])["case_index"],
            "summary_sha256": replacement["summary_sha256"],
        }
    )
    return replacement


def _gate_audit_summary(decision: str = "passed_nonstatistical") -> dict[str, object]:
    design = build_hidden_regime_factorial_calibration_design()
    results: list[dict[str, object]] = []
    for requirement in design.audits:
        reference = {
            "kind": "synthetic_exact_reference",
            "requirement_id": requirement.requirement_id,
        }
        result_body: dict[str, object] = {
            **requirement.to_payload(),
            "evaluation_mode": "synthetic_gate_matrix_unit",
            "threshold_independent": True,
            "thresholds_consulted": False,
            "decision": decision,
            "required_reference_count": 1,
            "required_references": [reference],
            "required_references_sha256": calibration.canonical_sha256([reference]),
            "descriptive_reference_count": 0,
            "descriptive_references": [],
            "descriptive_references_sha256": calibration.canonical_sha256([]),
            "failed_case_indices": [],
        }
        results.append(
            {
                **result_body,
                "requirement_result_sha256": calibration.canonical_sha256(result_body),
            }
        )
    return {
        "schema": calibration.CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "threshold_independent": True,
        "thresholds_consulted": False,
        "decision": decision,
        "requirement_results": results,
        "requirement_results_sha256": calibration.canonical_sha256(results),
    }


def test_protocol_digest_and_exact_cardinality_are_bound() -> None:
    design = calibration._design()
    assert len(design.cases) == 240
    assert len(design.seed_pairs) == 30
    assert len(design.condition_runtime_bindings) == 8
    assert CALIBRATION_DESIGN_PAYLOAD_SHA256 == (
        "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
    )


def test_float_hex_round_trip_is_exact_and_nonfinite_fails() -> None:
    value = math.nextafter(1.0, 2.0)
    encoded = calibration._float_hex(value, "value")
    assert calibration._parse_float_hex(encoded, "value") == value
    with pytest.raises(calibration.CalibrationError, match="canonical"):
        calibration._parse_float_hex("0x1p+0", "value")
    with pytest.raises(calibration.CalibrationError, match="finite"):
        calibration._float_hex(float("nan"), "value")
    with pytest.raises(calibration.CalibrationError, match="finite"):
        calibration._float_hex(float("inf"), "value")


def test_payload_digest_rejects_outcome_tamper() -> None:
    payload = calibration._payload_with_digest({"schema": "test", "value_hex": (1.0).hex()})
    assert calibration._validate_payload_digest(payload, "test")["value_hex"] == (1.0).hex()
    tampered = dict(payload)
    tampered["value_hex"] = (0.0).hex()
    with pytest.raises(calibration.CalibrationError, match="digest mismatch"):
        calibration._validate_payload_digest(tampered, "test")


def test_zip_worker_provenance_enforces_exact_no_site_runtime_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeZipImporter:
        pass

    archive_directory = tmp_path / "archive"
    archive_directory.mkdir()
    archive = archive_directory / "source.zip"
    member = "alberta_framework/fake.py"
    with calibration.zipfile.ZipFile(archive, "w") as source_zip:
        source_zip.writestr(member, b'"""fake"""\n')
    archive.chmod(0o444)
    cache = tmp_path / "bytecode-cache"
    cache.mkdir()
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    prefix = (tmp_path / "runtime").as_posix()
    purelib = f"{prefix}/lib/python3.12/site-packages"
    stdlib_paths = (
        (tmp_path / "runtime-base/python312.zip").as_posix(),
        (tmp_path / "runtime-base/lib/python3.12").as_posix(),
        (tmp_path / "runtime-base/lib/python3.12/lib-dynload").as_posix(),
    )
    exact_path = [archive.resolve().as_posix(), *stdlib_paths, purelib]
    fake_module = SimpleNamespace(
        __loader__=FakeZipImporter(),
        __file__=f"{archive.resolve().as_posix()}/{member}",
    )
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(no_site=1),
        modules={"alberta_framework.fake": fake_module},
        prefix=prefix,
        exec_prefix=prefix,
        path=list(exact_path),
        dont_write_bytecode=True,
        pycache_prefix=cache.as_posix(),
        _xoptions={"pycache_prefix": cache.as_posix()},
    )
    bundle = SimpleNamespace(
        payload={
            "body": {
                "runtime_identity": {
                    "python": {
                        "prefix": prefix,
                        "exec_prefix": prefix,
                        "purelib": purelib,
                        "platlib": purelib,
                        "no_site_stdlib_search_paths": list(stdlib_paths),
                    }
                }
            }
        },
        source_archive_sha256=calibration.hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(calibration, "sys", fake_sys)
    monkeypatch.setattr(calibration.zipimport, "zipimporter", FakeZipImporter)

    provenance = calibration._zip_worker_provenance(archive, cast(Any, bundle))
    assert provenance["no_site_startup"] is True
    assert provenance["prebootstrap_pth_hook_absent"] is True
    assert provenance["receipt_bound_runtime_prefix"] is True
    assert provenance["exact_receipt_bound_site_search_paths"] is True

    fake_sys.flags = SimpleNamespace(no_site=0)
    with pytest.raises(calibration.CalibrationError, match="automatic site"):
        calibration._zip_worker_provenance(archive, cast(Any, bundle))
    fake_sys.flags = SimpleNamespace(no_site=1)
    fake_sys.modules["_virtualenv"] = SimpleNamespace()
    with pytest.raises(calibration.CalibrationError, match="path-hook"):
        calibration._zip_worker_provenance(archive, cast(Any, bundle))
    del fake_sys.modules["_virtualenv"]
    fake_sys.prefix = f"{prefix}-drift"
    with pytest.raises(calibration.CalibrationError, match="runtime prefix"):
        calibration._zip_worker_provenance(archive, cast(Any, bundle))
    fake_sys.prefix = prefix
    fake_sys.path.append("/unbound/dependency-path")
    with pytest.raises(calibration.CalibrationError, match="exact readiness-bound"):
        calibration._zip_worker_provenance(archive, cast(Any, bundle))


def test_all_predeclared_metrics_have_direct_extractors() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    records = [_record((3, 0, 1, 1))]
    summary = _summary(records)
    observations = [
        calibration._metric_observation(contract, records, summary) for contract in design.metrics
    ]
    assert tuple(item["metric_id"] for item in observations) == tuple(
        contract.metric_id for contract in design.metrics
    )
    assert all(item["structural_missing_n"] == 0 for item in observations)


def test_qualified_failures_are_zero_not_best_lineage_substitution() -> None:
    records = [
        _record((3, 0, 1, 1), qualified=True, selected=False),
        _record((5, 1, 1, 1), qualified=False, selected=False),
    ]
    value, eligible, observed = calibration._metric_value(
        "selected_lineage_joint_bit_exact_preservation_rate",
        records,
        _summary(records),
    )
    assert value == 0.0
    assert eligible == ((3, 0, 1, 1),)
    assert observed == eligible


def test_paired_metric_uses_exact_qualification_intersection() -> None:
    identity_a = (3, 0, 1, 1)
    identity_b = (5, 1, 1, 1)
    conditions = {
        "selective_full": _compact_shard(
            [
                _record(identity_a, qualified=True, errors=0),
                _record(identity_b, qualified=False, errors=0),
            ]
        ),
        "writable_evidence": _compact_shard(
            [
                _record(identity_a, qualified=True, errors=16),
                _record(identity_b, qualified=True, errors=16),
            ]
        ),
    }
    paired = calibration._paired_seed_metric(
        _metric("qualified_first_entry_window_error_rate", orientation="lower"),
        _estimand("qualified_first_entry_window_error_rate"),
        conditions,
    )
    assert paired["included_recurrence_identities"] == [list(identity_a)]
    assert paired["excluded_recurrence_identities"] == [list(identity_b)]
    assert calibration._parse_float_hex(paired["oriented_delta_hex"], "delta") == 1.0


def test_paired_metric_rejects_misaligned_recurrence_identity() -> None:
    left = _compact_shard([_record((3, 0, 1, 1))])
    right = _compact_shard([_record((4, 0, 1, 1))])
    with pytest.raises(calibration.CalibrationError, match="populations differ"):
        calibration._paired_seed_metric(
            _metric("qualified_first_entry_window_error_rate", orientation="lower"),
            _estimand("qualified_first_entry_window_error_rate"),
            {"selective_full": left, "writable_evidence": right},
        )


def test_statistics_use_sample_sd_se_and_one_sided_t_bound() -> None:
    values = [0.25, 0.5, 0.75, 1.0]
    result = calibration._sample_statistics(values, null=0.0)
    mean = sum(values) / len(values)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 3)
    se = sample_sd / math.sqrt(4)
    expected_bound = mean - float(student_t.ppf(0.95, 3)) * se
    assert calibration._parse_float_hex(result["mean_hex"], "mean") == mean
    assert calibration._parse_float_hex(
        result["sample_standard_deviation_hex"], "sd"
    ) == pytest.approx(sample_sd, rel=0.0, abs=2e-16)
    assert calibration._parse_float_hex(result["standard_error_hex"], "se") == pytest.approx(
        se, rel=0.0, abs=2e-16
    )
    assert calibration._parse_float_hex(
        result["one_sided_95_percent_lower_confidence_bound_hex"], "bound"
    ) == pytest.approx(expected_bound, rel=0.0, abs=2e-16)
    assert (result["wins"], result["ties"], result["losses"]) == (4, 0, 0)


def test_strata_are_exact_n30_and_n10_with_explicit_missingness() -> None:
    rows: list[tuple[int, str, float | None, bool, bool]] = []
    for seed in range(30):
        manifest = CALIBRATION_MANIFEST_ORDER[seed % 3]
        if seed == 0:
            rows.append((seed, manifest, None, True, False))
        elif seed == 1:
            rows.append((seed, manifest, None, False, True))
        else:
            rows.append((seed, manifest, float(seed), False, False))
    result = calibration._stratified_statistics(rows, null=0.0)
    pooled = cast(dict[str, object], result["pooled"])
    assert pooled["eligible_n"] == 30
    assert pooled["observed_n"] == 28
    assert pooled["conditional_unobserved_seed_indices"] == [0]
    assert pooled["structural_missing_seed_indices"] == [1]
    assert [cast(dict[str, object], item)["eligible_n"] for item in result["by_manifest"]] == [
        10,
        10,
        10,
    ]


def test_worker_result_parser_rejects_trailing_or_noncanonical_data() -> None:
    payload = calibration._payload_with_digest({"schema": "synthetic"})
    raw = canonical_json_bytes(payload)
    encoded = calibration.WORKER_RESULT_PREFIX + calibration.base64.b64encode(raw)
    assert calibration._parse_worker_result(encoded) == payload
    with pytest.raises(calibration.CalibrationError, match="output differs"):
        calibration._parse_worker_result(encoded + b"\n")


def test_coordinator_refuses_execution_without_explicit_boolean(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_case_subprocess(
            case_index=0,
            readiness_directory=tmp_path / "missing-readiness",
            managed_ledger_directory=tmp_path / "missing-ledger",
            shard_publication_root=tmp_path,
            explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
            authorize_calibration_execution=False,
        )


def test_request_refuses_wrong_acknowledgement_before_any_execution() -> None:
    with pytest.raises(calibration.CalibrationError, match="exact explicit"):
        calibration.build_calibration_case_request(
            0,
            cast("calibration.ValidatedReadinessBundle", object()),
            managed_ledger_directory=Path("unused"),
            explicit_acknowledgement="yes",
        )


def test_immutable_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(calibration.CalibrationError, match="symlinked"):
        calibration._read_regular_file(link, max_bytes=100, label="symlink")


def test_immutable_reader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o444)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration._read_regular_file(linked / "target.json", max_bytes=100, label="ancestor")


@pytest.mark.parametrize("mutation", ["writable", "hardlink"])
def test_immutable_reader_rejects_mode_or_link_mutation_during_read(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "payload.json"
    path.write_bytes(b"immutable-input")
    path.chmod(0o444)
    alias = tmp_path / "payload-alias.json"
    real_read = os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        raw = real_read(descriptor, size)
        if raw and not mutated:
            mutated = True
            if mutation == "writable":
                path.chmod(0o644)
            else:
                os.link(path, alias)
        return raw

    monkeypatch.setattr(calibration.os, "read", mutate_after_first_read)
    with pytest.raises(calibration.CalibrationError, match="changed while reading"):
        calibration._read_regular_file(
            path,
            max_bytes=1024,
            label="mutating immutable input",
        )


def test_immutable_reader_rejects_parent_directory_substitution_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "bound"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    parent.mkdir()
    replacement.mkdir()
    path = parent / "payload.json"
    path.write_bytes(b"original-input")
    path.chmod(0o444)
    replacement_path = replacement / path.name
    replacement_path.write_bytes(b"replacement-input")
    replacement_path.chmod(0o444)
    real_read = os.read
    mutated = False

    def substitute_parent_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        raw = real_read(descriptor, size)
        if raw and not mutated:
            mutated = True
            parent.rename(moved)
            replacement.rename(parent)
        return raw

    monkeypatch.setattr(calibration.os, "read", substitute_parent_after_first_read)
    with pytest.raises(calibration.CalibrationError, match="parent changed while reading"):
        calibration._read_regular_file(
            path,
            max_bytes=1024,
            label="substituted immutable input",
        )


def test_new_only_shard_duplicate_must_be_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    body = {
        "case": {"case_index": 0},
        "readiness_binding": readiness,
    }
    shard = calibration._payload_with_digest(body)
    monkeypatch.setattr(
        calibration,
        "validate_finalized_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    path = calibration.publish_calibration_case_shard_new_only(
        tmp_path,
        shard,
        expected_readiness_binding=readiness,
        managed_ledger_directory=tmp_path,
    )
    assert path.stat().st_mode & 0o777 == 0o444
    assert (
        calibration.publish_calibration_case_shard_new_only(
            tmp_path,
            shard,
            expected_readiness_binding=readiness,
            managed_ledger_directory=tmp_path,
        )
        == path
    )
    tampered = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": readiness, "extra": True}
    )
    with pytest.raises(calibration.CalibrationError, match="not byte-identical"):
        calibration.publish_calibration_case_shard_new_only(
            tmp_path,
            tampered,
            expected_readiness_binding=readiness,
            managed_ledger_directory=tmp_path,
        )


def test_new_only_publication_installs_only_complete_immutable_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"complete-canonical-payload"
    target = tmp_path / "case-000.json"
    real_install = calibration.atomic_install_new_immutable
    install_observed = False

    def checked_install(
        directory_fd: int,
        name: str,
        payload: bytes,
        *,
        max_bytes: int,
        label: str,
    ) -> None:
        nonlocal install_observed
        assert name == target.name
        assert payload == raw
        assert max_bytes == calibration._MAX_SHARD_BYTES
        assert label == "case shard"
        assert not target.exists()
        assert not tuple(tmp_path.glob(".staging-*"))
        real_install(
            directory_fd,
            name,
            payload,
            max_bytes=max_bytes,
            label=label,
        )
        install_observed = True

    monkeypatch.setattr(calibration, "atomic_install_new_immutable", checked_install)
    calibration._write_new_immutable(
        tmp_path,
        target.name,
        raw,
        max_bytes=calibration._MAX_SHARD_BYTES,
        label="case shard",
    )
    assert install_observed
    assert target.read_bytes() == raw
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert target.stat().st_nlink == 1
    assert not tuple(tmp_path.glob(".staging-*"))

    def interrupted_install(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic crash before atomic install")

    second = tmp_path / "case-001.json"
    monkeypatch.setattr(calibration, "atomic_install_new_immutable", interrupted_install)
    with pytest.raises(OSError, match="synthetic crash"):
        calibration._write_new_immutable(
            tmp_path,
            second.name,
            raw,
            max_bytes=calibration._MAX_SHARD_BYTES,
            label="case shard",
        )
    assert not second.exists()
    assert not tuple(tmp_path.glob(".staging-*"))


def test_shard_publication_rejects_symlinked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    shard = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": readiness}
    )
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(
        calibration,
        "validate_finalized_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration.publish_calibration_case_shard_new_only(
            linked_root,
            shard,
            expected_readiness_binding=readiness,
            managed_ledger_directory=tmp_path,
        )


def test_verified_aggregate_worker_output_install_is_new_only_and_symlink_safe(
    tmp_path: Path,
) -> None:
    aggregate = calibration._payload_with_digest({"schema": "synthetic", "value": 1})
    raw = canonical_json_bytes(aggregate)
    published = calibration._install_verified_aggregate_worker_output_new_only(
        tmp_path,
        aggregate,
        raw,
    )
    digest = cast(str, aggregate["payload_sha256"])
    assert published.path.name == f"{digest}.json"
    assert published.path.stat().st_mode & 0o777 == 0o444
    duplicate = calibration._install_verified_aggregate_worker_output_new_only(
        tmp_path,
        aggregate,
        raw,
    )
    assert duplicate.path == published.path
    tampered = {**aggregate, "value": 2}
    with pytest.raises(calibration.CalibrationError, match="not byte-identical"):
        calibration._install_verified_aggregate_worker_output_new_only(
            tmp_path,
            tampered,
            canonical_json_bytes(tampered),
        )

    real_root = tmp_path / "aggregate-real"
    real_root.mkdir()
    linked_root = tmp_path / "aggregate-linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration._install_verified_aggregate_worker_output_new_only(
            linked_root,
            aggregate,
            raw,
        )


def test_verified_aggregate_installer_rejects_nonworker_bytes(tmp_path: Path) -> None:
    aggregate = calibration._payload_with_digest({"schema": "synthetic"})
    with pytest.raises(calibration.CalibrationError, match="worker bytes differ"):
        calibration._install_verified_aggregate_worker_output_new_only(
            tmp_path,
            aggregate,
            b"not-the-worker-output",
        )


def test_completed_aggregation_executes_only_in_the_bound_source_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_sha256 = "a" * 64
    genesis_sha256 = "b" * 64
    readiness = {"readiness_receipt_sha256": receipt_sha256}
    audit_summary = _gate_audit_summary()
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "case_count": calibration.EXPECTED_CASES,
            "readiness_binding": readiness,
            "managed_ledger_content_address": genesis_sha256,
            "mandatory_audit_summary": audit_summary,
            "mandatory_audit_summary_sha256": calibration.canonical_sha256(audit_summary),
            "mandatory_audit_decision": "passed_nonstatistical",
            "gate_decision_status": (
                "mandatory_audits_passed_statistical_thresholds_unset"
            ),
        }
    )
    aggregate_raw = canonical_json_bytes(aggregate)
    aggregate_root = tmp_path / "aggregates"
    aggregate_root.mkdir()
    aggregate_path = calibration.calibration_aggregate_path(
        aggregate_root,
        cast(str, aggregate["payload_sha256"]),
    )
    stdout = calibration.AGGREGATE_RESULT_PREFIX + base64.b64encode(
        aggregate_raw
    )
    calls: list[tuple[object, ...]] = []

    def execute(directory: Path, arguments: tuple[str, ...], **kwargs: object) -> object:
        calls.append((directory, arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(
            receipt_sha256=receipt_sha256,
            execution_genesis_sha256=genesis_sha256,
            payload={"body": {"runtime_identity": {}}},
        ),
    )
    monkeypatch.setattr(calibration, "execute_bound_calibration_worker", execute)
    monkeypatch.setattr(calibration, "_readiness_binding", lambda bundle: readiness)
    monkeypatch.setattr(
        calibration,
        "_validate_aggregate_provenance_bindings",
        lambda body, bundle: None,
    )
    monkeypatch.setattr(calibration, "require_current_full_runtime_identity", lambda runtime: None)
    monkeypatch.setattr(
        calibration,
        "aggregate_hidden_regime_factorial_calibration",
        lambda *a, **k: pytest.fail("mutable checkout computed the aggregate"),
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_aggregate",
        lambda *a, **k: pytest.fail("mutable checkout validated the aggregate"),
    )
    published = calibration.aggregate_and_publish_completed_calibration(
        readiness_directory=tmp_path / "readiness",
        shard_publication_root=tmp_path / "shards",
        managed_ledger_directory=tmp_path / "ledger",
        aggregate_publication_root=aggregate_root,
        authorize_publication=True,
        timeout_seconds=123,
    )
    assert published.path == aggregate_path
    assert published.payload == aggregate
    assert len(calls) == 1
    _, arguments, kwargs = calls[0]
    assert cast(tuple[str, ...], arguments) == (
        "--worker-aggregate-v1",
        (tmp_path / "readiness").absolute().as_posix(),
        (tmp_path / "ledger").absolute().as_posix(),
        (tmp_path / "shards").absolute().as_posix(),
    )
    assert cast(dict[str, object], kwargs) == {
        "authorize_calibration_execution": True,
        "timeout_seconds": 123,
    }


def test_aggregate_output_cannot_overlap_any_exact_input_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    for path in (readiness, shards, ledger):
        path.mkdir()
    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: pytest.fail("overlapping aggregate root launched a worker"),
    )
    for output_root, label in (
        (readiness, "readiness publication"),
        (shards, "shard publication"),
        (ledger, "managed ledger"),
    ):
        with pytest.raises(calibration.CalibrationError, match=label):
            calibration.aggregate_and_publish_completed_calibration(
                readiness_directory=readiness,
                shard_publication_root=shards,
                managed_ledger_directory=ledger,
                aggregate_publication_root=output_root,
                authorize_publication=True,
            )


def test_failed_worker_postcheck_cannot_install_returned_aggregate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    for path in (readiness, shards, ledger, aggregates):
        path.mkdir()
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "case_count": calibration.EXPECTED_CASES,
        }
    )
    stdout = calibration.AGGREGATE_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(aggregate)
    )
    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=stdout, stderr=b"postcheck"),
    )
    with pytest.raises(calibration.CalibrationError, match="isolated calibration aggregation"):
        calibration.aggregate_and_publish_completed_calibration(
            readiness_directory=readiness,
            shard_publication_root=shards,
            managed_ledger_directory=ledger,
            aggregate_publication_root=aggregates,
            authorize_publication=True,
        )
    assert not tuple(aggregates.iterdir())


def _synthetic_threshold_receipt(
    aggregate_payload_sha256: str,
    readiness_receipt_sha256: str,
    *,
    rejection: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "receipt_schema": (
            "alberta.hidden-regime-factorial.threshold-freeze-receipt.v1"
        ),
        "decision_status": (
            calibration.THRESHOLD_FREEZE_DECISION_REJECTION
            if rejection
            else calibration.THRESHOLD_FREEZE_DECISION_FROZEN
        ),
        "development_only": True,
        "claim_accepted": False,
        "thresholds_frozen": not rejection,
        "calibration_outcomes_payload_sha256": aggregate_payload_sha256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "frozen_thresholds": [] if rejection else [{"endpoint_id": "synthetic"}],
        "rejection_reasons": (
            [{"endpoint_id": "synthetic", "reasons": ["insufficient margin"]}]
            if rejection
            else []
        ),
        "scientific_promotion_allowed": False,
        "amendments_allowed": False,
    }
    return {
        **body,
        "receipt_payload_sha256": calibration.canonical_sha256(body),
    }


def _synthetic_threshold_exact_input_binding(
    aggregate_payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema": calibration.THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA,
        "calibration_aggregate_payload_sha256": aggregate_payload_sha256,
        "managed_ledger_inventory_sha256": "1" * 64,
        "managed_ledger_snapshot_sha256": "2" * 64,
        "case_ledger_sha256": "3" * 64,
        "case_shard_payloads_sha256": "4" * 64,
        "case_count": calibration.EXPECTED_CASES,
    }


def _synthetic_successful_protected_receipt(
    aggregate_payload_sha256: str,
    readiness_receipt_sha256: str,
) -> dict[str, object]:
    """Build a test-only receipt shape; it contains no official protected seed material."""

    frozen_thresholds = [
        {
            "endpoint_id": calibration.canonical_sha256(
                {"synthetic_protected_endpoint_index": index}
            ),
            "test_only_threshold_index": index,
        }
        for index in range(calibration.MANDATORY_STATISTICAL_ENDPOINT_COUNT)
    ]
    body: dict[str, object] = {
        "receipt_schema": calibration.THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        "decision_status": calibration.THRESHOLD_FREEZE_DECISION_FROZEN,
        "development_only": True,
        "claim_accepted": False,
        "thresholds_frozen": True,
        "calibration_outcomes_payload_sha256": aggregate_payload_sha256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "frozen_thresholds": frozen_thresholds,
        "rejection_reasons": [],
        "scientific_promotion_allowed": False,
        "amendments_allowed": False,
        "mandatory_statistical_endpoint_count": (
            calibration.MANDATORY_STATISTICAL_ENDPOINT_COUNT
        ),
        "mandatory_statistical_endpoint_identities_sha256": (
            calibration.MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
        ),
        "mandatory_statistical_endpoint_ids_sha256": (
            calibration.MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
        ),
        "protocol_payload_sha256": "1" * 64,
        "seed_snapshot_sha256": "2" * 64,
        "gate_matrix_sha256": "3" * 64,
        "source_closure_sha256": "4" * 64,
        "source_archive_sha256": "5" * 64,
        "environment_identity_sha256": "6" * 64,
        "managed_ledger_snapshot_sha256": "7" * 64,
        "managed_ledger_content_address": "8" * 64,
        "execution_governance_genesis_sha256": "9" * 64,
        "case_ledger_sha256": "a" * 64,
        "aggregation_readiness_certification_binding_sha256": "b" * 64,
        "mandatory_audit_summary_sha256": "c" * 64,
    }
    return {
        **body,
        "receipt_payload_sha256": calibration.canonical_sha256(body),
    }


def _synthetic_protected_plan(
    aggregate: dict[str, object],
    receipt: dict[str, object],
) -> dict[str, object]:
    """Build a structural test envelope, never an official protected plan or seed set."""

    aggregate_digest = cast(str, aggregate["payload_sha256"])
    receipt_digest = cast(str, receipt["receipt_payload_sha256"])
    frozen_thresholds = cast(list[object], receipt["frozen_thresholds"])
    seed_snapshot: dict[str, object] = {
        "schema": "synthetic.test-only.protected-seed-snapshot.v1",
        "official": False,
        "pairs": [],
    }
    disjointness: dict[str, object] = {"test_only": True, "verified": False}
    manifest_order: list[object] = ["synthetic-a", "synthetic-b", "synthetic-c"]
    manifest_bindings: list[object] = []
    recurrence_bindings: list[object] = []
    assignments: list[object] = [
        {"seed_index": index, "test_only": True}
        for index in range(calibration.EXPECTED_SEED_PAIRS)
    ]
    condition_order: list[object] = [f"synthetic-condition-{index}" for index in range(8)]
    cases: list[object] = [
        {"case_index": index, "test_only": True}
        for index in range(calibration.EXPECTED_CASES)
    ]
    evaluation_contract: dict[str, object] = {"test_only": True}
    decision_rule: dict[str, object] = {"test_only": True}
    body: dict[str, object] = {
        "schema": calibration.PROTECTED_PLAN_SCHEMA,
        "status_time_scope": "synthetic test-only facts",
        "plan_status": "preregistered_unexecuted",
        "use_partition": calibration.PROTECTED_CANDIDATE_PARTITION,
        "scientific_evidence_eligible_if_validated": True,
        "scientific_promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "amendments_allowed": False,
        "thresholds_frozen": True,
        "threshold_adjustment_permitted": False,
        "protected_namespace_derived": True,
        "protected_outcomes_observed": False,
        "learner_outcomes_executed": False,
        "learner_execution_authorized": False,
        "protected_execution_permitted": False,
        "execution_issuer_available": False,
        "protected_readiness_receipt_sha256": None,
        "protected_execution_ledger_genesis_sha256": None,
        "calibration_binding": {
            "protocol_payload_sha256": receipt["protocol_payload_sha256"],
            "calibration_seed_snapshot_sha256": receipt["seed_snapshot_sha256"],
            "calibration_aggregate_schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "calibration_aggregate_payload_sha256": aggregate_digest,
            "calibration_readiness_receipt_sha256": receipt["readiness_receipt_sha256"],
            "calibration_gate_matrix_sha256": receipt["gate_matrix_sha256"],
            "calibration_source_closure_sha256": receipt["source_closure_sha256"],
            "calibration_source_archive_sha256": receipt["source_archive_sha256"],
            "calibration_environment_identity_sha256": receipt[
                "environment_identity_sha256"
            ],
            "calibration_managed_ledger_snapshot_sha256": receipt[
                "managed_ledger_snapshot_sha256"
            ],
            "calibration_managed_ledger_content_address": receipt[
                "managed_ledger_content_address"
            ],
            "calibration_execution_governance_genesis_sha256": receipt[
                "execution_governance_genesis_sha256"
            ],
            "calibration_case_ledger_sha256": receipt["case_ledger_sha256"],
            "aggregation_readiness_certification_binding_sha256": receipt[
                "aggregation_readiness_certification_binding_sha256"
            ],
            "mandatory_audit_summary_sha256": receipt[
                "mandatory_audit_summary_sha256"
            ],
        },
        "threshold_freeze_receipt_binding": {
            "receipt_schema": calibration.THRESHOLD_FREEZE_RECEIPT_SCHEMA,
            "receipt_payload_sha256": receipt_digest,
            "decision_status": calibration.THRESHOLD_FREEZE_DECISION_FROZEN,
            "mandatory_statistical_endpoint_count": (
                calibration.MANDATORY_STATISTICAL_ENDPOINT_COUNT
            ),
            "mandatory_statistical_endpoint_identities_sha256": (
                calibration.MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
            ),
            "mandatory_statistical_endpoint_ids_sha256": (
                calibration.MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256
            ),
        },
        "frozen_thresholds": frozen_thresholds,
        "frozen_thresholds_sha256": calibration.canonical_sha256(frozen_thresholds),
        "protected_seed_snapshot": seed_snapshot,
        "protected_seed_snapshot_sha256": calibration.canonical_sha256(seed_snapshot),
        "seed_disjointness_proof": disjointness,
        "seed_disjointness_proof_sha256": calibration.canonical_sha256(disjointness),
        "structural_manifest_order": manifest_order,
        "structural_manifest_order_sha256": calibration.canonical_sha256(manifest_order),
        "manifest_bindings": manifest_bindings,
        "manifest_bindings_sha256": calibration.canonical_sha256(manifest_bindings),
        "recurrence_eligibility_bindings": recurrence_bindings,
        "recurrence_eligibility_bindings_sha256": calibration.canonical_sha256(
            recurrence_bindings
        ),
        "assignment_rule": "synthetic test-only assignment",
        "assignments": assignments,
        "assignments_sha256": calibration.canonical_sha256(assignments),
        "condition_order": condition_order,
        "condition_order_sha256": calibration.canonical_sha256(condition_order),
        "cases": cases,
        "cases_sha256": calibration.canonical_sha256(cases),
        "seed_pair_count": calibration.EXPECTED_SEED_PAIRS,
        "condition_count": 8,
        "matched_case_count": calibration.EXPECTED_CASES,
        "manifest_seed_pair_counts": [],
        "manifest_case_counts": [],
        "condition_case_counts": [],
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": calibration.canonical_sha256(evaluation_contract),
        "protected_decision_rule": decision_rule,
        "protected_decision_rule_sha256": calibration.canonical_sha256(decision_rule),
        "claim_scope": "synthetic test-only nonclaim",
        "limitations": ["not an official protected plan"],
    }
    return calibration._payload_with_digest(body)


def _synthetic_protected_worker_result(
    aggregate: dict[str, object],
    receipt: dict[str, object],
    plan: dict[str, object],
    *,
    readiness_receipt_sha256: str,
) -> dict[str, object]:
    return calibration._protected_plan_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        protected_plan=plan,
        readiness_receipt_sha256=readiness_receipt_sha256,
        exact_input_binding=_synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
        worker_readiness_certification_binding={},
        worker_provenance={},
        zip_provenance_attestation={},
    )


def test_threshold_worker_recomputes_exact_inputs_before_freezing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_receipt_sha256 = "a" * 64
    aggregate_worker_provenance = {"kind": "original-aggregate-worker"}
    aggregate_zip_attestation = {"kind": "original-aggregate-attestation"}
    certification_binding = _readiness_certification_binding()
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "readiness_binding": {
                "readiness_receipt_sha256": readiness_receipt_sha256,
            },
            "aggregation_worker_provenance": aggregate_worker_provenance,
            "aggregation_zip_provenance_attestation": aggregate_zip_attestation,
            "aggregation_readiness_certification_binding": certification_binding,
        }
    )
    aggregate_payload_sha256 = cast(str, aggregate["payload_sha256"])
    receipt = _synthetic_threshold_receipt(
        aggregate_payload_sha256,
        readiness_receipt_sha256,
        rejection=True,
    )
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=readiness_receipt_sha256,
    )
    shards = ({"synthetic": "shard"},)
    inventory = {"synthetic": "inventory"}
    current_worker_provenance = {"kind": "threshold-worker"}
    current_zip_attestation = {"kind": "threshold-attestation"}
    events: list[str] = []

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: {"runtime": "expected"},
    )

    def current_runtime() -> dict[str, str]:
        events.append("runtime")
        return {"runtime": "expected"}

    monkeypatch.setattr(calibration, "build_runtime_execution_identity", current_runtime)
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: {"readiness_receipt_sha256": readiness_receipt_sha256},
    )
    monkeypatch.setattr(
        calibration,
        "_aggregation_readiness_certification_binding",
        lambda value: certification_binding,
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda *a, **k: events.append("worker_provenance") or current_worker_provenance,
    )
    monkeypatch.setattr(
        calibration,
        "_read_regular_file",
        lambda *a, **k: events.append("read_source_zip") or b"source-zip",
    )
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: events.append("zip_attestation")
        or SimpleNamespace(payload=current_zip_attestation),
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_calibration_aggregate",
        lambda *a, **k: events.append("read_aggregate") or aggregate,
    )
    monkeypatch.setattr(
        calibration,
        "_validate_aggregate_provenance_bindings",
        lambda *a, **k: events.append("aggregate_provenance"),
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: events.append("load_shards") or shards,
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: events.append("snapshot_ledger") or inventory,
    )

    def validate_aggregate(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (aggregate, shards)
        assert kwargs == {
            "managed_ledger_snapshot": inventory,
            "managed_ledger_directory": tmp_path / "ledger",
            "aggregation_worker_provenance": current_worker_provenance,
            "aggregation_zip_provenance_attestation": current_zip_attestation,
            "aggregation_readiness_certification_binding": certification_binding,
        }
        events.append("exact_recompute")
        return aggregate

    monkeypatch.setattr(calibration, "validate_calibration_aggregate", validate_aggregate)
    exact_input_binding = _synthetic_threshold_exact_input_binding(
        aggregate_payload_sha256
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: events.append("input_binding") or exact_input_binding,
    )
    monkeypatch.setattr(
        calibration,
        "materialize_hidden_regime_factorial_threshold_freeze_receipt",
        lambda value: events.append("freeze") or receipt,
    )
    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        lambda payload, **kwargs: events.append("validate_receipt") or receipt,
    )
    monkeypatch.setattr(
        calibration,
        "require_current_full_runtime_identity",
        lambda runtime: events.append("full_runtime"),
    )
    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: pytest.fail("threshold freezing executed a learner"),
    )

    result = calibration._worker_threshold_freeze(
        readiness_directory=tmp_path / "readiness",
        ledger_directory=tmp_path / "ledger",
        shard_publication_root=tmp_path / "shards",
        aggregate_publication_root=tmp_path / "aggregates",
        aggregate_payload_sha256=aggregate_payload_sha256,
    )
    result_body = calibration._validate_payload_digest(result, "threshold worker result")
    assert result_body["threshold_freeze_receipt"] == receipt
    assert result_body["threshold_exact_input_binding"] == exact_input_binding
    assert result_body["threshold_worker_provenance"] == current_worker_provenance
    assert result_body["threshold_zip_provenance_attestation"] == current_zip_attestation
    assert events == [
        "runtime",
        "worker_provenance",
        "read_source_zip",
        "zip_attestation",
        "read_aggregate",
        "aggregate_provenance",
        "load_shards",
        "snapshot_ledger",
        "exact_recompute",
        "input_binding",
        "freeze",
        "validate_receipt",
        "runtime",
        "full_runtime",
    ]


def test_threshold_exact_input_binding_joins_ordered_shards_and_live_inventory() -> None:
    inventory_body = {"schema": "synthetic-completed-inventory"}
    inventory = {
        **inventory_body,
        "inventory_sha256": calibration.canonical_sha256(inventory_body),
    }
    shards = tuple(
        calibration._payload_with_digest({"case": {"case_index": case_index}})
        for case_index in range(calibration.EXPECTED_CASES)
    )
    case_ledger = [
        {
            "case_index": case_index,
            "case_shard_payload_sha256": shard["payload_sha256"],
        }
        for case_index, shard in enumerate(shards)
    ]
    aggregate = calibration._payload_with_digest(
        {
            "managed_ledger_snapshot": inventory,
            "managed_ledger_snapshot_sha256": calibration.canonical_sha256(inventory),
            "case_ledger": case_ledger,
            "case_ledger_sha256": calibration.canonical_sha256(case_ledger),
        }
    )
    binding = calibration._threshold_freeze_exact_input_binding(
        aggregate,
        shards,
        inventory,
    )
    assert binding == {
        "schema": calibration.THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA,
        "calibration_aggregate_payload_sha256": aggregate["payload_sha256"],
        "managed_ledger_inventory_sha256": inventory["inventory_sha256"],
        "managed_ledger_snapshot_sha256": calibration.canonical_sha256(inventory),
        "case_ledger_sha256": aggregate["case_ledger_sha256"],
        "case_shard_payloads_sha256": calibration.canonical_sha256(case_ledger),
        "case_count": calibration.EXPECTED_CASES,
    }

    changed_inventory = {**inventory, "late_replay": True}
    with pytest.raises(calibration.CalibrationError, match="managed ledger changed"):
        calibration._threshold_freeze_exact_input_binding(
            aggregate,
            shards,
            changed_inventory,
        )
    reordered_shards = (shards[1], shards[0], *shards[2:])
    with pytest.raises(calibration.CalibrationError, match="case shards changed"):
        calibration._threshold_freeze_exact_input_binding(
            aggregate,
            reordered_shards,
            inventory,
        )


def test_threshold_worker_main_dispatches_exact_content_addressed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    readiness = tmp_path / "readiness"
    ledger = tmp_path / "ledger"
    shards = tmp_path / "shards"
    aggregates = tmp_path / "aggregates"
    aggregate = calibration._payload_with_digest(
        {
            "readiness_binding": {"readiness_receipt_sha256": "a" * 64},
        }
    )
    receipt = _synthetic_threshold_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
        rejection=True,
    )
    result = calibration._threshold_freeze_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        readiness_receipt_sha256="a" * 64,
        threshold_exact_input_binding=_synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
        threshold_worker_readiness_certification_binding={},
        threshold_worker_provenance={},
        threshold_zip_provenance_attestation={},
    )
    observed: list[dict[str, object]] = []

    def dispatch(**kwargs: object) -> dict[str, object]:
        observed.append(kwargs)
        return result

    monkeypatch.setattr(calibration, "_worker_threshold_freeze", dispatch)
    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: pytest.fail("threshold worker dispatch executed a learner"),
    )
    assert (
        calibration.main(
            (
                "--worker-threshold-freeze-v1",
                readiness.as_posix(),
                ledger.as_posix(),
                shards.as_posix(),
                aggregates.as_posix(),
                cast(str, aggregate["payload_sha256"]),
            )
        )
        == 0
    )
    captured = capsysbinary.readouterr()
    assert captured.out == calibration.THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    assert captured.err == b""
    assert observed == [
        {
            "readiness_directory": readiness.absolute(),
            "ledger_directory": ledger.absolute(),
            "shard_publication_root": shards.absolute(),
            "aggregate_publication_root": aggregates.absolute(),
            "aggregate_payload_sha256": aggregate["payload_sha256"],
        }
    ]

    with pytest.raises(calibration.CalibrationError, match="arguments are not exact"):
        calibration.main(
            (
                "--worker-threshold-freeze-v1",
                readiness.as_posix(),
            )
        )


def test_protected_plan_worker_main_dispatches_exact_nonauthorizing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    readiness = tmp_path / "readiness"
    ledger = tmp_path / "ledger"
    shards = tmp_path / "shards"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": "a" * 64}}
    )
    receipt = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    result = _synthetic_protected_worker_result(
        aggregate,
        receipt,
        plan,
        readiness_receipt_sha256="a" * 64,
    )
    observed: list[dict[str, object]] = []

    def dispatch(**kwargs: object) -> dict[str, object]:
        observed.append(kwargs)
        return result

    monkeypatch.setattr(calibration, "_worker_protected_plan", dispatch)
    for forbidden_name in (
        "run_hidden_regime_condition",
        "issue_calibration_execution_authorization",
        "initialize_calibration_execution_ledger",
        "materialize_hidden_regime_factorial_threshold_freeze_receipt",
    ):
        monkeypatch.setattr(
            calibration,
            forbidden_name,
            lambda *a, _name=forbidden_name, **k: pytest.fail(
                f"protected-plan dispatch invoked forbidden {_name}"
            ),
        )
    assert (
        calibration.main(
            (
                "--worker-protected-plan-v1",
                readiness.as_posix(),
                ledger.as_posix(),
                shards.as_posix(),
                aggregates.as_posix(),
                cast(str, aggregate["payload_sha256"]),
                thresholds.as_posix(),
                cast(str, receipt["receipt_payload_sha256"]),
            )
        )
        == 0
    )
    captured = capsysbinary.readouterr()
    assert captured.out == calibration.PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    assert captured.err == b""
    assert observed == [
        {
            "readiness_directory": readiness.absolute(),
            "ledger_directory": ledger.absolute(),
            "shard_publication_root": shards.absolute(),
            "aggregate_publication_root": aggregates.absolute(),
            "aggregate_payload_sha256": aggregate["payload_sha256"],
            "threshold_receipt_publication_root": thresholds.absolute(),
            "threshold_receipt_payload_sha256": receipt["receipt_payload_sha256"],
        }
    ]

    with pytest.raises(calibration.CalibrationError, match="arguments are not exact"):
        calibration.main(("--worker-protected-plan-v1", readiness.as_posix()))


def test_protected_plan_worker_validates_receipt_before_deriving_and_never_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_digest = "a" * 64
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": readiness_digest}}
    )
    receipt = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        readiness_digest,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=readiness_digest,
    )
    exact_binding = _synthetic_threshold_exact_input_binding(
        cast(str, aggregate["payload_sha256"])
    )
    events: list[str] = []
    expected_runtime = {"test_runtime": "exact"}
    current_worker_provenance = {"test_worker": "protected-plan"}
    current_attestation = {"test_attestation": "protected-plan"}
    certification = {"test_certification": "protected-plan"}

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda value: expected_runtime,
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: events.append("runtime") or expected_runtime,
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: {"readiness_receipt_sha256": readiness_digest},
    )
    monkeypatch.setattr(
        calibration,
        "_aggregation_readiness_certification_binding",
        lambda value: certification,
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda *a, **k: events.append("worker_provenance") or current_worker_provenance,
    )
    monkeypatch.setattr(
        calibration,
        "_read_regular_file",
        lambda *a, **k: events.append("source_zip") or b"source-zip",
    )
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: events.append("zip_attestation")
        or SimpleNamespace(payload=current_attestation),
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_calibration_aggregate",
        lambda *a, **k: events.append("load_aggregate") or aggregate,
    )
    monkeypatch.setattr(
        calibration,
        "_validate_aggregate_provenance_bindings",
        lambda *a, **k: events.append("aggregate_provenance"),
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_threshold_freeze_receipt",
        lambda *a, **k: events.append("load_receipt") or receipt,
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    shards = ({"test_only": "shard"},)
    inventory = {"test_only": "ledger"}
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: events.append("load_240_shards") or shards,
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: events.append("snapshot_live_ledger") or inventory,
    )

    def exact_recompute(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (aggregate, shards)
        assert kwargs["managed_ledger_snapshot"] == inventory
        assert kwargs["aggregation_worker_provenance"] == current_worker_provenance
        assert kwargs["aggregation_zip_provenance_attestation"] == current_attestation
        assert kwargs["aggregation_readiness_certification_binding"] == certification
        events.append("exact_recompute_aggregate")
        return aggregate

    monkeypatch.setattr(calibration, "validate_calibration_aggregate", exact_recompute)
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: events.append("exact_input_binding") or exact_binding,
    )

    def validate_receipt(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (receipt,)
        assert kwargs == {"calibration_aggregate": aggregate}
        events.append("full_validate_receipt")
        return receipt

    def derive_plan(*args: object, **kwargs: object) -> dict[str, object]:
        assert events[-1] == "full_validate_receipt"
        assert args == (receipt,)
        assert kwargs == {"calibration_aggregate": aggregate}
        events.append("derive_plan")
        return plan

    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(
        calibration,
        "build_hidden_regime_factorial_protected_plan",
        derive_plan,
    )
    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_protected_plan",
        lambda *a, **k: events.append("validate_plan") or plan,
    )
    monkeypatch.setattr(
        calibration,
        "require_current_full_runtime_identity",
        lambda value: events.append("full_runtime"),
    )
    for forbidden_name in (
        "run_hidden_regime_condition",
        "issue_calibration_execution_authorization",
        "initialize_calibration_execution_ledger",
        "materialize_hidden_regime_factorial_threshold_freeze_receipt",
    ):
        monkeypatch.setattr(
            calibration,
            forbidden_name,
            lambda *a, _name=forbidden_name, **k: pytest.fail(
                f"protected-plan worker invoked forbidden {_name}"
            ),
        )

    result = calibration._worker_protected_plan(
        readiness_directory=tmp_path / "readiness",
        ledger_directory=tmp_path / "ledger",
        shard_publication_root=tmp_path / "shards",
        aggregate_publication_root=tmp_path / "aggregates",
        aggregate_payload_sha256=cast(str, aggregate["payload_sha256"]),
        threshold_receipt_publication_root=tmp_path / "thresholds",
        threshold_receipt_payload_sha256=cast(str, receipt["receipt_payload_sha256"]),
    )
    body = calibration._validate_protected_plan_worker_result_payload(result)
    assert body["nonauthorizing"] is True
    assert body["protected_plan"] == plan
    assert body["exact_input_binding"] == exact_binding
    assert all(
        body[field] is False
        for field in (
            "scientific_promotion_allowed",
            "automatic_promotion_allowed",
            "protected_outcomes_observed",
            "learner_outcomes_executed",
            "learner_execution_authorized",
            "protected_execution_permitted",
            "execution_issuer_available",
            "protected_readiness_created",
            "protected_execution_ledger_created",
        )
    )
    assert events == [
        "runtime",
        "worker_provenance",
        "source_zip",
        "zip_attestation",
        "load_aggregate",
        "aggregate_provenance",
        "load_receipt",
        "load_240_shards",
        "snapshot_live_ledger",
        "exact_recompute_aggregate",
        "exact_input_binding",
        "full_validate_receipt",
        "derive_plan",
        "validate_plan",
        "runtime",
        "full_runtime",
    ]


@pytest.mark.parametrize("failure_kind", ["valid_rejection", "malformed_receipt"])
def test_protected_plan_worker_rejection_or_malformed_receipt_derives_nothing_and_emits_nothing(
    failure_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    readiness_digest = "a" * 64
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": readiness_digest}}
    )
    successful = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        readiness_digest,
    )
    receipt = (
        _synthetic_threshold_receipt(
            cast(str, aggregate["payload_sha256"]),
            readiness_digest,
            rejection=True,
        )
        if failure_kind == "valid_rejection"
        else successful
    )
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=readiness_digest,
    )
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda value: {"runtime": "exact"},
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: {"runtime": "exact"},
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: {"readiness_receipt_sha256": readiness_digest},
    )
    monkeypatch.setattr(
        calibration,
        "_aggregation_readiness_certification_binding",
        lambda value: {},
    )
    monkeypatch.setattr(calibration, "_zip_worker_provenance", lambda *a, **k: {})
    monkeypatch.setattr(calibration, "_read_regular_file", lambda *a, **k: b"source-zip")
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: SimpleNamespace(payload={}),
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_calibration_aggregate",
        lambda *a, **k: aggregate,
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_threshold_freeze_receipt",
        lambda *a, **k: receipt,
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"test_only": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"test_only": "ledger"},
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_aggregate",
        lambda *a, **k: aggregate,
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: _synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
    )
    if failure_kind == "valid_rejection":
        monkeypatch.setattr(
            calibration,
            "validate_hidden_regime_factorial_threshold_freeze_receipt",
            lambda *a, **k: receipt,
        )
        expected = "not a successful threshold freeze"
    else:
        monkeypatch.setattr(
            calibration,
            "validate_hidden_regime_factorial_threshold_freeze_receipt",
            lambda *a, **k: (_ for _ in ()).throw(
                calibration.ThresholdFreezeError("malformed threshold receipt")
            ),
        )
        expected = "threshold receipt validation failed"
    monkeypatch.setattr(
        calibration,
        "build_hidden_regime_factorial_protected_plan",
        lambda *a, **k: pytest.fail("invalid threshold receipt derived protected seeds"),
    )
    with pytest.raises(calibration.CalibrationError, match=expected):
        calibration.main(
            (
                "--worker-protected-plan-v1",
                (tmp_path / "readiness").as_posix(),
                (tmp_path / "ledger").as_posix(),
                (tmp_path / "shards").as_posix(),
                (tmp_path / "aggregates").as_posix(),
                cast(str, aggregate["payload_sha256"]),
                (tmp_path / "thresholds").as_posix(),
                cast(str, receipt["receipt_payload_sha256"]),
            )
        )
    captured = capsysbinary.readouterr()
    assert captured.out == b""


def test_threshold_worker_structural_failure_emits_no_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    monkeypatch.setattr(
        calibration,
        "_worker_threshold_freeze",
        lambda **kwargs: (_ for _ in ()).throw(
            calibration.CalibrationError("structural aggregate failure")
        ),
    )
    with pytest.raises(calibration.CalibrationError, match="structural aggregate failure"):
        calibration.main(
            (
                "--worker-threshold-freeze-v1",
                (tmp_path / "readiness").as_posix(),
                (tmp_path / "ledger").as_posix(),
                (tmp_path / "shards").as_posix(),
                (tmp_path / "aggregates").as_posix(),
                "a" * 64,
            )
        )
    captured = capsysbinary.readouterr()
    assert captured.out == b""


def test_threshold_worker_parser_requires_exact_prefix_and_canonical_bytes() -> None:
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": "a" * 64}}
    )
    receipt = _synthetic_threshold_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
        rejection=True,
    )
    result = calibration._threshold_freeze_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        readiness_receipt_sha256="a" * 64,
        threshold_exact_input_binding=_synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
        threshold_worker_readiness_certification_binding={},
        threshold_worker_provenance={},
        threshold_zip_provenance_attestation={},
    )
    raw = canonical_json_bytes(result)
    valid = calibration.THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(raw)
    assert calibration._parse_threshold_freeze_worker_result(valid) == result
    with pytest.raises(calibration.CalibrationError, match="prefix differs"):
        calibration._parse_threshold_freeze_worker_result(b"WRONG:" + base64.b64encode(raw))
    noncanonical = calibration.THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(raw + b" ")
    with pytest.raises(calibration.CalibrationError, match="byte-canonical"):
        calibration._parse_threshold_freeze_worker_result(noncanonical)


def test_protected_plan_worker_parser_is_canonical_and_all_authority_flags_are_false() -> None:
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": "a" * 64}}
    )
    receipt = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    result = _synthetic_protected_worker_result(
        aggregate,
        receipt,
        plan,
        readiness_receipt_sha256="a" * 64,
    )
    raw = canonical_json_bytes(result)
    valid = calibration.PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(raw)
    assert calibration._parse_protected_plan_worker_result(valid) == result
    with pytest.raises(calibration.CalibrationError, match="prefix differs"):
        calibration._parse_protected_plan_worker_result(
            b"WRONG:" + base64.b64encode(raw)
        )
    noncanonical = calibration.PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(raw + b" ")
    with pytest.raises(calibration.CalibrationError, match="byte-canonical"):
        calibration._parse_protected_plan_worker_result(noncanonical)

    tampered_body = dict(result)
    tampered_body.pop("payload_sha256")
    tampered_body["learner_execution_authorized"] = True
    tampered = calibration._payload_with_digest(tampered_body)
    with pytest.raises(calibration.CalibrationError, match="must be false"):
        calibration._parse_protected_plan_worker_result(
            calibration.PROTECTED_PLAN_RESULT_PREFIX
            + base64.b64encode(canonical_json_bytes(tampered))
        )


@pytest.mark.parametrize(
    ("tamper_kind", "tampered_fields"),
    [
        (
            "rehashed_forged_statistics",
            {
                "estimand_summaries": [
                    {
                        "estimand_id": "forged",
                        "statistics": {"mean_hex": float("999").hex()},
                    }
                ]
            },
        ),
        (
            "duplicated_ledger_records_and_case_references",
            {
                "managed_ledger_snapshot": {
                    "completed_records": [
                        {"case_index": 0, "completed_record_sha256": "c" * 64},
                        {"case_index": 0, "completed_record_sha256": "c" * 64},
                    ]
                },
                "case_ledger": [
                    {"case_index": 0, "case_shard_payload_sha256": "d" * 64},
                    {"case_index": 0, "case_shard_payload_sha256": "d" * 64},
                ],
            },
        ),
        (
            "fabricated_audit_references",
            {
                "mandatory_audit_summary": {
                    "decision": "passed_nonstatistical",
                    "case_audit_references": [{"case_index": 999}],
                    "requirement_results": [
                        {
                            "requirement_id": "invented",
                            "decision": "passed_nonstatistical",
                        }
                    ],
                }
            },
        ),
        (
            "self_consistent_provenance_tamper",
            {
                "aggregation_worker_provenance": {
                    "source_archive_sha256": "e" * 64,
                    "project_modules_sha256": "f" * 64,
                },
                "aggregation_zip_provenance_attestation": {
                    "zip_provenance_attestation_sha256": "1" * 64,
                },
            },
        ),
    ],
)
def test_threshold_worker_exact_recomputation_rejects_rehashed_structural_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
    tampered_fields: dict[str, object],
) -> None:
    aggregate = calibration._payload_with_digest(
        {
            "readiness_binding": {},
            "aggregation_worker_provenance": {},
            "aggregation_zip_provenance_attestation": {},
            "aggregation_readiness_certification_binding": {},
            **copy.deepcopy(tampered_fields),
        }
    )
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256="a" * 64,
    )
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: {"runtime": "expected"},
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: {"runtime": "expected"},
    )
    monkeypatch.setattr(calibration, "_readiness_binding", lambda value: {})
    monkeypatch.setattr(
        calibration,
        "_aggregation_readiness_certification_binding",
        lambda value: {},
    )
    monkeypatch.setattr(calibration, "_zip_worker_provenance", lambda *a, **k: {})
    monkeypatch.setattr(calibration, "_read_regular_file", lambda *a, **k: b"source-zip")
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: SimpleNamespace(payload={}),
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_calibration_aggregate",
        lambda *a, **k: aggregate,
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"synthetic": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"synthetic": "inventory"},
    )
    exact_recomputations: list[str] = []

    def reject_exact_recomputation(*args: object, **kwargs: object) -> dict[str, object]:
        assert args[0] == aggregate
        assert kwargs["aggregation_worker_provenance"] == {}
        assert kwargs["aggregation_zip_provenance_attestation"] == {}
        assert kwargs["aggregation_readiness_certification_binding"] == {}
        exact_recomputations.append(tamper_kind)
        raise calibration.CalibrationError(f"exact aggregate mismatch: {tamper_kind}")

    monkeypatch.setattr(
        calibration,
        "validate_calibration_aggregate",
        reject_exact_recomputation,
    )
    monkeypatch.setattr(
        calibration,
        "materialize_hidden_regime_factorial_threshold_freeze_receipt",
        lambda value: pytest.fail("structurally invalid aggregate reached threshold engine"),
    )
    with pytest.raises(calibration.CalibrationError, match=tamper_kind):
        calibration._worker_threshold_freeze(
            readiness_directory=tmp_path / "readiness",
            ledger_directory=tmp_path / "ledger",
            shard_publication_root=tmp_path / "shards",
            aggregate_publication_root=tmp_path / "aggregates",
            aggregate_payload_sha256=cast(str, aggregate["payload_sha256"]),
        )
    assert exact_recomputations == [tamper_kind]


def test_threshold_receipt_install_is_content_addressed_new_only_and_symlink_safe(
    tmp_path: Path,
) -> None:
    receipt = _synthetic_threshold_receipt("a" * 64, "b" * 64)
    raw = canonical_json_bytes(receipt)
    published = calibration._install_verified_threshold_freeze_receipt_new_only(
        tmp_path,
        receipt,
        raw,
    )
    digest = cast(str, receipt["receipt_payload_sha256"])
    assert published.path.name == f"{digest}.json"
    assert published.path.stat().st_mode & 0o777 == 0o444
    assert published.path.stat().st_nlink == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        calibration._install_verified_threshold_freeze_receipt_new_only(
            tmp_path,
            receipt,
            raw,
        )

    real_root = tmp_path / "threshold-real"
    real_root.mkdir()
    linked_root = tmp_path / "threshold-linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration._install_verified_threshold_freeze_receipt_new_only(
            linked_root,
            receipt,
            raw,
        )


def test_protected_plan_install_is_content_addressed_strict_new_only_and_symlink_safe(
    tmp_path: Path,
) -> None:
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": "a" * 64}}
    )
    receipt = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    raw = canonical_json_bytes(plan)
    published = calibration._install_verified_protected_plan_new_only(
        tmp_path,
        plan,
        raw,
    )
    digest = cast(str, plan["payload_sha256"])
    assert published.path.name == f"{digest}.json"
    assert published.path.read_bytes() == raw
    assert stat.S_IMODE(published.path.stat().st_mode) == 0o444
    assert published.path.stat().st_nlink == 1
    with pytest.raises(FileExistsError, match="overwrite or reuse"):
        calibration._install_verified_protected_plan_new_only(tmp_path, plan, raw)

    real_root = tmp_path / "plan-real"
    real_root.mkdir()
    linked_root = tmp_path / "plan-linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    second = copy.deepcopy(plan)
    second_body = dict(second)
    second_body.pop("payload_sha256")
    second_body["status_time_scope"] = "second synthetic plan"
    second = calibration._payload_with_digest(second_body)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration._install_verified_protected_plan_new_only(
            linked_root,
            second,
            canonical_json_bytes(second),
        )


def test_threshold_parent_publishes_canonical_valid_rejection_only_after_rechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    for path in (readiness, shards, ledger, aggregates, thresholds):
        path.mkdir()
    readiness_receipt_sha256 = "a" * 64
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "readiness_binding": {
                "readiness_receipt_sha256": readiness_receipt_sha256,
            },
            "managed_ledger_content_address": "b" * 64,
        }
    )
    aggregate_raw = canonical_json_bytes(aggregate)
    calibration._write_new_immutable(
        aggregates,
        f"{aggregate['payload_sha256']}.json",
        aggregate_raw,
        max_bytes=calibration._MAX_AGGREGATE_BYTES,
        label="synthetic aggregate",
    )
    receipt = _synthetic_threshold_receipt(
        cast(str, aggregate["payload_sha256"]),
        readiness_receipt_sha256,
        rejection=True,
    )
    certification = _readiness_certification_binding()
    result = calibration._threshold_freeze_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        readiness_receipt_sha256=readiness_receipt_sha256,
        threshold_exact_input_binding=_synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
        threshold_worker_readiness_certification_binding=certification,
        threshold_worker_provenance={"kind": "threshold-worker"},
        threshold_zip_provenance_attestation={"kind": "threshold-attestation"},
    )
    stdout = calibration.THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    calls: list[tuple[object, ...]] = []

    def execute(directory: Path, arguments: tuple[str, ...], **kwargs: object) -> object:
        calls.append((directory, arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    bundle = SimpleNamespace(
        receipt_sha256=readiness_receipt_sha256,
        execution_genesis_sha256="b" * 64,
        payload={"body": {"runtime_identity": {}}},
    )
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(calibration, "execute_bound_calibration_worker", execute)
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: aggregate["readiness_binding"],
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_validate_threshold_worker_provenance_bindings",
        lambda *a: None,
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"synthetic": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"synthetic": "inventory"},
    )
    exact_input_binding = _synthetic_threshold_exact_input_binding(
        cast(str, aggregate["payload_sha256"])
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: exact_input_binding,
    )
    receipt_validations: list[dict[str, object]] = []

    def validate_receipt(payload: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["calibration_aggregate"] == aggregate
        receipt_validations.append(cast(dict[str, object], payload))
        return receipt

    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(calibration, "require_current_full_runtime_identity", lambda value: None)
    published = calibration.freeze_and_publish_completed_calibration_thresholds(
        readiness_directory=readiness,
        shard_publication_root=shards,
        managed_ledger_directory=ledger,
        aggregate_publication_root=aggregates,
        aggregate_payload_sha256=cast(str, aggregate["payload_sha256"]),
        threshold_receipt_publication_root=thresholds,
        authorize_publication=True,
        timeout_seconds=123,
    )
    assert published.payload == receipt
    assert published.path.read_bytes() == canonical_json_bytes(receipt)
    assert receipt_validations == [receipt]
    assert len(calls) == 1
    _, arguments, kwargs = calls[0]
    assert arguments == (
        "--worker-threshold-freeze-v1",
        readiness.absolute().as_posix(),
        ledger.absolute().as_posix(),
        shards.absolute().as_posix(),
        aggregates.absolute().as_posix(),
        aggregate["payload_sha256"],
    )
    assert kwargs == {
        "authorize_calibration_execution": True,
        "timeout_seconds": 123,
    }


def test_protected_plan_parent_twice_rechecks_and_publishes_exact_worker_bytes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    plans = tmp_path / "plans"
    for path in (readiness, shards, ledger, aggregates, thresholds, plans):
        path.mkdir()
    readiness_digest = "a" * 64
    genesis_digest = "b" * 64
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "readiness_binding": {"readiness_receipt_sha256": readiness_digest},
            "managed_ledger_content_address": genesis_digest,
        }
    )
    aggregate_digest = cast(str, aggregate["payload_sha256"])
    receipt = _synthetic_successful_protected_receipt(
        aggregate_digest,
        readiness_digest,
    )
    receipt_digest = cast(str, receipt["receipt_payload_sha256"])
    plan = _synthetic_protected_plan(aggregate, receipt)
    result = _synthetic_protected_worker_result(
        aggregate,
        receipt,
        plan,
        readiness_receipt_sha256=readiness_digest,
    )
    calibration._write_new_immutable(
        aggregates,
        f"{aggregate_digest}.json",
        canonical_json_bytes(aggregate),
        max_bytes=calibration._MAX_AGGREGATE_BYTES,
        label="synthetic aggregate",
    )
    calibration._write_new_immutable(
        thresholds,
        f"{receipt_digest}.json",
        canonical_json_bytes(receipt),
        max_bytes=calibration._MAX_RECEIPT_BYTES,
        label="synthetic threshold receipt",
    )
    stdout = calibration.PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=readiness_digest,
        source_archive_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
        execution_genesis_sha256=genesis_digest,
    )
    readiness_loads: list[Path] = []

    def load_bundle(directory: Path, **kwargs: object) -> object:
        readiness_loads.append(directory)
        return bundle

    worker_calls: list[tuple[Path, tuple[str, ...], dict[str, object]]] = []

    def execute(directory: Path, arguments: tuple[str, ...], **kwargs: object) -> object:
        worker_calls.append((directory, arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", load_bundle)
    monkeypatch.setattr(calibration, "execute_bound_calibration_worker", execute)
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: {"readiness_receipt_sha256": readiness_digest},
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_validate_protected_plan_worker_provenance_bindings",
        lambda *a: None,
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"test_only": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"test_only": "ledger"},
    )
    exact_binding = _synthetic_threshold_exact_input_binding(aggregate_digest)
    exact_binding_calls: list[str] = []
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: exact_binding_calls.append("exact") or exact_binding,
    )
    receipt_validations: list[str] = []

    def validate_receipt(payload: object, **kwargs: object) -> dict[str, object]:
        assert payload == receipt
        assert kwargs["calibration_aggregate"] == aggregate
        receipt_validations.append("validated")
        return receipt

    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(calibration, "require_current_full_runtime_identity", lambda value: None)
    for forbidden_name in (
        "build_hidden_regime_factorial_protected_plan",
        "validate_hidden_regime_factorial_protected_plan",
        "run_hidden_regime_condition",
        "issue_calibration_execution_authorization",
        "initialize_calibration_execution_ledger",
    ):
        monkeypatch.setattr(
            calibration,
            forbidden_name,
            lambda *a, _name=forbidden_name, **k: pytest.fail(
                f"parent publication invoked forbidden {_name}"
            ),
        )

    published = calibration.derive_and_publish_completed_calibration_protected_plan(
        readiness_directory=readiness,
        shard_publication_root=shards,
        managed_ledger_directory=ledger,
        aggregate_publication_root=aggregates,
        aggregate_payload_sha256=aggregate_digest,
        threshold_receipt_publication_root=thresholds,
        threshold_receipt_payload_sha256=receipt_digest,
        protected_plan_publication_root=plans,
        authorize_publication=True,
        timeout_seconds=321,
    )
    assert published.payload == plan
    assert published.path.read_bytes() == canonical_json_bytes(plan)
    assert stat.S_IMODE(published.path.stat().st_mode) == 0o444
    assert published.path.stat().st_nlink == 1
    assert len(readiness_loads) == 2
    assert receipt_validations == ["validated", "validated"]
    assert exact_binding_calls == ["exact", "exact"]
    assert worker_calls == [
        (
            readiness,
            (
                "--worker-protected-plan-v1",
                readiness.absolute().as_posix(),
                ledger.absolute().as_posix(),
                shards.absolute().as_posix(),
                aggregates.absolute().as_posix(),
                aggregate_digest,
                thresholds.absolute().as_posix(),
                receipt_digest,
            ),
            {
                "authorize_calibration_execution": True,
                "timeout_seconds": 321,
            },
        )
    ]


def test_threshold_parent_provenance_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    for path in (readiness, shards, ledger, aggregates, thresholds):
        path.mkdir()
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "readiness_binding": {"readiness_receipt_sha256": "a" * 64},
            "managed_ledger_content_address": "b" * 64,
        }
    )
    calibration._write_new_immutable(
        aggregates,
        f"{aggregate['payload_sha256']}.json",
        canonical_json_bytes(aggregate),
        max_bytes=calibration._MAX_AGGREGATE_BYTES,
        label="synthetic aggregate",
    )
    receipt = _synthetic_threshold_receipt(
        cast(str, aggregate["payload_sha256"]),
        "a" * 64,
    )
    result = calibration._threshold_freeze_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        readiness_receipt_sha256="a" * 64,
        threshold_exact_input_binding=_synthetic_threshold_exact_input_binding(
            cast(str, aggregate["payload_sha256"])
        ),
        threshold_worker_readiness_certification_binding=_readiness_certification_binding(),
        threshold_worker_provenance={},
        threshold_zip_provenance_attestation={},
    )
    stdout = calibration.THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    bundle = SimpleNamespace(
        receipt_sha256="a" * 64,
        execution_genesis_sha256="b" * 64,
        payload={"body": {"runtime_identity": {}}},
    )
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=b""),
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_input_publication_guard",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: aggregate["readiness_binding"],
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_validate_threshold_worker_provenance_bindings",
        lambda *a: (_ for _ in ()).throw(
            calibration.CalibrationError("threshold worker provenance differs")
        ),
    )
    with pytest.raises(calibration.CalibrationError, match="worker provenance differs"):
        calibration.freeze_and_publish_completed_calibration_thresholds(
            readiness_directory=readiness,
            shard_publication_root=shards,
            managed_ledger_directory=ledger,
            aggregate_publication_root=aggregates,
            aggregate_payload_sha256=cast(str, aggregate["payload_sha256"]),
            threshold_receipt_publication_root=thresholds,
            authorize_publication=True,
        )
    assert not tuple(thresholds.iterdir())


def test_threshold_parent_rechecks_live_inputs_immediately_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    threshold_receipts = tmp_path / "thresholds"
    for path in (readiness, shards, ledger, aggregates, threshold_receipts):
        path.mkdir()
    aggregate = calibration._payload_with_digest(
        {
            "schema": calibration.CALIBRATION_AGGREGATE_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_accepted": False,
            "thresholds_frozen": False,
            "promotion_artifact": False,
            "readiness_binding": {"readiness_receipt_sha256": "a" * 64},
            "managed_ledger_content_address": "b" * 64,
        }
    )
    calibration._write_new_immutable(
        aggregates,
        f"{aggregate['payload_sha256']}.json",
        canonical_json_bytes(aggregate),
        max_bytes=calibration._MAX_AGGREGATE_BYTES,
        label="synthetic aggregate",
    )
    aggregate_digest = cast(str, aggregate["payload_sha256"])
    expected_binding = _synthetic_threshold_exact_input_binding(aggregate_digest)
    receipt = _synthetic_threshold_receipt(aggregate_digest, "a" * 64, rejection=True)
    result = calibration._threshold_freeze_worker_result(
        calibration_aggregate=aggregate,
        threshold_freeze_receipt=receipt,
        readiness_receipt_sha256="a" * 64,
        threshold_exact_input_binding=expected_binding,
        threshold_worker_readiness_certification_binding={},
        threshold_worker_provenance={},
        threshold_zip_provenance_attestation={},
    )
    bundle = SimpleNamespace(
        receipt_sha256="a" * 64,
        execution_genesis_sha256="b" * 64,
        payload={"body": {"runtime_identity": {}}},
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: aggregate["readiness_binding"],
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_validate_threshold_worker_provenance_bindings",
        lambda *a: None,
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"synthetic": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"synthetic": "inventory"},
    )
    drifted_binding = {
        **expected_binding,
        "managed_ledger_inventory_sha256": "9" * 64,
    }
    live_bindings = iter((expected_binding, drifted_binding))
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: next(live_bindings),
    )
    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        lambda *a, **k: receipt,
    )
    monkeypatch.setattr(calibration, "require_current_full_runtime_identity", lambda value: None)
    with pytest.raises(calibration.CalibrationError, match="input changed before"):
        calibration._verify_and_install_threshold_freeze_worker_result(
            result=result,
            bundle=cast(Any, bundle),
            readiness_directory=readiness,
            shard_publication_root=shards,
            managed_ledger_directory=ledger,
            aggregate_publication_root=aggregates,
            aggregate_payload_sha256=aggregate_digest,
            threshold_receipt_publication_root=threshold_receipts,
        )
    assert not tuple(threshold_receipts.iterdir())


def test_protected_plan_parent_rejects_second_live_input_snapshot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    plans = tmp_path / "plans"
    for path in (readiness, shards, ledger, aggregates, thresholds, plans):
        path.mkdir()
    readiness_digest = "a" * 64
    aggregate = calibration._payload_with_digest(
        {
            "readiness_binding": {"readiness_receipt_sha256": readiness_digest},
            "managed_ledger_content_address": "b" * 64,
        }
    )
    aggregate_digest = cast(str, aggregate["payload_sha256"])
    receipt = _synthetic_successful_protected_receipt(
        aggregate_digest,
        readiness_digest,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    result = _synthetic_protected_worker_result(
        aggregate,
        receipt,
        plan,
        readiness_receipt_sha256=readiness_digest,
    )
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=readiness_digest,
        source_archive_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
        execution_genesis_sha256="b" * 64,
    )
    expected = _synthetic_threshold_exact_input_binding(aggregate_digest)
    drifted = {**expected, "managed_ledger_inventory_sha256": "f" * 64}
    bindings = iter((expected, drifted))
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_calibration_aggregate",
        lambda *a, **k: aggregate,
    )
    monkeypatch.setattr(
        calibration,
        "_load_content_addressed_threshold_freeze_receipt",
        lambda *a, **k: receipt,
    )
    monkeypatch.setattr(
        calibration,
        "_readiness_binding",
        lambda value: {"readiness_receipt_sha256": readiness_digest},
    )
    monkeypatch.setattr(calibration, "_validate_aggregate_provenance_bindings", lambda *a: None)
    monkeypatch.setattr(
        calibration,
        "_validate_protected_plan_worker_provenance_bindings",
        lambda *a: None,
    )
    monkeypatch.setattr(
        calibration,
        "validate_hidden_regime_factorial_threshold_freeze_receipt",
        lambda *a, **k: receipt,
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: ({"test_only": "shard"},),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: {"test_only": "ledger"},
    )
    monkeypatch.setattr(
        calibration,
        "_threshold_freeze_exact_input_binding",
        lambda *a, **k: next(bindings),
    )
    monkeypatch.setattr(calibration, "require_current_full_runtime_identity", lambda value: None)
    monkeypatch.setattr(
        calibration,
        "_install_verified_protected_plan_new_only",
        lambda *a, **k: pytest.fail("drifted inputs reached protected-plan installation"),
    )
    with pytest.raises(calibration.CalibrationError, match="input changed before"):
        calibration._verify_and_install_protected_plan_worker_result(
            result=result,
            bundle=cast(Any, bundle),
            readiness_directory=readiness,
            shard_publication_root=shards,
            managed_ledger_directory=ledger,
            aggregate_publication_root=aggregates,
            aggregate_payload_sha256=aggregate_digest,
            threshold_receipt_publication_root=thresholds,
            threshold_receipt_payload_sha256=cast(
                str,
                receipt["receipt_payload_sha256"],
            ),
            protected_plan_publication_root=plans,
        )
    assert not tuple(plans.iterdir())


def test_threshold_output_root_cannot_overlap_any_input_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    for path in (readiness, shards, ledger, aggregates):
        path.mkdir()
    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: pytest.fail("overlapping threshold root launched a worker"),
    )
    for output_root, label in (
        (readiness, "readiness publication"),
        (shards, "shard publication"),
        (ledger, "managed ledger"),
        (aggregates, "aggregate publication"),
    ):
        with pytest.raises(calibration.CalibrationError, match=label):
            calibration.freeze_and_publish_completed_calibration_thresholds(
                readiness_directory=readiness,
                shard_publication_root=shards,
                managed_ledger_directory=ledger,
                aggregate_publication_root=aggregates,
                aggregate_payload_sha256="a" * 64,
                threshold_receipt_publication_root=output_root,
                authorize_publication=True,
            )


def test_protected_plan_output_root_cannot_overlap_or_symlink_any_input_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    shards = tmp_path / "shards"
    ledger = tmp_path / "ledger"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    plans = tmp_path / "plans"
    for path in (readiness, shards, ledger, aggregates, thresholds, plans):
        path.mkdir()
    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: pytest.fail("invalid protected-plan root launched a worker"),
    )
    for output_root, label in (
        (readiness, "readiness publication"),
        (shards, "shard publication"),
        (ledger, "managed ledger"),
        (aggregates, "aggregate publication"),
        (thresholds, "threshold receipt publication"),
    ):
        with pytest.raises(calibration.CalibrationError, match=label):
            calibration.derive_and_publish_completed_calibration_protected_plan(
                readiness_directory=readiness,
                shard_publication_root=shards,
                managed_ledger_directory=ledger,
                aggregate_publication_root=aggregates,
                aggregate_payload_sha256="a" * 64,
                threshold_receipt_publication_root=thresholds,
                threshold_receipt_payload_sha256="b" * 64,
                protected_plan_publication_root=output_root,
                authorize_publication=True,
            )

    real_output = tmp_path / "real-plan-output"
    real_output.mkdir()
    symlinked_output = tmp_path / "symlinked-plan-output"
    symlinked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="symlinked directory component"):
        calibration.derive_and_publish_completed_calibration_protected_plan(
            readiness_directory=readiness,
            shard_publication_root=shards,
            managed_ledger_directory=ledger,
            aggregate_publication_root=aggregates,
            aggregate_payload_sha256="a" * 64,
            threshold_receipt_publication_root=thresholds,
            threshold_receipt_payload_sha256="b" * 64,
            protected_plan_publication_root=symlinked_output,
            authorize_publication=True,
        )


def test_protected_plan_parent_fails_without_output_when_mutation_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_digest = "a" * 64
    readiness = tmp_path / "readiness"
    shard_root = tmp_path / "shards"
    shard_directory = shard_root / readiness_digest
    ledger = tmp_path / "ledger"
    cases = ledger / "cases"
    aggregates = tmp_path / "aggregates"
    thresholds = tmp_path / "thresholds"
    plans = tmp_path / "plans"
    readiness.mkdir()
    shard_directory.mkdir(parents=True)
    cases.mkdir(parents=True)
    aggregates.mkdir()
    thresholds.mkdir()
    plans.mkdir()
    for case_index in range(calibration.EXPECTED_CASES):
        (cases / f"case-{case_index:03d}").mkdir()
    aggregate = calibration._payload_with_digest(
        {"readiness_binding": {"readiness_receipt_sha256": readiness_digest}}
    )
    receipt = _synthetic_successful_protected_receipt(
        cast(str, aggregate["payload_sha256"]),
        readiness_digest,
    )
    plan = _synthetic_protected_plan(aggregate, receipt)
    result = _synthetic_protected_worker_result(
        aggregate,
        receipt,
        plan,
        readiness_receipt_sha256=readiness_digest,
    )
    stdout = calibration.PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(
        canonical_json_bytes(result)
    )
    monkeypatch.setattr(
        calibration,
        "load_validated_readiness_bundle",
        lambda *a, **k: SimpleNamespace(receipt_sha256=readiness_digest),
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=b""),
    )
    locked_case_fd = os.open(cases / "case-007", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(locked_case_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(calibration.CalibrationError, match="active execution or mutation"):
            calibration.derive_and_publish_completed_calibration_protected_plan(
                readiness_directory=readiness,
                shard_publication_root=shard_root,
                managed_ledger_directory=ledger,
                aggregate_publication_root=aggregates,
                aggregate_payload_sha256=cast(str, aggregate["payload_sha256"]),
                threshold_receipt_publication_root=thresholds,
                threshold_receipt_payload_sha256=cast(
                    str,
                    receipt["receipt_payload_sha256"],
                ),
                protected_plan_publication_root=plans,
                authorize_publication=True,
            )
    finally:
        fcntl.flock(locked_case_fd, fcntl.LOCK_UN)
        os.close(locked_case_fd)
    assert not tuple(plans.iterdir())


def test_threshold_input_guard_fails_without_output_when_mutation_lock_is_held(
    tmp_path: Path,
) -> None:
    readiness_receipt_sha256 = "a" * 64
    shard_root = tmp_path / "shards"
    shard_directory = shard_root / readiness_receipt_sha256
    ledger = tmp_path / "ledger"
    cases = ledger / "cases"
    threshold_receipts = tmp_path / "thresholds"
    shard_directory.mkdir(parents=True)
    cases.mkdir(parents=True)
    threshold_receipts.mkdir()
    for case_index in range(calibration.EXPECTED_CASES):
        (cases / f"case-{case_index:03d}").mkdir()

    locked_case_fd = os.open(cases / "case-007", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(locked_case_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(calibration.CalibrationError, match="active execution or mutation"):
            with calibration._threshold_input_publication_guard(
                shard_publication_root=shard_root,
                readiness_receipt_sha256=readiness_receipt_sha256,
                managed_ledger_directory=ledger,
            ):
                pytest.fail("guard entered while an execution mutation lock was held")
    finally:
        fcntl.flock(locked_case_fd, fcntl.LOCK_UN)
        os.close(locked_case_fd)
    assert not tuple(threshold_receipts.iterdir())

    first_case_fd = os.open(cases / "case-000", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(first_case_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(first_case_fd, fcntl.LOCK_UN)
        os.close(first_case_fd)


def test_exact_240_ledger_rejects_missing_and_duplicate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(calibration.CalibrationError, match="exactly 240"):
        calibration._validated_shard_bodies([])
    duplicate = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": {}}
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    with pytest.raises(calibration.CalibrationError, match="duplicate shard payload"):
        calibration._validated_shard_bodies([copy.deepcopy(duplicate) for _ in range(240)])


def test_execution_record_binding_includes_direct_component_digests() -> None:
    inventory = {
        "genesis_sha256": "1" * 64,
        "started_records": [
            {
                "case_index": 7,
                "started_record_sha256": "2" * 64,
                "zip_provenance_binding_sha256": "8" * 64,
                "zip_provenance_attestation_sha256": "9" * 64,
            }
        ],
        "attempt_records": [
            {
                "case_index": 7,
                "managed_execution_attempt_count": 1,
                "attempt_records_sha256": calibration.canonical_sha256(
                    [
                        {
                            "attempt_index": 0,
                            "execution_mode": "first_execution",
                            "attempt_record_schema": CALIBRATION_EXECUTION_STARTED_SCHEMA,
                            "attempt_record_sha256": "2" * 64,
                            "attempt_request_payload_sha256": "b" * 64,
                            "exact_replay_consent": False,
                            "zip_provenance_attestation_sha256": "9" * 64,
                        }
                    ]
                ),
                "attempts": [
                    {
                        "attempt_index": 0,
                        "execution_mode": "first_execution",
                        "attempt_record_schema": CALIBRATION_EXECUTION_STARTED_SCHEMA,
                        "attempt_record_sha256": "2" * 64,
                        "attempt_request_payload_sha256": "b" * 64,
                        "exact_replay_consent": False,
                        "zip_provenance_attestation_sha256": "9" * 64,
                    }
                ],
            }
        ],
        "completed_records": [
            {
                "case_index": 7,
                "started_record_sha256": "2" * 64,
                "completed_record_sha256": "3" * 64,
                "summary_sha256": "4" * 64,
                "resource_sha256": "5" * 64,
                "primitive_trace_sha256": "6" * 64,
                "final_state_sha256": "a" * 64,
                "outcome_sha256": "7" * 64,
            }
        ],
    }
    assert calibration._execution_record_binding(inventory, 7) == {
        "case_index": 7,
        "genesis_sha256": "1" * 64,
        "started_record_sha256": "2" * 64,
        "completed_record_sha256": "3" * 64,
        "summary_sha256": "4" * 64,
        "resource_sha256": "5" * 64,
        "primitive_trace_sha256": "6" * 64,
        "final_state_sha256": "a" * 64,
        "outcome_sha256": "7" * 64,
        "managed_execution_attempt_count": 1,
        "attempt_records_sha256": inventory["attempt_records"][0]["attempt_records_sha256"],
        "zip_provenance_binding_sha256": "8" * 64,
        "zip_provenance_attestation_sha256": "9" * 64,
    }


def test_worker_attests_zip_then_validates_audits_and_finalizes_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = calibration._design()
    case = design.cases[0]
    readiness = {
        "source_archive_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
    }
    request = calibration.CalibrationCaseRequest(
        case_index=0,
        case_binding=case.to_payload(),
        readiness_binding=readiness,
        managed_ledger_genesis_sha256="c" * 64,
        allow_exact_replay=False,
        explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
    )
    bundle = SimpleNamespace(payload={"body": {"runtime_identity": {}}})
    authorization = object()
    provenance_capability = object()
    run = object()
    audit = object()
    shard = calibration._payload_with_digest({"schema": "synthetic", "case": case.to_payload()})
    events: list[str] = []
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    runtime_execution_identity = {"schema": "synthetic-runtime"}
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: runtime_execution_identity,
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: dict(runtime_execution_identity),
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_request",
        lambda payload, validated_bundle: request,
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda directory: {"genesis_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda archive, validated_bundle: events.append("runner_provenance") or {},
    )
    monkeypatch.setattr(calibration, "_read_regular_file", lambda *a, **k: b"source-zip")
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: events.append("governance_provenance") or provenance_capability,
    )

    def issue(**kwargs: object) -> object:
        assert kwargs["zip_provenance_capability"] is provenance_capability
        events.append("issue")
        return authorization

    monkeypatch.setattr(calibration, "issue_calibration_execution_authorization", issue)
    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: events.append("run") or run,
    )
    monkeypatch.setattr(
        calibration,
        "_execution_record_binding",
        lambda inventory, case_index: events.append("completion_binding") or {},
    )
    monkeypatch.setattr(
        calibration,
        "audit_hidden_regime_run_result",
        lambda result: events.append("audit") or audit,
    )
    monkeypatch.setattr(
        calibration,
        "extract_calibration_case_shard",
        lambda *a, **k: events.append("extract") or shard,
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **kwargs: events.append("validate") or dict(payload),
    )
    monkeypatch.setattr(
        calibration,
        "require_current_full_runtime_identity",
        lambda runtime: events.append("full_runtime"),
    )

    def finalize(auth: object, **kwargs: object) -> dict[str, object]:
        assert auth is authorization
        assert kwargs["run_result"] is run
        assert kwargs["shard_payload"] == shard
        events.append("finalize")
        return {}

    monkeypatch.setattr(calibration, "finalize_calibration_case_shard", finalize)
    monkeypatch.setattr(
        calibration,
        "load_finalized_calibration_case_shard",
        lambda directory, case_index: events.append("recover") or shard,
    )
    result = calibration._worker_case(
        readiness_directory=tmp_path,
        ledger_directory=tmp_path,
        request_payload={},
    )
    assert result == shard
    assert events == [
        "runner_provenance",
        "governance_provenance",
        "issue",
        "run",
        "completion_binding",
        "audit",
        "extract",
        "validate",
        "full_runtime",
        "finalize",
        "recover",
    ]

    events.clear()

    def reject_full_runtime(runtime: object) -> None:
        del runtime
        events.append("full_runtime")
        raise calibration.ReadinessError("synthetic complete-runtime drift")

    monkeypatch.setattr(
        calibration,
        "require_current_full_runtime_identity",
        reject_full_runtime,
    )
    with pytest.raises(calibration.ReadinessError, match="complete-runtime drift"):
        calibration._worker_case(
            readiness_directory=tmp_path,
            ledger_directory=tmp_path,
            request_payload={},
        )
    assert "full_runtime" in events
    assert "finalize" not in events
    assert "recover" not in events


def test_worker_aggregate_attests_zip_and_full_scans_before_returning_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_sha256 = "a" * 64
    genesis_sha256 = "b" * 64
    readiness = {"readiness_receipt_sha256": receipt_sha256}
    bundle = SimpleNamespace(
        payload={"body": {"runtime_identity": {}}},
        receipt_sha256=receipt_sha256,
    )
    shards = ({"synthetic": "shard"},)
    inventory = {"synthetic": "inventory"}
    aggregate = {"synthetic": "aggregate"}
    worker_provenance = {"synthetic": "worker-provenance"}
    zip_attestation = {"synthetic": "zip-attestation"}
    certification_binding = _readiness_certification_binding()
    validated = calibration._payload_with_digest(
        {
            "managed_ledger_content_address": genesis_sha256,
            "case_count": calibration.EXPECTED_CASES,
            "readiness_binding": readiness,
            "aggregation_worker_provenance": worker_provenance,
            "aggregation_zip_provenance_attestation": zip_attestation,
        }
    )
    events: list[str] = []

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: {"runtime": "expected"},
    )

    def cheap_runtime() -> dict[str, str]:
        events.append("cheap_runtime")
        return {"runtime": "expected"}

    monkeypatch.setattr(calibration, "build_runtime_execution_identity", cheap_runtime)
    monkeypatch.setattr(calibration, "_readiness_binding", lambda value: readiness)
    monkeypatch.setattr(
        calibration,
        "_aggregation_readiness_certification_binding",
        lambda value: events.append("certification_binding") or certification_binding,
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda *a, **k: events.append("runner_provenance") or worker_provenance,
    )
    monkeypatch.setattr(
        calibration,
        "_read_regular_file",
        lambda *a, **k: events.append("read_archive") or b"source-zip",
    )
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: events.append("governance_provenance")
        or SimpleNamespace(payload=zip_attestation),
    )
    monkeypatch.setattr(
        calibration,
        "load_complete_calibration_case_shards",
        lambda *a, **k: events.append("load_shards") or shards,
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda path: events.append("snapshot") or inventory,
    )
    monkeypatch.setattr(
        calibration,
        "aggregate_hidden_regime_factorial_calibration",
        lambda *a, **k: events.append("aggregate") or aggregate,
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_aggregate",
        lambda *a, **k: events.append("validate") or validated,
    )
    monkeypatch.setattr(
        calibration,
        "_validate_aggregate_provenance_bindings",
        lambda *a, **k: events.append("validate_provenance"),
    )
    monkeypatch.setattr(
        calibration,
        "require_current_full_runtime_identity",
        lambda runtime: events.append("full_runtime"),
    )

    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: pytest.fail("aggregation executed a learner"),
    )
    result = calibration._worker_aggregate(
        readiness_directory=tmp_path / "readiness",
        ledger_directory=tmp_path / "ledger",
        shard_publication_root=tmp_path / "shards",
    )
    assert result == validated
    assert events == [
        "cheap_runtime",
        "certification_binding",
        "runner_provenance",
        "read_archive",
        "governance_provenance",
        "load_shards",
        "snapshot",
        "aggregate",
        "validate",
        "validate_provenance",
        "cheap_runtime",
        "full_runtime",
    ]


def test_finalized_case_recovers_without_replay_after_parent_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    shard = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": readiness}
    )
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: object())
    monkeypatch.setattr(calibration, "_readiness_binding", lambda bundle: readiness)
    monkeypatch.setattr(
        calibration,
        "_read_regular_file",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    inventory = {"finalized_case_indices": [0]}
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda directory: inventory,
    )
    monkeypatch.setattr(
        calibration,
        "require_valid_calibration_execution_inventory",
        lambda snapshot, directory: snapshot,
    )
    monkeypatch.setattr(
        calibration,
        "load_finalized_calibration_case_shard",
        lambda directory, case_index: shard,
    )
    monkeypatch.setattr(
        calibration,
        "validate_finalized_calibration_case_shard",
        lambda payload, **kwargs: dict(payload),
    )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        calibration,
        "publish_calibration_case_shard_new_only",
        lambda root, payload, **kwargs: published.append(dict(payload)) or tmp_path / "case.json",
    )
    monkeypatch.setattr(
        calibration,
        "execute_bound_calibration_worker",
        lambda *a, **k: pytest.fail("finalized recovery replayed the learner"),
    )
    result = calibration.run_calibration_case_subprocess(
        case_index=0,
        readiness_directory=tmp_path,
        managed_ledger_directory=tmp_path,
        shard_publication_root=tmp_path,
        explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
        authorize_calibration_execution=True,
    )
    assert result == shard
    assert published == [shard]


def test_rehashed_compact_audit_tamper_cannot_cross_finalization_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    audit = {
        "trace_audit_schema": "schema",
        "trace_audit_report_sha256": "1" * 64,
        "valid": True,
        "expected_steps": 16_528,
        "rows_checked": 16_528,
        "helper_transitions_checked": 16_528,
        "beneficiary_transitions_checked": 16_528,
        "world_transitions_checked": 16_528,
        "commit_lineages_checked": 1,
        "recurrence_records_checked": 1,
        "retention_aggregate_fields_checked": 1,
        "summary_fields_checked": 1,
        "resource_fields_checked": 1,
        "mismatch_count": 0,
        "mismatches_sha256": "2" * 64,
        "accepted_float32_contraction_count": 1,
        "accepted_float32_contractions_sha256": "3" * 64,
        "unobserved_transition_fields": [],
        "evidence_boundary_sha256": "4" * 64,
        "lineage_oracle_schema": "lineage",
        "lineage_oracle_valid": True,
        "lineage_oracle_mismatches_sha256": "5" * 64,
        "lineage_commit_lineages_checked": 1,
        "lineage_recurrence_records_checked": 1,
        "lineage_aggregate_fields_checked": 1,
        "audited_summary_sha256": "6" * 64,
    }
    finalized = calibration._payload_with_digest(
        {
            "case": {"case_index": 0},
            "readiness_binding": readiness,
            "audit": audit,
        }
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **kwargs: dict(payload),
    )
    monkeypatch.setattr(
        calibration,
        "load_finalized_calibration_case_shard",
        lambda directory, case_index: finalized,
    )

    def mutate(value: object) -> object:
        if type(value) is bool:
            return not value
        if type(value) is int:
            return value + 1
        if type(value) is str:
            return ("f" if value != "f" * 64 else "e") * len(value)
        if type(value) is list:
            return ["tampered"]
        raise AssertionError(type(value))

    for field in audit:
        tampered_body = calibration._validate_payload_digest(finalized, "finalized shard")
        tampered_audit = copy.deepcopy(cast(dict[str, object], tampered_body["audit"]))
        tampered_audit[field] = mutate(tampered_audit[field])
        tampered = calibration._payload_with_digest({**tampered_body, "audit": tampered_audit})
        with pytest.raises(calibration.CalibrationError, match="post-audit finalization"):
            calibration.validate_finalized_calibration_case_shard(
                tampered,
                expected_readiness_binding=readiness,
                managed_ledger_directory=tmp_path,
            )
        with pytest.raises(calibration.CalibrationError, match="post-audit finalization"):
            calibration.publish_calibration_case_shard_new_only(
                tmp_path,
                tampered,
                expected_readiness_binding=readiness,
                managed_ledger_directory=tmp_path,
            )


def test_compact_audit_binds_contraction_count_type_and_frozen_evidence_boundary() -> None:
    summary: dict[str, object] = {
        "commit_generation_lineages": [],
        "recurrence_retention": [],
    }
    aggregate_fields = len(calibration.dataclasses.fields(calibration.RetentionAggregateSummary))
    audit: dict[str, object] = {
        "trace_audit_schema": calibration.HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
        "trace_audit_report_sha256": "1" * 64,
        "valid": True,
        "expected_steps": calibration.EXPECTED_STEPS,
        "rows_checked": calibration.EXPECTED_STEPS,
        "helper_transitions_checked": calibration.EXPECTED_STEPS,
        "beneficiary_transitions_checked": calibration.EXPECTED_STEPS,
        "world_transitions_checked": calibration.EXPECTED_STEPS,
        "commit_lineages_checked": 0,
        "recurrence_records_checked": 0,
        "retention_aggregate_fields_checked": aggregate_fields,
        "summary_fields_checked": len(
            calibration.dataclasses.fields(calibration.HiddenRegimeRunSummary)
        ),
        "resource_fields_checked": len(
            calibration.dataclasses.fields(calibration.HiddenRegimeResourceReport)
        ),
        "mismatch_count": 0,
        "mismatches_sha256": calibration.canonical_sha256([]),
        "accepted_float32_contraction_count": 0,
        "accepted_float32_contractions_sha256": calibration.canonical_sha256([]),
        "unobserved_transition_fields": [],
        "evidence_boundary_sha256": calibration.hashlib.sha256(
            calibration.EVIDENCE_BOUNDARY.encode("utf-8")
        ).hexdigest(),
        "lineage_oracle_schema": calibration.HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
        "lineage_oracle_valid": True,
        "lineage_oracle_mismatches_sha256": calibration.canonical_sha256([]),
        "lineage_commit_lineages_checked": 0,
        "lineage_recurrence_records_checked": 0,
        "lineage_aggregate_fields_checked": aggregate_fields,
        "audited_summary_sha256": calibration.canonical_sha256(calibration._encode_exact(summary)),
    }
    calibration._validate_audit_payload(audit, summary)
    wrong_count = {**audit, "accepted_float32_contraction_count": True}
    with pytest.raises(calibration.CalibrationError, match="strict integer"):
        calibration._validate_audit_payload(wrong_count, summary)
    wrong_boundary = {**audit, "evidence_boundary_sha256": "f" * 64}
    with pytest.raises(calibration.CalibrationError, match="evidence boundary"):
        calibration._validate_audit_payload(wrong_boundary, summary)


def test_batch_coordinator_refuses_defaults_before_touching_readiness(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_cases_subprocess(
            case_indices=range(2),
            max_workers=1,
            readiness_directory=tmp_path / "missing",
            managed_ledger_directory=tmp_path / "missing",
            shard_publication_root=tmp_path,
        )


def test_batch_coordinator_returns_caller_order_without_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def run_case(**kwargs: object) -> dict[str, object]:
        index = cast(int, kwargs["case_index"])
        calls.append(index)
        return {"case_index": index}

    monkeypatch.setattr(calibration, "run_calibration_case_subprocess", run_case)
    monkeypatch.setattr(
        calibration,
        "bound_calibration_runtime_batch",
        lambda *args, **kwargs: nullcontext(object()),
    )
    shards = calibration.run_calibration_cases_subprocess(
        case_indices=(2, 0, 1),
        max_workers=2,
        readiness_directory=tmp_path,
        managed_ledger_directory=tmp_path,
        shard_publication_root=tmp_path,
        explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
        authorize_calibration_execution=True,
    )
    assert tuple(item["case_index"] for item in shards) == (2, 0, 1)
    assert sorted(calls) == [0, 1, 2]


def test_batch_failure_does_not_start_a_later_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def run_case(**kwargs: object) -> dict[str, object]:
        index = cast(int, kwargs["case_index"])
        calls.append(index)
        if index == 1:
            raise RuntimeError("synthetic failure")
        return {"case_index": index}

    monkeypatch.setattr(calibration, "run_calibration_case_subprocess", run_case)
    monkeypatch.setattr(
        calibration,
        "bound_calibration_runtime_batch",
        lambda *args, **kwargs: nullcontext(object()),
    )
    with pytest.raises(calibration.CalibrationError, match="without retry or substitution"):
        calibration.run_calibration_cases_subprocess(
            case_indices=(0, 1, 2),
            max_workers=2,
            readiness_directory=tmp_path,
            managed_ledger_directory=tmp_path,
            shard_publication_root=tmp_path,
            explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
            authorize_calibration_execution=True,
        )
    assert sorted(calls) == [0, 1]


def test_preflight_refuses_defaults_before_touching_readiness(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_preflight_subprocess(
            readiness_directory=tmp_path / "missing",
            managed_ledger_directory=tmp_path / "missing",
        )


def test_preflight_uses_one_two_scan_runtime_batch_without_eager_full_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(receipt_sha256="c" * 64)
    request = calibration._payload_with_digest(
        {"managed_ledger_genesis_sha256": "a" * 64}
    )
    report = calibration._payload_with_digest({"inventory_after_sha256": "b" * 64})
    load_calls: list[dict[str, object]] = []
    batch_events: list[str] = []
    guard = object()

    def load_bundle(*args: object, **kwargs: object) -> object:
        del args
        load_calls.append(dict(kwargs))
        return bundle

    @contextmanager
    def runtime_batch(*args: object, **kwargs: object) -> Any:
        del args
        assert kwargs == {"authorize_batch_execution": True}
        batch_events.append("before_full_scan")
        try:
            yield guard
        finally:
            batch_events.append("after_full_scan")

    def execute(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        assert kwargs["runtime_batch_guard"] is guard
        assert batch_events == ["before_full_scan"]
        return SimpleNamespace(returncode=0, stdout=b"report", stderr=b"")

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", load_bundle)
    monkeypatch.setattr(calibration, "build_calibration_preflight_request", lambda *a, **k: request)
    monkeypatch.setattr(calibration, "bound_calibration_runtime_batch", runtime_batch)
    monkeypatch.setattr(calibration, "execute_bound_calibration_worker", execute)
    monkeypatch.setattr(calibration, "_parse_preflight_result", lambda stdout: report)
    monkeypatch.setattr(
        calibration,
        "validate_calibration_preflight_report",
        lambda payload, request_payload, validated_bundle: payload,
    )
    inventory = {"inventory_sha256": "b" * 64}
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda directory: inventory,
    )
    monkeypatch.setattr(
        calibration,
        "require_valid_calibration_execution_inventory",
        lambda snapshot, directory: snapshot,
    )
    monkeypatch.setattr(calibration, "_require_pristine_execution_inventory", lambda *a, **k: None)

    assert (
        calibration.run_calibration_preflight_subprocess(
            readiness_directory=tmp_path / "readiness",
            managed_ledger_directory=tmp_path / "ledger",
            explicit_acknowledgement=calibration.PREFLIGHT_ACKNOWLEDGEMENT,
            authorize_calibration_preflight=True,
        )
        == report
    )
    assert load_calls == [{"recheck_current": False, "recheck_runtime": False}]
    assert batch_events == ["before_full_scan", "after_full_scan"]


def test_preflight_case_rows_cover_all_frozen_bindings_without_seed_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"execution_governance": {"genesis_sha256": "a" * 64}}
    request = calibration._payload_with_digest(
        {
            "readiness_binding": readiness,
            "managed_ledger_genesis_sha256": "a" * 64,
        }
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_request",
        lambda payload, bundle: cast(Any, payload),
    )
    rows = calibration._preflight_case_binding_rows(request, cast(Any, object()))
    assert tuple(item["case_index"] for item in rows) == tuple(range(240))
    assert len({cast(str, item["case_request_payload_sha256"]) for item in rows}) == 240
    assert all(
        set(item)
        == {
            "case_index",
            "case_binding_sha256",
            "condition_runtime_binding_sha256",
            "manifest_binding_sha256",
            "recurrence_binding_sha256",
            "configuration_sha256",
            "seed_pair_binding_sha256",
            "case_request_binding_sha256",
            "case_request_payload_sha256",
        }
        for item in rows
    )


def test_zip_preflight_never_calls_learner_and_leaves_inventory_equal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "schema": calibration.CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "genesis_sha256": "a" * 64,
        "expected_case_count": 240,
        "started_case_indices": [],
        "completed_case_indices": [],
        "finalized_case_indices": [],
        "learner_interrupted_case_indices": [],
        "post_audit_unfinalized_case_indices": [],
        "started_record_count": 0,
        "completed_record_count": 0,
        "finalized_record_count": 0,
        "managed_execution_attempt_count": 0,
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": True,
        "started_records": [],
        "completed_records": [],
        "finalized_records": [],
        "attempt_records": [],
        "managed_boundary_scope": "synthetic",
        "inventory_sha256": "b" * 64,
    }
    request = calibration._payload_with_digest(
        {
            "managed_ledger_genesis_sha256": "a" * 64,
            "pristine_inventory_sha256": "b" * 64,
            "issue_process_local_authorizations": True,
        }
    )
    bundle = type(
        "Bundle",
        (),
        {
            "payload": {"body": {"runtime_identity": {}}},
            "receipt_sha256": "c" * 64,
            "source_archive_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
        },
    )()
    rows = tuple(
        {
            "case_request_binding_sha256": f"{index:064x}",
            "case_request_payload_sha256": f"{index + 240:064x}",
        }
        for index in range(240)
    )
    issued: list[int] = []
    runtime_identity_calls = 0

    def current_runtime_identity() -> dict[str, str]:
        nonlocal runtime_identity_calls
        runtime_identity_calls += 1
        return {"runtime": "expected"}

    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: {"runtime": "expected"},
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        current_runtime_identity,
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_preflight_request",
        lambda payload, validated_bundle: dict(payload),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda directory: copy.deepcopy(inventory),
    )
    monkeypatch.setattr(
        calibration,
        "require_valid_calibration_execution_inventory",
        lambda snapshot, directory: dict(snapshot),
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda archive, validated_bundle: {"source_archive_sha256": "d" * 64},
    )
    monkeypatch.setattr(calibration, "_read_regular_file", lambda *a, **k: b"zip")
    monkeypatch.setattr(
        calibration,
        "attest_calibration_zip_provenance",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        calibration,
        "_preflight_case_binding_rows",
        lambda payload, validated_bundle: rows,
    )
    monkeypatch.setattr(
        calibration,
        "issue_calibration_execution_authorization",
        lambda **kwargs: issued.append(cast(int, kwargs["case_index"])),
    )
    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: pytest.fail("preflight called learner"),
    )
    report = calibration._worker_preflight(
        readiness_directory=tmp_path,
        ledger_directory=tmp_path,
        request_payload=request,
    )
    body = calibration._validate_payload_digest(report, "preflight report")
    assert issued == list(range(240))
    assert body["learner_execution_called"] is False
    assert body["outcome_observed"] is False
    assert body["authorization_material_serialized"] is False
    assert body["inventory_byte_equal_before_after"] is True
    assert runtime_identity_calls == 2

    runtime_identities = iter(({"runtime": "expected"}, {"runtime": "drifted"}))
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: next(runtime_identities),
    )
    with pytest.raises(calibration.CalibrationError, match="drifted during preflight"):
        calibration._worker_preflight(
            readiness_directory=tmp_path,
            ledger_directory=tmp_path,
            request_payload=request,
        )


def test_zip_preflight_rejects_runtime_execution_identity_before_other_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(payload={"body": {"runtime_identity": {}}})
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "runtime_execution_identity_from_receipt",
        lambda runtime: {"runtime": "expected"},
    )
    monkeypatch.setattr(
        calibration,
        "build_runtime_execution_identity",
        lambda: {"runtime": "drifted"},
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_preflight_request",
        lambda *a, **k: pytest.fail("preflight continued after runtime mismatch"),
    )
    with pytest.raises(calibration.CalibrationError, match="differs before preflight"):
        calibration._worker_preflight(
            readiness_directory=tmp_path,
            ledger_directory=tmp_path,
            request_payload={},
        )


def test_preflight_worker_provenance_requires_no_site_and_exact_bound_paths() -> None:
    bundle = type(
        "Bundle",
        (),
        {
            "source_archive_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
        },
    )()
    provenance: dict[str, object] = {
        "source_archive_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "loader": "zipimport.zipimporter",
        "source_zip_first": True,
        "mutable_project_path_count": 0,
        "fresh_empty_working_directory": True,
        "no_site_startup": True,
        "prebootstrap_pth_hook_absent": True,
        "receipt_bound_runtime_prefix": True,
        "exact_receipt_bound_site_search_paths": True,
        "dont_write_bytecode": True,
        "command_line_pycache_prefix": True,
        "pycache_prefix_fresh_empty_nonsymlink": True,
        "pycache_prefix_outside_bound_paths": True,
        "project_module_count": 1,
        "project_modules_sha256": "c" * 64,
    }
    assert calibration._validate_preflight_worker_provenance(provenance, cast(Any, bundle)) == (
        provenance
    )
    for field in (
        "no_site_startup",
        "prebootstrap_pth_hook_absent",
        "receipt_bound_runtime_prefix",
        "exact_receipt_bound_site_search_paths",
    ):
        tampered = {**provenance, field: False}
        with pytest.raises(calibration.CalibrationError, match=field):
            calibration._validate_preflight_worker_provenance(tampered, cast(Any, bundle))


def test_gate_matrix_decides_only_nonstatistical_audits_before_thresholds() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    levels = [
        {"condition": condition, "metric_id": metric.metric_id, "statistics": {}}
        for condition in CANONICAL_CONDITION_ORDER
        for metric in design.metrics
    ]
    estimands = [
        {
            "estimand_id": estimand.estimand_id,
            "metrics": [
                {"metric_id": metric_id, "statistics": {}} for metric_id in estimand.metrics
            ],
        }
        for estimand in design.factorial_estimands + design.control_estimands
    ]
    supports = [
        {
            "metric_id": support.metric_id,
            "estimand_id": support.estimand_id,
            "statistics": {},
        }
        for support in design.paired_population_support_metrics
    ]
    mandatory, descriptive = calibration._gate_result_matrix(
        design,
        levels,
        estimands,
        supports,
        _gate_audit_summary(),
    )
    assert len(mandatory) + len(descriptive) == len(design.gate_families)
    audit_gate = next(
        item
        for item in mandatory
        if item["gate_family_id"] == "mandatory_trace_and_lifecycle_audits"
    )
    assert audit_gate["decision"] == "passed_nonstatistical"
    assert audit_gate["threshold_status"] == "not_applicable_nonstatistical"
    assert len(cast(list[object], audit_gate["references"])) == len(design.audits)
    statistical = [item for item in mandatory if item is not audit_gate]
    assert all(item["decision"] == "not_evaluated_no_thresholds" for item in statistical)
    assert all(
        item["threshold_status"] == "unset_pending_consumed_calibration_outcomes"
        for item in statistical + descriptive
    )


def test_mandatory_audit_builder_has_exact_nonempty_scoped_references() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    summary = calibration._build_mandatory_audit_summary(
        design,
        _semantic_audit_shards(),
        _readiness_certification_binding(),
    )
    assert summary["decision"] == "passed_nonstatistical"
    assert summary["case_audit_reference_count"] == 240
    results = {
        cast(str, item["requirement_id"]): item
        for item in cast(list[dict[str, object]], summary["requirement_results"])
    }
    assert tuple(results) == tuple(item.requirement_id for item in design.audits)
    assert all(cast(int, item["required_reference_count"]) > 0 for item in results.values())
    assert results["both_roles_learning"]["required_reference_count"] == 120
    assert results["atomic_c_old_to_c_new_replacement"]["required_reference_count"] == 30
    assert results["atomic_c_old_to_c_new_replacement"]["descriptive_reference_count"] == 90
    assert results["d_short_non_displacement"]["required_reference_count"] == 30
    for requirement_id in (
        "decentralized_role_equivalence",
        "checkpoint_resume_equivalence",
    ):
        result = results[requirement_id]
        references = cast(list[dict[str, object]], result["required_references"])
        assert result["evaluation_mode"] == "readiness_certification_not_per_case_execution"
        assert len(references) == 1
        assert references[0]["kind"] == "readiness_certification"
        assert "case_index" not in references[0]
    immutability = cast(dict[str, object], summary["selective_immutability_result"])
    assert immutability["decision"] == "passed_nonstatistical"
    assert immutability["required_reference_count"] == 180


def test_false_required_atomic_predicate_invalidates_but_descriptive_we_does_not() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    shards = _semantic_audit_shards()
    false_atomic = {
        "c_old_to_c_new_replacement_count": 0,
        "c_old_to_c_new_target_slots": [],
        "c_old_to_c_new_generation_pairs": [],
        "c_old_to_c_new_exactly_one_target": False,
    }

    descriptive_only = dict(shards)
    descriptive_only[1] = _replace_semantic_summary(shards[1], **false_atomic)
    descriptive_summary = calibration._build_mandatory_audit_summary(
        design,
        descriptive_only,
        _readiness_certification_binding(),
    )
    assert descriptive_summary["decision"] == "passed_nonstatistical"

    required = dict(shards)
    required[0] = _replace_semantic_summary(shards[0], **false_atomic)
    rejected = calibration._build_mandatory_audit_summary(
        design,
        required,
        _readiness_certification_binding(),
    )
    assert rejected["decision"] == "invalid_calibration"
    assert rejected["failed_requirement_ids"] == ["atomic_c_old_to_c_new_replacement"]
    result = next(
        item
        for item in cast(list[dict[str, object]], rejected["requirement_results"])
        if item["requirement_id"] == "atomic_c_old_to_c_new_replacement"
    )
    assert result["decision"] == "invalid_calibration"
    assert result["failed_case_indices"] == [0]
    assert cast(list[object], result["required_references"])


def test_atomic_replacement_accepts_durable_slot_three_and_rejects_scratch_slot_zero() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    shards = _semantic_audit_shards()

    durable_slot_three = dict(shards)
    durable_slot_three[0] = _replace_semantic_summary(
        shards[0],
        c_old_to_c_new_target_slots=[3],
    )
    accepted = calibration._build_mandatory_audit_summary(
        design,
        durable_slot_three,
        _readiness_certification_binding(),
    )
    assert accepted["decision"] == "passed_nonstatistical"

    scratch_slot_zero = dict(shards)
    scratch_slot_zero[0] = _replace_semantic_summary(
        shards[0],
        c_old_to_c_new_target_slots=[0],
    )
    rejected = calibration._build_mandatory_audit_summary(
        design,
        scratch_slot_zero,
        _readiness_certification_binding(),
    )
    assert rejected["decision"] == "invalid_calibration"
    assert rejected["failed_requirement_ids"] == ["atomic_c_old_to_c_new_replacement"]

    zero_generation = dict(shards)
    zero_generation[0] = _replace_semantic_summary(
        shards[0],
        c_old_to_c_new_generation_pairs=[[0, 1]],
    )
    rejected_generation = calibration._build_mandatory_audit_summary(
        design,
        zero_generation,
        _readiness_certification_binding(),
    )
    assert rejected_generation["decision"] == "invalid_calibration"
    assert rejected_generation["failed_requirement_ids"] == [
        "atomic_c_old_to_c_new_replacement"
    ]


def test_d_short_requirement_cannot_pass_a_selective_durable_mutation() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    shards = _semantic_audit_shards()
    shards[0] = _replace_semantic_summary(
        shards[0],
        helper_selective_mutation_violations=1,
        selective_durable_bit_immutable_until_atomic_replacement=False,
    )
    rejected = calibration._build_mandatory_audit_summary(
        design,
        shards,
        _readiness_certification_binding(),
    )
    assert rejected["decision"] == "invalid_calibration"
    assert rejected["failed_requirement_ids"] == [
        "d_short_non_displacement",
        "complete_role_lifecycle_oracle",
    ]
