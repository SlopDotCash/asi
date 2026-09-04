"""Fail-closed evidence-artifact tests for the frozen recurring-feature gate."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import alberta_framework.evaluation.recurring_feature_artifact as recurring_artifact_module
import alberta_framework.evaluation.recurring_feature_cli as recurring_cli_module
from alberta_framework.evaluation.recurring_feature_artifact import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    artifact_json,
    build_recurring_feature_artifact,
    load_recurring_feature_artifact,
    scientific_payload_sha256,
    validate_recurring_feature_artifact,
)
from alberta_framework.evaluation.recurring_feature_cli import (
    main as recurring_feature_cli_main,
)
from alberta_framework.recurring_feature_gate import (
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    PHASE_TASKS,
    TASK_PAIRS,
    RecurringFeatureGateResult,
    RecurringFeatureSeedEvidence,
    run_recurring_feature_gate,
)

pytestmark = pytest.mark.scientific


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _scientific(artifact: dict[str, object]) -> dict[str, object]:
    return _as_dict(artifact["scientific_payload"])


def _rehash(artifact: dict[str, object]) -> None:
    digest = _as_dict(artifact["scientific_digest"])
    digest["sha256"] = scientific_payload_sha256(_scientific(artifact))


@pytest.fixture(scope="module")
def frozen_result() -> RecurringFeatureGateResult:
    """Run only the preregistered promoted schedule, once for this module."""

    return run_recurring_feature_gate()


@pytest.fixture(scope="module")
def frozen_artifact(frozen_result: RecurringFeatureGateResult) -> dict[str, object]:
    return build_recurring_feature_artifact(
        frozen_result,
        gate_wall_seconds=6.5,
        generated_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )


def test_frozen_artifact_is_valid_accepted_and_narrow(
    frozen_artifact: dict[str, object],
) -> None:
    validation = validate_recurring_feature_artifact(frozen_artifact)

    assert validation.valid
    assert validation.accepted
    assert validation.errors == ()
    assert frozen_artifact["schema_version"] == SCHEMA_VERSION
    scientific = _scientific(frozen_artifact)
    protocol = _as_dict(scientific["protocol"])
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    seed_roles = _as_dict(protocol["seed_roles"])
    assert seed_roles["development_and_threshold_calibration"] == list(DEVELOPMENT_SEEDS)
    assert seed_roles["promoted_held_out_evidence"] == list(EVIDENCE_SEEDS)
    assert set(DEVELOPMENT_SEEDS).isdisjoint(EVIDENCE_SEEDS)
    excluded = _as_list(protocol["excluded_claims"])
    assert "general feature discovery" in excluded
    assert "indefinite-memory learning" in excluded
    assert "completion of the Alberta Plan" in excluded
    provenance = _as_dict(scientific["source_provenance"])
    assert "not signatures" in str(provenance["interpretation"])
    assert "authenticity" in str(provenance["interpretation"])


def test_artifact_remains_valid_after_checkout_head_advances_without_source_changes(
    frozen_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committing the artifact must not invalidate its exact source-byte binding."""

    generation_head = str(_as_dict(_scientific(frozen_artifact)["source_provenance"])["git_head"])
    later_head = "1" * 40 if generation_head != "1" * 40 else "2" * 40
    monkeypatch.setattr(recurring_artifact_module, "_git_head", lambda: later_head)

    validation = validate_recurring_feature_artifact(frozen_artifact)

    assert validation.valid
    assert validation.accepted


def test_artifact_counts_archive_memory_and_preserves_primitive_seed_evidence(
    frozen_artifact: dict[str, object],
) -> None:
    scientific = _scientific(frozen_artifact)
    memory = _as_dict(scientific["memory_budget"])
    assert memory == {
        "active_pair_descriptor_slots": 3,
        "candidate_pair_descriptor_slots": 15,
        "total_pair_descriptor_slots": 18,
        "output_heads": 4,
        "active_output_weight_slots": 12,
        "candidate_output_weight_slots": 60,
        "total_output_weight_slots": 72,
        "candidate_archive_is_counted_memory": True,
    }
    summaries = _as_list(scientific["seed_summaries"])
    assert len(summaries) == 30
    assert [_as_dict(summary)["seed"] for summary in summaries] == list(EVIDENCE_SEEDS)
    critical_pairs = {tuple(pair) for pair in TASK_PAIRS[:3]}
    for summary_value in summaries:
        summary = _as_dict(summary_value)
        for variant_name in ("retained", "no_retention"):
            variant = _as_dict(summary[variant_name])
            candidates = _as_list(variant["candidate_pairs"])
            assert len(candidates) == 15
            assert len({tuple(_as_list(pair)) for pair in candidates}) == 15
            assert list(TASK_PAIRS[3]) in candidates
            phases = _as_list(variant["phase_evidence"])
            assert len(phases) == len(PHASE_TASKS)
            assert [_as_dict(phase)["task"] for phase in phases] == list(PHASE_TASKS)
            assert variant["steps_seen"] == 3_600
        retained = _as_dict(summary["retained"])
        assert {tuple(_as_list(pair)) for pair in _as_list(retained["active_pairs"])} == (
            critical_pairs
        )


def test_exact_heldout_aggregate_and_deterministic_paired_intervals(
    frozen_artifact: dict[str, object],
) -> None:
    aggregate = _as_dict(_scientific(frozen_artifact)["aggregate"])
    retained = _as_dict(aggregate["retained"])
    baseline = _as_dict(aggregate["no_retention"])
    assert retained["all_critical_retention_rate"] == 1.0
    assert baseline["all_critical_retention_rate"] == 0.0
    assert retained["obsolete_active_bank_eviction_rate"] == 1.0
    assert baseline["obsolete_active_bank_eviction_rate"] == 0.9
    assert _as_dict(retained["median_heldout_nmse_by_task"]) == pytest.approx(
        {
            "A": 0.003930832042281673,
            "B": 1.0998805432321312e-09,
            "C": 1.157750888133892e-13,
            "D": 1.0075389632299214,
        }
    )
    assert _as_dict(baseline["median_heldout_nmse_by_task"]) == pytest.approx(
        {
            "A": 1.0035049204860838,
            "B": 1.0167260865937773,
            "C": 0.02733832230147328,
            "D": 1.0000559292370013,
        }
    )
    assert retained["median_acquisition_recovery_steps"] == 112.5
    assert retained["median_recurrence_recovery_steps"] == 40.0

    effects = _as_dict(aggregate["paired_effects"])
    retention = _as_dict(effects["retention_rate_gain_over_no_retention"])
    assert (retention["estimate"], retention["lower"], retention["upper"]) == (
        1.0,
        1.0,
        1.0,
    )
    critical = _as_dict(effects["per_seed_maximum_critical_nmse_reduction"])
    assert critical["estimate"] == pytest.approx(1.0121495562829679)
    assert critical["lower"] == pytest.approx(1.0019829686174402)
    assert critical["upper"] == pytest.approx(1.0244659246323935)
    obsolete = _as_dict(effects["obsolete_nmse_increase_over_no_retention"])
    assert obsolete["estimate"] == pytest.approx(0.027811657222801912)
    assert obsolete["lower"] == pytest.approx(0.011941954070848035)
    assert obsolete["upper"] == pytest.approx(0.04664394406306361)
    recovery = _as_dict(effects["retained_acquisition_minus_recurrence_steps"])
    assert recovery["estimate"] == pytest.approx(71.9)
    assert recovery["lower"] == pytest.approx(67.63333333333334)
    assert recovery["upper"] == pytest.approx(76.40083333333332)
    assert all(_as_dict(interval)["sample_size"] == 30 for interval in effects.values())


def _without_recovery(seed: RecurringFeatureSeedEvidence) -> RecurringFeatureSeedEvidence:
    return replace(
        seed,
        phase_evidence=tuple(
            replace(phase, recovery_steps=None) for phase in seed.phase_evidence
        ),
        task_recovery=tuple(
            replace(
                recovery,
                acquisition_steps=None,
                recurrence_steps=tuple(None for _ in recovery.recurrence_steps),
            )
            for recovery in seed.task_recovery
        ),
    )


def test_never_recovered_seeds_build_instead_of_crashing(
    frozen_result: RecurringFeatureGateResult,
) -> None:
    """Never-recovered seeds are permitted gate evidence, not artifact build errors.

    One unrecovered seed shrinks the paired recovery interval to the seeds that
    did recover and the artifact stays fully valid. When no seed recovers the
    build used to raise from the paired bootstrap before anything was recorded;
    it must now produce the artifact with an explicit ``unavailable`` interval
    and a failed recurrence check, so the validator can name the missing
    finite evidence instead of the CLI dying with a traceback.
    """
    retained = frozen_result.retained
    first, *rest = retained.seeds

    one_unrecovered = replace(
        frozen_result, retained=replace(retained, seeds=(_without_recovery(first), *rest))
    )
    artifact = build_recurring_feature_artifact(
        one_unrecovered,
        gate_wall_seconds=6.5,
        generated_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )
    validation = validate_recurring_feature_artifact(artifact)
    assert validation.valid, validation.errors
    effects = _as_dict(_as_dict(_scientific(artifact)["aggregate"])["paired_effects"])
    recovery = _as_dict(effects["retained_acquisition_minus_recurrence_steps"])
    assert recovery["estimate"] is not None
    assert recovery["sample_size"] == len(retained.seeds) - 1

    none_recovered = replace(
        frozen_result,
        retained=replace(retained, seeds=tuple(_without_recovery(seed) for seed in retained.seeds)),
    )
    artifact = build_recurring_feature_artifact(
        none_recovered,
        gate_wall_seconds=6.5,
        generated_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )
    effects = _as_dict(_as_dict(_scientific(artifact)["aggregate"])["paired_effects"])
    recovery = _as_dict(effects["retained_acquisition_minus_recurrence_steps"])
    assert recovery["estimate"] is None
    assert recovery["sample_size"] == 0
    assert recovery["unavailable_reason"] == "no paired differences"
    checks = {
        _as_dict(check)["name"]: _as_dict(check)
        for check in _as_list(_as_dict(_scientific(artifact)["acceptance"])["checks"])
    }
    assert checks["recurrence_faster_than_acquisition"]["actual"] is None
    assert checks["recurrence_faster_than_acquisition"]["passed"] is False
    validation = validate_recurring_feature_artifact(artifact)
    assert not validation.accepted
    # v1 requires finite recovery medians, so the validator names that gap
    # rather than the build crashing before any artifact exists.
    assert not validation.valid
    assert any("recurrence_faster_than_acquisition" in error or "numeric comparison" in error
               for error in validation.errors), validation.errors


def test_paired_interval_reports_non_finite_differences_as_unavailable() -> None:
    """Non-finite paired differences (``inf`` NMSE) must not crash the builder."""
    unavailable = recurring_artifact_module._paired_interval_or_unavailable(
        [0.5, math.inf, -0.25], seed_offset=1
    )
    assert unavailable["estimate"] is None
    assert unavailable["sample_size"] == 3
    assert unavailable["finite_sample_size"] == 2
    assert unavailable["unavailable_reason"] == "non-finite paired differences"
    available = recurring_artifact_module._paired_interval_or_unavailable(
        [0.5, 0.25, -0.25], seed_offset=1
    )
    assert available == recurring_artifact_module.paired_bootstrap_mean_interval(
        [0.5, 0.25, -0.25], seed_offset=1
    )


def test_scientific_digest_is_deterministic_and_excludes_operational_metadata(
    frozen_result: RecurringFeatureGateResult,
    frozen_artifact: dict[str, object],
) -> None:
    later = build_recurring_feature_artifact(
        frozen_result,
        gate_wall_seconds=123.456,
        generated_at=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert _scientific(later) == _scientific(frozen_artifact)
    assert later["scientific_digest"] == frozen_artifact["scientific_digest"]
    assert later["operational_metadata"] != frozen_artifact["operational_metadata"]
    assert validate_recurring_feature_artifact(later).accepted
    digest = _as_dict(frozen_artifact["scientific_digest"])
    assert digest["sha256"] == scientific_payload_sha256(_scientific(frozen_artifact))


def test_strict_json_round_trip_and_nonstandard_numbers(
    frozen_artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evidence.json"
    artifact_path.write_text(artifact_json(frozen_artifact), encoding="utf-8")
    loaded = load_recurring_feature_artifact(artifact_path)
    assert loaded == frozen_artifact
    assert validate_recurring_feature_artifact(loaded).accepted

    invalid_path = tmp_path / "nan.json"
    invalid_path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_recurring_feature_artifact(invalid_path)


def test_unrehashed_scientific_tampering_fails_digest(
    frozen_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(frozen_artifact)
    aggregate = _as_dict(_scientific(tampered)["aggregate"])
    _as_dict(aggregate["retained"])["all_critical_retention_rate"] = 0.0

    validation = validate_recurring_feature_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert "scientific_digest.sha256 does not match scientific payload" in validation.errors


def test_rehashed_active_pair_claim_fabrication_fails_raw_reconstruction(
    frozen_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(frozen_artifact)
    first = _as_dict(_as_list(_scientific(fabricated)["seed_summaries"])[0])
    retained = _as_dict(first["retained"])
    retained["critical_pairs_retained"] = [False, False, False]
    retained["retained_all_critical_pairs"] = False
    _rehash(fabricated)

    validation = validate_recurring_feature_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("disagrees with active_pairs" in error for error in validation.errors)


def test_rehashed_phase_and_recovery_fabrication_fails_primitive_consistency(
    frozen_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(frozen_artifact)
    first = _as_dict(_as_list(_scientific(fabricated)["seed_summaries"])[0])
    retained = _as_dict(first["retained"])
    recovery = _as_dict(_as_list(retained["task_recovery"])[0])
    recovery["acquisition_steps"] = 1
    phase = _as_dict(_as_list(retained["phase_evidence"])[0])
    phase["prequential_nmse"] = -1.0
    _rehash(fabricated)

    validation = validate_recurring_feature_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("task_recovery is not derived" in error for error in validation.errors)
    assert any("prequential_nmse is invalid" in error for error in validation.errors)


def test_rehashed_schema_threshold_and_provenance_tampering_fail_closed(
    frozen_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(frozen_artifact)
    scientific = _scientific(fabricated)
    protocol = _as_dict(scientific["protocol"])
    protocol["unexpected"] = True
    provenance = _as_dict(scientific["source_provenance"])
    provenance["git_head"] = "0" * 40
    source_hashes = _as_dict(provenance["source_sha256"])
    first_source = next(iter(source_hashes))
    source_hashes[first_source] = "0" * 64
    acceptance = _as_dict(scientific["acceptance"])
    checks = _as_list(acceptance["checks"])
    retention_check = next(
        _as_dict(check)
        for check in checks
        if _as_dict(check)["name"] == "retained_all_critical_rate"
    )
    retention_check["threshold"] = 0.0
    _rehash(fabricated)

    validation = validate_recurring_feature_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("protocol is not canonical" in error for error in validation.errors)
    assert any("pinned source hashes" in error for error in validation.errors)
    assert any("frozen comparison" in error for error in validation.errors)


def test_rehashed_duplicate_and_underpowered_seed_evidence_fail_closed(
    frozen_result: RecurringFeatureGateResult,
    frozen_artifact: dict[str, object],
) -> None:
    duplicate = copy.deepcopy(frozen_artifact)
    summaries = _as_list(_scientific(duplicate)["seed_summaries"])
    _as_dict(summaries[1])["seed"] = _as_dict(summaries[0])["seed"]
    _rehash(duplicate)
    duplicate_validation = validate_recurring_feature_artifact(duplicate)
    assert not duplicate_validation.valid
    assert not duplicate_validation.accepted
    assert any("strictly increasing and unique" in error for error in duplicate_validation.errors)

    underpowered_result = replace(
        frozen_result,
        retained=replace(frozen_result.retained, seeds=frozen_result.retained.seeds[:3]),
        no_retention=replace(
            frozen_result.no_retention,
            seeds=frozen_result.no_retention.seeds[:3],
        ),
    )
    underpowered = build_recurring_feature_artifact(
        underpowered_result,
        gate_wall_seconds=0.0,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    underpowered_validation = validate_recurring_feature_artifact(underpowered)
    assert underpowered_validation.valid
    assert not underpowered_validation.accepted
    acceptance = _as_dict(_scientific(underpowered)["acceptance"])
    failed_names = {
        _as_dict(check)["name"]
        for check in _as_list(acceptance["checks"])
        if _as_dict(check)["passed"] is False
    }
    assert {
        "evidence_seed_schedule",
        "minimum_seed_count",
        "upstream_gate_decision",
    } <= failed_names


def test_operational_metadata_is_outside_digest_but_still_validated(
    frozen_artifact: dict[str, object],
) -> None:
    changed = copy.deepcopy(frozen_artifact)
    operational = _as_dict(changed["operational_metadata"])
    operational["gate_wall_seconds"] = 999.0
    operational["generated_at_utc"] = (
        datetime(2026, 7, 30, tzinfo=UTC) + timedelta(days=1)
    ).isoformat()
    assert changed["scientific_digest"] == frozen_artifact["scientific_digest"]
    assert validate_recurring_feature_artifact(changed).accepted

    operational["gate_wall_seconds"] = -1.0
    validation = validate_recurring_feature_artifact(changed)
    assert not validation.valid
    assert not validation.accepted


def test_cli_requires_explicit_output_before_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repository_root)
    pinned = recurring_cli_module.DEFAULT_OUTPUT.resolve()
    assert pinned.is_file()
    original = pinned.read_bytes()

    def forbidden_run() -> RecurringFeatureGateResult:
        raise AssertionError("existing output must be rejected before the gate runs")

    monkeypatch.setattr(recurring_cli_module, "run_recurring_feature_gate", forbidden_run)
    status = recurring_cli_module.main([])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "generation requires --output with a new path" in emitted["errors"][0]
    assert pinned.read_bytes() == original


def test_cli_refuses_missing_reserved_canonical_path_before_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reserved = tmp_path / "reserved" / "evidence.v1.json"
    assert not reserved.exists()
    monkeypatch.setattr(recurring_cli_module, "DEFAULT_OUTPUT", reserved)

    def forbidden_run() -> RecurringFeatureGateResult:
        raise AssertionError("the reserved canonical path must never be regenerated")

    monkeypatch.setattr(recurring_cli_module, "run_recurring_feature_gate", forbidden_run)
    status = recurring_cli_module.main(["--output", str(reserved)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "pinned canonical artifact path" in emitted["errors"][0]
    assert not reserved.exists()


def test_cli_refuses_arbitrary_existing_output_before_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "existing.json"
    sentinel = b"existing artifact must survive"
    path.write_bytes(sentinel)

    def forbidden_run() -> RecurringFeatureGateResult:
        raise AssertionError("existing output must be rejected before the gate runs")

    monkeypatch.setattr(recurring_cli_module, "run_recurring_feature_gate", forbidden_run)
    status = recurring_cli_module.main(["--output", str(path)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "existing output path" in emitted["errors"][0]
    assert path.read_bytes() == sentinel


def test_cli_atomically_writes_new_path_verifies_and_rejects_tampering_without_rerun(
    frozen_result: RecurringFeatureGateResult,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "accepted.json"
    linked_destinations: list[Path] = []
    real_link = os.link

    def observed_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert not destination_path.exists()
        assert source_path.read_text(encoding="utf-8").endswith("\n")
        linked_destinations.append(destination_path)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", observed_link)
    status = recurring_feature_cli_main(
        ["--output", str(path)],
        result=frozen_result,
        gate_wall_seconds=6.5,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["valid"] is True
    assert emitted["accepted"] is True
    assert emitted["seed_count"] == 30
    assert linked_destinations == [path.resolve()]
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    verify_status = recurring_feature_cli_main(["--verify", str(path)])
    verified = json.loads(capsys.readouterr().out)
    assert verify_status == 0
    assert verified["valid"] is True
    assert verified["accepted"] is True

    tampered = load_recurring_feature_artifact(path)
    _as_dict(_scientific(tampered)["aggregate"])["seed_count"] = 1
    path.write_text(artifact_json(tampered), encoding="utf-8")
    tampered_status = recurring_feature_cli_main(["--verify", str(path)])
    rejected = json.loads(capsys.readouterr().out)
    assert tampered_status == 2
    assert rejected["valid"] is False
    assert rejected["accepted"] is False


def test_cli_returns_rejection_for_underpowered_injected_evidence(
    frozen_result: RecurringFeatureGateResult,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    underpowered = replace(
        frozen_result,
        retained=replace(frozen_result.retained, seeds=frozen_result.retained.seeds[:3]),
        no_retention=replace(
            frozen_result.no_retention,
            seeds=frozen_result.no_retention.seeds[:3],
        ),
    )
    path = tmp_path / "underpowered.json"
    status = recurring_feature_cli_main(
        ["--output", str(path)],
        result=underpowered,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    emitted = json.loads(capsys.readouterr().out)
    assert status == 1
    assert emitted["valid"] is True
    assert emitted["accepted"] is False
    assert path.exists()


def test_cli_exposes_no_seed_or_threshold_retuning_options(
    frozen_result: RecurringFeatureGateResult,
) -> None:
    with pytest.raises(SystemExit):
        recurring_feature_cli_main(
            ["--seed-start", "0"],
            result=frozen_result,
        )
    with pytest.raises(SystemExit):
        recurring_feature_cli_main(
            ["--minimum-retained-all-critical-rate", "0"],
            result=frozen_result,
        )
