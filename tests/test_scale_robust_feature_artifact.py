"""Synthetic-only tests for the frozen-v2 scale-robust feature artifact.

These tests never execute the 30-seed JAX protocol.  They construct compact
phase-window records whose arithmetic is known exactly, then exercise
aggregation, bootstrap reconstruction, source/digest binding, strict JSON,
and the no-tuning CLI injection seam.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from alberta_framework.evaluation import (
    scale_robust_feature_artifact,
    scale_robust_feature_cli,
)
from alberta_framework.evaluation.scale_robust_feature import (
    CONDITION_LEGACY,
    CONDITION_NAMES,
    CONDITION_NO_RETENTION,
    CONDITION_PRIMARY,
    EVIDENCE_SEED_DERIVATION,
    EVIDENCE_SEED_NAMESPACE,
    EVIDENCE_SEEDS,
    EXPECTED_MEMORY,
    ConditionSeedRecord,
    PhaseWindowRecord,
    ScaleRobustFeatureReport,
    count_relevant_context_pairs,
    count_relevant_context_pairs_by_task,
    frozen_configuration_payload,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_RECORD_RELATIVE_PATH,
    CALIBRATION_RECORD_SHA256,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ArtifactValidation,
    artifact_json,
    build_evidence_artifact,
    current_source_fingerprint,
    load_evidence_artifact,
    paired_bootstrap_mean_interval,
    scientific_payload_sha256,
    threshold_calibration_ready,
    validate_evidence_artifact,
)
from alberta_framework.streams.gauntlet import SEGMENT_NAMES

pytestmark = pytest.mark.scientific


def _phase(
    phase_index: int,
    *,
    phase_mse: float = 1.0,
    early_mse: float = 1.0,
    tail_mse: float = 1.0,
    asymptotic_mse: float = 1.0,
    nonfinite_steps: int = 0,
) -> PhaseWindowRecord:
    return PhaseWindowRecord(
        phase_index=phase_index,
        phase_name=SEGMENT_NAMES[phase_index],
        step_count=3_000,
        nonfinite_steps=nonfinite_steps,
        phase_squared_error_sum=phase_mse * 3_000,
        early_squared_error_sum=early_mse * 200,
        early_count=200,
        tail_squared_error_sum=tail_mse * 500,
        tail_count=500,
        asymptotic_squared_error_sum=asymptotic_mse * 1_500,
        asymptotic_count=1_500,
    )


def _phase_program(condition: str) -> tuple[PhaseWindowRecord, ...]:
    parameters: dict[int, dict[str, float]] = {}
    if condition == CONDITION_PRIMARY:
        parameters = {
            2: {"early_mse": 10.0, "tail_mse": 0.5},
            3: {"early_mse": 12.0, "tail_mse": 0.7},
            4: {"early_mse": 1.0, "tail_mse": 0.5},
            5: {"early_mse": 1.5, "tail_mse": 0.6},
            6: {"phase_mse": 8.0, "early_mse": 20.0, "tail_mse": 4.0},
            7: {"asymptotic_mse": 0.05},
            8: {"early_mse": 1.0, "tail_mse": 0.05},
        }
    elif condition == CONDITION_LEGACY:
        parameters = {
            2: {"early_mse": 10.0, "tail_mse": 1.0},
            3: {"early_mse": 12.0, "tail_mse": 2.0},
            4: {"early_mse": 5.0, "tail_mse": 5.0},
            5: {"early_mse": 6.0, "tail_mse": 5.0},
            6: {"phase_mse": 80.0, "early_mse": 120.0, "tail_mse": 40.0},
            7: {"asymptotic_mse": 1.0},
            8: {"early_mse": 5.0, "tail_mse": 10.0},
        }
    elif condition == CONDITION_NO_RETENTION:
        parameters = {
            2: {"early_mse": 10.0, "tail_mse": 0.5},
            3: {"early_mse": 12.0, "tail_mse": 0.8},
            4: {"early_mse": 2.0, "tail_mse": 1.0},
            5: {"early_mse": 2.0, "tail_mse": 1.0},
            6: {"phase_mse": 10.0, "early_mse": 24.0, "tail_mse": 5.0},
            7: {"asymptotic_mse": 0.08},
            8: {"early_mse": 2.0, "tail_mse": 0.5},
        }
    else:
        raise AssertionError(f"unknown synthetic condition: {condition}")
    return tuple(_phase(index, **parameters.get(index, {})) for index in range(len(SEGMENT_NAMES)))


def _pairs(task_c_count: int, task_d_count: int) -> tuple[tuple[int, int], ...]:
    context_pairs = tuple((index, 12) for index in range(task_c_count)) + tuple(
        (index, 13) for index in range(task_d_count)
    )
    filler_candidates = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (0, 10),
        (0, 11),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 10),
        (2, 11),
    )
    return context_pairs + filler_candidates[: 24 - len(context_pairs)]


def _condition_pairs(
    condition: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    if condition == CONDITION_PRIMARY:
        return _pairs(8, 8), _pairs(7, 6), _pairs(8, 6)
    if condition == CONDITION_LEGACY:
        return _pairs(1, 1), _pairs(0, 1), _pairs(2, 0)
    if condition == CONDITION_NO_RETENTION:
        return _pairs(6, 5), _pairs(2, 2), _pairs(7, 2)
    raise AssertionError(f"unknown synthetic condition: {condition}")


def synthetic_report(
    seeds: tuple[int, ...] = EVIDENCE_SEEDS,
) -> ScaleRobustFeatureReport:
    records = tuple(
        ConditionSeedRecord(
            seed=seed,
            condition=condition,
            phases=_phase_program(condition),
            end_segment_5_active_pairs=_condition_pairs(condition)[0],
            end_segment_7_active_pairs=_condition_pairs(condition)[1],
            final_active_pairs=_condition_pairs(condition)[2],
        )
        for seed in seeds
        for condition in CONDITION_NAMES
    )
    return ScaleRobustFeatureReport(
        seeds=seeds,
        records=records,
        memory_by_condition=copy.deepcopy(EXPECTED_MEMORY),
        wall_time_seconds_by_condition={condition: 0.25 for condition in CONDITION_NAMES},
    )


def _report_with_primary_phase_field(
    *,
    phase_index: int,
    field: str,
    value: float,
    seeds: tuple[int, ...] = EVIDENCE_SEEDS,
) -> ScaleRobustFeatureReport:
    report = synthetic_report()
    changed_records: list[ConditionSeedRecord] = []
    for record in report.records:
        if record.condition != CONDITION_PRIMARY or record.seed not in seeds:
            changed_records.append(record)
            continue
        phases = list(record.phases)
        phases[phase_index] = dataclasses.replace(
            phases[phase_index],
            **{field: value},
        )
        changed_records.append(dataclasses.replace(record, phases=tuple(phases)))
    return dataclasses.replace(report, records=tuple(changed_records))


def _report_with_snapshot_pairs(
    *,
    condition: str,
    snapshot: str,
    task_c_count: int,
    task_d_count: int,
    seeds: tuple[int, ...] = EVIDENCE_SEEDS,
) -> ScaleRobustFeatureReport:
    report = synthetic_report()
    changed_records: list[ConditionSeedRecord] = []
    pairs = _pairs(task_c_count, task_d_count)
    for record in report.records:
        if record.condition != condition or record.seed not in seeds:
            changed_records.append(record)
        elif snapshot == "end_segment_5":
            changed_records.append(dataclasses.replace(record, end_segment_5_active_pairs=pairs))
        elif snapshot == "end_segment_7":
            changed_records.append(dataclasses.replace(record, end_segment_7_active_pairs=pairs))
        elif snapshot == "final":
            changed_records.append(dataclasses.replace(record, final_active_pairs=pairs))
        else:
            raise AssertionError(f"unknown snapshot: {snapshot}")
    return dataclasses.replace(report, records=tuple(changed_records))


def _acceptance_checks(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    scientific = artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    acceptance = scientific["acceptance"]
    assert isinstance(acceptance, dict)
    checks = acceptance["checks"]
    assert isinstance(checks, list)
    return {str(check["name"]): check for check in checks}


def _rehash(artifact: dict[str, object]) -> None:
    scientific = artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    digest = artifact["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = scientific_payload_sha256(scientific)


@pytest.fixture(scope="module")
def accepted_artifact() -> dict[str, object]:
    return build_evidence_artifact(synthetic_report())


def test_frozen_configuration_binds_direct_seed_keys_and_all_three_arms():
    assert scale_robust_feature_cli.DEFAULT_OUTPUT == Path(
        "outputs/scale_robust_feature/evidence.v2.json"
    )
    configuration = frozen_configuration_payload()
    assert configuration["seed_key_derivation"] == "jax.random.key(seed_id)"
    assert configuration["evidence_seed_namespace"] == EVIDENCE_SEED_NAMESPACE
    assert configuration["evidence_seed_derivation"] == EVIDENCE_SEED_DERIVATION
    assert configuration["learner_init_seed"] == 123
    assert configuration["windows"] == {
        "full_phase_steps": 3_000,
        "early_steps": 200,
        "tail_steps": 500,
        "asymptotic_steps": 1_500,
    }
    conditions = configuration["conditions"]
    assert isinstance(conditions, dict)
    assert set(conditions) == set(CONDITION_NAMES)
    assert conditions[CONDITION_PRIMARY]["scale_robust"] is True
    assert conditions[CONDITION_PRIMARY]["use_obgd"] is False
    assert conditions[CONDITION_LEGACY]["scale_robust"] is False
    assert conditions[CONDITION_LEGACY]["use_obgd"] is True
    assert conditions[CONDITION_NO_RETENTION]["utility_retention_decay"] is None


def test_fresh_evidence_schedule_matches_frozen_sha256_derivation():
    derived = tuple(
        int.from_bytes(
            hashlib.sha256(f"{EVIDENCE_SEED_NAMESPACE}{index}".encode("ascii")).digest()[:4],
            "big",
        )
        for index in range(30)
    )
    assert EVIDENCE_SEEDS == derived
    assert len(EVIDENCE_SEEDS) == len(set(EVIDENCE_SEEDS)) == 30
    assert all(0 <= seed <= (2**32 - 1) for seed in EVIDENCE_SEEDS)
    assert not set(EVIDENCE_SEEDS) & set(range(30, 60))


def test_artifact_preserves_registered_nonmonotonic_seed_order():
    assert EVIDENCE_SEEDS != tuple(sorted(EVIDENCE_SEEDS))
    report = synthetic_report()
    artifact = build_evidence_artifact(
        dataclasses.replace(report, records=tuple(reversed(report.records)))
    )
    scientific = artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    seed_records = scientific["seed_records"]
    assert isinstance(seed_records, list)
    assert [record["seed"] for record in seed_records] == list(EVIDENCE_SEEDS)
    aggregate = scientific["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["seeds"] == list(EVIDENCE_SEEDS)
    comparisons = scientific["comparisons"]
    assert isinstance(comparisons, dict)
    for comparison in comparisons.values():
        assert [row["seed"] for row in comparison["per_seed"]] == list(EVIDENCE_SEEDS)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert validation.accepted, validation.errors


def test_validator_rejects_reordered_registered_seed_records(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    seed_records = changed["scientific_payload"]["seed_records"]
    seed_records[0], seed_records[1] = seed_records[1], seed_records[0]
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any("namespace-derived evidence-seed order" in error for error in validation.errors)


def test_builder_rejects_duplicate_primitive_seed_condition_record():
    report = synthetic_report()
    changed = dataclasses.replace(report, records=(*report.records, report.records[0]))
    with pytest.raises(ValueError, match="duplicate .* record for seed"):
        build_evidence_artifact(changed)


def test_builder_rejects_duplicate_declared_seed():
    report = synthetic_report()
    with pytest.raises(ValueError, match="must contain unique seeds"):
        dataclasses.replace(report, seeds=(report.seeds[0], *report.seeds))


def test_builder_rejects_missing_primitive_seed_records():
    report = synthetic_report()
    missing_seed = report.seeds[0]
    changed = dataclasses.replace(
        report,
        records=tuple(record for record in report.records if record.seed != missing_seed),
    )
    with pytest.raises(ValueError, match="missing primitive records for seeds"):
        build_evidence_artifact(changed)


def test_builder_rejects_extra_primitive_seed_records():
    report = synthetic_report()
    source_seed = report.seeds[0]
    extra_seed = 30
    extra_records = tuple(
        dataclasses.replace(record, seed=extra_seed)
        for record in report.records
        if record.seed == source_seed
    )
    changed = dataclasses.replace(report, records=(*report.records, *extra_records))
    with pytest.raises(ValueError, match="extra primitive records for seeds"):
        build_evidence_artifact(changed)


def test_calibration_record_is_hash_bound_and_marks_missing_primitives():
    repository_root = Path(__file__).resolve().parents[1]
    record_path = repository_root / CALIBRATION_RECORD_RELATIVE_PATH
    raw = record_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CALIBRATION_RECORD_SHA256
    record = json.loads(raw)
    assert record["status"] == "development_only_threshold_calibration_not_promoted_evidence"
    assert record["protocol"]["development_seeds"] == list(range(8, 16))
    fresh = record["protocol"]["fresh_evidence_schedule"]
    assert fresh["status_at_freeze"] == "not_run"
    assert fresh["namespace_ascii"] == EVIDENCE_SEED_NAMESPACE
    assert fresh["derivation"] == EVIDENCE_SEED_DERIVATION
    assert fresh["seeds"] == list(EVIDENCE_SEEDS)
    exposure = record["protocol"]["historical_exposure"]
    assert exposure["seeds"] == {"first": 30, "last": 45}
    assert "not the v2 evidence schedule" in exposure["consequence"]
    assert record["capture"]["primitive_condition_rows"] is None
    assert "not_reconstructed" in record["capture"]["primitive_condition_rows_status"]
    retained_c = record["structural_comparison_gate_inputs"][
        "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs"
    ]
    assert [row["difference"] for row in retained_c["per_seed_differences"]] == [
        0.0,
        0.0,
        2.0,
        1.0,
        3.0,
        0.0,
        2.0,
        1.0,
    ]


def test_source_fingerprint_covers_exactly_five_pinned_sources():
    assert set(current_source_fingerprint()) == {
        "alberta_framework/core/interaction_features.py",
        "alberta_framework/streams/gauntlet.py",
        "alberta_framework/evaluation/scale_robust_feature.py",
        "alberta_framework/evaluation/scale_robust_feature_artifact.py",
        "alberta_framework/evaluation/scale_robust_feature_cli.py",
    }


def test_pair_descriptor_count_uses_only_unique_relevant_context_products():
    primary_end_5, primary_end_7, primary_final = _condition_pairs(CONDITION_PRIMARY)
    assert count_relevant_context_pairs(primary_end_5) == 16
    assert count_relevant_context_pairs(primary_end_7) == 13
    assert count_relevant_context_pairs(primary_final) == 14
    assert count_relevant_context_pairs_by_task(primary_end_7) == (7, 6)
    duplicated = (
        (0, 12),
        (0, 12),
        (12, 0),
        (1, 13),
        (1, 13),
        (13, 1),
    )
    assert count_relevant_context_pairs(duplicated) == 2


def test_artifact_unique_context_metric_cannot_be_inflated_by_duplicate_slots():
    report = synthetic_report()
    first = report.records[0]
    duplicated_pairs = ((0, 12),) * 16 + _pairs(8, 8)[16:]
    assert len(duplicated_pairs) == 24
    duplicated_record = ConditionSeedRecord(
        seed=first.seed,
        condition=first.condition,
        phases=first.phases,
        end_segment_5_active_pairs=duplicated_pairs,
        end_segment_7_active_pairs=duplicated_pairs,
        final_active_pairs=duplicated_pairs,
    )
    changed_report = ScaleRobustFeatureReport(
        seeds=report.seeds,
        records=(duplicated_record, *report.records[1:]),
        memory_by_condition=report.memory_by_condition,
        wall_time_seconds_by_condition=report.wall_time_seconds_by_condition,
    )

    artifact = build_evidence_artifact(changed_report)
    scientific = artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    seed_records = scientific["seed_records"]
    assert isinstance(seed_records, list)
    seed_record = seed_records[0]
    assert isinstance(seed_record, dict)
    conditions = seed_record["conditions"]
    assert isinstance(conditions, dict)
    primary = conditions[CONDITION_PRIMARY]
    assert isinstance(primary, dict)
    assert len(primary["final_active_pairs"]) == 24
    metrics = primary["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["final_unique_relevant_c_context_pairs"] == 1
    assert metrics["final_unique_relevant_d_context_pairs"] == 0
    assert metrics["final_unique_relevant_context_pairs_total"] == 1
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted


def test_synthetic_artifact_is_valid_and_accepted_under_frozen_contract(
    accepted_artifact: dict[str, object],
):
    validation = validate_evidence_artifact(accepted_artifact)
    assert validation.valid, validation.errors
    assert validation.accepted, validation.errors
    assert accepted_artifact["schema_version"] == SCHEMA_VERSION
    assert PROTOCOL_VERSION.endswith(".v2")
    assert threshold_calibration_ready()

    scientific = accepted_artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    protocol = scientific["protocol"]
    assert isinstance(protocol, dict)
    seed_roles = protocol["seed_roles"]
    assert isinstance(seed_roles, dict)
    assert seed_roles["fresh_promoted_evidence"] == list(EVIDENCE_SEEDS)
    assert seed_roles["fresh_evidence_namespace"] == EVIDENCE_SEED_NAMESPACE
    assert seed_roles["fresh_evidence_derivation"] == EVIDENCE_SEED_DERIVATION
    historical_exposure = seed_roles["historical_exposure"]
    assert historical_exposure["seeds"] == list(range(30, 46))
    operational = accepted_artifact["operational_metadata"]
    assert isinstance(operational, dict)
    source = scientific["source_provenance"]
    assert isinstance(source, dict)
    assert operational["git_head"] == source["git_head"]
    thresholds = scientific["thresholds"]
    assert isinstance(thresholds, dict)
    calibration = thresholds["calibration"]
    assert isinstance(calibration, dict)
    assert calibration["required_seed_schedule"] == list(range(8, 16))
    assert calibration["learner_init_seed"] == 123
    assert calibration["status"] == "frozen_direct_key_development_8_15"
    assert calibration["unset_thresholds"] == []
    assert calibration["record"] == {
        "path": CALIBRATION_RECORD_RELATIVE_PATH.as_posix(),
        "sha256": CALIBRATION_RECORD_SHA256,
    }
    assert all(value is not None for name, value in thresholds.items() if name != "calibration")
    assert {
        name: thresholds[name]
        for name in (
            "maximum_median_first_d_early_mse",
            "maximum_median_first_d_tail_mse",
            "maximum_median_recurrent_d_early_mse",
            "maximum_median_recurrent_d_tail_mse",
            "maximum_median_scaled_early_mse",
            "maximum_per_seed_scaled_early_mse",
            "maximum_median_scaled_cumulative_mse",
            "maximum_per_seed_scaled_cumulative_mse",
        )
    } == {
        "maximum_median_first_d_early_mse": 15.0,
        "maximum_median_first_d_tail_mse": 2.0,
        "maximum_median_recurrent_d_early_mse": 2.0,
        "maximum_median_recurrent_d_tail_mse": 2.0,
        "maximum_median_scaled_early_mse": 100.0,
        "maximum_per_seed_scaled_early_mse": 200.0,
        "maximum_median_scaled_cumulative_mse": 50.0,
        "maximum_per_seed_scaled_cumulative_mse": 100.0,
    }
    assert {
        thresholds[name]
        for name in (
            "minimum_unique_relevant_c_context_pairs_end_segment_5",
            "minimum_unique_relevant_d_context_pairs_end_segment_5",
            "minimum_unique_relevant_c_context_pairs_end_segment_7",
            "minimum_unique_relevant_d_context_pairs_end_segment_7",
            "minimum_unique_relevant_c_context_pairs_final",
            "minimum_unique_relevant_d_context_pairs_final",
            "minimum_all_seed_primary_minus_legacy_final_unique_c_context",
            "minimum_all_seed_primary_minus_legacy_final_unique_d_context",
            "minimum_paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
            "minimum_paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
        )
    } == {4.0}
    acceptance = scientific["acceptance"]
    assert isinstance(acceptance, dict)
    checks = acceptance["checks"]
    assert isinstance(checks, list)
    names = {check["name"] for check in checks}
    assert {
        "all_required_thresholds_frozen",
        "paired_mean_primary_minus_legacy_final_savings_ci_lower",
        "paired_mean_legacy_minus_primary_final_tail_mse_ci_lower",
        "paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
        "paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
        ("paired_mean_primary_minus_no_retention_end_segment_7_unique_c_context_ci_lower"),
        ("paired_mean_primary_minus_no_retention_end_segment_7_unique_d_context_ci_lower"),
        "all_seed_primary_pre_final_c_context_noninferior_to_no_retention",
        "all_seed_primary_pre_final_d_context_noninferior_to_no_retention",
    } <= names
    assert not any(check["comparator"] == "calibration_pending" for check in checks)
    assert all(check["passed"] for check in checks)
    strict_retention = {
        check["name"]: check
        for check in checks
        if check["name"].startswith("paired_mean_primary_minus_no_retention")
    }
    assert all(check["comparator"] == ">" for check in strict_retention.values())
    assert all(check["threshold"] == 0.0 for check in strict_retention.values())
    assert not any("no_retention_final_tail" in name for name in names)
    assert not any("retention" in name and "savings" in name for name in names)


def test_readiness_requires_empty_unset_list(monkeypatch: pytest.MonkeyPatch):
    thresholds = scale_robust_feature_artifact._threshold_payload()
    calibration = thresholds["calibration"]
    assert isinstance(calibration, dict)
    calibration["unset_thresholds"] = ["maximum_median_first_d_early_mse"]
    monkeypatch.setattr(
        scale_robust_feature_artifact,
        "_threshold_payload",
        lambda: thresholds,
    )
    assert not threshold_calibration_ready()


def test_readiness_requires_exact_calibration_record_hash(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        scale_robust_feature_artifact,
        "_calibration_record_sha256",
        lambda: "0" * 64,
    )
    assert not threshold_calibration_ready()


@pytest.mark.parametrize(
    ("phase_index", "field", "value", "failed_check"),
    [
        (3, "early_squared_error_sum", 15.01 * 200, "primary_median_first_d_early_mse"),
        (3, "tail_squared_error_sum", 2.01 * 500, "primary_median_first_d_tail_mse"),
        (
            5,
            "early_squared_error_sum",
            2.01 * 200,
            "primary_median_recurrent_d_early_mse",
        ),
        (
            5,
            "tail_squared_error_sum",
            2.01 * 500,
            "primary_median_recurrent_d_tail_mse",
        ),
        (
            6,
            "early_squared_error_sum",
            100.01 * 200,
            "primary_median_scaled_early_mse",
        ),
        (
            6,
            "phase_squared_error_sum",
            50.01 * 3_000,
            "primary_median_scaled_cumulative_mse",
        ),
    ],
)
def test_new_median_error_boundaries_fail_closed(
    phase_index: int,
    field: str,
    value: float,
    failed_check: str,
):
    report = _report_with_primary_phase_field(
        phase_index=phase_index,
        field=field,
        value=value,
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not _acceptance_checks(artifact)[failed_check]["passed"]


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        (
            "early_squared_error_sum",
            200.01 * 200,
            "primary_maximum_scaled_early_mse",
        ),
        (
            "phase_squared_error_sum",
            100.01 * 3_000,
            "primary_maximum_scaled_cumulative_mse",
        ),
    ],
)
def test_new_per_seed_scale_boundaries_fail_closed(
    field: str,
    value: float,
    failed_check: str,
):
    report = _report_with_primary_phase_field(
        phase_index=6,
        field=field,
        value=value,
        seeds=(EVIDENCE_SEEDS[0],),
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not _acceptance_checks(artifact)[failed_check]["passed"]


@pytest.mark.parametrize(
    ("snapshot", "task_c_count", "task_d_count", "failed_check"),
    [
        (
            "end_segment_5",
            3,
            8,
            "primary_unique_relevant_c_context_pairs_end_segment_5",
        ),
        (
            "end_segment_5",
            8,
            3,
            "primary_unique_relevant_d_context_pairs_end_segment_5",
        ),
        (
            "end_segment_7",
            3,
            6,
            "primary_unique_relevant_c_context_pairs_end_segment_7",
        ),
        (
            "end_segment_7",
            7,
            3,
            "primary_unique_relevant_d_context_pairs_end_segment_7",
        ),
        ("final", 3, 6, "primary_unique_relevant_c_context_pairs_final"),
        ("final", 8, 3, "primary_unique_relevant_d_context_pairs_final"),
    ],
)
def test_each_per_task_representation_floor_fails_closed(
    snapshot: str,
    task_c_count: int,
    task_d_count: int,
    failed_check: str,
):
    report = _report_with_snapshot_pairs(
        condition=CONDITION_PRIMARY,
        snapshot=snapshot,
        task_c_count=task_c_count,
        task_d_count=task_d_count,
        seeds=(EVIDENCE_SEEDS[0],),
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not _acceptance_checks(artifact)[failed_check]["passed"]


@pytest.mark.parametrize(
    ("task_c_count", "task_d_count", "check_name", "interval_check_name"),
    [
        (
            4,
            0,
            "all_seed_primary_final_unique_c_context_improvement_over_legacy",
            "paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
        ),
        (
            2,
            2,
            "all_seed_primary_final_unique_d_context_improvement_over_legacy",
            "paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
        ),
    ],
)
def test_primary_legacy_representation_bound_accepts_exactly_four(
    task_c_count: int,
    task_d_count: int,
    check_name: str,
    interval_check_name: str,
):
    report = _report_with_snapshot_pairs(
        condition=CONDITION_LEGACY,
        snapshot="final",
        task_c_count=task_c_count,
        task_d_count=task_d_count,
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert validation.accepted, validation.errors
    checks = _acceptance_checks(artifact)
    assert checks[check_name]["actual"] == 4.0
    assert checks[check_name]["comparator"] == ">="
    assert checks[interval_check_name]["actual"] == 4.0
    assert checks[interval_check_name]["comparator"] == ">="


@pytest.mark.parametrize(
    ("task_c_count", "task_d_count", "failed_check"),
    [
        (
            5,
            0,
            "all_seed_primary_final_unique_c_context_improvement_over_legacy",
        ),
        (
            2,
            3,
            "all_seed_primary_final_unique_d_context_improvement_over_legacy",
        ),
    ],
)
def test_primary_legacy_representation_below_four_is_rejected(
    task_c_count: int,
    task_d_count: int,
    failed_check: str,
):
    report = _report_with_snapshot_pairs(
        condition=CONDITION_LEGACY,
        snapshot="final",
        task_c_count=task_c_count,
        task_d_count=task_d_count,
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not _acceptance_checks(artifact)[failed_check]["passed"]


@pytest.mark.parametrize(
    ("task_c_count", "task_d_count", "failed_check"),
    [
        (
            8,
            2,
            "all_seed_primary_pre_final_c_context_noninferior_to_no_retention",
        ),
        (
            2,
            7,
            "all_seed_primary_pre_final_d_context_noninferior_to_no_retention",
        ),
    ],
)
def test_retention_ablation_rejects_any_seed_with_structural_harm(
    task_c_count: int,
    task_d_count: int,
    failed_check: str,
):
    report = _report_with_snapshot_pairs(
        condition=CONDITION_NO_RETENTION,
        snapshot="end_segment_7",
        task_c_count=task_c_count,
        task_d_count=task_d_count,
        seeds=(EVIDENCE_SEEDS[0],),
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not _acceptance_checks(artifact)[failed_check]["passed"]


@pytest.mark.parametrize(
    ("task_c_count", "task_d_count", "strict_check"),
    [
        (
            7,
            2,
            "paired_mean_primary_minus_no_retention_end_segment_7_unique_c_context_ci_lower",
        ),
        (
            2,
            6,
            "paired_mean_primary_minus_no_retention_end_segment_7_unique_d_context_ci_lower",
        ),
    ],
)
def test_retention_interval_must_be_strictly_positive(
    task_c_count: int,
    task_d_count: int,
    strict_check: str,
):
    report = _report_with_snapshot_pairs(
        condition=CONDITION_NO_RETENTION,
        snapshot="end_segment_7",
        task_c_count=task_c_count,
        task_d_count=task_d_count,
    )
    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    check = _acceptance_checks(artifact)[strict_check]
    assert check["actual"] == 0.0
    assert check["threshold"] == 0.0
    assert check["comparator"] == ">"
    assert not check["passed"]


def test_v1_schema_is_rejected_without_upgrade(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    changed["schema_version"] = "alberta.scale_robust_pair_feature_evidence.v1"
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert not validation.accepted
    assert any("schema_version must be" in error for error in validation.errors)


def test_metrics_reconstruct_exact_windows_and_scorecard_nonlinear_semantics(
    accepted_artifact: dict[str, object],
):
    scientific = accepted_artifact["scientific_payload"]
    assert isinstance(scientific, dict)
    seed_records = scientific["seed_records"]
    assert isinstance(seed_records, list)
    conditions = seed_records[0]["conditions"]
    primary = conditions[CONDITION_PRIMARY]
    assert primary["end_segment_7_active_pairs"] != primary["final_active_pairs"]
    metrics = primary["metrics"]
    assert metrics["savings_c"] == pytest.approx(10.0)
    assert metrics["savings_d"] == pytest.approx(8.0)
    assert metrics["savings_c_final"] == pytest.approx(10.0)
    assert metrics["first_d_early_mse"] == pytest.approx(12.0)
    assert metrics["first_d_tail_mse"] == pytest.approx(0.7)
    assert metrics["recurrent_d_early_mse"] == pytest.approx(1.5)
    assert metrics["recurrent_d_tail_mse"] == pytest.approx(0.6)
    assert metrics["scaled_early_mse"] == pytest.approx(20.0)
    assert metrics["scaled_cumulative_mse"] == pytest.approx(8.0)
    assert metrics["scaled_tail_mse"] == pytest.approx(4.0)
    assert metrics["nonlinear_mse"] == pytest.approx(0.05)
    assert metrics["final_c_tail_mse"] == pytest.approx(0.05)
    assert metrics["end_segment_5_unique_relevant_c_context_pairs"] == 8
    assert metrics["end_segment_5_unique_relevant_d_context_pairs"] == 8
    assert metrics["end_segment_7_unique_relevant_c_context_pairs"] == 7
    assert metrics["end_segment_7_unique_relevant_d_context_pairs"] == 6
    assert metrics["final_unique_relevant_c_context_pairs"] == 8
    assert metrics["final_unique_relevant_d_context_pairs"] == 6
    comparisons = scientific["comparisons"]
    retained_c = comparisons[
        "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs"
    ]
    retained_d = comparisons[
        "primary_minus_no_retention_end_segment_7_unique_relevant_d_context_pairs"
    ]
    assert retained_c["minimum_difference"] == pytest.approx(5.0)
    assert retained_d["minimum_difference"] == pytest.approx(4.0)


def test_paired_bootstrap_is_deterministic_mean_interval():
    first = paired_bootstrap_mean_interval([8.0] * 30)
    second = paired_bootstrap_mean_interval([8.0] * 30)
    assert first == second
    assert first["estimate"] == pytest.approx(8.0)
    assert first["lower"] == pytest.approx(8.0)
    assert first["upper"] == pytest.approx(8.0)
    assert first["resamples"] == BOOTSTRAP_RESAMPLES
    assert first["statistic"] == "mean of paired per-seed differences"


def test_roundtrip_strict_json_preserves_acceptance(
    accepted_artifact: dict[str, object],
    tmp_path: Path,
):
    path = tmp_path / "evidence.json"
    path.write_text(artifact_json(accepted_artifact), encoding="utf-8")
    loaded = load_evidence_artifact(path)
    assert loaded == accepted_artifact
    validation = validate_evidence_artifact(loaded)
    assert validation.valid
    assert validation.accepted


def test_digest_excludes_host_timing_but_timing_schema_remains_strict(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    digest_before = changed["content_digest"]["sha256"]
    timing = changed["operational_metadata"]["wall_time_seconds_by_condition"]
    timing[CONDITION_PRIMARY] = 999.0
    assert changed["content_digest"]["sha256"] == digest_before
    validation = validate_evidence_artifact(changed)
    assert validation.valid
    assert validation.accepted

    timing["unknown_arm"] = 1.0
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any("unknown keys" in error for error in validation.errors)


def test_unknown_scientific_key_fails_even_when_rehashed(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    changed["scientific_payload"]["unexpected"] = True
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert not validation.accepted
    assert any("unknown keys" in error for error in validation.errors)


def test_type_punned_gate_values_fail_even_when_rehashed(
    accepted_artifact: dict[str, object],
):
    """Python equality treats True == 1.0 and 30 == 30.0; the validator must not."""
    changed = copy.deepcopy(accepted_artifact)
    check = _acceptance_checks(changed)[
        "all_seed_primary_pre_final_d_context_noninferior_to_no_retention"
    ]
    assert float(check["actual"]).is_integer() and float(check["threshold"]).is_integer()
    check["actual"] = int(check["actual"])  # JSON 4 in place of 4.0
    check["threshold"] = bool(check["threshold"]) if check["threshold"] in (0.0, 1.0) else int(
        check["threshold"]
    )
    changed["scientific_payload"]["thresholds"]["minimum_seed_count"] = 30.0
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert not validation.accepted
    errors = validation.errors
    assert any("acceptance does not match recomputed gate" in error for error in errors)
    assert any("thresholds are not the frozen v2 thresholds" in error for error in errors)


def test_primitive_tamper_fails_internal_binding_even_when_rehashed(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    phase = changed["scientific_payload"]["seed_records"][0]["conditions"][CONDITION_PRIMARY][
        "phase_windows"
    ][8]
    phase["tail_squared_error_sum"] = 0.0
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any(
        "metrics do not match primitive" in error or "aggregate does not match primitive" in error
        for error in validation.errors
    )


def test_scaled_phase_sum_tamper_fails_cumulative_reconstruction(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    phase = changed["scientific_payload"]["seed_records"][0]["conditions"][CONDITION_PRIMARY][
        "phase_windows"
    ][6]
    phase["phase_squared_error_sum"] = 0.0
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any("metrics do not match primitive" in error for error in validation.errors)


def test_source_hash_tamper_fails_current_source_binding_even_when_rehashed(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    provenance = changed["scientific_payload"]["source_provenance"]
    hashes = provenance["source_sha256"]
    path = next(iter(hashes))
    hashes[path] = "0" * 64
    _rehash(changed)
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any(
        "does not match current pinned source hashes" in error for error in validation.errors
    )


def test_validator_allows_current_git_head_to_advance(
    accepted_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
):
    recorded_head = accepted_artifact["scientific_payload"]["source_provenance"]["git_head"]
    advanced_head = "f" * 40 if recorded_head != "f" * 40 else "e" * 40
    monkeypatch.setattr(scale_robust_feature_artifact, "_git_head", lambda: advanced_head)
    assert scale_robust_feature_artifact._git_head() == advanced_head

    validation = validate_evidence_artifact(accepted_artifact)
    assert validation.valid, validation.errors
    assert validation.accepted


def test_scientific_and_operational_generation_heads_must_match(
    accepted_artifact: dict[str, object],
):
    changed = copy.deepcopy(accepted_artifact)
    scientific_head = changed["scientific_payload"]["source_provenance"]["git_head"]
    changed["operational_metadata"]["git_head"] = (
        "f" * 40 if scientific_head != "f" * 40 else "e" * 40
    )
    validation = validate_evidence_artifact(changed)
    assert not validation.valid
    assert any(
        "generation git_head values are inconsistent" in error for error in validation.errors
    )


def test_wrong_seed_schedule_is_valid_diagnostic_but_not_accepted(
    tmp_path: Path,
):
    artifact = build_evidence_artifact(synthetic_report(tuple(range(31, 60))))
    validation = validate_evidence_artifact(artifact)
    assert validation.valid, validation.errors
    assert not validation.accepted
    scientific = artifact["scientific_payload"]
    acceptance = scientific["acceptance"]
    failed = {check["name"] for check in acceptance["checks"] if not check["passed"]}
    assert "canonical_evidence_seed_schedule" in failed
    assert "minimum_seed_count" in failed
    path = tmp_path / "valid-rejection.json"
    path.write_text(artifact_json(artifact), encoding="utf-8")
    assert scale_robust_feature_cli.main(["--verify", str(path)]) == 1


def test_cli_returns_two_for_integrity_invalid_artifact(
    accepted_artifact: dict[str, object],
    tmp_path: Path,
):
    changed = copy.deepcopy(accepted_artifact)
    changed["content_digest"]["sha256"] = "0" * 64
    path = tmp_path / "invalid.json"
    path.write_text(artifact_json(changed), encoding="utf-8")
    assert scale_robust_feature_cli.main(["--verify", str(path)]) == 2


def test_cli_does_not_write_validator_invalid_generation(
    accepted_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def injected_build(report: ScaleRobustFeatureReport) -> dict[str, object]:
        del report
        return copy.deepcopy(accepted_artifact)

    def invalid_validation(artifact: dict[str, object]) -> ArtifactValidation:
        del artifact
        return ArtifactValidation(
            valid=False,
            accepted=False,
            errors=("synthetic validator failure",),
        )

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "build_evidence_artifact",
        injected_build,
    )
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "validate_evidence_artifact",
        invalid_validation,
    )
    output = tmp_path / "must-not-exist.json"
    assert (
        scale_robust_feature_cli.main(
            ["--output", str(output)],
            report=synthetic_report(),
        )
        == 2
    )
    assert not output.exists()


def test_strict_loader_rejects_duplicate_keys_and_nonstandard_constants(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_evidence_artifact(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_evidence_artifact(nonfinite)


def test_cli_requires_explicit_output_before_running_protocol(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repository_root)
    pinned = scale_robust_feature_cli.DEFAULT_OUTPUT.resolve()
    assert pinned.is_file()
    original = pinned.read_bytes()

    def forbidden_run() -> ScaleRobustFeatureReport:
        raise AssertionError("existing output must be rejected before the protocol runs")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        forbidden_run,
    )
    status = scale_robust_feature_cli.main([])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "generation requires --output with a new path" in emitted["errors"][0]
    assert pinned.read_bytes() == original


def test_cli_refuses_missing_reserved_canonical_path_before_running_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reserved = tmp_path / "reserved" / "evidence.v2.json"
    assert not reserved.exists()
    monkeypatch.setattr(scale_robust_feature_cli, "DEFAULT_OUTPUT", reserved)

    def forbidden_run() -> ScaleRobustFeatureReport:
        raise AssertionError("the reserved canonical path must never be regenerated")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        forbidden_run,
    )
    status = scale_robust_feature_cli.main(["--output", str(reserved)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "pinned canonical artifact path" in emitted["errors"][0]
    assert not reserved.exists()


def test_cli_refuses_arbitrary_existing_output_before_running_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "existing.json"
    sentinel = b"existing artifact must survive"
    path.write_bytes(sentinel)

    def forbidden_run() -> ScaleRobustFeatureReport:
        raise AssertionError("existing output must be rejected before the protocol runs")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        forbidden_run,
    )
    status = scale_robust_feature_cli.main(["--output", str(path)])
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert "existing output path" in emitted["errors"][0]
    assert path.read_bytes() == sentinel


def test_cli_injected_report_atomically_generates_and_verifies_without_running_protocol(
    accepted_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    del accepted_artifact

    def forbidden_run() -> ScaleRobustFeatureReport:
        raise AssertionError("scientific protocol must not run in this test")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        forbidden_run,
    )
    output = tmp_path / "evidence.json"
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
    assert (
        scale_robust_feature_cli.main(
            ["--output", str(output)],
            report=synthetic_report(),
        )
        == 0
    )
    assert output.exists()
    assert linked_destinations == [output.resolve()]
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert scale_robust_feature_cli.main(["--verify", str(output)]) == 0


def test_production_cli_refuses_to_run_when_calibration_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def forbidden_run() -> ScaleRobustFeatureReport:
        raise AssertionError("failed readiness must block evidence execution")

    monkeypatch.setattr(scale_robust_feature_cli, "threshold_calibration_ready", lambda: False)
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        forbidden_run,
    )
    output = tmp_path / "must-not-run.json"
    assert scale_robust_feature_cli.main(["--output", str(output)]) == 2
    assert not output.exists()


def test_production_cli_rejects_calibration_change_during_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    readiness = iter((True, False))
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "threshold_calibration_ready",
        lambda: next(readiness),
    )
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        synthetic_report,
    )

    def forbidden_build(report: ScaleRobustFeatureReport) -> dict[str, object]:
        del report
        raise AssertionError("changed calibration must be rejected before artifact build")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "build_evidence_artifact",
        forbidden_build,
    )
    output = tmp_path / "must-not-exist.json"
    assert scale_robust_feature_cli.main(["--output", str(output)]) == 2
    assert not output.exists()


def test_production_cli_rejects_source_change_during_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    before = current_source_fingerprint()
    after = dict(before)
    path = next(iter(after))
    after[path] = "0" * 64
    fingerprints = iter((before, after))

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "current_source_fingerprint",
        lambda: next(fingerprints),
    )
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        synthetic_report,
    )

    def forbidden_build(report: ScaleRobustFeatureReport) -> dict[str, object]:
        del report
        raise AssertionError("changed source must be rejected before artifact build")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "build_evidence_artifact",
        forbidden_build,
    )
    output = tmp_path / "must-not-exist.json"
    assert scale_robust_feature_cli.main(["--output", str(output)]) == 2
    assert not output.exists()


def test_production_cli_accepts_stable_five_source_fingerprint_without_jax(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fingerprint = current_source_fingerprint()
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "current_source_fingerprint",
        lambda: dict(fingerprint),
    )
    monkeypatch.setattr(
        scale_robust_feature_cli,
        "run_scale_robust_feature_evaluation",
        synthetic_report,
    )
    output = tmp_path / "evidence.json"
    assert scale_robust_feature_cli.main(["--output", str(output)]) == 0
    assert output.exists()


def test_cli_rejects_injected_noncanonical_schedule_before_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def forbidden_build(report: ScaleRobustFeatureReport) -> dict[str, object]:
        del report
        raise AssertionError("noncanonical report must be rejected before build")

    monkeypatch.setattr(
        scale_robust_feature_cli,
        "build_evidence_artifact",
        forbidden_build,
    )
    output = tmp_path / "should-not-exist.json"
    code = scale_robust_feature_cli.main(
        ["--output", str(output)],
        report=synthetic_report(tuple(range(8, 16))),
    )
    assert code == 2
    assert not output.exists()
