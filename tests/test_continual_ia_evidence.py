"""Held-out, fail-closed evidence tests for the narrow Step-12 IA prototype."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.evaluation.continual_ia import (
    CONDITION_NAMES,
    DEVELOPMENT_SEEDS,
    PROMOTED_EVIDENCE_SEEDS,
    RECOMMENDATION_CONDITIONS,
    ConditionTiming,
    ContinualIAConfig,
    ContinualIAReport,
    IAAcceptanceThresholds,
    IAConditionName,
    IAConditionResult,
    aggregate_ia_evidence,
    evaluate_ia_acceptance,
    paired_bootstrap_mean_interval,
    run_continual_ia_benchmark,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    build_ia_evidence_artifact,
    ia_artifact_json,
    load_ia_evidence_artifact,
    scientific_content_sha256,
    validate_ia_evidence_artifact,
    write_ia_evidence_artifact,
)
from alberta_framework.evaluation.continual_ia_cli import (
    main as continual_ia_cli_main,
)

pytestmark = [pytest.mark.scientific, pytest.mark.slow]


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _as_int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _as_float(value: object) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _content(artifact: dict[str, object]) -> dict[str, object]:
    return _as_dict(artifact["content"])


def _rehash(artifact: dict[str, object]) -> None:
    digest = _as_dict(artifact["content_digest"])
    digest["sha256"] = scientific_content_sha256(_content(artifact))


def _results(
    report: ContinualIAReport,
    condition: IAConditionName,
) -> tuple[IAConditionResult, ...]:
    return tuple(result for result in report.condition_results if result.condition == condition)


def _valid_rejected_report(report: ContinualIAReport) -> ContinualIAReport:
    """Inject one explicit credit failure and bind every derived field to it."""

    changed_results: list[IAConditionResult] = []
    changed = False
    for result in report.condition_results:
        if not changed and result.condition == "recommendation_p05":
            credits = result.credited_actions.copy()
            credits[0] = 1 - result.executed_actions[0]
            changed_results.append(
                replace(
                    result,
                    credited_actions=credits,
                    executed_action_credit_mismatches=1,
                )
            )
            changed = True
        else:
            changed_results.append(result)
    assert changed
    aggregate = aggregate_ia_evidence(changed_results, config=report.config)
    acceptance = evaluate_ia_acceptance(aggregate, report.thresholds)
    assert not acceptance.passed
    return replace(
        report,
        condition_results=tuple(changed_results),
        aggregate=aggregate,
        acceptance=acceptance,
    )


@pytest.fixture(scope="module")
def heldout_report() -> ContinualIAReport:
    """Execute the exact promoted schedule once; no development seed is rerun."""

    return run_continual_ia_benchmark()


@pytest.fixture(scope="module")
def heldout_artifact(
    heldout_report: ContinualIAReport,
) -> dict[str, object]:
    return build_ia_evidence_artifact(heldout_report)


def test_frozen_seed_roles_configuration_and_primitive_shapes(
    heldout_report: ContinualIAReport,
) -> None:
    assert DEVELOPMENT_SEEDS == tuple(range(12))
    assert PROMOTED_EVIDENCE_SEEDS == tuple(range(30, 60))
    assert set(DEVELOPMENT_SEEDS).isdisjoint(PROMOTED_EVIDENCE_SEEDS)
    assert heldout_report.config == ContinualIAConfig()
    assert heldout_report.thresholds == IAAcceptanceThresholds()
    assert heldout_report.aggregate.seeds == PROMOTED_EVIDENCE_SEEDS
    assert len(heldout_report.condition_results) == 30 * len(CONDITION_NAMES)

    for result in heldout_report.condition_results:
        assert result.rewards.shape == (heldout_report.config.num_steps,)
        assert result.executed_actions.shape == result.rewards.shape
        assert result.credited_actions.shape == result.rewards.shape
        assert result.recommendations.shape == result.rewards.shape
        assert result.partner_proposals.shape == result.rewards.shape
        assert result.accepted_recommendations.shape == result.rewards.shape
        assert result.phase_mean_rewards.shape == (heldout_report.config.n_phases,)
        assert result.recovery_lengths.shape == (heldout_report.config.n_phases - 1,)
        assert np.all(np.isfinite(result.rewards))


def test_frozen_primary_result_is_a_valid_intervention_rate_rejection(
    heldout_report: ContinualIAReport,
) -> None:
    aggregate = heldout_report.aggregate
    interval = aggregate.primary_uplift_interval

    assert interval.method == "paired-percentile-bootstrap"
    assert interval.sample_size == 30
    assert interval.resamples == heldout_report.config.bootstrap_resamples
    assert interval.confidence_level == heldout_report.config.confidence_level
    assert interval.estimate == pytest.approx(0.26702777777777775)
    assert interval.lower == pytest.approx(0.2550548611111111)
    assert interval.upper == pytest.approx(0.2783888888888889)
    assert aggregate.mean_changed_action_intervention_rate == pytest.approx(
        0.08727777777777779
    )
    assert aggregate.total_action_changing_interventions == 3_142
    assert aggregate.primary_state_budget_matched
    assert aggregate.primary_interaction_budget_matched
    assert aggregate.executed_action_credit_mismatches == 0
    assert aggregate.all_values_finite
    assert not heldout_report.acceptance.primary_passed
    assert not heldout_report.acceptance.passed
    assert tuple(check.name for check in heldout_report.acceptance.failures) == (
        "changed_action_intervention_rate",
    )


def test_observe_only_is_bitwise_identical_to_partner_alone(
    heldout_report: ContinualIAReport,
) -> None:
    alone = _results(heldout_report, "partner_alone")
    observe = _results(heldout_report, "observe_only")

    assert heldout_report.aggregate.observe_only_exact_reward_identity
    assert heldout_report.aggregate.observe_only_exact_action_identity
    for control, attached in zip(alone, observe, strict=True):
        assert control.seed == attached.seed
        assert np.array_equal(control.rewards, attached.rewards)
        assert np.array_equal(
            control.executed_actions,
            attached.executed_actions,
        )
        assert attached.nominal_accepted_recommendations == 0
        assert not np.any(attached.accepted_recommendations)


def test_primitive_interventions_and_executed_action_credit_are_recomputed(
    heldout_report: ContinualIAReport,
) -> None:
    treatment_changed = 0
    for condition in RECOMMENDATION_CONDITIONS:
        for result in _results(heldout_report, condition):
            accepted = result.accepted_recommendations
            recommendations = result.recommendations
            proposals = result.partner_proposals
            expected_actions = np.where(accepted, recommendations, proposals)
            changed = accepted & (recommendations != proposals)

            assert not bool(accepted[0])
            assert recommendations[0] == result.executed_actions[0]
            assert proposals[0] == result.executed_actions[0]
            assert np.array_equal(result.executed_actions, expected_actions)
            assert np.array_equal(
                result.credited_actions,
                result.executed_actions,
            )
            assert result.action_changing_interventions == int(np.count_nonzero(changed))
            assert result.changed_action_intervention_rate == pytest.approx(
                float(np.mean(changed)),
                abs=0.0,
            )
            executed_accepted = int(np.count_nonzero(accepted))
            assert result.nominal_accepted_recommendations in {
                executed_accepted,
                executed_accepted + 1,
            }
            if condition == "recommendation_p05":
                treatment_changed += int(np.count_nonzero(changed))

    assert treatment_changed == heldout_report.aggregate.total_action_changing_interventions


def test_accept_always_is_a_finite_negative_diagnostic_not_an_uplift_gate(
    heldout_report: ContinualIAReport,
) -> None:
    accept_always = _results(heldout_report, "accept_always")

    assert all(np.all(np.isfinite(result.rewards)) for result in accept_always)
    assert all(
        result.nominal_accepted_recommendations == heldout_report.config.num_steps
        for result in accept_always
    )
    assert all(
        result.executed_accepted_recommendations == heldout_report.config.num_steps - 1
        for result in accept_always
    )
    check_names = {check.name for check in heldout_report.acceptance.checks}
    assert "accept_always_uplift" not in check_names


def test_augmentation_controls_pass_paired_effect_and_budget_gates(
    heldout_report: ContinualIAReport,
) -> None:
    aggregate = heldout_report.aggregate
    prediction_budget = aggregate.condition_budgets["augmented_predictions"]
    noise_budget = aggregate.condition_budgets["augmented_noise"]
    alone_budget = aggregate.condition_budgets["partner_alone"]

    assert prediction_budget == noise_budget
    assert prediction_budget.state_bytes > alone_budget.state_bytes
    assert aggregate.augmentation_noise_state_budget_matched
    assert aggregate.augmentation_state_bytes_above_alone > 0
    assert aggregate.augmentation_vs_alone_interval.lower >= 0.05
    assert aggregate.augmentation_vs_noise_interval.lower >= 0.05
    assert heldout_report.acceptance.secondary_passed
    assert not heldout_report.acceptance.passed


def test_paired_bootstrap_is_deterministic_and_resamples_pairs() -> None:
    differences = np.asarray([0.2, 0.4, -0.1, 0.3], dtype=np.float64)
    first = paired_bootstrap_mean_interval(
        differences,
        confidence_level=0.95,
        resamples=10_000,
        seed=91,
    )
    second = paired_bootstrap_mean_interval(
        differences,
        confidence_level=0.95,
        resamples=10_000,
        seed=91,
    )

    assert first == second
    assert first.estimate == pytest.approx(float(np.mean(differences)))
    assert first.sample_size == differences.size
    assert first.lower <= first.estimate <= first.upper


def test_artifact_is_valid_bound_deterministic_and_explicitly_narrow(
    heldout_report: ContinualIAReport,
    heldout_artifact: dict[str, object],
) -> None:
    validation = validate_ia_evidence_artifact(heldout_artifact)

    assert validation.valid
    assert not validation.accepted
    assert validation.errors == ()
    assert heldout_artifact["schema_version"] == SCHEMA_VERSION
    content = _content(heldout_artifact)
    protocol = _as_dict(content["protocol"])
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    limitations = _as_list(protocol["limitations"])
    assert "no autonomous feature discovery" in limitations
    assert "not completion of the Alberta Plan" in limitations
    assert any("not origin authentication" in str(item) for item in limitations)

    changed_timings = tuple(
        replace(
            result,
            timing=ConditionTiming(
                wall_seconds=result.timing.wall_seconds + 1.0,
                mean_step_latency_ms=result.timing.mean_step_latency_ms + 1.0,
            ),
        )
        for result in heldout_report.condition_results
    )
    later = build_ia_evidence_artifact(replace(heldout_report, condition_results=changed_timings))
    assert _content(later) == content
    assert later["content_digest"] == heldout_artifact["content_digest"]
    assert later["operational_diagnostics"] != (heldout_artifact["operational_diagnostics"])
    later_validation = validate_ia_evidence_artifact(later)
    assert later_validation.valid
    assert not later_validation.accepted


def test_strict_json_round_trip_rejects_nonstandard_numbers(
    heldout_artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    path = tmp_path / "ia-evidence.json"
    path.write_text(ia_artifact_json(heldout_artifact), encoding="utf-8")
    loaded = load_ia_evidence_artifact(path)
    assert loaded == heldout_artifact
    loaded_validation = validate_ia_evidence_artifact(loaded)
    assert loaded_validation.valid
    assert not loaded_validation.accepted

    invalid = tmp_path / "nan.json"
    invalid.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_ia_evidence_artifact(invalid)


def test_unrehashed_primitive_tampering_fails_digest(
    heldout_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(heldout_artifact)
    first = _as_dict(_as_list(_content(tampered)["seed_summaries"])[0])
    treatment = _as_dict(_as_dict(first["conditions"])["recommendation_p05"])
    primitive = _as_dict(treatment["primitive_records"])
    rewards = _as_list(primitive["rewards"])
    rewards[0] = 1.0 - _as_float(rewards[0])

    validation = validate_ia_evidence_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert "content_digest.sha256 does not match content" in validation.errors


def test_rehashed_credit_and_intervention_fabrication_fails_primitives(
    heldout_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(heldout_artifact)
    first = _as_dict(_as_list(_content(fabricated)["seed_summaries"])[0])
    treatment = _as_dict(_as_dict(first["conditions"])["recommendation_p05"])
    primitive = _as_dict(treatment["primitive_records"])
    actions = _as_list(primitive["executed_actions"])
    credits = _as_list(primitive["credited_actions"])
    accepted = _as_list(primitive["accepted_recommendations"])
    credits[0] = 1 - _as_int(actions[0])
    accepted[0] = True
    _rehash(fabricated)

    validation = validate_ia_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("first transition" in error for error in validation.errors)
    assert any("inconsistent with primitives" in error for error in validation.errors)


def test_rehashed_budget_nominal_count_and_threshold_tampering_fail_closed(
    heldout_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(heldout_artifact)
    summaries = _as_list(_content(fabricated)["seed_summaries"])
    second = _as_dict(summaries[1])
    treatment = _as_dict(_as_dict(second["conditions"])["recommendation_p05"])
    summary = _as_dict(treatment["summary"])
    budget = _as_dict(summary["controller_budget"])
    budget["state_bytes"] = _as_int(budget["state_bytes"]) + 4
    executed = _as_int(summary["executed_accepted_recommendations"])
    summary["nominal_accepted_recommendations"] = executed + 2
    thresholds = _as_dict(_content(fabricated)["thresholds"])
    thresholds["minimum_primary_uplift_lower_ci"] = 0.11
    _rehash(fabricated)

    validation = validate_ia_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("exact frozen v1 thresholds" in error for error in validation.errors)
    assert any("not bound to executed primitives" in error for error in validation.errors)
    assert any("canonical recommendation_p05 state" in error for error in validation.errors)


def test_rehashed_protocol_provenance_seed_and_aggregate_tampering_fail_closed(
    heldout_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(heldout_artifact)
    content = _content(fabricated)
    _as_dict(content["protocol"])["supported_claim"] = "general"
    provenance = _as_dict(content["source_provenance"])
    hashes = _as_dict(provenance["source_sha256"])
    first_source = next(iter(hashes))
    hashes[first_source] = "0" * 64
    summaries = _as_list(content["seed_summaries"])
    _as_dict(summaries[1])["seed"] = 30
    aggregate = _as_dict(content["aggregate"])
    _as_dict(aggregate["primary_uplift_interval"])["lower"] = 1.0
    _rehash(fabricated)

    validation = validate_ia_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("protocol is not" in error for error in validation.errors)
    assert any("current pinned sources" in error for error in validation.errors)
    assert any("exactly unique seeds 30-59" in error for error in validation.errors)


def test_unknown_operational_timing_and_digest_keys_fail_closed(
    heldout_artifact: dict[str, object],
) -> None:
    fabricated = copy.deepcopy(heldout_artifact)
    operational = _as_dict(fabricated["operational_diagnostics"])
    operational["unknown"] = True
    first_timing = _as_dict(_as_list(operational["condition_timings"])[0])
    first_timing["unknown"] = True
    digest = _as_dict(fabricated["content_digest"])
    digest["unknown"] = True

    validation = validate_ia_evidence_artifact(fabricated)
    assert not validation.valid
    assert not validation.accepted
    assert any("operational_diagnostics keys do not match" in error for error in validation.errors)
    assert any("condition_timings[0] keys do not match" in error for error in validation.errors)
    assert any("content_digest keys do not match" in error for error in validation.errors)


def test_writer_refuses_to_overwrite_an_existing_artifact(
    heldout_report: ContinualIAReport,
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.json"
    path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_ia_evidence_artifact(path, heldout_report)

    assert path.read_text(encoding="utf-8") == "sentinel\n"


def test_writer_allows_only_one_simultaneous_creator(
    heldout_report: ContinualIAReport,
    tmp_path: Path,
) -> None:
    path = tmp_path / "raced.json"
    barrier = threading.Barrier(2)

    def attempt() -> dict[str, object]:
        barrier.wait()
        return write_ia_evidence_artifact(path, heldout_report)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt) for _ in range(2)]

    artifacts: list[dict[str, object]] = []
    collisions = 0
    for future in futures:
        try:
            artifacts.append(future.result())
        except FileExistsError:
            collisions += 1

    assert len(artifacts) == 1
    assert collisions == 1
    assert path.read_text(encoding="utf-8") == ia_artifact_json(artifacts[0])


def test_cli_writes_verifies_and_returns_two_for_invalid_artifacts(
    heldout_report: ContinualIAReport,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "rejected.json"
    status = continual_ia_cli_main(["--output", str(path)], report=heldout_report)
    emitted = json.loads(capsys.readouterr().out)
    assert status == 1
    assert emitted["valid"] is True
    assert emitted["accepted"] is False
    assert emitted["seed_count"] == 30
    assert path.exists()

    verify_status = continual_ia_cli_main(["--verify", str(path)])
    verified = json.loads(capsys.readouterr().out)
    assert verify_status == 1
    assert verified["valid"] is True
    assert verified["accepted"] is False

    tampered = load_ia_evidence_artifact(path)
    _as_dict(_content(tampered)["aggregate"])["seed_count"] = 1
    path.write_text(ia_artifact_json(tampered), encoding="utf-8")
    invalid_status = continual_ia_cli_main(["--verify", str(path)])
    invalid = json.loads(capsys.readouterr().out)
    assert invalid_status == 2
    assert invalid["valid"] is False
    assert invalid["accepted"] is False


def test_cli_writes_valid_scientific_rejection_with_status_one(
    heldout_report: ContinualIAReport,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected_report = _valid_rejected_report(heldout_report)
    path = tmp_path / "rejected.json"
    status = continual_ia_cli_main(["--output", str(path)], report=rejected_report)
    emitted = json.loads(capsys.readouterr().out)

    assert status == 1
    assert emitted["valid"] is True
    assert emitted["accepted"] is False
    assert path.exists()
    validation = validate_ia_evidence_artifact(load_ia_evidence_artifact(path))
    assert validation.valid
    assert not validation.accepted


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--seed-start", "0"),
        ("--seed-count", "1"),
        ("--minimum-primary-uplift-lower-ci", "0"),
        ("--minimum-changed-action-intervention-rate", "0"),
        ("--minimum-augmentation-vs-alone-lower-ci", "0"),
        ("--minimum-augmentation-vs-noise-lower-ci", "0"),
    ),
)
def test_cli_exposes_no_seed_or_threshold_retuning_options(
    heldout_report: ContinualIAReport,
    flag: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        continual_ia_cli_main([flag, value], report=heldout_report)
    assert exit_info.value.code == 2


def test_cli_rejects_an_injected_non_promoted_schedule_without_writing(
    heldout_report: ContinualIAReport,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = tuple(
        result
        for result in heldout_report.condition_results
        if result.seed in PROMOTED_EVIDENCE_SEEDS[:3]
    )
    aggregate = aggregate_ia_evidence(selected, config=heldout_report.config)
    underpowered = replace(
        heldout_report,
        condition_results=selected,
        aggregate=aggregate,
        acceptance=evaluate_ia_acceptance(
            aggregate,
            heldout_report.thresholds,
        ),
    )
    path = tmp_path / "must-not-exist.json"
    status = continual_ia_cli_main(["--output", str(path)], report=underpowered)
    emitted = json.loads(capsys.readouterr().out)

    assert status == 2
    assert emitted["valid"] is False
    assert emitted["accepted"] is False
    assert not path.exists()
