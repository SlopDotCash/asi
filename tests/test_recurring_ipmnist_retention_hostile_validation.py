"""Hostile input and boundary validation for recurring IPMNIST retention records."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.recurring_ipmnist_retention import (
    PhaseOnlineSummary,
    RecurrenceRetentionMetrics,
    RecurringIPMNISTPhase,
    RecurringIPMNISTProtocol,
    RecurringIPMNISTRetentionReport,
    SentinelProbeBinding,
    SentinelProbeScore,
)


def _make_dummy_phase_summary() -> PhaseOnlineSummary:
    return PhaseOnlineSummary(
        phase_index=0,
        permutation_id="perm.v1",
        exposure_index=0,
        observation_count=1000,
        mean_pre_update_online_accuracy=0.85,
        online_mistakes=150,
        mean_post_update_one_step_plasticity=0.05,
    )


def _make_dummy_sentinel_score() -> SentinelProbeScore:
    return SentinelProbeScore(
        phase_index=0,
        checkpoint_step=1000,
        permutation_id="perm.v1",
        exposure_index=0,
        sentinel_set_id="sentinel.v1",
        sentinel_case_count=100,
        correct_count=85,
        accuracy=0.85,
        learner_state_sha256="a" * 64,
    )


def _make_dummy_recurrence() -> RecurrenceRetentionMetrics:
    return RecurrenceRetentionMetrics(
        permutation_id="perm.v1",
        first_exposure_index=0,
        revisit_exposure_index=1,
        relearning_window=200,
        acquisition_end_sentinel_accuracy=0.85,
        peak_before_revisit_sentinel_accuracy=0.85,
        pre_revisit_sentinel_accuracy=0.70,
        retention_change_from_acquisition=-0.15,
        peak_to_revisit_forgetting=0.15,
        revisit_end_sentinel_accuracy=0.90,
        revisit_recovery=0.20,
        first_exposure_leading_pre_update_accuracy=0.40,
        revisit_leading_pre_update_accuracy=0.60,
        first_exposure_leading_mistakes=120,
        revisit_leading_mistakes=80,
        relearning_savings_mistakes=40,
        relearning_savings_accuracy=0.20,
        first_exposure_leading_one_step_plasticity=0.05,
        revisit_leading_one_step_plasticity=0.04,
    )


def test_phase_online_summary_validation() -> None:
    summary = _make_dummy_phase_summary()
    assert summary.phase_index == 0
    assert summary.permutation_id == "perm.v1"

    # Reject non-integer / bool phase index
    with pytest.raises(ValueError, match="phase_index must be a non-negative integer"):
        PhaseOnlineSummary(
            phase_index=True,
            permutation_id="perm.v1",
            exposure_index=0,
            observation_count=1000,
            mean_pre_update_online_accuracy=0.85,
            online_mistakes=150,
            mean_post_update_one_step_plasticity=0.05,
        )

    # Reject negative mistakes
    with pytest.raises(ValueError, match="online_mistakes must be a non-negative integer"):
        PhaseOnlineSummary(
            phase_index=0,
            permutation_id="perm.v1",
            exposure_index=0,
            observation_count=1000,
            mean_pre_update_online_accuracy=0.85,
            online_mistakes=-1,
            mean_post_update_one_step_plasticity=0.05,
        )

    # Reject non-finite accuracy
    with pytest.raises(ValueError, match="mean_pre_update_online_accuracy must be finite in"):
        PhaseOnlineSummary(
            phase_index=0,
            permutation_id="perm.v1",
            exposure_index=0,
            observation_count=1000,
            mean_pre_update_online_accuracy=float("nan"),
            online_mistakes=150,
            mean_post_update_one_step_plasticity=0.05,
        )


def test_sentinel_probe_score_validation() -> None:
    score = _make_dummy_sentinel_score()
    assert score.checkpoint_step == 1000

    # Reject non-positive checkpoint step
    with pytest.raises(ValueError, match="checkpoint_step must be a positive integer"):
        SentinelProbeScore(
            phase_index=0,
            checkpoint_step=0,
            permutation_id="perm.v1",
            exposure_index=0,
            sentinel_set_id="sentinel.v1",
            sentinel_case_count=100,
            correct_count=85,
            accuracy=0.85,
            learner_state_sha256="a" * 64,
        )

    # Reject invalid sha256
    with pytest.raises(ValueError, match="learner_state_sha256 must be a lowercase SHA-256 digest"):
        SentinelProbeScore(
            phase_index=0,
            checkpoint_step=1000,
            permutation_id="perm.v1",
            exposure_index=0,
            sentinel_set_id="sentinel.v1",
            sentinel_case_count=100,
            correct_count=85,
            accuracy=0.85,
            learner_state_sha256="invalid",
        )


def test_recurrence_retention_metrics_validation() -> None:
    metrics = _make_dummy_recurrence()
    assert metrics.relearning_savings_mistakes == 40

    # Reject non-unit float accuracy
    with pytest.raises(ValueError, match="acquisition_end_sentinel_accuracy must be finite in"):
        RecurrenceRetentionMetrics(
            permutation_id="perm.v1",
            first_exposure_index=0,
            revisit_exposure_index=1,
            relearning_window=200,
            acquisition_end_sentinel_accuracy=1.5,
            peak_before_revisit_sentinel_accuracy=0.85,
            pre_revisit_sentinel_accuracy=0.70,
            retention_change_from_acquisition=-0.15,
            peak_to_revisit_forgetting=0.15,
            revisit_end_sentinel_accuracy=0.90,
            revisit_recovery=0.20,
            first_exposure_leading_pre_update_accuracy=0.40,
            revisit_leading_pre_update_accuracy=0.60,
            first_exposure_leading_mistakes=120,
            revisit_leading_mistakes=80,
            relearning_savings_mistakes=40,
            relearning_savings_accuracy=0.20,
            first_exposure_leading_one_step_plasticity=0.05,
            revisit_leading_one_step_plasticity=0.04,
        )


def test_recurring_ipmnist_retention_report_validation() -> None:
    protocol = RecurringIPMNISTProtocol(
        protocol_id="tests.recurring-ipmnist-retention.v1",
        phases=(
            RecurringIPMNISTPhase(
                phase_index=0,
                start_step=0,
                length=4,
                permutation_id="permutation-a.v1",
                exposure_index=0,
            ),
            RecurringIPMNISTPhase(
                phase_index=1,
                start_step=4,
                length=4,
                permutation_id="permutation-b.v1",
                exposure_index=0,
            ),
            RecurringIPMNISTPhase(
                phase_index=2,
                start_step=8,
                length=4,
                permutation_id="permutation-a.v1",
                exposure_index=1,
            ),
        ),
        sentinel_bindings=(
            SentinelProbeBinding(
                permutation_id="permutation-a.v1",
                permutation_sha256="a" * 64,
                sentinel_set_id="sentinel-a.v1",
                sentinel_set_sha256="c" * 64,
                sentinel_case_count=4,
            ),
            SentinelProbeBinding(
                permutation_id="permutation-b.v1",
                permutation_sha256="b" * 64,
                sentinel_set_id="sentinel-b.v1",
                sentinel_set_sha256="d" * 64,
                sentinel_case_count=4,
            ),
        ),
        relearning_window=2,
    )
    report = RecurringIPMNISTRetentionReport(
        protocol=protocol,
        protocol_sha256="c" * 64,
        trace_sha256="d" * 64,
        sentinel_snapshots_sha256="e" * 64,
        phase_summaries=(_make_dummy_phase_summary(),),
        sentinel_scores=(_make_dummy_sentinel_score(),),
        recurrence=_make_dummy_recurrence(),
    )
    assert report.protocol_sha256 == "c" * 64

    # Reject invalid protocol type
    with pytest.raises(TypeError, match="protocol must be a RecurringIPMNISTProtocol"):
        RecurringIPMNISTRetentionReport(
            protocol="invalid",  # type: ignore[arg-type]
            protocol_sha256="c" * 64,
            trace_sha256="d" * 64,
            sentinel_snapshots_sha256="e" * 64,
            phase_summaries=(_make_dummy_phase_summary(),),
            sentinel_scores=(_make_dummy_sentinel_score(),),
            recurrence=_make_dummy_recurrence(),
        )
