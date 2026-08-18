from __future__ import annotations

from dataclasses import replace

import pytest

from alberta_framework.evaluation.matched_development import (
    ArmRecord,
    MatchedDevelopmentPlan,
    build_matched_report,
)


def _record(arm: str, seed: int, metric: float) -> ArmRecord:
    return ArmRecord(
        arm=arm,
        seed=seed,
        observation_sha256=f"{seed:064x}",
        updates=4,
        data_steps=4,
        model_queries=4,
        persistent_bytes=8,
        peak_working_set_bytes=16,
        timing_seconds=0.1,
        metric=metric,
        transaction_valid=True,
    )


def _plan(*, seeds: tuple[int, ...] = (0, 1)) -> MatchedDevelopmentPlan:
    return MatchedDevelopmentPlan(
        protocol_id="test.v1",
        paper_revision="paper.v1",
        reference_code_revision="not-used",
        source_sha256="a" * 64,
        report_namespace="outputs/test.v1",
        metric_name="score",
        control="off",
        arms=("off", "on"),
        seeds=seeds,
        steps=4,
    )


def test_report_requires_complete_matched_cross_product() -> None:
    plan = _plan()
    records = (
        _record("off", 0, 1.0),
        _record("on", 0, 2.0),
        _record("off", 1, 3.0),
        _record("on", 1, 5.0),
    )
    report = build_matched_report(plan, records)
    assert report["paired_differences"] == (1.0, 2.0)
    assert report["mean_paired_difference"] == pytest.approx(1.5)
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False


def test_report_rejects_observation_or_counter_mismatch() -> None:
    plan = _plan(seeds=(0,))
    bad_hash = replace(_record("on", 0, 2.0), observation_sha256="f" * 64)
    with pytest.raises(ValueError, match="observation identity"):
        build_matched_report(plan, (_record("off", 0, 1.0), bad_hash))
    bad_steps = replace(_record("on", 0, 2.0), updates=3)
    with pytest.raises(ValueError, match="matched steps"):
        build_matched_report(plan, (_record("off", 0, 1.0), bad_steps))


def test_report_rejects_invalid_transaction_and_incomplete_plan() -> None:
    plan = _plan(seeds=(0,))
    invalid = replace(_record("on", 0, 2.0), transaction_valid=False)
    with pytest.raises(ValueError, match="transaction"):
        build_matched_report(plan, (_record("off", 0, 1.0), invalid))
    with pytest.raises(ValueError, match="complete arm-seed cross product"):
        build_matched_report(plan, (_record("off", 0, 1.0),))


def test_plan_and_report_reject_path_escape_and_numeric_overflow() -> None:
    with pytest.raises(ValueError, match="outputs path"):
        replace(_plan(), report_namespace="outputs/../escape")
    plan = _plan(seeds=(0,))
    with pytest.raises(ValueError, match="paired differences"):
        build_matched_report(
            plan,
            (
                _record("off", 0, -float.fromhex("0x1.fffffffffffffp+1023")),
                _record("on", 0, float.fromhex("0x1.fffffffffffffp+1023")),
            ),
        )
