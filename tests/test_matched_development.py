from __future__ import annotations

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


def test_report_requires_complete_matched_cross_product() -> None:
    plan = MatchedDevelopmentPlan(
        protocol_id="test.v1", control="off", arms=("off", "on"), seeds=(0, 1), steps=4
    )
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
    plan = MatchedDevelopmentPlan(
        protocol_id="test.v1", control="off", arms=("off", "on"), seeds=(0,), steps=4
    )
    bad_hash = _record("on", 0, 2.0)
    object.__setattr__(bad_hash, "observation_sha256", "f" * 64)
    with pytest.raises(ValueError, match="observation identity"):
        build_matched_report(plan, (_record("off", 0, 1.0), bad_hash))
    bad_steps = _record("on", 0, 2.0)
    object.__setattr__(bad_steps, "updates", 3)
    with pytest.raises(ValueError, match="matched steps"):
        build_matched_report(plan, (_record("off", 0, 1.0), bad_steps))


def test_report_rejects_invalid_transaction_and_incomplete_plan() -> None:
    plan = MatchedDevelopmentPlan(
        protocol_id="test.v1", control="off", arms=("off", "on"), seeds=(0,), steps=4
    )
    invalid = _record("on", 0, 2.0)
    object.__setattr__(invalid, "transaction_valid", False)
    with pytest.raises(ValueError, match="transaction"):
        build_matched_report(plan, (_record("off", 0, 1.0), invalid))
    with pytest.raises(ValueError, match="complete arm-seed cross product"):
        build_matched_report(plan, (_record("off", 0, 1.0),))
