from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alberta_framework.evaluation.matched_development import (
    ArmRecord,
    MatchedDevelopmentPlan,
    build_matched_report,
    canonical_report_bytes,
    plan_sha256,
    retain_matched_report,
    validate_matched_report,
)


def _plan(
    *, seeds: tuple[int, ...] = (0, 1), higher_is_better: bool = True
) -> MatchedDevelopmentPlan:
    return MatchedDevelopmentPlan(
        protocol_id="test.v1",
        paper_revision="paper.v1",
        reference_code_revision="code.v1",
        source_sha256="a" * 64,
        report_namespace="outputs/test.v1",
        metric_name="score",
        higher_is_better=higher_is_better,
        tie_tolerance=0.0,
        control="off",
        arms=("off", "on"),
        seeds=seeds,
        expected_updates=4,
        expected_observations=5,
        expected_data_steps=6,
        expected_model_queries=0,
        persistent_byte_budget=64,
        working_set_byte_budget=128,
    )


def _record(plan: MatchedDevelopmentPlan, arm: str, seed: int, metric: float) -> ArmRecord:
    return ArmRecord(
        plan_sha256=plan_sha256(plan),
        arm=arm,
        seed=seed,
        observation_sha256=f"{seed:064x}",
        updates=4,
        observations=5,
        data_steps=6,
        model_queries=0,
        persistent_bytes=8,
        peak_working_set_bytes=16,
        timing_seconds=0.1,
        metric=metric,
        transaction_valid=True,
    )


def _records(plan: MatchedDevelopmentPlan) -> tuple[ArmRecord, ...]:
    return (
        _record(plan, "on", 1, 5.0),
        _record(plan, "off", 0, 1.0),
        _record(plan, "off", 1, 3.0),
        _record(plan, "on", 0, 2.0),
    )


def test_report_is_canonical_paired_and_resource_explicit() -> None:
    plan = _plan()
    report = build_matched_report(plan, _records(plan))
    assert [(row["seed"], row["arm"]) for row in report["records"]] == [
        (0, "off"),
        (0, "on"),
        (1, "off"),
        (1, "on"),
    ]
    assert report["paired_differences"] == (1.0, 2.0)
    assert report["mean_paired_difference"] == pytest.approx(1.5)
    assert report["paired_stderr"] == pytest.approx(0.5)
    assert report["outcome"] == "improved"
    assert report["arm_summaries"]["off"]["total_updates"] == 8
    assert report["arm_summaries"]["on"]["mean_metric"] == pytest.approx(3.5)
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert len(report["report_sha256"]) == 64


def test_direction_and_single_seed_stderr_are_frozen() -> None:
    plan = _plan(seeds=(0,), higher_is_better=False)
    records = (_record(plan, "on", 0, 1.0), _record(plan, "off", 0, 2.0))
    report = build_matched_report(plan, records)
    assert report["paired_differences"] == (-1.0,)
    assert report["paired_stderr"] == 0.0
    assert report["outcome"] == "improved"


def test_report_rejects_identity_observation_or_counter_mismatch() -> None:
    plan = _plan(seeds=(0,))
    off = _record(plan, "off", 0, 1.0)
    on = _record(plan, "on", 0, 2.0)
    with pytest.raises(ValueError, match="plan identity"):
        build_matched_report(plan, (off, replace(on, plan_sha256="f" * 64)))
    with pytest.raises(ValueError, match="observation identity"):
        build_matched_report(plan, (off, replace(on, observation_sha256="f" * 64)))
    with pytest.raises(ValueError, match="expected counters"):
        build_matched_report(plan, (off, replace(on, model_queries=1)))


def test_distinct_matched_counter_budgets_are_representable() -> None:
    plan = replace(
        _plan(seeds=(0,)),
        expected_updates=10_000,
        expected_observations=10_000,
        expected_data_steps=10_000,
        expected_model_queries=10_128,
    )
    records = tuple(
        replace(
            _record(plan, arm, 0, metric),
            updates=10_000,
            observations=10_000,
            data_steps=10_000,
            model_queries=10_128,
        )
        for arm, metric in (("off", 1.0), ("on", 2.0))
    )
    report = build_matched_report(plan, records)
    assert report["arm_summaries"]["off"]["total_model_queries"] == 10_128
    assert report["arm_summaries"]["on"]["total_data_steps"] == 10_000


def test_consumption_revalidates_mutated_frozen_values() -> None:
    plan = _plan(seeds=(0,))
    off = _record(plan, "off", 0, 1.0)
    on = _record(plan, "on", 0, 2.0)
    object.__setattr__(plan, "seeds", [0])
    with pytest.raises(ValueError, match="seeds"):
        build_matched_report(plan, (off, on))
    plan = _plan(seeds=(0,))
    off = _record(plan, "off", 0, 1.0)
    object.__setattr__(off, "updates", 2**100)
    with pytest.raises(ValueError, match="updates"):
        build_matched_report(plan, (off, _record(plan, "on", 0, 2.0)))


@pytest.mark.parametrize(
    "namespace",
    ("outputs/../escape", "outputs/./x", "outputs//x", "outputs/x/", "elsewhere/x"),
)
def test_plan_rejects_noncanonical_namespaces(namespace: str) -> None:
    with pytest.raises(ValueError, match="outputs path"):
        replace(_plan(), report_namespace=namespace)


def test_plan_rejects_hostile_or_unbounded_values_without_iteration() -> None:
    class HostileTuple:
        def __iter__(self) -> object:
            raise AssertionError("must not iterate")

        def __len__(self) -> int:
            raise AssertionError("must not inspect")

    with pytest.raises(ValueError, match="arms"):
        replace(_plan(), arms=HostileTuple())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounded"):
        replace(_plan(), metric_name="é" * 300)
    with pytest.raises(ValueError, match="valid UTF-8"):
        replace(_plan(), metric_name="\ud800")
    with pytest.raises(ValueError, match="expected_updates"):
        replace(_plan(), expected_updates=2**100)


def test_report_rejects_non_string_mapping_keys_without_hooks() -> None:
    class HostileKey:
        calls = 0

        def __hash__(self) -> int:
            return hash("plan")

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            raise AssertionError("hostile key equality executed")

    report = build_matched_report(_plan(), _records(_plan()))
    hostile = HostileKey()
    forged = dict(report)
    del forged["plan"]
    forged[hostile] = None  # type: ignore[index]
    HostileKey.calls = 0
    with pytest.raises(ValueError, match="keys must be exact strings"):
        validate_matched_report(forged)
    assert HostileKey.calls == 0


def test_report_rejects_invalid_transaction_numeric_and_total_resource_overflow() -> None:
    plan = _plan(seeds=(0,))
    off = _record(plan, "off", 0, 1.0)
    on = _record(plan, "on", 0, 2.0)
    with pytest.raises(ValueError, match="transaction"):
        build_matched_report(plan, (off, replace(on, transaction_valid=False)))
    with pytest.raises(ValueError, match="paired differences"):
        build_matched_report(
            plan,
            (
                replace(off, metric=-float.fromhex("0x1.fffffffffffffp+1023")),
                replace(on, metric=float.fromhex("0x1.fffffffffffffp+1023")),
            ),
        )
    large_plan = replace(plan, persistent_byte_budget=256 * 1024 * 1024)
    large_off = replace(
        off,
        plan_sha256=plan_sha256(large_plan),
        persistent_bytes=200 * 1024 * 1024,
    )
    large_on = replace(
        on,
        plan_sha256=plan_sha256(large_plan),
        persistent_bytes=200 * 1024 * 1024,
    )
    with pytest.raises(ValueError, match="aggregate resource"):
        build_matched_report(large_plan, (large_off, large_on))


def test_canonical_serialization_strict_validation_and_exclusive_retention(tmp_path: Path) -> None:
    plan = _plan()
    report = build_matched_report(plan, _records(plan))
    encoded = canonical_report_bytes(report)
    assert encoded == canonical_report_bytes(validate_matched_report(json.loads(encoded)))
    destination = retain_matched_report(report, repository_root=tmp_path)
    assert destination == tmp_path / "outputs/test.v1/report.v2.json"
    assert destination.read_bytes() == encoded
    with pytest.raises(FileExistsError):
        retain_matched_report(report, repository_root=tmp_path)
    forged = dict(report)
    forged["outcome"] = "worse"
    with pytest.raises(ValueError, match="canonical report"):
        validate_matched_report(forged)


def test_retention_rejects_namespace_symlink(tmp_path: Path) -> None:
    plan = _plan()
    report = build_matched_report(plan, _records(plan))
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        retain_matched_report(report, repository_root=tmp_path)
    assert not (outside / "test.v1").exists()


def test_serialization_rejects_nan() -> None:
    report = build_matched_report(_plan(), _records(_plan()))
    forged = dict(report)
    forged["mean_paired_difference"] = float("nan")
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_report_bytes(forged)
