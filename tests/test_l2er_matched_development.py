from __future__ import annotations

import dataclasses
import os
from copy import deepcopy
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from alberta_framework.benchmarks import l2er_matched_development as matched
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult, screening_spec


def _results() -> list[ScreeningRunResult]:
    offsets = {
        "l2er_mechanism_off": 0.0,
        "l2er_l2_only": 0.02,
        "l2er_er_only": -0.02,
        "l2er_combined": 0.03,
    }
    results = []
    for seed in matched.SEEDS:
        for arm in matched.ARMS:
            spec = screening_spec(arm)
            accuracy = 0.5 + offsets[arm]
            results.append(
                ScreeningRunResult(
                    config_name=arm,
                    base_learner=spec.base_learner,
                    hyperparameters=dict(spec.hyperparameters),
                    seed=seed,
                    config=matched.CONFIG,
                    per_task_accuracy=np.asarray([accuracy, accuracy], dtype=np.float64),
                    per_task_loss=np.asarray([0.8, 0.7], dtype=np.float64),
                    per_task_plasticity=np.asarray([0.1, 0.2], dtype=np.float64),
                    wall_clock_seconds=1.0,
                )
            )
    return results


def test_matched_report_is_complete_paired_and_permanently_nonpromoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(),
        source_provenance={},
        dataset_provenance={},
        environment={},
    )
    assert len(report["records"]) == len(matched.SEEDS) * len(matched.ARMS)
    paired = report["paired_comparisons"]
    assert paired["l2er_l2_only"]["outcome"] == "supported"
    assert paired["l2er_er_only"]["outcome"] == "rejected"
    assert report["policy"]["scientific_promotion_allowed"] is False
    assert matched.validate_report(report, require_current_source=False) == report


def test_validator_rejects_hostile_plan_container_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile iteration")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

    hostile = deepcopy(report)
    hostile["plan"]["arms"] = HostileList()
    with pytest.raises(ValueError, match="exact string list"):
        matched.validate_report(hostile, require_current_source=False)
    assert HostileList.calls == 0

    hostile_runtime = deepcopy(report)
    hostile_runtime["environment"] = {"devices": HostileList()}
    with pytest.raises(ValueError, match="finite exact JSON"):
        matched.validate_report(hostile_runtime, require_current_source=False)
    assert HostileList.calls == 0

    shared_bomb: object = "leaf"
    for _ in range(3):
        shared_bomb = [shared_bomb] * 64
    hostile_runtime = deepcopy(report)
    hostile_runtime["environment"] = {"bomb": shared_bomb}
    with pytest.raises(ValueError, match="aggregate JSON node limit"):
        matched.validate_report(hostile_runtime, require_current_source=False)


def test_validator_preflights_hostile_dict_keys_before_hash_or_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileStr(str):
        armed = False
        calls = 0

        def __hash__(self) -> int:
            if type(self).armed:
                type(self).calls += 1
                raise AssertionError("hostile hash")
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            if type(self).armed:
                type(self).calls += 1
                raise AssertionError("hostile equality")
            return super().__eq__(other)

    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    hostile = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    value = hostile.pop("schema")
    hostile[HostileStr("schema")] = value
    HostileStr.armed = True
    with pytest.raises(ValueError, match="keys do not match"):
        matched.validate_report(hostile, require_current_source=False)
    assert HostileStr.calls == 0


@pytest.mark.parametrize("nested", ("record", "hyperparameters", "metrics", "resources"))
def test_report_preflights_nested_receipt_keys_before_hostile_dispatch(
    nested: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HostileStr(str):
        armed = False
        calls = 0

        def __hash__(self) -> int:
            if type(self).armed:
                type(self).calls += 1
                raise AssertionError("hostile hash")
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            if type(self).armed:
                type(self).calls += 1
                raise AssertionError("hostile equality")
            return super().__eq__(other)

    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    records = report["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    target = record if nested == "record" else record[nested]
    assert isinstance(target, dict)
    key = next(iter(target))
    assert isinstance(key, str)
    value = target.pop(key)
    target[HostileStr(key)] = value
    HostileStr.armed = True
    with pytest.raises(ValueError, match="keys must be exactly"):
        matched.validate_report(report, require_current_source=False)
    assert HostileStr.calls == 0


def test_builder_revalidates_forged_result_before_field_dispatch() -> None:
    class HostileInt(int):
        calls = 0

        def __mul__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile multiply")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

        def __hash__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile hash")

    results = _results()
    forged_config = matched.IPMNISTConfig(**matched.CONFIG.to_config())
    object.__setattr__(forged_config, "n_tasks", HostileInt(2))
    object.__setattr__(results[0], "config", forged_config)
    with pytest.raises(ValueError, match="n_tasks"):
        matched.build_report(
            results, source_provenance={}, dataset_provenance={}, environment={}
        )
    assert HostileInt.calls == 0


def test_validator_recomputes_paired_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    hostile = deepcopy(report)
    hostile["paired_comparisons"]["l2er_combined"]["mean_delta"] = 0.0
    with pytest.raises(ValueError, match="mean_delta is inconsistent"):
        matched.validate_report(hostile, require_current_source=False)

    reordered = deepcopy(report)
    reordered["records"][0], reordered["records"][1] = (
        reordered["records"][1],
        reordered["records"][0],
    )
    with pytest.raises(ValueError, match="deterministic frozen"):
        matched.validate_report(reordered, require_current_source=False)


def test_three_seed_interval_uses_student_t_not_normal_critical_value() -> None:
    mean, lower, upper, outcome = matched._outcome((0.01, 0.01, 0.0))
    assert mean > 0.0
    assert lower < 0.0 < upper
    assert outcome == "inconclusive"
    assert matched.frozen_plan()["confidence_method"] == "two_sided_student_t"
    critical = matched.frozen_plan()["confidence_critical"]
    assert critical == 4.302652729749464
    assert isinstance(critical, float)
    assert critical.hex() == "0x1.135ea98e146bbp+2"


def test_validator_rejects_obsolete_v1_report_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    report["schema"] = "asi.l2er-ipmnist.matched-development-report.v1"
    with pytest.raises(ValueError, match="schema does not match"):
        matched.validate_report(report, require_current_source=False)


def test_report_revalidates_result_identity_before_reading_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    results = _results()
    results[0] = dataclasses.replace(results[0], base_learner="adamw")
    with pytest.raises(ValueError, match="base learner"):
        matched.build_report(
            results, source_provenance={}, dataset_provenance={}, environment={}
        )


def test_output_namespace_is_one_new_development_path() -> None:
    assert matched.SCHEMA == "asi.l2er-ipmnist.matched-development-report.v2"
    assert matched.PLAN_ID == "asi.l2er-ipmnist.cheap-screen.v2"
    assert matched.OUTPUT_PATH.relative_to(matched._REPO_ROOT).as_posix() == (
        "outputs/l2er_matched_development/report.v2.json"
    )
    assert matched.SEEDS == (1711, 1712, 1713)
    critical = matched.frozen_plan()["confidence_critical"]
    assert isinstance(critical, float)
    assert critical.hex() == "0x1.135ea98e146bbp+2"
    assert matched.frozen_plan()["statistical_correction_seed_policy"] == (
        "a pre-execution statistical correction does not authorize seed churn"
    )
    assert matched.frozen_plan()["consumed_preplan_audit_seeds"] == [1701]
    invalid = matched.frozen_plan()["invalid_execution_history"]
    assert invalid[1]["seeds"] == [1721, 1722, 1723]
    assert invalid[1]["artifact_sha256"] == (
        "579c400412d3c50898c16a8fd02fa82e2cd712b5278b5a99974b5e89560707ec"
    )
    assert invalid[1]["disposition"] == "invalid_unmerged_seed_churn_attempt"
    assert matched.frozen_plan()["development_only"] is True
    assert matched.frozen_plan()["scientific_promotion_allowed"] is False


def test_output_directory_rejects_symlinked_segments_and_occupied_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(matched, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        matched,
        "OUTPUT_PATH",
        tmp_path / "outputs/l2er_matched_development/report.v2.json",
    )
    with pytest.raises(OSError):
        matched._open_output_transaction()

    (tmp_path / "outputs").unlink()
    target = tmp_path / "outputs/l2er_matched_development/report.v2.json"
    target.parent.mkdir(parents=True)
    target.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        matched._open_output_transaction()


def test_output_publication_uses_pinned_dirfd_and_strict_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(matched, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        matched,
        "OUTPUT_PATH",
        tmp_path / "outputs/l2er_matched_development/report.v2.json",
    )
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    monkeypatch.setattr(matched, "_screening_source_provenance", lambda: {})
    monkeypatch.setattr(matched, "_screening_runtime_environment", lambda: {})
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    directory_fd, temporary_fd, temporary_name = matched._open_output_transaction()
    try:
        with pytest.raises(FileExistsError, match="already reserved"):
            matched._open_output_transaction()
        matched._publish_report(directory_fd, temporary_fd, temporary_name, report)
    finally:
        os.close(temporary_fd)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)
    assert matched.OUTPUT_PATH.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        matched._open_output_transaction()
