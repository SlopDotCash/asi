"""Prospectively frozen matched-development campaign for issue #1560."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from alberta_framework.benchmarks import adamo_matched_development as matched
from alberta_framework.benchmarks.adamo_diagnostic import run_adamo_diagnostic

pytestmark = pytest.mark.integration


def _patch_identities(
    monkeypatch: pytest.MonkeyPatch, receipts: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run_runtime = receipts[0]["runtime"]
    dataset = {
        "x": {"shape": [60_000, 784], "sha256": matched.CANONICAL_X_SHA256},
        "y": {"shape": [60_000], "sha256": matched.CANONICAL_Y_SHA256},
    }
    source = {"git_commit": "c" * 40, "relevant_source_sha256": "a" * 64}
    runtime = {
        "python": {"version": run_runtime["python"]},
        "packages": {"jax": run_runtime["jax"], "numpy": run_runtime["numpy"]},
        "jax": {"backend": run_runtime["backend"]},
    }
    monkeypatch.setattr(
        matched, "_current_source_provenance", lambda: source
    )
    monkeypatch.setattr(matched, "_current_runtime_environment", lambda: runtime)
    monkeypatch.setattr(matched, "validate_adamo_diagnostic", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    return dataset, source, runtime


def _build_report(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    dataset, source, runtime = _patch_identities(monkeypatch, receipts)
    return matched.build_report(
        receipts,
        dataset_provenance=dataset,
        source_provenance=source,
        runtime_environment=runtime,
    )


@pytest.fixture(scope="module")
def receipts() -> list[dict[str, object]]:
    inputs = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4)
    labels = np.arange(8, dtype=np.int32) % 2
    first = run_adamo_diagnostic(
        inputs,
        labels,
        profile="contract-smoke",
        seed=matched.CONSUMED_QUALIFICATION_SEEDS[0],
    )
    result = []
    for seed in matched.SEEDS:
        receipt = copy.deepcopy(first)
        receipt["seed"] = seed
        receipt["profile"] = matched.PROFILE
        receipt["frozen_development_seeds"] = list(matched.SEEDS)
        receipt["dataset"]["x_sha256"] = matched.CANONICAL_X_SHA256
        receipt["dataset"]["y_sha256"] = matched.CANONICAL_Y_SHA256
        receipt["dataset"]["rows"] = 60_000
        receipt["dataset"]["loaded_numeric_bytes"] = 60_000 * (4 * 4 + 4)
        result.append(receipt)
    return result


def test_plan_is_prospective_exact_and_permanently_nonpromoting() -> None:
    plan = matched.frozen_plan()
    assert plan["seeds"] == list(matched.SEEDS)
    assert plan["profile"] == "bounded-development"
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["outcome_retention_required"] is True
    assert plan["consumed_qualification_seeds"] == [15600, 15601, 15602, 15603]
    assert plan["quarantined_preplan_seeds"] == [25600, 25601, 25602, 25603]
    assert set(plan["seeds"]).isdisjoint(plan["consumed_qualification_seeds"])
    assert set(plan["seeds"]).isdisjoint(plan["quarantined_preplan_seeds"])
    assert plan["dataset"]["source"] == {
        "provider": "openml",
        "name": "mnist_784",
        "version": 1,
        "row_start": 0,
        "row_stop_exclusive": 60_000,
    }
    assert plan["dataset"]["arrays"] == {
        "x": {
            "dtype": "<f4",
            "shape": [60_000, 784],
            "sha256": matched.CANONICAL_X_SHA256,
        },
        "y": {
            "dtype": "<i4",
            "shape": [60_000],
            "sha256": matched.CANONICAL_Y_SHA256,
        },
    }


def test_canonical_dataset_hashes_are_required_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = {
        "x": {"shape": [60_000, 784], "sha256": "0" * 64},
        "y": {"shape": [60_000], "sha256": "1" * 64},
    }
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    destination = tmp_path / "report.json"
    monkeypatch.setattr(matched, "OUTPUT_PATH", destination)
    monkeypatch.setattr(matched, "_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(matched, "_current_source_provenance", lambda: {})
    monkeypatch.setattr(matched, "_current_runtime_environment", lambda: {})
    monkeypatch.setattr(
        matched,
        "load_mnist_train",
        lambda _: (np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32)),
    )
    monkeypatch.setattr(matched, "_screening_dataset_provenance", lambda *_: fake)
    dispatched = 0

    def unexpected_dispatch(*_: object, **__: object) -> Never:
        nonlocal dispatched
        dispatched += 1
        raise AssertionError("reserved seeds must not dispatch")

    monkeypatch.setattr(matched, "_run_matched_adamo_diagnostic", unexpected_dispatch)
    with pytest.raises(ValueError, match="canonical OpenML materialization"):
        matched.run_campaign(Path("unused"), destination)
    assert dispatched == 0
    assert not destination.exists()


def test_report_recomputes_paired_statistics_and_retains_every_outcome(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _build_report(receipts, monkeypatch)
    assert len(report["runs"]) == len(matched.SEEDS)
    assert set(report["paired_comparisons"]) == {
        "adamo_l1e3",
        "adam_iso_joint_l1e3",
    }
    assert all(
        comparison["outcome"] in {"supported", "rejected", "inconclusive"}
        for comparison in report["paired_comparisons"].values()
    )
    assert report["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retained": True,
        "timing_is_telemetry_only": True,
    }
    assert matched.validate_report(report, require_current_execution_identity=True) == report


def test_validator_rejects_missing_seed_tampering_and_promotion(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _build_report(receipts, monkeypatch)

    missing = copy.deepcopy(report)
    missing["runs"].pop()
    with pytest.raises(ValueError, match="complete frozen seed schedule"):
        matched.validate_report(missing, require_current_execution_identity=True)

    arithmetic = copy.deepcopy(report)
    arithmetic["paired_comparisons"]["adamo_l1e3"]["mean_accuracy_delta"] = 1.0
    with pytest.raises(ValueError, match="paired arithmetic"):
        matched.validate_report(arithmetic, require_current_execution_identity=True)

    promoting = copy.deepcopy(report)
    promoting["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="permanently nonpromoting"):
        matched.validate_report(promoting, require_current_execution_identity=True)


def test_atomic_publication_refuses_overwrite(
    tmp_path: Path,
    receipts: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(matched, "OUTPUT_PATH", destination)
    monkeypatch.setattr(matched, "_EXECUTION_AUTHORIZED", True)
    report = _build_report(receipts, monkeypatch)
    matched.publish_report(destination, report)
    with pytest.raises(FileExistsError):
        matched.publish_report(destination, report)


def test_execution_gate_is_closed_until_plan_review() -> None:
    assert not hasattr(matched, "run_adamo_diagnostic")
    with pytest.raises(RuntimeError, match="not authorized"):
        matched.run_campaign(Path("unused.npz"), Path("unused.json"))


def test_student_t_df3_constant_is_exact() -> None:
    assert matched.T95_DF3 == 3.1824463052837078
    assert matched.T95_DF3.hex() == "0x1.975a66893c1a7p+1"


def test_validator_preflights_hostile_nested_plan_without_dispatch(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _build_report(receipts, monkeypatch)

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile iteration")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

    hostile = copy.deepcopy(report)
    hostile["plan"]["arms"] = HostileList(hostile["plan"]["arms"])
    with pytest.raises(ValueError, match="exact JSON"):
        matched.validate_report(hostile, require_current_execution_identity=True)
    assert HostileList.calls == 0


def test_offline_validation_does_not_require_the_execution_runtime(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _build_report(receipts, monkeypatch)
    monkeypatch.setattr(matched, "_current_source_provenance", lambda: {"different": True})
    monkeypatch.setattr(matched, "_current_runtime_environment", lambda: {"different": True})
    assert matched.validate_report(report) == report
    with pytest.raises(ValueError, match="current source"):
        matched.validate_report(report, require_current_execution_identity=True)


def test_build_report_rejects_hostile_sequence_without_hashing_its_metaclass() -> None:
    class ExplosiveMeta(type):
        def __hash__(cls) -> Never:
            raise AssertionError("must not hash a hostile sequence type")

    class HostileSequence(list[object], metaclass=ExplosiveMeta):
        pass

    with pytest.raises(ValueError, match="complete frozen seed schedule"):
        matched.build_report(
            HostileSequence(),
            dataset_provenance={},
            source_provenance={},
            runtime_environment={},
        )


def test_report_schema_subclass_is_rejected_without_equality_hook(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build_report(receipts, monkeypatch)

    class HostileString(str):
        calls = 0

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("must not compare hostile string")

    hostile = copy.deepcopy(report)
    hostile["schema"] = HostileString(matched.SCHEMA)
    with pytest.raises(ValueError, match="schema"):
        matched.validate_report(hostile)
    assert HostileString.calls == 0


def test_dataset_provenance_subclass_is_rejected_before_validator_hooks(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _build_report(receipts, monkeypatch)

    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("must not iterate hostile provenance")

    hostile = copy.deepcopy(report)
    hostile["dataset_provenance"] = HostileDict(hostile["dataset_provenance"])
    with pytest.raises(ValueError, match="exact JSON"):
        matched.validate_report(hostile)
    assert HostileDict.calls == 0


def test_publication_rejects_symlink_parent_without_touching_target(
    tmp_path: Path,
    receipts: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    destination = linked_parent / "report.json"
    monkeypatch.setattr(matched, "OUTPUT_PATH", destination)
    monkeypatch.setattr(matched, "_EXECUTION_AUTHORIZED", True)
    report = _build_report(receipts, monkeypatch)
    with pytest.raises(OSError):
        matched.publish_report(destination, report)
    assert not (real_parent / "report.json").exists()
