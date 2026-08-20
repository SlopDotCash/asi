from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.calibrated_partial_reset_campaign as lane
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

SMALL = IPMNISTConfig(n_tasks=1, task_length=8, input_dim=4, hidden1=3, hidden2=2, n_classes=2)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0,
        np.arange(8, dtype=np.int32) % 2,
    )


def _fake_run(
    data_x: np.ndarray,
    data_y: np.ndarray,
    spec: object,
    seed: int,
    config: IPMNISTConfig,
) -> ScreeningRunResult:
    del data_x, data_y
    checked = cast(Any, spec)
    value = 0.4 + 0.01 * (lane.TEST_ONLY_SEEDS.index(seed) + lane.ARMS.index(checked.name))
    return ScreeningRunResult(
        config_name=checked.name,
        base_learner=checked.base_learner,
        hyperparameters=checked.hyperparameters,
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
    )


def _run_for_test(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(lane, "run_screening_config", _fake_run)
    return lane._run(
        *_data(),
        config=SMALL,
        seeds=lane.TEST_ONLY_SEEDS,
        capability=lane._TEST_EXECUTION_CAPABILITY,
    )


def _resign(report: dict[str, object]) -> None:
    unsigned = dict(report)
    unsigned.pop("sha256", None)
    report["sha256"] = hashlib.sha256(lane._canonical(unsigned)).hexdigest()


def test_plan_is_prospective_exact_and_nonpromoting() -> None:
    plan = lane.frozen_plan()
    assert plan["seeds"] == [61_563_001, 61_563_002, 61_563_003, 61_563_004, 61_563_005]
    assert set(plan["seeds"]).isdisjoint(plan["test_only_seeds"])
    assert plan["reviewed_execution_transition"] is False
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["arms"] == list(lane.ARMS)
    assert "pyproject.toml" in plan["source_identity_policy"]["hashed_files"]
    assert "uv.lock" in plan["source_identity_policy"]["hashed_files"]
    assert plan["resources"]["combined_numeric_bytes"] <= 256 * 1024 * 1024
    assert plan["transaction_resources"] == {
        "campaign_rows": 25,
        "initial_runner_dispatches": 25,
        "strict_reexecution_dispatches": 25,
        "total_runner_dispatches": 50,
        "total_observations": 2_000_000,
        "total_updates": 2_000_000,
        "total_data_steps": 2_000_000,
        "total_environment_steps": 0,
        "total_model_queries": 4_000_000,
    }
    runtime = lane._runtime_identity()
    assert runtime["jax"]["config"]["jax_default_prng_impl"] == "threefry2x32"
    assert runtime["jax"]["config"]["jax_random_seed_offset"] == 0


def test_public_transaction_is_closed_before_reservation_or_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("consumer or reservation ran before authorization")

    monkeypatch.setattr(lane, "_reserve", forbidden)
    monkeypatch.setattr(lane, "load_mnist_train", forbidden)
    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_and_publish(tmp_path, tmp_path / "report.json")
    assert calls == 0


def test_private_runner_covers_complete_roster_and_strict_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    assert [(row["seed"], row["arm"]) for row in report["rows"]] == [
        (seed, arm) for seed in lane.TEST_ONLY_SEEDS for arm in lane.ARMS
    ]
    lane.validate_report(
        report,
        *_data(),
        config=SMALL,
        seeds=lane.TEST_ONLY_SEEDS,
        reexecute=True,
    )


def test_validator_rejects_resource_and_aggregate_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    forged = copy.deepcopy(report)
    forged["rows"][0]["result"]["resources"]["updates"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="resource"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )
    forged = copy.deepcopy(report)
    forged["aggregate"]["row_count"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="aggregate"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )


def test_json_boundary_rejects_hostile_nested_type_without_hooks() -> None:
    calls = 0

    class Meta(type):
        def __hash__(cls) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type hash dispatched")

        def __eq__(cls, other: object) -> bool:
            del other
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type equality dispatched")

    class Hostile(metaclass=Meta):
        pass

    with pytest.raises(ValueError, match="exact JSON"):
        lane._bounded_json({"nested": [Hostile()]})
    assert calls == 0


def test_runtime_device_bound_precedes_device_attribute_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Device:
        def __getattribute__(self, name: str) -> object:
            del name
            nonlocal calls
            calls += 1
            raise AssertionError("device attribute accessed before inventory bound")

    monkeypatch.setattr(lane.jax, "devices", lambda: [Device()] * 65)
    with pytest.raises(RuntimeError, match="inventory"):
        lane._runtime_identity()
    assert calls == 0


def test_json_boundary_rejects_unbounded_integer() -> None:
    with pytest.raises(ValueError, match="out-of-bounds integer"):
        lane._bounded_json({"nested": [2**64]})


def test_combined_numeric_bound_precedes_dataset_copy_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _data()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("allocation or runner preceded aggregate bound")

    monkeypatch.setattr(lane, "_MAX_NUMERIC_BYTES", 1)
    monkeypatch.setattr(lane.np, "array", forbidden)
    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="combined numeric allocation"):
        lane._run(
            x,
            y,
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            capability=lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_noncontiguous_dataset_is_rejected_before_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _data()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner preceded dataset allocation validation")

    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="C-contiguous"):
        lane._run(
            x[:, ::-1],
            y,
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            capability=lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_transaction_reserves_before_load_and_retains_tombstone_after_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    marker = destination.with_name(f".{destination.name}.reservation")
    marker.write_bytes(b"occupied")
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("consumer failure")

    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", fail)
    with pytest.raises(FileExistsError):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0
    marker.unlink()
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    monkeypatch.setattr(lane, "run_screening_config", fail)
    with pytest.raises(RuntimeError, match="consumer failure"):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert marker.read_bytes() == b"asi-cpr-consumed-without-result-v1\n"
    with pytest.raises(FileExistsError):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 1


def test_transaction_strictly_publishes_and_removes_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    monkeypatch.setattr(lane, "run_screening_config", _fake_run)
    report = lane._run_and_publish(
        tmp_path,
        destination,
        SMALL,
        lane.TEST_ONLY_SEEDS,
        lane._TEST_EXECUTION_CAPABILITY,
    )
    assert json.loads(destination.read_bytes()) == report
    assert not destination.with_name(f".{destination.name}.reservation").exists()
