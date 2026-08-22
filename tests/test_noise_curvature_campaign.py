"""Fail-closed tests for the issue #1567 noise-curvature campaign contract."""

from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.noise_curvature_campaign as campaign
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult, screening_spec


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((500, 784), dtype=np.float32)
    y = np.arange(500, dtype=np.int32) % 10
    monkeypatch.setattr(campaign, "_canonical_dataset_shapes", lambda: (x.shape, y.shape))
    monkeypatch.setattr(
        campaign, "_canonical_dataset_hashes", lambda: campaign._dataset_hashes(x, y)
    )
    monkeypatch.setattr(campaign, "SEEDS", campaign.CONSUMED_SEEDS)
    return x, y


def _run(arm: str, seed: int, value: float) -> ScreeningRunResult:
    spec = screening_spec(arm)
    config = campaign.CONFIG
    return ScreeningRunResult(
        config_name=arm,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
        noise_mode="step",
    )


def _shards(
    plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    offsets = {
        campaign.LIVE_CONTROL: 0.000,
        "noise_curvature_fixed_adam_l2": 0.010,
        "noise_curvature_gradient_only": 0.020,
        "noise_curvature_volatility_only": 0.025,
        "noise_curvature_combined": 0.050,
    }
    for index, seed in enumerate(campaign.SEEDS):
        for arm in campaign.ARM_ROSTER:
            result = _run(arm, seed, 0.50 + index / 1000 + offsets[arm])
            values.append(campaign._build_shard_for_test(plan, data[0], data[1], result))
    return values


def test_plan_is_fresh_hard_disabled_and_identity_bound(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    plan = campaign.build_plan(*data)
    assert plan["matrix"] == {
        "arms": list(campaign.ARM_ROSTER),
        "seeds": list(campaign.SEEDS),
        "shard_count": 25,
        "ordering": "seed_major_then_arm_roster",
        "execution": "one_shard_per_fresh_python_process",
    }
    assert cast(dict[str, object], plan["execution_gate"])["execution_authorized"] is False
    identity = cast(dict[str, Any], plan["identity"])
    assert identity["dataset"]["sha256"] == campaign._dataset_sha256(*data)
    assert identity["runtime"]["jax"]["version"]
    assert identity["source_sha256"]["pyproject.toml"]
    assert identity["source_sha256"]["uv.lock"]
    assert campaign.validate_plan(copy.deepcopy(plan), data_x=data[0], data_y=data[1]) == plan


def test_production_roster_is_disjoint_from_every_consumed_root() -> None:
    assert campaign.SEEDS == tuple(range(3_186_771_201, 3_186_771_206))
    assert set(campaign.SEEDS).isdisjoint(campaign.CONSUMED_SEEDS)


def test_execution_stops_before_dataset_or_runner_work(
    data: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = campaign.build_plan(*data)
    touched = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal touched
        touched = True
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(campaign, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="separate reviewed authorization transition"):
        campaign.run_shard(
            plan, data[0], data[1], arm=campaign.ARM_ROSTER[0], seed=campaign.SEEDS[0]
        )
    assert touched is False


def test_complete_panel_includes_exact_live_control_and_causal_rules(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    plan = campaign.build_plan(*data)
    aggregate = campaign.build_aggregate(plan, _shards(plan, data))
    summary = cast(dict[str, Any], aggregate["summary"])
    assert summary["mechanism"]["outcome"] == "supported"
    assert summary["causal"]["outcome"] == "supported"
    assert summary["hillclimb"]["outcome"] == "supported"
    assert summary["overall_outcome"] == "supported"
    assert campaign.validate_aggregate(copy.deepcopy(aggregate)) == aggregate


def test_aggregate_rejects_missing_control_identity_and_tampering(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    plan = campaign.build_plan(*data)
    shards = _shards(plan, data)
    with pytest.raises(ValueError, match="complete 5-arm by 5-seed matrix"):
        campaign.build_aggregate(plan, shards[1:])
    hostile = copy.deepcopy(shards)
    hostile[0]["execution_identity"]["schedule_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="execution identity"):
        campaign.build_aggregate(plan, hostile)


def test_replay_recomputes_dataset_schedule_and_initialization(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    plan = campaign.build_plan(*data)
    shard = _shards(plan, data)[0]
    assert campaign.validate_shard_against_dataset(shard, plan, data[0], data[1]) == shard
    changed = data[0].copy()
    changed[0, 0] = 0.5
    with pytest.raises(ValueError, match="dataset"):
        campaign.validate_shard_against_dataset(shard, plan, changed, data[1])

    touched = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal touched
        touched = True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(campaign, "run_screening_config", forbidden)
    try:
        with pytest.raises(RuntimeError, match="authorization transition"):
            campaign.replay_shard(shard, plan, data[0], data[1])
    finally:
        monkeypatch.undo()
    assert touched is False


def test_live_control_contract_rejects_scheduler_or_hyperparameter_drift(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    result = _run(campaign.LIVE_CONTROL, campaign.SEEDS[0], 0.5)
    receipt = campaign.live_control_result_payload(result)
    assert receipt["arm"] == campaign.LIVE_CONTROL
    drift = copy.deepcopy(result)
    drift.hyperparameters["step_size"] = 99.0
    with pytest.raises(ValueError, match="hyperparameters"):
        campaign.live_control_result_payload(drift)


def test_publication_is_reserved_create_only_and_strictly_reloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_EXECUTION_AUTHORIZED", True)
    destination = tmp_path / "result.json"
    assert campaign.write_new_json(destination, {"schema": "test", "value": 1}) == destination
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    assert not (tmp_path / ".result.json.reservation").exists()
    with pytest.raises(FileExistsError, match="immutable output"):
        campaign.write_new_json(destination, {"schema": "test", "value": 2})


def test_cli_reserves_before_dataset_work_and_cleans_failed_first_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_EXECUTION_AUTHORIZED", True)
    destination = tmp_path / "plan.json"
    marker = tmp_path / ".plan.json.reservation"
    failure = tmp_path / ".plan.json.failure"

    def fail_after_reservation(_path: Path) -> tuple[np.ndarray, np.ndarray]:
        assert marker.is_file()
        raise RuntimeError("first dispatch failed")

    monkeypatch.setattr(campaign, "load_mnist_train", fail_after_reservation)
    with pytest.raises(RuntimeError, match="first dispatch failed"):
        campaign.main(["plan", "--data-home", str(tmp_path), "--output", str(destination)])
    assert not destination.exists()
    assert not marker.exists()
    assert failure.is_file()
    with pytest.raises(RuntimeError, match="requires disposition"):
        campaign.main(["plan", "--data-home", str(tmp_path), "--output", str(destination)])
