"""Regression tests for finite metric aggregation in historical_forager."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerArtifactError,
    HistoricalForagerContractError,
    HistoricalForagerRunConfig,
    HistoricalUpdateKernel,
    _HistoricalMetricAccumulator,
    development_historical_environment_adapter,
    historical_fov_metrics,
    run_historical_forager,
    validate_historical_forager_artifact,
)

pytestmark = pytest.mark.unit


class _SimpleEnvironment:
    def __init__(self, rewards: list[float]) -> None:
        self._rewards = rewards
        self._offset = 0

    def start(self) -> int:
        return 0

    def step(self, action: int) -> tuple[float, int, bool, Mapping[str, Any]]:
        _ = action
        reward = self._rewards[self._offset]
        self._offset += 1
        return reward, self._offset, False, {}


def _make_lane(
    tmp_path: Path,
    rewards: list[float],
) -> tuple[Any, HistoricalUpdateKernel[int], HistoricalForagerRunConfig]:
    adapter = development_historical_environment_adapter(
        lambda seed, aperture_size: _SimpleEnvironment(rewards)
    )
    kernel = HistoricalUpdateKernel(
        name="simple_kernel",
        start_kernel=lambda obs: (0, 0),
        update_kernel=lambda state, reward, obs: (state + 1, 0),
        metadata={"implementation": "test"},
    )
    config = HistoricalForagerRunConfig(
        seed=0,
        steps=len(rewards),
        aperture_size=9,
        output_directory=tmp_path / "run",
        allow_unverified_development_adapter=True,
    )
    return adapter, kernel, config


def test_historical_fov_metrics_rejects_overflowing_finite_rewards() -> None:
    rewards = np.asarray([1e308, 1e308], dtype=np.float64)
    with pytest.raises(HistoricalForagerContractError, match="non-finite"):
        historical_fov_metrics(rewards)


def test_historical_metric_accumulator_finalize_rejects_overflowing_tail_mean() -> None:
    acc = _HistoricalMetricAccumulator(steps=1001)
    for _ in range(1001):
        acc.append(0.0)
    acc.tail_samples = [1e308, 1e308]
    with pytest.raises(HistoricalForagerContractError, match="non-finite"):
        acc.finalize()


def test_validate_historical_forager_artifact_translates_overflowing_sidecar_to_artifact_error(
    tmp_path: Path,
) -> None:
    adapter, kernel, config = _make_lane(tmp_path, [1.0, 2.0])
    run_historical_forager(adapter, kernel, config)
    reward_path = config.output_directory / "rewards.npy"
    result_path = config.output_directory / "result.json"

    reward_path.chmod(0o600)
    overflow_rewards = np.asarray([1e308, 1e308], dtype=np.dtype("<f8"))
    np.save(reward_path, overflow_rewards, allow_pickle=False)
    reward_bytes = reward_path.read_bytes()
    reward_path.chmod(0o444)

    result_path.chmod(0o600)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["reward_sidecar"]["sha256"] = hashlib.sha256(reward_bytes).hexdigest()
    manifest["reward_sidecar"]["size"] = len(reward_bytes)
    result_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o444)

    with pytest.raises(HistoricalForagerArtifactError, match="non-finite"):
        validate_historical_forager_artifact(config.output_directory)


def test_historical_fov_metrics_preserves_exact_finite_values() -> None:
    rewards = np.asarray([1.0, -0.5, 2.25, 0.0, 1.5], dtype=np.float64)
    metrics = historical_fov_metrics(rewards)
    assert metrics["total_reward"] == float(np.sum(rewards))
    assert np.isfinite(metrics["fov_last_10pct_ema_auc"])
    assert np.isfinite(metrics["final_unadjusted_ema"])
