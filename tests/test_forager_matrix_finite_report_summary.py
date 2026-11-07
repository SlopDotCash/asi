"""Regression tests for finite summary and report aggregation in forager_matrix."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matrix as matrix
from alberta_framework.benchmarks.forager import (
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    AlbertaForagerConfig,
    ForagerBatchMode,
    ForagerBenchmarkConfig,
    ForagerRunResult,
    environment_rng_schedule_sha256,
    forager_metric_contract,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "source-root"
    package = root / "alberta_framework"
    benchmarks = package / "benchmarks"
    benchmarks.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='matrix-test'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "FORAGER_BENCHMARK.md").write_text(
        "# Matrix test\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text('"""test package"""\n', encoding="utf-8")
    (package / "py.typed").write_bytes(b"")
    (benchmarks / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(matrix, "REPO_ROOT", root)
    return root


def _manifest_payload(**updates: Any) -> dict[str, Any]:
    seeds = list(updates.get("seeds", [1, 2]))
    counterpart = [matrix._MAX_JAX_SEED - index for index in range(len(seeds))]
    payload: dict[str, Any] = {
        "schema_version": "2.2",
        "preset": "field_of_view",
        "stage": "tuning",
        "steps": 12,
        "seeds": seeds,
        "jax_chunk_size": 4,
        "seed_batch_size": 1,
        "mode": "strict",
        "source_execution_mode": "live_tree_unsealed",
        "metric_evidence_mode": "scalar_summary_unsealed",
        "selection_rule": {
            "metric": "fov_last_10pct_ema_auc",
            "direction": "maximize",
            "statistic": "mean",
            "confidence": 0.9,
            "bootstrap_resamples": 128,
            "bootstrap_seed": 7,
            "tie_break": "variant_id_ascending",
        },
        "variants": {
            "base": {
                "kind": "alberta_horde_ac",
                "selection_group": "policy",
                "config": {},
            }
        },
        "tuning_seeds": seeds,
        "evaluation_seeds": counterpart,
    }
    payload.update(updates)
    return payload


def _make_run_result(
    config: AlbertaForagerConfig,
    benchmark: ForagerBenchmarkConfig,
    seed: int,
    *,
    batch_seeds: Sequence[int],
    mode: ForagerBatchMode = "strict",
    value: float = 1.0,
) -> ForagerRunResult:
    return ForagerRunResult(
        agent="alberta_horde_ac",
        privileged=False,
        seed=seed,
        steps=benchmark.steps,
        total_reward=float(benchmark.steps),
        mean_reward=1.0,
        final_window_mean_reward=1.0,
        final_ewm_reward=1.0,
        mean_ewm_reward=1.0,
        fov_last_10pct_ema_auc=value,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1, benchmark.steps),
        curve_ewm_reward=(1.0, 1.0),
        curve_window_reward=(1.0, 1.0),
        duration_s=1e-12,
        frames_per_second=float(benchmark.steps / 1e-12),
        environment=benchmark.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=benchmark.ewm_decay,
            final_window=benchmark.final_window,
            record_every=benchmark.record_every,
            steps=benchmark.steps,
        ),
        agent_metadata={
            "name": "alberta_horde_ac",
            "privileged": False,
            "seed": seed,
            "config": config.to_dict(),
            "environment_rng_schedule": FORAGER_ENVIRONMENT_RNG_SCHEDULE,
            "environment_rng_schedule_sha256": environment_rng_schedule_sha256(),
            "runner": {
                "kind": "jax_batched_scan",
                "batch_mode": mode,
                "batch_seeds": list(batch_seeds),
                "batch_size": len(batch_seeds),
                "chunk_size": benchmark.jax_chunk_size,
                "overall_duration_s": 1e-12,
                "setup_duration_s": 0.0,
                "compile_duration_s": 0.0,
                "execution_duration_s": 1e-12,
                "aggregate_transitions_per_second": (
                    len(batch_seeds) * benchmark.steps / 1e-12
                ),
                "per_seed_effective_frames_per_second": benchmark.steps / 1e-12,
            },
        },
    )


def test_summary_payload_raises_state_error_on_overflowing_cross_seed_metric() -> None:
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload())
    benchmark = matrix._build_benchmark_config(manifest)
    config = AlbertaForagerConfig()
    runs = (
        _make_run_result(config, benchmark, 1, batch_seeds=(1,), value=1e308),
        _make_run_result(config, benchmark, 2, batch_seeds=(2,), value=1e308),
    )

    with pytest.raises(
        matrix.ForagerMatrixStateError,
        match="variant runs cannot produce a finite benchmark summary",
    ):
        matrix._summary_payload(runs, rule=manifest.selection_rule)


def test_report_payload_raises_state_error_on_overflowing_batch_wall_time_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_horde(
        config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        ordered = tuple(seeds)
        return tuple(
            _make_run_result(
                config,
                benchmark,
                seed,
                batch_seeds=ordered,
                mode=mode,
                value=float(seed),
            )
            for seed in ordered
        )

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", fake_horde)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload())
    output = tmp_path / "matrix"
    report = matrix.run_forager_matrix(manifest, output)
    assert report["status"] == "complete"

    execution_manifest = json.loads(
        (output / matrix.EXECUTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    plan = matrix._batch_plan(manifest)
    batch_payloads: dict[str, dict[str, Any]] = {}
    for item in plan:
        raw_batch = json.loads(
            (output / item.relative_path).read_text(encoding="utf-8")
        )
        raw_batch["wall_time_s"] = 1e308
        batch_payloads[item.relative_path] = raw_batch

    config = AlbertaForagerConfig()
    benchmark = matrix._build_benchmark_config(manifest)
    runs_by_variant = {
        "base": (
            _make_run_result(config, benchmark, 1, batch_seeds=(1,), value=1.0),
            _make_run_result(config, benchmark, 2, batch_seeds=(2,), value=2.0),
        )
    }

    with pytest.raises(
        matrix.ForagerMatrixStateError,
        match="batch wall_time_s values produce a non-finite benchmark_wall_time_s",
    ):
        matrix._report_payload(
            manifest=manifest,
            execution_manifest=execution_manifest,
            plan=plan,
            batch_payloads=batch_payloads,
            runs_by_variant=runs_by_variant,
        )


def test_finite_summary_and_report_payload_remain_exact() -> None:
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload())
    benchmark = matrix._build_benchmark_config(manifest)
    config = AlbertaForagerConfig()
    runs = (
        _make_run_result(config, benchmark, 1, batch_seeds=(1,), value=1.5),
        _make_run_result(config, benchmark, 2, batch_seeds=(2,), value=2.5),
    )
    summary = matrix._summary_payload(runs, rule=manifest.selection_rule)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["seeds"] == [1, 2]
    assert dataclasses.replace(manifest.selection_rule).metric == summary["metric"]
