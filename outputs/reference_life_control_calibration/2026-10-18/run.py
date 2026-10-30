"""Run and summarize the frozen nonpromoting RiverSwim calibration screen."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from alberta_framework.reference_life import (
    ExactDispatchConfig,
    LifePhase,
    ReferenceLifeMetricsConfig,
    ReferenceLifeRunner,
    RiverSwimReferenceEnvironment,
)
from alberta_framework.reference_life_controls import (
    DiscountedSARSAReferenceAdapter,
    DiscountedSARSAReferenceConfig,
    control_state_resource_usage,
)
from alberta_framework.streams.closed_loop import RiverSwimConfig

SCHEMA = "asi.reference_life_riverswim_control_calibration.development.v1"
SUMMARY_SCHEMA = "asi.reference_life_riverswim_control_calibration_summary.development.v1"
ROOT = Path(__file__).resolve().parents[3]
SEEDS = tuple(range(81010, 81016))
ARMS = {"baseline": None, "optimistic": 10.0}
HORIZON = 5_000
WINDOW = 1_000
POLICY = {
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
}
SOURCE_PATHS = (
    "alberta_framework/reference_life_controls.py",
    "alberta_framework/reference_life.py",
    "alberta_framework/streams/closed_loop.py",
    "outputs/reference_life_control_calibration/2026-10-18/run.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, Any]:
    return {
        "git_head": _git_head(),
        "files": {relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS},
    }


def _verify_source_identity(recorded: object) -> dict[str, Any]:
    """Reject shards whose bound sources differ from the summarizing checkout.

    Every recorded source-file digest must equal the digest of the same file in
    the current checkout. The recorded commit must be the current ``HEAD`` or
    one of its ancestors: the commit that retains the shards cannot be named by
    the shards themselves, so equality is required at the file-digest level and
    ancestry at the commit level.
    """
    if type(recorded) is not dict:
        raise ValueError("source identity must be an exact object")
    current = _source_identity()
    if recorded.get("files") != current["files"]:
        raise ValueError("recorded source digests differ from the current checkout")
    recorded_head = recorded.get("git_head")
    if type(recorded_head) is not str or len(recorded_head) != 40:
        raise ValueError("recorded git head is invalid")
    if recorded_head != current["git_head"]:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", recorded_head, "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestry.returncode != 0:
            raise ValueError("recorded git head is not the current HEAD or one of its ancestors")
    return current


def _runtime_identity() -> dict[str, Any]:
    packages = ("jax", "jaxlib", "numpy", "chex")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in packages
        },
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _environment_config() -> RiverSwimConfig:
    return RiverSwimConfig(  # type: ignore[call-arg]
        n_states=6,
        p_right_up=0.35,
        p_right_down=0.05,
        reward_left=0.005,
        reward_right=1.0,
        initial_state=0,
    )


def _control_config(arm: str) -> DiscountedSARSAReferenceConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return DiscountedSARSAReferenceConfig.for_riverswim(
        _environment_config(),
        gamma=0.9,
        epsilon_start=0.5,
        epsilon_end=0.02,
        epsilon_decay_steps=2_500,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        leaky_relu_slope=0.01,
        use_layer_norm=False,
        lamda=0.0,
        initial_q_value=ARMS[arm],
    )


def _run(arm: str, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed is outside the frozen development roster")
    config = _control_config(arm)
    adapter = DiscountedSARSAReferenceAdapter(config)
    dispatch = ExactDispatchConfig(
        executor_id="asi.riverswim.executor",
        executor_epoch="asi.riverswim.executor_epoch.1",
    )
    environment = RiverSwimReferenceEnvironment(
        _environment_config(),
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=f"calibration.{arm}.{seed}",
        seed=seed,
        max_accepted_events=HORIZON,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="stationary"),
    )
    state = runner.init()
    initial_resources = control_state_resource_usage(state.agent_state)
    high_end_visits = 0
    first_high_end_event: int | None = None
    early_reward = 0.0
    late_reward = 0.0
    started = time.perf_counter()
    while state.phase is LifePhase.QUIESCENT:
        step = runner.step(state)
        state = step.state
        if step.event is None:
            continue
        event_index = state.accepted_events - 1
        reward = float(step.event.transaction.reward)
        if event_index < WINDOW:
            early_reward += reward
        if event_index >= HORIZON - WINDOW:
            late_reward += reward
        if int(state.environment_state.state_index) == 5:
            high_end_visits += 1
            if first_high_end_event is None:
                first_high_end_event = state.accepted_events
    elapsed = time.perf_counter() - started
    final_resources = control_state_resource_usage(state.agent_state)
    if state.phase is not LifePhase.COMPLETED or state.accepted_events != HORIZON:
        raise RuntimeError(f"life did not complete: {state.phase.value}")
    return {
        "schema": SCHEMA,
        "evidence_policy": POLICY,
        "arm": arm,
        "seed": seed,
        "horizon": HORIZON,
        "window": WINDOW,
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "agent_manifest_id": adapter.manifest.manifest_id,
        "agent_config": config.to_config(),
        "result": {
            "phase": state.phase.value,
            "accepted_events": state.accepted_events,
            "parameter_change_events": state.metrics.parameter_change_events,
            "reward_sum": state.metrics.reward_sum,
            "oracle_reward_sum": state.metrics.oracle_reward_sum,
            "regret_sum": state.metrics.regret_sum,
            "early_reward_sum": early_reward,
            "late_reward_sum": late_reward,
            "high_end_visit_count": high_end_visits,
            "high_end_visit_rate": high_end_visits / HORIZON,
            "first_high_end_event": first_high_end_event,
            "transcript_sha256": state.transcript_sha256,
            "initial_resources": dataclasses.asdict(initial_resources),
            "final_resources": dataclasses.asdict(final_resources),
            "wall_clock_seconds_telemetry_only": elapsed,
        },
    }


def _validate_record(record: object) -> dict[str, Any]:
    if type(record) is not dict:
        raise ValueError("record must be an exact object")
    payload = record
    if payload.get("schema") != SCHEMA or payload.get("evidence_policy") != POLICY:
        raise ValueError("record schema or evidence policy mismatch")
    arm = payload.get("arm")
    seed = payload.get("seed")
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("record arm/seed is outside the plan")
    if payload.get("horizon") != HORIZON or payload.get("window") != WINDOW:
        raise ValueError("record horizon differs from the plan")
    if payload.get("agent_config") != _control_config(arm).to_config():
        raise ValueError("record agent config differs from the plan")
    result = payload.get("result")
    if type(result) is not dict:
        raise ValueError("record result must be an exact object")
    if result.get("phase") != "completed" or result.get("accepted_events") != HORIZON:
        raise ValueError("record is incomplete")
    for name in ("reward_sum", "oracle_reward_sum", "regret_sum"):
        value = result.get(name)
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"record {name} is invalid")
    if result.get("initial_resources") != result.get("final_resources"):
        raise ValueError("record persistent numeric resources grew")
    return payload


def _summarize(paths: list[Path]) -> dict[str, Any]:
    records = [_validate_record(json.loads(path.read_text())) for path in paths]
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = {(record["arm"], record["seed"]) for record in records}
    if len(records) != len(expected) or observed != expected:
        raise ValueError("records do not exactly cover the frozen roster")
    sources = {json.dumps(record["source_identity"], sort_keys=True) for record in records}
    runtimes = {json.dumps(record["runtime_identity"], sort_keys=True) for record in records}
    if len(sources) != 1 or len(runtimes) != 1:
        raise ValueError("records do not share one source/runtime identity")
    summarizing_identity = _verify_source_identity(records[0]["source_identity"])
    by_pair = {(record["arm"], record["seed"]): record for record in records}
    paired = []
    optimistic_high_end = 0
    resources_equal = True
    for seed in SEEDS:
        baseline = by_pair[("baseline", seed)]["result"]
        optimistic = by_pair[("optimistic", seed)]["result"]
        paired.append(optimistic["reward_sum"] - baseline["reward_sum"])
        optimistic_high_end += int(optimistic["high_end_visit_count"] > 0)
        resources_equal &= (
            baseline["initial_resources"]["persistent_bytes"]
            == optimistic["initial_resources"]["persistent_bytes"]
        )
    positive_pairs = sum(difference > 0.0 for difference in paired)
    mean_difference = statistics.mean(paired)
    retained = (
        positive_pairs >= 5
        and mean_difference > 0.0
        and optimistic_high_end >= 5
        and resources_equal
    )
    arm_summaries = {}
    for arm in ARMS:
        arm_records = [by_pair[(arm, seed)]["result"] for seed in SEEDS]
        arm_summaries[arm] = {
            "mean_reward_sum": statistics.mean(row["reward_sum"] for row in arm_records),
            "mean_late_reward_sum": statistics.mean(
                row["late_reward_sum"] for row in arm_records
            ),
            "high_end_discovery_count": sum(
                row["high_end_visit_count"] > 0 for row in arm_records
            ),
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "evidence_policy": POLICY,
        "source_identity": records[0]["source_identity"],
        "summarizing_source_identity": summarizing_identity,
        "runtime_identity": records[0]["runtime_identity"],
        "seed_roster": list(SEEDS),
        "arm_summaries": arm_summaries,
        "paired_reward_differences": [
            {"seed": seed, "optimistic_minus_baseline": difference}
            for seed, difference in zip(SEEDS, paired, strict=True)
        ],
        "decision": {
            "positive_pair_count": positive_pairs,
            "mean_paired_reward_difference": mean_difference,
            "optimistic_high_end_discovery_count": optimistic_high_end,
            "persistent_resources_equal": resources_equal,
            "retain_mechanism_for_next_calibration": retained,
            "classification": "development_screen_not_scientific_inference",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--shards", type=Path, nargs="+", required=True)
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        _write_new_json(args.out, _run(args.arm, args.seed))
    else:
        _write_new_json(args.out, _summarize(args.shards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
