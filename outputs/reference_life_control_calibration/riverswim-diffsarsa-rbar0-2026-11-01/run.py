"""Full-length RiverSwim calibration of differential SARSA with an optimistic initial R-bar.

Development-only and permanently nonpromoting. Executes one reference life per
(arm, seed) through the same adapters, environment, dispatch and metrics as the
frozen reference-life scorecard, on fresh seeds. The only difference between
``differential_sarsa`` and ``differential_sarsa_rbar0`` is ``initial_average_reward``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from alberta_framework.benchmarks import reference_life_scorecard as sc
from alberta_framework.reference_life import (
    ExactDispatchConfig,
    LifePhase,
    ReferenceLifeMetricsConfig,
    ReferenceLifeRunner,
    RiverSwimReferenceEnvironment,
)
from alberta_framework.reference_life_controls import (
    DifferentialSARSAReferenceAdapter,
    DifferentialSARSAReferenceConfig,
    DiscountedSARSAReferenceAdapter,
    DiscountedSARSAReferenceConfig,
    control_state_resource_usage,
)

SCHEMA = "asi.riverswim_diffsarsa_rbar0_calibration.life.development.v1"
PROTOCOL = "asi.riverswim_diffsarsa_rbar0_calibration.development.v1"
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(91_510, 91_522))
ARMS = ("differential_sarsa", "differential_sarsa_rbar0", "random", "privileged_oracle")
# The largest per-step RiverSwim reward (RIGHT in the last state), so R-bar0 is at least the
# gain of every policy, including the optimal 0.857150...; exact in float32.
RBAR0 = 1.0
Q0_3308 = 10.0
HORIZON = sc.RIVERSWIM_HORIZON  # 20,000: the scorecard's RiverSwim protocol length
EARLY = sc.RIVERSWIM_EARLY_WINDOW
LATE = sc.RIVERSWIM_LATE_WINDOW
HIGH_END_STATE = sc.RIVERSWIM_N_STATES - 1
POLICY = {
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
}
SOURCE_PATHS = (
    "alberta_framework/reference_life_controls.py",
    "alberta_framework/reference_life.py",
    "alberta_framework/core/average_reward.py",
    "alberta_framework/streams/closed_loop.py",
    "alberta_framework/benchmarks/reference_life_scorecard.py",
    "outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/run.py",
)
SCORECARD_SHARDS = ROOT / "outputs/reference_life_scorecard_development/run2-2026-10-09/shards"
Q0_REFERENCE = ROOT / "outputs/reference_life_control_calibration/2026-10-18/optimistic-81010.json"
ENVCHECK_CASES = {
    # case: (arm, seed, horizon, lifecycle source)
    "scorecard_differential_70000": ("differential_sarsa", 70_000, HORIZON, "scorecard"),
    "scorecard_random_70000": ("random", 70_000, HORIZON, "scorecard"),
    "scorecard_oracle_70000": ("privileged_oracle", 70_000, HORIZON, "scorecard"),
    "pr3308_q0_81010": ("sarsa_q0", 81_010, 5_000, "calibration.optimistic.81010"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"git_head": head, "files": {p: _sha256(ROOT / p) for p in SOURCE_PATHS}}


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "dependencies": {
            n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "numpy", "chex")
        },
    }


def lifecycle_id(arm: str, seed: int) -> str:
    digest = hashlib.sha256(f"{PROTOCOL}:riverswim:{arm}:{seed}".encode("ascii")).hexdigest()
    return f"prototype.{digest[:16]}"


def build_adapter(arm: str, horizon: int) -> Any:
    """Scorecard-identical adapters; the R-bar0 arm adds only ``initial_average_reward``.

    ``sarsa_q0`` is the #3308 discounted Q0 = 10 control, used only by the environment check.
    """
    plan = sc.build_development_plan()
    river = sc._river_environment_config(plan)
    if arm == "differential_sarsa_rbar0":
        config = dict(plan.arm_definition("differential_sarsa")["config"])
        return DifferentialSARSAReferenceAdapter(
            DifferentialSARSAReferenceConfig.for_riverswim(
                river, **config, initial_average_reward=RBAR0
            )
        )
    if arm == "sarsa_q0":
        config = dict(plan.arm_definition("sarsa")["config"])
        return DiscountedSARSAReferenceAdapter(
            DiscountedSARSAReferenceConfig.for_riverswim(river, **config, initial_q_value=Q0_3308)
        )
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return sc._control_adapter(
        arm=arm,
        arm_definition=plan.arm_definition(arm),
        environment_kind="riverswim",
        switching_config=None,
        river_config=river,
        horizon=horizon,
    )


def run_life(arm: str, seed: int, *, horizon: int, life_id: str) -> dict[str, Any]:
    plan = sc.build_development_plan()
    river = sc._river_environment_config(plan)
    adapter = build_adapter(arm, horizon)
    dispatch = ExactDispatchConfig(
        executor_id="asi.riverswim.executor", executor_epoch="asi.riverswim.executor_epoch.1"
    )
    environment = RiverSwimReferenceEnvironment(
        river,
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=life_id,
        seed=seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="stationary"),
    )
    started = time.perf_counter()
    outcome: dict[str, Any] = {"status": "failed"}
    try:
        state = runner.init()
        initial = control_state_resource_usage(state.agent_state)
        early = late = 0.0
        high = 0
        first_high: int | None = None
        while state.phase is LifePhase.QUIESCENT:
            step = runner.step(state)
            state = step.state
            if step.event is None:
                continue
            index = state.accepted_events - 1
            reward = float(step.event.transaction.reward)
            if index < EARLY:
                early += reward
            if index >= horizon - LATE:
                late += reward
            if int(state.environment_state.state_index) == HIGH_END_STATE:
                high += 1
                if first_high is None:
                    first_high = state.accepted_events
        final = control_state_resource_usage(state.agent_state)
        reward_sum = float(state.metrics.reward_sum)
        completed = state.phase is LifePhase.COMPLETED and state.accepted_events == horizon
        outcome = {
            "status": "completed" if completed and math.isfinite(reward_sum) else "failed",
            "phase": state.phase.value,
            "accepted_events": int(state.accepted_events),
            "parameter_change_events": int(state.metrics.parameter_change_events),
            "reward_sum": reward_sum,
            "oracle_reward_sum": float(state.metrics.oracle_reward_sum),
            "regret_sum": float(state.metrics.regret_sum),
            "early_reward_sum": early,
            "late_reward_sum": late,
            "high_end_visit_count": high,
            "first_high_end_event": first_high,
            "transcript_sha256": state.transcript_sha256,
            "persistent_bytes_initial": int(initial.persistent_bytes),
            "persistent_bytes_final": int(final.persistent_bytes),
        }
        learner = getattr(state.agent_state, "agent_state", None)
        if hasattr(learner, "average_reward"):
            outcome["final_average_reward"] = float(learner.average_reward)
    except Exception as error:  # retained as a failed life; analysis voids
        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:500]}
    return {
        "schema": SCHEMA,
        "evidence_policy": POLICY,
        "arm": arm,
        "seed": seed,
        "horizon": horizon,
        "lifecycle_id": life_id,
        "initial_average_reward": RBAR0 if arm == "differential_sarsa_rbar0" else None,
        "agent_manifest_id": adapter.manifest.manifest_id,
        "source_identity": source_identity(),
        "runtime_identity": runtime_identity(),
        "outcome": outcome,
        "wall_clock_seconds_telemetry_only": round(time.perf_counter() - started, 3),
    }


def write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as handle:
        handle.write(data)


def pack(raw: Path, out: Path) -> None:
    """One compact line per seed holding its four lives, in fixed arm order."""
    for seed in SEEDS:
        lives = [json.loads((raw / f"{arm}__{seed}.json").read_text()) for arm in ARMS]
        write_new(out / f"seed{seed}.json", {"seed": seed, "lives": lives})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    life = sub.add_parser("life")
    life.add_argument("--arm", choices=ARMS, required=True)
    life.add_argument("--seed", type=int, required=True)
    life.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("envcheck-life")
    check.add_argument("--case", choices=tuple(ENVCHECK_CASES), required=True)
    check.add_argument("--out", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--arm", choices=ARMS, required=True)
    smoke.add_argument("--events", type=int, required=True)
    packer = sub.add_parser("pack")
    packer.add_argument("--raw", type=Path, required=True)
    packer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "life":
        if args.seed not in SEEDS:
            raise SystemExit("seed is outside the fixed development roster")
        record = run_life(
            args.arm, args.seed, horizon=HORIZON, life_id=lifecycle_id(args.arm, args.seed)
        )
        write_new(args.out, record)
        return 0 if record["outcome"]["status"] == "completed" else 1
    if args.command == "envcheck-life":
        arm, seed, horizon, source = ENVCHECK_CASES[args.case]
        if source == "scorecard":
            stored = json.loads((SCORECARD_SHARDS / f"riverswim__{arm}__{seed}.json").read_text())
            source = stored["lifecycle_id"]
        record = run_life(arm, seed, horizon=horizon, life_id=source)
        write_new(args.out, record)
        return 0 if record["outcome"]["status"] == "completed" else 1
    if args.command == "smoke":
        # Plumbing check only, on consumed seed 70000 with a short horizon; prints, writes nothing.
        if not 1 <= args.events <= 100:
            raise SystemExit("smoke is limited to 100 events")
        record = run_life(args.arm, 70_000, horizon=args.events, life_id="smoke.rbar0")
        print(json.dumps(record["outcome"], sort_keys=True))
        return 0 if record["outcome"]["status"] == "completed" else 1
    pack(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
