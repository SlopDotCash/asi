"""Switching development probe: the Prototype base step size 0.05 -> 0.10.

Development-only and permanently nonpromoting. This tunes the CANDIDATE (the Prototype),
not a control. Every life runs through the scorecard's own builders, environment,
dispatch, metrics and streaming summary on fresh seeds. ``prototype_step010`` is the
scorecard ``prototype`` arm with exactly one field changed, ``oak.stomp.base_step_size``,
applied here as a runner-level override; no registered source is edited.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    SwitchingTwoStateReferenceEnvironment,
    build_prototype_switching_life,
)

SCHEMA = "asi.switching_prototype_step_probe.life.development.v1"
PROTOCOL = "asi.switching_prototype_step_probe.development.v1"
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ENVIRONMENT = "switching_two_state"
SEEDS = tuple(range(93_610, 93_622))
PROBE_ARM = "prototype_step010"
PROBE_STEP = 0.10  # fixed in DEVELOPMENT_PLAN.md before any run; never retuned
ARMS = (
    PROBE_ARM,
    "prototype",
    "prototype_frozen",
    "differential_sarsa",
    "sarsa",
    "random",
    "privileged_oracle",
)
HORIZON = sc.SWITCHING_HORIZON  # 4,000: the scorecard's Switching protocol length
POLICY = {
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
}
SOURCE_PATHS = (
    "alberta_framework/benchmarks/reference_life_scorecard.py",
    "alberta_framework/reference_life.py",
    "alberta_framework/reference_life_controls.py",
    "alberta_framework/prototype_reference_adapter.py",
    "alberta_framework/core/prototype_agent.py",
    "alberta_framework/core/options.py",
    "alberta_framework/core/multi_head_learner.py",
    "alberta_framework/streams/closed_loop.py",
    "outputs/reference_life_control_calibration/switching-prototype-step-2026-11-01/run.py",
)
SCORECARD_SHARDS = ROOT / "outputs/reference_life_scorecard_development/run2-2026-10-09/shards"
STORED_ARMS = (
    "prototype",
    "prototype_frozen",
    "differential_sarsa",
    "sarsa",
    "random",
    "privileged_oracle",
)
# Environment check on consumed scorecard seed 70000: every stored arm, plus the probe arm.
ENVCHECK_CASES = {f"scorecard_{arm}_70000": (arm, 70_000) for arm in STORED_ARMS} | {
    "probe_step010_70000": (PROBE_ARM, 70_000)
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
    digest = hashlib.sha256(f"{PROTOCOL}:{ENVIRONMENT}:{arm}:{seed}".encode("ascii")).hexdigest()
    return f"prototype.{digest[:16]}"


def scorecard_arm(arm: str) -> str:
    """The scorecard arm whose definition (and parameter-change rule) an arm inherits."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return "prototype" if arm == PROBE_ARM else arm


def prototype_config(arm: str) -> Any:
    """The scorecard Prototype config; the probe arm replaces only the base step size."""
    plan = sc.build_development_plan()
    config = sc._prototype_agent_config(plan, environment_kind=ENVIRONMENT, arm=scorecard_arm(arm))
    if arm != PROBE_ARM:
        return config
    stomp = dataclasses.replace(config.oak.stomp, base_step_size=PROBE_STEP)
    return dataclasses.replace(config, oak=dataclasses.replace(config.oak, stomp=stomp))


def build_runner(arm: str, seed: int, *, horizon: int, life_id: str) -> ReferenceLifeRunner:
    """Mirror of ``sc.build_scorecard_runner`` for Switching, free of the fixed seed roster."""
    plan = sc.build_development_plan()
    switching = sc._switching_environment_config(plan)
    if scorecard_arm(arm) in ("prototype", "prototype_frozen"):
        return build_prototype_switching_life(
            agent_config=prototype_config(arm),
            environment_config=switching,
            lifecycle_id=life_id,
            seed=seed,
            max_accepted_events=horizon,
        )
    adapter = sc._control_adapter(
        arm=arm,
        arm_definition=plan.arm_definition(arm),
        environment_kind=ENVIRONMENT,
        switching_config=switching,
        river_config=None,
        horizon=horizon,
    )
    dispatch = ExactDispatchConfig()
    environment = SwitchingTwoStateReferenceEnvironment(
        switching,
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=life_id,
        seed=seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="switching_two_phase"),
    )


def run_life(arm: str, seed: int, *, horizon: int, life_id: str) -> dict[str, Any]:
    """One life through the scorecard's own step loop and streaming summary."""
    import numpy as np

    plan = sc.build_development_plan()
    protocol = plan.protocol(ENVIRONMENT)
    # The streaming summary needs the full A/B/A windows; short smoke lives skip it.
    streaming = (
        sc.StreamingRunSummary.for_switching(
            horizon=horizon,
            phase_length=protocol["phase_length"],
            post_switch_window=protocol["post_switch_window"],
        )
        if horizon == HORIZON
        else None
    )
    started = time.perf_counter()
    manifest_id: str | None = None
    manifest: Any = None
    environment_manifest_id: str | None = None
    setup_seconds: float | None = None
    step_seconds: list[float] = []
    try:
        runner = build_runner(arm, seed, horizon=horizon, life_id=life_id)
        manifest_id = runner.agent_adapter.manifest.manifest_id
        manifest = sc._agent_manifest_descriptor(runner.agent_adapter.manifest)
        environment_manifest_id = runner.environment_adapter.manifest.descriptor()["manifest_id"]
        state = runner.init()
        initial = sc._agent_resource_payload(runner.agent_adapter, state.agent_state)
        setup_seconds = time.perf_counter() - started
        while state.phase is LifePhase.QUIESCENT:
            step_started = time.perf_counter()
            step = runner.step(state)
            step_seconds.append(time.perf_counter() - step_started)
            if step.accepted and streaming is not None:
                if step.event is None:
                    raise RuntimeError("accepted runner step did not expose its event")
                observation = step.event.transaction.next_decision_observation
                if observation is None:
                    raise RuntimeError("continuing event lacks its next observation")
                streaming.observe(
                    reward=step.event.transaction.reward,
                    oracle_reward=step.event.oracle_reward,
                    regime_id=step.event.regime_id,
                    parameters_changed=step.event.step_result.parameters_changed,
                    next_state_index=int(np.argmax(observation.to_numpy())),
                )
            state = step.state
            if not step.accepted:
                raise RuntimeError(step.rejection_reason or "reference life rejected a step")
        if state.phase is not LifePhase.COMPLETED:
            raise RuntimeError("life left quiescent execution without completing")
        outcome = (
            streaming.finalize()
            if streaming is not None
            else {
                "accepted_events": int(state.accepted_events),
                "reward_sum": float(state.metrics.reward_sum),
                "parameter_change_events": int(state.metrics.parameter_change_events),
            }
        )
        if state.accepted_events != outcome["accepted_events"]:
            raise RuntimeError("runner and streaming accepted-event counts disagree")
        outcome.update(
            {
                "environment_rng_cursor": state.environment_rng_cursor,
                "transcript_sha256": state.transcript_sha256,
                "resources": {
                    "initial": initial,
                    "final": sc._agent_resource_payload(runner.agent_adapter, state.agent_state),
                },
                "parameter_change_check": sc.parameter_change_check(
                    scorecard_arm(arm), state.metrics.parameter_change_events
                ),
            }
        )
        finite = math.isfinite(float(outcome["reward_sum"]))
        outcome["status"] = "completed" if finite else "failed"
    except Exception as error:  # retained as a failed life; analysis voids
        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:500]}
    base_step = (
        prototype_config(arm).oak.stomp.base_step_size
        if scorecard_arm(arm) in ("prototype", "prototype_frozen")
        else None
    )
    return {
        "schema": SCHEMA,
        "evidence_policy": POLICY,
        "environment_kind": ENVIRONMENT,
        "arm": arm,
        "seed": seed,
        "horizon": horizon,
        "lifecycle_id": life_id,
        "prototype_base_step_size": base_step,
        "agent_manifest_id": manifest_id,
        "agent_manifest": manifest,
        "environment_manifest_id": environment_manifest_id,
        "source_identity": source_identity(),
        "runtime_identity": runtime_identity(),
        "outcome": outcome,
        "wall_clock_seconds_telemetry_only": round(time.perf_counter() - started, 3),
        "telemetry_only": {
            "setup_seconds": None if setup_seconds is None else round(setup_seconds, 4),
            "cold_step_seconds": round(step_seconds[0], 4) if step_seconds else None,
            "warm_step_seconds_mean": (
                round(math.fsum(step_seconds[1:]) / (len(step_seconds) - 1), 6)
                if len(step_seconds) > 1
                else None
            ),
        },
    }


def write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as handle:
        handle.write(data)


def pack(raw: Path, out: Path) -> None:
    """One compact file per seed holding its lives in fixed arm order."""
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
        arm, seed = ENVCHECK_CASES[args.case]
        stored_arm = scorecard_arm(arm)
        stored = json.loads(
            (SCORECARD_SHARDS / f"{ENVIRONMENT}__{stored_arm}__{seed}.json").read_text()
        )
        record = run_life(arm, seed, horizon=HORIZON, life_id=stored["lifecycle_id"])
        write_new(args.out, record)
        return 0 if record["outcome"]["status"] == "completed" else 1
    if args.command == "smoke":
        # Plumbing and timing check only, on consumed seed 70000; prints, writes nothing.
        if not 1 <= args.events <= 100:
            raise SystemExit("smoke is limited to 100 events")
        record = run_life(args.arm, 70_000, horizon=args.events, life_id=lifecycle_id("smoke", 0))
        out = record["outcome"]
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "status": out.get("status"),
                    "error": out.get("error"),
                    "reward_sum": out.get("reward_sum"),
                    "parameter_change_events": out.get("parameter_change_events"),
                    "base_step": record["prototype_base_step_size"],
                    "manifest": record["agent_manifest_id"],
                    "seconds": record["wall_clock_seconds_telemetry_only"],
                    "telemetry": record["telemetry_only"],
                }
            )
        )
        return 0 if out["status"] == "completed" else 1
    pack(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
