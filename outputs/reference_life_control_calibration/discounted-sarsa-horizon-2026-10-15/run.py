"""Run one preregistered full reference-life SARSA discount-horizon development shard."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np

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


_REQUIRED_THREAD_SETTINGS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def _require_fixed_thread_settings() -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in _REQUIRED_THREAD_SETTINGS}
    if actual != _REQUIRED_THREAD_SETTINGS:
        raise RuntimeError(
            "comparison requires fixed thread settings "
            f"{_REQUIRED_THREAD_SETTINGS!r}, got {actual!r}"
        )
    return actual


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root: Path) -> dict[str, str]:
    files = sorted((root / "alberta_framework").rglob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    return {p.relative_to(root).as_posix(): sha(p) for p in files}


def diagnostics(state: object) -> dict[str, object]:
    learner = state.agent_state.agent_state.learner_state
    params = learner.head_params
    weights = np.stack([np.asarray(w)[0] for w in params.weights], axis=1)
    biases = np.asarray([np.asarray(b)[0] for b in params.biases])
    traces = np.concatenate(
        [np.asarray(a).ravel() for pair in learner.head_traces for a in pair]
    ).astype(np.float64)
    parameters = np.concatenate([weights.ravel(), biases.ravel()]).astype(np.float64)
    return {
        "q_by_state_action": (weights + biases[None, :]).astype(np.float64).tolist(),
        "eligibility_l2": float(np.linalg.norm(traces)),
        "eligibility_max_abs": float(np.max(np.abs(traces))),
        "parameter_l2": float(np.linalg.norm(parameters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(101001, 101002, 101003, 101099), required=True)
    parser.add_argument(
        "--arm",
        choices=("gamma_090", "gamma_099"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="one excluded event at seed101099")
    args = parser.parse_args()
    thread_settings = _require_fixed_thread_settings()
    if args.out.exists():
        raise SystemExit("Output already exists; overwriting is forbidden")
    root = Path.cwd()
    plan_path = Path(__file__).with_name("plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["scientific_promotion_allowed"] is False
    assert plan["horizon"] == 20000
    if (args.preflight and args.seed != 101099) or (
        not args.preflight and args.seed not in plan["seeds"]
    ):
        parser.error("preflight uses only seed101099; comparison uses only the declared roster")
    horizon = 1 if args.preflight else plan["horizon"]
    source_before = source_identity(root)
    script_hash, plan_hash = sha(Path(__file__)), sha(plan_path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty:
        raise RuntimeError("Commit experiment inputs before starting comparison")
    environment_config = RiverSwimConfig()
    config = DiscountedSARSAReferenceConfig.for_riverswim(
        environment_config, **plan["arms"][args.arm]
    )
    adapter = DiscountedSARSAReferenceAdapter(config)
    dispatch = ExactDispatchConfig(
        executor_id="asi.riverswim.executor",
        executor_epoch="asi.riverswim.executor_epoch.1",
    )
    environment = RiverSwimReferenceEnvironment(
        environment_config,
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=f"discount-horizon.{args.seed}.{args.arm}",
        seed=args.seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="stationary"),
    )
    started = time.monotonic()
    started_at_utc = datetime.now(UTC).isoformat()
    state = runner.init()
    initial_resource = dataclasses.asdict(control_state_resource_usage(state.agent_state))
    snapshots = [{"accepted_events": 0, **diagnostics(state)}]
    rows = []
    try:
        while state.phase is LifePhase.QUIESCENT:
            action = int(state.agent_state.agent_state.last_action)
            result = runner.step(state)
            if not result.accepted:
                raise RuntimeError(result.rejection_reason)
            state = result.state
            rows.append(
                (result.event.transaction.reward, action, int(state.environment_state.state_index))
            )
            if state.accepted_events in plan["fixed_checkpoints"]:
                snapshots.append(
                    {
                        "accepted_events": state.accepted_events,
                        **diagnostics(state),
                    }
                )
            if state.accepted_events % 1000 == 0:
                print(
                    f"seed={args.seed} arm={args.arm} accepted={state.accepted_events}/20000 "
                    f"seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    except BaseException as exc:
        failure = {
            "schema": "asi.riverswim_sarsa_discount_horizon.failure.v1",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "plan_sha256": plan_hash,
            "script_sha256": script_hash,
            "source_commit": commit,
            "seed": args.seed,
            "arm": args.arm,
            "started_at_utc": started_at_utc,
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "accepted_prefix_rows": rows,
            "accepted_events": state.accepted_events,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.with_suffix(".failure.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(failure, handle, sort_keys=True, allow_nan=False)
        raise
    if state.phase is not LifePhase.COMPLETED or state.accepted_events != horizon:
        raise RuntimeError("Incomplete life")
    if source_identity(root) != source_before or sha(plan_path) != plan_hash:
        raise RuntimeError("Source or plan drift during execution")
    if sha(Path(__file__)) != script_hash:
        raise RuntimeError("Experiment script drift during execution")
    trace = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(trace).all():
        raise RuntimeError("Nonfinite trajectory")
    rewards = trace[:, 0]
    upstream = np.flatnonzero(rewards == float(np.float32(environment_config.reward_right)))
    record = {
        "schema": (
            "asi.riverswim_sarsa_discount_horizon.preflight.v1"
            if args.preflight
            else "asi.riverswim_sarsa_discount_horizon.shard.v1"
        ),
        "comparison_eligible": not args.preflight,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "plan_sha256": plan_hash,
        "script_sha256": script_hash,
        "source_commit": commit,
        "worktree_clean_at_start": not dirty,
        "source_sha256": source_before,
        "seed": args.seed,
        "arm": args.arm,
        "config": config.to_config(),
        "life_config": runner.config.config,
        "environment": dataclasses.asdict(environment_config),
        "accepted_events": state.accepted_events,
        "executed_events": state.executed_events,
        "dispatch_attempts": state.dispatch_attempts,
        "environment_rng_cursor": state.environment_rng_cursor,
        "transcript_sha256": state.transcript_sha256,
        "parameter_change_events": state.metrics.parameter_change_events,
        "mean_reward": float(rewards.mean()),
        "late_mean_reward": float(rewards[-2000:].mean()),
        "rightmost_state_visits": int((trace[:, 2] == 5).sum()),
        "first_upstream_reward_event": int(upstream[0] + 1) if upstream.size else None,
        "action_counts": np.bincount(trace[:, 1].astype(np.int64), minlength=2).tolist(),
        "resources": {
            "initial": initial_resource,
            "final": dataclasses.asdict(control_state_resource_usage(state.agent_state)),
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "backend": jax.default_backend(),
            "versions": {
                n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "numpy", "chex")
            },
            "thread_settings": thread_settings,
        },
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds_telemetry": time.monotonic() - started,
        "rewards": rewards.tolist(),
        "actions": trace[:, 1].astype(np.int64).tolist(),
        "positions": trace[:, 2].astype(np.int64).tolist(),
        "snapshots": snapshots,
        "limit": plan["scope_limit"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "seed",
                    "arm",
                    "mean_reward",
                    "late_mean_reward",
                    "rightmost_state_visits",
                    "first_upstream_reward_event",
                    "wall_seconds_telemetry",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
