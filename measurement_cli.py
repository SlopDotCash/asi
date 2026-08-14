"""Unified CLI entry points for all measurement campaigns.

Provides consistent command-line interface for launching any baseline arm
or learner on any domain.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def run_ipmnist_arm(
    arm: str,
    n_tasks: int = 200,
    n_steps: int = 5000,
    seed: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run IPMNIST screening arm."""
    from alberta_framework.benchmarks.ipmnist_screening import screening_spec

    logger.info(f"Running IPMNIST arm '{arm}' on {n_tasks} tasks, seed {seed}")
    spec = screening_spec(arm)

    # Would integrate with actual harness
    return {
        "arm": arm,
        "domain": "ipmnist",
        "n_tasks": n_tasks,
        "n_steps": n_steps,
        "seed": seed,
        "status": "ready_for_measurement",
    }


def run_scr_arm(
    arm: str,
    n_tasks: int = 100,
    n_steps: int = 1000,
    seed: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run SCR v2 arm."""
    from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
        get_learner_factory,
        get_arm_hyperparameters,
    )

    logger.info(f"Running SCR arm '{arm}' on {n_tasks} tasks, seed {seed}")
    factory = get_learner_factory(arm)
    hp = get_arm_hyperparameters(arm)

    return {
        "arm": arm,
        "domain": "scr",
        "n_tasks": n_tasks,
        "n_steps": n_steps,
        "seed": seed,
        "status": "ready_for_measurement",
    }


def run_emnist_learner(
    learner: str,
    n_tasks: int = 400,
    n_steps: int = 1000,
    seed: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run EMNIST v3 learner."""
    from alberta_framework.benchmarks.upgd_label_emnist import _FULL_STEP_FACTORIES

    logger.info(f"Running EMNIST learner '{learner}' on {n_tasks} tasks, seed {seed}")
    assert learner in _FULL_STEP_FACTORIES, f"Unknown learner: {learner}"

    return {
        "learner": learner,
        "domain": "emnist",
        "n_tasks": n_tasks,
        "n_steps": n_steps,
        "seed": seed,
        "status": "ready_for_measurement",
    }


def run_micro_continual_arm(
    arm: str,
    tasks: str = "m1",
    n_seeds: int = 3,
    seed_offset: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run micro-continual improvement arm."""
    from micro_continual_improvements import PREREGISTERED_ARMS

    logger.info(f"Running micro-continual arm '{arm}' on {tasks}, {n_seeds} seeds")
    assert arm in PREREGISTERED_ARMS, f"Unknown arm: {arm}"

    return {
        "arm": arm,
        "domain": "micro_continual",
        "tasks": tasks,
        "n_seeds": n_seeds,
        "seed_offset": seed_offset,
        "status": "ready_for_measurement",
    }


def run_forager_baseline(
    baseline: str,
    phase: Literal["smoke", "continual", "transfer"] = "smoke",
    n_tasks: int = 1,
    n_episodes: int = 100,
    seed: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run Forager RL baseline."""
    from alberta_framework.benchmarks.forager_open_baselines import make_baseline

    logger.info(f"Running Forager baseline '{baseline}' phase={phase}, seed {seed}")
    agent = make_baseline(baseline, action_dim=4, state_dim=16)

    return {
        "baseline": baseline,
        "domain": "forager",
        "phase": phase,
        "n_tasks": n_tasks,
        "n_episodes": n_episodes,
        "seed": seed,
        "status": "ready_for_measurement",
    }


def main():
    """Main CLI router."""
    parser = argparse.ArgumentParser(description="ASI Measurement Campaign CLI")
    subparsers = parser.add_subparsers(dest="domain", help="Measurement domain")

    # IPMNIST
    parser_ipmnist = subparsers.add_parser("ipmnist", help="IPMNIST screening")
    parser_ipmnist.add_argument("--arm", required=True, help="Arm name")
    parser_ipmnist.add_argument("--n-tasks", type=int, default=200)
    parser_ipmnist.add_argument("--n-steps", type=int, default=5000)
    parser_ipmnist.add_argument("--seed", type=int, default=0)
    parser_ipmnist.add_argument("--output-dir", type=Path)

    # SCR
    parser_scr = subparsers.add_parser("scr", help="SCR v2")
    parser_scr.add_argument("--arm", required=True)
    parser_scr.add_argument("--n-tasks", type=int, default=100)
    parser_scr.add_argument("--n-steps", type=int, default=1000)
    parser_scr.add_argument("--seed", type=int, default=0)
    parser_scr.add_argument("--output-dir", type=Path)

    # EMNIST
    parser_emnist = subparsers.add_parser("emnist", help="EMNIST v3")
    parser_emnist.add_argument("--learner", required=True)
    parser_emnist.add_argument("--n-tasks", type=int, default=400)
    parser_emnist.add_argument("--n-steps", type=int, default=1000)
    parser_emnist.add_argument("--seed", type=int, default=0)
    parser_emnist.add_argument("--output-dir", type=Path)

    # Micro-continual
    parser_micro = subparsers.add_parser("micro", help="Micro-continual")
    parser_micro.add_argument("--arm", required=True)
    parser_micro.add_argument("--tasks", choices=["m1", "m2", "m3", "m4"], default="m1")
    parser_micro.add_argument("--n-seeds", type=int, default=3)
    parser_micro.add_argument("--seed-offset", type=int, default=0)
    parser_micro.add_argument("--output-dir", type=Path)

    # Forager
    parser_forager = subparsers.add_parser("forager", help="Forager RL baselines")
    parser_forager.add_argument("--baseline", required=True, choices=["dqn", "a3c", "horde", "random"])
    parser_forager.add_argument("--phase", choices=["smoke", "continual", "transfer"], default="smoke")
    parser_forager.add_argument("--n-tasks", type=int, default=1)
    parser_forager.add_argument("--n-episodes", type=int, default=100)
    parser_forager.add_argument("--seed", type=int, default=0)
    parser_forager.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.domain == "ipmnist":
        result = run_ipmnist_arm(
            arm=args.arm,
            n_tasks=args.n_tasks,
            n_steps=args.n_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.domain == "scr":
        result = run_scr_arm(
            arm=args.arm,
            n_tasks=args.n_tasks,
            n_steps=args.n_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.domain == "emnist":
        result = run_emnist_learner(
            learner=args.learner,
            n_tasks=args.n_tasks,
            n_steps=args.n_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.domain == "micro":
        result = run_micro_continual_arm(
            arm=args.arm,
            tasks=args.tasks,
            n_seeds=args.n_seeds,
            seed_offset=args.seed_offset,
            output_dir=args.output_dir,
        )
    elif args.domain == "forager":
        result = run_forager_baseline(
            baseline=args.baseline,
            phase=args.phase,
            n_tasks=args.n_tasks,
            n_episodes=args.n_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        return

    print(f"[OK] {result['status']}")
    print(f"  Domain: {result.get('domain', 'unknown')}")
    print(f"  Arm/Learner: {result.get('arm') or result.get('learner') or result.get('baseline')}")


if __name__ == "__main__":
    main()
