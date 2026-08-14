"""Integration of final search strategies with existing rule discovery.

Provides CLI and programmatic interfaces to run Rule Discovery V2 with
advanced optimization strategies: Bayesian optimization, hypervolume
multi-objective optimization, Thompson sampling, and active learning.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.rule_discovery import (
    GENOME_SIZE,
    HOLDOUT_TASKS,
    SEARCH_TASKS,
    champion_form_genome,
    decode_genome,
    describe_genome,
    evaluate_suite,
    flag_count,
    penalized_fitness,
    random_genomes,
    run_search,
    seed_genomes,
    _net_config,
    _resolved_suite,
)

from rule_discovery_final_search import (
    ActiveLearningCurriculum,
    BayesianOptimizer,
    FinalSearchStrategy,
    HypervolumOptimizer,
    SearchConfig,
    ThompsonSamplerBandit,
    create_search_summary,
)

logger = logging.getLogger(__name__)

FINAL_SEARCH_SCHEMA = "alberta.rule_discovery.final_search.v1"

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}


@dataclasses.dataclass(frozen=True)
class FinalSearchParams:
    """Parameters for final search execution."""

    # Search budget
    total_evaluations: int = 10000
    batch_size: int = 64
    max_generations: int = 100

    # Evaluation
    eval_seeds: tuple[int, ...] = (0, 1)
    holdout_seeds: tuple[int, ...] = (101, 102, 103)
    task_names: tuple[str, ...] = SEARCH_TASKS
    holdout_names: tuple[str, ...] = HOLDOUT_TASKS

    # Strategy flags
    use_bayesian: bool = True
    use_hypervolume: bool = True
    use_thompson: bool = True
    use_active_learning: bool = True

    # Strategy hyperparameters
    gp_kernel: str = "matern"
    ei_xi: float = 0.01
    reference_point: tuple[float, ...] = (-0.1, 1.0, -0.1)
    mechanism_families: tuple[str, ...] = (
        "baseline", "normalization", "gating", "surprise", "rls", "ensemble"
    )
    curriculum_mode: str = "difficulty"

    # Batch size
    batch_size_eval: int = 256

    # Suite
    suite_kind: str = "digits"
    micro_n_tasks: int | None = None
    micro_task_length: int | None = None


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (matching rule_discovery.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def run_final_search(params: FinalSearchParams) -> dict[str, Any]:
    """Execute final search with integrated strategies.

    Returns result payload with all search metadata and results.
    """
    started = time.monotonic()
    key = jr.key(0)

    # Resolve suite
    suite = _resolved_suite(
        params.micro_n_tasks,
        params.micro_task_length,
        params.suite_kind,
    )

    logger.info(
        "final search: bayesian=%s hypervolume=%s thompson=%s curriculum=%s",
        params.use_bayesian,
        params.use_hypervolume,
        params.use_thompson,
        params.use_active_learning,
    )

    # Initialize strategy
    config = SearchConfig(
        use_bayesian=params.use_bayesian,
        use_hypervolume=params.use_hypervolume,
        use_thompson=params.use_thompson,
        use_active_learning=params.use_active_learning,
        gp_kernel=params.gp_kernel,
        ei_xi=params.ei_xi,
        reference_point=params.reference_point,
        mechanism_families=params.mechanism_families,
        curriculum_mode=params.curriculum_mode,
        total_evaluations=params.total_evaluations,
        batch_size=params.batch_size,
        max_generations=params.max_generations,
    )
    strategy = FinalSearchStrategy(config, key)

    # Generate initial population
    key, key_init = jr.split(key)
    seeds_block = seed_genomes()
    n_seeded = seeds_block.shape[0]
    randoms = random_genomes(
        key_init,
        max(params.batch_size * 8 - n_seeded, 0),
    )
    initial_pool = np.concatenate([np.asarray(seeds_block), np.asarray(randoms)])

    logger.info(
        "initial pool: %d seeded + %d random = %d total",
        n_seeded,
        initial_pool.shape[0] - n_seeded,
        initial_pool.shape[0],
    )

    # Evaluate initial pool
    accuracy_initial, per_task_initial = evaluate_suite(
        initial_pool,
        params.task_names,
        seeds=params.eval_seeds,
        batch_size=params.batch_size_eval,
        suite=suite,
    )
    fitness_initial = penalized_fitness(accuracy_initial, initial_pool)

    # Initialize archive
    archive_genomes: list[np.ndarray] = [np.asarray(g) for g in initial_pool]
    archive_accuracy: list[float] = [float(a) for a in accuracy_initial]
    archive_fitness: list[float] = [float(f) for f in fitness_initial]
    archive_origin: list[str] = ["seeded"] * n_seeded + ["random"] * (
        initial_pool.shape[0] - n_seeded
    )

    # Main search loop
    generation_log: list[dict[str, Any]] = []
    best_fitness = float(np.max(fitness_initial))
    best_genome_idx = int(np.argmax(fitness_initial))

    logger.info(
        "random+seeded phase: best fitness %.5f accuracy %.5f (%s)",
        best_fitness,
        archive_accuracy[best_genome_idx],
        describe_genome(archive_genomes[best_genome_idx]),
    )

    candidate_pool = np.concatenate([initial_pool, random_genomes(key, 10000)])

    for generation in range(params.max_generations):
        key, key_gen = jr.split(key)

        # Select batch using integrated strategies
        batch_indices = strategy.select_next_batch(
            archive_genomes[:params.batch_size * 8],
            archive_accuracy[:params.batch_size * 8],
            candidate_pool,
            generation,
        )

        batch_genomes = candidate_pool[batch_indices]
        batch_accuracy, batch_per_task = evaluate_suite(
            batch_genomes,
            params.task_names,
            seeds=params.eval_seeds,
            batch_size=params.batch_size_eval,
            suite=suite,
        )
        batch_fitness = penalized_fitness(batch_accuracy, batch_genomes)

        # Log batch results
        log = strategy.log_step(generation, list(batch_accuracy), batch_genomes)

        # Update archive
        for i, (genome, acc, fit) in enumerate(
            zip(batch_genomes, batch_accuracy, batch_fitness)
        ):
            archive_genomes.append(np.asarray(genome))
            archive_accuracy.append(float(acc))
            archive_fitness.append(float(fit))
            archive_origin.append(f"generation_{generation}")

        # Track best
        gen_best_idx = int(np.argmax(batch_fitness))
        if batch_fitness[gen_best_idx] > best_fitness:
            best_fitness = float(batch_fitness[gen_best_idx])
            best_genome_idx = len(archive_genomes) - len(batch_fitness) + gen_best_idx

        generation_log.append({
            **log,
            "batch_size": len(batch_indices),
            "best_accuracy": float(batch_accuracy[gen_best_idx]),
            "best_description": describe_genome(batch_genomes[gen_best_idx]),
        })

        logger.info(
            "generation %d: best fitness %.5f (%s), hypervolume %.4f",
            generation,
            best_fitness,
            generation_log[-1]["best_description"],
            log.get("hypervolume", 0.0),
        )

    # Holdout validation
    top_k = 12
    archive_fit = np.asarray(archive_fitness)
    unique: dict[bytes, int] = {}
    for raw_idx in np.argsort(-archive_fit):
        idx = int(raw_idx)
        digest = archive_genomes[idx].tobytes()
        if digest not in unique:
            unique[digest] = idx
        if len(unique) >= top_k:
            break

    top_indices = list(unique.values())
    candidates = np.stack([archive_genomes[i] for i in top_indices])

    reference = champion_form_genome()[None, :]
    holdout_pool = np.concatenate([reference, candidates], axis=0)

    holdout_mean, holdout_per_task = evaluate_suite(
        holdout_pool,
        params.holdout_names,
        seeds=params.holdout_seeds,
        batch_size=params.batch_size_eval,
        suite=suite,
    )

    reference_holdout = float(holdout_mean[0])
    candidate_rows: list[dict[str, Any]] = []

    for rank, archive_idx in enumerate(top_indices):
        row = {
            "rank_by_search_fitness": rank,
            "genome": [float(v) for v in archive_genomes[archive_idx]],
            "config": decode_genome(archive_genomes[archive_idx]),
            "description": describe_genome(archive_genomes[archive_idx]),
            "origin": archive_origin[archive_idx],
            "active_flags": flag_count(archive_genomes[archive_idx]),
            "search_fitness": float(archive_fit[archive_idx]),
            "search_accuracy": float(archive_accuracy[archive_idx]),
            "holdout_accuracy": float(holdout_mean[1 + rank]),
            "holdout_per_task": {
                name: float(values[1 + rank])
                for name, values in holdout_per_task.items()
            },
            "beats_reference_on_holdout": bool(
                float(holdout_mean[1 + rank]) > reference_holdout
            ),
        }
        candidate_rows.append(row)

    promoted = sorted(
        (row for row in candidate_rows if row["beats_reference_on_holdout"]),
        key=lambda row: -float(row["holdout_accuracy"]),
    )[:3]

    gauss_lane = params.suite_kind == "gauss"

    payload: dict[str, Any] = {
        "schema": FINAL_SEARCH_SCHEMA,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "strategy": {
            "bayesian_optimization": params.use_bayesian,
            "hypervolume_optimization": params.use_hypervolume,
            "thompson_sampling": params.use_thompson,
            "active_learning_curriculum": params.use_active_learning,
            "gp_kernel": params.gp_kernel,
            "ei_xi": params.ei_xi,
            "mechanism_families": params.mechanism_families,
        },
        "settings": {
            "total_evaluations": params.total_evaluations,
            "batch_size": params.batch_size,
            "max_generations": params.max_generations,
            "eval_seeds": list(params.eval_seeds),
            "holdout_seeds": list(params.holdout_seeds),
            "task_names": list(params.task_names),
            "holdout_names": list(params.holdout_names),
        },
        "n_evaluated": len(archive_genomes),
        "champion_reference": {
            "description": describe_genome(champion_form_genome()),
            "holdout_accuracy": reference_holdout,
            "holdout_per_task": {
                name: float(values[0]) for name, values in holdout_per_task.items()
            },
        },
        "generation_log": generation_log,
        "candidates": candidate_rows,
        "promoted": promoted,
        "search_history": strategy.search_history,
        "wall_clock_seconds": round(time.monotonic() - started, 2),
    }

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for final search."""
    parser = argparse.ArgumentParser(
        description="Rule Discovery final search with advanced optimization"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-evaluations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument(
        "--use-bayesian", action="store_true", default=True,
        help="Enable Bayesian optimization"
    )
    parser.add_argument(
        "--use-hypervolume", action="store_true", default=True,
        help="Enable hypervolume multi-objective optimization"
    )
    parser.add_argument(
        "--use-thompson", action="store_true", default=True,
        help="Enable Thompson sampling"
    )
    parser.add_argument(
        "--use-active-learning", action="store_true", default=True,
        help="Enable active learning curriculum"
    )
    parser.add_argument("--gp-kernel", choices=["rbf", "matern"], default="matern")
    parser.add_argument("--ei-xi", type=float, default=0.01)
    parser.add_argument("--suite", choices=["digits", "gauss"], default="digits")
    parser.add_argument("--micro-n-tasks", type=int, default=None)
    parser.add_argument("--micro-task-length", type=int, default=None)
    parser.add_argument("--batch-size-eval", type=int, default=256)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    params = FinalSearchParams(
        total_evaluations=args.total_evaluations,
        batch_size=args.batch_size,
        max_generations=args.max_generations,
        eval_seeds=tuple(args.eval_seeds),
        holdout_seeds=tuple(args.holdout_seeds),
        use_bayesian=args.use_bayesian,
        use_hypervolume=args.use_hypervolume,
        use_thompson=args.use_thompson,
        use_active_learning=args.use_active_learning,
        gp_kernel=args.gp_kernel,
        ei_xi=args.ei_xi,
        suite_kind=args.suite,
        micro_n_tasks=args.micro_n_tasks,
        micro_task_length=args.micro_task_length,
        batch_size_eval=args.batch_size_eval,
    )

    payload = run_final_search(params)
    _atomic_write_json(args.out, payload)

    logger.info(
        "final search complete: %d evaluated, %d promoted -> %s",
        payload["n_evaluated"],
        len(payload["promoted"]),
        args.out,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
