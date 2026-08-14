"""Micro-continual improvements utilities - arm orchestration and analysis.

Tools for managing micro-continual measurement campaigns and result analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MicroContinualOrchestrator:
    """Orchestrate micro-continual arm measurements."""

    @staticmethod
    def get_preregistered_arms() -> dict[str, str]:
        """Get all preregistered micro-continual arms."""
        return {
            "rls_head_resid": "RLS readout on penultimate features + residual learning",
            "alignment_first": "Alignment learning before body training",
            "naive_bayes_extended": "Streaming class-conditional Gaussians",
            "dual_speed_rfs_rls": "Dual-speed RFS with RLS head",
            "actor_critic_micro": "Policy gradient + value critic for continual learning",
        }

    @staticmethod
    def get_task_suites() -> dict[str, str]:
        """Get all task suites for micro-continual."""
        return {
            "m1": "Gaussian M1 (40 regimes) - baseline",
            "m2": "Gaussian M2 (60 regimes) - increased complexity",
            "m3": "Gaussian M3 (80 regimes) - high complexity",
            "m4": "Gaussian M4 + M1' transfer - transfer validation",
        }

    @staticmethod
    def get_benchmark_performance() -> dict[str, dict[str, float]]:
        """Get preregistered benchmark performance targets."""
        return {
            "rls_head_resid": {
                "m1": 0.87114,  # ± 0.00010 (n=20)
                "m2": 0.86500,
                "m3": 0.85800,
                "m4_transfer": 0.86200,
            },
            "alignment_first": {
                "m1": 0.86800,
                "m2": 0.86000,
                "m3": 0.85200,
                "m4_transfer": 0.85800,
            },
            "naive_bayes_extended": {
                "m1": 0.86500,
                "m2": 0.85800,
                "m3": 0.84900,
                "m4_transfer": 0.85400,
            },
        }

    @staticmethod
    def plan_measurement_campaign(
        arms: list[str] = None,
        task_suites: list[str] = None,
        n_seeds: int = 3,
    ) -> dict[str, Any]:
        """Plan complete micro-continual measurement campaign."""
        if arms is None:
            arms = list(MicroContinualOrchestrator.get_preregistered_arms().keys())
        if task_suites is None:
            task_suites = list(MicroContinualOrchestrator.get_task_suites().keys())

        total_runs = len(arms) * len(task_suites) * n_seeds
        hours_per_run = 0.2  # Estimated

        return {
            "campaign": "micro_continual_full",
            "arms": arms,
            "task_suites": task_suites,
            "n_seeds": n_seeds,
            "total_runs": total_runs,
            "estimated_compute_hours": total_runs * hours_per_run,
            "measurement_matrix": {
                "rows": arms,
                "columns": task_suites,
                "per_cell_seeds": n_seeds,
            },
        }


class MicroContinualAnalyzer:
    """Analyze micro-continual measurement results."""

    @staticmethod
    def load_arm_results(
        results_dir: Path,
        arm: str,
    ) -> dict[str, list[float]]:
        """Load results for specific arm across all task suites."""
        results_dir = Path(results_dir)
        arm_results = {}

        for task_suite in ["m1", "m2", "m3", "m4"]:
            result_file = results_dir / f"{arm}_{task_suite}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    arm_results[task_suite] = data.get("accuracies", [])

        return arm_results

    @staticmethod
    def compute_arm_statistics(arm_results: dict[str, list[float]]) -> dict[str, Any]:
        """Compute statistics for an arm across task suites."""
        stats = {}

        for task_suite, accuracies in arm_results.items():
            if accuracies:
                stats[task_suite] = {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies)),
                    "n_seeds": len(accuracies),
                }

        return stats

    @staticmethod
    def rank_arms_by_performance(
        results: dict[str, dict[str, float]],
        task_suite: str = "m1",
    ) -> list[tuple[str, float]]:
        """Rank arms by performance on specific task suite."""
        rankings = []

        for arm, stats in results.items():
            if task_suite in stats:
                mean_perf = stats[task_suite].get("mean", 0)
                rankings.append((arm, mean_perf))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    @staticmethod
    def compute_transfer_scores(
        results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute transfer learning scores (M1->M4 correlation)."""
        transfer_scores = {}

        for arm, stats in results.items():
            m1_mean = stats.get("m1", {}).get("mean", 0)
            m4_mean = stats.get("m4", {}).get("mean", 0)

            if m1_mean > 0:
                transfer_score = m4_mean / m1_mean
                transfer_scores[arm] = transfer_score

        return transfer_scores

    @staticmethod
    def identify_best_generalist(
        results: dict[str, dict[str, float]],
    ) -> tuple[str, float]:
        """Identify arm with best average performance across all suites."""
        avg_performances = {}

        for arm, stats in results.items():
            means = [s.get("mean", 0) for s in stats.values()]
            avg_performances[arm] = float(np.mean(means))

        best_arm = max(avg_performances, key=avg_performances.get)
        best_score = avg_performances[best_arm]

        return best_arm, best_score

    @staticmethod
    def identify_suite_specialists(
        results: dict[str, dict[str, float]],
    ) -> dict[str, tuple[str, float]]:
        """Identify best arm for each task suite."""
        specialists = {}

        for task_suite in ["m1", "m2", "m3", "m4"]:
            best_arm = None
            best_score = -1

            for arm, stats in results.items():
                if task_suite in stats:
                    score = stats[task_suite].get("mean", 0)
                    if score > best_score:
                        best_score = score
                        best_arm = arm

            if best_arm:
                specialists[task_suite] = (best_arm, best_score)

        return specialists


def generate_micro_continual_report(
    results_dir: Path,
) -> dict[str, Any]:
    """Generate comprehensive micro-continual report."""
    results_dir = Path(results_dir)

    # Load all results
    all_results = {}
    orchestrator = MicroContinualOrchestrator()

    for arm in orchestrator.get_preregistered_arms():
        analyzer = MicroContinualAnalyzer()
        arm_results = analyzer.load_arm_results(results_dir, arm)
        if arm_results:
            all_results[arm] = analyzer.compute_arm_statistics(arm_results)

    if not all_results:
        return {"error": "No results found"}

    analyzer = MicroContinualAnalyzer()

    report = {
        "campaign": "micro_continual",
        "n_arms": len(all_results),
        "task_suites": ["m1", "m2", "m3", "m4"],
        "arm_statistics": all_results,
        "ranking_m1": analyzer.rank_arms_by_performance(all_results, "m1"),
        "ranking_m4": analyzer.rank_arms_by_performance(all_results, "m4"),
        "transfer_scores": analyzer.compute_transfer_scores(all_results),
        "best_generalist": analyzer.identify_best_generalist(all_results),
        "suite_specialists": analyzer.identify_suite_specialists(all_results),
    }

    return report
