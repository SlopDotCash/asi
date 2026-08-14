"""Forager RL baseline utilities and measurement infrastructure.

Tools for orchestrating and analyzing Forager RL baseline campaigns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ForagerBaselineOrchestrator:
    """Orchestrate Forager RL baseline measurements."""

    @staticmethod
    def get_baselines() -> dict[str, str]:
        """Get all Forager RL baselines."""
        return {
            "dqn": "Deep Q-Network - off-policy value-based learning",
            "a3c": "Asynchronous Advantage Actor-Critic - on-policy policy gradient",
            "horde": "Hierarchical Off-policy Reinforcement learning Demons - GVF framework",
            "random": "Random action selection - oracle baseline",
        }

    @staticmethod
    def get_phases() -> dict[str, str]:
        """Get Forager measurement phases."""
        return {
            "smoke": "Smoke test - 1 task, 100 episodes - validation",
            "continual": "Continual learning - 5 tasks, 20 episodes each - phase 1",
            "transfer": "Transfer validation - new task distribution - phase 2",
        }

    @staticmethod
    def get_benchmark_performance() -> dict[str, dict[str, float]]:
        """Get preregistered benchmark performance targets."""
        return {
            "dqn": {
                "smoke_mean_return": 0.65,
                "smoke_success_rate": 0.45,
                "continual_mean_return": 0.58,
                "transfer_mean_return": 0.60,
            },
            "a3c": {
                "smoke_mean_return": 0.62,
                "smoke_success_rate": 0.40,
                "continual_mean_return": 0.55,
                "transfer_mean_return": 0.57,
            },
            "horde": {
                "smoke_mean_return": 0.58,
                "smoke_success_rate": 0.35,
                "continual_mean_return": 0.52,
                "transfer_mean_return": 0.54,
            },
            "random": {
                "smoke_mean_return": 0.25,
                "smoke_success_rate": 0.10,
                "continual_mean_return": 0.25,
                "transfer_mean_return": 0.25,
            },
        }

    @staticmethod
    def plan_measurement_campaign(
        baselines: list[str] = None,
        phases: list[str] = None,
        n_seeds: int = 3,
    ) -> dict[str, Any]:
        """Plan complete Forager measurement campaign."""
        if baselines is None:
            baselines = list(ForagerBaselineOrchestrator.get_baselines().keys())
        if phases is None:
            phases = list(ForagerBaselineOrchestrator.get_phases().keys())

        total_runs = len(baselines) * len(phases) * n_seeds
        hours_per_run = 0.5  # Forager is more expensive

        return {
            "campaign": "forager_baselines",
            "baselines": baselines,
            "phases": phases,
            "n_seeds": n_seeds,
            "total_runs": total_runs,
            "estimated_compute_hours": total_runs * hours_per_run,
            "measurement_matrix": {
                "rows": baselines,
                "columns": phases,
                "per_cell_seeds": n_seeds,
            },
        }


class ForagerAnalyzer:
    """Analyze Forager RL baseline results."""

    @staticmethod
    def load_baseline_results(
        results_dir: Path,
        baseline: str,
    ) -> dict[str, list[float]]:
        """Load results for specific baseline across all phases."""
        results_dir = Path(results_dir)
        baseline_results = {}

        for phase in ["smoke", "continual", "transfer"]:
            result_file = results_dir / f"{baseline}_{phase}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    baseline_results[phase] = {
                        "returns": data.get("returns", []),
                        "success_rates": data.get("success_rates", []),
                    }

        return baseline_results

    @staticmethod
    def compute_baseline_statistics(
        baseline_results: dict[str, dict[str, list[float]]],
    ) -> dict[str, Any]:
        """Compute statistics for a baseline across phases."""
        stats = {}

        for phase, data in baseline_results.items():
            returns = data.get("returns", [])
            successes = data.get("success_rates", [])

            if returns:
                stats[phase] = {
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "min_return": float(np.min(returns)),
                    "max_return": float(np.max(returns)),
                    "mean_success_rate": float(np.mean(successes)) if successes else 0.0,
                    "n_runs": len(returns),
                }

        return stats

    @staticmethod
    def rank_baselines_by_performance(
        results: dict[str, dict[str, Any]],
        phase: str = "smoke",
    ) -> list[tuple[str, float]]:
        """Rank baselines by performance on specific phase."""
        rankings = []

        for baseline, stats in results.items():
            if phase in stats:
                mean_return = stats[phase].get("mean_return", 0)
                rankings.append((baseline, mean_return))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    @staticmethod
    def analyze_learning_progression(
        results: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Analyze learning progression across phases (smoke -> continual -> transfer)."""
        progression = {}

        for baseline, stats in results.items():
            if "smoke" in stats and "continual" in stats:
                smoke_return = stats["smoke"].get("mean_return", 0)
                continual_return = stats["continual"].get("mean_return", 0)

                if smoke_return > 0:
                    progression[baseline] = continual_return / smoke_return

        return progression

    @staticmethod
    def compute_sample_efficiency(
        results: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Compute sample efficiency (performance per episode)."""
        efficiency = {}

        for baseline, stats in results.items():
            if "smoke" in stats:
                mean_return = stats["smoke"].get("mean_return", 0)
                n_episodes = 100  # Smoke phase default

                efficiency[baseline] = mean_return / n_episodes if n_episodes > 0 else 0

        return efficiency

    @staticmethod
    def identify_best_baseline(
        results: dict[str, dict[str, Any]],
        phase: str = "smoke",
    ) -> tuple[str, float]:
        """Identify best baseline for specific phase."""
        rankings = ForagerAnalyzer.rank_baselines_by_performance(results, phase)

        if rankings:
            return rankings[0]
        return ("unknown", 0.0)


def generate_forager_report(
    results_dir: Path,
) -> dict[str, Any]:
    """Generate comprehensive Forager baseline report."""
    results_dir = Path(results_dir)

    # Load all results
    all_results = {}
    orchestrator = ForagerBaselineOrchestrator()

    for baseline in orchestrator.get_baselines():
        analyzer = ForagerAnalyzer()
        baseline_results = analyzer.load_baseline_results(results_dir, baseline)
        if baseline_results:
            all_results[baseline] = analyzer.compute_baseline_statistics(baseline_results)

    if not all_results:
        return {"error": "No results found"}

    analyzer = ForagerAnalyzer()

    report = {
        "campaign": "forager_baselines",
        "n_baselines": len(all_results),
        "phases": ["smoke", "continual", "transfer"],
        "baseline_statistics": all_results,
        "ranking_smoke": analyzer.rank_baselines_by_performance(all_results, "smoke"),
        "ranking_continual": analyzer.rank_baselines_by_performance(all_results, "continual"),
        "ranking_transfer": analyzer.rank_baselines_by_performance(all_results, "transfer"),
        "learning_progression": analyzer.analyze_learning_progression(all_results),
        "sample_efficiency": analyzer.compute_sample_efficiency(all_results),
        "best_overall": analyzer.identify_best_baseline(all_results, "smoke"),
    }

    return report
