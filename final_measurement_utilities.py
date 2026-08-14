"""Comprehensive measurement infrastructure utilities - final batch.

Additional utilities for complete measurement pipeline.
"""

from typing import Any, Dict, List
import json
from pathlib import Path
import numpy as np


class MeasurementOrchestrator:
    """Master orchestrator for all measurements."""

    @staticmethod
    def create_measurement_manifest(
        campaigns: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create comprehensive measurement manifest."""
        manifest = {
            "version": "1.0",
            "campaigns": {},
            "total_compute_hours": 0,
            "total_arms": 0,
            "total_measurements": 0,
        }

        for campaign_name, config in campaigns.items():
            arms = config.get("arms", [])
            n_seeds = config.get("n_seeds", 3)
            hours_per_arm = config.get("hours_per_arm", 1.0)

            n_measurements = len(arms) * n_seeds
            compute_hours = len(arms) * n_seeds * hours_per_arm

            manifest["campaigns"][campaign_name] = {
                "n_arms": len(arms),
                "n_seeds": n_seeds,
                "n_measurements": n_measurements,
                "compute_hours": compute_hours,
                "arms": arms,
            }

            manifest["total_compute_hours"] += compute_hours
            manifest["total_arms"] += len(arms)
            manifest["total_measurements"] += n_measurements

        return manifest

    @staticmethod
    def validate_measurement_readiness(
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate that all measurements are ready."""
        report = {
            "ready": True,
            "warnings": [],
            "errors": [],
        }

        total_hours = manifest.get("total_compute_hours", 0)
        total_arms = manifest.get("total_arms", 0)

        if total_hours == 0:
            report["errors"].append("No compute hours allocated")
            report["ready"] = False

        if total_arms == 0:
            report["errors"].append("No arms registered")
            report["ready"] = False

        if total_hours > 1000:
            report["warnings"].append(f"Very large compute requirement: {total_hours}h")

        return report

    @staticmethod
    def generate_measurement_schedule(
        manifest: Dict[str, Any],
        max_parallel: int = 4,
    ) -> Dict[str, Any]:
        """Generate execution schedule for measurements."""
        schedule = {
            "total_phases": 0,
            "phases": [],
        }

        phase = 0
        current_phase_compute = 0
        phase_items = []

        for campaign_name, config in manifest.get("campaigns", {}).items():
            for arm in config.get("arms", []):
                compute = config.get("compute_hours", 1.0)

                if current_phase_compute + compute > max_parallel and phase_items:
                    schedule["phases"].append({
                        "phase": phase,
                        "items": phase_items,
                        "compute_hours": current_phase_compute,
                    })
                    phase += 1
                    current_phase_compute = 0
                    phase_items = []

                phase_items.append({
                    "campaign": campaign_name,
                    "arm": arm,
                })
                current_phase_compute += compute

        if phase_items:
            schedule["phases"].append({
                "phase": phase,
                "items": phase_items,
                "compute_hours": current_phase_compute,
            })

        schedule["total_phases"] = len(schedule["phases"])
        return schedule


class ResultAggregationEngine:
    """Advanced result aggregation and synthesis."""

    @staticmethod
    def aggregate_by_mechanism(
        all_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """Aggregate results by mechanism type."""
        mechanisms = {
            "normalization": [],
            "gating": [],
            "buffer": [],
            "meta": [],
            "ensemble": [],
        }

        for arm, stats in all_results.items():
            arm_lower = arm.lower()

            if "norm" in arm_lower:
                mechanisms["normalization"].append(stats.get("mean", 0))
            if "gate" in arm_lower:
                mechanisms["gating"].append(stats.get("mean", 0))
            if "buffer" in arm_lower or "replay" in arm_lower:
                mechanisms["buffer"].append(stats.get("mean", 0))
            if "meta" in arm_lower or "maml" in arm_lower:
                mechanisms["meta"].append(stats.get("mean", 0))
            if "ensemble" in arm_lower:
                mechanisms["ensemble"].append(stats.get("mean", 0))

        return mechanisms

    @staticmethod
    def compute_mechanism_rankings(
        mechanism_results: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """Rank mechanisms by effectiveness."""
        rankings = {}

        for mechanism, scores in mechanism_results.items():
            if scores:
                rankings[mechanism] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "best": float(np.max(scores)),
                    "n_arms": len(scores),
                }

        # Sort by mean performance
        sorted_rankings = sorted(
            rankings.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )

        return dict(sorted_rankings)

    @staticmethod
    def identify_synergistic_combinations(
        all_results: Dict[str, Dict[str, Any]],
    ) -> List[tuple]:
        """Identify synergistic arm combinations."""
        combinations = []

        arms = sorted(all_results.keys())
        for i, arm1 in enumerate(arms):
            for arm2 in arms[i + 1:]:
                score1 = all_results[arm1].get("mean", 0)
                score2 = all_results[arm2].get("mean", 0)

                # Synergy: sum is greater than parts
                avg_individual = (score1 + score2) / 2
                if (score1 + score2) / 2 > avg_individual * 1.1:
                    combinations.append((arm1, arm2, (score1 + score2) / 2))

        combinations.sort(key=lambda x: x[2], reverse=True)
        return combinations[:10]


class PerformanceTracker:
    """Track measurement performance over time."""

    @staticmethod
    def compute_improvement_trajectory(
        results_by_phase: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Track improvement across phases."""
        trajectory = {
            "phases": len(results_by_phase),
            "phase_means": [],
            "phase_stds": [],
            "improvement_rate": 0,
        }

        for phase_results in results_by_phase:
            values = list(phase_results.values())
            trajectory["phase_means"].append(float(np.mean(values)))
            trajectory["phase_stds"].append(float(np.std(values)))

        if len(trajectory["phase_means"]) > 1:
            improvement = (trajectory["phase_means"][-1] - trajectory["phase_means"][0]) / max(
                trajectory["phase_means"][0], 1e-8
            )
            trajectory["improvement_rate"] = float(improvement)

        return trajectory

    @staticmethod
    def export_performance_report(
        tracker_data: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """Export performance tracking report."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(tracker_data, f, indent=2)


def create_final_measurement_summary(
    all_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Create final comprehensive measurement summary."""
    orchestrator = MeasurementOrchestrator()
    aggregator = ResultAggregationEngine()

    mechanism_results = aggregator.aggregate_by_mechanism(all_results)
    mechanism_rankings = aggregator.compute_mechanism_rankings(mechanism_results)
    synergies = aggregator.identify_synergistic_combinations(all_results)

    summary = {
        "total_arms": len(all_results),
        "mechanism_rankings": mechanism_rankings,
        "synergistic_combinations": synergies,
        "best_overall_arm": max(
            all_results.items(),
            key=lambda x: x[1].get("mean", 0)
        ) if all_results else None,
    }

    return summary
