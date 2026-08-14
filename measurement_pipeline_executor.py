"""Final infrastructure: Measurement pipeline orchestration and execution utilities.

Complete end-to-end measurement execution framework.
"""

from typing import Dict, List, Any, Callable
import json
from pathlib import Path


class MeasurementPipelineExecutor:
    """Execute complete measurement pipeline end-to-end."""

    @staticmethod
    def create_execution_plan(
        campaigns: Dict[str, Dict[str, Any]],
        max_parallel_jobs: int = 4,
    ) -> Dict[str, Any]:
        """Create execution plan for all campaigns."""
        plan = {
            "campaigns": len(campaigns),
            "total_arms": sum(len(c.get("arms", [])) for c in campaigns.values()),
            "total_compute_hours": 0,
            "phases": [],
            "phase_count": 0,
        }

        current_phase = 0
        phase_items = []
        phase_hours = 0

        for campaign_name, config in campaigns.items():
            arms = config.get("arms", [])
            hours_per_arm = config.get("estimated_hours", 3.5) / len(arms) if arms else 0

            for arm in arms:
                if phase_hours + hours_per_arm > max_parallel_jobs and phase_items:
                    plan["phases"].append({
                        "phase": current_phase,
                        "items": phase_items,
                        "compute_hours": phase_hours,
                    })
                    current_phase += 1
                    phase_items = []
                    phase_hours = 0

                phase_items.append({
                    "campaign": campaign_name,
                    "arm": arm,
                    "hours": hours_per_arm,
                })
                phase_hours += hours_per_arm
                plan["total_compute_hours"] += hours_per_arm

        if phase_items:
            plan["phases"].append({
                "phase": current_phase,
                "items": phase_items,
                "compute_hours": phase_hours,
            })

        plan["phase_count"] = len(plan["phases"])
        return plan

    @staticmethod
    def validate_pipeline(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate measurement pipeline."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if plan["phase_count"] == 0:
            validation["errors"].append("No phases in plan")
            validation["valid"] = False

        if plan["total_compute_hours"] == 0:
            validation["errors"].append("No compute hours allocated")
            validation["valid"] = False

        if plan["total_arms"] == 0:
            validation["errors"].append("No arms registered")
            validation["valid"] = False

        if plan["total_compute_hours"] > 500:
            validation["warnings"].append(f"Large compute requirement: {plan['total_compute_hours']}h")

        return validation

    @staticmethod
    def generate_execution_script(plan: Dict[str, Any], output_file: Path) -> None:
        """Generate execution script."""
        script = "#!/bin/bash\n"
        script += "# Measurement Pipeline Execution Script\n"
        script += f"# Total phases: {plan['phase_count']}\n"
        script += f"# Total compute: {plan['total_compute_hours']:.1f}h\n\n"

        for phase_info in plan["phases"]:
            phase = phase_info["phase"]
            script += f"echo 'Phase {phase}: {phase_info[\"compute_hours\"]:.1f}h'\n"

            for item in phase_info["items"]:
                campaign = item["campaign"]
                arm = item["arm"]
                script += f"python measurement_cli.py {campaign} --arm {arm}\n"

            script += "\n"

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(script)


class ResultCollectionOrchestrator:
    """Orchestrate result collection and aggregation."""

    @staticmethod
    def collect_phase_results(
        phase_dir: Path,
        campaign_names: List[str],
    ) -> Dict[str, Any]:
        """Collect all results from phase."""
        phase_dir = Path(phase_dir)
        results = {
            "phase": phase_dir.name,
            "campaigns": {},
            "collection_time": None,
        }

        for campaign in campaign_names:
            campaign_dir = phase_dir / campaign
            if campaign_dir.exists():
                result_files = list(campaign_dir.glob("*.json"))
                results["campaigns"][campaign] = {
                    "result_files": [str(f) for f in result_files],
                    "n_files": len(result_files),
                }

        return results

    @staticmethod
    def aggregate_all_phases(
        output_base: Path,
        phase_count: int,
    ) -> Dict[str, Any]:
        """Aggregate results from all phases."""
        output_base = Path(output_base)
        aggregation = {
            "phases": {},
            "total_phases": phase_count,
            "global_summary": {},
        }

        for phase in range(phase_count):
            phase_dir = output_base / f"phase_{phase}"
            if phase_dir.exists():
                phase_results = ResultCollectionOrchestrator.collect_phase_results(
                    phase_dir,
                    ["ipmnist", "scr", "emnist", "micro_continual", "forager", "rule_discovery"]
                )
                aggregation["phases"][f"phase_{phase}"] = phase_results

        return aggregation


class MeasurementMetricsComputer:
    """Compute aggregate metrics across all measurements."""

    @staticmethod
    def compute_campaign_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute campaign-level statistics."""
        stats = {}

        for campaign_name, campaign_data in results.items():
            measurements = campaign_data.get("measurements", [])

            if measurements:
                means = [m.get("mean", 0) for m in measurements]
                stds = [m.get("std", 0) for m in measurements]

                stats[campaign_name] = {
                    "n_arms": len(measurements),
                    "mean_performance": float(__import__("numpy").mean(means)),
                    "std_performance": float(__import__("numpy").std(means)),
                    "best_arm": max(measurements, key=lambda m: m.get("mean", 0)).get("arm"),
                    "best_score": max(m.get("mean", 0) for m in measurements),
                }

        return stats

    @staticmethod
    def compute_system_metrics(all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute system-wide metrics."""
        metrics = {
            "total_campaigns": len(all_results),
            "total_arms": sum(len(r.get("measurements", [])) for r in all_results.values()),
            "overall_best_arm": None,
            "overall_best_score": 0,
            "campaign_rankings": [],
        }

        # Find overall best
        all_arms = []
        for campaign, data in all_results.items():
            for measurement in data.get("measurements", []):
                all_arms.append((measurement.get("arm"), measurement.get("mean", 0), campaign))

        if all_arms:
            ranked = sorted(all_arms, key=lambda x: x[1], reverse=True)
            metrics["overall_best_arm"] = ranked[0][0]
            metrics["overall_best_score"] = ranked[0][1]
            metrics["campaign_rankings"] = [(c, s) for _, s, c in ranked[:10]]

        return metrics


def generate_final_measurement_report(
    all_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate final comprehensive measurement report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    campaign_stats = MeasurementMetricsComputer.compute_campaign_statistics(all_results)
    system_metrics = MeasurementMetricsComputer.compute_system_metrics(all_results)

    report = {
        "title": "Complete ASI Measurement Campaign Report",
        "campaign_statistics": campaign_stats,
        "system_metrics": system_metrics,
        "timestamp": str(__import__("datetime").datetime.now()),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
