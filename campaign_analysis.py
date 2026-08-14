"""Batch result analysis and reporting utilities.

Generates comprehensive reports from completed measurement campaigns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CampaignAnalyzer:
    """Analyze and report on completed campaigns."""

    @staticmethod
    def load_campaign_results(campaign_dir: Path) -> dict[str, Any]:
        """Load all results from campaign directory."""
        campaign_dir = Path(campaign_dir)
        results_file = campaign_dir / "campaign_results.json"

        if not results_file.exists():
            return {"error": f"No results found in {campaign_dir}"}

        with open(results_file) as f:
            return json.load(f)

    @staticmethod
    def generate_summary_report(results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary statistics from results."""
        measurements = results.get("measurements", [])

        if not measurements:
            return {"error": "No measurements found"}

        # Group by arm
        by_arm = {}
        for m in measurements:
            arm = m.get("arm") or m.get("learner") or m.get("baseline")
            if arm not in by_arm:
                by_arm[arm] = []
            by_arm[arm].append(m)

        # Compute stats
        summary = {
            "campaign": results.get("campaign", "unknown"),
            "n_measurements": len(measurements),
            "n_arms": len(by_arm),
            "arms": {},
        }

        for arm, arm_measurements in by_arm.items():
            if "mean" in arm_measurements[0]:
                values = [m["mean"] for m in arm_measurements if "mean" in m]
                summary["arms"][arm] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return summary

    @staticmethod
    def rank_arms(results: dict[str, Any]) -> list[tuple[str, float]]:
        """Rank arms by performance."""
        summary = CampaignAnalyzer.generate_summary_report(results)

        arms = summary.get("arms", {})
        ranked = [(arm, stats["mean"]) for arm, stats in arms.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    @staticmethod
    def cross_campaign_comparison(
        campaign_results: dict[str, dict],
    ) -> dict[str, Any]:
        """Compare results across multiple campaigns."""
        comparison = {
            "campaigns": {},
            "common_arms": set(),
        }

        # Load all campaigns
        for campaign_name, results in campaign_results.items():
            summary = CampaignAnalyzer.generate_summary_report(results)
            comparison["campaigns"][campaign_name] = summary

            if campaign_name == list(campaign_results.keys())[0]:
                comparison["common_arms"] = set(summary.get("arms", {}).keys())
            else:
                comparison["common_arms"] &= set(summary.get("arms", {}).keys())

        return comparison

    @staticmethod
    def identify_winners(results: dict[str, Any], top_k: int = 5) -> list[tuple[str, float]]:
        """Identify top-k performing arms."""
        ranked = CampaignAnalyzer.rank_arms(results)
        return ranked[:top_k]

    @staticmethod
    def export_report(
        results: dict[str, Any],
        output_file: Path,
    ) -> None:
        """Export comprehensive report to JSON."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "campaign": results.get("campaign"),
            "summary": CampaignAnalyzer.generate_summary_report(results),
            "ranking": CampaignAnalyzer.rank_arms(results),
            "top_5_winners": CampaignAnalyzer.identify_winners(results, top_k=5),
            "metadata": {
                "total_measurements": len(results.get("measurements", [])),
                "timestamp": str(Path.cwd()),
            },
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)


def batch_analyze_campaigns(output_base: Path = None) -> dict[str, Any]:
    """Analyze all campaigns in output directory."""
    output_base = Path(output_base or "outputs")

    if not output_base.exists():
        return {"error": f"Output directory not found: {output_base}"}

    all_results = {}

    for campaign_dir in output_base.iterdir():
        if campaign_dir.is_dir():
            results = CampaignAnalyzer.load_campaign_results(campaign_dir)
            if "error" not in results:
                all_results[campaign_dir.name] = results

    # Generate comparison
    comparison = CampaignAnalyzer.cross_campaign_comparison(all_results)

    return {
        "campaigns": all_results,
        "comparison": comparison,
        "n_campaigns": len(all_results),
    }


def generate_publication_summary(output_base: Path = None) -> dict[str, Any]:
    """Generate publication-ready summary from all campaigns."""
    output_base = Path(output_base or "outputs")

    analysis = batch_analyze_campaigns(output_base)

    summary = {
        "title": "ASI Measurement Campaign Results",
        "campaigns": {},
        "key_findings": [],
    }

    for campaign_name, results in analysis.get("campaigns", {}).items():
        summary["campaigns"][campaign_name] = {
            "summary": CampaignAnalyzer.generate_summary_report(results),
            "top_performers": CampaignAnalyzer.identify_winners(results, top_k=3),
        }

    # Identify overall winners
    all_rankings = []
    for campaign_name, results in analysis.get("campaigns", {}).items():
        ranked = CampaignAnalyzer.rank_arms(results)
        for rank, (arm, score) in enumerate(ranked[:3]):
            all_rankings.append((arm, score, campaign_name))

    summary["key_findings"] = {
        "top_arms_across_campaigns": sorted(
            all_rankings, key=lambda x: x[1], reverse=True
        )[:10],
        "consistency_check": "Arms appearing in multiple top-3 lists",
    }

    return summary
