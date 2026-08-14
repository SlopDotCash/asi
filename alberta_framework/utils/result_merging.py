"""Result merging and consolidation utilities for multi-run campaigns.

Merges results from parallel campaign runs, handles seed aggregation,
and validates result consistency across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ResultMerger:
    """Merge results from multiple campaign runs."""

    @staticmethod
    def merge_seed_results(
        result_files: list[Path],
        output_path: Path = None,
    ) -> dict[str, Any]:
        """Merge results from multiple seed runs.

        Args:
            result_files: List of result JSON files (one per seed)
            output_path: Where to save merged results

        Returns:
            Merged result dictionary
        """
        all_results = []
        metadata = {"n_files": len(result_files), "input_files": []}

        for result_file in result_files:
            with open(result_file) as f:
                data = json.load(f)
                all_results.append(data)
                metadata["input_files"].append(str(result_file))

        # Merge all measurements
        merged = {
            "campaign": all_results[0].get("campaign", "unknown") if all_results else "unknown",
            "num_runs": len(all_results),
            "metadata": metadata,
            "measurements": [],
        }

        # Consolidate measurements across seeds
        if all_results and "measurements" in all_results[0]:
            # Collect all measurements
            all_measurements = []
            for result in all_results:
                all_measurements.extend(result.get("measurements", []))

            # Group by arm/domain
            grouped = {}
            for measurement in all_measurements:
                key = (measurement.get("arm"), measurement.get("domain"))
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(measurement)

            # Aggregate each group
            for (arm, domain), measurements in grouped.items():
                returns = []
                for m in measurements:
                    if "mean_return" in m:
                        returns.append(m["mean_return"])
                    elif "episodes" in m:
                        eps_returns = [e.get("return_", 0) for e in m.get("episodes", [])]
                        returns.extend(eps_returns)

                if returns:
                    merged["measurements"].append({
                        "arm": arm,
                        "domain": domain,
                        "n_seeds": len(measurements),
                        "mean": float(np.mean(returns)),
                        "std": float(np.std(returns)),
                        "min": float(np.min(returns)),
                        "max": float(np.max(returns)),
                        "sem": float(np.std(returns) / np.sqrt(len(returns))),
                    })

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(merged, f, indent=2)

        return merged

    @staticmethod
    def merge_campaign_results(
        campaign_dirs: list[Path],
        output_path: Path = None,
    ) -> dict[str, Any]:
        """Merge results from multiple campaign directories.

        Args:
            campaign_dirs: List of campaign output directories
            output_path: Where to save merged results

        Returns:
            Merged result dictionary
        """
        all_campaigns = {}

        for campaign_dir in campaign_dirs:
            campaign_dir = Path(campaign_dir)
            campaign_name = campaign_dir.name

            # Find all result files in this campaign
            result_files = list(campaign_dir.glob("**/result.json"))

            if result_files:
                merged = ResultMerger.merge_seed_results(result_files)
                all_campaigns[campaign_name] = merged

        # Consolidate across campaigns
        consolidated = {
            "campaigns": all_campaigns,
            "n_campaigns": len(all_campaigns),
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(consolidated, f, indent=2)

        return consolidated

    @staticmethod
    def validate_result_consistency(
        results1: dict[str, Any],
        results2: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two result sets for consistency.

        Args:
            results1: First result set
            results2: Second result set

        Returns:
            Consistency report
        """
        report = {
            "consistent": True,
            "issues": [],
            "stats": {},
        }

        # Extract measurements
        meas1 = {(m.get("arm"), m.get("domain")): m for m in results1.get("measurements", [])}
        meas2 = {(m.get("arm"), m.get("domain")): m for m in results2.get("measurements", [])}

        # Check coverage
        keys1 = set(meas1.keys())
        keys2 = set(meas2.keys())

        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                report["issues"].append(f"Missing in results2: {missing_in_2}")
            if missing_in_1:
                report["issues"].append(f"Missing in results1: {missing_in_1}")
            report["consistent"] = False

        # Compare values
        common_keys = keys1 & keys2
        diffs = []

        for key in common_keys:
            mean1 = meas1[key].get("mean", 0)
            mean2 = meas2[key].get("mean", 0)
            diff = abs(mean1 - mean2)
            rel_diff = diff / (abs(mean1) + 1e-8)

            if rel_diff > 0.05:  # >5% difference
                diffs.append({
                    "arm_domain": key,
                    "mean1": mean1,
                    "mean2": mean2,
                    "diff": diff,
                    "rel_diff": rel_diff,
                })

        if diffs:
            report["issues"].append(f"Large differences found: {len(diffs)} measurements")
            report["consistent"] = False

        report["stats"] = {
            "common_measurements": len(common_keys),
            "large_differences": len(diffs),
            "consistency_score": 1.0 - min(len(diffs) / max(len(common_keys), 1), 1.0),
        }

        return report


def merge_all_campaigns(output_base: Path = None) -> dict[str, Any]:
    """Convenience function to merge all campaigns in output directory.

    Args:
        output_base: Base output directory (default: outputs/)

    Returns:
        Consolidated results
    """
    output_base = Path(output_base or "outputs")

    if not output_base.exists():
        return {"error": f"Output directory not found: {output_base}"}

    campaign_dirs = [d for d in output_base.iterdir() if d.is_dir()]

    return ResultMerger.merge_campaign_results(
        campaign_dirs,
        output_path=output_base / "consolidated_results.json",
    )
