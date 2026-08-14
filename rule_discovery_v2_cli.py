"""Rule Discovery V2 Integration CLI and utilities.

Provides command-line tools and utilities for Phase 1-2 rule discovery campaigns
including template injection, result validation, and comparison infrastructure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from alberta_framework.benchmarks.rule_discovery import (
    GENOME_SIZE,
    FLAG_NAMES,
    describe_genome,
    seed_genomes,
)
from rule_discovery_v2_integration import expand_seed_genomes_with_templates
from rule_discovery_v2_templates import RULE_DISCOVERY_V2_TEMPLATES

logger = logging.getLogger(__name__)


class RuleDiscoveryV2Runner:
    """Orchestrate Rule Discovery v2 phases."""

    def __init__(self, output_dir: Path = Path("outputs/rule_discovery")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def phase_1a_validate_ordering(
        self,
        ipmnist_results: dict[str, float],
        gaussian_results: dict[str, float],
    ) -> dict[str, Any]:
        """Phase 1a: Verify that IPMNIST and Gaussian orderings match.

        Args:
            ipmnist_results: {arm_name: performance}
            gaussian_results: {arm_name: performance}

        Returns:
            Validation report with Spearman correlation
        """
        common_arms = set(ipmnist_results.keys()) & set(gaussian_results.keys())

        if len(common_arms) < 3:
            logger.warning("Not enough common arms for statistical comparison")
            return {"status": "insufficient_data", "n_common": len(common_arms)}

        # Rank both domains
        ipmnist_ranked = sorted(
            [(arm, score) for arm, score in ipmnist_results.items() if arm in common_arms],
            key=lambda x: x[1],
            reverse=True,
        )
        gaussian_ranked = sorted(
            [(arm, score) for arm, score in gaussian_results.items() if arm in common_arms],
            key=lambda x: x[1],
            reverse=True,
        )

        # Compute Spearman correlation
        ipmnist_ranks = {arm: i for i, (arm, _) in enumerate(ipmnist_ranked)}
        gaussian_ranks = {arm: i for i, (arm, _) in enumerate(gaussian_ranked)}

        ranks_x = [ipmnist_ranks[arm] for arm in sorted(common_arms)]
        ranks_y = [gaussian_ranks[arm] for arm in sorted(common_arms)]

        correlation = np.corrcoef(ranks_x, ranks_y)[0, 1]

        return {
            "status": "complete",
            "n_common": len(common_arms),
            "spearman_correlation": float(np.nan_to_num(correlation, nan=0.0)),
            "ipmnist_ranking": ipmnist_ranked[:5],
            "gaussian_ranking": gaussian_ranked[:5],
            "ordering_match": float(np.nan_to_num(correlation, nan=0.0)) > 0.8,
        }

    def phase_1b_prepare_rerun(self) -> dict[str, Any]:
        """Phase 1b: Prepare seed genomes for rerun on Gaussian fitness.

        Returns:
            Metadata for v1 search candidates rerun
        """
        original_seeds = np.asarray(seed_genomes(), dtype=np.float32)
        expanded_seeds = expand_seed_genomes_with_templates()

        return {
            "original_seed_count": len(original_seeds),
            "expanded_seed_count": len(expanded_seeds),
            "template_count": len(RULE_DISCOVERY_V2_TEMPLATES),
            "new_genomes": len(expanded_seeds) - len(original_seeds),
            "genome_size": GENOME_SIZE,
            "flag_names": FLAG_NAMES,
        }

    def phase_1c_compare_results(
        self,
        v1_ipmnist: dict[str, float],
        v1_gaussian_rerun: dict[str, float],
        v2_gaussian: dict[str, float],
    ) -> dict[str, Any]:
        """Phase 1c: Compare v1 baseline, v1 rerun, and v2 search.

        Args:
            v1_ipmnist: Original v1 results on IPMNIST
            v1_gaussian_rerun: v1 candidates rerun on Gaussian fitness
            v2_gaussian: v2 expanded search on Gaussian fitness

        Returns:
            Comparison report with ranking shifts and new winners
        """
        all_arms = set(v1_ipmnist.keys()) | set(v1_gaussian_rerun.keys()) | set(v2_gaussian.keys())

        report = {
            "n_total_arms": len(all_arms),
            "v1_ipmnist_count": len(v1_ipmnist),
            "v1_gaussian_rerun_count": len(v1_gaussian_rerun),
            "v2_gaussian_count": len(v2_gaussian),
            "new_arms_v2": len(set(v2_gaussian.keys()) - set(v1_gaussian_rerun.keys())),
            "top_5_v1_ipmnist": sorted(v1_ipmnist.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_5_v1_gaussian_rerun": sorted(
                v1_gaussian_rerun.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_5_v2_gaussian": sorted(v2_gaussian.items(), key=lambda x: x[1], reverse=True)[:5],
        }

        # Identify winners
        v1_rerun_champion = max(v1_gaussian_rerun.items(), key=lambda x: x[1])[1] if v1_gaussian_rerun else 0
        v2_best = max(v2_gaussian.items(), key=lambda x: x[1])[1] if v2_gaussian else 0

        report["v2_improvement_over_v1_gaussian"] = float(v2_best - v1_rerun_champion)
        report["has_new_winner"] = v2_best > v1_rerun_champion * 1.01  # >1% improvement

        return report

    def save_phase_1_report(self, report: dict[str, Any], phase: str) -> Path:
        """Save phase 1 report to JSON."""
        output_file = self.output_dir / f"phase_1{phase}_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Phase 1{phase} report saved to {output_file}")
        return output_file


def main():
    """CLI entry point for rule discovery v2 integration."""
    parser = argparse.ArgumentParser(description="Rule Discovery V2 Integration")
    subparsers = parser.add_subparsers(dest="phase", help="Phase to execute")

    # Phase 1a: Validate ordering
    parser_1a = subparsers.add_parser("phase_1a", help="Validate IPMNIST->Gaussian ordering")
    parser_1a.add_argument("--ipmnist-results", type=Path, required=True)
    parser_1a.add_argument("--gaussian-results", type=Path, required=True)
    parser_1a.add_argument("--output-dir", type=Path, default=Path("outputs/rule_discovery"))

    # Phase 1b: Prepare rerun
    parser_1b = subparsers.add_parser("phase_1b", help="Prepare expanded seed genomes")
    parser_1b.add_argument("--output-dir", type=Path, default=Path("outputs/rule_discovery"))

    # Phase 1c: Compare results
    parser_1c = subparsers.add_parser("phase_1c", help="Compare v1 vs v2 results")
    parser_1c.add_argument("--v1-ipmnist", type=Path, required=True)
    parser_1c.add_argument("--v1-gaussian-rerun", type=Path, required=True)
    parser_1c.add_argument("--v2-gaussian", type=Path, required=True)
    parser_1c.add_argument("--output-dir", type=Path, default=Path("outputs/rule_discovery"))

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runner = RuleDiscoveryV2Runner(output_dir=args.output_dir)

    if args.phase == "phase_1a":
        with open(args.ipmnist_results) as f:
            ipmnist = json.load(f)
        with open(args.gaussian_results) as f:
            gaussian = json.load(f)
        report = runner.phase_1a_validate_ordering(ipmnist, gaussian)
        runner.save_phase_1_report(report, "a")

    elif args.phase == "phase_1b":
        report = runner.phase_1b_prepare_rerun()
        runner.save_phase_1_report(report, "b")

    elif args.phase == "phase_1c":
        with open(args.v1_ipmnist) as f:
            v1_ipmnist = json.load(f)
        with open(args.v1_gaussian_rerun) as f:
            v1_gaussian = json.load(f)
        with open(args.v2_gaussian) as f:
            v2_gaussian = json.load(f)
        report = runner.phase_1c_compare_results(v1_ipmnist, v1_gaussian, v2_gaussian)
        runner.save_phase_1_report(report, "c")


if __name__ == "__main__":
    main()
