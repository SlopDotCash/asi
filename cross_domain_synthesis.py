"""Cross-domain benchmark comparison and synthesis utilities.

Compare results across all measurement domains for holistic analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CrossDomainComparator:
    """Compare and synthesize results across all domains."""

    DOMAINS = ["ipmnist", "scr", "emnist", "micro_continual", "forager"]

    @staticmethod
    def load_all_domain_results(results_base: Path) -> dict[str, dict[str, Any]]:
        """Load results from all measurement domains."""
        results_base = Path(results_base)
        all_results = {}

        for domain in CrossDomainComparator.DOMAINS:
            domain_file = results_base / f"{domain}_results.json"
            if domain_file.exists():
                with open(domain_file) as f:
                    all_results[domain] = json.load(f)

        return all_results

    @staticmethod
    def extract_arm_performances(
        domain_results: dict[str, Any],
        domain_name: str,
    ) -> dict[str, float]:
        """Extract mean performance for each arm in a domain."""
        performances = {}

        measurements = domain_results.get("measurements", [])
        for measurement in measurements:
            arm = (
                measurement.get("arm")
                or measurement.get("learner")
                or measurement.get("baseline")
            )
            performance = measurement.get("mean", 0)

            if arm:
                performances[arm] = performance

        return performances

    @staticmethod
    def compute_domain_statistics(
        all_results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Compute statistics for each domain."""
        stats = {}

        for domain, results in all_results.items():
            performances = CrossDomainComparator.extract_arm_performances(results, domain)

            if performances:
                values = list(performances.values())
                stats[domain] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_arms": len(performances),
                }

        return stats

    @staticmethod
    def identify_domain_specialists(
        all_results: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[str, float]]:
        """Identify best arm in each domain."""
        specialists = {}

        for domain, results in all_results.items():
            performances = CrossDomainComparator.extract_arm_performances(results, domain)

            if performances:
                best_arm = max(performances, key=performances.get)
                best_score = performances[best_arm]
                specialists[domain] = (best_arm, best_score)

        return specialists

    @staticmethod
    def identify_generalists(
        all_results: dict[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        """Identify arms that perform well across multiple domains."""
        arm_scores = {}

        for domain, results in all_results.items():
            performances = CrossDomainComparator.extract_arm_performances(results, domain)

            for arm, score in performances.items():
                if arm not in arm_scores:
                    arm_scores[arm] = []
                arm_scores[arm].append(score)

        # Score by mean performance across domains
        generalist_scores = []
        for arm, scores in arm_scores.items():
            if len(scores) >= 2:  # Must appear in multiple domains
                avg_score = np.mean(scores)
                generalist_scores.append((arm, float(avg_score)))

        generalist_scores.sort(key=lambda x: x[1], reverse=True)
        return generalist_scores

    @staticmethod
    def compute_transfer_potential(
        all_results: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Compute transfer learning potential between domains."""
        transfer_scores = {}

        domains_list = list(all_results.keys())
        for i, source_domain in enumerate(domains_list):
            for target_domain in domains_list[i + 1 :]:
                source_perfs = CrossDomainComparator.extract_arm_performances(
                    all_results[source_domain], source_domain
                )
                target_perfs = CrossDomainComparator.extract_arm_performances(
                    all_results[target_domain], target_domain
                )

                # Find common arms
                common_arms = set(source_perfs.keys()) & set(target_perfs.keys())

                if common_arms:
                    source_scores = [source_perfs[arm] for arm in common_arms]
                    target_scores = [target_perfs[arm] for arm in common_arms]

                    # Spearman correlation of rankings
                    source_ranks = sorted(range(len(source_scores)), key=lambda k: source_scores[k])
                    target_ranks = sorted(range(len(target_scores)), key=lambda k: target_scores[k])

                    correlation = np.corrcoef(source_ranks, target_ranks)[0, 1]
                    key = f"{source_domain}_to_{target_domain}"
                    transfer_scores[key] = float(np.nan_to_num(correlation, nan=0.0))

        return transfer_scores

    @staticmethod
    def identify_robustness_leaders(
        all_results: dict[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        """Identify arms with most consistent performance across domains."""
        arm_consistency = {}

        for domain, results in all_results.items():
            performances = CrossDomainComparator.extract_arm_performances(results, domain)

            for arm, score in performances.items():
                if arm not in arm_consistency:
                    arm_consistency[arm] = []
                arm_consistency[arm].append(score)

        # Score by consistency (low coefficient of variation)
        consistency_scores = []
        for arm, scores in arm_consistency.items():
            if len(scores) >= 2:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv = std_score / (mean_score + 1e-8)  # Coefficient of variation
                consistency_scores.append((arm, 1.0 / (1.0 + cv)))  # Higher is better

        consistency_scores.sort(key=lambda x: x[1], reverse=True)
        return consistency_scores


def generate_cross_domain_synthesis_report(
    results_base: Path = None,
) -> dict[str, Any]:
    """Generate comprehensive cross-domain synthesis report."""
    results_base = Path(results_base or "outputs")

    # Load all results
    all_results = CrossDomainComparator.load_all_domain_results(results_base)

    if not all_results:
        return {"error": "No results found"}

    comparator = CrossDomainComparator()

    report = {
        "title": "Cross-Domain Benchmark Synthesis",
        "domains_analyzed": list(all_results.keys()),
        "domain_statistics": comparator.compute_domain_statistics(all_results),
        "domain_specialists": comparator.identify_domain_specialists(all_results),
        "generalists": comparator.identify_generalists(all_results),
        "transfer_potential": comparator.compute_transfer_potential(all_results),
        "robustness_leaders": comparator.identify_robustness_leaders(all_results),
        "key_findings": {
            "best_generalist": comparator.identify_generalists(all_results)[0]
            if comparator.identify_generalists(all_results)
            else None,
            "most_robust": comparator.identify_robustness_leaders(all_results)[0]
            if comparator.identify_robustness_leaders(all_results)
            else None,
            "transfer_insights": "Arms with high transfer scores generalize well across domains",
        },
    }

    return report


def export_synthesis_report(
    report: dict[str, Any],
    output_file: Path,
) -> None:
    """Export synthesis report to JSON."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
