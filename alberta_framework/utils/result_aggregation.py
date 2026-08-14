"""Result aggregation utilities for multi-seed, multi-domain experiments.

Consolidates measurement results across seeds, domains, and arms for statistical
analysis, comparison, and publication-ready summaries.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import numpy as np
import json


@dataclasses.dataclass
class ArmResult:
    """Single-arm result summary across seeds."""

    arm_name: str
    domain: str
    metric: str
    results: list[float]  # Per-seed results
    seeds: list[int]

    def mean(self) -> float:
        """Mean performance across seeds."""
        return float(np.mean(self.results))

    def std(self) -> float:
        """Standard deviation across seeds."""
        return float(np.std(self.results))

    def sem(self) -> float:
        """Standard error of the mean."""
        return float(np.std(self.results) / np.sqrt(len(self.results)))

    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval."""
        mean = self.mean()
        sem = self.sem()
        return (mean - 1.96 * sem, mean + 1.96 * sem)

    def summary(self) -> dict[str, Any]:
        """Return summary statistics dict."""
        return {
            "arm": self.arm_name,
            "domain": self.domain,
            "metric": self.metric,
            "n_seeds": len(self.results),
            "mean": self.mean(),
            "std": self.std(),
            "sem": self.sem(),
            "ci_lower": self.ci_95()[0],
            "ci_upper": self.ci_95()[1],
            "min": float(np.min(self.results)),
            "max": float(np.max(self.results)),
            "results": self.results,
        }


@dataclasses.dataclass
class DomainResults:
    """Results for one domain across multiple arms."""

    domain: str
    arms: dict[str, ArmResult]  # arm_name -> ArmResult

    def ranking(self) -> list[tuple[str, float]]:
        """Rank arms by mean performance (descending)."""
        ranking = [(name, result.mean()) for name, result in self.arms.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def pairwise_comparison(
        self, arm1: str, arm2: str, use_sem: bool = False
    ) -> dict[str, Any]:
        """Compare two arms via t-test-style effect size."""
        if arm1 not in self.arms or arm2 not in self.arms:
            raise ValueError(f"Arm {arm1} or {arm2} not found")

        r1 = self.arms[arm1]
        r2 = self.arms[arm2]

        mean_diff = r1.mean() - r2.mean()
        std_pooled = np.sqrt((r1.std() ** 2 + r2.std() ** 2) / 2)
        cohens_d = mean_diff / std_pooled if std_pooled > 0 else 0.0

        sem_combined = np.sqrt(r1.sem() ** 2 + r2.sem() ** 2)

        return {
            "arm1": arm1,
            "arm2": arm2,
            "mean1": r1.mean(),
            "mean2": r2.mean(),
            "mean_diff": float(mean_diff),
            "cohens_d": float(cohens_d),
            "sem_combined": float(sem_combined),
            "likely_winner": arm1 if mean_diff > 0 else arm2,
            "effect_size": "large"
            if abs(cohens_d) > 0.8
            else "medium"
            if abs(cohens_d) > 0.5
            else "small",
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "domain": self.domain,
            "arms": {name: result.summary() for name, result in self.arms.items()},
            "ranking": self.ranking(),
        }


class ResultAggregator:
    """Aggregate and analyze results across multiple domains and seeds."""

    def __init__(self):
        self.domains: dict[str, DomainResults] = {}

    def add_result(
        self, domain: str, arm: str, metric: str, seed: int, value: float
    ) -> None:
        """Add a single result."""
        if domain not in self.domains:
            self.domains[domain] = DomainResults(domain=domain, arms={})

        if arm not in self.domains[domain].arms:
            self.domains[domain].arms[arm] = ArmResult(
                arm_name=arm,
                domain=domain,
                metric=metric,
                results=[],
                seeds=[],
            )

        result = self.domains[domain].arms[arm]
        result.results.append(value)
        result.seeds.append(seed)

    def add_results_from_file(self, json_path: Path) -> None:
        """Load results from JSON measurement file."""
        with open(json_path) as f:
            data = json.load(f)

        # Expect format: {domain: {arm: {seed: value}}}
        for domain, arms_dict in data.items():
            for arm, seed_dict in arms_dict.items():
                for seed, value in seed_dict.items():
                    self.add_result(domain, arm, "metric", int(seed), float(value))

    def cross_domain_comparison(self, arm: str) -> dict[str, float]:
        """How does arm perform across domains?"""
        results = {}
        for domain_name, domain_results in self.domains.items():
            if arm in domain_results.arms:
                results[domain_name] = domain_results.arms[arm].mean()
        return results

    def arm_ranking_across_domains(self) -> dict[str, list[tuple[str, float]]]:
        """Get ranking of all arms in each domain."""
        return {
            domain_name: domain_results.ranking()
            for domain_name, domain_results in self.domains.items()
        }

    def transfer_score(self, source_domain: str, target_domain: str) -> float:
        """Score transfer from source to target domain.

        Measures correlation of arm rankings between domains.
        """
        if source_domain not in self.domains or target_domain not in self.domains:
            return 0.0

        source_ranking = self.domains[source_domain].ranking()
        target_ranking = self.domains[target_domain].ranking()

        # Get arms in both
        source_arms = {name: rank for rank, (name, _) in enumerate(source_ranking)}
        target_arms = {name: rank for rank, (name, _) in enumerate(target_ranking)}

        common_arms = set(source_arms.keys()) & set(target_arms.keys())
        if len(common_arms) < 2:
            return 0.0

        # Spearman correlation
        source_ranks = [source_arms[arm] for arm in sorted(common_arms)]
        target_ranks = [target_arms[arm] for arm in sorted(common_arms)]

        correlation = np.corrcoef(source_ranks, target_ranks)[0, 1]
        return float(np.nan_to_num(correlation, nan=0.0))

    def robustness_score(self, arm: str) -> float:
        """How consistent is arm performance across domains?

        Measures coefficient of variation of mean performance.
        """
        perf_by_domain = self.cross_domain_comparison(arm)
        if not perf_by_domain:
            return 0.0

        perf_values = list(perf_by_domain.values())
        mean_perf = np.mean(perf_values)
        if mean_perf == 0:
            return 0.0

        cv = np.std(perf_values) / mean_perf
        return float(1.0 / (1.0 + cv))  # Higher is more robust

    def summary_report(self) -> dict[str, Any]:
        """Generate comprehensive summary report."""
        report = {
            "domains": {
                domain_name: domain_results.to_dict()
                for domain_name, domain_results in self.domains.items()
            },
            "transfer_scores": {},
            "arm_robustness": {},
        }

        # Transfer scores between domains
        domain_names = list(self.domains.keys())
        for i, source in enumerate(domain_names):
            for target in domain_names[i + 1 :]:
                key = f"{source}_to_{target}"
                report["transfer_scores"][key] = self.transfer_score(source, target)

        # Robustness for each arm
        all_arms = set()
        for domain_results in self.domains.values():
            all_arms.update(domain_results.arms.keys())

        for arm in sorted(all_arms):
            report["arm_robustness"][arm] = self.robustness_score(arm)

        return report

    def to_json(self, output_path: Path) -> None:
        """Save aggregated results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.summary_report(), f, indent=2)


def compare_arms_statistical(results1: list[float], results2: list[float]) -> dict[str, Any]:
    """Perform statistical comparison of two result sets."""
    mean1, mean2 = np.mean(results1), np.mean(results2)
    std1, std2 = np.std(results1), np.std(results2)

    # Welch's t-test (unequal variance)
    t_stat = (mean1 - mean2) / np.sqrt(std1 ** 2 / len(results1) + std2 ** 2 / len(results2))
    df = len(results1) + len(results2) - 2

    return {
        "mean1": float(mean1),
        "mean2": float(mean2),
        "std1": float(std1),
        "std2": float(std2),
        "t_statistic": float(t_stat),
        "df": df,
        "likely_better": 1 if t_stat > 0 else 2,
        "magnitude": "large" if abs(t_stat) > 2 else "medium" if abs(t_stat) > 1 else "small",
    }
