"""Result validation and visualization utilities for measurement analysis.

Provides statistical validation, significance testing, and visualization tools
for comparing measurement results across domains and arms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class ResultValidator:
    """Validate measurement results for statistical significance."""

    @staticmethod
    def bootstrap_ci(
        values: list[float],
        ci: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        np.random.seed(seed)
        bootstraps = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, len(values), replace=True)
            bootstraps.append(np.mean(sample))

        alpha = (1 - ci) / 2
        return (
            float(np.percentile(bootstraps, alpha * 100)),
            float(np.percentile(bootstraps, (1 - alpha) * 100)),
        )

    @staticmethod
    def significance_test(
        group1: list[float],
        group2: list[float],
        test: str = "ttest",
    ) -> dict[str, Any]:
        """Test statistical significance between two groups."""
        if test == "ttest":
            t_stat, p_value = stats.ttest_ind(group1, group2)
        elif test == "mannwhitneyu":
            t_stat, p_value = stats.mannwhitneyu(group1, group2)
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "test": test,
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01,
            "mean_diff": float(np.mean(group1) - np.mean(group2)),
        }

    @staticmethod
    def effect_size(group1: list[float], group2: list[float]) -> dict[str, float]:
        """Compute effect sizes between groups."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)

        # Cohen's d
        pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Hedges' g (bias-corrected Cohen's d)
        n1, n2 = len(group1), len(group2)
        hedges_g = cohens_d * (1 - 3 / (4 * (n1 + n2) - 9))

        return {
            "cohens_d": float(cohens_d),
            "hedges_g": float(hedges_g),
            "magnitude": "large"
            if abs(cohens_d) > 0.8
            else "medium"
            if abs(cohens_d) > 0.5
            else "small",
        }


class ResultVisualizer:
    """Visualization utilities for measurement results."""

    @staticmethod
    def plot_arm_comparison(
        results: dict[str, list[float]],
        title: str = "Arm Comparison",
        output_path: Path = None,
    ) -> None:
        """Plot comparison of arms."""
        fig, ax = plt.subplots(figsize=(10, 6))

        arms = list(results.keys())
        means = [np.mean(results[arm]) for arm in arms]
        stds = [np.std(results[arm]) for arm in arms]

        ax.bar(arms, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_ylabel("Performance")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_transfer_curve(
        source_results: dict[str, list[float]],
        target_results: dict[str, list[float]],
        title: str = "Transfer Analysis",
        output_path: Path = None,
    ) -> None:
        """Plot transfer learning curve."""
        fig, ax = plt.subplots(figsize=(10, 6))

        common_arms = set(source_results.keys()) & set(target_results.keys())
        if not common_arms:
            print("No common arms to plot")
            return

        arms = sorted(common_arms)
        source_means = [np.mean(source_results[arm]) for arm in arms]
        target_means = [np.mean(target_results[arm]) for arm in arms]

        x = np.arange(len(arms))
        width = 0.35

        ax.bar(x - width / 2, source_means, width, label="Source", alpha=0.7)
        ax.bar(x + width / 2, target_means, width, label="Target", alpha=0.7)

        ax.set_xlabel("Arms")
        ax.set_ylabel("Performance")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_heatmap(
        data: dict[str, dict[str, float]],
        title: str = "Arm x Domain Heatmap",
        output_path: Path = None,
    ) -> None:
        """Plot heatmap of arms across domains."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract domains and arms
        domains = list(data.keys())
        arms = set()
        for domain_data in data.values():
            arms.update(domain_data.keys())
        arms = sorted(arms)

        # Build matrix
        matrix = np.zeros((len(arms), len(domains)))
        for j, domain in enumerate(domains):
            for i, arm in enumerate(arms):
                matrix[i, j] = data[domain].get(arm, 0)

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(np.arange(len(domains)))
        ax.set_yticks(np.arange(len(arms)))
        ax.set_xticklabels(domains)
        ax.set_yticklabels(arms)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_convergence(
        results: dict[str, list[float]],
        title: str = "Convergence Analysis",
        output_path: Path = None,
    ) -> None:
        """Plot convergence curves (if results have temporal structure)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for arm, values in results.items():
            if isinstance(values[0], (list, tuple)):
                # Temporal structure: plot mean over time
                mean_over_time = np.mean(values, axis=0)
                ax.plot(mean_over_time, label=arm, marker="o")
            else:
                # No temporal structure: plot cumulative mean
                cumulative_mean = np.cumsum(values) / np.arange(1, len(values) + 1)
                ax.plot(cumulative_mean, label=arm, marker="o")

        ax.set_xlabel("Step/Seed")
        ax.set_ylabel("Performance")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


def validate_and_summarize(
    results_file: Path,
    output_dir: Path = None,
) -> dict[str, Any]:
    """Comprehensive validation and summary of results."""
    with open(results_file) as f:
        results = json.load(f)

    output_dir = Path(output_dir) or Path("outputs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = ResultValidator()
    summary = {
        "input_file": str(results_file),
        "validations": {},
    }

    # Validate each arm
    if "measurements" in results:
        for measurement in results["measurements"]:
            arm = measurement.get("arm", "unknown")
            eps = measurement.get("episodes", [])

            if eps:
                returns = [e.get("return_", 0) for e in eps]
                ci_low, ci_high = validator.bootstrap_ci(returns)

                summary["validations"][arm] = {
                    "n_episodes": len(eps),
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "min_return": float(np.min(returns)),
                    "max_return": float(np.max(returns)),
                }

    # Save summary
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
