"""Rule Discovery V2 result analysis and filtering utilities.

Analyze and filter discovered rules for quality and interpretability.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class RuleQualityAnalyzer:
    """Analyze quality of discovered rules."""

    @staticmethod
    def compute_rule_complexity(genome: np.ndarray) -> float:
        """Compute complexity score for rule (lower = simpler)."""
        # Sparsity: how many non-zero elements
        sparsity = np.sum(np.abs(genome) > 0.1) / len(genome)

        # Variance: how diverse are the weights
        variance = np.var(genome)

        # Complexity = sparsity + variance normalization
        complexity = sparsity + variance / 10
        return float(complexity)

    @staticmethod
    def compute_rule_interpretability(genome: np.ndarray) -> float:
        """Compute interpretability score (1.0 = fully interpretable)."""
        # More sparse = more interpretable
        sparsity = np.sum(np.abs(genome) > 0.5) / len(genome)

        # Lower variance = more interpretable
        variance = np.var(genome)
        interpretability = (1.0 - sparsity) * (1.0 / (1.0 + variance))

        return float(np.clip(interpretability, 0, 1))

    @staticmethod
    def compute_rule_robustness(
        genome: np.ndarray,
        performance_values: List[float],
    ) -> float:
        """Compute robustness score based on performance variance."""
        if not performance_values:
            return 0.0

        mean_perf = np.mean(performance_values)
        std_perf = np.std(performance_values)

        # Lower variance = more robust
        robustness = 1.0 / (1.0 + std_perf / (mean_perf + 1e-8))
        return float(robustness)

    @staticmethod
    def rank_rules(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
        weights: Dict[str, float] = None,
    ) -> List[Tuple[str, float]]:
        """Rank rules by composite score."""
        if weights is None:
            weights = {
                "performance": 0.5,
                "interpretability": 0.3,
                "robustness": 0.2,
            }

        scores = {}

        for rule_name, genome in genomes.items():
            perf_values = performances.get(rule_name, [0.88])
            mean_perf = np.mean(perf_values) / 0.9  # Normalize to baseline
            interpretability = RuleQualityAnalyzer.compute_rule_interpretability(genome)
            robustness = RuleQualityAnalyzer.compute_rule_robustness(genome, perf_values)

            composite_score = (
                weights.get("performance", 0.5) * mean_perf +
                weights.get("interpretability", 0.3) * interpretability +
                weights.get("robustness", 0.2) * robustness
            )

            scores[rule_name] = float(composite_score)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


class RuleFilteringEngine:
    """Filter discovered rules by quality criteria."""

    @staticmethod
    def filter_by_performance(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
        threshold: float = 0.87,
    ) -> Dict[str, np.ndarray]:
        """Keep only rules above performance threshold."""
        filtered = {}

        for rule_name, genome in genomes.items():
            perf_values = performances.get(rule_name, [])
            if perf_values and np.mean(perf_values) >= threshold:
                filtered[rule_name] = genome

        return filtered

    @staticmethod
    def filter_by_complexity(
        genomes: Dict[str, np.ndarray],
        max_complexity: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Keep only interpretable (simple) rules."""
        filtered = {}

        for rule_name, genome in genomes.items():
            complexity = RuleQualityAnalyzer.compute_rule_complexity(genome)
            if complexity <= max_complexity:
                filtered[rule_name] = genome

        return filtered

    @staticmethod
    def filter_by_robustness(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
        max_cv: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Keep only robust rules (low variance across seeds)."""
        filtered = {}

        for rule_name, genome in genomes.items():
            perf_values = performances.get(rule_name, [])
            if perf_values:
                cv = np.std(perf_values) / (np.mean(perf_values) + 1e-8)
                if cv <= max_cv:
                    filtered[rule_name] = genome

        return filtered

    @staticmethod
    def filter_pareto_frontier(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
    ) -> Dict[str, np.ndarray]:
        """Keep only Pareto-optimal rules (performance vs complexity)."""
        scores = {}

        for rule_name, genome in genomes.items():
            perf = np.mean(performances.get(rule_name, [0.88]))
            complexity = RuleQualityAnalyzer.compute_rule_complexity(genome)
            scores[rule_name] = (perf, complexity)

        # Pareto filter: maximize performance, minimize complexity
        pareto = {}
        for rule1, (perf1, comp1) in scores.items():
            dominated = False
            for rule2, (perf2, comp2) in scores.items():
                if perf2 > perf1 and comp2 < comp1:
                    dominated = True
                    break
            if not dominated:
                pareto[rule1] = genomes[rule1]

        return pareto


class RuleEnsembleBuilder:
    """Build ensemble of complementary rules."""

    @staticmethod
    def build_performance_ensemble(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
        n_rules: int = 3,
    ) -> List[str]:
        """Select top-performing rules for ensemble."""
        ranked = RuleQualityAnalyzer.rank_rules(genomes, performances)
        return [rule_name for rule_name, _ in ranked[:n_rules]]

    @staticmethod
    def build_diversity_ensemble(
        genomes: Dict[str, np.ndarray],
        performances: Dict[str, List[float]],
        n_rules: int = 3,
    ) -> List[str]:
        """Select diverse rules for ensemble."""
        # Compute pairwise dissimilarity
        dissimilarity = {}
        rule_names = list(genomes.keys())

        for i, rule1 in enumerate(rule_names):
            for j, rule2 in enumerate(rule_names):
                if i < j:
                    dist = np.linalg.norm(genomes[rule1] - genomes[rule2])
                    dissimilarity[(rule1, rule2)] = dist

        # Greedy selection: pick diverse rules
        selected = [rule_names[0]]
        remaining = set(rule_names[1:])

        for _ in range(n_rules - 1):
            best_rule = None
            best_diversity = -1

            for rule in remaining:
                diversity = sum(dissimilarity.get((min(r, rule), max(r, rule)), 0)
                               for r in selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_rule = rule

            if best_rule:
                selected.append(best_rule)
                remaining.remove(best_rule)

        return selected[:n_rules]


def generate_rule_discovery_report(
    genomes: Dict[str, np.ndarray],
    performances: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Generate comprehensive rule discovery report."""
    analyzer = RuleQualityAnalyzer()
    filter_engine = RuleFilteringEngine()
    ensemble_builder = RuleEnsembleBuilder()

    ranked = analyzer.rank_rules(genomes, performances)

    report = {
        "total_rules": len(genomes),
        "top_10_rules": ranked[:10],
        "pareto_frontier": list(filter_engine.filter_pareto_frontier(genomes, performances).keys()),
        "performance_ensemble": ensemble_builder.build_performance_ensemble(genomes, performances),
        "diversity_ensemble": ensemble_builder.build_diversity_ensemble(genomes, performances),
        "quality_summary": {
            "avg_complexity": float(np.mean([analyzer.compute_rule_complexity(g) for g in genomes.values()])),
            "avg_interpretability": float(np.mean([analyzer.compute_rule_interpretability(g) for g in genomes.values()])),
            "best_rule": ranked[0][0] if ranked else None,
            "best_score": ranked[0][1] if ranked else 0,
        },
    }

    return report
