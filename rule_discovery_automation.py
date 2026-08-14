"""Rule Discovery V2 automation - genome generation and search infrastructure.

Automated tools for expanding rule discovery search space systematically.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class GenomeGenerator:
    """Generate rule genomes programmatically for expanded search."""

    @staticmethod
    def generate_flag_combinations(
        n_flags: int = 50,
        sparsity: float = 0.3,
        seed: int = 0,
    ) -> list[np.ndarray]:
        """Generate flag combinations with controlled sparsity."""
        np.random.seed(seed)
        genomes = []

        for i in range(10):  # Generate 10 combinations
            genome = np.zeros(n_flags, dtype=np.float32)
            n_active = int(n_flags * sparsity)
            active_indices = np.random.choice(n_flags, n_active, replace=False)
            genome[active_indices] = np.random.uniform(0.5, 1.0, n_active)
            genomes.append(genome)

        return genomes

    @staticmethod
    def generate_parameter_sweeps(
        param_ranges: dict[str, tuple[float, float]],
        n_per_param: int = 3,
    ) -> list[dict[str, float]]:
        """Generate parameter sweep configurations."""
        configs = []

        for param_name, (min_val, max_val) in param_ranges.items():
            for value in np.linspace(min_val, max_val, n_per_param):
                config = {param_name: float(value)}
                configs.append(config)

        return configs

    @staticmethod
    def generate_mechanism_ablations(
        base_mechanism: dict[str, float],
        mechanisms_to_ablate: list[str],
    ) -> list[dict[str, float]]:
        """Generate ablation study configurations."""
        ablations = []

        # Full mechanism
        ablations.append(dict(base_mechanism))

        # Individual ablations (remove each mechanism)
        for mechanism in mechanisms_to_ablate:
            ablated = dict(base_mechanism)
            ablated[mechanism] = 0.0
            ablations.append(ablated)

        return ablations

    @staticmethod
    def generate_crossover_population(
        parent1: np.ndarray,
        parent2: np.ndarray,
        n_offspring: int = 5,
        mutation_rate: float = 0.1,
        seed: int = 0,
    ) -> list[np.ndarray]:
        """Generate offspring via crossover and mutation."""
        np.random.seed(seed)
        offspring = []

        for _ in range(n_offspring):
            # Crossover
            crossover_point = np.random.randint(len(parent1))
            child = np.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_indices = np.random.choice(len(child), size=2, replace=False)
                child[mutation_indices] = np.random.uniform(-1, 1, 2)

            offspring.append(child.astype(np.float32))

        return offspring


class RuleDiscoveryAutomation:
    """Automate rule discovery search phases."""

    @staticmethod
    def phase_1a_generate_candidates(
        n_candidates: int = 20,
        sparsity: float = 0.3,
    ) -> dict[str, Any]:
        """Phase 1a: Generate initial search candidates."""
        generator = GenomeGenerator()

        candidates = {
            "flag_based": generator.generate_flag_combinations(
                sparsity=sparsity,
                seed=0
            ),
            "parameter_sweeps": generator.generate_parameter_sweeps(
                {
                    "learning_rate": (0.001, 0.1),
                    "momentum": (0.0, 0.99),
                    "weight_decay": (0.0, 0.1),
                },
                n_per_param=3,
            ),
            "n_total": n_candidates,
        }

        return candidates

    @staticmethod
    def phase_1b_ablation_studies(
        champion_config: dict[str, float],
        mechanisms: list[str],
    ) -> dict[str, Any]:
        """Phase 1b: Generate ablation studies."""
        generator = GenomeGenerator()

        ablations = generator.generate_mechanism_ablations(
            champion_config,
            mechanisms,
        )

        return {
            "champion": champion_config,
            "ablations": ablations,
            "n_ablations": len(ablations),
            "mechanisms_tested": mechanisms,
        }

    @staticmethod
    def phase_1c_crossover_search(
        top_candidates: list[tuple[np.ndarray, float]],
        n_generations: int = 3,
    ) -> dict[str, Any]:
        """Phase 1c: Genetic algorithm crossover."""
        generator = GenomeGenerator()
        population = []

        # Initial population from top candidates
        for genome, score in top_candidates:
            population.append(genome)

        # Generate offspring across generations
        for gen in range(n_generations):
            offspring = []
            for i in range(0, len(population) - 1, 2):
                parent1, parent2 = population[i], population[i + 1]
                children = generator.generate_crossover_population(
                    parent1,
                    parent2,
                    n_offspring=3,
                    seed=gen * 100 + i,
                )
                offspring.extend(children)

            population.extend(offspring)

        return {
            "generations": n_generations,
            "initial_population": len(top_candidates),
            "final_population": len(population),
            "population_growth": len(population) / len(top_candidates),
        }

    @staticmethod
    def generate_search_curriculum(
        total_budget: int = 100,
    ) -> dict[str, Any]:
        """Generate structured search curriculum."""
        return {
            "phase_1a_screening": {
                "name": "Candidate generation",
                "budget": int(0.3 * total_budget),
                "task": "Generate diverse initial candidates",
            },
            "phase_1b_ablations": {
                "name": "Ablation studies",
                "budget": int(0.2 * total_budget),
                "task": "Test mechanism importance",
            },
            "phase_1c_crossover": {
                "name": "Genetic search",
                "budget": int(0.3 * total_budget),
                "task": "Evolve top candidates",
            },
            "phase_1d_refinement": {
                "name": "Fine-tuning",
                "budget": int(0.2 * total_budget),
                "task": "Local optimization of top performers",
            },
        }


def estimate_rule_discovery_compute(
    n_candidates: int = 50,
    hours_per_eval: float = 4.0,
) -> float:
    """Estimate compute hours for rule discovery campaign."""
    return n_candidates * hours_per_eval


def generate_rule_discovery_report(
    phase_results: dict[str, Any],
) -> dict[str, Any]:
    """Generate summary report of rule discovery progress."""
    return {
        "phases_completed": list(phase_results.keys()),
        "total_candidates_evaluated": sum(
            len(v.get("population", [])) for v in phase_results.values()
        ),
        "estimated_compute_hours": estimate_rule_discovery_compute(
            sum(len(v.get("population", [])) for v in phase_results.values())
        ),
        "next_phase": "validation_on_held_out",
    }
