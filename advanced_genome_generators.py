"""Rule Discovery V2 advanced genome generation strategies.

Implements sophisticated genome generation for expanded rule search.
"""

from typing import List, Dict, Any
import numpy as np


class AdvancedGenomeGenerator:
    """Generate rule genomes using advanced strategies."""

    @staticmethod
    def generate_from_genetic_algorithm(
        population_size: int = 50,
        generations: int = 5,
        mutation_rate: float = 0.1,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Generate genomes via genetic algorithm."""
        np.random.seed(seed)
        genomes = []

        # Initial population
        population = [np.random.uniform(-1, 1, 256) for _ in range(population_size)]

        for gen in range(generations):
            # Fitness: simple heuristic (sparse + moderate variance)
            fitness = []
            for genome in population:
                sparsity = np.sum(np.abs(genome) > 0.5) / len(genome)
                variance = np.var(genome)
                f = -abs(sparsity - 0.3) + variance  # Target 30% sparsity
                fitness.append(f)

            # Selection: top 50%
            top_indices = np.argsort(fitness)[-population_size // 2:]
            selected = [population[i] for i in top_indices]

            # Crossover + mutation
            new_population = selected.copy()
            for _ in range(population_size - len(selected)):
                parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
                crossover_point = np.random.randint(len(selected[0]))

                child = np.concatenate([
                    selected[parent1][:crossover_point],
                    selected[parent2][crossover_point:]
                ])

                if np.random.rand() < mutation_rate:
                    mut_points = np.random.choice(len(child), 5, replace=False)
                    child[mut_points] = np.random.uniform(-1, 1, 5)

                new_population.append(child)

            population = new_population[:population_size]
            genomes.extend(population)

        return genomes


    @staticmethod
    def generate_from_latin_hypercube(
        n_samples: int = 30,
        n_dims: int = 256,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Generate genomes via Latin Hypercube Sampling."""
        np.random.seed(seed)
        genomes = []

        # LHS: divide each dimension into n_samples bins
        for dim in range(n_dims):
            for i in range(n_samples):
                genome = np.random.uniform(-1, 1, n_dims)
                genome[dim] = -1 + (2 * (i + np.random.rand())) / n_samples
                genomes.append(genome)

        return genomes[:n_samples]


    @staticmethod
    def generate_from_sobol_sequence(
        n_samples: int = 32,
        n_dims: int = 256,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Generate quasi-random genomes via Sobol sequence."""
        np.random.seed(seed)
        genomes = []

        for i in range(1, n_samples + 1):
            # Simplified Sobol-like: bit-reversed integer
            g = i ^ (i >> 1)
            sobol_dim = np.array([(g >> k) & 1 for k in range(32)])

            genome = np.random.uniform(-1, 1, n_dims)
            # Modulate by Sobol bits
            genome = genome * (2 * sobol_dim[:min(len(sobol_dim), n_dims)] - 1)
            genomes.append(genome.astype(np.float32))

        return genomes


    @staticmethod
    def generate_from_mechanism_library(
        mechanisms: List[str],
        n_per_mechanism: int = 3,
    ) -> List[np.ndarray]:
        """Generate genomes from mechanism library combinations."""
        genomes = []

        mechanism_configs = {
            "rls_head": {"rls_head": 1.0, "rls_lambda": 0.99},
            "normalization": {"norm": 1.0, "norm_decay": 0.99},
            "gating": {"gate_signal": 1.0, "gate_beta": 0.5},
            "buffering": {"buffer_size": 1000},
            "meta_learning": {"meta_decay": 0.9},
        }

        for mechanism in mechanisms:
            config = mechanism_configs.get(mechanism, {})

            for i in range(n_per_mechanism):
                genome = np.zeros(256, dtype=np.float32)

                # Encode config in genome
                for j, (key, value) in enumerate(config.items()):
                    if j < len(genome):
                        genome[j] = value

                # Add noise
                genome += np.random.normal(0, 0.01, genome.shape)
                genomes.append(genome)

        return genomes


    @staticmethod
    def generate_interpolation_between_champions(
        champion1: np.ndarray,
        champion2: np.ndarray,
        n_interpolations: int = 5,
    ) -> List[np.ndarray]:
        """Generate intermediate genomes between two champions."""
        genomes = []

        for alpha in np.linspace(0, 1, n_interpolations + 2)[1:-1]:
            interpolated = (1 - alpha) * champion1 + alpha * champion2
            genomes.append(interpolated.astype(np.float32))

        return genomes


    @staticmethod
    def generate_perturbation_clouds(
        base_genome: np.ndarray,
        n_samples: int = 20,
        perturbation_scale: float = 0.1,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Generate cloud of perturbations around base genome."""
        np.random.seed(seed)
        genomes = [base_genome.copy()]

        for _ in range(n_samples):
            perturbation = np.random.normal(0, perturbation_scale, base_genome.shape)
            perturbed = base_genome + perturbation
            perturbed = np.clip(perturbed, -1, 1)
            genomes.append(perturbed.astype(np.float32))

        return genomes


def register_advanced_genome_generators() -> Dict[str, Any]:
    """Register all advanced genome generation strategies."""
    generator = AdvancedGenomeGenerator()

    return {
        "genetic_algorithm": generator.generate_from_genetic_algorithm,
        "latin_hypercube": generator.generate_from_latin_hypercube,
        "sobol_sequence": generator.generate_from_sobol_sequence,
        "mechanism_library": generator.generate_from_mechanism_library,
        "interpolation": generator.generate_interpolation_between_champions,
        "perturbation_clouds": generator.generate_perturbation_clouds,
    }


def generate_comprehensive_genome_population(
    total_size: int = 100,
    seed: int = 0,
) -> List[np.ndarray]:
    """Generate comprehensive genome population using all strategies."""
    np.random.seed(seed)
    genomes = []

    generator = AdvancedGenomeGenerator()

    # Genetic algorithm: 30 genomes
    genomes.extend(generator.generate_from_genetic_algorithm(30, seed=seed))

    # Latin hypercube: 20 genomes
    genomes.extend(generator.generate_from_latin_hypercube(20, seed=seed + 1))

    # Sobol sequence: 20 genomes
    genomes.extend(generator.generate_from_sobol_sequence(20, seed=seed + 2))

    # Mechanism library: 15 genomes
    mechanisms = ["rls_head", "normalization", "gating", "buffering", "meta_learning"]
    genomes.extend(generator.generate_from_mechanism_library(mechanisms, n_per_mechanism=3))

    # Perturbation clouds: 15 genomes
    base = np.random.uniform(-1, 1, 256)
    genomes.extend(generator.generate_perturbation_clouds(base, 15, seed=seed + 3))

    return genomes[:total_size]
