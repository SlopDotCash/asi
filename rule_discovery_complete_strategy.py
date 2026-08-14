"""Rule Discovery V2 final specialized genome variants.

Implements specialized genome initialization for specific discovery phases.
"""

from typing import List, Dict, Any
import numpy as np


class SpecializedGenomeInitializers:
    """Create specialized genomes for targeted discovery."""

    @staticmethod
    def create_minimalist_genomes(n_samples: int = 10) -> List[np.ndarray]:
        """Ultra-sparse genomes - test extreme simplicity."""
        genomes = []
        for i in range(n_samples):
            genome = np.zeros(256, dtype=np.float32)
            # Only 1-3 active elements
            n_active = np.random.randint(1, 4)
            active_idx = np.random.choice(256, n_active, replace=False)
            genome[active_idx] = np.random.uniform(0.5, 1.0, n_active)
            genomes.append(genome)
        return genomes

    @staticmethod
    def create_dense_exploration_genomes(n_samples: int = 10) -> List[np.ndarray]:
        """Dense genomes - test high-capacity models."""
        genomes = []
        for i in range(n_samples):
            # 50%+ active
            genome = np.random.uniform(-1, 1, 256).astype(np.float32)
            mask = np.random.rand(256) > 0.3
            genome = genome * mask
            genomes.append(genome)
        return genomes

    @staticmethod
    def create_mechanism_focused_genomes(
        mechanisms: List[str],
        n_per_mechanism: int = 5,
    ) -> List[np.ndarray]:
        """Genomes focused on specific mechanisms."""
        genomes = []
        mechanism_ranges = {
            "normalization": (0, 50),
            "gating": (50, 100),
            "buffering": (100, 150),
            "meta": (150, 200),
            "rls": (200, 256),
        }

        for mechanism in mechanisms:
            start, end = mechanism_ranges.get(mechanism, (0, 256))
            for _ in range(n_per_mechanism):
                genome = np.zeros(256, dtype=np.float32)
                active_size = end - start
                n_active = max(1, active_size // 4)
                active_idx = np.random.choice(
                    np.arange(start, end), n_active, replace=False
                )
                genome[active_idx] = np.random.uniform(0.5, 1.0, n_active)
                genomes.append(genome)

        return genomes

    @staticmethod
    def create_adversarial_genomes(n_samples: int = 10) -> List[np.ndarray]:
        """Genomes designed to challenge baselines."""
        genomes = []
        for i in range(n_samples):
            genome = np.zeros(256, dtype=np.float32)
            # Concentrate weight on few dimensions
            n_clusters = np.random.randint(2, 5)
            cluster_size = 256 // n_clusters

            for c in range(n_clusters):
                start = c * cluster_size
                end = (c + 1) * cluster_size
                cluster_center = np.random.randint(start, end)
                genome[cluster_center] = np.random.uniform(0.8, 1.0)

            genomes.append(genome)
        return genomes

    @staticmethod
    def create_transfer_friendly_genomes(
        source_champion: np.ndarray,
        n_samples: int = 10,
        perturbation: float = 0.2,
    ) -> List[np.ndarray]:
        """Genomes derived from champion for transfer."""
        genomes = []
        for i in range(n_samples):
            # Small perturbation around champion
            noise = np.random.normal(0, perturbation, source_champion.shape)
            genome = np.clip(source_champion + noise, -1, 1).astype(np.float32)
            genomes.append(genome)
        return genomes

    @staticmethod
    def create_diversity_maximizing_genomes(
        n_samples: int = 20,
        seed: int = 0,
    ) -> List[np.ndarray]:
        """Genomes maximizing pairwise diversity."""
        np.random.seed(seed)
        genomes = [np.random.uniform(-1, 1, 256).astype(np.float32)]

        for _ in range(n_samples - 1):
            best_candidate = None
            best_diversity = -1

            for attempt in range(10):
                candidate = np.random.uniform(-1, 1, 256).astype(np.float32)

                # Compute diversity: average distance to existing
                diversity = 0
                for existing in genomes:
                    dist = np.linalg.norm(candidate - existing)
                    diversity += dist

                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate is not None:
                genomes.append(best_candidate)

        return genomes


class PhaseSpecificGenomeStrategies:
    """Strategies for different discovery phases."""

    @staticmethod
    def get_phase_1a_strategy() -> Dict[str, Any]:
        """Phase 1a: Candidate generation strategy."""
        initializer = SpecializedGenomeInitializers()
        return {
            "phase": "1a",
            "name": "Candidate Generation",
            "strategies": [
                ("minimalist", initializer.create_minimalist_genomes),
                ("dense", initializer.create_dense_exploration_genomes),
                ("mechanism_focused",
                 lambda: initializer.create_mechanism_focused_genomes(
                     ["normalization", "gating", "buffering"]
                 )),
            ],
            "total_genomes": 30,
        }

    @staticmethod
    def get_phase_1b_strategy() -> Dict[str, Any]:
        """Phase 1b: Ablation study strategy."""
        return {
            "phase": "1b",
            "name": "Ablation Studies",
            "ablation_mechanisms": [
                "normalization", "gating", "buffering", "meta_decay", "rls_head"
            ],
            "genomes_per_ablation": 5,
            "total_genomes": 30,
        }

    @staticmethod
    def get_phase_1c_strategy() -> Dict[str, Any]:
        """Phase 1c: Genetic algorithm strategy."""
        initializer = SpecializedGenomeInitializers()
        return {
            "phase": "1c",
            "name": "Genetic Search",
            "initial_population": initializer.create_diversity_maximizing_genomes(20),
            "generations": 5,
            "mutation_rate": 0.15,
            "crossover_type": "uniform",
            "total_candidates": 50,
        }

    @staticmethod
    def get_phase_1d_strategy() -> Dict[str, Any]:
        """Phase 1d: Refinement strategy."""
        return {
            "phase": "1d",
            "name": "Fine-tuning",
            "refinement_window": 0.1,
            "n_top_to_refine": 10,
            "refinement_steps": 20,
            "total_candidates": 20,
        }


def create_complete_discovery_strategy() -> Dict[str, Any]:
    """Create complete Rule Discovery V2 strategy."""
    return {
        "discovery_name": "Rule Discovery V2 Complete",
        "phases": [
            PhaseSpecificGenomeStrategies.get_phase_1a_strategy(),
            PhaseSpecificGenomeStrategies.get_phase_1b_strategy(),
            PhaseSpecificGenomeStrategies.get_phase_1c_strategy(),
            PhaseSpecificGenomeStrategies.get_phase_1d_strategy(),
        ],
        "total_genomes": 130,
        "estimated_hours": 120,
    }
