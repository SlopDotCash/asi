"""Rule Discovery final search: advanced optimization strategies.

Implements Bayesian optimization for genome search, hypervolume multi-objective
optimization, Thompson sampling with multi-armed bandits, and active learning
curriculum for systematic rule discovery.

Search strategies:
1. Bayesian Optimization: Gaussian process-based surrogate model with expected
   improvement acquisition function for efficient exploration-exploitation.
2. Hypervolume Optimization: Multi-objective optimization tracking Pareto front,
   maximizing hypervolume over multiple objectives (accuracy, complexity, diversity).
3. Thompson Sampling: Probabilistic arm bandit strategy for adaptive allocation
   of evaluation budget across genome families and mechanism combinations.
4. Active Learning Curriculum: Progressive difficulty curriculum that learns
   which tasks/seeds are most informative for discovery.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

logger = logging.getLogger(__name__)


# ============================================================================
# Bayesian Optimization for Genome Search
# ============================================================================

@chex.dataclass(frozen=True)
class GaussianProcessModel:
    """Lightweight GP surrogate model for Bayesian optimization."""

    X: Array  # (n_observations, n_features)
    y: Array  # (n_observations,) - observed fitnesses
    length_scale: float = 0.2
    noise_variance: float = 1e-4
    signal_variance: float = 1.0


class BayesianOptimizer:
    """Bayesian optimization using GP surrogate + expected improvement."""

    def __init__(
        self,
        search_space_dim: int,
        key: Array,
        kernel: str = "rbf",
    ):
        self.search_space_dim = search_space_dim
        self.key = key
        self.kernel_type = kernel
        self.observations: list[tuple[np.ndarray, float]] = []

    def rbf_kernel(self, x1: Array, x2: Array, length_scale: float) -> Array:
        """Radial basis function kernel."""
        distances = jnp.sqrt(
            jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2) + 1e-8
        )
        return jnp.exp(-(distances ** 2) / (2 * length_scale ** 2))

    def matern_kernel(self, x1: Array, x2: Array, length_scale: float, nu: float = 2.5) -> Array:
        """Matérn kernel (nu=2.5)."""
        distances = jnp.sqrt(
            jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2) + 1e-8
        )
        sqrt3 = jnp.sqrt(3.0)
        scaled_dist = sqrt3 * distances / length_scale
        kernel = (1.0 + scaled_dist) * jnp.exp(-scaled_dist)
        return kernel

    def fit_gp(self, X: Array, y: Array) -> GaussianProcessModel:
        """Fit GP to observations, return model."""
        n = X.shape[0]

        # Normalize y
        y_mean = jnp.mean(y)
        y_std = jnp.std(y) + 1e-8
        y_norm = (y - y_mean) / y_std

        # Kernel matrix with noise
        if self.kernel_type == "rbf":
            K = self.rbf_kernel(X, X, length_scale=0.2)
        else:
            K = self.matern_kernel(X, X, length_scale=0.2)

        K_inv = jnp.linalg.inv(K + 1e-4 * jnp.eye(n))

        return GaussianProcessModel(
            X=X,
            y=y_norm,
            length_scale=0.2,
            noise_variance=1e-4,
            signal_variance=float(y_std),
        )

    def predict(self, model: GaussianProcessModel, x_test: Array) -> tuple[Array, Array]:
        """Predict mean and variance at test points."""
        X = model.X
        y = model.y

        if self.kernel_type == "rbf":
            K_star = self.rbf_kernel(X, x_test, model.length_scale)
            K_ss = self.rbf_kernel(x_test, x_test, model.length_scale)
        else:
            K_star = self.matern_kernel(X, x_test, model.length_scale)
            K_ss = self.matern_kernel(x_test, x_test, model.length_scale)

        n = X.shape[0]
        K = self.rbf_kernel(X, X, model.length_scale) if self.kernel_type == "rbf" else self.matern_kernel(X, X, model.length_scale)
        K_inv = jnp.linalg.inv(K + model.noise_variance * jnp.eye(n))

        mean = K_star.T @ K_inv @ y
        var = jnp.diag(K_ss) - jnp.sum(K_star.T @ K_inv * K_star.T, axis=1)
        var = jnp.maximum(var, model.noise_variance)

        return mean, jnp.sqrt(var)

    def expected_improvement(
        self,
        model: GaussianProcessModel,
        x_candidates: Array,
        y_best: float,
        xi: float = 0.01,
    ) -> Array:
        """Expected improvement acquisition function."""
        mean, std = self.predict(model, x_candidates)

        Z = (mean - y_best - xi) / (std + 1e-8)
        ei = (mean - y_best - xi) * jax.scipy.stats.norm.cdf(Z) + std * jax.scipy.stats.norm.pdf(Z)
        ei = jnp.maximum(ei, 0.0)

        return ei

    def suggest_batch(
        self,
        model: GaussianProcessModel,
        candidate_pool: Array,
        batch_size: int = 32,
    ) -> list[int]:
        """Suggest batch of candidate indices via EI."""
        y_best = jnp.max(model.y)
        ei_scores = self.expected_improvement(model, candidate_pool, float(y_best))

        selected = []
        for _ in range(min(batch_size, candidate_pool.shape[0])):
            idx = int(jnp.argmax(ei_scores))
            selected.append(idx)
            ei_scores = ei_scores.at[idx].set(-jnp.inf)

        return selected


# ============================================================================
# Hypervolume Multi-Objective Optimization
# ============================================================================

@chex.dataclass(frozen=True)
class ParetoPoint:
    """Point on the Pareto front."""

    genome: np.ndarray
    objectives: np.ndarray  # (n_objectives,) - [accuracy, -complexity, diversity]
    crowding_distance: float = 0.0


class HypervolumOptimizer:
    """Multi-objective optimization via hypervolume indicator."""

    def __init__(self, reference_point: np.ndarray):
        """Reference point for hypervolume calculation (minimization side)."""
        self.reference_point = reference_point  # (n_objectives,)
        self.pareto_front: list[ParetoPoint] = []

    def dominate(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (all maximized)."""
        better = np.all(obj1 >= obj2)
        same = np.all(obj1 == obj2)
        return better and not same

    def compute_crowding_distance(self, front: list[ParetoPoint]) -> list[float]:
        """Compute crowding distance for diversity preservation."""
        if len(front) <= 2:
            return [float('inf')] * len(front)

        n_objectives = front[0].objectives.shape[0]
        distances = np.zeros(len(front))

        for m in range(n_objectives):
            objectives_m = np.array([p.objectives[m] for p in front])
            sorted_indices = np.argsort(objectives_m)

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Interior points get distance = normalized gap
            obj_min = objectives_m[sorted_indices[0]]
            obj_max = objectives_m[sorted_indices[-1]]
            obj_range = obj_max - obj_min + 1e-8

            for i in range(1, len(sorted_indices) - 1):
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                gap = (objectives_m[next_idx] - objectives_m[prev_idx]) / obj_range
                distances[curr_idx] += gap

        return list(distances)

    def fast_non_dominated_sort(
        self,
        genomes: list[np.ndarray],
        objectives: list[np.ndarray],
    ) -> list[list[int]]:
        """Fast non-dominated sorting (NSGA-II)."""
        n = len(genomes)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominate(objectives[i], objectives[j]):
                    dominated_by[i].append(j)
                elif self.dominate(objectives[j], objectives[i]):
                    domination_count[i] += 1

        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def update_front(
        self,
        new_genomes: list[np.ndarray],
        new_objectives: list[np.ndarray],
    ) -> None:
        """Update Pareto front with new points."""
        # Combine with existing front
        all_genomes = [p.genome for p in self.pareto_front] + new_genomes
        all_objectives = [p.objectives for p in self.pareto_front] + new_objectives

        # Non-dominated sort
        fronts = self.fast_non_dominated_sort(all_genomes, all_objectives)

        # Keep first front as approximate Pareto (can extend to multiple fronts)
        if fronts:
            front_indices = fronts[0]
            front_points = [
                ParetoPoint(genome=all_genomes[i], objectives=all_objectives[i])
                for i in front_indices
            ]

            # Add crowding distance
            distances = self.compute_crowding_distance(front_points)
            self.pareto_front = [
                dataclasses.replace(p, crowding_distance=d)
                for p, d in zip(front_points, distances)
            ]

    def hypervolume_indicator(self) -> float:
        """Estimate hypervolume of Pareto front (simplified WFG algorithm)."""
        if not self.pareto_front:
            return 0.0

        # 2D approximation: use first two objectives
        points = np.array([p.objectives[:2] for p in self.pareto_front])

        # Sort by first objective
        points = points[np.argsort(-points[:, 0])]

        hv = 0.0
        for i in range(len(points)):
            width = points[i, 0] - self.reference_point[0]
            height = points[i, 1] - self.reference_point[1]
            if i > 0:
                height -= points[i - 1, 1] - self.reference_point[1]
            hv += width * max(height, 0.0)

        return hv

    def select_batch(self, batch_size: int) -> list[ParetoPoint]:
        """Select most promising points from front for evaluation."""
        if len(self.pareto_front) <= batch_size:
            return self.pareto_front

        # Select by crowding distance and objective trade-off
        sorted_indices = np.argsort([
            -p.crowding_distance + 0.1 * np.mean(p.objectives)
            for p in self.pareto_front
        ])
        return [self.pareto_front[i] for i in sorted_indices[:batch_size]]


# ============================================================================
# Thompson Sampling with Multi-Armed Bandits
# ============================================================================

@chex.dataclass(frozen=True)
class BanditArm:
    """Bayesian arm state for Thompson sampling."""

    name: str
    success_count: int = 0
    failure_count: int = 0
    mean_reward: float = 0.5
    variance: float = 1.0


class ThompsonSamplerBandit:
    """Multi-armed bandit via Thompson sampling for budget allocation."""

    def __init__(self, arm_names: Sequence[str], key: Array):
        self.arms: dict[str, BanditArm] = {name: BanditArm(name=name) for name in arm_names}
        self.key = key
        self.history: list[dict[str, Any]] = []

    def sample_arm(self, key: Array) -> str:
        """Sample arm via Thompson sampling (Beta-Bernoulli model)."""
        thetas = {}
        for name, arm in self.arms.items():
            alpha = arm.success_count + 1.0
            beta = arm.failure_count + 1.0
            theta = np.random.beta(alpha, beta)
            thetas[name] = theta

        best_arm = max(thetas, key=thetas.get)
        return best_arm

    def update_arm(self, arm_name: str, success: bool, reward: float) -> None:
        """Update arm posterior with observation."""
        arm = self.arms[arm_name]

        # Update counts
        if success:
            new_arm = dataclasses.replace(
                arm,
                success_count=arm.success_count + 1,
                mean_reward=(
                    (arm.success_count * arm.mean_reward + reward) /
                    (arm.success_count + 1)
                ),
            )
        else:
            new_arm = dataclasses.replace(
                arm,
                failure_count=arm.failure_count + 1,
            )

        self.arms[arm_name] = new_arm

        self.history.append({
            "arm": arm_name,
            "success": success,
            "reward": reward,
            "success_count": new_arm.success_count,
            "failure_count": new_arm.failure_count,
        })

    def get_allocation(self) -> dict[str, float]:
        """Get probability allocation across arms."""
        total_pulls = sum(
            arm.success_count + arm.failure_count for arm in self.arms.values()
        )
        if total_pulls == 0:
            return {name: 1.0 / len(self.arms) for name in self.arms}

        allocation = {}
        for name, arm in self.arms.items():
            pulls = arm.success_count + arm.failure_count
            allocation[name] = float(pulls) / float(total_pulls)

        return allocation


# ============================================================================
# Active Learning Curriculum
# ============================================================================

@chex.dataclass(frozen=True)
class InformativenessScore:
    """Task/seed informativenss for active learning."""

    task_name: str
    seed: int
    information_gain: float  # Reduction in uncertainty
    difficulty: float  # Average loss across current candidates
    variance: float  # Variance across candidates
    selected_count: int = 0


class ActiveLearningCurriculum:
    """Progressive curriculum: learn which tasks/seeds most reduce uncertainty."""

    def __init__(self, task_names: Sequence[str], seed_pool: Sequence[int]):
        self.task_names = list(task_names)
        self.seed_pool = list(seed_pool)
        self.informativeness: dict[str, InformativenessScore] = {}

        # Initialize uniform
        for task in task_names:
            for seed in seed_pool:
                key = f"{task}_{seed}"
                self.informativeness[key] = InformativenessScore(
                    task_name=task,
                    seed=seed,
                    information_gain=1.0,
                    difficulty=0.5,
                    variance=0.25,
                )

    def update_informativeness(
        self,
        task_name: str,
        seed: int,
        accuracies: np.ndarray,
    ) -> None:
        """Update informativeness based on observed accuracies."""
        key = f"{task_name}_{seed}"

        # Information gain: variance of accuracies (high variance = informative)
        information_gain = float(np.var(accuracies))

        # Difficulty: average loss
        difficulty = float(1.0 - np.mean(accuracies))

        # Variance: normalized across range
        variance = float(np.std(accuracies))

        old_score = self.informativeness[key]
        self.informativeness[key] = InformativenessScore(
            task_name=task_name,
            seed=seed,
            information_gain=information_gain,
            difficulty=difficulty,
            variance=variance,
            selected_count=old_score.selected_count + 1,
        )

    def select_batch(self, batch_size: int, phase: float = 0.5) -> list[tuple[str, int]]:
        """Select most informative tasks/seeds.

        Args:
            batch_size: Number of (task, seed) pairs to select
            phase: Schedule parameter [0, 1]; 0=easy-first, 1=hard-first
        """
        scores = []
        for key, score in self.informativeness.items():
            # Blend difficulty and information gain
            # Early phase: prefer easier tasks (high accuracy variance)
            # Late phase: prefer harder, more informative tasks
            priority = (
                (1.0 - phase) * score.information_gain +
                phase * (score.difficulty + score.variance)
            )

            # Exploration bonus for undersampled tasks
            exploration_bonus = 1.0 / (1.0 + score.selected_count)

            combined_score = priority * exploration_bonus
            scores.append((key, combined_score))

        # Sort by score and select top
        scores.sort(key=lambda x: -x[1])
        selected_keys = [key for key, _ in scores[:batch_size]]

        return [
            (self.informativeness[key].task_name, self.informativeness[key].seed)
            for key in selected_keys
        ]

    def get_curriculum_schedule(self, step: int, total_steps: int) -> float:
        """Get phase parameter (0=easy, 1=hard) based on progress."""
        return float(step) / float(total_steps)


# ============================================================================
# Integrated Final Search Strategy
# ============================================================================

@chex.dataclass(frozen=True)
class SearchConfig:
    """Configuration for final search."""

    # Bayesian optimization
    use_bayesian: bool = True
    gp_kernel: str = "matern"
    ei_xi: float = 0.01

    # Hypervolume multi-objective
    use_hypervolume: bool = True
    reference_point: tuple[float, ...] = (-0.1, 1.0, -0.1)

    # Thompson sampling
    use_thompson: bool = True
    mechanism_families: tuple[str, ...] = (
        "baseline", "normalization", "gating", "surprise", "rls", "ensemble"
    )

    # Active learning
    use_active_learning: bool = True
    curriculum_mode: str = "difficulty"  # or "random"

    # Budget and convergence
    total_evaluations: int = 10000
    batch_size: int = 64
    max_generations: int = 100


class FinalSearchStrategy:
    """Integrated final search combining all strategies."""

    def __init__(
        self,
        config: SearchConfig,
        key: Array,
    ):
        self.config = config
        self.key = key

        # Initialize components
        self.bayesian = BayesianOptimizer(
            search_space_dim=34,  # Genome size
            key=key,
            kernel=config.gp_kernel,
        ) if config.use_bayesian else None

        self.hypervolume = HypervolumOptimizer(
            reference_point=np.array(config.reference_point),
        ) if config.use_hypervolume else None

        self.thompson = ThompsonSamplerBandit(
            arm_names=config.mechanism_families,
            key=key,
        ) if config.use_thompson else None

        self.curriculum = ActiveLearningCurriculum(
            task_names=("M1", "M2", "M3", "M4"),
            seed_pool=(0, 1, 2),
        ) if config.use_active_learning else None

        self.search_history: list[dict[str, Any]] = []

    def compute_objectives(
        self,
        accuracy: float,
        genome: np.ndarray,
        complexity: float | None = None,
    ) -> np.ndarray:
        """Compute multi-objective vector [accuracy, complexity, diversity]."""
        # Accuracy (maximize)
        obj_accuracy = accuracy

        # Complexity (minimize) - count active flags
        if complexity is None:
            complexity = float(np.sum(genome[:16] > 0.5)) / 16.0

        # Diversity (maximize) - entropy of genome
        active_fraction = float(np.sum(genome > 0.5)) / len(genome)
        diversity = -active_fraction * np.log(active_fraction + 1e-8) - (1 - active_fraction) * np.log(1 - active_fraction + 1e-8)

        return np.array([obj_accuracy, -complexity, diversity])

    def select_next_batch(
        self,
        evaluated_genomes: list[np.ndarray],
        evaluated_accuracies: list[float],
        candidate_pool: np.ndarray,
        generation: int,
    ) -> list[int]:
        """Select next batch using integrated strategy."""
        selected = []

        # 1. Thompson sampling: allocate budget to mechanism families
        if self.thompson:
            allocation = self.thompson.get_allocation()
            logger.info(f"Thompson allocation: {allocation}")

        # 2. Bayesian optimization: fit GP and get promising candidates
        if self.bayesian and len(evaluated_genomes) >= 5:
            X = np.stack(evaluated_genomes)
            y = np.array(evaluated_accuracies)

            gp_model = self.bayesian.fit_gp(jnp.asarray(X), jnp.asarray(y))
            ei_scores = self.bayesian.expected_improvement(
                gp_model,
                jnp.asarray(candidate_pool),
                float(np.max(y)),
                xi=self.config.ei_xi,
            )
            ei_indices = np.argsort(-np.asarray(ei_scores))
            selected.extend(ei_indices[:self.config.batch_size // 2])

        # 3. Hypervolume: select diverse Pareto-optimal solutions
        if self.hypervolume:
            hv_batch = self.hypervolume.select_batch(self.config.batch_size // 4)
            for point in hv_batch:
                # Find closest candidate in pool
                distances = np.linalg.norm(
                    candidate_pool - point.genome[None, :],
                    axis=1,
                )
                selected.append(int(np.argmin(distances)))

        # 4. Active learning curriculum: select informative tasks
        if self.curriculum:
            phase = min(
                float(generation) / float(self.config.max_generations),
                1.0,
            )
            curriculum_schedule = self.curriculum.get_curriculum_schedule(
                generation, self.config.max_generations
            )
            logger.info(f"Curriculum phase: {curriculum_schedule:.2f}")

        # Remove duplicates
        selected = list(set(selected))

        # Fill remaining budget with random exploration
        remaining = self.config.batch_size - len(selected)
        if remaining > 0:
            all_indices = set(range(candidate_pool.shape[0]))
            available = list(all_indices - set(selected))
            random_indices = np.random.choice(
                available,
                size=min(remaining, len(available)),
                replace=False,
            )
            selected.extend(random_indices)

        return selected[:self.config.batch_size]

    def log_step(
        self,
        generation: int,
        batch_accuracies: list[float],
        batch_genomes: list[np.ndarray],
    ) -> dict[str, Any]:
        """Log search progress."""
        best_accuracy = float(np.max(batch_accuracies))
        mean_accuracy = float(np.mean(batch_accuracies))

        # Compute multi-objectives
        objectives_list = [
            self.compute_objectives(acc, genome)
            for acc, genome in zip(batch_accuracies, batch_genomes)
        ]

        step_log = {
            "generation": generation,
            "batch_size": len(batch_accuracies),
            "best_accuracy": best_accuracy,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": float(np.std(batch_accuracies)),
            "hypervolume": float(self.hypervolume.hypervolume_indicator())
            if self.hypervolume else 0.0,
            "thompson_entropy": self._compute_arm_entropy()
            if self.thompson else 0.0,
        }

        self.search_history.append(step_log)
        return step_log

    def _compute_arm_entropy(self) -> float:
        """Compute entropy of Thompson arm distribution."""
        if not self.thompson:
            return 0.0

        allocation = self.thompson.get_allocation()
        probs = np.array(list(allocation.values()))
        probs = probs / (np.sum(probs) + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy)


# ============================================================================
# Summary and Export Functions
# ============================================================================

def create_search_summary(
    final_strategy: FinalSearchStrategy,
    final_results: dict[str, Any],
) -> dict[str, Any]:
    """Create comprehensive summary of search execution."""
    return {
        "schema": "rule_discovery.final_search.v1",
        "strategy": {
            "bayesian_optimization": final_strategy.config.use_bayesian,
            "hypervolume_optimization": final_strategy.config.use_hypervolume,
            "thompson_sampling": final_strategy.config.use_thompson,
            "active_learning": final_strategy.config.use_active_learning,
        },
        "search_history": final_strategy.search_history,
        "final_results": final_results,
        "total_evaluations": final_strategy.config.total_evaluations,
        "batch_size": final_strategy.config.batch_size,
    }


if __name__ == "__main__":
    # Example usage
    key = jr.key(0)
    config = SearchConfig(
        use_bayesian=True,
        use_hypervolume=True,
        use_thompson=True,
        use_active_learning=True,
        total_evaluations=10000,
    )

    strategy = FinalSearchStrategy(config, key)
    logger.info("Final search strategy initialized")
