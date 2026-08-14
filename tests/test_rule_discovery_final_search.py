"""Tests for Rule Discovery final search strategies.

Tests Bayesian optimization, hypervolume optimization, Thompson sampling,
and active learning curriculum implementations.
"""

import numpy as np
import pytest
import jax.random as jr
from jax import Array

from rule_discovery_final_search import (
    BayesianOptimizer,
    HypervolumOptimizer,
    ThompsonSamplerBandit,
    ActiveLearningCurriculum,
    FinalSearchStrategy,
    SearchConfig,
    GaussianProcessModel,
    ParetoPoint,
    BanditArm,
)


class TestBayesianOptimizer:
    """Tests for Bayesian optimization."""

    def test_rbf_kernel(self):
        """Test RBF kernel computation."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0), kernel="rbf")
        x1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        x2 = np.array([[0.0, 0.0]])

        kernel = opt.rbf_kernel(x1, x2, length_scale=1.0)
        assert kernel.shape == (2, 1)
        assert kernel[0, 0] > 0.9  # Same point
        assert kernel[1, 0] < kernel[0, 0]  # Different point

    def test_matern_kernel(self):
        """Test Matérn kernel computation."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0), kernel="matern")
        x1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        x2 = np.array([[0.0, 0.0]])

        kernel = opt.matern_kernel(x1, x2, length_scale=1.0)
        assert kernel.shape == (2, 1)
        assert kernel[0, 0] == 1.0  # Same point

    def test_fit_gp(self):
        """Test GP fitting."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0))

        # Create synthetic data
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        model = opt.fit_gp(X, y)
        assert model.X.shape == (10, 5)
        assert model.y.shape == (10,)

    def test_gp_prediction(self):
        """Test GP mean and variance predictions."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0))

        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model = opt.fit_gp(X, y)

        x_test = np.random.randn(5, 5)
        mean, std = opt.predict(model, x_test)

        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0)

    def test_expected_improvement(self):
        """Test EI acquisition function."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0))

        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model = opt.fit_gp(X, y)

        candidates = np.random.randn(20, 5)
        ei = opt.expected_improvement(model, candidates, y_best=0.0, xi=0.01)

        assert ei.shape == (20,)
        assert np.all(ei >= 0)

    def test_suggest_batch(self):
        """Test batch suggestion."""
        opt = BayesianOptimizer(search_space_dim=5, key=jr.key(0))

        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model = opt.fit_gp(X, y)

        candidates = np.random.randn(100, 5)
        batch = opt.suggest_batch(model, candidates, batch_size=10)

        assert len(batch) == 10
        assert len(set(batch)) == 10  # All unique
        assert all(0 <= idx < 100 for idx in batch)


class TestHypervolumOptimizer:
    """Tests for hypervolume multi-objective optimization."""

    def test_dominance(self):
        """Test Pareto dominance check."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        obj1 = np.array([0.8, 0.9])
        obj2 = np.array([0.7, 0.8])
        obj3 = np.array([0.8, 0.8])

        assert opt.dominate(obj1, obj2)
        assert not opt.dominate(obj2, obj1)
        assert not opt.dominate(obj1, obj3)

    def test_crowding_distance(self):
        """Test crowding distance computation."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        points = [
            ParetoPoint(genome=np.zeros(10), objectives=np.array([0.5, 0.5])),
            ParetoPoint(genome=np.zeros(10), objectives=np.array([0.6, 0.4])),
            ParetoPoint(genome=np.zeros(10), objectives=np.array([0.7, 0.6])),
        ]

        distances = opt.compute_crowding_distance(points)
        assert len(distances) == 3
        assert distances[0] == float('inf')  # Boundary
        assert distances[2] == float('inf')  # Boundary
        assert distances[1] > 0  # Interior

    def test_fast_non_dominated_sort(self):
        """Test NSGA-II sorting."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        genomes = [np.zeros(10), np.zeros(10), np.zeros(10)]
        objectives = [
            np.array([0.9, 0.9]),  # Dominates others
            np.array([0.5, 0.8]),  # Dominated by first
            np.array([0.8, 0.5]),  # Dominated by first
        ]

        fronts = opt.fast_non_dominated_sort(genomes, objectives)
        assert len(fronts) >= 1
        assert fronts[0] == [0]  # Best point in first front

    def test_update_front(self):
        """Test Pareto front update."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        genomes = [np.zeros(10), np.zeros(10)]
        objectives = [np.array([0.8, 0.7]), np.array([0.6, 0.9])]

        opt.update_front(genomes, objectives)
        assert len(opt.pareto_front) == 2

    def test_hypervolume_indicator(self):
        """Test hypervolume computation."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        genomes = [np.zeros(10), np.zeros(10)]
        objectives = [np.array([0.8, 0.7]), np.array([0.6, 0.9])]
        opt.update_front(genomes, objectives)

        hv = opt.hypervolume_indicator()
        assert hv > 0

    def test_select_batch(self):
        """Test batch selection from Pareto front."""
        opt = HypervolumOptimizer(reference_point=np.array([0.0, 0.0]))

        genomes = [np.random.randn(10) for _ in range(10)]
        objectives = [np.random.rand(2) for _ in range(10)]
        opt.update_front(genomes, objectives)

        batch = opt.select_batch(batch_size=3)
        assert len(batch) <= 3
        assert all(isinstance(p, ParetoPoint) for p in batch)


class TestThompsonSamplerBandit:
    """Tests for Thompson sampling bandit."""

    def test_arm_creation(self):
        """Test bandit arm initialization."""
        arms = ["arm1", "arm2", "arm3"]
        bandit = ThompsonSamplerBandit(arms, jr.key(0))

        assert len(bandit.arms) == 3
        for name in arms:
            assert name in bandit.arms
            assert bandit.arms[name].success_count == 0

    def test_sample_arm(self):
        """Test arm sampling."""
        arms = ["arm1", "arm2"]
        bandit = ThompsonSamplerBandit(arms, jr.key(0))

        sampled = bandit.sample_arm(jr.key(1))
        assert sampled in arms

    def test_update_arm(self):
        """Test arm update with feedback."""
        bandit = ThompsonSamplerBandit(["arm1"], jr.key(0))

        bandit.update_arm("arm1", success=True, reward=0.8)
        assert bandit.arms["arm1"].success_count == 1

        bandit.update_arm("arm1", success=False, reward=0.0)
        assert bandit.arms["arm1"].failure_count == 1

    def test_get_allocation(self):
        """Test arm allocation computation."""
        bandit = ThompsonSamplerBandit(["arm1", "arm2"], jr.key(0))

        # Uniform initially
        alloc = bandit.get_allocation()
        assert len(alloc) == 2
        assert sum(alloc.values()) == pytest.approx(1.0, abs=1e-6)

        # Update one arm
        bandit.update_arm("arm1", success=True, reward=0.8)
        alloc = bandit.get_allocation()
        assert alloc["arm1"] == 1.0  # Only one was tried

    def test_arm_history(self):
        """Test history logging."""
        bandit = ThompsonSamplerBandit(["arm1"], jr.key(0))

        bandit.update_arm("arm1", success=True, reward=0.8)
        bandit.update_arm("arm1", success=False, reward=0.2)

        assert len(bandit.history) == 2
        assert bandit.history[0]["reward"] == 0.8


class TestActiveLearningCurriculum:
    """Tests for active learning curriculum."""

    def test_curriculum_initialization(self):
        """Test curriculum initialization."""
        tasks = ["M1", "M2"]
        seeds = [0, 1]

        curriculum = ActiveLearningCurriculum(tasks, seeds)
        assert len(curriculum.informativeness) == 4  # 2 tasks x 2 seeds

    def test_update_informativeness(self):
        """Test informativeness score update."""
        curriculum = ActiveLearningCurriculum(["M1"], [0])

        accuracies = np.array([0.7, 0.8, 0.75, 0.85])
        curriculum.update_informativeness("M1", 0, accuracies)

        key = "M1_0"
        score = curriculum.informativeness[key]
        assert score.information_gain > 0
        assert score.difficulty > 0
        assert score.selected_count == 1

    def test_select_batch_easy_first(self):
        """Test batch selection with easy-first schedule."""
        curriculum = ActiveLearningCurriculum(["M1", "M2"], [0, 1])

        # Mark M1 as easy (high accuracy)
        curriculum.update_informativeness("M1", 0, np.array([0.9, 0.92]))
        curriculum.update_informativeness("M2", 0, np.array([0.5, 0.6]))

        batch = curriculum.select_batch(batch_size=2, phase=0.0)
        assert len(batch) <= 2

    def test_select_batch_hard_first(self):
        """Test batch selection with hard-first schedule."""
        curriculum = ActiveLearningCurriculum(["M1", "M2"], [0, 1])

        # Mark M1 as easy, M2 as hard
        curriculum.update_informativeness("M1", 0, np.array([0.9, 0.92]))
        curriculum.update_informativeness("M2", 0, np.array([0.5, 0.6]))

        batch = curriculum.select_batch(batch_size=2, phase=1.0)
        assert len(batch) <= 2

    def test_curriculum_schedule(self):
        """Test curriculum phase scheduling."""
        curriculum = ActiveLearningCurriculum(["M1"], [0])

        phase_0 = curriculum.get_curriculum_schedule(0, 100)
        phase_50 = curriculum.get_curriculum_schedule(50, 100)
        phase_100 = curriculum.get_curriculum_schedule(100, 100)

        assert phase_0 == 0.0
        assert 0.4 < phase_50 < 0.6
        assert phase_100 == 1.0


class TestFinalSearchStrategy:
    """Tests for integrated final search strategy."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = SearchConfig(
            use_bayesian=True,
            use_hypervolume=True,
            use_thompson=True,
            use_active_learning=True,
        )
        strategy = FinalSearchStrategy(config, jr.key(0))

        assert strategy.bayesian is not None
        assert strategy.hypervolume is not None
        assert strategy.thompson is not None
        assert strategy.curriculum is not None

    def test_compute_objectives(self):
        """Test multi-objective computation."""
        config = SearchConfig()
        strategy = FinalSearchStrategy(config, jr.key(0))

        genome = np.random.randn(34)
        objectives = strategy.compute_objectives(accuracy=0.85, genome=genome)

        assert len(objectives) == 3
        assert objectives[0] == 0.85  # Accuracy
        assert objectives[1] <= 0  # Negative complexity
        assert -1 <= objectives[2] <= 1  # Diversity (entropy)

    def test_select_next_batch(self):
        """Test batch selection integration."""
        config = SearchConfig(batch_size=32)
        strategy = FinalSearchStrategy(config, jr.key(0))

        # Dummy data
        evaluated_genomes = [np.random.randn(34) for _ in range(5)]
        evaluated_accuracies = [0.7, 0.75, 0.8, 0.72, 0.78]
        candidate_pool = np.random.randn(100, 34)

        batch = strategy.select_next_batch(
            evaluated_genomes, evaluated_accuracies, candidate_pool, generation=0
        )

        assert len(batch) <= config.batch_size
        assert all(0 <= idx < 100 for idx in batch)

    def test_log_step(self):
        """Test search progress logging."""
        config = SearchConfig()
        strategy = FinalSearchStrategy(config, jr.key(0))

        batch_genomes = [np.random.randn(34) for _ in range(10)]
        batch_accuracies = [0.7 + 0.1 * np.random.randn() for _ in range(10)]

        log = strategy.log_step(generation=0, batch_genomes=batch_genomes,
                                batch_accuracies=batch_accuracies)

        assert "generation" in log
        assert "best_accuracy" in log
        assert "mean_accuracy" in log
        assert log["generation"] == 0

    def test_search_history(self):
        """Test search history accumulation."""
        config = SearchConfig()
        strategy = FinalSearchStrategy(config, jr.key(0))

        for gen in range(3):
            batch_genomes = [np.random.randn(34) for _ in range(5)]
            batch_accuracies = [0.7 + 0.01 * gen for _ in range(5)]
            strategy.log_step(gen, batch_accuracies, batch_genomes)

        assert len(strategy.search_history) == 3

    def test_compute_arm_entropy(self):
        """Test Thompson arm entropy computation."""
        config = SearchConfig(use_thompson=True)
        strategy = FinalSearchStrategy(config, jr.key(0))

        entropy = strategy._compute_arm_entropy()
        assert entropy >= 0
        assert entropy <= np.log(len(config.mechanism_families))


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_bayesian_with_hypervolume(self):
        """Test Bayesian optimization with hypervolume."""
        config = SearchConfig(
            use_bayesian=True,
            use_hypervolume=True,
            use_thompson=False,
            use_active_learning=False,
        )
        strategy = FinalSearchStrategy(config, jr.key(0))

        # Simulate search loop
        evaluated_genomes = [np.random.randn(34) for _ in range(10)]
        evaluated_accuracies = np.random.rand(10) * 0.3 + 0.6

        for gen in range(2):
            batch_genomes = evaluated_genomes[-5:]
            batch_accuracies = list(evaluated_accuracies[-5:])

            for genome, acc in zip(batch_genomes, batch_accuracies):
                objectives = strategy.compute_objectives(acc, genome)
                strategy.hypervolume.update_front([genome], [objectives])

            hv = strategy.hypervolume.hypervolume_indicator()
            assert hv >= 0

    def test_thompson_bandit_convergence(self):
        """Test Thompson sampling convergence to best arm."""
        bandit = ThompsonSamplerBandit(
            ["good_arm", "bad_arm"], jr.key(0)
        )

        # Simulate trials
        for _ in range(100):
            arm = bandit.sample_arm(jr.key(0))
            if arm == "good_arm":
                bandit.update_arm(arm, success=True, reward=0.9)
            else:
                bandit.update_arm(arm, success=False, reward=0.3)

        # Good arm should have more pulls
        good_pulls = (
            bandit.arms["good_arm"].success_count +
            bandit.arms["good_arm"].failure_count
        )
        bad_pulls = (
            bandit.arms["bad_arm"].success_count +
            bandit.arms["bad_arm"].failure_count
        )
        assert good_pulls > bad_pulls

    def test_full_search_workflow(self):
        """Test complete search workflow."""
        config = SearchConfig(
            use_bayesian=True,
            use_hypervolume=True,
            use_thompson=True,
            use_active_learning=True,
            total_evaluations=500,
            batch_size=32,
            max_generations=5,
        )
        strategy = FinalSearchStrategy(config, jr.key(0))

        # Simulate search
        candidate_pool = np.random.randn(1000, 34)
        evaluated_genomes = []
        evaluated_accuracies = []

        for generation in range(config.max_generations):
            # Select batch
            batch_indices = strategy.select_next_batch(
                evaluated_genomes, evaluated_accuracies,
                candidate_pool, generation
            )

            # Simulate evaluation
            batch_genomes = candidate_pool[batch_indices]
            batch_accuracies = [
                0.7 + 0.1 * np.random.randn() for _ in batch_indices
            ]

            # Update strategy
            evaluated_genomes.extend(batch_genomes)
            evaluated_accuracies.extend(batch_accuracies)

            log = strategy.log_step(generation, batch_accuracies, batch_genomes)
            assert log["generation"] == generation

        assert len(strategy.search_history) == config.max_generations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
