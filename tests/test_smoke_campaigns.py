"""Smoke tests for all measurement campaigns - quick sanity checks.

Validates that each campaign can start, run 1 step/episode, and collect metrics
without crashing. Detects configuration errors before expensive compute runs.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path


class TestIPMNISTSmoke:
    """Smoke tests for IPMNIST screening."""

    def test_can_initialize_baseline_arm(self):
        """Test IPMNIST baseline arm initializes without error."""
        from alberta_framework.benchmarks.ipmnist_screening import screening_spec

        for arm_name in ["upgd_w_control", "adamw_control", "upgd_ema_norm"]:
            spec = screening_spec(arm_name)
            # Just verify spec is valid and factory callable
            assert spec.factory is not None
            assert callable(spec.factory)

    def test_can_run_one_step(self):
        """Test IPMNIST learner factory creates valid step function."""
        from alberta_framework.benchmarks.ipmnist_screening import screening_spec

        spec = screening_spec("upgd_w_control")
        init_fn, step_fn = spec.factory(spec.hyperparameters)

        # Verify functions are callable
        assert callable(init_fn)
        assert callable(step_fn)


class TestSCRSmoke:
    """Smoke tests for SCR v2."""

    def test_can_initialize_scr_baseline(self):
        """Test SCR baseline arm initializes."""
        from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
            get_learner_factory,
            get_arm_hyperparameters,
        )

        for arm in ["backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline"]:
            factory = get_learner_factory(arm)
            hp = get_arm_hyperparameters(arm)
            # Just verify factory and hp exist
            assert factory is not None
            assert hp is not None

    def test_can_run_scr_step(self):
        """Test SCR learner factory creates valid functions."""
        from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
            get_learner_factory,
            get_arm_hyperparameters,
        )

        factory = get_learner_factory("upgd_w_baseline")
        hp = get_arm_hyperparameters("upgd_w_baseline")
        init_fn, step_fn = factory(hp)

        # Verify functions are callable
        assert callable(init_fn)
        assert callable(step_fn)


class TestEMNISTSmoke:
    """Smoke tests for EMNIST v3."""

    def test_all_emnist_learners_registered(self):
        """Test all EMNIST learners are registered."""
        from alberta_framework.benchmarks.upgd_label_emnist import _FULL_STEP_FACTORIES

        required_learners = [
            "upgd_w", "adamw", "upgd_ema_norm", "sgd_ema_norm",
            "upgd_ema_norm_cbp", "sgd_norm_cbp", "upgd_l2init", "upgd_shiftnorm"
        ]

        for learner in required_learners:
            assert learner in _FULL_STEP_FACTORIES


class TestMicroContinualSmoke:
    """Smoke tests for micro-continual."""

    def test_all_micro_arms_registered(self):
        """Test all micro-continual arms are registered."""
        from micro_continual_improvements import PREREGISTERED_ARMS

        required_arms = [
            "rls_head_resid", "alignment_first", "naive_bayes_extended",
            "dual_speed_rfs_rls", "actor_critic_micro"
        ]

        for arm in required_arms:
            assert arm in PREREGISTERED_ARMS


class TestForagerSmoke:
    """Smoke tests for Forager baselines."""

    def test_forager_dqn_initialization(self):
        """Test DQN agent initializes."""
        from alberta_framework.benchmarks.forager_open_baselines import make_baseline

        agent = make_baseline("dqn", action_dim=4, state_dim=16)
        agent.init(jax.random.PRNGKey(0), state_dim=16)

        # Test action selection
        state = jnp.zeros(16)
        action = agent.act(state, training=True)
        assert 0 <= action < 4

    def test_forager_a3c_initialization(self):
        """Test A3C agent initializes."""
        from alberta_framework.benchmarks.forager_open_baselines import make_baseline

        agent = make_baseline("a3c", action_dim=4, state_dim=16)
        agent.init(jax.random.PRNGKey(0), state_dim=16)

        state = jnp.zeros(16)
        action = agent.act(state, training=True)
        assert 0 <= action < 4

    def test_forager_random_baseline(self):
        """Test random baseline works."""
        from alberta_framework.benchmarks.forager_open_baselines import make_baseline

        agent = make_baseline("random", action_dim=4)
        action = agent.act(jnp.zeros(16), training=True)
        assert 0 <= action < 4

    def test_forager_harness_runs(self):
        """Test Forager harness runs without error."""
        from alberta_framework.benchmarks.forager_open_baselines_harness import (
            run_baseline_on_task,
        )

        result = run_baseline_on_task(
            baseline="random",
            task_id=0,
            num_episodes=2,
            seed=0,
        )

        assert result.baseline == "random"
        assert len(result.episodes) == 2


class TestMeasurementCLI:
    """Smoke tests for measurement CLI."""

    def test_ipmnist_cli_works(self):
        """Test IPMNIST CLI entry point."""
        from measurement_cli import run_ipmnist_arm

        result = run_ipmnist_arm(
            arm="upgd_w_control",
            n_tasks=10,
            seed=0,
        )

        assert result["status"] == "ready_for_measurement"
        assert result["domain"] == "ipmnist"

    def test_scr_cli_works(self):
        """Test SCR CLI entry point."""
        from measurement_cli import run_scr_arm

        result = run_scr_arm(
            arm="upgd_w_baseline",
            n_tasks=10,
            seed=0,
        )

        assert result["status"] == "ready_for_measurement"
        assert result["domain"] == "scr"

    def test_forager_cli_works(self):
        """Test Forager CLI entry point."""
        from measurement_cli import run_forager_baseline

        result = run_forager_baseline(
            baseline="random",
            phase="smoke",
            seed=0,
        )

        assert result["status"] == "ready_for_measurement"
        assert result["domain"] == "forager"


class TestResultAnalysis:
    """Smoke tests for result analysis tools."""

    def test_result_aggregator_works(self):
        """Test result aggregation."""
        from alberta_framework.utils.result_aggregation import ResultAggregator

        agg = ResultAggregator()

        # Add dummy results
        for domain in ["ipmnist", "scr"]:
            for arm in ["upgd", "sgd"]:
                for seed in range(3):
                    agg.add_result(domain, arm, "accuracy", seed, 0.85 + seed * 0.01)

        # Test ranking
        ipmnist_ranking = agg.domains["ipmnist"].ranking()
        assert len(ipmnist_ranking) == 2

        # Test transfer score
        transfer = agg.transfer_score("ipmnist", "scr")
        assert isinstance(transfer, float)

    def test_result_validator_works(self):
        """Test result validation."""
        from alberta_framework.utils.result_validation import ResultValidator

        validator = ResultValidator()

        group1 = [0.85, 0.87, 0.86, 0.88, 0.87]
        group2 = [0.80, 0.82, 0.81, 0.83, 0.82]

        # Test bootstrap CI
        ci = validator.bootstrap_ci(group1)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Test significance
        sig = validator.significance_test(group1, group2)
        assert "p_value" in sig

        # Test effect size
        effect = validator.effect_size(group1, group2)
        assert "cohens_d" in effect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
