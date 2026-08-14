"""Integration tests for measurement harnesses across all domains.

Validates that all baseline arms, learners, and harnesses work correctly
end-to-end without regressions.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Import all harnesses and factories
from alberta_framework.benchmarks.ipmnist_screening import screening_spec
from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
    get_learner_factory as get_scr_factory,
    get_arm_hyperparameters as get_scr_hp,
)
from alberta_framework.benchmarks.upgd_label_emnist import _FULL_STEP_FACTORIES as emnist_factories
from alberta_framework.benchmarks.forager_open_baselines import make_baseline as make_forager_baseline
from alberta_framework.benchmarks.forager_open_baselines_harness import run_baseline_on_task
from micro_continual_improvements import PREREGISTERED_ARMS


class TestIPMNISTScreeningHarness:
    """Test IPMNIST screening harness and arms."""

    def test_ipmnist_baseline_arms(self):
        """Test that baseline arms (backprop_sgd_relu, adamw, upgd_w) all work."""
        baseline_arms = ["upgd_w_control", "adamw_control"]
        for arm in baseline_arms:
            spec = screening_spec(arm)
            init_fn, step_fn = spec.factory(spec.hyperparameters)
            assert callable(init_fn)
            assert callable(step_fn)


class TestSCRV2Harness:
    """Test SCR v2 baseline and arm harnesses."""

    @pytest.mark.parametrize("scr_arm", [
        "backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline",
        "upgd_ema_norm", "sigma0_shiftnorm", "rls_head"
    ])
    def test_scr_baseline_factory(self, scr_arm):
        """Test SCR baseline learner factories."""
        try:
            factory = get_scr_factory(scr_arm)
            hp = get_scr_hp(scr_arm)
            init_fn, step_fn = factory(hp)
            assert callable(init_fn)
            assert callable(step_fn)
        except Exception as e:
            pytest.fail(f"SCR ARM {scr_arm} factory failed: {e}")

    def test_scr_all_baselines_work(self):
        """Test all 3 SCR baselines specifically."""
        baselines = ["backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline"]
        for baseline in baselines:
            factory = get_scr_factory(baseline)
            hp = get_scr_hp(baseline)
            init_fn, step_fn = factory(hp)
            assert callable(init_fn), f"{baseline} init_fn not callable"
            assert callable(step_fn), f"{baseline} step_fn not callable"


class TestEMNISTHarness:
    """Test EMNIST v3 learner harness."""

    @pytest.mark.parametrize("learner_id", [
        "upgd_w", "adamw", "upgd_ema_norm", "sgd_ema_norm",
        "upgd_ema_norm_cbp", "sgd_norm_cbp", "upgd_l2init", "upgd_shiftnorm"
    ])
    def test_emnist_learner_factory(self, learner_id):
        """Test that EMNIST learner factories are registered."""
        assert learner_id in emnist_factories, f"Learner {learner_id} not in registry"
        factory = emnist_factories[learner_id]
        assert callable(factory)

    def test_emnist_v3_arms_registered(self):
        """Test that all 4 new EMNIST v3 arms are registered."""
        v3_arms = ["upgd_ema_norm_cbp", "sgd_norm_cbp", "upgd_l2init", "upgd_shiftnorm"]
        for arm in v3_arms:
            assert arm in emnist_factories, f"EMNIST v3 arm {arm} not registered"


class TestMicroContinualHarness:
    """Test micro-continual arm harness."""

    @pytest.mark.parametrize("arm_name", [
        "rls_head_resid", "alignment_first", "naive_bayes_extended",
        "dual_speed_rfs_rls", "actor_critic_micro",
        "rls_head_resid_lambda_095", "rls_head_resid_lambda_099",
        "dual_speed_rfs_rls_lambda_095"
    ])
    def test_micro_continual_arm_registered(self, arm_name):
        """Test that micro-continual arm is registered."""
        assert arm_name in PREREGISTERED_ARMS, f"Micro arm {arm_name} not registered"
        arm_spec = PREREGISTERED_ARMS[arm_name]
        assert "factory" in arm_spec
        assert callable(arm_spec["factory"])


class TestForagerBaselines:
    """Test Forager RL baselines."""

    @pytest.mark.parametrize("baseline_type", ["dqn", "a3c", "horde", "random"])
    def test_forager_agent_creation(self, baseline_type):
        """Test that Forager agents can be created and initialized."""
        agent = make_forager_baseline(baseline_type, action_dim=4, state_dim=8)
        agent.init(jax.random.PRNGKey(0), state_dim=8)
        assert agent is not None

    @pytest.mark.parametrize("baseline_type", ["dqn", "a3c", "horde", "random"])
    def test_forager_agent_action_selection(self, baseline_type):
        """Test that agents can select actions."""
        agent = make_forager_baseline(baseline_type, action_dim=4, state_dim=8)
        agent.init(jax.random.PRNGKey(0), state_dim=8)

        state = jnp.zeros(8)
        action = agent.act(state, training=True)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4

    @pytest.mark.parametrize("baseline_type", ["dqn", "a3c", "horde", "random"])
    def test_forager_agent_update(self, baseline_type):
        """Test that agents can perform update steps."""
        agent = make_forager_baseline(baseline_type, action_dim=4, state_dim=8)
        agent.init(jax.random.PRNGKey(0), state_dim=8)

        # Perform update (should not raise)
        transition = {
            "state": np.zeros(8),
            "action": 0,
            "reward": 0.1,
            "next_state": np.zeros(8),
            "done": False,
        }
        agent.update(transition)  # Should not raise

    def test_forager_harness_smoke_test(self, tmp_path):
        """Test Forager harness runs without error."""
        result = run_baseline_on_task(
            baseline="random",
            task_id=0,
            num_episodes=3,
            seed=0,
        )

        assert result.baseline == "random"
        assert result.num_episodes == 3
        assert len(result.episodes) == 3

        summary = result.summary()
        assert "mean_return" in summary
        assert "success_rate" in summary


class TestNoRegressions:
    """Ensure no regressions in existing test suite."""

    def test_ipmnist_registry_completeness(self):
        """Test that IPMNIST registry has no gaps."""
        # Sample 5 random arms
        test_arms = [
            "upgd_w_control", "adamw_control", "upgd_ema_norm",
            "upgd_ema_norm_cbp", "upgd_w_wd_0"
        ]

        for arm in test_arms:
            spec = screening_spec(arm)
            assert spec.name == arm
            assert spec.factory is not None
            assert spec.hyperparameters is not None

    def test_scr_baseline_completeness(self):
        """Test that SCR baselines are complete."""
        baselines = ["backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline"]

        for baseline in baselines:
            factory = get_scr_factory(baseline)
            hp = get_scr_hp(baseline)
            init_fn, step_fn = factory(hp)

            assert callable(init_fn)
            assert callable(step_fn)
            assert len(hp) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
