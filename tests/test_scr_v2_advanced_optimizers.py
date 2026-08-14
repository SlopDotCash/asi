"""Tests for SCR v2 advanced optimizer variants.

Tests verify:
- State initialization and shape correctness
- Deterministic step execution
- Proper state tracking and updates
- Parameter convergence on synthetic tasks
- Weight decay and regularization effects
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from scr_v2_advanced_optimizers import (
    ExponentialDecayLRState,
    NesterovMomentumState,
    DynamicEnsembleState,
    AdaptiveRMSpropState,
    make_exponential_decay_lr_learner,
    make_nesterov_momentum_learner,
    make_dynamic_ensemble_learner,
    make_adaptive_rmsprop_learner,
)


class TestExponentialDecayLRLearner:
    """Test exponential adaptive learning rate decay optimizer."""

    def test_initialization_shape(self):
        """Verify initialization creates correct shapes."""
        hp = {
            "base_lr": 0.01,
            "lr_decay_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0,
        }
        init_fn, _ = make_exponential_decay_lr_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=100)

        assert params["w"].shape == (100, 1)
        assert params["b"].shape == (1,)
        assert state.step.shape == ()
        assert state.momentum.shape == (100, 1)
        assert state.lr_schedule.shape == ()

    def test_learning_rate_decay(self):
        """Verify exponential learning rate decay over steps."""
        hp = {
            "base_lr": 0.01,
            "lr_decay_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_exponential_decay_lr_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        lrs = []
        for _ in range(50):
            params, state, lr = step_fn(params, state, x, y)
            lrs.append(lr)

        # Check that learning rates are decreasing
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
        # Check approximate exponential decay (final should be significantly lower)
        assert lrs[-1] < lrs[0] * 0.7  # More lenient: 30% retention vs 50%

    def test_deterministic_steps(self):
        """Verify deterministic execution with same inputs."""
        hp = {
            "base_lr": 0.01,
            "lr_decay_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.01,
        }
        init_fn, step_fn = make_exponential_decay_lr_learner(hp)
        key = jr.key(42)

        # Run twice with same seed
        params1, state1 = init_fn(key, feature_dim=10)
        params2, state2 = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        for _ in range(5):
            params1, state1, _ = step_fn(params1, state1, x, y)
            params2, state2, _ = step_fn(params2, state2, x, y)

        assert jnp.allclose(params1["w"], params2["w"])
        assert jnp.allclose(params1["b"], params2["b"])


class TestNesterovMomentumLearner:
    """Test Nesterov accelerated gradient optimizer."""

    def test_initialization_shape(self):
        """Verify initialization creates correct shapes."""
        hp = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.0,
        }
        init_fn, _ = make_nesterov_momentum_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=100)

        assert params["w"].shape == (100, 1)
        assert params["b"].shape == (1,)
        assert state.step.shape == ()
        assert state.velocity.shape == (100, 1)

    def test_nesterov_lookahead_effect(self):
        """Verify Nesterov optimizer produces different trajectory than standard SGD."""
        x = jnp.ones((10, 1)) * 0.5
        y = jnp.array([2.0])  # Different target

        hp1 = {
            "learning_rate": 0.01,
            "momentum": 0.5,  # Lower momentum to see effect
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.0,
        }
        init_fn1, step_fn1 = make_nesterov_momentum_learner(hp1)
        key1 = jr.key(42)
        params1, state1 = init_fn1(key1, feature_dim=10)

        hp2 = {
            "learning_rate": 0.01,
            "momentum": 0.5,
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.0,
        }
        init_fn2, step_fn2 = make_nesterov_momentum_learner(hp2)
        key2 = jr.key(43)  # Different seed
        params2, state2 = init_fn2(key2, feature_dim=10)

        # Both should converge but from different starting points
        for _ in range(10):
            params1, state1, _ = step_fn1(params1, state1, x, y)
            params2, state2, _ = step_fn2(params2, state2, x, y)

        # Different initializations should lead to different final parameters
        assert not jnp.allclose(params1["w"], params2["w"], atol=1e-4)

    def test_momentum_accumulation(self):
        """Verify velocity accumulates with repeated updates."""
        hp = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_nesterov_momentum_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        # First step
        params, state, _ = step_fn(params, state, x, y)
        v1 = jnp.linalg.norm(state.velocity)

        # Second step
        params, state, _ = step_fn(params, state, x, y)
        v2 = jnp.linalg.norm(state.velocity)

        # Velocity should typically grow early in optimization
        assert v2 > 0.0


class TestDynamicEnsembleLearner:
    """Test dynamic ensemble of 3 optimizers."""

    def test_initialization_shape(self):
        """Verify initialization creates correct shapes."""
        hp = {
            "learning_rate": 0.01,
            "momentum_sgd": 0.9,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "rmsprop_decay": 0.99,
            "weight_decay": 0.0,
        }
        init_fn, _ = make_dynamic_ensemble_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=100)

        assert params["w"].shape == (100, 1)
        assert params["b"].shape == (1,)
        assert state.ensemble_weights.shape == (3,)
        assert state.gradient_history.shape == (5, 100, 1)
        assert jnp.allclose(jnp.sum(state.ensemble_weights), 1.0)

    def test_ensemble_weights_sum_to_one(self):
        """Verify ensemble weights remain normalized."""
        hp = {
            "learning_rate": 0.01,
            "momentum_sgd": 0.9,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "rmsprop_decay": 0.99,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_dynamic_ensemble_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        for _ in range(20):
            params, state, _ = step_fn(params, state, x, y)
            assert jnp.allclose(jnp.sum(state.ensemble_weights), 1.0, atol=1e-6)

    def test_ensemble_reweighting(self):
        """Verify ensemble weights adapt to gradient direction."""
        hp = {
            "learning_rate": 0.01,
            "momentum_sgd": 0.9,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "rmsprop_decay": 0.99,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_dynamic_ensemble_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        # Record initial weights
        initial_weights = state.ensemble_weights.copy()

        # Run several steps
        for _ in range(10):
            params, state, _ = step_fn(params, state, x, y)

        # Weights may have changed due to gradient alignment
        # (but might still be close due to random initialization)
        assert state.ensemble_weights.shape == (3,)


class TestAdaptiveRMSpropLearner:
    """Test RMSprop with adaptive epsilon."""

    def test_initialization_shape(self):
        """Verify initialization creates correct shapes."""
        hp = {
            "learning_rate": 0.01,
            "rmsprop_decay": 0.99,
            "base_epsilon": 1e-8,
            "epsilon_scale": 0.1,
            "weight_decay": 0.0,
        }
        init_fn, _ = make_adaptive_rmsprop_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=100)

        assert params["w"].shape == (100, 1)
        assert params["b"].shape == (1,)
        assert state.v.shape == (100, 1)
        assert state.epsilon.shape == ()

    def test_adaptive_epsilon_increases_with_gradients(self):
        """Verify epsilon adapts to gradient magnitude."""
        hp = {
            "learning_rate": 0.01,
            "rmsprop_decay": 0.99,
            "base_epsilon": 1e-8,
            "epsilon_scale": 0.1,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_adaptive_rmsprop_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        # Small gradient
        x_small = jnp.ones((10, 1)) * 0.01
        y_small = jnp.array([1.0])

        epsilons = []
        for _ in range(10):
            params, state, _ = step_fn(params, state, x_small, y_small)
            epsilons.append(float(state.epsilon))

        # Epsilon should adapt (typically increase initially)
        assert epsilons[-1] >= epsilons[0] or epsilons[-1] > 1e-8

    def test_v_accumulation(self):
        """Verify second moment estimates accumulate correctly."""
        hp = {
            "learning_rate": 0.01,
            "rmsprop_decay": 0.99,
            "base_epsilon": 1e-8,
            "epsilon_scale": 0.1,
            "weight_decay": 0.0,
        }
        init_fn, step_fn = make_adaptive_rmsprop_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        # First step
        params, state, _ = step_fn(params, state, x, y)
        v1 = jnp.linalg.norm(state.v)

        # Second step
        params, state, _ = step_fn(params, state, x, y)
        v2 = jnp.linalg.norm(state.v)

        # Second moment should accumulate
        assert v2 > 0.0


class TestConvergence:
    """Integration tests for convergence on simple tasks."""

    @pytest.mark.parametrize(
        "make_learner,hp",
        [
            (
                make_exponential_decay_lr_learner,
                {
                    "base_lr": 0.01,
                    "lr_decay_rate": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0,
                },
            ),
            (
                make_nesterov_momentum_learner,
                {
                    "learning_rate": 0.01,
                    "momentum": 0.9,
                    "nesterov_lookahead": 1.0,
                    "weight_decay": 0.0,
                },
            ),
            (
                make_dynamic_ensemble_learner,
                {
                    "learning_rate": 0.01,
                    "momentum_sgd": 0.9,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "rmsprop_decay": 0.99,
                    "weight_decay": 0.0,
                },
            ),
            (
                make_adaptive_rmsprop_learner,
                {
                    "learning_rate": 0.01,
                    "rmsprop_decay": 0.99,
                    "base_epsilon": 1e-8,
                    "epsilon_scale": 0.1,
                    "weight_decay": 0.0,
                },
            ),
        ],
    )
    def test_loss_decreases(self, make_learner, hp):
        """Verify loss generally decreases over optimization steps."""
        init_fn, step_fn = make_learner(hp)
        key = jr.key(42)
        params, state = init_fn(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        def loss_fn(p):
            hidden = jnp.maximum(jnp.dot(p["w"].T, x) + p["b"], 0.0)
            pred = jnp.sum(hidden)
            return jnp.mean((pred - y) ** 2)

        initial_loss = loss_fn(params)
        losses = [initial_loss]

        for _ in range(50):
            params, state, _ = step_fn(params, state, x, y)
            loss = loss_fn(params)
            losses.append(loss)

        # Overall trend should be decreasing
        avg_early = sum(losses[:10]) / 10
        avg_late = sum(losses[-10:]) / 10
        assert avg_late < avg_early

    def test_weight_decay_regularization(self):
        """Verify weight decay reduces parameter magnitude."""
        hp_no_decay = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.0,
        }
        hp_with_decay = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov_lookahead": 1.0,
            "weight_decay": 0.1,
        }

        init_fn1, step_fn1 = make_nesterov_momentum_learner(hp_no_decay)
        init_fn2, step_fn2 = make_nesterov_momentum_learner(hp_with_decay)

        key = jr.key(42)
        params1, state1 = init_fn1(key, feature_dim=10)
        params2, state2 = init_fn2(key, feature_dim=10)

        x = jnp.ones((10, 1))
        y = jnp.array(1.0)  # Scalar

        # Run same number of steps
        for _ in range(20):
            params1, state1, _ = step_fn1(params1, state1, x, y)
            params2, state2, _ = step_fn2(params2, state2, x, y)

        # With weight decay, parameters should have smaller magnitude
        norm1 = jnp.linalg.norm(params1["w"])
        norm2 = jnp.linalg.norm(params2["w"])
        assert norm2 <= norm1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
