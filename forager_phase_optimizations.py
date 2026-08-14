"""Forager phase-specific optimizations and advanced variants.

Implements learner variants optimized for different Forager phases.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


class ForagerPhaseOptimizer:
    """Create phase-specific Forager agent optimizations."""

    @staticmethod
    def create_smoke_optimized_dqn(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """DQN optimized for smoke test (single task, fast learning)."""
        return {
            "name": "dqn_smoke_opt",
            "config": {
                "learning_rate": 0.01,  # Higher for fast convergence
                "epsilon_start": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "replay_buffer_size": 5000,
                "batch_size": 32,
                "target_update_freq": 1000,
                "discount_factor": 0.99,
            },
            "description": "DQN optimized for smoke: fast learning, high exploration"
        }

    @staticmethod
    def create_continual_optimized_dqn(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """DQN optimized for continual learning (multiple tasks, stability)."""
        return {
            "name": "dqn_continual_opt",
            "config": {
                "learning_rate": 0.005,  # Lower for stability
                "epsilon_start": 0.5,  # Start lower (less exploration needed)
                "epsilon_decay": 0.99,
                "epsilon_min": 0.05,
                "replay_buffer_size": 10000,
                "batch_size": 64,
                "target_update_freq": 2000,
                "discount_factor": 0.95,  # Lower discount (shorter horizons)
            },
            "description": "DQN optimized for continual: stability, memory, consistency"
        }

    @staticmethod
    def create_transfer_optimized_dqn(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """DQN optimized for transfer (new distribution, fast adaptation)."""
        return {
            "name": "dqn_transfer_opt",
            "config": {
                "learning_rate": 0.02,  # High for fast adaptation to new dist
                "epsilon_start": 1.0,  # High exploration for new tasks
                "epsilon_decay": 0.98,  # Fast decay (quick to exploitation)
                "epsilon_min": 0.05,
                "replay_buffer_size": 3000,  # Smaller buffer (forget old dist)
                "batch_size": 16,  # Smaller batches for responsiveness
                "target_update_freq": 500,  # Frequent updates
                "discount_factor": 0.99,
            },
            "description": "DQN optimized for transfer: fast adaptation to new distribution"
        }

    @staticmethod
    def create_smoke_optimized_a3c(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """A3C optimized for smoke test (policy gradient, exploration)."""
        return {
            "name": "a3c_smoke_opt",
            "config": {
                "actor_learning_rate": 0.005,
                "critic_learning_rate": 0.01,
                "entropy_coeff": 0.1,  # Higher entropy (encourage exploration)
                "value_loss_coeff": 0.5,
                "gamma": 0.99,
                "max_grad_norm": 0.5,
            },
            "description": "A3C optimized for smoke: exploration, policy learning"
        }

    @staticmethod
    def create_continual_optimized_a3c(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """A3C optimized for continual learning (stability, consistency)."""
        return {
            "name": "a3c_continual_opt",
            "config": {
                "actor_learning_rate": 0.001,  # Conservative
                "critic_learning_rate": 0.002,
                "entropy_coeff": 0.01,  # Low entropy (exploit learned policy)
                "value_loss_coeff": 1.0,  # Weight value estimation
                "gamma": 0.95,
                "max_grad_norm": 1.0,
            },
            "description": "A3C optimized for continual: stability, value consistency"
        }

    @staticmethod
    def create_transfer_optimized_a3c(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """A3C optimized for transfer learning (new distribution)."""
        return {
            "name": "a3c_transfer_opt",
            "config": {
                "actor_learning_rate": 0.01,  # High for new distribution
                "critic_learning_rate": 0.02,
                "entropy_coeff": 0.05,  # Moderate exploration
                "value_loss_coeff": 0.3,  # Less weight on old value estimates
                "gamma": 0.99,
                "max_grad_norm": 0.3,
            },
            "description": "A3C optimized for transfer: adapt to new distribution"
        }


class ForagerHybridVariants:
    """Create hybrid Forager agent variants."""

    @staticmethod
    def create_dqn_with_curiosity(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """DQN with intrinsic curiosity motivation."""
        return {
            "name": "dqn_curiosity",
            "config": {
                "base_learning_rate": 0.01,
                "curiosity_driven": True,
                "curiosity_weight": 0.5,
                "prediction_error_scale": 1.0,
            },
            "description": "DQN with curiosity bonus: explore novel states"
        }

    @staticmethod
    def create_a3c_with_entropy_regularization(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """A3C with dynamic entropy regularization."""
        return {
            "name": "a3c_entropy_reg",
            "config": {
                "actor_learning_rate": 0.005,
                "critic_learning_rate": 0.01,
                "entropy_coeff_init": 0.1,
                "entropy_coeff_decay": 0.998,  # Gradually reduce exploration
            },
            "description": "A3C with entropy decay: smooth exploration reduction"
        }

    @staticmethod
    def create_dqn_a3c_ensemble(
        action_dim: int = 4,
        state_dim: int = 16,
    ) -> dict[str, Any]:
        """Ensemble of DQN and A3C for robustness."""
        return {
            "name": "dqn_a3c_ensemble",
            "config": {
                "ensemble_size": 2,
                "dqn_weight": 0.5,
                "a3c_weight": 0.5,
                "action_selection": "average_q",
            },
            "description": "Ensemble of DQN + A3C: combine value and policy"
        }


def register_forager_phase_optimizations():
    """Register all Forager phase-specific variants."""
    optimizer = ForagerPhaseOptimizer()

    # Smoke-optimized variants
    smoke_dqn = optimizer.create_smoke_optimized_dqn()
    smoke_a3c = optimizer.create_smoke_optimized_a3c()

    # Continual-optimized variants
    continual_dqn = optimizer.create_continual_optimized_dqn()
    continual_a3c = optimizer.create_continual_optimized_a3c()

    # Transfer-optimized variants
    transfer_dqn = optimizer.create_transfer_optimized_dqn()
    transfer_a3c = optimizer.create_transfer_optimized_a3c()

    # Hybrid variants
    hybrid = ForagerHybridVariants()
    dqn_curiosity = hybrid.create_dqn_with_curiosity()
    a3c_entropy = hybrid.create_a3c_with_entropy_regularization()
    ensemble = hybrid.create_dqn_a3c_ensemble()

    variants = [
        smoke_dqn, smoke_a3c,
        continual_dqn, continual_a3c,
        transfer_dqn, transfer_a3c,
        dqn_curiosity, a3c_entropy, ensemble,
    ]

    print(f"[OK] Registered {len(variants)} Forager phase-optimized variants")
    return variants
