"""Final batch: Forager hybrid agent variants combining multiple strategies.

Implements advanced hybrid agents for Forager RL.
"""

from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp


def make_dqn_a3c_weighted_ensemble(hp: Dict[str, float]) -> Dict[str, Any]:
    """DQN + A3C with learned weighting."""
    return {
        "name": "dqn_a3c_weighted",
        "config": {
            "dqn_weight_init": hp.get("dqn_weight", 0.5),
            "a3c_weight_init": hp.get("a3c_weight", 0.5),
            "weight_adaptation": True,
            "dqn_lr": 0.01,
            "a3c_lr": 0.005,
            "weight_lr": 0.001,
        },
        "description": "Weighted ensemble with learned component weights"
    }


def make_curiosity_entropy_hybrid(hp: Dict[str, float]) -> Dict[str, Any]:
    """Curiosity-driven exploration + entropy regularization."""
    return {
        "name": "curiosity_entropy",
        "config": {
            "curiosity_weight": hp.get("curiosity", 0.5),
            "entropy_coeff_init": hp.get("entropy", 0.1),
            "entropy_decay": 0.998,
            "prediction_error_scale": 1.0,
        },
        "description": "Intrinsic curiosity + policy entropy for balanced exploration"
    }


def make_distributional_rls_hybrid(hp: Dict[str, float]) -> Dict[str, Any]:
    """Distributional RL + RLS head."""
    return {
        "name": "distributional_rls",
        "config": {
            "n_atoms": 51,
            "v_min": -10,
            "v_max": 10,
            "rls_lambda": 0.99,
            "rls_reset_threshold": 0.5,
        },
        "description": "Distributional value + RLS readout for stability"
    }


def make_dueling_advantage_hybrid(hp: Dict[str, float]) -> Dict[str, Any]:
    """Dueling architecture + advantage learning."""
    return {
        "name": "dueling_advantage",
        "config": {
            "value_stream_hidden": 128,
            "advantage_stream_hidden": 128,
            "advantage_learning_rate": 0.01,
            "dueling_aggregation": "mean",
        },
        "description": "Dueling value/advantage decomposition with advantage optimization"
    }


def make_multi_step_bootstrap_hybrid(hp: Dict[str, float]) -> Dict[str, Any]:
    """Multi-step returns + bootstrapped uncertainty."""
    return {
        "name": "multi_step_bootstrap",
        "config": {
            "n_steps": [1, 3, 5],
            "n_bootstrap_heads": 5,
            "uncertainty_coefficient": 0.5,
        },
        "description": "Multi-step returns with bootstrapped uncertainty estimates"
    }


def make_hindsight_relabeling_hybrid(hp: Dict[str, float]) -> Dict[str, Any]:
    """Hindsight experience replay + goal relabeling."""
    return {
        "name": "hindsight_relabeling",
        "config": {
            "hindsight_fraction": 0.8,
            "goal_space_dim": 8,
            "relabeling_strategy": "uniform",
        },
        "description": "HER with dynamic goal relabeling for better exploration"
    }


FORAGER_HYBRID_AGENTS = {
    "dqn_a3c_weighted": make_dqn_a3c_weighted_ensemble,
    "curiosity_entropy": make_curiosity_entropy_hybrid,
    "distributional_rls": make_distributional_rls_hybrid,
    "dueling_advantage": make_dueling_advantage_hybrid,
    "multi_step_bootstrap": make_multi_step_bootstrap_hybrid,
    "hindsight_relabeling": make_hindsight_relabeling_hybrid,
}


def register_forager_hybrid_agents():
    """Register all Forager hybrid agents."""
    print(f"[OK] Registered {len(FORAGER_HYBRID_AGENTS)} Forager hybrid agents")
    return FORAGER_HYBRID_AGENTS
