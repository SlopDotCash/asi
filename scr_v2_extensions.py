"""Additional SCR v2 arms - quick-win variants for sensitivity analysis.

Extends SCR v2 with parametric sensitivity exploration.
"""

from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
    ARM_REGISTRY,
    make_learner_factory,
    make_hyperparameters,
)


# =============================================================================
# Extended Step Size Variants for SCR
# =============================================================================

def register_scr_step_size_variants():
    """Register SCR arms with different step sizes."""

    # Very aggressive (2x baseline)
    ARM_REGISTRY["upgd_w_scr_step_002"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.02,
            "weight_decay": 0.01,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with aggressive step (0.02) - fast convergence, stability test"
    }

    # Conservative (0.5x baseline)
    ARM_REGISTRY["upgd_w_scr_step_005"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.005,
            "weight_decay": 0.01,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with conservative step (0.005) - stability focus"
    }

    # Very aggressive (4x baseline)
    ARM_REGISTRY["upgd_w_scr_step_004"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.04,
            "weight_decay": 0.01,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with very aggressive step (0.04) - extreme learning rate test"
    }


# =============================================================================
# Extended Weight Decay Variants for SCR
# =============================================================================

def register_scr_weight_decay_variants():
    """Register SCR arms with different weight decay values."""

    # High regularization (10x baseline)
    ARM_REGISTRY["upgd_w_scr_wd_01"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.1,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with high weight decay (0.1) - strong regularization"
    }

    # Light regularization (0.5x baseline)
    ARM_REGISTRY["upgd_w_scr_wd_005"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.005,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with light weight decay (0.005) - minimal regularization"
    }

    # No regularization
    ARM_REGISTRY["upgd_w_scr_wd_0"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.0,
            "norm_decay": 0.99,
        },
        "description": "UPGD-W SCR with no weight decay (0.0) - pure SGD"
    }


# =============================================================================
# Extended Norm Decay Variants for SCR
# =============================================================================

def register_scr_norm_decay_variants():
    """Register SCR arms with different normalizer decay values."""

    # Very fast adaptation (0.90)
    ARM_REGISTRY["upgd_w_scr_norm_09"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "norm_decay": 0.90,
        },
        "description": "UPGD-W SCR with fast norm decay (0.90) - aggressive shift adaptation"
    }

    # Moderate adaptation (0.95)
    ARM_REGISTRY["upgd_w_scr_norm_095"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "norm_decay": 0.95,
        },
        "description": "UPGD-W SCR with moderate norm decay (0.95) - balanced adaptation"
    }

    # Very slow adaptation (0.999)
    ARM_REGISTRY["upgd_w_scr_norm_999"] = {
        "factory": make_learner_factory("upgd_w"),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "norm_decay": 0.999,
        },
        "description": "UPGD-W SCR with slow norm decay (0.999) - conservative stability"
    }


# =============================================================================
# Register All SCR Extensions
# =============================================================================

def register_all_scr_extensions():
    """Register all SCR extension arms."""
    register_scr_step_size_variants()
    register_scr_weight_decay_variants()
    register_scr_norm_decay_variants()
    print("[OK] Registered 9 additional SCR v2 arms")


if __name__ == "__main__":
    register_all_scr_extensions()
