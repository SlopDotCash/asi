"""Additional quick-win arm variants for IPMNIST sensitivity analysis.

These arms test additional hyperparameter values to expand the search space
without requiring complex new mechanisms.
"""

from alberta_framework.benchmarks.ipmnist_screening import ScreeningSpec, register_arm
from alberta_framework.benchmarks.upgd_label_emnist import make_upgd_w_learner
import jax.numpy as jnp


# =============================================================================
# Item 6.1: Extended Norm Decay Variants (3 arms)
# =============================================================================

def register_norm_decay_variants():
    """Register additional norm_decay sensitivity variants."""
    base_hp = {
        "step_size": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }

    # Very aggressive decay (fast adaptation)
    hp_decay_09 = dict(base_hp)
    hp_decay_09["norm_decay"] = 0.90
    register_arm(
        ScreeningSpec(
            name="upgd_w_norm_decay_09",
            factory=make_upgd_w_learner,
            hyperparameters=hp_decay_09,
            description="UPGD + very fast norm decay (0.90) - aggressive adaptation to shifts"
        )
    )

    # Moderate decay (middle ground)
    hp_decay_095 = dict(base_hp)
    hp_decay_095["norm_decay"] = 0.95
    register_arm(
        ScreeningSpec(
            name="upgd_w_norm_decay_095",
            factory=make_upgd_w_learner,
            hyperparameters=hp_decay_095,
            description="UPGD + moderate norm decay (0.95) - balanced adaptation"
        )
    )

    # Very slow decay (conservative)
    hp_decay_999 = dict(base_hp)
    hp_decay_999["norm_decay"] = 0.999
    register_arm(
        ScreeningSpec(
            name="upgd_w_norm_decay_999",
            factory=make_upgd_w_learner,
            hyperparameters=hp_decay_999,
            description="UPGD + very slow norm decay (0.999) - conservative stability"
        )
    )


# =============================================================================
# Item 6.2: Extended Step Size Variants (3 arms)
# =============================================================================

def register_step_size_variants():
    """Register additional step_size sensitivity variants."""
    base_hp = {
        "step_size": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }

    # Aggressive learning (2x baseline)
    hp_step_002 = dict(base_hp)
    hp_step_002["step_size"] = 0.02
    register_arm(
        ScreeningSpec(
            name="upgd_w_step_size_002",
            factory=make_upgd_w_learner,
            hyperparameters=hp_step_002,
            description="UPGD + aggressive step size (0.02) - faster convergence"
        )
    )

    # Conservative learning (0.5x baseline)
    hp_step_005 = dict(base_hp)
    hp_step_005["step_size"] = 0.005
    register_arm(
        ScreeningSpec(
            name="upgd_w_step_size_005",
            factory=make_upgd_w_learner,
            hyperparameters=hp_step_005,
            description="UPGD + conservative step size (0.005) - stable but slower"
        )
    )

    # Very aggressive learning (4x baseline)
    hp_step_004 = dict(base_hp)
    hp_step_004["step_size"] = 0.04
    register_arm(
        ScreeningSpec(
            name="upgd_w_step_size_004",
            factory=make_upgd_w_learner,
            hyperparameters=hp_step_004,
            description="UPGD + very aggressive step size (0.04) - test stability limits"
        )
    )


# =============================================================================
# Item 6.3: Extended Weight Decay Variants (3 arms)
# =============================================================================

def register_weight_decay_variants():
    """Register additional weight_decay sensitivity variants."""
    base_hp = {
        "step_size": 0.01,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }

    # Very high regularization (10x baseline)
    hp_wd_01 = dict(base_hp)
    hp_wd_01["weight_decay"] = 0.1
    register_arm(
        ScreeningSpec(
            name="upgd_w_weight_decay_01",
            factory=make_upgd_w_learner,
            hyperparameters=hp_wd_01,
            description="UPGD + high weight decay (0.1) - strong regularization"
        )
    )

    # Moderate regularization (0.5x baseline)
    hp_wd_005 = dict(base_hp)
    hp_wd_005["weight_decay"] = 0.005
    register_arm(
        ScreeningSpec(
            name="upgd_w_weight_decay_005",
            factory=make_upgd_w_learner,
            hyperparameters=hp_wd_005,
            description="UPGD + light weight decay (0.005) - less regularization"
        )
    )

    # Zero regularization (test without decay)
    hp_wd_0 = dict(base_hp)
    hp_wd_0["weight_decay"] = 0.0
    register_arm(
        ScreeningSpec(
            name="upgd_w_weight_decay_0",
            factory=make_upgd_w_learner,
            hyperparameters=hp_wd_0,
            description="UPGD + no weight decay (0.0) - baseline without regularization"
        )
    )


# =============================================================================
# Item 6.4: Combination Sensitivity (3 arms)
# =============================================================================

def register_combination_variants():
    """Register combination sensitivity tests."""

    # Aggressive on all fronts
    hp_aggressive = {
        "step_size": 0.02,
        "weight_decay": 0.005,
        "norm_decay": 0.95,
        "norm_epsilon": 1e-8,
    }
    register_arm(
        ScreeningSpec(
            name="upgd_w_aggressive_combo",
            factory=make_upgd_w_learner,
            hyperparameters=hp_aggressive,
            description="UPGD + aggressive combo (fast learning, light reg, moderate decay)"
        )
    )

    # Conservative on all fronts
    hp_conservative = {
        "step_size": 0.005,
        "weight_decay": 0.1,
        "norm_decay": 0.999,
        "norm_epsilon": 1e-8,
    }
    register_arm(
        ScreeningSpec(
            name="upgd_w_conservative_combo",
            factory=make_upgd_w_learner,
            hyperparameters=hp_conservative,
            description="UPGD + conservative combo (slow learning, high reg, slow decay)"
        )
    )

    # Balanced mix
    hp_balanced = {
        "step_size": 0.015,
        "weight_decay": 0.01,
        "norm_decay": 0.99,
        "norm_epsilon": 1e-8,
    }
    register_arm(
        ScreeningSpec(
            name="upgd_w_balanced_combo",
            factory=make_upgd_w_learner,
            hyperparameters=hp_balanced,
            description="UPGD + balanced combo (moderate learning, moderate reg, moderate decay)"
        )
    )


# =============================================================================
# Registration Entry Point
# =============================================================================

def register_all_quick_win_variants():
    """Register all quick-win arm variants."""
    register_norm_decay_variants()
    register_step_size_variants()
    register_weight_decay_variants()
    register_combination_variants()
    print("[OK] All 12 quick-win variants registered")


if __name__ == "__main__":
    register_all_quick_win_variants()
