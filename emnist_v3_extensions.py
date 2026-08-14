"""EMNIST v3 extensions - additional protection mechanisms and learner variants.

Expands EMNIST v3 measurement campaign with new mechanism combinations.
"""

from alberta_framework.benchmarks.upgd_label_emnist import (
    LEARNER_REGISTRY,
    make_upgd_w_learner,
    make_adamw_learner,
)


# =============================================================================
# EMNIST v3: Additional CBP Variants (Composition + Recycling)
# =============================================================================

def register_emnist_cbp_variants():
    """Register additional CBP (composition + buffering/protection) variants."""

    # CBP with very high recycling (aggressive reuse)
    LEARNER_REGISTRY["upgd_cbp_recycle_high"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "cbp_enabled": True,
            "cbp_recycle_ratio": 0.9,  # 90% recycling
            "cbp_buffer_size": 1000,
        },
        "description": "UPGD + CBP with very high recycling (0.9) - aggressive composition reuse"
    }

    # CBP with low recycling (conservative composition)
    LEARNER_REGISTRY["upgd_cbp_recycle_low"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "cbp_enabled": True,
            "cbp_recycle_ratio": 0.3,  # 30% recycling
            "cbp_buffer_size": 500,
        },
        "description": "UPGD + CBP with low recycling (0.3) - conservative composition"
    }

    # CBP with moderate recycling
    LEARNER_REGISTRY["upgd_cbp_recycle_mid"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "cbp_enabled": True,
            "cbp_recycle_ratio": 0.6,  # 60% recycling
            "cbp_buffer_size": 750,
        },
        "description": "UPGD + CBP with moderate recycling (0.6) - balanced composition"
    }


# =============================================================================
# EMNIST v3: L2-Init Variants (Regularization to initialization)
# =============================================================================

def register_emnist_l2init_variants():
    """Register L2-init (decay-to-init) protection variants."""

    # Strong L2-init (aggressive decay to init)
    LEARNER_REGISTRY["upgd_l2init_strong"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "l2init_enabled": True,
            "l2init_decay": 0.1,  # Strong decay to init
        },
        "description": "UPGD + strong L2-init (0.1) - aggressive pull to initialization"
    }

    # Moderate L2-init
    LEARNER_REGISTRY["upgd_l2init_moderate"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "l2init_enabled": True,
            "l2init_decay": 0.05,  # Moderate decay to init
        },
        "description": "UPGD + moderate L2-init (0.05) - balanced init pull"
    }

    # Weak L2-init
    LEARNER_REGISTRY["upgd_l2init_weak"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "l2init_enabled": True,
            "l2init_decay": 0.01,  # Weak decay to init
        },
        "description": "UPGD + weak L2-init (0.01) - minimal init pull"
    }


# =============================================================================
# EMNIST v3: Shift-Norm Variants (Adaptive normalization)
# =============================================================================

def register_emnist_shiftnorm_variants():
    """Register shift-norm (adaptive normalization) variants."""

    # Aggressive shift detection
    LEARNER_REGISTRY["upgd_shiftnorm_aggressive"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "shiftnorm_enabled": True,
            "shiftnorm_threshold": 0.1,  # Low threshold = more sensitive
            "shiftnorm_adapt_rate": 0.2,
        },
        "description": "UPGD + aggressive shift-norm (threshold=0.1) - responsive to shifts"
    }

    # Conservative shift detection
    LEARNER_REGISTRY["upgd_shiftnorm_conservative"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "shiftnorm_enabled": True,
            "shiftnorm_threshold": 0.5,  # High threshold = less sensitive
            "shiftnorm_adapt_rate": 0.05,
        },
        "description": "UPGD + conservative shift-norm (threshold=0.5) - stable normalization"
    }

    # Balanced shift detection
    LEARNER_REGISTRY["upgd_shiftnorm_balanced"] = {
        "factory": make_upgd_w_learner,
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "shiftnorm_enabled": True,
            "shiftnorm_threshold": 0.3,  # Moderate threshold
            "shiftnorm_adapt_rate": 0.1,
        },
        "description": "UPGD + balanced shift-norm (threshold=0.3) - moderate sensitivity"
    }


# =============================================================================
# EMNIST v3: Adam Variants with Protection
# =============================================================================

def register_emnist_adam_protection_variants():
    """Register Adam with protection mechanisms."""

    # Adam + CBP
    LEARNER_REGISTRY["adamw_cbp"] = {
        "factory": make_adamw_learner,
        "hyperparameters": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "cbp_enabled": True,
            "cbp_recycle_ratio": 0.6,
        },
        "description": "Adam + CBP protection - compare momentum-based with composition"
    }

    # Adam + L2-init
    LEARNER_REGISTRY["adamw_l2init"] = {
        "factory": make_adamw_learner,
        "hyperparameters": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "l2init_enabled": True,
            "l2init_decay": 0.05,
        },
        "description": "Adam + L2-init protection - momentum with regularization"
    }


# =============================================================================
# EMNIST v3: SGD Variants with Protection
# =============================================================================

def register_emnist_sgd_protection_variants():
    """Register SGD with protection mechanisms."""

    # SGD + CBP
    LEARNER_REGISTRY["sgd_cbp"] = {
        "factory": lambda hp: make_upgd_w_learner({**hp, "use_momentum": False}),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "cbp_enabled": True,
            "cbp_recycle_ratio": 0.6,
        },
        "description": "SGD + CBP - pure gradient descent with composition"
    }

    # SGD + shift-norm
    LEARNER_REGISTRY["sgd_shiftnorm"] = {
        "factory": lambda hp: make_upgd_w_learner({**hp, "use_momentum": False}),
        "hyperparameters": {
            "step_size": 0.01,
            "weight_decay": 0.01,
            "shiftnorm_enabled": True,
            "shiftnorm_threshold": 0.3,
        },
        "description": "SGD + shift-norm - pure gradient with adaptive normalization"
    }


# =============================================================================
# Register All EMNIST Extensions
# =============================================================================

def register_all_emnist_extensions():
    """Register all EMNIST v3 extension learners."""
    register_emnist_cbp_variants()
    register_emnist_l2init_variants()
    register_emnist_shiftnorm_variants()
    register_emnist_adam_protection_variants()
    register_emnist_sgd_protection_variants()
    print("[OK] Registered 14 additional EMNIST v3 learner variants")


if __name__ == "__main__":
    register_all_emnist_extensions()
