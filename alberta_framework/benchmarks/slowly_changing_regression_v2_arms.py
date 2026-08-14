"""Arm registry, hyperparameter definitions, and state dataclasses for SCR v2.

This module provides the infrastructure for the slowly-changing regression v2
preregistration (SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION.md). It defines:

- Hyperparameter constants for all baseline and Alberta-local arms
- ARM_REGISTRY: the complete arm specification mapping
- State dataclasses for arms requiring additional tracked state

All arms run on the same deterministic task instance (seeded environment),
with identical task configuration across methods. The registry serves as the
immutable specification used by shard executors and merge pipelines.

Reference: SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION.md
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import chex
from jax import Array
from jaxtyping import Float

from alberta_framework.benchmarks.slowly_changing_regression import SCRLearnerParams

__all__ = [
    "SCR_V2_BASELINE_ARMS",
    "SCR_V2_ALBERTA_ARMS",
    "SCR_V2_ALL_ARMS",
    "ARM_REGISTRY",
    "get_arm_hyperparameters",
    "get_arm_description",
    "EMANormalizerState",
    "ShiftDetectorState",
]

# =============================================================================
# Baseline arms (publication reference + published controls)
# =============================================================================

#: Nature reference arm: ordinary backprop with ReLU, SGD lr=0.01
#: Kaiming-uniform initialization, true MSE gradients (factor of 2).
BACKPROP_SGD_RELU_HYPERPARAMETERS: dict[str, float] = {
    "hidden_units": 5,
    "step_size": 0.01,
    "cbp_replacement_rate": 0.0,
    "cbp_maturity_threshold": 0,
    "cbp_decay_rate": 0.0,
    "upgd_utility_decay": 0.0,
    "upgd_sigma": 0.0,
    "upgd_beta": 0.0,
}

#: AdamW control arm from published configuration.
#: Includes per-parameter moment estimates and weight decay.
ADAMW_BASELINE_HYPERPARAMETERS: dict[str, float] = {
    "hidden_units": 5,
    "step_size": 0.001,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.01,
    "cbp_replacement_rate": 0.0,
    "cbp_maturity_threshold": 0,
    "cbp_decay_rate": 0.0,
    "upgd_utility_decay": 0.0,
    "upgd_sigma": 0.0,
    "upgd_beta": 0.0,
}

#: UPGD-W control arm (published regression configuration).
#: Utility-gated SGD with perturbation noise and decoupled weight decay.
UPGD_W_BASELINE_HYPERPARAMETERS: dict[str, float] = {
    "hidden_units": 5,
    "step_size": 0.01,
    "weight_decay": 0.01,
    "upgd_utility_decay": 0.9999,
    "upgd_sigma": 0.01,
    "upgd_beta": 0.0,
    "cbp_replacement_rate": 0.0,
    "cbp_maturity_threshold": 0,
    "cbp_decay_rate": 0.0,
}

# =============================================================================
# Alberta-local arms (domain-transfer and mechanism-extension)
# =============================================================================

#: Input-statistics normalization + utility gate (from IPMNIST screening).
#: EMA mean/variance tracking with scale normalization before the learner.
#: Hypothesis: conditioning mechanism transfers across domains.
UPGD_EMA_NORM_HYPERPARAMETERS: dict[str, float] = {
    **UPGD_W_BASELINE_HYPERPARAMETERS,
    "norm_enabled": 1.0,
    "norm_decay": 0.999,
    "norm_epsilon": 1e-8,
}

#: Shift-triggered re-conditioning (IPMNIST champion, ported to regression).
#: Detects output shifts and resets normalizer state (sigma0_shiftnorm from IPMNIST).
#: Single-axis extension of upgd_ema_norm for shift detection.
SIGMA0_SHIFTNORM_HYPERPARAMETERS: dict[str, float] = {
    **UPGD_EMA_NORM_HYPERPARAMETERS,
    "shift_detector_enabled": 1.0,
    "shift_detector_window": 100,
    "shift_detector_threshold": 0.05,
    "shift_detector_reset": 1.0,
}

#: RLS readout on final-layer features (prototype for regression domain).
#: Replaces learned output layer with streaming recursive-least-squares.
#: Tests whether RLS solves a fundamental problem on regression tasks.
RLS_HEAD_HYPERPARAMETERS: dict[str, float] = {
    "hidden_units": 5,
    "step_size": 0.01,
    "weight_decay": 0.01,
    "upgd_utility_decay": 0.9999,
    "upgd_sigma": 0.01,
    "upgd_beta": 0.0,
    "cbp_replacement_rate": 0.0,
    "cbp_maturity_threshold": 0,
    "cbp_decay_rate": 0.0,
    "rls_enabled": 1.0,
    "rls_forgetting_factor": 0.999,
    "rls_initial_covariance": 1.0,
}

# =============================================================================
# Arm registry (immutable specification)
# =============================================================================

SCR_V2_BASELINE_ARMS = frozenset(("backprop_sgd_relu", "adamw_baseline", "upgd_w_baseline"))
SCR_V2_ALBERTA_ARMS = frozenset(
    ("upgd_ema_norm", "sigma0_shiftnorm", "rls_head")
)
SCR_V2_SENSITIVITY_ARMS = frozenset(
    ("upgd_ema_norm_d095", "upgd_ema_norm_d0999", "upgd_ema_norm_d09999")
)
SCR_V2_ALL_ARMS = SCR_V2_BASELINE_ARMS | SCR_V2_ALBERTA_ARMS | SCR_V2_SENSITIVITY_ARMS


@chex.dataclass(frozen=True)
class ArmSpecification:
    """Immutable specification for one preregistered arm.

    Attributes:
        name: Unique arm identifier (used in shard paths, merge specs).
        role: Arm role in the protocol ("baseline_publication", "baseline_control",
            "alberta_domain_transfer", "alberta_mechanism_extension").
        hyperparameters: Frozen dict of hyperparameter name->value pairs.
        description: Human-readable summary for the preregistration.
        reference: Citation or design rationale (IPMNIST transfer, paper figure, etc.).
    """

    name: str
    role: str
    hyperparameters: MappingProxyType[str, float]
    description: str
    reference: str


ARM_REGISTRY: MappingProxyType[str, ArmSpecification] = MappingProxyType(
    {
        "backprop_sgd_relu": ArmSpecification(
            name="backprop_sgd_relu",
            role="baseline_publication",
            hyperparameters=MappingProxyType(BACKPROP_SGD_RELU_HYPERPARAMETERS),
            description=(
                "Ordinary backprop with ReLU activation, SGD lr=0.01. "
                "Nature reference arm; Kaiming-uniform init, true MSE gradients."
            ),
            reference=(
                "Dohare et al. 2024, Nature Methods 'Loss of plasticity in deep continual learning'"
            ),
        ),
        "adamw_baseline": ArmSpecification(
            name="adamw_baseline",
            role="baseline_control",
            hyperparameters=MappingProxyType(ADAMW_BASELINE_HYPERPARAMETERS),
            description=(
                "AdamW optimizer with per-parameter adaptive learning rates, "
                "moment estimates (beta1=0.9, beta2=0.999), and decoupled weight decay (0.01)."
            ),
            reference=(
                "Loshchilov & Hutter 2019, 'Decoupled Weight Decay Regularization' (ICLR); "
                "control for plasticity loss under adaptive methods."
            ),
        ),
        "upgd_w_baseline": ArmSpecification(
            name="upgd_w_baseline",
            role="baseline_control",
            hyperparameters=MappingProxyType(UPGD_W_BASELINE_HYPERPARAMETERS),
            description=(
                "UPGD-W: utility-gated SGD with perturbation noise (sigma=0.01) and "
                "decoupled weight decay (0.01). Published regression configuration."
            ),
            reference=(
                "Mahmood et al. 2012, 'Online Gradient Boosting'; "
                "UPGD gate mechanism for plasticity protection."
            ),
        ),
        "upgd_ema_norm": ArmSpecification(
            name="upgd_ema_norm",
            role="alberta_domain_transfer",
            hyperparameters=MappingProxyType(UPGD_EMA_NORM_HYPERPARAMETERS),
            description=(
                "UPGD-W behind EMA input normalization (decay=0.999, epsilon=1e-8). "
                "Input-statistics tracking transferred from IPMNIST screening. "
                "Tests whether conditioning mechanism generalizes across non-stationarity types."
            ),
            reference=(
                "Alberta IPMNIST screening: upgd_ema_norm champion (0.8514 ± 0.0001, n=20). "
                "Hypothesis: input conditioning generalizes from permutation to output-shift domains."
            ),
        ),
        "sigma0_shiftnorm": ArmSpecification(
            name="sigma0_shiftnorm",
            role="alberta_mechanism_extension",
            hyperparameters=MappingProxyType(SIGMA0_SHIFTNORM_HYPERPARAMETERS),
            description=(
                "Shift-triggered re-conditioning: detects output target shifts and resets "
                "normalizer state (window=100 examples, threshold=0.05 loss jump). "
                "Port of IPMNIST sigma0_shiftnorm champion to regression domain."
            ),
            reference=(
                "Alberta IPMNIST screening: sigma0_shiftnorm_d099 champion (0.8645 ± 0.0001, n=20); "
                "paired gain over upgd_ema_norm +0.0065. "
                "Tests shift-detector generality on different non-stationarity type."
            ),
        ),
        "rls_head": ArmSpecification(
            name="rls_head",
            role="alberta_mechanism_extension",
            hyperparameters=MappingProxyType(RLS_HEAD_HYPERPARAMETERS),
            description=(
                "RLS (recursive-least-squares) streaming readout on final-layer features. "
                "Replaces learned output layer with forgetting-factor RLS (lambda=0.999). "
                "Tests whether RLS solves a fundamental learning problem on regression."
            ),
            reference=(
                "Alberta IPMNIST: rls_head_resid_l1_preset005 standing record (0.8711 ± 0.0001, n=20). "
                "Prototype: does RLS readout architecture transfer to regression-on-features?"
            ),
        ),
        # --- Wave 10: norm_decay sensitivity (2026-08-14 addition).
        # Hypothesis: Test whether IPMNIST's decay=0.999 transfers to SCR or if
        # faster (0.95) or slower (0.9999) decay improves performance on output-shift.
        "upgd_ema_norm_d095": ArmSpecification(
            name="upgd_ema_norm_d095",
            role="mechanism_sensitivity",
            hyperparameters=MappingProxyType({**UPGD_EMA_NORM_HYPERPARAMETERS, "norm_decay": 0.95}),
            description=(
                "upgd_ema_norm with faster EMA decay (0.95 vs 0.999). "
                "Tests if quicker normalization adaptation helps with output-shift."
            ),
            reference="SCR v2 domain-transfer sensitivity test (SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION).",
        ),
        "upgd_ema_norm_d0999": ArmSpecification(
            name="upgd_ema_norm_d0999",
            role="mechanism_sensitivity",
            hyperparameters=MappingProxyType({**UPGD_EMA_NORM_HYPERPARAMETERS, "norm_decay": 0.999}),
            description=(
                "upgd_ema_norm with very slow EMA decay (0.999 vs 0.999). "
                "Conservative baseline; replicates IPMNIST champion decay exactly."
            ),
            reference="SCR v2 domain-transfer sensitivity test (SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION).",
        ),
        "upgd_ema_norm_d09999": ArmSpecification(
            name="upgd_ema_norm_d09999",
            role="mechanism_sensitivity",
            hyperparameters=MappingProxyType({**UPGD_EMA_NORM_HYPERPARAMETERS, "norm_decay": 0.9999}),
            description=(
                "upgd_ema_norm with extremely slow EMA decay (0.9999 vs 0.999). "
                "Tests if even slower normalization helps prevent over-adaptation."
            ),
            reference="SCR v2 domain-transfer sensitivity test (SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION).",
        ),
    }
)


def get_arm_hyperparameters(arm_name: str) -> dict[str, float]:
    """Retrieve the frozen hyperparameter dict for an arm.

    Args:
        arm_name: Key in ARM_REGISTRY.

    Returns:
        A dict copy of the arm's hyperparameters (unfrozen for mutation if needed).

    Raises:
        KeyError: if arm_name is not in the registry.
    """
    if arm_name not in ARM_REGISTRY:
        raise KeyError(
            f"arm {arm_name!r} not in registry. "
            f"Valid arms: {sorted(ARM_REGISTRY)}"
        )
    return dict(ARM_REGISTRY[arm_name].hyperparameters)


def get_arm_description(arm_name: str) -> str:
    """Retrieve the human-readable description of an arm.

    Args:
        arm_name: Key in ARM_REGISTRY.

    Returns:
        The arm's description string.

    Raises:
        KeyError: if arm_name is not in the registry.
    """
    if arm_name not in ARM_REGISTRY:
        raise KeyError(f"arm {arm_name!r} not in registry")
    return ARM_REGISTRY[arm_name].description


# =============================================================================
# State dataclasses (for arms with additional tracked state)
# =============================================================================


@chex.dataclass(frozen=True)
class EMANormalizerState:
    """State for EMA input normalization (upgd_ema_norm, sigma0_shiftnorm).

    Tracks running mean and variance of input activations for zero-mean,
    unit-variance normalization before the learner.

    Attributes:
        mean: EMA mean estimate, shape (input_dim,).
        variance: EMA variance estimate, shape (input_dim,).
        count: Number of examples processed (for bias correction).
    """

    mean: Float[Array, " input_dim"]
    variance: Float[Array, " input_dim"]
    count: Array


@chex.dataclass(frozen=True)
class ShiftDetectorState:
    """State for shift-triggered re-conditioning (sigma0_shiftnorm).

    Detects abrupt changes in loss/output distribution and signals
    normalizer reset.

    Attributes:
        loss_window: Rolling loss history (FIFO buffer), shape (window_size,).
        window_idx: Current insertion index in the loss window.
        shift_detected: Boolean flag if a shift has triggered reset.
        resets_count: Total number of shift-triggered resets so far.
    """

    loss_window: Float[Array, " window_size"]
    window_idx: Array
    shift_detected: Array
    resets_count: Array


@chex.dataclass(frozen=True)
class RLSHeadState:
    """State for RLS readout layer (rls_head).

    Recursive-least-squares maintains the inverse covariance matrix and
    accumulated cross-terms to compute the optimal linear readout on
    final-layer features.

    Attributes:
        covariance_inv: Inverse covariance of feature outer products,
            shape (hidden_units, hidden_units).
        cross_term: Accumulated (feature * target) cross-terms,
            shape (hidden_units,).
        step_count: Number of examples processed.
    """

    covariance_inv: Float[Array, "hidden_units hidden_units"]
    cross_term: Float[Array, " hidden_units"]
    step_count: Array
