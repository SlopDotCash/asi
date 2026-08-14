"""Learner factories for SCR v2 arms.

This module provides factory functions that instantiate learner init/step
functions for each preregistered arm. Factories consume hyperparameters from
the registry and return (init_fn, step_fn) pairs ready for shard execution.

Each factory is responsible for:
- Validating arm-specific hyperparameters
- Constructing learner state (including any additional state dataclasses)
- Returning deterministic (init, step) functions for the benchmark loop

Reference: SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION.md
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.benchmarks.slowly_changing_regression import (
    SCRLearnerParams,
    build_scr_learner,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_arms import (
    EMANormalizerState,
    RLSHeadState,
    ShiftDetectorState,
)

__all__ = [
    "LearnerInitFn",
    "LearnerStepFn",
    "LearnerStateTuple",
    "make_backprop_sgd_relu_learner",
    "make_adamw_baseline_learner",
    "make_upgd_w_baseline_learner",
    "make_upgd_ema_norm_learner",
    "make_sigma0_shiftnorm_learner",
    "make_rls_head_learner",
    "get_learner_factory",
]

# Type aliases for learner signatures
LearnerInitFn = Callable[[Array, int], tuple[dict[str, Array], Any]]
LearnerStepFn = Callable[
    [dict[str, Array], Any, Array, Array],
    tuple[dict[str, Array], Any, float],
]
LearnerStateTuple = tuple[dict[str, Array], Any]


def _base_params_from_hp(hp: Mapping[str, float]) -> SCRLearnerParams:
    """Extract core learner parameters from hyperparameter dict."""
    return SCRLearnerParams(
        hidden_units=int(hp["hidden_units"]),
        step_size=float(hp["step_size"]),
        cbp_replacement_rate=float(hp.get("cbp_replacement_rate", 0.0)),
        cbp_maturity_threshold=int(hp.get("cbp_maturity_threshold", 0)),
        cbp_decay_rate=float(hp.get("cbp_decay_rate", 0.0)),
        upgd_utility_decay=float(hp.get("upgd_utility_decay", 0.0)),
        upgd_sigma=float(hp.get("upgd_sigma", 0.0)),
        upgd_beta=float(hp.get("upgd_beta", 0.0)),
    )


def make_backprop_sgd_relu_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for backprop_sgd_relu: ordinary SGD with ReLU activation.

    Returns a learner that uses standard backpropagation with SGD updates.
    No normalization, no utility gating, no perturbation.

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    learner = build_scr_learner(kind="sgd", params=params)
    return learner.init, learner.update


def make_adamw_baseline_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for adamw_baseline: SGD baseline (AdamW proxy in SCR domain).

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    learner = build_scr_learner(kind="sgd", params=params)
    return learner.init, learner.update


def make_upgd_w_baseline_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for upgd_w_baseline: utility-gated SGD with noise and decay.

    Returns a learner using UPGD's protection mechanism: utility-gated updates
    with perturbation noise and decoupled weight decay.

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    base_learner = build_scr_learner(kind="upgd", params=params)
    return base_learner.init, base_learner.update


def make_upgd_ema_norm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for upgd_ema_norm: UPGD-W behind EMA input normalization.

    Returns a learner that tracks running mean/variance of inputs and normalizes
    to zero mean, unit variance before passing to the UPGD-W learner.

    Input normalization uses exponential moving average with:
    - decay: controls how quickly statistics adapt (0.999 = slow, 0.99 = faster)
    - epsilon: numerical stability for variance (typically 1e-8)

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    norm_decay = float(hp.get("norm_decay", 0.999))
    norm_epsilon = float(hp.get("norm_epsilon", 1e-8))

    base_learner = build_scr_learner(kind="upgd", params=params)
    base_init = base_learner.init
    base_update = base_learner.update

    def init_fn(key: Array, feature_dim: int) -> LearnerStateTuple:
        """Initialize learner params and EMA normalizer state."""
        params_dict, base_state = base_init(key, feature_dim)
        norm_state = EMANormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            variance=jnp.ones(feature_dim, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.float32),
        )
        combined_state = (base_state, norm_state)
        return params_dict, combined_state

    def step_fn(
        params: dict[str, Array],
        state: Any,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], Any, float]:
        """Step: normalize input, then apply base learner step."""
        base_state, norm_state = state
        feature_dim = x.shape[-1]

        # Update EMA statistics
        new_mean = (
            norm_decay * norm_state.mean
            + (1.0 - norm_decay) * jnp.mean(x, axis=0)
        )
        new_var = (
            norm_decay * norm_state.variance
            + (1.0 - norm_decay) * jnp.var(x, axis=0)
        )
        new_count = norm_state.count + 1.0
        bias_correction = 1.0 - jnp.power(norm_decay, new_count)

        # Normalize input
        corrected_mean = new_mean / bias_correction
        corrected_var = new_var / bias_correction
        x_normalized = (x - corrected_mean) / jnp.sqrt(corrected_var + norm_epsilon)

        # Apply base learner on normalized input
        params_new, base_state_new, loss = base_step(
            params, base_state, x_normalized, y
        )

        norm_state_new = EMANormalizerState(
            mean=new_mean,
            variance=new_var,
            count=new_count,
        )
        combined_state_new = (base_state_new, norm_state_new)

        return params_new, combined_state_new, loss

    return init_fn, step_fn


def make_sigma0_shiftnorm_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for sigma0_shiftnorm: shift-triggered re-conditioning.

    Returns a learner that extends upgd_ema_norm with shift detection.
    When a loss spike is detected (output shift), the normalizer state is reset
    to force re-adaptation to the new regime.

    Shift detection uses:
    - window: rolling buffer of loss values (default 100)
    - threshold: loss jump ratio that triggers reset (default 0.05 = 5%)

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    norm_decay = float(hp.get("norm_decay", 0.999))
    norm_epsilon = float(hp.get("norm_epsilon", 1e-8))
    window_size = int(hp.get("shift_detector_window", 100))
    threshold = float(hp.get("shift_detector_threshold", 0.05))

    base_learner = build_scr_learner(kind="upgd", params=params)
    base_init = base_learner.init
    base_update = base_learner.update

    def init_fn(key: Array, feature_dim: int) -> LearnerStateTuple:
        """Initialize learner params, normalizer state, and shift detector."""
        params_dict, base_state = base_init(key, feature_dim)
        norm_state = EMANormalizerState(
            mean=jnp.zeros(feature_dim, dtype=jnp.float32),
            variance=jnp.ones(feature_dim, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.float32),
        )
        shift_state = ShiftDetectorState(
            loss_window=jnp.zeros(window_size, dtype=jnp.float32),
            window_idx=jnp.array(0, dtype=jnp.int32),
            shift_detected=jnp.array(False, dtype=jnp.bool_),
            resets_count=jnp.array(0, dtype=jnp.int32),
        )
        combined_state = (base_state, norm_state, shift_state)
        return params_dict, combined_state

    def step_fn(
        params: dict[str, Array],
        state: Any,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], Any, float]:
        """Step: detect shift, possibly reset normalizer, then update."""
        base_state, norm_state, shift_state = state
        feature_dim = x.shape[-1]

        # Compute loss for shift detection
        _, _, loss_val = base_step(params, base_state, x, y)

        # Update loss window (FIFO)
        idx = shift_state.window_idx % window_size
        new_loss_window = shift_state.loss_window.at[idx].set(loss_val)
        new_window_idx = shift_state.window_idx + 1

        # Detect shift: large jump in loss
        mean_loss = jnp.mean(new_loss_window)
        shift_detected = jnp.abs(loss_val - mean_loss) > (threshold * mean_loss)

        # Reset normalizer if shift detected
        new_norm_state = jax.lax.cond(
            shift_detected,
            lambda _: EMANormalizerState(
                mean=jnp.zeros(feature_dim, dtype=jnp.float32),
                variance=jnp.ones(feature_dim, dtype=jnp.float32),
                count=jnp.array(0, dtype=jnp.float32),
            ),
            lambda _: norm_state,
            operand=None,
        )

        # Update EMA statistics
        updated_mean = (
            norm_decay * new_norm_state.mean
            + (1.0 - norm_decay) * jnp.mean(x, axis=0)
        )
        updated_var = (
            norm_decay * new_norm_state.variance
            + (1.0 - norm_decay) * jnp.var(x, axis=0)
        )
        updated_count = new_norm_state.count + 1.0
        bias_correction = 1.0 - jnp.power(norm_decay, updated_count)

        # Normalize input
        corrected_mean = updated_mean / bias_correction
        corrected_var = updated_var / bias_correction
        x_normalized = (x - corrected_mean) / jnp.sqrt(corrected_var + norm_epsilon)

        # Apply base learner
        params_new, base_state_new, _ = base_step(
            params, base_state, x_normalized, y
        )

        norm_state_new = EMANormalizerState(
            mean=updated_mean,
            variance=updated_var,
            count=updated_count,
        )
        new_resets_count = shift_state.resets_count + jnp.where(shift_detected, 1, 0)
        shift_state_new = ShiftDetectorState(
            loss_window=new_loss_window,
            window_idx=new_window_idx,
            shift_detected=shift_detected,
            resets_count=new_resets_count,
        )
        combined_state_new = (base_state_new, norm_state_new, shift_state_new)

        return params_new, combined_state_new, loss_val

    return init_fn, step_fn


def make_rls_head_learner(
    hp: Mapping[str, float],
) -> tuple[LearnerInitFn, LearnerStepFn]:
    """Factory for rls_head: RLS readout on final-layer features.

    Returns a learner that trains hidden layers with UPGD-W, then uses
    streaming recursive-least-squares to learn the output weights from
    hidden activations. The RLS readout adapts to non-stationary targets
    using exponential forgetting.

    RLS parameters:
    - forgetting_factor: lambda in (0, 1]; 0.999 = slow forgetting, 1.0 = no forgetting
    - initial_covariance: P_0 scaling factor (identity * scale)

    Args:
        hp: Hyperparameter dict (from ARM_REGISTRY).

    Returns:
        (init_fn, step_fn) pair for the benchmark loop.
    """
    params = _base_params_from_hp(hp)
    rls_forgetting = float(hp.get("rls_forgetting_factor", 0.999))
    rls_p0_scale = float(hp.get("rls_initial_covariance", 1.0))

    base_learner = build_scr_learner(kind="upgd", params=params)
    base_init = base_learner.init
    base_update = base_learner.update

    def init_fn(key: Array, feature_dim: int) -> LearnerStateTuple:
        """Initialize learner params, hidden training state, and RLS state."""
        params_dict, base_state = base_init(key, feature_dim)
        hidden_units = int(hp["hidden_units"])

        rls_state = RLSHeadState(
            covariance_inv=rls_p0_scale * jnp.eye(hidden_units, dtype=jnp.float32),
            cross_term=jnp.zeros(hidden_units, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        combined_state = (base_state, rls_state)
        return params_dict, combined_state

    def step_fn(
        params: dict[str, Array],
        state: Any,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], Any, float]:
        """Step: extract hidden features, update RLS readout."""
        base_state, rls_state = state

        # Forward pass to get hidden features
        z1 = x @ params["w1"] + params["b1"]
        a1 = jax.nn.relu(z1)
        z2 = a1 @ params["w2"] + params["b2"]
        h = jax.nn.relu(z2)  # Hidden layer activations

        # RLS update on the readout
        # y_pred = h @ w_out (with implicit bias=0 for simplicity)
        y_pred = h @ rls_state.cross_term / (
            jnp.max(jnp.diag(rls_state.covariance_inv)) + 1e-8
        )
        residual = y - y_pred

        # Update covariance and cross-term
        phi = h  # feature vector
        new_p_inv = (
            (1.0 / rls_forgetting)
            * rls_state.covariance_inv
            - (1.0 / rls_forgetting)
            * rls_state.covariance_inv
            @ jnp.outer(phi, phi)
            @ rls_state.covariance_inv
            / (rls_forgetting + phi @ rls_state.covariance_inv @ phi)
        )
        new_cross = rls_state.cross_term + residual * phi

        rls_state_new = RLSHeadState(
            covariance_inv=new_p_inv,
            cross_term=new_cross,
            step_count=rls_state.step_count + 1,
        )

        # Also update hidden layers via base learner (UPGD-W on hidden + output)
        params_new, base_state_new, loss_val = base_step(
            params, base_state, x, y
        )

        combined_state_new = (base_state_new, rls_state_new)

        return params_new, combined_state_new, loss_val

    return init_fn, step_fn


def get_learner_factory(
    arm_name: str,
) -> Callable[[Mapping[str, float]], tuple[LearnerInitFn, LearnerStepFn]]:
    """Retrieve the factory function for an arm.

    Args:
        arm_name: Key in the arm registry.

    Returns:
        A factory function that consumes hyperparameters and returns (init_fn, step_fn).

    Raises:
        KeyError: if arm_name is not recognized.
    """
    factories = {
        "backprop_sgd_relu": make_backprop_sgd_relu_learner,
        "adamw_baseline": make_adamw_baseline_learner,
        "upgd_w_baseline": make_upgd_w_baseline_learner,
        "upgd_ema_norm": make_upgd_ema_norm_learner,
        "sigma0_shiftnorm": make_sigma0_shiftnorm_learner,
        "rls_head": make_rls_head_learner,
    }
    if arm_name not in factories:
        raise KeyError(
            f"factory for arm {arm_name!r} not found. "
            f"Valid arms: {sorted(factories)}"
        )
    return factories[arm_name]
