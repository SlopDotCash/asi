"""SCR v2 Advanced Optimizer Variants with state management.

Implements four advanced optimization mechanisms:
1. Exponential adaptive learning rate decay
2. Gradient acceleration with Nesterov momentum
3. Dynamic ensemble of 3 optimizers
4. RMSprop with adaptive epsilon

Each optimizer includes proper state initialization, tracking, and deterministic updates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

__all__ = [
    "ExponentialDecayLRState",
    "NesterovMomentumState",
    "DynamicEnsembleState",
    "AdaptiveRMSpropState",
    "make_exponential_decay_lr_learner",
    "make_nesterov_momentum_learner",
    "make_dynamic_ensemble_learner",
    "make_adaptive_rmsprop_learner",
]


# =============================================================================
# State Dataclasses for Advanced Optimizers
# =============================================================================


@chex.dataclass(frozen=False)
class ExponentialDecayLRState:
    """State for exponential adaptive learning rate decay optimizer.

    Tracks momentum and exponentially decaying learning rates based on
    gradient history. Allows for aggressive initial learning followed by
    smooth convergence.

    Attributes:
        step: Current optimization step counter.
        momentum: Exponential moving average of gradients.
        lr_schedule: Current learning rate value (updated per step).
        base_lr: Initial learning rate (immutable).
        decay_rate: Exponential decay rate per step.
    """
    step: Array  # scalar
    momentum: Array  # shape of parameters
    lr_schedule: Array  # scalar
    base_lr: Array  # scalar
    decay_rate: Array  # scalar


@chex.dataclass(frozen=False)
class NesterovMomentumState:
    """State for Nesterov accelerated gradient optimizer.

    Implements gradient acceleration with Nesterov momentum, which looks ahead
    in the gradient direction before applying the update. Provides faster
    convergence than vanilla momentum.

    Attributes:
        step: Current optimization step counter.
        velocity: Velocity vector (accumulated momentum).
        momentum_coeff: Momentum coefficient (typically 0.9).
        nesterov_lookahead: Lookahead factor for Nesterov term.
    """
    step: Array  # scalar
    velocity: Array  # shape of parameters
    momentum_coeff: Array  # scalar
    nesterov_lookahead: Array  # scalar


@chex.dataclass(frozen=False)
class DynamicEnsembleState:
    """State for dynamic ensemble of 3 optimizers.

    Maintains three independent optimizer states (SGD, Adam, RMSprop) and
    dynamically weights them based on recent gradient alignment. The ensemble
    adapts its composition based on which optimizer is most aligned with
    current gradients.

    Attributes:
        step: Current optimization step counter.
        sgd_momentum: SGD momentum state.
        adam_m: Adam first moment estimate.
        adam_v: Adam second moment estimate.
        rmsprop_v: RMSprop second moment estimate.
        ensemble_weights: (3,) array of optimizer weights that sum to 1.
        gradient_history: Recent gradient buffer for alignment computation.
    """
    step: Array  # scalar
    sgd_momentum: Array  # shape of parameters
    adam_m: Array  # shape of parameters
    adam_v: Array  # shape of parameters
    rmsprop_v: Array  # shape of parameters
    ensemble_weights: Array  # shape (3,)
    gradient_history: Array  # shape (5, *param_shape) for alignment


@chex.dataclass(frozen=False)
class AdaptiveRMSpropState:
    """State for RMSprop with adaptive epsilon.

    Extends standard RMSprop with dynamic epsilon adjustment based on
    gradient magnitude. When gradients are small, epsilon increases to prevent
    excessive scaling. When gradients are large, epsilon decreases for
    responsive updates.

    Attributes:
        step: Current optimization step counter.
        v: Second moment estimates (squared gradient EMA).
        epsilon: Current adaptive epsilon value.
        base_epsilon: Base epsilon for scaling.
        grad_magnitude_ema: EMA of gradient magnitude for epsilon adjustment.
        decay: EMA decay rate for second moments.
    """
    step: Array  # scalar
    v: Array  # shape of parameters
    epsilon: Array  # scalar
    base_epsilon: Array  # scalar
    grad_magnitude_ema: Array  # scalar
    decay: Array  # scalar


# =============================================================================
# 1. Exponential Adaptive Learning Rate Decay
# =============================================================================


def make_exponential_decay_lr_learner(
    hp: Mapping[str, float],
) -> tuple[Callable, Callable]:
    """Factory for exponential adaptive learning rate decay optimizer.

    Implements learning rate schedule: lr(t) = base_lr * exp(-decay_rate * t)
    with momentum-based updates. This provides aggressive initial learning
    that smoothly decays to fine-tuning regime.

    Hyperparameters:
        - base_lr: Initial learning rate (default 0.01)
        - lr_decay_rate: Exponential decay rate (default 0.001)
        - momentum: Momentum coefficient (default 0.9)
        - weight_decay: L2 regularization (default 0.0)

    Args:
        hp: Hyperparameter dictionary.

    Returns:
        Tuple of (init_fn, step_fn) for learner loop.
    """
    base_lr = jnp.asarray(float(hp.get("base_lr", 0.01)))
    decay_rate = jnp.asarray(float(hp.get("lr_decay_rate", 0.001)))
    momentum_coeff = float(hp.get("momentum", 0.9))
    weight_decay = float(hp.get("weight_decay", 0.0))

    def init_fn(key: Array, feature_dim: int = 100) -> tuple[dict[str, Array], ExponentialDecayLRState]:
        """Initialize parameters and optimizer state."""
        params = {
            "w": jr.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }
        state = ExponentialDecayLRState(
            step=jnp.asarray(0, dtype=jnp.int32),
            momentum=jnp.zeros((feature_dim, 1)),
            lr_schedule=base_lr,
            base_lr=base_lr,
            decay_rate=decay_rate,
        )
        return params, state

    def step_fn(
        params: dict[str, Array],
        state: ExponentialDecayLRState,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], ExponentialDecayLRState, float]:
        """Perform one optimization step with exponentially decaying learning rate."""
        # Compute loss and gradients
        def loss_fn(p):
            # x shape: (feature_dim, 1) or (batch, feature_dim)
            if x.shape[0] == p["w"].shape[0]:  # Single sample: x shape (feature_dim, 1)
                pred = jnp.dot(p["w"].T, x) + p["b"]
            else:  # Batch: x shape (batch, feature_dim)
                pred = jnp.dot(x, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Update step counter
        new_step = state.step + 1

        # Exponential decay schedule: lr(t) = base_lr * exp(-decay_rate * t)
        new_lr = base_lr * jnp.exp(-decay_rate * new_step)

        # Momentum-based update
        new_momentum_w = (
            momentum_coeff * state.momentum
            + (1 - momentum_coeff) * grads["w"]
        )
        new_momentum_b = (
            momentum_coeff * jnp.array([0.0])
            + (1 - momentum_coeff) * grads["b"]
        )

        # Parameter update with weight decay
        new_w = params["w"] - new_lr * (new_momentum_w + weight_decay * params["w"])
        new_b = params["b"] - new_lr * new_momentum_b

        new_params = {"w": new_w, "b": new_b}
        new_state = ExponentialDecayLRState(
            step=new_step,
            momentum=new_momentum_w,
            lr_schedule=new_lr,
            base_lr=base_lr,
            decay_rate=decay_rate,
        )

        return new_params, new_state, float(new_lr)

    return init_fn, step_fn


# =============================================================================
# 2. Gradient Acceleration with Nesterov Momentum
# =============================================================================


def make_nesterov_momentum_learner(
    hp: Mapping[str, float],
) -> tuple[Callable, Callable]:
    """Factory for Nesterov accelerated gradient optimizer.

    Implements Nesterov momentum: v(t+1) = mu*v(t) - lr*grad(theta + mu*v(t))
    This "looks ahead" in the gradient direction, providing faster convergence
    than vanilla momentum, especially in poorly conditioned problems.

    Hyperparameters:
        - learning_rate: Step size (default 0.01)
        - momentum: Momentum coefficient (default 0.9)
        - nesterov_lookahead: Lookahead factor (default 1.0)
        - weight_decay: L2 regularization (default 0.0)

    Args:
        hp: Hyperparameter dictionary.

    Returns:
        Tuple of (init_fn, step_fn) for learner loop.
    """
    learning_rate = float(hp.get("learning_rate", 0.01))
    momentum_coeff = jnp.asarray(float(hp.get("momentum", 0.9)))
    nesterov_lookahead = jnp.asarray(float(hp.get("nesterov_lookahead", 1.0)))
    weight_decay = float(hp.get("weight_decay", 0.0))

    def init_fn(key: Array, feature_dim: int = 100) -> tuple[dict[str, Array], NesterovMomentumState]:
        """Initialize parameters and Nesterov momentum state."""
        params = {
            "w": jr.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }
        state = NesterovMomentumState(
            step=jnp.asarray(0, dtype=jnp.int32),
            velocity=jnp.zeros((feature_dim, 1)),
            momentum_coeff=momentum_coeff,
            nesterov_lookahead=nesterov_lookahead,
        )
        return params, state

    def step_fn(
        params: dict[str, Array],
        state: NesterovMomentumState,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], NesterovMomentumState, float]:
        """Perform one Nesterov accelerated gradient step."""
        # Lookahead: compute gradient at params + momentum_coeff * velocity
        lookahead_params = {
            "w": params["w"] + state.momentum_coeff * state.velocity,
            "b": params["b"],
        }

        def loss_fn(p):
            if x.shape[0] == p["w"].shape[0]:  # Single sample: x shape (feature_dim, 1)
                pred = jnp.dot(p["w"].T, x) + p["b"]
            else:  # Batch: x shape (batch, feature_dim)
                pred = jnp.dot(x, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)

        # Gradient at lookahead point
        grads_lookahead = jax.grad(loss_fn)(lookahead_params)

        # Update velocity: v = momentum * v - lr * grad
        new_velocity = (
            state.momentum_coeff * state.velocity
            - learning_rate * grads_lookahead["w"]
        )

        # Update parameters: theta = theta + v + weight_decay term
        new_w = params["w"] + new_velocity - weight_decay * learning_rate * params["w"]
        new_b = params["b"] - learning_rate * grads_lookahead["b"]

        new_params = {"w": new_w, "b": new_b}
        new_state = NesterovMomentumState(
            step=state.step + 1,
            velocity=new_velocity,
            momentum_coeff=state.momentum_coeff,
            nesterov_lookahead=state.nesterov_lookahead,
        )

        return new_params, new_state, learning_rate

    return init_fn, step_fn


# =============================================================================
# 3. Dynamic Ensemble of 3 Optimizers
# =============================================================================


def make_dynamic_ensemble_learner(
    hp: Mapping[str, float],
) -> tuple[Callable, Callable]:
    """Factory for dynamic ensemble of SGD, Adam, and RMSprop.

    Maintains three independent optimizers and dynamically reweights them
    based on gradient alignment. Weights are updated using:
    - Softmax of recent gradient alignment scores
    - Alignment = cosine similarity between optimizer direction and recent gradients

    This allows the ensemble to automatically select which optimizer is best
    suited for the current optimization landscape.

    Hyperparameters:
        - learning_rate: Base learning rate (default 0.01)
        - momentum_sgd: SGD momentum (default 0.9)
        - adam_beta1: Adam momentum (default 0.9)
        - adam_beta2: Adam second moment decay (default 0.999)
        - rmsprop_decay: RMSprop decay (default 0.99)
        - weight_decay: L2 regularization (default 0.0)

    Args:
        hp: Hyperparameter dictionary.

    Returns:
        Tuple of (init_fn, step_fn) for learner loop.
    """
    lr = float(hp.get("learning_rate", 0.01))
    mom_sgd = float(hp.get("momentum_sgd", 0.9))
    beta1 = float(hp.get("adam_beta1", 0.9))
    beta2 = float(hp.get("adam_beta2", 0.999))
    decay_rmsprop = float(hp.get("rmsprop_decay", 0.99))
    weight_decay = float(hp.get("weight_decay", 0.0))

    def init_fn(key: Array, feature_dim: int = 100) -> tuple[dict[str, Array], DynamicEnsembleState]:
        """Initialize parameters and ensemble state."""
        params = {
            "w": jr.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }

        # Initialize ensemble state with equal weights
        state = DynamicEnsembleState(
            step=jnp.asarray(0, dtype=jnp.int32),
            sgd_momentum=jnp.zeros((feature_dim, 1)),
            adam_m=jnp.zeros((feature_dim, 1)),
            adam_v=jnp.zeros((feature_dim, 1)),
            rmsprop_v=jnp.zeros((feature_dim, 1)),
            ensemble_weights=jnp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            gradient_history=jnp.zeros((5, feature_dim, 1)),
        )
        return params, state

    def step_fn(
        params: dict[str, Array],
        state: DynamicEnsembleState,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], DynamicEnsembleState, float]:
        """Perform one ensemble optimization step."""
        def loss_fn(p):
            if x.shape[0] == p["w"].shape[0]:  # Single sample: x shape (feature_dim, 1)
                pred = jnp.dot(p["w"].T, x) + p["b"]
            else:  # Batch: x shape (batch, feature_dim)
                pred = jnp.dot(x, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)

        grads = jax.grad(loss_fn)(params)
        new_step = state.step + 1

        # --- SGD Update ---
        new_sgd_mom = mom_sgd * state.sgd_momentum + grads["w"]
        sgd_update = lr * (new_sgd_mom + weight_decay * params["w"])

        # ---- Adam Update ---
        new_adam_m = beta1 * state.adam_m + (1 - beta1) * grads["w"]
        new_adam_v = beta2 * state.adam_v + (1 - beta2) * (grads["w"] ** 2)
        m_hat = new_adam_m / (1.0 - beta1 ** new_step)
        v_hat = new_adam_v / (1.0 - beta2 ** new_step)
        adam_update = lr * (m_hat / (jnp.sqrt(v_hat) + 1e-8) + weight_decay * params["w"])

        # ---- RMSprop Update ---
        new_rmsprop_v = decay_rmsprop * state.rmsprop_v + (1 - decay_rmsprop) * (grads["w"] ** 2)
        rmsprop_update = lr * (grads["w"] / (jnp.sqrt(new_rmsprop_v) + 1e-8) + weight_decay * params["w"])

        # ---- Compute gradient alignment for reweighting ----
        # Shift gradient history and add new gradient
        new_grad_history = jnp.concatenate(
            [state.gradient_history[1:], jnp.expand_dims(grads["w"], 0)],
            axis=0
        )

        # Compute alignment scores (cosine similarity with recent gradient average)
        avg_recent_grad = jnp.mean(new_grad_history, axis=0)

        def cosine_sim(u, v):
            norm_u = jnp.linalg.norm(u) + 1e-8
            norm_v = jnp.linalg.norm(v) + 1e-8
            return jnp.sum(u * v) / (norm_u * norm_v)

        alignment_sgd = cosine_sim(new_sgd_mom, avg_recent_grad)
        alignment_adam = cosine_sim(new_adam_m, avg_recent_grad)
        alignment_rmsprop = cosine_sim(grads["w"], avg_recent_grad)

        # Soft update of weights using temperature-scaled softmax
        alignments = jnp.array([alignment_sgd, alignment_adam, alignment_rmsprop])
        temperature = 2.0
        new_weights = jax.nn.softmax(alignments / temperature)

        # Blend updates according to ensemble weights
        blended_update = (
            state.ensemble_weights[0] * sgd_update
            + state.ensemble_weights[1] * adam_update
            + state.ensemble_weights[2] * rmsprop_update
        )

        new_w = params["w"] - blended_update
        new_b = params["b"] - lr * grads["b"]

        new_params = {"w": new_w, "b": new_b}
        new_state = DynamicEnsembleState(
            step=new_step,
            sgd_momentum=new_sgd_mom,
            adam_m=new_adam_m,
            adam_v=new_adam_v,
            rmsprop_v=new_rmsprop_v,
            ensemble_weights=new_weights,
            gradient_history=new_grad_history,
        )

        return new_params, new_state, lr

    return init_fn, step_fn


# =============================================================================
# 4. RMSprop with Adaptive Epsilon
# =============================================================================


def make_adaptive_rmsprop_learner(
    hp: Mapping[str, float],
) -> tuple[Callable, Callable]:
    """Factory for RMSprop with adaptive epsilon.

    Extends RMSprop with dynamic epsilon adjustment. Epsilon scales based on
    gradient magnitude EMA:
    - When gradients are small: epsilon increases (smoother updates)
    - When gradients are large: epsilon decreases (more aggressive scaling)

    Formula: epsilon(t) = base_eps * (1 + grad_magnitude_ema)
    This prevents both numerical instability and overly conservative steps.

    Hyperparameters:
        - learning_rate: Step size (default 0.01)
        - rmsprop_decay: EMA decay for second moments (default 0.99)
        - base_epsilon: Base epsilon value (default 1e-8)
        - epsilon_scale: How much gradient magnitude affects epsilon (default 0.1)
        - weight_decay: L2 regularization (default 0.0)

    Args:
        hp: Hyperparameter dictionary.

    Returns:
        Tuple of (init_fn, step_fn) for learner loop.
    """
    lr = float(hp.get("learning_rate", 0.01))
    decay = jnp.asarray(float(hp.get("rmsprop_decay", 0.99)))
    base_eps = jnp.asarray(float(hp.get("base_epsilon", 1e-8)))
    eps_scale = float(hp.get("epsilon_scale", 0.1))
    weight_decay = float(hp.get("weight_decay", 0.0))

    def init_fn(key: Array, feature_dim: int = 100) -> tuple[dict[str, Array], AdaptiveRMSpropState]:
        """Initialize parameters and adaptive RMSprop state."""
        params = {
            "w": jr.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }
        state = AdaptiveRMSpropState(
            step=jnp.asarray(0, dtype=jnp.int32),
            v=jnp.zeros((feature_dim, 1)),
            epsilon=base_eps,
            base_epsilon=base_eps,
            grad_magnitude_ema=jnp.asarray(0.0),
            decay=decay,
        )
        return params, state

    def step_fn(
        params: dict[str, Array],
        state: AdaptiveRMSpropState,
        x: Array,
        y: Array,
    ) -> tuple[dict[str, Array], AdaptiveRMSpropState, float]:
        """Perform one RMSprop step with adaptive epsilon."""
        def loss_fn(p):
            if x.shape[0] == p["w"].shape[0]:  # Single sample: x shape (feature_dim, 1)
                pred = jnp.dot(p["w"].T, x) + p["b"]
            else:  # Batch: x shape (batch, feature_dim)
                pred = jnp.dot(x, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)

        grads = jax.grad(loss_fn)(params)
        new_step = state.step + 1

        # Update second moment estimate
        new_v = state.decay * state.v + (1 - state.decay) * (grads["w"] ** 2)

        # Compute gradient magnitude and update its EMA
        grad_magnitude = jnp.linalg.norm(grads["w"])
        new_grad_mag_ema = (
            0.999 * state.grad_magnitude_ema + 0.001 * grad_magnitude
        )

        # Adaptive epsilon: increases with gradient magnitude
        new_epsilon = state.base_epsilon * (1.0 + eps_scale * new_grad_mag_ema)

        # RMSprop update with adaptive epsilon
        scaled_grads = grads["w"] / (jnp.sqrt(new_v) + new_epsilon)
        new_w = params["w"] - lr * (scaled_grads + weight_decay * params["w"])
        new_b = params["b"] - lr * grads["b"]

        new_params = {"w": new_w, "b": new_b}
        new_state = AdaptiveRMSpropState(
            step=new_step,
            v=new_v,
            epsilon=new_epsilon,
            base_epsilon=state.base_epsilon,
            grad_magnitude_ema=new_grad_mag_ema,
            decay=state.decay,
        )

        return new_params, new_state, lr

    return init_fn, step_fn


# =============================================================================
# Registry and helpers
# =============================================================================

SCR_ADVANCED_OPTIMIZERS = {
    "exponential_decay_lr": make_exponential_decay_lr_learner,
    "nesterov_momentum": make_nesterov_momentum_learner,
    "dynamic_ensemble": make_dynamic_ensemble_learner,
    "adaptive_rmsprop": make_adaptive_rmsprop_learner,
}


def register_scr_advanced_optimizers() -> dict[str, Callable]:
    """Register all advanced optimizer variants.

    Returns:
        Dictionary mapping optimizer names to factory functions.
    """
    return SCR_ADVANCED_OPTIMIZERS
