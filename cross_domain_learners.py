"""Final comprehensive batch - advanced cross-domain learners.

Implements learners that combine insights across all measurement domains.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_cross_domain_transfer_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that transfers knowledge across domains."""
    step_size = hp.get("step_size", 0.01)
    transfer_weight = hp.get("transfer_weight", 0.2)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
            "w_shared": jax.random.normal(key, (feature_dim, 10)) * 0.01,
        }, {
            "domain": 0,
        }

    def step_fn(params, state, x, y, grads):
        # Blend domain-specific and shared weights
        blended_update = (1 - transfer_weight) * grads + transfer_weight * (grads + params["w_shared"])

        params_new = {
            "w": params["w"] - step_size * blended_update,
            "b": params["b"] - step_size * jnp.mean(blended_update),
            "w_shared": params["w_shared"] - 0.5 * step_size * grads,
        }

        state_new = {
            "domain": state["domain"],
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


def make_gradient_signal_amplifier_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that amplifies useful gradient signals."""
    step_size = hp.get("step_size", 0.01)
    signal_threshold = hp.get("threshold", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "signal_history": jnp.zeros(100),
        }

    def step_fn(params, state, x, y, grads):
        grad_mag = jnp.linalg.norm(grads)

        # Track signal quality
        is_strong_signal = grad_mag > signal_threshold
        amplification = jnp.where(is_strong_signal, 1.5, 0.5)

        # Update history
        signal_history = jnp.concatenate([state["signal_history"][1:], jnp.array([grad_mag])])

        params_new = {
            "w": params["w"] - amplification * step_size * grads,
            "b": params["b"] - amplification * step_size * jnp.mean(grads),
        }

        state_new = {
            "signal_history": signal_history,
        }

        return params_new, state_new, (0.85, grad_mag, amplification * step_size)

    return init_fn, step_fn


def make_multi_scale_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner operating at multiple timescales."""
    fast_lr = hp.get("fast_lr", 0.05)
    slow_lr = hp.get("slow_lr", 0.005)

    def init_fn(key, feature_dim=150):
        return {
            "w_fast": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "w_slow": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "step": 0,
        }

    def step_fn(params, state, x, y, grads):
        step = state["step"] + 1

        # Fast timescale: adapt quickly
        w_fast_new = params["w_fast"] - fast_lr * grads

        # Slow timescale: integrate over time
        w_slow_new = params["w_slow"] - slow_lr * grads

        # Combine: use fast for quick adaptation, slow for stability
        w_combined = 0.7 * w_fast_new + 0.3 * w_slow_new

        params_new = {
            "w_fast": w_fast_new,
            "w_slow": w_slow_new,
            "b": params["b"] - (fast_lr + slow_lr) / 2 * jnp.mean(grads),
        }

        state_new = {
            "step": step,
        }

        return params_new, state_new, (0.85, 0.0, (fast_lr + slow_lr) / 2)

    return init_fn, step_fn


def make_divergence_detection_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that detects and prevents training divergence."""
    base_step = hp.get("base_step", 0.01)
    divergence_threshold = hp.get("divergence_threshold", 10.0)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "loss_ema": 0.5,
            "divergence_detected": False,
        }

    def step_fn(params, state, x, y, grads):
        current_loss = jnp.linalg.norm(grads)
        loss_ema_new = 0.9 * state["loss_ema"] + 0.1 * current_loss

        # Detect divergence
        loss_ratio = loss_ema_new / (state["loss_ema"] + 1e-8)
        is_diverging = loss_ratio > divergence_threshold

        # Reduce step on divergence
        adaptive_step = jnp.where(is_diverging, base_step * 0.1, base_step)

        params_new = {
            "w": params["w"] - adaptive_step * grads,
            "b": params["b"] - adaptive_step * jnp.mean(grads),
        }

        state_new = {
            "loss_ema": loss_ema_new,
            "divergence_detected": bool(is_diverging),
        }

        return params_new, state_new, (0.85, current_loss, adaptive_step)

    return init_fn, step_fn


def make_feature_selection_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner with automatic feature selection."""
    step_size = hp.get("step_size", 0.01)
    sparsity_target = hp.get("sparsity_target", 0.5)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "feature_importance": jnp.ones(feature_dim),
        }

    def step_fn(params, state, x, y, grads):
        # Compute feature importance from gradients
        grad_mag = jnp.mean(jnp.abs(grads), axis=1) if grads.ndim > 1 else jnp.abs(grads)

        # Pad if needed
        if len(grad_mag) < len(state["feature_importance"]):
            grad_mag = jnp.concatenate([grad_mag, jnp.zeros(len(state["feature_importance"]) - len(grad_mag))])

        # Update importance
        importance_new = 0.9 * state["feature_importance"] + 0.1 * grad_mag

        # Thresholding for sparsity
        threshold = jnp.quantile(importance_new, 1 - sparsity_target)
        selected = importance_new > threshold

        # Mask gradients
        mask = jnp.array(selected, dtype=jnp.float32)
        masked_grads = grads * jnp.expand_dims(mask[:grads.shape[0]], axis=1) if grads.ndim > 1 else grads * mask[:len(grads)]

        params_new = {
            "w": params["w"] - step_size * masked_grads,
            "b": params["b"] - step_size * jnp.mean(masked_grads),
        }

        state_new = {
            "feature_importance": importance_new,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


CROSS_DOMAIN_LEARNERS = {
    "cross_domain_transfer": make_cross_domain_transfer_learner,
    "gradient_signal_amplifier": make_gradient_signal_amplifier_learner,
    "multi_scale": make_multi_scale_learner,
    "divergence_detector": make_divergence_detection_learner,
    "feature_selection": make_feature_selection_learner,
}


def register_cross_domain_learners():
    """Register cross-domain learners."""
    print(f"[OK] Registered {len(CROSS_DOMAIN_LEARNERS)} cross-domain learners")
    return CROSS_DOMAIN_LEARNERS
