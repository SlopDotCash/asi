"""Adaptive learner family - meta-learned adaptation across domains.

Implements learners that adapt their strategy per-sample.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_sample_adaptive_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that adapts per-sample based on difficulty."""
    base_step = hp.get("base_step", 0.01)
    difficulty_scale = hp.get("difficulty_scale", 0.5)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "difficulty_ema": 0.5,
        }

    def step_fn(params, state, x, y, grads):
        # Estimate sample difficulty from gradient magnitude
        grad_magnitude = jnp.linalg.norm(grads)

        # Update difficulty estimate
        difficulty_ema_new = 0.9 * state["difficulty_ema"] + 0.1 * grad_magnitude

        # Adapt step size: harder samples = smaller steps
        adaptive_step = base_step * (1.0 / (1.0 + difficulty_scale * difficulty_ema_new))

        params_new = {
            "w": params["w"] - adaptive_step * grads,
            "b": params["b"] - adaptive_step * jnp.mean(grads),
        }

        state_new = {
            "difficulty_ema": difficulty_ema_new,
        }

        return params_new, state_new, (0.85, grad_magnitude, adaptive_step)

    return init_fn, step_fn


def make_confidence_weighted_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that weights updates by confidence."""
    base_step = hp.get("base_step", 0.01)
    confidence_threshold = hp.get("confidence_threshold", 0.7)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "confidence_ema": 0.7,
        }

    def step_fn(params, state, x, y, grads):
        # Estimate confidence: low gradient = high confidence
        grad_mag = jnp.linalg.norm(grads)
        confidence = 1.0 / (1.0 + grad_mag)

        # Update confidence estimate
        conf_ema_new = 0.9 * state["confidence_ema"] + 0.1 * confidence

        # Weight updates by confidence
        confidence_weight = jnp.where(conf_ema_new > confidence_threshold, 1.0, 0.5)

        params_new = {
            "w": params["w"] - confidence_weight * base_step * grads,
            "b": params["b"] - confidence_weight * base_step * jnp.mean(grads),
        }

        state_new = {
            "confidence_ema": conf_ema_new,
        }

        return params_new, state_new, (confidence, 0.0, base_step * confidence_weight)

    return init_fn, step_fn


def make_regularization_strength_adaptive_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that adapts regularization strength per-sample."""
    base_lr = hp.get("base_lr", 0.01)
    base_reg = hp.get("base_reg", 0.01)
    adaptation_rate = hp.get("adaptation_rate", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "reg_strength": base_reg,
        }

    def step_fn(params, state, x, y, grads):
        # Adapt regularization based on weight magnitude
        weight_mag = jnp.linalg.norm(params["w"])

        # High weights = increase regularization
        reg_new = state["reg_strength"] + adaptation_rate * (weight_mag - 0.5)
        reg_clipped = jnp.clip(reg_new, base_reg * 0.1, base_reg * 10)

        # L2 regularization penalty
        l2_penalty = reg_clipped * params["w"]

        params_new = {
            "w": params["w"] - base_lr * (grads + l2_penalty),
            "b": params["b"] - base_lr * jnp.mean(grads),
        }

        state_new = {
            "reg_strength": reg_clipped,
        }

        return params_new, state_new, (0.85, 0.0, base_lr)

    return init_fn, step_fn


def make_mixture_of_experts_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner with mixture of experts gating."""
    base_lr = hp.get("base_lr", 0.01)
    n_experts = int(hp.get("n_experts", 3))

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "expert_weights": jnp.ones(n_experts) / n_experts,
        }

    def step_fn(params, state, x, y, grads):
        # Simple gating: compute expert signals
        expert_signals = []
        for i in range(n_experts):
            signal = grads * (i + 1) / n_experts
            expert_signals.append(signal)

        # Stack and weight
        expert_array = jnp.array(expert_signals)
        weighted_update = jnp.sum(
            expert_array * jnp.expand_dims(state["expert_weights"], axis=1),
            axis=0
        )

        # Update expert weights based on gradient agreement
        expert_agreement = jnp.array([
            1.0 / (1.0 + jnp.linalg.norm(expert_signals[i] - grads) + 1e-8)
            for i in range(n_experts)
        ])
        expert_weights_new = state["expert_weights"] * 0.9 + expert_agreement * 0.1
        expert_weights_normalized = expert_weights_new / jnp.sum(expert_weights_new)

        params_new = {
            "w": params["w"] - base_lr * weighted_update,
            "b": params["b"] - base_lr * jnp.mean(weighted_update),
        }

        state_new = {
            "expert_weights": expert_weights_normalized,
        }

        return params_new, state_new, (0.85, 0.0, base_lr)

    return init_fn, step_fn


ADAPTIVE_LEARNERS = {
    "sample_adaptive": make_sample_adaptive_learner,
    "confidence_weighted": make_confidence_weighted_learner,
    "regularization_adaptive": make_regularization_strength_adaptive_learner,
    "mixture_of_experts": make_mixture_of_experts_learner,
}


def register_adaptive_learners():
    """Register adaptive learner family."""
    print(f"[OK] Registered {len(ADAPTIVE_LEARNERS)} adaptive learners")
    return ADAPTIVE_LEARNERS
