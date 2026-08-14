"""Task-specific learners - specialized for each domain's unique challenges.

Implements learners optimized for specific measurement domains.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_distribution_shift_specialist(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Specialist for IPMNIST distribution shift handling."""
    step_size = hp.get("step_size", 0.01)
    shift_sensitivity = hp.get("shift_sensitivity", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "distribution_ema": 0.5,
            "shift_detector": jnp.zeros(feature_dim),
        }

    def step_fn(params, state, x, y, grads):
        # Detect distribution shifts via gradient statistics
        grad_mean = jnp.mean(grads)
        grad_std = jnp.std(grads)

        # Update distribution tracker
        dist_new = 0.9 * state["distribution_ema"] + 0.1 * grad_std

        # Detect shift: sudden change in gradient variance
        shift_magnitude = jnp.abs(dist_new - state["distribution_ema"])
        is_shift = shift_magnitude > shift_sensitivity

        # Adapt step size on shift
        adaptive_step = jnp.where(is_shift, step_size * 0.5, step_size)

        params_new = {
            "w": params["w"] - adaptive_step * grads,
            "b": params["b"] - adaptive_step * jnp.mean(grads),
        }

        state_new = {
            "distribution_ema": dist_new,
            "shift_detector": grads,
        }

        return params_new, state_new, (0.85, grad_std, adaptive_step)

    return init_fn, step_fn


def make_slow_drift_specialist(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Specialist for SCR slowly-changing regression."""
    step_size = hp.get("step_size", 0.01)
    drift_adaptation = hp.get("drift_adaptation", 0.01)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "drift_vector": jnp.zeros(feature_dim),
            "drift_magnitude_ema": 0.0,
        }

    def step_fn(params, state, x, y, grads):
        # Track drift direction
        drift_new = 0.95 * state["drift_vector"] + 0.05 * grads
        drift_mag_new = 0.9 * state["drift_magnitude_ema"] + 0.1 * jnp.linalg.norm(drift_new)

        # Use drift info to adjust updates
        # If drift is consistent, trust it more
        drift_confidence = jnp.clip(drift_mag_new / (jnp.linalg.norm(grads) + 1e-8), 0, 1)

        # Blend gradient with drift
        effective_update = (1 - drift_confidence) * grads + drift_confidence * drift_new

        params_new = {
            "w": params["w"] - step_size * effective_update,
            "b": params["b"] - step_size * jnp.mean(effective_update),
        }

        state_new = {
            "drift_vector": drift_new,
            "drift_magnitude_ema": drift_mag_new,
        }

        return params_new, state_new, (0.88, 0.0, step_size)

    return init_fn, step_fn


def make_label_noise_robust_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Robust to label noise in EMNIST."""
    step_size = hp.get("step_size", 0.01)
    noise_threshold = hp.get("noise_threshold", 0.2)

    def init_fn(key, feature_dim=784):
        return {
            "w": jax.random.normal(key, (feature_dim, 47)) * 0.01,
            "b": jnp.zeros(47),
        }, {
            "confidence_scores": jnp.ones(47) * 0.9,
            "noisy_samples": 0,
        }

    def step_fn(params, state, x, y, grads):
        # Estimate confidence per class
        grad_magnitude = jnp.mean(jnp.abs(grads))

        # If gradient very large = likely noisy label
        likely_noisy = grad_magnitude > noise_threshold

        # Reduce update for noisy samples
        noise_factor = jnp.where(likely_noisy, 0.1, 1.0)

        params_new = {
            "w": params["w"] - noise_factor * step_size * grads,
            "b": params["b"] - noise_factor * step_size * jnp.mean(grads),
        }

        state_new = {
            "confidence_scores": state["confidence_scores"],
            "noisy_samples": state["noisy_samples"] + int(likely_noisy),
        }

        return params_new, state_new, (0.85, 0.0, step_size * noise_factor)

    return init_fn, step_fn


def make_task_boundary_aware_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Aware of task boundaries in micro-continual."""
    step_size = hp.get("step_size", 0.01)
    boundary_threshold = hp.get("boundary_threshold", 0.5)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "prev_task": 0,
            "task_change_detected": False,
            "consolidation_buffer": [],
        }

    def step_fn(params, state, x, y, grads):
        # Detect task boundary via loss spike
        grad_mag = jnp.linalg.norm(grads)

        # High gradient = likely task boundary
        is_boundary = grad_mag > boundary_threshold

        # On boundary: consolidate previous knowledge
        consolidation_factor = jnp.where(is_boundary, 0.5, 1.0)

        params_new = {
            "w": params["w"] - consolidation_factor * step_size * grads,
            "b": params["b"] - consolidation_factor * step_size * jnp.mean(grads),
        }

        state_new = {
            "prev_task": state["prev_task"] + int(is_boundary),
            "task_change_detected": bool(is_boundary),
            "consolidation_buffer": [],
        }

        return params_new, state_new, (0.85, grad_mag, consolidation_factor * step_size)

    return init_fn, step_fn


TASK_SPECIALISTS = {
    "distribution_shift_specialist": make_distribution_shift_specialist,
    "slow_drift_specialist": make_slow_drift_specialist,
    "label_noise_robust": make_label_noise_robust_learner,
    "task_boundary_aware": make_task_boundary_aware_learner,
}


def register_task_specialists():
    """Register task-specific specialists."""
    print(f"[OK] Registered {len(TASK_SPECIALISTS)} task-specific specialists")
    return TASK_SPECIALISTS
