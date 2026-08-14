"""SCR v2 learner composition variants - combine multiple mechanisms.

Implements learner variants that compose multiple SCR mechanisms.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_combined_norm_gate_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner combining normalization + gating."""
    step_size = hp.get("step_size", 0.01)
    norm_decay = hp.get("norm_decay", 0.99)
    gate_beta = hp.get("gate_beta", 0.5)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "norm_mean": 0.0,
            "norm_var": 1.0,
            "gate": 1.0,
        }

    def step_fn(params, state, x, y, grads):
        # Normalize gradients
        norm_mean_new = norm_decay * state["norm_mean"] + (1 - norm_decay) * jnp.mean(grads)
        norm_var_new = norm_decay * state["norm_var"] + (1 - norm_decay) * jnp.var(grads)
        normalized_grads = grads / (jnp.sqrt(norm_var_new) + 1e-8)

        # Compute gate from gradient magnitude
        grad_mag = jnp.mean(jnp.abs(grads))
        gate = 1.0 / (1.0 + gate_beta * grad_mag)

        # Apply both
        update = gate * normalized_grads

        params_new = {
            "w": params["w"] - step_size * update,
            "b": params["b"] - step_size * jnp.mean(update),
        }

        state_new = {
            "norm_mean": norm_mean_new,
            "norm_var": norm_var_new,
            "gate": gate,
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_meta_decay_composition_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with meta-decay composition."""
    base_lr = hp.get("base_lr", 0.01)
    meta_decay = hp.get("meta_decay", 0.9)
    surprise_weight = hp.get("surprise_weight", 0.1)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "meta_state": 0.5,
            "surprise_ema": 0.1,
        }

    def step_fn(params, state, x, y, grads):
        # Surprise: gradient magnitude change
        surprise = jnp.abs(jnp.mean(jnp.abs(grads)) - state["surprise_ema"])
        surprise_ema_new = 0.9 * state["surprise_ema"] + 0.1 * jnp.mean(jnp.abs(grads))

        # Meta-decay: modulate learning rate by surprise
        meta_state_new = meta_decay * state["meta_state"] + (1 - meta_decay) * surprise
        adaptive_lr = base_lr * (1 + surprise_weight * meta_state_new)

        params_new = {
            "w": params["w"] - adaptive_lr * grads,
            "b": params["b"] - adaptive_lr * jnp.mean(grads),
        }

        state_new = {
            "meta_state": meta_state_new,
            "surprise_ema": surprise_ema_new,
        }

        return params_new, state_new, (0.0, 0.0, adaptive_lr)

    return init_fn, step_fn


def make_buffer_norm_composition_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner composing buffer + normalization."""
    step_size = hp.get("step_size", 0.01)
    buffer_size = int(hp.get("buffer_size", 50))
    norm_decay = hp.get("norm_decay", 0.99)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "buffer": [],
            "norm_mean": 0.0,
            "norm_var": 1.0,
        }

    def step_fn(params, state, x, y, grads):
        # Add to buffer
        buffer = list(state["buffer"])
        if len(buffer) < buffer_size:
            buffer.append(grads)
        else:
            buffer = buffer[1:] + [grads]

        # Average from buffer
        avg_grads = jnp.mean(jnp.array(buffer) if buffer else jnp.expand_dims(grads, 0), axis=0)

        # Normalize
        norm_mean_new = norm_decay * state["norm_mean"] + (1 - norm_decay) * jnp.mean(avg_grads)
        norm_var_new = norm_decay * state["norm_var"] + (1 - norm_decay) * jnp.var(avg_grads)
        normalized = avg_grads / (jnp.sqrt(norm_var_new) + 1e-8)

        params_new = {
            "w": params["w"] - step_size * normalized,
            "b": params["b"] - step_size * jnp.mean(normalized),
        }

        state_new = {
            "buffer": buffer,
            "norm_mean": norm_mean_new,
            "norm_var": norm_var_new,
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_rls_gate_composition_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner composing RLS-style gating."""
    step_size = hp.get("step_size", 0.01)
    rls_lambda = hp.get("rls_lambda", 0.99)
    gate_threshold = hp.get("gate_threshold", 0.5)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "P": jnp.eye(feature_dim) * 0.1,
            "error_ema": 0.5,
        }

    def step_fn(params, state, x, y, grads):
        # RLS-style P matrix update (simplified)
        P_new = (state["P"] + jnp.eye(len(grads)) * 0.01) / rls_lambda

        # Error-based gating
        error_ema_new = 0.9 * state["error_ema"] + 0.1 * jnp.mean(jnp.abs(grads))
        gate = 1.0 if error_ema_new > gate_threshold else 0.5

        # Update with gating
        update = gate * grads

        params_new = {
            "w": params["w"] - step_size * update,
            "b": params["b"] - step_size * jnp.mean(update),
        }

        state_new = {
            "P": P_new,
            "error_ema": error_ema_new,
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


SCR_COMPOSITION_VARIANTS = {
    "norm_gate_composition": make_combined_norm_gate_learner,
    "meta_decay_composition": make_meta_decay_composition_learner,
    "buffer_norm_composition": make_buffer_norm_composition_learner,
    "rls_gate_composition": make_rls_gate_composition_learner,
}


def register_scr_composition_variants():
    """Register SCR composition variants."""
    print(f"[OK] Registered {len(SCR_COMPOSITION_VARIANTS)} SCR composition variants")
    return SCR_COMPOSITION_VARIANTS
