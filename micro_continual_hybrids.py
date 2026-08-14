"""Micro-continual hybrid mechanism variants - combined learning strategies.

Implements hybrid micro-continual learners combining multiple mechanisms.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_rls_meta_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual combining RLS head + meta-learning."""
    step_size = hp.get("step_size", 0.01)
    meta_step = hp.get("meta_step", 0.001)
    rls_lambda = hp.get("rls_lambda", 0.99)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
            "meta_w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
        }, {
            "rls_P": jnp.eye(feature_dim) * 0.1,
            "meta_step_ema": 0.01,
        }

    def step_fn(params, state, x, y, grads):
        # RLS update on P matrix
        P_new = (state["rls_P"] + jnp.eye(len(grads)) * 0.001) / rls_lambda

        # Meta-learning: adapt meta parameters
        meta_step_new = 0.9 * state["meta_step_ema"] + 0.1 * jnp.mean(jnp.abs(grads))
        meta_grads = grads + meta_step * (params["w"] - params["meta_w"])

        # Combine RLS + meta
        params_new = {
            "w": params["w"] - step_size * (grads + 0.1 * meta_grads),
            "b": params["b"] - step_size * jnp.mean(grads),
            "meta_w": params["meta_w"] - meta_step * meta_grads,
        }

        state_new = {
            "rls_P": P_new,
            "meta_step_ema": meta_step_new,
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_buffer_plasticity_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual combining experience buffer + plasticity modulation."""
    step_size = hp.get("step_size", 0.01)
    buffer_size = int(hp.get("buffer_size", 100))
    error_scale = hp.get("error_scale", 1.0)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "buffer": [],
            "error_ema": 0.5,
        }

    def step_fn(params, state, x, y, grads):
        # Buffer management
        buffer = list(state["buffer"])
        buffer.append(grads)
        if len(buffer) > buffer_size:
            buffer = buffer[-(buffer_size):]

        # Error-based plasticity modulation
        error_ema_new = 0.9 * state["error_ema"] + 0.1 * jnp.mean(jnp.abs(grads))
        plasticity = 0.5 + error_scale * error_ema_new  # Modulate between 0.5 and 1.5

        # Average from buffer
        buffer_avg = jnp.mean(jnp.array(buffer) if buffer else jnp.expand_dims(grads, 0), axis=0)

        # Mix: current + buffer average
        mixed = 0.5 * grads + 0.5 * buffer_avg

        params_new = {
            "w": params["w"] - plasticity * step_size * mixed,
            "b": params["b"] - plasticity * step_size * jnp.mean(mixed),
        }

        state_new = {
            "buffer": buffer,
            "error_ema": error_ema_new,
        }

        return params_new, state_new, (0.0, 0.0, step_size * plasticity)

    return init_fn, step_fn


def make_gate_boundary_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual combining gating + task boundary detection."""
    step_size = hp.get("step_size", 0.01)
    gate_threshold = hp.get("gate_threshold", 0.5)
    boundary_threshold = hp.get("boundary_threshold", 0.3)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "prev_error": 0.0,
            "gate_ema": 0.5,
        }

    def step_fn(params, state, x, y, grads):
        current_error = jnp.mean(jnp.abs(grads))
        error_jump = current_error - state["prev_error"]

        # Boundary detection
        is_boundary = error_jump > boundary_threshold
        boundary_factor = 0.5 if is_boundary else 1.0

        # Gating
        gate_ema_new = 0.9 * state["gate_ema"] + 0.1 * current_error
        gate = 1.0 if gate_ema_new > gate_threshold else 0.3

        # Combined: gate + boundary adaptation
        update = boundary_factor * gate * grads

        params_new = {
            "w": params["w"] - step_size * update,
            "b": params["b"] - step_size * jnp.mean(update),
        }

        state_new = {
            "prev_error": current_error,
            "gate_ema": gate_ema_new,
        }

        return params_new, state_new, (0.0, 0.0, step_size * gate * boundary_factor)

    return init_fn, step_fn


def make_episodic_meta_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual combining episodic memory + meta-learning."""
    step_size = hp.get("step_size", 0.01)
    meta_step = hp.get("meta_step", 0.001)
    memory_size = int(hp.get("memory_size", 50))

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
            "meta_w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
        }, {
            "memory": [],
        }

    def step_fn(params, state, x, y, grads):
        # Episodic memory
        memory = list(state["memory"])
        memory.append(grads)
        if len(memory) > memory_size:
            memory = memory[-memory_size:]

        # Retrieve similar episodes
        if len(memory) > 5:
            memory_avg = jnp.mean(jnp.array(memory[-5:]), axis=0)
        else:
            memory_avg = grads

        # Meta-update
        meta_grads = grads - 0.1 * (params["w"] - params["meta_w"])

        # Combine episodic + meta
        combined = 0.6 * memory_avg + 0.4 * meta_grads

        params_new = {
            "w": params["w"] - step_size * combined,
            "b": params["b"] - step_size * jnp.mean(combined),
            "meta_w": params["meta_w"] - meta_step * meta_grads,
        }

        state_new = {"memory": memory}
        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


MICRO_CONTINUAL_HYBRIDS = {
    "rls_meta_hybrid": make_rls_meta_hybrid_learner,
    "buffer_plasticity_hybrid": make_buffer_plasticity_hybrid_learner,
    "gate_boundary_hybrid": make_gate_boundary_hybrid_learner,
    "episodic_meta_hybrid": make_episodic_meta_hybrid_learner,
}


def register_micro_continual_hybrids():
    """Register all micro-continual hybrid variants."""
    print(f"[OK] Registered {len(MICRO_CONTINUAL_HYBRIDS)} micro-continual hybrids")
    return MICRO_CONTINUAL_HYBRIDS
