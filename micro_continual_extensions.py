"""Additional micro-continual mechanisms - expansion beyond preregistered arms.

Extends micro-continual with new learner variants.
"""

from typing import Mapping, Tuple
import jax
import jax.numpy as jnp


# Type definitions
LearnerInitFn = Callable
ScreeningStepFn = Callable


def _make_replay_buffer_learner(
    hp: Mapping[str, float],
) -> Tuple[LearnerInitFn, ScreeningStepFn]:
    """Micro-continual with experience replay buffer for stability."""
    step_size = hp.get("step_size", 0.01)
    weight_decay = hp.get("weight_decay", 0.01)
    buffer_size = int(hp.get("buffer_size", 1000))
    sample_batch_size = int(hp.get("sample_batch", 32))

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "replay_buffer": [],
            "buffer_idx": 0,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Add to replay buffer
        buffer = list(state["replay_buffer"])
        if len(buffer) < buffer_size:
            buffer.append((x, y, grads))
        else:
            buffer[state["buffer_idx"] % buffer_size] = (x, y, grads)

        # Sample from buffer
        if len(buffer) >= sample_batch_size:
            indices = jax.random.choice(jnp.arange(len(buffer)), (sample_batch_size,))
            batch = [buffer[i] for i in indices]
        else:
            batch = buffer

        # Compute average gradient from batch
        avg_grads = jnp.mean(jnp.array([g for _, _, g in batch]), axis=0)

        # Update with L2 regularization
        params_new = {}
        for key, param in params.items():
            params_new[key] = param - step_size * (avg_grads + weight_decay * param)

        state_new = {
            "replay_buffer": buffer,
            "buffer_idx": state["buffer_idx"] + 1,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(0.0, dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_plasticity_modulated_learner(
    hp: Mapping[str, float],
) -> Tuple[LearnerInitFn, ScreeningStepFn]:
    """Learner with online plasticity modulation based on prediction error."""
    base_step = hp.get("step_size", 0.01)
    min_step = hp.get("min_step", 0.001)
    max_step = hp.get("max_step", 0.1)
    error_scale = hp.get("error_scale", 1.0)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "error_ema": 0.5,
            "plasticity": base_step,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Modulate step size based on error
        error_ema_new = 0.9 * state["error_ema"] + 0.1 * jnp.mean(jnp.abs(grads))
        plasticity_new = jnp.clip(
            base_step * (1.0 + error_scale * error_ema_new),
            min_step, max_step
        )

        # Update with modulated step
        params_new = {}
        for key, param in params.items():
            params_new[key] = param - plasticity_new * grads

        state_new = {
            "error_ema": error_ema_new,
            "plasticity": plasticity_new,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = float(plasticity_new)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_task_boundary_detector_learner(
    hp: Mapping[str, float],
) -> Tuple[LearnerInitFn, ScreeningStepFn]:
    """Learner that detects task boundaries and resets accordingly."""
    step_size = hp.get("step_size", 0.01)
    boundary_threshold = hp.get("boundary_threshold", 0.5)
    reset_factor = hp.get("reset_factor", 0.1)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "prev_error": 0.0,
            "error_history": [],
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        current_error = jnp.mean(jnp.abs(grads))

        # Detect boundary (sudden error increase)
        error_jump = current_error - state["prev_error"]
        is_boundary = error_jump > boundary_threshold

        # Partial reset if boundary detected
        reset_mask = reset_factor if is_boundary else 0.0

        params_new = {}
        for key, param in params.items():
            params_new[key] = (1.0 - reset_mask) * param - step_size * grads

        state_new = {
            "prev_error": current_error,
            "error_history": state["error_history"] + [float(current_error)],
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(0.0, dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


# Registry for new micro-continual variants
MICRO_CONTINUAL_EXTENSIONS = {
    "replay_buffer_learner": _make_replay_buffer_learner,
    "plasticity_modulated": _make_plasticity_modulated_learner,
    "task_boundary_detector": _make_task_boundary_detector_learner,
}


def register_micro_continual_extensions():
    """Register all micro-continual extension mechanisms."""
    print(f"[OK] Registered {len(MICRO_CONTINUAL_EXTENSIONS)} micro-continual extensions")
    return MICRO_CONTINUAL_EXTENSIONS
