"""Additional micro-continual gate mechanisms - signal-based plasticity control.

Implements various gating strategies for plasticity modulation.
"""

from typing import Mapping, Tuple, Callable
import jax
import jax.numpy as jnp


def _make_loss_gated_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Loss-based gating - modulate learning by prediction error."""
    base_step = hp.get("step_size", 0.01)
    loss_threshold = hp.get("loss_threshold", 0.5)
    gate_smooth = hp.get("gate_smooth", 0.9)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "loss_ema": 0.5,
            "gate": 1.0,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        current_loss = jnp.mean(jnp.abs(grads))
        loss_ema = gate_smooth * state["loss_ema"] + (1 - gate_smooth) * current_loss

        # Gate: 1 if error high, 0 if error low
        gate = jnp.where(loss_ema > loss_threshold, 1.0, 0.1)

        # Modulate update
        params_new = {}
        for key, param in params.items():
            params_new[key] = param - gate * base_step * grads

        state_new = {
            "loss_ema": loss_ema,
            "gate": gate,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(gate * base_step), dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_gradient_norm_gated_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Gradient norm gating - gate based on gradient magnitude."""
    base_step = hp.get("step_size", 0.01)
    grad_threshold = hp.get("grad_threshold", 0.1)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "grad_history": [],
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        grad_norm = jnp.linalg.norm(grads)

        # Gate: strong gate for large gradients, weak for small
        gate = jnp.clip(grad_norm / (grad_threshold + 1e-8), 0.1, 1.0)

        params_new = {}
        for key, param in params.items():
            params_new[key] = param - gate * base_step * grads

        state_new = {
            "grad_history": list(state["grad_history"])[-100:] + [float(grad_norm)],
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(gate * base_step), dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_variance_gated_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Variance-based gating - gate based on gradient variance."""
    base_step = hp.get("step_size", 0.01)
    variance_window = int(hp.get("window", 20))

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "grad_window": [],
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        window = list(state["grad_window"]) + [grads][-variance_window:]

        if len(window) > 1:
            grad_variance = jnp.var(jnp.array(window))
            # High variance = uncertain = reduce learning
            gate = 1.0 / (1.0 + grad_variance)
        else:
            gate = 1.0

        params_new = {}
        for key, param in params.items():
            params_new[key] = param - gate * base_step * grads

        state_new = {
            "grad_window": window,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(gate * base_step), dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_confidence_gated_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Confidence-based gating - gate based on model confidence."""
    base_step = hp.get("step_size", 0.01)
    confidence_threshold = hp.get("conf_threshold", 0.5)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "confidence_ema": 0.5,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Estimate confidence as 1 - normalized_error
        error = jnp.mean(jnp.abs(grads))
        confidence = 1.0 / (1.0 + error)
        confidence_ema = 0.9 * state["confidence_ema"] + 0.1 * confidence

        # Gate: high learning when confident, low when uncertain
        gate = jnp.where(confidence_ema > confidence_threshold, 1.0, 0.3)

        params_new = {}
        for key, param in params.items():
            params_new[key] = param - gate * base_step * grads

        state_new = {
            "confidence_ema": confidence_ema,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(gate * base_step), dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


GATE_VARIANTS = {
    "loss_gated": _make_loss_gated_learner,
    "gradient_norm_gated": _make_gradient_norm_gated_learner,
    "variance_gated": _make_variance_gated_learner,
    "confidence_gated": _make_confidence_gated_learner,
}


def register_gate_variants():
    """Register all gate mechanism variants."""
    print(f"[OK] Registered {len(GATE_VARIANTS)} gate variants")
    return GATE_VARIANTS
