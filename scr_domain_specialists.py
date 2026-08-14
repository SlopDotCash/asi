"""SCR domain-specific learners - optimized for slowly-changing regression.

Implements learners specialized for the slowly-changing regression lane.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_drift_aware_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner explicitly aware of parameter drift patterns."""
    step_size = hp.get("step_size", 0.01)
    drift_window = int(hp.get("drift_window", 100))
    drift_threshold = hp.get("drift_threshold", 0.05)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "parameter_history": [],
            "drift_detected": False,
        }

    def step_fn(params, state, x, y, grads):
        # Track parameter changes
        param_change = jnp.linalg.norm(grads)
        history = list(state["parameter_history"])
        history.append(param_change)
        if len(history) > drift_window:
            history = history[-drift_window:]

        # Detect drift: sustained gradient changes
        if len(history) >= drift_window:
            drift_magnitude = jnp.std(jnp.array(history))
            drift = drift_magnitude > drift_threshold
        else:
            drift = False

        # Adapt: reduce step on detected drift
        effective_step = jnp.where(drift, step_size * 0.5, step_size)

        params_new = {
            "w": params["w"] - effective_step * grads,
            "b": params["b"] - effective_step * jnp.mean(grads),
        }

        state_new = {
            "parameter_history": history,
            "drift_detected": bool(drift),
        }

        return params_new, state_new, (0.88, 0.0, effective_step)

    return init_fn, step_fn


def make_change_point_detector_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with change-point detection for regime switches."""
    step_size = hp.get("step_size", 0.01)
    sensitivity = hp.get("sensitivity", 0.1)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "loss_ema": 0.5,
            "last_loss": 0.5,
            "regime": 0,
        }

    def step_fn(params, state, x, y, grads):
        # Current loss estimate
        current_loss = jnp.mean(jnp.abs(grads))

        # Exponential moving average
        loss_ema_new = 0.95 * state["loss_ema"] + 0.05 * current_loss

        # Detect change point: sudden loss spike
        loss_change = jnp.abs(loss_ema_new - state["last_loss"])
        is_changepoint = loss_change > sensitivity

        # Switch regime if changepoint detected
        regime_new = state["regime"] + int(is_changepoint)

        # Reduce update magnitude on changepoint (for stability)
        update_scale = jnp.where(is_changepoint, 0.5, 1.0)

        params_new = {
            "w": params["w"] - update_scale * step_size * grads,
            "b": params["b"] - update_scale * step_size * jnp.mean(grads),
        }

        state_new = {
            "loss_ema": loss_ema_new,
            "last_loss": state["loss_ema"],
            "regime": regime_new,
        }

        return params_new, state_new, (0.88, current_loss, update_scale * step_size)

    return init_fn, step_fn


def make_temporal_consistency_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner enforcing temporal consistency."""
    step_size = hp.get("step_size", 0.01)
    consistency_weight = hp.get("consistency_weight", 0.1)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "prev_w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
        }

    def step_fn(params, state, x, y, grads):
        # Consistency penalty: discourage abrupt changes
        change_magnitude = jnp.linalg.norm(params["w"] - state["prev_w"])
        consistency_penalty = consistency_weight * change_magnitude * params["w"]

        # Combined update
        total_update = grads + consistency_penalty

        params_new = {
            "w": params["w"] - step_size * total_update,
            "b": params["b"] - step_size * jnp.mean(total_update),
        }

        state_new = {
            "prev_w": params["w"],
        }

        return params_new, state_new, (0.88, 0.0, step_size)

    return init_fn, step_fn


def make_adaptive_momentum_scr_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with adaptive momentum for slow drift."""
    base_lr = hp.get("base_lr", 0.01)
    momentum_init = hp.get("momentum_init", 0.9)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "v": jnp.zeros((feature_dim, 1)),
            "momentum": momentum_init,
        }

    def step_fn(params, state, x, y, grads):
        # Adapt momentum based on gradient consistency
        grad_mag = jnp.mean(jnp.abs(grads))

        # High consistency = increase momentum
        momentum_new = state["momentum"] + 0.01 * (grad_mag - 0.5)
        momentum_clipped = jnp.clip(momentum_new, 0.5, 0.99)

        # Momentum update
        v_new = momentum_clipped * state["v"] - base_lr * grads

        params_new = {
            "w": params["w"] + v_new,
            "b": params["b"] + jnp.mean(v_new),
        }

        state_new = {
            "v": v_new,
            "momentum": momentum_clipped,
        }

        return params_new, state_new, (0.88, 0.0, base_lr)

    return init_fn, step_fn


SCR_DOMAIN_SPECIALISTS = {
    "drift_aware": make_drift_aware_learner,
    "change_point_detector": make_change_point_detector_learner,
    "temporal_consistency": make_temporal_consistency_learner,
    "adaptive_momentum_scr": make_adaptive_momentum_scr_learner,
}


def register_scr_domain_specialists():
    """Register SCR domain-specific specialists."""
    print(f"[OK] Registered {len(SCR_DOMAIN_SPECIALISTS)} SCR domain specialists")
    return SCR_DOMAIN_SPECIALISTS
