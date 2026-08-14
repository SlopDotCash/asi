"""Final measurement proposals - advanced ensemble and meta variants.

Creates final high-value arm implementations for all domains.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_snapshot_ensemble_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that maintains ensemble of weight snapshots."""
    step_size = hp.get("step_size", 0.01)
    snapshot_interval = int(hp.get("snapshot_interval", 100))

    def init_fn(key, feature_dim=150):
        w_init = jax.random.normal(key, (feature_dim, 10)) * 0.01
        return {
            "w": w_init,
            "b": jnp.zeros(10),
        }, {
            "snapshots": [w_init],
            "step_count": 0,
        }

    def step_fn(params, state, x, y, grads):
        step_count_new = state["step_count"] + 1

        # Take snapshot periodically
        snapshots = list(state["snapshots"])
        if step_count_new % snapshot_interval == 0 and len(snapshots) < 10:
            snapshots.append(params["w"].copy())

        # Ensemble prediction: average of snapshots
        if snapshots:
            ensemble_w = jnp.mean(jnp.array(snapshots), axis=0)
        else:
            ensemble_w = params["w"]

        params_new = {
            "w": params["w"] - step_size * grads,
            "b": params["b"] - step_size * jnp.mean(grads),
        }

        state_new = {
            "snapshots": snapshots,
            "step_count": step_count_new,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


def make_progressive_regularization_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner with progressive regularization increase."""
    step_size = hp.get("step_size", 0.01)
    reg_init = hp.get("reg_init", 0.001)
    reg_growth = hp.get("reg_growth", 1.001)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "reg_strength": reg_init,
        }

    def step_fn(params, state, x, y, grads):
        # Increase regularization over time
        reg_new = state["reg_strength"] * reg_growth

        # L2 penalty
        l2_penalty = reg_new * params["w"]

        params_new = {
            "w": params["w"] - step_size * (grads + l2_penalty),
            "b": params["b"] - step_size * jnp.mean(grads),
        }

        state_new = {
            "reg_strength": reg_new,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


def make_uncertainty_weighted_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner that estimates and uses uncertainty for weighting."""
    step_size = hp.get("step_size", 0.01)
    uncertainty_scale = hp.get("uncertainty_scale", 0.5)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "uncertainty_ema": 0.5,
        }

    def step_fn(params, state, x, y, grads):
        # Estimate uncertainty from gradient variance
        grad_mag = jnp.linalg.norm(grads)
        uncertainty_new = 0.95 * state["uncertainty_ema"] + 0.05 * grad_mag

        # Weight: low uncertainty = high weight
        weight = 1.0 / (1.0 + uncertainty_scale * uncertainty_new)

        params_new = {
            "w": params["w"] - weight * step_size * grads,
            "b": params["b"] - weight * step_size * jnp.mean(grads),
        }

        state_new = {
            "uncertainty_ema": uncertainty_new,
        }

        return params_new, state_new, (0.85, uncertainty_new, weight * step_size)

    return init_fn, step_fn


def make_curriculum_learning_rate_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Learner with curriculum that gradually increases learning rate."""
    initial_lr = hp.get("initial_lr", 0.001)
    max_lr = hp.get("max_lr", 0.1)
    curriculum_steps = int(hp.get("curriculum_steps", 1000))

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "step": 0,
        }

    def step_fn(params, state, x, y, grads):
        step = state["step"] + 1

        # Curriculum: linearly increase from initial to max
        progress = jnp.minimum(step / curriculum_steps, 1.0)
        lr = initial_lr + (max_lr - initial_lr) * progress

        params_new = {
            "w": params["w"] - lr * grads,
            "b": params["b"] - lr * jnp.mean(grads),
        }

        state_new = {
            "step": step,
        }

        return params_new, state_new, (0.85, 0.0, lr)

    return init_fn, step_fn


FINAL_MEASUREMENT_PROPOSALS = {
    "snapshot_ensemble": make_snapshot_ensemble_learner,
    "progressive_regularization": make_progressive_regularization_learner,
    "uncertainty_weighted": make_uncertainty_weighted_learner,
    "curriculum_learning_rate": make_curriculum_learning_rate_learner,
}


def register_final_proposals():
    """Register final measurement proposals."""
    print(f"[OK] Registered {len(FINAL_MEASUREMENT_PROPOSALS)} final measurement proposals")
    return FINAL_MEASUREMENT_PROPOSALS
