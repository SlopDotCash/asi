"""IPMNIST advanced mechanism variants - final expansion.

Additional high-value IPMNIST arms for complete sensitivity coverage.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_exponential_moving_average_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """UPGD with exponential moving average for smoothing."""
    step_size = hp.get("step_size", 0.01)
    ema_decay = hp.get("ema_decay", 0.9)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {"ema_w": jnp.zeros((feature_dim, 10)), "ema_b": jnp.zeros(10)}

    def step_fn(params, state, x, y, grads):
        ema_w = ema_decay * state["ema_w"] + (1 - ema_decay) * grads
        ema_b = ema_decay * state["ema_b"] + (1 - ema_decay) * jnp.mean(grads)

        params_new = {
            "w": params["w"] - step_size * ema_w,
            "b": params["b"] - step_size * ema_b,
        }
        state_new = {"ema_w": ema_w, "ema_b": ema_b}
        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_second_order_momentum_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """UPGD with second-order momentum (acceleration)."""
    step_size = hp.get("step_size", 0.01)
    momentum = hp.get("momentum", 0.9)
    damping = hp.get("damping", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "v": jnp.zeros((feature_dim, 10)),
            "a": jnp.zeros((feature_dim, 10)),
        }

    def step_fn(params, state, x, y, grads):
        v_new = momentum * state["v"] - step_size * grads
        a_new = damping * state["a"] + v_new

        params_new = {
            "w": params["w"] + a_new,
            "b": params["b"] + jnp.mean(a_new),
        }
        state_new = {"v": v_new, "a": a_new}
        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_adaptive_lr_schedule_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """UPGD with adaptive learning rate schedule."""
    base_lr = hp.get("base_lr", 0.01)
    schedule_type = hp.get("schedule", "cosine")

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {"step": 0}

    def step_fn(params, state, x, y, grads):
        step = state["step"] + 1

        if schedule_type == "cosine":
            lr = base_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / 1000))
        elif schedule_type == "exponential":
            lr = base_lr * jnp.exp(-step / 1000)
        else:
            lr = base_lr / (1 + step / 100)

        params_new = {
            "w": params["w"] - lr * grads,
            "b": params["b"] - lr * jnp.mean(grads),
        }
        state_new = {"step": step}
        return params_new, state_new, (0.0, 0.0, lr)

    return init_fn, step_fn


def make_gradient_clipping_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """UPGD with gradient clipping for stability."""
    step_size = hp.get("step_size", 0.01)
    clip_threshold = hp.get("clip_threshold", 1.0)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {}

    def step_fn(params, state, x, y, grads):
        grad_norm = jnp.linalg.norm(grads)
        clipped_grads = grads * jnp.minimum(1.0, clip_threshold / (grad_norm + 1e-8))

        params_new = {
            "w": params["w"] - step_size * clipped_grads,
            "b": params["b"] - step_size * jnp.mean(clipped_grads),
        }
        return params_new, state, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_lookahead_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """UPGD with Lookahead meta-optimizer."""
    inner_step = hp.get("inner_step", 0.01)
    outer_step = hp.get("outer_step", 0.001)
    lookahead_k = int(hp.get("lookahead_k", 5))

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {"fast_w": jnp.zeros((feature_dim, 10)), "step_count": 0}

    def step_fn(params, state, x, y, grads):
        fast_w_new = state["fast_w"] - inner_step * grads
        step_count = state["step_count"] + 1

        if step_count % lookahead_k == 0:
            w_new = params["w"] + outer_step * (fast_w_new - params["w"])
        else:
            w_new = params["w"]

        params_new = {
            "w": w_new,
            "b": params["b"] - inner_step * jnp.mean(grads),
        }
        state_new = {"fast_w": fast_w_new, "step_count": step_count}
        return params_new, state_new, (0.0, 0.0, inner_step)

    return init_fn, step_fn


IPMNIST_ADVANCED_VARIANTS = {
    "ema_smoothing": make_exponential_moving_average_learner,
    "second_order_momentum": make_second_order_momentum_learner,
    "adaptive_schedule": make_adaptive_lr_schedule_learner,
    "gradient_clipping": make_gradient_clipping_learner,
    "lookahead": make_lookahead_learner,
}


def register_ipmnist_advanced_variants():
    """Register all advanced IPMNIST variants."""
    print(f"[OK] Registered {len(IPMNIST_ADVANCED_VARIANTS)} IPMNIST advanced variants")
    return IPMNIST_ADVANCED_VARIANTS
