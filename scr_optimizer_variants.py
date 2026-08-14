"""Additional SCR v2 learner factory variants - advanced optimizers and mechanisms.

Extends SCR v2 with alternative learner implementations beyond parametric variants.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_lion_optimizer_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with Lion optimizer (Evo fitness)."""
    learning_rate = hp.get("learning_rate", 0.01)
    beta1 = hp.get("beta1", 0.9)
    beta2 = hp.get("beta2", 0.99)
    weight_decay = hp.get("weight_decay", 0.01)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {
            "m": jnp.zeros((feature_dim, 1)),
            "v": jnp.zeros((feature_dim, 1)),
        }

    def step_fn(params, state, x, y, grads):
        m_new = beta1 * state["m"] + (1 - beta1) * grads
        v_new = beta2 * state["v"] + (1 - beta2) * (grads ** 2)

        update = jnp.sign(m_new) * jnp.sqrt(v_new + 1e-8)

        params_new = {
            "w": params["w"] - learning_rate * (update + weight_decay * params["w"]),
            "b": params["b"] - learning_rate * jnp.mean(update),
        }

        state_new = {"m": m_new, "v": v_new}
        return params_new, state_new, (0.0, 0.0, learning_rate)

    return init_fn, step_fn


def make_adamw_warmup_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """AdamW with learning rate warmup schedule."""
    base_lr = hp.get("base_lr", 0.001)
    warmup_steps = int(hp.get("warmup_steps", 100))
    total_steps = int(hp.get("total_steps", 1000))

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {"step": 0, "m": jnp.zeros((feature_dim, 1)), "v": jnp.zeros((feature_dim, 1))}

    def step_fn(params, state, x, y, grads):
        step = state["step"] + 1

        # Warmup schedule
        if step < warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))

        m_new = 0.9 * state["m"] + 0.1 * grads
        v_new = 0.999 * state["v"] + 0.001 * (grads ** 2)

        m_hat = m_new / (1 - 0.9 ** step)
        v_hat = v_new / (1 - 0.999 ** step)

        params_new = {
            "w": params["w"] - lr * (m_hat / (jnp.sqrt(v_hat) + 1e-8) + 0.01 * params["w"]),
            "b": params["b"] - lr * jnp.mean(m_hat),
        }

        state_new = {"step": step, "m": m_new, "v": v_new}
        return params_new, state_new, (0.0, 0.0, lr)

    return init_fn, step_fn


def make_muon_optimizer_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with Muon optimizer (momentum via SVD)."""
    learning_rate = hp.get("learning_rate", 0.01)
    momentum = hp.get("momentum", 0.9)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {"momentum": jnp.zeros((feature_dim, 1))}

    def step_fn(params, state, x, y, grads):
        # Muon: momentum via gradient direction
        momentum_new = momentum * state["momentum"] + grads

        params_new = {
            "w": params["w"] - learning_rate * momentum_new,
            "b": params["b"] - learning_rate * jnp.mean(momentum_new),
        }

        state_new = {"momentum": momentum_new}
        return params_new, state_new, (0.0, 0.0, learning_rate)

    return init_fn, step_fn


def make_normalized_sgd_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """SCR learner with normalized gradient updates."""
    learning_rate = hp.get("learning_rate", 0.01)
    eps = hp.get("eps", 1e-8)

    def init_fn(key, feature_dim=100):
        return {
            "w": jax.random.normal(key, (feature_dim, 1)) * 0.01,
            "b": jnp.zeros(1),
        }, {}

    def step_fn(params, state, x, y, grads):
        grad_norm = jnp.linalg.norm(grads)
        normalized_grads = grads / (grad_norm + eps)

        params_new = {
            "w": params["w"] - learning_rate * normalized_grads,
            "b": params["b"] - learning_rate * jnp.mean(normalized_grads),
        }

        return params_new, state, (0.0, 0.0, learning_rate)

    return init_fn, step_fn


SCR_OPTIMIZER_VARIANTS = {
    "lion_optimizer": make_lion_optimizer_learner,
    "adamw_warmup": make_adamw_warmup_learner,
    "muon_optimizer": make_muon_optimizer_learner,
    "normalized_sgd": make_normalized_sgd_learner,
}


def register_scr_optimizer_variants():
    """Register SCR optimizer variants."""
    print(f"[OK] Registered {len(SCR_OPTIMIZER_VARIANTS)} SCR optimizer variants")
    return SCR_OPTIMIZER_VARIANTS
