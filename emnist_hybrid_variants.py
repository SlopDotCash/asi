"""EMNIST v3 learner composition and hybrid variants.

Implements hybrid learner combinations for EMNIST.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_cbp_l2init_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """EMNIST learner combining CBP + L2-init protection."""
    step_size = hp.get("step_size", 0.01)
    cbp_ratio = hp.get("cbp_ratio", 0.6)
    l2init_decay = hp.get("l2init_decay", 0.05)

    def init_fn(key, feature_dim=784):
        w_init = jax.random.normal(key, (feature_dim, 47)) * 0.01
        return {
            "w": w_init,
            "b": jnp.zeros(47),
            "w_init": w_init,
        }, {"cbp_buffer": []}

    def step_fn(params, state, x, y, grads):
        # CBP: composition via buffer
        buffer = list(state["cbp_buffer"])
        buffer.append(grads)
        if len(buffer) > 10:
            buffer = buffer[-10:]

        # Mix: current gradient + buffer average
        if buffer:
            avg_buffer = jnp.mean(jnp.array(buffer), axis=0)
            mixed_grads = cbp_ratio * grads + (1 - cbp_ratio) * avg_buffer
        else:
            mixed_grads = grads

        # L2-init: pull towards initialization
        l2init_penalty = l2init_decay * (params["w"] - params["w_init"])

        # Combined update
        params_new = {
            "w": params["w"] - step_size * (mixed_grads + l2init_penalty),
            "b": params["b"] - step_size * jnp.mean(mixed_grads),
            "w_init": params["w_init"],
        }

        state_new = {"cbp_buffer": buffer}
        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_shiftnorm_cbp_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """EMNIST learner combining shift-norm + CBP."""
    step_size = hp.get("step_size", 0.01)
    shift_threshold = hp.get("shift_threshold", 0.3)
    cbp_ratio = hp.get("cbp_ratio", 0.6)

    def init_fn(key, feature_dim=784):
        return {
            "w": jax.random.normal(key, (feature_dim, 47)) * 0.01,
            "b": jnp.zeros(47),
        }, {
            "norm_mean": 0.0,
            "norm_var": 1.0,
            "cbp_buffer": [],
            "shift_detected": False,
        }

    def step_fn(params, state, x, y, grads):
        # Shift detection
        norm_mean_new = 0.9 * state["norm_mean"] + 0.1 * jnp.mean(grads)
        norm_var_new = 0.9 * state["norm_var"] + 0.1 * jnp.var(grads)

        shift = jnp.abs(norm_mean_new - state["norm_mean"]) > shift_threshold
        shift_detected = bool(shift)

        # Normalize
        normalized = grads / (jnp.sqrt(norm_var_new) + 1e-8)

        # CBP composition
        buffer = list(state["cbp_buffer"])
        buffer.append(normalized)
        if len(buffer) > 10:
            buffer = buffer[-10:]

        if buffer:
            avg_buffer = jnp.mean(jnp.array(buffer), axis=0)
            mixed = cbp_ratio * normalized + (1 - cbp_ratio) * avg_buffer
        else:
            mixed = normalized

        # If shift detected, use buffer more (more composition)
        if shift_detected:
            mixed = 0.3 * normalized + 0.7 * avg_buffer if buffer else mixed

        params_new = {
            "w": params["w"] - step_size * mixed,
            "b": params["b"] - step_size * jnp.mean(mixed),
        }

        state_new = {
            "norm_mean": norm_mean_new,
            "norm_var": norm_var_new,
            "cbp_buffer": buffer,
            "shift_detected": shift_detected,
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_adversarial_cbp_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """EMNIST learner combining adversarial training + CBP."""
    step_size = hp.get("step_size", 0.01)
    epsilon = hp.get("epsilon", 0.1)
    cbp_ratio = hp.get("cbp_ratio", 0.5)

    def init_fn(key, feature_dim=784):
        return {
            "w": jax.random.normal(key, (feature_dim, 47)) * 0.01,
            "b": jnp.zeros(47),
        }, {"cbp_buffer": []}

    def step_fn(params, state, x, y, grads):
        # Adversarial perturbation
        adv_grads = grads + epsilon * jnp.sign(grads + 1e-8)

        # Mix adversarial + clean
        mixed_adv = 0.5 * grads + 0.5 * adv_grads

        # CBP composition
        buffer = list(state["cbp_buffer"])
        buffer.append(mixed_adv)
        if len(buffer) > 10:
            buffer = buffer[-10:]

        if buffer:
            avg_buffer = jnp.mean(jnp.array(buffer), axis=0)
            final = cbp_ratio * mixed_adv + (1 - cbp_ratio) * avg_buffer
        else:
            final = mixed_adv

        params_new = {
            "w": params["w"] - step_size * final,
            "b": params["b"] - step_size * jnp.mean(final),
        }

        state_new = {"cbp_buffer": buffer}
        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


def make_ensemble_protection_hybrid_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """EMNIST learner with ensemble of protection mechanisms."""
    step_size = hp.get("step_size", 0.01)
    n_protections = int(hp.get("n_protections", 3))

    def init_fn(key, feature_dim=784):
        return {
            "w": jax.random.normal(key, (feature_dim, 47)) * 0.01,
            "b": jnp.zeros(47),
        }, {
            "buffers": [[] for _ in range(n_protections)],
            "l2init_w": jax.random.normal(key, (feature_dim, 47)) * 0.01,
        }

    def step_fn(params, state, x, y, grads):
        updates = []

        # Protection 1: L2-init
        l2init_update = grads + 0.01 * (params["w"] - state["l2init_w"])
        updates.append(l2init_update)

        # Protection 2: Buffer averaging
        buf1 = list(state["buffers"][0])
        buf1.append(grads)
        if len(buf1) > 5:
            buf1 = buf1[-5:]
        buffer_update = jnp.mean(jnp.array(buf1) if buf1 else jnp.expand_dims(grads, 0), axis=0)
        updates.append(buffer_update)

        # Protection 3: Clipping
        clipped_update = jnp.clip(grads, -1.0, 1.0)
        updates.append(clipped_update)

        # Ensemble: average all protections
        ensemble_update = jnp.mean(jnp.array(updates), axis=0)

        params_new = {
            "w": params["w"] - step_size * ensemble_update,
            "b": params["b"] - step_size * jnp.mean(ensemble_update),
        }

        buffers_new = [list(state["buffers"][0])] + [[] for _ in range(n_protections - 1)]
        buffers_new[0] = buf1

        state_new = {
            "buffers": buffers_new,
            "l2init_w": state["l2init_w"],
        }

        return params_new, state_new, (0.0, 0.0, step_size)

    return init_fn, step_fn


EMNIST_HYBRID_VARIANTS = {
    "cbp_l2init_hybrid": make_cbp_l2init_hybrid_learner,
    "shiftnorm_cbp_hybrid": make_shiftnorm_cbp_hybrid_learner,
    "adversarial_cbp_hybrid": make_adversarial_cbp_hybrid_learner,
    "ensemble_protection": make_ensemble_protection_hybrid_learner,
}


def register_emnist_hybrid_variants():
    """Register EMNIST hybrid variants."""
    print(f"[OK] Registered {len(EMNIST_HYBRID_VARIANTS)} EMNIST hybrid variants")
    return EMNIST_HYBRID_VARIANTS
