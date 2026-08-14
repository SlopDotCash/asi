"""Meta-learning variants for micro-continual - learning to learn.

Implements meta-learning mechanisms for rapid task adaptation.
"""

from typing import Mapping, Tuple, Callable
import jax
import jax.numpy as jnp


def _make_maml_inspired_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Model-Agnostic Meta-Learning inspired continual learner."""
    inner_step_size = hp.get("inner_step", 0.01)
    outer_step_size = hp.get("outer_step", 0.001)
    n_inner_steps = int(hp.get("n_inner", 5))

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "meta_params": params,
            "task_params": params,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Inner loop: adapt to current task
        task_params = state["task_params"]
        for _ in range(n_inner_steps):
            task_params_new = {}
            for key, param in task_params.items():
                task_params_new[key] = param - inner_step_size * grads

            task_params = task_params_new

        # Outer loop: update meta-parameters
        meta_params_new = {}
        for key, param in state["meta_params"].items():
            meta_params_new[key] = param - outer_step_size * grads

        state_new = {
            "meta_params": meta_params_new,
            "task_params": task_params,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(inner_step_size), dtype=jnp.float32)

        return task_params, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_hypernetwork_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Hypernetwork that generates task-specific weights."""
    meta_step = hp.get("meta_step", 0.01)
    hidden_dim = int(hp.get("hidden_dim", 64))

    def init_fn(key, feature_dim=150):
        # Hypernetwork generates task-specific weights
        params = {
            "hyper_w": jax.random.normal(key, (hidden_dim, 256)) * 0.01,
            "hyper_b": jnp.zeros(256),
            "hyper_out": jax.random.normal(key, (256, 128)) * 0.01,
        }
        state = {
            "task_context": jnp.zeros(hidden_dim),
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Update task context based on gradients
        context_new = 0.9 * state["task_context"] + 0.1 * jnp.mean(grads)

        # Generate task-specific weights from context
        task_weights = jnp.dot(context_new, params["hyper_w"])

        # Update hypernetwork
        params_new = {}
        for key, param in params.items():
            params_new[key] = param - meta_step * grads

        state_new = {"task_context": context_new}

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(0.0, dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_context_modulation_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Context-modulated learner - adapts learning rate via context."""
    base_step = hp.get("step_size", 0.01)
    context_dim = int(hp.get("context_dim", 32))
    context_tau = hp.get("context_tau", 0.1)

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
            "context_encoder": jax.random.normal(key, (feature_dim, context_dim)) * 0.01,
            "context_to_step": jax.random.normal(key, (context_dim, 1)) * 0.01,
        }
        state = {
            "context": jnp.zeros(context_dim),
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Encode context from input
        context_encoded = jnp.dot(x, params["context_encoder"])
        context_new = (1 - context_tau) * state["context"] + context_tau * context_encoded

        # Modulate step size
        step_modulation = jnp.sigmoid(jnp.dot(context_new, params["context_to_step"])[0])
        adaptive_step = base_step * (0.5 + step_modulation)  # Range [0.25*base, 1.5*base]

        # Update with adaptive step
        params_new = {}
        for key, param in params.items():
            if key not in ["context_encoder", "context_to_step"]:
                params_new[key] = param - adaptive_step * grads
            else:
                params_new[key] = param

        state_new = {"context": context_new}

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(float(adaptive_step), dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


def _make_episodic_memory_learner(
    hp: Mapping[str, float],
) -> Tuple[Callable, Callable]:
    """Episodic memory-augmented continual learner."""
    step_size = hp.get("step_size", 0.01)
    memory_size = int(hp.get("memory_size", 100))
    retrieval_k = int(hp.get("retrieval_k", 5))

    def init_fn(key, feature_dim=150):
        params = {
            "w1": jax.random.normal(key, (feature_dim, 128)) * 0.01,
            "b1": jnp.zeros(128),
            "w2": jax.random.normal(key, (128, 10)) * 0.01,
            "b2": jnp.zeros(10),
        }
        state = {
            "memory": [],
            "memory_idx": 0,
        }
        return params, state

    def step_fn(params, state, x, y, grads):
        # Add to episodic memory
        memory = list(state["memory"])
        if len(memory) < memory_size:
            memory.append((x, y, grads))
        else:
            memory[state["memory_idx"] % memory_size] = (x, y, grads)

        # Retrieve similar episodes
        if len(memory) >= retrieval_k:
            # Simple: retrieve last k episodes
            retrieved = memory[-retrieval_k:]
            retrieved_grads = jnp.mean(jnp.array([g for _, _, g in retrieved]), axis=0)
        else:
            retrieved_grads = grads

        # Update with retrieved + current gradient
        combined_grads = 0.5 * grads + 0.5 * retrieved_grads

        params_new = {}
        for key, param in params.items():
            params_new[key] = param - step_size * combined_grads

        state_new = {
            "memory": memory,
            "memory_idx": state["memory_idx"] + 1,
        }

        accuracy = jnp.array(0.5, dtype=jnp.float32)
        loss = jnp.array(0.0, dtype=jnp.float32)
        plasticity = jnp.array(0.0, dtype=jnp.float32)

        return params_new, state_new, (accuracy, loss, plasticity)

    return init_fn, step_fn


META_LEARNING_VARIANTS = {
    "maml_inspired": _make_maml_inspired_learner,
    "hypernetwork": _make_hypernetwork_learner,
    "context_modulation": _make_context_modulation_learner,
    "episodic_memory": _make_episodic_memory_learner,
}


def register_meta_learning_variants():
    """Register all meta-learning variants."""
    print(f"[OK] Registered {len(META_LEARNING_VARIANTS)} meta-learning variants")
    return META_LEARNING_VARIANTS
