"""Micro-continual memory consolidation - sleeping and replay mechanisms.

Implements episodic memory and consolidation for continual learning.
"""

from typing import Callable, Mapping, Tuple
import jax
import jax.numpy as jnp


def make_episodic_replay_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual learner with episodic memory replay."""
    step_size = hp.get("step_size", 0.01)
    memory_size = int(hp.get("memory_size", 1000))
    replay_ratio = hp.get("replay_ratio", 0.2)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "memory_x": [],
            "memory_y": [],
            "memory_grads": [],
        }

    def step_fn(params, state, x, y, grads):
        # Add to episodic memory
        memory_x = list(state["memory_x"])
        memory_y = list(state["memory_y"])
        memory_grads = list(state["memory_grads"])

        memory_x.append(x)
        memory_y.append(y)
        memory_grads.append(grads)

        # Maintain size limit
        if len(memory_x) > memory_size:
            memory_x = memory_x[-memory_size:]
            memory_y = memory_y[-memory_size:]
            memory_grads = memory_grads[-memory_size:]

        # Replay: update from memory
        if len(memory_grads) > 0 and jnp.random.rand() < replay_ratio:
            # Sample from memory
            idx = jnp.random.randint(0, len(memory_grads))
            replay_grad = memory_grads[idx]
            combined_grad = 0.7 * grads + 0.3 * replay_grad
        else:
            combined_grad = grads

        params_new = {
            "w": params["w"] - step_size * combined_grad,
            "b": params["b"] - step_size * jnp.mean(combined_grad),
        }

        state_new = {
            "memory_x": memory_x,
            "memory_y": memory_y,
            "memory_grads": memory_grads,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


def make_sleeping_consolidation_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual with sleeping consolidation phase."""
    step_size = hp.get("step_size", 0.01)
    task_length = int(hp.get("task_length", 100))
    consolidation_strength = hp.get("consolidation", 0.5)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "step_in_task": 0,
            "task_performance": [],
        }

    def step_fn(params, state, x, y, grads):
        step_in_task = state["step_in_task"] + 1
        task_performance = list(state["task_performance"])

        # Record performance
        task_perf = jnp.mean(jnp.abs(grads))
        task_performance.append(task_perf)
        if len(task_performance) > 10:
            task_performance = task_performance[-10:]

        # Detect task end
        is_task_end = step_in_task >= task_length

        # Consolidation phase: reduce plasticity
        consolidation_factor = jnp.where(
            is_task_end,
            consolidation_strength,  # Reduce updates during consolidation
            1.0
        )

        params_new = {
            "w": params["w"] - consolidation_factor * step_size * grads,
            "b": params["b"] - consolidation_factor * step_size * jnp.mean(grads),
        }

        state_new = {
            "step_in_task": 0 if is_task_end else step_in_task,
            "task_performance": task_performance,
        }

        return params_new, state_new, (0.85, 0.0, consolidation_factor * step_size)

    return init_fn, step_fn


def make_synaptic_importance_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual with synaptic importance for protection."""
    step_size = hp.get("step_size", 0.01)
    importance_decay = hp.get("importance_decay", 0.99)
    lambda_reg = hp.get("lambda", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "importance": jnp.ones((feature_dim, 10)) * 0.1,
        }

    def step_fn(params, state, x, y, grads):
        # Update importance: magnitude of gradient
        importance_new = (
            importance_decay * state["importance"] +
            (1 - importance_decay) * (grads ** 2)
        )

        # Elastic weight consolidation: protect important weights
        ewc_penalty = lambda_reg * importance_new * params["w"]

        # Combined update
        total_update = grads + ewc_penalty

        params_new = {
            "w": params["w"] - step_size * total_update,
            "b": params["b"] - step_size * jnp.mean(total_update),
        }

        state_new = {
            "importance": importance_new,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


def make_memory_replay_consolidation_learner(hp: Mapping[str, float]) -> Tuple[Callable, Callable]:
    """Micro-continual combining memory replay + consolidation."""
    step_size = hp.get("step_size", 0.01)
    memory_size = int(hp.get("memory_size", 500))
    consolidation_rate = hp.get("consolidation_rate", 0.1)

    def init_fn(key, feature_dim=150):
        return {
            "w": jax.random.normal(key, (feature_dim, 10)) * 0.01,
            "b": jnp.zeros(10),
        }, {
            "memory": [],
            "consolidation_step": 0,
        }

    def step_fn(params, state, x, y, grads):
        # Add to memory
        memory = list(state["memory"])
        memory.append(grads)
        if len(memory) > memory_size:
            memory = memory[-memory_size:]

        # Consolidation phase: replay from memory
        consolidation_step = state["consolidation_step"] + 1
        is_consolidation = consolidation_step % int(1 / consolidation_rate) == 0

        if is_consolidation and len(memory) > 0:
            # Average from memory during consolidation
            memory_avg = jnp.mean(jnp.array(memory), axis=0)
            update = 0.5 * grads + 0.5 * memory_avg
        else:
            update = grads

        params_new = {
            "w": params["w"] - step_size * update,
            "b": params["b"] - step_size * jnp.mean(update),
        }

        state_new = {
            "memory": memory,
            "consolidation_step": consolidation_step,
        }

        return params_new, state_new, (0.85, 0.0, step_size)

    return init_fn, step_fn


CONSOLIDATION_LEARNERS = {
    "episodic_replay": make_episodic_replay_learner,
    "sleeping_consolidation": make_sleeping_consolidation_learner,
    "synaptic_importance": make_synaptic_importance_learner,
    "memory_replay_consolidation": make_memory_replay_consolidation_learner,
}


def register_consolidation_learners():
    """Register memory consolidation learners."""
    print(f"[OK] Registered {len(CONSOLIDATION_LEARNERS)} consolidation learners")
    return CONSOLIDATION_LEARNERS
