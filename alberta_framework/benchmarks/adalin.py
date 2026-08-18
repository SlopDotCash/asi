"""AdaLin equation primitives and non-comparable PMNIST protocol declaration."""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import Array

ADALIN_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.adalin.protocol.v1",
        "paper_revision": "arXiv:2505.09486v1",
        "paper_revision_date": "2025-05-14",
        "paper_pmnist_tasks": 400,
        "paper_examples_per_task": 10_000,
        "paper_batch_size": 16,
        "paper_hidden_widths": (100, 100),
        "asi_target_tasks": 200,
        "asi_examples_per_task": 5_000,
        "asi_batch_size": 1,
        "asi_hidden_widths": (300, 150),
        "learner_observes_task_boundary": False,
        "mechanism_off": "alpha_zero_exact_base_activation",
        "matched_axes": ("seed", "updates", "observations"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _adalin(x: Array, alpha: Array, *, activation: Array, derivative: Array) -> Array:
    gate = jax.lax.stop_gradient(jnp.cos(0.5 * jnp.pi * jnp.abs(derivative)))
    return activation + alpha * x * gate


def adalin_relu(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for ReLU (algebraically PReLU)."""
    value = jnp.asarray(x)
    coefficient = jnp.asarray(alpha)
    return _adalin(
        value,
        coefficient,
        activation=jax.nn.relu(value),
        derivative=(value > 0).astype(value.dtype),
    )


def adalin_tanh(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for tanh, whose Lipschitz constant is one."""
    value = jnp.asarray(x)
    coefficient = jnp.asarray(alpha)
    activation = jnp.tanh(value)
    return _adalin(
        value,
        coefficient,
        activation=activation,
        derivative=1.0 - activation**2,
    )
