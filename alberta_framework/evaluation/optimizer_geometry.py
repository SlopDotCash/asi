"""Small-matrix primitives for staging continual optimizer geometry controls."""

from __future__ import annotations

from types import MappingProxyType

import jax.numpy as jnp
from jax import Array

GEOMETRY_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimizer-geometry.protocol.v1",
        "paper_revisions": (
            "arXiv:2605.08949v2",
            "arXiv:2606.10406v1",
            "arXiv:2601.07636v1",
        ),
        "stage": "small_streaming_matrix_pre_ipmnist",
        "protocol_difference": "equation primitives only; no LLM or batch-CL claim",
        "mechanism_off": "empty_basis_or_zero_gradient_exact_reduction",
        "matched_axes": ("seed", "updates", "observations"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def orthogonal_correction(update: Array, protected_basis: Array) -> Array:
    """Project a vector away from row-wise orthonormal protected directions."""
    vector = jnp.asarray(update)
    basis = jnp.asarray(protected_basis)
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[1] != vector.shape[0]:
        raise ValueError("update must be a vector and basis rows must match its width")
    return vector - basis.T @ (basis @ vector)


def spectral_matrix_sign(matrix: Array, *, steps: int = 5) -> Array:
    """Muon-style Newton--Schulz matrix-sign approximation for a small matrix."""
    value = jnp.asarray(matrix)
    if (
        value.ndim != 2
        or value.size == 0
        or not jnp.issubdtype(value.dtype, jnp.floating)
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError("matrix must be non-empty and steps a positive integer")
    x = value / jnp.maximum(jnp.linalg.norm(value), jnp.asarray(1e-12, dtype=value.dtype))
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x - 4.7750 * a @ x + 2.0315 * a @ a @ x
    return x.T if transposed else x


def flad_noise_component(perturbation: Array, gradient: Array) -> Array:
    """Remove FLAD's gradient-aligned perturbation component."""
    delta = jnp.asarray(perturbation)
    direction = jnp.asarray(gradient)
    if delta.shape != direction.shape or delta.ndim != 1:
        raise ValueError("perturbation and gradient must be equal-width vectors")
    squared_norm = jnp.vdot(direction, direction).real
    projection = jnp.where(
        squared_norm > 0.0,
        direction * (jnp.vdot(direction, delta).real / squared_norm),
        jnp.zeros_like(delta),
    )
    return delta - projection
