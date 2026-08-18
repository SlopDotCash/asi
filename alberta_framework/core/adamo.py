"""AdamO: Adam with decoupled pseudo-orthogonality updates.

This implements equations 16, 19, and 20 of Rosseau, Müller, and Nowé,
arXiv:2606.09762v1. Adam moments observe only the task gradient; the Gram
penalty gradient is applied as a separate descent step. Bias/vector leaves
are intentionally not regularized.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.baseline_optimizers import Adam, AdamParamState
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    select_transaction,
)

ADAMO_PAPER_REVISION = "arXiv:2606.09762v1"
ADAMO_PROTOCOL = MappingProxyType({
    "paper_revision": ADAMO_PAPER_REVISION,
    "paper_url": "https://arxiv.org/abs/2606.09762v1",
    "equations": (16, 19, 20),
    "official_code_available": False,
    "official_code_search_date": "2026-08-17",
    "paper_mlp_architecture": "depth-4 width-512 ReLU",
    "adapter_architecture": "IPMNIST protocol 784-300-150-10 ReLU MLP",
    "paper_initialization": "orthogonal weights",
    "adapter_initialization": "IPMNIST protocol PyTorch-default Linear initialization",
    "paper_mlp_step_size": 1e-4,
    "adapter_step_size": 1e-4,
    "bias_regularization": False,
    "convolution_kernel_adapter": "not_implemented",
    "timing_is_telemetry_only": True,
    "development_only": True,
    "scientific_promotion_allowed": False,
})

_INT32_MAX = (1 << 31) - 1
_MAX_ARRAY_ELEMENTS = 1_000_000
_MAX_WORKING_BYTES = 256 * 1024 * 1024


def _trusted_float_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(
        actual_type, (jax.Array, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.size < 1
        or array.size > _MAX_ARRAY_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise ValueError(f"{name} must be a bounded non-empty floating array")
    return array


def _gram_bytes(rows: int, columns: int, dtype_bytes: int) -> int:
    smaller = min(rows, columns)
    if smaller > _INT32_MAX // smaller:
        raise ValueError("Gram element count exceeds signed int32")
    count = smaller * smaller
    if dtype_bytes > _MAX_WORKING_BYTES // count:
        raise ValueError("Gram working bytes exceed 256 MiB")
    return count * dtype_bytes


@dataclass(frozen=True, slots=True)
class AdamOConfig:
    """Validated AdamO hyperparameters."""

    step_size: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    isometry_strength: float = 1e-3
    isometry_step_size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_size", validated_float32_scalar("step_size", self.step_size, positive=True)
        )
        object.__setattr__(
            self,
            "beta1",
            validated_float32_scalar(
                "beta1", self.beta1, lower=0.0, upper=1.0, upper_inclusive=False
            ),
        )
        object.__setattr__(
            self,
            "beta2",
            validated_float32_scalar(
                "beta2", self.beta2, lower=0.0, upper=1.0, upper_inclusive=False
            ),
        )
        object.__setattr__(self, "eps", validated_float32_scalar("eps", self.eps, positive=True))
        object.__setattr__(
            self,
            "isometry_strength",
            validated_float32_scalar(
                "isometry_strength", self.isometry_strength, lower=0.0
            ),
        )
        resolved_iso_step = (
            self.step_size if self.isometry_step_size is None else self.isometry_step_size
        )
        object.__setattr__(
            self,
            "isometry_step_size",
            validated_float32_scalar("isometry_step_size", resolved_iso_step, positive=True),
        )


AdamOState = AdamParamState


@chex.dataclass(frozen=True)
class AdamOUpdate:
    """Checked descent step and state transition."""

    step: Array
    new_state: AdamOState
    task_step: Array
    isometry_step: Array
    update_applied: Array


def isometry_penalty(weight: Array) -> Array:
    """Return the rectangular Gram-deviation penalty from paper equation 16."""

    matrix = _trusted_float_array(weight, name="weight")
    if matrix.ndim != 2:
        raise ValueError("isometry_penalty requires a rank-two weight matrix")
    rows, columns = matrix.shape
    _gram_bytes(rows, columns, matrix.dtype.itemsize)
    if rows >= columns:
        gram = matrix.T @ matrix
        identity = jnp.eye(columns, dtype=matrix.dtype)
    else:
        gram = matrix @ matrix.T
        identity = jnp.eye(rows, dtype=matrix.dtype)
    candidate = jnp.sum(jnp.square(gram - identity))
    valid = jnp.isfinite(candidate)
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, candidate, jnp.full_like(candidate, jnp.nan))
    if not bool(valid):
        raise ValueError("isometry penalty must be finite")
    return candidate


def isometry_gradient(weight: Array) -> Array:
    """Return the closed-form gradient of :func:`isometry_penalty`."""

    matrix = _trusted_float_array(weight, name="weight")
    if matrix.ndim != 2:
        raise ValueError("isometry_gradient requires a rank-two weight matrix")
    rows, columns = matrix.shape
    _gram_bytes(rows, columns, matrix.dtype.itemsize)
    if rows >= columns:
        identity = jnp.eye(columns, dtype=matrix.dtype)
        candidate = 4.0 * matrix @ (matrix.T @ matrix - identity)
    else:
        identity = jnp.eye(rows, dtype=matrix.dtype)
        candidate = 4.0 * (matrix @ matrix.T - identity) @ matrix
    valid = jnp.all(jnp.isfinite(candidate))
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, candidate, jnp.full_like(candidate, jnp.nan))
    if not bool(valid):
        raise ValueError("isometry gradient must be finite")
    return candidate


def state_persistent_bytes(state: AdamOState) -> int:
    """Count the complete persisted moment state from concrete arrays."""

    if type(state) is not AdamParamState:
        raise TypeError("state must be an exact AdamParamState")

    leaves = (
        state.m,
        state.v,
        state.t,
        state.step_size,
        state.beta1,
        state.beta2,
        state.eps,
    )
    total = 0
    for leaf in leaves:
        actual_type = type(leaf)
        if not issubclass(actual_type, jax.Array):
            raise ValueError("state must contain exact JAX arrays")
        size = int(leaf.nbytes)
        if size > _MAX_WORKING_BYTES - total:
            raise ValueError("AdamO state exceeds 256 MiB")
        total += size
    return total


def gram_working_bytes(shape: tuple[int, ...], *, dtype_bytes: int = 4) -> int:
    """Return the named Gram-matrix working allocation for one weight leaf."""

    if type(shape) is not tuple:
        raise TypeError("shape must be an exact tuple")
    if len(shape) != 2 or any(
        type(dimension) is not int or not 1 <= dimension <= _INT32_MAX
        for dimension in shape
    ):
        raise ValueError("shape must contain exactly two positive built-in integers")
    if type(dtype_bytes) is not int or not 1 <= dtype_bytes <= _INT32_MAX:
        raise ValueError("dtype_bytes must be a positive built-in integer")
    return _gram_bytes(shape[0], shape[1], dtype_bytes)


class AdamO:
    """Pure per-parameter AdamO transformation."""

    def __init__(self, config: AdamOConfig = AdamOConfig()):
        if type(config) is not AdamOConfig:
            raise TypeError("config must be an exact AdamOConfig")
        self.config = config
        self._adam = Adam(
            step_size=config.step_size,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        )

    def init_for_shape(self, shape: tuple[int, ...]) -> AdamOState:
        if type(shape) is not tuple or not shape:
            raise ValueError("shape must be a non-empty tuple")
        if any(type(dimension) is not int or dimension < 1 for dimension in shape):
            raise ValueError("shape dimensions must be positive built-in integers")
        return self._adam.init_for_shape(shape)

    def update_from_gradient_checked(
        self,
        state: AdamOState,
        task_gradient: Array,
        param: Array,
        *,
        regularize: bool,
    ) -> AdamOUpdate:
        """Apply equation 20 while keeping equation 19 moments task-only."""

        if type(regularize) is not bool:
            raise TypeError("regularize must be a built-in bool")
        if type(state) is not AdamParamState:
            raise TypeError("state must be an exact AdamParamState")
        parameter = _trusted_float_array(param, name="parameter")
        gradient = _trusted_float_array(task_gradient, name="task_gradient")
        if gradient.dtype != parameter.dtype:
            raise ValueError("gradient and parameter dtypes must match")
        scalar_leaves = (state.t, state.step_size, state.beta1, state.beta2, state.eps)
        if (
            gradient.shape != parameter.shape
            or state.m.shape != parameter.shape
            or state.v.shape != parameter.shape
            or state.m.dtype != parameter.dtype
            or state.v.dtype != parameter.dtype
            or any(leaf.shape != () or leaf.dtype != parameter.dtype for leaf in scalar_leaves)
        ):
            raise ValueError("state, gradient, and parameter shapes must match")
        if regularize and parameter.ndim != 2:
            raise ValueError("only rank-two weight leaves can be isometry-regularized")
        if regularize:
            _gram_bytes(parameter.shape[0], parameter.shape[1], parameter.dtype.itemsize)
        config = self.config
        task_update = self._adam.update_from_gradient_checked(
            state, gradient, error=None, param=parameter
        )
        task_step = task_update.step
        if regularize and config.isometry_strength != 0.0:
            isometry_step = (
                cast(float, config.isometry_step_size)
                * config.isometry_strength
                * isometry_gradient(parameter)
            )
            step = task_step + isometry_step
        else:
            isometry_step = jnp.zeros_like(parameter)
            step = task_step
        candidate_state = cast(AdamParamState, task_update.new_state)
        update_applied = (
            task_update.update_applied
            & floating_tree_is_finite(state)
            & jnp.all(jnp.isfinite(parameter))
            & jnp.all(jnp.isfinite(gradient))
            & floating_tree_is_finite(candidate_state)
            & jnp.all(jnp.isfinite(step))
        )
        return AdamOUpdate(  # type: ignore[call-arg]
            step=neutralize_array(update_applied, step),
            new_state=select_transaction(update_applied, candidate_state, state),
            task_step=neutralize_array(update_applied, task_step),
            isometry_step=neutralize_array(update_applied, isometry_step),
            update_applied=update_applied,
        )
