# mypy: disable-error-code="call-arg"
"""Sliced isotropic Gaussian regularization for latent representations.

SIGReg is the anti-collapse regularizer introduced with LeJEPA (Balestriero
& LeCun 2025): project a latent batch onto random unit directions and
penalize each one-dimensional projection's departure from ``N(0, 1)`` with
an Epps-Pulley/BHEP normality statistic, driving the batch toward an
isotropic Gaussian without covariance-matrix machinery.

SIGReg is useful whenever a learner emits a batch of latent features that
should avoid collapse while retaining a simple Gaussian prior. The functions
here are model-agnostic: callers can use them as a diagnostic, an auxiliary
loss in a batch learner, or a gate before trusting imagined rollouts.

References:
    Balestriero & LeCun (2025). "LeJEPA: Provable and Scalable
        Self-Supervised Learning Without the Heuristics."
    Epps & Pulley (1983). "A Test for Normality Based on the Empirical
        Characteristic Function."
    Baringhaus & Henze (1988). "A Consistent Test for Multivariate
        Normality Based on the Empirical Characteristic Function."
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_KERNEL_WIDTH_ERROR = "kernel_width must be positive and finite in float32 kernel arithmetic"
_SAMPLES_FINITE_ERROR = "samples must be finite"
_EMBEDDINGS_FINITE_ERROR = "embeddings must be finite"
_DIRECTIONS_FINITE_ERROR = "directions must be finite"
_PROJECTIONS_FINITE_ERROR = "projected samples must be finite"


def _normalized_positive_float32(
    name: str,
    value: object,
    *,
    squared: bool,
) -> float:
    """Return a concrete real as ``float`` after float32-domain validation."""
    message = f"{name} must be positive and finite in float32 arithmetic"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(message)
    try:
        concrete = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(message)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value32 = np.float32(concrete)
        squared32 = np.float32(value32 * value32)
    if not math.isfinite(float(value32)) or float(value32) < _FLOAT32_TINY:
        raise ValueError(message)
    if squared and (not math.isfinite(float(squared32)) or float(squared32) < _FLOAT32_TINY):
        raise ValueError(message)
    return concrete


@dataclasses.dataclass(frozen=True)
class SIGRegConfig:
    """Configuration for sliced Gaussian regularization.

    Args:
        n_projections: Number of random unit directions.  More directions
            tighten the sliced approximation to full isotropy, at cost
            linear in ``n_projections``.
        kernel_width: Gaussian-kernel width used in the Epps-Pulley/BHEP
            statistic. ``1.0`` is the standard simple setting.
        eps: Numerical floor for norms and square roots.
    """

    n_projections: int = 32
    kernel_width: float = 1.0
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate the configuration."""
        n_projections = _require_int32("n_projections", self.n_projections, minimum=1)
        kernel_width = _normalized_positive_float32(
            "kernel_width",
            self.kernel_width,
            squared=True,
        )
        eps = _normalized_positive_float32("eps", self.eps, squared=False)
        object.__setattr__(self, "n_projections", n_projections)
        object.__setattr__(self, "kernel_width", kernel_width)
        object.__setattr__(self, "eps", eps)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "SIGRegConfig"
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SIGRegConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class SIGRegDiagnostics:
    """Distribution diagnostics for a latent batch."""

    loss: Float[Array, ""]
    latent_mean_abs: Float[Array, ""]
    latent_std_mean: Float[Array, ""]
    latent_std_min: Float[Array, ""]
    projected_mean_abs: Float[Array, ""]
    projected_std_mean: Float[Array, ""]


def _runtime_checked_value(
    value: Array,
    predicate: Array,
    message: str,
) -> Array:
    """Enforce a scalar predicate without breaking valid JAX transforms.

    Eager invalid inputs are rejected before this helper. Under compiled JAX
    transforms, a failed check surfaces on synchronization as
    :class:`jax.errors.JaxRuntimeError` wrapping the host ``ValueError``;
    an uncompiled ``jax.vmap`` may surface the host ``ValueError`` directly.
    """
    predicate = jnp.reshape(jnp.asarray(predicate, dtype=jnp.bool_), ())

    def valid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, _ = operand
        return checked_value

    def invalid_branch(operand: tuple[Array, Array]) -> Array:
        checked_value, runtime_predicate = operand

        def raise_if_false(concrete_predicate: Any) -> None:
            if not bool(concrete_predicate):
                raise ValueError(message)

        jax.debug.callback(raise_if_false, runtime_predicate, ordered=True)
        return checked_value

    return cast(
        Array,
        jax.lax.cond(
            predicate,
            valid_branch,
            invalid_branch,
            (value, predicate),
        ),
    )


def _validated_kernel_width(kernel_width: float | Array) -> Array:
    """Validate the width in the float32 domain used by the statistic."""
    if isinstance(kernel_width, bool):
        raise ValueError(_KERNEL_WIDTH_ERROR)
    try:
        if isinstance(kernel_width, numbers.Real):
            uncast = jnp.asarray(float(kernel_width), dtype=jnp.float32)
        else:
            uncast = jnp.asarray(kernel_width)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_KERNEL_WIDTH_ERROR) from exc
    if uncast.shape != ():
        raise ValueError(_KERNEL_WIDTH_ERROR)
    if not (
        jnp.issubdtype(uncast.dtype, jnp.integer) or jnp.issubdtype(uncast.dtype, jnp.floating)
    ):
        raise ValueError(_KERNEL_WIDTH_ERROR)

    width = jnp.asarray(uncast, dtype=jnp.float32)
    width_squared = width * width
    predicate = (
        jnp.isfinite(width)
        & (width > 0.0)
        & jnp.isfinite(width_squared)
        & (width_squared >= jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32))
    )
    if not isinstance(width, jax.core.Tracer):
        if not bool(predicate):
            raise ValueError(_KERNEL_WIDTH_ERROR)
        return width
    return _runtime_checked_value(width, predicate, _KERNEL_WIDTH_ERROR)


def _require_finite_array(values: Array, message: str) -> Array:
    """Reject non-finite coordinates without rewriting them to a fake zero."""
    array = jnp.asarray(values)
    predicate = jnp.all(jnp.isfinite(array))
    if not isinstance(predicate, jax.core.Tracer):
        if not bool(predicate):
            raise ValueError(message)
        return array
    return _runtime_checked_value(array, predicate, message)


def _epps_pulley_gaussian_statistic(samples: Array, width: Array) -> Array:
    """Compute the statistic after ``width`` has passed boundary validation."""
    x = jnp.ravel(jnp.asarray(samples, dtype=jnp.float32))
    width_squared = width * width
    diffs = x[:, None] - x[None, :]
    empirical = jnp.mean(jnp.exp(-(diffs**2) / (2.0 * width_squared)))
    cross_scale = jnp.sqrt(width_squared / (width_squared + 1.0))
    cross = jnp.mean(cross_scale * jnp.exp(-(x**2) / (2.0 * (width_squared + 1.0))))
    target = jnp.sqrt(width_squared / (width_squared + 2.0))
    return jnp.maximum(empirical - 2.0 * cross + target, 0.0)


def sample_sigreg_directions(
    key: Array,
    latent_dim: int,
    config: SIGRegConfig | None = None,
) -> Float[Array, "n_projections latent_dim"]:
    """Sample random unit directions for sliced SIGReg."""
    cfg = config or SIGRegConfig()
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    directions = jax.random.normal(
        key,
        (cfg.n_projections, latent_dim),
        dtype=jnp.float32,
    )
    norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
    return directions / jnp.maximum(norms, jnp.asarray(cfg.eps, dtype=jnp.float32))


def epps_pulley_gaussian_statistic(
    samples: Array,
    *,
    kernel_width: float | Array = 1.0,
) -> Float[Array, ""]:
    """Return the one-dimensional Epps-Pulley/BHEP Gaussian statistic.

    This is the biased Gaussian-kernel MMD between the empirical one-
    dimensional sample distribution and ``N(0, 1)`` (Epps & Pulley 1983; the
    BHEP form of Baringhaus & Henze 1988). It is zero only when the projected
    distribution matches the target Gaussian in the population limit.

    ``kernel_width`` may be a dynamic scalar under JAX transforms. Invalid
    eager values raise :class:`ValueError`; invalid compiled values raise a
    :class:`jax.errors.JaxRuntimeError` when the result is synchronized.
    """
    width = _validated_kernel_width(kernel_width)
    checked = _require_finite_array(
        jnp.asarray(samples, dtype=jnp.float32),
        _SAMPLES_FINITE_ERROR,
    )
    return _epps_pulley_gaussian_statistic(checked, width)


def sliced_sigreg_loss(
    embeddings: Array,
    directions: Array,
    *,
    kernel_width: float | Array = 1.0,
) -> Float[Array, ""]:
    """Compute sliced isotropic Gaussian regularization loss.

    Args:
        embeddings: Array with shape ``(..., latent_dim)``.
        directions: Unit projection directions with shape
            ``(n_projections, latent_dim)``.
        kernel_width: Gaussian-kernel width for each projected normality test.

    Returns:
        Mean one-dimensional Epps-Pulley/BHEP statistic across directions.
    """
    z = jnp.asarray(embeddings, dtype=jnp.float32)
    dirs = jnp.asarray(directions, dtype=jnp.float32)
    if z.ndim < 2:
        raise ValueError("embeddings must have shape (..., latent_dim)")
    width = _validated_kernel_width(kernel_width)
    z = _require_finite_array(z, _EMBEDDINGS_FINITE_ERROR)
    dirs = _require_finite_array(dirs, _DIRECTIONS_FINITE_ERROR)
    flat = jnp.reshape(z, (-1, z.shape[-1]))
    projections = _require_finite_array(flat @ dirs.T, _PROJECTIONS_FINITE_ERROR)
    per_projection = jax.vmap(
        lambda values: _epps_pulley_gaussian_statistic(values, width),
        in_axes=1,
    )(projections)
    return jnp.mean(per_projection)


def sigreg_diagnostics(
    embeddings: Array,
    directions: Array,
    config: SIGRegConfig | None = None,
) -> SIGRegDiagnostics:
    """Return SIGReg loss plus simple latent distribution diagnostics."""
    cfg = config or SIGRegConfig()
    z = jnp.asarray(embeddings, dtype=jnp.float32)
    dirs = jnp.asarray(directions, dtype=jnp.float32)
    z = _require_finite_array(z, _EMBEDDINGS_FINITE_ERROR)
    dirs = _require_finite_array(dirs, _DIRECTIONS_FINITE_ERROR)
    flat = jnp.reshape(z, (-1, z.shape[-1]))
    projected = _require_finite_array(flat @ dirs.T, _PROJECTIONS_FINITE_ERROR)
    latent_std = jnp.std(flat, axis=0)
    projected_std = jnp.std(projected, axis=0)
    return SIGRegDiagnostics(
        loss=sliced_sigreg_loss(flat, dirs, kernel_width=cfg.kernel_width),
        latent_mean_abs=jnp.mean(jnp.abs(jnp.mean(flat, axis=0))),
        latent_std_mean=jnp.mean(latent_std),
        latent_std_min=jnp.min(latent_std),
        projected_mean_abs=jnp.mean(jnp.abs(jnp.mean(projected, axis=0))),
        projected_std_mean=jnp.mean(projected_std),
    )


__all__ = [
    "SIGRegConfig",
    "SIGRegDiagnostics",
    "epps_pulley_gaussian_statistic",
    "sample_sigreg_directions",
    "sigreg_diagnostics",
    "sliced_sigreg_loss",
]
