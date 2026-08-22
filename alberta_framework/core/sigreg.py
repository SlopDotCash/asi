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

from alberta_framework._bounded_containers import require_bounded_container_tree
from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1
# Origin ``jnp.asarray`` still accepts depth 64 and RecursionErrors at 10_000.
_MAX_ARRAY_NESTING_DEPTH = 64
_MAX_HOST_ARRAY_CONTAINER_ITEMS = 4096
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        *(np.dtype(code).type
          for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")),
    }
)
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)
_FLOAT32_TINY = float.fromhex("0x1.000000p-126")
_MIN_KERNEL_WIDTH = float.fromhex("0x1.000000p-63")
_MAX_KERNEL_WIDTH = float.fromhex("0x1.fffffep+63")
_KERNEL_WIDTH_ERROR = (
    "kernel_width must be positive and finite in float32 kernel arithmetic"
)
_SAMPLES_FINITE_ERROR = "samples must be finite"
_EMBEDDINGS_FINITE_ERROR = "embeddings must be finite"
_DIRECTIONS_FINITE_ERROR = "directions must be finite"
_PROJECTIONS_FINITE_ERROR = "projected samples must be finite"


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return canonical


def _require_float32_resource(name: str, scalar_count: int) -> None:
    if not 1 <= scalar_count <= _INT32_MAX or scalar_count > _INT32_MAX // 4:
        raise ValueError(f"derived {name} float32 allocation must fit in signed int32 bytes")


def _normalized_positive_float32(
    name: str,
    value: object,
    *,
    squared: bool,
) -> float:
    """Return one operation-safe exact-host and float32 scalar."""
    message = f"{name} must be positive and finite in float32 arithmetic"
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(message)
    try:
        return validated_float32_scalar(
            name,
            value,
            lower=_MIN_KERNEL_WIDTH if squared else _FLOAT32_TINY,
            upper=_MAX_KERNEL_WIDTH if squared else None,
        )
    except ValueError as error:
        raise ValueError(message) from error


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
        _require_float32_resource("SIGReg directions", n_projections)
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
        if type(config) is not dict:
            raise ValueError("SIGRegConfig must be an actual dict")
        payload = dict(config)
        if any(type(key) is not str for key in payload) or set(payload) != {
            "type",
            "n_projections",
            "kernel_width",
            "eps",
        }:
            raise ValueError("SIGRegConfig fields do not match its schema")
        type_name = payload.pop("type")
        if type(type_name) is not str or type_name != "SIGRegConfig":
            raise ValueError("type must be SIGRegConfig")
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
        jnp.issubdtype(uncast.dtype, jnp.integer)
        or jnp.issubdtype(uncast.dtype, jnp.floating)
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


def _require_host_array_nesting(value: object, *, name: str) -> None:
    """Reject cycles and nesting that RecursionError ``jnp.asarray``."""
    def children(node: object) -> list[Any] | tuple[Any, ...] | None:
        if type(node) in (list, tuple):
            sequence = cast(list[Any] | tuple[Any, ...], node)
            if len(sequence) > _MAX_HOST_ARRAY_CONTAINER_ITEMS:
                raise ValueError(
                    f"{name} length must be an integer in [0, {_MAX_HOST_ARRAY_CONTAINER_ITEMS}]"
                )
            return sequence
        return None

    require_bounded_container_tree(
        value,
        children=children,
        max_depth=_MAX_ARRAY_NESTING_DEPTH,
        max_nodes=None,
        name=name,
        kind="host array",
    )


def _asarray_float32(value: object, *, name: str) -> Array:
    _require_host_array_nesting(value, name=name)
    try:
        return jnp.asarray(value, dtype=jnp.float32)
    except RecursionError as exc:
        raise ValueError(f"{name} exceeds the maximum host array nesting depth") from exc


def _require_finite_array(values: Array, message: str) -> Array:
    """Reject non-finite coordinates without rewriting them to a fake zero."""
    array = jnp.asarray(values)
    predicate = jnp.all(jnp.isfinite(array))
    if not isinstance(predicate, jax.core.Tracer):
        if not bool(predicate):
            raise ValueError(message)
        return array
    return _runtime_checked_value(array, predicate, message)


def _preflight_projected_statistic(
    embeddings: Array,
    directions: Array,
) -> tuple[Array, Array, int, int]:
    z = _asarray_float32(embeddings, name="embeddings")
    dirs = _asarray_float32(directions, name="directions")
    if z.ndim < 2:
        raise ValueError("embeddings must have shape (..., latent_dim)")
    if dirs.ndim != 2 or dirs.shape[0] < 1 or dirs.shape[1] != z.shape[-1]:
        raise ValueError("directions must have shape (n_projections, latent_dim)")
    sample_count = math.prod(z.shape[:-1])
    if sample_count < 1:
        raise ValueError("embeddings must contain at least one sample")
    n_projections = dirs.shape[0]
    projection_scalars = sample_count * n_projections
    pairwise_scalars = sample_count * sample_count * n_projections
    _require_float32_resource(
        "projected pairwise workspace",
        projection_scalars + 3 * pairwise_scalars,
    )
    return z, dirs, sample_count, n_projections


def _stable_std(values: Array, *, axis: int) -> Array:
    """Compute float32 standard deviation without overflowing finite inputs."""
    scale = jnp.max(jnp.abs(values), axis=axis, keepdims=True)
    safe_scale = jnp.where(scale == 0.0, jnp.ones_like(scale), scale)
    normalized = values / safe_scale
    return jnp.std(normalized, axis=axis) * jnp.squeeze(scale, axis=axis)


def _epps_pulley_gaussian_statistic(samples: Array, width: Array) -> Array:
    """Compute the statistic after ``width`` has passed boundary validation."""
    x = jnp.ravel(jnp.asarray(samples, dtype=jnp.float32))
    width_squared = width * width
    diffs = x[:, None] - x[None, :]
    empirical = jnp.mean(jnp.exp(-(diffs**2) / (2.0 * width_squared)))
    cross_scale = jnp.sqrt(width_squared / (width_squared + 1.0))
    cross = jnp.mean(
        cross_scale * jnp.exp(-(x**2) / (2.0 * (width_squared + 1.0)))
    )
    target = jnp.sqrt(width_squared / (width_squared + 2.0))
    return jnp.maximum(empirical - 2.0 * cross + target, 0.0)


def sample_sigreg_directions(
    key: Array,
    latent_dim: int,
    config: SIGRegConfig | None = None,
) -> Float[Array, "n_projections latent_dim"]:
    """Sample random unit directions for sliced SIGReg."""
    cfg = config or SIGRegConfig()
    latent_dim = _require_int32("latent_dim", latent_dim, minimum=1)
    _require_float32_resource("SIGReg directions", cfg.n_projections * latent_dim)
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
    flattened = jnp.ravel(_asarray_float32(samples, name="samples"))
    width = _validated_kernel_width(kernel_width)
    if flattened.size < 1:
        raise ValueError("samples must be non-empty")
    _require_float32_resource(
        "Epps-Pulley pairwise workspace", 3 * flattened.size * flattened.size
    )
    checked = _require_finite_array(flattened, _SAMPLES_FINITE_ERROR)
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
    z, dirs, _, _ = _preflight_projected_statistic(embeddings, directions)
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
    z, dirs, _, _ = _preflight_projected_statistic(embeddings, directions)
    z = _require_finite_array(z, _EMBEDDINGS_FINITE_ERROR)
    dirs = _require_finite_array(dirs, _DIRECTIONS_FINITE_ERROR)
    flat = jnp.reshape(z, (-1, z.shape[-1]))
    projected = _require_finite_array(flat @ dirs.T, _PROJECTIONS_FINITE_ERROR)
    latent_std = _stable_std(flat, axis=0)
    projected_std = _stable_std(projected, axis=0)
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
