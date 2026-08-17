# mypy: disable-error-code="call-arg"
"""Stateless, fail-closed mixing of two representation-gradient sources.

The mixer gives behavior prediction and grounded-world prediction named,
fixed-weight paths into one representation update.  It owns no parameters,
optimizer state, history, or RNG.  Mixing modes are static configuration: both
gradient arrays and both dynamic validity predicates remain present in every
call so matched evaluator arms keep the same input and diagnostic surface.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Mapping
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar_with_ratio,
)

GradientMixMode = Literal["full", "behavior_only", "world_only", "discard"]
GradientNormalization = Literal["none", "unit_l2"]

REPRESENTATION_GRADIENT_MIXER_CONFIG_SCHEMA = (
    "alberta.representation-gradient-mixer.config.v1"
)

_CONFIG_TYPE = "RepresentationGradientMixerConfig"
_VALID_MODES = frozenset({"full", "behavior_only", "world_only", "discard"})
_VALID_NORMALIZATIONS = frozenset({"none", "unit_l2"})
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_INT32_MAX = 2**31 - 1
_DIAGNOSTIC_FLOAT32_SCALARS = 12
_OUTPUT_BOOL_SCALARS = 12
_FIXED_OUTPUT_BYTES = 4 * _DIAGNOSTIC_FLOAT32_SCALARS + _OUTPUT_BOOL_SCALARS
_MAX_REPRESENTATION_DIM = (_INT32_MAX - _FIXED_OUTPUT_BYTES) // 4
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")),
    }
)
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _strict_float32_scalar(
    value: object,
    label: str,
    *,
    allow_zero: bool,
) -> float:
    """Validate a finite nonnegative scalar with a stable float32 encoding."""

    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{label} must be a finite real scalar")

    normalized, numerator, denominator = validated_float32_scalar_with_ratio(
        label,
        value,
        positive=not allow_zero,
        lower=0.0 if allow_zero else None,
        upper=_FLOAT32_MAX,
    )
    tiny_numerator, tiny_denominator = _FLOAT32_TINY.as_integer_ratio()
    exact_is_subnormal = (
        numerator != 0
        and numerator * tiny_denominator < tiny_numerator * denominator
    )
    if exact_is_subnormal or (normalized != 0.0 and normalized < _FLOAT32_TINY):
        raise ValueError(f"{label} is below the normal float32 range")
    return 0.0 if numerator == 0 else normalized


@dataclasses.dataclass(frozen=True)
class RepresentationGradientMixerConfig:
    """Static, serializable contract for two-source gradient mixing.

    ``unit_l2`` divides a source by ``max(source_norm,
    normalization_epsilon)``.  Source clips are applied after optional
    normalization and before the disclosed weight.  The final clip is applied
    after summing the two weighted source contributions.
    """

    representation_dim: int
    mode: GradientMixMode = "full"
    behavior_weight: float = 1.0
    grounded_world_weight: float = 1.0
    behavior_normalization: GradientNormalization = "none"
    grounded_world_normalization: GradientNormalization = "none"
    normalization_epsilon: float = 1e-8
    behavior_clip_norm: float | None = None
    grounded_world_clip_norm: float | None = None
    final_clip_norm: float | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous, dynamic, or non-float32-representable rules."""

        object.__setattr__(
            self,
            "representation_dim",
            _require_int32(
                "representation_dim",
                self.representation_dim,
                minimum=1,
                maximum=_MAX_REPRESENTATION_DIM,
            ),
        )
        if type(self.mode) is not str or self.mode not in _VALID_MODES:
            raise ValueError(
                "mode must be full, behavior_only, world_only, or discard"
            )
        if (
            type(self.behavior_normalization) is not str
            or self.behavior_normalization not in _VALID_NORMALIZATIONS
        ):
            raise ValueError("behavior_normalization must be none or unit_l2")
        if (
            type(self.grounded_world_normalization) is not str
            or self.grounded_world_normalization not in _VALID_NORMALIZATIONS
        ):
            raise ValueError("grounded_world_normalization must be none or unit_l2")
        object.__setattr__(
            self,
            "behavior_weight",
            _strict_float32_scalar(
                self.behavior_weight,
                "behavior_weight",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "grounded_world_weight",
            _strict_float32_scalar(
                self.grounded_world_weight,
                "grounded_world_weight",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "normalization_epsilon",
            _strict_float32_scalar(
                self.normalization_epsilon,
                "normalization_epsilon",
                allow_zero=False,
            ),
        )
        for label, value in (
            ("behavior_clip_norm", self.behavior_clip_norm),
            ("grounded_world_clip_norm", self.grounded_world_clip_norm),
            ("final_clip_norm", self.final_clip_norm),
        ):
            if value is not None:
                object.__setattr__(
                    self,
                    label,
                    _strict_float32_scalar(value, label, allow_zero=False),
                )

    def to_config(self) -> dict[str, object]:
        """Return the complete JSON-compatible mixing rule."""

        return {
            "schema": REPRESENTATION_GRADIENT_MIXER_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "representation_dim": self.representation_dim,
            "mode": self.mode,
            "behavior_weight": float(self.behavior_weight),
            "grounded_world_weight": float(self.grounded_world_weight),
            "behavior_normalization": self.behavior_normalization,
            "grounded_world_normalization": self.grounded_world_normalization,
            "normalization_epsilon": float(self.normalization_epsilon),
            "behavior_clip_norm": (
                None
                if self.behavior_clip_norm is None
                else float(self.behavior_clip_norm)
            ),
            "grounded_world_clip_norm": (
                None
                if self.grounded_world_clip_norm is None
                else float(self.grounded_world_clip_norm)
            ),
            "final_clip_norm": (
                None if self.final_clip_norm is None else float(self.final_clip_norm)
            ),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RepresentationGradientMixerConfig:
        """Strictly reconstruct only an exact :meth:`to_config` payload."""

        if type(config) is not dict:
            raise ValueError("config must be an exact built-in dict")
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "representation_dim",
            "mode",
            "behavior_weight",
            "grounded_world_weight",
            "behavior_normalization",
            "grounded_world_normalization",
            "normalization_epsilon",
            "behavior_clip_norm",
            "grounded_world_clip_norm",
            "final_clip_norm",
        }
        if any(type(key) is not str for key in payload) or set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
        schema = payload.pop("schema")
        if (
            type(schema) is not str
            or schema != REPRESENTATION_GRADIENT_MIXER_CONFIG_SCHEMA
        ):
            raise ValueError("config schema differs")
        config_type = payload.pop("type")
        if type(config_type) is not str or config_type != _CONFIG_TYPE:
            raise ValueError("config type differs")
        return cls(**cast(dict[str, Any], payload))

    @property
    def resource_budget(self) -> RepresentationGradientMixerResourceBudget:
        """Return the allocation contract for this stateless mixer."""

        return RepresentationGradientMixerResourceBudget(
            representation_dim=self.representation_dim,
            persistent_state_scalars=0,
            persistent_state_bytes=0,
            output_float32_scalars=(
                self.representation_dim + _DIAGNOSTIC_FLOAT32_SCALARS
            ),
            output_bool_scalars=_OUTPUT_BOOL_SCALARS,
            output_scalars=(
                self.representation_dim
                + _DIAGNOSTIC_FLOAT32_SCALARS
                + _OUTPUT_BOOL_SCALARS
            ),
            output_nbytes=(
                4 * (self.representation_dim + _DIAGNOSTIC_FLOAT32_SCALARS)
                + _OUTPUT_BOOL_SCALARS
            ),
        )


@dataclasses.dataclass(frozen=True)
class RepresentationGradientMixerResourceBudget:
    """Exact persistent state and logical public-result payload accounting.

    Output counts include the mixed gradient, diagnostics, and repeated
    top-level decision fields. They intentionally exclude compiler-dependent
    temporaries and do not assume that repeated result leaves alias buffers.
    """

    representation_dim: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    output_float32_scalars: int
    output_bool_scalars: int
    output_scalars: int
    output_nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "representation_dim",
            _require_int32("representation_dim", self.representation_dim, minimum=0),
        )
        object.__setattr__(
            self,
            "persistent_state_scalars",
            _require_int32(
                "persistent_state_scalars", self.persistent_state_scalars, minimum=0
            ),
        )
        object.__setattr__(
            self,
            "persistent_state_bytes",
            _require_int32(
                "persistent_state_bytes", self.persistent_state_bytes, minimum=0
            ),
        )
        object.__setattr__(
            self,
            "output_float32_scalars",
            _require_int32(
                "output_float32_scalars", self.output_float32_scalars, minimum=0
            ),
        )
        object.__setattr__(
            self,
            "output_bool_scalars",
            _require_int32("output_bool_scalars", self.output_bool_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "output_scalars",
            _require_int32("output_scalars", self.output_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "output_nbytes",
            _require_int32("output_nbytes", self.output_nbytes, minimum=0),
        )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class RepresentationGradientMixDiagnostics:
    """Immutable scalar diagnostics for one two-source mix."""

    behavior_raw_norm: Float[Array, ""]
    grounded_world_raw_norm: Float[Array, ""]
    behavior_used_norm: Float[Array, ""]
    grounded_world_used_norm: Float[Array, ""]
    behavior_weight: Float[Array, ""]
    grounded_world_weight: Float[Array, ""]
    behavior_effective_weight: Float[Array, ""]
    grounded_world_effective_weight: Float[Array, ""]
    dot_product: Float[Array, ""]
    cosine_similarity: Float[Array, ""]
    conflict: Bool[Array, ""]
    unclipped_mixed_norm: Float[Array, ""]
    final_mixed_norm: Float[Array, ""]
    behavior_valid: Bool[Array, ""]
    grounded_world_valid: Bool[Array, ""]
    behavior_active: Bool[Array, ""]
    grounded_world_active: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    zero_output: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RepresentationGradientMixResult:
    """Final gradient and top-level fail-closed decision flags."""

    gradient: Float[Array, " representation_dim"]
    valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    zero_output: Bool[Array, ""]
    diagnostics: RepresentationGradientMixDiagnostics


def _gradient_array(value: object, *, dimension: int, label: str) -> Array:
    """Return one fixed-width float32 gradient or reject its static contract."""

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a floating array") from exc
    if array.shape != (dimension,):
        raise ValueError(f"{label} shape must be ({dimension},)")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{label} must be a floating array")
    return jnp.asarray(array, dtype=jnp.float32)


def _scalar_boolean(value: object, *, label: str) -> Array:
    """Accept Python or traced scalar booleans without coercing integers."""

    try:
        predicate = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a scalar boolean") from exc
    if predicate.shape != ():
        raise ValueError(f"{label} must be a scalar boolean")
    if predicate.dtype != jnp.bool_:
        raise TypeError(f"{label} must be a scalar boolean")
    return predicate


def _scaled_norm_parts(value: Array) -> tuple[Array, Array, Array]:
    """Return max scale, scaled vector, and scaled L2 norm without overflow."""

    scale = jnp.max(jnp.abs(value))
    scale_mantissa, scale_exponent = jnp.frexp(scale)
    safe_mantissa = jnp.where(
        scale > 0.0,
        scale_mantissa,
        jnp.float32(1.0),
    )
    scaled = jnp.ldexp(value, -scale_exponent) / safe_mantissa
    scaled_norm = jnp.sqrt(jnp.sum(jnp.square(scaled)))
    return scale, scaled, scaled_norm


def _finite_saturated_norm(value: Array) -> Array:
    """Compute an L2 norm with finite saturation instead of overflow."""

    scale, _, scaled_norm = _scaled_norm_parts(value)
    safe_scaled_norm = jnp.where(
        scaled_norm > 0.0,
        scaled_norm,
        jnp.float32(1.0),
    )
    overflow = scale > jnp.float32(_FLOAT32_MAX) / safe_scaled_norm
    ordinary = scale * scaled_norm
    return jnp.where(
        scaled_norm == 0.0,
        jnp.float32(0.0),
        jnp.where(overflow, jnp.float32(_FLOAT32_MAX), ordinary),
    )


def _normalize_unit_l2(value: Array, epsilon: float) -> Array:
    """Divide by max(L2 norm, epsilon) without forming an overflowing norm."""

    scale, scaled, scaled_norm = _scaled_norm_parts(value)
    safe_scaled_norm = jnp.where(
        scaled_norm > 0.0,
        scaled_norm,
        jnp.float32(1.0),
    )
    epsilon_value = jnp.float32(epsilon)
    at_least_epsilon = jnp.logical_and(
        scaled_norm > 0.0,
        scale >= epsilon_value / safe_scaled_norm,
    )
    unit = scaled / safe_scaled_norm
    below_epsilon = scaled * (scale / epsilon_value)
    return jnp.where(at_least_epsilon, unit, below_epsilon)


def _clip_l2(value: Array, limit: float | None) -> Array:
    """Apply a global L2 clip with max-scale arithmetic and exact no-op behavior."""

    if limit is None:
        return value
    scale, scaled, scaled_norm = _scaled_norm_parts(value)
    safe_scaled_norm = jnp.where(
        scaled_norm > 0.0,
        scaled_norm,
        jnp.float32(1.0),
    )
    limit_value = jnp.float32(limit)
    needs_clip = jnp.logical_and(
        scaled_norm > 0.0,
        scale > limit_value / safe_scaled_norm,
    )
    clipped = (scaled / safe_scaled_norm) * limit_value
    return jnp.where(needs_clip, clipped, value)


def _prepare_source(
    value: Array,
    *,
    normalization: GradientNormalization,
    epsilon: float,
    clip_norm: float | None,
) -> Array:
    normalized = (
        value if normalization == "none" else _normalize_unit_l2(value, epsilon)
    )
    return _clip_l2(normalized, clip_norm)


def _safe_cosine(first: Array, second: Array) -> Array:
    """Return finite cosine similarity, with exact zero for either zero vector."""

    _, first_scaled, first_scaled_norm = _scaled_norm_parts(first)
    _, second_scaled, second_scaled_norm = _scaled_norm_parts(second)
    denominator = first_scaled_norm * second_scaled_norm
    safe_denominator = jnp.where(denominator > 0.0, denominator, jnp.float32(1.0))
    cosine = jnp.sum(first_scaled * second_scaled) / safe_denominator
    return jnp.where(
        denominator > 0.0,
        jnp.clip(cosine, -1.0, 1.0),
        jnp.float32(0.0),
    )


def _safe_dot(first: Array, second: Array, cosine: Array) -> Array:
    """Return a signed, finite-saturated dot product."""

    first_norm = _finite_saturated_norm(first)
    second_norm = _finite_saturated_norm(second)
    raw = cosine * first_norm * second_norm
    return jnp.nan_to_num(
        raw,
        nan=0.0,
        posinf=_FLOAT32_MAX,
        neginf=-_FLOAT32_MAX,
    )


def mix_representation_gradients(
    config: RepresentationGradientMixerConfig,
    behavior_gradient: Float[Array, " representation_dim"],
    grounded_world_gradient: Float[Array, " representation_dim"],
    *,
    behavior_valid: Bool[Array, ""] | bool = True,
    grounded_world_valid: Bool[Array, ""] | bool = True,
) -> RepresentationGradientMixResult:
    """Mix two fixed-width gradients or atomically reject to an exact zero.

    Dynamic validity predicates must be scalar booleans.  A source is active
    only when its configured mode includes it and its fixed weight is positive.
    Invalid inactive sources remain visible in diagnostics but do not reject a
    valid active source.  Any invalid active source or nonfinite weighted mix
    rejects the entire output.
    """

    behavior = _gradient_array(
        behavior_gradient,
        dimension=config.representation_dim,
        label="behavior_gradient",
    )
    world = _gradient_array(
        grounded_world_gradient,
        dimension=config.representation_dim,
        label="grounded_world_gradient",
    )
    behavior_predicate = _scalar_boolean(
        behavior_valid,
        label="behavior_valid",
    )
    world_predicate = _scalar_boolean(
        grounded_world_valid,
        label="grounded_world_valid",
    )

    behavior_finite = jnp.all(jnp.isfinite(behavior))
    world_finite = jnp.all(jnp.isfinite(world))
    behavior_source_valid = jnp.logical_and(behavior_predicate, behavior_finite)
    world_source_valid = jnp.logical_and(world_predicate, world_finite)
    safe_behavior = jnp.where(jnp.isfinite(behavior), behavior, jnp.float32(0.0))
    safe_world = jnp.where(jnp.isfinite(world), world, jnp.float32(0.0))

    behavior_raw_finite_norm = _finite_saturated_norm(safe_behavior)
    world_raw_finite_norm = _finite_saturated_norm(safe_world)
    behavior_raw_norm = jnp.where(
        behavior_finite,
        behavior_raw_finite_norm,
        jnp.float32(jnp.inf),
    )
    world_raw_norm = jnp.where(
        world_finite,
        world_raw_finite_norm,
        jnp.float32(jnp.inf),
    )

    prepared_behavior = _prepare_source(
        safe_behavior,
        normalization=config.behavior_normalization,
        epsilon=config.normalization_epsilon,
        clip_norm=config.behavior_clip_norm,
    )
    prepared_world = _prepare_source(
        safe_world,
        normalization=config.grounded_world_normalization,
        epsilon=config.normalization_epsilon,
        clip_norm=config.grounded_world_clip_norm,
    )

    behavior_mode_enabled = config.mode in {"full", "behavior_only"}
    world_mode_enabled = config.mode in {"full", "world_only"}
    behavior_active = jnp.asarray(
        behavior_mode_enabled and config.behavior_weight > 0.0,
        dtype=jnp.bool_,
    )
    world_active = jnp.asarray(
        world_mode_enabled and config.grounded_world_weight > 0.0,
        dtype=jnp.bool_,
    )
    behavior_weight_value = jnp.float32(config.behavior_weight)
    world_weight_value = jnp.float32(config.grounded_world_weight)
    behavior_effective_weight = jnp.float32(
        config.behavior_weight if behavior_mode_enabled else 0.0
    )
    world_effective_weight = jnp.float32(
        config.grounded_world_weight if world_mode_enabled else 0.0
    )

    zeros = jnp.zeros((config.representation_dim,), dtype=jnp.float32)
    behavior_for_use = jnp.where(behavior_source_valid, prepared_behavior, zeros)
    world_for_use = jnp.where(world_source_valid, prepared_world, zeros)
    behavior_contribution = behavior_effective_weight * behavior_for_use
    world_contribution = world_effective_weight * world_for_use
    behavior_contribution_finite = jnp.all(jnp.isfinite(behavior_contribution))
    world_contribution_finite = jnp.all(jnp.isfinite(world_contribution))
    safe_behavior_contribution = jnp.where(
        jnp.isfinite(behavior_contribution),
        behavior_contribution,
        jnp.float32(0.0),
    )
    safe_world_contribution = jnp.where(
        jnp.isfinite(world_contribution),
        world_contribution,
        jnp.float32(0.0),
    )

    mixed_unclipped = safe_behavior_contribution + safe_world_contribution
    mixed_finite = jnp.all(jnp.isfinite(mixed_unclipped))
    safe_mixed_unclipped = jnp.where(
        jnp.isfinite(mixed_unclipped),
        mixed_unclipped,
        jnp.float32(0.0),
    )
    numerical_mix_valid = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_or(jnp.logical_not(behavior_active), behavior_contribution_finite),
            jnp.logical_or(jnp.logical_not(world_active), world_contribution_finite),
        ),
        mixed_finite,
    )
    active_sources_valid = jnp.logical_and(
        jnp.logical_or(jnp.logical_not(behavior_active), behavior_source_valid),
        jnp.logical_or(jnp.logical_not(world_active), world_source_valid),
    )
    valid = jnp.logical_and(active_sources_valid, numerical_mix_valid)
    final_candidate = _clip_l2(safe_mixed_unclipped, config.final_clip_norm)
    valid = jnp.logical_and(valid, jnp.all(jnp.isfinite(final_candidate)))
    gradient = jnp.where(valid, final_candidate, zeros)

    any_active = jnp.logical_or(behavior_active, world_active)
    applied = jnp.logical_and(valid, any_active)
    rejected = jnp.logical_not(valid)
    zero_output = jnp.all(gradient == jnp.float32(0.0))
    behavior_used_norm = jnp.where(
        behavior_contribution_finite,
        _finite_saturated_norm(safe_behavior_contribution),
        jnp.float32(jnp.inf),
    )
    world_used_norm = jnp.where(
        world_contribution_finite,
        _finite_saturated_norm(safe_world_contribution),
        jnp.float32(jnp.inf),
    )
    cosine = _safe_cosine(safe_behavior_contribution, safe_world_contribution)
    dot = _safe_dot(safe_behavior_contribution, safe_world_contribution, cosine)
    conflict = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(behavior_active, world_active),
                jnp.logical_and(behavior_source_valid, world_source_valid),
            ),
            jnp.logical_and(
                behavior_contribution_finite,
                world_contribution_finite,
            ),
        ),
        jnp.logical_and(
            jnp.logical_and(behavior_used_norm > 0.0, world_used_norm > 0.0),
            cosine < jnp.float32(0.0),
        ),
    )
    unclipped_norm = jnp.where(
        numerical_mix_valid,
        _finite_saturated_norm(safe_mixed_unclipped),
        jnp.float32(jnp.inf),
    )
    final_norm = _finite_saturated_norm(gradient)

    diagnostics = RepresentationGradientMixDiagnostics(
        behavior_raw_norm=behavior_raw_norm,
        grounded_world_raw_norm=world_raw_norm,
        behavior_used_norm=behavior_used_norm,
        grounded_world_used_norm=world_used_norm,
        behavior_weight=behavior_weight_value,
        grounded_world_weight=world_weight_value,
        behavior_effective_weight=behavior_effective_weight,
        grounded_world_effective_weight=world_effective_weight,
        dot_product=dot,
        cosine_similarity=cosine,
        conflict=conflict,
        unclipped_mixed_norm=unclipped_norm,
        final_mixed_norm=final_norm,
        behavior_valid=behavior_source_valid,
        grounded_world_valid=world_source_valid,
        behavior_active=behavior_active,
        grounded_world_active=world_active,
        applied=applied,
        rejected=rejected,
        zero_output=zero_output,
    )
    return RepresentationGradientMixResult(
        gradient=gradient,
        valid=valid,
        applied=applied,
        rejected=rejected,
        zero_output=zero_output,
        diagnostics=diagnostics,
    )


__all__ = [
    "GradientMixMode",
    "GradientNormalization",
    "REPRESENTATION_GRADIENT_MIXER_CONFIG_SCHEMA",
    "RepresentationGradientMixDiagnostics",
    "RepresentationGradientMixResult",
    "RepresentationGradientMixerConfig",
    "RepresentationGradientMixerResourceBudget",
    "mix_representation_gradients",
]
