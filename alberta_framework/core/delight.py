# mypy: disable-error-code="call-arg"
"""Candidate-update safety audit and paper-defined delight signals.

:func:`assess_gradient_joy` is the historical API name for a multi-objective
candidate-update safety audit. It checks first-order effects on independent
objective, retention, and safety probes plus an update-norm trust bound. It is
optimizer control-plane logic and is *not* the paper-defined meaning of
``delight`` or “sparks joy.” New prose should call it the candidate-update
audit; compatibility names remain to avoid checkpoint/API breakage.

:func:`discrete_delightful_policy_gradient` implements the bounded
discrete-action experiment described in WP5 of
``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``. Its paper-defined signal is
``stop_gradient(advantage * -log_probability)``. Callers provide selected
categorical action log-probabilities and detached advantages, then apply the
returned scalar loss only to actor parameters.

Critic, world-model, representation, and safety losses must remain outside the
policy-gradient helper. Callers that select samples before autodiff must gather
a smaller PyTree; masking a loss inside a full graph does not demonstrate
skipped backward work.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio

PolicyGradientMode = Literal["ordinary_pg", "delightful_pg"]
GradientCandidateSemantics = Literal["gradient", "update"]
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)
_ALIGNMENT_ENDPOINT_TOLERANCE = 4.0 * _FLOAT32_EPSILON
_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES: frozenset[type] = frozenset(
    {int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")}
)
_ACTUAL_FLOAT_TYPES: frozenset[type] = frozenset(
    {float, Fraction, *(np.dtype(c).type for c in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES: frozenset[type] = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_int(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer")
    return canonical


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an actual bool")
    return value


def _require_choice(name: str, value: object, choices: tuple[str, ...]) -> str:
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} is unsupported")
    return value


def _validated_config_float(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    if numerator != 0 and abs(numerator) * (1 << 149) <= denominator:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _copy_mapping(payload: object, *, name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    try:
        data = dict(payload)
    except Exception as exc:
        raise ValueError(f"{name} payload could not be read") from exc
    for key in data:
        if type(key) is not str:
            raise ValueError(f"{name} payload has exact strings as keys")
    return data


def _require_float32_resource(
    name: str, *, vector_scalars: int, fixed_scalars: int = 0
) -> None:
    total = vector_scalars + fixed_scalars
    if total > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _validate_float32_config_value(name: str, value: float) -> None:
    """Reject values that become zero or non-finite in float32 computation."""
    # Hostile-safe gate: delegate to validated_float32_scalar_with_ratio
    _validated_config_float(name, value)


@chex.dataclass(frozen=True)
class LearningValue:
    """Typed learning-value channels with no implicit aggregate score.

    Every field is supplied by a named producer and retains its own units:

    - ``advantage``: return or reward units from the actor's baseline.
    - ``action_surprisal``: nats from a selected action log-probability.
    - ``delight``: paper-defined actor-sample delight, in return-nats,
      ``advantage * action_surprisal``. It is not the separate candidate-update
      safety-audit verdict.
    - ``epistemic_surprise``: calibrated reducible-uncertainty units.
    - ``aleatoric_uncertainty``: calibrated outcome-variance units.
    - ``learning_progress``: predictive-loss reduction per model window/version.
    - ``change_probability``: probability in ``[0, 1]`` from a change detector.
    - ``safety_cost``: the safety learner's native cost units.

    The structure deliberately defines no sum, ordering, or universal score.
    Consumers must select the channel appropriate to their decision.
    """

    advantage: Array
    action_surprisal: Array
    delight: Array
    epistemic_surprise: Array
    aleatoric_uncertainty: Array
    learning_progress: Array
    change_probability: Array
    safety_cost: Array


@dataclasses.dataclass(frozen=True)
class GradientJoyConfig:
    """Static contract for a first-order candidate-gradient assessment.

    The candidate is interpreted either as a loss gradient ``g`` whose proposed
    plain-gradient update is ``u = -gradient_step_size * g``, or directly as an
    already formed update ``u``. Callers using momentum, preconditioning,
    clipping, or another optimizer transform must assess the formed update.
    Probe inputs are gradients of quantities to *minimize*, evaluated at the
    same parameters as the candidate.

    A candidate is accepted only when probe independence is caller-attested, all
    evidence is available and finite, the objective probe predicts a strict
    decrease, retention and safety probes remain within their declared
    tolerances, and the update stays inside the norm trust bound. The soft
    weight is the minimum of four named sigmoid factors. Both the raw candidate
    and its tentative soft-weighted update must satisfy the configured
    objective/retention/safety magnitude gates. The tentative update is formed
    elementwise in float32 and receives fresh norm and dot certificates; scalar
    multiplication of candidate diagnostics is not accepted as evidence for
    that rounded tree. It is not a sum over learning-value channels.

    All numeric values are consumed as float32. Every nonzero value must be a
    finite normal float32, booleans are rejected, and ``max_update_norm`` must
    exceed the norm-resolution floor ``diagnostics_epsilon``. Probe dots and
    norms use fixed balanced reductions with conservative accumulated-roundoff
    intervals; a sign, trust bound, norm floor, or alignment threshold that the
    intervals cannot certify fails closed. Cosine diagnostics are clipped to
    ``[-1, 1]``; hard alignment gates and soft factors use a certified lower
    bound. An exact ``1.0`` threshold applies a four-machine-epsilon endpoint
    tolerance to that lower bound.
    """

    candidate_semantics: GradientCandidateSemantics = "gradient"
    gradient_step_size: float = 1.0
    max_update_norm: float = 1.0
    min_objective_decrease: float = 0.0
    max_retention_loss_increase: float = 0.0
    max_safety_cost_increase: float = 0.0
    min_objective_descent_alignment: float = 0.0
    min_retention_descent_alignment: float = 0.0
    min_safety_descent_alignment: float = 0.0
    alignment_temperature: float = 0.1
    norm_temperature: float = 0.1
    diagnostics_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate units, thresholds, and static candidate semantics."""
        object.__setattr__(
            self, "candidate_semantics", _require_choice(
                "candidate_semantics", self.candidate_semantics, ("gradient", "update")
            )
        )
        for name in (
            "gradient_step_size",
            "max_update_norm",
            "alignment_temperature",
            "norm_temperature",
            "diagnostics_epsilon",
        ):
            object.__setattr__(
                self,
                name,
                _validated_config_float(
                    name, getattr(self, name), lower=0.0, upper=None
                ),
            )
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "min_objective_decrease",
            "max_retention_loss_increase",
            "max_safety_cost_increase",
        ):
            object.__setattr__(
                self, name, _validated_config_float(name, getattr(self, name))
            )
        if self.min_objective_decrease < 0.0:
            raise ValueError("min_objective_decrease must be nonnegative")
        for name in (
            "min_objective_descent_alignment",
            "min_retention_descent_alignment",
            "min_safety_descent_alignment",
        ):
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), lower=-1.0, upper=1.0),
            )
        if self.max_update_norm <= self.diagnostics_epsilon:
            raise ValueError(
                "max_update_norm must be greater than diagnostics_epsilon "
                "so a nonzero trusted update can exist"
            )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "GradientJoyConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> GradientJoyConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = _copy_mapping(config, name="GradientJoyConfig")
        type_value = payload.pop("type", None)
        if type(type_value) is not str or type_value != "GradientJoyConfig":
            raise ValueError("GradientJoyConfig type is unsupported")
        allowed = {
            "candidate_semantics",
            "gradient_step_size",
            "max_update_norm",
            "min_objective_decrease",
            "max_retention_loss_increase",
            "max_safety_cost_increase",
            "min_objective_descent_alignment",
            "min_retention_descent_alignment",
            "min_safety_descent_alignment",
            "alignment_temperature",
            "norm_temperature",
            "diagnostics_epsilon",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"GradientJoyConfig has unsupported fields: {unknown}")
        return cls(**payload)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class LearningValueAvailability:
    """Named semantic-validity flags for all eight learning-value channels."""

    advantage: Bool[Array, ""]
    action_surprisal: Bool[Array, ""]
    delight: Bool[Array, ""]
    epistemic_surprise: Bool[Array, ""]
    aleatoric_uncertainty: Bool[Array, ""]
    learning_progress: Bool[Array, ""]
    change_probability: Bool[Array, ""]
    safety_cost: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientJoyEvidence:
    """Caller-supplied evidence for one candidate update.

    ``objective_probe_gradient`` is accepted under an explicit independence
    contract: callers must obtain it from a validation/probe objective rather
    than reuse the candidate gradient. ``retention_probe_gradient`` measures a
    protected old-skill loss and ``safety_cost_gradient`` measures a separately
    trained safety cost. The implementation cannot infer independence from
    arrays, so ``probe_independence_attested`` must explicitly attest that these
    source requirements hold. ``None``, a false attestation, or a false probe
    availability flag fails closed.

    ``learning_value`` retains the eight typed channels and
    ``learning_value_availability`` declares which producers actually emitted
    them. A channel must be both declared available and semantically valid.
    In particular, the historical ``delight`` field must exactly equal its
    float32 paper-specific DG ``advantage * action_surprisal`` identity.
    Channels are reported separately and are never aggregated into the joy
    weight.
    """

    objective_probe_gradient: Any | None
    retention_probe_gradient: Any | None
    safety_cost_gradient: Any | None
    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    learning_value: LearningValue
    learning_value_availability: LearningValueAvailability


@chex.dataclass(frozen=True)
class GradientJoyDiagnostics:
    """Named first-order diagnostics; signs follow minimization convention.

    Unqualified update/change fields describe the raw candidate. ``tentative``
    fields describe the soft-weighted candidate before the final hard veto.
    Dot error bounds are conservative raw-unit radii; their corresponding
    ``resolved`` flags are false whenever cancellation or underflow leaves the
    first-order sign uncertain. Norm and alignment bounds are outward-rounded:
    hard gates and soft factors consume the conservative edge rather than the
    reported point estimate.
    """

    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    objective_probe_finite: Bool[Array, ""]
    retention_probe_finite: Bool[Array, ""]
    safety_probe_finite: Bool[Array, ""]
    learning_value_complete: Bool[Array, ""]
    derived_numerics_valid: Bool[Array, ""]
    evidence_complete: Bool[Array, ""]
    nonzero_update: Bool[Array, ""]
    tentative_nonzero_update: Bool[Array, ""]
    objective_improves: Bool[Array, ""]
    retention_preserved: Bool[Array, ""]
    safety_preserved: Bool[Array, ""]
    within_trust_region: Bool[Array, ""]
    update_norm: Float[Array, ""]
    update_norm_lower_bound: Float[Array, ""]
    update_norm_upper_bound: Float[Array, ""]
    update_norm_resolved: Bool[Array, ""]
    objective_probe_norm: Float[Array, ""]
    retention_probe_norm: Float[Array, ""]
    safety_probe_norm: Float[Array, ""]
    predicted_objective_decrease: Float[Array, ""]
    predicted_retention_loss_change: Float[Array, ""]
    predicted_safety_cost_change: Float[Array, ""]
    objective_dot_error_bound: Float[Array, ""]
    retention_dot_error_bound: Float[Array, ""]
    safety_dot_error_bound: Float[Array, ""]
    objective_dot_resolved: Bool[Array, ""]
    retention_dot_resolved: Bool[Array, ""]
    safety_dot_resolved: Bool[Array, ""]
    objective_descent_alignment: Float[Array, ""]
    retention_descent_alignment: Float[Array, ""]
    safety_descent_alignment: Float[Array, ""]
    objective_descent_alignment_lower_bound: Float[Array, ""]
    retention_descent_alignment_lower_bound: Float[Array, ""]
    safety_descent_alignment_lower_bound: Float[Array, ""]
    objective_factor: Float[Array, ""]
    retention_factor: Float[Array, ""]
    safety_factor: Float[Array, ""]
    trust_factor: Float[Array, ""]
    tentative_weight: Float[Array, ""]
    tentative_update_norm: Float[Array, ""]
    tentative_update_norm_lower_bound: Float[Array, ""]
    tentative_update_norm_upper_bound: Float[Array, ""]
    tentative_update_norm_resolved: Bool[Array, ""]
    tentative_objective_dot_error_bound: Float[Array, ""]
    tentative_retention_dot_error_bound: Float[Array, ""]
    tentative_safety_dot_error_bound: Float[Array, ""]
    tentative_objective_dot_resolved: Bool[Array, ""]
    tentative_retention_dot_resolved: Bool[Array, ""]
    tentative_safety_dot_resolved: Bool[Array, ""]
    predicted_tentative_objective_decrease: Float[Array, ""]
    predicted_tentative_retention_loss_change: Float[Array, ""]
    predicted_tentative_safety_cost_change: Float[Array, ""]


@chex.dataclass(frozen=True)
class GradientJoyAssessment:
    """Detached candidate-update audit and auditable evidence.

    Acceptance certifies the candidate audit only.  It does not assert that an
    update was representable in a parameter dtype or committed to parameters;
    :class:`GradientJoyApplicationResult` reports that separate boundary.

    ``sparks_joy`` is retained as a historical read-only alias of ``accepted``.
    In paper-defined terminology, “sparks joy” instead means sample selection
    for a backward pass.
    """

    accepted: Bool[Array, ""]
    weight: Float[Array, ""]
    candidate_update: Any
    weighted_update: Any
    learning_value: LearningValue
    channel_availability: LearningValueAvailability
    diagnostics: GradientJoyDiagnostics

    @property
    def candidate_update_audit_passed(self) -> Bool[Array, ""]:
        """Return whether the multi-objective candidate-update audit passed."""
        return self.accepted

    @property
    def sparks_joy(self) -> Bool[Array, ""]:
        """Historical compatibility alias; prefer ``candidate_update_audit_passed``."""
        return self.candidate_update_audit_passed


@chex.dataclass(frozen=True)
class GradientJoyApplicationResult:
    """Auditable result of atomically attempting an accepted parameter update.

    ``assessment.accepted`` records the formed candidate audit, while
    ``effective_assessment`` separately audits the finite-precision stored
    delta produced by dtype casting and parameter addition. ``applied`` is true
    only when both audits accept, the complete proposed parameter tree is
    finite, and at least one stored parameter value changes. The effective
    assessment is a validation pass: its ``candidate_update`` is the stored
    delta, while its further soft-weighted update is diagnostic and is not
    recursively applied.

    ``proposed_change_nonzero`` reports that comparison independently of the
    three finiteness checks and both audits, so the named fields distinguish a
    safety no-op from an original- or effective-assessment veto.
    """

    parameters: Any
    assessment: GradientJoyAssessment
    effective_assessment: GradientJoyAssessment
    applied: Bool[Array, ""]
    parameters_finite: Bool[Array, ""]
    cast_update_finite: Bool[Array, ""]
    proposed_parameters_finite: Bool[Array, ""]
    proposed_change_nonzero: Bool[Array, ""]


def _detached_float_tree(tree: Any, *, name: str) -> tuple[Any, Any]:
    """Convert a non-empty floating PyTree to detached float32 arrays."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves:
        raise ValueError(f"{name} must be a non-empty PyTree of floating arrays")
    converted: list[Array] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError(f"{name} leaves must have floating dtypes")
        converted.append(jax.lax.stop_gradient(jnp.asarray(array, dtype=jnp.float32)))
    if sum(array.size for array in converted) == 0:
        raise ValueError(f"{name} must contain at least one floating value")
    return jax.tree_util.tree_unflatten(structure, converted), structure


def _prepare_probe_tree(
    probe: Any | None,
    candidate: Any,
    candidate_structure: Any,
    *,
    name: str,
) -> tuple[Any, Array]:
    """Prepare a probe matching the candidate, or a zero unavailable probe."""
    if probe is None:
        return (
            jax.tree_util.tree_map(jnp.zeros_like, candidate),
            jnp.array(False, dtype=jnp.bool_),
        )
    prepared, structure = _detached_float_tree(probe, name=name)
    if structure != candidate_structure:
        raise ValueError(f"{name} PyTree structure must match candidate")
    candidate_leaves = jax.tree_util.tree_leaves(candidate)
    probe_leaves = jax.tree_util.tree_leaves(prepared)
    for candidate_leaf, probe_leaf in zip(
        candidate_leaves,
        probe_leaves,
        strict=True,
    ):
        if candidate_leaf.shape != probe_leaf.shape:
            raise ValueError(f"{name} leaf shapes must match candidate")
    return prepared, jnp.array(True, dtype=jnp.bool_)


def _pairwise_sum(values: Array) -> Array:
    """Sum a nonempty flat float32 array through an explicit balanced tree."""
    level = jnp.ravel(values)
    level_size = level.size
    while level_size > 1:
        if level_size % 2:
            level = jnp.pad(level, (0, 1))
            level_size += 1
        pairs = jnp.reshape(level, (-1, 2))
        level = pairs[:, 0] + pairs[:, 1]
        level_size //= 2
    return level[0]


def _tree_product_vector(left: Any, right: Any) -> Array:
    """Flatten elementwise products from two matching float PyTrees."""
    products = [
        jnp.ravel(left_leaf) * jnp.ravel(right_leaf)
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    ]
    return jnp.concatenate(products)


def _tree_dot(left: Any, right: Any) -> Array:
    """Return a deterministic balanced-tree inner product for float PyTrees."""
    return _pairwise_sum(_tree_product_vector(left, right))


def _tree_norm_certificate(tree: Any) -> tuple[Array, Array, Array, Array]:
    """Return a scale-safe L2 point, lower/upper bounds, and resolution flag.

    The input leaves are stored float32 values. After max scaling, a fixed
    balanced sum of squares computes the diagnostic point. A standard
    ``gamma_n`` bound covers division, squaring, balanced accumulation, square
    root, and final rescaling; the interval endpoints are then rounded
    outwards. Nonzero values lost to zero/subnormal intermediates fail closed.
    A tree with exactly one nonzero element has an exact norm equal to its
    stored absolute value and therefore receives a zero-width certificate.
    """
    absolute_values = jnp.concatenate(
        [jnp.ravel(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    )
    scale = jnp.max(absolute_values)
    safe_scale = jnp.where(
        jnp.isfinite(scale) & (scale > 0.0),
        scale,
        jnp.array(1.0, dtype=jnp.float32),
    )
    scaled_values = absolute_values / safe_scale
    scaled_squares = scaled_values * scaled_values
    scaled_square_sum = _pairwise_sum(scaled_squares)
    point = scale * jnp.sqrt(jnp.maximum(scaled_square_sum, 0.0))

    reduction_depth = max(0, (scaled_values.size - 1).bit_length())
    roundoff_budget = (reduction_depth + 8) * _FLOAT32_EPSILON
    if roundoff_budget >= 0.5:
        relative_error = jnp.array(jnp.inf, dtype=jnp.float32)
    else:
        relative_error = jnp.asarray(
            roundoff_budget / (1.0 - roundoff_budget),
            dtype=jnp.float32,
        )
    negative_infinity = jnp.array(-jnp.inf, dtype=jnp.float32)
    positive_infinity = jnp.array(jnp.inf, dtype=jnp.float32)
    lower = jnp.nextafter(
        point / (jnp.array(1.0, dtype=jnp.float32) + relative_error),
        negative_infinity,
    )
    upper = jnp.nextafter(
        point / (jnp.array(1.0, dtype=jnp.float32) - relative_error),
        positive_infinity,
    )
    lower = jnp.maximum(lower, scale)

    source_nonzero = absolute_values != 0.0
    nonzero_count = jnp.sum(source_nonzero.astype(jnp.int32))
    exactly_one_nonzero = nonzero_count == 1
    lower = jnp.where(exactly_one_nonzero, scale, lower)
    upper = jnp.where(exactly_one_nonzero, scale, upper)
    point = jnp.where(exactly_one_nonzero, scale, point)
    zero_tree = nonzero_count == 0
    lower = jnp.where(zero_tree, jnp.array(0.0, dtype=jnp.float32), lower)
    upper = jnp.where(zero_tree, jnp.array(0.0, dtype=jnp.float32), upper)
    point = jnp.where(zero_tree, jnp.array(0.0, dtype=jnp.float32), point)

    scaled_underflowed = jnp.any(
        source_nonzero
        & (
            (scaled_values == 0.0)
            | (
                (scaled_values != 0.0)
                & (scaled_values < jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32))
            )
        )
    )
    square_underflowed = jnp.any(
        (scaled_values != 0.0)
        & (
            (scaled_squares == 0.0)
            | (
                (scaled_squares != 0.0)
                & (scaled_squares < jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32))
            )
        )
    )
    scale_subnormal = (scale != 0.0) & (
        scale < jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32)
    )
    resolved = (
        ~scaled_underflowed
        & ~square_underflowed
        & ~scale_subnormal
        & jnp.all(jnp.isfinite(jnp.stack([point, lower, upper])))
        & (lower >= 0.0)
        & (upper >= lower)
        & (zero_tree | (lower > 0.0))
    )
    return point, lower, upper, resolved


def _tree_normalized_dot_certificate(
    left: Any,
    right: Any,
    left_norm: Array,
    right_norm: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return normalized dot, normalized/raw error bounds, and resolution.

    The dot is accumulated through a fixed balanced tree. A conservative
    float32 roundoff bound covers both normalizations, elementwise products,
    the tree depth, and positive absolute-product accumulation. A nonzero
    product lost to underflow or an uncertainty interval containing zero is
    unresolved. Exact elementwise orthogonality (all source products zero)
    remains resolved.
    """
    safe_left_norm = jnp.where(
        jnp.isfinite(left_norm) & (left_norm > 0.0),
        left_norm,
        jnp.array(1.0, dtype=jnp.float32),
    )
    safe_right_norm = jnp.where(
        jnp.isfinite(right_norm) & (right_norm > 0.0),
        right_norm,
        jnp.array(1.0, dtype=jnp.float32),
    )
    normalized_left = jax.tree_util.tree_map(
        lambda leaf: leaf / safe_left_norm,
        left,
    )
    normalized_right = jax.tree_util.tree_map(
        lambda leaf: leaf / safe_right_norm,
        right,
    )
    normalized_products = _tree_product_vector(normalized_left, normalized_right)
    normalized_dot = _pairwise_sum(normalized_products)
    absolute_product_sum = _pairwise_sum(jnp.abs(normalized_products))
    source_product_mask = jnp.concatenate(
        [
            jnp.ravel((left_leaf != 0.0) & (right_leaf != 0.0))
            for left_leaf, right_leaf in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
                strict=True,
            )
        ]
    )
    product_underflowed = jnp.any(
        source_product_mask
        & (
            (normalized_products == 0.0)
            | (
                (jnp.abs(normalized_products) < _FLOAT32_TINY)
                & (normalized_products != 0.0)
            )
        )
    )

    reduction_depth = max(0, (normalized_products.size - 1).bit_length())
    roundoff_budget = (reduction_depth + 6) * _FLOAT32_EPSILON
    if roundoff_budget >= 0.5:
        normalized_error_bound = jnp.array(jnp.inf, dtype=jnp.float32)
    else:
        gamma = roundoff_budget / (1.0 - roundoff_budget)
        # The second factor covers rounding in the positive absolute-product
        # accumulation used to estimate the first bound.
        inflated_gamma = gamma / (1.0 - gamma)
        normalized_error_bound = jnp.asarray(
            inflated_gamma,
            dtype=jnp.float32,
        ) * absolute_product_sum
    if normalized_products.size == 1:
        left_value = jnp.concatenate(
            [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(left)]
        )[0]
        right_value = jnp.concatenate(
            [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(right)]
        )[0]
        both_nonzero = (left_value != 0.0) & (right_value != 0.0)
        same_sign = (left_value > 0.0) == (right_value > 0.0)
        normalized_dot = jnp.where(
            both_nonzero,
            jnp.where(same_sign, 1.0, -1.0),
            0.0,
        ).astype(jnp.float32)
        direction_error_bound = jnp.array(0.0, dtype=jnp.float32)
    else:
        direction_error_bound = normalized_error_bound
    normalized_bound_underflowed = (
        (absolute_product_sum > 0.0) & (normalized_error_bound == 0.0)
    )

    smaller_norm = jnp.minimum(left_norm, right_norm)
    larger_norm = jnp.maximum(left_norm, right_norm)
    first_scale = normalized_error_bound * smaller_norm
    raw_error_bound = first_scale * larger_norm
    raw_error_bound = jnp.where(
        raw_error_bound > 0.0,
        jnp.nextafter(raw_error_bound, jnp.array(jnp.inf, dtype=jnp.float32)),
        raw_error_bound,
    )
    bound_underflowed = (
        (normalized_error_bound > 0.0)
        & (smaller_norm > 0.0)
        & (larger_norm > 0.0)
        & ((first_scale == 0.0) | (raw_error_bound == 0.0))
    )
    dot_resolved = (
        ~product_underflowed
        & ~normalized_bound_underflowed
        & ~bound_underflowed
        & jnp.isfinite(normalized_dot)
        & jnp.isfinite(normalized_error_bound)
        & jnp.isfinite(raw_error_bound)
        & (
            (absolute_product_sum == 0.0)
            | (jnp.abs(normalized_dot) > direction_error_bound)
        )
    )
    return normalized_dot, direction_error_bound, raw_error_bound, dot_resolved


def _descent_alignment_certificate(
    normalized_dot: Array,
    normalized_dot_error: Array,
    left_norm: Array,
    left_norm_lower: Array,
    left_norm_upper: Array,
    right_norm: Array,
    right_norm_lower: Array,
    right_norm_upper: Array,
    dot_resolved: Array,
) -> tuple[Array, Array, Array]:
    """Return point/lower-bound descent cosine and a resolution verdict.

    ``normalized_dot`` is relative to the two computed norm points. Its
    roundoff interval is expanded by the certified norm intervals before the
    sign is negated. The lower endpoint is the only alignment value permitted
    to drive a hard gate or a soft weight.
    """
    negative_infinity = jnp.array(-jnp.inf, dtype=jnp.float32)
    positive_infinity = jnp.array(jnp.inf, dtype=jnp.float32)
    one = jnp.array(1.0, dtype=jnp.float32)
    minus_one = jnp.array(-1.0, dtype=jnp.float32)
    left_nonzero = left_norm_lower > 0.0
    right_nonzero = right_norm_lower > 0.0
    norms_nonzero = left_nonzero & right_nonzero
    safe_left_lower = jnp.where(left_nonzero, left_norm_lower, one)
    safe_left_upper = jnp.where(left_nonzero, left_norm_upper, one)
    safe_right_lower = jnp.where(right_nonzero, right_norm_lower, one)
    safe_right_upper = jnp.where(right_nonzero, right_norm_upper, one)

    dot_lower = jnp.nextafter(
        normalized_dot - normalized_dot_error,
        negative_infinity,
    )
    dot_upper = jnp.nextafter(
        normalized_dot + normalized_dot_error,
        positive_infinity,
    )
    left_ratio_lower = jnp.nextafter(left_norm / safe_left_upper, negative_infinity)
    left_ratio_upper = jnp.nextafter(left_norm / safe_left_lower, positive_infinity)
    right_ratio_lower = jnp.nextafter(
        right_norm / safe_right_upper,
        negative_infinity,
    )
    right_ratio_upper = jnp.nextafter(
        right_norm / safe_right_lower,
        positive_infinity,
    )
    scale_lower = jnp.nextafter(
        left_ratio_lower * right_ratio_lower,
        negative_infinity,
    )
    scale_upper = jnp.nextafter(
        left_ratio_upper * right_ratio_upper,
        positive_infinity,
    )
    quotient_endpoints = jnp.stack(
        [
            dot_lower * scale_lower,
            dot_lower * scale_upper,
            dot_upper * scale_lower,
            dot_upper * scale_upper,
        ]
    )
    cosine_lower = jnp.nextafter(jnp.min(quotient_endpoints), negative_infinity)
    cosine_upper = jnp.nextafter(jnp.max(quotient_endpoints), positive_infinity)
    point = jnp.clip(-normalized_dot, minus_one, one)
    lower = jnp.clip(jnp.nextafter(-cosine_upper, negative_infinity), minus_one, one)
    zero_case = (~norms_nonzero) & (normalized_dot == 0.0)
    point = jnp.where(zero_case, jnp.array(0.0, dtype=jnp.float32), point)
    lower = jnp.where(zero_case, jnp.array(0.0, dtype=jnp.float32), lower)
    resolved = dot_resolved & (
        zero_case
        | (
            norms_nonzero
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        [
                            point,
                            lower,
                            dot_lower,
                            dot_upper,
                            scale_lower,
                            scale_upper,
                            cosine_lower,
                            cosine_upper,
                        ]
                    )
                )
            )
        )
    )
    return point, lower, resolved


def _tree_is_finite(tree: Any) -> Array:
    """Return whether every element of a PyTree is finite."""
    finite = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.all(jnp.stack(finite))


def _detached_flag(value: Array, *, name: str) -> Array:
    """Convert an availability input to one detached scalar boolean."""
    flag = jnp.asarray(value)
    if flag.dtype != jnp.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    if flag.shape != ():
        raise ValueError(f"{name} must be scalar")
    return jax.lax.stop_gradient(flag)


def _learning_value_for_gradient_audit(
    learning_value: LearningValue,
    declared_availability: LearningValueAvailability,
) -> tuple[LearningValue, LearningValueAvailability, Array]:
    """Detach, sanitize, and semantically validate eight available channels.

    The historical ``LearningValue.delight`` field is accepted only when its
    float32 bits exactly equal the declared paper-defined identity
    ``advantage * action_surprisal``. It remains evidence supplied to the
    separate candidate-update safety audit; it never performs Kondo selection.
    """
    channel_names = (
        "advantage",
        "action_surprisal",
        "delight",
        "epistemic_surprise",
        "aleatoric_uncertainty",
        "learning_progress",
        "change_probability",
        "safety_cost",
    )
    values: dict[str, Array] = {}
    valid: dict[str, Array] = {}
    for name in channel_names:
        raw_value = jnp.asarray(getattr(learning_value, name))
        if jnp.issubdtype(raw_value.dtype, jnp.complexfloating):
            raise ValueError(f"learning_value.{name} must be real-valued")
        if not jnp.issubdtype(raw_value.dtype, jnp.floating):
            raise ValueError(f"learning_value.{name} must have a floating dtype")
        value = jnp.asarray(raw_value, dtype=jnp.float32)
        if value.shape != ():
            raise ValueError(f"learning_value.{name} must be scalar for one gradient audit")
        scalar = jax.lax.stop_gradient(value)
        is_valid = jnp.isfinite(scalar)
        if name in (
            "action_surprisal",
            "epistemic_surprise",
            "aleatoric_uncertainty",
        ):
            is_valid = is_valid & (scalar >= 0.0)
        if name == "change_probability":
            is_valid = is_valid & (scalar >= 0.0) & (scalar <= 1.0)
        declared = _detached_flag(
            getattr(declared_availability, name),
            name=f"learning_value_availability.{name}",
        )
        valid[name] = jax.lax.stop_gradient(declared & is_valid)
        values[name] = jax.lax.stop_gradient(
            jnp.where(is_valid, scalar, jnp.array(0.0, dtype=jnp.float32))
        )

    expected_paper_dg_delight = values["advantage"] * values["action_surprisal"]
    expected_paper_dg_bits = jax.lax.bitcast_convert_type(
        expected_paper_dg_delight,
        jnp.uint32,
    )
    supplied_paper_dg_bits = jax.lax.bitcast_convert_type(
        values["delight"],
        jnp.uint32,
    )
    paper_dg_identity_valid = (
        valid["advantage"]
        & valid["action_surprisal"]
        & valid["delight"]
        & jnp.isfinite(expected_paper_dg_delight)
        & (supplied_paper_dg_bits == expected_paper_dg_bits)
    )
    valid["delight"] = jax.lax.stop_gradient(paper_dg_identity_valid)

    detached = LearningValue(**values)
    availability = LearningValueAvailability(**valid)
    complete = jnp.all(jnp.stack([valid[name] for name in channel_names]))
    return detached, availability, jax.lax.stop_gradient(complete)


def assess_gradient_joy(
    candidate: Any,
    evidence: GradientJoyEvidence,
    config: GradientJoyConfig | None = None,
) -> GradientJoyAssessment:
    """Assess whether a candidate gradient/update passes a local joy audit.

    Let ``u`` be the proposed parameter update and let ``g_o``, ``g_r``, and
    ``g_s`` be gradients of an independent objective probe, a retention loss,
    and a safety cost. First-order predicted changes are

    ``delta_o = <g_o, u>``, ``delta_r = <g_r, u>``, and
    ``delta_s = <g_s, u>``.

    Negative changes improve their corresponding minimization objectives.
    Candidate alignments and norm produce a tentative weakest-link weight. The
    elementwise-rounded tentative update receives fresh norm and dot
    certificates. Hard objective, retention, and safety magnitude gates audit
    both the raw candidate and that actual tentative update. This prevents soft scaling
    from silently violating a required improvement and prevents scaling from
    rescuing a raw harmful candidate. Candidate direction and trust gates remain
    valid because a positive weight preserves direction and only shrinks the
    norm. Missing or invalid evidence and any hard-gate failure zero the final
    weight. No learning-value channel is added to another channel.

    Alignment uses scale-safe global norms and a normalized-coordinate dot
    accumulated through a fixed balanced tree. Conservative float32 roundoff
    intervals cover norms, normalization, products, and reduction. Any
    non-finite derived dot/norm, unresolved norm of a nonzero input, nonzero
    sign disagreement between raw and normalized-coordinate dots, or interval
    that cannot resolve a cancellation-sensitive dot invalidates the evidence
    instead of becoming a passing zero. The trust gate consumes the norm upper
    bound, the nonzero floor consumes its lower bound, and positive magnitude
    gates use the conservative edge of the dot interval. At an exact zero
    threshold, both the raw dot and certified normalized direction must permit
    the hard verdict; this still permits a raw dot that underflows to exact zero
    when the normalized sign remains resolved. Alignment gates and their soft
    factors consume the certified cosine lower bound. A configured endpoint
    threshold of exactly ``1.0`` applies its four-machine-epsilon float32
    tolerance to that lower bound; other thresholds remain exact.

    All inputs, diagnostics, the decision, and the returned update are detached.
    This is an optimizer control-plane assessment, not a differentiable
    meta-objective.
    """
    cfg = config or GradientJoyConfig()
    candidate_tree, candidate_structure = _detached_float_tree(
        candidate,
        name="candidate",
    )
    step_size = jnp.asarray(cfg.gradient_step_size, dtype=jnp.float32)
    if cfg.candidate_semantics == "gradient":
        candidate_update = jax.tree_util.tree_map(
            lambda leaf: -step_size * leaf,
            candidate_tree,
        )
    else:
        candidate_update = candidate_tree

    objective_probe, objective_present = _prepare_probe_tree(
        evidence.objective_probe_gradient,
        candidate_update,
        candidate_structure,
        name="objective_probe_gradient",
    )
    retention_probe, retention_present = _prepare_probe_tree(
        evidence.retention_probe_gradient,
        candidate_update,
        candidate_structure,
        name="retention_probe_gradient",
    )
    safety_probe, safety_present = _prepare_probe_tree(
        evidence.safety_cost_gradient,
        candidate_update,
        candidate_structure,
        name="safety_cost_gradient",
    )
    objective_available = (
        _detached_flag(
            evidence.objective_probe_available,
            name="objective_probe_available",
        )
        & objective_present
    )
    retention_available = (
        _detached_flag(
            evidence.retention_probe_available,
            name="retention_probe_available",
        )
        & retention_present
    )
    safety_available = (
        _detached_flag(
            evidence.safety_probe_available,
            name="safety_probe_available",
        )
        & safety_present
    )
    probe_independence_attested = _detached_flag(
        evidence.probe_independence_attested,
        name="probe_independence_attested",
    )
    (
        detached_learning_value,
        channel_availability,
        learning_value_complete,
    ) = _learning_value_for_gradient_audit(
        evidence.learning_value,
        evidence.learning_value_availability,
    )

    candidate_finite = _tree_is_finite(candidate_update)
    objective_probe_finite = _tree_is_finite(objective_probe)
    retention_probe_finite = _tree_is_finite(retention_probe)
    safety_probe_finite = _tree_is_finite(safety_probe)
    input_evidence_complete = (
        candidate_finite
        & objective_available
        & retention_available
        & safety_available
        & probe_independence_attested
        & objective_probe_finite
        & retention_probe_finite
        & safety_probe_finite
        & learning_value_complete
    )

    (
        update_norm_raw,
        update_norm_lower_raw,
        update_norm_upper_raw,
        update_norm_resolved,
    ) = _tree_norm_certificate(candidate_update)
    (
        objective_norm_raw,
        objective_norm_lower_raw,
        objective_norm_upper_raw,
        objective_norm_resolved,
    ) = _tree_norm_certificate(objective_probe)
    (
        retention_norm_raw,
        retention_norm_lower_raw,
        retention_norm_upper_raw,
        retention_norm_resolved,
    ) = _tree_norm_certificate(retention_probe)
    (
        safety_norm_raw,
        safety_norm_lower_raw,
        safety_norm_upper_raw,
        safety_norm_resolved,
    ) = _tree_norm_certificate(safety_probe)
    objective_change_raw = _tree_dot(objective_probe, candidate_update)
    retention_change_raw = _tree_dot(retention_probe, candidate_update)
    safety_change_raw = _tree_dot(safety_probe, candidate_update)
    (
        objective_direction_raw,
        objective_direction_error_raw,
        objective_dot_error_bound_raw,
        objective_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        objective_probe,
        candidate_update,
        objective_norm_raw,
        update_norm_raw,
    )
    (
        retention_direction_raw,
        retention_direction_error_raw,
        retention_dot_error_bound_raw,
        retention_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        retention_probe,
        candidate_update,
        retention_norm_raw,
        update_norm_raw,
    )
    (
        safety_direction_raw,
        safety_direction_error_raw,
        safety_dot_error_bound_raw,
        safety_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        safety_probe,
        candidate_update,
        safety_norm_raw,
        update_norm_raw,
    )

    def _dot_signs_are_consistent(raw_dot: Array, normalized_dot: Array) -> Array:
        """Reject cancellation-sensitive dots whose nonzero signs disagree."""
        return ~(
            ((raw_dot > 0.0) & (normalized_dot < 0.0))
            | ((raw_dot < 0.0) & (normalized_dot > 0.0))
        )

    derived_numerics_valid = (
        jnp.all(
            jnp.isfinite(
                jnp.stack(
                    [
                        update_norm_raw,
                        objective_norm_raw,
                        retention_norm_raw,
                        safety_norm_raw,
                        objective_change_raw,
                        retention_change_raw,
                        safety_change_raw,
                        objective_direction_raw,
                        retention_direction_raw,
                        safety_direction_raw,
                        objective_dot_error_bound_raw,
                        retention_dot_error_bound_raw,
                        safety_dot_error_bound_raw,
                        objective_direction_error_raw,
                        retention_direction_error_raw,
                        safety_direction_error_raw,
                        update_norm_lower_raw,
                        update_norm_upper_raw,
                        objective_norm_lower_raw,
                        objective_norm_upper_raw,
                        retention_norm_lower_raw,
                        retention_norm_upper_raw,
                        safety_norm_lower_raw,
                        safety_norm_upper_raw,
                    ]
                )
            )
        )
        & update_norm_resolved
        & objective_norm_resolved
        & retention_norm_resolved
        & safety_norm_resolved
        & _dot_signs_are_consistent(objective_change_raw, objective_direction_raw)
        & _dot_signs_are_consistent(retention_change_raw, retention_direction_raw)
        & _dot_signs_are_consistent(safety_change_raw, safety_direction_raw)
        & objective_dot_resolved
        & retention_dot_resolved
        & safety_dot_resolved
    )
    evidence_complete = input_evidence_complete & derived_numerics_valid

    def _finite_or_zero(value: Array) -> Array:
        return jnp.where(
            jnp.isfinite(value),
            value,
            jnp.array(0.0, dtype=jnp.float32),
        )

    update_norm = _finite_or_zero(update_norm_raw)
    update_norm_lower = _finite_or_zero(update_norm_lower_raw)
    update_norm_upper = _finite_or_zero(update_norm_upper_raw)
    objective_norm = _finite_or_zero(objective_norm_raw)
    objective_norm_lower = _finite_or_zero(objective_norm_lower_raw)
    objective_norm_upper = _finite_or_zero(objective_norm_upper_raw)
    retention_norm = _finite_or_zero(retention_norm_raw)
    retention_norm_lower = _finite_or_zero(retention_norm_lower_raw)
    retention_norm_upper = _finite_or_zero(retention_norm_upper_raw)
    safety_norm = _finite_or_zero(safety_norm_raw)
    safety_norm_lower = _finite_or_zero(safety_norm_lower_raw)
    safety_norm_upper = _finite_or_zero(safety_norm_upper_raw)
    objective_change = _finite_or_zero(objective_change_raw)
    retention_change = _finite_or_zero(retention_change_raw)
    safety_change = _finite_or_zero(safety_change_raw)
    objective_dot_error_bound = _finite_or_zero(objective_dot_error_bound_raw)
    retention_dot_error_bound = _finite_or_zero(retention_dot_error_bound_raw)
    safety_dot_error_bound = _finite_or_zero(safety_dot_error_bound_raw)
    objective_direction = _finite_or_zero(objective_direction_raw)
    objective_direction_error = _finite_or_zero(objective_direction_error_raw)
    retention_direction = _finite_or_zero(retention_direction_raw)
    retention_direction_error = _finite_or_zero(retention_direction_error_raw)
    safety_direction = _finite_or_zero(safety_direction_raw)
    safety_direction_error = _finite_or_zero(safety_direction_error_raw)
    epsilon = jnp.asarray(cfg.diagnostics_epsilon, dtype=jnp.float32)

    def _meets_alignment_threshold(
        certified_lower_bound: Array,
        configured_threshold: float,
    ) -> Array:
        threshold = jnp.asarray(configured_threshold, dtype=jnp.float32)
        endpoint_tolerance = jnp.asarray(
            _ALIGNMENT_ENDPOINT_TOLERANCE,
            dtype=jnp.float32,
        )
        effective_threshold = jnp.where(
            threshold == jnp.array(1.0, dtype=jnp.float32),
            threshold - endpoint_tolerance,
            threshold,
        )
        return certified_lower_bound >= effective_threshold

    (
        objective_alignment,
        objective_alignment_lower,
        objective_alignment_resolved,
    ) = _descent_alignment_certificate(
        objective_direction,
        objective_direction_error,
        objective_norm,
        objective_norm_lower,
        objective_norm_upper,
        update_norm,
        update_norm_lower,
        update_norm_upper,
        objective_dot_resolved,
    )
    (
        retention_alignment,
        retention_alignment_lower,
        retention_alignment_resolved,
    ) = _descent_alignment_certificate(
        retention_direction,
        retention_direction_error,
        retention_norm,
        retention_norm_lower,
        retention_norm_upper,
        update_norm,
        update_norm_lower,
        update_norm_upper,
        retention_dot_resolved,
    )
    (
        safety_alignment,
        safety_alignment_lower,
        safety_alignment_resolved,
    ) = _descent_alignment_certificate(
        safety_direction,
        safety_direction_error,
        safety_norm,
        safety_norm_lower,
        safety_norm_upper,
        update_norm,
        update_norm_lower,
        update_norm_upper,
        safety_dot_resolved,
    )
    derived_numerics_valid = derived_numerics_valid & (
        objective_alignment_resolved
        & retention_alignment_resolved
        & safety_alignment_resolved
    )
    evidence_complete = input_evidence_complete & derived_numerics_valid
    predicted_objective_decrease = -objective_change
    nonzero_update = update_norm_lower > epsilon
    within_trust_region = nonzero_update & (
        update_norm_upper <= jnp.asarray(cfg.max_update_norm, dtype=jnp.float32)
    )

    alignment_temperature = jnp.asarray(
        cfg.alignment_temperature,
        dtype=jnp.float32,
    )
    norm_temperature = jnp.asarray(cfg.norm_temperature, dtype=jnp.float32)
    objective_factor = jax.nn.sigmoid(
        (
            objective_alignment_lower
            - jnp.asarray(
                cfg.min_objective_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    retention_factor = jax.nn.sigmoid(
        (
            retention_alignment_lower
            - jnp.asarray(
                cfg.min_retention_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    safety_factor = jax.nn.sigmoid(
        (
            safety_alignment_lower
            - jnp.asarray(
                cfg.min_safety_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    trust_factor = jax.nn.sigmoid(
        (jnp.asarray(cfg.max_update_norm, dtype=jnp.float32) - update_norm_upper)
        / norm_temperature
    )
    weakest_link_weight = jnp.min(
        jnp.stack(
            [
                objective_factor,
                retention_factor,
                safety_factor,
                trust_factor,
            ]
        )
    )
    tentative_weight = jax.lax.stop_gradient(weakest_link_weight)
    safe_update = jax.tree_util.tree_map(
        lambda leaf: jnp.where(jnp.isfinite(leaf), leaf, jnp.zeros_like(leaf)),
        candidate_update,
    )
    tentative_update = jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(tentative_weight * leaf),
        safe_update,
    )
    (
        tentative_update_norm_raw,
        tentative_update_norm_lower_raw,
        tentative_update_norm_upper_raw,
        tentative_update_norm_resolved,
    ) = _tree_norm_certificate(tentative_update)
    tentative_objective_change_raw = _tree_dot(objective_probe, tentative_update)
    tentative_retention_change_raw = _tree_dot(retention_probe, tentative_update)
    tentative_safety_change_raw = _tree_dot(safety_probe, tentative_update)
    (
        tentative_objective_direction_raw,
        _,
        tentative_objective_dot_error_bound_raw,
        tentative_objective_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        objective_probe,
        tentative_update,
        objective_norm_raw,
        tentative_update_norm_raw,
    )
    (
        tentative_retention_direction_raw,
        _,
        tentative_retention_dot_error_bound_raw,
        tentative_retention_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        retention_probe,
        tentative_update,
        retention_norm_raw,
        tentative_update_norm_raw,
    )
    (
        tentative_safety_direction_raw,
        _,
        tentative_safety_dot_error_bound_raw,
        tentative_safety_dot_resolved,
    ) = _tree_normalized_dot_certificate(
        safety_probe,
        tentative_update,
        safety_norm_raw,
        tentative_update_norm_raw,
    )

    tentative_derived_numerics_valid = (
        jnp.all(
            jnp.isfinite(
                jnp.stack(
                    [
                        tentative_update_norm_raw,
                        tentative_update_norm_lower_raw,
                        tentative_update_norm_upper_raw,
                        tentative_objective_change_raw,
                        tentative_retention_change_raw,
                        tentative_safety_change_raw,
                        tentative_objective_direction_raw,
                        tentative_retention_direction_raw,
                        tentative_safety_direction_raw,
                        tentative_objective_dot_error_bound_raw,
                        tentative_retention_dot_error_bound_raw,
                        tentative_safety_dot_error_bound_raw,
                    ]
                )
            )
        )
        & tentative_update_norm_resolved
        & _dot_signs_are_consistent(
            tentative_objective_change_raw,
            tentative_objective_direction_raw,
        )
        & _dot_signs_are_consistent(
            tentative_retention_change_raw,
            tentative_retention_direction_raw,
        )
        & _dot_signs_are_consistent(
            tentative_safety_change_raw,
            tentative_safety_direction_raw,
        )
        & tentative_objective_dot_resolved
        & tentative_retention_dot_resolved
        & tentative_safety_dot_resolved
    )
    derived_numerics_valid = derived_numerics_valid & tentative_derived_numerics_valid
    evidence_complete = input_evidence_complete & derived_numerics_valid

    tentative_update_norm = _finite_or_zero(tentative_update_norm_raw)
    tentative_update_norm_lower = _finite_or_zero(tentative_update_norm_lower_raw)
    tentative_update_norm_upper = _finite_or_zero(tentative_update_norm_upper_raw)
    tentative_objective_change = _finite_or_zero(tentative_objective_change_raw)
    tentative_retention_change = _finite_or_zero(tentative_retention_change_raw)
    tentative_safety_change = _finite_or_zero(tentative_safety_change_raw)
    tentative_objective_direction = _finite_or_zero(tentative_objective_direction_raw)
    tentative_retention_direction = _finite_or_zero(tentative_retention_direction_raw)
    tentative_safety_direction = _finite_or_zero(tentative_safety_direction_raw)
    tentative_objective_dot_error_bound = _finite_or_zero(
        tentative_objective_dot_error_bound_raw
    )
    tentative_retention_dot_error_bound = _finite_or_zero(
        tentative_retention_dot_error_bound_raw
    )
    tentative_safety_dot_error_bound = _finite_or_zero(
        tentative_safety_dot_error_bound_raw
    )
    predicted_tentative_objective_decrease = -tentative_objective_change
    tentative_nonzero_update = tentative_update_norm_lower > epsilon

    def _strict_decrease_exceeds(
        predicted_decrease: Array,
        directional_change: Array,
        dot_error_bound: Array,
        dot_resolved: Array,
        configured_threshold: float,
    ) -> Array:
        threshold = jnp.asarray(configured_threshold, dtype=jnp.float32)
        conservative_decrease = jnp.nextafter(
            predicted_decrease - dot_error_bound,
            jnp.array(-jnp.inf, dtype=jnp.float32),
        )
        return dot_resolved & jnp.where(
            threshold == 0.0,
            (predicted_decrease >= 0.0) & (directional_change < 0.0),
            conservative_decrease > threshold,
        )

    def _change_within_limit(
        change: Array,
        directional_change: Array,
        dot_error_bound: Array,
        dot_resolved: Array,
        configured_limit: float,
    ) -> Array:
        limit = jnp.asarray(configured_limit, dtype=jnp.float32)
        conservative_change = jnp.nextafter(
            change + dot_error_bound,
            jnp.array(jnp.inf, dtype=jnp.float32),
        )
        return dot_resolved & jnp.where(
            limit == 0.0,
            (change <= 0.0) & (directional_change <= 0.0),
            conservative_change <= limit,
        )

    objective_improves = (
        (objective_norm > 0.0)
        & _strict_decrease_exceeds(
            predicted_objective_decrease,
            objective_direction,
            objective_dot_error_bound,
            objective_dot_resolved,
            cfg.min_objective_decrease,
        )
        & _strict_decrease_exceeds(
            predicted_tentative_objective_decrease,
            tentative_objective_direction,
            tentative_objective_dot_error_bound,
            tentative_objective_dot_resolved,
            cfg.min_objective_decrease,
        )
        & _meets_alignment_threshold(
            objective_alignment_lower,
            cfg.min_objective_descent_alignment,
        )
    )
    retention_preserved = (
        _change_within_limit(
            retention_change,
            retention_direction,
            retention_dot_error_bound,
            retention_dot_resolved,
            cfg.max_retention_loss_increase,
        )
        & _change_within_limit(
            tentative_retention_change,
            tentative_retention_direction,
            tentative_retention_dot_error_bound,
            tentative_retention_dot_resolved,
            cfg.max_retention_loss_increase,
        )
    ) & _meets_alignment_threshold(
        retention_alignment_lower,
        cfg.min_retention_descent_alignment,
    )
    safety_preserved = (
        _change_within_limit(
            safety_change,
            safety_direction,
            safety_dot_error_bound,
            safety_dot_resolved,
            cfg.max_safety_cost_increase,
        )
        & _change_within_limit(
            tentative_safety_change,
            tentative_safety_direction,
            tentative_safety_dot_error_bound,
            tentative_safety_dot_resolved,
            cfg.max_safety_cost_increase,
        )
    ) & _meets_alignment_threshold(
        safety_alignment_lower,
        cfg.min_safety_descent_alignment,
    )
    accepted = (
        evidence_complete
        & objective_improves
        & retention_preserved
        & safety_preserved
        & within_trust_region
        & tentative_nonzero_update
    )
    weight = jax.lax.stop_gradient(
        jnp.where(
            accepted,
            weakest_link_weight,
            jnp.array(0.0, dtype=jnp.float32),
        )
    )
    accepted = jax.lax.stop_gradient(accepted)
    weighted_update = jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(
            jnp.where(accepted, leaf, jnp.zeros_like(leaf))
        ),
        tentative_update,
    )
    safe_update = jax.tree_util.tree_map(jax.lax.stop_gradient, safe_update)

    diagnostics = GradientJoyDiagnostics(
        objective_probe_available=jax.lax.stop_gradient(objective_available),
        retention_probe_available=jax.lax.stop_gradient(retention_available),
        safety_probe_available=jax.lax.stop_gradient(safety_available),
        probe_independence_attested=jax.lax.stop_gradient(probe_independence_attested),
        candidate_finite=jax.lax.stop_gradient(candidate_finite),
        objective_probe_finite=jax.lax.stop_gradient(objective_probe_finite),
        retention_probe_finite=jax.lax.stop_gradient(retention_probe_finite),
        safety_probe_finite=jax.lax.stop_gradient(safety_probe_finite),
        learning_value_complete=learning_value_complete,
        derived_numerics_valid=jax.lax.stop_gradient(derived_numerics_valid),
        evidence_complete=jax.lax.stop_gradient(evidence_complete),
        nonzero_update=jax.lax.stop_gradient(nonzero_update),
        tentative_nonzero_update=jax.lax.stop_gradient(tentative_nonzero_update),
        objective_improves=jax.lax.stop_gradient(objective_improves),
        retention_preserved=jax.lax.stop_gradient(retention_preserved),
        safety_preserved=jax.lax.stop_gradient(safety_preserved),
        within_trust_region=jax.lax.stop_gradient(within_trust_region),
        update_norm=jax.lax.stop_gradient(update_norm),
        update_norm_lower_bound=jax.lax.stop_gradient(update_norm_lower),
        update_norm_upper_bound=jax.lax.stop_gradient(update_norm_upper),
        update_norm_resolved=jax.lax.stop_gradient(update_norm_resolved),
        objective_probe_norm=jax.lax.stop_gradient(objective_norm),
        retention_probe_norm=jax.lax.stop_gradient(retention_norm),
        safety_probe_norm=jax.lax.stop_gradient(safety_norm),
        predicted_objective_decrease=jax.lax.stop_gradient(predicted_objective_decrease),
        predicted_retention_loss_change=jax.lax.stop_gradient(retention_change),
        predicted_safety_cost_change=jax.lax.stop_gradient(safety_change),
        objective_dot_error_bound=jax.lax.stop_gradient(objective_dot_error_bound),
        retention_dot_error_bound=jax.lax.stop_gradient(retention_dot_error_bound),
        safety_dot_error_bound=jax.lax.stop_gradient(safety_dot_error_bound),
        objective_dot_resolved=jax.lax.stop_gradient(objective_dot_resolved),
        retention_dot_resolved=jax.lax.stop_gradient(retention_dot_resolved),
        safety_dot_resolved=jax.lax.stop_gradient(safety_dot_resolved),
        objective_descent_alignment=jax.lax.stop_gradient(objective_alignment),
        retention_descent_alignment=jax.lax.stop_gradient(retention_alignment),
        safety_descent_alignment=jax.lax.stop_gradient(safety_alignment),
        objective_descent_alignment_lower_bound=jax.lax.stop_gradient(
            objective_alignment_lower
        ),
        retention_descent_alignment_lower_bound=jax.lax.stop_gradient(
            retention_alignment_lower
        ),
        safety_descent_alignment_lower_bound=jax.lax.stop_gradient(
            safety_alignment_lower
        ),
        objective_factor=jax.lax.stop_gradient(objective_factor),
        retention_factor=jax.lax.stop_gradient(retention_factor),
        safety_factor=jax.lax.stop_gradient(safety_factor),
        trust_factor=jax.lax.stop_gradient(trust_factor),
        tentative_weight=tentative_weight,
        tentative_update_norm=jax.lax.stop_gradient(tentative_update_norm),
        tentative_update_norm_lower_bound=jax.lax.stop_gradient(
            tentative_update_norm_lower
        ),
        tentative_update_norm_upper_bound=jax.lax.stop_gradient(
            tentative_update_norm_upper
        ),
        tentative_update_norm_resolved=jax.lax.stop_gradient(
            tentative_update_norm_resolved
        ),
        tentative_objective_dot_error_bound=jax.lax.stop_gradient(
            tentative_objective_dot_error_bound
        ),
        tentative_retention_dot_error_bound=jax.lax.stop_gradient(
            tentative_retention_dot_error_bound
        ),
        tentative_safety_dot_error_bound=jax.lax.stop_gradient(
            tentative_safety_dot_error_bound
        ),
        tentative_objective_dot_resolved=jax.lax.stop_gradient(
            tentative_objective_dot_resolved
        ),
        tentative_retention_dot_resolved=jax.lax.stop_gradient(
            tentative_retention_dot_resolved
        ),
        tentative_safety_dot_resolved=jax.lax.stop_gradient(
            tentative_safety_dot_resolved
        ),
        predicted_tentative_objective_decrease=jax.lax.stop_gradient(
            predicted_tentative_objective_decrease
        ),
        predicted_tentative_retention_loss_change=jax.lax.stop_gradient(
            tentative_retention_change
        ),
        predicted_tentative_safety_cost_change=jax.lax.stop_gradient(
            tentative_safety_change
        ),
    )
    return GradientJoyAssessment(
        accepted=accepted,
        weight=weight,
        candidate_update=safe_update,
        weighted_update=weighted_update,
        learning_value=detached_learning_value,
        channel_availability=channel_availability,
        diagnostics=diagnostics,
    )


def apply_gradient_joy_update(
    parameters: Any,
    candidate: Any,
    evidence: GradientJoyEvidence,
    config: GradientJoyConfig | None = None,
) -> GradientJoyApplicationResult:
    """Assess and atomically apply one joy-weighted parameter update.

    The assessment is created internally so callers cannot substitute a forged
    acceptance decision.  ``parameters`` and the assessed update must be
    non-empty PyTrees with exactly the same structure and leaf shapes; every
    leaf must have a real floating dtype.  These static contract violations
    raise :class:`ValueError` instead of permitting JAX broadcasting.

    Update leaves are cast to their corresponding parameter dtypes before
    addition. The effective stored delta (proposed parameters minus input
    parameters) is computed after promoting both stored endpoints to at least
    float32, then independently re-audited under ``candidate_semantics =
    "update"`` with the same evidence and thresholds. Endpoint promotion
    prevents float16/bfloat16 subtraction rounding from understating a stored
    change. This conservative second audit can veto a finite quantized update
    whose direction, magnitude, or trust-bound status no longer matches the
    accepted float32 candidate.

    The update is applied only when both assessments accept it, every parameter,
    cast update, and proposed parameter is finite, and at least one proposed
    stored value differs from its input. Otherwise every parameter leaf is
    returned unchanged, so a rejection, unsafe cast, overflow, non-finite leaf,
    or update wholly lost to finite-precision addition cannot produce or claim
    a partial update. Existing non-finite parameters are not repaired; they are
    preserved without applying any part of the candidate.

    The typed result keeps the original and effective audit decisions distinct
    from ``applied`` and exposes detached scalar booleans for every application
    check. The helper is compatible with :func:`jax.jit` when the PyTree
    structures and ``config`` are static.
    """
    assessment = assess_gradient_joy(candidate, evidence, config)
    parameter_structure: Any
    update_structure: Any
    parameter_leaves, parameter_structure = jax.tree_util.tree_flatten(parameters)
    update_leaves, update_structure = jax.tree_util.tree_flatten(
        assessment.weighted_update
    )
    if not parameter_leaves:
        raise ValueError("parameters must be a non-empty PyTree of floating arrays")
    if parameter_structure != update_structure:
        raise ValueError("parameter and joy-update PyTree structures must match exactly")

    prepared_parameters: list[Array] = []
    prepared_updates: list[Array] = []
    for position, (parameter_leaf, update_leaf) in enumerate(
        zip(parameter_leaves, update_leaves, strict=True)
    ):
        parameter_array = jnp.asarray(parameter_leaf)
        update_array = jnp.asarray(update_leaf)
        if not jnp.issubdtype(parameter_array.dtype, jnp.floating):
            raise ValueError(
                f"parameter leaf {position} must have a real floating dtype"
            )
        if not jnp.issubdtype(update_array.dtype, jnp.floating):
            raise ValueError(
                f"joy-update leaf {position} must have a real floating dtype"
            )
        if parameter_array.shape != update_array.shape:
            raise ValueError(
                f"parameter and joy-update leaf {position} shapes must match exactly"
            )
        prepared_parameters.append(parameter_array)
        prepared_updates.append(
            jnp.asarray(update_array, dtype=parameter_array.dtype)
        )

    parameter_tree = jax.tree_util.tree_unflatten(
        parameter_structure,
        prepared_parameters,
    )
    update_tree = jax.tree_util.tree_unflatten(
        parameter_structure,
        prepared_updates,
    )
    proposed_parameters = jax.tree_util.tree_map(
        lambda parameter, update: parameter + update,
        parameter_tree,
        update_tree,
    )
    def _stored_endpoint_delta(parameter: Array, proposed: Array) -> Array:
        """Subtract stored endpoints without low-precision subtraction rounding."""
        audit_dtype = jnp.promote_types(parameter.dtype, jnp.float32)
        return jnp.asarray(proposed, dtype=audit_dtype) - jnp.asarray(
            parameter,
            dtype=audit_dtype,
        )

    effective_update = jax.tree_util.tree_map(
        _stored_endpoint_delta,
        parameter_tree,
        proposed_parameters,
    )
    effective_config = dataclasses.replace(
        config or GradientJoyConfig(),
        candidate_semantics="update",
    )
    effective_assessment = assess_gradient_joy(
        effective_update,
        evidence,
        effective_config,
    )
    parameters_finite = jax.lax.stop_gradient(_tree_is_finite(parameter_tree))
    cast_update_finite = jax.lax.stop_gradient(_tree_is_finite(update_tree))
    proposed_parameters_finite = jax.lax.stop_gradient(
        _tree_is_finite(proposed_parameters)
    )
    proposed_change_raw = jnp.any(
        jnp.stack(
            [
                jnp.any(proposed != parameter)
                for parameter, proposed in zip(
                    jax.tree_util.tree_leaves(parameter_tree),
                    jax.tree_util.tree_leaves(proposed_parameters),
                    strict=True,
                )
            ]
        )
    )
    proposed_change_nonzero = jax.lax.stop_gradient(proposed_change_raw)
    applied = jax.lax.stop_gradient(
        assessment.accepted
        & effective_assessment.accepted
        & parameters_finite
        & cast_update_finite
        & proposed_parameters_finite
        & proposed_change_nonzero
    )
    updated_parameters = jax.tree_util.tree_map(
        lambda parameter, proposed: jnp.where(
            applied,
            proposed,
            parameter,
        ),
        parameter_tree,
        proposed_parameters,
    )
    return GradientJoyApplicationResult(
        parameters=updated_parameters,
        assessment=assessment,
        effective_assessment=effective_assessment,
        applied=applied,
        parameters_finite=parameters_finite,
        cast_update_finite=cast_update_finite,
        proposed_parameters_finite=proposed_parameters_finite,
        proposed_change_nonzero=proposed_change_nonzero,
    )


@dataclasses.dataclass(frozen=True)
class DelightfulPolicyGradientConfig:
    """Configuration for the discrete Delightful Policy Gradient experiment.

    Args:
        mode: ``"ordinary_pg"`` applies the matched ordinary policy-gradient
            loss. ``"delightful_pg"`` applies the detached sigmoid
            paper-defined delight
            weight.
        temperature: Positive sigmoid temperature in return-nats.
        actor_trace_lambda: Must be exactly zero. A current-sample
            paper-defined delight gate
            cannot multiply an eligibility trace containing historical score
            gradients.
        diagnostics_epsilon: Positive denominator floor used only in reported
            diagnostics.
        kondo_enabled: Reserved fail-closed flag. ``True`` is rejected because
            this full-batch loss helper cannot skip compiled backward work;
            perform selection and gather before invoking autodiff instead.

    Nonzero numeric controls must be finite normal float32 values; this prevents
    a positive Python value from becoming zero or infinity inside the JAX loss.
    """

    mode: PolicyGradientMode = "delightful_pg"
    temperature: float = 1.0
    actor_trace_lambda: float = 0.0
    diagnostics_epsilon: float = 1.0e-8
    kondo_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate the experimental contract."""
        object.__setattr__(
            self, "mode", _require_choice("mode", self.mode, ("ordinary_pg", "delightful_pg"))
        )
        object.__setattr__(
            self, "temperature", _validated_config_float("temperature", self.temperature, lower=0.0)
        )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(
            self,
            "actor_trace_lambda",
            _validated_config_float("actor_trace_lambda", self.actor_trace_lambda),
        )
        if self.actor_trace_lambda != 0.0:
            raise ValueError(
                "Delightful Policy Gradient requires actor_trace_lambda=0; "
                "a current-sample gate cannot reweight historical score gradients"
            )
        object.__setattr__(
            self,
            "diagnostics_epsilon",
            _validated_config_float("diagnostics_epsilon", self.diagnostics_epsilon, lower=0.0),
        )
        if self.diagnostics_epsilon <= 0.0:
            raise ValueError("diagnostics_epsilon must be positive")
        object.__setattr__(
            self, "kondo_enabled", _require_bool("kondo_enabled", self.kondo_enabled)
        )
        if self.kondo_enabled:
            raise ValueError(
                "Kondo compute gating is unavailable in this full-batch helper; "
                "screen and gather before invoking autodiff"
            )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "DelightfulPolicyGradientConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> DelightfulPolicyGradientConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = _copy_mapping(config, name="DelightfulPolicyGradientConfig")
        type_value = payload.pop("type", None)
        if type(type_value) is not str or type_value != "DelightfulPolicyGradientConfig":
            raise ValueError("DelightfulPolicyGradientConfig type is unsupported")
        allowed = {
            "mode",
            "temperature",
            "actor_trace_lambda",
            "diagnostics_epsilon",
            "kondo_enabled",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"DelightfulPolicyGradientConfig has unsupported fields: {unknown}")
        return cls(**payload)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class DelightfulPolicyGradientDiagnostics:
    """Batch diagnostics for a policy-gradient gate.

    Conditional gate rates are mean soft gate probabilities, not claims of
    skipped gradient computations. Effective sample size is
    ``sum(w)^2 / sum(w^2)``.
    """

    effective_sample_size: Float[Array, ""]
    effective_sample_fraction: Float[Array, ""]
    mean_gate_rate: Float[Array, ""]
    positive_advantage_gate_rate: Float[Array, ""]
    negative_advantage_gate_rate: Float[Array, ""]
    positive_advantage_fraction: Float[Array, ""]
    negative_advantage_fraction: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulPolicyGradientResult:
    """Actor loss, typed actor signals, and detached sample coefficients.

    The ``delight`` field is the paper-defined actor-sample quantity, not the
    separate candidate-update safety-audit verdict.
    """

    actor_loss: Float[Array, ""]
    ordinary_actor_loss: Float[Array, ""]
    advantage: Array
    action_surprisal: Array
    delight: Array
    sample_weights: Array
    actor_coefficients: Array
    diagnostics: DelightfulPolicyGradientDiagnostics


@chex.dataclass(frozen=True)
class DelightStratumDiagnostics:
    """Diagnostics for one of common/rare × success/failure.

    ``mean_delight`` means mean paper-specific DG actor-sample delight.
    """

    count: Int[Array, ""]
    fraction: Float[Array, ""]
    mean_advantage: Float[Array, ""]
    mean_action_surprisal: Float[Array, ""]
    mean_delight: Float[Array, ""]
    mean_gate_rate: Float[Array, ""]
    mean_gated_advantage: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightOutcomeStratification:
    """Rare/common success/failure diagnostics for gambling audits."""

    common_success: DelightStratumDiagnostics
    common_failure: DelightStratumDiagnostics
    rare_success: DelightStratumDiagnostics
    rare_failure: DelightStratumDiagnostics
    common_advantage_mean: Float[Array, ""]
    rare_advantage_mean: Float[Array, ""]
    common_gated_advantage_mean: Float[Array, ""]
    rare_gated_advantage_mean: Float[Array, ""]
    common_advantage_variance: Float[Array, ""]
    rare_advantage_variance: Float[Array, ""]


def _masked_mean(values: Array, mask: Array) -> Array:
    """Return a finite conditional mean, or zero when the mask is empty."""
    value_arr = jnp.ravel(jnp.asarray(values, dtype=jnp.float32))
    mask_arr = jnp.ravel(jnp.asarray(mask, dtype=jnp.bool_))
    mask_float = mask_arr.astype(jnp.float32)
    count = jnp.sum(mask_float)
    return jnp.sum(jnp.where(mask_arr, value_arr, 0.0)) / jnp.maximum(count, 1.0)


def discrete_delightful_policy_gradient(
    selected_log_probabilities: Array,
    advantages: Array,
    config: DelightfulPolicyGradientConfig | None = None,
) -> DelightfulPolicyGradientResult:
    """Return a matched ordinary or delightful discrete actor loss.

    Args:
        selected_log_probabilities: Finite log-probabilities of the sampled
            categorical actions under the explicitly chosen actor objective.
            Off-policy callers must perform their declared importance
            correction separately; paper-defined delight is never a
            substitute for it.
        advantages: Baseline-derived per-sample advantages. The actor
            coefficient is stopped so this loss cannot train the baseline.
        config: Static experimental configuration. Close over it when using
            :func:`jax.jit`.

    Returns:
        Per-sample actor signals, detached coefficients, scalar losses, and
        diagnostics. The selected loss is exactly
        ``-mean(stop_gradient(weight * advantage) * log_probability)``.
    """
    cfg = config or DelightfulPolicyGradientConfig()
    log_prob = jnp.asarray(selected_log_probabilities, dtype=jnp.float32)
    advantage = jnp.asarray(advantages, dtype=jnp.float32)
    if log_prob.shape != advantage.shape:
        raise ValueError("selected_log_probabilities and advantages must have the same shape")
    if log_prob.size == 0:
        raise ValueError("at least one policy-gradient sample is required")

    action_surprisal = -log_prob
    delight = jax.lax.stop_gradient(advantage * action_surprisal)
    if cfg.mode == "ordinary_pg":
        sample_weights = jnp.ones_like(delight)
    else:
        temperature = jnp.asarray(cfg.temperature, dtype=jnp.float32)
        sample_weights = jax.nn.sigmoid(delight / temperature)
    sample_weights = jax.lax.stop_gradient(sample_weights)
    actor_coefficients = jax.lax.stop_gradient(sample_weights * advantage)
    ordinary_coefficients = jax.lax.stop_gradient(advantage)

    actor_loss = -jnp.mean(actor_coefficients * log_prob)
    ordinary_actor_loss = -jnp.mean(ordinary_coefficients * log_prob)

    flat_weights = jnp.ravel(sample_weights)
    flat_advantages = jnp.ravel(advantage)
    positive = flat_advantages > 0.0
    negative = flat_advantages < 0.0
    n_samples = jnp.asarray(flat_weights.size, dtype=jnp.float32)
    epsilon = jnp.asarray(cfg.diagnostics_epsilon, dtype=jnp.float32)
    weight_sum = jnp.sum(flat_weights)
    effective_sample_size = weight_sum**2 / jnp.maximum(
        jnp.sum(flat_weights**2),
        epsilon,
    )
    diagnostics = DelightfulPolicyGradientDiagnostics(
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / jnp.maximum(n_samples, 1.0),
        mean_gate_rate=jnp.mean(flat_weights),
        positive_advantage_gate_rate=_masked_mean(flat_weights, positive),
        negative_advantage_gate_rate=_masked_mean(flat_weights, negative),
        positive_advantage_fraction=jnp.mean(positive.astype(jnp.float32)),
        negative_advantage_fraction=jnp.mean(negative.astype(jnp.float32)),
    )
    return DelightfulPolicyGradientResult(
        actor_loss=actor_loss,
        ordinary_actor_loss=ordinary_actor_loss,
        advantage=advantage,
        action_surprisal=action_surprisal,
        delight=delight,
        sample_weights=sample_weights,
        actor_coefficients=actor_coefficients,
        diagnostics=diagnostics,
    )


def _stratum_diagnostics(
    result: DelightfulPolicyGradientResult,
    mask: Array,
) -> DelightStratumDiagnostics:
    """Summarize one explicit outcome stratum."""
    mask_arr = jnp.ravel(jnp.asarray(mask, dtype=jnp.bool_))
    count = jnp.sum(mask_arr.astype(jnp.int32))
    n_samples = jnp.asarray(jnp.ravel(result.advantage).size, dtype=jnp.float32)
    return DelightStratumDiagnostics(
        count=count,
        fraction=count.astype(jnp.float32) / jnp.maximum(n_samples, 1.0),
        mean_advantage=_masked_mean(result.advantage, mask_arr),
        mean_action_surprisal=_masked_mean(result.action_surprisal, mask_arr),
        mean_delight=_masked_mean(result.delight, mask_arr),
        mean_gate_rate=_masked_mean(result.sample_weights, mask_arr),
        mean_gated_advantage=_masked_mean(result.actor_coefficients, mask_arr),
    )


def stratify_delight_outcomes(
    result: DelightfulPolicyGradientResult,
    rare_mask: Array,
) -> DelightOutcomeStratification:
    """Stratify DG diagnostics into common/rare success/failure groups.

    ``rare_mask`` must be defined by the experiment protocol independently of
    observed reward or advantage—for example, from the behavior action
    probability. Positive advantage denotes success and negative advantage
    denotes failure. Zero-advantage samples are retained in aggregate moments
    but belong to neither success nor failure stratum.
    """
    advantages = jnp.ravel(jnp.asarray(result.advantage, dtype=jnp.float32))
    rare_input = jnp.asarray(rare_mask)
    if rare_input.dtype != jnp.bool_:
        raise ValueError("rare_mask must have boolean dtype")
    rare = jnp.ravel(rare_input)
    if rare.shape != advantages.shape:
        raise ValueError("rare_mask must match the flattened advantage shape")
    common = ~rare
    success = advantages > 0.0
    failure = advantages < 0.0

    common_mean = _masked_mean(advantages, common)
    rare_mean = _masked_mean(advantages, rare)
    common_variance = _masked_mean((advantages - common_mean) ** 2, common)
    rare_variance = _masked_mean((advantages - rare_mean) ** 2, rare)
    return DelightOutcomeStratification(
        common_success=_stratum_diagnostics(result, common & success),
        common_failure=_stratum_diagnostics(result, common & failure),
        rare_success=_stratum_diagnostics(result, rare & success),
        rare_failure=_stratum_diagnostics(result, rare & failure),
        common_advantage_mean=common_mean,
        rare_advantage_mean=rare_mean,
        common_gated_advantage_mean=_masked_mean(result.actor_coefficients, common),
        rare_gated_advantage_mean=_masked_mean(result.actor_coefficients, rare),
        common_advantage_variance=common_variance,
        rare_advantage_variance=rare_variance,
    )


__all__ = [
    "DelightOutcomeStratification",
    "DelightStratumDiagnostics",
    "DelightfulPolicyGradientConfig",
    "DelightfulPolicyGradientDiagnostics",
    "DelightfulPolicyGradientResult",
    "GradientCandidateSemantics",
    "GradientJoyApplicationResult",
    "GradientJoyAssessment",
    "GradientJoyConfig",
    "GradientJoyDiagnostics",
    "GradientJoyEvidence",
    "LearningValue",
    "LearningValueAvailability",
    "PolicyGradientMode",
    "apply_gradient_joy_update",
    "assess_gradient_joy",
    "discrete_delightful_policy_gradient",
    "stratify_delight_outcomes",
]
