# mypy: disable-error-code="call-arg"
"""Bounded contextual partner-message policy fusion.

This module is a development-level (L0) mechanism.  It does not infer a task
or regime identifier, does not establish calibrated confidence, and does not
support an evidence or promotion claim.  A fixed-size online model predicts
partner reliability from bounded context and the partner's declared
confidence.  That model is updated only from a realized assistance value and
an observed safety outcome after a partner-influenced action.

Feedback is deliberately transactional.  A decision can arm at most one
feedback record, containing the exact decision identity, executed event,
action, partner, and model features used at decision time.  Stale, duplicate,
or misattributed feedback returns the byte-for-byte same state.  Mere policy
agreement is neither accepted nor represented as a learning target.

The caller supplies the hard discrete-action safety mask.  No route can emit
an action outside that mask.  Invalid, expired, missing, non-finite, or unsafe
partner messages fall back to the caller's counterfactual base action.  If the
base action itself is not mask-allowed, the decision fails closed with action
``-1`` and leaves state unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework._bounded_containers import require_bounded_container_tree

PARTNER_POLICY_FUSION_CONFIG_SCHEMA = "alberta.partner-policy-fusion.config.v1"
PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA = "alberta.partner-policy-fusion.checkpoint.v1"
MECHANISM_STATUS = "development_l0_mechanism_only"
SCIENTIFIC_PROMOTION_ALLOWED = False

ROUTE_IGNORE = 0
ROUTE_QUERY = 1
ROUTE_ACCEPT = 2
ROUTE_BLEND = 3
ROUTE_REQUEST_CLARIFICATION = 4
ROUTE_COUNT = 5

SOURCE_BASE = 0
SOURCE_OPTION_KEYBOARD = 1
SOURCE_PARTNER = 2

RELIABILITY_HISTORY_UNAVAILABLE = 0
RELIABILITY_HISTORY_DEVELOPMENT_ONLINE = 1

_CONFIG_TYPE = "PartnerPolicyFusionConfig"
_FUSION_TYPE = "PartnerPolicyFusion"
_INT32_MAX = 2_147_483_647
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
# Largest non-INT32 constructor bound in this L0 module (`max_partners`).
# Sequence T is the remaining unbounded scan axis on origin.
_FUSION_SEQUENCE_MAX_STEPS = 1_024
# Same ceiling as security/checkpoints. Origin json.dumps RecursionErrors
# a 16_000-deep fusion mapping during from_checkpoint_payload digest.
_CHECKPOINT_JSON_MAX_DEPTH = 32
_CHECKPOINT_JSON_MAX_NODES = 4096



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
_ACTUAL_FLOAT_TYPES = frozenset((float, np.float16, np.float32, np.float64, np.longdouble))


def _strict_positive_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    """Return a strict positive integer within an explicit bound."""

    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not 1 <= canonical <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return canonical


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_fusion_sequence_vector(name: str, value: object) -> int:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank one")
    length = int(value.shape[0])
    if length < 1 or length > _FUSION_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"{name} length must be an integer in [1, {_FUSION_SEQUENCE_MAX_STEPS}]"
        )
    return length


def _strict_float32(
    value: object,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
    positive: bool = False,
    allow_zero: bool = True,
) -> float:
    """Return a finite normal float32-compatible configuration scalar."""

    actual_type = type(value)
    if actual_type not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a real scalar, not boolean")
    normalized = float(cast(Real, value))
    if not math.isfinite(normalized) or abs(normalized) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and float32-compatible")
    if positive and (normalized < 0.0 or (not allow_zero and normalized == 0.0)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    if normalized != 0.0 and abs(normalized) < _FLOAT32_TINY:
        raise ValueError(f"{name} must be zero or a normal float32 value")
    if lower is not None and normalized < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and normalized > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return normalized


def _json_container_children(node: object) -> tuple[object, ...] | None:
    """Return JSON-container children, or None for a scalar leaf.

    Uses :mod:`collections.abc` checks aligned with ``json.dumps`` so that
    dict/list/tuple subclasses are treated as containers, not scalars.
    """

    if isinstance(node, Mapping):
        return tuple(cast(Mapping[str, Any], node).values())
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return tuple(cast(Sequence[object], node))
    return None


def _canonical_json_bytes(payload: object) -> bytes:
    """Return the canonical encoding used by checkpoint digests."""

    require_bounded_container_tree(
        payload,
        children=_json_container_children,
        max_depth=_CHECKPOINT_JSON_MAX_DEPTH,
        max_nodes=_CHECKPOINT_JSON_MAX_NODES,
        name="checkpoint payload",
        kind="JSON",
    )
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except RecursionError as exc:
        raise ValueError("checkpoint payload exceeds the JSON nesting limit") from exc


def _payload_digest(payload: object) -> str:
    """Return a SHA-256 digest for one canonical JSON payload."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _saturating_increment(value: Array, cap: int) -> Array:
    """Increment a non-negative int32 counter without overflow."""

    cap_array = jnp.asarray(cap, dtype=jnp.int32)
    can_increment = value < cap_array
    safe_value = jnp.where(can_increment, value, jnp.int32(0))
    return jnp.where(can_increment, safe_value + jnp.int32(1), cap_array)


def _saturating_sum(values: Array, cap: int) -> Array:
    """Sum non-negative int32 counters without intermediate overflow."""

    cap_array = jnp.asarray(cap, dtype=jnp.int32)

    def add(carry: Array, value: Array) -> tuple[Array, None]:
        remaining = cap_array - carry
        return carry + jnp.minimum(value, remaining), None

    total, _ = jax.lax.scan(add, jnp.asarray(0, dtype=jnp.int32), values)
    return total


def _finite_l2_norm(value: Array) -> Array:
    """Return a finite scale-safe float32 L2 norm."""

    scale = jnp.max(jnp.abs(value))
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = value / safe_scale
    norm = scale * jnp.sqrt(jnp.sum(jnp.square(normalized)))
    return jnp.where(jnp.isfinite(norm), norm, jnp.float32(_FLOAT32_MAX))


def _safe_action_lookup(mask: Array, action: Array, n_actions: int) -> Array:
    """Read an action mask without allowing an out-of-range gather."""

    in_range = (action >= 0) & (action < n_actions)
    safe_index = jnp.clip(action, 0, n_actions - 1)
    return in_range & mask[safe_index]


@dataclasses.dataclass(frozen=True)
class PartnerPolicyFusionConfig:
    """Static dimensions, bounds, routing thresholds, and update rule.

    The reliability feature vector is ``[1, context / max_abs_context,
    declared_confidence]``.  Its logistic prediction is a development-only
    online estimate, not an empirical calibration claim.  Before a partner has
    ``min_feedback_for_learned_routing`` bound outcomes, an ``accept``
    route above the configured threshold is explicitly cold-start development
    exploration for collecting realized feedback.  It is not calibrated
    trust.  Lower-scoring uncalibrated proposals can only route to ``query``
    (within the query-cost bound) or ``ignore``.
    """

    max_partners: int
    context_dim: int
    n_actions: int
    max_message_horizon: int = 8
    min_feedback_for_learned_routing: int = 4
    counter_cap: int = 1_000_000
    learning_rate: float = 0.05
    max_abs_weight: float = 8.0
    max_abs_context: float = 1.0e4
    max_communication_cost: float = 1.0e4
    communication_cost_weight: float = 1.0
    assistance_value_bound: float = 1.0e4
    safety_target_weight: float = 0.5
    accept_net_value_threshold: float = 0.40
    blend_net_value_threshold: float = 0.15
    clarification_confidence_threshold: float = 0.20
    max_query_cost: float = 0.25
    base_blend_weight: float = 1.0
    option_blend_weight: float = 1.0
    partner_blend_weight: float = 1.0
    max_abs_declared_score: float = 1.0e4

    def __post_init__(self) -> None:
        """Reject dynamic dimensions, ambiguous thresholds, and unsafe bounds."""

        object.__setattr__(
            self,
            "max_partners",
            _strict_positive_int(self.max_partners, name="max_partners", maximum=1024),
        )
        object.__setattr__(
            self,
            "context_dim",
            _strict_positive_int(self.context_dim, name="context_dim", maximum=65_536),
        )
        object.__setattr__(
            self,
            "n_actions",
            _strict_positive_int(self.n_actions, name="n_actions", maximum=65_536),
        )
        object.__setattr__(
            self,
            "max_message_horizon",
            _strict_positive_int(
                self.max_message_horizon,
                name="max_message_horizon",
                maximum=_INT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "min_feedback_for_learned_routing",
            _strict_positive_int(
                self.min_feedback_for_learned_routing,
                name="min_feedback_for_learned_routing",
                maximum=_INT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "counter_cap",
            _strict_positive_int(self.counter_cap, name="counter_cap", maximum=_INT32_MAX),
        )
        if self.min_feedback_for_learned_routing > self.counter_cap:
            raise ValueError("min_feedback_for_learned_routing exceeds counter_cap")

        for name in (
            "learning_rate",
            "max_abs_weight",
            "max_abs_context",
            "max_communication_cost",
            "assistance_value_bound",
            "max_abs_declared_score",
        ):
            object.__setattr__(
                self,
                name,
                _strict_float32(
                    getattr(self, name),
                    name=name,
                    positive=True,
                    allow_zero=False,
                ),
            )
        for name in (
            "communication_cost_weight",
            "base_blend_weight",
            "option_blend_weight",
            "partner_blend_weight",
        ):
            object.__setattr__(
                self,
                name,
                _strict_float32(
                    getattr(self, name),
                    name=name,
                    positive=True,
                    allow_zero=True,
                ),
            )
        for name, lower, upper in (
            ("safety_target_weight", 0.0, 1.0),
            ("clarification_confidence_threshold", 0.0, 1.0),
            ("max_query_cost", 0.0, float(self.max_communication_cost)),
            ("blend_net_value_threshold", None, None),
            ("accept_net_value_threshold", None, None),
        ):
            object.__setattr__(
                self,
                name,
                _strict_float32(getattr(self, name), name=name, lower=lower, upper=upper),
            )
        if self.blend_net_value_threshold >= self.accept_net_value_threshold:
            raise ValueError("blend_net_value_threshold must be below accept threshold")

        feature_width = self.model_feature_dim
        partners = self.max_partners
        persistent_f32 = partners * feature_width + feature_width + 1
        persistent_i32 = 2 * partners + 9
        decision_f32 = self.context_dim + 2 + 2 * partners
        decision_i32 = 6 + 9 * partners
        decision_bool = self.n_actions + 1 + partners
        derived_resources = {
            "model_feature_dim": feature_width,
            "reliability_weight_scalars": partners * feature_width,
            "reliability_weight_nbytes": 4 * partners * feature_width,
            "persistent_state_scalars": persistent_f32 + persistent_i32 + 2,
            "persistent_state_nbytes": 4 * (persistent_f32 + persistent_i32) + 2,
            "partner_message_scalars": 12 * partners,
            "partner_message_nbytes": 45 * partners,
            "pairwise_comparisons": partners * partners,
            "context_input_nbytes": 4 * self.context_dim,
            "action_mask_nbytes": self.n_actions,
            "decision_input_float32_scalars": decision_f32,
            "decision_input_int32_scalars": decision_i32,
            "decision_input_bool_scalars": decision_bool,
            "decision_input_nbytes": 4 * (decision_f32 + decision_i32) + decision_bool,
        }
        for name, value in derived_resources.items():
            if value > _INT32_MAX:
                raise ValueError(f"derived {name} must be at most {_INT32_MAX}")
        maximum_update = (
            float(self.learning_rate)
            * math.sqrt(float(feature_width))
            + float(self.max_abs_weight)
        )
        if not math.isfinite(maximum_update) or maximum_update > _FLOAT32_MAX:
            raise ValueError("learning rule cannot be kept finite in float32")
        maximum_logit = float(feature_width) * float(self.max_abs_weight)
        maximum_net_magnitude = 1.0 + (
            float(self.communication_cost_weight) * float(self.max_communication_cost)
        )
        maximum_blend_score = max(
            float(self.base_blend_weight) * float(self.max_abs_declared_score),
            float(self.option_blend_weight) * float(self.max_abs_declared_score),
            float(self.partner_blend_weight) * maximum_net_magnitude,
        )
        if any(
            not math.isfinite(value) or value > _FLOAT32_MAX / 4.0
            for value in (maximum_logit, maximum_net_magnitude, maximum_blend_score)
        ):
            raise ValueError("model and score bounds cannot remain finite in float32")

    @property
    def model_feature_dim(self) -> int:
        """Return bias + normalized context + declared-confidence width."""

        return self.context_dim + 2

    def to_config(self) -> dict[str, object]:
        """Return the exact canonical JSON-compatible configuration."""

        payload: dict[str, object] = {}
        integer_fields = {
            "max_partners",
            "context_dim",
            "n_actions",
            "max_message_horizon",
            "min_feedback_for_learned_routing",
            "counter_cap",
        }
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            payload[field.name] = value if field.name in integer_fields else float(value)
        return {
            "schema": PARTNER_POLICY_FUSION_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": MECHANISM_STATUS,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            **payload,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> PartnerPolicyFusionConfig:
        """Strictly reconstruct only an exact :meth:`to_config` record."""

        if type(config) is not dict:
            raise ValueError("partner-fusion config must be an actual dict")
        if not all(type(key) is str for key in config):
            raise ValueError("partner-fusion config keys must be strings")
        payload = config.copy()
        field_names = {field.name for field in dataclasses.fields(cls)}
        expected = field_names | {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the partner-fusion v1 schema")
        schema = payload.pop("schema")
        if type(schema) is not str or schema != PARTNER_POLICY_FUSION_CONFIG_SCHEMA:
            raise ValueError("unexpected partner-fusion config schema")
        config_type = payload.pop("type")
        if type(config_type) is not str or config_type != _CONFIG_TYPE:
            raise ValueError("unexpected partner-fusion config type")
        mechanism_status = payload.pop("mechanism_status")
        if type(mechanism_status) is not str or mechanism_status != MECHANISM_STATUS:
            raise ValueError("partner fusion must remain a development L0 mechanism")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("partner fusion configuration cannot claim promotion")
        integer_fields = {
            "max_partners",
            "context_dim",
            "n_actions",
            "max_message_horizon",
            "min_feedback_for_learned_routing",
            "counter_cap",
        }
        for name, value in payload.items():
            if name in integer_fields:
                if type(value) is not int:
                    raise ValueError(f"serialized {name} must be a JSON integer")
            elif type(value) is not float:
                raise ValueError(f"serialized {name} must be a JSON float")
        return cls(**cast(dict[str, Any], payload))


@dataclasses.dataclass(frozen=True)
class PartnerPolicyFusionResourceBudget:
    """Exact fixed-state and maximum per-call logical resource accounting.

    Logical byte counts use four bytes for float32/int32 and one byte for
    booleans.  They exclude Python objects, compiler workspace, alignment,
    executable caches, and checkpoint JSON syntax.  Touched-scalar counts are
    bounds, not FLOP, latency, energy, or physical-memory measurements.
    ``partner_id_pairwise_equality_comparisons_per_decision`` explicitly
    accounts for the quadratic duplicate-identifier audit.  It is a logical
    comparison count; compiler fusion determines transient physical storage.
    """

    max_partners: int
    context_dim: int
    n_actions: int
    model_feature_dim: int
    trainable_float32_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_bool_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    max_messages_per_decision: int
    max_model_scores_per_decision: int
    partner_id_pairwise_equality_comparisons_per_decision: int
    max_trainable_scalars_touched_per_feedback: int
    decision_input_float32_scalars: int
    decision_input_int32_scalars: int
    decision_input_bool_scalars: int
    feedback_input_float32_scalars: int
    feedback_input_int32_scalars: int
    feedback_input_bool_scalars: int
    max_parameter_updates_per_feedback: int
    rng_state_bytes: int
    replay_capacity: int
    dynamic_partner_capacity: int

    def __post_init__(self) -> None:
        for name in ("max_partners", "context_dim", "n_actions", "model_feature_dim"):
            object.__setattr__(
                self,
                name,
                _require_int32(name, getattr(self, name), minimum=1),
            )
        for name in (
            "trainable_float32_scalars",
            "persistent_float32_scalars",
            "persistent_int32_scalars",
            "persistent_bool_scalars",
            "persistent_state_scalars",
            "persistent_state_bytes",
            "max_messages_per_decision",
            "max_model_scores_per_decision",
            "partner_id_pairwise_equality_comparisons_per_decision",
            "max_trainable_scalars_touched_per_feedback",
            "decision_input_float32_scalars",
            "decision_input_int32_scalars",
            "decision_input_bool_scalars",
            "feedback_input_float32_scalars",
            "feedback_input_int32_scalars",
            "feedback_input_bool_scalars",
            "max_parameter_updates_per_feedback",
            "rng_state_bytes",
            "replay_capacity",
            "dynamic_partner_capacity",
        ):
            object.__setattr__(
                self,
                name,
                _require_int32(name, getattr(self, name), minimum=0),
            )
        partners = self.max_partners
        features = self.context_dim + 2
        persistent_f32 = partners * features + features + 1
        persistent_i32 = 2 * partners + 9
        persistent_bool = 2
        expected = {
            "model_feature_dim": features,
            "trainable_float32_scalars": partners * features,
            "persistent_float32_scalars": persistent_f32,
            "persistent_int32_scalars": persistent_i32,
            "persistent_bool_scalars": persistent_bool,
            "persistent_state_scalars": persistent_f32 + persistent_i32 + persistent_bool,
            "persistent_state_bytes": 4 * (persistent_f32 + persistent_i32) + persistent_bool,
            "max_messages_per_decision": partners,
            "max_model_scores_per_decision": partners,
            "partner_id_pairwise_equality_comparisons_per_decision": partners * partners,
            "max_trainable_scalars_touched_per_feedback": features,
            "decision_input_float32_scalars": self.context_dim + 2 + 2 * partners,
            "decision_input_int32_scalars": 6 + 9 * partners,
            "decision_input_bool_scalars": self.n_actions + 1 + partners,
            "feedback_input_float32_scalars": 1,
            "feedback_input_int32_scalars": 4,
            "feedback_input_bool_scalars": 4,
            "max_parameter_updates_per_feedback": 1,
            "rng_state_bytes": 0,
            "replay_capacity": 0,
            "dynamic_partner_capacity": 0,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"{name} does not match the partner-fusion implementation")

    def to_config(self) -> dict[str, int]:
        """Return the fixed JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PartnerMessageBatch:
    """Fixed-capacity partner messages for one action decision.

    All integer references are caller-owned opaque non-negative identities.
    There is intentionally no task or regime identifier.  An available
    message is valid only when observation, context, decision, and event
    bindings match the current call exactly and its finite validity horizon
    has not expired.
    """

    available: Bool[Array, " partners"]
    partner_id: Int[Array, " partners"]
    observation_id: Int[Array, " partners"]
    context_id: Int[Array, " partners"]
    suggested_action: Int[Array, " partners"]
    declared_confidence: Float[Array, " partners"]
    rationale_reference: Int[Array, " partners"]
    provenance_reference: Int[Array, " partners"]
    communication_cost: Float[Array, " partners"]
    issued_decision_id: Int[Array, " partners"]
    issued_event_id: Int[Array, " partners"]
    valid_through_event_id: Int[Array, " partners"]


@chex.dataclass(frozen=True)
class OptionKeyboardProposal:
    """Optional discrete option/keyboard proposal with a declared score."""

    available: Bool[Array, ""]
    action: Int[Array, ""]
    declared_score: Float[Array, ""]


@chex.dataclass(frozen=True)
class PartnerPolicyFusionFeedback:
    """Realized outcome bound to one executed partner-influenced decision.

    There is no agreement field.  Both realized assistance value and observed
    safety outcome must be available for any model update.
    """

    available: Bool[Array, ""]
    decision_id: Int[Array, ""]
    executed_event_id: Int[Array, ""]
    executed_action: Int[Array, ""]
    partner_id: Int[Array, ""]
    assistance_value_available: Bool[Array, ""]
    realized_assistance_value: Float[Array, ""]
    safety_outcome_available: Bool[Array, ""]
    safety_outcome_ok: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PartnerPolicyFusionState:
    """Fixed-size contextual model, saturating counters, and armed record."""

    reliability_weights: Float[Array, "partners model_features"]
    feedback_counts: Int[Array, " partners"]
    safe_feedback_counts: Int[Array, " partners"]
    decision_count: Int[Array, ""]
    feedback_applied_count: Int[Array, ""]
    has_last_decision: Bool[Array, ""]
    last_decision_id: Int[Array, ""]
    last_event_id: Int[Array, ""]
    feedback_armed: Bool[Array, ""]
    armed_decision_id: Int[Array, ""]
    armed_event_id: Int[Array, ""]
    armed_action: Int[Array, ""]
    armed_partner_id: Int[Array, ""]
    armed_route: Int[Array, ""]
    armed_model_features: Float[Array, " model_features"]
    armed_predicted_reliability: Float[Array, ""]


@chex.dataclass(frozen=True)
class PartnerFusionAvailability:
    """Explicit availability and validity surface for one decision."""

    state_valid: Bool[Array, ""]
    decision_identity_valid: Bool[Array, ""]
    base_action_available: Bool[Array, ""]
    base_declared_score_valid: Bool[Array, ""]
    context_features_valid: Bool[Array, ""]
    option_declared: Bool[Array, ""]
    option_valid: Bool[Array, ""]
    messages_declared: Bool[Array, " partners"]
    messages_binding_valid: Bool[Array, " partners"]
    messages_horizon_valid: Bool[Array, " partners"]
    messages_numeric_valid: Bool[Array, " partners"]
    messages_safety_valid: Bool[Array, " partners"]
    messages_unique_partner: Bool[Array, " partners"]
    messages_valid: Bool[Array, " partners"]
    reliability_prediction_available: Bool[Array, " partners"]
    reliability_history_available: Bool[Array, " partners"]
    feedback_slot_available: Bool[Array, ""]
    route_available: Bool[Array, ""]
    action_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PartnerFusionScoreAudit:
    """Finite declared and derived scores; discrete IDs are never averaged."""

    base_declared_score: Float[Array, ""]
    option_declared_score: Float[Array, ""]
    predicted_reliability: Float[Array, " partners"]
    predicted_net_value: Float[Array, " partners"]
    selected_predicted_reliability: Float[Array, ""]
    selected_predicted_net_value: Float[Array, ""]
    blend_candidate_scores: Float[Array, " three_sources"]
    blend_candidate_available: Bool[Array, " three_sources"]
    blend_selected_source: Int[Array, ""]


@chex.dataclass(frozen=True)
class PartnerFusionShieldResult:
    """Audit of the inviolable caller-supplied action mask."""

    caller_action_mask: Bool[Array, " actions"]
    base_action_allowed: Bool[Array, ""]
    proposed_action: Int[Array, ""]
    proposed_action_allowed: Bool[Array, ""]
    shield_intervened: Bool[Array, ""]
    failed_closed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PartnerFusionDecision:
    """Complete bounded audit record for one fusion attempt."""

    decision_id: Int[Array, ""]
    event_id: Int[Array, ""]
    observation_id: Int[Array, ""]
    context_id: Int[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    option_keyboard_action: Int[Array, ""]
    selected_partner_slot: Int[Array, ""]
    selected_partner_id: Int[Array, ""]
    selected_partner_action: Int[Array, ""]
    route: Int[Array, ""]
    route_one_hot: Bool[Array, " routes"]
    partner_influenced: Bool[Array, ""]
    option_keyboard_influenced: Bool[Array, ""]
    effective_action: Int[Array, ""]
    reliability_history_status: Int[Array, ""]
    reliability_history_available: Bool[Array, ""]
    uncalibrated_development_exploration: Bool[Array, ""]
    feedback_armed: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    availability: PartnerFusionAvailability
    scores: PartnerFusionScoreAudit
    shield: PartnerFusionShieldResult


@chex.dataclass(frozen=True)
class PartnerFusionDecisionResult:
    """Next state plus one complete fusion decision log."""

    state: PartnerPolicyFusionState
    decision: PartnerFusionDecision


@chex.dataclass(frozen=True)
class PartnerFusionFeedbackResult:
    """Atomic feedback result with finite bounded diagnostics."""

    state: PartnerPolicyFusionState
    feedback_available: Bool[Array, ""]
    armed_record_available: Bool[Array, ""]
    identity_match: Bool[Array, ""]
    action_match: Bool[Array, ""]
    partner_match: Bool[Array, ""]
    realized_outcomes_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    counter_saturated: Bool[Array, ""]
    predicted_reliability_before: Float[Array, ""]
    realized_training_target: Float[Array, ""]
    parameter_update_l2_norm: Float[Array, ""]


class PartnerPolicyFusion:
    """Fixed-capacity discrete partner policy fusion and reliability learner."""

    def __init__(self, config: PartnerPolicyFusionConfig):
        self._config = config

    @property
    def config(self) -> PartnerPolicyFusionConfig:
        """Return the immutable construction contract."""

        return self._config

    @property
    def resource_budget(self) -> PartnerPolicyFusionResourceBudget:
        """Return exact logical fixed-state and maximum per-call counts."""

        cfg = self._config
        partners = cfg.max_partners
        features = cfg.model_feature_dim
        persistent_f32 = partners * features + features + 1
        persistent_i32 = 2 * partners + 9
        persistent_bool = 2
        persistent_scalars = persistent_f32 + persistent_i32 + persistent_bool
        return PartnerPolicyFusionResourceBudget(
            max_partners=partners,
            context_dim=cfg.context_dim,
            n_actions=cfg.n_actions,
            model_feature_dim=features,
            trainable_float32_scalars=partners * features,
            persistent_float32_scalars=persistent_f32,
            persistent_int32_scalars=persistent_i32,
            persistent_bool_scalars=persistent_bool,
            persistent_state_scalars=persistent_scalars,
            persistent_state_bytes=4 * (persistent_f32 + persistent_i32) + persistent_bool,
            max_messages_per_decision=partners,
            max_model_scores_per_decision=partners,
            partner_id_pairwise_equality_comparisons_per_decision=partners * partners,
            max_trainable_scalars_touched_per_feedback=features,
            decision_input_float32_scalars=cfg.context_dim + 2 + 2 * partners,
            decision_input_int32_scalars=6 + 9 * partners,
            decision_input_bool_scalars=cfg.n_actions + 1 + partners,
            feedback_input_float32_scalars=1,
            feedback_input_int32_scalars=4,
            feedback_input_bool_scalars=4,
            max_parameter_updates_per_feedback=1,
            rng_state_bytes=0,
            replay_capacity=0,
            dynamic_partner_capacity=0,
        )

    def to_config(self) -> dict[str, object]:
        """Serialize the complete fusion construction."""

        return {"type": _FUSION_TYPE, "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> PartnerPolicyFusion:
        """Strictly restore :class:`PartnerPolicyFusion` construction."""

        if type(config) is not dict:
            raise ValueError("fusion construction must be an actual dict")
        if not all(type(key) is str for key in config):
            raise ValueError("fusion construction keys must be strings")
        payload = config.copy()
        if set(payload) != {"type", "config"}:
            raise ValueError("fusion construction fields do not match the v1 schema")
        config_type = payload.get("type")
        if type(config_type) is not str or config_type != _FUSION_TYPE:
            raise ValueError("unexpected partner-fusion construction type")
        nested = payload.get("config")
        if type(nested) is not dict:
            raise ValueError("fusion construction config must be an actual dict")
        return cls(PartnerPolicyFusionConfig.from_config(nested))

    def init(self) -> PartnerPolicyFusionState:
        """Return deterministic zero state; this component owns no RNG."""

        cfg = self._config
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        missing_i = jnp.asarray(-1, dtype=jnp.int32)
        return PartnerPolicyFusionState(
            reliability_weights=jnp.zeros(
                (cfg.max_partners, cfg.model_feature_dim), dtype=jnp.float32
            ),
            feedback_counts=jnp.zeros((cfg.max_partners,), dtype=jnp.int32),
            safe_feedback_counts=jnp.zeros((cfg.max_partners,), dtype=jnp.int32),
            decision_count=zero_i,
            feedback_applied_count=zero_i,
            has_last_decision=jnp.asarray(False, dtype=jnp.bool_),
            last_decision_id=missing_i,
            last_event_id=missing_i,
            feedback_armed=jnp.asarray(False, dtype=jnp.bool_),
            armed_decision_id=missing_i,
            armed_event_id=missing_i,
            armed_action=missing_i,
            armed_partner_id=missing_i,
            armed_route=missing_i,
            armed_model_features=jnp.zeros((cfg.model_feature_dim,), dtype=jnp.float32),
            armed_predicted_reliability=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def empty_messages(self) -> PartnerMessageBatch:
        """Return a canonical fixed-capacity batch with no messages."""

        partners = self._config.max_partners
        missing = jnp.full((partners,), -1, dtype=jnp.int32)
        return PartnerMessageBatch(
            available=jnp.zeros((partners,), dtype=jnp.bool_),
            partner_id=missing,
            observation_id=missing,
            context_id=missing,
            suggested_action=missing,
            declared_confidence=jnp.zeros((partners,), dtype=jnp.float32),
            rationale_reference=missing,
            provenance_reference=missing,
            communication_cost=jnp.zeros((partners,), dtype=jnp.float32),
            issued_decision_id=missing,
            issued_event_id=missing,
            valid_through_event_id=missing,
        )

    @staticmethod
    def empty_option_proposal() -> OptionKeyboardProposal:
        """Return a canonical unavailable option/keyboard proposal."""

        return OptionKeyboardProposal(
            available=jnp.asarray(False, dtype=jnp.bool_),
            action=jnp.asarray(-1, dtype=jnp.int32),
            declared_score=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _validate_state_static_contract(self, state: PartnerPolicyFusionState) -> None:
        """Raise for state shapes or dtypes that cannot be handled under JIT."""

        cfg = self._config
        contracts = (
            (state.reliability_weights, (cfg.max_partners, cfg.model_feature_dim), jnp.float32),
            (state.feedback_counts, (cfg.max_partners,), jnp.int32),
            (state.safe_feedback_counts, (cfg.max_partners,), jnp.int32),
            (state.decision_count, (), jnp.int32),
            (state.feedback_applied_count, (), jnp.int32),
            (state.has_last_decision, (), jnp.bool_),
            (state.last_decision_id, (), jnp.int32),
            (state.last_event_id, (), jnp.int32),
            (state.feedback_armed, (), jnp.bool_),
            (state.armed_decision_id, (), jnp.int32),
            (state.armed_event_id, (), jnp.int32),
            (state.armed_action, (), jnp.int32),
            (state.armed_partner_id, (), jnp.int32),
            (state.armed_route, (), jnp.int32),
            (state.armed_model_features, (cfg.model_feature_dim,), jnp.float32),
            (state.armed_predicted_reliability, (), jnp.float32),
        )
        for value, shape, dtype in contracts:
            if getattr(value, "shape", None) != shape or getattr(value, "dtype", None) != dtype:
                raise ValueError(f"state leaf must have shape {shape} and dtype {dtype}")

    def _state_valid_predicate(self, state: PartnerPolicyFusionState) -> Array:
        """Return the full dynamic state invariant as a scalar predicate."""

        cfg = self._config
        cap = jnp.asarray(cfg.counter_cap, dtype=jnp.int32)
        weights_valid = jnp.all(jnp.isfinite(state.reliability_weights)) & jnp.all(
            jnp.abs(state.reliability_weights) <= cfg.max_abs_weight
        )
        features_valid = jnp.all(jnp.isfinite(state.armed_model_features)) & jnp.all(
            jnp.abs(state.armed_model_features) <= 1.0
        )
        counters_valid = (
            jnp.all((state.feedback_counts >= 0) & (state.feedback_counts <= cap))
            & jnp.all(
                (state.safe_feedback_counts >= 0)
                & (state.safe_feedback_counts <= state.feedback_counts)
            )
            & (state.decision_count >= 0)
            & (state.decision_count <= cap)
            & (state.feedback_applied_count >= 0)
            & (state.feedback_applied_count <= cap)
            & (state.feedback_applied_count <= state.decision_count)
        )
        expected_total = _saturating_sum(state.feedback_counts, cfg.counter_cap)
        counters_valid &= state.feedback_applied_count == expected_total

        last_absent = (
            (~state.has_last_decision)
            & (state.last_decision_id == -1)
            & (state.last_event_id == -1)
            & (state.decision_count == 0)
        )
        last_present = (
            state.has_last_decision
            & (state.last_decision_id >= 0)
            & (state.last_event_id >= 0)
            & (state.decision_count > 0)
        )
        last_valid = last_absent | last_present

        armed_absent = (
            (~state.feedback_armed)
            & (state.armed_decision_id == -1)
            & (state.armed_event_id == -1)
            & (state.armed_action == -1)
            & (state.armed_partner_id == -1)
            & (state.armed_route == -1)
            & jnp.all(state.armed_model_features == 0.0)
            & (state.armed_predicted_reliability == 0.0)
        )
        armed_present = (
            state.feedback_armed
            & state.has_last_decision
            & (state.armed_decision_id >= 0)
            & (state.armed_decision_id <= state.last_decision_id)
            & (state.armed_event_id >= 0)
            & (state.armed_event_id <= state.last_event_id)
            & (state.armed_action >= 0)
            & (state.armed_action < cfg.n_actions)
            & (state.armed_partner_id >= 0)
            & (state.armed_partner_id < cfg.max_partners)
            & ((state.armed_route == ROUTE_ACCEPT) | (state.armed_route == ROUTE_BLEND))
            & jnp.isfinite(state.armed_predicted_reliability)
            & (state.armed_predicted_reliability >= 0.0)
            & (state.armed_predicted_reliability <= 1.0)
        )
        return weights_valid & features_valid & counters_valid & last_valid & (
            armed_absent | armed_present
        )

    def validate_state(self, state: PartnerPolicyFusionState) -> None:
        """Eagerly reject any state violating the exact persistent contract."""

        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_valid_predicate(state))):
            raise ValueError("partner-fusion state violates its dynamic invariants")

    def _validate_decision_static_contract(
        self,
        *,
        decision_id: Array,
        event_id: Array,
        observation_id: Array,
        context_id: Array,
        context_features: Array,
        base_action: Array,
        base_declared_score: Array,
        safety_action_mask: Array,
        option_proposal: OptionKeyboardProposal,
        messages: PartnerMessageBatch,
    ) -> None:
        """Raise only for malformed shapes/dtypes, never dynamic contents."""

        cfg = self._config
        scalar_ints = (decision_id, event_id, observation_id, context_id, base_action)
        for value in scalar_ints:
            if value.shape != () or value.dtype != jnp.int32:
                raise ValueError("decision identities and actions must be scalar int32")
        if context_features.shape != (cfg.context_dim,) or context_features.dtype != jnp.float32:
            raise ValueError("context_features has the wrong shape or dtype")
        if base_declared_score.shape != () or base_declared_score.dtype != jnp.float32:
            raise ValueError("base_declared_score must be scalar float32")
        if safety_action_mask.shape != (cfg.n_actions,) or safety_action_mask.dtype != jnp.bool_:
            raise ValueError("safety_action_mask has the wrong shape or dtype")
        if option_proposal.available.shape != () or option_proposal.available.dtype != jnp.bool_:
            raise ValueError("option available must be scalar bool")
        if option_proposal.action.shape != () or option_proposal.action.dtype != jnp.int32:
            raise ValueError("option action must be scalar int32")
        if (
            option_proposal.declared_score.shape != ()
            or option_proposal.declared_score.dtype != jnp.float32
        ):
            raise ValueError("option declared_score must be scalar float32")
        message_ints = (
            messages.partner_id,
            messages.observation_id,
            messages.context_id,
            messages.suggested_action,
            messages.rationale_reference,
            messages.provenance_reference,
            messages.issued_decision_id,
            messages.issued_event_id,
            messages.valid_through_event_id,
        )
        for value in message_ints:
            if value.shape != (cfg.max_partners,) or value.dtype != jnp.int32:
                raise ValueError("message integer fields have the wrong shape or dtype")
        message_floats = (messages.declared_confidence, messages.communication_cost)
        for value in message_floats:
            if value.shape != (cfg.max_partners,) or value.dtype != jnp.float32:
                raise ValueError("message float fields have the wrong shape or dtype")
        if messages.available.shape != (cfg.max_partners,) or messages.available.dtype != jnp.bool_:
            raise ValueError("message availability has the wrong shape or dtype")

    def decide(
        self,
        state: PartnerPolicyFusionState,
        *,
        decision_id: Array,
        event_id: Array,
        observation_id: Array,
        context_id: Array,
        context_features: Array,
        base_action: Array,
        base_declared_score: Array,
        safety_action_mask: Array,
        option_proposal: OptionKeyboardProposal,
        messages: PartnerMessageBatch,
    ) -> PartnerFusionDecisionResult:
        """Fuse one bounded message batch into one hard-shielded action.

        ``blend`` is discrete argmax over three declared/derived scores:
        ``base_blend_weight * base_declared_score``,
        ``option_blend_weight * option_declared_score``, and
        ``partner_blend_weight * predicted_net_value``.  Action identifiers
        are never averaged.  Ties use the fixed source order base, option,
        partner.  An uncalibrated message can reach ``accept`` only through
        the configured high net-value threshold; the decision then marks
        ``uncalibrated_development_exploration``.  That cold-start mechanism
        gathers realized outcomes and must not be read as calibrated trust.
        """

        self._validate_state_static_contract(state)
        self._validate_decision_static_contract(
            decision_id=decision_id,
            event_id=event_id,
            observation_id=observation_id,
            context_id=context_id,
            context_features=context_features,
            base_action=base_action,
            base_declared_score=base_declared_score,
            safety_action_mask=safety_action_mask,
            option_proposal=option_proposal,
            messages=messages,
        )
        cfg = self._config
        state_valid = self._state_valid_predicate(state)
        ids_valid = (
            (decision_id >= 0)
            & (event_id >= 0)
            & (observation_id >= 0)
            & (context_id >= 0)
            & (
                (~state.has_last_decision)
                | ((decision_id > state.last_decision_id) & (event_id > state.last_event_id))
            )
        )
        base_allowed = _safe_action_lookup(safety_action_mask, base_action, cfg.n_actions)
        base_score_valid = jnp.isfinite(base_declared_score) & (
            jnp.abs(base_declared_score) <= cfg.max_abs_declared_score
        )
        safe_base_score = jnp.where(base_score_valid, base_declared_score, 0.0)
        context_valid = jnp.all(jnp.isfinite(context_features)) & jnp.all(
            jnp.abs(context_features) <= cfg.max_abs_context
        )
        safety_envelope_valid = state_valid & ids_valid & base_allowed
        fusion_input_valid = safety_envelope_valid & context_valid & base_score_valid

        option_action_allowed = _safe_action_lookup(
            safety_action_mask, option_proposal.action, cfg.n_actions
        )
        option_score_valid = jnp.isfinite(option_proposal.declared_score) & (
            jnp.abs(option_proposal.declared_score) <= cfg.max_abs_declared_score
        )
        option_valid = option_proposal.available & option_action_allowed & option_score_valid
        safe_option_score = jnp.where(option_valid, option_proposal.declared_score, 0.0)

        partner_in_range = (messages.partner_id >= 0) & (
            messages.partner_id < cfg.max_partners
        )
        safe_partner_ids = jnp.clip(messages.partner_id, 0, cfg.max_partners - 1)
        equality = messages.partner_id[:, None] == messages.partner_id[None, :]
        both_available = messages.available[:, None] & messages.available[None, :]
        off_diagonal = ~jnp.eye(cfg.max_partners, dtype=jnp.bool_)
        duplicate = jnp.any(equality & both_available & off_diagonal, axis=1)
        unique_partner = partner_in_range & (~duplicate)

        message_binding_valid = (
            (messages.observation_id == observation_id)
            & (messages.context_id == context_id)
            & (messages.issued_decision_id == decision_id)
        )
        nonnegative_events = (
            (messages.issued_event_id >= 0) & (messages.valid_through_event_id >= 0)
        )
        ordered_horizon = nonnegative_events & (
            messages.valid_through_event_id >= messages.issued_event_id
        )
        # Select equal operands on unordered/sentinel rows before subtraction.
        # On ordered non-negative int32 rows, the mathematical difference is
        # in [0, INT32_MAX], including issued=INT32_MAX-1 and through=INT32_MAX.
        safe_horizon_start = jnp.where(
            ordered_horizon,
            messages.issued_event_id,
            messages.valid_through_event_id,
        )
        horizon_width = messages.valid_through_event_id - safe_horizon_start
        horizon_valid = (
            ordered_horizon
            & (messages.issued_event_id <= event_id)
            & (event_id <= messages.valid_through_event_id)
            & (horizon_width <= cfg.max_message_horizon)
        )
        numeric_valid = (
            context_valid
            & jnp.isfinite(messages.declared_confidence)
            & (messages.declared_confidence >= 0.0)
            & (messages.declared_confidence <= 1.0)
            & jnp.isfinite(messages.communication_cost)
            & (messages.communication_cost >= 0.0)
            & (messages.communication_cost <= cfg.max_communication_cost)
            & (messages.rationale_reference >= 0)
            & (messages.provenance_reference >= 0)
        )
        message_safety_valid = _safe_action_lookup(
            safety_action_mask, messages.suggested_action, cfg.n_actions
        )
        messages_valid = (
            messages.available
            & unique_partner
            & message_binding_valid
            & horizon_valid
            & numeric_valid
            & message_safety_valid
        )

        safe_context = jnp.where(context_valid, context_features, 0.0)
        normalized_context = jnp.clip(
            safe_context / jnp.float32(cfg.max_abs_context), -1.0, 1.0
        )
        safe_confidences = jnp.where(
            jnp.isfinite(messages.declared_confidence),
            jnp.clip(messages.declared_confidence, 0.0, 1.0),
            0.0,
        )
        shared = jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.float32), normalized_context)
        )
        model_features = jnp.concatenate(
            (
                jnp.broadcast_to(shared, (cfg.max_partners, cfg.context_dim + 1)),
                safe_confidences[:, None],
            ),
            axis=1,
        )
        safe_reliability_weights = jnp.where(
            jnp.isfinite(state.reliability_weights),
            jnp.clip(
                state.reliability_weights,
                -jnp.float32(cfg.max_abs_weight),
                jnp.float32(cfg.max_abs_weight),
            ),
            0.0,
        )
        selected_weights = safe_reliability_weights[safe_partner_ids]
        logits = jnp.sum(selected_weights * model_features, axis=1)
        predicted_reliability = jax.nn.sigmoid(logits)
        predicted_net = predicted_reliability * safe_confidences - jnp.float32(
            cfg.communication_cost_weight
        ) * jnp.where(
            jnp.isfinite(messages.communication_cost),
            jnp.clip(messages.communication_cost, 0.0, cfg.max_communication_cost),
            cfg.max_communication_cost,
        )
        score_floor = jnp.float32(-cfg.max_communication_cost * cfg.communication_cost_weight - 2.0)
        selectable_net = jnp.where(messages_valid, predicted_net, score_floor)
        selected_slot_raw = jnp.argmax(selectable_net).astype(jnp.int32)
        any_valid_message = jnp.any(messages_valid)
        selected_slot = jnp.where(any_valid_message, selected_slot_raw, jnp.int32(-1))
        safe_slot = jnp.clip(selected_slot_raw, 0, cfg.max_partners - 1)
        selected_partner_id_raw = safe_partner_ids[safe_slot]
        selected_partner_id = jnp.where(
            any_valid_message, selected_partner_id_raw, jnp.int32(-1)
        )
        selected_partner_action = jnp.where(
            any_valid_message, messages.suggested_action[safe_slot], jnp.int32(-1)
        )
        selected_reliability = jnp.where(
            any_valid_message, predicted_reliability[safe_slot], 0.0
        )
        selected_net = jnp.where(any_valid_message, predicted_net[safe_slot], 0.0)
        selected_confidence = jnp.where(
            any_valid_message, safe_confidences[safe_slot], 0.0
        )
        selected_cost = jnp.where(
            any_valid_message,
            jnp.clip(messages.communication_cost[safe_slot], 0.0, cfg.max_communication_cost),
            0.0,
        )
        selected_count = state.feedback_counts[selected_partner_id_raw]
        selected_history_available = any_valid_message & (
            selected_count >= cfg.min_feedback_for_learned_routing
        )

        selected_low_confidence = selected_confidence < cfg.clarification_confidence_threshold
        route = jnp.asarray(ROUTE_IGNORE, dtype=jnp.int32)
        route = jnp.where(
            any_valid_message & selected_low_confidence,
            jnp.int32(ROUTE_REQUEST_CLARIFICATION),
            route,
        )
        route = jnp.where(
            any_valid_message
            & (~selected_low_confidence)
            & (selected_net >= cfg.accept_net_value_threshold),
            jnp.int32(ROUTE_ACCEPT),
            route,
        )
        route = jnp.where(
            any_valid_message
            & (~selected_low_confidence)
            & (selected_net >= cfg.blend_net_value_threshold)
            & (selected_net < cfg.accept_net_value_threshold)
            & selected_history_available,
            jnp.int32(ROUTE_BLEND),
            route,
        )
        route = jnp.where(
            any_valid_message
            & (~selected_low_confidence)
            & (selected_net >= cfg.blend_net_value_threshold)
            & (selected_net < cfg.accept_net_value_threshold)
            & (~selected_history_available)
            & (selected_cost <= cfg.max_query_cost),
            jnp.int32(ROUTE_QUERY),
            route,
        )
        feedback_slot_available = ~state.feedback_armed
        route_available = fusion_input_valid & feedback_slot_available
        route = jnp.where(route_available, route, jnp.int32(ROUTE_IGNORE))

        blend_scores = jnp.asarray(
            [
                cfg.base_blend_weight * safe_base_score,
                cfg.option_blend_weight * safe_option_score,
                cfg.partner_blend_weight * selected_net,
            ],
            dtype=jnp.float32,
        )
        blend_available = jnp.asarray(
            [base_allowed, option_valid, any_valid_message], dtype=jnp.bool_
        )
        blend_score_floor = jnp.float32(
            -max(
                cfg.max_abs_declared_score
                * max(cfg.base_blend_weight, cfg.option_blend_weight),
                cfg.max_communication_cost * cfg.communication_cost_weight
                * cfg.partner_blend_weight,
            )
            - 2.0
        )
        blend_selected_source = jnp.argmax(
            jnp.where(blend_available, blend_scores, blend_score_floor)
        ).astype(jnp.int32)
        blend_action = jnp.where(
            blend_selected_source == SOURCE_OPTION_KEYBOARD,
            option_proposal.action,
            jnp.where(
                blend_selected_source == SOURCE_PARTNER,
                selected_partner_action,
                base_action,
            ),
        )
        proposed_action = jnp.where(
            route == ROUTE_ACCEPT,
            selected_partner_action,
            jnp.where(route == ROUTE_BLEND, blend_action, base_action),
        )
        proposed_allowed = _safe_action_lookup(
            safety_action_mask, proposed_action, cfg.n_actions
        )
        shield_intervened = fusion_input_valid & (~proposed_allowed)
        shielded_action = jnp.where(proposed_allowed, proposed_action, base_action)
        # Corrupt context or declared base score disables learned fusion and
        # emits the independently shield-checked base action without advancing
        # state.  Invalid state/identity/base safety fails fully closed.
        safe_fallback_action = jnp.where(fusion_input_valid, shielded_action, base_action)
        effective_action = jnp.where(
            safety_envelope_valid, safe_fallback_action, jnp.int32(-1)
        )
        action_available = safety_envelope_valid & _safe_action_lookup(
            safety_action_mask, effective_action, cfg.n_actions
        )
        partner_selected_as_source = (route == ROUTE_ACCEPT) | (
            (route == ROUTE_BLEND) & (blend_selected_source == SOURCE_PARTNER)
        )
        partner_influenced = (
            route_available
            & partner_selected_as_source
            & (effective_action != base_action)
            & action_available
        )
        option_influenced = (
            route_available
            & (route == ROUTE_BLEND)
            & (blend_selected_source == SOURCE_OPTION_KEYBOARD)
            & (effective_action != base_action)
            & action_available
        )
        feedback_armed = partner_influenced

        updated_state = PartnerPolicyFusionState(
            reliability_weights=state.reliability_weights,
            feedback_counts=state.feedback_counts,
            safe_feedback_counts=state.safe_feedback_counts,
            decision_count=_saturating_increment(state.decision_count, cfg.counter_cap),
            feedback_applied_count=state.feedback_applied_count,
            has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
            last_decision_id=decision_id,
            last_event_id=event_id,
            feedback_armed=state.feedback_armed | feedback_armed,
            armed_decision_id=jnp.where(feedback_armed, decision_id, state.armed_decision_id),
            armed_event_id=jnp.where(feedback_armed, event_id, state.armed_event_id),
            armed_action=jnp.where(feedback_armed, effective_action, state.armed_action),
            armed_partner_id=jnp.where(
                feedback_armed, selected_partner_id, state.armed_partner_id
            ),
            armed_route=jnp.where(feedback_armed, route, state.armed_route),
            armed_model_features=jnp.where(
                feedback_armed,
                model_features[safe_slot],
                state.armed_model_features,
            ),
            armed_predicted_reliability=jnp.where(
                feedback_armed,
                selected_reliability,
                state.armed_predicted_reliability,
            ),
        )
        next_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(fusion_input_valid, new, old),
            updated_state,
            state,
        )

        route_one_hot = jax.nn.one_hot(
            route, ROUTE_COUNT, dtype=jnp.int32
        ).astype(jnp.bool_) & route_available
        history_status = jnp.where(
            selected_history_available,
            jnp.int32(RELIABILITY_HISTORY_DEVELOPMENT_ONLINE),
            jnp.int32(RELIABILITY_HISTORY_UNAVAILABLE),
        )
        availability = PartnerFusionAvailability(
            state_valid=state_valid,
            decision_identity_valid=ids_valid,
            base_action_available=base_allowed,
            base_declared_score_valid=base_score_valid,
            context_features_valid=context_valid,
            option_declared=option_proposal.available,
            option_valid=option_valid,
            messages_declared=messages.available,
            messages_binding_valid=message_binding_valid,
            messages_horizon_valid=horizon_valid,
            messages_numeric_valid=numeric_valid,
            messages_safety_valid=message_safety_valid,
            messages_unique_partner=unique_partner,
            messages_valid=messages_valid,
            reliability_prediction_available=messages_valid,
            reliability_history_available=messages_valid
            & (state.feedback_counts[safe_partner_ids] >= cfg.min_feedback_for_learned_routing),
            feedback_slot_available=feedback_slot_available,
            route_available=route_available,
            action_available=action_available,
        )
        scores = PartnerFusionScoreAudit(
            base_declared_score=safe_base_score,
            option_declared_score=safe_option_score,
            predicted_reliability=jnp.where(messages_valid, predicted_reliability, 0.0),
            predicted_net_value=jnp.where(messages_valid, predicted_net, 0.0),
            selected_predicted_reliability=selected_reliability,
            selected_predicted_net_value=selected_net,
            blend_candidate_scores=jnp.where(blend_available, blend_scores, 0.0),
            blend_candidate_available=blend_available,
            blend_selected_source=jnp.where(
                route == ROUTE_BLEND, blend_selected_source, jnp.int32(SOURCE_BASE)
            ),
        )
        shield = PartnerFusionShieldResult(
            caller_action_mask=safety_action_mask,
            base_action_allowed=base_allowed,
            proposed_action=jnp.where(fusion_input_valid, proposed_action, jnp.int32(-1)),
            proposed_action_allowed=fusion_input_valid & proposed_allowed,
            shield_intervened=shield_intervened,
            failed_closed=~safety_envelope_valid,
        )
        decision = PartnerFusionDecision(
            decision_id=decision_id,
            event_id=event_id,
            observation_id=observation_id,
            context_id=context_id,
            counterfactual_base_action=base_action,
            option_keyboard_action=jnp.where(
                option_proposal.available, option_proposal.action, jnp.int32(-1)
            ),
            selected_partner_slot=selected_slot,
            selected_partner_id=selected_partner_id,
            selected_partner_action=selected_partner_action,
            route=route,
            route_one_hot=route_one_hot,
            partner_influenced=partner_influenced,
            option_keyboard_influenced=option_influenced,
            effective_action=effective_action,
            reliability_history_status=history_status,
            reliability_history_available=selected_history_available,
            uncalibrated_development_exploration=(route == ROUTE_ACCEPT)
            & any_valid_message
            & (~selected_history_available)
            & route_available,
            feedback_armed=feedback_armed,
            input_valid=fusion_input_valid,
            applied=fusion_input_valid,
            availability=availability,
            scores=scores,
            shield=shield,
        )
        return PartnerFusionDecisionResult(state=next_state, decision=decision)

    def apply_feedback(
        self,
        state: PartnerPolicyFusionState,
        feedback: PartnerPolicyFusionFeedback,
    ) -> PartnerFusionFeedbackResult:
        """Apply exactly bound realized feedback or perform an atomic no-op."""

        self._validate_state_static_contract(state)
        scalar_contracts = (
            (feedback.available, jnp.bool_),
            (feedback.decision_id, jnp.int32),
            (feedback.executed_event_id, jnp.int32),
            (feedback.executed_action, jnp.int32),
            (feedback.partner_id, jnp.int32),
            (feedback.assistance_value_available, jnp.bool_),
            (feedback.realized_assistance_value, jnp.float32),
            (feedback.safety_outcome_available, jnp.bool_),
            (feedback.safety_outcome_ok, jnp.bool_),
        )
        for value, dtype in scalar_contracts:
            if value.shape != () or value.dtype != dtype:
                raise ValueError("feedback fields must be scalar with their declared dtype")

        cfg = self._config
        state_valid = self._state_valid_predicate(state)
        identity_match = (
            (feedback.decision_id == state.armed_decision_id)
            & (feedback.executed_event_id == state.armed_event_id)
        )
        action_match = feedback.executed_action == state.armed_action
        partner_match = feedback.partner_id == state.armed_partner_id
        outcomes_available = (
            feedback.assistance_value_available & feedback.safety_outcome_available
        )
        assistance_valid = jnp.isfinite(feedback.realized_assistance_value) & (
            jnp.abs(feedback.realized_assistance_value) <= cfg.assistance_value_bound
        )
        input_valid = (
            feedback.available
            & state_valid
            & state.feedback_armed
            & identity_match
            & action_match
            & partner_match
            & outcomes_available
            & assistance_valid
        )
        safe_partner = jnp.clip(state.armed_partner_id, 0, cfg.max_partners - 1)
        prediction = jnp.where(
            jnp.isfinite(state.armed_predicted_reliability),
            jnp.clip(state.armed_predicted_reliability, 0.0, 1.0),
            0.0,
        )
        assistance_target = jnp.clip(
            0.5
            * (
                feedback.realized_assistance_value
                / jnp.float32(cfg.assistance_value_bound)
                + 1.0
            ),
            0.0,
            1.0,
        )
        safe_target = jnp.float32(cfg.safety_target_weight) + jnp.float32(
            1.0 - cfg.safety_target_weight
        ) * assistance_target
        training_target = jnp.where(feedback.safety_outcome_ok, safe_target, 0.0)
        error = training_target - prediction
        parameter_delta = jnp.float32(cfg.learning_rate) * error * state.armed_model_features
        old_row = state.reliability_weights[safe_partner]
        new_row = jnp.clip(
            old_row + parameter_delta,
            -jnp.float32(cfg.max_abs_weight),
            jnp.float32(cfg.max_abs_weight),
        )
        actual_delta = new_row - old_row
        next_weights = state.reliability_weights.at[safe_partner].set(new_row)
        next_feedback_counts = state.feedback_counts.at[safe_partner].set(
            _saturating_increment(state.feedback_counts[safe_partner], cfg.counter_cap)
        )
        next_safe_counts = state.safe_feedback_counts.at[safe_partner].set(
            jnp.where(
                feedback.safety_outcome_ok,
                _saturating_increment(
                    state.safe_feedback_counts[safe_partner], cfg.counter_cap
                ),
                state.safe_feedback_counts[safe_partner],
            )
        )
        missing_i = jnp.asarray(-1, dtype=jnp.int32)
        proposed = PartnerPolicyFusionState(
            reliability_weights=next_weights,
            feedback_counts=next_feedback_counts,
            safe_feedback_counts=next_safe_counts,
            decision_count=state.decision_count,
            feedback_applied_count=_saturating_increment(
                state.feedback_applied_count, cfg.counter_cap
            ),
            has_last_decision=state.has_last_decision,
            last_decision_id=state.last_decision_id,
            last_event_id=state.last_event_id,
            feedback_armed=jnp.asarray(False, dtype=jnp.bool_),
            armed_decision_id=missing_i,
            armed_event_id=missing_i,
            armed_action=missing_i,
            armed_partner_id=missing_i,
            armed_route=missing_i,
            armed_model_features=jnp.zeros(
                (cfg.model_feature_dim,), dtype=jnp.float32
            ),
            armed_predicted_reliability=jnp.asarray(0.0, dtype=jnp.float32),
        )
        next_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(input_valid, new, old), proposed, state
        )
        update_norm = _finite_l2_norm(actual_delta)
        counter_saturated = (
            (state.feedback_counts[safe_partner] >= cfg.counter_cap)
            | (state.feedback_applied_count >= cfg.counter_cap)
            | (
                feedback.safety_outcome_ok
                & (state.safe_feedback_counts[safe_partner] >= cfg.counter_cap)
            )
        )
        return PartnerFusionFeedbackResult(
            state=next_state,
            feedback_available=feedback.available,
            armed_record_available=state.feedback_armed,
            identity_match=identity_match,
            action_match=action_match,
            partner_match=partner_match,
            realized_outcomes_available=outcomes_available,
            input_valid=input_valid,
            applied=input_valid,
            counter_saturated=input_valid & counter_saturated,
            predicted_reliability_before=jnp.where(input_valid, prediction, 0.0),
            realized_training_target=jnp.where(input_valid, training_target, 0.0),
            parameter_update_l2_norm=jnp.where(input_valid, update_norm, 0.0),
        )

    def decide_sequence(
        self,
        state: PartnerPolicyFusionState,
        *,
        decision_ids: Array,
        event_ids: Array,
        observation_ids: Array,
        context_ids: Array,
        context_features: Array,
        base_actions: Array,
        base_declared_scores: Array,
        safety_action_masks: Array,
        option_proposals: OptionKeyboardProposal,
        messages: PartnerMessageBatch,
    ) -> tuple[PartnerPolicyFusionState, PartnerFusionDecision]:
        """Run a fixed-length uninterrupted decision sequence with ``lax.scan``."""

        length = _require_fusion_sequence_vector("decision_ids", decision_ids)
        if (
            decision_ids.dtype != jnp.int32
            or event_ids.shape != (length,)
            or event_ids.dtype != jnp.int32
            or observation_ids.shape != (length,)
            or observation_ids.dtype != jnp.int32
            or context_ids.shape != (length,)
            or context_ids.dtype != jnp.int32
            or context_features.shape != (length, self._config.context_dim)
            or context_features.dtype != jnp.float32
            or base_actions.shape != (length,)
            or base_actions.dtype != jnp.int32
            or base_declared_scores.shape != (length,)
            or base_declared_scores.dtype != jnp.float32
            or safety_action_masks.shape != (length, self._config.n_actions)
            or safety_action_masks.dtype != jnp.bool_
        ):
            raise ValueError("decision sequence arrays violate their fixed contract")

        def step(
            carry: PartnerPolicyFusionState,
            inputs: tuple[
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                OptionKeyboardProposal,
                PartnerMessageBatch,
            ],
        ) -> tuple[PartnerPolicyFusionState, PartnerFusionDecision]:
            result = self.decide(
                carry,
                decision_id=inputs[0],
                event_id=inputs[1],
                observation_id=inputs[2],
                context_id=inputs[3],
                context_features=inputs[4],
                base_action=inputs[5],
                base_declared_score=inputs[6],
                safety_action_mask=inputs[7],
                option_proposal=inputs[8],
                messages=inputs[9],
            )
            return result.state, result.decision

        return jax.lax.scan(
            step,
            state,
            (
                decision_ids,
                event_ids,
                observation_ids,
                context_ids,
                context_features,
                base_actions,
                base_declared_scores,
                safety_action_masks,
                option_proposals,
                messages,
            ),
        )

    def feedback_sequence(
        self,
        state: PartnerPolicyFusionState,
        feedback: PartnerPolicyFusionFeedback,
    ) -> tuple[PartnerPolicyFusionState, PartnerFusionFeedbackResult]:
        """Run a fixed-length feedback sequence with :func:`jax.lax.scan`."""

        _require_fusion_sequence_vector("feedback.available", feedback.available)

        def step(
            carry: PartnerPolicyFusionState,
            item: PartnerPolicyFusionFeedback,
        ) -> tuple[PartnerPolicyFusionState, PartnerFusionFeedbackResult]:
            result = self.apply_feedback(carry, item)
            return result.state, result

        return jax.lax.scan(step, state, feedback)

    def checkpoint_payload(self, state: PartnerPolicyFusionState) -> dict[str, object]:
        """Return a strict digest-bound JSON-compatible checkpoint."""

        self.validate_state(state)
        construction = self.to_config()
        state_payload: dict[str, object] = {
            "reliability_weights": np.asarray(
                jax.device_get(state.reliability_weights), dtype=np.float32
            ).tolist(),
            "feedback_counts": np.asarray(
                jax.device_get(state.feedback_counts), dtype=np.int32
            ).tolist(),
            "safe_feedback_counts": np.asarray(
                jax.device_get(state.safe_feedback_counts), dtype=np.int32
            ).tolist(),
            "decision_count": int(jax.device_get(state.decision_count)),
            "feedback_applied_count": int(jax.device_get(state.feedback_applied_count)),
            "has_last_decision": bool(jax.device_get(state.has_last_decision)),
            "last_decision_id": int(jax.device_get(state.last_decision_id)),
            "last_event_id": int(jax.device_get(state.last_event_id)),
            "feedback_armed": bool(jax.device_get(state.feedback_armed)),
            "armed_decision_id": int(jax.device_get(state.armed_decision_id)),
            "armed_event_id": int(jax.device_get(state.armed_event_id)),
            "armed_action": int(jax.device_get(state.armed_action)),
            "armed_partner_id": int(jax.device_get(state.armed_partner_id)),
            "armed_route": int(jax.device_get(state.armed_route)),
            "armed_model_features": np.asarray(
                jax.device_get(state.armed_model_features), dtype=np.float32
            ).tolist(),
            "armed_predicted_reliability": float(
                jax.device_get(state.armed_predicted_reliability)
            ),
        }
        return {
            "schema": PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
            "mechanism_status": MECHANISM_STATUS,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "fusion": construction,
            "config_digest": _payload_digest(construction),
            "resource_budget": self.resource_budget.to_config(),
            "state": state_payload,
            "state_digest": _payload_digest(state_payload),
        }

    @staticmethod
    def _checkpoint_int_vector(value: object, *, length: int, name: str) -> Array:
        """Parse an exact JSON int32 vector."""

        try:
            untyped = np.asarray(value, dtype=object)
        except ValueError as error:
            raise ValueError(f"{name} must be a rectangular vector") from error
        if untyped.shape != (length,) or any(type(item) is not int for item in untyped.flat):
            raise ValueError(f"{name} must contain {length} JSON integers")
        if any(item < 0 or item > _INT32_MAX for item in untyped.flat):
            raise ValueError(f"{name} contains an out-of-range counter")
        return jnp.asarray(value, dtype=jnp.int32)

    @staticmethod
    def _checkpoint_float_array(value: object, *, shape: tuple[int, ...], name: str) -> Array:
        """Parse an exact finite JSON float32 array."""

        try:
            untyped = np.asarray(value, dtype=object)
        except ValueError as error:
            raise ValueError(f"{name} must be rectangular") from error
        if untyped.shape != shape or any(type(item) is not float for item in untyped.flat):
            raise ValueError(f"{name} must contain JSON floats with shape {shape}")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            parsed: npt.NDArray[np.float32] = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(parsed)):
            raise ValueError(f"{name} must contain finite float32 values")
        return jnp.asarray(parsed, dtype=jnp.float32)

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, object],
    ) -> tuple[PartnerPolicyFusion, PartnerPolicyFusionState]:
        """Strictly restore an exact v1 construction and state."""

        expected = {
            "schema",
            "mechanism_status",
            "scientific_promotion_allowed",
            "fusion",
            "config_digest",
            "resource_budget",
            "state",
            "state_digest",
        }
        if set(checkpoint) != expected:
            raise ValueError("checkpoint fields do not match partner-fusion v1")
        if checkpoint.get("schema") != PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA:
            raise ValueError("unexpected partner-fusion checkpoint schema")
        if checkpoint.get("mechanism_status") != MECHANISM_STATUS:
            raise ValueError("partner-fusion checkpoint must remain development L0")
        if checkpoint.get("scientific_promotion_allowed") is not False:
            raise ValueError("partner-fusion checkpoint cannot claim promotion")
        construction = checkpoint.get("fusion")
        state_payload = checkpoint.get("state")
        if not isinstance(construction, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint fusion and state fields must be mappings")
        if checkpoint.get("config_digest") != _payload_digest(construction):
            raise ValueError("partner-fusion checkpoint config digest mismatch")
        if checkpoint.get("state_digest") != _payload_digest(state_payload):
            raise ValueError("partner-fusion checkpoint state digest mismatch")
        fusion = cls.from_config(construction)
        if checkpoint.get("resource_budget") != fusion.resource_budget.to_config():
            raise ValueError("checkpoint resource budget differs from construction")
        expected_state = {
            "reliability_weights",
            "feedback_counts",
            "safe_feedback_counts",
            "decision_count",
            "feedback_applied_count",
            "has_last_decision",
            "last_decision_id",
            "last_event_id",
            "feedback_armed",
            "armed_decision_id",
            "armed_event_id",
            "armed_action",
            "armed_partner_id",
            "armed_route",
            "armed_model_features",
            "armed_predicted_reliability",
        }
        if set(state_payload) != expected_state:
            raise ValueError("checkpoint state fields do not match partner-fusion v1")
        cfg = fusion.config

        def scalar_int(name: str, *, allow_missing: bool) -> Array:
            raw = state_payload.get(name)
            lower = -1 if allow_missing else 0
            if type(raw) is not int or not lower <= raw <= _INT32_MAX:
                raise ValueError(f"checkpoint state.{name} is not a valid int32 scalar")
            return jnp.asarray(raw, dtype=jnp.int32)

        def scalar_bool(name: str) -> Array:
            raw = state_payload.get(name)
            if type(raw) is not bool:
                raise ValueError(f"checkpoint state.{name} is not a JSON boolean")
            return jnp.asarray(raw, dtype=jnp.bool_)

        predicted = state_payload.get("armed_predicted_reliability")
        if type(predicted) is not float:
            raise ValueError("armed_predicted_reliability must be a JSON float")
        parsed_predicted = np.float32(predicted)
        if not np.isfinite(parsed_predicted):
            raise ValueError("armed_predicted_reliability must be finite float32")
        state = PartnerPolicyFusionState(
            reliability_weights=cls._checkpoint_float_array(
                state_payload.get("reliability_weights"),
                shape=(cfg.max_partners, cfg.model_feature_dim),
                name="checkpoint state.reliability_weights",
            ),
            feedback_counts=cls._checkpoint_int_vector(
                state_payload.get("feedback_counts"),
                length=cfg.max_partners,
                name="checkpoint state.feedback_counts",
            ),
            safe_feedback_counts=cls._checkpoint_int_vector(
                state_payload.get("safe_feedback_counts"),
                length=cfg.max_partners,
                name="checkpoint state.safe_feedback_counts",
            ),
            decision_count=scalar_int("decision_count", allow_missing=False),
            feedback_applied_count=scalar_int(
                "feedback_applied_count", allow_missing=False
            ),
            has_last_decision=scalar_bool("has_last_decision"),
            last_decision_id=scalar_int("last_decision_id", allow_missing=True),
            last_event_id=scalar_int("last_event_id", allow_missing=True),
            feedback_armed=scalar_bool("feedback_armed"),
            armed_decision_id=scalar_int("armed_decision_id", allow_missing=True),
            armed_event_id=scalar_int("armed_event_id", allow_missing=True),
            armed_action=scalar_int("armed_action", allow_missing=True),
            armed_partner_id=scalar_int("armed_partner_id", allow_missing=True),
            armed_route=scalar_int("armed_route", allow_missing=True),
            armed_model_features=cls._checkpoint_float_array(
                state_payload.get("armed_model_features"),
                shape=(cfg.model_feature_dim,),
                name="checkpoint state.armed_model_features",
            ),
            armed_predicted_reliability=jnp.asarray(parsed_predicted, dtype=jnp.float32),
        )
        fusion.validate_state(state)
        return fusion, state


__all__ = [
    "MECHANISM_STATUS",
    "PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA",
    "PARTNER_POLICY_FUSION_CONFIG_SCHEMA",
    "PartnerFusionAvailability",
    "PartnerFusionDecision",
    "PartnerFusionDecisionResult",
    "PartnerFusionFeedbackResult",
    "PartnerFusionScoreAudit",
    "PartnerFusionShieldResult",
    "PartnerMessageBatch",
    "PartnerPolicyFusion",
    "PartnerPolicyFusionConfig",
    "PartnerPolicyFusionFeedback",
    "PartnerPolicyFusionResourceBudget",
    "PartnerPolicyFusionState",
    "ROUTE_ACCEPT",
    "ROUTE_BLEND",
    "ROUTE_COUNT",
    "ROUTE_IGNORE",
    "ROUTE_QUERY",
    "ROUTE_REQUEST_CLARIFICATION",
    "RELIABILITY_HISTORY_DEVELOPMENT_ONLINE",
    "RELIABILITY_HISTORY_UNAVAILABLE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_BASE",
    "SOURCE_OPTION_KEYBOARD",
    "SOURCE_PARTNER",
    "OptionKeyboardProposal",
]
