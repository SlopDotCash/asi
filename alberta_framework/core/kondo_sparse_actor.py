# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""A bounded nonlinear actor whose real backward pass is selected by Kondo.

This L0 mechanism is the concrete consumer of :mod:`kondo_gate`.  It computes
full-batch categorical logits and selected-action log probabilities, derives a
detached baseline advantage, and lets :class:`KondoGate` form forward admission
intent.  A sample *sparks joy* only when its gradient contribution then enters
an actor backward pass that this consumer actually executes.  ``delight``
retains the paper quantity ``advantage * selected-action surprisal``.

For a sparse transaction, ``KondoGate.gather_sparse`` first creates a fixed
capacity-shaped actor batch.  Only then is ``jax.value_and_grad`` invoked.  A
full-batch loss with zero weights is never reported as sparse work.  When the
gate reports that forced/guardrail or stochastic survivors exceed capacity,
the actor instead performs the explicitly diagnosed full-shape masked
backward, preserving every selected sample.

The protected arrays are returned byte-for-byte at full batch shape with a
canonical digest.  Return targets and baseline predictions enter the actor
path only through a detached advantage; critic and safety features do not
enter the actor loss.  Their downstream learners remain full-batch and
ungated.  This module does not implement those learners, dispatch actions, or
make wall-clock, efficacy, safety, or evidence-promotion claims.

The screen/gather decision is intentionally a host orchestration boundary,
matching ``KondoGate.gather_sparse``.  The fixed-shape nonlinear backward
kernel is pure JAX and supports eager execution, ``jax.jit``, and
``jax.lax.scan``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.kondo_gate import (
    KONDO_GATE_LEGACY_V1_SCHEMA,
    KONDO_GATE_SCHEMA,
    KondoGate,
    KondoGateConfig,
    KondoGateResult,
    KondoGateState,
)

KONDO_SPARSE_ACTOR_SCHEMA = "alberta.kondo-sparse-actor.v1"

_INT32_MAX = 2_147_483_647
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_BATCH_SIZE = 4_096
_MAX_FEATURE_DIM = 4_096
_MAX_HIDDEN_DIM = 2_048
_MAX_ACTION_COUNT = 4_096
_MAX_PROTECTED_DIM = 4_096
_MAX_PARAMETER_COUNT = 16_777_216
_CHECKPOINT_FIELDS = {
    "schema",
    "type",
    "source_sha256",
    "config",
    "state",
    "checkpoint_sha256",
}
_STATE_FIELDS = {
    "parameters",
    "gate_checkpoint",
    "policy_revision",
    "actor_backward_count",
    "sparse_backward_count",
    "full_fallback_count",
}
_PARAMETER_FIELDS = {"hidden_weight", "hidden_bias", "output_weight", "output_bias"}


def _finite_positive_float32(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < _FLOAT32_TINY
        or number > _FLOAT32_MAX
    ):
        raise ValueError(f"{name} must be a positive normal float32 value")
    return float(np.float32(number))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    actual_shape = tuple(cast(Any, value).shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(cast(Any, value).dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_words(data: bytes) -> UInt[Array, " 8"]:
    digest = hashlib.sha256(data).digest()
    return jnp.asarray(
        [
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, 32, 4)
        ],
        dtype=jnp.uint32,
    )


def kondo_sparse_actor_source_sha256() -> str:
    """Return the exact SHA-256 of this source file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class KondoSparseActorConfig:
    """Static dimensions, optimizer step, and exact Kondo allocation."""

    feature_dim: int
    hidden_dim: int
    action_count: int
    critic_dim: int
    safety_dim: int
    learning_rate: float
    gate: KondoGateConfig

    def __post_init__(self) -> None:
        integer_bounds = {
            "feature_dim": (self.feature_dim, _MAX_FEATURE_DIM),
            "hidden_dim": (self.hidden_dim, _MAX_HIDDEN_DIM),
            "action_count": (self.action_count, _MAX_ACTION_COUNT),
            "critic_dim": (self.critic_dim, _MAX_PROTECTED_DIM),
            "safety_dim": (self.safety_dim, _MAX_PROTECTED_DIM),
        }
        for name, (value, maximum) in integer_bounds.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}]")
        if not isinstance(self.gate, KondoGateConfig):
            raise TypeError("gate must be KondoGateConfig")
        if not 1 <= self.gate.batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(f"gate.batch_size must be in [1, {_MAX_BATCH_SIZE}]")
        parameter_count = (
            self.feature_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.action_count
            + self.action_count
        )
        if parameter_count > _MAX_PARAMETER_COUNT:
            raise ValueError("nonlinear actor parameter count exceeds the finite cap")
        for name, product in (
            ("actor feature slots", self.gate.batch_size * self.feature_dim),
            ("critic feature slots", self.gate.batch_size * self.critic_dim),
            ("safety feature slots", self.gate.batch_size * self.safety_dim),
            ("screening accounting", self.gate.batch_size * self.gate.max_screenings),
        ):
            if product > _INT32_MAX:
                raise ValueError(f"{name} exceeds signed-int32 accounting")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_positive_float32("learning_rate", self.learning_rate),
        )

    @property
    def batch_size(self) -> int:
        return self.gate.batch_size

    @property
    def backward_capacity(self) -> int:
        return self.gate.backward_capacity

    @property
    def parameter_count(self) -> int:
        return (
            self.feature_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.action_count
            + self.action_count
        )

    def _to_config_for_gate_schema(self, gate_schema: str) -> dict[str, object]:
        return {
            "schema": KONDO_SPARSE_ACTOR_SCHEMA,
            "type": "KondoSparseActorConfig",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "action_count": self.action_count,
            "critic_dim": self.critic_dim,
            "safety_dim": self.safety_dim,
            "learning_rate": self.learning_rate,
            "gate": self.gate._to_config_for_schema(gate_schema),
            "delight_semantics": "advantage-times-selected-action-surprisal",
            "sparks_joy_semantics": (
                "gradient-contribution-entered-executed-actor-backward"
            ),
            "baseline_gradient_gated": False,
            "critic_gradient_gated": False,
            "safety_gradient_gated": False,
            "wall_clock_claimed": False,
            "efficacy_claimed": False,
            "safety_claimed": False,
            "evidence_promotion_claimed": False,
        }

    def to_config(self) -> dict[str, object]:
        """Emit the actor contract with a canonical v2 embedded gate."""
        return self._to_config_for_gate_schema(KONDO_GATE_SCHEMA)

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoSparseActorConfig:
        expected = {
            "schema",
            "type",
            "feature_dim",
            "hidden_dim",
            "action_count",
            "critic_dim",
            "safety_dim",
            "learning_rate",
            "gate",
            "delight_semantics",
            "sparks_joy_semantics",
            "baseline_gradient_gated",
            "critic_gradient_gated",
            "safety_gradient_gated",
            "wall_clock_claimed",
            "efficacy_claimed",
            "safety_claimed",
            "evidence_promotion_claimed",
        }
        if set(payload) != expected:
            raise ValueError("Kondo sparse actor config fields do not match v1")
        fixed: dict[str, object] = {
            "schema": KONDO_SPARSE_ACTOR_SCHEMA,
            "type": "KondoSparseActorConfig",
            "delight_semantics": "advantage-times-selected-action-surprisal",
            "sparks_joy_semantics": (
                "gradient-contribution-entered-executed-actor-backward"
            ),
            "baseline_gradient_gated": False,
            "critic_gradient_gated": False,
            "safety_gradient_gated": False,
            "wall_clock_claimed": False,
            "efficacy_claimed": False,
            "safety_claimed": False,
            "evidence_promotion_claimed": False,
        }
        for name, value in fixed.items():
            if type(payload.get(name)) is not type(value) or payload.get(name) != value:
                raise ValueError(f"Kondo sparse actor {name} is invalid")
        for name in ("feature_dim", "hidden_dim", "action_count", "critic_dim", "safety_dim"):
            if type(payload[name]) is not int:
                raise ValueError(f"Kondo sparse actor {name} must be an integer")
        if type(payload["learning_rate"]) is not float:
            raise ValueError("Kondo sparse actor learning_rate must be a float")
        gate_payload = payload["gate"]
        if not isinstance(gate_payload, Mapping):
            raise ValueError("Kondo sparse actor gate must be an object")
        result = cls(
            feature_dim=cast(int, payload["feature_dim"]),
            hidden_dim=cast(int, payload["hidden_dim"]),
            action_count=cast(int, payload["action_count"]),
            critic_dim=cast(int, payload["critic_dim"]),
            safety_dim=cast(int, payload["safety_dim"]),
            learning_rate=payload["learning_rate"],
            gate=KondoGateConfig.from_config(cast(Mapping[str, object], gate_payload)),
        )
        gate_schema = gate_payload.get("schema")
        if gate_schema not in (KONDO_GATE_SCHEMA, KONDO_GATE_LEGACY_V1_SCHEMA):
            raise ValueError("Kondo sparse actor gate schema is invalid")
        if result._to_config_for_gate_schema(cast(str, gate_schema)) != dict(payload):
            raise ValueError("Kondo sparse actor config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class KondoActorParameters:
    """Nonlinear tanh categorical actor parameters."""

    hidden_weight: Float[Array, " feature hidden"]
    hidden_bias: Float[Array, " hidden"]
    output_weight: Float[Array, " hidden action"]
    output_bias: Float[Array, " action"]


@chex.dataclass(frozen=True)
class KondoActorProtectedInputs:
    """Full-fidelity learner inputs that are never actor-gated."""

    critic_features: Float[Array, " batch critic"]
    baseline_predictions: Float[Array, " batch"]
    return_targets: Float[Array, " batch"]
    safety_features: Float[Array, " batch safety"]


@chex.dataclass(frozen=True)
class KondoSparseActorBatch:
    """One exact on-policy actor batch plus protected full-batch channels."""

    actor_features: Float[Array, " batch feature"]
    actions: Int[Array, " batch"]
    action_identity: Int[Array, " batch"]
    policy_revision: Int[Array, " batch"]
    behavior_log_probability: Float[Array, " batch"]
    valid_mask: Bool[Array, " batch"]
    force_keep_mask: Bool[Array, " batch"]
    protected: KondoActorProtectedInputs


@chex.dataclass(frozen=True)
class KondoActorBackwardBatch:
    """Fixed-shape input to the actual nonlinear actor backward kernel."""

    actor_features: Float[Array, " slots feature"]
    actions: Int[Array, " slots"]
    advantage: Float[Array, " slots"]
    sample_mask: Bool[Array, " slots"]


@chex.dataclass(frozen=True)
class KondoSparseActorState:
    """Actor snapshot and exact committed backward accounting."""

    parameters: KondoActorParameters
    gate_state: KondoGateState
    policy_revision: Int[Array, ""]
    actor_backward_count: Int[Array, ""]
    sparse_backward_count: Int[Array, ""]
    full_fallback_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class KondoActorBackwardResult:
    """Loss and parameter gradient from one fixed-shape actor batch."""

    loss: Float[Array, ""]
    gradient: KondoActorParameters
    selected_log_probability: Float[Array, " slots"]
    delight: Float[Array, " slots"]
    selected_count: Int[Array, ""]
    gradient_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class KondoActorBackwardSummary:
    """Committed backward accounting without the per-slot autodiff auxiliaries.

    The full :class:`KondoActorBackwardResult` also carries the per-slot
    ``selected_log_probability`` and ``delight`` auxiliaries that fall out of the
    canonical ``value_and_grad(_actor_loss, has_aux=True)`` backward pass.  Some
    evaluation and development lanes reconstruct a backward record from an
    already-committed :class:`KondoSparseActorResult` transaction, which does not
    carry those two per-slot quantities; there is no honest local source for
    them at those sites.  Every consumer of a backward record reads only ``loss``,
    ``gradient``, ``selected_count``, and ``gradient_finite`` — so those lanes
    carry exactly that committed subset here rather than fabricating per-slot
    values to satisfy the full constructor.  Keeping the two types distinct means
    the core producer at :func:`_actor_backward` still errors if it ever omits a
    real auxiliary.
    """

    loss: Float[Array, ""]
    gradient: KondoActorParameters
    selected_count: Int[Array, ""]
    gradient_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class KondoSparseActorResult:
    """One proposed/committed Kondo actor transaction."""

    state: KondoSparseActorState
    screen: KondoGateResult
    gradient: KondoActorParameters
    actor_loss: Float[Array, ""]
    advantage: Float[Array, " batch"]
    current_action_log_probability: Float[Array, " batch"]
    protected: KondoActorProtectedInputs
    protected_digest: UInt[Array, " 8"]
    protected_slots: Int[Array, ""]
    input_finite: Bool[Array, ""]
    action_identity_valid: Bool[Array, ""]
    policy_revision_valid: Bool[Array, ""]
    behavior_log_probability_valid: Bool[Array, ""]
    force_keep_contract_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    gradient_finite: Bool[Array, ""]
    updated_parameters_finite: Bool[Array, ""]
    sparse_backward_used: Bool[Array, ""]
    full_shape_masked_backward_used: Bool[Array, ""]
    backward_batch_size: Int[Array, ""]
    backward_selected_count: Int[Array, ""]
    executed_actor_backward_mask: Bool[Array, " batch"]
    executed_delight: Float[Array, " batch"]
    backward_delight_exact: Bool[Array, ""]
    policy_revision_before: Int[Array, ""]
    policy_revision_after: Int[Array, ""]
    transaction_applied: Bool[Array, ""]

    @property
    def sparks_joy(self) -> Bool[Array, " batch"]:
        """Samples whose contribution entered an actor backward that executed.

        This records backward execution, independently of later gradient
        finiteness or parameter-transaction acceptance.
        """
        return self.executed_actor_backward_mask


@dataclasses.dataclass(frozen=True)
class KondoSparseActorResourceDeclaration:
    """Exact logical allocation, never a latency or measured-FLOP claim."""

    full_forward_batch_size: int
    sparse_backward_capacity: int
    full_fallback_batch_size: int
    nonlinear_parameter_count: int
    protected_full_fidelity_slots_per_step: int
    persistent_state_bytes: int
    maximum_actor_backward_passes_per_step: int
    maximum_random_draws_per_step: int
    maximum_delight_products_per_step: int
    full_shape_fallback_possible: bool
    source_sha256: str
    wall_clock_savings_claimed: bool = False
    efficacy_claimed: bool = False
    safety_claimed: bool = False
    evidence_promotion_claimed: bool = False

    def to_config(self) -> dict[str, int | bool | str]:
        return dataclasses.asdict(self)


def _actor_selected_log_probability(
    parameters: KondoActorParameters,
    actor_features: Array,
    actions: Array,
) -> Array:
    hidden = jnp.tanh(
        actor_features @ parameters.hidden_weight + parameters.hidden_bias
    )
    logits = hidden @ parameters.output_weight + parameters.output_bias
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probabilities, actions[:, None], axis=1)[:, 0]


def _actor_loss(
    parameters: KondoActorParameters,
    batch: KondoActorBackwardBatch,
) -> tuple[Array, tuple[Array, Array]]:
    selected_log_probability = _actor_selected_log_probability(
        parameters,
        batch.actor_features,
        batch.actions,
    )
    sample_weight = batch.sample_mask.astype(jnp.float32)
    advantage = jax.lax.stop_gradient(batch.advantage)
    delight = advantage * -selected_log_probability
    coefficient = advantage * sample_weight
    denominator = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = -jnp.sum(coefficient * selected_log_probability) / denominator
    return loss, (selected_log_probability, delight)


def _tree_all_finite(tree: object) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    result = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in leaves:
        result = result & jnp.all(jnp.isfinite(leaf))
    return result


@functools.partial(jax.jit, static_argnums=())
def kondo_actor_backward_kernel(
    parameters: KondoActorParameters,
    batch: KondoActorBackwardBatch,
) -> KondoActorBackwardResult:
    """Run one real fixed-shape actor autodiff backward pass.

    Callers are responsible for supplying either the audited gathered capacity
    batch or the explicit full-shape fallback batch.  The host consumer below
    is the authority for that choice.
    """
    (loss, (selected_log_probability, delight)), gradient = jax.value_and_grad(
        _actor_loss,
        has_aux=True,
    )(parameters, batch)
    selected_count = jnp.sum(batch.sample_mask.astype(jnp.int32))
    gradient_finite = jnp.isfinite(loss) & _tree_all_finite(gradient)
    return KondoActorBackwardResult(
        loss=loss,
        gradient=gradient,
        selected_log_probability=selected_log_probability,
        delight=delight,
        selected_count=selected_count,
        gradient_finite=gradient_finite,
    )


def _encode_float32_array(array: Array) -> dict[str, object]:
    host = np.asarray(array)
    if host.dtype != np.float32:
        raise TypeError("checkpoint parameter arrays must be float32")
    return {
        "dtype": "float32",
        "shape": list(host.shape),
        "data_hex": np.ascontiguousarray(host).tobytes().hex(),
    }


def _decode_float32_array(
    payload: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Array:
    if not isinstance(payload, Mapping) or set(payload) != {"dtype", "shape", "data_hex"}:
        raise ValueError(f"checkpoint {name} must be a canonical array object")
    if payload.get("dtype") != "float32" or payload.get("shape") != list(shape):
        raise ValueError(f"checkpoint {name} dtype/shape is invalid")
    data_hex = payload.get("data_hex")
    if type(data_hex) is not str or len(data_hex) != 8 * math.prod(shape):
        raise ValueError(f"checkpoint {name} data_hex length is invalid")
    try:
        raw = bytes.fromhex(data_hex)
    except ValueError as error:
        raise ValueError(f"checkpoint {name} data_hex is invalid") from error
    host = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    return jnp.asarray(host, dtype=jnp.float32)


class KondoSparseActor:
    """Host-audited Kondo screen plus a real nonlinear actor backward."""

    def __init__(self, config: KondoSparseActorConfig):
        if not isinstance(config, KondoSparseActorConfig):
            raise TypeError("config must be KondoSparseActorConfig")
        self._config = config
        self._gate = KondoGate(config.gate)
        self._config_id = _sha256_words(_canonical_json_bytes(config.to_config()))

    @property
    def config(self) -> KondoSparseActorConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoSparseActor:
        return cls(KondoSparseActorConfig.from_config(payload))

    def _validate_parameters_static(self, parameters: KondoActorParameters) -> None:
        if not isinstance(parameters, KondoActorParameters):
            raise TypeError("parameters must be KondoActorParameters")
        cfg = self._config
        _require_array(
            parameters.hidden_weight,
            name="parameters.hidden_weight",
            shape=(cfg.feature_dim, cfg.hidden_dim),
            dtype=jnp.float32,
        )
        _require_array(
            parameters.hidden_bias,
            name="parameters.hidden_bias",
            shape=(cfg.hidden_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            parameters.output_weight,
            name="parameters.output_weight",
            shape=(cfg.hidden_dim, cfg.action_count),
            dtype=jnp.float32,
        )
        _require_array(
            parameters.output_bias,
            name="parameters.output_bias",
            shape=(cfg.action_count,),
            dtype=jnp.float32,
        )

    def _validate_protected_static(self, protected: KondoActorProtectedInputs) -> None:
        if not isinstance(protected, KondoActorProtectedInputs):
            raise TypeError("batch.protected must be KondoActorProtectedInputs")
        cfg = self._config
        _require_array(
            protected.critic_features,
            name="protected.critic_features",
            shape=(cfg.batch_size, cfg.critic_dim),
            dtype=jnp.float32,
        )
        for name in ("baseline_predictions", "return_targets"):
            _require_array(
                getattr(protected, name),
                name=f"protected.{name}",
                shape=(cfg.batch_size,),
                dtype=jnp.float32,
            )
        _require_array(
            protected.safety_features,
            name="protected.safety_features",
            shape=(cfg.batch_size, cfg.safety_dim),
            dtype=jnp.float32,
        )

    def _validate_batch_static(self, batch: KondoSparseActorBatch) -> None:
        if not isinstance(batch, KondoSparseActorBatch):
            raise TypeError("batch must be KondoSparseActorBatch")
        cfg = self._config
        _require_array(
            batch.actor_features,
            name="batch.actor_features",
            shape=(cfg.batch_size, cfg.feature_dim),
            dtype=jnp.float32,
        )
        for name in ("actions", "action_identity", "policy_revision"):
            _require_array(
                getattr(batch, name),
                name=f"batch.{name}",
                shape=(cfg.batch_size,),
                dtype=jnp.int32,
            )
        _require_array(
            batch.behavior_log_probability,
            name="batch.behavior_log_probability",
            shape=(cfg.batch_size,),
            dtype=jnp.float32,
        )
        for name in ("valid_mask", "force_keep_mask"):
            _require_array(
                getattr(batch, name),
                name=f"batch.{name}",
                shape=(cfg.batch_size,),
                dtype=jnp.bool_,
            )
        self._validate_protected_static(batch.protected)

    def _validate_state_static(self, state: KondoSparseActorState) -> None:
        if not isinstance(state, KondoSparseActorState):
            raise TypeError("state must be KondoSparseActorState")
        self._validate_parameters_static(state.parameters)
        self._gate._validate_state_static(state.gate_state)
        for name in (
            "policy_revision",
            "actor_backward_count",
            "sparse_backward_count",
            "full_fallback_count",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(),
                dtype=jnp.int32,
            )

    def _state_valid(self, state: KondoSparseActorState) -> Array:
        return (
            _tree_all_finite(state.parameters)
            & self._gate._state_valid(state.gate_state)
            & (state.policy_revision >= 0)
            & (state.policy_revision == state.actor_backward_count)
            & (state.actor_backward_count >= 0)
            & (state.actor_backward_count <= self._config.gate.max_screenings)
            & (state.sparse_backward_count >= 0)
            & (state.full_fallback_count >= 0)
            & (
                state.sparse_backward_count + state.full_fallback_count
                == state.actor_backward_count
            )
            & (state.gate_state.screen_count == state.actor_backward_count)
            & (state.gate_state.sparse_batch_count == state.sparse_backward_count)
            & (state.gate_state.full_fallback_count == state.full_fallback_count)
        )

    def init(
        self,
        parameters: KondoActorParameters,
        gate_key: Array,
    ) -> KondoSparseActorState:
        """Initialize from caller-owned deterministic parameters and Kondo RNG."""
        self._validate_parameters_static(parameters)
        if not bool(np.asarray(_tree_all_finite(parameters))):
            raise ValueError("initial actor parameters must be finite")
        zero = jnp.asarray(0, dtype=jnp.int32)
        return KondoSparseActorState(
            parameters=parameters,
            gate_state=self._gate.init(gate_key),
            policy_revision=zero,
            actor_backward_count=zero,
            sparse_backward_count=zero,
            full_fallback_count=zero,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _selected_log_probability_jit(
        self,
        parameters: KondoActorParameters,
        actor_features: Array,
        actions: Array,
    ) -> Array:
        return _actor_selected_log_probability(parameters, actor_features, actions)

    def behavior_log_probability(
        self,
        state: KondoSparseActorState,
        actor_features: Float[Array, " batch feature"],
        actions: Int[Array, " batch"],
    ) -> Float[Array, " batch"]:
        """Produce the exact behavior log-probability contract for a snapshot."""
        self._validate_state_static(state)
        _require_array(
            actor_features,
            name="actor_features",
            shape=(self._config.batch_size, self._config.feature_dim),
            dtype=jnp.float32,
        )
        _require_array(
            actions,
            name="actions",
            shape=(self._config.batch_size,),
            dtype=jnp.int32,
        )
        host_actions = np.asarray(actions)
        if np.any(host_actions < 0) or np.any(host_actions >= self._config.action_count):
            raise ValueError("actions must name configured categorical actions")
        return cast(
            Float[Array, " batch"],
            self._selected_log_probability_jit(
                state.parameters,
                actor_features,
                actions,
            ),
        )

    def _protected_digest(self, protected: KondoActorProtectedInputs) -> Array:
        digest = hashlib.sha256()
        for name in (
            "critic_features",
            "baseline_predictions",
            "return_targets",
            "safety_features",
        ):
            value = np.asarray(getattr(protected, name))
            header = {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            digest.update(_canonical_json_bytes(header))
            digest.update(np.ascontiguousarray(value).tobytes())
        return _sha256_words(digest.digest())

    def _zero_gradient(self) -> KondoActorParameters:
        cfg = self._config
        return KondoActorParameters(
            hidden_weight=jnp.zeros((cfg.feature_dim, cfg.hidden_dim), dtype=jnp.float32),
            hidden_bias=jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32),
            output_weight=jnp.zeros((cfg.hidden_dim, cfg.action_count), dtype=jnp.float32),
            output_bias=jnp.zeros((cfg.action_count,), dtype=jnp.float32),
        )

    def _backward_batch_static(
        self,
        batch: KondoActorBackwardBatch,
        *,
        slots: int,
    ) -> None:
        if not isinstance(batch, KondoActorBackwardBatch):
            raise TypeError("backward batch must be KondoActorBackwardBatch")
        _require_array(
            batch.actor_features,
            name="backward.actor_features",
            shape=(slots, self._config.feature_dim),
            dtype=jnp.float32,
        )
        _require_array(
            batch.actions,
            name="backward.actions",
            shape=(slots,),
            dtype=jnp.int32,
        )
        _require_array(
            batch.advantage,
            name="backward.advantage",
            shape=(slots,),
            dtype=jnp.float32,
        )
        _require_array(
            batch.sample_mask,
            name="backward.sample_mask",
            shape=(slots,),
            dtype=jnp.bool_,
        )

    def sparse_backward(
        self,
        parameters: KondoActorParameters,
        batch: KondoActorBackwardBatch,
    ) -> KondoActorBackwardResult:
        """Pure JAX capacity-shape kernel exposed for JIT/scan verification."""
        self._validate_parameters_static(parameters)
        self._backward_batch_static(batch, slots=self._config.backward_capacity)
        return kondo_actor_backward_kernel(parameters, batch)

    def full_shape_masked_backward(
        self,
        parameters: KondoActorParameters,
        batch: KondoActorBackwardBatch,
    ) -> KondoActorBackwardResult:
        """Pure JAX full-shape fallback kernel; it is never called sparse."""
        self._validate_parameters_static(parameters)
        self._backward_batch_static(batch, slots=self._config.batch_size)
        return kondo_actor_backward_kernel(parameters, batch)

    def _input_diagnostics(
        self,
        state: KondoSparseActorState,
        batch: KondoSparseActorBatch,
        current_log_probability: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        arrays = (
            batch.actor_features,
            batch.behavior_log_probability,
            batch.protected.critic_features,
            batch.protected.baseline_predictions,
            batch.protected.return_targets,
            batch.protected.safety_features,
        )
        input_finite = jnp.asarray(True, dtype=jnp.bool_)
        for array in arrays:
            input_finite = input_finite & jnp.all(jnp.isfinite(array))
        action_identity_valid = (
            jnp.all(batch.actions >= 0)
            & jnp.all(batch.actions < self._config.action_count)
            & jnp.all(batch.action_identity == batch.actions)
        )
        policy_revision_valid = jnp.all(batch.policy_revision == state.policy_revision)
        current_bits = jax.lax.bitcast_convert_type(current_log_probability, jnp.uint32)
        behavior_bits = jax.lax.bitcast_convert_type(
            batch.behavior_log_probability,
            jnp.uint32,
        )
        behavior_valid = jnp.all(current_bits == behavior_bits)
        force_keep_valid = jnp.all(~batch.force_keep_mask | batch.valid_mask)
        has_valid_sample = jnp.any(batch.valid_mask)
        return (
            input_finite,
            action_identity_valid,
            policy_revision_valid,
            behavior_valid,
            force_keep_valid,
            has_valid_sample,
        )

    def step(
        self,
        state: KondoSparseActorState,
        batch: KondoSparseActorBatch,
    ) -> KondoSparseActorResult:
        """Screen, gather, and only then execute one actor backward pass.

        This method is host orchestration by design.  Invalid dynamic inputs or
        state reject the whole transaction without consuming Kondo RNG or
        changing actor parameters.  Static shape/dtype violations raise before
        tracing.
        """
        self._validate_state_static(state)
        self._validate_batch_static(batch)
        cfg = self._config
        protected_digest = self._protected_digest(batch.protected)
        safe_actions = jnp.clip(batch.actions, 0, cfg.action_count - 1)
        current_log_probability = self._selected_log_probability_jit(
            state.parameters,
            batch.actor_features,
            safe_actions,
        )
        advantage = jax.lax.stop_gradient(
            batch.protected.return_targets - batch.protected.baseline_predictions
        )
        (
            input_finite,
            action_identity_valid,
            policy_revision_valid,
            behavior_valid,
            force_keep_valid,
            has_valid_sample,
        ) = self._input_diagnostics(state, batch, current_log_probability)
        state_valid = self._state_valid(state)
        contract_valid = (
            input_finite
            & action_identity_valid
            & policy_revision_valid
            & behavior_valid
            & force_keep_valid
            & has_valid_sample
            & state_valid
            & jnp.all(jnp.isfinite(advantage))
            & jnp.all(jnp.isfinite(current_log_probability))
            & jnp.all(current_log_probability <= 0.0)
        )
        rejected_marker = jnp.asarray(jnp.nan, dtype=jnp.float32)
        screened_advantage = jnp.where(contract_valid, advantage, rejected_marker)
        screened_log_probability = jnp.where(
            contract_valid,
            current_log_probability,
            rejected_marker,
        )
        screen = self._gate.screen(
            state.gate_state,
            screened_advantage,
            screened_log_probability,
            batch.valid_mask,
            batch.force_keep_mask,
        )

        zero_gradient = self._zero_gradient()
        screen_applied = bool(np.asarray(screen.transaction_applied))
        if not screen_applied:
            return KondoSparseActorResult(
                state=state,
                screen=screen,
                gradient=zero_gradient,
                actor_loss=jnp.asarray(0.0, dtype=jnp.float32),
                advantage=advantage,
                current_action_log_probability=current_log_probability,
                protected=batch.protected,
                protected_digest=protected_digest,
                protected_slots=jnp.asarray(cfg.batch_size, dtype=jnp.int32),
                input_finite=input_finite,
                action_identity_valid=action_identity_valid,
                policy_revision_valid=policy_revision_valid,
                behavior_log_probability_valid=behavior_valid,
                force_keep_contract_valid=force_keep_valid,
                state_valid=state_valid,
                gradient_finite=jnp.asarray(False, dtype=jnp.bool_),
                updated_parameters_finite=jnp.asarray(False, dtype=jnp.bool_),
                sparse_backward_used=jnp.asarray(False, dtype=jnp.bool_),
                full_shape_masked_backward_used=jnp.asarray(False, dtype=jnp.bool_),
                backward_batch_size=jnp.asarray(0, dtype=jnp.int32),
                backward_selected_count=jnp.asarray(0, dtype=jnp.int32),
                executed_actor_backward_mask=jnp.zeros(
                    (cfg.batch_size,),
                    dtype=jnp.bool_,
                ),
                executed_delight=jnp.zeros(
                    (cfg.batch_size,),
                    dtype=jnp.float32,
                ),
                backward_delight_exact=jnp.asarray(False, dtype=jnp.bool_),
                policy_revision_before=state.policy_revision,
                policy_revision_after=state.policy_revision,
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
            )

        sparse_used = bool(np.asarray(screen.sparse_backward_available))
        full_used = bool(np.asarray(screen.full_shape_masked_backward_required))
        if sparse_used == full_used:
            raise ValueError("Kondo screen must select exactly one backward shape")
        if sparse_used:
            gathered = self._gate.gather_sparse(
                {
                    "actor_features": batch.actor_features,
                    "actions": batch.actions,
                    "advantage": advantage,
                },
                screen,
            )
            backward_batch = KondoActorBackwardBatch(
                actor_features=gathered.data["actor_features"],
                actions=gathered.data["actions"],
                advantage=gathered.data["advantage"],
                sample_mask=gathered.sample_mask,
            )
            # value_and_grad is reached only after the audited capacity gather.
            backward = self.sparse_backward(state.parameters, backward_batch)
            backward_batch_size = cfg.backward_capacity
            backward_entry_mask = jnp.any(
                (
                    jnp.arange(cfg.batch_size, dtype=jnp.int32)[:, None]
                    == gathered.source_indices[None, :]
                )
                & gathered.sample_mask[None, :],
                axis=1,
            )
            executed_delight = jnp.zeros((cfg.batch_size,), dtype=jnp.float32)
            source_rows = jnp.arange(cfg.batch_size, dtype=jnp.int32)
            for slot in range(cfg.backward_capacity):
                source_matches = source_rows == gathered.source_indices[slot]
                executed_delight = jnp.where(
                    gathered.sample_mask[slot] & source_matches,
                    backward.delight[slot],
                    executed_delight,
                )
            expected_backward_log_probability = current_log_probability[
                gathered.source_indices
            ]
            expected_backward_delight = screen.delight[gathered.source_indices]
        else:
            backward_batch = KondoActorBackwardBatch(
                actor_features=batch.actor_features,
                actions=batch.actions,
                advantage=advantage,
                sample_mask=screen.selected_mask,
            )
            # Every selected/forced row is retained in this full-shape fallback.
            backward = self.full_shape_masked_backward(state.parameters, backward_batch)
            backward_batch_size = cfg.batch_size
            backward_entry_mask = backward_batch.sample_mask
            executed_delight = jnp.where(
                backward_batch.sample_mask,
                backward.delight,
                jnp.zeros_like(backward.delight),
            )
            expected_backward_log_probability = current_log_probability
            expected_backward_delight = screen.delight

        backward_log_probability_exact = jnp.all(
            ~backward_batch.sample_mask
            | (
                jax.lax.bitcast_convert_type(
                    backward.selected_log_probability,
                    jnp.uint32,
                )
                == jax.lax.bitcast_convert_type(
                    expected_backward_log_probability,
                    jnp.uint32,
                )
            )
        )
        backward_delight_exact = jnp.all(
            ~backward_batch.sample_mask
            | (
                jax.lax.bitcast_convert_type(backward.delight, jnp.uint32)
                == jax.lax.bitcast_convert_type(
                    expected_backward_delight,
                    jnp.uint32,
                )
            )
        )
        backward_entry_exact = (
            jnp.array_equal(backward_entry_mask, screen.selected_mask)
            & (backward.selected_count == jnp.sum(backward_entry_mask.astype(jnp.int32)))
            & (backward.selected_count == screen.selected_count)
            & backward_log_probability_exact
            & backward_delight_exact
        )
        if not bool(np.asarray(backward_entry_exact)):
            raise ValueError("actor backward entries differ from the audited Kondo selection")

        learning_rate = jnp.asarray(cfg.learning_rate, dtype=jnp.float32)
        updated_parameters = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter - learning_rate * gradient,
            state.parameters,
            backward.gradient,
        )
        updated_parameters_finite = _tree_all_finite(updated_parameters)
        transaction_applied = backward.gradient_finite & updated_parameters_finite
        applied = bool(np.asarray(transaction_applied))
        if applied:
            one = jnp.asarray(1, dtype=jnp.int32)
            next_state = KondoSparseActorState(
                parameters=cast(KondoActorParameters, updated_parameters),
                gate_state=screen.state,
                policy_revision=state.policy_revision + one,
                actor_backward_count=state.actor_backward_count + one,
                sparse_backward_count=(
                    state.sparse_backward_count
                    + jnp.asarray(int(sparse_used), dtype=jnp.int32)
                ),
                full_fallback_count=(
                    state.full_fallback_count
                    + jnp.asarray(int(full_used), dtype=jnp.int32)
                ),
            )
            if not bool(np.asarray(self._state_valid(next_state))):
                raise ValueError("committed Kondo sparse actor state is inconsistent")
        else:
            next_state = state
        return KondoSparseActorResult(
            state=next_state,
            screen=screen,
            gradient=backward.gradient,
            actor_loss=backward.loss,
            advantage=advantage,
            current_action_log_probability=current_log_probability,
            protected=batch.protected,
            protected_digest=protected_digest,
            protected_slots=jnp.asarray(cfg.batch_size, dtype=jnp.int32),
            input_finite=input_finite,
            action_identity_valid=action_identity_valid,
            policy_revision_valid=policy_revision_valid,
            behavior_log_probability_valid=behavior_valid,
            force_keep_contract_valid=force_keep_valid,
            state_valid=state_valid,
            gradient_finite=backward.gradient_finite,
            updated_parameters_finite=updated_parameters_finite,
            sparse_backward_used=jnp.asarray(sparse_used, dtype=jnp.bool_),
            full_shape_masked_backward_used=jnp.asarray(full_used, dtype=jnp.bool_),
            backward_batch_size=jnp.asarray(backward_batch_size, dtype=jnp.int32),
            backward_selected_count=backward.selected_count,
            executed_actor_backward_mask=backward_entry_mask,
            executed_delight=executed_delight,
            backward_delight_exact=backward_delight_exact,
            policy_revision_before=state.policy_revision,
            policy_revision_after=next_state.policy_revision,
            transaction_applied=transaction_applied,
        )

    def resource_declaration(
        self,
        state: KondoSparseActorState | None = None,
    ) -> KondoSparseActorResourceDeclaration:
        sample = state
        if sample is None:
            zero_parameters = self._zero_gradient()
            sample = self.init(zero_parameters, jr.key(0, impl="threefry2x32"))
        self._validate_state_static(sample)
        if not bool(np.asarray(self._state_valid(sample))):
            raise ValueError("resource accounting requires a valid actor state")
        state_bytes = sum(
            int(cast(Any, leaf).nbytes) for leaf in jax.tree_util.tree_leaves(sample)
        )
        gate_resources = self._gate.resource_declaration(sample.gate_state)
        cfg = self._config
        return KondoSparseActorResourceDeclaration(
            full_forward_batch_size=cfg.batch_size,
            sparse_backward_capacity=cfg.backward_capacity,
            full_fallback_batch_size=cfg.batch_size,
            nonlinear_parameter_count=cfg.parameter_count,
            protected_full_fidelity_slots_per_step=cfg.batch_size,
            persistent_state_bytes=state_bytes,
            maximum_actor_backward_passes_per_step=1,
            maximum_random_draws_per_step=gate_resources.maximum_random_draws_per_screen,
            maximum_delight_products_per_step=(
                gate_resources.maximum_delight_products_per_screen
            ),
            full_shape_fallback_possible=gate_resources.full_shape_fallback_possible,
            source_sha256=kondo_sparse_actor_source_sha256(),
        )

    def _checkpoint_payload_for_gate_schema(
        self,
        state: KondoSparseActorState,
        gate_schema: str,
    ) -> dict[str, object]:
        self._validate_state_static(state)
        if not bool(np.asarray(self._state_valid(state))):
            raise ValueError("Kondo sparse actor state is invalid")
        parameters = {
            name: _encode_float32_array(getattr(state.parameters, name))
            for name in sorted(_PARAMETER_FIELDS)
        }
        body: dict[str, object] = {
            "schema": KONDO_SPARSE_ACTOR_SCHEMA,
            "type": "KondoSparseActorCheckpoint",
            "source_sha256": kondo_sparse_actor_source_sha256(),
            "config": self._config._to_config_for_gate_schema(gate_schema),
            "state": {
                "parameters": parameters,
                "gate_checkpoint": self._gate._checkpoint_payload_for_schema(
                    state.gate_state,
                    gate_schema,
                ),
                "policy_revision": int(np.asarray(state.policy_revision)),
                "actor_backward_count": int(np.asarray(state.actor_backward_count)),
                "sparse_backward_count": int(np.asarray(state.sparse_backward_count)),
                "full_fallback_count": int(np.asarray(state.full_fallback_count)),
            },
        }
        return {
            **body,
            "checkpoint_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        }

    def checkpoint_payload(self, state: KondoSparseActorState) -> dict[str, object]:
        """Serialize exact actor bits with only the canonical v2 embedded gate."""
        return self._checkpoint_payload_for_gate_schema(state, KONDO_GATE_SCHEMA)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> tuple[KondoSparseActor, KondoSparseActorState]:
        """Restore an integrity-checked, source-bound outer-v1 checkpoint.

        New outer-v1 payloads embed the canonical v2 gate. Exact historical
        outer-v1 payloads with matching legacy-v1 gate config and checkpoint
        layers remain importable and normalize to the v2 gate on re-emission.

        The SHA-256 checksum is unkeyed: it detects ordinary corruption and
        noncanonical state, but it is not a cryptographic authenticity proof.
        """
        if set(payload) != _CHECKPOINT_FIELDS:
            raise ValueError("Kondo sparse actor checkpoint fields do not match v1")
        if payload.get("schema") != KONDO_SPARSE_ACTOR_SCHEMA:
            raise ValueError("Kondo sparse actor checkpoint schema is invalid")
        if payload.get("type") != "KondoSparseActorCheckpoint":
            raise ValueError("Kondo sparse actor checkpoint type is invalid")
        if payload.get("source_sha256") != kondo_sparse_actor_source_sha256():
            raise ValueError("Kondo sparse actor checkpoint source hash is invalid")
        checkpoint_sha256 = payload.get("checkpoint_sha256")
        if type(checkpoint_sha256) is not str or len(checkpoint_sha256) != 64:
            raise ValueError("Kondo sparse actor checkpoint digest is invalid")
        body = {name: payload[name] for name in payload if name != "checkpoint_sha256"}
        expected_digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        if checkpoint_sha256 != expected_digest:
            raise ValueError("Kondo sparse actor checkpoint integrity check failed")
        config_payload = payload.get("config")
        state_payload = payload.get("state")
        if not isinstance(config_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("Kondo sparse actor checkpoint config/state must be objects")
        if set(state_payload) != _STATE_FIELDS:
            raise ValueError("Kondo sparse actor checkpoint state fields do not match v1")
        actor = cls.from_config(cast(Mapping[str, object], config_payload))
        parameter_payload = state_payload.get("parameters")
        if not isinstance(parameter_payload, Mapping) or (
            set(parameter_payload) != _PARAMETER_FIELDS
        ):
            raise ValueError("Kondo sparse actor checkpoint parameters are invalid")
        cfg = actor.config
        parameters = KondoActorParameters(
            hidden_weight=_decode_float32_array(
                parameter_payload["hidden_weight"],
                name="hidden_weight",
                shape=(cfg.feature_dim, cfg.hidden_dim),
            ),
            hidden_bias=_decode_float32_array(
                parameter_payload["hidden_bias"],
                name="hidden_bias",
                shape=(cfg.hidden_dim,),
            ),
            output_weight=_decode_float32_array(
                parameter_payload["output_weight"],
                name="output_weight",
                shape=(cfg.hidden_dim, cfg.action_count),
            ),
            output_bias=_decode_float32_array(
                parameter_payload["output_bias"],
                name="output_bias",
                shape=(cfg.action_count,),
            ),
        )
        gate_payload = state_payload.get("gate_checkpoint")
        if not isinstance(gate_payload, Mapping):
            raise ValueError("Kondo sparse actor gate checkpoint must be an object")
        config_gate_payload = config_payload.get("gate")
        if not isinstance(config_gate_payload, Mapping):
            raise ValueError("Kondo sparse actor config gate must be an object")
        gate_schema = gate_payload.get("schema")
        if gate_schema != config_gate_payload.get("schema"):
            raise ValueError(
                "Kondo sparse actor embedded gate schemas are inconsistent"
            )
        restored_gate, gate_state = KondoGate.from_checkpoint_payload(
            cast(Mapping[str, object], gate_payload)
        )
        if restored_gate.to_config() != actor._gate.to_config():
            raise ValueError("Kondo sparse actor gate checkpoint config is inconsistent")

        def counter(name: str) -> Array:
            value = state_payload.get(name)
            if type(value) is not int or not 0 <= value <= _INT32_MAX:
                raise ValueError(f"Kondo sparse actor {name} must be a nonnegative int32")
            return jnp.asarray(value, dtype=jnp.int32)

        state = KondoSparseActorState(
            parameters=parameters,
            gate_state=gate_state,
            policy_revision=counter("policy_revision"),
            actor_backward_count=counter("actor_backward_count"),
            sparse_backward_count=counter("sparse_backward_count"),
            full_fallback_count=counter("full_fallback_count"),
        )
        if not bool(np.asarray(actor._state_valid(state))):
            raise ValueError("Kondo sparse actor checkpoint state is invalid")
        if gate_schema not in (KONDO_GATE_SCHEMA, KONDO_GATE_LEGACY_V1_SCHEMA):
            raise ValueError("Kondo sparse actor embedded gate schema is invalid")
        if actor._checkpoint_payload_for_gate_schema(
            state,
            cast(str, gate_schema),
        ) != dict(payload):
            raise ValueError("Kondo sparse actor checkpoint is noncanonical")
        return actor, state


__all__ = [
    "KONDO_SPARSE_ACTOR_SCHEMA",
    "KondoActorBackwardBatch",
    "KondoActorBackwardResult",
    "KondoActorBackwardSummary",
    "KondoActorParameters",
    "KondoActorProtectedInputs",
    "KondoSparseActor",
    "KondoSparseActorBatch",
    "KondoSparseActorConfig",
    "KondoSparseActorResourceDeclaration",
    "KondoSparseActorResult",
    "KondoSparseActorState",
    "kondo_actor_backward_kernel",
    "kondo_sparse_actor_source_sha256",
]
