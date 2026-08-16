"""Development-only exact-dispatch adapter for the ASI transaction preview.

The adapter deliberately exposes only primitive-action ``PrototypeAgent``
configurations with no optional sidecars or Python-level curation.  It is an L0
interoperability slice, not ``reference-dev``, a whole-life runner, an exact
resume implementation, a safety boundary, or evidence of learning benefit.
Each adapter instance owns its emitted state through an opaque process-local
token; neither the adapter nor that state is a serialization or restore surface.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.oak import OaKState
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    PrototypeTransitionDiagnostics,
)
from alberta_framework.reference_agent import (
    MAX_DECISION_INDEX,
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAuthorization,
    DispatchStatus,
    SpaceSpec,
    Transaction,
)

PROTOTYPE_REFERENCE_ADAPTER_SCHEMA = "asi.prototype_exact_adapter.preview1"
PROTOTYPE_REFERENCE_IMPLEMENTATION_ID = "asi.prototype_exact_adapter.preview1"
PROTOTYPE_REFERENCE_STATE_SCHEMA = "asi.prototype_reference_state.preview1"
PROTOTYPE_OBSERVATION_SEMANTIC_ID = "asi.prototype_raw_observation.preview1"
PROTOTYPE_ACTION_SEMANTIC_ID = "asi.prototype_primitive_action.preview1"

_LIFECYCLE_CODEC = "prototype_uint32x2_hex.preview1"
_SCALAR_CODEC = "finite_float32_round_to_nearest.preview1"
_LIFECYCLE_PATTERN = re.compile(r"^prototype\.([0-9a-f]{8})([0-9a-f]{8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeReferenceState:
    """Process-local, adapter-owned envelope around functional Prototype state."""

    schema: str
    manifest_id: str
    config_sha256: str
    agent_state: PrototypeAgentState
    lifecycle_id: str
    decision_index: int
    current_observation_id: str | None
    _owner_token: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != PROTOTYPE_REFERENCE_STATE_SCHEMA:
            raise ValueError(
                f"schema must be {PROTOTYPE_REFERENCE_STATE_SCHEMA!r}, got {self.schema!r}"
            )
        if not isinstance(self.manifest_id, str) or _SHA256_PATTERN.fullmatch(
            self.manifest_id
        ) is None:
            raise ValueError("manifest_id must be a lowercase SHA-256 digest")
        if not isinstance(self.config_sha256, str) or _SHA256_PATTERN.fullmatch(
            self.config_sha256
        ) is None:
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.agent_state, PrototypeAgentState):
            raise ValueError("agent_state must be a PrototypeAgentState")
        if not isinstance(self.lifecycle_id, str) or _LIFECYCLE_PATTERN.fullmatch(
            self.lifecycle_id
        ) is None:
            raise ValueError("lifecycle_id must use the Prototype lifecycle codec")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
            or self.decision_index > MAX_DECISION_INDEX
        ):
            raise ValueError("decision_index must be a uint64 integer")
        if self.current_observation_id is not None and (
            not isinstance(self.current_observation_id, str)
            or not self.current_observation_id.strip()
        ):
            raise ValueError("current_observation_id must be None or a nonempty string")

    def __reduce__(self) -> Any:
        raise TypeError(
            "Prototype reference state is process-local and cannot be serialized; "
            "whole-life restore is not implemented"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeAdapterUpdate:
    """Staged functional result for a host runner to accept or reject."""

    state: PrototypeReferenceState
    next_decision: Decision | None
    accepted: bool
    parameters_changed: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, PrototypeReferenceState):
            raise ValueError("state must be a PrototypeReferenceState")
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be boolean")
        if not isinstance(self.parameters_changed, bool):
            raise ValueError("parameters_changed must be boolean")
        if self.accepted:
            if self.rejection_reason is not None:
                raise ValueError("an accepted update cannot carry a rejection reason")
        else:
            if self.parameters_changed:
                raise ValueError("a rejected update cannot change parameters")
            if self.next_decision is not None:
                raise ValueError("a rejected update cannot arm a next decision")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ValueError("a rejected update requires a nonempty reason")


def _lifecycle_words(lifecycle_id: str) -> Array:
    if not isinstance(lifecycle_id, str):
        raise ValueError("lifecycle_id must use the Prototype lifecycle codec")
    match = _LIFECYCLE_PATTERN.fullmatch(lifecycle_id)
    if match is None:
        raise ValueError(
            "lifecycle_id must use 'prototype.' followed by exactly sixteen "
            "lowercase hexadecimal digits"
        )
    return jnp.asarray(
        (int(match.group(1), 16), int(match.group(2), 16)),
        dtype=jnp.uint32,
    )


def _decision_index(words: Any) -> int:
    array = np.asarray(words)
    if array.shape != (4,) or array.dtype != np.dtype(np.uint32):
        raise DecisionOwnershipError(
            "Prototype decision identity must contain exactly four uint32 words"
        )
    return (int(array[2]) << 32) | int(array[3])


def _bool_scalar(value: Any, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.bool_):
        raise DecisionOwnershipError(f"{name} must be a scalar boolean")
    return bool(array)


def _float32_scalar(value: float, *, name: str) -> np.float32:
    try:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as a float32 scalar") from exc
    if converted.shape != () or not bool(np.isfinite(converted)):
        raise ValueError(f"{name} is not finitely representable as float32")
    return np.float32(converted)


def _tree_exactly_equal(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if cast(Any, left_structure) != right_structure or len(left_leaves) != len(
        right_leaves
    ):
        return False
    return all(
        np.asarray(left_leaf).dtype == np.asarray(right_leaf).dtype
        and np.asarray(left_leaf).shape == np.asarray(right_leaf).shape
        and np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _parameter_tree(state: PrototypeAgentState) -> tuple[Any, Any, Any]:
    oak_state = state.oak_state
    if not isinstance(oak_state, OaKState):
        raise DecisionOwnershipError(
            "the exact Prototype adapter requires an unwrapped primitive-only OaKState"
        )
    base = oak_state.stomp_state.base_learner_state
    return base.trunk_params, base.head_params, oak_state.stomp_state.base_average_reward


def _diagnostic_rejection_reason(diagnostics: PrototypeTransitionDiagnostics) -> str:
    names = (
        "started",
        "inputs_finite",
        "action_in_range",
        "observation_matches",
        "action_matches",
        "decision_id_matches",
        "discount_valid",
        "boundary_semantics_valid",
        "state_consistent",
        "post_update_finite",
        "post_update_consistent",
    )
    failed = [
        name
        for name in names
        if not _bool_scalar(getattr(diagnostics, name), name=f"diagnostics.{name}")
    ]
    detail = ", ".join(failed) if failed else "unknown invariant"
    return f"Prototype transition rejected: {detail}"


class PrototypeReferenceAdapter:
    """Primitive-only, exact-dispatch bridge into the host transaction preview."""

    __slots__ = ("_agent", "_config", "_manifest", "_state_owner_token")

    def __init__(self, config: PrototypeAgentConfig) -> None:
        self._validate_config(config)
        self._config = config
        self._agent = PrototypeAgent(config)
        self._state_owner_token = object()
        prototype_config = self._agent.to_config()
        observation_dim = config.oak.observation_dim
        action_count = config.oak.n_primitive_actions
        self._manifest = AgentManifest.from_config(
            schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
            implementation_id=PROTOTYPE_REFERENCE_IMPLEMENTATION_ID,
            state_schema=PROTOTYPE_REFERENCE_STATE_SCHEMA,
            config={
                "adapter_schema": PROTOTYPE_REFERENCE_ADAPTER_SCHEMA,
                "dispatch_mode": "exact_only",
                "environment_mode": "continuing_only",
                "lifecycle_codec": _LIFECYCLE_CODEC,
                "prototype_agent": prototype_config,
                "prototype_state_schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "reward_discount_codec": _SCALAR_CODEC,
            },
            observation_spec=SpaceSpec.box(
                shape=(observation_dim,),
                dtype="float32",
                low=None,
                high=None,
                semantic_id=PROTOTYPE_OBSERVATION_SEMANTIC_ID,
            ),
            action_spec=SpaceSpec.discrete(
                cardinality=action_count,
                dtype="int32",
                semantic_id=PROTOTYPE_ACTION_SEMANTIC_ID,
            ),
            capabilities=AgentCapabilities(dispatch_rebinding=False),
        )

    @classmethod
    def from_config(cls, config: PrototypeAgentConfig) -> PrototypeReferenceAdapter:
        """Build the adapter after rejecting every unsupported Prototype lane."""

        return cls(config)

    @staticmethod
    def _validate_config(config: PrototypeAgentConfig) -> None:
        if not isinstance(config, PrototypeAgentConfig):
            raise ValueError("config must be a PrototypeAgentConfig")
        if config.oak.n_options != 0 or config.oak.stomp.subtask_specs:
            raise ValueError("the adapter requires primitive-only OaK with options disabled")
        if config.oak.stomp.option_planning_backups_per_step != 0:
            raise ValueError("primitive-only option planning must be disabled")

        unsupported = {
            "option_search_control": config.option_search_control,
            "world_model": config.world_model,
            "world_model_ensemble": config.world_model_ensemble,
            "model_replay_rehearsal": config.model_replay_rehearsal,
            "recurrent_latent_world_model_ensemble": (
                config.recurrent_latent_world_model_ensemble
            ),
            "dreaming": config.dreaming,
            "horde_spec": config.horde_spec,
            "ia": config.ia,
            "partner_policy_fusion": config.partner_policy_fusion,
            "experiential_memory": config.experiential_memory,
            "gru_perception": config.gru_perception,
            "state_builder": config.state_builder,
            "representation_gradient_mixer": config.representation_gradient_mixer,
            "gradient_joy": config.gradient_joy,
            "prototype_feature_lifecycle": config.prototype_feature_lifecycle,
        }
        enabled = sorted(name for name, value in unsupported.items() if value is not None)
        if enabled:
            raise ValueError(
                "unsupported Prototype lanes must be disabled: " + ", ".join(enabled)
            )
        if config.n_dreams_per_step != 0:
            raise ValueError("dreaming must be disabled")
        if config.dream_next_observation_mode != "model_prediction":
            raise ValueError("nondefault dream observation handling is unsupported")
        if config.learn_state_builder_from_world_model:
            raise ValueError("state-builder learning must be disabled")
        if config.auto_curate_every != 0:
            raise ValueError("Python-level auto-curation must be disabled")

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    @property
    def config(self) -> PrototypeAgentConfig:
        return self._config

    def __reduce__(self) -> Any:
        raise TypeError(
            "Prototype reference adapter is process-local and cannot be serialized; "
            "whole-life restore is not implemented"
        )

    def _require_bound_state(self, state: PrototypeReferenceState) -> None:
        if not isinstance(state, PrototypeReferenceState):
            raise DecisionOwnershipError("state must be a PrototypeReferenceState")
        if state.schema != PROTOTYPE_REFERENCE_STATE_SCHEMA:
            raise DecisionOwnershipError("state schema does not match the adapter")
        if state.manifest_id != self.manifest.manifest_id:
            raise DecisionOwnershipError("state manifest identity does not match the adapter")
        if state.config_sha256 != self.manifest.config_sha256:
            raise DecisionOwnershipError("state config digest does not match the adapter")
        if state._owner_token is not self._state_owner_token:
            raise DecisionOwnershipError(
                "state owner is another PrototypeReferenceAdapter instance"
            )

    def init(self, key: Array, *, lifecycle_id: str) -> PrototypeReferenceState:
        """Initialize complete functional agent state for one host lifecycle."""

        agent_state = self._agent.init(
            key,
            lifecycle_id=_lifecycle_words(lifecycle_id),
        )
        return PrototypeReferenceState(
            schema=PROTOTYPE_REFERENCE_STATE_SCHEMA,
            manifest_id=self.manifest.manifest_id,
            config_sha256=self.manifest.config_sha256,
            agent_state=agent_state,
            lifecycle_id=lifecycle_id,
            decision_index=0,
            current_observation_id=None,
            _owner_token=self._state_owner_token,
        )

    def start(
        self,
        state: PrototypeReferenceState,
        *,
        observation_id: str,
        observation: Any,
    ) -> tuple[PrototypeReferenceState, Decision]:
        """Prime a fresh Prototype state and expose its first host decision."""

        self._require_bound_state(state)
        if state.current_observation_id is not None:
            raise DecisionOwnershipError("start requires a fresh reference state")
        encoded = self.manifest.observation_spec.encode(observation)
        started_agent_state = self._agent.start(
            state.agent_state,
            jnp.asarray(encoded.to_numpy(), dtype=jnp.float32),
        )
        started = dataclasses.replace(
            state,
            agent_state=started_agent_state,
            current_observation_id=observation_id,
        )
        decision = self.current_decision(started)
        return started, decision

    def current_decision(
        self,
        state: PrototypeReferenceState,
    ) -> Decision:
        """Translate and cross-check the currently armed Prototype decision."""

        self._require_bound_state(state)
        if state.current_observation_id is None:
            raise DecisionOwnershipError("fresh reference state has no current observation")
        expected_lifecycle_words = np.asarray(_lifecycle_words(state.lifecycle_id))
        try:
            prototype_decision = self._agent.decision(state.agent_state)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise DecisionOwnershipError(
                "Prototype state does not contain a valid current decision"
            ) from exc
        if not _bool_scalar(prototype_decision.armed, name="Prototype decision armed"):
            raise DecisionOwnershipError("Prototype decision is disarmed")
        internal_words = np.asarray(prototype_decision.decision_id)
        if (
            internal_words.shape != (4,)
            or internal_words.dtype != np.dtype(np.uint32)
            or not np.array_equal(internal_words[:2], expected_lifecycle_words)
        ):
            raise DecisionOwnershipError("Prototype decision belongs to another lifecycle")
        internal_index = _decision_index(internal_words)
        if internal_index != state.decision_index:
            raise DecisionOwnershipError(
                "Prototype generation/index mismatch: "
                f"expected {state.decision_index}, got {internal_index}"
            )
        return self.manifest.make_decision(
            lifecycle_id=state.lifecycle_id,
            decision_index=state.decision_index,
            observation_id=state.current_observation_id,
            observation=prototype_decision.observation,
            proposed_action=prototype_decision.action,
            armed=True,
        )

    def _require_current_host_decision(
        self,
        state: PrototypeReferenceState,
        decision: Decision,
    ) -> None:
        try:
            self.manifest.validate_decision(decision)
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError("decision does not match the adapter manifest") from exc
        expected = self.current_decision(state)
        if expected != decision:
            raise DecisionOwnershipError(
                "host decision does not match the Prototype observation/action cache"
            )

    def settle_dispatch(
        self,
        state: PrototypeReferenceState,
        authorization: DispatchAuthorization,
    ) -> tuple[PrototypeReferenceState, Literal[False]]:
        """Assert exact settlement without altering agent state or credit owners."""

        if not isinstance(authorization, DispatchAuthorization):
            raise DecisionOwnershipError("settlement requires a DispatchAuthorization")
        self._require_current_host_decision(state, authorization.decision)
        if authorization.status is AuthorizationStatus.VETOED:
            raise DecisionOwnershipError("a vetoed authorization cannot settle")
        if authorization.status is not AuthorizationStatus.EXACT:
            raise DecisionOwnershipError(
                "the exact-only adapter cannot settle an action replacement or rebind credit"
            )
        if authorization.authorized_action != authorization.decision.proposed_action:
            raise DecisionOwnershipError("exact authorization action does not match the proposal")
        return state, False

    @staticmethod
    def _rejected(
        state: PrototypeReferenceState,
        reason: str,
    ) -> PrototypeAdapterUpdate:
        return PrototypeAdapterUpdate(
            state=state,
            next_decision=None,
            accepted=False,
            parameters_changed=False,
            rejection_reason=reason,
        )

    def apply_outcome(
        self,
        state: PrototypeReferenceState,
        transaction: Transaction,
    ) -> PrototypeAdapterUpdate:
        """Stage one continuing update or return an exact state-preserving rejection."""

        if not isinstance(state, PrototypeReferenceState):
            raise DecisionOwnershipError("state must be a PrototypeReferenceState")
        try:
            self._require_bound_state(state)
        except DecisionOwnershipError as exc:
            return self._rejected(state, str(exc))
        if not isinstance(transaction, Transaction):
            return self._rejected(state, "outcome must be a Transaction")
        if transaction.is_boundary:
            return self._rejected(
                state,
                "execution boundary is unsupported by the continuing-only Prototype adapter",
            )
        if transaction.next_decision_observation is None:
            return self._rejected(state, "continuing outcome lacks a next decision observation")
        if transaction.next_decision_observation_id is None:
            return self._rejected(state, "continuing outcome lacks a next observation identity")

        try:
            self._require_current_host_decision(state, transaction.decision)
            if transaction.receipt.dispatch.status is not DispatchStatus.EXACT:
                raise DecisionOwnershipError("outcome uses a rebound dispatch")
            if (
                transaction.receipt.dispatch.authorization.status
                is not AuthorizationStatus.EXACT
            ):
                raise DecisionOwnershipError("outcome does not use an exact authorization")
            if transaction.receipt.effective_action != transaction.decision.proposed_action:
                raise DecisionOwnershipError("receipt action differs from the current proposal")
            self.manifest.action_spec.encode(transaction.receipt.effective_action)
            self.manifest.observation_spec.encode(transaction.bootstrap_observation)
            self.manifest.observation_spec.encode(
                transaction.next_decision_observation
            )

            prototype_decision = self._agent.decision(state.agent_state)
            observation = transaction.decision.observation.to_numpy()
            action = transaction.receipt.effective_action.to_numpy()
            bootstrap_observation = transaction.bootstrap_observation.to_numpy()
            next_observation = transaction.next_decision_observation.to_numpy()
            reward = _float32_scalar(transaction.reward, name="reward")
            discount = _float32_scalar(transaction.discount, name="discount")
            if (transaction.discount == 0.0) != (float(discount) == 0.0):
                raise ValueError("discount changes zero/nonzero semantics in float32")

            result = self._agent.update_transition(
                state.agent_state,
                PrototypeTransition(  # type: ignore[call-arg]
                    observation=observation,
                    action=action,
                    decision_id=prototype_decision.decision_id,
                    reward=reward,
                    discount=discount,
                    terminated=jnp.asarray(transaction.terminated, dtype=jnp.bool_),
                    truncated=jnp.asarray(transaction.truncated, dtype=jnp.bool_),
                    next_observation=bootstrap_observation,
                    next_decision_observation=next_observation,
                ),
            )
            if not _bool_scalar(
                result.transition_diagnostics.valid,
                name="transition diagnostics valid",
            ):
                return self._rejected(
                    state,
                    _diagnostic_rejection_reason(result.transition_diagnostics),
                )

            candidate_prototype_decision = self._agent.decision(result.state)
            candidate_armed = _bool_scalar(
                candidate_prototype_decision.armed,
                name="candidate Prototype decision armed",
            )
            if transaction.decision.decision_index == MAX_DECISION_INDEX:
                if candidate_armed:
                    return self._rejected(
                        state,
                        "Prototype remained armed after the final host decision index",
                    )
                next_decision = None
                candidate_state = dataclasses.replace(
                    state,
                    agent_state=result.state,
                    current_observation_id=transaction.next_decision_observation_id,
                )
            else:
                if not candidate_armed:
                    return self._rejected(
                        state,
                        "Prototype disarmed before host lifecycle capacity was exhausted",
                    )
                candidate_state = dataclasses.replace(
                    state,
                    agent_state=result.state,
                    decision_index=state.decision_index + 1,
                    current_observation_id=transaction.next_decision_observation_id,
                )
                next_decision = self.current_decision(candidate_state)
                if next_decision.observation != transaction.next_decision_observation:
                    return self._rejected(
                        state,
                        "Prototype next observation does not match the host outcome",
                    )

            parameters_changed = not _tree_exactly_equal(
                _parameter_tree(state.agent_state),
                _parameter_tree(result.state),
            )
            return PrototypeAdapterUpdate(
                state=candidate_state,
                next_decision=next_decision,
                accepted=True,
                parameters_changed=parameters_changed,
                rejection_reason=None,
            )
        except (
            DecisionOwnershipError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return self._rejected(state, str(exc))


__all__ = [
    "PROTOTYPE_ACTION_SEMANTIC_ID",
    "PROTOTYPE_OBSERVATION_SEMANTIC_ID",
    "PROTOTYPE_REFERENCE_ADAPTER_SCHEMA",
    "PROTOTYPE_REFERENCE_IMPLEMENTATION_ID",
    "PROTOTYPE_REFERENCE_STATE_SCHEMA",
    "PrototypeAdapterUpdate",
    "PrototypeReferenceAdapter",
    "PrototypeReferenceState",
]
