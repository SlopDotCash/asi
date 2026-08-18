# mypy: disable-error-code="attr-defined,call-arg"
"""Model-only dual-replay rehearsal composition.

This development-only mechanism composes :class:`DualReplayMemory` with a
bounded :class:`WorldModelEnsemble`.  One real event is handled causally:

1. the ensemble predicts and performs its real evidence-bearing update;
2. the returned pre-update learning signals are recorded with the transition;
3. the newly committed replay memory is sampled with a fixed stratified quota;
4. available samples update only ensemble member models through the dedicated
   replay lane, while padding positions execute explicit no-ops.

Replay never calls the learning-signal estimator.  It cannot advance real
event counts, residual/aleatoric calibration, change evidence, real bootstrap
keys, or real member-update counts.  This module has no actor, critic, state
builder, control update, effectiveness claim, or accepted scientific evidence.

Rehearsal on stored experience follows Lin (1992).  Replay is restricted to
the world model because one-step model targets
``(observation, action) -> (next_observation, reward, discount)`` are
policy-independent: stored transitions remain valid supervised examples no
matter how far the current policy has drifted.  Replaying into a critic or
actor would instead re-serve returns generated under stale behavior, so
control-side replay is deliberately out of scope here.

References:
    Lin (1992). "Self-Improving Reactive Agents Based on Reinforcement
        Learning, Planning and Teaching."
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.dual_replay import (
    DualReplayConfig,
    DualReplayMemory,
    DualReplayState,
    ReplayEntries,
    ReplayOutcome,
    ReplayPrediction,
    _allocation_sizes,
)
from alberta_framework.core.learning_signals import (
    LearningSignalAvailability,
    TypedLearningSignals,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleDiagnostics,
    WorldModelEnsemblePrediction,
    WorldModelEnsembleState,
    _ensemble_state_resource_counts,
)

MODEL_REPLAY_REHEARSAL_SCHEMA = "alberta.model_replay_rehearsal.v1"
MECHANISM_STATUS = "model-only-replay-mechanism-no-scientific-claim"
_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_MAX_EXACT_FLOAT32_INTEGER = 16_777_216
_ACTUAL_INT_TYPES: tuple[type, ...] = (
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
)


def _require_int(
    name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_float32_resource(
    name: str, *, vector_scalars: int, fixed_scalars: int = 0
) -> None:
    total_scalars = vector_scalars + fixed_scalars
    if total_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not issubclass(type(payload), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        values = dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} keys must be exact strings")
    return values


def _preflight_model_replay_state_resources(config: ModelReplayRehearsalConfig) -> None:
    """Reject an oversized combined state before either child allocates arrays."""
    _, ensemble_bytes = _ensemble_state_resource_counts(
        model=config.ensemble.model,
        ensemble_size=config.ensemble.ensemble_size,
    )
    _, replay_bytes = _allocation_sizes(config.replay)
    persistent_bytes = ensemble_bytes + replay_bytes + 7 * 4
    if persistent_bytes > _INT32_MAX:
        raise ValueError("model replay rehearsal state byte count must fit signed int32")
    # Source ensemble, proposed real-update ensemble, and committed replay.
    update_working_set_bytes = 2 * ensemble_bytes + replay_bytes
    if update_working_set_bytes > _INT32_MAX:
        raise ValueError(
            "model replay rehearsal update working set byte count must fit signed int32"
        )


ReplayActionEncoding = Literal["scalar_index", "one_hot"]


@dataclasses.dataclass(frozen=True)
class ModelReplayRehearsalConfig:
    """Static child mechanisms and exact action-storage conversion."""

    ensemble: WorldModelEnsembleConfig
    replay: DualReplayConfig
    action_encoding: ReplayActionEncoding = "one_hot"

    def __post_init__(self) -> None:
        if type(self.ensemble) is not WorldModelEnsembleConfig:
            raise TypeError("ensemble must be an exact WorldModelEnsembleConfig")
        if type(self.replay) is not DualReplayConfig:
            raise TypeError("replay must be an exact DualReplayConfig")
        if type(self.action_encoding) is not str:
            raise ValueError("action_encoding must be 'scalar_index' or 'one_hot'")
        if self.ensemble.model.observation_dim != self.replay.observation_dim:
            raise ValueError("ensemble and replay observation dimensions must match")
        if self.ensemble.model.n_actions > _MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError("n_actions exceeds exact float32 action-index storage")
        _require_int(
            "ensemble.model.n_actions",
            self.ensemble.model.n_actions,
            minimum=1,
            maximum=_INT32_MAX,
        )
        _require_int(
            "replay.action_dim",
            self.replay.action_dim,
            minimum=1,
            maximum=_INT32_MAX,
        )
        _require_int(
            "replay.batch_size",
            self.replay.batch_size,
            minimum=1,
            maximum=_INT32_MAX,
        )
        if self.action_encoding == "scalar_index":
            if self.replay.action_dim != 1:
                raise ValueError("scalar_index requires replay.action_dim == 1")
        elif self.action_encoding == "one_hot":
            if self.replay.action_dim != self.ensemble.model.n_actions:
                raise ValueError("one_hot requires replay.action_dim == model.n_actions")
        else:
            raise ValueError("action_encoding must be 'scalar_index' or 'one_hot'")
        if self.ensemble.ensemble_size * (self.replay_quota + 1) > _INT32_MAX:
            raise ValueError("derived per-event model update candidates must fit signed int32")

    @property
    def replay_quota(self) -> int:
        """Fixed per-real-event quota, including explicit padding positions."""
        return self.replay.batch_size

    def to_config(self) -> dict[str, Any]:
        """Return a strict development-only configuration payload."""
        return {
            "schema": MODEL_REPLAY_REHEARSAL_SCHEMA,
            "type": "ModelReplayRehearsalConfig",
            "mechanism_status": MECHANISM_STATUS,
            "ensemble": self.ensemble.to_config(),
            "replay": self.replay.to_config(),
            "action_encoding": self.action_encoding,
            "accepted_scientific_evidence": False,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> ModelReplayRehearsalConfig:
        """Reconstruct an exact v1 configuration."""
        values = _copy_mapping(payload, name="model replay config")
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "ensemble",
            "replay",
            "action_encoding",
            "accepted_scientific_evidence",
        }
        if set(values) != expected:
            raise ValueError("model replay config fields do not match the v1 schema")
        if (
            type(values.get("schema")) is not str
            or values["schema"] != MODEL_REPLAY_REHEARSAL_SCHEMA
        ):
            raise ValueError("unexpected model replay rehearsal schema")
        if type(values.get("type")) is not str or values["type"] != "ModelReplayRehearsalConfig":
            raise ValueError("unexpected model replay rehearsal config type")
        if (
            type(values.get("mechanism_status")) is not str
            or values["mechanism_status"] != MECHANISM_STATUS
        ):
            raise ValueError("model replay rehearsal must remain mechanism-only")
        if values.get("accepted_scientific_evidence") is not False:
            raise ValueError("model replay rehearsal has no accepted scientific evidence")
        ensemble_payload = values.get("ensemble")
        replay_payload = values.get("replay")
        ensemble_values = _copy_mapping(ensemble_payload, name="ensemble config")
        replay_values = _copy_mapping(replay_payload, name="replay config")
        return cls(
            ensemble=WorldModelEnsembleConfig.from_config(ensemble_values),
            replay=DualReplayConfig.from_config(replay_values),
            action_encoding=cast(ReplayActionEncoding, values.get("action_encoding")),
        )


@chex.dataclass(frozen=True)
class RealModelReplayEvent:
    """One authoritative real transition for model learning and replay.

    ``next_observation`` is the final/bootstrap transition observation.  A
    post-reset decision observation is intentionally absent: this composer does
    not choose the next action, and storing a reset observation in its place
    would corrupt a censored time-limit model target.
    """

    observation: Array
    action: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    next_observation: Array
    representation_version: Array
    provenance_id: Array
    source_id: Array
    safety_cost: Array
    safety_cost_available: Array
    valid: Array


@chex.dataclass(frozen=True)
class ModelReplayRehearsalState:
    """Complete atomic composer state and bounded lifetime counters."""

    ensemble_state: WorldModelEnsembleState
    replay_state: DualReplayState
    real_attempt_count: Array
    accepted_real_event_count: Array
    rejected_real_event_count: Array
    rehearsal_attempt_count: Array
    rehearsal_applied_count: Array
    rehearsal_padding_count: Array
    persistent_bytes: Array


@chex.dataclass(frozen=True)
class ReplayActionConversion:
    """Exact stored-action to ensemble-index conversion."""

    action: Array
    valid: Array


@chex.dataclass(frozen=True)
class ModelReplayRehearsalTrace:
    """Fixed-quota per-position rehearsal audit."""

    sample_valid: Array
    padding: Array
    action_conversion_valid: Array
    actions: Array
    provenance_ids: Array
    representation_versions: Array
    observed_losses: Array
    member_updates_applied: Array
    model_updates_applied: Array
    fresh_evidence_observed: Array


@chex.dataclass(frozen=True)
class ModelReplayRehearsalDiagnostics:
    """Atomic transaction, filtering, and evidence-isolation diagnostics.

    Sub-operation fields describe candidate work.  Only
    ``transaction_applied`` states that the composed child states committed.
    """

    state_valid: Array
    event_valid: Array
    counter_available: Array
    real_update_applied: Array
    replay_recorded: Array
    replay_sampled: Array
    action_conversions_valid: Array
    rehearsal_updates_valid: Array
    calibration_unchanged: Array
    candidate_state_valid: Array
    stale_short_term_count: Array
    stale_long_term_count: Array
    future_short_term_count: Array
    future_long_term_count: Array
    transaction_applied: Array
    rejected: Array


@chex.dataclass(frozen=True)
class ModelReplayRehearsalResult:
    """One real update, record, fixed-quota rehearsal transaction.

    ``real_update_diagnostics`` audits the causal ensemble sub-operation and
    may therefore report ``applied=True`` even when a later record/rehearsal
    check rolls the whole composition back.  ``diagnostics.transaction_applied``
    and the commit-gated real surfaces are the authoritative commit verdict.
    """

    state: ModelReplayRehearsalState
    real_prediction: WorldModelEnsemblePrediction
    real_signals: TypedLearningSignals
    real_signals_committed: Array
    real_observed_loss: Array
    real_representation_gradient: Array
    real_representation_gradient_valid: Array
    real_update_diagnostics: WorldModelEnsembleDiagnostics
    trace: ModelReplayRehearsalTrace
    diagnostics: ModelReplayRehearsalDiagnostics


@dataclasses.dataclass(frozen=True)
class ModelReplayRehearsalResourceBudget:
    """Exact persistent allocation and statically bounded per-event work.

    Transient returned diagnostics, compiler/autodiff storage, and device
    alignment are intentionally outside the persistent checkpoint allocation.
    """

    persistent_state_scalars: int
    persistent_state_bytes: int
    ensemble_state_bytes: int
    replay_state_bytes: int
    composer_accounting_bytes: int
    replay_total_capacity: int
    short_term_capacity: int
    long_term_capacity: int
    fixed_replay_quota: int
    max_real_model_update_candidates_per_event: int
    max_replay_model_update_candidates_per_event: int
    max_total_model_update_candidates_per_event: int
    max_actor_updates_per_event: int
    max_critic_updates_per_event: int
    max_state_builder_updates_per_event: int
    max_real_event_count: int
    max_rehearsal_attempt_count: int

    def __post_init__(self) -> None:
        for name in (
            "persistent_state_scalars",
            "persistent_state_bytes",
            "ensemble_state_bytes",
            "replay_state_bytes",
            "composer_accounting_bytes",
            "replay_total_capacity",
            "short_term_capacity",
            "long_term_capacity",
            "fixed_replay_quota",
            "max_real_model_update_candidates_per_event",
            "max_replay_model_update_candidates_per_event",
            "max_total_model_update_candidates_per_event",
            "max_actor_updates_per_event",
            "max_critic_updates_per_event",
            "max_state_builder_updates_per_event",
            "max_real_event_count",
            "max_rehearsal_attempt_count",
        ):
            object.__setattr__(
                self,
                name,
                _require_int(name, getattr(self, name), minimum=0),
            )
        if self.persistent_state_scalars <= 0:
            raise ValueError("persistent_state_scalars must be positive")
        if self.ensemble_state_bytes <= 0:
            raise ValueError("ensemble_state_bytes must be positive")
        if self.replay_state_bytes <= 0:
            raise ValueError("replay_state_bytes must be positive")
        if self.fixed_replay_quota < 1:
            raise ValueError("fixed_replay_quota must be positive")
        if self.short_term_capacity < 1 or self.long_term_capacity < 1:
            raise ValueError("replay tier capacities must be positive")
        if self.fixed_replay_quota > self.replay_total_capacity:
            raise ValueError("fixed_replay_quota cannot exceed replay_total_capacity")
        if self.max_real_model_update_candidates_per_event < 2:
            raise ValueError("max_real_model_update_candidates_per_event must be at least two")
        expected = {
            "composer_accounting_bytes": 28,
            "persistent_state_bytes": (
                self.ensemble_state_bytes + self.replay_state_bytes + 28
            ),
            "replay_total_capacity": self.short_term_capacity + self.long_term_capacity,
            "max_replay_model_update_candidates_per_event": (
                self.fixed_replay_quota
                * self.max_real_model_update_candidates_per_event
            ),
            "max_total_model_update_candidates_per_event": (
                (self.fixed_replay_quota + 1)
                * self.max_real_model_update_candidates_per_event
            ),
            "max_actor_updates_per_event": 0,
            "max_critic_updates_per_event": 0,
            "max_state_builder_updates_per_event": 0,
            "max_real_event_count": _INT32_MAX // self.fixed_replay_quota,
            "max_rehearsal_attempt_count": (
                (_INT32_MAX // self.fixed_replay_quota) * self.fixed_replay_quota
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the model-replay implementation")

    def to_config(self) -> dict[str, int]:
        """Return JSON-compatible exact accounting."""
        return dataclasses.asdict(self)


def _strict_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    array = jnp.asarray(value)
    if array.shape != shape or array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have shape {shape} and dtype {jnp.dtype(dtype)}")
    return array


def _materialize_key(value: Any) -> Any:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        return jr.key_data(value)
    return value


def _logical_tree_size(tree: object) -> tuple[int, int]:
    arrays = [jnp.asarray(_materialize_key(leaf)) for leaf in jax.tree.leaves(tree)]
    return (
        sum(int(array.size) for array in arrays),
        sum(int(array.nbytes) for array in arrays),
    )


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    same_structure = str(left_structure) == str(right_structure)
    if not same_structure or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False)
    equal = jnp.asarray(True)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = _materialize_key(left_leaf)
        right_array = _materialize_key(right_leaf)
        equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _gate_signals(signals: TypedLearningSignals, available: Array) -> TypedLearningSignals:
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    def gate(value: Array, flag: Array) -> Array:
        return jnp.where(available & flag, value, zero)

    source = signals.availability
    return TypedLearningSignals(
        epistemic_disagreement=gate(signals.epistemic_disagreement, source.epistemic),
        epistemic_surprise=gate(signals.epistemic_surprise, source.epistemic),
        aleatoric_uncertainty=gate(signals.aleatoric_uncertainty, source.aleatoric),
        normalized_residual=gate(signals.normalized_residual, source.normalized_residual),
        learning_progress=gate(signals.learning_progress, source.learning_progress),
        calibrated_residual_z=gate(signals.calibrated_residual_z, source.change_probability),
        instantaneous_change_probability=gate(
            signals.instantaneous_change_probability, source.change_probability
        ),
        change_probability=gate(signals.change_probability, source.change_probability),
        availability=LearningSignalAvailability(
            input_valid=available & source.input_valid,
            epistemic=available & source.epistemic,
            aleatoric=available & source.aleatoric,
            normalized_residual=available & source.normalized_residual,
            learning_progress=available & source.learning_progress,
            change_probability=available & source.change_probability,
        ),
    )


def _config_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _saturating_increment(value: Array) -> Array:
    return jnp.minimum(value, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + jnp.asarray(
        1, dtype=jnp.int32
    )


class ModelReplayRehearsal:
    """Causal, atomic composition of real model learning and model-only replay."""

    def __init__(self, config: ModelReplayRehearsalConfig):
        if type(config) is not ModelReplayRehearsalConfig:
            raise ValueError("config must be an exact ModelReplayRehearsalConfig")
        _preflight_model_replay_state_resources(config)
        self._config = config
        self._ensemble = WorldModelEnsemble(config.ensemble)
        self._replay = DualReplayMemory(config.replay)
        template = self._make_initial_state(jr.key(0), persistent_bytes=0)
        persistent_scalars, persistent_bytes = _logical_tree_size(template)
        _require_float32_resource(
            "ModelReplayRehearsal state",
            vector_scalars=persistent_scalars,
        )
        if persistent_bytes > _INT32_MAX:
            raise ValueError("ModelReplayRehearsal state byte count must fit signed int32")
        if persistent_bytes > _UINT32_MAX:
            raise ValueError("model replay rehearsal state exceeds uint32 byte accounting")
        self._persistent_bytes = persistent_bytes
        self._persistent_scalars = persistent_scalars
        child_bytes = (
            self._ensemble.resource_budget(template.ensemble_state).persistent_state_bytes
            + self._replay.persistent_bytes
        )
        if persistent_bytes - child_bytes != 7 * jnp.dtype(jnp.int32).itemsize:
            raise RuntimeError("composer resource accounting disagrees with allocated state")

    @property
    def config(self) -> ModelReplayRehearsalConfig:
        """Return the immutable composition configuration."""
        return self._config

    @property
    def ensemble(self) -> WorldModelEnsemble:
        """Return the bounded child ensemble."""
        return self._ensemble

    @property
    def replay(self) -> DualReplayMemory:
        """Return the fixed-capacity child replay memory."""
        return self._replay

    def to_config(self) -> dict[str, Any]:
        """Return the strict mechanism-only construction payload."""
        return {"type": "ModelReplayRehearsal", "config": self._config.to_config()}

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> ModelReplayRehearsal:
        """Reconstruct from an exact :meth:`to_config` payload."""
        values = _copy_mapping(payload, name="model replay rehearsal construction")
        if set(values) != {"type", "config"}:
            raise ValueError("model replay rehearsal construction fields are invalid")
        if type(values.get("type")) is not str or values["type"] != "ModelReplayRehearsal":
            raise ValueError("unexpected model replay rehearsal construction type")
        config = values.get("config")
        config_values = _copy_mapping(config, name="model replay rehearsal config")
        return cls(ModelReplayRehearsalConfig.from_config(config_values))

    def _make_initial_state(
        self,
        key: Array,
        *,
        persistent_bytes: int,
    ) -> ModelReplayRehearsalState:
        ensemble_key, replay_key = jr.split(key)
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ModelReplayRehearsalState(
            ensemble_state=self._ensemble.init(ensemble_key),
            replay_state=self._replay.init(replay_key),
            real_attempt_count=zero,
            accepted_real_event_count=zero,
            rejected_real_event_count=zero,
            rehearsal_attempt_count=zero,
            rehearsal_applied_count=zero,
            rehearsal_padding_count=zero,
            persistent_bytes=jnp.asarray(persistent_bytes, dtype=jnp.uint32),
        )

    def init(self, key: Array) -> ModelReplayRehearsalState:
        """Initialize both isolated child key streams and exact accounting."""
        state = self._make_initial_state(key, persistent_bytes=self._persistent_bytes)
        scalars, persistent_bytes = _logical_tree_size(state)
        if scalars != self._persistent_scalars or persistent_bytes != self._persistent_bytes:
            raise RuntimeError("model replay rehearsal allocation changed during initialization")
        return state

    def _validate_state_static_contract(self, state: ModelReplayRehearsalState) -> None:
        if not isinstance(state, ModelReplayRehearsalState):
            raise TypeError("state must be a ModelReplayRehearsalState")
        self._ensemble.state_valid(state.ensemble_state)
        self._replay.state_valid(state.replay_state)
        for name in (
            "real_attempt_count",
            "accepted_real_event_count",
            "rejected_real_event_count",
            "rehearsal_attempt_count",
            "rehearsal_applied_count",
            "rehearsal_padding_count",
        ):
            _strict_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        _strict_array(
            state.persistent_bytes,
            name="state.persistent_bytes",
            shape=(),
            dtype=jnp.uint32,
        )

    def _validate_event_static_contract(self, event: RealModelReplayEvent) -> None:
        if not isinstance(event, RealModelReplayEvent):
            raise TypeError("event must be a RealModelReplayEvent")
        observation_dim = self._config.ensemble.model.observation_dim
        _strict_array(
            event.observation,
            name="event.observation",
            shape=(observation_dim,),
            dtype=jnp.float32,
        )
        _strict_array(
            event.next_observation,
            name="event.next_observation",
            shape=(observation_dim,),
            dtype=jnp.float32,
        )
        for name in ("reward", "discount", "safety_cost"):
            _strict_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(),
                dtype=jnp.float32,
            )
        for name in ("action", "representation_version", "provenance_id", "source_id"):
            _strict_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        for name in ("terminated", "truncated", "safety_cost_available", "valid"):
            _strict_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(),
                dtype=jnp.bool_,
            )

    def _event_valid(self, event: RealModelReplayEvent) -> Array:
        magnitude_bound = self._config.ensemble.signal_estimator.max_input_magnitude
        return (
            event.valid
            & jnp.all(jnp.isfinite(event.observation))
            & jnp.all(jnp.abs(event.observation) <= magnitude_bound)
            & jnp.all(jnp.isfinite(event.next_observation))
            & jnp.all(jnp.abs(event.next_observation) <= magnitude_bound)
            & jnp.isfinite(event.reward)
            & (jnp.abs(event.reward) <= magnitude_bound)
            & jnp.isfinite(event.discount)
            & (event.discount >= 0.0)
            & (event.discount <= 1.0)
            & ((event.discount == 0.0) == event.terminated)
            & (event.action >= 0)
            & (event.action < self._config.ensemble.model.n_actions)
            & (event.representation_version >= 0)
            & (event.provenance_id >= 0)
            & (event.source_id >= 0)
            & jnp.isfinite(event.safety_cost)
            & (event.safety_cost >= 0.0)
            & (event.safety_cost_available | (event.safety_cost == 0.0))
        )

    def _stored_actions_valid(self, entries: ReplayEntries) -> Array:
        actions = entries.actions
        n_actions = self._config.ensemble.model.n_actions
        if self._config.action_encoding == "scalar_index":
            values = actions[:, 0]
            finite = jnp.isfinite(values)
            safe_values = jnp.where(finite, values, 0.0)
            indices = safe_values.astype(jnp.int32)
            encoded = (
                finite
                & (safe_values == indices.astype(jnp.float32))
                & (indices >= 0)
                & (indices < n_actions)
            )
        else:
            encoded = (
                jnp.all(jnp.isfinite(actions), axis=1)
                & jnp.all((actions == 0.0) | (actions == 1.0), axis=1)
                & (jnp.sum(actions, axis=1) == 1.0)
            )
        return jnp.all((~entries.valid) | encoded)

    def _state_is_valid(self, state: ModelReplayRehearsalState) -> Array:
        counters = jnp.stack(
            (
                state.real_attempt_count,
                state.accepted_real_event_count,
                state.rejected_real_event_count,
                state.rehearsal_attempt_count,
                state.rehearsal_applied_count,
                state.rehearsal_padding_count,
            )
        )
        quota = jnp.asarray(self._config.replay_quota, dtype=jnp.int32)
        accepted = state.accepted_real_event_count
        memory = state.replay_state
        ensemble = state.ensemble_state
        return (
            jnp.all(counters >= 0)
            & jnp.all(counters <= _INT32_MAX)
            & (
                accepted
                <= jnp.asarray(_INT32_MAX, jnp.int32)
                - state.rejected_real_event_count
            )
            & (state.real_attempt_count == accepted + state.rejected_real_event_count)
            & (accepted <= jnp.asarray(_INT32_MAX // self._config.replay_quota, jnp.int32))
            & (state.rehearsal_attempt_count == accepted * quota)
            & (
                state.rehearsal_attempt_count
                == state.rehearsal_applied_count + state.rehearsal_padding_count
            )
            & (ensemble.event_count == accepted)
            & (ensemble.replay_event_count == state.rehearsal_applied_count)
            & (memory.write_attempt_count == accepted)
            & (memory.accepted_transition_count == accepted)
            & (memory.rejected_transition_count == 0)
            & (memory.sample_count == accepted)
            & self._stored_actions_valid(memory.short_term)
            & self._stored_actions_valid(memory.long_term)
            & self._ensemble.state_valid(ensemble)
            & self._replay.state_valid(memory)
            & (state.persistent_bytes == jnp.asarray(self._persistent_bytes, jnp.uint32))
        )

    def state_valid(self, state: ModelReplayRehearsalState) -> Array:
        """Return the complete dynamic composition verdict."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def validate_state(self, state: ModelReplayRehearsalState) -> None:
        """Raise when a restored or externally supplied state is inconsistent."""
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("model replay rehearsal state violates dynamic invariants")

    def encode_action(self, action: Array) -> Array:
        """Encode one exact int32 action for replay storage."""
        _strict_array(action, name="action", shape=(), dtype=jnp.int32)
        if self._config.action_encoding == "scalar_index":
            return jnp.reshape(action.astype(jnp.float32), (1,))
        return jax.nn.one_hot(
            action,
            self._config.ensemble.model.n_actions,
            dtype=jnp.float32,
        )

    def decode_action(self, stored_action: Array) -> ReplayActionConversion:
        """Decode only an exact scalar index or exact one-hot replay action."""
        _strict_array(
            stored_action,
            name="stored_action",
            shape=(self._config.replay.action_dim,),
            dtype=jnp.float32,
        )
        n_actions = self._config.ensemble.model.n_actions
        if self._config.action_encoding == "scalar_index":
            value = stored_action[0]
            finite = jnp.isfinite(value)
            safe_value = jnp.where(finite, value, 0.0)
            action = safe_value.astype(jnp.int32)
            valid = (
                finite
                & (safe_value == action.astype(jnp.float32))
                & (action >= 0)
                & (action < n_actions)
            )
            return ReplayActionConversion(
                action=jnp.where(valid, action, 0).astype(jnp.int32),
                valid=valid,
            )
        finite = jnp.all(jnp.isfinite(stored_action))
        binary = jnp.all((stored_action == 0.0) | (stored_action == 1.0))
        exactly_one = jnp.sum(stored_action) == 1.0
        action = jnp.argmax(stored_action).astype(jnp.int32)
        valid = finite & binary & exactly_one & (action >= 0) & (action < n_actions)
        return ReplayActionConversion(
            action=jnp.where(valid, action, 0).astype(jnp.int32),
            valid=valid,
        )

    def _counter_available(self, state: ModelReplayRehearsalState) -> Array:
        quota = self._config.replay_quota
        max_real = _INT32_MAX // quota
        return (
            (state.real_attempt_count < _INT32_MAX)
            & (state.accepted_real_event_count < max_real)
            & (state.rejected_real_event_count < _INT32_MAX)
            & (state.rehearsal_attempt_count <= _INT32_MAX - quota)
            & (state.rehearsal_applied_count <= _INT32_MAX - quota)
            & (state.rehearsal_padding_count <= _INT32_MAX - quota)
        )

    @staticmethod
    def _calibration_unchanged(
        real_state: WorldModelEnsembleState,
        replay_state: WorldModelEnsembleState,
    ) -> Array:
        return (
            _tree_equal(real_state.signal_state, replay_state.signal_state)
            & jnp.array_equal(real_state.residual_variances, replay_state.residual_variances)
            & jnp.array_equal(
                jr.key_data(real_state.bootstrap_key), jr.key_data(replay_state.bootstrap_key)
            )
            & jnp.array_equal(
                real_state.last_bootstrap_mask, replay_state.last_bootstrap_mask
            )
            & jnp.array_equal(
                real_state.member_update_counts, replay_state.member_update_counts
            )
            & (real_state.event_count == replay_state.event_count)
        )

    def _rehearse_batch(
        self,
        state: WorldModelEnsembleState,
        entries: ReplayEntries,
        sample_valid: Array,
    ) -> tuple[WorldModelEnsembleState, Array, Array, Array, Array, Array]:
        def body(
            ensemble_state: WorldModelEnsembleState,
            position: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[
            WorldModelEnsembleState,
            tuple[Array, Array, Array, Array, Array],
        ]:
            observation, stored_action, reward, discount, next_observation, available = position
            conversion = self.decode_action(stored_action)
            update = self._ensemble.replay_update(
                ensemble_state,
                observation,
                conversion.action,
                reward,
                discount,
                next_observation,
                available & conversion.valid,
            )
            conversion_valid = (~available) | conversion.valid
            return update.state, (
                conversion.action,
                conversion_valid,
                update.observed_loss,
                update.member_updates_applied,
                update.diagnostics.applied,
            )

        final_state, outputs = jax.lax.scan(
            body,
            state,
            (
                entries.observations,
                entries.actions,
                entries.rewards,
                entries.discounts,
                entries.next_observations,
                sample_valid,
            ),
        )
        actions, conversions_valid, losses, member_updates, model_updates = outputs
        return (
            final_state,
            actions,
            conversions_valid,
            losses,
            member_updates,
            model_updates,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: ModelReplayRehearsalState,
        event: RealModelReplayEvent,
    ) -> ModelReplayRehearsalResult:
        state_valid = self._state_is_valid(state)
        event_valid = self._event_valid(event)
        counter_available = self._counter_available(state)

        # This returns a pre-update prediction and its real evidence-bearing
        # signals before any outcome can cross the replay record boundary.
        real_update = self._ensemble.update(
            state.ensemble_state,
            event.observation,
            event.action,
            event.reward,
            event.discount,
            event.next_observation,
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        prediction = ReplayPrediction(
            observation=event.observation,
            action=self.encode_action(event.action),
            old_behavior_probability=zero,
            old_behavior_probability_available=false,
            old_behavior_logit=zero,
            old_behavior_logit_available=false,
            old_value_target=zero,
            old_value_target_available=false,
            epistemic_surprise=real_update.signals.epistemic_surprise,
            epistemic_surprise_available=real_update.signals.availability.epistemic,
            aleatoric_uncertainty=real_update.signals.aleatoric_uncertainty,
            aleatoric_uncertainty_available=real_update.signals.availability.aleatoric,
            representation_version=event.representation_version,
            provenance_id=event.provenance_id,
            source_id=event.source_id,
            valid=event_valid & real_update.diagnostics.applied,
        )
        outcome = ReplayOutcome(
            next_observation=event.next_observation,
            reward=event.reward,
            discount=event.discount,
            terminated=event.terminated,
            truncated=event.truncated,
            learning_progress=real_update.signals.learning_progress,
            learning_progress_available=real_update.signals.availability.learning_progress,
            safety_cost=event.safety_cost,
            safety_cost_available=event.safety_cost_available,
            valid=event_valid & real_update.diagnostics.applied,
        )
        write = self._replay.record(state.replay_state, prediction, outcome)
        recorded = (
            write.state_valid
            & write.input_valid
            & write.counter_available
            & write.wrote_short_term
            & (
                write.state.accepted_transition_count
                == state.replay_state.accepted_transition_count + 1
            )
        )
        sample = self._replay.sample(write.state, event.representation_version)
        sampled = (
            sample.state_valid
            & sample.representation_version_valid
            & sample.counter_available
            & (sample.state.sample_count == write.state.sample_count + 1)
        )
        (
            rehearsed_ensemble_state,
            actions,
            action_conversion_valid,
            observed_losses,
            member_updates_applied,
            model_updates_applied,
        ) = self._rehearse_batch(
            real_update.state,
            sample.batch.entries,
            sample.batch.valid,
        )
        action_conversions_valid = jnp.all(action_conversion_valid)
        rehearsal_updates_valid = jnp.all(
            (~sample.batch.valid) | model_updates_applied
        ) & jnp.all((~model_updates_applied) | sample.batch.valid)
        calibration_unchanged = self._calibration_unchanged(
            real_update.state,
            rehearsed_ensemble_state,
        )
        applied_count = jnp.sum(model_updates_applied.astype(jnp.int32))
        padding_count = jnp.asarray(self._config.replay_quota, jnp.int32) - applied_count
        accepted_candidate = ModelReplayRehearsalState(
            ensemble_state=rehearsed_ensemble_state,
            replay_state=sample.state,
            real_attempt_count=_saturating_increment(state.real_attempt_count),
            accepted_real_event_count=_saturating_increment(
                state.accepted_real_event_count
            ),
            rejected_real_event_count=state.rejected_real_event_count,
            rehearsal_attempt_count=(
                jnp.minimum(
                    state.rehearsal_attempt_count,
                    jnp.asarray(_INT32_MAX - self._config.replay_quota, jnp.int32),
                )
                + jnp.asarray(self._config.replay_quota, jnp.int32)
            ),
            rehearsal_applied_count=state.rehearsal_applied_count + applied_count,
            rehearsal_padding_count=state.rehearsal_padding_count + padding_count,
            persistent_bytes=state.persistent_bytes,
        )
        candidate_state_valid = self._state_is_valid(accepted_candidate)
        transaction_applied = (
            state_valid
            & event_valid
            & counter_available
            & real_update.diagnostics.applied
            & recorded
            & sampled
            & action_conversions_valid
            & rehearsal_updates_valid
            & calibration_unchanged
            & candidate_state_valid
        )
        rejected_candidate = state.replace(
            real_attempt_count=_saturating_increment(state.real_attempt_count),
            rejected_real_event_count=_saturating_increment(
                state.rejected_real_event_count
            ),
        )
        next_state = cast(
            ModelReplayRehearsalState,
            jax.lax.cond(
                transaction_applied,
                lambda: accepted_candidate,
                lambda: jax.lax.cond(
                    state_valid & counter_available,
                    lambda: rejected_candidate,
                    lambda: state,
                ),
            ),
        )
        trace = ModelReplayRehearsalTrace(
            sample_valid=sample.batch.valid,
            padding=~sample.batch.valid,
            action_conversion_valid=action_conversion_valid,
            actions=actions,
            provenance_ids=sample.batch.entries.provenance_ids,
            representation_versions=sample.batch.entries.representation_versions,
            observed_losses=observed_losses,
            member_updates_applied=member_updates_applied,
            model_updates_applied=model_updates_applied,
            fresh_evidence_observed=jnp.zeros_like(sample.batch.valid),
        )
        diagnostics = ModelReplayRehearsalDiagnostics(
            state_valid=state_valid,
            event_valid=event_valid,
            counter_available=counter_available,
            real_update_applied=real_update.diagnostics.applied,
            replay_recorded=recorded,
            replay_sampled=sampled,
            action_conversions_valid=action_conversions_valid,
            rehearsal_updates_valid=rehearsal_updates_valid,
            calibration_unchanged=calibration_unchanged,
            candidate_state_valid=candidate_state_valid,
            stale_short_term_count=sample.stale_short_term_count,
            stale_long_term_count=sample.stale_long_term_count,
            future_short_term_count=sample.future_short_term_count,
            future_long_term_count=sample.future_long_term_count,
            transaction_applied=transaction_applied,
            rejected=~transaction_applied,
        )
        return ModelReplayRehearsalResult(
            state=next_state,
            real_prediction=real_update.prediction,
            real_signals=_gate_signals(real_update.signals, transaction_applied),
            real_signals_committed=transaction_applied,
            real_observed_loss=jnp.where(
                transaction_applied,
                real_update.observed_loss,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            real_representation_gradient=jnp.where(
                transaction_applied & real_update.representation_gradient_valid,
                real_update.representation_gradient,
                jnp.zeros_like(real_update.representation_gradient),
            ),
            real_representation_gradient_valid=(
                transaction_applied & real_update.representation_gradient_valid
            ),
            real_update_diagnostics=real_update.diagnostics,
            trace=trace,
            diagnostics=diagnostics,
        )

    def step(
        self,
        state: ModelReplayRehearsalState,
        event: RealModelReplayEvent,
    ) -> ModelReplayRehearsalResult:
        """Process one all-or-nothing real-update/record/rehearsal event."""
        self._validate_state_static_contract(state)
        self._validate_event_static_contract(event)
        return cast(ModelReplayRehearsalResult, self._step_jit(state, event))

    def resource_budget(
        self,
        state: ModelReplayRehearsalState | None = None,
    ) -> ModelReplayRehearsalResourceBudget:
        """Return exact persistent allocation and bounded update candidates."""
        measured = self.init(jr.key(0)) if state is None else state
        self._validate_state_static_contract(measured)
        scalars, persistent_bytes = _logical_tree_size(measured)
        ensemble_bytes = self._ensemble.resource_budget(
            measured.ensemble_state
        ).persistent_state_bytes
        replay_bytes = self._replay.persistent_bytes
        composer_bytes = persistent_bytes - ensemble_bytes - replay_bytes
        if persistent_bytes != self._persistent_bytes or composer_bytes != 28:
            raise ValueError("model replay rehearsal resource allocation is invalid")
        _require_float32_resource(
            "ModelReplayRehearsal state",
            vector_scalars=scalars,
        )
        if persistent_bytes > _INT32_MAX:
            raise ValueError("ModelReplayRehearsal state byte count must fit signed int32")
        if persistent_bytes > _UINT32_MAX:
            raise ValueError("model replay rehearsal state exceeds uint32 byte accounting")
        ensemble_size = self._config.ensemble.ensemble_size
        quota = self._config.replay_quota
        max_real = _INT32_MAX // quota
        real_candidates = ensemble_size
        replay_candidates = quota * ensemble_size
        return ModelReplayRehearsalResourceBudget(
            persistent_state_scalars=scalars,
            persistent_state_bytes=persistent_bytes,
            ensemble_state_bytes=ensemble_bytes,
            replay_state_bytes=replay_bytes,
            composer_accounting_bytes=composer_bytes,
            replay_total_capacity=self._config.replay.total_capacity,
            short_term_capacity=self._config.replay.short_term_capacity,
            long_term_capacity=self._config.replay.long_term_capacity,
            fixed_replay_quota=quota,
            max_real_model_update_candidates_per_event=real_candidates,
            max_replay_model_update_candidates_per_event=replay_candidates,
            max_total_model_update_candidates_per_event=(real_candidates + replay_candidates),
            max_actor_updates_per_event=0,
            max_critic_updates_per_event=0,
            max_state_builder_updates_per_event=0,
            max_real_event_count=max_real,
            max_rehearsal_attempt_count=max_real * quota,
        )


def save_model_replay_rehearsal_checkpoint(
    composer: ModelReplayRehearsal,
    state: ModelReplayRehearsalState,
    path: str | Path,
) -> None:
    """Persist the atomic child states, keys, counters, and strict metadata."""
    composer.validate_state(state)
    config = composer.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": MODEL_REPLAY_REHEARSAL_SCHEMA,
            "mechanism_status": MECHANISM_STATUS,
            "accepted_scientific_evidence": False,
            "composer_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": composer.resource_budget(state).to_config(),
        },
    )


def load_model_replay_rehearsal_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[ModelReplayRehearsal, ModelReplayRehearsalState]:
    """Restore a v1 composer and fail closed on metadata or state mismatch."""
    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != MODEL_REPLAY_REHEARSAL_SCHEMA:
        raise ValueError("checkpoint is not a ModelReplayRehearsal v1 checkpoint")
    if metadata.get("mechanism_status") != MECHANISM_STATUS:
        raise ValueError("checkpoint does not describe the model-only mechanism")
    if metadata.get("accepted_scientific_evidence") is not False:
        raise ValueError("checkpoint cannot claim accepted scientific evidence")
    config = metadata.get("composer_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing composer_config")
    digest = metadata.get("config_sha256")
    if not isinstance(digest, str) or digest != _config_digest(config):
        raise ValueError("model replay rehearsal config digest does not match")
    composer = ModelReplayRehearsal.from_config(config)
    if composer.to_config() != config:
        raise ValueError("model replay rehearsal checkpoint config is not canonical")
    key = jr.key(0) if template_key is None else template_key
    template = composer.init(key)
    expected_budget = composer.resource_budget(template).to_config()
    if metadata.get("resource_budget") != expected_budget:
        raise ValueError("model replay rehearsal resource budget does not match config")
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("model replay rehearsal checkpoint metadata changed between reads")
    state = cast(ModelReplayRehearsalState, restored)
    composer.validate_state(state)
    if composer.resource_budget(state).to_config() != expected_budget:
        raise ValueError("restored model replay rehearsal allocation is invalid")
    return composer, state


__all__ = [
    "MECHANISM_STATUS",
    "MODEL_REPLAY_REHEARSAL_SCHEMA",
    "ModelReplayRehearsal",
    "ModelReplayRehearsalConfig",
    "ModelReplayRehearsalDiagnostics",
    "ModelReplayRehearsalResourceBudget",
    "ModelReplayRehearsalResult",
    "ModelReplayRehearsalState",
    "ModelReplayRehearsalTrace",
    "RealModelReplayEvent",
    "ReplayActionConversion",
    "ReplayActionEncoding",
    "load_model_replay_rehearsal_checkpoint",
    "save_model_replay_rehearsal_checkpoint",
]
