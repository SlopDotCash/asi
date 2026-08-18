"""Development-only aggregate runner for primitive Prototype reference lives.

The module owns synchronous, functional slices across the host transaction
reducer, a primitive-only :class:`PrototypeReferenceAdapter`, and the selected
SwitchingTwoState/RiverSwim environments.  It is an L0 integration mechanism;
the separate current-schema checkpoint module covers only supported quiescent
states.  Neither module establishes ``reference-dev``, learning benefit,
portable recovery, robotics readiness, or scientific evidence.  The declared
in-process authority is only a static identity for the exact-action lane, not
an independent safety policy or evidence of veto conformance. The synchronous
fault boundary covers ordinary :class:`Exception` failures, not process death
or :class:`BaseException` cancellation.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
import struct
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.prototype_reference_adapter import (
    PrototypeReferenceAdapter,
    PrototypeReferenceState,
)
from alberta_framework.reference_agent import (
    MAX_DECISION_INDEX,
    AgentManifest,
    ArrayValue,
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAuthorization,
    DispatchCommand,
    DispatchReceipt,
    ReferenceAgentUpdate,
    ReferenceTransactionReducer,
    ReferenceTransactionState,
    SpaceSpec,
    StepResult,
    Transaction,
    TransactionPhase,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    RiverSwimState,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
    SwitchingTwoStateState,
)

REFERENCE_LIFE_CONFIG_SCHEMA = "asi.reference_life_config.preview1"
REFERENCE_LIFE_STATE_SCHEMA = "asi.reference_life_state.preview1"
REFERENCE_LIFE_METRICS_SCHEMA = "asi.reference_life_metrics.preview1"
REFERENCE_ENVIRONMENT_MANIFEST_SCHEMA = "asi.reference_environment_manifest.preview1"
SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID = "asi.switching_two_state_environment.preview1"
SWITCHING_ENVIRONMENT_STATE_SCHEMA = "asi.switching_two_state_state.preview1"
RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID = "asi.riverswim_environment.preview1"
RIVERSWIM_ENVIRONMENT_STATE_SCHEMA = "asi.riverswim_state.preview1"
EXACT_DISPATCH_SCHEMA = "asi.exact_dispatch.preview1"
EXACT_DISPATCH_STATE_SCHEMA = "asi.exact_dispatch_state.preview1"
REFERENCE_METRICS_CONFIG_SCHEMA = "asi.reference_life_metrics_config.preview1"
REFERENCE_LIFE_PRNG_IMPLEMENTATION = "threefry2x32"
REFERENCE_LIFE_RNG_SCHEDULE = "jax_threefry2x32_fold_in_environment_cursor.preview1"

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTOTYPE_LIFECYCLE = re.compile(r"^prototype\.[0-9a-f]{16}$")
_MAX_ID_LENGTH = 256
_MAX_SWITCHING_EXECUTIONS = int(np.iinfo(np.int32).max)
RIVERSWIM_REFERENCE_MAX_STATES = 12
_CheckpointPublishResult = TypeVar("_CheckpointPublishResult")


def _require_id(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_ID_LENGTH
        or _SAFE_ID.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a lowercase safe identifier")


def _require_digest(value: str, *, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_exact_bool(name: str, value: object) -> bool:
    """Reject leftover int/string identities before they become a verdict."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _require_regime_id(value: object, *, name: str = "regime_id") -> int:
    """Reject leftover bool identities that compare equal to 0 or 1."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return value


def _require_finite_real(name: str, value: object) -> float:
    """Reject leftover bool/NaN identities before they become recorded rewards."""
    if type(value) is bool or type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite real number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _require_prototype_lifecycle_id(value: str) -> None:
    if not isinstance(value, str) or _PROTOTYPE_LIFECYCLE.fullmatch(value) is None:
        raise ValueError(
            "lifecycle_id must use the Prototype lifecycle codec "
            "'prototype.' followed by 16 lowercase hexadecimal digits"
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reference-life configuration must be canonical finite JSON") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("reference-life configuration must be a JSON object")
    return encoded


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _space_descriptor(spec: SpaceSpec) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "semantic_id": spec.semantic_id,
        "cardinality": spec.cardinality,
        "low": None if spec.low is None else list(spec.low),
        "high": None if spec.high is None else list(spec.high),
    }


def _array_descriptor(value: ArrayValue) -> dict[str, Any]:
    return {
        "semantic_id": value.semantic_id,
        "dtype": value.dtype,
        "shape": list(value.shape),
        "payload_sha256": hashlib.sha256(value.payload).hexdigest(),
    }


def _replace_tuple_value(
    values: tuple[Any, Any],
    index: int,
    value: Any,
) -> tuple[Any, Any]:
    return (value, values[1]) if index == 0 else (values[0], value)


def _switching_metric_schedule(
    accepted_events: int,
    phase_length: int,
) -> tuple[tuple[int, int], int, int, int]:
    """Return the deterministic phase counters in constant time."""

    if accepted_events < 0 or phase_length <= 0:
        raise ValueError("switching metric schedule requires nonnegative events and a phase")
    full_segments, remainder = divmod(accepted_events, phase_length)
    phase_zero = ((full_segments + 1) // 2) * phase_length
    phase_one = (full_segments // 2) * phase_length
    if full_segments % 2 == 0:
        phase_zero += remainder
    else:
        phase_one += remainder
    current_phase = 0 if accepted_events == 0 else ((accepted_events - 1) // phase_length) % 2
    phase_switches = 0 if accepted_events == 0 else (accepted_events - 1) // phase_length
    segment_events = 0 if accepted_events == 0 else ((accepted_events - 1) % phase_length) + 1
    return (
        (phase_zero, phase_one),
        current_phase,
        phase_switches,
        segment_events,
    )


class LifePhase(enum.Enum):
    QUIESCENT = "quiescent"
    HALTED = "halted"
    COMPLETED = "completed"
    ABORTED = "aborted"


class HaltStage(enum.Enum):
    PRE_DISPATCH = "pre_dispatch"
    OUTCOME_PENDING = "outcome_pending"
    DISPATCH_UNCERTAIN = "dispatch_uncertain"
    POST_EXECUTION_DIVERGENCE = "post_execution_divergence"


class RecoveryMode(enum.Enum):
    RETRY_OUTCOME = "retry_outcome"
    RECONCILE_ONLY = "reconcile_only"
    ABORT_ONLY = "abort_only"


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceEnvironmentManifest:
    """Canonical declaration for one functional environment adapter."""

    schema: str
    implementation_id: str
    state_schema: str
    config_sha256: str
    manifest_id: str
    observation_spec: SpaceSpec
    action_spec: SpaceSpec
    max_executions: int
    _config_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_ENVIRONMENT_MANIFEST_SCHEMA:
            raise ValueError("unsupported environment manifest schema")
        _require_id(self.implementation_id, name="implementation_id")
        _require_id(self.state_schema, name="state_schema")
        _require_digest(self.config_sha256, name="config_sha256")
        _require_digest(self.manifest_id, name="manifest_id")
        if not isinstance(self.observation_spec, SpaceSpec):
            raise ValueError("observation_spec must be a SpaceSpec")
        if not isinstance(self.action_spec, SpaceSpec):
            raise ValueError("action_spec must be a SpaceSpec")
        if (
            isinstance(self.max_executions, bool)
            or not isinstance(self.max_executions, int)
            or self.max_executions <= 0
        ):
            raise ValueError("max_executions must be a positive integer")
        config = self.config
        if _canonical_json(config) != self._config_json:
            raise ValueError("environment config is not canonical JSON")
        if _digest_json(config) != self.config_sha256:
            raise ValueError("environment config digest mismatch")
        if _digest_json(self._manifest_payload()) != self.manifest_id:
            raise ValueError("environment manifest digest mismatch")

    @classmethod
    def from_config(
        cls,
        *,
        implementation_id: str,
        state_schema: str,
        config: Mapping[str, Any],
        observation_spec: SpaceSpec,
        action_spec: SpaceSpec,
        max_executions: int,
    ) -> ReferenceEnvironmentManifest:
        config_json = _canonical_json(config)
        config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        provisional = {
            "schema": REFERENCE_ENVIRONMENT_MANIFEST_SCHEMA,
            "implementation_id": implementation_id,
            "state_schema": state_schema,
            "config_sha256": config_sha256,
            "config": json.loads(config_json),
            "observation_spec": _space_descriptor(observation_spec),
            "action_spec": _space_descriptor(action_spec),
            "max_executions": max_executions,
        }
        return cls(
            schema=REFERENCE_ENVIRONMENT_MANIFEST_SCHEMA,
            implementation_id=implementation_id,
            state_schema=state_schema,
            config_sha256=config_sha256,
            manifest_id=_digest_json(provisional),
            observation_spec=observation_spec,
            action_spec=action_spec,
            max_executions=max_executions,
            _config_json=config_json,
        )

    @property
    def config(self) -> dict[str, Any]:
        value = json.loads(self._config_json)
        assert isinstance(value, dict)
        return value

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "implementation_id": self.implementation_id,
            "state_schema": self.state_schema,
            "config_sha256": self.config_sha256,
            "config": self.config,
            "observation_spec": _space_descriptor(self.observation_spec),
            "action_spec": _space_descriptor(self.action_spec),
            "max_executions": self.max_executions,
        }

    def descriptor(self) -> dict[str, Any]:
        return {**self._manifest_payload(), "manifest_id": self.manifest_id}


@dataclasses.dataclass(frozen=True, slots=True)
class ExactDispatchConfig:
    """Declared static authority and executor identity for the exact lane."""

    authority_id: str = "asi.reference_life.authority"
    policy_version: str = "asi.reference_life.exact_policy.1"
    executor_id: str = "asi.switching_two_state.executor"
    executor_epoch: str = "asi.switching_two_state.executor_epoch.1"
    mode: str = "exact_only"
    schema: str = EXACT_DISPATCH_SCHEMA
    config_sha256: str = dataclasses.field(init=False)
    manifest_id: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.schema != EXACT_DISPATCH_SCHEMA or self.mode != "exact_only":
            raise ValueError("the development runner supports exact dispatch only")
        for name in ("authority_id", "policy_version", "executor_id", "executor_epoch"):
            _require_id(getattr(self, name), name=name)
        config_sha256 = _digest_json(self.config)
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(
            self,
            "manifest_id",
            _digest_json(
                {
                    "schema": self.schema,
                    "config_sha256": config_sha256,
                    "config": self.config,
                    "max_dispatches": MAX_DECISION_INDEX + 1,
                }
            ),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "authority_role": "declared_static_identity",
            "safety_policy": "unimplemented",
            "veto_conformance": False,
            "authority_id": self.authority_id,
            "policy_version": self.policy_version,
            "executor_id": self.executor_id,
            "executor_epoch": self.executor_epoch,
        }

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config_sha256": self.config_sha256,
            "manifest_id": self.manifest_id,
            "config": self.config,
            "max_dispatches": MAX_DECISION_INDEX + 1,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ExactDispatchState:
    schema: str
    manifest_id: str
    authorizations: int
    commands_issued: int
    receipts_recorded: int
    last_command_id: str | None

    def __post_init__(self) -> None:
        if self.schema != EXACT_DISPATCH_STATE_SCHEMA:
            raise ValueError("unsupported exact-dispatch state schema")
        _require_digest(self.manifest_id, name="manifest_id")
        for name in ("authorizations", "commands_issued", "receipts_recorded"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not (self.receipts_recorded <= self.commands_issued <= self.authorizations):
            raise ValueError("dispatch counters are out of order")
        if self.last_command_id is not None:
            _require_id(self.last_command_id, name="last_command_id")


class ExactDispatchAdapter:
    """Stateless-policy adapter whose counters live entirely in its state."""

    __slots__ = ("_config",)

    def __init__(self, config: ExactDispatchConfig | None = None) -> None:
        self._config = ExactDispatchConfig() if config is None else config

    @property
    def config(self) -> ExactDispatchConfig:
        return self._config

    @property
    def max_dispatches(self) -> int:
        return MAX_DECISION_INDEX + 1

    def init(self) -> ExactDispatchState:
        return ExactDispatchState(
            schema=EXACT_DISPATCH_STATE_SCHEMA,
            manifest_id=self.config.manifest_id,
            authorizations=0,
            commands_issued=0,
            receipts_recorded=0,
            last_command_id=None,
        )

    def authorized(self, state: ExactDispatchState) -> ExactDispatchState:
        self._validate(state)
        return dataclasses.replace(state, authorizations=state.authorizations + 1)

    def issued(
        self,
        state: ExactDispatchState,
        command: DispatchCommand,
    ) -> ExactDispatchState:
        self._validate(state)
        return dataclasses.replace(
            state,
            commands_issued=state.commands_issued + 1,
            last_command_id=command.command_id,
        )

    def receipted(
        self,
        state: ExactDispatchState,
        receipt: DispatchReceipt,
    ) -> ExactDispatchState:
        self._validate(state)
        if state.last_command_id != receipt.command.command_id:
            raise DecisionOwnershipError("receipt does not own the last issued command")
        return dataclasses.replace(
            state,
            receipts_recorded=state.receipts_recorded + 1,
        )

    def _validate(self, state: ExactDispatchState) -> None:
        if not isinstance(state, ExactDispatchState):
            raise DecisionOwnershipError("dispatch state has the wrong type")
        if state.manifest_id != self.config.manifest_id:
            raise DecisionOwnershipError("dispatch state manifest mismatch")

    def validate_state(self, state: ExactDispatchState) -> None:
        """Validate a candidate before it becomes an aggregate trust root."""

        self._validate(state)


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeMetricsConfig:
    schema: str = REFERENCE_METRICS_CONFIG_SCHEMA
    oracle_regret: bool = True
    mode: str = "switching_two_phase"
    config_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_METRICS_CONFIG_SCHEMA:
            raise ValueError("unsupported metrics config schema")
        if not isinstance(self.oracle_regret, bool) or not self.oracle_regret:
            raise ValueError("the development metric requires oracle regret")
        if self.mode not in ("switching_two_phase", "stationary"):
            raise ValueError("metrics mode must be switching_two_phase or stationary")
        object.__setattr__(self, "config_sha256", _digest_json(self.config))

    @property
    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "oracle_regret": self.oracle_regret}

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config_sha256": self.config_sha256,
            "config": self.config,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeMetricsState:
    """Bounded deterministic online metrics; wall-clock telemetry is excluded."""

    schema: str
    config_sha256: str
    accepted_events: int
    parameter_change_events: int
    reward_sum: float
    oracle_reward_sum: float
    regret_sum: float
    phase_event_counts: tuple[int, int]
    phase_reward_sums: tuple[float, float]
    phase_regret_sums: tuple[float, float]
    phase_switches: int
    current_phase: int
    current_segment_events: int
    current_segment_reward: float
    current_segment_regret: float
    first_completed_segment_reward: tuple[float | None, float | None]
    latest_completed_segment_reward: tuple[float | None, float | None]

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_LIFE_METRICS_SCHEMA:
            raise ValueError("unsupported reference-life metrics schema")
        _require_digest(self.config_sha256, name="config_sha256")
        for name in (
            "accepted_events",
            "parameter_change_events",
            "phase_switches",
            "current_segment_events",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.current_phase not in (0, 1):
            raise ValueError("current_phase must be 0 or 1")
        pairs = (
            self.phase_event_counts,
            self.phase_reward_sums,
            self.phase_regret_sums,
            self.first_completed_segment_reward,
            self.latest_completed_segment_reward,
        )
        if any(len(values) != 2 for values in pairs):
            raise ValueError("phase and segment metric tuples must have length two")
        if any(value < 0 for value in self.phase_event_counts):
            raise ValueError("phase event counts must be nonnegative")
        if self.parameter_change_events > self.accepted_events:
            raise ValueError("parameter changes cannot exceed accepted events")
        if sum(self.phase_event_counts) != self.accepted_events:
            raise ValueError("phase event counts must sum to accepted events")
        if self.phase_switches > self.accepted_events:
            raise ValueError("phase switches cannot exceed accepted events")
        if self.current_segment_events > self.phase_event_counts[self.current_phase]:
            raise ValueError("current segment cannot exceed its phase event count")
        optional = (
            *self.first_completed_segment_reward,
            *self.latest_completed_segment_reward,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("optional segment metrics must be finite when present")
        numeric = (
            self.reward_sum,
            self.oracle_reward_sum,
            self.regret_sum,
            *self.phase_reward_sums,
            *self.phase_regret_sums,
            self.current_segment_reward,
            self.current_segment_regret,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("metric accumulators must be finite")


class ReferenceLifeMetricsAdapter:
    __slots__ = ("_config",)

    def __init__(self, config: ReferenceLifeMetricsConfig | None = None) -> None:
        self._config = ReferenceLifeMetricsConfig() if config is None else config

    @property
    def config(self) -> ReferenceLifeMetricsConfig:
        return self._config

    def init(self, *, initial_phase: int) -> ReferenceLifeMetricsState:
        if initial_phase not in (0, 1):
            raise ValueError("initial_phase must be 0 or 1")
        if self.config.mode == "stationary" and initial_phase != 0:
            raise ValueError("stationary metrics require regime zero")
        return ReferenceLifeMetricsState(
            schema=REFERENCE_LIFE_METRICS_SCHEMA,
            config_sha256=self.config.config_sha256,
            accepted_events=0,
            parameter_change_events=0,
            reward_sum=0.0,
            oracle_reward_sum=0.0,
            regret_sum=0.0,
            phase_event_counts=(0, 0),
            phase_reward_sums=(0.0, 0.0),
            phase_regret_sums=(0.0, 0.0),
            phase_switches=0,
            current_phase=initial_phase,
            current_segment_events=0,
            current_segment_reward=0.0,
            current_segment_regret=0.0,
            first_completed_segment_reward=(None, None),
            latest_completed_segment_reward=(None, None),
        )

    def observe(
        self,
        state: ReferenceLifeMetricsState,
        *,
        phase: int,
        reward: float,
        oracle_reward: float,
        parameters_changed: bool,
    ) -> ReferenceLifeMetricsState:
        if state.config_sha256 != self.config.config_sha256:
            raise DecisionOwnershipError("metrics state config mismatch")
        if phase not in (0, 1):
            raise ValueError("phase must be 0 or 1")
        if self.config.mode == "stationary" and phase != 0:
            raise ValueError("stationary metrics require regime zero")
        reward = float(reward)
        oracle_reward = float(oracle_reward)
        regret = oracle_reward - reward
        if not all(math.isfinite(value) for value in (reward, oracle_reward, regret)):
            raise ValueError("metric inputs must be finite")

        first = state.first_completed_segment_reward
        latest = state.latest_completed_segment_reward
        switches = state.phase_switches
        segment_events = state.current_segment_events
        segment_reward = state.current_segment_reward
        segment_regret = state.current_segment_regret
        if phase != state.current_phase:
            previous = state.current_phase
            if first[previous] is None:
                first = _replace_tuple_value(first, previous, segment_reward)
            latest = _replace_tuple_value(latest, previous, segment_reward)
            switches += 1
            segment_events = 0
            segment_reward = 0.0
            segment_regret = 0.0

        phase_counts = _replace_tuple_value(
            state.phase_event_counts,
            phase,
            state.phase_event_counts[phase] + 1,
        )
        phase_rewards = _replace_tuple_value(
            state.phase_reward_sums,
            phase,
            state.phase_reward_sums[phase] + reward,
        )
        phase_regrets = _replace_tuple_value(
            state.phase_regret_sums,
            phase,
            state.phase_regret_sums[phase] + regret,
        )
        oracle_reward_sum = (
            float(state.accepted_events + 1) * oracle_reward
            if self.config.mode == "stationary"
            else state.oracle_reward_sum + oracle_reward
        )
        return ReferenceLifeMetricsState(
            schema=state.schema,
            config_sha256=state.config_sha256,
            accepted_events=state.accepted_events + 1,
            parameter_change_events=(state.parameter_change_events + int(parameters_changed)),
            reward_sum=state.reward_sum + reward,
            oracle_reward_sum=oracle_reward_sum,
            regret_sum=state.regret_sum + regret,
            phase_event_counts=phase_counts,
            phase_reward_sums=phase_rewards,
            phase_regret_sums=phase_regrets,
            phase_switches=switches,
            current_phase=phase,
            current_segment_events=segment_events + 1,
            current_segment_reward=segment_reward + reward,
            current_segment_regret=segment_regret + regret,
            first_completed_segment_reward=first,
            latest_completed_segment_reward=latest,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceEnvironmentStart:
    """Generic functional environment initialization result."""

    state: Any
    observation: ArrayValue
    regime_id: int

    def __post_init__(self) -> None:
        _require_regime_id(self.regime_id, name="environment start regime_id")
        if type(self.observation) is not ArrayValue:
            raise ValueError("environment start observation must be an ArrayValue")


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceEnvironmentExecution:
    """Generic post-command result from one functional environment executor."""

    command: DispatchCommand
    state: Any
    applied_action: ArrayValue
    next_observation: ArrayValue
    reward: float
    discount: float
    terminated: bool
    truncated: bool
    autoreset: bool
    regime_id: int
    oracle_reward: float

    def __post_init__(self) -> None:
        reward = _require_finite_real("reward", self.reward)
        discount = _require_finite_real("discount", self.discount)
        oracle_reward = _require_finite_real("oracle_reward", self.oracle_reward)
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        _require_exact_bool("terminated", self.terminated)
        _require_exact_bool("truncated", self.truncated)
        _require_exact_bool("autoreset", self.autoreset)
        _require_regime_id(self.regime_id)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "oracle_reward", oracle_reward)
        if type(self.command) is not DispatchCommand:
            raise ValueError("environment execution requires a DispatchCommand")
        if type(self.applied_action) is not ArrayValue:
            raise ValueError("applied_action must be an ArrayValue")
        if type(self.next_observation) is not ArrayValue:
            raise ValueError("next_observation must be an ArrayValue")


# Compatibility names retained for the original deterministic slice.
SwitchingEnvironmentStart = ReferenceEnvironmentStart
SwitchingEnvironmentExecution = ReferenceEnvironmentExecution


class ReferenceAgentAdapter(Protocol):
    @property
    def manifest(self) -> AgentManifest: ...

    @property
    def max_accepted_events(self) -> int: ...

    def init(self, key: Array, *, lifecycle_id: str) -> Any: ...

    def start(
        self,
        state: Any,
        *,
        observation_id: str,
        observation: Any,
    ) -> tuple[Any, Decision]: ...

    def settle_dispatch(
        self,
        state: Any,
        authorization: DispatchAuthorization,
    ) -> tuple[Any, bool]: ...

    def apply_outcome(self, state: Any, transaction: Transaction) -> ReferenceAgentUpdate: ...

    def current_decision(self, state: Any) -> Decision: ...

    def validate_state(self, state: Any) -> None: ...


class ReferenceEnvironmentAdapter(Protocol):
    @property
    def manifest(self) -> ReferenceEnvironmentManifest: ...

    @property
    def max_executions(self) -> int: ...

    def init(self, key: Array) -> ReferenceEnvironmentStart: ...

    def execute(
        self,
        state: Any,
        command: DispatchCommand,
        *,
        key: Array,
    ) -> ReferenceEnvironmentExecution: ...

    def validate_execution(
        self,
        previous_state: Any,
        command: DispatchCommand,
        execution: ReferenceEnvironmentExecution,
        *,
        key: Array,
    ) -> None: ...

    def current_observation(self, state: Any) -> ArrayValue: ...

    def validate_state(self, state: Any) -> None: ...


class _ImmutableSwitchingTwoStateMDP(SwitchingTwoStateMDP):
    """Switching kernel whose behavior-defining fields cannot drift in-process."""

    __slots__ = ("_reference_life_frozen",)

    def __init__(self, config: SwitchingTwoStateConfig) -> None:
        object.__setattr__(self, "_reference_life_frozen", False)
        super().__init__(config)
        payoff_bytes = np.ascontiguousarray(self._payoffs_np, dtype="<f4").tobytes()
        immutable_payoffs = np.frombuffer(payoff_bytes, dtype="<f4").reshape((2, 2, 2))
        object.__setattr__(self, "_payoffs_np", immutable_payoffs)
        object.__setattr__(self, "_payoffs", jnp.asarray(immutable_payoffs))
        object.__setattr__(self, "_reference_life_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_reference_life_frozen", False):
            raise AttributeError("reference-life switching environment is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_reference_life_frozen", False):
            raise AttributeError("reference-life switching environment is immutable")
        object.__delattr__(self, name)


class SwitchingTwoStateReferenceEnvironment:
    """Strict codec wrapper around the otherwise clipping switching MDP."""

    __slots__ = ("_environment", "_frozen", "_manifest")

    def __init__(
        self,
        config: SwitchingTwoStateConfig,
        *,
        observation_spec: SpaceSpec,
        action_spec: SpaceSpec,
        executor_id: str = "asi.switching_two_state.executor",
        executor_epoch: str = "asi.switching_two_state.executor_epoch.1",
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        _require_id(executor_id, name="executor_id")
        _require_id(executor_epoch, name="executor_epoch")
        phase_length = config.phase_length
        if (
            isinstance(phase_length, bool)
            or not isinstance(phase_length, int)
            or phase_length <= 0
            or phase_length > _MAX_SWITCHING_EXECUTIONS
        ):
            raise ValueError(
                "phase_length must be an integer within signed-int32 capacity "
                f"[1, {_MAX_SWITCHING_EXECUTIONS}]"
            )
        if (
            observation_spec.kind != "box"
            or observation_spec.shape != (2,)
            or observation_spec.dtype != "float32"
        ):
            raise ValueError(
                "switching environment requires a float32 box observation of shape (2,)"
            )
        if (
            action_spec.kind != "discrete"
            or action_spec.shape != ()
            or action_spec.dtype != "int32"
            or action_spec.cardinality != 2
        ):
            raise ValueError(
                "switching environment requires a scalar int32 action of cardinality 2"
            )
        try:
            payoffs_a = np.asarray(config.payoffs_a, dtype=np.float32)
            payoffs_b = np.asarray(config.payoffs_b, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("switching payoffs must be numeric 2x2 matrices") from exc
        if payoffs_a.shape != (2, 2) or payoffs_b.shape != (2, 2):
            raise ValueError("switching payoffs must be 2x2 (state x action) matrices")
        if not np.all(np.isfinite(payoffs_a)) or not np.all(np.isfinite(payoffs_b)):
            raise ValueError("switching payoffs must be finite")
        canonical_config = SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=phase_length,
            payoffs_a=tuple(tuple(float(value) for value in row) for row in payoffs_a),
            payoffs_b=tuple(tuple(float(value) for value in row) for row in payoffs_b),
        )
        environment = _ImmutableSwitchingTwoStateMDP(canonical_config)
        self._environment = environment
        self._manifest = ReferenceEnvironmentManifest.from_config(
            implementation_id=SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID,
            state_schema=SWITCHING_ENVIRONMENT_STATE_SCHEMA,
            config={
                "environment_kind": "switching_two_state",
                "metrics_mode": "switching_two_phase",
                "phase_length": phase_length,
                "payoffs_a": [list(row) for row in canonical_config.payoffs_a],
                "payoffs_b": [list(row) for row in canonical_config.payoffs_b],
                "executor_id": executor_id,
                "executor_epoch": executor_epoch,
                "dynamics": "next_state_equals_action",
                "rng_mode": "key_schedule_bound_but_dynamics_deterministic",
                "boundary_mode": "continuing_only",
            },
            observation_spec=observation_spec,
            action_spec=action_spec,
            max_executions=_MAX_SWITCHING_EXECUTIONS,
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life switching environment is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life switching environment is immutable")
        object.__delattr__(self, name)

    @property
    def manifest(self) -> ReferenceEnvironmentManifest:
        return self._manifest

    @property
    def max_executions(self) -> int:
        return self.manifest.max_executions

    @property
    def executor_id(self) -> str:
        value = self.manifest.config["executor_id"]
        assert isinstance(value, str)
        return value

    @property
    def executor_epoch(self) -> str:
        value = self.manifest.config["executor_epoch"]
        assert isinstance(value, str)
        return value

    def _validate_state(
        self,
        state: SwitchingTwoStateState,
        *,
        require_capacity: bool,
    ) -> None:
        if not isinstance(state, SwitchingTwoStateState):
            raise DecisionOwnershipError("environment state has the wrong type")
        state_index = np.asarray(state.state_index)
        step_count = np.asarray(state.step_count)
        if state_index.shape != () or state_index.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("environment state_index must be scalar int32")
        if step_count.shape != () or step_count.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("environment step_count must be scalar int32")
        if int(state_index) not in (0, 1):
            raise DecisionOwnershipError("environment state_index is outside {0, 1}")
        if int(step_count) < 0 or (require_capacity and int(step_count) >= self.max_executions):
            raise DecisionOwnershipError("environment step counter capacity is exhausted")

    def validate_state(self, state: SwitchingTwoStateState) -> None:
        """Validate a restorable environment state at a quiescent boundary."""

        self._validate_state(state, require_capacity=True)

    def current_observation(self, state: SwitchingTwoStateState) -> ArrayValue:
        """Encode the observation owned by one validated environment state."""

        self.validate_state(state)
        return self.manifest.observation_spec.encode(
            np.asarray(self._environment.observe(state), dtype=np.float32)
        )

    def init(self, key: Array) -> SwitchingEnvironmentStart:
        state = self._environment.init(key)
        self._validate_state(state, require_capacity=True)
        observation = self.manifest.observation_spec.encode(
            np.asarray(self._environment.observe(state), dtype=np.float32)
        )
        return SwitchingEnvironmentStart(
            state=state,
            observation=observation,
            regime_id=int(self._environment.phase_id(state)),
        )

    def execute(
        self,
        state: SwitchingTwoStateState,
        command: DispatchCommand,
        *,
        key: Array,
    ) -> SwitchingEnvironmentExecution:
        """Execute only an already-valid exact int32 command, never a coerced action."""

        self._validate_state(state, require_capacity=True)
        if not isinstance(command, DispatchCommand):
            raise DecisionOwnershipError("environment execute requires a DispatchCommand")
        if command.executor_id != self.executor_id or command.executor_epoch != self.executor_epoch:
            raise DecisionOwnershipError("command targets another executor identity or epoch")
        # This validation occurs before the underlying environment, whose
        # public step method deliberately clips/casts for its legacy API.
        action = self.manifest.action_spec.encode(command.effective_action)
        action_array = action.to_numpy()
        if action_array.shape != () or action_array.dtype != np.dtype(np.int32):
            raise ValueError("executor action must remain a scalar int32")
        action_index = int(action_array)
        if action_index < 0 or action_index >= 2:
            raise ValueError("executor action is outside the discrete cardinality range")

        phase = int(self._environment.phase_id(state))
        next_observation, reward, next_state = self._environment.step(
            state,
            jnp.asarray(action_array, dtype=jnp.int32),
            key,
        )
        self._validate_state(next_state, require_capacity=False)
        encoded_observation = self.manifest.observation_spec.encode(
            np.asarray(next_observation, dtype=np.float32)
        )
        reward_value = float(np.asarray(reward, dtype=np.float32))
        return SwitchingEnvironmentExecution(
            command=command,
            state=next_state,
            applied_action=action,
            next_observation=encoded_observation,
            reward=reward_value,
            discount=1.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            regime_id=phase,
            oracle_reward=self._environment.optimal_average_reward(phase),
        )

    def validate_execution(
        self,
        previous_state: SwitchingTwoStateState,
        command: DispatchCommand,
        execution: ReferenceEnvironmentExecution,
        *,
        key: Array,
    ) -> None:
        """Recompute every deterministic result field before it reaches learning."""

        del key
        self._validate_state(previous_state, require_capacity=True)
        self._validate_state(execution.state, require_capacity=False)
        if execution.command != command:
            raise DecisionOwnershipError("execution result belongs to another command")
        previous_step = int(np.asarray(previous_state.step_count))
        next_step = int(np.asarray(execution.state.step_count))
        if next_step != previous_step + 1:
            raise DecisionOwnershipError("execution step counter did not advance exactly once")
        applied = self.manifest.action_spec.encode(execution.applied_action)
        applied_value = applied.to_python()
        if not isinstance(applied_value, int):
            raise DecisionOwnershipError("execution applied action is not a scalar integer")
        next_state_index = int(np.asarray(execution.state.state_index))
        if next_state_index != applied_value:
            raise DecisionOwnershipError(
                "execution state does not equal the deterministic applied action"
            )
        expected_observation = self.manifest.observation_spec.encode(
            np.asarray(self._environment.observe(execution.state), dtype=np.float32)
        )
        if execution.next_observation != expected_observation:
            raise DecisionOwnershipError("execution observation does not describe its state")
        phase = int(self._environment.phase_id(previous_state))
        if execution.regime_id != phase:
            raise DecisionOwnershipError("execution regime does not match the previous state")
        payoff_config = (
            self._environment.config.payoffs_a if phase == 0 else self._environment.config.payoffs_b
        )
        expected_reward = float(
            np.asarray(
                payoff_config[int(np.asarray(previous_state.state_index))][applied_value],
                dtype=np.float32,
            )
        )
        if execution.reward != expected_reward:
            raise DecisionOwnershipError("execution reward does not match switching dynamics")
        expected_oracle = self._environment.optimal_average_reward(phase)
        if execution.oracle_reward != expected_oracle:
            raise DecisionOwnershipError("execution oracle reward does not match its regime")
        if (
            execution.discount != 1.0
            or execution.terminated
            or execution.truncated
            or execution.autoreset
        ):
            raise DecisionOwnershipError(
                "switching execution must retain continuing-task boundary semantics"
            )


class _ImmutableRiverSwimMDP(RiverSwimMDP):
    """RiverSwim kernel whose behavior-defining fields cannot drift in-process."""

    __slots__ = ("_reference_life_frozen",)

    def __init__(self, config: RiverSwimConfig) -> None:
        object.__setattr__(self, "_reference_life_frozen", False)
        super().__init__(config)
        transition_shape = (2, config.n_states, config.n_states)
        transition_bytes = np.ascontiguousarray(self._transitions_np, dtype="<f4").tobytes()
        immutable_transitions = np.frombuffer(transition_bytes, dtype="<f4").reshape(
            transition_shape
        )
        reward_bytes = np.ascontiguousarray(self._rewards_np, dtype="<f4").tobytes()
        immutable_rewards = np.frombuffer(reward_bytes, dtype="<f4").reshape(
            (config.n_states, 2)
        )
        object.__setattr__(self, "_transitions_np", immutable_transitions)
        object.__setattr__(self, "_rewards_np", immutable_rewards)
        object.__setattr__(
            self,
            "_transition_logits",
            jnp.where(
                jnp.asarray(immutable_transitions) > 0.0,
                jnp.log(jnp.clip(jnp.asarray(immutable_transitions), 1e-30, 1.0)),
                -jnp.inf,
            ),
        )
        object.__setattr__(self, "_rewards", jnp.asarray(immutable_rewards))
        object.__setattr__(self, "_reference_life_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_reference_life_frozen", False):
            raise AttributeError("reference-life RiverSwim environment is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_reference_life_frozen", False):
            raise AttributeError("reference-life RiverSwim environment is immutable")
        object.__delattr__(self, name)


class RiverSwimReferenceEnvironment:
    """Strict keyed codec/executor wrapper for the stochastic RiverSwim MDP."""

    __slots__ = ("_environment", "_frozen", "_manifest", "_oracle_reward")

    def __init__(
        self,
        config: RiverSwimConfig,
        *,
        observation_spec: SpaceSpec,
        action_spec: SpaceSpec,
        executor_id: str = "asi.riverswim.executor",
        executor_epoch: str = "asi.riverswim.executor_epoch.1",
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        _require_id(executor_id, name="executor_id")
        _require_id(executor_epoch, name="executor_epoch")
        if not isinstance(config, RiverSwimConfig):
            raise ValueError("RiverSwim config has the wrong type")
        n_states = config.n_states
        if (
            isinstance(n_states, bool)
            or not isinstance(n_states, int)
            or n_states < 2
            or n_states > RIVERSWIM_REFERENCE_MAX_STATES
        ):
            raise ValueError(
                "RiverSwim n_states must be an integer in "
                f"[2, {RIVERSWIM_REFERENCE_MAX_STATES}] before exact 2**n oracle enumeration"
            )
        initial_state = config.initial_state
        if (
            isinstance(initial_state, bool)
            or not isinstance(initial_state, int)
            or initial_state < 0
            or initial_state >= n_states
        ):
            raise ValueError("RiverSwim initial_state is outside the configured chain")

        def canonical_float32(value: Any, *, name: str) -> float:
            if isinstance(value, bool):
                raise ValueError(f"RiverSwim {name} must be a finite float32 value")
            try:
                raw = float(value)
                cast = float(np.asarray(raw, dtype=np.float32))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"RiverSwim {name} must be a finite float32 value") from exc
            if not math.isfinite(raw) or not math.isfinite(cast):
                raise ValueError(f"RiverSwim {name} must be a finite float32 value")
            return cast

        p_right_up = canonical_float32(config.p_right_up, name="p_right_up")
        p_right_down = canonical_float32(config.p_right_down, name="p_right_down")
        reward_left = canonical_float32(config.reward_left, name="reward_left")
        reward_right = canonical_float32(config.reward_right, name="reward_right")
        if p_right_up <= 0.0 or p_right_down <= 0.0:
            raise ValueError("RiverSwim right-transition probabilities must be positive")
        if p_right_up + p_right_down > 1.0:
            raise ValueError("RiverSwim right-transition probabilities must sum to at most one")
        if (
            observation_spec.kind != "box"
            or observation_spec.shape != (n_states,)
            or observation_spec.dtype != "float32"
        ):
            raise ValueError(
                f"RiverSwim requires a float32 box observation of shape ({n_states},)"
            )
        if (
            action_spec.kind != "discrete"
            or action_spec.shape != ()
            or action_spec.dtype != "int32"
            or action_spec.cardinality != 2
        ):
            raise ValueError("RiverSwim requires a scalar int32 action of cardinality 2")

        canonical_config = RiverSwimConfig(  # type: ignore[call-arg]
            n_states=n_states,
            p_right_up=p_right_up,
            p_right_down=p_right_down,
            reward_left=reward_left,
            reward_right=reward_right,
            initial_state=initial_state,
        )
        environment = _ImmutableRiverSwimMDP(canonical_config)
        oracle_reward = environment.optimal_average_reward()
        if not math.isfinite(oracle_reward):
            raise ValueError("RiverSwim exact stationary oracle must be finite")
        self._environment = environment
        self._oracle_reward = oracle_reward
        self._manifest = ReferenceEnvironmentManifest.from_config(
            implementation_id=RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID,
            state_schema=RIVERSWIM_ENVIRONMENT_STATE_SCHEMA,
            config={
                "environment_kind": "riverswim",
                "metrics_mode": "stationary",
                "n_states": n_states,
                "p_right_up": p_right_up,
                "p_right_down": p_right_down,
                "reward_left": reward_left,
                "reward_right": reward_right,
                "initial_state": initial_state,
                "oracle_average_reward": oracle_reward,
                "executor_id": executor_id,
                "executor_epoch": executor_epoch,
                "dynamics": "riverswim_stochastic_chain",
                "rng_mode": "exact_jax_key_replay",
                "boundary_mode": "continuing_only",
            },
            observation_spec=observation_spec,
            action_spec=action_spec,
            max_executions=_MAX_SWITCHING_EXECUTIONS,
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life RiverSwim environment is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life RiverSwim environment is immutable")
        object.__delattr__(self, name)

    @property
    def manifest(self) -> ReferenceEnvironmentManifest:
        return self._manifest

    @property
    def max_executions(self) -> int:
        return self.manifest.max_executions

    @property
    def executor_id(self) -> str:
        value = self.manifest.config["executor_id"]
        assert isinstance(value, str)
        return value

    @property
    def executor_epoch(self) -> str:
        value = self.manifest.config["executor_epoch"]
        assert isinstance(value, str)
        return value

    def _validate_state(self, state: RiverSwimState, *, require_capacity: bool) -> None:
        if not isinstance(state, RiverSwimState):
            raise DecisionOwnershipError("RiverSwim state has the wrong type")
        state_index = np.asarray(state.state_index)
        step_count = np.asarray(state.step_count)
        if state_index.shape != () or state_index.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("RiverSwim state_index must be scalar int32")
        if step_count.shape != () or step_count.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("RiverSwim step_count must be scalar int32")
        if int(state_index) < 0 or int(state_index) >= self._environment.n_states:
            raise DecisionOwnershipError("RiverSwim state_index is outside the chain")
        if int(step_count) < 0 or (require_capacity and int(step_count) >= self.max_executions):
            raise DecisionOwnershipError("RiverSwim step counter capacity is exhausted")

    def validate_state(self, state: RiverSwimState) -> None:
        self._validate_state(state, require_capacity=True)

    def current_observation(self, state: RiverSwimState) -> ArrayValue:
        self.validate_state(state)
        return self.manifest.observation_spec.encode(
            np.asarray(self._environment.observe(state), dtype=np.float32)
        )

    def init(self, key: Array) -> ReferenceEnvironmentStart:
        state = self._environment.init(key)
        self._validate_state(state, require_capacity=True)
        observation = self.manifest.observation_spec.encode(
            np.asarray(self._environment.observe(state), dtype=np.float32)
        )
        return ReferenceEnvironmentStart(state=state, observation=observation, regime_id=0)

    def execute(
        self,
        state: RiverSwimState,
        command: DispatchCommand,
        *,
        key: Array,
    ) -> ReferenceEnvironmentExecution:
        self._validate_state(state, require_capacity=True)
        if not isinstance(command, DispatchCommand):
            raise DecisionOwnershipError("RiverSwim execute requires a DispatchCommand")
        if command.executor_id != self.executor_id or command.executor_epoch != self.executor_epoch:
            raise DecisionOwnershipError("command targets another RiverSwim executor")
        action = self.manifest.action_spec.encode(command.effective_action)
        action_array = action.to_numpy()
        if action_array.shape != () or action_array.dtype != np.dtype(np.int32):
            raise ValueError("RiverSwim executor action must remain scalar int32")
        action_index = int(action_array)
        if action_index < 0 or action_index >= 2:
            raise ValueError("RiverSwim executor action is outside cardinality range")
        next_observation, reward, next_state = self._environment.step(
            state,
            jnp.asarray(action_array, dtype=jnp.int32),
            key,
        )
        self._validate_state(next_state, require_capacity=False)
        return ReferenceEnvironmentExecution(
            command=command,
            state=next_state,
            applied_action=action,
            next_observation=self.manifest.observation_spec.encode(
                np.asarray(next_observation, dtype=np.float32)
            ),
            reward=float(np.asarray(reward, dtype=np.float32)),
            discount=1.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            regime_id=0,
            oracle_reward=self._oracle_reward,
        )

    def validate_execution(
        self,
        previous_state: RiverSwimState,
        command: DispatchCommand,
        execution: ReferenceEnvironmentExecution,
        *,
        key: Array,
    ) -> None:
        """Replay the stochastic transition with the exact scheduled key."""

        self._validate_state(previous_state, require_capacity=True)
        self._validate_state(execution.state, require_capacity=False)
        if execution.command != command:
            raise DecisionOwnershipError("RiverSwim execution belongs to another command")
        applied = self.manifest.action_spec.encode(execution.applied_action)
        applied_value = applied.to_python()
        if not isinstance(applied_value, int):
            raise DecisionOwnershipError("RiverSwim applied action is not scalar int32")
        try:
            expected_observation, expected_reward_array, expected_state = self._environment.step(
                previous_state,
                jnp.asarray(applied_value, dtype=jnp.int32),
                key,
            )
        except Exception as exc:
            raise DecisionOwnershipError("RiverSwim stochastic key replay failed") from exc
        expected_state_index = np.asarray(expected_state.state_index)
        actual_state_index = np.asarray(execution.state.state_index)
        expected_step_count = np.asarray(expected_state.step_count)
        actual_step_count = np.asarray(execution.state.step_count)
        if (
            expected_state_index.dtype != actual_state_index.dtype
            or expected_state_index.tobytes() != actual_state_index.tobytes()
            or expected_step_count.dtype != actual_step_count.dtype
            or expected_step_count.tobytes() != actual_step_count.tobytes()
        ):
            raise DecisionOwnershipError(
                "RiverSwim stochastic transition does not match exact key replay"
            )
        expected_encoded_observation = self.manifest.observation_spec.encode(
            np.asarray(expected_observation, dtype=np.float32)
        )
        if execution.next_observation != expected_encoded_observation:
            raise DecisionOwnershipError("RiverSwim observation does not match key replay")
        expected_reward = float(np.asarray(expected_reward_array, dtype=np.float32))
        if struct.pack(">d", execution.reward) != struct.pack(">d", expected_reward):
            raise DecisionOwnershipError("RiverSwim reward does not match key replay")
        if execution.regime_id != 0:
            raise DecisionOwnershipError("RiverSwim stationary regime must be zero")
        if struct.pack(">d", execution.oracle_reward) != struct.pack(">d", self._oracle_reward):
            raise DecisionOwnershipError("RiverSwim stationary oracle reward is inconsistent")
        if (
            execution.discount != 1.0
            or execution.terminated
            or execution.truncated
            or execution.autoreset
        ):
            raise DecisionOwnershipError(
                "RiverSwim execution must retain continuing-task boundary semantics"
            )


def _agent_descriptor(
    manifest: AgentManifest,
    *,
    max_accepted_events: int,
) -> dict[str, Any]:
    return {
        "schema": manifest.schema,
        "implementation_id": manifest.implementation_id,
        "state_schema": manifest.state_schema,
        "config_sha256": manifest.config_sha256,
        "manifest_id": manifest.manifest_id,
        "config": manifest.config,
        "observation_spec": _space_descriptor(manifest.observation_spec),
        "action_spec": _space_descriptor(manifest.action_spec),
        "capabilities": {
            "dispatch_rebinding": manifest.capabilities.dispatch_rebinding,
        },
        "max_accepted_events": max_accepted_events,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeConfig:
    """Deeply immutable canonical binding for one development life."""

    schema: str
    lifecycle_id: str
    seed: int
    max_accepted_events: int
    agent_manifest_id: str
    environment_manifest_id: str
    dispatch_manifest_id: str
    metrics_config_sha256: str
    config_sha256: str
    _config_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_LIFE_CONFIG_SCHEMA:
            raise ValueError("unsupported reference-life config schema")
        _require_id(self.lifecycle_id, name="lifecycle_id")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > int(np.iinfo(np.uint32).max)
        ):
            raise ValueError("seed must be a uint32 integer")
        if (
            isinstance(self.max_accepted_events, bool)
            or not isinstance(self.max_accepted_events, int)
            or self.max_accepted_events <= 0
            or self.max_accepted_events > MAX_DECISION_INDEX + 1
        ):
            raise ValueError("max_accepted_events must be a positive host-capacity count")
        for name in (
            "agent_manifest_id",
            "environment_manifest_id",
            "dispatch_manifest_id",
            "metrics_config_sha256",
            "config_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        payload = self.config
        if _canonical_json(payload) != self._config_json:
            raise ValueError("life configuration is not canonical JSON")
        if _digest_json(payload) != self.config_sha256:
            raise ValueError("life configuration digest mismatch")
        expected = {
            "lifecycle_id": self.lifecycle_id,
            "seed": self.seed,
            "max_accepted_events": self.max_accepted_events,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"life configuration field {key!r} is not bound")
        if payload.get("agent", {}).get("manifest_id") != self.agent_manifest_id:
            raise ValueError("agent manifest identity is not bound")
        if payload.get("environment", {}).get("manifest_id") != self.environment_manifest_id:
            raise ValueError("environment manifest identity is not bound")
        if payload.get("dispatch", {}).get("manifest_id") != self.dispatch_manifest_id:
            raise ValueError("dispatch manifest identity is not bound")
        if payload.get("metrics", {}).get("config_sha256") != self.metrics_config_sha256:
            raise ValueError("metrics configuration identity is not bound")

    @classmethod
    def from_components(
        cls,
        *,
        lifecycle_id: str,
        seed: int,
        max_accepted_events: int,
        agent_manifest: AgentManifest,
        agent_capacity: int,
        environment_manifest: ReferenceEnvironmentManifest,
        dispatch_config: ExactDispatchConfig,
        metrics_config: ReferenceLifeMetricsConfig,
    ) -> ReferenceLifeConfig:
        payload = {
            "schema": REFERENCE_LIFE_CONFIG_SCHEMA,
            "lifecycle_id": lifecycle_id,
            "seed": seed,
            "max_accepted_events": max_accepted_events,
            "rng_schedule": REFERENCE_LIFE_RNG_SCHEDULE,
            "observation_id_schedule": "lifecycle_observation_index.preview1",
            "checkpoint_policy": "atomic_quiescent_bundle.preview1",
            "agent": _agent_descriptor(
                agent_manifest,
                max_accepted_events=agent_capacity,
            ),
            "environment": environment_manifest.descriptor(),
            "dispatch": dispatch_config.descriptor(),
            "metrics": metrics_config.descriptor(),
        }
        config_json = _canonical_json(payload)
        return cls(
            schema=REFERENCE_LIFE_CONFIG_SCHEMA,
            lifecycle_id=lifecycle_id,
            seed=seed,
            max_accepted_events=max_accepted_events,
            agent_manifest_id=agent_manifest.manifest_id,
            environment_manifest_id=environment_manifest.manifest_id,
            dispatch_manifest_id=dispatch_config.manifest_id,
            metrics_config_sha256=metrics_config.config_sha256,
            config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            _config_json=config_json,
        )

    @property
    def config(self) -> dict[str, Any]:
        value = json.loads(self._config_json)
        assert isinstance(value, dict)
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class LifeHalt:
    stage: HaltStage
    recovery_mode: RecoveryMode
    reason: str
    recovery_attempts: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, HaltStage):
            raise ValueError("halt stage must be a HaltStage")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise ValueError("recovery_mode must be a RecoveryMode")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("halt reason must be nonempty")
        if (
            isinstance(self.recovery_attempts, bool)
            or not isinstance(self.recovery_attempts, int)
            or self.recovery_attempts < 0
        ):
            raise ValueError("recovery_attempts must be nonnegative")


@dataclasses.dataclass(frozen=True, slots=True)
class PendingOutcome:
    transaction: Transaction
    regime_id: int
    oracle_reward: float

    def __post_init__(self) -> None:
        _require_regime_id(self.regime_id, name="pending outcome regime_id")
        oracle_reward = _require_finite_real(
            "pending outcome oracle reward", self.oracle_reward
        )
        object.__setattr__(self, "oracle_reward", oracle_reward)
        if type(self.transaction) is not Transaction:
            raise ValueError("pending outcome requires a Transaction")


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeState:
    """One immutable aggregate owning every mutable development-life component."""

    schema: str
    config_sha256: str
    lifecycle_id: str
    phase: LifePhase
    accepted_events: int
    dispatch_attempts: int
    executed_events: int
    environment_rng_cursor: int
    commit_generation: int
    checkpoint_generation: int
    agent_state: Any
    environment_state: Any
    transaction_state: ReferenceTransactionState
    dispatch_state: ExactDispatchState
    metrics: ReferenceLifeMetricsState
    pending_outcome: PendingOutcome | None
    halt: LifeHalt | None
    transcript_sha256: str

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_LIFE_STATE_SCHEMA:
            raise ValueError("unsupported reference-life state schema")
        _require_digest(self.config_sha256, name="config_sha256")
        _require_id(self.lifecycle_id, name="lifecycle_id")
        _require_digest(self.transcript_sha256, name="transcript_sha256")
        if not isinstance(self.phase, LifePhase):
            raise ValueError("phase must be a LifePhase")
        for name in (
            "accepted_events",
            "dispatch_attempts",
            "executed_events",
            "environment_rng_cursor",
            "commit_generation",
            "checkpoint_generation",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not (self.accepted_events <= self.executed_events <= self.dispatch_attempts):
            raise ValueError("life counters violate accepted <= outcomes <= dispatches")
        if self.environment_rng_cursor != self.dispatch_attempts:
            raise ValueError("environment RNG cursor must count dispatch attempts")
        if not isinstance(self.transaction_state, ReferenceTransactionState):
            raise ValueError("transaction_state has the wrong type")
        if not isinstance(self.dispatch_state, ExactDispatchState):
            raise ValueError("dispatch_state has the wrong type")
        if not isinstance(self.metrics, ReferenceLifeMetricsState):
            raise ValueError("metrics has the wrong type")
        if self.metrics.accepted_events != self.accepted_events:
            raise ValueError("metrics and runner accepted-event counters disagree")

        if self.phase in (LifePhase.QUIESCENT, LifePhase.COMPLETED):
            if self.pending_outcome is not None or self.halt is not None:
                raise ValueError("quiescent/completed life cannot retain a halt")
            if not (self.accepted_events == self.executed_events == self.dispatch_attempts):
                raise ValueError("quiescent/completed life counters must align")
            if self.transaction_state.phase not in (
                TransactionPhase.ARMED,
                TransactionPhase.EXHAUSTED,
            ):
                raise ValueError("quiescent/completed life must own an armed/exhausted transaction")
        elif self.phase is LifePhase.HALTED:
            if self.halt is None:
                raise ValueError("halted life requires halt metadata")
            if self.halt.stage is HaltStage.OUTCOME_PENDING:
                if self.pending_outcome is None:
                    raise ValueError("outcome-pending halt requires the retained outcome")
                if self.transaction_state.transaction != self.pending_outcome.transaction:
                    raise ValueError("pending outcome does not match transaction state")
            elif self.pending_outcome is not None:
                raise ValueError("only an outcome-pending halt may retain a transaction")
        elif self.phase is LifePhase.ABORTED and self.halt is None:
            raise ValueError("aborted life requires terminal halt metadata")


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeEvent:
    command: DispatchCommand
    receipt: DispatchReceipt
    transaction: Transaction
    step_result: StepResult
    regime_id: int
    oracle_reward: float
    transcript_sha256: str
    recovered: bool

    def __post_init__(self) -> None:
        """Reject leftover recovered/regime/reward identities before they record."""
        _require_exact_bool("recovered", self.recovered)
        _require_regime_id(self.regime_id)
        oracle_reward = _require_finite_real("oracle_reward", self.oracle_reward)
        _require_digest(self.transcript_sha256, name="transcript_sha256")
        object.__setattr__(self, "oracle_reward", oracle_reward)
        for name, expected_type in (
            ("command", DispatchCommand),
            ("receipt", DispatchReceipt),
            ("transaction", Transaction),
            ("step_result", StepResult),
        ):
            if type(getattr(self, name)) is not expected_type:
                raise ValueError(f"{name} has the wrong exact reference-life type")


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeStep:
    state: ReferenceLifeState
    event: ReferenceLifeEvent | None
    accepted: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        _require_exact_bool("accepted", self.accepted)
        if type(self.state) is not ReferenceLifeState:
            raise ValueError("state must be a ReferenceLifeState")
        if self.event is not None and type(self.event) is not ReferenceLifeEvent:
            raise ValueError("event must be a ReferenceLifeEvent or None")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted life step cannot carry a rejection reason")
        if not self.accepted and (
            type(self.rejection_reason) is not str or not self.rejection_reason.strip()
        ):
            raise ValueError("rejected life step requires a reason")
        if self.accepted and self.event is None:
            raise ValueError("accepted life step requires an event")
        if self.event is not None:
            if self.event.step_result.transaction_accepted is not self.accepted:
                raise ValueError("life step acceptance disagrees with its event")
            if self.event.transcript_sha256 != self.state.transcript_sha256:
                raise ValueError("life step event transcript disagrees with state")
            if not self.accepted and self.event.step_result.rejection_reason != (
                self.rejection_reason
            ):
                raise ValueError("life step rejection reason disagrees with its event")

    @property
    def recovery_required(self) -> bool:
        return not self.accepted and self.state.phase is LifePhase.HALTED


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeRun:
    state: ReferenceLifeState
    events: tuple[ReferenceLifeEvent, ...]


@dataclasses.dataclass(slots=True)
class _PostIssueContext:
    """Ephemeral deepest-valid prefix for the post-issue emergency CAS."""

    transaction_state: ReferenceTransactionState
    dispatch_state: ExactDispatchState
    issued_state: ReferenceTransactionState | None = None
    command: DispatchCommand | None = None
    execution: ReferenceEnvironmentExecution | None = None
    receipt: DispatchReceipt | None = None
    transcript_sha256: str | None = None


def _initial_transcript(
    config: ReferenceLifeConfig,
    observation: ArrayValue,
) -> str:
    return _digest_json(
        {
            "schema": "asi.reference_life_transcript_init.preview1",
            "config_sha256": config.config_sha256,
            "lifecycle_id": config.lifecycle_id,
            "observation": _array_descriptor(observation),
        }
    )


def _advance_transcript(
    previous_digest: str,
    *,
    execution: ReferenceEnvironmentExecution,
    receipt: DispatchReceipt,
    next_observation_id: str,
) -> str:
    command = receipt.command
    payload = {
        "schema": "asi.reference_life_transcript_event.preview1",
        "previous_sha256": previous_digest,
        "decision_id": command.decision_id,
        "observation_id": command.decision.observation_id,
        "command_id": command.command_id,
        "executor_result_command_id": execution.command.command_id,
        "receipt_id": receipt.receipt_id,
        "proposed_action": _array_descriptor(command.decision.proposed_action)
        if command.decision.proposed_action is not None
        else None,
        "settled_action": _array_descriptor(command.effective_action),
        "applied_action": _array_descriptor(receipt.applied_action),
        "reward_hex": float(execution.reward).hex(),
        "discount_hex": float(execution.discount).hex(),
        "terminated": execution.terminated,
        "truncated": execution.truncated,
        "autoreset": execution.autoreset,
        "next_observation_id": next_observation_id,
        "next_observation": _array_descriptor(execution.next_observation),
        "regime_id": execution.regime_id,
        "oracle_reward_hex": float(execution.oracle_reward).hex(),
    }
    return _digest_json(payload)


class ReferenceLifeRunner:
    """Single-writer synchronous owner for one aggregate development life."""

    __slots__ = (
        "_agent_adapter",
        "_config",
        "_current_state",
        "_dispatch_adapter",
        "_environment_adapter",
        "_lock",
        "_metrics_adapter",
        "_reducer",
    )

    def __init__(
        self,
        *,
        config: ReferenceLifeConfig,
        agent_adapter: ReferenceAgentAdapter,
        environment_adapter: ReferenceEnvironmentAdapter,
        dispatch_adapter: ExactDispatchAdapter,
        metrics_adapter: ReferenceLifeMetricsAdapter,
    ) -> None:
        if (
            agent_adapter.manifest.config.get("lifecycle_codec")
            == "prototype_uint32x2_hex.preview1"
        ):
            _require_prototype_lifecycle_id(config.lifecycle_id)
        canonical_config = ReferenceLifeConfig.from_components(
            lifecycle_id=config.lifecycle_id,
            seed=config.seed,
            max_accepted_events=config.max_accepted_events,
            agent_manifest=agent_adapter.manifest,
            agent_capacity=agent_adapter.max_accepted_events,
            environment_manifest=environment_adapter.manifest,
            dispatch_config=dispatch_adapter.config,
            metrics_config=metrics_adapter.config,
        )
        if config != canonical_config:
            raise ValueError(
                "life configuration must exactly equal the canonical live-component binding"
            )
        if config.agent_manifest_id != agent_adapter.manifest.manifest_id:
            raise ValueError("life config agent manifest mismatch")
        if config.environment_manifest_id != environment_adapter.manifest.manifest_id:
            raise ValueError("life config environment manifest mismatch")
        if config.dispatch_manifest_id != dispatch_adapter.config.manifest_id:
            raise ValueError("life config dispatch manifest mismatch")
        if config.metrics_config_sha256 != metrics_adapter.config.config_sha256:
            raise ValueError("life config metrics mismatch")
        environment_config = environment_adapter.manifest.config
        agent_config = agent_adapter.manifest.config
        if environment_config.get("executor_id") != dispatch_adapter.config.executor_id:
            raise ValueError("dispatch and environment executor IDs differ")
        if environment_config.get("executor_epoch") != dispatch_adapter.config.executor_epoch:
            raise ValueError("dispatch and environment executor epochs differ")
        if environment_config.get("metrics_mode") != metrics_adapter.config.mode:
            raise ValueError("metrics and environment modes differ")
        agent_environment_kind = agent_config.get("environment_kind")
        if (
            agent_environment_kind is not None
            and agent_environment_kind != environment_config.get("environment_kind")
        ):
            raise ValueError("agent and environment kinds differ")
        bound_environment = agent_config.get("environment_config")
        if bound_environment is not None:
            environment_kind = environment_config.get("environment_kind")
            if environment_kind == "switching_two_state":
                expected_bound_environment = {
                    "phase_length": environment_config.get("phase_length"),
                    "payoffs_a": environment_config.get("payoffs_a"),
                    "payoffs_b": environment_config.get("payoffs_b"),
                }
            elif environment_kind == "riverswim":
                expected_bound_environment = {
                    "n_states": environment_config.get("n_states"),
                    "p_right_up": environment_config.get("p_right_up"),
                    "p_right_down": environment_config.get("p_right_down"),
                    "reward_left": environment_config.get("reward_left"),
                    "reward_right": environment_config.get("reward_right"),
                    "initial_state": environment_config.get("initial_state"),
                }
            else:
                raise ValueError("environment-bound agent selected an unsupported environment")
            if bound_environment != expected_bound_environment:
                raise ValueError("agent's bound environment differs from the live environment")
        if agent_adapter.manifest.observation_spec != environment_adapter.manifest.observation_spec:
            raise ValueError("agent and environment observation codecs must match exactly")
        if agent_adapter.manifest.action_spec != environment_adapter.manifest.action_spec:
            raise ValueError("agent and environment action codecs must match exactly")
        capacities = {
            "agent": agent_adapter.max_accepted_events,
            "environment": environment_adapter.max_executions,
            "dispatch": dispatch_adapter.max_dispatches,
            "host": MAX_DECISION_INDEX + 1,
        }
        invalid_capacity = [
            name for name, capacity in capacities.items() if config.max_accepted_events > capacity
        ]
        if invalid_capacity:
            detail = ", ".join(f"{name}={capacities[name]}" for name in sorted(invalid_capacity))
            raise ValueError("max_accepted_events exceeds selected component capacity: " + detail)
        self._config = config
        self._agent_adapter = agent_adapter
        self._environment_adapter = environment_adapter
        self._dispatch_adapter = dispatch_adapter
        self._metrics_adapter = metrics_adapter
        self._reducer = ReferenceTransactionReducer(agent_adapter.manifest)
        self._current_state: ReferenceLifeState | None = None
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        *,
        agent_adapter: ReferenceAgentAdapter,
        environment_adapter: ReferenceEnvironmentAdapter,
        lifecycle_id: str,
        seed: int,
        max_accepted_events: int,
        dispatch_config: ExactDispatchConfig | None = None,
        metrics_config: ReferenceLifeMetricsConfig | None = None,
    ) -> ReferenceLifeRunner:
        dispatch_adapter = ExactDispatchAdapter(dispatch_config)
        metrics_adapter = ReferenceLifeMetricsAdapter(metrics_config)
        environment_config = environment_adapter.manifest.config
        if environment_config.get("executor_id") != dispatch_adapter.config.executor_id:
            raise ValueError("dispatch and environment executor IDs differ")
        if environment_config.get("executor_epoch") != dispatch_adapter.config.executor_epoch:
            raise ValueError("dispatch and environment executor epochs differ")
        config = ReferenceLifeConfig.from_components(
            lifecycle_id=lifecycle_id,
            seed=seed,
            max_accepted_events=max_accepted_events,
            agent_manifest=agent_adapter.manifest,
            agent_capacity=agent_adapter.max_accepted_events,
            environment_manifest=environment_adapter.manifest,
            dispatch_config=dispatch_adapter.config,
            metrics_config=metrics_adapter.config,
        )
        return cls(
            config=config,
            agent_adapter=agent_adapter,
            environment_adapter=environment_adapter,
            dispatch_adapter=dispatch_adapter,
            metrics_adapter=metrics_adapter,
        )

    @property
    def config(self) -> ReferenceLifeConfig:
        return self._config

    @property
    def agent_adapter(self) -> ReferenceAgentAdapter:
        return self._agent_adapter

    @property
    def environment_adapter(self) -> ReferenceEnvironmentAdapter:
        return self._environment_adapter

    @property
    def current_state(self) -> ReferenceLifeState | None:
        with self._lock:
            return self._current_state

    def __reduce__(self) -> Any:
        raise TypeError(
            "live reference-life runner cannot be serialized; use the quiescent "
            "whole-life checkpoint API"
        )

    def _require_current_locked(self, state: ReferenceLifeState) -> None:
        if not isinstance(state, ReferenceLifeState):
            raise DecisionOwnershipError("state must be a ReferenceLifeState")
        if state.config_sha256 != self.config.config_sha256:
            raise DecisionOwnershipError("life state configuration mismatch")
        if state.lifecycle_id != self.config.lifecycle_id:
            raise DecisionOwnershipError("life state lifecycle mismatch")
        if self._current_state is None:
            raise DecisionOwnershipError("life runner must be initialized before use")
        if state is not self._current_state:
            raise DecisionOwnershipError("life state is stale, replayed, or not current")
        self._reducer.validate_state(state.transaction_state)

    def _commit_locked(
        self,
        previous: ReferenceLifeState,
        candidate: ReferenceLifeState,
    ) -> ReferenceLifeState:
        if self._current_state is not previous:
            raise DecisionOwnershipError("life state changed before aggregate commit")
        if candidate.config_sha256 != self.config.config_sha256:
            raise DecisionOwnershipError("candidate life configuration mismatch")
        self._reducer.validate_state(candidate.transaction_state)
        self._current_state = candidate
        return candidate

    @staticmethod
    def _int32_scalar(value: Any, *, name: str) -> int:
        array = np.asarray(value)
        if array.shape != () or array.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError(f"{name} must be a scalar int32")
        return int(array)

    def validate_checkpoint_state(self, state: ReferenceLifeState) -> None:
        """Validate the exact cross-component quiescent checkpoint boundary."""

        if not isinstance(state, ReferenceLifeState):
            raise DecisionOwnershipError("checkpoint state must be a ReferenceLifeState")
        if not isinstance(self._agent_adapter, PrototypeReferenceAdapter):
            raise DecisionOwnershipError("checkpoint runner must use the Prototype adapter")
        if state.config_sha256 != self.config.config_sha256:
            raise DecisionOwnershipError("checkpoint life configuration mismatch")
        if state.lifecycle_id != self.config.lifecycle_id:
            raise DecisionOwnershipError("checkpoint lifecycle mismatch")
        if state.phase is not LifePhase.QUIESCENT:
            raise DecisionOwnershipError("checkpoint state must be quiescent")
        if state.transaction_state.phase is not TransactionPhase.ARMED:
            raise DecisionOwnershipError("checkpoint transaction must be armed")
        if state.pending_outcome is not None or state.halt is not None:
            raise DecisionOwnershipError("checkpoint state cannot retain recovery state")
        if state.accepted_events >= self.config.max_accepted_events:
            raise DecisionOwnershipError("checkpoint state must precede life completion")
        if state.checkpoint_generation > state.commit_generation:
            raise DecisionOwnershipError("checkpoint generation exceeds commit generation")
        if state.commit_generation < (state.accepted_events + state.checkpoint_generation):
            raise DecisionOwnershipError("checkpoint generation history predates accepted events")

        self._reducer.validate_state(state.transaction_state)
        self._dispatch_adapter.validate_state(state.dispatch_state)
        self._environment_adapter.validate_state(state.environment_state)
        self._agent_adapter.validate_state(state.agent_state)
        if state.metrics.config_sha256 != self._metrics_adapter.config.config_sha256:
            raise DecisionOwnershipError("checkpoint metrics configuration mismatch")

        accepted = state.accepted_events
        if not (
            state.dispatch_attempts
            == state.executed_events
            == state.environment_rng_cursor
            == state.metrics.accepted_events
            == accepted
        ):
            raise DecisionOwnershipError("checkpoint event counters do not align")
        if not (
            state.dispatch_state.authorizations
            == state.dispatch_state.commands_issued
            == state.dispatch_state.receipts_recorded
            == accepted
        ):
            raise DecisionOwnershipError("checkpoint dispatch counters do not align")
        expected_last_command = (
            None if accepted == 0 else f"{state.lifecycle_id}:{accepted - 1}:command"
        )
        if state.dispatch_state.last_command_id != expected_last_command:
            raise DecisionOwnershipError("checkpoint last command identity is inconsistent")

        if not isinstance(state.agent_state, PrototypeReferenceState):
            raise DecisionOwnershipError("checkpoint agent wrapper has the wrong type")
        agent_state = state.agent_state
        if (
            agent_state.manifest_id != self._agent_adapter.manifest.manifest_id
            or agent_state.config_sha256 != self._agent_adapter.manifest.config_sha256
            or agent_state.lifecycle_id != state.lifecycle_id
            or agent_state.decision_index != accepted
        ):
            raise DecisionOwnershipError("checkpoint agent wrapper identity is inconsistent")
        expected_observation_id = f"{state.lifecycle_id}:observation:{accepted}"
        if agent_state.current_observation_id != expected_observation_id:
            raise DecisionOwnershipError("checkpoint observation identity is inconsistent")

        prototype_step = self._int32_scalar(
            agent_state.agent_state.step_count,
            name="Prototype step_count",
        )
        prototype_observations = self._int32_scalar(
            agent_state.agent_state.observation_event_count,
            name="Prototype observation_event_count",
        )
        environment_step = self._int32_scalar(
            state.environment_state.step_count,
            name="environment step_count",
        )
        if prototype_step != accepted or environment_step != accepted:
            raise DecisionOwnershipError("checkpoint component step counters do not align")
        if prototype_observations != accepted + 1:
            raise DecisionOwnershipError("checkpoint observation counter is inconsistent")

        transaction = state.transaction_state
        if transaction.next_decision_index != accepted or transaction.decision is None:
            raise DecisionOwnershipError("checkpoint ledger decision index is inconsistent")
        decision = self._agent_adapter.current_decision(agent_state)
        if transaction.decision != decision:
            raise DecisionOwnershipError("checkpoint current decision lineage is inconsistent")
        if decision.observation_id != expected_observation_id:
            raise DecisionOwnershipError("checkpoint decision observation identity is inconsistent")
        environment_observation = self._environment_adapter.current_observation(
            state.environment_state
        )
        if decision.observation != environment_observation:
            raise DecisionOwnershipError("checkpoint decision does not observe environment state")

        metrics = state.metrics
        environment_config = self._environment_adapter.manifest.config
        environment_kind = environment_config.get("environment_kind")
        metrics_mode = environment_config.get("metrics_mode")
        if metrics_mode != self._metrics_adapter.config.mode:
            raise DecisionOwnershipError("checkpoint environment/metrics modes differ")
        positive_float64_zero = b"\x00" * 8
        oracle_reward_must_match_bitwise = False
        if environment_kind == "switching_two_state":
            if metrics_mode != "switching_two_phase":
                raise DecisionOwnershipError("checkpoint switching metrics mode is invalid")
            phase_length = environment_config.get("phase_length")
            if isinstance(phase_length, bool) or not isinstance(phase_length, int):
                raise DecisionOwnershipError("checkpoint environment phase schedule is invalid")
            if phase_length <= 0:
                raise DecisionOwnershipError("checkpoint environment phase schedule is invalid")
            payoff_matrices = (
                np.asarray(environment_config.get("payoffs_a"), dtype=np.float32),
                np.asarray(environment_config.get("payoffs_b"), dtype=np.float32),
            )
            if any(payoff.shape != (2, 2) for payoff in payoff_matrices):
                raise DecisionOwnershipError("checkpoint environment payoff schedule is invalid")
            oracle_rewards = tuple(
                max(
                    float(payoff[0, 0]),
                    float(payoff[1, 1]),
                    0.5 * float(payoff[0, 1] + payoff[1, 0]),
                )
                for payoff in payoff_matrices
            )
            (
                expected_phase_counts,
                expected_current_phase,
                expected_phase_switches,
                expected_segment_events,
            ) = _switching_metric_schedule(accepted, phase_length)
            expected_oracle_reward = (
                expected_phase_counts[0] * oracle_rewards[0]
                + expected_phase_counts[1] * oracle_rewards[1]
            )
        elif environment_kind == "riverswim":
            if metrics_mode != "stationary":
                raise DecisionOwnershipError("checkpoint RiverSwim metrics mode is invalid")
            oracle_value = environment_config.get("oracle_average_reward")
            if isinstance(oracle_value, bool) or not isinstance(oracle_value, (int, float)):
                raise DecisionOwnershipError("checkpoint RiverSwim oracle is invalid")
            oracle_reward = float(oracle_value)
            if not math.isfinite(oracle_reward):
                raise DecisionOwnershipError("checkpoint RiverSwim oracle is invalid")
            expected_phase_counts = (accepted, 0)
            expected_current_phase = 0
            expected_phase_switches = 0
            expected_segment_events = accepted
            expected_oracle_reward = accepted * oracle_reward
            oracle_reward_must_match_bitwise = True
            stationary_exact_pairs = (
                (metrics.current_segment_reward, metrics.reward_sum),
                (metrics.current_segment_regret, metrics.regret_sum),
                (metrics.phase_reward_sums[0], metrics.reward_sum),
                (metrics.phase_regret_sums[0], metrics.regret_sum),
            )
            if (
                metrics.first_completed_segment_reward != (None, None)
                or metrics.latest_completed_segment_reward != (None, None)
                or any(
                    type(left) is not float
                    or type(right) is not float
                    or struct.pack(">d", left) != struct.pack(">d", right)
                    for left, right in stationary_exact_pairs
                )
            ):
                raise DecisionOwnershipError(
                    "checkpoint stationary reward/regret accumulators are not bit-exact"
                )
        else:
            raise DecisionOwnershipError("checkpoint environment kind is unsupported")

        for phase, count in enumerate(expected_phase_counts):
            if count != 0:
                continue
            reward_sum = metrics.phase_reward_sums[phase]
            regret_sum = metrics.phase_regret_sums[phase]
            if (
                type(reward_sum) is not float
                or struct.pack(">d", reward_sum) != positive_float64_zero
                or type(regret_sum) is not float
                or struct.pack(">d", regret_sum) != positive_float64_zero
                or metrics.first_completed_segment_reward[phase] is not None
                or metrics.latest_completed_segment_reward[phase] is not None
            ):
                raise DecisionOwnershipError(
                    "checkpoint zero-event phase metrics must be exact positive zero "
                    "with no completed-segment values"
                )
        oracle_reward_matches = (
            struct.pack(">d", metrics.oracle_reward_sum)
            == struct.pack(">d", expected_oracle_reward)
            if oracle_reward_must_match_bitwise
            else math.isclose(
                metrics.oracle_reward_sum,
                expected_oracle_reward,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
        if (
            metrics.phase_event_counts != expected_phase_counts
            or metrics.current_phase != expected_current_phase
            or metrics.phase_switches != expected_phase_switches
            or metrics.current_segment_events != expected_segment_events
            or not oracle_reward_matches
        ):
            raise DecisionOwnershipError("checkpoint metrics do not match the environment schedule")
        # Reward/regret history is preserved exactly by the recursively hashed
        # semantic bundle, but without a retained event log it is integrity-bound,
        # not independently authenticated or reconstructable. Validate only the
        # algebra and deterministic schedule fields derivable from current state.
        if not math.isclose(
            sum(metrics.phase_reward_sums),
            metrics.reward_sum,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            sum(metrics.phase_regret_sums),
            metrics.regret_sum,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DecisionOwnershipError("checkpoint metric phase totals are inconsistent")
        if not math.isclose(
            metrics.oracle_reward_sum - metrics.reward_sum,
            metrics.regret_sum,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise DecisionOwnershipError("checkpoint metric regret algebra is inconsistent")

    def checkpoint_barrier(
        self,
        state: ReferenceLifeState,
        publisher: Callable[
            [ReferenceLifeState, Callable[[], None]],
            _CheckpointPublishResult,
        ],
    ) -> tuple[ReferenceLifeState, _CheckpointPublishResult]:
        """Publish one immutable checkpoint generation under the runner lock."""

        if not callable(publisher):
            raise TypeError("checkpoint publisher must be callable")
        with self._lock:
            self._require_current_locked(state)
            self.validate_checkpoint_state(state)
            candidate = dataclasses.replace(
                state,
                commit_generation=state.commit_generation + 1,
                checkpoint_generation=state.checkpoint_generation + 1,
            )
            self.validate_checkpoint_state(candidate)
            installed = False

            def install_committed_candidate() -> None:
                nonlocal installed
                if installed or self._current_state is not state:
                    raise DecisionOwnershipError(
                        "checkpoint candidate installation is stale or repeated"
                    )
                self._current_state = candidate
                installed = True

            published = publisher(candidate, install_committed_candidate)
            if not installed:
                raise RuntimeError("checkpoint publisher did not install its committed state")
            return candidate, published

    def adopt_checkpoint_state(self, state: ReferenceLifeState) -> ReferenceLifeState:
        """Install one fully validated checkpoint into a fresh runner owner."""

        with self._lock:
            if self._current_state is not None:
                raise DecisionOwnershipError("checkpoint adoption requires a fresh runner")
            if state.checkpoint_generation <= 0:
                raise DecisionOwnershipError("restored checkpoint generation must be positive")
            self.validate_checkpoint_state(state)
            self._current_state = state
            return state

    def _environment_key(self, cursor: int) -> Array:
        root = jr.fold_in(
            jr.key(self.config.seed, impl=REFERENCE_LIFE_PRNG_IMPLEMENTATION),
            0x415349,
        )
        return jr.fold_in(root, cursor)

    def _validate_accepted_agent_update(
        self,
        update: ReferenceAgentUpdate,
        *,
        previous_agent_state: Any,
        transaction: Transaction,
    ) -> None:
        """Cross-check staged agent state before metrics or ledger acceptance."""

        if not isinstance(update, ReferenceAgentUpdate) or not update.accepted:
            raise DecisionOwnershipError("agent update is not an accepted reference-agent update")
        if update.state is previous_agent_state:
            raise DecisionOwnershipError("accepted agent update retained the stale input state")
        self._agent_adapter.validate_state(update.state)
        current_decision = self._agent_adapter.current_decision(update.state)
        if update.next_decision != current_decision:
            raise DecisionOwnershipError(
                "accepted update next_decision does not match its agent state"
            )
        if transaction.decision.decision_index >= MAX_DECISION_INDEX:
            raise DecisionOwnershipError(
                "selected reference life cannot accept the final uint64 decision"
            )
        expected_index = transaction.decision.decision_index + 1
        if (
            current_decision.lifecycle_id != transaction.decision.lifecycle_id
            or current_decision.decision_index != expected_index
            or current_decision.observation_id != transaction.next_decision_observation_id
            or current_decision.observation != transaction.next_decision_observation
        ):
            raise DecisionOwnershipError(
                "accepted agent state does not advance the retained outcome exactly once"
            )

    def init(self) -> ReferenceLifeState:
        """Initialize every state owner exactly once and arm decision zero."""

        with self._lock:
            if self._current_state is not None:
                raise DecisionOwnershipError("life runner is already initialized")
            root_key = jr.key(
                self.config.seed,
                impl=REFERENCE_LIFE_PRNG_IMPLEMENTATION,
            )
            agent_key, environment_key = jr.split(root_key, 2)
            environment_start = self.environment_adapter.init(environment_key)
            initial_observation_id = f"{self.config.lifecycle_id}:observation:0"
            agent_state = self.agent_adapter.init(
                agent_key,
                lifecycle_id=self.config.lifecycle_id,
            )
            agent_state, decision = self.agent_adapter.start(
                agent_state,
                observation_id=initial_observation_id,
                observation=environment_start.observation,
            )
            transaction_state = self._reducer.arm(self._reducer.init(), decision)
            state = ReferenceLifeState(
                schema=REFERENCE_LIFE_STATE_SCHEMA,
                config_sha256=self.config.config_sha256,
                lifecycle_id=self.config.lifecycle_id,
                phase=LifePhase.QUIESCENT,
                accepted_events=0,
                dispatch_attempts=0,
                executed_events=0,
                environment_rng_cursor=0,
                commit_generation=0,
                checkpoint_generation=0,
                agent_state=agent_state,
                environment_state=environment_start.state,
                transaction_state=transaction_state,
                dispatch_state=self._dispatch_adapter.init(),
                metrics=self._metrics_adapter.init(initial_phase=environment_start.regime_id),
                pending_outcome=None,
                halt=None,
                transcript_sha256=_initial_transcript(
                    self.config,
                    environment_start.observation,
                ),
            )
            self._current_state = state
            return state

    def _halt_before_execution_locked(
        self,
        previous: ReferenceLifeState,
        transaction_state: ReferenceTransactionState,
        dispatch_state: ExactDispatchState,
        *,
        reason: str,
    ) -> ReferenceLifeStep:
        if transaction_state.phase is not TransactionPhase.HALTED:
            transaction_state = self._reducer.halt(
                transaction_state,
                reason=reason,
            )
        candidate = dataclasses.replace(
            previous,
            phase=LifePhase.HALTED,
            transaction_state=transaction_state,
            dispatch_state=dispatch_state,
            commit_generation=previous.commit_generation + 1,
            halt=LifeHalt(
                stage=HaltStage.PRE_DISPATCH,
                recovery_mode=RecoveryMode.ABORT_ONLY,
                reason=reason,
            ),
        )
        return ReferenceLifeStep(
            state=self._commit_locked(previous, candidate),
            event=None,
            accepted=False,
            rejection_reason=reason,
        )

    def _halt_uncertain_locked(
        self,
        previous: ReferenceLifeState,
        transaction_state: ReferenceTransactionState,
        dispatch_state: ExactDispatchState,
        *,
        reason: str,
    ) -> ReferenceLifeStep:
        if transaction_state.phase is not TransactionPhase.HALTED:
            transaction_state = self._reducer.halt(
                transaction_state,
                reason=reason,
            )
        candidate = dataclasses.replace(
            previous,
            phase=LifePhase.HALTED,
            transaction_state=transaction_state,
            dispatch_state=dispatch_state,
            dispatch_attempts=previous.dispatch_attempts + 1,
            environment_rng_cursor=previous.environment_rng_cursor + 1,
            commit_generation=previous.commit_generation + 1,
            halt=LifeHalt(
                stage=HaltStage.DISPATCH_UNCERTAIN,
                recovery_mode=RecoveryMode.RECONCILE_ONLY,
                reason=reason,
            ),
        )
        return ReferenceLifeStep(
            state=self._commit_locked(previous, candidate),
            event=None,
            accepted=False,
            rejection_reason=reason,
        )

    def _halt_after_execution_locked(
        self,
        previous: ReferenceLifeState,
        transaction_state: ReferenceTransactionState,
        dispatch_state: ExactDispatchState,
        execution: ReferenceEnvironmentExecution,
        receipt: DispatchReceipt | None,
        *,
        transcript_sha256: str,
        reason: str,
    ) -> ReferenceLifeStep:
        """Commit a known execution while discarding all staged learning."""

        if transaction_state.phase is TransactionPhase.ISSUED and receipt is not None:
            transaction_state = self._reducer.halt_after_execution(
                transaction_state,
                receipt,
                reason=reason,
            )
        elif transaction_state.phase is not TransactionPhase.HALTED:
            transaction_state = self._reducer.halt(
                transaction_state,
                reason=reason,
            )
        candidate = dataclasses.replace(
            previous,
            phase=LifePhase.HALTED,
            environment_state=execution.state,
            transaction_state=transaction_state,
            dispatch_state=dispatch_state,
            dispatch_attempts=previous.dispatch_attempts + 1,
            executed_events=previous.executed_events + 1,
            environment_rng_cursor=previous.environment_rng_cursor + 1,
            commit_generation=previous.commit_generation + 1,
            pending_outcome=None,
            halt=LifeHalt(
                stage=HaltStage.POST_EXECUTION_DIVERGENCE,
                recovery_mode=RecoveryMode.RECONCILE_ONLY,
                reason=reason,
            ),
            transcript_sha256=transcript_sha256,
        )
        return ReferenceLifeStep(
            state=self._commit_locked(previous, candidate),
            event=None,
            accepted=False,
            rejection_reason=reason,
        )

    def _halt_pending_outcome_locked(
        self,
        previous: ReferenceLifeState,
        transaction_state: ReferenceTransactionState,
        dispatch_state: ExactDispatchState,
        execution: ReferenceEnvironmentExecution,
        transaction: Transaction,
        *,
        transcript_sha256: str,
        reason: str,
    ) -> ReferenceLifeStep:
        transaction_state, result = self._reducer.reject(
            transaction_state,
            reason=reason,
        )
        event = ReferenceLifeEvent(
            command=transaction.receipt.command,
            receipt=transaction.receipt,
            transaction=transaction,
            step_result=result,
            regime_id=execution.regime_id,
            oracle_reward=execution.oracle_reward,
            transcript_sha256=transcript_sha256,
            recovered=False,
        )
        candidate = dataclasses.replace(
            previous,
            phase=LifePhase.HALTED,
            environment_state=execution.state,
            transaction_state=transaction_state,
            dispatch_state=dispatch_state,
            dispatch_attempts=previous.dispatch_attempts + 1,
            executed_events=previous.executed_events + 1,
            environment_rng_cursor=previous.environment_rng_cursor + 1,
            commit_generation=previous.commit_generation + 1,
            pending_outcome=PendingOutcome(
                transaction=transaction,
                regime_id=execution.regime_id,
                oracle_reward=execution.oracle_reward,
            ),
            halt=LifeHalt(
                stage=HaltStage.OUTCOME_PENDING,
                recovery_mode=RecoveryMode.RETRY_OUTCOME,
                reason=reason,
            ),
            transcript_sha256=transcript_sha256,
        )
        return ReferenceLifeStep(
            state=self._commit_locked(previous, candidate),
            event=event,
            accepted=False,
            rejection_reason=reason,
        )

    def _emergency_halt_locked(
        self,
        previous: ReferenceLifeState,
        context: _PostIssueContext,
        *,
        cause: Exception,
    ) -> ReferenceLifeStep:
        """Last-resort CAS that does not invoke a possibly failing reducer method."""

        if self._current_state is not previous:
            raise cause
        reason = f"post-issue emergency halt after {type(cause).__name__}: {cause}"
        deepest = context.transaction_state
        try:
            if deepest.phase is TransactionPhase.HALTED:
                halted_transaction = deepest
            elif deepest.phase is TransactionPhase.ISSUED and context.receipt is not None:
                halted_transaction = dataclasses.replace(
                    deepest,
                    phase=TransactionPhase.HALTED,
                    receipt=context.receipt,
                    halt_reason=reason,
                )
            else:
                halted_transaction = dataclasses.replace(
                    deepest,
                    phase=TransactionPhase.HALTED,
                    halt_reason=reason,
                )
        except Exception:
            issued = context.issued_state
            if issued is None:
                raise cause
            halted_transaction = dataclasses.replace(
                issued,
                phase=TransactionPhase.HALTED,
                halt_reason=reason,
            )

        execution = context.execution
        execution_known = execution is not None
        candidate = dataclasses.replace(
            previous,
            phase=LifePhase.HALTED,
            environment_state=(
                execution.state if execution is not None else previous.environment_state
            ),
            transaction_state=halted_transaction,
            dispatch_state=context.dispatch_state,
            dispatch_attempts=previous.dispatch_attempts + 1,
            executed_events=previous.executed_events + int(execution_known),
            environment_rng_cursor=previous.environment_rng_cursor + 1,
            commit_generation=previous.commit_generation + 1,
            pending_outcome=None,
            halt=LifeHalt(
                stage=(
                    HaltStage.POST_EXECUTION_DIVERGENCE
                    if execution_known
                    else HaltStage.DISPATCH_UNCERTAIN
                ),
                recovery_mode=RecoveryMode.RECONCILE_ONLY,
                reason=reason,
            ),
            transcript_sha256=(
                context.transcript_sha256
                if execution_known and context.transcript_sha256 is not None
                else previous.transcript_sha256
            ),
        )
        if self._current_state is not previous:
            raise cause
        self._current_state = candidate
        return ReferenceLifeStep(
            state=candidate,
            event=None,
            accepted=False,
            rejection_reason=reason,
        )

    def step(self, state: ReferenceLifeState) -> ReferenceLifeStep:
        """Run one event under a post-issue emergency fault boundary."""

        context = _PostIssueContext(
            transaction_state=state.transaction_state,
            dispatch_state=state.dispatch_state,
            transcript_sha256=state.transcript_sha256,
        )
        with self._lock:
            try:
                return self._step_locked(state, context)
            except Exception as exc:
                if self._current_state is not state or context.command is None:
                    raise
                return self._emergency_halt_locked(
                    state,
                    context,
                    cause=exc,
                )

    def _step_locked(
        self,
        state: ReferenceLifeState,
        context: _PostIssueContext,
    ) -> ReferenceLifeStep:
        """Execute, learn from, measure, and commit one synchronous event."""

        # The lock deliberately spans validation through executor invocation
        # and aggregate commit. Two callers holding the same immutable snapshot
        # therefore cannot both dispatch before one loses a final CAS.
        with self._lock:
            self._require_current_locked(state)
            if state.phase is LifePhase.COMPLETED:
                raise DecisionOwnershipError("life is completed; no further dispatch is valid")
            if state.phase is LifePhase.ABORTED:
                raise DecisionOwnershipError("life is aborted; no further dispatch is valid")
            if state.phase is LifePhase.HALTED:
                raise DecisionOwnershipError("life is halted and requires explicit recovery/abort")
            if state.accepted_events >= self.config.max_accepted_events:
                raise DecisionOwnershipError("configured life horizon is completed")
            decision = state.transaction_state.decision
            if decision is None:
                raise DecisionOwnershipError("quiescent life lacks an armed decision")

            transaction_state = state.transaction_state
            dispatch_state = state.dispatch_state
            settled_agent_state = state.agent_state
            try:
                candidate_transaction_state, authorization = self._reducer.authorize(
                    transaction_state,
                    decision,
                    authorized_action=None,
                    authority_id=self._dispatch_adapter.config.authority_id,
                    policy_version=self._dispatch_adapter.config.policy_version,
                    authorization_id=f"{decision.decision_id}:authorization",
                )
                self._reducer.validate_state(candidate_transaction_state)
                if (
                    candidate_transaction_state.phase
                    not in (TransactionPhase.AUTHORIZED, TransactionPhase.HALTED)
                    or candidate_transaction_state.authorization != authorization
                ):
                    raise DecisionOwnershipError(
                        "authorization returned an invalid transaction prefix"
                    )
                transaction_state = candidate_transaction_state
                candidate_dispatch_state = self._dispatch_adapter.authorized(dispatch_state)
                self._dispatch_adapter.validate_state(candidate_dispatch_state)
                dispatch_state = candidate_dispatch_state
                context.dispatch_state = dispatch_state
                if authorization.status is AuthorizationStatus.VETOED:
                    reason = transaction_state.halt_reason or "dispatch authorization vetoed"
                    return self._halt_before_execution_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        reason=reason,
                    )
                settled_agent_state, rebinding_applied = self.agent_adapter.settle_dispatch(
                    state.agent_state,
                    authorization,
                )
                candidate_transaction_state, dispatch = self._reducer.settle_dispatch(
                    transaction_state,
                    authorization,
                    rebinding_applied=rebinding_applied,
                    settlement_id=f"{decision.decision_id}:settlement",
                )
                self._reducer.validate_state(candidate_transaction_state)
                if dispatch is None:
                    if candidate_transaction_state.phase is not TransactionPhase.HALTED:
                        raise DecisionOwnershipError(
                            "empty settlement did not return a halted transaction"
                        )
                elif (
                    candidate_transaction_state.phase is not TransactionPhase.SETTLED
                    or candidate_transaction_state.dispatch != dispatch
                ):
                    raise DecisionOwnershipError(
                        "settlement returned an invalid transaction prefix"
                    )
                transaction_state = candidate_transaction_state
                if dispatch is None:
                    reason = transaction_state.halt_reason or "dispatch settlement halted"
                    return self._halt_before_execution_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        reason=reason,
                    )
                candidate_transaction_state, command = self._reducer.issue_dispatch(
                    transaction_state,
                    dispatch,
                    command_id=f"{decision.decision_id}:command",
                    executor_id=self._dispatch_adapter.config.executor_id,
                    executor_epoch=self._dispatch_adapter.config.executor_epoch,
                )
                self._reducer.validate_state(candidate_transaction_state)
                if (
                    candidate_transaction_state.phase is not TransactionPhase.ISSUED
                    or candidate_transaction_state.command != command
                ):
                    raise DecisionOwnershipError(
                        "command issuance returned an invalid transaction prefix"
                    )
                transaction_state = candidate_transaction_state
                context.transaction_state = transaction_state
                context.issued_state = transaction_state
                context.command = command
                candidate_dispatch_state = self._dispatch_adapter.issued(
                    dispatch_state,
                    command,
                )
                self._dispatch_adapter.validate_state(candidate_dispatch_state)
                dispatch_state = candidate_dispatch_state
                context.dispatch_state = dispatch_state
            except Exception as exc:
                if context.command is not None:
                    raise
                return self._halt_before_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    reason=f"pre-dispatch failure: {exc}",
                )

            environment_key = self._environment_key(state.environment_rng_cursor)
            try:
                execution = self.environment_adapter.execute(
                    state.environment_state,
                    command,
                    key=environment_key,
                )
            except Exception as exc:
                return self._halt_uncertain_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    reason=f"executor completion is uncertain: {exc}",
                )

            if not isinstance(execution, ReferenceEnvironmentExecution):
                return self._halt_uncertain_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    reason="executor returned no trustworthy execution result",
                )
            if execution.command != command:
                return self._halt_uncertain_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    reason="executor result belongs to another issued command",
                )
            try:
                self.environment_adapter.validate_execution(
                    state.environment_state,
                    command,
                    execution,
                    key=environment_key,
                )
            except Exception as exc:
                return self._halt_uncertain_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    reason=f"executor result failed semantic validation: {exc}",
                )
            context.execution = execution
            next_executed_events = state.executed_events + 1
            next_observation_id = f"{self.config.lifecycle_id}:observation:{next_executed_events}"
            receipt: DispatchReceipt | None = None
            transcript_sha256 = state.transcript_sha256
            try:
                receipt = DispatchReceipt(
                    command=command,
                    applied_action=execution.applied_action,
                    receipt_id=f"{decision.decision_id}:receipt",
                )
                context.receipt = receipt
                transcript_sha256 = _advance_transcript(
                    state.transcript_sha256,
                    execution=execution,
                    receipt=receipt,
                    next_observation_id=next_observation_id,
                )
                context.transcript_sha256 = transcript_sha256
            except Exception as exc:
                return self._halt_after_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    execution,
                    receipt,
                    transcript_sha256=transcript_sha256,
                    reason=f"post-execution receipt staging failed: {exc}",
                )
            try:
                candidate_transaction_state, recorded_receipt = self._reducer.record_dispatch(
                    transaction_state,
                    receipt,
                )
                self._reducer.validate_state(candidate_transaction_state)
                if (
                    candidate_transaction_state.phase
                    not in (TransactionPhase.DISPATCHED, TransactionPhase.HALTED)
                    or candidate_transaction_state.receipt != recorded_receipt
                ):
                    raise DecisionOwnershipError(
                        "receipt recording returned an invalid transaction prefix"
                    )
                transaction_state = candidate_transaction_state
                receipt = recorded_receipt
                context.transaction_state = transaction_state
                context.receipt = receipt
                candidate_dispatch_state = self._dispatch_adapter.receipted(
                    dispatch_state,
                    receipt,
                )
                self._dispatch_adapter.validate_state(candidate_dispatch_state)
                dispatch_state = candidate_dispatch_state
                context.dispatch_state = dispatch_state
            except Exception as exc:
                return self._halt_after_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    execution,
                    receipt,
                    transcript_sha256=transcript_sha256,
                    reason=f"post-execution receipt recording failed: {exc}",
                )
            if transaction_state.phase is TransactionPhase.HALTED:
                reason = transaction_state.halt_reason or (
                    "executor applied action differs from issued command"
                )
                return self._halt_after_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    execution,
                    receipt,
                    transcript_sha256=transcript_sha256,
                    reason=reason,
                )

            try:
                candidate_transaction_state, transaction = self._reducer.record_outcome(
                    transaction_state,
                    receipt,
                    reward=execution.reward,
                    discount=execution.discount,
                    terminated=execution.terminated,
                    truncated=execution.truncated,
                    autoreset=execution.autoreset,
                    bootstrap_observation_id=next_observation_id,
                    bootstrap_observation=execution.next_observation,
                    next_decision_observation_id=next_observation_id,
                    next_decision_observation=execution.next_observation,
                )
                self._reducer.validate_state(candidate_transaction_state)
                if (
                    candidate_transaction_state.phase is not TransactionPhase.OUTCOME
                    or candidate_transaction_state.transaction != transaction
                ):
                    raise DecisionOwnershipError(
                        "outcome recording returned an invalid transaction prefix"
                    )
                transaction_state = candidate_transaction_state
                context.transaction_state = transaction_state
            except Exception as exc:
                return self._halt_after_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    execution,
                    receipt,
                    transcript_sha256=transcript_sha256,
                    reason=f"post-execution outcome recording failed: {exc}",
                )

            update: ReferenceAgentUpdate
            try:
                update = self.agent_adapter.apply_outcome(
                    settled_agent_state,
                    transaction,
                )
            except Exception as exc:
                update = ReferenceAgentUpdate(
                    state=settled_agent_state,
                    next_decision=None,
                    accepted=False,
                    parameters_changed=False,
                    rejection_reason=f"agent update raised: {exc}",
                )
            if not isinstance(update, ReferenceAgentUpdate):
                update = ReferenceAgentUpdate(
                    state=settled_agent_state,
                    next_decision=None,
                    accepted=False,
                    parameters_changed=False,
                    rejection_reason="agent returned a malformed update",
                )
            if not update.accepted:
                reason = update.rejection_reason or "agent rejected the retained outcome"
                try:
                    return self._halt_pending_outcome_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        transaction,
                        transcript_sha256=transcript_sha256,
                        reason=reason,
                    )
                except Exception as exc:
                    return self._halt_after_execution_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        receipt,
                        transcript_sha256=transcript_sha256,
                        reason=f"post-execution rejection staging failed: {exc}",
                    )

            try:
                self._validate_accepted_agent_update(
                    update,
                    previous_agent_state=settled_agent_state,
                    transaction=transaction,
                )
            except Exception as exc:
                reason = f"accepted agent update failed validation: {exc}"
                try:
                    return self._halt_pending_outcome_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        transaction,
                        transcript_sha256=transcript_sha256,
                        reason=reason,
                    )
                except Exception as rejection_exc:
                    return self._halt_after_execution_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        receipt,
                        transcript_sha256=transcript_sha256,
                        reason=(
                            "post-execution invalid-agent rejection staging failed: "
                            f"{rejection_exc}"
                        ),
                    )

            try:
                metrics = self._metrics_adapter.observe(
                    state.metrics,
                    phase=execution.regime_id,
                    reward=transaction.reward,
                    oracle_reward=execution.oracle_reward,
                    parameters_changed=update.parameters_changed,
                )
            except Exception as exc:
                reason = f"metric update rejected retained outcome: {exc}"
                try:
                    return self._halt_pending_outcome_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        transaction,
                        transcript_sha256=transcript_sha256,
                        reason=reason,
                    )
                except Exception as rejection_exc:
                    return self._halt_after_execution_locked(
                        state,
                        transaction_state,
                        dispatch_state,
                        execution,
                        receipt,
                        transcript_sha256=transcript_sha256,
                        reason=(f"post-execution metric rejection staging failed: {rejection_exc}"),
                    )

            try:
                accepted_transaction_state, result = self._reducer.accept(
                    transaction_state,
                    next_decision=update.next_decision,
                    parameters_changed=update.parameters_changed,
                )
                self._reducer.validate_state(accepted_transaction_state)
                if accepted_transaction_state.phase not in (
                    TransactionPhase.ARMED,
                    TransactionPhase.READY,
                    TransactionPhase.EXHAUSTED,
                ):
                    raise DecisionOwnershipError("acceptance returned an invalid transaction phase")
            except Exception as exc:
                return self._halt_after_execution_locked(
                    state,
                    transaction_state,
                    dispatch_state,
                    execution,
                    receipt,
                    transcript_sha256=transcript_sha256,
                    reason=f"post-execution transaction acceptance failed: {exc}",
                )
            next_accepted_events = state.accepted_events + 1
            phase = (
                LifePhase.COMPLETED
                if next_accepted_events == self.config.max_accepted_events
                else LifePhase.QUIESCENT
            )
            candidate = dataclasses.replace(
                state,
                phase=phase,
                accepted_events=next_accepted_events,
                dispatch_attempts=state.dispatch_attempts + 1,
                executed_events=next_executed_events,
                environment_rng_cursor=state.environment_rng_cursor + 1,
                commit_generation=state.commit_generation + 1,
                agent_state=update.state,
                environment_state=execution.state,
                transaction_state=accepted_transaction_state,
                dispatch_state=dispatch_state,
                metrics=metrics,
                transcript_sha256=transcript_sha256,
            )
            event = ReferenceLifeEvent(
                command=command,
                receipt=receipt,
                transaction=transaction,
                step_result=result,
                regime_id=execution.regime_id,
                oracle_reward=execution.oracle_reward,
                transcript_sha256=transcript_sha256,
                recovered=False,
            )
            return ReferenceLifeStep(
                state=self._commit_locked(state, candidate),
                event=event,
                accepted=True,
                rejection_reason=None,
            )

    def _retain_pending_recovery_locked(
        self,
        state: ReferenceLifeState,
        pending: PendingOutcome,
        *,
        reason: str,
    ) -> ReferenceLifeStep:
        """Record one failed staging retry without consuming its retained outcome."""

        if state.halt is None:
            raise DecisionOwnershipError("pending-outcome recovery lacks halt metadata")
        transaction = pending.transaction
        halt = dataclasses.replace(
            state.halt,
            reason=reason,
            recovery_attempts=state.halt.recovery_attempts + 1,
        )
        candidate = dataclasses.replace(
            state,
            commit_generation=state.commit_generation + 1,
            halt=halt,
        )
        result = StepResult(
            transaction=transaction,
            next_decision=None,
            transaction_accepted=False,
            parameters_changed=False,
            rejection_reason=reason,
        )
        event = ReferenceLifeEvent(
            command=transaction.receipt.command,
            receipt=transaction.receipt,
            transaction=transaction,
            step_result=result,
            regime_id=pending.regime_id,
            oracle_reward=pending.oracle_reward,
            transcript_sha256=state.transcript_sha256,
            recovered=True,
        )
        return ReferenceLifeStep(
            state=self._commit_locked(state, candidate),
            event=event,
            accepted=False,
            rejection_reason=reason,
        )

    def recover_pending_outcome(
        self,
        state: ReferenceLifeState,
    ) -> ReferenceLifeStep:
        """Retry only agent/metric staging for one already-executed event."""

        with self._lock:
            self._require_current_locked(state)
            if (
                state.phase is not LifePhase.HALTED
                or state.halt is None
                or state.halt.stage is not HaltStage.OUTCOME_PENDING
                or state.halt.recovery_mode is not RecoveryMode.RETRY_OUTCOME
                or state.pending_outcome is None
            ):
                raise DecisionOwnershipError(
                    "recovery requires the current complete outcome-pending halt"
                )
            pending = state.pending_outcome
            transaction = pending.transaction
            command = transaction.receipt.command
            receipt = transaction.receipt
            try:
                update = self.agent_adapter.apply_outcome(
                    state.agent_state,
                    transaction,
                )
            except Exception as exc:
                update = ReferenceAgentUpdate(
                    state=state.agent_state,
                    next_decision=None,
                    accepted=False,
                    parameters_changed=False,
                    rejection_reason=f"agent recovery update raised: {exc}",
                )
            if not isinstance(update, ReferenceAgentUpdate):
                update = ReferenceAgentUpdate(
                    state=state.agent_state,
                    next_decision=None,
                    accepted=False,
                    parameters_changed=False,
                    rejection_reason="agent recovery returned a malformed update",
                )
            if not update.accepted:
                reason = update.rejection_reason or "agent recovery rejected outcome"
                return self._retain_pending_recovery_locked(
                    state,
                    pending,
                    reason=reason,
                )

            try:
                self._validate_accepted_agent_update(
                    update,
                    previous_agent_state=state.agent_state,
                    transaction=transaction,
                )
            except Exception as exc:
                return self._retain_pending_recovery_locked(
                    state,
                    pending,
                    reason=f"accepted agent update failed recovery validation: {exc}",
                )

            try:
                metrics = self._metrics_adapter.observe(
                    state.metrics,
                    phase=pending.regime_id,
                    reward=transaction.reward,
                    oracle_reward=pending.oracle_reward,
                    parameters_changed=update.parameters_changed,
                )
            except (DecisionOwnershipError, RuntimeError, TypeError, ValueError) as exc:
                reason = f"metric recovery rejected retained outcome: {exc}"
                return self._retain_pending_recovery_locked(
                    state,
                    pending,
                    reason=reason,
                )

            transaction_state, result = self._reducer.recover_accept(
                state.transaction_state,
                next_decision=update.next_decision,
                parameters_changed=update.parameters_changed,
            )
            next_accepted_events = state.accepted_events + 1
            phase = (
                LifePhase.COMPLETED
                if next_accepted_events == self.config.max_accepted_events
                else LifePhase.QUIESCENT
            )
            candidate = dataclasses.replace(
                state,
                phase=phase,
                accepted_events=next_accepted_events,
                commit_generation=state.commit_generation + 1,
                agent_state=update.state,
                transaction_state=transaction_state,
                metrics=metrics,
                pending_outcome=None,
                halt=None,
            )
            event = ReferenceLifeEvent(
                command=command,
                receipt=receipt,
                transaction=transaction,
                step_result=result,
                regime_id=pending.regime_id,
                oracle_reward=pending.oracle_reward,
                transcript_sha256=state.transcript_sha256,
                recovered=True,
            )
            return ReferenceLifeStep(
                state=self._commit_locked(state, candidate),
                event=event,
                accepted=True,
                rejection_reason=None,
            )

    def abort(
        self,
        state: ReferenceLifeState,
        *,
        reason: str,
    ) -> ReferenceLifeState:
        """Terminate the current life without relabelling an unconsumed event."""

        if type(reason) is not str or not reason.strip():
            raise ValueError("abort reason must be nonempty")
        with self._lock:
            self._require_current_locked(state)
            if state.phase is LifePhase.ABORTED:
                raise DecisionOwnershipError("life is already aborted")
            if state.phase is LifePhase.COMPLETED:
                raise DecisionOwnershipError("completed life cannot be aborted")
            stage = state.halt.stage if state.halt is not None else HaltStage.PRE_DISPATCH
            candidate = dataclasses.replace(
                state,
                phase=LifePhase.ABORTED,
                commit_generation=state.commit_generation + 1,
                halt=LifeHalt(
                    stage=stage,
                    recovery_mode=RecoveryMode.ABORT_ONLY,
                    reason=reason,
                    recovery_attempts=(0 if state.halt is None else state.halt.recovery_attempts),
                ),
            )
            return self._commit_locked(state, candidate)

    def run_to_completion(self, state: ReferenceLifeState) -> ReferenceLifeRun:
        """Drive the configured small development horizon until complete/halted."""

        events: list[ReferenceLifeEvent] = []
        current = state
        while current.phase is LifePhase.QUIESCENT:
            step = self.step(current)
            if step.event is not None:
                events.append(step.event)
            current = step.state
        return ReferenceLifeRun(state=current, events=tuple(events))


def build_prototype_switching_life(
    *,
    agent_config: PrototypeAgentConfig,
    environment_config: SwitchingTwoStateConfig,
    lifecycle_id: str,
    seed: int,
    max_accepted_events: int,
    dispatch_config: ExactDispatchConfig | None = None,
    metrics_config: ReferenceLifeMetricsConfig | None = None,
) -> ReferenceLifeRunner:
    """Build, but do not initialize, the canonical primitive switching slice."""

    _require_prototype_lifecycle_id(lifecycle_id)
    agent_adapter = PrototypeReferenceAdapter.from_config(agent_config)
    effective_dispatch = ExactDispatchConfig() if dispatch_config is None else dispatch_config
    effective_metrics = (
        ReferenceLifeMetricsConfig(mode="switching_two_phase")
        if metrics_config is None
        else metrics_config
    )
    if effective_metrics.mode != "switching_two_phase":
        raise ValueError("Switching life requires switching_two_phase metrics")
    environment_adapter = SwitchingTwoStateReferenceEnvironment(
        environment_config,
        observation_spec=agent_adapter.manifest.observation_spec,
        action_spec=agent_adapter.manifest.action_spec,
        executor_id=effective_dispatch.executor_id,
        executor_epoch=effective_dispatch.executor_epoch,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=agent_adapter,
        environment_adapter=environment_adapter,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=max_accepted_events,
        dispatch_config=effective_dispatch,
        metrics_config=effective_metrics,
    )


def build_prototype_riverswim_life(
    *,
    agent_config: PrototypeAgentConfig,
    environment_config: RiverSwimConfig,
    lifecycle_id: str,
    seed: int,
    max_accepted_events: int,
    dispatch_config: ExactDispatchConfig | None = None,
    metrics_config: ReferenceLifeMetricsConfig | None = None,
) -> ReferenceLifeRunner:
    """Build, but do not initialize, the canonical stochastic RiverSwim slice."""

    _require_prototype_lifecycle_id(lifecycle_id)
    agent_adapter = PrototypeReferenceAdapter.from_config(agent_config)
    effective_dispatch = (
        ExactDispatchConfig(
            executor_id="asi.riverswim.executor",
            executor_epoch="asi.riverswim.executor_epoch.1",
        )
        if dispatch_config is None
        else dispatch_config
    )
    effective_metrics = (
        ReferenceLifeMetricsConfig(mode="stationary")
        if metrics_config is None
        else metrics_config
    )
    if effective_metrics.mode != "stationary":
        raise ValueError("RiverSwim life requires stationary metrics")
    environment_adapter = RiverSwimReferenceEnvironment(
        environment_config,
        observation_spec=agent_adapter.manifest.observation_spec,
        action_spec=agent_adapter.manifest.action_spec,
        executor_id=effective_dispatch.executor_id,
        executor_epoch=effective_dispatch.executor_epoch,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=agent_adapter,
        environment_adapter=environment_adapter,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=max_accepted_events,
        dispatch_config=effective_dispatch,
        metrics_config=effective_metrics,
    )


__all__ = [
    "EXACT_DISPATCH_SCHEMA",
    "EXACT_DISPATCH_STATE_SCHEMA",
    "REFERENCE_ENVIRONMENT_MANIFEST_SCHEMA",
    "REFERENCE_LIFE_CONFIG_SCHEMA",
    "REFERENCE_LIFE_METRICS_SCHEMA",
    "REFERENCE_LIFE_STATE_SCHEMA",
    "RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID",
    "RIVERSWIM_ENVIRONMENT_STATE_SCHEMA",
    "RIVERSWIM_REFERENCE_MAX_STATES",
    "ExactDispatchAdapter",
    "ExactDispatchConfig",
    "ExactDispatchState",
    "HaltStage",
    "LifeHalt",
    "LifePhase",
    "PendingOutcome",
    "RecoveryMode",
    "ReferenceAgentAdapter",
    "ReferenceEnvironmentAdapter",
    "ReferenceEnvironmentExecution",
    "ReferenceEnvironmentManifest",
    "ReferenceEnvironmentStart",
    "ReferenceLifeConfig",
    "ReferenceLifeEvent",
    "ReferenceLifeMetricsAdapter",
    "ReferenceLifeMetricsConfig",
    "ReferenceLifeMetricsState",
    "ReferenceLifeRun",
    "ReferenceLifeRunner",
    "ReferenceLifeState",
    "ReferenceLifeStep",
    "RiverSwimReferenceEnvironment",
    "SwitchingEnvironmentExecution",
    "SwitchingEnvironmentStart",
    "SwitchingTwoStateReferenceEnvironment",
    "build_prototype_riverswim_life",
    "build_prototype_switching_life",
]
