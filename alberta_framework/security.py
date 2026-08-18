"""Security-gym integration contracts for downstream active-defense agents.

This module is intentionally small and dependency-free. It gives ``rlsecd`` and
``security-gym`` a stable framework-side contract for discrete actions, reward
components, feature schemas, rollout records, and throughput timing without
requiring either sibling repository at import time.
"""

from __future__ import annotations

import dataclasses
import json
import math
import operator
import time
from collections.abc import Mapping, Sequence
from enum import IntEnum
from fractions import Fraction
from types import MappingProxyType
from typing import Any, SupportsIndex, cast

import numpy as np

from alberta_framework.core._float32_scalars import validated_float32_scalar


def _require_rfc_json_mapping(payload: Mapping[str, Any], *, name: str) -> None:
    """Raise ``ValueError`` unless ``payload`` serializes as RFC-compliant JSON."""
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be RFC-compliant JSON") from exc


_INT32_MAX: int = 2**31 - 1

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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if type(payload) not in (dict, MappingProxyType):
        raise ValueError(f"{name} must be a mapping")
    try:
        values = dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} keys must be exact strings")
    return values


def _copy_json_value(value: object, *, name: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be RFC-compliant JSON with finite numbers")
        return value
    if type(value) in (list, tuple):
        return [_copy_json_value(item, name=name) for item in cast(Sequence[object], value)]
    if type(value) in (dict, MappingProxyType):
        payload = _copy_mapping(value, name=name)
        return {key: _copy_json_value(item, name=name) for key, item in payload.items()}
    raise ValueError(f"{name} must contain exact JSON values")


def _copy_json_mapping(payload: object, *, name: str) -> dict[str, Any]:
    copied = _copy_mapping(payload, name=name)
    return {key: _copy_json_value(value, name=name) for key, value in copied.items()}


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_finite_real(name: str, value: object) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a real number")
    # validated_float32_scalar ensures finite and canonical float32 domain;
    # use permissive domain to allow any finite real then check isfinite via validator.
    try:
        return validated_float32_scalar(name, value)
    except ValueError:
        raise ValueError(f"{name} must be a finite real number") from None


class SecurityAction(IntEnum):
    """Stable six-action active-defense vocabulary.

    The integer values are the action-head indices expected by SARSA/Horde and
    actor-critic agents. Downstream environment wrappers should translate these
    semantic actions to their local actuation APIs without changing the values.
    """

    PASS = 0
    ALERT = 1
    THROTTLE = 2
    BLOCK = 3
    UNBLOCK = 4
    ISOLATE = 5


SECURITY_ACTION_NAMES: tuple[str, ...] = tuple(action.name.lower() for action in SecurityAction)
SECURITY_GYM_ACTION_NAMES: tuple[str, ...] = (
    "pass",
    "alert",
    "throttle",
    "block_source",
    "unblock",
    "isolate",
)
N_SECURITY_ACTIONS = len(SecurityAction)

_ACTION_ALIASES = {
    "block_source": SecurityAction.BLOCK,
    "block": SecurityAction.BLOCK,
}

_SECURITY_GYM_ATTACK_REWARDS = {
    SecurityAction.PASS: -0.5,
    SecurityAction.ALERT: 0.5,
    SecurityAction.THROTTLE: 0.75,
    SecurityAction.BLOCK: 1.0,
    SecurityAction.UNBLOCK: -0.5,
    SecurityAction.ISOLATE: 0.25,
}

_SECURITY_GYM_BENIGN_REWARDS = {
    SecurityAction.PASS: 0.0,
    SecurityAction.ALERT: -0.3,
    SecurityAction.THROTTLE: -0.5,
    SecurityAction.BLOCK: -1.0,
    SecurityAction.UNBLOCK: 0.0,
    SecurityAction.ISOLATE: -2.0,
}


@dataclasses.dataclass(frozen=True)
class SecurityRewardWeights:
    """Linear reward weights for active-defense rollouts.

    Positive components reward correct protection and service restoration.
    Negative components penalize operational disruption and missed threats. The
    defaults are conservative integration-test baselines; production
    experiments should record the exact weights in rollout metadata.
    """

    threat_blocked: float = 1.0
    false_positive: float = -0.5
    service_disruption: float = -0.2
    alert_cost: float = -0.05
    latency_cost: float = -0.1
    compromise_cost: float = -1.0
    recovery: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "threat_blocked",
            "false_positive",
            "service_disruption",
            "alert_cost",
            "latency_cost",
            "compromise_cost",
            "recovery",
        ):
            if type(getattr(self, name)) is bool:
                raise ValueError(f"{name} must be a finite real number")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable weight mapping."""
        payload = dataclasses.asdict(self)
        _require_rfc_json_mapping(payload, name="security reward weights")
        return payload


def security_reward(
    components: Mapping[str, float],
    weights: SecurityRewardWeights | Mapping[str, float] | None = None,
) -> float:
    """Compute scalar reward from named security outcome components.

    Unknown component names are ignored so sibling environments can log richer
    diagnostics while keeping the learning reward contract stable.
    """
    component_map = _copy_mapping(components, name="security reward components")
    if weights is None:
        weight_map = SecurityRewardWeights().to_dict()
    elif type(weights) is SecurityRewardWeights:
        weight_map = weights.to_dict()
    else:
        weight_map = _copy_mapping(weights, name="security reward weights")
    terms = []
    for name, raw_weight in weight_map.items():
        weight = _require_finite_real(f"security reward weight {name}", raw_weight)
        component = _require_finite_real(
            f"security reward component {name}", component_map.get(name, 0.0)
        )
        terms.append(component * weight)
    reward = math.fsum(terms)
    if not math.isfinite(reward):
        raise ValueError("security reward must be finite")
    return reward


@dataclasses.dataclass(frozen=True)
class SecurityFeatureSchema:
    """Versioned flat feature schema for rlsecd/security-gym observations."""

    names: tuple[str, ...]
    version: str = "security-gym-v1"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if type(self.names) is not tuple:
            raise ValueError("feature schema names must be an exact tuple")
        if not self.names:
            raise ValueError("feature schema must contain at least one feature")
        for idx, name in enumerate(self.names):
            _require_exact_str(f"names[{idx}]", name)
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        _require_exact_str("version", self.version)
        _require_exact_str("dtype", self.dtype)

    @property
    def feature_dim(self) -> int:
        """Number of features in this schema."""
        return len(self.names)

    def validate_observation(self, observation: Sequence[float]) -> None:
        """Raise ``ValueError`` if an observation does not match this schema."""
        if len(observation) != self.feature_dim:
            raise ValueError(
                f"observation length {len(observation)} does not match "
                f"schema feature_dim {self.feature_dim}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable schema mapping."""
        payload = {
            "version": self.version,
            "dtype": self.dtype,
            "names": list(self.names),
            "feature_dim": self.feature_dim,
        }
        _require_rfc_json_mapping(payload, name="security feature schema")
        return payload

    @classmethod
    def from_dict(cls, data: object) -> SecurityFeatureSchema:
        """Reconstruct a schema from ``to_dict`` output."""
        payload = _copy_mapping(data, name="security feature schema")
        raw_names = payload.get("names")
        if type(raw_names) not in (list, tuple):
            raise ValueError("security feature schema names must be a list or tuple")
        names = tuple(
            _require_exact_str(f"names[{i}]", n)
            for i, n in enumerate(cast(Sequence[Any], raw_names))
        )
        version = payload.get("version", "security-gym-v1")
        dtype = payload.get("dtype", "float32")
        _require_exact_str("version", version)
        _require_exact_str("dtype", dtype)
        feature_dim = payload.get("feature_dim")
        if feature_dim is not None and (type(feature_dim) is not int or feature_dim != len(names)):
            raise ValueError("feature_dim must match the number of feature names")
        return cls(names=names, version=version, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class SecurityRolloutStep:
    """Serializable transition record for reproducible active-defense rollouts."""

    state: tuple[float, ...]
    action: SecurityAction
    reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool = False
    policy_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable transition mapping."""
        payload = {
            "state": list(self.state),
            "action": int(self.action),
            "action_name": self.action.name.lower(),
            "reward": self.reward,
            "next_state": list(self.next_state),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "policy_metadata": _copy_json_mapping(
                self.policy_metadata, name="policy_metadata"
            ),
        }
        _require_rfc_json_mapping(payload, name="security rollout step")
        return payload

    @classmethod
    def from_dict(cls, data: object) -> SecurityRolloutStep:
        """Reconstruct a rollout step from ``to_dict`` output."""
        payload = _copy_mapping(data, name="security rollout step")
        raw_state = payload.get("state")
        raw_next = payload.get("next_state")
        if type(raw_state) not in (list, tuple) or type(raw_next) not in (list, tuple):
            raise ValueError("security rollout step state must be a list or tuple")
        state = tuple(
            _require_finite_real(f"state[{i}]", v)
            for i, v in enumerate(cast(Sequence[Any], raw_state))
        )
        next_state = tuple(
            _require_finite_real(f"next_state[{i}]", v)
            for i, v in enumerate(cast(Sequence[Any], raw_next))
        )
        action = coerce_security_action(payload.get("action"))
        action_name = payload.get("action_name")
        if action_name is not None and (
            type(action_name) is not str or action_name != action.name.lower()
        ):
            raise ValueError("action_name must match action")
        reward = _require_finite_real("reward", payload.get("reward"))
        terminated = payload.get("terminated")
        truncated = payload.get("truncated", False)
        if type(terminated) is not bool:
            raise ValueError("terminated must be an exact bool")
        if type(truncated) is not bool:
            raise ValueError("truncated must be an exact bool")
        raw_meta = payload.get("policy_metadata", {})
        meta = _copy_json_mapping(raw_meta, name="policy_metadata")
        return cls(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
            policy_metadata=meta,
        )


@dataclasses.dataclass(frozen=True)
class SecurityOracleExperience:
    """Serializable oracle-review record derived from a security rollout step."""

    state: tuple[float, ...]
    action: SecurityAction
    reward: float
    outcome: Mapping[str, Any]
    policy_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema: str = "alberta.security_gym.oracle_experience.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable oracle experience mapping."""
        payload = {
            "schema": self.schema,
            "state": list(self.state),
            "action": int(self.action),
            "action_name": security_gym_action_name(self.action),
            "reward": self.reward,
            "outcome": _copy_json_mapping(self.outcome, name="security oracle outcome"),
            "policy_metadata": _copy_json_mapping(
                self.policy_metadata, name="policy_metadata"
            ),
        }
        _require_rfc_json_mapping(payload, name="security oracle experience")
        return payload


def security_rollout_step_to_oracle_experience(
    step: SecurityRolloutStep,
) -> SecurityOracleExperience:
    """Convert a rollout transition to a compact oracle-review record."""
    policy_metadata = _copy_json_mapping(step.policy_metadata, name="policy_metadata")
    is_malicious = policy_metadata.get("is_malicious", False)
    if type(is_malicious) is not bool:
        raise ValueError("policy_metadata is_malicious must be an exact bool")
    defensive_action = step.action in (
        SecurityAction.THROTTLE,
        SecurityAction.BLOCK,
        SecurityAction.ISOLATE,
    )
    if is_malicious and defensive_action:
        label = "true_positive"
    elif is_malicious:
        label = "false_negative"
    elif defensive_action:
        label = "false_positive"
    else:
        label = "true_negative"
    return SecurityOracleExperience(
        state=step.state,
        action=step.action,
        reward=step.reward,
        outcome={
            "label": label,
            "terminated": step.terminated,
            "truncated": step.truncated,
        },
        policy_metadata=policy_metadata,
    )


def validate_security_oracle_experience(
    records: Sequence[SecurityOracleExperience],
    schema: SecurityFeatureSchema,
) -> None:
    """Validate oracle-review records against a feature schema."""
    for idx, record in enumerate(records):
        try:
            schema.validate_observation(record.state)
        except ValueError as exc:
            raise ValueError(f"invalid oracle experience {idx}: {exc}") from exc
        if not isinstance(record.outcome.get("label"), str) or not record.outcome["label"]:
            raise ValueError(f"invalid oracle experience {idx}: missing outcome label")


def coerce_security_action(action: object) -> SecurityAction:
    """Coerce an integer or name to ``SecurityAction``."""
    if type(action) is bool:
        raise ValueError("security action must not be a boolean")
    if type(action) is SecurityAction:
        return action
    if type(action) is int:
        try:
            return SecurityAction(action)
        except ValueError as exc:
            raise ValueError("unknown security action") from exc
    if type(action) is str:
        _require_exact_str("security action", action)
        normalized = action.strip().lower()
        if normalized in _ACTION_ALIASES:
            return _ACTION_ALIASES[normalized]
        for candidate in SecurityAction:
            if candidate.name.lower() == normalized:
                return candidate
        raise ValueError("unknown security action")
    raise ValueError("unknown security action")


def security_gym_action_name(action: object) -> str:
    """Return the action name expected by ``security-gym``."""
    return SECURITY_GYM_ACTION_NAMES[int(coerce_security_action(action))]


def to_security_gym_action(
    action: object,
    risk_score: object = 0.0,
) -> dict[str, int | tuple[float]]:
    """Convert a framework action into a ``security-gym`` action dict.

    ``security-gym`` uses a Gymnasium ``Dict`` action space with a discrete
    ``action`` id and a one-element ``risk_score`` array. A one-element tuple is
    accepted by the environment and keeps this module dependency-free.
    """
    if type(risk_score) not in _ALLOWED_REAL_TYPES:
        raise ValueError("risk_score must be a finite real number")
    try:
        val = validated_float32_scalar("risk_score", risk_score)
    except ValueError as exc:
        raise ValueError("risk_score must be a finite real number") from exc
    if not math.isfinite(val):
        raise ValueError("risk_score must be a finite real number")
    clipped_risk = min(10.0, max(0.0, val))
    return {
        "action": int(coerce_security_action(action)),
        "risk_score": (clipped_risk,),
    }


def security_gym_action_reward(
    action: object,
    *,
    is_malicious: object,
) -> float:
    """Return the immediate action reward from ``security-gym`` v0.4.x."""
    if type(is_malicious) is not bool:
        raise ValueError("is_malicious must be an exact bool")
    table = _SECURITY_GYM_ATTACK_REWARDS if is_malicious else _SECURITY_GYM_BENIGN_REWARDS
    return table[coerce_security_action(action)]


def validate_security_rollout(
    steps: Sequence[SecurityRolloutStep],
    schema: SecurityFeatureSchema,
) -> None:
    """Validate that rollout transitions satisfy the active-defense contract."""
    for idx, step in enumerate(steps):
        try:
            schema.validate_observation(step.state)
            schema.validate_observation(step.next_state)
        except ValueError as exc:
            raise ValueError(f"invalid rollout step {idx}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class ThroughputMeasurement:
    """Measured events-per-second summary."""

    n_events: int
    elapsed_s: float

    def __post_init__(self) -> None:
        n_events = _require_int("n_events", self.n_events, minimum=0, maximum=_INT32_MAX)
        if type(self.elapsed_s) not in _ALLOWED_REAL_TYPES:
            raise ValueError("elapsed_s must be a real number")
        try:
            elapsed_s = float(self.elapsed_s)
        except (OverflowError, ValueError) as exc:
            raise ValueError("elapsed_s must be representable as a float") from exc
        object.__setattr__(self, "n_events", n_events)
        object.__setattr__(self, "elapsed_s", elapsed_s)
        # Non-finite and nonpositive telemetry remains constructible for compatibility;
        # strict JSON publication rejects its derived non-finite rate in ``to_dict``.

    @property
    def events_per_second(self) -> float:
        """Throughput in events per second."""
        if self.elapsed_s <= 0.0:
            return float("inf")
        return self.n_events / self.elapsed_s

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable measurement mapping."""
        payload = {
            "n_events": self.n_events,
            "elapsed_s": self.elapsed_s,
            "events_per_second": self.events_per_second,
        }
        _require_rfc_json_mapping(payload, name="throughput measurement")
        return payload


class ThroughputMeter:
    """Wall-clock throughput hook for daemon integration smoke tests."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._n_events = 0

    def tick(self, n_events: object = 1) -> None:
        """Record completed events."""
        n_events_int = _require_int("n_events", n_events, minimum=0, maximum=_INT32_MAX)
        if n_events_int > _INT32_MAX - self._n_events:
            raise ValueError("cumulative n_events must fit signed int32")
        self._n_events += n_events_int

    def measure(self) -> ThroughputMeasurement:
        """Return current throughput measurement."""
        return ThroughputMeasurement(
            n_events=self._n_events,
            elapsed_s=time.perf_counter() - self._start,
        )


__all__ = [
    "N_SECURITY_ACTIONS",
    "SECURITY_GYM_ACTION_NAMES",
    "SECURITY_ACTION_NAMES",
    "SecurityAction",
    "SecurityFeatureSchema",
    "SecurityOracleExperience",
    "SecurityRewardWeights",
    "SecurityRolloutStep",
    "ThroughputMeasurement",
    "ThroughputMeter",
    "coerce_security_action",
    "security_gym_action_name",
    "security_gym_action_reward",
    "security_rollout_step_to_oracle_experience",
    "security_reward",
    "to_security_gym_action",
    "validate_security_oracle_experience",
    "validate_security_rollout",
]
