"""Versioned host-facing transaction protocol for ASI reference-agent adapters.

This module provides immutable records and a functional ownership ledger. It is
an L0 interoperability contract only: it does not select reference-dev, prove
learning benefit, certify safety or robotics readiness, or establish exact
whole-life checkpoint resume.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

REFERENCE_AGENT_API_VERSION = "asi.reference_agent.v1"
REFERENCE_AGENT_MANIFEST_SCHEMA = "asi.reference_agent_manifest.v1"
REFERENCE_TRANSACTION_STATE_SCHEMA = "asi.reference_transaction_state.v1"

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DecisionOwnershipError(ValueError):
    """A record does not own the current lifecycle transaction."""


def _require_safe_id(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a nonempty lowercase identifier containing only "
            "letters, digits, '.', '_', ':', or '-'"
        )


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} JSON object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} is not a canonical JSON value")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError("config must be a JSON mapping")
    _validate_json_value(value, path="config")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("config must have a finite canonical JSON encoding") from exc


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of one finite, canonical JSON configuration."""

    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _numeric_array(value: Any, *, name: str) -> np.ndarray[Any, Any]:
    try:
        array = np.array(value, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real numeric scalar or array") from exc
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must contain only real numeric values, not bool")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cast_representable(
    array: np.ndarray[Any, Any],
    *,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray[Any, Any]:
    if dtype.kind in {"i", "u"}:
        if array.dtype.kind == "f" and np.any(array != np.trunc(array)):
            raise ValueError(f"{name} must contain integral values representable by {dtype.name}")
        limits = np.iinfo(dtype)
        if np.any(array < limits.min) or np.any(array > limits.max):
            raise ValueError(f"{name} contains a value not representable by {dtype.name}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            cast = array.astype(dtype, copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a value not representable by {dtype.name}") from exc
    if dtype.kind == "f" and not np.all(np.isfinite(cast)):
        raise ValueError(f"{name} contains a value not finitely representable by {dtype.name}")
    return cast


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayValue:
    """Exact-dtype, immutable numeric value carried by protocol records."""

    semantic_id: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    def __post_init__(self) -> None:
        _require_safe_id(self.semantic_id, name="semantic_id")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from exc
        if dtype.kind not in {"f", "i", "u"} or dtype.name != self.dtype:
            raise ValueError("ArrayValue dtype must be a canonical real numeric dtype")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("ArrayValue shape must be a tuple of positive dimensions or ()")
        if not isinstance(self.payload, bytes):
            raise ValueError("ArrayValue payload must be immutable bytes")
        expected = _shape_size(self.shape) * dtype.itemsize
        if len(self.payload) != expected:
            raise ValueError(
                f"ArrayValue payload has {len(self.payload)} bytes; expected {expected}"
            )
        if not np.all(np.isfinite(self.to_numpy())):
            raise ValueError("ArrayValue payload must contain only finite values")

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Return a caller-owned NumPy copy."""

        dtype = np.dtype(self.dtype)
        return np.frombuffer(self.payload, dtype=dtype).copy().reshape(self.shape)

    def to_python(self) -> int | float | tuple[Any, ...]:
        """Return a deeply immutable Python scalar or nested tuple."""

        array = self.to_numpy()
        if array.shape == ():
            value = array.item()
            return float(value) if array.dtype.kind == "f" else int(value)
        frozen = _nested_tuple(array.tolist())
        assert isinstance(frozen, tuple)
        return frozen


@dataclasses.dataclass(frozen=True, slots=True)
class SpaceSpec:
    """Fixed numeric observation/action codec."""

    kind: str
    shape: tuple[int, ...]
    dtype: str
    semantic_id: str
    cardinality: int | None = None
    low: tuple[float | int, ...] | None = None
    high: tuple[float | int, ...] | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.semantic_id, name="semantic_id")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("shape must be a tuple of positive integer dimensions or ()")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from exc
        if dtype.kind not in {"f", "i", "u"}:
            raise ValueError("dtype must be a real floating-point or integer dtype")
        if dtype.name != self.dtype:
            raise ValueError(f"dtype must use canonical spelling {dtype.name!r}")

        if self.kind == "discrete":
            if self.shape != ():
                raise ValueError("a discrete space must have scalar shape ()")
            if dtype.kind not in {"i", "u"}:
                raise ValueError("a discrete space dtype must be integer")
            if (
                isinstance(self.cardinality, bool)
                or not isinstance(self.cardinality, int)
                or self.cardinality <= 0
            ):
                raise ValueError("discrete cardinality must be a positive integer")
            if self.cardinality - 1 > int(np.iinfo(dtype).max):
                raise ValueError(
                    f"discrete cardinality is not representable by declared dtype {dtype.name}"
                )
            if self.low is not None or self.high is not None:
                raise ValueError("a discrete space cannot declare box bounds")
            return

        if self.kind != "box":
            raise ValueError("space kind must be 'discrete' or 'box'")
        if self.cardinality is not None:
            raise ValueError("a box space cannot declare a cardinality")
        if (self.low is None) != (self.high is None):
            raise ValueError("box low and high bounds must both be present or both be absent")
        if self.low is None:
            return
        assert self.high is not None
        lows = _numeric_array(self.low, name="box low bounds")
        highs = _numeric_array(self.high, name="box high bounds")
        size = _shape_size(self.shape)
        if lows.shape != (size,) or highs.shape != (size,):
            raise ValueError("box bounds must contain one value per flattened shape entry")
        if np.any(lows > highs):
            raise ValueError("every box low bound must be <= its high bound")
        cast_lows = _cast_representable(lows, dtype=dtype, name="box low bounds")
        cast_highs = _cast_representable(highs, dtype=dtype, name="box high bounds")
        if np.any(cast_lows > cast_highs):
            raise ValueError("box bounds reverse in the declared dtype")
        object.__setattr__(
            self,
            "low",
            tuple(
                float(value) if dtype.kind == "f" else int(value)
                for value in cast_lows.tolist()
            ),
        )
        object.__setattr__(
            self,
            "high",
            tuple(
                float(value) if dtype.kind == "f" else int(value)
                for value in cast_highs.tolist()
            ),
        )

    @classmethod
    def discrete(
        cls,
        *,
        cardinality: int,
        dtype: str,
        semantic_id: str,
    ) -> SpaceSpec:
        return cls(
            kind="discrete",
            shape=(),
            dtype=dtype,
            semantic_id=semantic_id,
            cardinality=cardinality,
        )

    @classmethod
    def box(
        cls,
        *,
        shape: tuple[int, ...],
        dtype: str,
        low: Sequence[float | int] | None,
        high: Sequence[float | int] | None,
        semantic_id: str,
    ) -> SpaceSpec:
        return cls(
            kind="box",
            shape=tuple(shape),
            dtype=dtype,
            semantic_id=semantic_id,
            low=None if low is None else tuple(low),
            high=None if high is None else tuple(high),
        )

    def encode(self, value: Any) -> ArrayValue:
        """Validate and encode a value without retaining caller aliases."""

        if isinstance(value, ArrayValue):
            if (
                value.semantic_id != self.semantic_id
                or value.dtype != self.dtype
                or value.shape != self.shape
            ):
                raise ValueError("encoded value does not match the declared space codec")
            array = value.to_numpy()
        else:
            array = _numeric_array(value, name="space value")
            if array.shape != self.shape:
                raise ValueError(f"space value shape must be {self.shape}, got {array.shape}")
            dtype = np.dtype(self.dtype)
            if isinstance(value, (np.ndarray, np.generic)) and array.dtype.name != self.dtype:
                raise ValueError(
                    f"space value dtype must be exactly {self.dtype}, got {array.dtype.name}"
                )
            array = _cast_representable(array, dtype=dtype, name="space value")

        if self.kind == "discrete":
            if array.dtype.kind not in {"i", "u"}:
                raise TypeError("discrete value must be an integer, not bool")
            integer = int(array)
            assert self.cardinality is not None
            if integer < 0 or integer >= self.cardinality:
                raise ValueError(
                    f"discrete value {integer} is outside cardinality range "
                    f"[0, {self.cardinality})"
                )
        elif self.low is not None:
            assert self.high is not None
            dtype = np.dtype(self.dtype)
            flattened = array.reshape(-1)
            lows = np.asarray(self.low, dtype=dtype)
            highs = np.asarray(self.high, dtype=dtype)
            if np.any(flattened < lows) or np.any(flattened > highs):
                raise ValueError("box value is outside the declared bounds/range")

        canonical = np.ascontiguousarray(array, dtype=np.dtype(self.dtype))
        return ArrayValue(
            semantic_id=self.semantic_id,
            dtype=self.dtype,
            shape=self.shape,
            payload=canonical.tobytes(order="C"),
        )

    def validate_value(self, value: Any) -> None:
        self.encode(value)

    def _descriptor(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "semantic_id": self.semantic_id,
            "cardinality": self.cardinality,
            "low": None if self.low is None else list(self.low),
            "high": None if self.high is None else list(self.high),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Mechanism disclosures, not performance claims."""

    dispatch_rebinding: bool = False
    explicit_discount: bool = True
    exact_checkpoint_resume: bool = False
    compiled_rollout: bool = False
    context_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "dispatch_rebinding",
            "explicit_discount",
            "exact_checkpoint_resume",
            "compiled_rollout",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.context_inputs, tuple):
            raise ValueError("context_inputs must be an immutable tuple")
        if len(set(self.context_inputs)) != len(self.context_inputs):
            raise ValueError("context_inputs must not contain duplicates")
        for context_input in self.context_inputs:
            _require_safe_id(context_input, name="context input")

    def _descriptor(self) -> dict[str, Any]:
        return {
            "dispatch_rebinding": self.dispatch_rebinding,
            "explicit_discount": self.explicit_discount,
            "exact_checkpoint_resume": self.exact_checkpoint_resume,
            "compiled_rollout": self.compiled_rollout,
            "context_inputs": list(self.context_inputs),
        }


def _manifest_digest(
    *,
    schema: str,
    implementation_id: str,
    state_schema: str,
    config_sha256: str,
    observation_spec: SpaceSpec,
    action_spec: SpaceSpec,
    capabilities: AgentCapabilities,
) -> str:
    payload = {
        "api_version": REFERENCE_AGENT_API_VERSION,
        "schema": schema,
        "implementation_id": implementation_id,
        "state_schema": state_schema,
        "config_sha256": config_sha256,
        "observation_spec": observation_spec._descriptor(),
        "action_spec": action_spec._descriptor(),
        "capabilities": capabilities._descriptor(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class AgentManifest:
    """Versioned declaration for one adapter configuration."""

    schema: str
    implementation_id: str
    state_schema: str
    config_sha256: str
    manifest_id: str
    observation_spec: SpaceSpec
    action_spec: SpaceSpec
    capabilities: AgentCapabilities
    _config_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_AGENT_MANIFEST_SCHEMA:
            raise ValueError(
                f"schema must be {REFERENCE_AGENT_MANIFEST_SCHEMA!r}, got {self.schema!r}"
            )
        _require_safe_id(self.implementation_id, name="implementation_id")
        _require_safe_id(self.state_schema, name="state_schema")
        _require_sha256(self.config_sha256, name="config_sha256")
        _require_sha256(self.manifest_id, name="manifest_id")
        if not isinstance(self.observation_spec, SpaceSpec):
            raise ValueError("observation_spec must be a SpaceSpec")
        if not isinstance(self.action_spec, SpaceSpec):
            raise ValueError("action_spec must be a SpaceSpec")
        if not isinstance(self.capabilities, AgentCapabilities):
            raise ValueError("capabilities must be AgentCapabilities")
        if not self.capabilities.explicit_discount:
            raise ValueError("reference-agent adapters must consume an explicit discount")
        try:
            config = json.loads(self._config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest config must be canonical JSON") from exc
        if not isinstance(config, dict):
            raise ValueError("manifest config must be a JSON object")
        if _canonical_json_bytes(config).decode("utf-8") != self._config_json:
            raise ValueError("manifest config must use canonical JSON encoding")
        if canonical_config_sha256(config) != self.config_sha256:
            raise ValueError("manifest config_sha256 does not match config")
        expected = _manifest_digest(
            schema=self.schema,
            implementation_id=self.implementation_id,
            state_schema=self.state_schema,
            config_sha256=self.config_sha256,
            observation_spec=self.observation_spec,
            action_spec=self.action_spec,
            capabilities=self.capabilities,
        )
        if self.manifest_id != expected:
            raise ValueError("manifest_id does not match the manifest fields")

    @classmethod
    def from_config(
        cls,
        *,
        schema: str,
        implementation_id: str,
        state_schema: str,
        config: Mapping[str, Any],
        observation_spec: SpaceSpec,
        action_spec: SpaceSpec,
        capabilities: AgentCapabilities,
    ) -> AgentManifest:
        config_bytes = _canonical_json_bytes(config)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        manifest_id = _manifest_digest(
            schema=schema,
            implementation_id=implementation_id,
            state_schema=state_schema,
            config_sha256=config_sha256,
            observation_spec=observation_spec,
            action_spec=action_spec,
            capabilities=capabilities,
        )
        return cls(
            schema=schema,
            implementation_id=implementation_id,
            state_schema=state_schema,
            config_sha256=config_sha256,
            manifest_id=manifest_id,
            observation_spec=observation_spec,
            action_spec=action_spec,
            capabilities=capabilities,
            _config_json=config_bytes.decode("utf-8"),
        )

    @property
    def config(self) -> dict[str, Any]:
        config = json.loads(self._config_json)
        assert isinstance(config, dict)
        return config

    def make_decision(
        self,
        *,
        lifecycle_id: str,
        decision_index: int,
        observation_id: str,
        observation: Any,
        proposed_action: Any | None,
        armed: bool,
    ) -> Decision:
        if not isinstance(armed, bool):
            raise ValueError("armed must be boolean")
        encoded_observation = self.observation_spec.encode(observation)
        if armed:
            if proposed_action is None:
                raise ValueError("an armed decision requires a proposed_action")
            encoded_action = self.action_spec.encode(proposed_action)
        else:
            if proposed_action is not None:
                raise ValueError("a disarmed decision cannot carry a proposed_action")
            encoded_action = None
        return Decision(
            manifest_id=self.manifest_id,
            lifecycle_id=lifecycle_id,
            decision_index=decision_index,
            decision_id=f"{lifecycle_id}:{decision_index}",
            observation_id=observation_id,
            observation=encoded_observation,
            proposed_action=encoded_action,
            armed=armed,
        )

    def validate_decision(self, decision: Decision) -> None:
        if not isinstance(decision, Decision):
            raise ValueError("decision must be a Decision")
        if decision.manifest_id != self.manifest_id:
            raise ValueError("decision manifest identity does not match this manifest")
        self.observation_spec.encode(decision.observation)
        if decision.armed:
            if decision.proposed_action is None:
                raise ValueError("armed decision lacks a proposed action")
            self.action_spec.encode(decision.proposed_action)
        elif decision.proposed_action is not None:
            raise ValueError("disarmed decision cannot carry a proposed action")


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """One manifest-bound decision proposal."""

    manifest_id: str
    lifecycle_id: str
    decision_index: int
    decision_id: str
    observation_id: str
    observation: ArrayValue
    proposed_action: ArrayValue | None
    armed: bool

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_id, name="manifest_id")
        _require_safe_id(self.lifecycle_id, name="lifecycle_id")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
        ):
            raise ValueError("decision_index must be a nonnegative integer")
        expected_id = f"{self.lifecycle_id}:{self.decision_index}"
        if self.decision_id != expected_id:
            raise ValueError(f"decision_id must use canonical value {expected_id!r}")
        _require_safe_id(self.decision_id, name="decision_id")
        _require_safe_id(self.observation_id, name="observation_id")
        if not isinstance(self.observation, ArrayValue):
            raise ValueError("decision observation must be an ArrayValue")
        if not isinstance(self.armed, bool):
            raise ValueError("armed must be boolean")
        if self.armed:
            if not isinstance(self.proposed_action, ArrayValue):
                raise ValueError("an armed decision requires an encoded proposed action")
        elif self.proposed_action is not None:
            raise ValueError("a disarmed decision cannot carry a proposed action")


class AuthorizationStatus(enum.Enum):
    EXACT = "exact"
    REPLACED = "replaced"
    VETOED = "vetoed"


class DispatchStatus(enum.Enum):
    EXACT = "exact"
    REBOUND = "rebound"


class TransactionPhase(enum.Enum):
    READY = "ready"
    ARMED = "armed"
    AUTHORIZED = "authorized"
    SETTLED = "settled"
    DISPATCHED = "dispatched"
    OUTCOME = "outcome"
    HALTED = "halted"


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchAuthorization:
    """Independent authority's decision-bound authorization."""

    decision: Decision
    status: AuthorizationStatus
    authorized_action: ArrayValue | None
    authority_id: str
    policy_version: str
    authorization_id: str
    veto_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision) or not self.decision.armed:
            raise ValueError("authorization requires an armed Decision")
        if not isinstance(self.status, AuthorizationStatus):
            raise ValueError("authorization status must be an AuthorizationStatus")
        _require_safe_id(self.authority_id, name="authority_id")
        _require_safe_id(self.policy_version, name="policy_version")
        _require_safe_id(self.authorization_id, name="authorization_id")
        if self.status is AuthorizationStatus.VETOED:
            if self.authorized_action is not None:
                raise ValueError("vetoed authorization cannot carry an action")
            if not isinstance(self.veto_reason, str) or not self.veto_reason.strip():
                raise ValueError("vetoed authorization requires a nonempty reason")
            return
        if not isinstance(self.authorized_action, ArrayValue):
            raise ValueError("non-vetoed authorization requires an encoded action")
        if self.veto_reason is not None:
            raise ValueError("non-vetoed authorization cannot carry a veto reason")
        assert self.decision.proposed_action is not None
        actions_equal = self.authorized_action == self.decision.proposed_action
        if self.status is AuthorizationStatus.EXACT and not actions_equal:
            raise ValueError("exact authorization action must equal the proposal")
        if self.status is AuthorizationStatus.REPLACED and actions_equal:
            raise ValueError("replacement authorization action must differ from the proposal")

    @property
    def lifecycle_id(self) -> str:
        return self.decision.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchAck:
    """Agent settlement of one non-vetoed authorization."""

    authorization: DispatchAuthorization
    status: DispatchStatus
    effective_action: ArrayValue
    settlement_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, DispatchAuthorization):
            raise ValueError("dispatch settlement requires a DispatchAuthorization")
        if self.authorization.status is AuthorizationStatus.VETOED:
            raise ValueError("a vetoed authorization cannot be settled")
        if not isinstance(self.status, DispatchStatus):
            raise ValueError("dispatch status must be a DispatchStatus")
        if not isinstance(self.effective_action, ArrayValue):
            raise ValueError("dispatch effective_action must be an ArrayValue")
        _require_safe_id(self.settlement_id, name="settlement_id")
        assert self.authorization.authorized_action is not None
        if self.effective_action != self.authorization.authorized_action:
            raise ValueError("settlement action must equal the independently authorized action")
        if (
            self.authorization.status is AuthorizationStatus.EXACT
            and self.status is not DispatchStatus.EXACT
        ):
            raise ValueError("exact authorization requires exact settlement")
        if (
            self.authorization.status is AuthorizationStatus.REPLACED
            and self.status is not DispatchStatus.REBOUND
        ):
            raise ValueError("replacement authorization requires rebound settlement")

    @property
    def decision(self) -> Decision:
        return self.authorization.decision

    @property
    def lifecycle_id(self) -> str:
        return self.decision.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def authorized(self) -> bool:
        return True

    @property
    def transition_expected(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchReceipt:
    """Record an executor's acknowledgement of one settled action.

    This L0 record is lineage, not external proof that a physical action
    occurred. A concrete dispatch adapter must define that stronger contract.
    """

    dispatch: DispatchAck
    receipt_id: str
    executor_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.dispatch, DispatchAck):
            raise ValueError("receipt requires a DispatchAck")
        _require_safe_id(self.receipt_id, name="receipt_id")
        _require_safe_id(self.executor_id, name="executor_id")

    @property
    def effective_action(self) -> ArrayValue:
        return self.dispatch.effective_action

    @property
    def decision(self) -> Decision:
        return self.dispatch.decision


def _finite_scalar(value: Any, *, name: str) -> float:
    array = _numeric_array(value, name=name)
    if array.shape != ():
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


@dataclasses.dataclass(frozen=True, slots=True)
class Transaction:
    """One receipt-bound environment outcome."""

    receipt: DispatchReceipt
    reward: float
    discount: float
    terminated: bool
    truncated: bool
    autoreset: bool
    bootstrap_observation_id: str
    bootstrap_observation: ArrayValue
    next_decision_observation_id: str | None
    next_decision_observation: ArrayValue | None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DispatchReceipt):
            raise ValueError("transaction requires a DispatchReceipt")
        reward = _finite_scalar(self.reward, name="reward")
        discount = _finite_scalar(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        for name in ("terminated", "truncated", "autoreset"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.terminated and discount != 0.0:
            raise ValueError("terminated transactions require discount == 0")
        if self.truncated and not self.terminated and discount <= 0.0:
            raise ValueError("a truncated nonterminal transaction requires discount > 0")
        _require_safe_id(self.bootstrap_observation_id, name="bootstrap_observation_id")
        if not isinstance(self.bootstrap_observation, ArrayValue):
            raise ValueError("bootstrap_observation must be an ArrayValue")
        if self.bootstrap_observation_id == self.decision.observation_id:
            raise ValueError("outcome observation identity must advance beyond the decision input")

        if self.is_boundary:
            if self.autoreset:
                if self.next_decision_observation_id is None:
                    raise ValueError("autoreset requires next_decision_observation_id")
                if not isinstance(self.next_decision_observation, ArrayValue):
                    raise ValueError("autoreset requires next_decision_observation")
                _require_safe_id(
                    self.next_decision_observation_id,
                    name="next_decision_observation_id",
                )
                if self.next_decision_observation_id == self.bootstrap_observation_id:
                    raise ValueError(
                        "autoreset final and reset observations need distinct identities"
                    )
            elif (
                self.next_decision_observation_id is not None
                or self.next_decision_observation is not None
            ):
                raise ValueError(
                    "a boundary without autoreset cannot carry a next decision observation"
                )
        else:
            if self.autoreset:
                raise ValueError("autoreset is valid only at a boundary")
            if self.next_decision_observation_id != self.bootstrap_observation_id:
                raise ValueError(
                    "continuing bootstrap and next observations must share identity"
                )
            if self.next_decision_observation != self.bootstrap_observation:
                raise ValueError(
                    "continuing bootstrap and next observations must have equal values"
                )

        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "discount", discount)

    @property
    def decision(self) -> Decision:
        return self.receipt.decision

    @property
    def lifecycle_id(self) -> str:
        return self.decision.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def is_boundary(self) -> bool:
        return self.terminated or self.truncated

    @property
    def is_autoreset(self) -> bool:
        return self.autoreset


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome acceptance or fail-closed retry result."""

    transaction: Transaction
    next_decision: Decision | None
    transaction_accepted: bool
    parameters_changed: bool
    retry_required: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, Transaction):
            raise ValueError("step result transaction must be a Transaction")
        for name in ("transaction_accepted", "parameters_changed", "retry_required"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not self.transaction_accepted:
            if self.parameters_changed:
                raise ValueError("a rejected transaction cannot change parameters")
            if not self.retry_required:
                raise ValueError("a rejected transaction requires retry_required=True")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ValueError("a rejected transaction requires a nonempty rejection_reason")
            if self.next_decision is not None:
                raise ValueError("a rejected transaction cannot arm a next decision")
            return
        if self.retry_required:
            raise ValueError("an accepted transaction requires retry_required=False")
        if self.rejection_reason is not None:
            raise ValueError("an accepted transaction cannot carry a rejection reason")
        if self.next_decision is None:
            if not self.transaction.is_boundary or self.transaction.is_autoreset:
                raise ValueError("a continuing/autoreset transaction must arm a next decision")
            return
        if not isinstance(self.next_decision, Decision):
            raise ValueError("next_decision must be a Decision or None")
        if not self.next_decision.armed:
            raise DecisionOwnershipError("next decision must be armed")
        if self.next_decision.manifest_id != self.transaction.decision.manifest_id:
            raise DecisionOwnershipError("next decision belongs to another manifest")
        if self.next_decision.lifecycle_id != self.transaction.lifecycle_id:
            raise DecisionOwnershipError("next decision belongs to another lifecycle")
        expected_index = self.transaction.decision.decision_index + 1
        if self.next_decision.decision_index != expected_index:
            raise DecisionOwnershipError(
                f"next decision index must be {expected_index}"
            )
        if (
            self.next_decision.observation_id
            != self.transaction.next_decision_observation_id
        ):
            raise DecisionOwnershipError("next decision does not own the next observation")
        if self.next_decision.observation != self.transaction.next_decision_observation:
            raise DecisionOwnershipError("next decision observation value does not match")


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceTransactionState:
    """Immutable state for one functional transaction ledger."""

    schema: str
    manifest_id: str
    phase: TransactionPhase
    lifecycle_id: str | None
    next_decision_index: int
    decision: Decision | None = None
    authorization: DispatchAuthorization | None = None
    dispatch: DispatchAck | None = None
    receipt: DispatchReceipt | None = None
    transaction: Transaction | None = None
    halt_reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_TRANSACTION_STATE_SCHEMA:
            raise ValueError("unsupported reference transaction state schema")
        _require_sha256(self.manifest_id, name="manifest_id")
        if not isinstance(self.phase, TransactionPhase):
            raise ValueError("phase must be a TransactionPhase")
        if self.lifecycle_id is not None:
            _require_safe_id(self.lifecycle_id, name="lifecycle_id")
        if (
            isinstance(self.next_decision_index, bool)
            or not isinstance(self.next_decision_index, int)
            or self.next_decision_index < 0
        ):
            raise ValueError("next_decision_index must be a nonnegative integer")
        if self.phase is TransactionPhase.HALTED:
            if not isinstance(self.halt_reason, str) or not self.halt_reason.strip():
                raise ValueError("halted state requires a nonempty halt_reason")
        elif self.halt_reason is not None:
            raise ValueError("non-halted state cannot carry halt_reason")

        records = (
            ("decision", self.decision, Decision),
            ("authorization", self.authorization, DispatchAuthorization),
            ("dispatch", self.dispatch, DispatchAck),
            ("receipt", self.receipt, DispatchReceipt),
            ("transaction", self.transaction, Transaction),
        )
        for name, value, expected_type in records:
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(f"state {name} must be a {expected_type.__name__} or None")

        present = tuple(value is not None for _, value, _ in records)
        expected_by_phase = {
            TransactionPhase.READY: (False, False, False, False, False),
            TransactionPhase.ARMED: (True, False, False, False, False),
            TransactionPhase.AUTHORIZED: (True, True, False, False, False),
            TransactionPhase.SETTLED: (True, True, True, False, False),
            TransactionPhase.DISPATCHED: (True, True, True, True, False),
            TransactionPhase.OUTCOME: (True, True, True, True, True),
        }
        if self.phase is TransactionPhase.HALTED:
            allowed_halted = {
                (True, True, False, False, False),
                (True, True, True, True, True),
            }
            if present not in allowed_halted:
                raise DecisionOwnershipError(
                    "halted state must retain an authorization-stage or outcome-stage chain"
                )
        elif present != expected_by_phase[self.phase]:
            raise DecisionOwnershipError(
                f"state record chain does not match phase {self.phase.value}"
            )

        if self.lifecycle_id is None:
            if self.phase is not TransactionPhase.READY or self.next_decision_index != 0:
                raise DecisionOwnershipError(
                    "only the initial ready state may omit lifecycle identity"
                )
        elif self.phase is TransactionPhase.READY and self.next_decision_index == 0:
            raise DecisionOwnershipError(
                "a lifecycle-bound ready state must have consumed a decision"
            )

        if self.decision is None:
            return
        if not self.decision.armed:
            raise DecisionOwnershipError("state decision must be armed")
        if self.decision.manifest_id != self.manifest_id:
            raise DecisionOwnershipError("state decision belongs to another manifest")
        if self.decision.lifecycle_id != self.lifecycle_id:
            raise DecisionOwnershipError("state decision belongs to another lifecycle")
        if self.decision.decision_index != self.next_decision_index:
            raise DecisionOwnershipError(
                "state decision index does not match next_decision_index"
            )
        if self.authorization is not None and self.authorization.decision != self.decision:
            raise DecisionOwnershipError("state authorization does not own its decision")
        if (
            self.dispatch is not None
            and self.dispatch.authorization != self.authorization
        ):
            raise DecisionOwnershipError("state dispatch does not own its authorization")
        if self.receipt is not None and self.receipt.dispatch != self.dispatch:
            raise DecisionOwnershipError("state receipt does not own its dispatch")
        if self.transaction is not None and self.transaction.receipt != self.receipt:
            raise DecisionOwnershipError("state transaction does not own its receipt")
        if (
            self.phase is TransactionPhase.AUTHORIZED
            and self.authorization is not None
            and self.authorization.status is AuthorizationStatus.VETOED
        ):
            raise DecisionOwnershipError("a vetoed authorization must halt the ledger")
        if self.phase is TransactionPhase.HALTED and self.transaction is None:
            assert self.authorization is not None
            if self.authorization.status is AuthorizationStatus.EXACT:
                raise DecisionOwnershipError(
                    "an exact authorization cannot produce a pre-dispatch halt"
                )


class ReferenceTransactionLedger:
    """Stateless transition functions for one manifest-bound agent life."""

    __slots__ = ("_manifest",)

    def __init__(self, manifest: AgentManifest) -> None:
        if not isinstance(manifest, AgentManifest):
            raise ValueError("manifest must be an AgentManifest")
        self._manifest = manifest

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    def init(self) -> ReferenceTransactionState:
        return ReferenceTransactionState(
            schema=REFERENCE_TRANSACTION_STATE_SCHEMA,
            manifest_id=self.manifest.manifest_id,
            phase=TransactionPhase.READY,
            lifecycle_id=None,
            next_decision_index=0,
        )

    def _validate_state(self, state: ReferenceTransactionState) -> None:
        if not isinstance(state, ReferenceTransactionState):
            raise DecisionOwnershipError("state must be a ReferenceTransactionState")
        if state.schema != REFERENCE_TRANSACTION_STATE_SCHEMA:
            raise DecisionOwnershipError("state schema does not match the ledger")
        if state.manifest_id != self.manifest.manifest_id:
            raise DecisionOwnershipError("state manifest does not match the ledger")
        if state.decision is None:
            return
        try:
            self.manifest.validate_decision(state.decision)
            if (
                state.authorization is not None
                and state.authorization.authorized_action is not None
            ):
                self.manifest.action_spec.validate_value(
                    state.authorization.authorized_action
                )
            if state.dispatch is not None:
                if (
                    state.authorization is not None
                    and state.authorization.status is AuthorizationStatus.REPLACED
                    and not self.manifest.capabilities.dispatch_rebinding
                ):
                    raise DecisionOwnershipError(
                        "rebound dispatch requires the manifest dispatch_rebinding capability"
                    )
                self.manifest.action_spec.validate_value(state.dispatch.effective_action)
            if state.transaction is not None:
                self.manifest.observation_spec.validate_value(
                    state.transaction.bootstrap_observation
                )
                if state.transaction.next_decision_observation is not None:
                    self.manifest.observation_spec.validate_value(
                        state.transaction.next_decision_observation
                    )
        except DecisionOwnershipError:
            raise
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError(
                "state records do not match the ledger manifest codecs"
            ) from exc

    def _require_phase(
        self,
        state: ReferenceTransactionState,
        expected: TransactionPhase,
    ) -> None:
        self._validate_state(state)
        if state.phase is not expected:
            raise DecisionOwnershipError(
                f"transaction phase must be {expected.value}, got {state.phase.value}"
            )

    def _halt(
        self,
        state: ReferenceTransactionState,
        *,
        reason: str,
    ) -> ReferenceTransactionState:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("halt reason must be nonempty")
        return dataclasses.replace(
            state,
            phase=TransactionPhase.HALTED,
            halt_reason=reason,
        )

    def arm(
        self,
        state: ReferenceTransactionState,
        decision: Decision,
    ) -> ReferenceTransactionState:
        self._require_phase(state, TransactionPhase.READY)
        self.manifest.validate_decision(decision)
        if not decision.armed:
            raise DecisionOwnershipError("cannot arm a disarmed decision")
        if decision.decision_index != state.next_decision_index:
            raise DecisionOwnershipError(
                f"decision index must equal expected {state.next_decision_index}"
            )
        if state.lifecycle_id is not None and decision.lifecycle_id != state.lifecycle_id:
            raise DecisionOwnershipError("decision belongs to a different lifecycle")
        lifecycle_id = decision.lifecycle_id
        return dataclasses.replace(
            state,
            phase=TransactionPhase.ARMED,
            lifecycle_id=lifecycle_id,
            decision=decision,
            authorization=None,
            dispatch=None,
            receipt=None,
            transaction=None,
            halt_reason=None,
        )

    def authorize(
        self,
        state: ReferenceTransactionState,
        decision: Decision,
        *,
        authorized_action: Any | None,
        authority_id: str,
        policy_version: str,
        authorization_id: str,
        veto_reason: str | None = None,
    ) -> tuple[ReferenceTransactionState, DispatchAuthorization]:
        self._require_phase(state, TransactionPhase.ARMED)
        if state.decision != decision:
            raise DecisionOwnershipError("authorization does not own the armed decision")
        self.manifest.validate_decision(decision)
        if veto_reason is not None:
            if authorized_action is not None:
                raise ValueError("veto cannot carry an authorized action")
            authorization = DispatchAuthorization(
                decision=decision,
                status=AuthorizationStatus.VETOED,
                authorized_action=None,
                authority_id=authority_id,
                policy_version=policy_version,
                authorization_id=authorization_id,
                veto_reason=veto_reason,
            )
            halted = dataclasses.replace(
                state,
                phase=TransactionPhase.HALTED,
                authorization=authorization,
                halt_reason=f"dispatch veto: {veto_reason}",
            )
            return halted, authorization

        assert decision.proposed_action is not None
        encoded_action = (
            decision.proposed_action
            if authorized_action is None
            else self.manifest.action_spec.encode(authorized_action)
        )
        status = (
            AuthorizationStatus.EXACT
            if encoded_action == decision.proposed_action
            else AuthorizationStatus.REPLACED
        )
        authorization = DispatchAuthorization(
            decision=decision,
            status=status,
            authorized_action=encoded_action,
            authority_id=authority_id,
            policy_version=policy_version,
            authorization_id=authorization_id,
        )
        return (
            dataclasses.replace(
                state,
                phase=TransactionPhase.AUTHORIZED,
                authorization=authorization,
            ),
            authorization,
        )

    def settle_dispatch(
        self,
        state: ReferenceTransactionState,
        authorization: DispatchAuthorization,
        *,
        rebinding_applied: bool,
        settlement_id: str,
        halt_on_unsupported: bool = True,
    ) -> tuple[ReferenceTransactionState, DispatchAck | None]:
        self._require_phase(state, TransactionPhase.AUTHORIZED)
        if state.authorization != authorization:
            raise DecisionOwnershipError("settlement does not own the authorization")
        if not isinstance(rebinding_applied, bool):
            raise ValueError("rebinding_applied must be boolean")
        if not isinstance(halt_on_unsupported, bool):
            raise ValueError("halt_on_unsupported must be boolean")
        if authorization.status is AuthorizationStatus.VETOED:
            raise DecisionOwnershipError("vetoed authorization cannot be settled")
        assert authorization.authorized_action is not None

        if authorization.status is AuthorizationStatus.REPLACED:
            supported = self.manifest.capabilities.dispatch_rebinding
            if not supported or not rebinding_applied:
                reason = "authorized replacement cannot rebind learning credit"
                if halt_on_unsupported:
                    return self._halt(state, reason=reason), None
                raise ValueError(
                    "rebinding_applied=True and declared dispatch_rebinding are required"
                )
            status = DispatchStatus.REBOUND
        else:
            if rebinding_applied:
                raise ValueError("rebinding_applied must be false for an exact authorization")
            status = DispatchStatus.EXACT

        dispatch = DispatchAck(
            authorization=authorization,
            status=status,
            effective_action=authorization.authorized_action,
            settlement_id=settlement_id,
        )
        return (
            dataclasses.replace(
                state,
                phase=TransactionPhase.SETTLED,
                dispatch=dispatch,
            ),
            dispatch,
        )

    def record_dispatch(
        self,
        state: ReferenceTransactionState,
        dispatch: DispatchAck,
        *,
        receipt_id: str,
        executor_id: str,
    ) -> tuple[ReferenceTransactionState, DispatchReceipt]:
        self._require_phase(state, TransactionPhase.SETTLED)
        if state.dispatch != dispatch:
            raise DecisionOwnershipError("dispatch receipt does not own the settled dispatch")
        receipt = DispatchReceipt(
            dispatch=dispatch,
            receipt_id=receipt_id,
            executor_id=executor_id,
        )
        return (
            dataclasses.replace(
                state,
                phase=TransactionPhase.DISPATCHED,
                receipt=receipt,
            ),
            receipt,
        )

    def record_outcome(
        self,
        state: ReferenceTransactionState,
        receipt: DispatchReceipt,
        *,
        reward: Any,
        discount: Any,
        terminated: bool,
        truncated: bool,
        autoreset: bool,
        bootstrap_observation_id: str,
        bootstrap_observation: Any,
        next_decision_observation_id: str | None,
        next_decision_observation: Any | None,
    ) -> tuple[ReferenceTransactionState, Transaction]:
        self._require_phase(state, TransactionPhase.DISPATCHED)
        if state.receipt != receipt:
            raise DecisionOwnershipError("outcome does not own the exact dispatch receipt")
        bootstrap = self.manifest.observation_spec.encode(bootstrap_observation)
        next_observation = (
            None
            if next_decision_observation is None
            else self.manifest.observation_spec.encode(next_decision_observation)
        )
        transaction = Transaction(
            receipt=receipt,
            reward=reward,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            autoreset=autoreset,
            bootstrap_observation_id=bootstrap_observation_id,
            bootstrap_observation=bootstrap,
            next_decision_observation_id=next_decision_observation_id,
            next_decision_observation=next_observation,
        )
        return (
            dataclasses.replace(
                state,
                phase=TransactionPhase.OUTCOME,
                transaction=transaction,
            ),
            transaction,
        )

    def accept(
        self,
        state: ReferenceTransactionState,
        *,
        next_decision: Decision | None,
        parameters_changed: bool,
    ) -> tuple[ReferenceTransactionState, StepResult]:
        self._require_phase(state, TransactionPhase.OUTCOME)
        if state.transaction is None:
            raise DecisionOwnershipError("outcome phase lacks a transaction")
        if not isinstance(parameters_changed, bool):
            raise ValueError("parameters_changed must be boolean")
        if next_decision is not None:
            self.manifest.validate_decision(next_decision)
        result = StepResult(
            transaction=state.transaction,
            next_decision=next_decision,
            transaction_accepted=True,
            parameters_changed=parameters_changed,
            retry_required=False,
            rejection_reason=None,
        )
        next_index = state.next_decision_index + 1
        if next_decision is None:
            next_state = dataclasses.replace(
                state,
                phase=TransactionPhase.READY,
                next_decision_index=next_index,
                decision=None,
                authorization=None,
                dispatch=None,
                receipt=None,
                transaction=None,
            )
        else:
            next_state = dataclasses.replace(
                state,
                phase=TransactionPhase.ARMED,
                next_decision_index=next_index,
                decision=next_decision,
                authorization=None,
                dispatch=None,
                receipt=None,
                transaction=None,
            )
        return next_state, result

    def reject(
        self,
        state: ReferenceTransactionState,
        *,
        reason: str,
    ) -> tuple[ReferenceTransactionState, StepResult]:
        self._require_phase(state, TransactionPhase.OUTCOME)
        if state.transaction is None:
            raise DecisionOwnershipError("outcome phase lacks a transaction")
        result = StepResult(
            transaction=state.transaction,
            next_decision=None,
            transaction_accepted=False,
            parameters_changed=False,
            retry_required=True,
            rejection_reason=reason,
        )
        return self._halt(state, reason=reason), result


__all__ = [
    "REFERENCE_AGENT_API_VERSION",
    "REFERENCE_AGENT_MANIFEST_SCHEMA",
    "REFERENCE_TRANSACTION_STATE_SCHEMA",
    "AgentCapabilities",
    "AgentManifest",
    "ArrayValue",
    "AuthorizationStatus",
    "Decision",
    "DecisionOwnershipError",
    "DispatchAck",
    "DispatchAuthorization",
    "DispatchReceipt",
    "DispatchStatus",
    "ReferenceTransactionLedger",
    "ReferenceTransactionState",
    "SpaceSpec",
    "StepResult",
    "Transaction",
    "TransactionPhase",
    "canonical_config_sha256",
]
