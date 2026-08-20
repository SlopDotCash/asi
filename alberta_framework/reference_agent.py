"""Versioned host-facing transaction protocol for ASI reference-agent adapters.

This module provides immutable records and an in-process, single-writer
ownership ledger. It is an L0 interoperability preview only: it does not select
reference-dev, prove learning benefit, certify safety or robotics readiness, or
provide standalone checkpointing. The separate reference-life checkpoint codec
supports its documented quiescent aggregate configurations.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alberta_framework._bounded_containers import require_json_text_nesting


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value

REFERENCE_AGENT_API_VERSION = "asi.reference_agent.preview1"
REFERENCE_AGENT_MANIFEST_SCHEMA = "asi.reference_agent_manifest.preview1"
REFERENCE_TRANSACTION_STATE_SCHEMA = "asi.reference_transaction_state.preview1"
MAX_DECISION_INDEX = (1 << 64) - 1

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ID_LENGTH = 256
_MAX_LIFECYCLE_ID_LENGTH = _MAX_ID_LENGTH - len(
    f":{MAX_DECISION_INDEX}:authorization"
)
_MAX_CONFIG_BYTES = 1 << 20
_MAX_JSON_NESTING_DEPTH = 64
_MAX_ARRAY_RANK = 8
_MAX_ARRAY_ELEMENTS = 1 << 20
_SUPPORTED_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


class DecisionOwnershipError(ValueError):
    """A record does not own the current lifecycle transaction."""


def _require_safe_id(value: str, *, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_ID_LENGTH
        or _SAFE_ID.fullmatch(value) is None
    ):
        raise ValueError(
            f"{name} must be a lowercase identifier of at most {_MAX_ID_LENGTH} "
            "characters containing only "
            "letters, digits, '.', '_', ':', or '-'"
        )


def _require_sha256(value: str, *, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _require_json_text_nesting(raw: str, *, name: str) -> None:
    """Reject nesting that would RecursionError ``json.loads`` before it runs."""
    if type(raw) is not str:
        raise ValueError(f"{name} must be canonical JSON")
    encoded_len = len(raw.encode("utf-8"))
    if encoded_len > _MAX_CONFIG_BYTES:
        raise ValueError(f"{name} exceeds {_MAX_CONFIG_BYTES} bytes")
    try:
        require_json_text_nesting(
            raw,
            max_depth=_MAX_JSON_NESTING_DEPTH,
            name=name,
        )
    except ValueError as exc:
        if "nesting limit" not in str(exc):
            raise
        raise ValueError(
            f"{name} exceeds the maximum canonical JSON nesting depth"
        ) from exc


def _load_manifest_config_json(raw: str) -> Any:
    _require_json_text_nesting(raw, name="manifest config")
    try:
        return json.loads(raw)
    except RecursionError as exc:
        raise ValueError(
            "manifest config exceeds the maximum canonical JSON nesting depth"
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest config must be canonical JSON") from exc


def _validate_json_value(value: Any, *, path: str, depth: int = 0) -> None:
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise ValueError(f"{path} exceeds the maximum canonical JSON nesting depth")
    value_type = type(value)
    if value is None or value_type is str or value_type is bool or value_type is int:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} JSON object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} is not a canonical JSON value")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError("config must be a JSON mapping")
    _validate_json_value(value, path="config")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("config must have a finite canonical JSON encoding") from exc
    if len(encoded) > _MAX_CONFIG_BYTES:
        raise ValueError(f"canonical config exceeds {_MAX_CONFIG_BYTES} bytes")
    return encoded


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of one finite, canonical JSON configuration."""

    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _require_bounded_sequence_length(
    value: object,
    *,
    name: str,
    maximum: int,
) -> int:
    try:
        count = len(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a sized sequence") from exc
    if type(count) is not int or count < 0 or count > maximum:
        raise ValueError(f"{name} exceed the array element limit")
    return count


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
        limits = np.iinfo(dtype)
        integers: list[int] = []
        for raw in array.reshape(-1):
            if array.dtype.kind == "f":
                floating = float(raw)
                if not math.isfinite(floating) or not floating.is_integer():
                    raise ValueError(
                        f"{name} must contain integral values representable by {dtype.name}"
                    )
            integer = int(raw)
            if integer < int(limits.min) or integer > int(limits.max):
                raise ValueError(f"{name} contains a value not representable by {dtype.name}")
            integers.append(integer)
        return np.asarray(integers, dtype=dtype).reshape(array.shape)
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
        host_dtype = _require_exact_str("dtype", self.dtype)
        try:
            dtype = np.dtype(host_dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype '{host_dtype}'") from exc
        if dtype.name not in _SUPPORTED_DTYPES or dtype.name != self.dtype:
            raise ValueError("ArrayValue dtype must be a portable canonical numeric dtype")
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("ArrayValue shape must be a tuple of positive dimensions or ()")
        if not isinstance(self.payload, bytes):
            raise ValueError("ArrayValue payload must be immutable bytes")
        if len(self.shape) > _MAX_ARRAY_RANK:
            raise ValueError(f"ArrayValue rank must be <= {_MAX_ARRAY_RANK}")
        elements = _shape_size(self.shape)
        if elements > _MAX_ARRAY_ELEMENTS:
            raise ValueError(f"ArrayValue must contain <= {_MAX_ARRAY_ELEMENTS} elements")
        expected = elements * dtype.itemsize
        if len(self.payload) != expected:
            raise ValueError(
                f"ArrayValue payload has {len(self.payload)} bytes; expected {expected}"
            )
        if not np.all(np.isfinite(self.to_numpy())):
            raise ValueError("ArrayValue payload must contain only finite values")

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Return a caller-owned NumPy copy."""

        dtype = np.dtype(self.dtype)
        storage_dtype = dtype.newbyteorder("<")
        return (
            np.frombuffer(self.payload, dtype=storage_dtype)
            .astype(dtype, copy=True)
            .reshape(self.shape)
        )

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
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("shape must be a tuple of positive integer dimensions or ()")
        host_dtype = _require_exact_str("dtype", self.dtype)
        try:
            dtype = np.dtype(host_dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype '{host_dtype}'") from exc
        if dtype.name not in _SUPPORTED_DTYPES:
            raise ValueError("dtype must be a supported portable floating-point or integer dtype")
        if dtype.name != self.dtype:
            host_canonical = _require_exact_str("dtype.name", dtype.name)
            raise ValueError(f"dtype must use canonical spelling '{host_canonical}'")
        if len(self.shape) > _MAX_ARRAY_RANK:
            raise ValueError(f"space rank must be <= {_MAX_ARRAY_RANK}")
        if _shape_size(self.shape) > _MAX_ARRAY_ELEMENTS:
            raise ValueError(f"space must contain <= {_MAX_ARRAY_ELEMENTS} elements")

        host_kind = _require_exact_str("kind", self.kind)
        if host_kind == "discrete":
            if self.shape != ():
                raise ValueError("a discrete space must have scalar shape ()")
            if dtype.kind not in {"i", "u"}:
                raise ValueError("a discrete space dtype must be integer")
            if (
                type(self.cardinality) is not int
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

        if host_kind != "box":
            raise ValueError("space kind must be 'discrete' or 'box'")
        if self.cardinality is not None:
            raise ValueError("a box space cannot declare a cardinality")
        if (self.low is None) != (self.high is None):
            raise ValueError("box low and high bounds must both be present or both be absent")
        if self.low is None:
            return
        assert self.high is not None
        size = _shape_size(self.shape)
        low_count = _require_bounded_sequence_length(
            self.low, name="box low bounds", maximum=_MAX_ARRAY_ELEMENTS
        )
        high_count = _require_bounded_sequence_length(
            self.high, name="box high bounds", maximum=_MAX_ARRAY_ELEMENTS
        )
        if low_count != size or high_count != size:
            raise ValueError("box bounds must contain one value per flattened shape entry")
        lows = _numeric_array(self.low, name="box low bounds")
        highs = _numeric_array(self.high, name="box high bounds")
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
        try:
            rank = len(shape)
        except TypeError as exc:
            raise ValueError("shape must be a tuple of positive integer dimensions or ()") from exc
        if rank > _MAX_ARRAY_RANK:
            raise ValueError(f"space rank must be <= {_MAX_ARRAY_RANK}")
        host_shape = tuple(shape)
        host_low: tuple[float | int, ...] | None = None
        host_high: tuple[float | int, ...] | None = None
        if low is not None or high is not None:
            if low is None or high is None:
                raise ValueError("box low and high bounds must both be present or both be absent")
            low_count = _require_bounded_sequence_length(
                low, name="box low bounds", maximum=_MAX_ARRAY_ELEMENTS
            )
            high_count = _require_bounded_sequence_length(
                high, name="box high bounds", maximum=_MAX_ARRAY_ELEMENTS
            )
            if any(type(dimension) is not int or dimension <= 0 for dimension in host_shape):
                if host_shape:
                    raise ValueError("shape must be a tuple of positive integer dimensions or ()")
            size = _shape_size(host_shape)
            if low_count != size or high_count != size:
                raise ValueError("box bounds must contain one value per flattened shape entry")
            host_low = tuple(low)
            host_high = tuple(high)
        return cls(
            kind="box",
            shape=host_shape,
            dtype=dtype,
            semantic_id=semantic_id,
            low=host_low,
            high=host_high,
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
            declared_dtype = np.dtype(self.dtype)
            supplied_dtype = getattr(value, "dtype", None)
            if supplied_dtype is not None:
                try:
                    supplied_name = np.dtype(supplied_dtype).name
                except TypeError as exc:
                    raise ValueError("space value has an unsupported dtype") from exc
                if supplied_name != self.dtype:
                    raise ValueError(
                        f"space value dtype must be exactly {self.dtype}, got {supplied_name}"
                    )
            if self.shape != () and not hasattr(value, "dtype"):
                expected = _shape_size(self.shape)
                count = _require_bounded_sequence_length(
                    value, name="space value", maximum=_MAX_ARRAY_ELEMENTS
                )
                if count != expected:
                    raise ValueError(
                        f"space value shape must be {self.shape}, got ({count},)"
                    )
            array = _numeric_array(value, name="space value")
            if array.shape != self.shape:
                raise ValueError(f"space value shape must be {self.shape}, got {array.shape}")
            if declared_dtype.kind in {"i", "u"} and array.dtype.kind == "f":
                raise TypeError("integer space value must be supplied as an integer, not float")
            array = _cast_representable(array, dtype=declared_dtype, name="space value")

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

        canonical = np.ascontiguousarray(
            array,
            dtype=np.dtype(self.dtype).newbyteorder("<"),
        )
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

    def __post_init__(self) -> None:
        if type(self.dispatch_rebinding) is not bool:
            raise ValueError("dispatch_rebinding must be boolean")

    def _descriptor(self) -> dict[str, Any]:
        return {"dispatch_rebinding": self.dispatch_rebinding}


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
        host_schema = _require_exact_str("schema", self.schema)
        if host_schema != REFERENCE_AGENT_MANIFEST_SCHEMA:
            host_expected = _require_exact_str("expected_schema", REFERENCE_AGENT_MANIFEST_SCHEMA)
            raise ValueError(
                f"schema must be '{host_expected}', got '{host_schema}'"
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
        config = _load_manifest_config_json(self._config_json)
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
        config = _load_manifest_config_json(self._config_json)
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
        if len(self.lifecycle_id) > _MAX_LIFECYCLE_ID_LENGTH:
            raise ValueError(
                "lifecycle_id is too long for canonical derived transaction IDs; "
                f"maximum length is {_MAX_LIFECYCLE_ID_LENGTH}"
            )
        if (
            type(self.decision_index) is not int
            or self.decision_index < 0
            or self.decision_index > MAX_DECISION_INDEX
        ):
            raise ValueError(
                f"decision_index must be a uint64 integer no greater than {MAX_DECISION_INDEX}"
            )
        expected_id = f"{self.lifecycle_id}:{self.decision_index}"
        _require_safe_id(self.decision_id, name="decision_id")
        if self.decision_id != expected_id:
            host_expected = _require_exact_str("expected_id", expected_id)
            raise ValueError(f"decision_id must use canonical value '{host_expected}'")
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
    ISSUED = "issued"
    DISPATCHED = "dispatched"
    OUTCOME = "outcome"
    EXHAUSTED = "exhausted"
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
        expected_id = f"{self.decision.decision_id}:authorization"
        if self.authorization_id != expected_id:
            host_expected = _require_exact_str("expected_id", expected_id)
            raise ValueError(f"authorization_id must use canonical value '{host_expected}'")
        if self.status is AuthorizationStatus.VETOED:
            if self.authorized_action is not None:
                raise ValueError("vetoed authorization cannot carry an action")
            if type(self.veto_reason) is not str or not self.veto_reason.strip():
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
        expected_id = f"{self.decision_id}:settlement"
        if self.settlement_id != expected_id:
            host_expected = _require_exact_str("expected_id", expected_id)
            raise ValueError(f"settlement_id must use canonical value '{host_expected}'")
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

@dataclasses.dataclass(frozen=True, slots=True)
class DispatchCommand:
    """Pre-execution command binding a settlement to one executor epoch."""

    dispatch: DispatchAck
    command_id: str
    executor_id: str
    executor_epoch: str

    def __post_init__(self) -> None:
        if not isinstance(self.dispatch, DispatchAck):
            raise ValueError("dispatch command requires a DispatchAck")
        _require_safe_id(self.command_id, name="command_id")
        _require_safe_id(self.executor_id, name="executor_id")
        _require_safe_id(self.executor_epoch, name="executor_epoch")
        expected_id = f"{self.decision_id}:command"
        if self.command_id != expected_id:
            host_expected = _require_exact_str("expected_id", expected_id)
            raise ValueError(f"command_id must use canonical value '{host_expected}'")

    @property
    def effective_action(self) -> ArrayValue:
        return self.dispatch.effective_action

    @property
    def decision(self) -> Decision:
        return self.dispatch.decision

    @property
    def lifecycle_id(self) -> str:
        return self.decision.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class DispatchReceipt:
    """Post-execution acknowledgement of the action an executor applied.

    The legacy ``dispatch``/``executor_id`` constructor remains available for
    preview compatibility and creates an explicit legacy executor command.
    New integrations must pass the command and independently reported action.
    """

    command: DispatchCommand
    applied_action: ArrayValue
    receipt_id: str

    def __init__(
        self,
        *,
        receipt_id: str,
        command: DispatchCommand | None = None,
        applied_action: ArrayValue | None = None,
        dispatch: DispatchAck | None = None,
        executor_id: str | None = None,
    ) -> None:
        if command is None:
            if not isinstance(dispatch, DispatchAck) or executor_id is None:
                raise ValueError(
                    "receipt requires command/applied_action or legacy dispatch/executor_id"
                )
            command = DispatchCommand(
                dispatch=dispatch,
                command_id=f"{dispatch.decision_id}:command",
                executor_id=executor_id,
                executor_epoch="legacy.preview1",
            )
            applied_action = dispatch.effective_action
        elif dispatch is not None:
            if executor_id is not None:
                raise ValueError("receipt constructors cannot mix executor_id with command")
            command = dataclasses.replace(command, dispatch=dispatch)
        elif executor_id is not None:
            raise ValueError("receipt constructors cannot mix executor_id with command")
        if not isinstance(command, DispatchCommand):
            raise ValueError("receipt requires a DispatchCommand")
        if not isinstance(applied_action, ArrayValue):
            raise ValueError("receipt applied_action must be an ArrayValue")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "applied_action", applied_action)
        object.__setattr__(self, "receipt_id", receipt_id)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_safe_id(self.receipt_id, name="receipt_id")
        expected_id = f"{self.decision.decision_id}:receipt"
        if self.receipt_id != expected_id:
            host_expected = _require_exact_str("expected_id", expected_id)
            raise ValueError(f"receipt_id must use canonical value '{host_expected}'")

    @property
    def effective_action(self) -> ArrayValue:
        return self.applied_action

    @property
    def dispatch(self) -> DispatchAck:
        return self.command.dispatch

    @property
    def executor_id(self) -> str:
        return self.command.executor_id

    @property
    def executor_epoch(self) -> str:
        return self.command.executor_epoch

    @property
    def decision(self) -> Decision:
        return self.command.decision


def _finite_scalar(value: Any, *, name: str) -> float:
    array = _numeric_array(value, name=name)
    if array.shape != ():
        raise ValueError(f"{name} must be a finite real scalar")
    original = array.item()
    try:
        converted = float(original)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must remain finite and representable as a protocol float"
        ) from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must remain finite and representable as a protocol float")
    if original != 0 and converted == 0.0:
        raise ValueError(f"{name} must not underflow when represented as a protocol float")
    return converted


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
        if not self.terminated and discount == 0.0:
            raise ValueError("discount == 0 requires terminated=True")
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
class ReferenceAgentUpdate:
    """State-agnostic staged result returned by a reference-agent adapter."""

    state: Any
    next_decision: Decision | None
    accepted: bool
    parameters_changed: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
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
            if type(self.rejection_reason) is not str or not self.rejection_reason.strip():
                raise ValueError("a rejected update requires a nonempty reason")


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome acceptance or fail-closed recovery result."""

    transaction: Transaction
    next_decision: Decision | None
    transaction_accepted: bool
    parameters_changed: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, Transaction):
            raise ValueError("step result transaction must be a Transaction")
        for name in ("transaction_accepted", "parameters_changed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not self.transaction_accepted:
            if self.parameters_changed:
                raise ValueError("a rejected transaction cannot change parameters")
            if type(self.rejection_reason) is not str or not self.rejection_reason.strip():
                raise ValueError("a rejected transaction requires a nonempty rejection_reason")
            if self.next_decision is not None:
                raise ValueError("a rejected transaction cannot arm a next decision")
            return
        if self.rejection_reason is not None:
            raise ValueError("an accepted transaction cannot carry a rejection reason")
        if self.next_decision is None and self.life_exhausted:
            return
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

    @property
    def event_consumed(self) -> bool:
        return self.transaction_accepted

    @property
    def recovery_required(self) -> bool:
        return not self.transaction_accepted

    @property
    def life_exhausted(self) -> bool:
        return (
            self.transaction_accepted
            and self.next_decision is None
            and self.transaction.decision.decision_index == MAX_DECISION_INDEX
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceTransactionState:
    """Immutable state for one in-process transaction ledger."""

    schema: str
    manifest_id: str
    phase: TransactionPhase
    lifecycle_id: str | None
    next_decision_index: int
    decision: Decision | None = None
    authorization: DispatchAuthorization | None = None
    dispatch: DispatchAck | None = None
    command: DispatchCommand | None = None
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
            type(self.next_decision_index) is not int
            or self.next_decision_index < 0
            or self.next_decision_index > MAX_DECISION_INDEX
        ):
            raise ValueError(
                "next_decision_index must be a uint64 integer no greater than "
                f"{MAX_DECISION_INDEX}"
            )
        if self.phase is TransactionPhase.HALTED:
            if type(self.halt_reason) is not str or not self.halt_reason.strip():
                raise ValueError("halted state requires a nonempty halt_reason")
        elif self.halt_reason is not None:
            raise ValueError("non-halted state cannot carry halt_reason")

        records = (
            self.decision,
            self.authorization,
            self.dispatch,
            self.command,
            self.receipt,
            self.transaction,
        )
        presence = tuple(record is not None for record in records)
        exact_presence = {
            TransactionPhase.READY: (False, False, False, False, False, False),
            TransactionPhase.ARMED: (True, False, False, False, False, False),
            TransactionPhase.AUTHORIZED: (True, True, False, False, False, False),
            TransactionPhase.SETTLED: (True, True, True, False, False, False),
            TransactionPhase.ISSUED: (True, True, True, True, False, False),
            TransactionPhase.DISPATCHED: (True, True, True, True, True, False),
            TransactionPhase.OUTCOME: (True, True, True, True, True, True),
            TransactionPhase.EXHAUSTED: (False, False, False, False, False, False),
        }
        if self.phase in exact_presence and presence != exact_presence[self.phase]:
            raise ValueError(f"{self.phase.value} phase carries inconsistent transaction records")
        if self.phase is TransactionPhase.HALTED:
            depth = sum(presence)
            if depth == 0 or presence != tuple(index < depth for index in range(6)):
                raise ValueError("halted phase requires one contiguous transaction record prefix")

        if self.phase is TransactionPhase.READY:
            if self.lifecycle_id is None and self.next_decision_index != 0:
                raise ValueError("an unstarted ready state must expect decision index 0")
            return
        if self.phase is TransactionPhase.EXHAUSTED:
            if self.lifecycle_id is None or self.next_decision_index != MAX_DECISION_INDEX:
                raise ValueError("exhausted state requires a lifecycle at the maximum index")
            return
        if self.lifecycle_id is None or self.decision is None:
            raise ValueError(f"{self.phase.value} phase requires an owned lifecycle decision")
        if not self.decision.armed:
            raise ValueError(f"{self.phase.value} phase cannot own a disarmed decision")
        if self.decision.manifest_id != self.manifest_id:
            raise ValueError("state decision manifest violates the ownership chain")
        if self.decision.lifecycle_id != self.lifecycle_id:
            raise ValueError("state decision lifecycle violates the ownership chain")
        if self.decision.decision_index != self.next_decision_index:
            raise ValueError("state decision index violates the ownership chain")
        if self.authorization is not None and self.authorization.decision != self.decision:
            raise ValueError("authorization decision violates the ownership chain")
        if self.dispatch is not None and self.dispatch.authorization != self.authorization:
            raise ValueError("dispatch authorization violates the ownership chain")
        if self.command is not None and self.command.dispatch != self.dispatch:
            raise ValueError("command dispatch violates the ownership chain")
        if self.receipt is not None and self.receipt.command != self.command:
            raise ValueError("receipt command violates the ownership chain")
        if (
            self.receipt is not None
            and self.command is not None
            and self.phase is not TransactionPhase.HALTED
            and self.receipt.applied_action != self.command.effective_action
        ):
            raise ValueError("a non-halted receipt applied action must equal the command")
        if self.transaction is not None and self.transaction.receipt != self.receipt:
            raise ValueError("outcome receipt violates the ownership chain")
        if (
            self.authorization is not None
            and self.authorization.status is AuthorizationStatus.VETOED
            and self.phase is not TransactionPhase.HALTED
        ):
            raise ValueError("vetoed authorization is valid only in a halted state")


class ReferenceTransactionReducer:
    """Pure semantic transitions for one manifest-bound transaction state.

    Aggregate runners use this reducer under their own outer commit. The live
    ledger remains the process-local compatibility/CAS owner.
    """

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

    def _ledger(self, state: ReferenceTransactionState) -> ReferenceTransactionLedger:
        ledger = ReferenceTransactionLedger(self.manifest)
        ledger._current_state = state
        return ledger

    def _validate_state(
        self,
        state: ReferenceTransactionState,
        *,
        require_current: bool = False,
    ) -> None:
        del require_current
        self._ledger(state)._validate_state(state)

    def validate_state(self, state: ReferenceTransactionState) -> None:
        self._validate_state(state)

    def _commit(
        self,
        previous: ReferenceTransactionState,
        next_state: ReferenceTransactionState,
    ) -> ReferenceTransactionState:
        self._validate_state(previous)
        self._validate_state(next_state)
        return next_state

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
        if type(reason) is not str or not reason.strip():
            raise ValueError("halt reason must be nonempty")
        return self._commit(
            state,
            dataclasses.replace(
                state,
                phase=TransactionPhase.HALTED,
                halt_reason=reason,
            ),
        )

    def halt(
        self,
        state: ReferenceTransactionState,
        *,
        reason: str,
    ) -> ReferenceTransactionState:
        self._validate_state(state)
        if state.phase in (TransactionPhase.HALTED, TransactionPhase.EXHAUSTED):
            raise DecisionOwnershipError(
                f"cannot halt a transaction already in phase {state.phase.value}"
            )
        return self._halt(state, reason=reason)

    def arm(
        self,
        state: ReferenceTransactionState,
        decision: Decision,
    ) -> ReferenceTransactionState:
        return self._ledger(state).arm(state, decision)

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
        return self._ledger(state).authorize(
            state,
            decision,
            authorized_action=authorized_action,
            authority_id=authority_id,
            policy_version=policy_version,
            authorization_id=authorization_id,
            veto_reason=veto_reason,
        )

    def settle_dispatch(
        self,
        state: ReferenceTransactionState,
        authorization: DispatchAuthorization,
        *,
        rebinding_applied: bool,
        settlement_id: str,
    ) -> tuple[ReferenceTransactionState, DispatchAck | None]:
        return self._ledger(state).settle_dispatch(
            state,
            authorization,
            rebinding_applied=rebinding_applied,
            settlement_id=settlement_id,
        )

    def issue_dispatch(
        self,
        state: ReferenceTransactionState,
        dispatch: DispatchAck,
        *,
        command_id: str,
        executor_id: str,
        executor_epoch: str,
    ) -> tuple[ReferenceTransactionState, DispatchCommand]:
        self._require_phase(state, TransactionPhase.SETTLED)
        if state.dispatch != dispatch:
            raise DecisionOwnershipError("dispatch command does not own the settlement")
        command = DispatchCommand(
            dispatch=dispatch,
            command_id=command_id,
            executor_id=executor_id,
            executor_epoch=executor_epoch,
        )
        return (
            self._commit(
                state,
                dataclasses.replace(
                    state,
                    phase=TransactionPhase.ISSUED,
                    command=command,
                ),
            ),
            command,
        )

    def record_dispatch(
        self,
        state: ReferenceTransactionState,
        receipt: DispatchReceipt,
    ) -> tuple[ReferenceTransactionState, DispatchReceipt]:
        self._require_phase(state, TransactionPhase.ISSUED)
        if not isinstance(receipt, DispatchReceipt):
            raise DecisionOwnershipError("dispatch acknowledgement must be a DispatchReceipt")
        if state.command != receipt.command:
            raise DecisionOwnershipError("dispatch receipt does not own the issued command")
        try:
            applied_action = self.manifest.action_spec.encode(receipt.applied_action)
        except (TypeError, ValueError) as exc:
            return self._halt(
                state,
                reason=f"executor applied action violates the manifest codec: {exc}",
            ), receipt
        assert state.command is not None
        if applied_action != state.command.effective_action:
            return (
                self._commit(
                    state,
                    dataclasses.replace(
                        state,
                        phase=TransactionPhase.HALTED,
                        receipt=receipt,
                        halt_reason="executor applied action differs from the settled command",
                    ),
                ),
                receipt,
            )
        return (
            self._commit(
                state,
                dataclasses.replace(
                    state,
                    phase=TransactionPhase.DISPATCHED,
                    receipt=receipt,
                ),
            ),
            receipt,
        )

    def halt_after_execution(
        self,
        state: ReferenceTransactionState,
        receipt: DispatchReceipt,
        *,
        reason: str,
    ) -> ReferenceTransactionState:
        self._require_phase(state, TransactionPhase.ISSUED)
        if not isinstance(receipt, DispatchReceipt) or state.command != receipt.command:
            raise DecisionOwnershipError(
                "post-execution halt receipt does not own the issued command"
            )
        try:
            self.manifest.action_spec.encode(receipt.applied_action)
        except (TypeError, ValueError) as exc:
            return self._halt(
                state,
                reason=f"{reason}; executor receipt could not be retained: {exc}",
            )
        return self._commit(
            state,
            dataclasses.replace(
                state,
                phase=TransactionPhase.HALTED,
                receipt=receipt,
                halt_reason=reason,
            ),
        )

    def record_outcome(
        self,
        state: ReferenceTransactionState,
        receipt: DispatchReceipt,
        **kwargs: Any,
    ) -> tuple[ReferenceTransactionState, Transaction]:
        return self._ledger(state).record_outcome(state, receipt, **kwargs)

    def accept(
        self,
        state: ReferenceTransactionState,
        *,
        next_decision: Decision | None,
        parameters_changed: bool,
    ) -> tuple[ReferenceTransactionState, StepResult]:
        return self._ledger(state).accept(
            state,
            next_decision=next_decision,
            parameters_changed=parameters_changed,
        )

    def reject(
        self,
        state: ReferenceTransactionState,
        *,
        reason: str,
    ) -> tuple[ReferenceTransactionState, StepResult]:
        return self._ledger(state).reject(state, reason=reason)

    def recover_accept(
        self,
        state: ReferenceTransactionState,
        *,
        next_decision: Decision | None,
        parameters_changed: bool,
    ) -> tuple[ReferenceTransactionState, StepResult]:
        self._validate_state(state)
        if state.phase is not TransactionPhase.HALTED or state.transaction is None:
            raise DecisionOwnershipError(
                "recovery requires a halted state retaining a complete outcome"
            )
        outcome = dataclasses.replace(
            state,
            phase=TransactionPhase.OUTCOME,
            halt_reason=None,
        )
        return self.accept(
            outcome,
            next_decision=next_decision,
            parameters_changed=parameters_changed,
        )


class ReferenceTransactionLedger:
    """Single-writer in-process transitions for one manifest-bound agent life.

    The ledger rejects stale snapshots within this object. It is deliberately
    not serializable and does not establish durable replay protection or exact
    resume across process restarts; those remain responsibilities of the future
    aggregate life runner.
    """

    __slots__ = ("_current_state", "_lock", "_manifest")

    def __init__(self, manifest: AgentManifest) -> None:
        if not isinstance(manifest, AgentManifest):
            raise ValueError("manifest must be an AgentManifest")
        self._manifest = manifest
        self._current_state: ReferenceTransactionState | None = None
        self._lock = threading.Lock()

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    def init(self) -> ReferenceTransactionState:
        state = ReferenceTransactionState(
            schema=REFERENCE_TRANSACTION_STATE_SCHEMA,
            manifest_id=self.manifest.manifest_id,
            phase=TransactionPhase.READY,
            lifecycle_id=None,
            next_decision_index=0,
        )
        with self._lock:
            if self._current_state is not None:
                raise DecisionOwnershipError(
                    "ledger is already initialized; replay is forbidden"
                )
            self._current_state = state
        return state

    def __reduce__(self) -> Any:
        raise TypeError(
            "live reference transaction ledger cannot be serialized; "
            "checkpoint the supported quiescent aggregate through "
            "reference_life_checkpoint"
        )

    def _validate_state(
        self,
        state: ReferenceTransactionState,
        *,
        require_current: bool = True,
    ) -> None:
        if not isinstance(state, ReferenceTransactionState):
            raise DecisionOwnershipError("state must be a ReferenceTransactionState")
        if state.schema != REFERENCE_TRANSACTION_STATE_SCHEMA:
            raise DecisionOwnershipError("state schema does not match the ledger")
        if state.manifest_id != self.manifest.manifest_id:
            raise DecisionOwnershipError("state manifest does not match the ledger")
        if require_current:
            if self._current_state is None:
                raise DecisionOwnershipError("ledger must be initialized before use")
            if state is not self._current_state:
                raise DecisionOwnershipError("state is stale, replayed, or not the current state")

        if state.decision is not None:
            self.manifest.validate_decision(state.decision)
        if state.authorization is not None:
            action = state.authorization.authorized_action
            if action is not None:
                self.manifest.action_spec.encode(action)
        if state.dispatch is not None:
            self.manifest.action_spec.encode(state.dispatch.effective_action)
            if (
                state.dispatch.status is DispatchStatus.REBOUND
                and not self.manifest.capabilities.dispatch_rebinding
            ):
                raise DecisionOwnershipError(
                    "rebound settlement is not declared by the agent manifest"
                )
        if state.command is not None:
            self.manifest.action_spec.encode(state.command.effective_action)
        if state.receipt is not None:
            self.manifest.action_spec.encode(state.receipt.applied_action)
        if state.transaction is not None:
            self.manifest.observation_spec.encode(state.transaction.bootstrap_observation)
            if state.transaction.next_decision_observation is not None:
                self.manifest.observation_spec.encode(
                    state.transaction.next_decision_observation
                )

    def _commit(
        self,
        previous: ReferenceTransactionState,
        next_state: ReferenceTransactionState,
    ) -> ReferenceTransactionState:
        self._validate_state(next_state, require_current=False)
        with self._lock:
            if self._current_state is not previous:
                raise DecisionOwnershipError("state changed before commit; replay is forbidden")
            self._current_state = next_state
        return next_state

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
        if type(reason) is not str or not reason.strip():
            raise ValueError("halt reason must be nonempty")
        halted = dataclasses.replace(
            state,
            phase=TransactionPhase.HALTED,
            halt_reason=reason,
        )
        return self._commit(state, halted)

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
        next_state = dataclasses.replace(
            state,
            phase=TransactionPhase.ARMED,
            lifecycle_id=lifecycle_id,
            decision=decision,
            authorization=None,
            dispatch=None,
            command=None,
            receipt=None,
            transaction=None,
            halt_reason=None,
        )
        return self._commit(state, next_state)

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
            return self._commit(state, halted), authorization

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
        next_state = dataclasses.replace(
            state,
            phase=TransactionPhase.AUTHORIZED,
            authorization=authorization,
        )
        return self._commit(state, next_state), authorization

    def settle_dispatch(
        self,
        state: ReferenceTransactionState,
        authorization: DispatchAuthorization,
        *,
        rebinding_applied: bool,
        settlement_id: str,
    ) -> tuple[ReferenceTransactionState, DispatchAck | None]:
        self._require_phase(state, TransactionPhase.AUTHORIZED)
        if state.authorization != authorization:
            raise DecisionOwnershipError("settlement does not own the authorization")
        if not isinstance(rebinding_applied, bool):
            raise ValueError("rebinding_applied must be boolean")
        if authorization.status is AuthorizationStatus.VETOED:
            raise DecisionOwnershipError("vetoed authorization cannot be settled")
        assert authorization.authorized_action is not None

        if authorization.status is AuthorizationStatus.REPLACED:
            supported = self.manifest.capabilities.dispatch_rebinding
            if not supported:
                reason = "authorized replacement cannot rebind learning credit"
                return self._halt(state, reason=reason), None
            if not rebinding_applied:
                raise ValueError("rebinding_applied=True is required for replacement")
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
        next_state = dataclasses.replace(
            state,
            phase=TransactionPhase.SETTLED,
            dispatch=dispatch,
        )
        return self._commit(state, next_state), dispatch

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
        next_state = dataclasses.replace(
            state,
            phase=TransactionPhase.DISPATCHED,
            command=receipt.command,
            receipt=receipt,
        )
        return self._commit(state, next_state), receipt

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
        next_state = dataclasses.replace(
            state,
            phase=TransactionPhase.OUTCOME,
            transaction=transaction,
        )
        return self._commit(state, next_state), transaction

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
        life_exhausted = state.next_decision_index == MAX_DECISION_INDEX
        if life_exhausted and next_decision is not None:
            raise DecisionOwnershipError("counter exhaustion cannot arm another decision")
        result = StepResult(
            transaction=state.transaction,
            next_decision=next_decision,
            transaction_accepted=True,
            parameters_changed=parameters_changed,
            rejection_reason=None,
        )
        if life_exhausted:
            next_state = dataclasses.replace(
                state,
                phase=TransactionPhase.EXHAUSTED,
                decision=None,
                authorization=None,
                dispatch=None,
                command=None,
                receipt=None,
                transaction=None,
            )
        elif next_decision is None:
            next_index = state.next_decision_index + 1
            next_state = dataclasses.replace(
                state,
                phase=TransactionPhase.READY,
                next_decision_index=next_index,
                decision=None,
                authorization=None,
                dispatch=None,
                command=None,
                receipt=None,
                transaction=None,
            )
        else:
            next_index = state.next_decision_index + 1
            next_state = dataclasses.replace(
                state,
                phase=TransactionPhase.ARMED,
                next_decision_index=next_index,
                decision=next_decision,
                authorization=None,
                dispatch=None,
                command=None,
                receipt=None,
                transaction=None,
            )
        return self._commit(state, next_state), result

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
            rejection_reason=reason,
        )
        return self._halt(state, reason=reason), result


__all__ = [
    "REFERENCE_AGENT_API_VERSION",
    "REFERENCE_AGENT_MANIFEST_SCHEMA",
    "REFERENCE_TRANSACTION_STATE_SCHEMA",
    "MAX_DECISION_INDEX",
    "AgentCapabilities",
    "AgentManifest",
    "ArrayValue",
    "AuthorizationStatus",
    "Decision",
    "DecisionOwnershipError",
    "DispatchAck",
    "DispatchAuthorization",
    "DispatchCommand",
    "DispatchReceipt",
    "DispatchStatus",
    "ReferenceAgentUpdate",
    "ReferenceTransactionLedger",
    "ReferenceTransactionReducer",
    "ReferenceTransactionState",
    "SpaceSpec",
    "StepResult",
    "Transaction",
    "TransactionPhase",
    "canonical_config_sha256",
]
