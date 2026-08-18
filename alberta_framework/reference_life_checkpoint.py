"""Canonical whole-life checkpoints for the development-only reference runner.

Only the primitive Prototype + SwitchingTwoState/RiverSwim reference slices
are supported. Checkpoints are taken between transactions; this module makes
no process-crash or durable physical-dispatch claim.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat as stat_module
import struct
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.prototype_reference_adapter import (
    PrototypeReferenceAdapter,
    PrototypeReferenceState,
)
from alberta_framework.reference_agent import (
    ArrayValue,
    Decision,
    ReferenceTransactionState,
    TransactionPhase,
)
from alberta_framework.reference_life import (
    REFERENCE_LIFE_CONFIG_SCHEMA,
    REFERENCE_LIFE_RNG_SCHEDULE,
    REFERENCE_LIFE_STATE_SCHEMA,
    RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID,
    RIVERSWIM_ENVIRONMENT_STATE_SCHEMA,
    RIVERSWIM_REFERENCE_MAX_STATES,
    SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID,
    SWITCHING_ENVIRONMENT_STATE_SCHEMA,
    ExactDispatchConfig,
    ExactDispatchState,
    LifePhase,
    ReferenceLifeMetricsConfig,
    ReferenceLifeMetricsState,
    ReferenceLifeRunner,
    ReferenceLifeState,
    build_prototype_riverswim_life,
    build_prototype_switching_life,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimState,
    SwitchingTwoStateConfig,
    SwitchingTwoStateState,
)

REFERENCE_LIFE_CHECKPOINT_SCHEMA = "asi.reference_life_checkpoint.preview1"
REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA = "asi.reference_life_checkpoint_state.preview1"
REFERENCE_LIFE_CHECKPOINT_SOURCE_SCHEMA = "asi.reference_life_checkpoint_source.preview1"
REFERENCE_LIFE_CHECKPOINT_RUNTIME_SCHEMA = "asi.reference_life_checkpoint_runtime.preview1"
REFERENCE_LIFE_CHECKPOINT_DEPENDENCIES_SCHEMA = (
    "asi.reference_life_checkpoint_dependencies.preview1"
)
REFERENCE_LIFE_CHECKPOINT_HASHES_SCHEMA = "asi.reference_life_checkpoint_hashes.preview1"

_MANIFEST_NAME = "manifest.json"
_STATE_NAME = "life_state.json"
_PROTOTYPE_NAME = "prototype"
_COMMITTED_NAME = "COMMITTED"
_SOURCE_SCOPE = "checkpoint_time_compatibility_only.preview1"
_GENERATION_WIDTH = 20
_SHA256_LENGTH = 64
_DEPENDENCIES = ("chex", "jax", "jaxlib", "numpy", "orbax-checkpoint")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_INVENTORY_FILES = 8192
_MAX_PATH_BYTES = 1024
_MAX_ARRAY_RANK = 16
_MAX_ARRAY_ELEMENTS = 1 << 24
_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_TREE_DEPTH = 32
_MAX_CHILD_FILE_BYTES = 128 * 1024 * 1024
_MAX_CHILD_TOTAL_BYTES = 512 * 1024 * 1024
_PROTOTYPE_EMPTY_ARRAY_CODEC = "alberta.prototype_agent.empty_array_projection.v1"
_PROTOTYPE_SUPPORTED_PRNG_IMPLS = frozenset({"threefry2x32", "rbg"})


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _require_keys(value: Any, expected: set[str], *, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{path} must be a JSON object")
    mapping = cast(dict[str, Any], value)
    if set(mapping) != expected:
        _fail(f"{path} fields do not match the checkpoint schema")
    return mapping


def _require_str(value: Any, *, path: str) -> str:
    if type(value) is not str:
        _fail(f"{path} must be a string")
    return value


def _require_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be boolean")
    return value


def _require_float(value: Any, *, path: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        _fail(f"{path} must be a finite JSON number")
    result = float(value)
    if not np.isfinite(result):
        _fail(f"{path} must be finite")
    return result


def _require_digest(value: Any, *, path: str) -> str:
    digest = _require_str(value, path=path)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        _fail(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"checkpoint JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint value is not canonical finite JSON") from exc


def _read_bounded_json_file(path: Path) -> bytes:
    """Read one bounded regular JSON file through a no-follow descriptor."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path.name} must be a regular file")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{path.name} exceeds the canonical JSON size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) != metadata.st_size or len(raw) > _MAX_JSON_BYTES:
            raise ValueError(f"{path.name} changed or exceeds the JSON size limit")
    finally:
        os.close(descriptor)
    return raw


def _load_canonical_json(path: Path) -> dict[str, Any]:
    raw = _read_bounded_json_file(path)
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not canonical JSON") from exc
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise ValueError(f"{path.name} is not canonically encoded")
    return cast(dict[str, Any], value)


def _sha256_file(
    path: Path,
    *,
    max_bytes: int = _MAX_CHILD_FILE_BYTES,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise ValueError(f"checkpoint file exceeds its byte limit: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(min(1024 * 1024, max_bytes - size + 1)):
                digest.update(chunk)
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"checkpoint file exceeds its byte limit: {path}")
        if size != metadata.st_size:
            raise ValueError(f"checkpoint file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_identity() -> dict[str, Any]:
    root = _repository_root()
    paths = [
        *sorted((root / "alberta_framework").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    if len(paths) > _MAX_INVENTORY_FILES:
        raise ValueError("checkpoint source inventory exceeds its file-count limit")
    files: dict[str, Any] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint source inventory entry is unavailable: {path}")
        digest, size = _sha256_file(path, max_bytes=_MAX_JSON_BYTES)
        relative = path.relative_to(root).as_posix()
        if len(relative.encode("utf-8")) > _MAX_PATH_BYTES:
            raise ValueError("checkpoint source inventory path is too long")
        files[relative] = {"sha256": digest, "size": size}
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_SOURCE_SCHEMA,
        "scope": _SOURCE_SCOPE,
        "files": files,
    }


def _runtime_identity() -> dict[str, Any]:
    devices = sorted(
        (
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
            }
            for device in jax.devices()
        ),
        key=lambda value: (
            value["process_index"],
            value["id"],
            value["platform"],
            value["device_kind"],
        ),
    )
    environment_names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    )
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_RUNTIME_SCHEMA,
        "python_implementation": platform.python_implementation(),
        "python_version": list(sys.version_info[:3]),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "machine": platform.machine(),
        "default_backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": devices,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _dependency_identity() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in _DEPENDENCIES}
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_DEPENDENCIES_SCHEMA,
        "versions": versions,
    }


def _encode_float(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("checkpoint floats must be finite")
    return struct.pack(">d", number).hex()


def _decode_float(value: Any, *, path: str) -> float:
    encoded = _require_str(value, path=path)
    if len(encoded) != 16 or any(c not in "0123456789abcdef" for c in encoded):
        _fail(f"{path} must be one canonical binary64 value")
    result = cast(float, struct.unpack(">d", bytes.fromhex(encoded))[0])
    if not np.isfinite(result):
        _fail(f"{path} must be finite")
    return result


def _encode_array_value(value: ArrayValue) -> dict[str, Any]:
    if not isinstance(value, ArrayValue):
        raise ValueError("checkpoint protocol value must be an ArrayValue")
    return {
        "semantic_id": value.semantic_id,
        "dtype": value.dtype,
        "shape": list(value.shape),
        "payload_hex": value.payload.hex(),
    }


def _decode_array_value(value: Any, *, path: str) -> ArrayValue:
    data = _require_keys(
        value,
        {"semantic_id", "dtype", "shape", "payload_hex"},
        path=path,
    )
    shape_raw = data["shape"]
    if type(shape_raw) is not list:
        _fail(f"{path}.shape must be a JSON array")
    if len(shape_raw) > _MAX_ARRAY_RANK:
        _fail(f"{path}.shape exceeds the rank limit")
    shape = tuple(
        _require_int(dimension, path=f"{path}.shape[{index}]", minimum=1)
        for index, dimension in enumerate(shape_raw)
    )
    payload_hex = _require_str(data["payload_hex"], path=f"{path}.payload_hex")
    if len(payload_hex) > 2 * _MAX_ARRAY_BYTES:
        _fail(f"{path}.payload_hex exceeds the byte limit")
    if len(payload_hex) % 2 or any(c not in "0123456789abcdef" for c in payload_hex):
        _fail(f"{path}.payload_hex must be lowercase hexadecimal")
    return ArrayValue(
        semantic_id=_require_str(data["semantic_id"], path=f"{path}.semantic_id"),
        dtype=_require_str(data["dtype"], path=f"{path}.dtype"),
        shape=shape,
        payload=bytes.fromhex(payload_hex),
    )


def _encode_decision(value: Decision) -> dict[str, Any]:
    if not isinstance(value, Decision):
        raise ValueError("checkpoint ledger must own a Decision")
    return {
        "manifest_id": value.manifest_id,
        "lifecycle_id": value.lifecycle_id,
        "decision_index": value.decision_index,
        "decision_id": value.decision_id,
        "observation_id": value.observation_id,
        "observation": _encode_array_value(value.observation),
        "proposed_action": (
            None if value.proposed_action is None else _encode_array_value(value.proposed_action)
        ),
        "armed": value.armed,
    }


def _decode_decision(value: Any, *, path: str) -> Decision:
    data = _require_keys(
        value,
        {
            "manifest_id",
            "lifecycle_id",
            "decision_index",
            "decision_id",
            "observation_id",
            "observation",
            "proposed_action",
            "armed",
        },
        path=path,
    )
    proposed_raw = data["proposed_action"]
    return Decision(
        manifest_id=_require_digest(data["manifest_id"], path=f"{path}.manifest_id"),
        lifecycle_id=_require_str(data["lifecycle_id"], path=f"{path}.lifecycle_id"),
        decision_index=_require_int(data["decision_index"], path=f"{path}.decision_index"),
        decision_id=_require_str(data["decision_id"], path=f"{path}.decision_id"),
        observation_id=_require_str(data["observation_id"], path=f"{path}.observation_id"),
        observation=_decode_array_value(data["observation"], path=f"{path}.observation"),
        proposed_action=(
            None
            if proposed_raw is None
            else _decode_array_value(proposed_raw, path=f"{path}.proposed_action")
        ),
        armed=_require_bool(data["armed"], path=f"{path}.armed"),
    )


def _encode_jax_array(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, jax.Array):
        raise ValueError(f"{path} must be a concrete JAX array")
    if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        value.dtype, jax.dtypes.prng_key
    ):
        raise ValueError(f"{path} must not be a typed PRNG key")
    array = np.asarray(value)
    dtype = np.dtype(array.dtype)
    if dtype.byteorder not in ("=", "|"):
        raise ValueError(f"{path} must use native byte order")
    storage = array.astype(dtype.newbyteorder("<"), copy=False)
    return {
        "dtype": dtype.name,
        "shape": list(array.shape),
        "payload_hex": storage.tobytes(order="C").hex(),
    }


def _decode_jax_array(value: Any, *, path: str) -> jax.Array:
    data = _require_keys(value, {"dtype", "shape", "payload_hex"}, path=path)
    dtype_name = _require_str(data["dtype"], path=f"{path}.dtype")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"{path}.dtype is unsupported") from exc
    if dtype.name != dtype_name or dtype.kind not in {"b", "f", "i", "u"}:
        _fail(f"{path}.dtype is not a canonical numeric dtype")
    shape_raw = data["shape"]
    if type(shape_raw) is not list:
        _fail(f"{path}.shape must be a JSON array")
    if len(shape_raw) > _MAX_ARRAY_RANK:
        _fail(f"{path}.shape exceeds the rank limit")
    shape = tuple(
        _require_int(dimension, path=f"{path}.shape[{index}]", minimum=0)
        for index, dimension in enumerate(shape_raw)
    )
    payload_hex = _require_str(data["payload_hex"], path=f"{path}.payload_hex")
    if len(payload_hex) > 2 * _MAX_ARRAY_BYTES:
        _fail(f"{path}.payload_hex exceeds the byte limit")
    if len(payload_hex) % 2 or any(c not in "0123456789abcdef" for c in payload_hex):
        _fail(f"{path}.payload_hex must be lowercase hexadecimal")
    raw = bytes.fromhex(payload_hex)
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements > _MAX_ARRAY_ELEMENTS:
            _fail(f"{path}.shape exceeds the element limit")
    if elements * dtype.itemsize > _MAX_ARRAY_BYTES:
        _fail(f"{path} exceeds the byte limit")
    if len(raw) != elements * dtype.itemsize:
        _fail(f"{path}.payload_hex length does not match shape and dtype")
    storage_dtype = dtype.newbyteorder("<")
    array = np.frombuffer(raw, dtype=storage_dtype).astype(dtype, copy=True).reshape(shape)
    result = jnp.asarray(array)
    if result.shape != shape or np.dtype(result.dtype) != dtype:
        _fail(f"{path} cannot preserve its encoded shape and dtype in this runtime")
    return result


def _encode_metrics(metrics: ReferenceLifeMetricsState) -> dict[str, Any]:
    return {
        "schema": metrics.schema,
        "config_sha256": metrics.config_sha256,
        "accepted_events": metrics.accepted_events,
        "parameter_change_events": metrics.parameter_change_events,
        "reward_sum": _encode_float(metrics.reward_sum),
        "oracle_reward_sum": _encode_float(metrics.oracle_reward_sum),
        "regret_sum": _encode_float(metrics.regret_sum),
        "phase_event_counts": list(metrics.phase_event_counts),
        "phase_reward_sums": [_encode_float(value) for value in metrics.phase_reward_sums],
        "phase_regret_sums": [_encode_float(value) for value in metrics.phase_regret_sums],
        "phase_switches": metrics.phase_switches,
        "current_phase": metrics.current_phase,
        "current_segment_events": metrics.current_segment_events,
        "current_segment_reward": _encode_float(metrics.current_segment_reward),
        "current_segment_regret": _encode_float(metrics.current_segment_regret),
        "first_completed_segment_reward": [
            None if value is None else _encode_float(value)
            for value in metrics.first_completed_segment_reward
        ],
        "latest_completed_segment_reward": [
            None if value is None else _encode_float(value)
            for value in metrics.latest_completed_segment_reward
        ],
    }


_METRICS_KEYS = {
    "schema",
    "config_sha256",
    "accepted_events",
    "parameter_change_events",
    "reward_sum",
    "oracle_reward_sum",
    "regret_sum",
    "phase_event_counts",
    "phase_reward_sums",
    "phase_regret_sums",
    "phase_switches",
    "current_phase",
    "current_segment_events",
    "current_segment_reward",
    "current_segment_regret",
    "first_completed_segment_reward",
    "latest_completed_segment_reward",
}


def _decode_pair(
    value: Any,
    *,
    path: str,
    decode: Any,
) -> tuple[Any, Any]:
    if type(value) is not list or len(value) != 2:
        _fail(f"{path} must be a two-element JSON array")
    return decode(value[0], path=f"{path}[0]"), decode(value[1], path=f"{path}[1]")


def _decode_metrics(value: Any, *, path: str) -> ReferenceLifeMetricsState:
    data = _require_keys(value, _METRICS_KEYS, path=path)

    def decode_int(item: Any, *, path: str) -> int:
        return _require_int(item, path=path)

    def decode_optional(item: Any, *, path: str) -> float | None:
        return None if item is None else _decode_float(item, path=path)

    return ReferenceLifeMetricsState(
        schema=_require_str(data["schema"], path=f"{path}.schema"),
        config_sha256=_require_digest(data["config_sha256"], path=f"{path}.config_sha256"),
        accepted_events=_require_int(data["accepted_events"], path=f"{path}.accepted_events"),
        parameter_change_events=_require_int(
            data["parameter_change_events"], path=f"{path}.parameter_change_events"
        ),
        reward_sum=_decode_float(data["reward_sum"], path=f"{path}.reward_sum"),
        oracle_reward_sum=_decode_float(
            data["oracle_reward_sum"], path=f"{path}.oracle_reward_sum"
        ),
        regret_sum=_decode_float(data["regret_sum"], path=f"{path}.regret_sum"),
        phase_event_counts=cast(
            tuple[int, int],
            _decode_pair(
                data["phase_event_counts"], path=f"{path}.phase_event_counts", decode=decode_int
            ),
        ),
        phase_reward_sums=cast(
            tuple[float, float],
            _decode_pair(
                data["phase_reward_sums"], path=f"{path}.phase_reward_sums", decode=_decode_float
            ),
        ),
        phase_regret_sums=cast(
            tuple[float, float],
            _decode_pair(
                data["phase_regret_sums"], path=f"{path}.phase_regret_sums", decode=_decode_float
            ),
        ),
        phase_switches=_require_int(data["phase_switches"], path=f"{path}.phase_switches"),
        current_phase=_require_int(data["current_phase"], path=f"{path}.current_phase"),
        current_segment_events=_require_int(
            data["current_segment_events"], path=f"{path}.current_segment_events"
        ),
        current_segment_reward=_decode_float(
            data["current_segment_reward"], path=f"{path}.current_segment_reward"
        ),
        current_segment_regret=_decode_float(
            data["current_segment_regret"], path=f"{path}.current_segment_regret"
        ),
        first_completed_segment_reward=cast(
            tuple[float | None, float | None],
            _decode_pair(
                data["first_completed_segment_reward"],
                path=f"{path}.first_completed_segment_reward",
                decode=decode_optional,
            ),
        ),
        latest_completed_segment_reward=cast(
            tuple[float | None, float | None],
            _decode_pair(
                data["latest_completed_segment_reward"],
                path=f"{path}.latest_completed_segment_reward",
                decode=decode_optional,
            ),
        ),
    )


def _encode_transaction(state: ReferenceTransactionState) -> dict[str, Any]:
    if state.phase is not TransactionPhase.ARMED or state.decision is None:
        raise ValueError("whole-life checkpoints require one armed transaction")
    return {
        "schema": state.schema,
        "manifest_id": state.manifest_id,
        "phase": state.phase.value,
        "lifecycle_id": state.lifecycle_id,
        "next_decision_index": state.next_decision_index,
        "decision": _encode_decision(state.decision),
        "authorization": None,
        "dispatch": None,
        "command": None,
        "receipt": None,
        "transaction": None,
        "halt_reason": None,
    }


def _decode_transaction(value: Any, *, path: str) -> ReferenceTransactionState:
    data = _require_keys(
        value,
        {
            "schema",
            "manifest_id",
            "phase",
            "lifecycle_id",
            "next_decision_index",
            "decision",
            "authorization",
            "dispatch",
            "command",
            "receipt",
            "transaction",
            "halt_reason",
        },
        path=path,
    )
    phase = _require_str(data["phase"], path=f"{path}.phase")
    if phase != TransactionPhase.ARMED.value:
        _fail(f"{path}.phase must be {TransactionPhase.ARMED.value!r}")
    for name in (
        "authorization",
        "dispatch",
        "command",
        "receipt",
        "transaction",
        "halt_reason",
    ):
        if data[name] is not None:
            _fail(f"{path}.{name} must be null at an armed checkpoint boundary")
    lifecycle_id = data["lifecycle_id"]
    if lifecycle_id is None:
        _fail(f"{path}.lifecycle_id cannot be null")
    return ReferenceTransactionState(
        schema=_require_str(data["schema"], path=f"{path}.schema"),
        manifest_id=_require_digest(data["manifest_id"], path=f"{path}.manifest_id"),
        phase=TransactionPhase.ARMED,
        lifecycle_id=_require_str(lifecycle_id, path=f"{path}.lifecycle_id"),
        next_decision_index=_require_int(
            data["next_decision_index"], path=f"{path}.next_decision_index"
        ),
        decision=_decode_decision(data["decision"], path=f"{path}.decision"),
    )


def _encode_environment_state(state: Any) -> dict[str, Any]:
    if isinstance(state, SwitchingTwoStateState):
        implementation_id = SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID
        state_schema = SWITCHING_ENVIRONMENT_STATE_SCHEMA
    elif isinstance(state, RiverSwimState):
        implementation_id = RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID
        state_schema = RIVERSWIM_ENVIRONMENT_STATE_SCHEMA
    else:
        raise ValueError("whole-life checkpoint environment state is unsupported")
    return {
        "implementation_id": implementation_id,
        "schema": state_schema,
        "state_index": _encode_jax_array(
            state.state_index,
            path="environment.state_index",
        ),
        "step_count": _encode_jax_array(
            state.step_count,
            path="environment.step_count",
        ),
    }


def _encode_state(state: ReferenceLifeState) -> dict[str, Any]:
    if state.phase is not LifePhase.QUIESCENT:
        raise ValueError("whole-life checkpoints require quiescent state")
    if not isinstance(state.agent_state, PrototypeReferenceState):
        raise ValueError("whole-life checkpoint agent wrapper is unsupported")
    wrapper = state.agent_state
    dispatch = state.dispatch_state
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA,
        "life_state_schema": state.schema,
        "config_sha256": state.config_sha256,
        "lifecycle_id": state.lifecycle_id,
        "phase": state.phase.value,
        "runner": {
            "accepted_events": state.accepted_events,
            "dispatch_attempts": state.dispatch_attempts,
            "executed_events": state.executed_events,
            "environment_rng_cursor": state.environment_rng_cursor,
            "commit_generation": state.commit_generation,
            "checkpoint_generation": state.checkpoint_generation,
            "transcript_sha256": state.transcript_sha256,
        },
        "agent": {
            "schema": wrapper.schema,
            "manifest_id": wrapper.manifest_id,
            "config_sha256": wrapper.config_sha256,
            "lifecycle_id": wrapper.lifecycle_id,
            "decision_index": wrapper.decision_index,
            "current_observation_id": wrapper.current_observation_id,
        },
        "environment": _encode_environment_state(state.environment_state),
        "transaction": _encode_transaction(state.transaction_state),
        "dispatch": {
            "schema": dispatch.schema,
            "manifest_id": dispatch.manifest_id,
            "authorizations": dispatch.authorizations,
            "commands_issued": dispatch.commands_issued,
            "receipts_recorded": dispatch.receipts_recorded,
            "last_command_id": dispatch.last_command_id,
        },
        "metrics": _encode_metrics(state.metrics),
        "pending_outcome": None,
        "halt": None,
    }


_STATE_KEYS = {
    "schema",
    "life_state_schema",
    "config_sha256",
    "lifecycle_id",
    "phase",
    "runner",
    "agent",
    "environment",
    "transaction",
    "dispatch",
    "metrics",
    "pending_outcome",
    "halt",
}


def _decode_state(
    value: Any,
    *,
    runner: ReferenceLifeRunner,
    prototype_state: PrototypeAgentState,
) -> ReferenceLifeState:
    data = _require_keys(value, _STATE_KEYS, path="life_state")
    if _require_str(data["schema"], path="life_state.schema") != (
        REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA
    ):
        _fail("life_state.schema is unsupported")
    if _require_str(data["life_state_schema"], path="life_state.life_state_schema") != (
        REFERENCE_LIFE_STATE_SCHEMA
    ):
        _fail("life_state.life_state_schema is unsupported")
    if _require_str(data["phase"], path="life_state.phase") != LifePhase.QUIESCENT.value:
        _fail("life_state.phase must be quiescent")
    if data["pending_outcome"] is not None or data["halt"] is not None:
        _fail("quiescent checkpoint state cannot carry recovery data")

    runner_data = _require_keys(
        data["runner"],
        {
            "accepted_events",
            "dispatch_attempts",
            "executed_events",
            "environment_rng_cursor",
            "commit_generation",
            "checkpoint_generation",
            "transcript_sha256",
        },
        path="life_state.runner",
    )
    wrapper_data = _require_keys(
        data["agent"],
        {
            "schema",
            "manifest_id",
            "config_sha256",
            "lifecycle_id",
            "decision_index",
            "current_observation_id",
        },
        path="life_state.agent",
    )
    environment_data = _require_keys(
        data["environment"],
        {"implementation_id", "schema", "state_index", "step_count"},
        path="life_state.environment",
    )
    environment_implementation_id = _require_str(
        environment_data["implementation_id"],
        path="life_state.environment.implementation_id",
    )
    environment_state_schema = _require_str(
        environment_data["schema"], path="life_state.environment.schema"
    )
    live_environment_manifest = runner.environment_adapter.manifest
    if (
        environment_implementation_id != live_environment_manifest.implementation_id
        or environment_state_schema != live_environment_manifest.state_schema
    ):
        _fail("life_state environment discriminator differs from the live manifest")
    state_index = _decode_jax_array(
        environment_data["state_index"], path="life_state.environment.state_index"
    )
    step_count = _decode_jax_array(
        environment_data["step_count"], path="life_state.environment.step_count"
    )
    if (
        environment_implementation_id == SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID
        and environment_state_schema == SWITCHING_ENVIRONMENT_STATE_SCHEMA
    ):
        environment_state: Any = SwitchingTwoStateState(
            state_index=state_index,
            step_count=step_count,
        )  # type: ignore[call-arg]
    elif (
        environment_implementation_id == RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID
        and environment_state_schema == RIVERSWIM_ENVIRONMENT_STATE_SCHEMA
    ):
        environment_state = RiverSwimState(
            state_index=state_index,
            step_count=step_count,
        )  # type: ignore[call-arg]
    else:
        _fail("life_state environment discriminator is unsupported or crossed")

    dispatch_data = _require_keys(
        data["dispatch"],
        {
            "schema",
            "manifest_id",
            "authorizations",
            "commands_issued",
            "receipts_recorded",
            "last_command_id",
        },
        path="life_state.dispatch",
    )
    last_command = dispatch_data["last_command_id"]
    dispatch_state = ExactDispatchState(
        schema=_require_str(dispatch_data["schema"], path="life_state.dispatch.schema"),
        manifest_id=_require_digest(
            dispatch_data["manifest_id"], path="life_state.dispatch.manifest_id"
        ),
        authorizations=_require_int(
            dispatch_data["authorizations"], path="life_state.dispatch.authorizations"
        ),
        commands_issued=_require_int(
            dispatch_data["commands_issued"], path="life_state.dispatch.commands_issued"
        ),
        receipts_recorded=_require_int(
            dispatch_data["receipts_recorded"], path="life_state.dispatch.receipts_recorded"
        ),
        last_command_id=(
            None
            if last_command is None
            else _require_str(last_command, path="life_state.dispatch.last_command_id")
        ),
    )

    current_observation = wrapper_data["current_observation_id"]
    if current_observation is None:
        _fail("life_state.agent.current_observation_id cannot be null")
    adapter = runner.agent_adapter
    if not isinstance(adapter, PrototypeReferenceAdapter):
        _fail("restored life does not use the Prototype adapter")
    wrapper_state = adapter.adopt_checkpoint_state(
        agent_state=prototype_state,
        lifecycle_id=_require_str(
            wrapper_data["lifecycle_id"], path="life_state.agent.lifecycle_id"
        ),
        decision_index=_require_int(
            wrapper_data["decision_index"], path="life_state.agent.decision_index"
        ),
        current_observation_id=_require_str(
            current_observation,
            path="life_state.agent.current_observation_id",
        ),
    )
    if (
        wrapper_state.schema != _require_str(wrapper_data["schema"], path="life_state.agent.schema")
        or wrapper_state.manifest_id
        != _require_digest(wrapper_data["manifest_id"], path="life_state.agent.manifest_id")
        or wrapper_state.config_sha256
        != _require_digest(wrapper_data["config_sha256"], path="life_state.agent.config_sha256")
    ):
        _fail("restored Prototype wrapper identity is inconsistent")

    result = ReferenceLifeState(
        schema=REFERENCE_LIFE_STATE_SCHEMA,
        config_sha256=_require_digest(data["config_sha256"], path="life_state.config_sha256"),
        lifecycle_id=_require_str(data["lifecycle_id"], path="life_state.lifecycle_id"),
        phase=LifePhase.QUIESCENT,
        accepted_events=_require_int(
            runner_data["accepted_events"], path="life_state.runner.accepted_events"
        ),
        dispatch_attempts=_require_int(
            runner_data["dispatch_attempts"], path="life_state.runner.dispatch_attempts"
        ),
        executed_events=_require_int(
            runner_data["executed_events"], path="life_state.runner.executed_events"
        ),
        environment_rng_cursor=_require_int(
            runner_data["environment_rng_cursor"],
            path="life_state.runner.environment_rng_cursor",
        ),
        commit_generation=_require_int(
            runner_data["commit_generation"], path="life_state.runner.commit_generation"
        ),
        checkpoint_generation=_require_int(
            runner_data["checkpoint_generation"],
            path="life_state.runner.checkpoint_generation",
        ),
        agent_state=wrapper_state,
        environment_state=environment_state,
        transaction_state=_decode_transaction(data["transaction"], path="life_state.transaction"),
        dispatch_state=dispatch_state,
        metrics=_decode_metrics(data["metrics"], path="life_state.metrics"),
        pending_outcome=None,
        halt=None,
        transcript_sha256=_require_digest(
            runner_data["transcript_sha256"], path="life_state.runner.transcript_sha256"
        ),
    )
    runner.validate_checkpoint_state(result)
    return result


_LIFE_CONFIG_KEYS = {
    "schema",
    "lifecycle_id",
    "seed",
    "max_accepted_events",
    "rng_schedule",
    "observation_id_schedule",
    "checkpoint_policy",
    "agent",
    "environment",
    "dispatch",
    "metrics",
}


def _build_runner(config_value: Any) -> tuple[ReferenceLifeRunner, dict[str, Any]]:
    config = _require_keys(config_value, _LIFE_CONFIG_KEYS, path="manifest.life_config")
    if _require_str(config["schema"], path="manifest.life_config.schema") != (
        REFERENCE_LIFE_CONFIG_SCHEMA
    ):
        _fail("manifest.life_config.schema is unsupported")
    schedules = {
        "rng_schedule": REFERENCE_LIFE_RNG_SCHEDULE,
        "observation_id_schedule": "lifecycle_observation_index.preview1",
        "checkpoint_policy": "atomic_quiescent_bundle.preview1",
    }
    for name, expected in schedules.items():
        if _require_str(config[name], path=f"manifest.life_config.{name}") != expected:
            _fail(f"manifest.life_config.{name} is unsupported")

    agent_descriptor = _require_keys(
        config["agent"],
        {
            "schema",
            "implementation_id",
            "state_schema",
            "config_sha256",
            "manifest_id",
            "config",
            "observation_spec",
            "action_spec",
            "capabilities",
            "max_accepted_events",
        },
        path="manifest.life_config.agent",
    )
    agent_binding = _require_keys(
        agent_descriptor["config"],
        {
            "adapter_schema",
            "dispatch_mode",
            "environment_mode",
            "lifecycle_codec",
            "prototype_agent",
            "prototype_state_schema",
            "reward_discount_codec",
        },
        path="manifest.life_config.agent.config",
    )
    prototype_config_raw = agent_binding["prototype_agent"]
    if type(prototype_config_raw) is not dict:
        _fail("manifest.life_config.agent.config.prototype_agent must be an object")
    prototype_config = PrototypeAgentConfig.from_config(cast(dict[str, Any], prototype_config_raw))

    environment_descriptor = _require_keys(
        config["environment"],
        {
            "schema",
            "implementation_id",
            "state_schema",
            "config_sha256",
            "manifest_id",
            "config",
            "observation_spec",
            "action_spec",
            "max_executions",
        },
        path="manifest.life_config.environment",
    )
    environment_implementation_id = _require_str(
        environment_descriptor["implementation_id"],
        path="manifest.life_config.environment.implementation_id",
    )
    environment_state_schema = _require_str(
        environment_descriptor["state_schema"],
        path="manifest.life_config.environment.state_schema",
    )
    environment_discriminator = (
        environment_implementation_id,
        environment_state_schema,
    )
    switching_discriminator = (
        SWITCHING_ENVIRONMENT_IMPLEMENTATION_ID,
        SWITCHING_ENVIRONMENT_STATE_SCHEMA,
    )
    riverswim_discriminator = (
        RIVERSWIM_ENVIRONMENT_IMPLEMENTATION_ID,
        RIVERSWIM_ENVIRONMENT_STATE_SCHEMA,
    )
    if environment_discriminator not in {
        switching_discriminator,
        riverswim_discriminator,
    }:
        _fail("manifest environment implementation/state discriminator is unsupported or crossed")

    def payoff_matrix(value: Any, *, path: str) -> tuple[tuple[float, float], ...]:
        if type(value) is not list or len(value) != 2:
            _fail(f"{path} must be a two-by-two array")
        rows: list[tuple[float, float]] = []
        for index, row in enumerate(value):
            if type(row) is not list or len(row) != 2:
                _fail(f"{path}[{index}] must contain two values")
            rows.append(
                (
                    _require_float(row[0], path=f"{path}[{index}][0]"),
                    _require_float(row[1], path=f"{path}[{index}][1]"),
                )
            )
        return tuple(rows)

    environment_path = "manifest.life_config.environment.config"
    environment_raw = environment_descriptor["config"]
    if environment_discriminator == switching_discriminator:
        environment = _require_keys(
            environment_raw,
            {
                "environment_kind",
                "metrics_mode",
                "phase_length",
                "payoffs_a",
                "payoffs_b",
                "executor_id",
                "executor_epoch",
                "dynamics",
                "rng_mode",
                "boundary_mode",
            },
            path=environment_path,
        )
        expected_strings = {
            "environment_kind": "switching_two_state",
            "metrics_mode": "switching_two_phase",
            "dynamics": "next_state_equals_action",
            "rng_mode": "key_schedule_bound_but_dynamics_deterministic",
            "boundary_mode": "continuing_only",
        }
        for name, expected in expected_strings.items():
            if _require_str(environment[name], path=f"{environment_path}.{name}") != expected:
                _fail(f"{environment_path}.{name} is unsupported")
        environment_config: SwitchingTwoStateConfig | RiverSwimConfig = (
            SwitchingTwoStateConfig(
                phase_length=_require_int(
                    environment["phase_length"],
                    path=f"{environment_path}.phase_length",
                    minimum=1,
                ),
                payoffs_a=payoff_matrix(
                    environment["payoffs_a"],
                    path=f"{environment_path}.payoffs_a",
                ),
                payoffs_b=payoff_matrix(
                    environment["payoffs_b"],
                    path=f"{environment_path}.payoffs_b",
                ),
            )  # type: ignore[call-arg]
        )
        environment_metrics_mode = "switching_two_phase"
    else:
        environment = _require_keys(
            environment_raw,
            {
                "environment_kind",
                "metrics_mode",
                "n_states",
                "p_right_up",
                "p_right_down",
                "reward_left",
                "reward_right",
                "initial_state",
                "oracle_average_reward",
                "executor_id",
                "executor_epoch",
                "dynamics",
                "rng_mode",
                "boundary_mode",
            },
            path=environment_path,
        )
        expected_strings = {
            "environment_kind": "riverswim",
            "metrics_mode": "stationary",
            "dynamics": "riverswim_stochastic_chain",
            "rng_mode": "exact_jax_key_replay",
            "boundary_mode": "continuing_only",
        }
        for name, expected in expected_strings.items():
            if _require_str(environment[name], path=f"{environment_path}.{name}") != expected:
                _fail(f"{environment_path}.{name} is unsupported")
        n_states = _require_int(
            environment["n_states"],
            path=f"{environment_path}.n_states",
            minimum=2,
        )
        if n_states > RIVERSWIM_REFERENCE_MAX_STATES:
            _fail(
                f"{environment_path}.n_states exceeds the RiverSwim reference bound"
            )
        initial_state = _require_int(
            environment["initial_state"],
            path=f"{environment_path}.initial_state",
        )
        if initial_state >= n_states:
            _fail(f"{environment_path}.initial_state is outside the state space")
        _require_float(
            environment["oracle_average_reward"],
            path=f"{environment_path}.oracle_average_reward",
        )
        environment_config = RiverSwimConfig(
            n_states=n_states,
            p_right_up=_require_float(
                environment["p_right_up"], path=f"{environment_path}.p_right_up"
            ),
            p_right_down=_require_float(
                environment["p_right_down"], path=f"{environment_path}.p_right_down"
            ),
            reward_left=_require_float(
                environment["reward_left"], path=f"{environment_path}.reward_left"
            ),
            reward_right=_require_float(
                environment["reward_right"], path=f"{environment_path}.reward_right"
            ),
            initial_state=initial_state,
        )  # type: ignore[call-arg]
        environment_metrics_mode = "stationary"

    _require_str(
        environment["executor_id"],
        path=f"{environment_path}.executor_id",
    )
    _require_str(
        environment["executor_epoch"],
        path=f"{environment_path}.executor_epoch",
    )

    dispatch_descriptor = _require_keys(
        config["dispatch"],
        {"schema", "config_sha256", "manifest_id", "config", "max_dispatches"},
        path="manifest.life_config.dispatch",
    )
    dispatch = _require_keys(
        dispatch_descriptor["config"],
        {
            "mode",
            "authority_role",
            "safety_policy",
            "veto_conformance",
            "authority_id",
            "policy_version",
            "executor_id",
            "executor_epoch",
        },
        path="manifest.life_config.dispatch.config",
    )
    dispatch_config = ExactDispatchConfig(
        authority_id=_require_str(
            dispatch["authority_id"],
            path="manifest.life_config.dispatch.config.authority_id",
        ),
        policy_version=_require_str(
            dispatch["policy_version"],
            path="manifest.life_config.dispatch.config.policy_version",
        ),
        executor_id=_require_str(
            dispatch["executor_id"],
            path="manifest.life_config.dispatch.config.executor_id",
        ),
        executor_epoch=_require_str(
            dispatch["executor_epoch"],
            path="manifest.life_config.dispatch.config.executor_epoch",
        ),
        mode=_require_str(dispatch["mode"], path="manifest.life_config.dispatch.config.mode"),
    )

    metrics_descriptor = _require_keys(
        config["metrics"],
        {"schema", "config_sha256", "config"},
        path="manifest.life_config.metrics",
    )
    metrics = _require_keys(
        metrics_descriptor["config"],
        {"mode", "oracle_regret"},
        path="manifest.life_config.metrics.config",
    )
    metrics_mode = _require_str(
        metrics["mode"],
        path="manifest.life_config.metrics.config.mode",
    )
    if metrics_mode != environment_metrics_mode:
        _fail("manifest metrics mode does not match the environment")
    metrics_config = ReferenceLifeMetricsConfig(
        mode=metrics_mode,
        oracle_regret=_require_bool(
            metrics["oracle_regret"],
            path="manifest.life_config.metrics.config.oracle_regret",
        )
    )

    lifecycle_id = _require_str(
        config["lifecycle_id"], path="manifest.life_config.lifecycle_id"
    )
    seed = _require_int(config["seed"], path="manifest.life_config.seed")
    max_accepted_events = _require_int(
        config["max_accepted_events"],
        path="manifest.life_config.max_accepted_events",
        minimum=1,
    )
    if environment_discriminator == switching_discriminator:
        runner = build_prototype_switching_life(
            agent_config=prototype_config,
            environment_config=cast(SwitchingTwoStateConfig, environment_config),
            lifecycle_id=lifecycle_id,
            seed=seed,
            max_accepted_events=max_accepted_events,
            dispatch_config=dispatch_config,
            metrics_config=metrics_config,
        )
    else:
        runner = build_prototype_riverswim_life(
            agent_config=prototype_config,
            environment_config=cast(RiverSwimConfig, environment_config),
            lifecycle_id=lifecycle_id,
            seed=seed,
            max_accepted_events=max_accepted_events,
            dispatch_config=dispatch_config,
            metrics_config=metrics_config,
        )
    if _canonical_json_bytes(runner.config.config) != _canonical_json_bytes(config):
        _fail("manifest.life_config is not the canonical reconstructed configuration")
    return runner, cast(dict[str, Any], prototype_config_raw)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _walk_regular_files(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"checkpoint directory is unavailable: {root}")
    files: dict[str, Path] = {}
    entries_seen = 0
    total_bytes = 0

    def visit(directory: Path, *, depth: int) -> None:
        nonlocal entries_seen, total_bytes
        if depth > _MAX_TREE_DEPTH:
            raise ValueError("checkpoint directory tree exceeds its depth limit")
        with os.scandir(directory) as entries:
            for entry in entries:
                entries_seen += 1
                if entries_seen > _MAX_INVENTORY_FILES:
                    raise ValueError("checkpoint directory tree exceeds its entry limit")
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if len(relative.encode("utf-8")) > _MAX_PATH_BYTES:
                    raise ValueError("checkpoint directory tree contains an overlong path")
                if entry.is_symlink():
                    raise ValueError(f"checkpoint cannot contain symlink {relative!r}")
                if entry.is_dir(follow_symlinks=False):
                    visit(path, depth=depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    size = entry.stat(follow_symlinks=False).st_size
                    if size > _MAX_CHILD_FILE_BYTES:
                        raise ValueError(
                            f"checkpoint child exceeds its per-file byte limit: {relative}"
                        )
                    total_bytes += size
                    if total_bytes > _MAX_CHILD_TOTAL_BYTES:
                        raise ValueError("checkpoint children exceed their total byte limit")
                    files[relative] = path
                else:
                    raise ValueError(f"checkpoint cannot contain non-regular entry {relative!r}")

    visit(root, depth=0)
    return files


def _child_hashes(root: Path) -> dict[str, Any]:
    files = _walk_regular_files(root)
    selected = {
        name: path
        for name, path in files.items()
        if name == _STATE_NAME or name.startswith(f"{_PROTOTYPE_NAME}/")
    }
    if _STATE_NAME not in selected or not any(
        name.startswith(f"{_PROTOTYPE_NAME}/") for name in selected
    ):
        raise ValueError("checkpoint is missing a required semantic child")
    encoded: dict[str, dict[str, Any]] = {}
    for name, path in sorted(selected.items()):
        digest, size = _sha256_file(path)
        encoded[name] = {"sha256": digest, "size": size}
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_HASHES_SCHEMA,
        "files": encoded,
    }


def _verify_child_hashes(root: Path, value: Any) -> None:
    data = _require_keys(value, {"schema", "files"}, path="manifest.children")
    if _require_str(data["schema"], path="manifest.children.schema") != (
        REFERENCE_LIFE_CHECKPOINT_HASHES_SCHEMA
    ):
        _fail("manifest.children.schema is unsupported")
    files_raw = data["files"]
    if type(files_raw) is not dict:
        _fail("manifest.children.files must be an object")
    if len(files_raw) > _MAX_INVENTORY_FILES:
        _fail("manifest.children.files exceeds the inventory limit")
    declared: dict[str, tuple[str, int]] = {}
    for name, raw in cast(dict[str, Any], files_raw).items():
        if (
            type(name) is not str
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or name == _MANIFEST_NAME
            or name == _COMMITTED_NAME
        ):
            _fail("manifest.children contains an unsafe path")
        if len(name.encode("utf-8")) > _MAX_PATH_BYTES:
            _fail("manifest.children contains an overlong path")
        entry = _require_keys(raw, {"sha256", "size"}, path=f"manifest.children.files.{name}")
        declared[name] = (
            _require_digest(entry["sha256"], path=f"manifest.children.files.{name}.sha256"),
            _require_int(entry["size"], path=f"manifest.children.files.{name}.size"),
        )
    actual = _child_hashes(root)["files"]
    actual_pairs = {
        name: (entry["sha256"], entry["size"])
        for name, entry in cast(dict[str, dict[str, Any]], actual).items()
    }
    if declared != actual_pairs:
        _fail("checkpoint child inventory or content hash does not match")


def _fsync_tree(root: Path) -> None:
    files = _walk_regular_files(root)
    for path in files.values():
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


_MANIFEST_PAYLOAD_KEYS = {
    "schema",
    "state_schema",
    "prototype_schema",
    "checkpoint_generation",
    "config_sha256",
    "life_config",
    "source_identity",
    "runtime_identity",
    "dependency_identity",
    "children",
}
_MANIFEST_KEYS = {*_MANIFEST_PAYLOAD_KEYS, "bundle_id"}
_PROTOTYPE_METADATA_KEYS = {
    "schema",
    "agent_config",
    "config_sha256",
    "empty_array_codec",
    "prng_impl",
}
_PROTOTYPE_RAW_METADATA_KEYS = {*_PROTOTYPE_METADATA_KEYS, "_format_version"}


def _load_raw_prototype_metadata(path: Path) -> dict[str, Any]:
    """Strictly parse Orbax JSON without requiring its incidental formatting."""

    raw = _read_bounded_json_file(path)
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("nested Prototype metadata is not valid bounded JSON") from exc
    data = _require_keys(
        value,
        _PROTOTYPE_RAW_METADATA_KEYS,
        path="prototype.metadata",
    )
    if (
        _require_int(
            data["_format_version"],
            path="prototype.metadata._format_version",
        )
        != 2
    ):
        _fail("nested Prototype metadata format version is unsupported")
    if _require_str(data["schema"], path="prototype.metadata.schema") != (
        PROTOTYPE_CHECKPOINT_SCHEMA
    ):
        _fail("nested Prototype checkpoint is not exact current v3")
    agent_config = data["agent_config"]
    if type(agent_config) is not dict:
        _fail("prototype.metadata.agent_config must be an object")
    expected_config_digest = hashlib.sha256(
        json.dumps(
            agent_config,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        _require_digest(
            data["config_sha256"],
            path="prototype.metadata.config_sha256",
        )
        != expected_config_digest
    ):
        _fail("nested Prototype metadata config digest does not match")
    if (
        _require_str(
            data["empty_array_codec"],
            path="prototype.metadata.empty_array_codec",
        )
        != _PROTOTYPE_EMPTY_ARRAY_CODEC
    ):
        _fail("nested Prototype metadata uses an unknown empty-array codec")
    if _require_str(data["prng_impl"], path="prototype.metadata.prng_impl") not in (
        _PROTOTYPE_SUPPORTED_PRNG_IMPLS
    ):
        _fail("nested Prototype metadata uses an unsupported PRNG implementation")
    return {name: data[name] for name in _PROTOTYPE_METADATA_KEYS}


def _bundle_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _manifest_payload(
    *,
    runner: ReferenceLifeRunner,
    state: ReferenceLifeState,
    root: Path,
) -> dict[str, Any]:
    return {
        "schema": REFERENCE_LIFE_CHECKPOINT_SCHEMA,
        "state_schema": REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA,
        "prototype_schema": PROTOTYPE_CHECKPOINT_SCHEMA,
        "checkpoint_generation": state.checkpoint_generation,
        "config_sha256": runner.config.config_sha256,
        "life_config": runner.config.config,
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "dependency_identity": _dependency_identity(),
        "children": _child_hashes(root),
    }


def _validate_identity(value: Any, expected: dict[str, Any], *, path: str) -> None:
    if type(value) is not dict:
        _fail(f"{path} must be an object")
    if path == "manifest.source_identity":
        source = _require_keys(value, {"schema", "scope", "files"}, path=path)
        files = source["files"]
        if type(files) is not dict or len(files) > _MAX_INVENTORY_FILES:
            _fail(f"{path}.files exceeds the inventory limit")
        for name in cast(dict[str, Any], files):
            if len(name.encode("utf-8")) > _MAX_PATH_BYTES:
                _fail(f"{path}.files contains an overlong path")
    if value != expected:
        _fail(f"{path} does not match the current checkpoint-time identity")


def _validate_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be a real directory, not a symlink")
    expected = {_MANIFEST_NAME, _STATE_NAME, _PROTOTYPE_NAME, _COMMITTED_NAME}
    with os.scandir(root) as entries:
        actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError("checkpoint root entries do not match the bundle schema")
    for name in (_MANIFEST_NAME, _STATE_NAME, _COMMITTED_NAME):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint {name} must be a regular file")
    prototype = root / _PROTOTYPE_NAME
    if prototype.is_symlink() or not prototype.is_dir():
        raise ValueError("checkpoint prototype child must be a real directory")
    _walk_regular_files(root)


def _trees_exact(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if cast(Any, left_structure) != right_structure or len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if not isinstance(left_leaf, jax.Array) or not isinstance(right_leaf, jax.Array):
            return False
        if left_leaf.shape != right_leaf.shape or left_leaf.dtype != right_leaf.dtype:
            return False
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            left_leaf.dtype, jax.dtypes.prng_key
        ):
            if str(jax.random.key_impl(left_leaf)) != str(jax.random.key_impl(right_leaf)):
                return False
            left_array = np.asarray(jax.random.key_data(left_leaf))
            right_array = np.asarray(jax.random.key_data(right_leaf))
            if (
                left_array.dtype != right_array.dtype
                or left_array.shape != right_array.shape
                or left_array.tobytes(order="C") != right_array.tobytes(order="C")
            ):
                return False
        else:
            left_array = np.asarray(left_leaf)
            right_array = np.asarray(right_leaf)
            if left_array.tobytes(order="C") != right_array.tobytes(order="C"):
                return False
    return True


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one directory without replacing any existing entry."""

    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "atomic no-replace rename requires Linux renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOTSUP, "libc does not expose renameat2") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _semantic_states_exact(left: ReferenceLifeState, right: ReferenceLifeState) -> bool:
    if _encode_state(left) != _encode_state(right):
        return False
    left_wrapper = cast(PrototypeReferenceState, left.agent_state)
    right_wrapper = cast(PrototypeReferenceState, right.agent_state)
    return _trees_exact(left_wrapper.agent_state, right_wrapper.agent_state)


def _snapshot_checkpoint(source: Path) -> Path:
    """Copy an untrusted bundle into one bounded, private, stable directory."""

    snapshot = Path(tempfile.mkdtemp(prefix=".reference-life-restore-"))
    source_descriptor = -1
    destination_descriptor = -1
    entries_seen = 0
    total_bytes = 0
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_descriptor = os.open(
            snapshot,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )

        def copy_directory(
            source_fd: int,
            destination_fd: int,
            *,
            relative: str,
            depth: int,
        ) -> None:
            nonlocal entries_seen, total_bytes
            if depth > _MAX_TREE_DEPTH:
                raise ValueError("checkpoint directory tree exceeds its depth limit")
            with os.scandir(source_fd) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_INVENTORY_FILES:
                        raise ValueError("checkpoint directory tree exceeds its entry limit")
                    name = entry.name
                    child_relative = f"{relative}/{name}" if relative else name
                    if len(child_relative.encode("utf-8")) > _MAX_PATH_BYTES:
                        raise ValueError("checkpoint directory tree contains an overlong path")
                    if entry.is_symlink():
                        raise ValueError(f"checkpoint cannot contain symlink {child_relative!r}")
                    if entry.is_dir(follow_symlinks=False):
                        child_source = os.open(
                            name,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=source_fd,
                        )
                        os.mkdir(name, mode=0o700, dir_fd=destination_fd)
                        child_destination = os.open(
                            name,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=destination_fd,
                        )
                        try:
                            copy_directory(
                                child_source,
                                child_destination,
                                relative=child_relative,
                                depth=depth + 1,
                            )
                        finally:
                            os.close(child_source)
                            os.close(child_destination)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError(
                            f"checkpoint contains non-regular entry {child_relative!r}"
                        )
                    child_source = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=source_fd,
                    )
                    try:
                        metadata = os.fstat(child_source)
                        if not stat_module.S_ISREG(metadata.st_mode):
                            raise ValueError(
                                f"checkpoint contains non-regular entry {child_relative!r}"
                            )
                        if metadata.st_size > _MAX_CHILD_FILE_BYTES:
                            raise ValueError(
                                f"checkpoint child exceeds its byte limit: {child_relative}"
                            )
                        total_bytes += metadata.st_size
                        if total_bytes > _MAX_CHILD_TOTAL_BYTES:
                            raise ValueError("checkpoint children exceed their total byte limit")
                        child_destination = os.open(
                            name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=destination_fd,
                        )
                        copied = 0
                        try:
                            while True:
                                chunk = os.read(
                                    child_source,
                                    min(
                                        1024 * 1024,
                                        _MAX_CHILD_FILE_BYTES - copied + 1,
                                    ),
                                )
                                if not chunk:
                                    break
                                copied += len(chunk)
                                if copied > _MAX_CHILD_FILE_BYTES:
                                    raise ValueError("checkpoint child grew beyond its byte limit")
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(child_destination, view)
                                    view = view[written:]
                            if copied != metadata.st_size:
                                raise ValueError(
                                    f"checkpoint child changed while copied: {child_relative}"
                                )
                            os.fsync(child_destination)
                        finally:
                            os.close(child_destination)
                    finally:
                        os.close(child_source)

        copy_directory(
            source_descriptor,
            destination_descriptor,
            relative="",
            depth=0,
        )
        os.fsync(destination_descriptor)
        return snapshot
    except Exception:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _restore_owned_bundle(root: Path) -> tuple[ReferenceLifeRunner, ReferenceLifeState]:
    _validate_root(root)
    manifest = _load_canonical_json(root / _MANIFEST_NAME)
    manifest_data = _require_keys(manifest, _MANIFEST_KEYS, path="manifest")
    if _require_str(manifest_data["schema"], path="manifest.schema") != (
        REFERENCE_LIFE_CHECKPOINT_SCHEMA
    ):
        _fail("manifest.schema is unsupported")
    if _require_str(manifest_data["state_schema"], path="manifest.state_schema") != (
        REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA
    ):
        _fail("manifest.state_schema is unsupported")
    if (
        _require_str(manifest_data["prototype_schema"], path="manifest.prototype_schema")
        != PROTOTYPE_CHECKPOINT_SCHEMA
    ):
        _fail("manifest.prototype_schema must be the exact current Prototype v3 schema")
    bundle_id = _require_digest(manifest_data["bundle_id"], path="manifest.bundle_id")
    payload = {name: manifest_data[name] for name in _MANIFEST_PAYLOAD_KEYS}
    if bundle_id != _bundle_id(payload):
        _fail("manifest bundle_id does not match its canonical payload")
    marker = (root / _COMMITTED_NAME).read_bytes()
    if marker != f"{bundle_id}\n".encode("ascii"):
        _fail("checkpoint COMMITTED marker is missing or inconsistent")

    _validate_identity(
        manifest_data["source_identity"],
        _source_identity(),
        path="manifest.source_identity",
    )
    _validate_identity(
        manifest_data["runtime_identity"],
        _runtime_identity(),
        path="manifest.runtime_identity",
    )
    _validate_identity(
        manifest_data["dependency_identity"],
        _dependency_identity(),
        path="manifest.dependency_identity",
    )
    _verify_child_hashes(root, manifest_data["children"])

    runner, prototype_config = _build_runner(manifest_data["life_config"])
    config_sha256 = _require_digest(manifest_data["config_sha256"], path="manifest.config_sha256")
    if runner.config.config_sha256 != config_sha256:
        _fail("manifest configuration digest is inconsistent")
    generation = _require_int(
        manifest_data["checkpoint_generation"],
        path="manifest.checkpoint_generation",
        minimum=1,
    )

    prototype_path = root / _PROTOTYPE_NAME
    raw_prototype_metadata = _load_raw_prototype_metadata(prototype_path / "metadata" / "metadata")
    if raw_prototype_metadata["agent_config"] != prototype_config:
        _fail("nested Prototype configuration differs from the life binding")
    prototype_metadata = load_checkpoint_metadata(prototype_path)
    if set(prototype_metadata) != _PROTOTYPE_METADATA_KEYS:
        _fail("nested Prototype checkpoint metadata fields are not exact current v3")
    if prototype_metadata != raw_prototype_metadata:
        _fail("Orbax did not preserve the exact nested Prototype metadata object")
    if prototype_metadata.get("schema") != PROTOTYPE_CHECKPOINT_SCHEMA:
        _fail("nested Prototype checkpoint is not exact current v3")
    if prototype_metadata.get("agent_config") != prototype_config:
        _fail("nested Prototype configuration differs from the life binding")
    prototype_agent, prototype_state = load_prototype_checkpoint(prototype_path)
    if prototype_agent.to_config() != prototype_config:
        _fail("restored Prototype configuration differs from the life binding")
    state_json = _load_canonical_json(root / _STATE_NAME)
    state = _decode_state(state_json, runner=runner, prototype_state=prototype_state)
    if state.config_sha256 != config_sha256:
        _fail("life state configuration digest differs from the manifest")
    if state.checkpoint_generation != generation:
        _fail("life state checkpoint generation differs from the manifest")
    # The source was copied into an owned mode-0700 snapshot before reaching
    # this function. Rechecking every child after all semantic reads nevertheless
    # detects an accidental same-process mutation during nested Orbax/state load.
    _verify_child_hashes(root, manifest_data["children"])
    runner.adopt_checkpoint_state(state)
    return runner, state


def save_reference_life_checkpoint(
    runner: ReferenceLifeRunner,
    state: ReferenceLifeState,
    parent: str | Path,
) -> tuple[ReferenceLifeState, Path]:
    """Atomically publish a new immutable quiescent whole-life generation."""

    if not isinstance(runner, ReferenceLifeRunner):
        raise TypeError("runner must be a ReferenceLifeRunner")
    parent_path = Path(parent).absolute()
    markers = (_MANIFEST_NAME, _STATE_NAME, _PROTOTYPE_NAME, _COMMITTED_NAME)

    def reject_bundle_destination() -> None:
        try:
            resolved_parent = parent_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError("checkpoint parent path cannot be resolved safely") from exc
        candidates = (parent_path, resolved_parent)
        for candidate in candidates:
            proposed_markers = tuple(os.path.lexists(candidate / marker) for marker in markers)
            if any(proposed_markers):
                raise ValueError("checkpoint parent cannot be an existing bundle root")
            for ancestor in candidate.parents:
                if all(os.path.lexists(ancestor / marker) for marker in markers):
                    raise ValueError("checkpoint parent cannot be inside an existing bundle root")

    def publish(
        candidate: ReferenceLifeState,
        install_committed_candidate: Any,
    ) -> Path:
        # checkpoint_barrier invokes this callback while holding the runner lock,
        # after current-state and quiescence validation. Keep every filesystem
        # mutation inside that same transaction so a racing step cannot stale the
        # candidate after a separate preflight has already created store entries.
        reject_bundle_destination()
        if os.path.lexists(parent_path):
            if parent_path.is_symlink() or not parent_path.is_dir():
                raise ValueError("checkpoint parent must be a real directory")
        else:
            parent_path.mkdir(parents=True, exist_ok=False)
        if parent_path.is_symlink() or not parent_path.is_dir():
            raise ValueError("checkpoint parent must be a real directory")
        reject_bundle_destination()

        lock_path = parent_path / ".reference-life-checkpoint.lock"
        lock_descriptor = -1
        parent_descriptor = -1
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat_module.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise ValueError("checkpoint lock must be a regular file")
            parent_descriptor = os.open(
                parent_path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            reject_bundle_destination()
            generation_name = f"generation-{candidate.checkpoint_generation:0{_GENERATION_WIDTH}d}"
            target = parent_path / generation_name
            if os.path.lexists(target):
                raise FileExistsError(f"checkpoint generation already exists: {target}")
            staging = Path(tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=parent_path))
            renamed = False
            try:
                wrapper = candidate.agent_state
                if not isinstance(wrapper, PrototypeReferenceState):
                    raise ValueError("checkpoint candidate lacks Prototype wrapper state")
                adapter = runner.agent_adapter
                if not isinstance(adapter, PrototypeReferenceAdapter):
                    raise ValueError("checkpoint runner does not use Prototype adapter")
                save_prototype_checkpoint(
                    PrototypeAgent(adapter.config),
                    wrapper.agent_state,
                    staging / _PROTOTYPE_NAME,
                )
                encoded_state = _encode_state(candidate)
                encoded_environment = cast(
                    dict[str, Any], encoded_state["environment"]
                )
                live_environment_manifest = runner.environment_adapter.manifest
                if (
                    encoded_environment["implementation_id"]
                    != live_environment_manifest.implementation_id
                    or encoded_environment["schema"]
                    != live_environment_manifest.state_schema
                ):
                    raise ValueError(
                        "checkpoint environment state discriminator differs from live manifest"
                    )
                _write_new(
                    staging / _STATE_NAME,
                    _canonical_json_bytes(encoded_state),
                )
                payload = _manifest_payload(runner=runner, state=candidate, root=staging)
                bundle_id = _bundle_id(payload)
                manifest = {**payload, "bundle_id": bundle_id}
                _write_new(staging / _MANIFEST_NAME, _canonical_json_bytes(manifest))
                _write_new(staging / _COMMITTED_NAME, f"{bundle_id}\n".encode("ascii"))
                _fsync_tree(staging)

                restored_runner, restored_state = _restore_owned_bundle(staging)
                if restored_runner.config != runner.config or not _semantic_states_exact(
                    restored_state, candidate
                ):
                    raise ValueError("staged checkpoint failed exact semantic self-validation")
                if os.path.lexists(target):
                    raise FileExistsError(f"checkpoint generation already exists: {target}")
                _rename_noreplace(staging, target)
                install_committed_candidate()
                renamed = True
                # The rename already made the generation visible and the runner now
                # owns its barrier state. A parent-directory fsync error cannot be
                # reported as an uncommitted save after that atomic visibility point.
                try:
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
                return target
            finally:
                if not renamed:
                    shutil.rmtree(staging, ignore_errors=True)
        finally:
            if lock_descriptor >= 0:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(lock_descriptor)
                except OSError:
                    pass
            if parent_descriptor >= 0:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass

    return runner.checkpoint_barrier(state, publish)


def load_reference_life_checkpoint(
    path: str | Path,
) -> tuple[ReferenceLifeRunner, ReferenceLifeState]:
    """Validate and adopt one exact current-schema whole-life generation."""

    snapshot = _snapshot_checkpoint(Path(path).absolute())
    try:
        return _restore_owned_bundle(snapshot)
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


__all__ = [
    "REFERENCE_LIFE_CHECKPOINT_SCHEMA",
    "REFERENCE_LIFE_CHECKPOINT_STATE_SCHEMA",
    "load_reference_life_checkpoint",
    "save_reference_life_checkpoint",
]
