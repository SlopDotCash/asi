"""Standalone, nonauthorizing execution boundary for matched-v3 local cells.

An exact isolated direct-byte load of this module constructs descriptors and
empty weak capability registries only.  It does not observe the filesystem,
import Alberta/JAX/NumPy/Foragax, create scratch state, launch a process, issue
a capability, or execute a candidate.  Every path, identity, seed, ceiling,
and opt-in is supplied explicitly by the caller.

Execution uses one fresh ``-I -S -B`` child.  The child verifies the caller-
carried full source snapshot before loading the exact pinned runner bytes,
neutralizes bytecode caches, consumes the runner's own capabilities, and
verifies the same snapshot again before returning bounded content through a
parent-retained regular-file descriptor.  The process group is terminated and
the direct child is waited; descendant reaping remains external.  Endpoint
equality does not prove continuous immutability.  Runtime, dependency,
interpreter, toolchain, and hardware qualification remain external.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import types
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, cast

LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_execution_bootstrap_descriptor.v1"
)
LOCAL_EXECUTION_BOOTSTRAP_CHILD_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_execution_bootstrap_child_record.v1"
)
LOCAL_EXECUTION_BOOTSTRAP_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_execution_bootstrap_receipt.v1"
)
LOCAL_EXECUTION_BOOTSTRAP_STATUS: Final = "implemented_unexecuted_non_authorizing"
LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_execution_bootstrap_isolated_v1"
)
_CHILD_MODULE_NAME: Final = "_alberta_forager_matched_v3_local_execution_bootstrap_child_v1"
_SNAPSHOT_MODULE_NAME: Final = "_alberta_forager_matched_v3_source_snapshot_bootstrap_v1"
_PARENT_RUNNER_PARSER_MODULE_NAME: Final = "_alberta_forager_matched_v3_runner_parser_v1"

PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256: Final = (
    "2237914749f353d2700bbb0f33a66d8789268a5e156f2961be2e626f42efd2a1"
)
PINNED_LOCAL_RUNNER_SOURCE_SHA256: Final = (
    "aa2eb0fd642dec7ef62a4cb0fc555f6aaede6570a55c49adfa8425a264be91aa"
)
PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256: Final = (
    "5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"
)
PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256: Final = (
    "cfb4c9df2b0d767a40aeeba4bd044ba50c2e595054db768966105a0df9233cbb"
)

_RUNNER_RELATIVE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_local_runner.py"
_SNAPSHOT_RELATIVE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_local_source_snapshot.py"
)
_SNAPSHOT_MANIFEST_SCHEMA: Final = "alberta.forager_matched_v3.local_source_snapshot_manifest.v1"
_SNAPSHOT_TREE_SCHEMA: Final = "alberta.forager_matched_v3.local_source_snapshot_tree.v1"
_SNAPSHOT_DESCRIPTOR_SCHEMA: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_descriptor.v1"
)
_RUNNER_DESCRIPTOR_SCHEMA: Final = "alberta.forager_matched_v3.local_runner_descriptor.v1"

_BOOTSTRAP_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_EXECUTION_BOOTSTRAP_SOURCE_SHA256"
)
_CHILD_SOURCE_SHA256_INPUT: Final = globals().get("_MATCHED_V3_BOOTSTRAP_CHILD_SOURCE_SHA256")
_CHILD_MODE_INPUT: Final = globals().get("_MATCHED_V3_BOOTSTRAP_CHILD_MODE")
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")

_FORBIDDEN_PREFIXES: Final = (
    "alberta_framework",
    "chex",
    "foragax",
    "jax",
    "jaxlib",
    "ml_dtypes",
    "numpy",
    "scipy",
)
_MODULE_KEYS_AT_LOAD: Final = tuple(sys.modules)
_NONEXACT_MODULE_KEYS_AT_LOAD: Final = tuple(
    type(name).__name__ for name in _MODULE_KEYS_AT_LOAD if type(name) is not str
)
_PRELOADED_FORBIDDEN_AT_LOAD: Final = tuple(
    sorted(
        name
        for name in _MODULE_KEYS_AT_LOAD
        if type(name) is str
        and any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES)
    )
)
_ISOLATED_PARENT_BOUNDARY: Final = (
    type(_MODULE_NAME_INPUT) is str
    and _MODULE_NAME_INPUT == LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME
    and (
        _MODULE_PACKAGE_INPUT is None
        or (type(_MODULE_PACKAGE_INPUT) is str and _MODULE_PACKAGE_INPUT == "")
    )
    and not _NONEXACT_MODULE_KEYS_AT_LOAD
    and not _PRELOADED_FORBIDDEN_AT_LOAD
)

_MAX_DESCRIPTOR_BYTES: Final = 1024 * 1024
_MAX_JSON_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_STRING_BYTES: Final = 24 * 1024 * 1024
_MAX_PATH_BYTES: Final = 4096
_MAX_PATH_COMPONENT_BYTES: Final = 255
_MAX_RUNTIME_ROOTS: Final = 32
_MAX_EXECUTABLE_BYTES: Final = 512 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
_MAX_CEILING: Final = 2**31 - 1
_UINT31_MAX: Final = 2**31 - 1
_PROCESS_POLL_SECONDS: Final = 0.025
_PROCESS_CLEANUP_SECONDS: Final = 2.0
_FRAME_MAGIC: Final = b"ALBERTA_V3_BOOT\x01"
_FRAME_HEADER: Final = struct.Struct(">16sQQQ")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z")
_CANDIDATE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SCRATCH_REQUEST_NAME: Final = "request.bin"
_SCRATCH_RESULT_NAME: Final = "result.bin"
_SCRATCH_CACHE_NAME: Final = "cache"
_SCRATCH_SOURCE_NAME: Final = "bootstrap.bin"

_MINIMAL_FIXED_ENVIRONMENT: Final[Mapping[str, str]] = {
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "JAX_PLATFORMS": "cpu",
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
}


def _child_environment(cache_proc_path: str) -> dict[str, str]:
    if (
        type(cache_proc_path) is not str
        or re.fullmatch(r"/proc/self/fd/[0-9]+", cache_proc_path) is None
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache environment path is invalid"
        )
    return {
        **_MINIMAL_FIXED_ENVIRONMENT,
        "HOME": cache_proc_path,
        "XDG_CACHE_HOME": cache_proc_path,
    }


class ForagerMatchedV3LocalExecutionBootstrapError(RuntimeError):
    """The bootstrap boundary, process, source, frame, or receipt failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalExecutionBootstrapError(
        f"bootstrap JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap JSON exceeds its node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap JSON exceeds its depth bound"
            )
        if type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES or any(
                byte < 0x20 or byte > 0x7E for byte in encoded
            ):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap JSON contains a non-plain value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalExecutionBootstrapError(
                        "bootstrap JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: Mapping[str, Any], *, maximum_bytes: int = _MAX_JSON_BYTES) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap canonical root must be a plain object"
        )
    _assert_plain_unaliased_json(value)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap canonical artifact exceeds its byte ceiling"
        )
    return raw


def _strict_json_load(raw: bytes, *, maximum_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact must be bounded exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact must have one trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalExecutionBootstrapError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact is not strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result, maximum_bytes=maximum_bytes), raw):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap artifact is not exactly canonical"
        )
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        right_list = cast(list[Any], right)
        return len(left) == len(right_list) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right_list, strict=True)
        )
    if type(left) is dict:
        left_map = cast(dict[str, Any], left)
        right_map = cast(dict[str, Any], right)
        return left_map.keys() == right_map.keys() and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    return False


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3LocalExecutionBootstrapError(f"{label} keys are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_uint31(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} must be one exact uint31 integer"
        )
    return value


def _require_positive_ceiling(value: Any, label: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_CEILING:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} must be one exact positive bounded integer"
        )
    return value


def _require_timeout(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 86_400:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "timeout_seconds must be one exact positive bounded integer"
        )
    return value


def _claims() -> dict[str, bool]:
    return {
        "execution_authority_serialized": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "A bootstrap receipt records one unqualified local execution only.",
        "Pre/post source equality does not prove continuous immutability between endpoints.",
        "Cache exclusion and source equality do not attest runtime dependency implementation.",
        (
            "Interpreter, runtime, dependency, toolchain, operating-system, and hardware "
            "closure remain external."
        ),
        (
            "Serialized requests, child records, receipts, traces, and descriptors grant "
            "no capability."
        ),
        (
            "No completion is scientific evidence or authorizes publication, promotion, "
            "or a SOTA claim."
        ),
        (
            "The bootstrap terminates the fresh process group and waits its direct child; "
            "it does not attest reaping of descendant processes."
        ),
        (
            "Post-run environment equality covers Python's os.environ mapping, not native "
            "environment state changed behind that mapping."
        ),
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_EXECUTION_BOOTSTRAP_STATUS,
        "classification": "standalone_local_execution_plumbing_non_authorizing",
        "pinned_components": {
            "local_runner": {
                "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
                "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
            },
            "local_source_snapshot": {
                "descriptor_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
                "source_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
            },
        },
        "module_import": {
            "scope": "exact_isolated_direct_byte_load_only_for_parent_capabilities",
            "bootstrap_source_sha256_injection_required": True,
            "filesystem_observation": False,
            "subprocess_launch": False,
            "heavy_or_alberta_import": False,
            "capability_issuance": False,
            "workload_execution": False,
        },
        "parent_capabilities": {
            "execution": {
                "explicit_opt_in": True,
                "opaque": True,
                "weak_registry": True,
                "pid_bound": True,
                "single_use": True,
                "serializable": False,
            },
            "outcome": {
                "second_explicit_opt_in": True,
                "opaque": True,
                "weak_registry": True,
                "pid_bound": True,
                "single_use": True,
                "serializable": False,
            },
            "serialized_content_grants_capability": False,
        },
        "process": {
            "fresh_processes_per_cell": 1,
            "required_interpreter_flags": ["-I", "-S", "-B"],
            "stdin": "devnull_closed_to_input",
            "stdout": "bounded_pipe",
            "stderr": "bounded_pipe",
            "result": "bounded_parent_retained_single_link_regular_fd",
            "new_session": True,
            "full_process_group_killed_on_all_exits": True,
            "direct_child_waited_and_reaped": True,
            "descendant_reaping_attested": False,
            "environment": {
                "fixed": dict(_MINIMAL_FIXED_ENVIRONMENT),
                "dynamic": {
                    "HOME": "private_cache_directory_proc_fd",
                    "XDG_CACHE_HOME": "private_cache_directory_proc_fd",
                },
                "exact_mapping_required": True,
                "python_os_environ_mapping_unchanged_after_execution_required": True,
                "jax_backend_selection": "cpu",
                "jax_default_backend_checked_after_execution": True,
                "jax_devices_all_cpu_checked_after_execution": True,
                "jax_persistent_compilation_cache_enabled": False,
            },
        },
        "source_order": {
            "independent_snapshot_source_identification_before_load": True,
            "full_snapshot_verification_before_runner_load": True,
            "runner_exact_bytes_direct_exec": True,
            "runner_capability_issue_run_consume": True,
            "full_snapshot_verification_after_outcome_consumption": True,
            "continuous_immutability_attested": False,
        },
        "cache": {
            "private_empty_mode": "0700",
            "interpreter_flag": "-B",
            "dont_write_bytecode": True,
            "pycache_prefix": "private_cache_directory_proc_fd",
            "site_initialization": False,
            "pth_processing": False,
            "empty_before_and_after": True,
        },
        "framing": {
            "magic_hex": _FRAME_MAGIC.hex(),
            "length_encoding": "three_unsigned_big_endian_uint64",
            "payloads": [
                "canonical_child_record",
                "canonical_local_completion_receipt",
                "raw_int8_reward_trace",
            ],
            "extra_or_truncated_bytes_allowed": False,
        },
        "caller_inputs": {
            "default_paths": False,
            "repository_root": "exact_absolute_non_symlink_path",
            "runtime_import_roots": "exact_tuple_of_absolute_non_symlink_paths",
            "scratch_parent": "new_private_empty_mode_0700_directory",
            "python_executable": "exact_absolute_non_symlink_regular_executable",
            "source_snapshot_bytes_and_independent_sha256": True,
            "candidate_environment_agent_seeds": True,
            "all_ceilings_explicit": True,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor(), maximum_bytes=_MAX_DESCRIPTOR_BYTES)
LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256: Final = (
    "6e62e2c6f2e1d157bee74c0866c96eededda21c5d77073d2adc05cc40dc72733"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 local execution bootstrap descriptor identity drifted")


class _ParentExecutionCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local bootstrap execution capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("bootstrap execution capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("bootstrap execution capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("bootstrap execution capabilities cannot be serialized")


@dataclass(slots=True)
class _ExecutionState:
    pid: int
    status: Literal["issued", "consumed"]


class _ParentOutcomeCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local bootstrap outcome capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("bootstrap outcome capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("bootstrap outcome capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("bootstrap outcome capabilities cannot be serialized")


@dataclass(slots=True)
class _OutcomeState:
    pid: int
    status: Literal["live", "consumed"]
    execution_capability: _ParentExecutionCapability
    execution_identity: int
    bootstrap_source_sha256: str
    receipt_sha256: str
    child_record_sha256: str
    local_receipt_sha256: str
    reward_trace_sha256: str
    completion: MatchedV3LocalBootstrapCompletion


_CAPABILITY_LOCK: Final = threading.Lock()
_EXECUTION_CAPABILITIES: Final[
    weakref.WeakKeyDictionary[_ParentExecutionCapability, _ExecutionState]
] = weakref.WeakKeyDictionary()
_OUTCOME_CAPABILITIES: Final[weakref.WeakKeyDictionary[_ParentOutcomeCapability, _OutcomeState]] = (
    weakref.WeakKeyDictionary()
)


def _live_forbidden_modules() -> tuple[str, ...]:
    try:
        items = tuple(sys.modules)
    except RuntimeError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "runtime module registry changed during bootstrap boundary observation"
        ) from exc
    if any(type(name) is not str for name in items):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "runtime module registry contains a non-exact-string key"
        )
    exact_items = items
    return tuple(
        sorted(
            name
            for name in exact_items
            if any(
                name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES
            )
        )
    )


def _require_parent_boundary(*, require_current_source: bool) -> str:
    if not _ISOLATED_PARENT_BOUNDARY:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap execution requires the exact isolated direct-byte parent module boundary"
        )
    expected = _require_sha256(
        _BOOTSTRAP_SOURCE_SHA256_INPUT,
        "bootstrap direct-byte source",
    )
    forbidden = _live_forbidden_modules()
    if forbidden:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap execution rejects preloaded runtime dependencies: "
            f"{', '.join(forbidden[:8])}"
        )
    if require_current_source:
        current = _current_bootstrap_source_sha256(_MAX_SOURCE_FILE_BYTES)
        if not hmac.compare_digest(current, expected):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap direct-byte source identity is stale or forged"
            )
    return expected


def issue_matched_v3_local_bootstrap_execution_capability(
    *,
    explicit_execution_opt_in: bool,
) -> object:
    """Issue one opaque parent capability; serialized requests never recreate it."""

    if type(explicit_execution_opt_in) is not bool or explicit_execution_opt_in is not True:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap capability issuance requires exact explicit opt-in"
        )
    _require_parent_boundary(require_current_source=True)
    capability = _ParentExecutionCapability()
    with _CAPABILITY_LOCK:
        _EXECUTION_CAPABILITIES[capability] = _ExecutionState(
            pid=os.getpid(),
            status="issued",
        )
    return capability


def _consume_execution_capability(capability: object) -> _ParentExecutionCapability:
    if type(capability) is not _ParentExecutionCapability:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap execution requires an authentic opaque parent capability"
        )
    exact = capability
    with _CAPABILITY_LOCK:
        state = _EXECUTION_CAPABILITIES.get(exact)
        if state is None or state.status != "issued":
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap execution capability is unknown or already consumed"
            )
        if state.pid != os.getpid():
            state.status = "consumed"
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap execution capability cannot cross a PID boundary"
            )
        state.status = "consumed"
    return exact


def _validate_path_component(component: Any, label: str) -> str:
    if (
        type(component) is not str
        or component in {"", ".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or _PATH_COMPONENT_RE.fullmatch(component) is None
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} is not one unambiguous ASCII path component"
        )
    if len(component.encode("ascii")) > _MAX_PATH_COMPONENT_BYTES:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} exceeds the path component byte bound"
        )
    return component


def _exact_absolute_path(path: Any, label: str) -> Path:
    if type(path) is not type(Path()):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} must be one exact concrete pathlib.Path"
        )
    exact = path
    raw = str(exact)
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(f"{label} must be ASCII") from exc
    if (
        not encoded
        or len(encoded) > _MAX_PATH_BYTES
        or not exact.is_absolute()
        or exact.anchor != os.sep
        or exact == Path(exact.anchor)
        or os.path.abspath(raw) != raw
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} must be a non-root exact absolute path without aliases"
        )
    for index, component in enumerate(exact.parts[1:]):
        _validate_path_component(component, f"{label} component {index}")
    return exact


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap traversal requires O_DIRECTORY and O_NOFOLLOW"
        )
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _file_flags(*, writable: bool = False) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap traversal requires O_NOFOLLOW"
        )
    access = os.O_RDWR if writable else os.O_RDONLY
    return access | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _locator_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))


@dataclass(slots=True)
class _DirectoryAnchor:
    path: Path
    descriptors: list[int]
    components: list[str]
    metadata: list[os.stat_result]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    @property
    def identity(self) -> tuple[int, int]:
        metadata = self.metadata[-1]
        return (metadata.st_dev, metadata.st_ino)

    def verify(self) -> None:
        for index in range(1, len(self.descriptors)):
            parent = self.descriptors[index - 1]
            descriptor = self.descriptors[index]
            component = self.components[index]
            expected = self.metadata[index]
            try:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
            except OSError as exc:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "anchored bootstrap path locator changed"
                ) from exc
            if _locator_identity(current) != _locator_identity(expected) or _locator_identity(
                opened
            ) != _locator_identity(expected):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "anchored bootstrap path locator changed"
                )

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_directory_anchor(path: Any, label: str) -> _DirectoryAnchor:
    exact = _exact_absolute_path(path, label)
    flags = _directory_flags()
    descriptors: list[int] = []
    metadata: list[os.stat_result] = []
    components = [exact.anchor, *exact.parts[1:]]
    try:
        anchor_fd = os.open(exact.anchor, flags)
        descriptors.append(anchor_fd)
        anchor_metadata = os.fstat(anchor_fd)
        if not stat.S_ISDIR(anchor_metadata.st_mode):
            raise ForagerMatchedV3LocalExecutionBootstrapError(f"{label} anchor is not a directory")
        metadata.append(anchor_metadata)
        for component in components[1:]:
            parent = descriptors[-1]
            try:
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    f"{label} cannot be inspected without following links"
                ) from exc
            if not stat.S_ISDIR(before.st_mode):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    f"{label} contains a symlink or non-directory component"
                )
            try:
                child = os.open(component, flags, dir_fd=parent)
            except OSError as exc:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    f"{label} cannot be opened without following links"
                ) from exc
            descriptors.append(child)
            opened = os.fstat(child)
            current = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
                opened
            ) != _stat_identity(current):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    f"{label} changed while being opened"
                )
            metadata.append(opened)
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    result = _DirectoryAnchor(exact, descriptors, components, metadata)
    result.verify()
    return result


@dataclass(slots=True)
class _ExecutableAnchor:
    path: Path
    parent: _DirectoryAnchor
    descriptor: int
    metadata: os.stat_result
    sha256: str

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def verify(self) -> None:
        self.parent.verify()
        name = self.path.name
        try:
            current = os.stat(name, dir_fd=self.parent.descriptor, follow_symlinks=False)
            opened = os.fstat(self.descriptor)
        except OSError as exc:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "python executable locator changed"
            ) from exc
        if _stat_identity(current) != _stat_identity(self.metadata) or _stat_identity(
            opened
        ) != _stat_identity(self.metadata):
            raise ForagerMatchedV3LocalExecutionBootstrapError("python executable locator changed")

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        finally:
            self.parent.close()


def _read_exact_fd(
    descriptor: int,
    *,
    maximum_bytes: int,
    require_single_link: bool,
    require_nonempty: bool,
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap file descriptor cannot be inspected"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or (require_single_link and before.st_nlink != 1)
        or before.st_size < int(require_nonempty)
        or before.st_size > maximum_bytes
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap file descriptor is not one bounded regular file"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap file ended while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap file grew while being read"
            )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap file could not be read exactly"
        ) from exc
    if _stat_identity(before) != _stat_identity(after):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap file changed while being read"
        )
    return b"".join(chunks), before


def _open_executable_anchor(path: Any) -> _ExecutableAnchor:
    exact = _exact_absolute_path(path, "python_executable")
    parent = _open_directory_anchor(exact.parent, "python_executable parent")
    descriptor = -1
    try:
        before = os.stat(exact.name, dir_fd=parent.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_EXECUTABLE_BYTES
            or before.st_mode & 0o111 == 0
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "python_executable must be one bounded executable single-link regular file"
            )
        descriptor = os.open(exact.name, _file_flags(), dir_fd=parent.descriptor)
        source, opened = _read_exact_fd(
            descriptor,
            maximum_bytes=_MAX_EXECUTABLE_BYTES,
            require_single_link=True,
            require_nonempty=True,
        )
        current = os.stat(exact.name, dir_fd=parent.descriptor, follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
            opened
        ) != _stat_identity(current):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "python_executable changed while being hashed"
            )
        result = _ExecutableAnchor(
            exact,
            parent,
            descriptor,
            opened,
            hashlib.sha256(source).hexdigest(),
        )
        result.verify()
        return result
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        parent.close()
        raise


def _read_anchored_relative_file(
    root: _DirectoryAnchor,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    components = relative_path.split("/")
    if not components or any(not component for component in components):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap source relative path is invalid"
        )
    for index, component in enumerate(components):
        _validate_path_component(component, f"bootstrap source component {index}")
    parent_fd = root.descriptor
    opened_directories: list[int] = []
    file_fd = -1
    try:
        for component in components[:-1]:
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap source path contains a link or non-directory"
                )
            child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            opened = os.fstat(child_fd)
            current = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
                opened
            ) != _stat_identity(current):
                os.close(child_fd)
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap source path changed while being opened"
                )
            opened_directories.append(child_fd)
            parent_fd = child_fd
        name = components[-1]
        before_file = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before_file.st_mode) or before_file.st_nlink != 1:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap source is not one single-link regular file"
            )
        file_fd = os.open(name, _file_flags(), dir_fd=parent_fd)
        source, opened_file = _read_exact_fd(
            file_fd,
            maximum_bytes=maximum_bytes,
            require_single_link=True,
            require_nonempty=True,
        )
        current_file = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _stat_identity(before_file) != _stat_identity(opened_file) or _stat_identity(
            opened_file
        ) != _stat_identity(current_file):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap source changed while being read"
            )
        root.verify()
        return source
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap source could not be opened exactly"
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _current_bootstrap_source_bytes(maximum_bytes: int) -> bytes:
    path = globals().get("__file__")
    if type(path) is not str:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap module has no exact source path"
        )
    exact = _exact_absolute_path(Path(path), "bootstrap module source")
    parent = _open_directory_anchor(exact.parent, "bootstrap module source parent")
    descriptor = -1
    try:
        before = os.stat(exact.name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(exact.name, _file_flags(), dir_fd=parent.descriptor)
        source, opened = _read_exact_fd(
            descriptor,
            maximum_bytes=maximum_bytes,
            require_single_link=True,
            require_nonempty=True,
        )
        current = os.stat(exact.name, dir_fd=parent.descriptor, follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
            opened
        ) != _stat_identity(current):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap module source changed while being read"
            )
        return source
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        parent.close()


def _current_bootstrap_source_sha256(maximum_bytes: int) -> str:
    return hashlib.sha256(_current_bootstrap_source_bytes(maximum_bytes)).hexdigest()


@dataclass(frozen=True, slots=True)
class _SnapshotManifestIdentity:
    full_sha256: str
    tree_sha256: str
    snapshot_source_size: int
    snapshot_source_sha256: str
    runner_source_size: int
    runner_source_sha256: str


def _independent_snapshot_manifest_identity(
    raw: bytes,
    *,
    expected_full_sha256: str,
    maximum_bytes: int,
) -> _SnapshotManifestIdentity:
    expected = _require_sha256(expected_full_sha256, "expected source snapshot manifest")
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot manifest full-file digest disagrees"
        )
    manifest = _strict_json_load(raw, maximum_bytes=maximum_bytes)
    required_top = frozenset(
        {
            "schema_version",
            "status",
            "classification",
            "descriptor_binding",
            "observation",
            "inventory",
            "directories",
            "files",
            "tree",
            "claims",
            "limitations",
            "manifest_body_sha256",
        }
    )
    _require_exact_keys(manifest, required_top, "source snapshot manifest")
    if manifest["schema_version"] != _SNAPSHOT_MANIFEST_SCHEMA:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot manifest schema drifted"
        )
    if not _exact_json_equal(
        manifest["descriptor_binding"],
        {
            "schema_version": _SNAPSHOT_DESCRIPTOR_SCHEMA,
            "sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot manifest descriptor binding drifted"
        )
    files = manifest["files"]
    directories = manifest["directories"]
    if type(files) is not list or type(directories) is not list:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot inventory arrays are not exact"
        )
    records: dict[str, tuple[int, str]] = {}
    ordered_paths: list[str] = []
    for index, item in enumerate(files):
        record = _require_exact_keys(
            item,
            frozenset({"path", "size_bytes", "sha256"}),
            f"source snapshot file {index}",
        )
        path = record["path"]
        size = record["size_bytes"]
        digest = _require_sha256(record["sha256"], f"source snapshot file {index}")
        if (
            type(path) is not str
            or type(size) is not int
            or not 0 <= size <= _MAX_CEILING
            or path in records
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "source snapshot file record is not exact"
            )
        records[path] = (size, digest)
        ordered_paths.append(path)
    if ordered_paths != sorted(ordered_paths, key=lambda value: value.encode("ascii")):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot file records are not ordered"
        )
    tree = _require_exact_keys(
        manifest["tree"],
        frozenset({"schema_version", "sha256"}),
        "source snapshot tree",
    )
    tree_sha256 = _require_sha256(tree["sha256"], "source snapshot tree")
    expected_tree = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": _SNAPSHOT_TREE_SCHEMA,
                "directories": directories,
                "files": files,
            },
            maximum_bytes=maximum_bytes,
        )
    ).hexdigest()
    if tree["schema_version"] != _SNAPSHOT_TREE_SCHEMA or not hmac.compare_digest(
        tree_sha256, expected_tree
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("source snapshot tree identity drifted")
    body = dict(manifest)
    supplied_body = _require_sha256(
        body.pop("manifest_body_sha256"),
        "source snapshot manifest body",
    )
    if not hmac.compare_digest(
        supplied_body,
        hashlib.sha256(_canonical_json(body, maximum_bytes=maximum_bytes)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot manifest body digest drifted"
        )
    snapshot_record = records.get(_SNAPSHOT_RELATIVE_PATH)
    runner_record = records.get(_RUNNER_RELATIVE_PATH)
    if snapshot_record is None or runner_record is None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot omits a bootstrap-critical source"
        )
    if snapshot_record[1] != PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "source snapshot module source identity drifted"
        )
    if runner_record[1] != PINNED_LOCAL_RUNNER_SOURCE_SHA256:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "local runner source identity drifted in the snapshot"
        )
    return _SnapshotManifestIdentity(
        full_sha256=expected,
        tree_sha256=tree_sha256,
        snapshot_source_size=snapshot_record[0],
        snapshot_source_sha256=snapshot_record[1],
        runner_source_size=runner_record[0],
        runner_source_sha256=runner_record[1],
    )


def _direct_load_snapshot_module(source: bytes, source_path: str) -> types.ModuleType:
    if type(source) is not bytes or not hmac.compare_digest(
        hashlib.sha256(source).hexdigest(),
        PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot module exact source bytes disagree"
        )
    module = types.ModuleType(_SNAPSHOT_MODULE_NAME)
    module.__file__ = source_path
    module.__package__ = ""
    previous = sys.modules.get(_SNAPSHOT_MODULE_NAME)
    if previous is not None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot bootstrap module name is already occupied"
        )
    sys.modules[_SNAPSHOT_MODULE_NAME] = module
    try:
        exec(compile(source, source_path, "exec"), module.__dict__)
        descriptor_sha256 = getattr(
            module,
            "LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256",
            None,
        )
        descriptor_bytes_function = getattr(
            module,
            "canonical_matched_v3_local_source_snapshot_descriptor_bytes",
            None,
        )
        if descriptor_sha256 != PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256 or not callable(
            descriptor_bytes_function
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "snapshot module descriptor identity drifted"
            )
        descriptor_bytes = descriptor_bytes_function()
        if type(descriptor_bytes) is not bytes or not hmac.compare_digest(
            hashlib.sha256(descriptor_bytes).hexdigest(),
            PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "snapshot module descriptor bytes drifted"
            )
        return module
    except BaseException:
        sys.modules.pop(_SNAPSHOT_MODULE_NAME, None)
        raise


def _parse_snapshot_with_exact_module(
    module: types.ModuleType,
    raw: bytes,
    expected_full_sha256: str,
) -> dict[str, Any]:
    parser = getattr(module, "parse_matched_v3_local_source_snapshot_manifest", None)
    if not callable(parser):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot module strict parser is unavailable"
        )
    try:
        value = parser(raw, expected_full_sha256=expected_full_sha256)
    except Exception as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot module rejected the caller-carried manifest"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot module parser returned a non-object"
        )
    return cast(dict[str, Any], value)


def _verify_snapshot_with_exact_module(
    module: types.ModuleType,
    *,
    repository_root: Path,
    raw: bytes,
    expected_full_sha256: str,
) -> Any:
    verifier = getattr(module, "verify_matched_v3_local_source_snapshot", None)
    if not callable(verifier):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot module verifier is unavailable"
        )
    try:
        result = verifier(
            repository_root=repository_root,
            expected_canonical_manifest_bytes=raw,
            expected_full_sha256=expected_full_sha256,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "full local source snapshot verification failed"
        ) from exc
    if (
        getattr(result, "full_sha256", None) != expected_full_sha256
        or type(getattr(result, "tree_sha256", None)) is not str
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "snapshot verifier result identity drifted"
        )
    return result


def _path_sha256(path: str) -> str:
    try:
        raw = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap process path must be ASCII"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _canonical_sequence_sha256(items: Sequence[str]) -> str:
    if type(items) not in {tuple, list} or any(type(item) is not str for item in items):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap string sequence is not exact")
    return hashlib.sha256(_canonical_json({"items": list(items)})).hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, str]) -> str:
    if type(value) is not dict or any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap string mapping is not exact")
    return hashlib.sha256(_canonical_json({"mapping": dict(value)})).hexdigest()


def _argv_sha256(items: Sequence[str]) -> str:
    if type(items) not in {tuple, list} or any(type(item) is not str for item in items):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap argv sequence is not exact")
    try:
        encoded = [base64.b64encode(item.encode("ascii")).decode("ascii") for item in items]
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap argv must be exact ASCII"
        ) from exc
    return hashlib.sha256(_canonical_json({"argv_base64": encoded})).hexdigest()


def _validate_runtime_root_tuple(value: Any) -> tuple[Path, ...]:
    if (
        type(value) is not tuple
        or not 1 <= len(value) <= _MAX_RUNTIME_ROOTS
        or any(type(item) is not type(Path()) for item in value)
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "runtime_import_roots must be one bounded exact tuple of concrete Paths"
        )
    result = tuple(
        _exact_absolute_path(item, f"runtime_import_roots[{index}]")
        for index, item in enumerate(value)
    )
    if len({str(item).casefold() for item in result}) != len(result):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "runtime_import_roots contain a duplicate or casefold alias"
        )
    return result


def _validate_candidate_id(value: Any) -> str:
    if type(value) is not str or _CANDIDATE_RE.fullmatch(value) is None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "candidate_id must be one bounded exact ASCII identifier"
        )
    return value


def _request_body(
    *,
    repository_root: Path,
    repository_identity: tuple[int, int],
    runtime_roots: tuple[Path, ...],
    runtime_identities: tuple[tuple[int, int], ...],
    manifest_raw: bytes,
    manifest_identity: _SnapshotManifestIdentity,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    ceilings: Mapping[str, int],
    process_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "alberta.forager_matched_v3.local_execution_bootstrap_request.v1",
        "descriptor_binding": {
            "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
        },
        "repository_root": {
            "path": str(repository_root),
            "device": repository_identity[0],
            "inode": repository_identity[1],
        },
        "runtime_import_roots": [
            {
                "path": str(path),
                "device": identity[0],
                "inode": identity[1],
            }
            for path, identity in zip(runtime_roots, runtime_identities, strict=True)
        ],
        "source_snapshot": {
            "canonical_manifest_base64": base64.b64encode(manifest_raw).decode("ascii"),
            "full_sha256": manifest_identity.full_sha256,
            "tree_sha256": manifest_identity.tree_sha256,
            "snapshot_source_size": manifest_identity.snapshot_source_size,
            "snapshot_source_sha256": manifest_identity.snapshot_source_sha256,
            "runner_source_size": manifest_identity.runner_source_size,
            "runner_source_sha256": manifest_identity.runner_source_sha256,
        },
        "cell": {
            "candidate_id": candidate_id,
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
        },
        "ceilings": dict(ceilings),
        "process_contract": dict(process_contract),
        "claims": _claims(),
    }


def _build_request(**kwargs: Any) -> tuple[dict[str, Any], bytes, str]:
    body = _request_body(**kwargs)
    request = {
        **body,
        "request_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    raw = _canonical_json(request)
    return request, raw, hashlib.sha256(raw).hexdigest()


def _parse_request(raw: bytes, *, maximum_bytes: int) -> tuple[dict[str, Any], str]:
    request = _strict_json_load(raw, maximum_bytes=maximum_bytes)
    expected_keys = frozenset(
        {
            "schema_version",
            "descriptor_binding",
            "repository_root",
            "runtime_import_roots",
            "source_snapshot",
            "cell",
            "ceilings",
            "process_contract",
            "claims",
            "request_body_sha256",
        }
    )
    _require_exact_keys(request, expected_keys, "bootstrap request")
    body = dict(request)
    supplied = _require_sha256(body.pop("request_body_sha256"), "bootstrap request body")
    if not hmac.compare_digest(supplied, hashlib.sha256(_canonical_json(body)).hexdigest()):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request body identity drifted"
        )
    if (
        request["schema_version"]
        != "alberta.forager_matched_v3.local_execution_bootstrap_request.v1"
        or not _exact_json_equal(
            request["descriptor_binding"],
            {
                "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
            },
        )
        or not _exact_json_equal(request["claims"], _claims())
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap request identity drifted")
    return request, hashlib.sha256(raw).hexdigest()


def _child_record_body(
    *,
    bootstrap_source_sha256: str,
    request_sha256: str,
    manifest_full_sha256: str,
    manifest_tree_sha256: str,
    pre_full_sha256: str,
    pre_tree_sha256: str,
    post_full_sha256: str,
    post_tree_sha256: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    process_contract: Mapping[str, Any],
    local_receipt: bytes,
    reward_trace: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_EXECUTION_BOOTSTRAP_CHILD_RECORD_SCHEMA_VERSION,
        "status": "child_completed_unqualified_non_authorizing",
        "classification": "verified_endpoint_execution_content_only",
        "descriptor_binding": {
            "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
        },
        "bootstrap_source_sha256": bootstrap_source_sha256,
        "request_sha256": request_sha256,
        "source_snapshot": {
            "descriptor_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
            "expected_full_sha256": manifest_full_sha256,
            "expected_tree_sha256": manifest_tree_sha256,
            "pre_full_sha256": pre_full_sha256,
            "pre_tree_sha256": pre_tree_sha256,
            "post_full_sha256": post_full_sha256,
            "post_tree_sha256": post_tree_sha256,
            "continuous_immutability_attested": False,
        },
        "runner": {
            "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
            "isolated_module_name": ("_alberta_forager_matched_v3_local_runner_isolated_v1"),
            "own_capabilities_issued_run_consumed": True,
        },
        "cell": {
            "candidate_id": candidate_id,
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
        },
        "process_contract": dict(process_contract),
        "completion": {
            "local_receipt_size_bytes": len(local_receipt),
            "local_receipt_sha256": hashlib.sha256(local_receipt).hexdigest(),
            "reward_trace_size_bytes": len(reward_trace),
            "reward_trace_sha256": hashlib.sha256(reward_trace).hexdigest(),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _build_child_record(**kwargs: Any) -> bytes:
    body = _child_record_body(**kwargs)
    return _canonical_json(
        {
            **body,
            "child_record_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
        }
    )


def _result_frame(child_record: bytes, local_receipt: bytes, reward_trace: bytes) -> bytes:
    for byte_value, label in (
        (child_record, "child record"),
        (local_receipt, "local completion receipt"),
        (reward_trace, "reward trace"),
    ):
        if type(byte_value) is not bytes:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap frame {label} must be exact bytes"
            )
    return (
        _FRAME_HEADER.pack(
            _FRAME_MAGIC,
            len(child_record),
            len(local_receipt),
            len(reward_trace),
        )
        + child_record
        + local_receipt
        + reward_trace
    )


def _parse_result_frame(
    raw: bytes,
    *,
    maximum_result_bytes: int,
) -> tuple[bytes, bytes, bytes]:
    maximum = _require_positive_ceiling(maximum_result_bytes, "maximum_result_bytes")
    if type(raw) is not bytes or len(raw) > maximum or len(raw) < _FRAME_HEADER.size:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap result frame is truncated or exceeds its ceiling"
        )
    try:
        magic, child_size, receipt_size, trace_size = _FRAME_HEADER.unpack_from(raw)
    except struct.error as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap result frame header is invalid"
        ) from exc
    if magic != _FRAME_MAGIC:
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap result frame magic drifted")
    sizes = (child_size, receipt_size, trace_size)
    if any(type(size) is not int or size < 1 or size > maximum for size in sizes):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap result frame length is invalid"
        )
    expected_total = _FRAME_HEADER.size + sum(sizes)
    if expected_total != len(raw):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap result frame has truncated or extra bytes"
        )
    start = _FRAME_HEADER.size
    child_end = start + child_size
    receipt_end = child_end + receipt_size
    return raw[start:child_end], raw[child_end:receipt_end], raw[receipt_end:]


def _write_all(descriptor: int, raw: bytes, *, maximum_bytes: int) -> None:
    if type(raw) is not bytes or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child result exceeds its byte ceiling"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap child result write made no progress"
                )
            written += count
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child result could not be written exactly"
        ) from exc
    if after.st_size != len(raw):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child result size drifted after writing"
        )


_CHILD_LAUNCHER: Final = """\
import hashlib, os, sys, types
source_fd = int(sys.argv[1])
request_fd = int(sys.argv[2])
result_fd = int(sys.argv[3])
cache_fd = int(sys.argv[4])
expected = sys.argv[5]
maximum = int(sys.argv[6])
metadata = os.fstat(source_fd)
if metadata.st_size < 1 or metadata.st_size > maximum:
    raise SystemExit(91)
os.lseek(source_fd, 0, os.SEEK_SET)
remaining = metadata.st_size
chunks = []
while remaining:
    chunk = os.read(source_fd, min(remaining, 1048576))
    if not chunk:
        raise SystemExit(92)
    chunks.append(chunk)
    remaining -= len(chunk)
if os.read(source_fd, 1):
    raise SystemExit(93)
source = b\"\".join(chunks)
if hashlib.sha256(source).hexdigest() != expected:
    raise SystemExit(94)
name = \"_alberta_forager_matched_v3_local_execution_bootstrap_child_v1\"
module = types.ModuleType(name)
module.__file__ = \"/proc/self/fd/\" + str(source_fd)
module.__package__ = \"\"
module.__dict__[\"_MATCHED_V3_BOOTSTRAP_CHILD_SOURCE_SHA256\"] = expected
module.__dict__[\"_MATCHED_V3_BOOTSTRAP_CHILD_MODE\"] = True
sys.modules[name] = module
exec(compile(source, module.__file__, \"exec\"), module.__dict__)
module._child_entry(
    source_fd=source_fd,
    request_fd=request_fd,
    result_fd=result_fd,
    cache_fd=cache_fd,
)
"""
_CHILD_LAUNCHER_SHA256: Final = hashlib.sha256(_CHILD_LAUNCHER.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap process group could not be terminated"
        ) from exc


def _run_bounded_child(
    *,
    argv: tuple[str, ...],
    executable_proc_path: str,
    cwd: str,
    environment: Mapping[str, str],
    pass_fds: tuple[int, ...],
    result_fd: int,
    maximum_result_bytes: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    timeout_seconds: int,
) -> _ProcessResult:
    """Launch once, kill the process group, and wait/reap the direct child."""

    result_maximum = _require_positive_ceiling(maximum_result_bytes, "maximum_result_bytes")
    stdout_maximum = _require_positive_ceiling(maximum_stdout_bytes, "maximum_stdout_bytes")
    stderr_maximum = _require_positive_ceiling(maximum_stderr_bytes, "maximum_stderr_bytes")
    timeout = _require_timeout(timeout_seconds)
    if type(argv) is not tuple or not argv or any(type(item) is not str for item in argv):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child argv is not exact")
    if type(pass_fds) is not tuple or any(type(item) is not int for item in pass_fds):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child pass_fds are not exact")
    try:
        process = subprocess.Popen(
            list(argv),
            executable=executable_proc_path,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child process could not start"
        ) from exc
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        process.wait()
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child process pipes are unavailable"
        )

    stdout = bytearray()
    stderr = bytearray()
    overflow: str | None = None
    reader_error: OSError | None = None
    result_overflow = False
    timed_out = False
    cleanup_timed_out = False
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout
    try:
        for stream, destination, maximum, label in (
            (process.stdout, stdout, stdout_maximum, "stdout"),
            (process.stderr, stderr, stderr_maximum, "stderr"),
        ):
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, (destination, maximum, label))
        while selector.get_map():
            try:
                if os.fstat(result_fd).st_size > result_maximum:
                    result_overflow = True
                    break
            except OSError as exc:
                reader_error = exc
                break
            exited = process.poll() is not None
            if exited:
                _terminate_process_group(process)
                selection_timeout = 0.0
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    break
                selection_timeout = min(remaining, _PROCESS_POLL_SECONDS)
            events = selector.select(selection_timeout)
            if not events:
                if exited:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                continue
            stop = False
            for key, _event in events:
                destination, maximum, label = cast(tuple[bytearray, int, str], key.data)
                while True:
                    try:
                        chunk = os.read(key.fd, 64 * 1024)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        reader_error = exc
                        stop = True
                        break
                    if not chunk:
                        selector.unregister(key.fd)
                        break
                    if len(destination) + len(chunk) > maximum:
                        overflow = label
                        stop = True
                        break
                    destination.extend(chunk)
                if stop:
                    break
            if stop:
                break
    finally:
        termination_error: BaseException | None = None
        try:
            _terminate_process_group(process)
        except BaseException as exc:
            termination_error = exc
        cleanup_deadline = time.monotonic() + _PROCESS_CLEANUP_SECONDS
        try:
            process.wait(timeout=max(0.001, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            cleanup_timed_out = True
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if termination_error is not None:
            raise termination_error
    if cleanup_timed_out:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child process could not be reaped within its cleanup bound"
        )
    if reader_error is not None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child output could not be read exactly"
        ) from reader_error
    if timed_out:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child process exceeded its wall-time ceiling"
        )
    if overflow is not None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"bootstrap child {overflow} exceeded its byte ceiling"
        )
    if result_overflow:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child result exceeded its byte ceiling"
        )
    if type(process.returncode) is not int:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child process has no exact return code"
        )
    return _ProcessResult(process.returncode, bytes(stdout), bytes(stderr))


def _directory_is_empty(descriptor: int) -> bool:
    try:
        with os.scandir(descriptor) as entries:
            return next(entries, None) is None
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap cache directory cannot be enumerated"
        ) from exc


def _activate_child_cache(cache_fd: int) -> str:
    if type(cache_fd) is not int or cache_fd < 0:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache descriptor is invalid"
        )
    metadata = os.fstat(cache_fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or not _directory_is_empty(cache_fd)
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache must be one private empty mode-0700 directory"
        )
    proc_path = f"/proc/self/fd/{cache_fd}"
    proc_metadata = os.stat(proc_path, follow_symlinks=True)
    if (proc_metadata.st_dev, proc_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache proc-fd identity drifted"
        )
    sys.dont_write_bytecode = True
    sys.pycache_prefix = proc_path
    if not sys.dont_write_bytecode or sys.pycache_prefix != proc_path:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache neutralization did not take effect"
        )
    return proc_path


def _child_initial_boundary(cache_proc_path: str) -> tuple[str, ...]:
    if (
        _CHILD_MODE_INPUT is not True
        or type(_MODULE_NAME_INPUT) is not str
        or _MODULE_NAME_INPUT != _CHILD_MODULE_NAME
        or not (_MODULE_PACKAGE_INPUT is None or _MODULE_PACKAGE_INPUT == "")
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child requires its exact direct-byte launcher boundary"
        )
    _require_sha256(_CHILD_SOURCE_SHA256_INPUT, "bootstrap child source")
    forbidden = _live_forbidden_modules()
    if forbidden:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"bootstrap child rejects preloaded runtime dependencies: {', '.join(forbidden[:8])}"
        )
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.ignore_environment != 1
        or not sys.flags.safe_path
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child interpreter flags are not exactly isolated"
        )
    if dict(os.environ) != _child_environment(cache_proc_path):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child environment is not the exact minimal mapping"
        )
    original = tuple(sys.path)
    if (
        not original
        or any(
            type(path) is not str
            or not path
            or not os.path.isabs(path)
            or "site-packages" in path
            or "dist-packages" in path
            for path in original
        )
        or len(set(original)) != len(original)
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child original sys.path is not stdlib-only"
        )
    return original


def _child_read_request(request_fd: int) -> tuple[dict[str, Any], bytes, str]:
    raw, _metadata = _read_exact_fd(
        request_fd,
        maximum_bytes=_MAX_JSON_BYTES,
        require_single_link=True,
        require_nonempty=True,
    )
    request, request_sha256 = _parse_request(raw, maximum_bytes=_MAX_JSON_BYTES)
    ceilings = _require_exact_keys(
        request["ceilings"],
        frozenset(
            {
                "maximum_request_bytes",
                "maximum_bootstrap_source_bytes",
                "maximum_result_bytes",
                "maximum_stdout_bytes",
                "maximum_stderr_bytes",
                "timeout_seconds",
            }
        ),
        "bootstrap request ceilings",
    )
    request_maximum = _require_positive_ceiling(
        ceilings["maximum_request_bytes"],
        "maximum_request_bytes",
    )
    if len(raw) > request_maximum:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request exceeds its caller ceiling"
        )
    _require_positive_ceiling(
        ceilings["maximum_bootstrap_source_bytes"],
        "maximum_bootstrap_source_bytes",
    )
    _require_positive_ceiling(ceilings["maximum_result_bytes"], "maximum_result_bytes")
    _require_positive_ceiling(ceilings["maximum_stdout_bytes"], "maximum_stdout_bytes")
    _require_positive_ceiling(ceilings["maximum_stderr_bytes"], "maximum_stderr_bytes")
    _require_timeout(ceilings["timeout_seconds"])
    return request, raw, request_sha256


def _decode_manifest_from_request(
    request: Mapping[str, Any],
    *,
    maximum_bytes: int,
) -> tuple[bytes, str, str]:
    source_snapshot = _require_exact_keys(
        request["source_snapshot"],
        frozenset(
            {
                "canonical_manifest_base64",
                "full_sha256",
                "tree_sha256",
                "snapshot_source_size",
                "snapshot_source_sha256",
                "runner_source_size",
                "runner_source_sha256",
            }
        ),
        "bootstrap request source snapshot",
    )
    encoded = source_snapshot["canonical_manifest_base64"]
    if type(encoded) is not str:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request manifest transport is not exact base64"
        )
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request manifest transport is not exact base64"
        ) from exc
    if base64.b64encode(raw).decode("ascii") != encoded or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request manifest transport is noncanonical or oversized"
        )
    full_sha256 = _require_sha256(source_snapshot["full_sha256"], "source snapshot full")
    tree_sha256 = _require_sha256(source_snapshot["tree_sha256"], "source snapshot tree")
    identity = _independent_snapshot_manifest_identity(
        raw,
        expected_full_sha256=full_sha256,
        maximum_bytes=maximum_bytes,
    )
    if (
        identity.tree_sha256 != tree_sha256
        or source_snapshot["snapshot_source_size"] != identity.snapshot_source_size
        or source_snapshot["snapshot_source_sha256"] != identity.snapshot_source_sha256
        or source_snapshot["runner_source_size"] != identity.runner_source_size
        or source_snapshot["runner_source_sha256"] != identity.runner_source_sha256
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request source snapshot identity drifted"
        )
    return raw, full_sha256, tree_sha256


def _child_validate_process_contract(
    request: Mapping[str, Any],
    *,
    source_fd: int,
    request_fd: int,
    result_fd: int,
    cache_fd: int,
    cache_proc_path: str,
) -> dict[str, Any]:
    contract = _require_exact_keys(
        request["process_contract"],
        frozenset(
            {
                "argv_sha256",
                "argv_count",
                "launcher_sha256",
                "environment_sha256",
                "cwd_path_sha256",
                "executable_path_sha256",
                "executable_source_sha256",
                "executable_proc_path_sha256",
                "executable_fd",
                "source_fd",
                "request_fd",
                "result_fd",
                "cache_fd",
                "cache_proc_path",
                "bootstrap_source_sha256",
                "jax_platform_selector",
                "jax_compilation_cache_enabled",
                "home_xdg_cache_proc_path",
                "required_flags",
                "stdin",
                "start_new_session",
                "site_initialization",
                "pth_processing",
            }
        ),
        "bootstrap child process contract",
    )
    expected = {
        "launcher_sha256": _CHILD_LAUNCHER_SHA256,
        "environment_sha256": _canonical_mapping_sha256(_child_environment(cache_proc_path)),
        "cwd_path_sha256": _path_sha256(os.getcwd()),
        "source_fd": source_fd,
        "request_fd": request_fd,
        "result_fd": result_fd,
        "cache_fd": cache_fd,
        "cache_proc_path": cache_proc_path,
        "bootstrap_source_sha256": _require_sha256(
            _CHILD_SOURCE_SHA256_INPUT,
            "bootstrap child source",
        ),
        "jax_platform_selector": "cpu",
        "jax_compilation_cache_enabled": False,
        "home_xdg_cache_proc_path": cache_proc_path,
        "required_flags": ["-I", "-S", "-B"],
        "stdin": "devnull_closed_to_input",
        "start_new_session": True,
        "site_initialization": False,
        "pth_processing": False,
    }
    for key, value in expected.items():
        if not _exact_json_equal(contract[key], value):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap child process contract drifted: {key}"
            )
    supplied_argv_sha256 = _require_sha256(
        contract["argv_sha256"],
        "bootstrap child argv",
    )
    original_argv = getattr(sys, "orig_argv", None)
    if (
        type(original_argv) is not list
        or any(type(item) is not str for item in original_argv)
        or len(original_argv) != 12
        or original_argv[1:6] != ["-I", "-S", "-B", "-c", _CHILD_LAUNCHER]
        or sys.argv[0] != "-c"
        or original_argv[6:] != sys.argv[1:]
        or not hmac.compare_digest(
            supplied_argv_sha256,
            _argv_sha256(tuple(original_argv)),
        )
        or contract["executable_path_sha256"] != _path_sha256(original_argv[0])
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child original argv drifted")
    if type(contract["argv_count"]) is not int or contract["argv_count"] != len(original_argv):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child argv count drifted")
    executable_path_sha256 = _require_sha256(
        contract["executable_path_sha256"], "python executable path"
    )
    executable_source_sha256 = _require_sha256(
        contract["executable_source_sha256"], "python executable source"
    )
    executable_proc_path_sha256 = _require_sha256(
        contract["executable_proc_path_sha256"], "python executable proc path"
    )
    executable_fd = contract["executable_fd"]
    if type(executable_fd) is not int or executable_fd < 0:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child executable descriptor is invalid"
        )
    executable_proc_path = f"/proc/self/fd/{executable_fd}"
    executable_source, executable_metadata = _read_exact_fd(
        executable_fd,
        maximum_bytes=_MAX_EXECUTABLE_BYTES,
        require_single_link=True,
        require_nonempty=True,
    )
    current_executable_metadata = os.stat("/proc/self/exe", follow_symlinks=True)
    if (
        executable_path_sha256 != _path_sha256(sys.executable)
        or executable_proc_path_sha256 != _path_sha256(executable_proc_path)
        or _locator_identity(executable_metadata) != _locator_identity(current_executable_metadata)
        or not hmac.compare_digest(
            executable_source_sha256,
            hashlib.sha256(executable_source).hexdigest(),
        )
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child executable identity drifted"
        )
    try:
        stdin_probe = os.read(0, 1)
    except OSError as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child stdin could not be checked"
        ) from exc
    if stdin_probe != b"" or os.getsid(0) != os.getpid():
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child stdin or new-session contract drifted"
        )
    result_metadata = os.fstat(result_fd)
    if (
        not stat.S_ISREG(result_metadata.st_mode)
        or result_metadata.st_nlink != 1
        or result_metadata.st_size != 0
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child result descriptor is not one empty single-link regular file"
        )
    return contract


def _child_open_requested_roots(
    request: Mapping[str, Any],
) -> tuple[_DirectoryAnchor, tuple[_DirectoryAnchor, ...]]:
    repository = _require_exact_keys(
        request["repository_root"],
        frozenset({"path", "device", "inode"}),
        "bootstrap request repository root",
    )
    if type(repository["path"]) is not str:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap request repository path is not exact"
        )
    repository_anchor = _open_directory_anchor(
        Path(repository["path"]),
        "child repository_root",
    )
    try:
        if (
            type(repository["device"]) is not int
            or type(repository["inode"]) is not int
            or repository_anchor.identity != (repository["device"], repository["inode"])
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "child repository_root identity drifted"
            )
        roots = request["runtime_import_roots"]
        if type(roots) is not list or not 1 <= len(roots) <= _MAX_RUNTIME_ROOTS:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "child runtime import roots are not exact"
            )
        runtime_anchors: list[_DirectoryAnchor] = []
        for index, item in enumerate(roots):
            record = _require_exact_keys(
                item,
                frozenset({"path", "device", "inode"}),
                f"child runtime root {index}",
            )
            if type(record["path"]) is not str:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "child runtime import path is not exact"
                )
            anchor = _open_directory_anchor(
                Path(record["path"]),
                f"child runtime_import_roots[{index}]",
            )
            if (
                type(record["device"]) is not int
                or type(record["inode"]) is not int
                or anchor.identity != (record["device"], record["inode"])
            ):
                anchor.close()
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "child runtime import root identity drifted"
                )
            runtime_anchors.append(anchor)
        identities = [repository_anchor.identity, *(anchor.identity for anchor in runtime_anchors)]
        if len(set(identities)) != len(identities):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "child repository and runtime roots alias"
            )
        return repository_anchor, tuple(runtime_anchors)
    except BaseException:
        repository_anchor.close()
        for anchor in locals().get("runtime_anchors", []):
            anchor.close()
        raise


def _child_process_record(
    *,
    request_contract: Mapping[str, Any],
    original_stdlib_paths: tuple[str, ...],
    final_sys_path: tuple[str, ...],
    repository_path: str,
    runtime_paths: tuple[str, ...],
    cache_proc_path: str,
) -> dict[str, Any]:
    original_path_digests = tuple(_path_sha256(path) for path in original_stdlib_paths)
    final_path_digests = tuple(_path_sha256(path) for path in final_sys_path)
    return {
        "argv_sha256": request_contract["argv_sha256"],
        "argv_count": request_contract["argv_count"],
        "launcher_sha256": _CHILD_LAUNCHER_SHA256,
        "environment_sha256": request_contract["environment_sha256"],
        "cwd_path_sha256": _path_sha256(repository_path),
        "executable_path_sha256": request_contract["executable_path_sha256"],
        "executable_source_sha256": request_contract["executable_source_sha256"],
        "executable_proc_path_sha256": request_contract["executable_proc_path_sha256"],
        "executable_fd": request_contract["executable_fd"],
        "required_flags": ["-I", "-S", "-B"],
        "isolated": True,
        "site_initialization": False,
        "pth_processing": False,
        "stdin": "devnull_closed_to_input",
        "new_session": True,
        "repository_path_sha256": _path_sha256(repository_path),
        "runtime_import_root_path_sha256": [_path_sha256(path) for path in runtime_paths],
        "original_stdlib_path_sha256": list(original_path_digests),
        "original_stdlib_path_sequence_sha256": _canonical_sequence_sha256(original_path_digests),
        "final_sys_path_entry_sha256": list(final_path_digests),
        "final_sys_path_sequence_sha256": _canonical_sequence_sha256(final_path_digests),
        "final_sys_path_entry_count": len(final_sys_path),
        "dont_write_bytecode": True,
        "pycache_prefix": cache_proc_path,
        "cache_empty_before": True,
        "cache_empty_after": True,
        "jax_platform_selector": "cpu",
        "jax_default_backend": "cpu",
        "jax_devices_all_cpu": True,
        "jax_compilation_cache_enabled": False,
        "home_xdg_cache_redirected": True,
        "python_os_environ_mapping_unchanged": True,
    }


def _verify_child_cpu_runtime() -> None:
    jax_module = sys.modules.get("jax")
    if type(jax_module) is not types.ModuleType:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child did not retain the runner-owned JAX runtime"
        )
    default_backend = getattr(jax_module, "default_backend", None)
    devices_function = getattr(jax_module, "devices", None)
    config = getattr(jax_module, "config", None)
    if not callable(default_backend) or not callable(devices_function) or config is None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child JAX runtime inspection surface is unavailable"
        )
    try:
        observed_backend = default_backend()
        observed_devices = devices_function()
        platforms = getattr(config, "jax_platforms", None)
        compilation_cache_enabled = getattr(
            config,
            "jax_enable_compilation_cache",
            None,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child JAX CPU contract could not be inspected"
        ) from exc
    if (
        observed_backend != "cpu"
        or type(observed_devices) is not list
        or not observed_devices
        or any(getattr(device, "platform", None) != "cpu" for device in observed_devices)
        or platforms != "cpu"
        or compilation_cache_enabled is not False
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child JAX CPU or compilation-cache contract drifted"
        )


def _child_entry(
    *,
    source_fd: int,
    request_fd: int,
    result_fd: int,
    cache_fd: int,
) -> None:
    """Execute the child contract.  Called only by the fixed direct-byte launcher."""

    cache_proc_path = f"/proc/self/fd/{cache_fd}"
    original_stdlib_paths = _child_initial_boundary(cache_proc_path)
    if _activate_child_cache(cache_fd) != cache_proc_path:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child cache activation path drifted"
        )
    request, request_raw, request_sha256 = _child_read_request(request_fd)
    process_contract = _child_validate_process_contract(
        request,
        source_fd=source_fd,
        request_fd=request_fd,
        result_fd=result_fd,
        cache_fd=cache_fd,
        cache_proc_path=cache_proc_path,
    )
    ceilings = cast(dict[str, Any], request["ceilings"])
    maximum_request_bytes = cast(int, ceilings["maximum_request_bytes"])
    if len(request_raw) > maximum_request_bytes:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child request exceeds its exact ceiling"
        )
    manifest_raw, manifest_full_sha256, manifest_tree_sha256 = _decode_manifest_from_request(
        request, maximum_bytes=maximum_request_bytes
    )
    cell = _require_exact_keys(
        request["cell"],
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "bootstrap child cell",
    )
    candidate_id = _validate_candidate_id(cell["candidate_id"])
    environment_seed = _require_uint31(cell["environment_seed"], "environment seed")
    agent_seed = _require_uint31(cell["agent_seed"], "agent seed")
    repository_anchor, runtime_anchors = _child_open_requested_roots(request)
    snapshot_module: types.ModuleType | None = None
    try:
        repository_path = str(repository_anchor.path)
        runtime_paths = tuple(str(anchor.path) for anchor in runtime_anchors)
        if os.getcwd() != repository_path:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap child working directory drifted"
            )
        snapshot_source = _read_anchored_relative_file(
            repository_anchor,
            _SNAPSHOT_RELATIVE_PATH,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        if len(snapshot_source) != request["source_snapshot"]["snapshot_source_size"]:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "snapshot module source size drifted"
            )
        snapshot_module = _direct_load_snapshot_module(
            snapshot_source,
            f"{repository_path}/{_SNAPSHOT_RELATIVE_PATH}",
        )
        _parse_snapshot_with_exact_module(
            snapshot_module,
            manifest_raw,
            manifest_full_sha256,
        )
        pre_snapshot = _verify_snapshot_with_exact_module(
            snapshot_module,
            repository_root=repository_anchor.path,
            raw=manifest_raw,
            expected_full_sha256=manifest_full_sha256,
        )
        if pre_snapshot.tree_sha256 != manifest_tree_sha256:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "pre-execution source snapshot tree drifted"
            )
        if not _directory_is_empty(cache_fd):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap cache changed before runtime imports"
            )
        if _live_forbidden_modules():
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "runtime dependencies appeared before the runner-owned import transition"
            )
        final_sys_path = (repository_path, *runtime_paths, *original_stdlib_paths)
        if len(set(final_sys_path)) != len(final_sys_path):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap child sys.path contains an alias"
            )
        sys.path[:] = list(final_sys_path)
        sys.path_importer_cache.clear()
        if tuple(sys.path) != final_sys_path:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap child sys.path sanitization drifted"
            )
        runner_source = _read_anchored_relative_file(
            repository_anchor,
            _RUNNER_RELATIVE_PATH,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        if len(runner_source) != request["source_snapshot"][
            "runner_source_size"
        ] or not hmac.compare_digest(
            hashlib.sha256(runner_source).hexdigest(),
            PINNED_LOCAL_RUNNER_SOURCE_SHA256,
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "local runner exact source bytes drifted"
            )
        runner_name = "_alberta_forager_matched_v3_local_runner_isolated_v1"
        if runner_name in sys.modules:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "local runner isolated module name is already occupied"
            )
        runner = types.ModuleType(runner_name)
        runner.__file__ = f"{repository_path}/{_RUNNER_RELATIVE_PATH}"
        runner.__package__ = ""
        runner.__dict__["_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"] = (
            PINNED_LOCAL_RUNNER_SOURCE_SHA256
        )
        sys.modules[runner_name] = runner
        try:
            exec(compile(runner_source, runner.__file__, "exec"), runner.__dict__)
            if (
                getattr(runner, "LOCAL_RUNNER_DESCRIPTOR_SHA256", None)
                != PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256
            ):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "local runner descriptor identity drifted"
                )
            issue = getattr(runner, "issue_matched_v3_local_execution_capability", None)
            run = getattr(runner, "run_matched_v3_local_candidate", None)
            consume = getattr(runner, "consume_matched_v3_local_outcome", None)
            if not callable(issue) or not callable(run) or not callable(consume):
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "local runner capability surface is unavailable"
                )
            runner_capability = issue(explicit_execution_opt_in=True)
            runner_outcome = run(
                candidate_id=candidate_id,
                environment_seed=environment_seed,
                agent_seed=agent_seed,
                explicit_execution_opt_in=True,
                execution_capability=runner_capability,
            )
            completion = consume(
                outcome_capability=runner_outcome,
                explicit_content_access_opt_in=True,
            )
        except ForagerMatchedV3LocalExecutionBootstrapError:
            raise
        except Exception as exc:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "local runner execution failed inside the verified child boundary"
            ) from exc
        local_receipt = getattr(completion, "canonical_receipt_bytes", None)
        reward_trace = getattr(completion, "reward_trace", None)
        if type(local_receipt) is not bytes or type(reward_trace) is not bytes:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "local runner completion content is not exact immutable bytes"
            )
        _verify_child_cpu_runtime()
        post_snapshot = _verify_snapshot_with_exact_module(
            snapshot_module,
            repository_root=repository_anchor.path,
            raw=manifest_raw,
            expected_full_sha256=manifest_full_sha256,
        )
        if (
            post_snapshot.tree_sha256 != manifest_tree_sha256
            or pre_snapshot.full_sha256 != post_snapshot.full_sha256
            or pre_snapshot.tree_sha256 != post_snapshot.tree_sha256
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "post-execution source snapshot drifted"
            )
        if (
            tuple(sys.path) != final_sys_path
            or not sys.dont_write_bytecode
            or sys.pycache_prefix != cache_proc_path
            or dict(os.environ) != _child_environment(cache_proc_path)
            or not _directory_is_empty(cache_fd)
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap import or cache contract drifted during execution"
            )
        repository_anchor.verify()
        for anchor in runtime_anchors:
            anchor.verify()
        child_process = _child_process_record(
            request_contract=process_contract,
            original_stdlib_paths=original_stdlib_paths,
            final_sys_path=final_sys_path,
            repository_path=repository_path,
            runtime_paths=runtime_paths,
            cache_proc_path=cache_proc_path,
        )
        child_record = _build_child_record(
            bootstrap_source_sha256=_require_sha256(
                _CHILD_SOURCE_SHA256_INPUT,
                "bootstrap child source",
            ),
            request_sha256=request_sha256,
            manifest_full_sha256=manifest_full_sha256,
            manifest_tree_sha256=manifest_tree_sha256,
            pre_full_sha256=pre_snapshot.full_sha256,
            pre_tree_sha256=pre_snapshot.tree_sha256,
            post_full_sha256=post_snapshot.full_sha256,
            post_tree_sha256=post_snapshot.tree_sha256,
            candidate_id=candidate_id,
            environment_seed=environment_seed,
            agent_seed=agent_seed,
            process_contract=child_process,
            local_receipt=local_receipt,
            reward_trace=reward_trace,
        )
        frame = _result_frame(child_record, local_receipt, reward_trace)
        _write_all(
            result_fd,
            frame,
            maximum_bytes=cast(int, ceilings["maximum_result_bytes"]),
        )
    finally:
        if snapshot_module is not None:
            sys.modules.pop(_SNAPSHOT_MODULE_NAME, None)
        for anchor in reversed(runtime_anchors):
            anchor.close()
        repository_anchor.close()


@dataclass(slots=True)
class _ScratchArtifacts:
    anchor: _DirectoryAnchor
    source_fd: int
    request_fd: int
    result_fd: int
    cache_fd: int
    source_identity: tuple[int, int]
    request_identity: tuple[int, int]
    result_identity: tuple[int, int]
    cache_identity: tuple[int, int]

    def _verify_named(
        self,
        name: str,
        descriptor: int,
        identity: tuple[int, int],
        mode: int,
    ) -> None:
        try:
            opened = os.fstat(descriptor)
            current = os.stat(name, dir_fd=self.anchor.descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap scratch artifact locator changed"
            ) from exc
        if (
            (opened.st_dev, opened.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or stat.S_IMODE(opened.st_mode) != mode
            or stat.S_IMODE(current.st_mode) != mode
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap scratch artifact locator changed"
            )

    def verify(self) -> None:
        self.anchor.verify()
        self._verify_named(_SCRATCH_SOURCE_NAME, self.source_fd, self.source_identity, 0o600)
        self._verify_named(_SCRATCH_REQUEST_NAME, self.request_fd, self.request_identity, 0o600)
        self._verify_named(_SCRATCH_RESULT_NAME, self.result_fd, self.result_identity, 0o600)
        self._verify_named(_SCRATCH_CACHE_NAME, self.cache_fd, self.cache_identity, 0o700)
        for descriptor in (self.source_fd, self.request_fd, self.result_fd):
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "bootstrap scratch file lost its single-link regular identity"
                )
        if not stat.S_ISDIR(os.fstat(self.cache_fd).st_mode):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap cache lost its directory identity"
            )

    def close_and_remove(self) -> None:
        for descriptor in (self.cache_fd, self.result_fd, self.request_fd, self.source_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass
        cleanup_errors: list[BaseException] = []
        try:
            os.rmdir(_SCRATCH_CACHE_NAME, dir_fd=self.anchor.descriptor)
        except OSError as exc:
            cleanup_errors.append(exc)
        for name in (_SCRATCH_RESULT_NAME, _SCRATCH_REQUEST_NAME, _SCRATCH_SOURCE_NAME):
            try:
                os.unlink(name, dir_fd=self.anchor.descriptor)
            except OSError as exc:
                cleanup_errors.append(exc)
        self.anchor.close()
        if cleanup_errors:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap scratch artifacts could not be removed exactly"
            ) from cleanup_errors[0]


def _create_scratch_artifacts(
    scratch_anchor: _DirectoryAnchor,
    *,
    bootstrap_source: bytes,
    maximum_bootstrap_source_bytes: int,
) -> _ScratchArtifacts:
    metadata = os.fstat(scratch_anchor.descriptor)
    if (
        stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or not _directory_is_empty(scratch_anchor.descriptor)
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "scratch_parent must be one new private empty mode-0700 directory"
        )
    if (
        type(bootstrap_source) is not bytes
        or not 1 <= len(bootstrap_source) <= maximum_bootstrap_source_bytes
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap source exceeds its exact caller ceiling"
        )
    descriptors: list[int] = []
    created_files: list[str] = []
    cache_created = False
    try:
        create_flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        source_fd = os.open(
            _SCRATCH_SOURCE_NAME,
            create_flags,
            0o600,
            dir_fd=scratch_anchor.descriptor,
        )
        descriptors.append(source_fd)
        created_files.append(_SCRATCH_SOURCE_NAME)
        _write_all(source_fd, bootstrap_source, maximum_bytes=maximum_bootstrap_source_bytes)
        os.lseek(source_fd, 0, os.SEEK_SET)
        request_fd = os.open(
            _SCRATCH_REQUEST_NAME,
            create_flags,
            0o600,
            dir_fd=scratch_anchor.descriptor,
        )
        descriptors.append(request_fd)
        created_files.append(_SCRATCH_REQUEST_NAME)
        result_fd = os.open(
            _SCRATCH_RESULT_NAME,
            create_flags,
            0o600,
            dir_fd=scratch_anchor.descriptor,
        )
        descriptors.append(result_fd)
        created_files.append(_SCRATCH_RESULT_NAME)
        os.mkdir(_SCRATCH_CACHE_NAME, 0o700, dir_fd=scratch_anchor.descriptor)
        cache_created = True
        cache_fd = os.open(
            _SCRATCH_CACHE_NAME,
            _directory_flags(),
            dir_fd=scratch_anchor.descriptor,
        )
        descriptors.append(cache_fd)
        source_stat = os.fstat(source_fd)
        request_stat = os.fstat(request_fd)
        result_stat = os.fstat(result_fd)
        cache_stat = os.fstat(cache_fd)
        artifacts = _ScratchArtifacts(
            scratch_anchor,
            source_fd,
            request_fd,
            result_fd,
            cache_fd,
            (source_stat.st_dev, source_stat.st_ino),
            (request_stat.st_dev, request_stat.st_ino),
            (result_stat.st_dev, result_stat.st_ino),
            (cache_stat.st_dev, cache_stat.st_ino),
        )
        artifacts.verify()
        return artifacts
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if cache_created:
            try:
                os.rmdir(_SCRATCH_CACHE_NAME, dir_fd=scratch_anchor.descriptor)
            except OSError:
                pass
        for name in reversed(created_files):
            try:
                os.unlink(name, dir_fd=scratch_anchor.descriptor)
            except OSError:
                pass
        raise


def _write_request_file(
    artifacts: _ScratchArtifacts,
    raw: bytes,
    *,
    maximum_request_bytes: int,
) -> None:
    _write_all(artifacts.request_fd, raw, maximum_bytes=maximum_request_bytes)
    os.lseek(artifacts.request_fd, 0, os.SEEK_SET)
    artifacts.verify()


def _validate_child_record(
    raw: bytes,
    *,
    bootstrap_source_sha256: str,
    request_sha256: str,
    manifest_identity: _SnapshotManifestIdentity,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    expected_process_contract: Mapping[str, Any],
    repository_path: str,
    runtime_paths: tuple[str, ...],
    local_receipt: bytes,
    reward_trace: bytes,
) -> dict[str, Any]:
    record = _strict_json_load(raw)
    _require_exact_keys(
        record,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "bootstrap_source_sha256",
                "request_sha256",
                "source_snapshot",
                "runner",
                "cell",
                "process_contract",
                "completion",
                "claims",
                "limitations",
                "child_record_body_sha256",
            }
        ),
        "bootstrap child record",
    )
    if (
        record["schema_version"] != LOCAL_EXECUTION_BOOTSTRAP_CHILD_RECORD_SCHEMA_VERSION
        or record["status"] != "child_completed_unqualified_non_authorizing"
        or record["classification"] != "verified_endpoint_execution_content_only"
        or not _exact_json_equal(
            record["descriptor_binding"],
            {
                "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
            },
        )
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child record identity drifted"
        )
    if (
        record["bootstrap_source_sha256"] != bootstrap_source_sha256
        or record["request_sha256"] != request_sha256
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child request or source binding drifted"
        )
    snapshot = _require_exact_keys(
        record["source_snapshot"],
        frozenset(
            {
                "descriptor_sha256",
                "source_sha256",
                "expected_full_sha256",
                "expected_tree_sha256",
                "pre_full_sha256",
                "pre_tree_sha256",
                "post_full_sha256",
                "post_tree_sha256",
                "continuous_immutability_attested",
            }
        ),
        "bootstrap child source snapshot",
    )
    if not _exact_json_equal(
        snapshot,
        {
            "descriptor_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
            "expected_full_sha256": manifest_identity.full_sha256,
            "expected_tree_sha256": manifest_identity.tree_sha256,
            "pre_full_sha256": manifest_identity.full_sha256,
            "pre_tree_sha256": manifest_identity.tree_sha256,
            "post_full_sha256": manifest_identity.full_sha256,
            "post_tree_sha256": manifest_identity.tree_sha256,
            "continuous_immutability_attested": False,
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child pre/post source linkage drifted"
        )
    if not _exact_json_equal(
        record["runner"],
        {
            "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
            "isolated_module_name": "_alberta_forager_matched_v3_local_runner_isolated_v1",
            "own_capabilities_issued_run_consumed": True,
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child runner binding drifted")
    if not _exact_json_equal(
        record["cell"],
        {
            "candidate_id": candidate_id,
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap child cell identity drifted")
    process = _require_exact_keys(
        record["process_contract"],
        frozenset(
            {
                "argv_sha256",
                "argv_count",
                "launcher_sha256",
                "environment_sha256",
                "cwd_path_sha256",
                "executable_path_sha256",
                "executable_source_sha256",
                "executable_proc_path_sha256",
                "executable_fd",
                "required_flags",
                "isolated",
                "site_initialization",
                "pth_processing",
                "stdin",
                "new_session",
                "repository_path_sha256",
                "runtime_import_root_path_sha256",
                "original_stdlib_path_sha256",
                "original_stdlib_path_sequence_sha256",
                "final_sys_path_entry_sha256",
                "final_sys_path_sequence_sha256",
                "final_sys_path_entry_count",
                "dont_write_bytecode",
                "pycache_prefix",
                "cache_empty_before",
                "cache_empty_after",
                "jax_platform_selector",
                "jax_default_backend",
                "jax_devices_all_cpu",
                "jax_compilation_cache_enabled",
                "home_xdg_cache_redirected",
                "python_os_environ_mapping_unchanged",
            }
        ),
        "bootstrap child process record",
    )
    exact_fields = {
        "argv_sha256": expected_process_contract["argv_sha256"],
        "argv_count": expected_process_contract["argv_count"],
        "launcher_sha256": _CHILD_LAUNCHER_SHA256,
        "environment_sha256": expected_process_contract["environment_sha256"],
        "cwd_path_sha256": _path_sha256(repository_path),
        "executable_path_sha256": expected_process_contract["executable_path_sha256"],
        "executable_source_sha256": expected_process_contract["executable_source_sha256"],
        "executable_proc_path_sha256": expected_process_contract["executable_proc_path_sha256"],
        "executable_fd": expected_process_contract["executable_fd"],
        "required_flags": ["-I", "-S", "-B"],
        "isolated": True,
        "site_initialization": False,
        "pth_processing": False,
        "stdin": "devnull_closed_to_input",
        "new_session": True,
        "repository_path_sha256": _path_sha256(repository_path),
        "runtime_import_root_path_sha256": [_path_sha256(path) for path in runtime_paths],
        "dont_write_bytecode": True,
        "pycache_prefix": expected_process_contract["cache_proc_path"],
        "cache_empty_before": True,
        "cache_empty_after": True,
        "jax_platform_selector": "cpu",
        "jax_default_backend": "cpu",
        "jax_devices_all_cpu": True,
        "jax_compilation_cache_enabled": False,
        "home_xdg_cache_redirected": True,
        "python_os_environ_mapping_unchanged": True,
    }
    for key, value in exact_fields.items():
        if not _exact_json_equal(process[key], value):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap child process linkage drifted: {key}"
            )
    original_digests = process["original_stdlib_path_sha256"]
    final_digests = process["final_sys_path_entry_sha256"]
    if (
        type(original_digests) is not list
        or not 1 <= len(original_digests) <= 32
        or any(
            type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64
            for value in original_digests
        )
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child original stdlib path identities drifted"
        )
    expected_final = [
        _path_sha256(repository_path),
        *(_path_sha256(path) for path in runtime_paths),
        *cast(list[str], original_digests),
    ]
    if (
        not _exact_json_equal(final_digests, expected_final)
        or process["original_stdlib_path_sequence_sha256"]
        != _canonical_sequence_sha256(tuple(cast(list[str], original_digests)))
        or process["final_sys_path_sequence_sha256"]
        != _canonical_sequence_sha256(tuple(expected_final))
        or type(process["final_sys_path_entry_count"]) is not int
        or process["final_sys_path_entry_count"] != len(expected_final)
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child final sys.path linkage drifted"
        )
    completion = _require_exact_keys(
        record["completion"],
        frozenset(
            {
                "local_receipt_size_bytes",
                "local_receipt_sha256",
                "reward_trace_size_bytes",
                "reward_trace_sha256",
            }
        ),
        "bootstrap child completion",
    )
    if not _exact_json_equal(
        completion,
        {
            "local_receipt_size_bytes": len(local_receipt),
            "local_receipt_sha256": hashlib.sha256(local_receipt).hexdigest(),
            "reward_trace_size_bytes": len(reward_trace),
            "reward_trace_sha256": hashlib.sha256(reward_trace).hexdigest(),
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child completion linkage drifted"
        )
    if not _exact_json_equal(record["claims"], _claims()) or not _exact_json_equal(
        record["limitations"], _limitations()
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child claims or limitations drifted"
        )
    body = dict(record)
    supplied_body = _require_sha256(
        body.pop("child_record_body_sha256"),
        "bootstrap child record body",
    )
    if not hmac.compare_digest(
        supplied_body,
        hashlib.sha256(_canonical_json(body)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap child record body identity drifted"
        )
    return record


def _direct_load_runner_parser(source: bytes, source_path: str) -> types.ModuleType:
    if type(source) is not bytes or not hmac.compare_digest(
        hashlib.sha256(source).hexdigest(),
        PINNED_LOCAL_RUNNER_SOURCE_SHA256,
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "parent runner parser source identity drifted"
        )
    if _PARENT_RUNNER_PARSER_MODULE_NAME in sys.modules:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "parent runner parser module name is already occupied"
        )
    module = types.ModuleType(_PARENT_RUNNER_PARSER_MODULE_NAME)
    module.__file__ = source_path
    module.__package__ = ""
    sys.modules[_PARENT_RUNNER_PARSER_MODULE_NAME] = module
    try:
        exec(compile(source, source_path, "exec"), module.__dict__)
        if (
            getattr(module, "LOCAL_RUNNER_DESCRIPTOR_SHA256", None)
            != PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "parent runner parser descriptor identity drifted"
            )
        return module
    except BaseException:
        sys.modules.pop(_PARENT_RUNNER_PARSER_MODULE_NAME, None)
        raise


def _replay_local_completion(
    runner_parser: types.ModuleType,
    *,
    local_receipt: bytes,
    reward_trace: bytes,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
) -> dict[str, Any]:
    parser = getattr(runner_parser, "parse_matched_v3_local_completion_receipt", None)
    if not callable(parser):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "parent runner strict completion parser is unavailable"
        )
    receipt_sha256 = hashlib.sha256(local_receipt).hexdigest()
    try:
        receipt = parser(
            local_receipt,
            reward_trace=reward_trace,
            expected_receipt_sha256=receipt_sha256,
        )
    except Exception as exc:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "parent runner strict completion replay failed"
        ) from exc
    if (
        type(receipt) is not dict
        or receipt["candidate"]["candidate_id"] != candidate_id
        or receipt["seed_transport"]["environment_seed"] != environment_seed
        or receipt["seed_transport"]["agent_seed"] != agent_seed
        or receipt["bindings"]["local_runner_descriptor"]["sha256"]
        != PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256
        or receipt["bindings"]["relevant_source_sha256"]["local_runner_observed"]
        != PINNED_LOCAL_RUNNER_SOURCE_SHA256
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "parent runner completion linkage drifted"
        )
    return cast(dict[str, Any], receipt)


def _bootstrap_receipt_body(
    *,
    bootstrap_source_sha256: str,
    manifest_identity: _SnapshotManifestIdentity,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    process_contract: Mapping[str, Any],
    process_result: _ProcessResult,
    frame: bytes,
    child_record: bytes,
    local_receipt: bytes,
    reward_trace: bytes,
    ceilings: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_EXECUTION_BOOTSTRAP_RECEIPT_SCHEMA_VERSION,
        "status": "completed_unqualified_non_authorizing",
        "classification": "parent_validated_bootstrap_content_only",
        "descriptor_binding": {
            "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
        },
        "bootstrap_source_sha256": bootstrap_source_sha256,
        "source_snapshot": {
            "descriptor_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256,
            "expected_full_sha256": manifest_identity.full_sha256,
            "expected_tree_sha256": manifest_identity.tree_sha256,
            "pre_full_sha256": manifest_identity.full_sha256,
            "pre_tree_sha256": manifest_identity.tree_sha256,
            "post_full_sha256": manifest_identity.full_sha256,
            "post_tree_sha256": manifest_identity.tree_sha256,
            "continuous_immutability_attested": False,
        },
        "runner": {
            "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
        },
        "cell": {
            "candidate_id": candidate_id,
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
        },
        "process": {
            "argv_sha256": process_contract["argv_sha256"],
            "argv_count": process_contract["argv_count"],
            "launcher_sha256": _CHILD_LAUNCHER_SHA256,
            "environment_sha256": process_contract["environment_sha256"],
            "cwd_path_sha256": process_contract["cwd_path_sha256"],
            "executable_path_sha256": process_contract["executable_path_sha256"],
            "executable_source_sha256": process_contract["executable_source_sha256"],
            "required_flags": ["-I", "-S", "-B"],
            "stdin": "devnull_closed_to_input",
            "new_session": True,
            "jax_platform_selector": process_contract["jax_platform_selector"],
            "jax_compilation_cache_enabled": process_contract["jax_compilation_cache_enabled"],
            "home_xdg_cache_redirected": True,
            "returncode": process_result.returncode,
            "stdout_size_bytes": len(process_result.stdout),
            "stdout_sha256": hashlib.sha256(process_result.stdout).hexdigest(),
            "stderr_size_bytes": len(process_result.stderr),
            "stderr_sha256": hashlib.sha256(process_result.stderr).hexdigest(),
            "ceilings": dict(ceilings),
        },
        "result": {
            "framing": "magic_three_u64_lengths_payloads_v1",
            "frame_size_bytes": len(frame),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "child_record_size_bytes": len(child_record),
            "child_record_sha256": hashlib.sha256(child_record).hexdigest(),
            "local_receipt_size_bytes": len(local_receipt),
            "local_receipt_sha256": hashlib.sha256(local_receipt).hexdigest(),
            "reward_trace_size_bytes": len(reward_trace),
            "reward_trace_sha256": hashlib.sha256(reward_trace).hexdigest(),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _build_bootstrap_receipt(**kwargs: Any) -> bytes:
    body = _bootstrap_receipt_body(**kwargs)
    return _canonical_json(
        {
            **body,
            "receipt_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
        }
    )


def _validate_bootstrap_receipt(
    value: Mapping[str, Any],
    *,
    child_record: bytes,
    local_receipt: bytes,
    reward_trace: bytes,
    stdout: bytes,
    stderr: bytes,
) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "bootstrap_source_sha256",
                "source_snapshot",
                "runner",
                "cell",
                "process",
                "result",
                "claims",
                "limitations",
                "receipt_body_sha256",
            }
        ),
        "bootstrap receipt",
    )
    if (
        receipt["schema_version"] != LOCAL_EXECUTION_BOOTSTRAP_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "completed_unqualified_non_authorizing"
        or receipt["classification"] != "parent_validated_bootstrap_content_only"
        or not _exact_json_equal(
            receipt["descriptor_binding"],
            {
                "schema_version": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
            },
        )
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap receipt identity drifted")
    _require_sha256(receipt["bootstrap_source_sha256"], "bootstrap receipt source")
    snapshot = _require_exact_keys(
        receipt["source_snapshot"],
        frozenset(
            {
                "descriptor_sha256",
                "source_sha256",
                "expected_full_sha256",
                "expected_tree_sha256",
                "pre_full_sha256",
                "pre_tree_sha256",
                "post_full_sha256",
                "post_tree_sha256",
                "continuous_immutability_attested",
            }
        ),
        "bootstrap receipt source snapshot",
    )
    for key in (
        "expected_full_sha256",
        "expected_tree_sha256",
        "pre_full_sha256",
        "pre_tree_sha256",
        "post_full_sha256",
        "post_tree_sha256",
    ):
        _require_sha256(snapshot[key], f"bootstrap receipt source snapshot {key}")
    if (
        snapshot["descriptor_sha256"] != PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256
        or snapshot["source_sha256"] != PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256
        or snapshot["expected_full_sha256"] != snapshot["pre_full_sha256"]
        or snapshot["expected_full_sha256"] != snapshot["post_full_sha256"]
        or snapshot["expected_tree_sha256"] != snapshot["pre_tree_sha256"]
        or snapshot["expected_tree_sha256"] != snapshot["post_tree_sha256"]
        or snapshot["continuous_immutability_attested"] is not False
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt pre/post source linkage drifted"
        )
    if not _exact_json_equal(
        receipt["runner"],
        {
            "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt runner binding drifted"
        )
    cell = _require_exact_keys(
        receipt["cell"],
        frozenset({"candidate_id", "environment_seed", "agent_seed"}),
        "bootstrap receipt cell",
    )
    _validate_candidate_id(cell["candidate_id"])
    _require_uint31(cell["environment_seed"], "bootstrap receipt environment seed")
    _require_uint31(cell["agent_seed"], "bootstrap receipt agent seed")
    child = _strict_json_load(child_record)
    _require_exact_keys(
        child,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "bootstrap_source_sha256",
                "request_sha256",
                "source_snapshot",
                "runner",
                "cell",
                "process_contract",
                "completion",
                "claims",
                "limitations",
                "child_record_body_sha256",
            }
        ),
        "bootstrap receipt child record",
    )
    if (
        child["schema_version"] != LOCAL_EXECUTION_BOOTSTRAP_CHILD_RECORD_SCHEMA_VERSION
        or child["status"] != "child_completed_unqualified_non_authorizing"
        or child["classification"] != "verified_endpoint_execution_content_only"
        or not _exact_json_equal(child["descriptor_binding"], receipt["descriptor_binding"])
        or child["bootstrap_source_sha256"] != receipt["bootstrap_source_sha256"]
        or not _exact_json_equal(child["source_snapshot"], receipt["source_snapshot"])
        or not _exact_json_equal(
            child["runner"],
            {
                "descriptor_sha256": PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256,
                "source_sha256": PINNED_LOCAL_RUNNER_SOURCE_SHA256,
                "isolated_module_name": ("_alberta_forager_matched_v3_local_runner_isolated_v1"),
                "own_capabilities_issued_run_consumed": True,
            },
        )
        or not _exact_json_equal(child["cell"], receipt["cell"])
        or not _exact_json_equal(child["claims"], _claims())
        or not _exact_json_equal(child["limitations"], _limitations())
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt child-record linkage drifted"
        )
    _require_sha256(child["request_sha256"], "bootstrap receipt child request")
    child_completion = _require_exact_keys(
        child["completion"],
        frozenset(
            {
                "local_receipt_size_bytes",
                "local_receipt_sha256",
                "reward_trace_size_bytes",
                "reward_trace_sha256",
            }
        ),
        "bootstrap receipt child completion",
    )
    if not _exact_json_equal(
        child_completion,
        {
            "local_receipt_size_bytes": len(local_receipt),
            "local_receipt_sha256": hashlib.sha256(local_receipt).hexdigest(),
            "reward_trace_size_bytes": len(reward_trace),
            "reward_trace_sha256": hashlib.sha256(reward_trace).hexdigest(),
        },
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt child completion linkage drifted"
        )
    child_body = dict(child)
    supplied_child_body = _require_sha256(
        child_body.pop("child_record_body_sha256"),
        "bootstrap receipt child body",
    )
    if not hmac.compare_digest(
        supplied_child_body,
        hashlib.sha256(_canonical_json(child_body)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt child body identity drifted"
        )
    process = _require_exact_keys(
        receipt["process"],
        frozenset(
            {
                "argv_sha256",
                "argv_count",
                "launcher_sha256",
                "environment_sha256",
                "cwd_path_sha256",
                "executable_path_sha256",
                "executable_source_sha256",
                "required_flags",
                "stdin",
                "new_session",
                "jax_platform_selector",
                "jax_compilation_cache_enabled",
                "home_xdg_cache_redirected",
                "returncode",
                "stdout_size_bytes",
                "stdout_sha256",
                "stderr_size_bytes",
                "stderr_sha256",
                "ceilings",
            }
        ),
        "bootstrap receipt process",
    )
    for key in (
        "argv_sha256",
        "launcher_sha256",
        "environment_sha256",
        "cwd_path_sha256",
        "executable_path_sha256",
        "executable_source_sha256",
        "stdout_sha256",
        "stderr_sha256",
    ):
        _require_sha256(process[key], f"bootstrap receipt process {key}")
    ceilings = _require_exact_keys(
        process["ceilings"],
        frozenset(
            {
                "maximum_request_bytes",
                "maximum_bootstrap_source_bytes",
                "maximum_result_bytes",
                "maximum_stdout_bytes",
                "maximum_stderr_bytes",
                "timeout_seconds",
            }
        ),
        "bootstrap receipt ceilings",
    )
    _bounded_execution_ceiling(
        ceilings["maximum_request_bytes"], "maximum_request_bytes", _MAX_JSON_BYTES
    )
    _bounded_execution_ceiling(
        ceilings["maximum_bootstrap_source_bytes"],
        "maximum_bootstrap_source_bytes",
        _MAX_SOURCE_FILE_BYTES,
    )
    _bounded_execution_ceiling(
        ceilings["maximum_result_bytes"], "maximum_result_bytes", _MAX_JSON_BYTES
    )
    _bounded_execution_ceiling(
        ceilings["maximum_stdout_bytes"], "maximum_stdout_bytes", _MAX_JSON_BYTES
    )
    _bounded_execution_ceiling(
        ceilings["maximum_stderr_bytes"], "maximum_stderr_bytes", _MAX_JSON_BYTES
    )
    _require_timeout(ceilings["timeout_seconds"])
    child_process = _require_exact_keys(
        child["process_contract"],
        frozenset(
            {
                "argv_sha256",
                "argv_count",
                "launcher_sha256",
                "environment_sha256",
                "cwd_path_sha256",
                "executable_path_sha256",
                "executable_source_sha256",
                "executable_proc_path_sha256",
                "executable_fd",
                "required_flags",
                "isolated",
                "site_initialization",
                "pth_processing",
                "stdin",
                "new_session",
                "repository_path_sha256",
                "runtime_import_root_path_sha256",
                "original_stdlib_path_sha256",
                "original_stdlib_path_sequence_sha256",
                "final_sys_path_entry_sha256",
                "final_sys_path_sequence_sha256",
                "final_sys_path_entry_count",
                "dont_write_bytecode",
                "pycache_prefix",
                "cache_empty_before",
                "cache_empty_after",
                "jax_platform_selector",
                "jax_default_backend",
                "jax_devices_all_cpu",
                "jax_compilation_cache_enabled",
                "home_xdg_cache_redirected",
                "python_os_environ_mapping_unchanged",
            }
        ),
        "bootstrap receipt child process",
    )
    for key in (
        "argv_sha256",
        "environment_sha256",
        "cwd_path_sha256",
        "executable_path_sha256",
        "executable_source_sha256",
    ):
        if child_process[key] != process[key]:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap receipt child process linkage drifted: {key}"
            )
    for key in (
        "argv_sha256",
        "launcher_sha256",
        "environment_sha256",
        "cwd_path_sha256",
        "executable_path_sha256",
        "executable_source_sha256",
        "executable_proc_path_sha256",
        "repository_path_sha256",
        "original_stdlib_path_sequence_sha256",
        "final_sys_path_sequence_sha256",
    ):
        _require_sha256(child_process[key], f"bootstrap receipt child process {key}")
    runtime_path_digests = child_process["runtime_import_root_path_sha256"]
    original_path_digests = child_process["original_stdlib_path_sha256"]
    final_path_digests = child_process["final_sys_path_entry_sha256"]
    for values, label, minimum in (
        (runtime_path_digests, "runtime roots", 1),
        (original_path_digests, "stdlib roots", 1),
        (final_path_digests, "final sys.path", 3),
    ):
        if (
            type(values) is not list
            or not minimum <= len(values) <= 64
            or any(
                type(item) is not str or _SHA256_RE.fullmatch(item) is None or item == "0" * 64
                for item in values
            )
            or len(set(values)) != len(values)
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap receipt child {label} identities drifted"
            )
    expected_final_path_digests = [
        child_process["repository_path_sha256"],
        *cast(list[str], runtime_path_digests),
        *cast(list[str], original_path_digests),
    ]
    if (
        not _exact_json_equal(final_path_digests, expected_final_path_digests)
        or child_process["cwd_path_sha256"] != child_process["repository_path_sha256"]
        or child_process["original_stdlib_path_sequence_sha256"]
        != _canonical_sequence_sha256(tuple(cast(list[str], original_path_digests)))
        or child_process["final_sys_path_sequence_sha256"]
        != _canonical_sequence_sha256(tuple(expected_final_path_digests))
        or type(child_process["final_sys_path_entry_count"]) is not int
        or child_process["final_sys_path_entry_count"] != len(expected_final_path_digests)
        or type(child_process["executable_fd"]) is not int
        or child_process["executable_fd"] < 0
        or type(child_process["pycache_prefix"]) is not str
        or re.fullmatch(r"/proc/self/fd/[0-9]+", child_process["pycache_prefix"]) is None
        or child_process["required_flags"] != ["-I", "-S", "-B"]
        or child_process["isolated"] is not True
        or child_process["site_initialization"] is not False
        or child_process["pth_processing"] is not False
        or child_process["stdin"] != "devnull_closed_to_input"
        or child_process["new_session"] is not True
        or child_process["dont_write_bytecode"] is not True
        or child_process["cache_empty_before"] is not True
        or child_process["cache_empty_after"] is not True
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt child import or process contract drifted"
        )
    if (
        child_process["launcher_sha256"] != _CHILD_LAUNCHER_SHA256
        or process["launcher_sha256"] != _CHILD_LAUNCHER_SHA256
        or type(process["argv_count"]) is not int
        or process["argv_count"] != 12
        or child_process["argv_count"] != 12
        or child_process["jax_platform_selector"] != "cpu"
        or child_process["jax_default_backend"] != "cpu"
        or child_process["jax_devices_all_cpu"] is not True
        or child_process["jax_compilation_cache_enabled"] is not False
        or child_process["home_xdg_cache_redirected"] is not True
        or child_process["python_os_environ_mapping_unchanged"] is not True
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt child CPU process contract drifted"
        )
    if (
        process["returncode"] != 0
        or type(process["returncode"]) is not int
        or type(process["stdout_size_bytes"]) is not int
        or type(process["stderr_size_bytes"]) is not int
        or process["stdout_size_bytes"] != len(stdout)
        or process["stdout_sha256"] != hashlib.sha256(stdout).hexdigest()
        or process["stderr_size_bytes"] != len(stderr)
        or process["stderr_sha256"] != hashlib.sha256(stderr).hexdigest()
        or process["required_flags"] != ["-I", "-S", "-B"]
        or process["stdin"] != "devnull_closed_to_input"
        or process["new_session"] is not True
        or process["jax_platform_selector"] != "cpu"
        or process["jax_compilation_cache_enabled"] is not False
        or process["home_xdg_cache_redirected"] is not True
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt process result drifted"
        )
    result = _require_exact_keys(
        receipt["result"],
        frozenset(
            {
                "framing",
                "frame_size_bytes",
                "frame_sha256",
                "child_record_size_bytes",
                "child_record_sha256",
                "local_receipt_size_bytes",
                "local_receipt_sha256",
                "reward_trace_size_bytes",
                "reward_trace_sha256",
            }
        ),
        "bootstrap receipt result",
    )
    frame = _result_frame(child_record, local_receipt, reward_trace)
    expected_result = {
        "frame_size_bytes": len(frame),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "child_record_size_bytes": len(child_record),
        "child_record_sha256": hashlib.sha256(child_record).hexdigest(),
        "local_receipt_size_bytes": len(local_receipt),
        "local_receipt_sha256": hashlib.sha256(local_receipt).hexdigest(),
        "reward_trace_size_bytes": len(reward_trace),
        "reward_trace_sha256": hashlib.sha256(reward_trace).hexdigest(),
    }
    for key, expected in expected_result.items():
        if type(result[key]) is not type(expected) or result[key] != expected:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap receipt result linkage drifted: {key}"
            )
    if result["framing"] != "magic_three_u64_lengths_payloads_v1":
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap receipt framing drifted")
    if not _exact_json_equal(receipt["claims"], _claims()) or not _exact_json_equal(
        receipt["limitations"], _limitations()
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt claims or limitations drifted"
        )
    body = dict(receipt)
    supplied_body = _require_sha256(body.pop("receipt_body_sha256"), "bootstrap receipt body")
    if not hmac.compare_digest(
        supplied_body,
        hashlib.sha256(_canonical_json(body)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt body identity drifted"
        )


def parse_matched_v3_local_bootstrap_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    canonical_child_record_bytes: bytes,
    local_completion_receipt_bytes: bytes,
    reward_trace: bytes,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    """Replay structural content only; this function grants no capability."""

    expected = _require_sha256(expected_receipt_sha256, "expected bootstrap receipt")
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap receipt full-file digest disagrees"
        )
    for byte_value, label in (
        (canonical_child_record_bytes, "child record"),
        (local_completion_receipt_bytes, "local completion receipt"),
        (reward_trace, "reward trace"),
        (stdout, "stdout"),
        (stderr, "stderr"),
    ):
        if type(byte_value) is not bytes:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap receipt {label} must be exact bytes"
            )
    value = _strict_json_load(raw)
    _validate_bootstrap_receipt(
        value,
        child_record=canonical_child_record_bytes,
        local_receipt=local_completion_receipt_bytes,
        reward_trace=reward_trace,
        stdout=stdout,
        stderr=stderr,
    )
    return value


@dataclass(frozen=True, slots=True)
class MatchedV3LocalBootstrapCompletion:
    """Immutable structural result; only a live parent outcome can expose it."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    canonical_receipt_bytes: bytes
    receipt_sha256: str
    canonical_child_record_bytes: bytes
    local_completion_receipt_bytes: bytes
    reward_trace: bytes
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id)
        _require_uint31(self.environment_seed, "bootstrap completion environment seed")
        _require_uint31(self.agent_seed, "bootstrap completion agent seed")
        _require_sha256(self.receipt_sha256, "bootstrap completion receipt")
        receipt = parse_matched_v3_local_bootstrap_receipt(
            self.canonical_receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
            canonical_child_record_bytes=self.canonical_child_record_bytes,
            local_completion_receipt_bytes=self.local_completion_receipt_bytes,
            reward_trace=self.reward_trace,
            stdout=self.stdout,
            stderr=self.stderr,
        )
        if not _exact_json_equal(
            receipt["cell"],
            {
                "candidate_id": self.candidate_id,
                "environment_seed": self.environment_seed,
                "agent_seed": self.agent_seed,
            },
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap completion dataclass identity drifted"
            )

    def receipt(self) -> dict[str, Any]:
        """Return detached nonauthorizing receipt content."""

        return parse_matched_v3_local_bootstrap_receipt(
            self.canonical_receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
            canonical_child_record_bytes=self.canonical_child_record_bytes,
            local_completion_receipt_bytes=self.local_completion_receipt_bytes,
            reward_trace=self.reward_trace,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _bounded_execution_ceiling(value: Any, label: str, hard_maximum: int) -> int:
    exact = _require_positive_ceiling(value, label)
    if exact > hard_maximum:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            f"{label} exceeds the bootstrap hard safety bound"
        )
    return exact


def _cleanup_execution_resources(
    *,
    inherited_read_fds: Sequence[int],
    artifacts: _ScratchArtifacts | None,
    scratch_anchor: _DirectoryAnchor | None,
    runtime_anchors: Sequence[_DirectoryAnchor],
    repository_anchor: _DirectoryAnchor | None,
    executable_anchor: _ExecutableAnchor | None,
) -> BaseException | None:
    errors: list[BaseException] = []
    for descriptor in inherited_read_fds:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if artifacts is not None:
        try:
            artifacts.close_and_remove()
        except BaseException as exc:
            errors.append(exc)
    elif scratch_anchor is not None:
        scratch_anchor.close()
    for anchor in reversed(runtime_anchors):
        anchor.close()
    if repository_anchor is not None:
        repository_anchor.close()
    if executable_anchor is not None:
        executable_anchor.close()
    return errors[0] if errors else None


def execute_matched_v3_local_bootstrap_cell(
    *,
    python_executable: Path,
    repository_root: Path,
    runtime_import_roots: tuple[Path, ...],
    scratch_parent: Path,
    expected_source_snapshot_bytes: bytes,
    expected_source_snapshot_sha256: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    maximum_request_bytes: int,
    maximum_bootstrap_source_bytes: int,
    maximum_result_bytes: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    timeout_seconds: int,
    explicit_execution_opt_in: bool,
    execution_capability: object,
) -> object:
    """Run exactly one cell and return an opaque, second-opt-in outcome handle."""

    if type(explicit_execution_opt_in) is not bool or explicit_execution_opt_in is not True:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap execution requires exact explicit opt-in"
        )
    _require_parent_boundary(require_current_source=False)
    exact_execution_capability = _consume_execution_capability(execution_capability)

    completion: MatchedV3LocalBootstrapCompletion | None = None
    bootstrap_source_sha256 = ""
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    executable_anchor: _ExecutableAnchor | None = None
    repository_anchor: _DirectoryAnchor | None = None
    runtime_anchors: tuple[_DirectoryAnchor, ...] = ()
    scratch_anchor: _DirectoryAnchor | None = None
    artifacts: _ScratchArtifacts | None = None
    inherited_read_fds: tuple[int, ...] = ()
    try:
        injected_source_sha256 = _require_parent_boundary(require_current_source=True)
        request_maximum = _bounded_execution_ceiling(
            maximum_request_bytes,
            "maximum_request_bytes",
            _MAX_JSON_BYTES,
        )
        source_maximum = _bounded_execution_ceiling(
            maximum_bootstrap_source_bytes,
            "maximum_bootstrap_source_bytes",
            _MAX_SOURCE_FILE_BYTES,
        )
        result_maximum = _bounded_execution_ceiling(
            maximum_result_bytes,
            "maximum_result_bytes",
            _MAX_JSON_BYTES,
        )
        stdout_maximum = _bounded_execution_ceiling(
            maximum_stdout_bytes,
            "maximum_stdout_bytes",
            _MAX_JSON_BYTES,
        )
        stderr_maximum = _bounded_execution_ceiling(
            maximum_stderr_bytes,
            "maximum_stderr_bytes",
            _MAX_JSON_BYTES,
        )
        timeout = _require_timeout(timeout_seconds)
        exact_candidate_id = _validate_candidate_id(candidate_id)
        exact_environment_seed = _require_uint31(environment_seed, "environment_seed")
        exact_agent_seed = _require_uint31(agent_seed, "agent_seed")
        manifest_sha256 = _require_sha256(
            expected_source_snapshot_sha256,
            "expected_source_snapshot_sha256",
        )
        if (
            type(expected_source_snapshot_bytes) is not bytes
            or not 1 <= len(expected_source_snapshot_bytes) <= request_maximum
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "expected_source_snapshot_bytes must be bounded exact bytes"
            )
        exact_runtime_roots = _validate_runtime_root_tuple(runtime_import_roots)
        exact_repository_root = _exact_absolute_path(repository_root, "repository_root")
        exact_scratch_parent = _exact_absolute_path(scratch_parent, "scratch_parent")

        executable_anchor = _open_executable_anchor(python_executable)
        repository_anchor = _open_directory_anchor(
            exact_repository_root,
            "repository_root",
        )
        opened_runtime_anchors: list[_DirectoryAnchor] = []
        try:
            for index, path in enumerate(exact_runtime_roots):
                opened_runtime_anchors.append(
                    _open_directory_anchor(path, f"runtime_import_roots[{index}]")
                )
        except BaseException:
            for anchor in reversed(opened_runtime_anchors):
                anchor.close()
            raise
        runtime_anchors = tuple(opened_runtime_anchors)
        scratch_anchor = _open_directory_anchor(exact_scratch_parent, "scratch_parent")
        all_root_anchors = (repository_anchor, *runtime_anchors, scratch_anchor)
        if len({anchor.identity for anchor in all_root_anchors}) != len(all_root_anchors):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "repository, runtime, and scratch roots must not alias"
            )
        all_root_paths = tuple(str(anchor.path) for anchor in all_root_anchors)
        if len({path.casefold() for path in all_root_paths}) != len(all_root_paths):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "repository, runtime, and scratch paths contain a casefold alias"
            )

        manifest_identity = _independent_snapshot_manifest_identity(
            expected_source_snapshot_bytes,
            expected_full_sha256=manifest_sha256,
            maximum_bytes=request_maximum,
        )
        snapshot_source = _read_anchored_relative_file(
            repository_anchor,
            _SNAPSHOT_RELATIVE_PATH,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        runner_source = _read_anchored_relative_file(
            repository_anchor,
            _RUNNER_RELATIVE_PATH,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        if (
            len(snapshot_source) != manifest_identity.snapshot_source_size
            or not hmac.compare_digest(
                hashlib.sha256(snapshot_source).hexdigest(),
                manifest_identity.snapshot_source_sha256,
            )
            or len(runner_source) != manifest_identity.runner_source_size
            or not hmac.compare_digest(
                hashlib.sha256(runner_source).hexdigest(),
                manifest_identity.runner_source_sha256,
            )
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap-critical source bytes disagree with the manifest"
            )
        snapshot_module = _direct_load_snapshot_module(
            snapshot_source,
            f"{repository_anchor.path}/{_SNAPSHOT_RELATIVE_PATH}",
        )
        try:
            parsed_manifest = _parse_snapshot_with_exact_module(
                snapshot_module,
                expected_source_snapshot_bytes,
                manifest_sha256,
            )
            if parsed_manifest["tree"]["sha256"] != manifest_identity.tree_sha256:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "snapshot module and independent tree identities disagree"
                )
            verified_snapshot = _verify_snapshot_with_exact_module(
                snapshot_module,
                repository_root=repository_anchor.path,
                raw=expected_source_snapshot_bytes,
                expected_full_sha256=manifest_sha256,
            )
            if verified_snapshot.tree_sha256 != manifest_identity.tree_sha256:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "parent full source verification tree identity drifted"
                )
        finally:
            sys.modules.pop(_SNAPSHOT_MODULE_NAME, None)

        bootstrap_source = _current_bootstrap_source_bytes(source_maximum)
        bootstrap_source_sha256 = hashlib.sha256(bootstrap_source).hexdigest()
        if not hmac.compare_digest(bootstrap_source_sha256, injected_source_sha256):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap source changed before child construction"
            )
        artifacts = _create_scratch_artifacts(
            scratch_anchor,
            bootstrap_source=bootstrap_source,
            maximum_bootstrap_source_bytes=source_maximum,
        )
        scratch_anchor = None
        child_source_fd = os.open(
            _SCRATCH_SOURCE_NAME,
            _file_flags(),
            dir_fd=artifacts.anchor.descriptor,
        )
        inherited_read_fds = (child_source_fd,)
        child_request_fd = os.open(
            _SCRATCH_REQUEST_NAME,
            _file_flags(),
            dir_fd=artifacts.anchor.descriptor,
        )
        inherited_read_fds = (child_source_fd, child_request_fd)
        if (
            os.fstat(child_source_fd).st_dev,
            os.fstat(child_source_fd).st_ino,
        ) != artifacts.source_identity or (
            os.fstat(child_request_fd).st_dev,
            os.fstat(child_request_fd).st_ino,
        ) != artifacts.request_identity:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap child read-only descriptor identity drifted"
            )
        cache_proc_path = f"/proc/self/fd/{artifacts.cache_fd}"
        argv = (
            str(executable_anchor.path),
            "-I",
            "-S",
            "-B",
            "-c",
            _CHILD_LAUNCHER,
            str(child_source_fd),
            str(child_request_fd),
            str(artifacts.result_fd),
            str(artifacts.cache_fd),
            bootstrap_source_sha256,
            str(source_maximum),
        )
        if (
            len(
                set(
                    (
                        executable_anchor.descriptor,
                        child_source_fd,
                        child_request_fd,
                        artifacts.result_fd,
                        artifacts.cache_fd,
                    )
                )
            )
            != 5
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap inherited descriptors contain an alias"
            )
        process_contract: dict[str, Any] = {
            "argv_sha256": _argv_sha256(argv),
            "argv_count": len(argv),
            "launcher_sha256": _CHILD_LAUNCHER_SHA256,
            "environment_sha256": _canonical_mapping_sha256(_child_environment(cache_proc_path)),
            "cwd_path_sha256": _path_sha256(str(repository_anchor.path)),
            "executable_path_sha256": _path_sha256(str(executable_anchor.path)),
            "executable_source_sha256": executable_anchor.sha256,
            "executable_proc_path_sha256": _path_sha256(executable_anchor.proc_path),
            "executable_fd": executable_anchor.descriptor,
            "source_fd": child_source_fd,
            "request_fd": child_request_fd,
            "result_fd": artifacts.result_fd,
            "cache_fd": artifacts.cache_fd,
            "cache_proc_path": cache_proc_path,
            "bootstrap_source_sha256": bootstrap_source_sha256,
            "jax_platform_selector": "cpu",
            "jax_compilation_cache_enabled": False,
            "home_xdg_cache_proc_path": cache_proc_path,
            "required_flags": ["-I", "-S", "-B"],
            "stdin": "devnull_closed_to_input",
            "start_new_session": True,
            "site_initialization": False,
            "pth_processing": False,
        }
        ceilings = {
            "maximum_request_bytes": request_maximum,
            "maximum_bootstrap_source_bytes": source_maximum,
            "maximum_result_bytes": result_maximum,
            "maximum_stdout_bytes": stdout_maximum,
            "maximum_stderr_bytes": stderr_maximum,
            "timeout_seconds": timeout,
        }
        request, request_raw, request_sha256 = _build_request(
            repository_root=repository_anchor.path,
            repository_identity=repository_anchor.identity,
            runtime_roots=tuple(anchor.path for anchor in runtime_anchors),
            runtime_identities=tuple(anchor.identity for anchor in runtime_anchors),
            manifest_raw=expected_source_snapshot_bytes,
            manifest_identity=manifest_identity,
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            ceilings=ceilings,
            process_contract=process_contract,
        )
        if len(request_raw) > request_maximum:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "canonical bootstrap request exceeds maximum_request_bytes"
            )
        reparsed_request, reparsed_request_sha256 = _parse_request(
            request_raw,
            maximum_bytes=request_maximum,
        )
        if not _exact_json_equal(reparsed_request, request) or not hmac.compare_digest(
            reparsed_request_sha256, request_sha256
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap request self-replay drifted"
            )
        _write_request_file(
            artifacts,
            request_raw,
            maximum_request_bytes=request_maximum,
        )
        process_result = _run_bounded_child(
            argv=argv,
            executable_proc_path=executable_anchor.proc_path,
            cwd=str(repository_anchor.path),
            environment=_child_environment(cache_proc_path),
            pass_fds=(
                executable_anchor.descriptor,
                child_source_fd,
                child_request_fd,
                artifacts.result_fd,
                artifacts.cache_fd,
            ),
            result_fd=artifacts.result_fd,
            maximum_result_bytes=result_maximum,
            maximum_stdout_bytes=stdout_maximum,
            maximum_stderr_bytes=stderr_maximum,
            timeout_seconds=timeout,
        )
        artifacts.verify()
        executable_anchor.verify()
        repository_anchor.verify()
        for anchor in runtime_anchors:
            anchor.verify()
        if not _directory_is_empty(artifacts.cache_fd):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap private bytecode cache is not empty after child exit"
            )
        observed_bootstrap_source, _source_metadata = _read_exact_fd(
            artifacts.source_fd,
            maximum_bytes=source_maximum,
            require_single_link=True,
            require_nonempty=True,
        )
        observed_request, _request_metadata = _read_exact_fd(
            artifacts.request_fd,
            maximum_bytes=request_maximum,
            require_single_link=True,
            require_nonempty=True,
        )
        if not hmac.compare_digest(
            observed_bootstrap_source,
            bootstrap_source,
        ) or not hmac.compare_digest(observed_request, request_raw):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap child mutated its immutable source or request transport"
            )
        post_snapshot_module = _direct_load_snapshot_module(
            snapshot_source,
            f"{repository_anchor.path}/{_SNAPSHOT_RELATIVE_PATH}",
        )
        try:
            parent_post_snapshot = _verify_snapshot_with_exact_module(
                post_snapshot_module,
                repository_root=repository_anchor.path,
                raw=expected_source_snapshot_bytes,
                expected_full_sha256=manifest_sha256,
            )
            if parent_post_snapshot.tree_sha256 != manifest_identity.tree_sha256:
                raise ForagerMatchedV3LocalExecutionBootstrapError(
                    "parent post-child source snapshot tree identity drifted"
                )
        finally:
            sys.modules.pop(_SNAPSHOT_MODULE_NAME, None)
        if process_result.returncode < 0:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap child process died from signal {-process_result.returncode}"
            )
        if process_result.returncode != 0:
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                f"bootstrap child process exited nonzero: {process_result.returncode}"
            )
        frame, _result_metadata = _read_exact_fd(
            artifacts.result_fd,
            maximum_bytes=result_maximum,
            require_single_link=True,
            require_nonempty=True,
        )
        artifacts.verify()
        child_record, local_receipt, reward_trace = _parse_result_frame(
            frame,
            maximum_result_bytes=result_maximum,
        )
        _validate_child_record(
            child_record,
            bootstrap_source_sha256=bootstrap_source_sha256,
            request_sha256=request_sha256,
            manifest_identity=manifest_identity,
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            expected_process_contract=process_contract,
            repository_path=str(repository_anchor.path),
            runtime_paths=tuple(str(anchor.path) for anchor in runtime_anchors),
            local_receipt=local_receipt,
            reward_trace=reward_trace,
        )
        post_runner_source = _read_anchored_relative_file(
            repository_anchor,
            _RUNNER_RELATIVE_PATH,
            maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        )
        if not hmac.compare_digest(post_runner_source, runner_source):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "parent runner parser source changed across child execution"
            )
        runner_parser = _direct_load_runner_parser(
            post_runner_source,
            f"{repository_anchor.path}/{_RUNNER_RELATIVE_PATH}",
        )
        try:
            _replay_local_completion(
                runner_parser,
                local_receipt=local_receipt,
                reward_trace=reward_trace,
                candidate_id=exact_candidate_id,
                environment_seed=exact_environment_seed,
                agent_seed=exact_agent_seed,
            )
        finally:
            sys.modules.pop(_PARENT_RUNNER_PARSER_MODULE_NAME, None)
        if not hmac.compare_digest(
            _current_bootstrap_source_sha256(source_maximum),
            bootstrap_source_sha256,
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap source changed across child execution"
            )
        receipt_raw = _build_bootstrap_receipt(
            bootstrap_source_sha256=bootstrap_source_sha256,
            manifest_identity=manifest_identity,
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            process_contract=process_contract,
            process_result=process_result,
            frame=frame,
            child_record=child_record,
            local_receipt=local_receipt,
            reward_trace=reward_trace,
            ceilings=ceilings,
        )
        receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        completion = MatchedV3LocalBootstrapCompletion(
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            canonical_receipt_bytes=receipt_raw,
            receipt_sha256=receipt_sha256,
            canonical_child_record_bytes=child_record,
            local_completion_receipt_bytes=local_receipt,
            reward_trace=reward_trace,
            stdout=process_result.stdout,
            stderr=process_result.stderr,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_error = _cleanup_execution_resources(
            inherited_read_fds=inherited_read_fds,
            artifacts=artifacts,
            scratch_anchor=scratch_anchor,
            runtime_anchors=runtime_anchors,
            repository_anchor=repository_anchor,
            executable_anchor=executable_anchor,
        )
    if primary_error is not None:
        if cleanup_error is not None:
            primary_error.add_note(
                f"bootstrap cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
    if completion is None:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap completion disappeared before outcome issuance"
        )
    current_source_sha256 = _require_parent_boundary(require_current_source=True)
    if not hmac.compare_digest(current_source_sha256, bootstrap_source_sha256):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap source changed before outcome issuance"
        )
    outcome = _ParentOutcomeCapability()
    with _CAPABILITY_LOCK:
        execution_state = _EXECUTION_CAPABILITIES.get(exact_execution_capability)
        if (
            execution_state is None
            or execution_state.pid != os.getpid()
            or execution_state.status != "consumed"
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "consumed execution capability disappeared before outcome binding"
            )
        _OUTCOME_CAPABILITIES[outcome] = _OutcomeState(
            pid=execution_state.pid,
            status="live",
            execution_capability=exact_execution_capability,
            execution_identity=id(exact_execution_capability),
            bootstrap_source_sha256=bootstrap_source_sha256,
            receipt_sha256=completion.receipt_sha256,
            child_record_sha256=hashlib.sha256(completion.canonical_child_record_bytes).hexdigest(),
            local_receipt_sha256=hashlib.sha256(
                completion.local_completion_receipt_bytes
            ).hexdigest(),
            reward_trace_sha256=hashlib.sha256(completion.reward_trace).hexdigest(),
            completion=completion,
        )
    return outcome


def consume_matched_v3_local_bootstrap_outcome(
    *,
    outcome_capability: object,
    explicit_content_access_opt_in: bool,
) -> MatchedV3LocalBootstrapCompletion:
    """Consume one authentic outcome and expose its nonauthorizing content."""

    if (
        type(explicit_content_access_opt_in) is not bool
        or explicit_content_access_opt_in is not True
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap outcome access requires exact explicit opt-in"
        )
    if type(outcome_capability) is not _ParentOutcomeCapability:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap outcome access requires an authentic opaque capability"
        )
    exact_outcome = outcome_capability
    with _CAPABILITY_LOCK:
        state = _OUTCOME_CAPABILITIES.get(exact_outcome)
        if state is None or state.status != "live":
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap outcome capability is unknown, stale, or already consumed"
            )
        current_pid = os.getpid()
        if state.pid != current_pid:
            state.status = "consumed"
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap outcome capability cannot cross a PID boundary"
            )
        execution_state = _EXECUTION_CAPABILITIES.get(state.execution_capability)
        if (
            execution_state is None
            or execution_state.pid != current_pid
            or execution_state.status != "consumed"
            or id(state.execution_capability) != state.execution_identity
        ):
            state.status = "consumed"
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap outcome lost its consumed execution-capability binding"
            )
        state.status = "consumed"
    current_source_sha256 = _require_parent_boundary(require_current_source=True)
    registered_source_sha256 = _require_sha256(
        state.bootstrap_source_sha256,
        "registered bootstrap outcome source",
    )
    if not hmac.compare_digest(current_source_sha256, registered_source_sha256):
        raise ForagerMatchedV3LocalExecutionBootstrapError("bootstrap outcome source is stale")
    completion = state.completion
    if type(completion) is not MatchedV3LocalBootstrapCompletion:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap outcome structural content is stale"
        )
    identities = (
        (completion.receipt_sha256, state.receipt_sha256, completion.canonical_receipt_bytes),
        (
            hashlib.sha256(completion.canonical_child_record_bytes).hexdigest(),
            state.child_record_sha256,
            completion.canonical_child_record_bytes,
        ),
        (
            hashlib.sha256(completion.local_completion_receipt_bytes).hexdigest(),
            state.local_receipt_sha256,
            completion.local_completion_receipt_bytes,
        ),
        (
            hashlib.sha256(completion.reward_trace).hexdigest(),
            state.reward_trace_sha256,
            completion.reward_trace,
        ),
    )
    for supplied, registered, raw in identities:
        if (
            type(raw) is not bytes
            or not hmac.compare_digest(
                _require_sha256(supplied, "bootstrap outcome content"),
                _require_sha256(registered, "registered bootstrap outcome content"),
            )
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), registered)
        ):
            raise ForagerMatchedV3LocalExecutionBootstrapError(
                "bootstrap outcome content identity is stale"
            )
    receipt = parse_matched_v3_local_bootstrap_receipt(
        completion.canonical_receipt_bytes,
        expected_receipt_sha256=completion.receipt_sha256,
        canonical_child_record_bytes=completion.canonical_child_record_bytes,
        local_completion_receipt_bytes=completion.local_completion_receipt_bytes,
        reward_trace=completion.reward_trace,
        stdout=completion.stdout,
        stderr=completion.stderr,
    )
    if receipt["bootstrap_source_sha256"] != registered_source_sha256:
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap outcome receipt source binding drifted"
        )
    return completion


def matched_v3_local_execution_bootstrap_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing bootstrap descriptor content."""

    return _strict_json_load(_DESCRIPTOR_BYTES, maximum_bytes=_MAX_DESCRIPTOR_BYTES)


def canonical_matched_v3_local_execution_bootstrap_descriptor_bytes() -> bytes:
    """Return the exact canonical bootstrap descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_local_execution_bootstrap_descriptor_sha256() -> str:
    """Return the exact bootstrap descriptor digest."""

    return LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256


def parse_matched_v3_local_execution_bootstrap_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen bootstrap descriptor."""

    value = _strict_json_load(raw, maximum_bytes=_MAX_DESCRIPTOR_BYTES)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3LocalExecutionBootstrapError(
            "bootstrap descriptor differs from its frozen identity"
        )
    return value


__all__ = [
    "ForagerMatchedV3LocalExecutionBootstrapError",
    "LOCAL_EXECUTION_BOOTSTRAP_CHILD_RECORD_SCHEMA_VERSION",
    "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_EXECUTION_BOOTSTRAP_DESCRIPTOR_SHA256",
    "LOCAL_EXECUTION_BOOTSTRAP_ISOLATED_MODULE_NAME",
    "LOCAL_EXECUTION_BOOTSTRAP_RECEIPT_SCHEMA_VERSION",
    "LOCAL_EXECUTION_BOOTSTRAP_STATUS",
    "MatchedV3LocalBootstrapCompletion",
    "PINNED_LOCAL_RUNNER_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_RUNNER_SOURCE_SHA256",
    "PINNED_LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_SOURCE_SNAPSHOT_SOURCE_SHA256",
    "canonical_matched_v3_local_execution_bootstrap_descriptor_bytes",
    "consume_matched_v3_local_bootstrap_outcome",
    "execute_matched_v3_local_bootstrap_cell",
    "issue_matched_v3_local_bootstrap_execution_capability",
    "matched_v3_local_execution_bootstrap_descriptor",
    "matched_v3_local_execution_bootstrap_descriptor_sha256",
    "parse_matched_v3_local_bootstrap_receipt",
    "parse_matched_v3_local_execution_bootstrap_descriptor",
]
