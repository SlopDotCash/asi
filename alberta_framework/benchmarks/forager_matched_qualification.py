"""Reward-blind capability qualification for the matched-current Forager panel.

This module is the provenance bridge between the frozen open-protocol builder
and :mod:`forager_matched_executor`.  It deliberately does **not** run a
benchmark horizon, inspect a reward archive, or authenticate its own output.
It performs three narrower jobs:

* stage content-addressed, deterministic source snapshots and configurations;
* run seed-zero structural probes in the already-qualified, networkless CPU
  image; and
* emit content-only capability receipts plus executor-ready local assets.

The receipts use the existing executor schema, whose ``status == "qualified"``
means that the named bytes passed the structural checks.  Their trust-anchor
identity is explicitly ``content_only_unendorsed_v1``.  No signature, external
endorsement, trust profile, promotion, or performance claim is created here.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import dataclasses
import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import selectors
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Final, Literal, NoReturn, cast

MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_current_qualification.v1"
)
MATCHED_CURRENT_PROBE_SCHEMA_VERSION: Final = "alberta.forager_matched_capability_probe.v1"
MATCHED_CURRENT_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_reviewed_source_snapshot.v1"
)
MATCHED_CURRENT_AUTHORITY_IDENTITY: Final = "content_only_unendorsed_v1"
PUBLIC_QUALIFICATION_SEED: Final = 0
_ISOLATED_AGENT_RNG_NAMESPACE: Final = 0xA63E7C11

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 512 * 1024 * 1024
_MAX_SOURCE_FILES: Final = 20_000
_MAX_SOURCE_DIRECTORIES: Final = 20_000
_MAX_SOURCE_ENTRIES: Final = _MAX_SOURCE_FILES + _MAX_SOURCE_DIRECTORIES
_MAX_QUALIFICATION_FILES: Final = 120_000
_MAX_QUALIFICATION_DIRECTORIES: Final = 120_000
_MAX_QUALIFICATION_ENTRIES: Final = 250_000
_MAX_QUALIFICATION_DEPTH: Final = 256
_MAX_QUALIFICATION_BYTES: Final = 8 * 1024**3
_MAX_QUALIFICATION_ARTIFACT_FILES: Final = 1_024
_MAX_QUALIFICATION_ARTIFACT_DIRECTORIES: Final = 1_024
_MAX_QUALIFICATION_ARTIFACT_ENTRIES: Final = 2_048
_MAX_QUALIFICATION_ARTIFACT_DEPTH: Final = 32
_MAX_QUALIFICATION_ARTIFACT_BYTES: Final = 4 * 1024**3
_MAX_PROBE_OUTPUT_BYTES: Final = 4 * 1024 * 1024
_MAX_GIT_METADATA_BYTES: Final = 4 * 1024
_MAX_CLEANUP_INSPECTION_BYTES: Final = 4 * 1024
_PROBE_TIMEOUT_SECONDS: Final = 10 * 60
_GIT_IDENTITY_TIMEOUT_SECONDS: Final = 60
_GIT_ARCHIVE_TIMEOUT_SECONDS: Final = 180
_PROCESS_REAP_TIMEOUT_SECONDS: Final = 10
_QUALIFIED_IMAGE_SHA256: Final = "5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
_QUALIFIED_RUNTIME_PROFILE_SHA256: Final = (
    "7170418e8082babbf17ebfbbb639ee75fcd8b5ae3931d35b3fb9199ea2bfd9b3"
)
_QUALIFIED_EXECUTOR_RECEIPT_SHA256: Final = (
    "7091147189debe9897d84a6ad55371381bf9a9d92b03ccc66b72e5859c0a4d13"
)
_QUALIFIED_UPSTREAM_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
_QUALIFIED_UPSTREAM_TREE: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
_QUALIFIED_UPSTREAM_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
_QUALIFIED_UPSTREAM_ARCHIVE_SIZE_BYTES: Final = 314_961_920
_QUALIFIED_UPSTREAM_INVENTORY_SHA256: Final = (
    "fcab40b01123250e837d9feb222d1c086303192dd24b806ab8cb8405cd7300d9"
)
_QUALIFIED_RNG_PATCH_SHA256: Final = (
    "46ac3d6c1ae5740bee97fea23abf002ffb161ab4b1b35c041b24b717645e076f"
)
_QUALIFIED_PYTHON: Final = "/opt/alberta-runtime/bin/python"
_CONTAINER_SOURCE_ROOT: Final = "/qualification/source"
_CONTAINER_CONFIG: Final = "/qualification/configuration.json"
_CONTAINER_PROBE: Final = "/qualification/probe.py"
_CONTAINER_BUNDLE_ROOT: Final = "/qualification/bundle"
_CONTAINER_REPLAY_SOURCE_ROOT: Final = (
    f"{_CONTAINER_BUNDLE_ROOT}/sources/alberta/source"
)
_CONTAINER_WORK_CONFIG: Final = "/run/alberta/configuration.json"
_CONTAINER_OUTPUT_BASE: Final = "/run/alberta/output"
_FRESH_SNAPSHOT_REPLAY_SCHEMA: Final = (
    "alberta.forager_matched_fresh_snapshot_replay.v1"
)
_FRESH_SNAPSHOT_REPLAY_SCRIPT: Final = r"""
import hashlib
import importlib
import json
import os
import stat
import sys
from pathlib import Path

MAX_MODULE_BYTES = 512 * 1024 * 1024

def bounded_sha256(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > MAX_MODULE_BYTES
        ):
            raise RuntimeError("qualification module is not a bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("qualification module ended while being read")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("qualification module grew while being read")
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError("qualification module changed while being read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)

source_root = Path(sys.argv[1]).resolve(strict=True)
qualification_root = Path(sys.argv[2]).resolve(strict=True)
expected_module_sha256 = sys.argv[3]
module_candidate = (
    source_root / "alberta_framework" / "benchmarks" / "forager_matched_qualification.py"
)
module_sha256_before = bounded_sha256(module_candidate)
if module_sha256_before != expected_module_sha256:
    raise RuntimeError("qualification module differs from its trusted parent binding")
sys.path.insert(0, source_root.as_posix())
module = importlib.import_module(
    "alberta_framework.benchmarks.forager_matched_qualification"
)
module_path = Path(module.__file__).resolve(strict=True)
relative_module = module_path.relative_to(source_root).as_posix()
module_sha256_after = bounded_sha256(module_path)
if module_sha256_after != module_sha256_before:
    raise RuntimeError("qualification module changed during import")
bundle = module.load_matched_current_qualification_bundle(qualification_root)
protocol, plan = module.build_open_protocol_and_execution_plan(bundle)
for name, loaded in sorted(sys.modules.items()):
    if name != "alberta_framework" and not name.startswith("alberta_framework."):
        continue
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is None:
        raise RuntimeError("loaded Alberta module has no concrete staged origin")
    Path(loaded_file).resolve(strict=True).relative_to(source_root)
payload = {
    "schema_version": "alberta.forager_matched_fresh_snapshot_replay.v1",
    "manifest_sha256": bundle.manifest_sha256,
    "protocol_sha256": protocol.protocol_sha256,
    "plan_sha256": plan.plan_sha256,
    "plan_qualification_manifest_sha256": plan.qualification_manifest_sha256,
    "qualification_module_path": relative_module,
    "qualification_module_sha256": module_sha256_after,
}
sys.stdout.buffer.write(
    json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
)
"""

SourceKey = Literal["alberta", "upstream", "upstream_rng_isolated"]

_RESOURCE_ACCOUNTING_SEMANTICS: Final = MappingProxyType(
    {
        "parameter_count": (
            "unique initialized online trainable scalar parameters; excludes target-network "
            "snapshots, optimizer state, and fixed nontrainable recurrent substrates"
        ),
        "optimizer_update_count": (
            "successful gradient optimizer applications under the exact frozen entrypoint and "
            "horizon; excludes target refreshes, ReDo recycling, and nonparametric updates"
        ),
        "replay_capacity_transitions": "maximum replay capacity measured in transitions",
        "recurrent_state_elements": (
            "candidate-specific disclosed recurrent/carry/nonparametric-state proxy, not total "
            "persistent state: Horde recurrent64 counts only its 64-element hidden carry while "
            "its fixed substrate is supplemental; local RTU counts only actor/critic carry and "
            "excludes sensitivities, eligibility traces, normalization/history, and RNG; causal "
            "maps count the full finite planner/map sufficient-statistic state, including "
            "control and RNG state, despite zero trainable parameters"
        ),
        "causal_map_scope": (
            "causal-map candidates have zero optimizer parameters and zero gradient updates; "
            "their one nonparametric learning update per transition is counted separately"
        ),
        "scope_limitation": (
            "the four ResourceAccounting values are disclosure fields, not total memory or "
            "compute, and candidate resource budgets are not matched"
        ),
    }
)


class ForagerMatchedQualificationError(ValueError):
    """A matched-current qualification input or artifact failed closed."""


class QualificationPublishedButUncertainError(ForagerMatchedQualificationError):
    """The immutable output name is visible but durability or replay is uncertain."""


class _BoundedProcessOutputError(RuntimeError):
    """A child crossed the active stdout or stderr byte limit."""


@dataclass(frozen=True, slots=True)
class QualificationProcessResult:
    """Bounded process result used by the injectable OCI runner."""

    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if (
            type(self.returncode) is not int
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
        ):
            raise TypeError(
                "qualification process result requires int/bytes/bytes fields"
            )


QualificationRunner = Callable[[Sequence[str]], QualificationProcessResult]


@dataclass(frozen=True, slots=True)
class _ProbeRuntimeIdentity:
    """Resolved live OCI executable, daemon version, and exact image identity."""

    executable: Path
    executable_sha256: str
    version: Mapping[str, Any]
    image_inspection: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _GitRuntimeIdentity:
    """Absolute host Git executable identity used only for pinned source staging."""

    executable: Path
    executable_sha256: str


@dataclass(frozen=True, slots=True)
class _FreshReplayClosure:
    """Exact parent/child closure that must survive atomic publication."""

    manifest_sha256: str
    protocol_sha256: str
    plan_sha256: str
    qualification_module_sha256: str


@dataclass(frozen=True, slots=True)
class ProbeInvocation:
    """Host-side, reward-free structural-probe request."""

    candidate_id: str
    source_key: SourceKey
    source_root: Path
    probe_path: Path
    probe_sha256: str
    configuration: Path
    configuration_sha256: str
    entrypoint_path: str
    entrypoint_sha256: str
    entrypoint_family: str
    implementation_kind: str
    invocation_style: str
    result_root: str
    seed_transport: str
    expected_agent: str
    horizon: int

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ForagerMatchedQualificationError("candidate_id must be a non-empty string")
        if self.source_key not in ("alberta", "upstream", "upstream_rng_isolated"):
            raise ForagerMatchedQualificationError("source_key is invalid")
        if not isinstance(self.source_root, Path):
            raise ForagerMatchedQualificationError("source_root must be a Path")
        if not isinstance(self.probe_path, Path):
            raise ForagerMatchedQualificationError("probe_path must be a Path")
        if type(self.probe_sha256) is not str or len(self.probe_sha256) != 64:
            raise ForagerMatchedQualificationError(
                "probe_sha256 must be a 64-character hex string"
            )
        if not isinstance(self.configuration, Path):
            raise ForagerMatchedQualificationError("configuration must be a Path")
        if type(self.configuration_sha256) is not str or len(self.configuration_sha256) != 64:
            raise ForagerMatchedQualificationError(
                "configuration_sha256 must be a 64-character hex string"
            )
        for name in (
            "entrypoint_path",
            "entrypoint_family",
            "implementation_kind",
            "invocation_style",
            "result_root",
            "seed_transport",
            "expected_agent",
        ):
            val = getattr(self, name)
            if type(val) is not str or not val:
                raise ForagerMatchedQualificationError(f"{name} must be a non-empty string")
        if type(self.entrypoint_sha256) is not str or len(self.entrypoint_sha256) != 64:
            raise ForagerMatchedQualificationError(
                "entrypoint_sha256 must be a 64-character hex string"
            )
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ForagerMatchedQualificationError("horizon must be a positive int")


@dataclass(frozen=True, slots=True)
class _StagedSource:
    key: SourceKey
    root: Path
    archive: Path
    inventory_path: Path
    inventory: Mapping[str, Any]
    binding: Any
    descriptor_path: Path | None
    patch_path: Path | None


@dataclass(frozen=True, slots=True)
class _MaterializedConfiguration:
    candidate_id: str
    original: Path
    derived: Path
    binding: Any


@dataclass(frozen=True, slots=True)
class _StagedExecutorQualifications:
    cpu_root: Path
    rng_root: Path
    cpu_inventory: Mapping[str, Any]
    rng_inventory: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _TreeWalkLimits:
    files: int
    directories: int
    entries: int
    depth: int
    bytes: int


@dataclass(slots=True)
class _TreeWalkState:
    files: int = 0
    directories: int = 0
    entries: int = 0
    bytes: int = 0


@dataclass(frozen=True, slots=True)
class MatchedCurrentQualificationBundle:
    """Verified values accepted by both matched-current builders.

    The mapping insertion order is the frozen matched-current candidate order.  ``Any`` is
    intentional in the annotations: imports of the project-side dataclasses
    stay lazy so this same file can execute as an isolated OCI probe.
    """

    output_root: Path
    cpu_qualification_root: Path
    rng_parity_qualification_root: Path
    runtime_qualification: Any
    candidate_qualifications: Mapping[str, Any]
    candidate_assets: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_qualifications, Mapping) or not isinstance(
            self.candidate_assets,
            Mapping,
        ):
            raise ForagerMatchedQualificationError(
                "qualification bundle candidate mappings are invalid"
            )
        if not isinstance(self.manifest, Mapping):
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest must be a mapping"
            )
        if (
            type(self.manifest_bytes) is not bytes
            or not self.manifest_bytes
            or len(self.manifest_bytes) > _MAX_JSON_BYTES
        ):
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest_bytes must be bounded exact bytes"
            )
        if type(self.manifest_sha256) is not str or _SHA256.fullmatch(
            self.manifest_sha256
        ) is None:
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest_sha256 is not lowercase SHA-256"
            )
        try:
            plain_manifest = _plain_json(
                self.manifest,
                "qualification bundle manifest",
            )
            canonical = _canonical_json_bytes(plain_manifest)
            frozen_manifest = _freeze_json(plain_manifest)
        except ForagerMatchedQualificationError:
            raise
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest is not bounded canonical JSON"
            ) from exc
        if not isinstance(plain_manifest, dict):
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest must be a JSON object"
            )
        if canonical != self.manifest_bytes:
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest bytes differ from its exact canonical content"
            )
        if hashlib.sha256(self.manifest_bytes).hexdigest() != self.manifest_sha256:
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest digest differs from its exact bytes"
            )
        if (
            plain_manifest.get("schema_version")
            != MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION
        ):
            raise ForagerMatchedQualificationError(
                "qualification bundle manifest schema is unsupported"
            )
        object.__setattr__(
            self,
            "manifest",
            cast(Mapping[str, Any], frozen_manifest),
        )
        object.__setattr__(
            self,
            "candidate_qualifications",
            MappingProxyType(dict(self.candidate_qualifications)),
        )
        object.__setattr__(
            self,
            "candidate_assets",
            MappingProxyType(dict(self.candidate_assets)),
        )


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _freeze_json(item)
                for key, item in cast(dict[str, Any], value).items()
            }
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedQualificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedQualificationError(f"non-finite JSON number {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ForagerMatchedQualificationError(f"non-finite JSON number {value!r}")
    return parsed


def _decode_json(raw: bytes, label: str) -> Any:
    if not raw or len(raw) > _MAX_JSON_BYTES or raw.startswith(b"\xef\xbb\xbf"):
        raise ForagerMatchedQualificationError(f"{label} violates the JSON byte contract")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_json_float,
        )
    except ForagerMatchedQualificationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ForagerMatchedQualificationError(f"{label} is not strict UTF-8 JSON") from exc


def _plain_json(value: Any, path: str = "value") -> Any:
    if type(value) is float:
        if not math.isfinite(value):
            raise ForagerMatchedQualificationError(
                f"{path} contains a non-finite JSON number"
            )
        return value
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) in (dict, MappingProxyType):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ForagerMatchedQualificationError(f"{path} has a non-string key")
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    if type(value) in (list, tuple):
        return [_plain_json(item, f"{path}[]") for item in value]
    raise ForagerMatchedQualificationError(f"{path} contains unsupported {type(value).__name__}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path, *, maximum: int = _MAX_SOURCE_BYTES) -> tuple[str, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedQualificationError(f"cannot safely open {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise ForagerMatchedQualificationError(f"{path} is not a bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedQualificationError(f"{path} ended while being read")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ForagerMatchedQualificationError(f"{path} changed while being read")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _read_stable(path: Path, label: str, *, maximum: int = _MAX_JSON_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ForagerMatchedQualificationError(f"{label} is not a singly linked regular file")
        if before.st_size > maximum:
            raise ForagerMatchedQualificationError(f"{label} exceeds its size limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedQualificationError(f"{label} ended while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedQualificationError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ForagerMatchedQualificationError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_canonical(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _safe_relative(value: str, label: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ForagerMatchedQualificationError(f"{label} is not a safe relative path")
    result = PurePosixPath(value)
    if result.is_absolute() or any(part in {"", ".", ".."} for part in result.parts):
        raise ForagerMatchedQualificationError(f"{label} is not a safe relative path")
    return result


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ForagerMatchedQualificationError(f"{label} is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ForagerMatchedQualificationError(f"{label} must be a non-symlink directory")
    return path.resolve()


def _bind_project_root(path: Path) -> Path:
    """Require staging from the project tree that loaded this verifier."""
    requested = _regular_directory(path, "project root")
    loaded = _regular_directory(
        Path(__file__).resolve().parents[2],
        "loaded Alberta project root",
    )
    if requested != loaded:
        raise ForagerMatchedQualificationError(
            "project root differs from the loaded Alberta project root"
        )
    framework_root = _regular_directory(
        requested / "alberta_framework",
        "loaded Alberta source tree",
    )
    try:
        Path(__file__).resolve(strict=True).relative_to(framework_root)
    except (OSError, ValueError) as exc:
        raise ForagerMatchedQualificationError(
            "qualification verifier is outside the loaded Alberta project root"
        ) from exc
    return requested


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


def _bounded_tree_walk(
    root: Path,
    *,
    label: str,
    limits: _TreeWalkLimits,
    skip_descendants: frozenset[Path] = frozenset(),
    fsync: bool = False,
    normalize_permissions: bool = False,
    require_normalized_permissions: bool = False,
) -> tuple[set[Path], set[Path]]:
    """Walk one tree with descriptor-relative race checks and hard global bounds.

    TOCTOU model: every entry is opened ``O_NOFOLLOW`` relative to its parent
    directory descriptor (``dir_fd``), and its stat identity (dev, inode, mode,
    nlink, size, mtime, ctime) must agree at three points — the pre-open
    ``stat``, the post-open ``fstat``, and a re-``stat`` through the parent
    after the entry (or its whole subtree) has been processed.  Any mismatch
    means the tree mutated mid-walk, and the walk fails closed rather than
    report a snapshot that never existed.  Only regular files with
    ``st_nlink == 1`` and non-symlink directories are admitted, so hardlink
    aliases and special files cannot smuggle content past the digests.
    ``limits`` bounds files, directories, total entries, depth, and bytes
    globally, capping what a hostile tree can make the host do.  With
    ``normalize_permissions=True``, chmod happens on the already-opened
    descriptor while its parent descriptor remains held; no absolute child
    path is reopened.  ``require_normalized_permissions`` verifies those
    modes through the same descriptor discipline.  With ``fsync=True`` every
    admitted file and directory is also made durable — that is how
    :func:`_durably_sync_verified_tree` reuses this walk before publication.
    """
    if any(
        type(value) is not int or value < 1
        for value in (
            limits.files,
            limits.directories,
            limits.entries,
            limits.depth,
            limits.bytes,
        )
    ):
        raise AssertionError("tree-walk limits must be positive integers")
    if normalize_permissions and require_normalized_permissions:
        raise AssertionError(
            "permission normalization and verification must be separate walks"
        )
    canonical_root = _regular_directory(root, label)
    canonical_skips: set[Path] = set()
    for path in skip_descendants:
        canonical = _regular_directory(path, f"{label} skipped subtree")
        try:
            canonical.relative_to(canonical_root)
        except ValueError as exc:
            raise ForagerMatchedQualificationError(
                f"{label} skip root escapes the walked tree"
            ) from exc
        if canonical == canonical_root:
            raise ForagerMatchedQualificationError(f"{label} cannot skip its own root")
        canonical_skips.add(canonical)

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        root_path_metadata = os.lstat(canonical_root)
        root_descriptor = os.open(canonical_root, directory_flags)
    except OSError as exc:
        raise ForagerMatchedQualificationError(f"cannot safely open {label}") from exc
    try:
        root_metadata = os.fstat(root_descriptor)
    except OSError as exc:
        os.close(root_descriptor)
        raise ForagerMatchedQualificationError(
            f"cannot inspect the opened root for {label}"
        ) from exc
    if (
        not stat.S_ISDIR(root_path_metadata.st_mode)
        or _stat_identity(root_path_metadata) != _stat_identity(root_metadata)
    ):
        os.close(root_descriptor)
        raise ForagerMatchedQualificationError(f"{label} changed before traversal")

    def permission_stable_fields(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )

    def normalize_descriptor(
        descriptor: int,
        before: os.stat_result,
        *,
        directory: bool,
    ) -> os.stat_result:
        if directory:
            if not stat.S_ISDIR(before.st_mode):
                raise ForagerMatchedQualificationError(
                    "qualification permission entry changed type"
                )
            target_mode = 0o755
        else:
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ForagerMatchedQualificationError(
                    "qualification permission entry changed type"
                )
            target_mode = 0o755 if stat.S_IMODE(before.st_mode) & 0o111 else 0o644
        try:
            os.fchmod(descriptor, target_mode)
            after = os.fstat(descriptor)
        except OSError as exc:
            raise ForagerMatchedQualificationError(
                "cannot safely normalize qualification permissions"
            ) from exc
        if (
            permission_stable_fields(before) != permission_stable_fields(after)
            or stat.S_IMODE(after.st_mode) != target_mode
        ):
            raise ForagerMatchedQualificationError(
                "qualification permission entry changed during normalization"
            )
        return after

    state = _TreeWalkState()
    files: set[Path] = set()
    directories: set[Path] = set()

    def enumerate_names(directory_descriptor: int) -> list[str]:
        names: list[str] = []
        try:
            iterator = os.scandir(directory_descriptor)
        except OSError as exc:
            raise ForagerMatchedQualificationError(f"cannot enumerate {label}") from exc
        with iterator:
            for entry in iterator:
                state.entries += 1
                if state.entries > limits.entries:
                    raise ForagerMatchedQualificationError(
                        f"{label} exceeds its global entry bound"
                    )
                name = entry.name
                if (
                    type(name) is not str
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                ):
                    raise ForagerMatchedQualificationError(f"{label} contains an unsafe name")
                names.append(name)
        try:
            names.sort(key=lambda value: value.encode("utf-8"))
        except UnicodeError as exc:
            raise ForagerMatchedQualificationError(
                f"{label} contains a non-UTF-8 name"
            ) from exc
        return names

    def check_opened_entry(
        parent_descriptor: int,
        name: str,
        expected: os.stat_result,
        descriptor: int,
        entry_label: str,
    ) -> None:
        opened = os.fstat(descriptor)
        try:
            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedQualificationError(f"cannot recheck {entry_label}") from exc
        if (
            _stat_identity(expected) != _stat_identity(opened)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedQualificationError(f"{entry_label} changed during traversal")

    def walk(
        directory_descriptor: int,
        current_path: Path,
        depth: int,
        opened_metadata: os.stat_result,
    ) -> None:
        names = enumerate_names(directory_descriptor)
        for name in names:
            child_depth = depth + 1
            if child_depth > limits.depth:
                raise ForagerMatchedQualificationError(
                    f"{label} exceeds its global depth bound"
                )
            child_path = current_path / name
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ForagerMatchedQualificationError(
                    f"cannot inspect an entry in {label}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                state.directories += 1
                if state.directories > limits.directories:
                    raise ForagerMatchedQualificationError(
                        f"{label} exceeds its global directory bound"
                    )
                directories.add(child_path)
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    raise ForagerMatchedQualificationError(
                        f"cannot safely open a directory in {label}"
                    ) from exc
                try:
                    child_opened = os.fstat(child_descriptor)
                    if _stat_identity(metadata) != _stat_identity(child_opened):
                        raise ForagerMatchedQualificationError(
                            f"a directory in {label} changed before traversal"
                        )
                    if normalize_permissions:
                        child_opened = normalize_descriptor(
                            child_descriptor,
                            child_opened,
                            directory=True,
                        )
                    elif (
                        require_normalized_permissions
                        and stat.S_IMODE(child_opened.st_mode) != 0o755
                    ):
                        raise ForagerMatchedQualificationError(
                            "qualification directory is not traversable by the fixed OCI user"
                        )
                    if child_path not in canonical_skips:
                        walk(child_descriptor, child_path, child_depth, child_opened)
                    check_opened_entry(
                        directory_descriptor,
                        name,
                        child_opened,
                        child_descriptor,
                        f"a directory in {label}",
                    )
                except OSError as exc:
                    raise ForagerMatchedQualificationError(
                        f"cannot durably sync a directory in {label}"
                    ) from exc
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ForagerMatchedQualificationError(
                    f"{label} contains a link or special file"
                )
            state.files += 1
            state.bytes += metadata.st_size
            if state.files > limits.files:
                raise ForagerMatchedQualificationError(
                    f"{label} exceeds its global file bound"
                )
            if metadata.st_size < 0 or state.bytes > limits.bytes:
                raise ForagerMatchedQualificationError(
                    f"{label} exceeds its global byte bound"
                )
            files.add(child_path)
            try:
                file_descriptor = os.open(
                    name,
                    file_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ForagerMatchedQualificationError(
                    f"cannot safely open a file in {label}"
                ) from exc
            try:
                opened = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or _stat_identity(metadata) != _stat_identity(opened)
                ):
                    raise ForagerMatchedQualificationError(
                        f"a file in {label} changed before traversal"
                    )
                if normalize_permissions:
                    opened = normalize_descriptor(
                        file_descriptor,
                        opened,
                        directory=False,
                    )
                elif (
                    require_normalized_permissions
                    and stat.S_IMODE(opened.st_mode) not in {0o644, 0o755}
                ):
                    raise ForagerMatchedQualificationError(
                        "qualification file is not readable by the fixed OCI user"
                    )
                if fsync:
                    os.fsync(file_descriptor)
                check_opened_entry(
                    directory_descriptor,
                    name,
                    opened,
                    file_descriptor,
                    f"a file in {label}",
                )
            except OSError as exc:
                raise ForagerMatchedQualificationError(
                    f"cannot durably sync a file in {label}"
                ) from exc
            finally:
                os.close(file_descriptor)
        if fsync:
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                raise ForagerMatchedQualificationError(
                    f"cannot durably sync a directory in {label}"
                ) from exc
        if _stat_identity(opened_metadata) != _stat_identity(os.fstat(directory_descriptor)):
            raise ForagerMatchedQualificationError(
                f"a directory in {label} changed during traversal"
            )

    try:
        if normalize_permissions:
            root_metadata = normalize_descriptor(
                root_descriptor,
                root_metadata,
                directory=True,
            )
            root_current = os.lstat(canonical_root)
            if _stat_identity(root_metadata) != _stat_identity(root_current):
                raise ForagerMatchedQualificationError(
                    f"{label} changed during root permission normalization"
                )
        elif (
            require_normalized_permissions
            and stat.S_IMODE(root_metadata.st_mode) != 0o755
        ):
            raise ForagerMatchedQualificationError(
                "qualification directory is not traversable by the fixed OCI user"
            )
        walk(root_descriptor, canonical_root, 0, root_metadata)
        root_after = os.fstat(root_descriptor)
        root_current = os.lstat(canonical_root)
        if (
            _stat_identity(root_metadata) != _stat_identity(root_after)
            or _stat_identity(root_after) != _stat_identity(root_current)
        ):
            raise ForagerMatchedQualificationError(f"{label} changed during traversal")
    finally:
        os.close(root_descriptor)
    return files, directories


def _normalize_qualification_tree_permissions(root: Path) -> None:
    """Make the tree OCI-readable without reopening absolute child paths."""
    limits = _TreeWalkLimits(
        files=_MAX_QUALIFICATION_FILES,
        directories=_MAX_QUALIFICATION_DIRECTORIES,
        entries=_MAX_QUALIFICATION_ENTRIES,
        depth=_MAX_QUALIFICATION_DEPTH,
        bytes=_MAX_QUALIFICATION_BYTES,
    )
    canonical_root = _regular_directory(root, "qualification permission tree")
    files, directories = _bounded_tree_walk(
        canonical_root,
        label="qualification permission tree",
        limits=limits,
        normalize_permissions=True,
    )
    verified_files, verified_directories = _bounded_tree_walk(
        canonical_root,
        label="OCI-readable qualification tree",
        limits=limits,
        require_normalized_permissions=True,
    )
    if verified_files != files or verified_directories != directories:
        raise ForagerMatchedQualificationError(
            "qualification tree changed during permission normalization"
        )


def _durably_sync_verified_tree(root: Path) -> None:
    """Make every verified staged file and directory durable before publication."""
    _bounded_tree_walk(
        root,
        label="verified qualification staging tree",
        limits=_TreeWalkLimits(
            files=_MAX_QUALIFICATION_FILES,
            directories=_MAX_QUALIFICATION_DIRECTORIES,
            entries=_MAX_QUALIFICATION_ENTRIES,
            depth=_MAX_QUALIFICATION_DEPTH,
            bytes=_MAX_QUALIFICATION_BYTES,
        ),
        fsync=True,
    )


def _verify_tree_entry_bounds(path: Path, label: str) -> tuple[int, int]:
    """Stream a tree inventory far enough to bound directory/inode exhaustion."""
    root = _regular_directory(path, label)
    pending = [root]
    file_count = 0
    directory_count = 0
    entry_count = 0
    while pending:
        current = pending.pop()
        try:
            iterator = os.scandir(current)
        except OSError as exc:
            raise ForagerMatchedQualificationError(f"cannot enumerate {label}") from exc
        with iterator:
            for entry in iterator:
                entry_count += 1
                if entry_count > _MAX_SOURCE_ENTRIES:
                    raise ForagerMatchedQualificationError(
                        f"{label} exceeds its total-entry bound"
                    )
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ForagerMatchedQualificationError(
                        f"cannot inspect an entry in {label}"
                    ) from exc
                if stat.S_ISDIR(metadata.st_mode):
                    directory_count += 1
                    if directory_count > _MAX_SOURCE_DIRECTORIES:
                        raise ForagerMatchedQualificationError(
                            f"{label} exceeds its directory bound"
                        )
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(metadata.st_mode):
                    file_count += 1
                    if file_count > _MAX_SOURCE_FILES:
                        raise ForagerMatchedQualificationError(
                            f"{label} exceeds its file bound"
                        )
                else:
                    raise ForagerMatchedQualificationError(
                        f"{label} contains a link or special file"
                    )
    return file_count, directory_count


def _copy_tree(source: Path, destination: Path, *, alberta_filter: bool) -> None:
    source = _regular_directory(source, "source tree")
    _verify_tree_entry_bounds(source, "source tree")
    destination.mkdir(parents=True)
    count = 0
    total = 0
    inode_ids: set[tuple[int, int]] = set()
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(source)
        directory_names[:] = sorted(
            name for name in directory_names if not (alberta_filter and name == "__pycache__")
        )
        for name in directory_names:
            child = current_path / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or child.is_symlink():
                raise ForagerMatchedQualificationError("source tree contains a linked directory")
            target = destination / relative_current / name
            target.mkdir(mode=0o755, parents=True, exist_ok=False)
        for name in sorted(file_names):
            if alberta_filter and (name.endswith(".pyc") or name.endswith(".pyo")):
                continue
            child = current_path / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ForagerMatchedQualificationError(
                    "source tree contains a link or special file"
                )
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in inode_ids:
                raise ForagerMatchedQualificationError("source tree contains an inode alias")
            inode_ids.add(inode)
            count += 1
            total += metadata.st_size
            if count > _MAX_SOURCE_FILES or total > _MAX_SOURCE_BYTES:
                raise ForagerMatchedQualificationError("source tree exceeds its bound")
            raw = _read_stable(child, "source file", maximum=_MAX_SOURCE_BYTES)
            target = destination / relative_current / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
            target.chmod(0o755 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o644)


def _alberta_snapshot_fingerprint(
    root: Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, bool, int, str], ...]]:
    """Fingerprint exactly the live Alberta entries admitted by ``_copy_tree``."""
    from alberta_framework.benchmarks import forager_matched_executor as executor

    try:
        directories, files = executor._source_tree_snapshot(  # noqa: SLF001
            _regular_directory(root, "Alberta source tree")
        )
    except ValueError as exc:
        raise ForagerMatchedQualificationError(
            "cannot capture a stable Alberta source snapshot"
        ) from exc

    def excluded(path: str) -> bool:
        parts = PurePosixPath(path).parts
        return "__pycache__" in parts or path.endswith((".pyc", ".pyo"))

    retained_directories = tuple(path for path in directories if not excluded(path))
    retained_files = tuple(
        (
            path,
            bool(mode & 0o111),
            len(raw),
            hashlib.sha256(raw).hexdigest(),
        )
        for path, mode, raw in files
        if not excluded(path)
    )
    if not retained_files:
        raise ForagerMatchedQualificationError(
            "filtered Alberta source snapshot must contain regular files"
        )
    return retained_directories, retained_files


def _copy_alberta_tree_stably(source: Path, destination: Path) -> None:
    """Copy one filtered Alberta tree and reject persistent concurrent drift."""
    before = _alberta_snapshot_fingerprint(source)
    _copy_tree(source, destination, alberta_filter=True)
    after = _alberta_snapshot_fingerprint(source)
    if after != before:
        raise ForagerMatchedQualificationError(
            "live Alberta source changed during snapshot staging"
        )
    copied = _alberta_snapshot_fingerprint(destination)
    if copied != before:
        raise ForagerMatchedQualificationError(
            "staged Alberta source differs from its stable live snapshot"
        )


def _archive_tree(source_root: Path, archive_path: Path) -> None:
    """Write one normalized deterministic USTAR archive."""
    source_root = _regular_directory(source_root, "archive source root")
    _verify_tree_entry_bounds(source_root, "archive source root")
    paths = sorted(
        source_root.rglob("*"),
        key=lambda item: (
            item.relative_to(source_root).as_posix() + ("/" if item.is_dir() else "")
        ).encode("utf-8"),
    )
    if len(paths) > _MAX_SOURCE_ENTRIES:
        raise ForagerMatchedQualificationError("archive source exceeds its total-entry bound")
    with archive_path.open("xb") as raw_handle:
        with tarfile.open(fileobj=raw_handle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path in paths:
                relative = path.relative_to(source_root).as_posix()
                metadata = path.lstat()
                info = tarfile.TarInfo(relative + ("/" if stat.S_ISDIR(metadata.st_mode) else ""))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if stat.S_ISDIR(metadata.st_mode):
                    if path.is_symlink():
                        raise ForagerMatchedQualificationError("archive source contains a symlink")
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.size = 0
                    archive.addfile(info)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    raw = _read_stable(path, "archive source file", maximum=_MAX_SOURCE_BYTES)
                    if len(raw) != metadata.st_size:
                        raise ForagerMatchedQualificationError(
                            "archive source file changed before it was archived"
                        )
                    info.type = tarfile.REGTYPE
                    info.mode = 0o755 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o644
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
                else:
                    raise ForagerMatchedQualificationError(
                        "archive source contains a link or special file"
                    )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _extract_git_archive(archive_path: Path, destination: Path) -> None:
    """Extract only bounded regular files/directories from the exact git archive."""
    destination.mkdir(parents=True)
    names: set[str] = set()
    file_count = 0
    directory_count = 0
    entry_count = 0
    total = 0
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            entry_count += 1
            if entry_count > _MAX_SOURCE_ENTRIES:
                raise ForagerMatchedQualificationError(
                    "git archive exceeds its total-member bound"
                )
            name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
            relative = _safe_relative(name, "git archive member")
            normalized = relative.as_posix()
            if normalized in names:
                raise ForagerMatchedQualificationError("git archive contains a duplicate member")
            names.add(normalized)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                directory_count += 1
                if directory_count > _MAX_SOURCE_DIRECTORIES:
                    raise ForagerMatchedQualificationError(
                        "git archive exceeds its directory bound"
                    )
                target.mkdir(mode=0o755, parents=True, exist_ok=False)
                continue
            if not member.isreg() or member.size < 0:
                raise ForagerMatchedQualificationError("git archive contains a non-regular member")
            file_count += 1
            total += member.size
            if file_count > _MAX_SOURCE_FILES or total > _MAX_SOURCE_BYTES:
                raise ForagerMatchedQualificationError("git archive exceeds its bound")
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ForagerMatchedQualificationError("git archive file has no payload")
            with target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ForagerMatchedQualificationError("git archive member ended early")
                    output.write(chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    raise ForagerMatchedQualificationError("git archive member exceeds its size")
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _git_environment() -> dict[str, str]:
    """Return the minimal host environment admitted for pinned Git reads."""
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TZ": "UTC",
        "XDG_CONFIG_HOME": "/nonexistent",
    }


def _bind_git_runtime(requested: str | Path = "git") -> _GitRuntimeIdentity:
    """Resolve and hash the one host Git executable admitted during staging."""
    if isinstance(requested, Path):
        requested_text = requested.as_posix()
    elif type(requested) is str and requested:
        requested_text = requested
    else:
        raise TypeError("Git runtime must be a non-empty str or Path")
    resolved_text: str | None
    if "/" in requested_text:
        resolved_text = requested_text
    else:
        resolved_text = shutil.which(requested_text, path=os.defpath)
    if resolved_text is None:
        raise ForagerMatchedQualificationError("cannot resolve the host Git executable")
    try:
        executable = Path(resolved_text).resolve(strict=True)
        metadata = executable.stat()
    except OSError as exc:
        raise ForagerMatchedQualificationError(
            "cannot inspect the host Git executable"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(executable, os.X_OK):
        raise ForagerMatchedQualificationError(
            "host Git executable is not an executable regular file"
        )
    executable_sha256, _size = _sha256_file(
        executable,
        maximum=_MAX_SOURCE_BYTES,
    )
    return _GitRuntimeIdentity(
        executable=executable,
        executable_sha256=executable_sha256,
    )


def _rebind_git_runtime(identity: _GitRuntimeIdentity) -> None:
    """Fail if the resolved host Git executable changes around a source read."""
    if type(identity) is not _GitRuntimeIdentity:
        raise TypeError("Git runtime identity must be a _GitRuntimeIdentity")
    try:
        executable_sha256, _size = _sha256_file(
            identity.executable,
            maximum=_MAX_SOURCE_BYTES,
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedQualificationError(
            "host Git executable changed after binding"
        ) from exc
    if (
        executable_sha256 != identity.executable_sha256
        or not os.access(identity.executable, os.X_OK)
    ):
        raise ForagerMatchedQualificationError(
            "host Git executable changed after binding"
        )


def _git_command(
    identity: _GitRuntimeIdentity,
    checkout: Path,
    *arguments: str,
) -> tuple[str, ...]:
    return (
        identity.executable.as_posix(),
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.pager=cat",
        "-c",
        "tar.umask=0002",
        "-C",
        os.fspath(checkout),
        *arguments,
    )


def _git_value(
    identity: _GitRuntimeIdentity,
    checkout: Path,
    *arguments: str,
) -> str:
    _rebind_git_runtime(identity)
    try:
        try:
            completed = _run_bounded_process(
                _git_command(identity, checkout, *arguments),
                timeout=_GIT_IDENTITY_TIMEOUT_SECONDS,
                maximum_stdout_bytes=_MAX_GIT_METADATA_BYTES,
                maximum_stderr_bytes=_MAX_GIT_METADATA_BYTES,
                environment=_git_environment(),
            )
        except _BoundedProcessOutputError as exc:
            raise ForagerMatchedQualificationError(
                "git identity query output exceeds its bound"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ForagerMatchedQualificationError(
                "git identity query could not run"
            ) from exc
    finally:
        _rebind_git_runtime(identity)
    if completed.returncode != 0 or completed.stderr:
        raise ForagerMatchedQualificationError("git identity query failed")
    try:
        raw = completed.stdout
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise ForagerMatchedQualificationError(
                "git identity query did not return one canonical line"
            )
        value = raw[:-1].decode("ascii")
    except UnicodeError as exc:
        raise ForagerMatchedQualificationError("git identity is not ASCII") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ForagerMatchedQualificationError("git identity is not a canonical SHA-1")
    return value


def _build_exact_git_archive(checkout: Path, output: Path) -> None:
    checkout = _regular_directory(checkout, "upstream checkout")
    git_runtime = _bind_git_runtime()
    commit = _git_value(git_runtime, checkout, "rev-parse", "HEAD")
    tree = _git_value(git_runtime, checkout, "rev-parse", "HEAD^{tree}")
    if commit != _QUALIFIED_UPSTREAM_COMMIT or tree != _QUALIFIED_UPSTREAM_TREE:
        raise ForagerMatchedQualificationError("upstream checkout is not the frozen commit/tree")
    with output.open("xb") as handle:
        _rebind_git_runtime(git_runtime)
        try:
            try:
                completed = _run_bounded_process(
                    _git_command(
                        git_runtime,
                        checkout,
                        "archive",
                        "--format=tar",
                        commit,
                    ),
                    timeout=_GIT_ARCHIVE_TIMEOUT_SECONDS,
                    maximum_stdout_bytes=_QUALIFIED_UPSTREAM_ARCHIVE_SIZE_BYTES,
                    maximum_stderr_bytes=_MAX_GIT_METADATA_BYTES,
                    stdout_sink=handle,
                    environment=_git_environment(),
                )
            except _BoundedProcessOutputError as exc:
                raise ForagerMatchedQualificationError(
                    "git archive output exceeds its bound"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise ForagerMatchedQualificationError(
                    "git archive generation could not run"
                ) from exc
        finally:
            _rebind_git_runtime(git_runtime)
        handle.flush()
        os.fsync(handle.fileno())
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise ForagerMatchedQualificationError("git archive generation failed")
    digest, size = _sha256_file(
        output,
        maximum=_QUALIFIED_UPSTREAM_ARCHIVE_SIZE_BYTES,
    )
    if (
        size != _QUALIFIED_UPSTREAM_ARCHIVE_SIZE_BYTES
        or digest != _QUALIFIED_UPSTREAM_ARCHIVE_SHA256
    ):
        raise ForagerMatchedQualificationError("generated upstream archive digest drifted")


def _stage_sources(
    project_root: Path,
    upstream_checkout: Path,
    root: Path,
) -> dict[SourceKey, _StagedSource]:
    """Stage the three content-addressed source trees every probe and executor run binds to.

    * ``upstream`` — a ``git archive`` of the frozen upstream commit/tree
      (identities pinned in ``_QUALIFIED_UPSTREAM_*``), extracted through the
      bounded extractor and bound as ``provenance_kind="git_tree"``: its
      identity is the git tree itself, cross-checked against the pinned
      archive and inventory digests.
    * ``upstream_rng_isolated`` — the same archive re-extracted, with
      ``src/rtu_ppo.py`` replaced by the deterministic RNG-isolation
      derivation (patch digest pinned in ``_QUALIFIED_RNG_PATCH_SHA256``).
      The result matches no git tree, so it is re-archived as a normalized
      USTAR snapshot and bound as ``provenance_kind="reviewed_snapshot"``
      with a full snapshot descriptor recording the derivation.
    * ``alberta`` — the live ``alberta_framework/`` working tree, copied with
      ``__pycache__``/bytecode filtered out.  A base commit is recorded as a
      reference point only; the working tree may differ from it, so the
      binding is likewise a reviewed snapshot whose identity is purely its
      normalized archive/inventory content digests.

    Each staged source carries its archive digest, size, and canonical
    inventory so downstream consumers rebind by content, never by path.
    """
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks import forager_matched_open_protocol as protocol_builder
    from alberta_framework.benchmarks import forager_rtu_ppo_rng_isolation as rng_patch
    from alberta_framework.benchmarks.forager_matched_protocol import SourceBinding

    source_parent = root / "sources"
    source_parent.mkdir()

    upstream_dir = source_parent / "upstream"
    upstream_dir.mkdir()
    upstream_archive = upstream_dir / "source.tar"
    _build_exact_git_archive(upstream_checkout, upstream_archive)
    upstream_root = upstream_dir / "source"
    _extract_git_archive(upstream_archive, upstream_root)
    upstream_inventory = executor.source_inventory(upstream_root)
    upstream_inventory_sha = executor.source_inventory_sha256(upstream_root)
    if upstream_inventory_sha != _QUALIFIED_UPSTREAM_INVENTORY_SHA256:
        raise ForagerMatchedQualificationError("extracted upstream inventory drifted")
    upstream_inventory_path = upstream_dir / "inventory.json"
    _write_canonical(upstream_inventory_path, cast(Mapping[str, Any], upstream_inventory))
    upstream_binding = SourceBinding(
        provenance_kind="git_tree",
        repository=protocol_builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        base_commit=_QUALIFIED_UPSTREAM_COMMIT,
        tree_git_sha1=_QUALIFIED_UPSTREAM_TREE,
        archive_sha256=_QUALIFIED_UPSTREAM_ARCHIVE_SHA256,
        inventory_sha256=upstream_inventory_sha,
        snapshot_descriptor_sha256=None,
    )

    isolated_dir = source_parent / "upstream_rng_isolated"
    isolated_dir.mkdir()
    isolated_root = isolated_dir / "source"
    _extract_git_archive(upstream_archive, isolated_root)
    patch_target = isolated_root / rng_patch.UPSTREAM_SOURCE_PATH
    upstream_rtu = _read_stable(patch_target, "upstream RTU/PPO source", maximum=_MAX_SOURCE_BYTES)
    derived = rng_patch.derive_isolated_rtu_ppo_source(upstream_rtu)
    if derived.patch_sha256 != _QUALIFIED_RNG_PATCH_SHA256:
        raise ForagerMatchedQualificationError("RNG-isolation patch digest drifted")
    patch_target.write_bytes(derived.source)
    patch_target.chmod(0o644)
    patch_path = isolated_dir / "rng-isolation.patch"
    patch_path.write_bytes(derived.patch)
    isolated_archive = isolated_dir / "source.tar"
    _archive_tree(isolated_root, isolated_archive)
    isolated_inventory = executor.source_inventory(isolated_root)
    isolated_inventory_sha = executor.source_inventory_sha256(isolated_root)
    isolated_archive_sha, isolated_archive_size = _sha256_file(isolated_archive)
    isolated_inventory_path = isolated_dir / "inventory.json"
    _write_canonical(isolated_inventory_path, cast(Mapping[str, Any], isolated_inventory))
    isolated_descriptor = {
        "schema_version": MATCHED_CURRENT_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
        "classification": "reviewed_snapshot_content_identity_only",
        "repository": protocol_builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        "base_commit": _QUALIFIED_UPSTREAM_COMMIT,
        "base_tree_git_sha1": _QUALIFIED_UPSTREAM_TREE,
        "base_archive_sha256": _QUALIFIED_UPSTREAM_ARCHIVE_SHA256,
        "archive": {
            "format": "normalized_ustar_uid_gid_mtime_zero_v1",
            "sha256": isolated_archive_sha,
            "size_bytes": isolated_archive_size,
        },
        "normalized_inventory_sha256": isolated_inventory_sha,
        "derivation": dict(derived.descriptor),
        "derivation_descriptor_sha256": derived.descriptor_sha256,
        "patch_sha256": derived.patch_sha256,
        "authority": {
            "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "promotion_authorized": False,
        },
    }
    isolated_descriptor_path = isolated_dir / "snapshot-descriptor.json"
    isolated_descriptor_sha = _write_canonical(isolated_descriptor_path, isolated_descriptor)
    isolated_binding = SourceBinding(
        provenance_kind="reviewed_snapshot",
        repository=protocol_builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        base_commit=_QUALIFIED_UPSTREAM_COMMIT,
        tree_git_sha1=None,
        archive_sha256=isolated_archive_sha,
        inventory_sha256=isolated_inventory_sha,
        snapshot_descriptor_sha256=isolated_descriptor_sha,
    )

    alberta_dir = source_parent / "alberta"
    alberta_dir.mkdir()
    alberta_root = alberta_dir / "source"
    alberta_root.mkdir()
    _copy_alberta_tree_stably(
        project_root / "alberta_framework",
        alberta_root / "alberta_framework",
    )
    alberta_archive = alberta_dir / "source.tar"
    _archive_tree(alberta_root, alberta_archive)
    alberta_inventory = executor.source_inventory(alberta_root)
    alberta_inventory_sha = executor.source_inventory_sha256(alberta_root)
    alberta_archive_sha, alberta_archive_size = _sha256_file(alberta_archive)
    alberta_inventory_path = alberta_dir / "inventory.json"
    _write_canonical(alberta_inventory_path, cast(Mapping[str, Any], alberta_inventory))
    alberta_descriptor = {
        "schema_version": MATCHED_CURRENT_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
        "classification": "reviewed_snapshot_content_identity_only",
        "repository": protocol_builder.MATCHED_CURRENT_ALBERTA_REPOSITORY,
        "base_commit": protocol_builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
        "selection": {
            "root": "alberta_framework",
            "included": "all_regular_files",
            "excluded": ["__pycache__", "*.pyc", "*.pyo"],
        },
        "archive": {
            "format": "normalized_ustar_uid_gid_mtime_zero_v1",
            "sha256": alberta_archive_sha,
            "size_bytes": alberta_archive_size,
        },
        "normalized_inventory_sha256": alberta_inventory_sha,
        "authority": {
            "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "promotion_authorized": False,
        },
    }
    alberta_descriptor_path = alberta_dir / "snapshot-descriptor.json"
    alberta_descriptor_sha = _write_canonical(alberta_descriptor_path, alberta_descriptor)
    alberta_binding = SourceBinding(
        provenance_kind="reviewed_snapshot",
        repository=protocol_builder.MATCHED_CURRENT_ALBERTA_REPOSITORY,
        base_commit=protocol_builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
        tree_git_sha1=None,
        archive_sha256=alberta_archive_sha,
        inventory_sha256=alberta_inventory_sha,
        snapshot_descriptor_sha256=alberta_descriptor_sha,
    )
    return {
        "alberta": _StagedSource(
            "alberta",
            alberta_root,
            alberta_archive,
            alberta_inventory_path,
            alberta_inventory,
            alberta_binding,
            alberta_descriptor_path,
            None,
        ),
        "upstream": _StagedSource(
            "upstream",
            upstream_root,
            upstream_archive,
            upstream_inventory_path,
            upstream_inventory,
            upstream_binding,
            None,
            None,
        ),
        "upstream_rng_isolated": _StagedSource(
            "upstream_rng_isolated",
            isolated_root,
            isolated_archive,
            isolated_inventory_path,
            isolated_inventory,
            isolated_binding,
            isolated_descriptor_path,
            patch_path,
        ),
    }


def _reverify_staged_source(source: _StagedSource) -> None:
    """Rebind every transitive source file immediately around a probe run."""
    if type(source) is not _StagedSource:
        raise TypeError("staged source must be a _StagedSource")
    from alberta_framework.benchmarks import forager_matched_executor as executor

    try:
        inventory = executor.source_inventory(source.root)
        normalized_sha256 = executor.source_inventory_sha256(source.root)
    except ValueError as exc:
        raise ForagerMatchedQualificationError(
            f"staged source changed for {source.key}"
        ) from exc
    expected_inventory_sha256 = source.binding.inventory_sha256
    if (
        type(expected_inventory_sha256) is not str
        or _SHA256.fullmatch(expected_inventory_sha256) is None
        or _plain_json(inventory) != _plain_json(source.inventory)
        or normalized_sha256 != expected_inventory_sha256
    ):
        raise ForagerMatchedQualificationError(
            f"staged source changed for {source.key}"
        )


def _stage_executor_qualification_roots(root: Path) -> _StagedExecutorQualifications:
    """Byte-copy the frozen executor dependencies without opening nested artifacts."""
    from alberta_framework.benchmarks import forager_matched_executor as executor

    parent = root / "executor-qualification"
    parent.mkdir()
    cpu_root = parent / "cpu"
    rng_root = parent / "rng-parity"
    _copy_tree(executor.DEFAULT_CPU_QUALIFICATION_ROOT, cpu_root, alberta_filter=False)
    _copy_tree(executor.DEFAULT_RNG_PARITY_QUALIFICATION_ROOT, rng_root, alberta_filter=False)
    cpu_inventory = executor.source_inventory(cpu_root)
    rng_inventory = executor.source_inventory(rng_root)
    # Parse only the five small JSON identities that the executor itself requires.
    # Other copied files (including nested tar bytes) remain opaque byte copies.
    executor.load_executor_qualification_artifacts(
        cpu_root=cpu_root,
        rng_parity_root=rng_root,
    )
    return _StagedExecutorQualifications(
        cpu_root=cpu_root,
        rng_root=rng_root,
        cpu_inventory=cpu_inventory,
        rng_inventory=rng_inventory,
    )


def _verify_archive_root_binding(
    archive_path: Path,
    source_root: Path,
    expected_inventory: Mapping[str, Any],
    expected_normalized_sha256: str,
) -> None:
    """Cross-bind one bounded archive to its materialized root and protocol identity."""
    from alberta_framework.benchmarks import forager_matched_executor as executor

    with tempfile.TemporaryDirectory(prefix="alberta-matched-archive-replay-") as temporary:
        replay_root = Path(temporary) / "source"
        _extract_git_archive(archive_path, replay_root)
        if executor.source_inventory(replay_root) != expected_inventory:
            raise ForagerMatchedQualificationError(
                "source archive content differs from its detailed root inventory"
            )
        if executor.source_inventory_sha256(replay_root) != expected_normalized_sha256:
            raise ForagerMatchedQualificationError(
                "source archive content differs from its normalized root identity"
            )
    if executor.source_inventory(source_root) != expected_inventory:
        raise ForagerMatchedQualificationError(
            "source root differs from its detailed archive inventory"
        )


def _replace_integer_literals(raw: bytes, transforms: Sequence[Any]) -> bytes:
    """Replay the protocol's exact byte-preserving integer transforms."""
    parsed = _decode_json(raw, "original configuration")
    if type(parsed) is not dict:
        raise ForagerMatchedQualificationError("original configuration must be an object")
    try:
        expected = cast(
            dict[str, Any],
            json.loads(json.dumps(parsed, allow_nan=False)),
        )
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedQualificationError(
            "original configuration is not finite canonical JSON"
        ) from exc
    transformed = raw
    for transform in transforms:
        if (
            transform.transform_type != "byte_preserving_unique_literal_replacement"
            or transform.value_type != "integer"
            or type(transform.value) is not int
        ):
            raise ForagerMatchedQualificationError("unsupported configuration transform")
        components = transform.target.split(".")
        current = expected
        for component in components[:-1]:
            nested = current.get(component)
            if type(nested) is not dict:
                raise ForagerMatchedQualificationError("configuration transform target is absent")
            current = cast(dict[str, Any], nested)
        leaf = components[-1]
        if leaf not in current or type(current[leaf]) is not int:
            raise ForagerMatchedQualificationError("configuration transform leaf is not integer")
        encoded_key = json.dumps(leaf, ensure_ascii=True).encode("ascii")
        pattern = re.compile(
            rb"(?P<prefix>"
            + re.escape(encoded_key)
            + rb"[ \t\r\n]*:[ \t\r\n]*)(?P<value>-?(?:0|[1-9][0-9]*))"
        )
        matches = list(pattern.finditer(transformed))
        if len(matches) != 1:
            raise ForagerMatchedQualificationError("configuration transform key is not byte-unique")
        match = matches[0]
        if int(match.group("value")) != current[leaf]:
            raise ForagerMatchedQualificationError("parsed and literal transform values differ")
        replacement = match.group("prefix") + str(transform.value).encode("ascii")
        transformed = transformed[: match.start()] + replacement + transformed[match.end() :]
        current[leaf] = transform.value
    if _decode_json(transformed, "derived configuration") != expected:
        raise ForagerMatchedQualificationError("configuration changed beyond declared transforms")
    return transformed


def _external_original_path(
    project_root: Path,
    upstream_root: Path,
    logical_path: str,
) -> Path:
    if logical_path.startswith("fov_baseline_screening_v1/"):
        return project_root / "outputs/forager" / logical_path
    if logical_path.startswith("fov_stateful_baseline_screening_v1/"):
        return project_root / "outputs/forager" / logical_path
    return upstream_root.joinpath(*_safe_relative(logical_path, "upstream config path").parts)


def _materialize_configurations(
    project_root: Path,
    sources: Mapping[SourceKey, _StagedSource],
    root: Path,
) -> dict[str, _MaterializedConfiguration]:
    """Materialize the original/derived configuration byte pair for every candidate.

    Alberta candidates use builder-generated worker envelopes: canonical JSON
    whose digest must equal the frozen fingerprint, with ``original ==
    derived`` and an empty transform list.  External candidates read the
    original bytes from the staged upstream tree (or the pinned
    ``outputs/forager`` FOV screening baselines), verify them against the
    frozen original digest, then replay the declared byte-preserving integer
    transforms (:func:`_replace_integer_literals`) to produce the derived
    bytes, which must match the frozen derived digest.  Both files are
    written side by side so any verifier can replay the transform offline
    and confirm nothing beyond the declared integers changed.
    """
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder
    from alberta_framework.benchmarks.forager_matched_protocol import ConfigurationBinding

    configuration_root = root / "configurations"
    configuration_root.mkdir()
    result: dict[str, _MaterializedConfiguration] = {}
    local_configs = builder.matched_current_alberta_configurations()
    local_fingerprints = builder.matched_current_alberta_configuration_fingerprints()
    requirements = cast(
        Mapping[str, tuple[str, str, str, tuple[Any, ...]]],
        getattr(builder, "_UPSTREAM_CONFIGURATION_REQUIREMENTS"),
    )
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        candidate_root = configuration_root / candidate_id
        candidate_root.mkdir()
        original_path = candidate_root / "original.json"
        derived_path = candidate_root / "derived.json"
        if candidate_id in local_configs:
            raw = executor.canonical_json_bytes(local_configs[candidate_id])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != local_fingerprints[candidate_id]:
                raise ForagerMatchedQualificationError("Alberta worker envelope digest drifted")
            original_path.write_bytes(raw)
            derived_path.write_bytes(raw)
            binding = ConfigurationBinding(
                original_path=f"matched_current/alberta_configs/{candidate_id}.json",
                original_sha256=digest,
                derived_sha256=digest,
                allowed_transforms=(),
            )
        else:
            try:
                logical_path, expected_original, expected_derived, transforms = requirements[
                    candidate_id
                ]
            except KeyError as exc:
                raise ForagerMatchedQualificationError(
                    f"missing external config rule for {candidate_id}"
                ) from exc
            source_path = _external_original_path(
                project_root,
                sources["upstream"].root,
                logical_path,
            )
            original = _read_stable(source_path, "external original configuration")
            if hashlib.sha256(original).hexdigest() != expected_original:
                raise ForagerMatchedQualificationError(
                    f"external original configuration drifted for {candidate_id}"
                )
            derived = _replace_integer_literals(original, transforms)
            if hashlib.sha256(derived).hexdigest() != expected_derived:
                raise ForagerMatchedQualificationError(
                    f"external derived configuration drifted for {candidate_id}"
                )
            original_path.write_bytes(original)
            derived_path.write_bytes(derived)
            binding = ConfigurationBinding(
                original_path=logical_path,
                original_sha256=expected_original,
                derived_sha256=expected_derived,
                allowed_transforms=transforms,
            )
        result[candidate_id] = _MaterializedConfiguration(
            candidate_id,
            original_path,
            derived_path,
            binding,
        )
    if tuple(result) != builder.MATCHED_CURRENT_CANDIDATE_IDS:
        raise AssertionError("configuration order drifted")
    return result


_EXTERNAL_EXECUTION: Final = MappingProxyType(
    {
        "external_dqn_ln": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/DQN_LN",
            "DQN_LN",
        ),
        "external_dqn_crelu": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/DQN_CReLU",
            "DQN_CReLU",
        ),
        "external_dqn_plain": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/DQN",
            "DQN",
        ),
        "external_dqn_redo": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/DQN_ReDo_PostLNScore",
            "DQN_ReDo_PostLNScore",
        ),
        "external_drqn_paper": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/DRQN",
            "DRQN",
        ),
        "isolated_ppo": (
            "upstream_rng_isolated",
            "src/rtu_ppo.py",
            "official_foragax_ppo_frozen_updates_v1",
            "results/results/run/alberta/PPO_2048_relu",
            "PPO_2048_relu",
        ),
        "isolated_rtu": (
            "upstream_rng_isolated",
            "src/rtu_ppo.py",
            "official_foragax_ppo_frozen_updates_v1",
            "results/results/run/alberta/PPO-RTU_LN_128_1_relu",
            "PPO-RTU_LN_128_1_relu",
        ),
        "exact_ppo": (
            "upstream",
            "src/rtu_ppo.py",
            "official_foragax_ppo_frozen_updates_v1",
            "results/results/run/alberta/PPO_2048_relu",
            "PPO_2048_relu",
        ),
        "search_oracle": (
            "upstream",
            "src/continuing_main.py",
            "official_foragax_continuing_main_v4",
            "results/results/run/alberta/Search-Oracle",
            "Search-Oracle",
        ),
    }
)


def _probe_invocations(
    sources: Mapping[SourceKey, _StagedSource],
    configurations: Mapping[str, _MaterializedConfiguration],
) -> tuple[ProbeInvocation, ...]:
    """Build the per-candidate probe matrix: source tree, entrypoint, style, and seed transport.

    Every digest an invocation carries (probe module, entrypoint, derived
    configuration) is hashed from the staged bytes here, so the container
    can re-verify exactly what the host claims to have mounted.
    """
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    probe_path = (
        sources["alberta"].root
        / "alberta_framework/benchmarks/forager_matched_qualification.py"
    )
    probe_sha256, _probe_size = _sha256_file(probe_path, maximum=_MAX_JSON_BYTES)
    invocations: list[ProbeInvocation] = []
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        if candidate_id in builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS:
            source_key: SourceKey = "alberta"
            entrypoint = "alberta_framework/benchmarks/_forager_matched_alberta_worker.py"
            style = "alberta_single_seed_v1"
            result_root = "results"
            expected_agent = candidate_id
        else:
            raw_source, entrypoint, style, result_root, expected_agent = _EXTERNAL_EXECUTION[
                candidate_id
            ]
            source_key = cast(SourceKey, raw_source)
        entrypoint_host = sources[source_key].root.joinpath(
            *_safe_relative(entrypoint, "entrypoint").parts
        )
        entrypoint_sha, _size = _sha256_file(entrypoint_host, maximum=_MAX_JSON_BYTES)
        config = configurations[candidate_id]
        config_sha, _config_size = _sha256_file(config.derived)
        if config_sha != config.binding.derived_sha256:
            raise ForagerMatchedQualificationError("materialized configuration changed")
        implementation = {
            **{
                value: "alberta_causal_map"
                for value in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
            },
            **{
                value: "alberta_horde_actor_critic"
                for value in builder.MATCHED_CURRENT_HORDE_CANDIDATE_IDS
            },
            "alberta_rtu_h08_taylor": "alberta_rtu_rtrl",
            "external_dqn_ln": "upstream_dqn_ln",
            "external_dqn_crelu": "upstream_dqn_crelu",
            "external_dqn_plain": "upstream_dqn_plain",
            "external_dqn_redo": "upstream_dqn_redo_post_ln",
            "external_drqn_paper": "upstream_drqn",
            "isolated_ppo": "upstream_ppo_isolated_rng",
            "isolated_rtu": "upstream_rtu_ppo_isolated_rng",
            "exact_ppo": "upstream_ppo",
            "search_oracle": "upstream_search_oracle",
        }[candidate_id]
        invocations.append(
            ProbeInvocation(
                candidate_id=candidate_id,
                source_key=source_key,
                source_root=sources[source_key].root,
                probe_path=probe_path,
                probe_sha256=probe_sha256,
                configuration=config.derived,
                configuration_sha256=config_sha,
                entrypoint_path=entrypoint,
                entrypoint_sha256=entrypoint_sha,
                entrypoint_family=(
                    "alberta_single_seed_worker"
                    if source_key == "alberta"
                    else (
                        "rtu_ppo_rng_isolation_adapter"
                        if candidate_id in {"isolated_ppo", "isolated_rtu"}
                        else ("rtu_ppo" if candidate_id == "exact_ppo" else "continuing_main")
                    )
                ),
                implementation_kind=implementation,
                invocation_style=style,
                result_root=result_root,
                seed_transport=(
                    "direct"
                    if source_key == "alberta"
                    else (
                        "adapter_injected"
                        if candidate_id in {"isolated_ppo", "isolated_rtu"}
                        else "top_level_seed"
                    )
                ),
                expected_agent=expected_agent,
                horizon=builder.MATCHED_CURRENT_HORIZON,
            )
        )
    return tuple(invocations)


def _tree_element_count(value: Any) -> int:
    import jax
    import numpy as np

    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        shape = getattr(leaf, "shape", None)
        if shape is not None:
            total += int(np.prod(shape, dtype=np.int64))
    return total


def _local_probe_resources(
    implementation_kind: str,
    configuration_path: Path,
    seed: int,
    horizon: int,
) -> tuple[dict[str, int], dict[str, Any], str]:
    """Initialize one local policy at reset only; consume no transition/reward."""
    import jax.numpy as jnp
    import jax.random as jr

    from alberta_framework.benchmarks._forager_matched_alberta_worker import (
        _load_configuration,
    )
    from alberta_framework.benchmarks.causal_map_forager import (
        CausalMapForagerAgent,
        CausalMapForagerConfig,
    )
    from alberta_framework.benchmarks.forager import (
        AlbertaForagerAgent,
        AlbertaForagerConfig,
        ForagerEnvConfig,
        RTURTRLForagerAgent,
        RTURTRLForagerConfig,
    )

    envelope = _load_configuration(configuration_path)
    if envelope.implementation_kind != implementation_kind:
        raise ForagerMatchedQualificationError("local implementation envelope drifted")
    environment, parameters = ForagerEnvConfig.paper_field_of_view(aperture_size=9).make()
    environment_key = jr.key(seed, impl="threefry2x32")
    _next_environment_key, reset_key = jr.split(environment_key)
    observation, _environment_state = environment.reset(reset_key, parameters)
    supplements: dict[str, Any] = {
        "fixed_substrate_parameter_count": 0,
        "target_snapshot_parameter_count": 0,
        "non_gradient_operations": {
            "causal_nonparametric_transition_updates": 0,
            "target_snapshot_refreshes": 0,
            "redo_recycles": 0,
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        if implementation_kind == "alberta_causal_map":
            causal_policy = CausalMapForagerAgent(
                cast(CausalMapForagerConfig, envelope.configuration),
                seed=seed,
            )
            causal_policy.start(observation)
            parameter_count = 0
            optimizer_update_count = 0
            recurrent_state_elements = _tree_element_count(causal_policy.state)
            supplements["non_gradient_operations"][
                "causal_nonparametric_transition_updates"
            ] = horizon
            parser_identity = "MatchedAlbertaWorkerConfiguration:CausalMapForagerConfig"
        elif implementation_kind == "alberta_horde_actor_critic":
            horde_policy = AlbertaForagerAgent(
                cast(AlbertaForagerConfig, envelope.configuration),
                seed=seed,
            )
            horde_policy.start(observation)
            if (  # noqa: SLF001
                horde_policy._state is None or horde_policy._recurrent_state is None
            ):
                raise ForagerMatchedQualificationError("Horde constructor did not initialize")
            horde_state = horde_policy._state  # noqa: SLF001
            parameter_count = (
                _tree_element_count(horde_state.actor_trunk)
                + _tree_element_count(horde_state.actor_head_w)
                + _tree_element_count(horde_state.actor_head_b)
                + _tree_element_count(horde_state.critic_state.trunk_params)
                + _tree_element_count(horde_state.critic_state.head_params)
            )
            optimizer_update_count = horizon
            recurrent_state_elements = int(  # noqa: SLF001
                horde_policy._recurrent_state.hidden.size
            )
            supplements["fixed_substrate_parameter_count"] = max(
                0,
                _tree_element_count(horde_policy._recurrent_state)  # noqa: SLF001
                - recurrent_state_elements,
            )
            parser_identity = "MatchedAlbertaWorkerConfiguration:AlbertaForagerConfig"
        elif implementation_kind == "alberta_rtu_rtrl":
            rtu_policy = RTURTRLForagerAgent(
                cast(RTURTRLForagerConfig, envelope.configuration),
                seed=seed,
            )
            rtu_policy.start(observation)
            if rtu_policy._state is None:  # noqa: SLF001
                raise ForagerMatchedQualificationError("RTU constructor did not initialize")
            rtu_state = rtu_policy._state  # noqa: SLF001
            parameter_count = _tree_element_count((rtu_state.actor_params, rtu_state.critic_params))
            optimizer_update_count = horizon
            recurrent_state_elements = _tree_element_count(
                (rtu_state.actor_rtu_state, rtu_state.critic_rtu_state)
            )
            parser_identity = "MatchedAlbertaWorkerConfiguration:RTURTRLForagerConfig"
        else:
            raise ForagerMatchedQualificationError("unsupported local implementation")
    if parameter_count < 0 or recurrent_state_elements < 0:
        raise ForagerMatchedQualificationError("local resource count is negative")
    if not bool(jnp.asarray(True)):
        raise AssertionError("JAX device check failed")
    return (
        {
            "parameter_count": parameter_count,
            "optimizer_update_count": optimizer_update_count,
            "replay_capacity_transitions": 0,
            "recurrent_state_elements": recurrent_state_elements,
        },
        supplements,
        parser_identity,
    )


def _ppo_parameter_resources(
    agent_name: str,
    hypers: Mapping[str, Any],
    seed: int,
    horizon: int,
) -> dict[str, int]:
    """Initialize the exact upstream Flax network without running a rollout."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from algorithms.nn.ACConv import ActorCriticConv  # type: ignore[import-not-found]
    from algorithms.nn.ACMLP import ActorCriticMLP  # type: ignore[import-not-found]
    from algorithms.nn.RealTimeACConv import (  # type: ignore[import-not-found]
        RealTimeActorCriticConv,
    )
    from algorithms.nn.RealTimeACConvHint import (  # type: ignore[import-not-found]
        RealTimeActorCriticConvHint,
    )
    from algorithms.nn.RealTimeACConvHintRTU import (  # type: ignore[import-not-found]
        RealTimeActorCriticConvHintRTU,
    )
    from algorithms.nn.RealTimeACConvPooling import (  # type: ignore[import-not-found]
        RealTimeActorCriticConvPooling,
    )
    from algorithms.nn.RealTimeACMLP import (  # type: ignore[import-not-found]
        RealTimeActorCriticMLP,
    )
    from algorithms.nn.RealTimeACMLPMulti import (  # type: ignore[import-not-found]
        RealTimeActorCriticMLPMulti,
    )
    from algorithms.PPORegistry import getAgent  # type: ignore[import-not-found]
    from foragax.registry import make

    representation = cast(Mapping[str, Any], hypers["representation"])
    environment_config = cast(Mapping[str, Any], hypers["environment"])
    environment = make(
        str(environment_config["env_id"]),
        aperture_size=int(environment_config["aperture_size"]),
    )
    root_key = jr.key(seed, impl="threefry2x32")
    root_key, reset_key = jr.split(root_key)
    observation, _environment_state = environment.reset(reset_key, environment.default_params)
    action_dim = 4
    agent_class = getAgent(agent_name)
    hidden_size = int(representation["hidden"])
    d_hidden = int(representation["d_hidden"])
    activation = str(representation.get("activation", "tanh")).lower()
    use_sinusoidal = bool(hypers.get("use_sinusoidal_encoding", False))
    use_reward_trace = bool(
        hypers.get("use_reward_trace", representation.get("use_reward_trace", False))
    )
    use_layernorm = bool(hypers.get("use_layernorm", representation.get("use_layernorm", False)))
    kwargs: dict[str, Any] = {}
    if representation.get("sparsity") is not None:
        kwargs["sparsity"] = representation["sparsity"]
    if representation.get("spectral_radius") is not None:
        kwargs["spectral_radius"] = representation["spectral_radius"]
    convolutional = (
        ActorCriticConv,
        RealTimeActorCriticConv,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
        RealTimeActorCriticConvHintRTU,
    )
    if agent_class in convolutional:
        kwargs["conv"] = str(representation.get("conv", "Conv2D"))
    if agent_class is ActorCriticMLP:
        kwargs["use_middle_layer"] = bool(representation.get("use_middle_layer", True))
        kwargs["use_midlayer_layernorm"] = bool(representation.get("use_midlayer_layernorm", False))
    if agent_class is RealTimeActorCriticMLP:
        kwargs["rtu_type"] = str(representation.get("rtu_type", "linear_rtu"))
        kwargs["alpha"] = float(representation.get("rtu_alpha", 0.9))
    network = agent_class(
        action_dim=action_dim,
        activation=activation,
        hidden_size=hidden_size,
        d_hidden=d_hidden,
        cont=False,
        use_sinusoidal_encoding=use_sinusoidal,
        use_reward_trace=use_reward_trace,
        use_layernorm=use_layernorm,
        **kwargs,
    )
    if isinstance(observation, Mapping):
        image_shape = observation["image"].shape
        hint_shape = (1 + observation["hint"].shape[-1],)
        hint_dim = int(observation["hint"].shape[-1])
    else:
        image_shape = observation.shape
        hint_shape = (1,)
        hint_dim = 1
    initial_input = (
        jnp.zeros((1, *image_shape)),
        jnp.zeros((1, action_dim)),
        jnp.zeros((1, *hint_shape)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    activation_multiplier = 2 if activation == "crelu" else 1
    plain_conv_rtu = agent_class is RealTimeActorCriticConv
    conv_rtu = agent_class in (
        RealTimeActorCriticConv,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
    )
    conv_hint_rtu = agent_class is RealTimeActorCriticConvHintRTU
    mlp_rtu = agent_class in (
        RealTimeActorCriticMLP,
        RealTimeActorCriticMLPMulti,
        ActorCriticMLP,
    )
    if plain_conv_rtu or mlp_rtu:
        input_width = hidden_size * activation_multiplier + action_dim + hint_shape[0]
        if use_sinusoidal:
            input_width += 2
        if use_reward_trace:
            input_width += 1
    elif conv_rtu:
        input_width = hidden_size * activation_multiplier
    elif conv_hint_rtu:
        input_width = hint_dim
    else:
        input_width = hidden_size
    if conv_hint_rtu:
        initial_memory = agent_class.initialize_memory(1, d_hidden, hint_dim)
    else:
        initial_memory = agent_class.initialize_memory(1, d_hidden, input_width)
    root_key, network_key = jr.split(root_key)
    with contextlib.redirect_stdout(io.StringIO()):
        network_parameters = network.init(network_key, initial_memory, initial_input)
    parameter_count = _tree_element_count(network_parameters)
    recurrent = _tree_element_count(initial_memory) if "RTU" in agent_name else 0
    rollout_steps = int(hypers["rollout_steps"])
    num_updates = int(hypers["num_updates"])
    if rollout_steps * num_updates != horizon:
        raise ForagerMatchedQualificationError("PPO rollout/update product differs from horizon")
    optimizer_updates = num_updates * int(hypers["epochs"]) * int(hypers["num_mini_batch"])
    if jax.default_backend() != "cpu":
        raise ForagerMatchedQualificationError("PPO probe did not use the CPU backend")
    return {
        "parameter_count": parameter_count,
        "optimizer_update_count": optimizer_updates,
        "replay_capacity_transitions": 0,
        "recurrent_state_elements": recurrent,
    }


def _continuing_main_v4_full_update_blocks(
    *,
    horizon: int,
    update_frequency: int,
    initial_agent_step: int,
) -> int:
    """Count full update blocks in the frozen one-chunk continuing_main v4 path."""
    if horizon < 0 or update_frequency < 2 or initial_agent_step != 1:
        raise ForagerMatchedQualificationError("invalid replay optimizer schedule")
    prefix = min(
        (update_frequency - (initial_agent_step % update_frequency)) % update_frequency,
        horizon,
    )
    # continuing_main intentionally sends the final partial block through
    # no_update when there is no freeze-boundary crossing.  For H=499712 and
    # f=4, this excludes the otherwise aligned t=499712 opportunity.
    return (horizon - prefix) // update_frequency


def _exact_continuing_main_v4_optimizer_update_count(
    *,
    horizon: int,
    buffer_min_size: int,
    update_frequency: int,
    initial_agent_step: int,
    freeze_steps: float,
) -> int:
    """Count successful gradient applications in the pinned one-chunk fast path."""
    if buffer_min_size < 1 or not math.isinf(freeze_steps):
        raise ForagerMatchedQualificationError("invalid replay optimizer schedule")
    full_blocks = _continuing_main_v4_full_update_blocks(
        horizon=horizon,
        update_frequency=update_frequency,
        initial_agent_step=initial_agent_step,
    )
    first_sampleable_block = (buffer_min_size + update_frequency - 1) // update_frequency
    return max(0, full_blocks - first_sampleable_block + 1)


def _external_probe_resources(
    configuration_path: Path,
    seed: int,
    horizon: int,
    invocation_style: str,
) -> tuple[dict[str, int], dict[str, Any], str, str, int, int]:
    """Parse exact PyExpUtils config and initialize only model structure."""
    import jax
    import numpy as np
    from experiment import ExperimentModel  # type: ignore[import-not-found]

    experiment = ExperimentModel.load(os.fspath(configuration_path))
    hypotheses = cast(dict[str, Any], experiment.get_hypers(seed))
    stored_seed = int(experiment.getRun(seed))
    if invocation_style == "official_foragax_ppo_frozen_updates_v1":
        offset = int(hypotheses.get("seed_offset", 0))
    else:
        experiment_hypers = cast(
            Mapping[str, Any],
            hypotheses.get("experiment", {}),
        )
        offset = int(experiment_hypers.get("seed_offset", 0))
    effective_seed = stored_seed + offset
    operations = {
        "causal_nonparametric_transition_updates": 0,
        "target_snapshot_refreshes": 0,
        "redo_recycles": 0,
    }
    supplements: dict[str, Any] = {
        "fixed_substrate_parameter_count": 0,
        "target_snapshot_parameter_count": 0,
        "non_gradient_operations": operations,
    }
    if experiment.agent in {"PPO_2048_relu", "PPO-RTU_LN_128_1_relu"}:
        resources = _ppo_parameter_resources(
            str(experiment.agent),
            hypotheses,
            effective_seed,
            horizon,
        )
        parser = "PyExpUtils.ExperimentModel+PPORegistry"
    else:
        from ml_instrumentation.Collector import Collector  # type: ignore[import-not-found]
        from ml_instrumentation.Sampler import Ignore  # type: ignore[import-not-found]
        from problems.registry import getProblem  # type: ignore[import-not-found]

        with contextlib.redirect_stdout(io.StringIO()):
            collector = Collector(config={}, default=Ignore())
            problem = getProblem(experiment.problem)(experiment, seed, collector)
            agent = problem.getAgent()
            # This is the one structural reset permitted by the qualification
            # boundary.  It proves that the effective public seed instantiates
            # the exact environment, but performs no environment transition.
            observation = problem.getEnvironment().start()
            agent.start(observation)
        if experiment.agent == "Search-Oracle":
            resources = {
                "parameter_count": 0,
                "optimizer_update_count": 0,
                "replay_capacity_transitions": 0,
                "recurrent_state_elements": 0,
            }
        else:
            experiment_settings = cast(Mapping[str, Any], hypotheses.get("experiment", {}))
            parameter_count = _tree_element_count(agent.state.params)
            supplements["target_snapshot_parameter_count"] = _tree_element_count(
                agent.state.target_params
            )
            update_frequency = int(agent.state.hypers.update_freq)
            buffer_min_size = int(agent.buffer_min_size)
            freeze_steps = float(agent.state.hypers.freeze_steps)
            initial_agent_step = int(np.asarray(agent.state.steps))
            save_every = int(experiment_settings.get("save_every", 10_001_000))
            video_every = int(experiment_settings.get("video_every", save_every))
            if (
                int(experiment_settings.get("ntk_freq", 0)) != 0
                or int(experiment_settings.get("video_length", 0)) != 0
                or save_every <= horizon
                or video_every <= horizon
            ):
                raise ForagerMatchedQualificationError(
                    "continuing_main resource proof requires its frozen one-chunk schedule"
                )
            optimizer_updates = _exact_continuing_main_v4_optimizer_update_count(
                horizon=horizon,
                buffer_min_size=buffer_min_size,
                update_frequency=update_frequency,
                initial_agent_step=initial_agent_step,
                freeze_steps=freeze_steps,
            )
            operations["target_snapshot_refreshes"] = optimizer_updates // int(
                agent.state.hypers.target_refresh
            )
            if experiment.agent == "DQN_ReDo_PostLNScore":
                full_blocks = _continuing_main_v4_full_update_blocks(
                    horizon=horizon,
                    update_frequency=update_frequency,
                    initial_agent_step=initial_agent_step,
                )
                operations["redo_recycles"] = full_blocks // int(
                    agent.state.hypers.redo_freq
                )
            resources = {
                # ``params`` is the unique online/trainable tree.  Deliberately
                # exclude target_params and the optimizer state.
                "parameter_count": parameter_count,
                "optimizer_update_count": optimizer_updates,
                "replay_capacity_transitions": int(agent.buffer_size),
                "recurrent_state_elements": (
                    _tree_element_count(agent.state.carry)
                    if experiment.agent == "DRQN"
                    else 0
                ),
            }
        parser = "PyExpUtils.ExperimentModel+problem.registry"
    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ForagerMatchedQualificationError("external probe did not use only CPU devices")
    if any(value < 0 for value in resources.values()) or not all(
        isinstance(value, (int, np.integer)) for value in resources.values()
    ):
        raise ForagerMatchedQualificationError("external resource accounting is invalid")
    return resources, supplements, parser, str(experiment.agent), stored_seed, effective_seed


def _entrypoint_contract(path: Path, invocation_style: str) -> dict[str, Any]:
    raw = _read_stable(path, "probe entrypoint", maximum=_MAX_JSON_BYTES)
    try:
        tree = ast.parse(raw, filename=path.as_posix())
    except (SyntaxError, ValueError) as exc:
        raise ForagerMatchedQualificationError("entrypoint does not parse as Python") from exc
    argument_literals: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        argument_literals.update(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
    required = {
        "alberta_single_seed_v1": {
            "--configuration",
            "--seed",
            "--horizon",
            "--output-root",
        },
        "official_foragax_continuing_main_v4": {
            "-e",
            "--exp",
            "-i",
            "--idxs",
            "--max_steps",
            "--save_path",
        },
        "official_foragax_ppo_frozen_updates_v1": {
            "-e",
            "--exp",
            "-i",
            "--idxs",
            "--save_path",
        },
    }[invocation_style]
    missing = sorted(required - argument_literals)
    if missing:
        raise ForagerMatchedQualificationError(
            f"entrypoint omits add_argument CLI bindings {missing}"
        )
    return {
        "ast_node_count": sum(1 for _node in ast.walk(tree)),
        "required_cli_literals": sorted(required),
        "required_cli_literals_present": True,
    }


def _probe_result_root(
    configuration_path: Path,
    seed: int,
    source_key: SourceKey,
) -> str:
    if source_key == "alberta":
        return "results"
    from experiment import ExperimentModel

    experiment = ExperimentModel.load(os.fspath(configuration_path))
    context = experiment.buildSaveContext(
        seed,
        base=f"{_CONTAINER_OUTPUT_BASE}/results",
    )
    resolved = Path(context.resolve(f"data/{seed}.npz"))
    try:
        relative = resolved.parent.parent.relative_to(_CONTAINER_OUTPUT_BASE)
    except ValueError as exc:
        raise ForagerMatchedQualificationError("result root escaped its output base") from exc
    return relative.as_posix()


def _container_probe(argv: Sequence[str]) -> int:
    """In-container half of one structural probe (the ``container-probe`` CLI operation).

    Re-verifies every identity the host claims to have mounted (this module's
    own bytes, image digest, configuration, entrypoint), resolves the
    candidate's agent and effective seed from the configuration, and derives
    the agent RNG provenance key words — with exactly one environment reset
    and zero transitions or reward reads, as the emitted
    ``reward_blind_boundary`` block attests.  The canonical JSON payload on
    stdout is the only channel back to the host.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--source-key",
        choices=("alberta", "upstream", "upstream_rng_isolated"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--entrypoint-path", required=True)
    parser.add_argument("--entrypoint-sha256", required=True)
    parser.add_argument("--entrypoint-family", required=True)
    parser.add_argument("--implementation-kind", required=True)
    parser.add_argument("--invocation-style", required=True)
    parser.add_argument("--expected-result-root", required=True)
    parser.add_argument("--seed-transport", required=True)
    parser.add_argument("--expected-agent", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--probe-sha256", required=True)
    arguments = parser.parse_args(argv)
    if arguments.seed != PUBLIC_QUALIFICATION_SEED:
        raise ForagerMatchedQualificationError("only public qualification seed zero is allowed")
    if arguments.image_sha256 != _QUALIFIED_IMAGE_SHA256:
        raise ForagerMatchedQualificationError("probe image identity drifted")
    probe_sha, _probe_size = _sha256_file(Path(__file__).resolve(), maximum=_MAX_JSON_BYTES)
    if probe_sha != arguments.probe_sha256:
        raise ForagerMatchedQualificationError("mounted qualification probe digest drifted")
    source_key = cast(SourceKey, arguments.source_key)
    source_root = _regular_directory(arguments.source_root, "container source root")
    config_raw = _read_stable(arguments.config, "container configuration")
    if hashlib.sha256(config_raw).hexdigest() != arguments.configuration_sha256:
        raise ForagerMatchedQualificationError("container configuration digest drifted")
    work_config = Path(_CONTAINER_WORK_CONFIG)
    if work_config.exists() or work_config.is_symlink():
        raise ForagerMatchedQualificationError("container work configuration already exists")
    work_config.write_bytes(config_raw)
    entrypoint_relative = _safe_relative(arguments.entrypoint_path, "entrypoint path")
    entrypoint = source_root.joinpath(*entrypoint_relative.parts)
    entrypoint_sha, _entrypoint_size = _sha256_file(entrypoint, maximum=_MAX_JSON_BYTES)
    if entrypoint_sha != arguments.entrypoint_sha256:
        raise ForagerMatchedQualificationError("entrypoint digest drifted inside container")
    entrypoint_contract = _entrypoint_contract(entrypoint, arguments.invocation_style)
    import_root = source_root if source_key == "alberta" else source_root / "src"
    sys.path.insert(0, os.fspath(import_root))
    if source_key == "alberta":
        resources, resource_supplement, parser_identity = _local_probe_resources(
            arguments.implementation_kind,
            work_config,
            arguments.seed,
            arguments.horizon,
        )
        resolved_agent = arguments.candidate_id
        effective_seed = arguments.seed
    else:
        (
            resources,
            resource_supplement,
            parser_identity,
            resolved_agent,
            stored_seed,
            effective_seed,
        ) = _external_probe_resources(
            work_config,
            arguments.seed,
            arguments.horizon,
            arguments.invocation_style,
        )
    if source_key == "alberta":
        stored_seed = arguments.seed
    if resolved_agent != arguments.expected_agent:
        raise ForagerMatchedQualificationError("configuration resolved a different agent")
    result_root = _probe_result_root(work_config, arguments.seed, source_key)
    if result_root != arguments.expected_result_root:
        raise ForagerMatchedQualificationError("result-root resolution drifted")
    import jax
    import jax.random as jr

    effective_seed_key = jr.key(effective_seed, impl="threefry2x32")
    key_words = [int(value) for value in jax.device_get(jr.key_data(effective_seed_key))]
    if source_key == "upstream_rng_isolated":
        agent_provenance_key = jr.fold_in(
            effective_seed_key,
            _ISOLATED_AGENT_RNG_NAMESPACE,
        )
        agent_provenance_derivation = "fold_in_isolated_agent_namespace_v1"
    elif arguments.invocation_style == "official_foragax_ppo_frozen_updates_v1":
        agent_provenance_key = jr.split(effective_seed_key)[0]
        agent_provenance_derivation = "shared_root_post_reset_split_v1"
    else:
        agent_provenance_key = effective_seed_key
        agent_provenance_derivation = "effective_seed_constructor_input_v1"
    agent_provenance_words = [
        int(value) for value in jax.device_get(jr.key_data(agent_provenance_key))
    ]
    seed_resolution = {
        "candidate_id": arguments.candidate_id,
        "qualification_seed_class": "public_nonbenchmark_seed",
        "requested_seed": arguments.seed,
        "stored_seed": stored_seed,
        "offset": effective_seed - stored_seed,
        "effective_seed": effective_seed,
        "transport": arguments.seed_transport,
        "prng_impl": "threefry2x32",
        "effective_seed_key_words": key_words,
        "agent_rng_provenance_derivation": agent_provenance_derivation,
        "agent_rng_provenance_key_words": agent_provenance_words,
        "environment_transition_count": 0,
        "reward_array_read_count": 0,
    }
    payload = {
        "schema_version": MATCHED_CURRENT_PROBE_SCHEMA_VERSION,
        "status": "structurally_qualified_content_only",
        "candidate_id": arguments.candidate_id,
        "qualification_seed": arguments.seed,
        "source_key": source_key,
        "qualification_probe": {
            "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
            "sha256": arguments.probe_sha256,
        },
        "configuration": {
            "path": _CONTAINER_WORK_CONFIG,
            "sha256": arguments.configuration_sha256,
            "parser_identity": parser_identity,
            "round_trip_accepted": True,
        },
        "entrypoint": {
            "family": arguments.entrypoint_family,
            "path": arguments.entrypoint_path,
            "sha256": entrypoint_sha,
            "python_ast_parsed": True,
            **entrypoint_contract,
        },
        "implementation_kind": arguments.implementation_kind,
        "resolved_agent": resolved_agent,
        "result_root": result_root,
        "seed_resolution": seed_resolution,
        "resources": resources,
        "resource_supplement": resource_supplement,
        "runtime": {
            "image_sha256": arguments.image_sha256,
            "python": sys.version.split()[0],
            "jax_backend": jax.default_backend(),
            "device_platforms": sorted({device.platform for device in jax.devices()}),
        },
        "reward_blind_boundary": {
            "environment_resets": 1,
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
            "benchmark_seeds_used": [],
        },
        "authority": {
            "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
            "promotion_authorized": False,
            "performance_claim": False,
        },
    }
    sys.stdout.buffer.write(_canonical_json_bytes(payload))
    return 0


def _cleanup_interrupted_probe_container(
    materialized: tuple[str, ...],
    cidfile: Path,
    container_name: str | None,
) -> Literal[
    "not_a_container_run",
    "force_removed_by_id",
    "force_removed_by_name",
    "already_absent_by_name",
]:
    """Force-remove a probe container after its foreground client is interrupted."""
    if container_name is None:
        return "not_a_container_run"
    if re.fullmatch(r"alberta-matched-qualification-[0-9a-f]{32}", container_name) is None:
        raise ForagerMatchedQualificationError(
            "OCI probe cleanup name violates its exact internal contract"
        )
    cleanup_target = container_name
    cleanup_state: Literal[
        "force_removed_by_id",
        "force_removed_by_name",
        "already_absent_by_name",
    ] = "force_removed_by_name"
    cidfile_error: BaseException | None = None
    if cidfile.exists():
        if not cidfile.is_file() or cidfile.is_symlink():
            cidfile_error = ForagerMatchedQualificationError(
                "OCI probe cidfile is not a regular file"
            )
        else:
            try:
                container_id = _read_stable(
                    cidfile,
                    "OCI probe cidfile",
                    maximum=128,
                ).decode("ascii").strip()
                if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                    raise ForagerMatchedQualificationError(
                        "OCI probe cidfile does not contain an exact container ID"
                    )
            except (
                OSError,
                UnicodeDecodeError,
                ForagerMatchedQualificationError,
            ) as exc:
                cidfile_error = exc
            else:
                cleanup_target = container_id
                cleanup_state = "force_removed_by_id"
    try:
        cleanup = subprocess.run(
            (materialized[0], "rm", "--force", cleanup_target),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForagerMatchedQualificationError(
            "OCI probe container cleanup could not be completed"
        ) from exc
    if cleanup.returncode != 0:
        try:
            absence = _run_bounded_process(
                (
                    materialized[0],
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    f"--filter=name=^/{container_name}$",
                ),
                timeout=60,
                maximum_stdout_bytes=_MAX_CLEANUP_INSPECTION_BYTES,
                maximum_stderr_bytes=_MAX_CLEANUP_INSPECTION_BYTES,
            )
        except (OSError, subprocess.SubprocessError, _BoundedProcessOutputError) as exc:
            raise ForagerMatchedQualificationError(
                "OCI probe cleanup could not prove the exact name absent"
            ) from exc
        if absence.returncode != 0 or absence.stdout or absence.stderr:
            raise ForagerMatchedQualificationError(
                "OCI probe cleanup did not remove or prove absent the exact name"
            )
        cleanup_state = "already_absent_by_name"
    if cidfile_error is not None:
        raise ForagerMatchedQualificationError(
            f"OCI probe cidfile contract failed after cleanup={cleanup_state}"
        ) from cidfile_error
    return cleanup_state


def _terminate_and_reap_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate one bounded child, then fail closed if it cannot be reaped."""
    termination_error: OSError | None = None
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            termination_error = exc
    try:
        process.wait(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise ForagerMatchedQualificationError(
            "bounded child could not be reaped after termination"
        ) from exc
    except OSError as exc:
        raise ForagerMatchedQualificationError(
            "bounded child could not be inspected after termination"
        ) from exc
    if termination_error is not None:
        raise ForagerMatchedQualificationError(
            "bounded child could not be terminated cleanly"
        ) from termination_error


def _run_bounded_process(
    command: Sequence[str],
    *,
    timeout: float,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    stdout_sink: BinaryIO | None = None,
    environment: Mapping[str, str] | None = None,
) -> QualificationProcessResult:
    """Drain both pipes into bounded memory or a caller-owned stdout sink."""
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("bounded process timeout must be finite and positive")
    if any(
        type(limit) is not int or limit < 0
        for limit in (maximum_stdout_bytes, maximum_stderr_bytes)
    ):
        raise ValueError("bounded process output limits must be nonnegative integers")
    materialized_environment: dict[str, str] | None = None
    if environment is not None:
        if not isinstance(environment, Mapping):
            raise TypeError("bounded process environment must be a mapping")
        materialized_environment = {}
        for env_key, value in environment.items():
            if (
                type(env_key) is not str
                or not env_key
                or "=" in env_key
                or "\x00" in env_key
                or type(value) is not str
                or "\x00" in value
            ):
                raise ValueError("bounded process environment is invalid")
            materialized_environment[env_key] = value
    materialized = tuple(command)
    process = subprocess.Popen(  # noqa: S603
        materialized,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=materialized_environment,
    )
    if process.stdout is None or process.stderr is None:
        _terminate_and_reap_process(process)
        raise AssertionError("bounded process pipes were not created")
    stdout = bytearray()
    stderr = bytearray()
    sizes = {"stdout": 0, "stderr": 0}
    deadline = time.monotonic() + float(timeout)
    streams: dict[str, tuple[BinaryIO, bytearray, int, BinaryIO | None]] = {
        "stdout": (
            cast(BinaryIO, process.stdout),
            stdout,
            maximum_stdout_bytes,
            stdout_sink,
        ),
        "stderr": (
            cast(BinaryIO, process.stderr),
            stderr,
            maximum_stderr_bytes,
            None,
        ),
    }
    selector: selectors.BaseSelector | None = None
    try:
        selector = selectors.DefaultSelector()
        for label, (stream, _buffer, _maximum, _sink) in streams.items():
            selector.register(stream, selectors.EVENT_READ, label)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    materialized,
                    timeout,
                    output=bytes(stdout),
                    stderr=bytes(stderr),
                )
            for selector_key, _events in selector.select(min(remaining, 1.0)):
                label = cast(Literal["stdout", "stderr"], selector_key.data)
                _stream, buffer, maximum, sink = streams[label]
                allowance = maximum - sizes[label]
                chunk = os.read(
                    selector_key.fd,
                    min(64 * 1024, allowance + 1),
                )
                if not chunk:
                    selector.unregister(selector_key.fileobj)
                    continue
                accepted = min(len(chunk), allowance)
                if accepted:
                    if sink is None:
                        buffer.extend(chunk[:accepted])
                        written = accepted
                    else:
                        written_result = sink.write(chunk[:accepted])
                        if written_result is None:
                            written = 0
                        else:
                            written = written_result
                        if written != accepted:
                            raise ForagerMatchedQualificationError(
                                "bounded process stdout sink accepted a partial write"
                            )
                    sizes[label] += written
                if accepted != len(chunk):
                    raise _BoundedProcessOutputError(
                        "child output exceeds its active byte limit"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            raise subprocess.TimeoutExpired(
                materialized,
                timeout,
                output=bytes(stdout),
                stderr=bytes(stderr),
            )
        returncode = process.wait(timeout=max(remaining, 0.001))
        return QualificationProcessResult(
            returncode,
            b"" if stdout_sink is not None else bytes(stdout),
            bytes(stderr),
        )
    except BaseException:
        _terminate_and_reap_process(process)
        raise
    finally:
        close_error: OSError | None = None
        if selector is not None:
            try:
                selector.close()
            except OSError as exc:
                close_error = exc
        if not process.stdout.closed:
            try:
                process.stdout.close()
            except OSError as exc:
                close_error = close_error or exc
        if not process.stderr.closed:
            try:
                process.stderr.close()
            except OSError as exc:
                close_error = close_error or exc
        if close_error is not None:
            raise ForagerMatchedQualificationError(
                "bounded child resources could not be closed cleanly"
            ) from close_error


def _default_runner(command: Sequence[str]) -> QualificationProcessResult:
    """Run one probe with active output/time bounds and interruption cleanup."""
    with tempfile.TemporaryDirectory(
        prefix="alberta-matched-qualification-runner-"
    ) as temporary:
        materialized = tuple(command)
        cidfile = Path(temporary) / "container.cid"
        container_name: str | None = None
        if len(materialized) >= 2 and materialized[1] == "run":
            if any(
                item == "--name"
                or item.startswith("--name=")
                or item == "--cidfile"
                or item.startswith("--cidfile=")
                for item in materialized[2:]
            ):
                raise ForagerMatchedQualificationError(
                    "OCI run command already contains a name or cidfile"
                )
            container_name = f"alberta-matched-qualification-{secrets.token_hex(16)}"
            materialized = (
                materialized[0],
                materialized[1],
                f"--cidfile={cidfile.as_posix()}",
                f"--name={container_name}",
                *materialized[2:],
            )
        try:
            completed = _run_bounded_process(
                materialized,
                timeout=_PROBE_TIMEOUT_SECONDS,
                maximum_stdout_bytes=_MAX_PROBE_OUTPUT_BYTES,
                maximum_stderr_bytes=_MAX_PROBE_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_state = _cleanup_interrupted_probe_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedQualificationError(
                f"OCI probe exceeded its timeout; cleanup={cleanup_state}"
            ) from exc
        except _BoundedProcessOutputError as exc:
            cleanup_state = _cleanup_interrupted_probe_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedQualificationError(
                f"OCI probe output exceeds its bound; cleanup={cleanup_state}"
            ) from exc
        except Exception as exc:
            cleanup_state = _cleanup_interrupted_probe_container(
                materialized, cidfile, container_name
            )
            raise ForagerMatchedQualificationError(
                f"OCI probe runner failed; cleanup={cleanup_state}"
            ) from exc
        except BaseException:
            _cleanup_interrupted_probe_container(materialized, cidfile, container_name)
            raise
        if completed.returncode != 0 and container_name is not None:
            _cleanup_interrupted_probe_container(
                materialized,
                cidfile,
                container_name,
            )
        return completed


def _invoke_qualification_runner(
    runner: QualificationRunner,
    command: Sequence[str],
    *,
    label: str,
) -> QualificationProcessResult:
    try:
        return runner(tuple(command))
    except ForagerMatchedQualificationError:
        raise
    except _BoundedProcessOutputError as exc:
        raise ForagerMatchedQualificationError(
            f"{label} runner output exceeds its bound"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForagerMatchedQualificationError(
            f"{label} runner could not run"
        ) from exc


def _executor_runner_adapter(runner: QualificationRunner) -> Any:
    from alberta_framework.benchmarks import forager_matched_executor as executor

    def adapted(command: Sequence[str]) -> Any:
        result = _invoke_qualification_runner(
            runner,
            command,
            label="qualification runtime inspector",
        )
        if type(result) is not QualificationProcessResult:
            raise ForagerMatchedQualificationError(
                "qualification runtime inspector returned an invalid result"
            )
        return executor.ProcessResult(result.returncode, result.stdout, result.stderr)

    return adapted


def _bind_probe_runtime(
    runtime: str | Path,
    runner: QualificationRunner,
) -> _ProbeRuntimeIdentity:
    """Resolve/hash the OCI CLI and inspect the exact image before any probe."""
    from alberta_framework.benchmarks import forager_matched_executor as executor

    try:
        executable = executor._resolve_runtime(runtime)  # noqa: SLF001
        executable_sha256, _size = _sha256_file(
            executable,
            maximum=512 * 1024 * 1024,
        )
        version, image_inspection = executor._inspect_runtime_bindings(  # noqa: SLF001
            executable,
            _executor_runner_adapter(runner),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ForagerMatchedQualificationError(
            f"cannot bind qualification OCI runtime: {exc}"
        ) from exc
    return _ProbeRuntimeIdentity(
        executable=executable,
        executable_sha256=executable_sha256,
        version=version,
        image_inspection=image_inspection,
    )


def _rebind_probe_runtime(
    identity: _ProbeRuntimeIdentity,
    runner: QualificationRunner,
) -> None:
    """Require the same executable, daemon version, and image around each probe."""
    if type(identity) is not _ProbeRuntimeIdentity:
        raise TypeError("probe runtime identity must be a _ProbeRuntimeIdentity")
    from alberta_framework.benchmarks import forager_matched_executor as executor

    try:
        executable_sha256, _size = _sha256_file(
            identity.executable,
            maximum=512 * 1024 * 1024,
        )
    except (ValueError, OSError) as exc:
        raise ForagerMatchedQualificationError(
            "qualification OCI runtime executable changed after binding"
        ) from exc
    if executable_sha256 != identity.executable_sha256:
        raise ForagerMatchedQualificationError(
            "qualification OCI runtime executable changed after binding"
        )
    try:
        version, image_inspection = executor._inspect_runtime_bindings(  # noqa: SLF001
            identity.executable,
            _executor_runner_adapter(runner),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ForagerMatchedQualificationError(
            f"cannot rebind qualification OCI runtime: {exc}"
        ) from exc
    if (
        _plain_json(version) != _plain_json(identity.version)
        or _plain_json(image_inspection) != _plain_json(identity.image_inspection)
    ):
        raise ForagerMatchedQualificationError(
            "qualification OCI daemon version or image identity changed after binding"
        )


def _probe_command(runtime: str | Path, invocation: ProbeInvocation) -> list[str]:
    runtime_path = os.fspath(runtime)
    for path in (invocation.source_root, invocation.configuration, invocation.probe_path):
        if "," in os.fspath(path) or "\n" in os.fspath(path):
            raise ForagerMatchedQualificationError("OCI mount path contains an unsafe character")
    return [
        runtime_path,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--cpus=2.0",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=256",
        "--tmpfs=/run/alberta:rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700",
        f"--mount=type=bind,source={invocation.source_root},destination={_CONTAINER_SOURCE_ROOT},readonly",
        f"--mount=type=bind,source={invocation.configuration},destination={_CONTAINER_CONFIG},readonly",
        f"--mount=type=bind,source={invocation.probe_path},destination={_CONTAINER_PROBE},readonly",
        "--env=HOME=/run/alberta",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=ALL_PROXY=",
        "--env=HTTP_PROXY=",
        "--env=HTTPS_PROXY=",
        "--env=all_proxy=",
        "--env=http_proxy=",
        "--env=https_proxy=",
        "--env=LD_LIBRARY_PATH=",
        "--env=LD_PRELOAD=",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=NO_PROXY=",
        "--env=no_proxy=",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=TMPDIR=/run/alberta",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta",
        f"--workdir={_CONTAINER_SOURCE_ROOT}",
        f"sha256:{_QUALIFIED_IMAGE_SHA256}",
        _QUALIFIED_PYTHON,
        "-I",
        "-B",
        _CONTAINER_PROBE,
        "container-probe",
        f"--candidate-id={invocation.candidate_id}",
        f"--source-key={invocation.source_key}",
        f"--source-root={_CONTAINER_SOURCE_ROOT}",
        f"--config={_CONTAINER_CONFIG}",
        f"--configuration-sha256={invocation.configuration_sha256}",
        f"--entrypoint-path={invocation.entrypoint_path}",
        f"--entrypoint-sha256={invocation.entrypoint_sha256}",
        f"--entrypoint-family={invocation.entrypoint_family}",
        f"--implementation-kind={invocation.implementation_kind}",
        f"--invocation-style={invocation.invocation_style}",
        f"--expected-result-root={invocation.result_root}",
        f"--seed-transport={invocation.seed_transport}",
        f"--expected-agent={invocation.expected_agent}",
        f"--horizon={invocation.horizon}",
        f"--seed={PUBLIC_QUALIFICATION_SEED}",
        f"--image-sha256={_QUALIFIED_IMAGE_SHA256}",
        f"--probe-sha256={invocation.probe_sha256}",
    ]


def _run_probe(
    runtime: str | Path,
    invocation: ProbeInvocation,
    runner: QualificationRunner,
) -> tuple[dict[str, Any], str]:
    probe_path = invocation.probe_path
    probe_sha_before, _probe_size = _sha256_file(probe_path, maximum=_MAX_JSON_BYTES)
    if probe_sha_before != invocation.probe_sha256:
        raise ForagerMatchedQualificationError("staged qualification probe digest drifted")
    result = _invoke_qualification_runner(
        runner,
        _probe_command(runtime, invocation),
        label="OCI probe",
    )
    probe_sha_after, _probe_size_after = _sha256_file(probe_path, maximum=_MAX_JSON_BYTES)
    if probe_sha_before != probe_sha_after:
        raise ForagerMatchedQualificationError("qualification probe changed during OCI execution")
    if type(result) is not QualificationProcessResult:
        raise ForagerMatchedQualificationError("probe runner returned the wrong result type")
    if result.returncode != 0:
        detail = result.stderr[-4096:].decode("utf-8", errors="replace")
        raise ForagerMatchedQualificationError(
            f"OCI probe for {invocation.candidate_id} failed: {detail}"
        )
    if result.stderr:
        raise ForagerMatchedQualificationError(
            f"OCI probe for {invocation.candidate_id} wrote unexpected stderr"
        )
    value = _decode_json(result.stdout, f"probe {invocation.candidate_id}")
    if type(value) is not dict or result.stdout != _canonical_json_bytes(value):
        raise ForagerMatchedQualificationError("OCI probe output is not canonical JSON")
    payload = cast(dict[str, Any], value)
    _verify_probe_payload(payload, invocation)
    stderr_sha = hashlib.sha256(b"").hexdigest()
    return payload, stderr_sha


def _run_bound_probe(
    runtime: _ProbeRuntimeIdentity,
    invocation: ProbeInvocation,
    source: _StagedSource,
    runner: QualificationRunner,
) -> tuple[dict[str, Any], str]:
    """Rebind the live runtime and complete source closure around one probe."""
    if (
        type(runtime) is not _ProbeRuntimeIdentity
        or type(invocation) is not ProbeInvocation
        or type(source) is not _StagedSource
        or source.key != invocation.source_key
        or source.root != invocation.source_root
    ):
        raise ForagerMatchedQualificationError(
            "probe request differs from its bound runtime/source closure"
        )
    _rebind_probe_runtime(runtime, runner)
    _reverify_staged_source(source)
    result = _run_probe(runtime.executable, invocation, runner)
    _reverify_staged_source(source)
    _rebind_probe_runtime(runtime, runner)
    return result


def _runtime_qualification() -> Any:
    from alberta_framework.benchmarks.forager_matched_open_protocol import (
        MatchedCurrentRuntimeQualification,
    )

    return MatchedCurrentRuntimeQualification(
        image_sha256=_QUALIFIED_IMAGE_SHA256,
        runtime_profile_sha256=_QUALIFIED_RUNTIME_PROFILE_SHA256,
        executor_qualification_receipt_sha256=_QUALIFIED_EXECUTOR_RECEIPT_SHA256,
        qualification_trust_anchor_identity=MATCHED_CURRENT_AUTHORITY_IDENTITY,
    )


def _source_key_for_candidate(candidate_id: str) -> SourceKey:
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    if candidate_id in builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS:
        return "alberta"
    if candidate_id in {"isolated_ppo", "isolated_rtu"}:
        return "upstream_rng_isolated"
    return "upstream"


def _candidate_qualifications(
    sources: Mapping[SourceKey, _StagedSource],
    configurations: Mapping[str, _MaterializedConfiguration],
    probes: Mapping[str, Mapping[str, Any]],
    receipt_digests: Mapping[str, str],
) -> dict[str, Any]:
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder
    from alberta_framework.benchmarks.forager_matched_protocol import ResourceAccounting

    result: dict[str, Any] = {}
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        probe = probes[candidate_id]
        raw_resources = cast(Mapping[str, int], probe["resources"])
        resources = ResourceAccounting(
            parameter_count=raw_resources["parameter_count"],
            optimizer_update_count=raw_resources["optimizer_update_count"],
            replay_capacity_transitions=raw_resources["replay_capacity_transitions"],
            recurrent_state_elements=raw_resources["recurrent_state_elements"],
        )
        seed_resolution = cast(Mapping[str, Any], probe["seed_resolution"])
        result[candidate_id] = builder.MatchedCurrentCandidateQualification(
            source=sources[_source_key_for_candidate(candidate_id)].binding,
            configuration=configurations[candidate_id].binding,
            effective_seed_proof_sha256=_canonical_sha256(seed_resolution),
            capability_qualification_receipt_sha256=receipt_digests[candidate_id],
            resources=resources,
        )
    return result


def _capability_receipt(candidate: Any, invocation: ProbeInvocation) -> dict[str, Any]:
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks.forager_matched_protocol import (
        candidate_capability_descriptor_sha256,
    )

    patch_sha256 = (
        _QUALIFIED_RNG_PATCH_SHA256
        if candidate.candidate_id in {"isolated_ppo", "isolated_rtu"}
        else None
    )
    return {
        "schema_version": executor.MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION,
        "status": "qualified",
        "candidate_id": candidate.candidate_id,
        "capability_descriptor_sha256": candidate_capability_descriptor_sha256(candidate),
        "qualification_trust_anchor_identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
        "source": candidate.source.to_dict(),
        "configuration_sha256": candidate.configuration.derived_sha256,
        "image_sha256": _QUALIFIED_IMAGE_SHA256,
        "runtime_profile_sha256": _QUALIFIED_RUNTIME_PROFILE_SHA256,
        "task_identity_sha256": candidate.runtime_binding.task_identity_sha256,
        "environment_rng_schedule_sha256": candidate.environment_rng.schedule_sha256,
        "rng_parity_contract_sha256": executor.RNG_PARITY_CONTRACT_SHA256,
        "entrypoint_family": candidate.entrypoint_family,
        "entrypoint_path": invocation.entrypoint_path,
        "python_import_root": "." if invocation.source_key == "alberta" else "src",
        "invocation_style": invocation.invocation_style,
        "result_root": invocation.result_root,
        "agent_rng_identity": candidate.agent_rng.identity,
        "environment_key_shared": candidate.agent_rng.environment_key_shared,
        "rng_isolation_patch_sha256": patch_sha256,
    }


def _relative_path(root: Path, path: Path | None) -> str | None:
    return None if path is None else path.relative_to(root).as_posix()


def _assemble_and_write(
    root: Path,
    executor_qualifications: _StagedExecutorQualifications,
    sources: Mapping[SourceKey, _StagedSource],
    configurations: Mapping[str, _MaterializedConfiguration],
    invocations: Sequence[ProbeInvocation],
    probes: Mapping[str, Mapping[str, Any]],
    probe_stderr: Mapping[str, str],
) -> None:
    """Assemble the bundle: receipts, two-pass protocol build, and the canonical manifest."""
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    runtime = _runtime_qualification()
    # Receipt <-> protocol bootstrap.  Each capability receipt embeds the
    # candidate's capability-descriptor digest, which exists only once a
    # protocol has been built — while the protocol's per-candidate
    # qualification embeds the receipt digest.  Pass 1 builds a provisional
    # protocol with placeholder receipt digests, the real receipts are
    # derived from it, and pass 2 rebuilds the protocol with the real
    # digests.  The check after both passes proves the capability descriptor
    # is byte-identical across them — i.e. the descriptor never depended on
    # the receipt digest, so the fixpoint is exact rather than circular.
    placeholder_receipts = {
        candidate_id: hashlib.sha256(f"pending:{candidate_id}".encode("ascii")).hexdigest()
        for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS
    }
    provisional_qualifications = _candidate_qualifications(
        sources,
        configurations,
        probes,
        placeholder_receipts,
    )
    provisional_protocol = builder.build_forager_matched_open_protocol(
        runtime=runtime,
        candidate_qualifications=provisional_qualifications,
    )
    invocation_index = {item.candidate_id: item for item in invocations}
    receipts: dict[str, dict[str, Any]] = {}
    receipt_digests: dict[str, str] = {}
    receipt_root = root / "receipts"
    receipt_root.mkdir()
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        candidate = provisional_protocol.candidate_index[candidate_id]
        receipt = _capability_receipt(candidate, invocation_index[candidate_id])
        digest = _canonical_sha256(receipt)
        _write_canonical(receipt_root / f"{candidate_id}.json", receipt)
        receipts[candidate_id] = receipt
        receipt_digests[candidate_id] = digest
    final_qualifications = _candidate_qualifications(
        sources,
        configurations,
        probes,
        receipt_digests,
    )
    final_protocol = builder.build_forager_matched_open_protocol(
        runtime=runtime,
        candidate_qualifications=final_qualifications,
    )
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        provisional = provisional_protocol.candidate_index[candidate_id]
        final = final_protocol.candidate_index[candidate_id]
        if (
            provisional.runtime_binding.qualified_capability_descriptor_sha256
            != final.runtime_binding.qualified_capability_descriptor_sha256
            or receipts[candidate_id]["capability_descriptor_sha256"]
            != final.runtime_binding.qualified_capability_descriptor_sha256
        ):
            raise ForagerMatchedQualificationError("capability descriptor became circular")

    qualification_artifacts = executor.load_executor_qualification_artifacts(
        cpu_root=executor_qualifications.cpu_root,
        rng_parity_root=executor_qualifications.rng_root,
    )
    source_payload: dict[str, Any] = {}
    for source_key, source in sources.items():
        archive_sha, archive_size = _sha256_file(source.archive)
        source_payload[source_key] = {
            "binding": source.binding.to_dict(),
            "root": _relative_path(root, source.root),
            "archive": {
                "path": _relative_path(root, source.archive),
                "sha256": archive_sha,
                "size_bytes": archive_size,
            },
            "inventory": {
                "path": _relative_path(root, source.inventory_path),
                "canonical_sha256": _canonical_sha256(source.inventory),
            },
            "snapshot_descriptor_path": _relative_path(root, source.descriptor_path),
            "patch_path": _relative_path(root, source.patch_path),
        }
    probe_root = root / "probes"
    probe_root.mkdir()
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        _write_canonical(probe_root / f"{candidate_id}.json", probes[candidate_id])
    candidate_payload: dict[str, Any] = {}
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        config = configurations[candidate_id]
        invocation = invocation_index[candidate_id]
        probe = probes[candidate_id]
        candidate_payload[candidate_id] = {
            "source_key": invocation.source_key,
            "configuration": {
                "binding": config.binding.to_dict(),
                "original_path": _relative_path(root, config.original),
                "derived_path": _relative_path(root, config.derived),
            },
            "probe": {
                "path": f"probes/{candidate_id}.json",
                "sha256": _canonical_sha256(probe),
                "stderr_sha256": probe_stderr[candidate_id],
            },
            "effective_seed_proof_sha256": _canonical_sha256(
                cast(Mapping[str, Any], probe["seed_resolution"])
            ),
            "resources": dict(cast(Mapping[str, int], probe["resources"])),
            "resource_supplement": dict(
                cast(Mapping[str, Any], probe["resource_supplement"])
            ),
            "capability_receipt": {
                "path": f"receipts/{candidate_id}.json",
                "sha256": receipt_digests[candidate_id],
            },
            "entrypoint": {
                "path": invocation.entrypoint_path,
                "sha256": invocation.entrypoint_sha256,
                "python_import_root": "." if invocation.source_key == "alberta" else "src",
                "invocation_style": invocation.invocation_style,
                "result_root": invocation.result_root,
            },
        }
    manifest = {
        "schema_version": MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "structurally_qualified_external_trust_resolution_required",
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "authority": {
            "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
        },
        "reward_blind_boundary": {
            "qualification_seed": PUBLIC_QUALIFICATION_SEED,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "tuning_seeds_used": [],
            "evaluation_seeds_used": [],
            "environment_resets": len(builder.MATCHED_CURRENT_CANDIDATE_IDS),
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
        },
        "runtime_qualification": dataclasses.asdict(runtime),
        "qualification_probe": {
            "source_key": "alberta",
            "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
            "sha256": invocations[0].probe_sha256,
        },
        "resource_accounting_semantics": _plain_json(_RESOURCE_ACCOUNTING_SEMANTICS),
        "executor_qualification_roots": {
            "cpu": {
                "path": _relative_path(root, executor_qualifications.cpu_root),
                "inventory": _plain_json(executor_qualifications.cpu_inventory),
                "inventory_sha256": _canonical_sha256(
                    executor_qualifications.cpu_inventory
                ),
            },
            "rng_parity": {
                "path": _relative_path(root, executor_qualifications.rng_root),
                "inventory": _plain_json(executor_qualifications.rng_inventory),
                "inventory_sha256": _canonical_sha256(
                    executor_qualifications.rng_inventory
                ),
            },
        },
        "frozen_executor_qualification_artifacts": dict(qualification_artifacts),
        "candidate_order": list(builder.MATCHED_CURRENT_CANDIDATE_IDS),
        "sources": source_payload,
        "candidates": candidate_payload,
        "open_protocol_sha256": final_protocol.protocol_sha256,
    }
    manifest_path = root / "manifest.json"
    manifest_sha = _write_canonical(manifest_path, manifest)
    (root / "manifest.json.sha256").write_text(f"{manifest_sha}\n", encoding="ascii")


def _directory_inode(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _directory_descriptor_matches_path(
    descriptor: int,
    path: Path,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and not path.is_symlink()
        and _directory_inode(opened) == expected_identity
        and _directory_inode(current) == expected_identity
    )


def _directory_entry_matches_descriptor(
    parent_descriptor: int,
    entry_name: str,
    held_descriptor: int,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        entry = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        held = os.fstat(held_descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry.st_mode)
        and stat.S_ISDIR(held.st_mode)
        and _directory_inode(entry) == expected_identity
        and _directory_inode(held) == expected_identity
    )


def _publish_directory_no_replace[PublishResult](
    source: Path,
    destination: Path,
    parent: Path,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_source_identity: tuple[int, int] | None = None,
    post_publish_validator: Callable[[Path], PublishResult] | None = None,
) -> PublishResult | None:
    """Atomically publish one sibling directory and replay it under held inodes."""
    if source.parent != parent or destination.parent != parent:
        raise ForagerMatchedQualificationError("qualification publication is not sibling-local")
    for name, label in (
        (source.name, "qualification staging directory"),
        (destination.name, "qualification destination directory"),
    ):
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise ForagerMatchedQualificationError(f"{label} name is unsafe")
    if source.name == destination.name:
        raise ForagerMatchedQualificationError(
            "qualification staging and destination names must differ"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = -1
    staging_fd = -1
    published_fd = -1
    published = False
    try:
        parent_path_metadata = parent.lstat()
        source_path_metadata = source.lstat()
        if not stat.S_ISDIR(parent_path_metadata.st_mode) or parent.is_symlink():
            raise ForagerMatchedQualificationError(
                "qualification output parent changed before publication"
            )
        if not stat.S_ISDIR(source_path_metadata.st_mode) or source.is_symlink():
            raise ForagerMatchedQualificationError(
                "qualification staging directory changed before publication"
            )
        parent_identity = (
            _directory_inode(parent_path_metadata)
            if expected_parent_identity is None
            else expected_parent_identity
        )
        source_identity = (
            _directory_inode(source_path_metadata)
            if expected_source_identity is None
            else expected_source_identity
        )
        if _directory_inode(parent_path_metadata) != parent_identity:
            raise ForagerMatchedQualificationError(
                "qualification output parent changed before publication"
            )
        if _directory_inode(source_path_metadata) != source_identity:
            raise ForagerMatchedQualificationError(
                "qualification staging directory changed before publication"
            )
        parent_fd = os.open(parent, directory_flags)
    except OSError as exc:
        raise ForagerMatchedQualificationError(
            "cannot safely open qualification output parent"
        ) from exc
    try:
        if not _directory_descriptor_matches_path(
            parent_fd,
            parent,
            parent_identity,
        ):
            raise ForagerMatchedQualificationError(
                "qualification output parent changed before publication"
            )
        try:
            staging_fd = os.open(source.name, directory_flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ForagerMatchedQualificationError(
                "cannot safely open qualification staging directory"
            ) from exc
        if not _directory_entry_matches_descriptor(
            parent_fd,
            source.name,
            staging_fd,
            source_identity,
        ):
            raise ForagerMatchedQualificationError(
                "qualification staging directory changed before publication"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise ForagerMatchedQualificationError(
                "atomic no-replace publication is unavailable on this platform"
            ) from exc
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(source.name),
            parent_fd,
            os.fsencode(destination.name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise ForagerMatchedQualificationError(
                    "qualification output root was created concurrently"
                )
            raise ForagerMatchedQualificationError(
                f"atomic qualification publication failed with errno {error}"
            )
        published = True
        os.fsync(parent_fd)
        if not _directory_descriptor_matches_path(
            parent_fd,
            parent,
            parent_identity,
        ):
            raise ForagerMatchedQualificationError(
                "qualification output parent changed after publication"
            )
        try:
            published_fd = os.open(destination.name, directory_flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ForagerMatchedQualificationError(
                "cannot safely open published qualification directory"
            ) from exc
        if not _directory_entry_matches_descriptor(
            parent_fd,
            destination.name,
            staging_fd,
            source_identity,
        ) or not _directory_entry_matches_descriptor(
            parent_fd,
            destination.name,
            published_fd,
            source_identity,
        ):
            raise ForagerMatchedQualificationError(
                "published qualification differs from the held staging inode"
            )
        try:
            os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForagerMatchedQualificationError(
                "qualification staging name was recreated after publication"
            )
        result = (
            None
            if post_publish_validator is None
            else post_publish_validator(destination)
        )
        if not _directory_descriptor_matches_path(
            parent_fd,
            parent,
            parent_identity,
        ) or not _directory_entry_matches_descriptor(
            parent_fd,
            destination.name,
            published_fd,
            source_identity,
        ):
            raise ForagerMatchedQualificationError(
                "published qualification changed during replay verification"
            )
        return result
    except BaseException as exc:
        if published:
            raise QualificationPublishedButUncertainError(
                f"qualification was published at {destination} but durability or replay "
                "verification is uncertain; do not reuse this output root"
            ) from exc
        raise
    finally:
        if published_fd >= 0:
            os.close(published_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either canonical path contains the other."""
    try:
        first.relative_to(second)
    except ValueError:
        try:
            second.relative_to(first)
        except ValueError:
            return False
    return True


def _reject_qualification_output_overlap(
    output_root: Path,
    staged_inputs: Mapping[str, Path],
) -> Path:
    """Reject a future output tree that could contain or be contained by an input."""
    try:
        canonical_output = output_root.resolve(strict=False)
    except OSError as exc:
        raise ForagerMatchedQualificationError(
            "cannot resolve the prospective qualification output root"
        ) from exc
    for label, raw_input in staged_inputs.items():
        canonical_input = _regular_directory(raw_input, label)
        if _paths_overlap(canonical_output, canonical_input):
            raise ForagerMatchedQualificationError(
                f"qualification output root overlaps {label}"
            )
    return canonical_output


def _fresh_snapshot_replay_command(
    runtime: Path,
    qualification_root: Path,
    *,
    qualification_module_sha256: str,
) -> list[str]:
    """Build the exact sandboxed OCI command for a fresh snapshot replay."""
    qualification_root = _regular_directory(
        qualification_root,
        "fresh replay qualification root",
    )
    if (
        type(qualification_module_sha256) is not str
        or _SHA256.fullmatch(qualification_module_sha256) is None
    ):
        raise ForagerMatchedQualificationError(
            "fresh replay qualification module digest is invalid"
        )
    if stat.S_IMODE(qualification_root.lstat().st_mode) != 0o755:
        raise ForagerMatchedQualificationError(
            "fresh replay qualification root is not OCI-readable"
        )
    root_value = qualification_root.as_posix()
    if any(character in root_value for character in (",", "\n", "\r", "\x00")):
        raise ForagerMatchedQualificationError(
            "fresh replay OCI mount path contains an unsafe character"
        )
    return [
        runtime.as_posix(),
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--cpus=2.0",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=256",
        "--tmpfs=/run/alberta:rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700",
        (
            "--mount=type=bind,"
            f"source={root_value},"
            f"destination={_CONTAINER_BUNDLE_ROOT},readonly"
        ),
        "--env=HOME=/run/alberta",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=ALL_PROXY=",
        "--env=HTTP_PROXY=",
        "--env=HTTPS_PROXY=",
        "--env=all_proxy=",
        "--env=http_proxy=",
        "--env=https_proxy=",
        "--env=LD_LIBRARY_PATH=",
        "--env=LD_PRELOAD=",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=NO_PROXY=",
        "--env=no_proxy=",
        "--env=PYTHONPATH=",
        "--env=TMPDIR=/run/alberta",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta",
        f"--workdir={_CONTAINER_BUNDLE_ROOT}",
        f"sha256:{_QUALIFIED_IMAGE_SHA256}",
        _QUALIFIED_PYTHON,
        "-I",
        "-B",
        "-c",
        _FRESH_SNAPSHOT_REPLAY_SCRIPT,
        _CONTAINER_REPLAY_SOURCE_ROOT,
        _CONTAINER_BUNDLE_ROOT,
        qualification_module_sha256,
    ]


def _run_fresh_snapshot_replay(
    qualification_root: Path,
    runtime: _ProbeRuntimeIdentity,
    runner: QualificationRunner,
    *,
    expected_manifest_sha256: str,
    expected_protocol_sha256: str,
    expected_plan_sha256: str,
    expected_qualification_module_sha256: str,
) -> None:
    """Replay staged loader/protocol/plan semantics inside the bound OCI image."""
    if type(runtime) is not _ProbeRuntimeIdentity:
        raise TypeError("fresh replay runtime must be a _ProbeRuntimeIdentity")
    qualification_root = _regular_directory(
        qualification_root,
        "fresh replay qualification root",
    )
    expected = {
        "manifest_sha256": expected_manifest_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "plan_sha256": expected_plan_sha256,
        "qualification_module_sha256": expected_qualification_module_sha256,
    }
    if any(
        type(value) is not str or _SHA256.fullmatch(value) is None
        for value in expected.values()
    ):
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay expected digests are invalid"
        )
    module_relative = "alberta_framework/benchmarks/forager_matched_qualification.py"
    source_root = _regular_directory(
        qualification_root / "sources" / "alberta" / "source",
        "fresh replay source root",
    )
    module_path = source_root.joinpath(*PurePosixPath(module_relative).parts)
    module_sha256, _module_size = _sha256_file(
        module_path,
        maximum=_MAX_SOURCE_BYTES,
    )
    if module_sha256 != expected_qualification_module_sha256:
        raise ForagerMatchedQualificationError(
            "fresh replay qualification module differs from its trusted binding"
        )
    completed = _invoke_qualification_runner(
        runner,
        _fresh_snapshot_replay_command(
            runtime.executable,
            qualification_root,
            qualification_module_sha256=module_sha256,
        ),
        label="fresh-process staged replay",
    )
    if type(completed) is not QualificationProcessResult:
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay runner returned the wrong result type"
        )
    stdout_raw = completed.stdout
    stderr_raw = completed.stderr
    if len(stdout_raw) > _MAX_JSON_BYTES or len(stderr_raw) > _MAX_PROBE_OUTPUT_BYTES:
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay output exceeds its bound"
        )
    if completed.returncode != 0:
        detail = stderr_raw[-4096:].decode("utf-8", errors="replace")
        raise ForagerMatchedQualificationError(
            f"fresh-process staged replay failed: {detail}"
        )
    if stderr_raw:
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay wrote unexpected stderr"
        )
    value = _decode_json(stdout_raw, "fresh-process staged replay")
    if type(value) is not dict or stdout_raw != _canonical_json_bytes(value):
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay output is not canonical JSON"
        )
    payload = cast(dict[str, Any], value)
    if set(payload) != {
        "schema_version",
        "manifest_sha256",
        "protocol_sha256",
        "plan_sha256",
        "plan_qualification_manifest_sha256",
        "qualification_module_path",
        "qualification_module_sha256",
    }:
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay fields drifted"
        )
    if (
        payload["schema_version"] != _FRESH_SNAPSHOT_REPLAY_SCHEMA
        or payload["manifest_sha256"] != expected_manifest_sha256
        or payload["protocol_sha256"] != expected_protocol_sha256
        or payload["plan_sha256"] != expected_plan_sha256
        or payload["plan_qualification_manifest_sha256"]
        != expected_manifest_sha256
        or payload["qualification_module_path"] != module_relative
        or payload["qualification_module_sha256"] != module_sha256
    ):
        raise ForagerMatchedQualificationError(
            "fresh-process staged replay differs from the parent closure"
        )


def _verify_staged_bundle_in_fresh_process(
    bundle: MatchedCurrentQualificationBundle,
    runtime: _ProbeRuntimeIdentity,
    runner: QualificationRunner,
) -> _FreshReplayClosure:
    """Compare parent and sandboxed child closures, then re-load every byte."""
    if type(bundle) is not MatchedCurrentQualificationBundle:
        raise TypeError("staged bundle must be a MatchedCurrentQualificationBundle")
    if type(runtime) is not _ProbeRuntimeIdentity:
        raise TypeError("staged replay runtime must be a _ProbeRuntimeIdentity")
    root = _regular_directory(bundle.output_root, "fresh replay qualification root")
    root_identity = _directory_inode(root.lstat())
    protocol, plan = build_open_protocol_and_execution_plan(bundle)
    trusted_module_path = Path(__file__).resolve(strict=True)
    trusted_module_sha256, _trusted_size = _sha256_file(
        trusted_module_path,
        maximum=_MAX_SOURCE_BYTES,
    )
    staged_module = (
        root
        / "sources"
        / "alberta"
        / "source"
        / "alberta_framework"
        / "benchmarks"
        / "forager_matched_qualification.py"
    )
    staged_module_sha256, _staged_size = _sha256_file(
        staged_module,
        maximum=_MAX_SOURCE_BYTES,
    )
    if staged_module_sha256 != trusted_module_sha256:
        raise ForagerMatchedQualificationError(
            "staged qualification module differs from the loaded trusted verifier"
        )
    closure = _FreshReplayClosure(
        manifest_sha256=bundle.manifest_sha256,
        protocol_sha256=plan.protocol.protocol_sha256,
        plan_sha256=plan.plan_sha256,
        qualification_module_sha256=trusted_module_sha256,
    )
    if protocol.protocol_sha256 != closure.protocol_sha256:
        raise ForagerMatchedQualificationError(
            "parent protocol and execution-plan protocol binding differ"
        )
    _rebind_probe_runtime(runtime, runner)
    try:
        _run_fresh_snapshot_replay(
            root,
            runtime,
            runner,
            expected_manifest_sha256=closure.manifest_sha256,
            expected_protocol_sha256=closure.protocol_sha256,
            expected_plan_sha256=closure.plan_sha256,
            expected_qualification_module_sha256=(
                closure.qualification_module_sha256
            ),
        )
    finally:
        _rebind_probe_runtime(runtime, runner)
    if _directory_inode(root.lstat()) != root_identity:
        raise ForagerMatchedQualificationError(
            "qualification root changed during fresh-process replay"
        )
    rebound = load_matched_current_qualification_bundle(root)
    rebound_protocol, rebound_plan = build_open_protocol_and_execution_plan(rebound)
    if rebound_plan.protocol.protocol_sha256 != rebound_protocol.protocol_sha256:
        raise ForagerMatchedQualificationError(
            "reloaded protocol and execution-plan protocol binding differ"
        )
    rebound_closure = _FreshReplayClosure(
        manifest_sha256=rebound.manifest_sha256,
        protocol_sha256=rebound_protocol.protocol_sha256,
        plan_sha256=rebound_plan.plan_sha256,
        qualification_module_sha256=_sha256_file(
            staged_module,
            maximum=_MAX_SOURCE_BYTES,
        )[0],
    )
    if rebound_closure != closure:
        raise ForagerMatchedQualificationError(
            "qualification closure changed during fresh-process replay"
        )
    return closure


def qualify_matched_current_candidates(
    project_root: Path,
    upstream_checkout: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: QualificationRunner | None = None,
) -> MatchedCurrentQualificationBundle:
    """Create and verify one new content-only matched-current qualification root.

    The output path must not already exist.  Only public seed zero is sent to
    the structural probe.  The probe initializes model/state structure but
    executes zero environment transitions and never opens an NPZ/result file.
    """
    if not all(isinstance(path, Path) for path in (project_root, upstream_checkout, output_root)):
        raise TypeError("project_root, upstream_checkout, and output_root must be Paths")
    project_root = _regular_directory(project_root, "project root")
    upstream_checkout = _regular_directory(upstream_checkout, "upstream checkout")
    from alberta_framework.benchmarks import forager_matched_executor as executor

    prospective_output = _reject_qualification_output_overlap(
        output_root,
        {
            "Alberta source tree": project_root / "alberta_framework",
            "upstream checkout": upstream_checkout,
            "CPU qualification root": executor.DEFAULT_CPU_QUALIFICATION_ROOT,
            "RNG parity qualification root": executor.DEFAULT_RNG_PARITY_QUALIFICATION_ROOT,
        },
    )
    project_root = _bind_project_root(project_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_parent = _regular_directory(output_root.parent, "qualification output parent")
    final_root = output_parent / output_root.name
    if final_root.resolve(strict=False) != prospective_output:
        raise ForagerMatchedQualificationError(
            "qualification output path changed while its parent was prepared"
        )
    if final_root.exists() or final_root.is_symlink():
        raise ForagerMatchedQualificationError("qualification output root already exists")
    parent_metadata = output_parent.lstat()
    parent_identity = _directory_inode(parent_metadata)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.partial-", dir=output_parent))
    temporary_metadata = temporary.lstat()
    temporary_identity = _directory_inode(temporary_metadata)
    try:
        executor_qualifications = _stage_executor_qualification_roots(temporary)
        sources = _stage_sources(project_root, upstream_checkout, temporary)
        configurations = _materialize_configurations(project_root, sources, temporary)
        invocations = _probe_invocations(sources, configurations)
        active_runner = _default_runner if runner is None else runner
        runtime_identity = _bind_probe_runtime(runtime, active_runner)
        probes: dict[str, Mapping[str, Any]] = {}
        probe_stderr: dict[str, str] = {}
        for invocation in invocations:
            probe, stderr_sha = _run_bound_probe(
                runtime_identity,
                invocation,
                sources[invocation.source_key],
                active_runner,
            )
            probes[invocation.candidate_id] = probe
            probe_stderr[invocation.candidate_id] = stderr_sha
        _assemble_and_write(
            temporary,
            executor_qualifications,
            sources,
            configurations,
            invocations,
            probes,
            probe_stderr,
        )
        _normalize_qualification_tree_permissions(temporary)
        # Replay every generated binding before any final path becomes visible.
        staged_bundle = load_matched_current_qualification_bundle(temporary)
        staged_closure = _verify_staged_bundle_in_fresh_process(
            staged_bundle,
            runtime_identity,
            active_runner,
        )
        _durably_sync_verified_tree(temporary)

        def validate_published(path: Path) -> MatchedCurrentQualificationBundle:
            published = load_matched_current_qualification_bundle(path)
            published_closure = _verify_staged_bundle_in_fresh_process(
                published,
                runtime_identity,
                active_runner,
            )
            if published_closure != staged_closure:
                raise ForagerMatchedQualificationError(
                    "published qualification differs from its staged fresh replay"
                )
            return published

        published_bundle = _publish_directory_no_replace(
            temporary,
            final_root,
            output_parent,
            expected_parent_identity=parent_identity,
            expected_source_identity=temporary_identity,
            post_publish_validator=validate_published,
        )
    except QualificationPublishedButUncertainError:
        raise
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if published_bundle is None:
        raise AssertionError("qualification publication omitted its final replay")
    return published_bundle


def _bound_path(root: Path, value: Any, label: str, *, directory: bool) -> Path:
    if type(value) is not str:
        raise ForagerMatchedQualificationError(f"{label} path must be a string")
    relative = _safe_relative(value, label)
    path = root.joinpath(*relative.parts)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ForagerMatchedQualificationError(f"{label} is missing") from exc
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or path.is_symlink() or (not directory and metadata.st_nlink != 1):
        raise ForagerMatchedQualificationError(f"{label} has an unsafe file type")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ForagerMatchedQualificationError(f"{label} escapes its artifact root") from exc
    return path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedQualificationError(f"{label} must be a plain object")
    return cast(dict[str, Any], value)


def _json_exact_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_json_bytes({"value": left}) == _canonical_json_bytes({"value": right})
    except (OverflowError, RecursionError, TypeError, ValueError):
        return False


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = _read_stable(path, label, maximum=_MAX_JSON_BYTES)
    value = _mapping(_decode_json(raw, label), label)
    if raw != _canonical_json_bytes(value):
        raise ForagerMatchedQualificationError(f"{label} is not canonical JSON")
    return value


def _verify_qualification_artifact_tree(
    root: Path,
    executor_qualifications: _StagedExecutorQualifications,
    sources: Mapping[SourceKey, _StagedSource],
    configurations: Mapping[str, _MaterializedConfiguration],
    candidate_records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject every unreferenced file or directory outside bound source roots."""
    expected_files = {
        root / "manifest.json",
        root / "manifest.json.sha256",
    }
    source_roots = {
        *(source.root for source in sources.values()),
        executor_qualifications.cpu_root,
        executor_qualifications.rng_root,
    }
    for source in sources.values():
        expected_files.update({source.archive, source.inventory_path})
        if source.descriptor_path is not None:
            expected_files.add(source.descriptor_path)
        if source.patch_path is not None:
            expected_files.add(source.patch_path)
    for configuration in configurations.values():
        expected_files.update({configuration.original, configuration.derived})
    for record in candidate_records.values():
        probe = _mapping(record["probe"], "candidate probe binding")
        receipt = _mapping(record["capability_receipt"], "capability receipt binding")
        expected_files.add(
            root.joinpath(*_safe_relative(probe["path"], "probe path").parts)
        )
        expected_files.add(
            root.joinpath(*_safe_relative(receipt["path"], "receipt path").parts)
        )
    expected_directories = set(source_roots)
    for path in expected_files | source_roots:
        parent = path.parent
        while parent != root:
            try:
                parent.relative_to(root)
            except ValueError as exc:
                raise ForagerMatchedQualificationError(
                    "qualification manifest names an artifact outside its root"
                ) from exc
            expected_directories.add(parent)
            parent = parent.parent

    actual_files, actual_directories = _bounded_tree_walk(
        root,
        label="qualification artifact tree",
        limits=_TreeWalkLimits(
            files=_MAX_QUALIFICATION_ARTIFACT_FILES,
            directories=_MAX_QUALIFICATION_ARTIFACT_DIRECTORIES,
            entries=_MAX_QUALIFICATION_ARTIFACT_ENTRIES,
            depth=_MAX_QUALIFICATION_ARTIFACT_DEPTH,
            bytes=_MAX_QUALIFICATION_ARTIFACT_BYTES,
        ),
        skip_descendants=frozenset(source_roots),
    )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ForagerMatchedQualificationError(
            "qualification artifact tree contains missing or unreferenced entries"
        )


def _parse_source_binding(value: Any) -> Any:
    from alberta_framework.benchmarks.forager_matched_protocol import SourceBinding

    payload = _mapping(value, "source binding")
    expected = {
        "provenance_kind",
        "repository",
        "base_commit",
        "tree_git_sha1",
        "archive_sha256",
        "inventory_sha256",
        "snapshot_descriptor_sha256",
    }
    if set(payload) != expected:
        raise ForagerMatchedQualificationError("source binding fields drifted")
    return SourceBinding(**payload)


def _parse_configuration_binding(value: Any) -> Any:
    from alberta_framework.benchmarks.forager_matched_protocol import (
        AllowedTransform,
        ConfigurationBinding,
    )

    payload = _mapping(value, "configuration binding")
    if set(payload) != {
        "original_path",
        "original_sha256",
        "derived_sha256",
        "allowed_transforms",
    }:
        raise ForagerMatchedQualificationError("configuration binding fields drifted")
    raw_transforms = payload["allowed_transforms"]
    if type(raw_transforms) is not list:
        raise ForagerMatchedQualificationError("configuration transforms must be an array")
    transforms = []
    for raw_transform in raw_transforms:
        transform = _mapping(raw_transform, "configuration transform")
        if set(transform) != {"transform_type", "target", "value_type", "value"}:
            raise ForagerMatchedQualificationError("configuration transform fields drifted")
        transforms.append(AllowedTransform(**transform))
    return ConfigurationBinding(
        original_path=payload["original_path"],
        original_sha256=payload["original_sha256"],
        derived_sha256=payload["derived_sha256"],
        allowed_transforms=tuple(transforms),
    )


def _verify_probe_payload(
    payload: Mapping[str, Any],
    invocation: ProbeInvocation,
) -> None:
    """Re-verify a persisted probe payload field-by-field against its invocation.

    The verifier accepts only exact expected values — fixed key sets, frozen
    parser identities per implementation kind, required entrypoint literals,
    and a zero-transition, zero-reward-read boundary block — so a probe
    payload that drifted in any dimension fails closed rather than being
    partially trusted.
    """
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    expected_top_level = {
        "schema_version",
        "status",
        "candidate_id",
        "qualification_seed",
        "source_key",
        "qualification_probe",
        "configuration",
        "entrypoint",
        "implementation_kind",
        "resolved_agent",
        "result_root",
        "seed_resolution",
        "resources",
        "resource_supplement",
        "runtime",
        "reward_blind_boundary",
        "authority",
    }
    if set(payload) != expected_top_level:
        raise ForagerMatchedQualificationError("persisted probe fields drifted")
    if (
        payload["schema_version"] != MATCHED_CURRENT_PROBE_SCHEMA_VERSION
        or payload["status"] != "structurally_qualified_content_only"
        or payload["candidate_id"] != invocation.candidate_id
        or not _json_exact_equal(payload["qualification_seed"], PUBLIC_QUALIFICATION_SEED)
        or payload["source_key"] != invocation.source_key
        or payload["implementation_kind"] != invocation.implementation_kind
        or payload["resolved_agent"] != invocation.expected_agent
        or payload["result_root"] != invocation.result_root
    ):
        raise ForagerMatchedQualificationError("persisted probe identity drifted")

    qualification_probe = _mapping(payload["qualification_probe"], "qualification probe")
    if qualification_probe != {
        "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
        "sha256": invocation.probe_sha256,
    }:
        raise ForagerMatchedQualificationError("persisted qualification probe drifted")

    configuration = _mapping(payload["configuration"], "probe configuration")
    parser_identity = {
        "alberta_causal_map": "MatchedAlbertaWorkerConfiguration:CausalMapForagerConfig",
        "alberta_horde_actor_critic": (
            "MatchedAlbertaWorkerConfiguration:AlbertaForagerConfig"
        ),
        "alberta_rtu_rtrl": "MatchedAlbertaWorkerConfiguration:RTURTRLForagerConfig",
        "upstream_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_rtu_ppo_isolated_rng": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_ppo": "PyExpUtils.ExperimentModel+PPORegistry",
        "upstream_dqn_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_crelu": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_plain": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_dqn_redo_post_ln": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_drqn": "PyExpUtils.ExperimentModel+problem.registry",
        "upstream_search_oracle": "PyExpUtils.ExperimentModel+problem.registry",
    }[invocation.implementation_kind]
    if not _json_exact_equal(
        configuration,
        {
            "path": _CONTAINER_WORK_CONFIG,
            "sha256": invocation.configuration_sha256,
            "parser_identity": parser_identity,
            "round_trip_accepted": True,
        },
    ):
        raise ForagerMatchedQualificationError("persisted probe configuration drifted")

    required_literals = {
        "alberta_single_seed_v1": {
            "--configuration",
            "--seed",
            "--horizon",
            "--output-root",
        },
        "official_foragax_continuing_main_v4": {
            "-e",
            "--exp",
            "-i",
            "--idxs",
            "--max_steps",
            "--save_path",
        },
        "official_foragax_ppo_frozen_updates_v1": {
            "-e",
            "--exp",
            "-i",
            "--idxs",
            "--save_path",
        },
    }[invocation.invocation_style]
    entrypoint = _mapping(payload["entrypoint"], "probe entrypoint")
    if set(entrypoint) != {
        "family",
        "path",
        "sha256",
        "python_ast_parsed",
        "ast_node_count",
        "required_cli_literals",
        "required_cli_literals_present",
    }:
        raise ForagerMatchedQualificationError("persisted probe entrypoint fields drifted")
    if (
        entrypoint["family"] != invocation.entrypoint_family
        or entrypoint["path"] != invocation.entrypoint_path
        or entrypoint["sha256"] != invocation.entrypoint_sha256
        or entrypoint["python_ast_parsed"] is not True
        or type(entrypoint["ast_node_count"]) is not int
        or entrypoint["ast_node_count"] <= 0
        or entrypoint["required_cli_literals"] != sorted(required_literals)
        or entrypoint["required_cli_literals_present"] is not True
    ):
        raise ForagerMatchedQualificationError("persisted probe entrypoint drifted")

    seed_resolution = _mapping(payload["seed_resolution"], "probe seed resolution")
    if invocation.source_key == "upstream_rng_isolated":
        expected_agent_words = [2795197240, 2837457689]
        expected_agent_derivation = "fold_in_isolated_agent_namespace_v1"
    elif invocation.invocation_style == "official_foragax_ppo_frozen_updates_v1":
        expected_agent_words = [1797259609, 2579123966]
        expected_agent_derivation = "shared_root_post_reset_split_v1"
    else:
        expected_agent_words = [0, 0]
        expected_agent_derivation = "effective_seed_constructor_input_v1"
    if not _json_exact_equal(
        seed_resolution,
        {
            "candidate_id": invocation.candidate_id,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "requested_seed": PUBLIC_QUALIFICATION_SEED,
            "stored_seed": PUBLIC_QUALIFICATION_SEED,
            "offset": 0,
            "effective_seed": PUBLIC_QUALIFICATION_SEED,
            "transport": invocation.seed_transport,
            "prng_impl": "threefry2x32",
            "effective_seed_key_words": [0, 0],
            "agent_rng_provenance_derivation": expected_agent_derivation,
            "agent_rng_provenance_key_words": expected_agent_words,
            "environment_transition_count": 0,
            "reward_array_read_count": 0,
        },
    ):
        raise ForagerMatchedQualificationError("persisted probe seed resolution drifted")

    resources = _mapping(payload["resources"], "probe resources")
    if set(resources) != {
        "parameter_count",
        "optimizer_update_count",
        "replay_capacity_transitions",
        "recurrent_state_elements",
    } or any(type(item) is not int or item < 0 for item in resources.values()):
        raise ForagerMatchedQualificationError("persisted probe resources drifted")
    expected_optimizer_updates: dict[str, int] = {
        **{
            candidate_id: 0
            for candidate_id in (
                *builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS,
                "search_oracle",
            )
        },
        **{
            candidate_id: invocation.horizon
            for candidate_id in (
                *builder.MATCHED_CURRENT_HORDE_CANDIDATE_IDS,
                "alberta_rtu_h08_taylor",
            )
        },
        "external_dqn_ln": 124_920,
        "external_dqn_crelu": 124_920,
        "external_dqn_plain": 124_920,
        "external_dqn_redo": 124_915,
        "external_drqn_paper": 124_915,
    }
    expected_updates = expected_optimizer_updates.get(invocation.candidate_id)
    if expected_updates is not None and resources["optimizer_update_count"] != expected_updates:
        raise ForagerMatchedQualificationError("persisted optimizer schedule drifted")

    supplement = _mapping(payload["resource_supplement"], "probe resource supplement")
    if set(supplement) != {
        "fixed_substrate_parameter_count",
        "target_snapshot_parameter_count",
        "non_gradient_operations",
    }:
        raise ForagerMatchedQualificationError("persisted resource supplement fields drifted")
    if any(
        type(supplement[key]) is not int or supplement[key] < 0
        for key in ("fixed_substrate_parameter_count", "target_snapshot_parameter_count")
    ):
        raise ForagerMatchedQualificationError("persisted resource supplement drifted")
    operations = _mapping(
        supplement["non_gradient_operations"],
        "probe non-gradient operations",
    )
    if set(operations) != {
        "causal_nonparametric_transition_updates",
        "target_snapshot_refreshes",
        "redo_recycles",
    } or any(type(value) is not int or value < 0 for value in operations.values()):
        raise ForagerMatchedQualificationError("persisted non-gradient operations drifted")
    if invocation.candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS:
        if operations["causal_nonparametric_transition_updates"] != invocation.horizon:
            raise ForagerMatchedQualificationError("causal-map update accounting drifted")
    elif operations["causal_nonparametric_transition_updates"] != 0:
        raise ForagerMatchedQualificationError("non-causal candidate reports causal updates")

    runtime = _mapping(payload["runtime"], "probe runtime")
    if runtime != {
        "image_sha256": _QUALIFIED_IMAGE_SHA256,
        "python": "3.12.3",
        "jax_backend": "cpu",
        "device_platforms": ["cpu"],
    }:
        raise ForagerMatchedQualificationError("persisted probe runtime drifted")

    boundary = _mapping(payload["reward_blind_boundary"], "probe reward boundary")
    authority = _mapping(payload["authority"], "probe authority")
    if (
        not _json_exact_equal(
            boundary,
            {
                "environment_resets": 1,
                "environment_transitions": 0,
                "reward_arrays_read": 0,
                "result_archives_opened": 0,
                "benchmark_seeds_used": [],
            },
        )
        or not _json_exact_equal(
            authority,
            {
                "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
                "content_only": True,
                "externally_endorsed": False,
                "external_signature_created": False,
                "trust_profile_created": False,
                "promotion_authorized": False,
                "performance_claim": False,
            },
        )
    ):
        raise ForagerMatchedQualificationError(
            "persisted probe crossed its reward-blind authority boundary"
        )


def _verify_snapshot_descriptor(
    *,
    source_key: SourceKey,
    descriptor: Mapping[str, Any],
    source_root: Path,
    upstream_root: Path | None,
    patch_path: Path | None,
    archive_sha256: str,
    archive_size: int,
    inventory_sha256: str,
) -> None:
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    authority = {
        "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
        "content_only": True,
        "externally_endorsed": False,
        "promotion_authorized": False,
    }
    archive = {
        "format": "normalized_ustar_uid_gid_mtime_zero_v1",
        "sha256": archive_sha256,
        "size_bytes": archive_size,
    }
    if source_key == "alberta":
        if patch_path is not None:
            raise ForagerMatchedQualificationError("Alberta snapshot must not name an RNG patch")
        expected = {
            "schema_version": MATCHED_CURRENT_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
            "classification": "reviewed_snapshot_content_identity_only",
            "repository": builder.MATCHED_CURRENT_ALBERTA_REPOSITORY,
            "base_commit": builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT,
            "selection": {
                "root": "alberta_framework",
                "included": "all_regular_files",
                "excluded": ["__pycache__", "*.pyc", "*.pyo"],
            },
            "archive": archive,
            "normalized_inventory_sha256": inventory_sha256,
            "authority": authority,
        }
        if descriptor != expected:
            raise ForagerMatchedQualificationError("Alberta snapshot descriptor drifted")
        return
    if source_key != "upstream_rng_isolated":
        raise ForagerMatchedQualificationError("git-tree source must not have a descriptor")
    if upstream_root is None or patch_path is None:
        raise ForagerMatchedQualificationError("isolated source is missing its base or patch")
    if set(descriptor) != {
        "schema_version",
        "classification",
        "repository",
        "base_commit",
        "base_tree_git_sha1",
        "base_archive_sha256",
        "archive",
        "normalized_inventory_sha256",
        "derivation",
        "derivation_descriptor_sha256",
        "patch_sha256",
        "authority",
    }:
        raise ForagerMatchedQualificationError("isolated snapshot descriptor fields drifted")
    derivation = _mapping(descriptor["derivation"], "RNG-isolation derivation")
    expected_outer = {
        "schema_version": MATCHED_CURRENT_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
        "classification": "reviewed_snapshot_content_identity_only",
        "repository": builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY,
        "base_commit": _QUALIFIED_UPSTREAM_COMMIT,
        "base_tree_git_sha1": _QUALIFIED_UPSTREAM_TREE,
        "base_archive_sha256": _QUALIFIED_UPSTREAM_ARCHIVE_SHA256,
        "archive": archive,
        "normalized_inventory_sha256": inventory_sha256,
        "derivation_descriptor_sha256": _canonical_sha256(derivation),
        "patch_sha256": _QUALIFIED_RNG_PATCH_SHA256,
        "authority": authority,
    }
    if any(descriptor[key] != value for key, value in expected_outer.items()):
        raise ForagerMatchedQualificationError("isolated snapshot descriptor drifted")
    from alberta_framework.benchmarks import forager_rtu_ppo_rng_isolation as rng_patch

    upstream_source = _read_stable(
        upstream_root / rng_patch.UPSTREAM_SOURCE_PATH,
        "base RTU/PPO source",
        maximum=_MAX_SOURCE_BYTES,
    )
    isolated_source = _read_stable(
        source_root / rng_patch.UPSTREAM_SOURCE_PATH,
        "isolated RTU/PPO source",
        maximum=_MAX_SOURCE_BYTES,
    )
    patch = _read_stable(patch_path, "RNG-isolation patch", maximum=_MAX_JSON_BYTES)
    try:
        rng_patch.validate_isolated_rtu_ppo_source(
            upstream_source,
            isolated_source,
            patch,
            derivation,
        )
    except rng_patch.RTUPPORngIsolationError as exc:
        raise ForagerMatchedQualificationError(
            "isolated source does not replay from its exact audited patch"
        ) from exc


def load_matched_current_qualification_bundle(
    output_root: Path,
) -> MatchedCurrentQualificationBundle:
    """Load, rehash, and reconstruct an existing nonpromoting qualification."""
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    root = _regular_directory(output_root, "qualification output root")
    manifest_path = _bound_path(root, "manifest.json", "qualification manifest", directory=False)
    manifest_raw = _read_stable(manifest_path, "qualification manifest", maximum=_MAX_JSON_BYTES)
    manifest = _mapping(_decode_json(manifest_raw, "qualification manifest"), "manifest")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise ForagerMatchedQualificationError("qualification manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "classification",
        "status",
        "promotion_authorized",
        "performance_claim",
        "external_verification_required",
        "authority",
        "reward_blind_boundary",
        "runtime_qualification",
        "qualification_probe",
        "resource_accounting_semantics",
        "executor_qualification_roots",
        "frozen_executor_qualification_artifacts",
        "candidate_order",
        "sources",
        "candidates",
        "open_protocol_sha256",
    }:
        raise ForagerMatchedQualificationError("qualification manifest fields drifted")
    digest = hashlib.sha256(manifest_raw).hexdigest()
    sidecar = _read_stable(
        _bound_path(root, "manifest.json.sha256", "manifest digest", directory=False),
        "manifest digest",
        maximum=128,
    )
    if sidecar != f"{digest}\n".encode("ascii"):
        raise ForagerMatchedQualificationError("qualification manifest digest mismatch")
    boundary = _mapping(manifest.get("reward_blind_boundary"), "manifest reward boundary")
    authority = _mapping(manifest.get("authority"), "manifest authority")
    if (
        manifest.get("schema_version") != MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION
        or manifest.get("classification") != "content_only_unendorsed_nonpromoting"
        or manifest.get("status") != "structurally_qualified_external_trust_resolution_required"
        or manifest.get("promotion_authorized") is not False
        or manifest.get("performance_claim") is not False
        or manifest.get("external_verification_required") is not True
        or not _json_exact_equal(
            authority,
            {
                "identity": MATCHED_CURRENT_AUTHORITY_IDENTITY,
                "content_only": True,
                "externally_endorsed": False,
                "external_signature_created": False,
                "trust_profile_created": False,
            },
        )
        or not _json_exact_equal(
            boundary,
            {
                "qualification_seed": PUBLIC_QUALIFICATION_SEED,
                "qualification_seed_class": "public_nonbenchmark_seed",
                "tuning_seeds_used": [],
                "evaluation_seeds_used": [],
                "environment_resets": len(builder.MATCHED_CURRENT_CANDIDATE_IDS),
                "environment_transitions": 0,
                "reward_arrays_read": 0,
                "result_archives_opened": 0,
            },
        )
    ):
        raise ForagerMatchedQualificationError("qualification authority boundary drifted")
    candidate_order = manifest.get("candidate_order")
    if candidate_order != list(builder.MATCHED_CURRENT_CANDIDATE_IDS):
        raise ForagerMatchedQualificationError("qualification candidate order drifted")
    runtime_payload = _mapping(manifest.get("runtime_qualification"), "runtime qualification")
    runtime = _runtime_qualification()
    if runtime_payload != dataclasses.asdict(runtime):
        raise ForagerMatchedQualificationError("runtime qualification drifted")
    if manifest.get("resource_accounting_semantics") != _plain_json(
        _RESOURCE_ACCOUNTING_SEMANTICS
    ):
        raise ForagerMatchedQualificationError("resource accounting semantics drifted")
    executor_root_records = _mapping(
        manifest.get("executor_qualification_roots"),
        "executor qualification roots",
    )
    if set(executor_root_records) != {"cpu", "rng_parity"}:
        raise ForagerMatchedQualificationError("executor qualification root set drifted")

    loaded_executor_roots: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for key in ("cpu", "rng_parity"):
        record = _mapping(
            executor_root_records[key],
            f"executor qualification root {key}",
        )
        if set(record) != {"path", "inventory", "inventory_sha256"}:
            raise ForagerMatchedQualificationError(
                "executor qualification root fields drifted"
            )
        qualification_root = _bound_path(
            root,
            record["path"],
            f"executor qualification root {key}",
            directory=True,
        )
        _verify_tree_entry_bounds(
            qualification_root,
            f"executor qualification root {key}",
        )
        inventory = _mapping(
            record["inventory"],
            f"executor qualification inventory {key}",
        )
        if (
            _canonical_sha256(inventory) != record["inventory_sha256"]
            or executor.source_inventory(qualification_root) != inventory
        ):
            raise ForagerMatchedQualificationError(
                "executor qualification root inventory drifted"
            )
        loaded_executor_roots[key] = (qualification_root, inventory)
    executor_qualifications = _StagedExecutorQualifications(
        cpu_root=loaded_executor_roots["cpu"][0],
        rng_root=loaded_executor_roots["rng_parity"][0],
        cpu_inventory=loaded_executor_roots["cpu"][1],
        rng_inventory=loaded_executor_roots["rng_parity"][1],
    )
    frozen_artifacts = executor.load_executor_qualification_artifacts(
        cpu_root=executor_qualifications.cpu_root,
        rng_parity_root=executor_qualifications.rng_root,
    )
    if manifest.get("frozen_executor_qualification_artifacts") != _plain_json(frozen_artifacts):
        raise ForagerMatchedQualificationError("frozen executor qualification content drifted")

    raw_sources = _mapping(manifest.get("sources"), "sources")
    if set(raw_sources) != {"alberta", "upstream", "upstream_rng_isolated"}:
        raise ForagerMatchedQualificationError("source set drifted")
    sources: dict[SourceKey, _StagedSource] = {}
    for source_key in ("alberta", "upstream", "upstream_rng_isolated"):
        record = _mapping(raw_sources[source_key], f"source {source_key}")
        if set(record) != {
            "binding",
            "root",
            "archive",
            "inventory",
            "snapshot_descriptor_path",
            "patch_path",
        }:
            raise ForagerMatchedQualificationError("source manifest fields drifted")
        binding = _parse_source_binding(record["binding"])
        source_root = _bound_path(root, record["root"], "source root", directory=True)
        _verify_tree_entry_bounds(source_root, f"source root {source_key}")
        archive_record = _mapping(record["archive"], "source archive")
        if set(archive_record) != {"path", "sha256", "size_bytes"}:
            raise ForagerMatchedQualificationError("source archive fields drifted")
        archive_path = _bound_path(
            root,
            archive_record["path"],
            "source archive",
            directory=False,
        )
        archive_sha, archive_size = _sha256_file(archive_path)
        if (
            archive_sha != archive_record["sha256"]
            or archive_size != archive_record["size_bytes"]
            or archive_sha != binding.archive_sha256
        ):
            raise ForagerMatchedQualificationError("source archive binding drifted")
        inventory_record = _mapping(record["inventory"], "source inventory")
        if set(inventory_record) != {"path", "canonical_sha256"}:
            raise ForagerMatchedQualificationError("source inventory fields drifted")
        inventory_path = _bound_path(
            root,
            inventory_record["path"],
            "source inventory",
            directory=False,
        )
        inventory = _load_canonical(inventory_path, "source inventory")
        if _canonical_sha256(inventory) != inventory_record["canonical_sha256"]:
            raise ForagerMatchedQualificationError("detailed source inventory drifted")
        if executor.source_inventory(source_root) != inventory:
            raise ForagerMatchedQualificationError("source root differs from detailed inventory")
        if executor.source_inventory_sha256(source_root) != binding.inventory_sha256:
            raise ForagerMatchedQualificationError("source normalized inventory drifted")
        _verify_archive_root_binding(
            archive_path,
            source_root,
            inventory,
            binding.inventory_sha256,
        )
        descriptor_value = record["snapshot_descriptor_path"]
        descriptor_path = (
            None
            if descriptor_value is None
            else _bound_path(root, descriptor_value, "snapshot descriptor", directory=False)
        )
        descriptor: Mapping[str, Any] | None = None
        if descriptor_path is None:
            if binding.snapshot_descriptor_sha256 is not None:
                raise ForagerMatchedQualificationError("snapshot descriptor is missing")
        else:
            descriptor = _load_canonical(descriptor_path, "snapshot descriptor")
            if _canonical_sha256(descriptor) != binding.snapshot_descriptor_sha256:
                raise ForagerMatchedQualificationError("snapshot descriptor digest drifted")
        patch_value = record["patch_path"]
        patch_path = (
            None
            if patch_value is None
            else _bound_path(root, patch_value, "RNG patch", directory=False)
        )
        if patch_path is not None and _sha256_file(patch_path, maximum=_MAX_JSON_BYTES)[0] != (
            _QUALIFIED_RNG_PATCH_SHA256
        ):
            raise ForagerMatchedQualificationError("RNG patch bytes drifted")
        if source_key == "upstream":
            if descriptor is not None or patch_path is not None:
                raise ForagerMatchedQualificationError(
                    "unmodified upstream source must not name a descriptor or patch"
                )
        else:
            if descriptor is None:
                raise ForagerMatchedQualificationError("reviewed snapshot descriptor is missing")
            _verify_snapshot_descriptor(
                source_key=source_key,
                descriptor=descriptor,
                source_root=source_root,
                upstream_root=(
                    sources["upstream"].root
                    if source_key == "upstream_rng_isolated"
                    else None
                ),
                patch_path=patch_path,
                archive_sha256=archive_sha,
                archive_size=archive_size,
                inventory_sha256=binding.inventory_sha256,
            )
        sources[source_key] = _StagedSource(
            source_key,
            source_root,
            archive_path,
            inventory_path,
            inventory,
            binding,
            descriptor_path,
            patch_path,
        )

    raw_candidates = _mapping(manifest.get("candidates"), "candidates")
    if set(raw_candidates) != set(builder.MATCHED_CURRENT_CANDIDATE_IDS):
        raise ForagerMatchedQualificationError("candidate manifest membership drifted")
    configurations: dict[str, _MaterializedConfiguration] = {}
    candidate_records: dict[str, dict[str, Any]] = {}
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        record = _mapping(raw_candidates[candidate_id], f"candidate {candidate_id}")
        if set(record) != {
            "source_key",
            "configuration",
            "probe",
            "effective_seed_proof_sha256",
            "resources",
            "resource_supplement",
            "capability_receipt",
            "entrypoint",
        }:
            raise ForagerMatchedQualificationError("candidate manifest fields drifted")
        source_key = cast(SourceKey, record["source_key"])
        if source_key != _source_key_for_candidate(candidate_id):
            raise ForagerMatchedQualificationError("candidate source family drifted")
        config_record = _mapping(record["configuration"], "candidate configuration")
        if set(config_record) != {"binding", "original_path", "derived_path"}:
            raise ForagerMatchedQualificationError("candidate configuration fields drifted")
        config_binding = _parse_configuration_binding(config_record["binding"])
        original_path = _bound_path(
            root,
            config_record["original_path"],
            "original configuration",
            directory=False,
        )
        derived_path = _bound_path(
            root,
            config_record["derived_path"],
            "derived configuration",
            directory=False,
        )
        original = _read_stable(original_path, "original configuration")
        derived = _read_stable(derived_path, "derived configuration")
        if (
            hashlib.sha256(original).hexdigest() != config_binding.original_sha256
            or hashlib.sha256(derived).hexdigest() != config_binding.derived_sha256
            or _replace_integer_literals(original, config_binding.allowed_transforms) != derived
        ):
            raise ForagerMatchedQualificationError("candidate configuration bytes drifted")
        configurations[candidate_id] = _MaterializedConfiguration(
            candidate_id,
            original_path,
            derived_path,
            config_binding,
        )
        candidate_records[candidate_id] = record

    invocations = _probe_invocations(sources, configurations)
    invocation_index = {item.candidate_id: item for item in invocations}
    if tuple(invocation_index) != builder.MATCHED_CURRENT_CANDIDATE_IDS:
        raise ForagerMatchedQualificationError("persisted probe invocation order drifted")
    qualification_probe = _mapping(
        manifest.get("qualification_probe"),
        "qualification probe binding",
    )
    expected_probe_binding = {
        "source_key": "alberta",
        "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
        "sha256": invocations[0].probe_sha256,
    }
    if qualification_probe != expected_probe_binding or any(
        invocation.probe_path != invocations[0].probe_path
        or invocation.probe_sha256 != invocations[0].probe_sha256
        for invocation in invocations
    ):
        raise ForagerMatchedQualificationError("qualification probe binding drifted")
    probes: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    receipt_digests: dict[str, str] = {}
    assets: dict[str, Any] = {}
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        record = candidate_records[candidate_id]
        invocation = invocation_index[candidate_id]
        entrypoint = _mapping(record["entrypoint"], "candidate entrypoint")
        expected_entrypoint = {
            "path": invocation.entrypoint_path,
            "sha256": invocation.entrypoint_sha256,
            "python_import_root": "." if invocation.source_key == "alberta" else "src",
            "invocation_style": invocation.invocation_style,
            "result_root": invocation.result_root,
        }
        if entrypoint != expected_entrypoint:
            raise ForagerMatchedQualificationError("candidate entrypoint binding drifted")
        probe_record = _mapping(record["probe"], "candidate probe binding")
        if set(probe_record) != {"path", "sha256", "stderr_sha256"}:
            raise ForagerMatchedQualificationError("candidate probe binding fields drifted")
        if probe_record["stderr_sha256"] != hashlib.sha256(b"").hexdigest():
            raise ForagerMatchedQualificationError("candidate probe stderr digest drifted")
        probe_path = _bound_path(root, probe_record["path"], "candidate probe", directory=False)
        probe = _load_canonical(probe_path, "candidate probe")
        _verify_probe_payload(probe, invocation)
        source_entrypoint = sources[invocation.source_key].root.joinpath(
            *_safe_relative(invocation.entrypoint_path, "entrypoint path").parts
        )
        replayed_entrypoint_contract = _entrypoint_contract(
            source_entrypoint,
            invocation.invocation_style,
        )
        persisted_entrypoint = _mapping(probe["entrypoint"], "probe entrypoint")
        if any(
            persisted_entrypoint[key] != value
            for key, value in replayed_entrypoint_contract.items()
        ):
            raise ForagerMatchedQualificationError(
                "candidate probe entrypoint AST contract did not replay"
            )
        if _canonical_sha256(probe) != probe_record["sha256"]:
            raise ForagerMatchedQualificationError("candidate probe digest drifted")
        if record["effective_seed_proof_sha256"] != _canonical_sha256(
            _mapping(probe["seed_resolution"], "seed resolution")
        ):
            raise ForagerMatchedQualificationError("effective-seed proof drifted")
        if record["resources"] != probe["resources"]:
            raise ForagerMatchedQualificationError("candidate resource accounting drifted")
        if record["resource_supplement"] != probe["resource_supplement"]:
            raise ForagerMatchedQualificationError(
                "candidate supplemental resource accounting drifted"
            )
        probes[candidate_id] = probe
        receipt_record = _mapping(record["capability_receipt"], "receipt binding")
        if set(receipt_record) != {"path", "sha256"}:
            raise ForagerMatchedQualificationError("capability receipt binding fields drifted")
        receipt_path = _bound_path(
            root,
            receipt_record["path"],
            "capability receipt",
            directory=False,
        )
        receipt = _load_canonical(receipt_path, "capability receipt")
        receipt_sha = _canonical_sha256(receipt)
        if receipt_sha != receipt_record["sha256"]:
            raise ForagerMatchedQualificationError("capability receipt digest drifted")
        if receipt.get("qualification_trust_anchor_identity") != MATCHED_CURRENT_AUTHORITY_IDENTITY:
            raise ForagerMatchedQualificationError("receipt is not explicitly content-only")
        receipts[candidate_id] = receipt
        receipt_digests[candidate_id] = receipt_sha
    _verify_qualification_artifact_tree(
        root,
        executor_qualifications,
        sources,
        configurations,
        candidate_records,
    )
    qualifications = _candidate_qualifications(
        sources,
        configurations,
        probes,
        receipt_digests,
    )
    protocol = builder.build_forager_matched_open_protocol(
        runtime=runtime,
        candidate_qualifications=qualifications,
    )
    if protocol.protocol_sha256 != manifest.get("open_protocol_sha256"):
        raise ForagerMatchedQualificationError("open protocol closure digest drifted")
    for candidate_id in builder.MATCHED_CURRENT_CANDIDATE_IDS:
        loaded_receipt = receipts[candidate_id]
        candidate = protocol.candidate_index[candidate_id]
        expected_receipt = _capability_receipt(
            candidate,
            invocation_index[candidate_id],
        )
        if loaded_receipt != expected_receipt:
            raise ForagerMatchedQualificationError("capability receipt closure drifted")
        assets[candidate_id] = executor.CandidateExecutionAssets(
            candidate_id=candidate_id,
            source_root=sources[_source_key_for_candidate(candidate_id)].root,
            source_archive=sources[_source_key_for_candidate(candidate_id)].archive,
            source_inventory=sources[_source_key_for_candidate(candidate_id)].inventory,
            original_configuration=configurations[candidate_id].original,
            configuration=configurations[candidate_id].derived,
            capability_receipt=loaded_receipt,
        )
    return MatchedCurrentQualificationBundle(
        output_root=root,
        cpu_qualification_root=executor_qualifications.cpu_root,
        rng_parity_qualification_root=executor_qualifications.rng_root,
        runtime_qualification=runtime,
        candidate_qualifications=MappingProxyType(qualifications),
        candidate_assets=assets,
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        manifest_bytes=manifest_raw,
        manifest_sha256=digest,
    )


def build_open_protocol_and_execution_plan(
    bundle: MatchedCurrentQualificationBundle,
) -> tuple[Any, Any]:
    """Exercise the complete builder closure for a verified bundle."""
    from alberta_framework.benchmarks import forager_matched_executor as executor
    from alberta_framework.benchmarks import forager_matched_open_protocol as builder

    if type(bundle) is not MatchedCurrentQualificationBundle:
        raise TypeError("bundle must be a MatchedCurrentQualificationBundle")
    protocol = builder.build_forager_matched_open_protocol(
        runtime=bundle.runtime_qualification,
        candidate_qualifications=bundle.candidate_qualifications,
    )
    plan = executor.build_execution_plan(
        protocol,
        dict(bundle.candidate_assets),
        qualification_manifest_sha256=bundle.manifest_sha256,
        candidate_ids=builder.MATCHED_CURRENT_CANDIDATE_IDS,
        cpu_qualification_root=bundle.cpu_qualification_root,
        rng_parity_qualification_root=bundle.rng_parity_qualification_root,
    )
    return protocol, plan


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    qualify = subparsers.add_parser("qualify", allow_abbrev=False)
    qualify.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    qualify.add_argument("--upstream-checkout", type=Path, required=True)
    qualify.add_argument("--output-root", type=Path, required=True)
    qualify.add_argument("--runtime", default="docker")
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.operation == "qualify":
        bundle = qualify_matched_current_candidates(
            arguments.project_root,
            arguments.upstream_checkout,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    else:
        bundle = load_matched_current_qualification_bundle(arguments.output_root)
    summary = {
        "schema_version": "alberta.forager_matched_qualification_cli_result.v1",
        "status": "verified_content_only_unendorsed_nonpromoting",
        "output_root": bundle.output_root.as_posix(),
        "candidate_count": len(bundle.candidate_qualifications),
        "candidate_order": list(bundle.candidate_qualifications),
        "manifest_sha256": bundle.manifest_sha256,
        "promotion_authorized": False,
        "external_verification_required": True,
    }
    sys.stdout.buffer.write(_canonical_json_bytes(summary))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint, including the private in-container probe operation."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "container-probe":
            return _container_probe(arguments[1:])
        return _cli(arguments)
    except QualificationPublishedButUncertainError as exc:
        sys.stderr.write(f"matched Forager qualification: PUBLISHED-UNCERTAIN: {exc}\n")
        return 3
    except (
        ValueError,
        OSError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        sys.stderr.write(f"matched Forager qualification: {exc}\n")
        return 2


__all__ = [
    "MATCHED_CURRENT_AUTHORITY_IDENTITY",
    "MATCHED_CURRENT_PROBE_SCHEMA_VERSION",
    "MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION",
    "MatchedCurrentQualificationBundle",
    "ForagerMatchedQualificationError",
    "QualificationPublishedButUncertainError",
    "ProbeInvocation",
    "QualificationProcessResult",
    "build_open_protocol_and_execution_plan",
    "load_matched_current_qualification_bundle",
    "main",
    "qualify_matched_current_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
