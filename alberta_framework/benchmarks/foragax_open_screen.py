"""Strict OCI harness for frozen, nonpromoting Foragax FOV development screens.

The harness registers the immutable v1/v2 histories but executes only the two
frozen CPU-v3 open-development protocols in ``outputs/forager``. It executes
every candidate before producing an aggregate, preserves the upstream archives
and process logs, and treats an exact completed attempt as immutable on resume.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final, NoReturn, cast

import numpy as np
import numpy.typing as npt

# Schema lattice.  Two protocol families (baseline and stateful baseline) exist in three
# generations: v1 (the GPU-era originals), CPU v2 (the first CPU overlays), and CPU v3.
# All six stay in SUPPORTED_SCHEMAS so their pinned artifacts can be re-validated
# byte-for-byte, but v1 and CPU v2 are immutable history — only the two CPU-v3
# protocols (EXECUTABLE_CPU_SCHEMAS) may actually be executed by this harness.
BASELINE_SCHEMA: Final = "alberta.forager_fov_baseline_screening.v1"
STATEFUL_SCHEMA: Final = "alberta.forager_fov_stateful_baseline_screening.v1"
BASELINE_CPU_SCHEMA: Final = "alberta.forager_fov_baseline_screening_cpu.v2"
STATEFUL_CPU_SCHEMA: Final = "alberta.forager_fov_stateful_baseline_screening_cpu.v2"
BASELINE_CPU_V3_SCHEMA: Final = "alberta.forager_fov_baseline_screening_cpu.v3"
STATEFUL_CPU_V3_SCHEMA: Final = "alberta.forager_fov_stateful_baseline_screening_cpu.v3"
BASELINE_CPU_SCHEMAS: Final = frozenset({BASELINE_CPU_SCHEMA, BASELINE_CPU_V3_SCHEMA})
STATEFUL_CPU_SCHEMAS: Final = frozenset({STATEFUL_CPU_SCHEMA, STATEFUL_CPU_V3_SCHEMA})
CPU_V2_SCHEMAS: Final = frozenset({BASELINE_CPU_SCHEMA, STATEFUL_CPU_SCHEMA})
CPU_V3_SCHEMAS: Final = frozenset({BASELINE_CPU_V3_SCHEMA, STATEFUL_CPU_V3_SCHEMA})
CPU_SCHEMAS: Final = CPU_V2_SCHEMAS | CPU_V3_SCHEMAS
EXECUTABLE_CPU_SCHEMAS: Final = CPU_V3_SCHEMAS
SUPPORTED_SCHEMAS: Final = frozenset(
    {BASELINE_SCHEMA, STATEFUL_SCHEMA, *CPU_SCHEMAS}
)
# SHA-256 of each frozen PROTOCOL.json under its _KNOWN_PROTOCOL_ROOTS directory in
# outputs/forager.  load_frozen_protocol() rejects any protocol whose bytes do not hash
# to the registered value, so these pins ARE the freeze: they are never regenerated.
# Editing a protocol changes its digest and correctly invalidates it; a genuinely new
# protocol gets a new schema version, a new root, and a new row here.
_FROZEN_PROTOCOL_SHA256: Final[dict[str, str]] = {
    BASELINE_SCHEMA: "b94c906f9d3c1f2abf226d7049bd9305700d7f5a81012b07b36cc10d458d3174",
    STATEFUL_SCHEMA: "df80dfafbd3bfc8eca7f1e6eccadc0ba93da4a3ec8f12ebe2befd430a21915e8",
    BASELINE_CPU_SCHEMA: "d384a44dcf8161e8d7c521ea3fda7720118cff93047bbb1261d84f158388e606",
    STATEFUL_CPU_SCHEMA: "83bb51ae792e90d27fb75f0215dc5c07a6a064468305f141508fe3ca57b13731",
    BASELINE_CPU_V3_SCHEMA: "e5a0f0fbe3fc9cd7245abe01a6a177eea030b7b533e6d85992b88b1b91c11dd0",
    STATEFUL_CPU_V3_SCHEMA: "a7cbca5735341ad580f09d705116f7131633b9cbe2494ef3bdc2d5ab6073c34d",
}
PLAN_SCHEMA: Final = "alberta.foragax_open_development_screen_plan.v4"
RUN_SCHEMA: Final = "alberta.foragax_open_development_screen_run.v4"
AGGREGATE_SCHEMA: Final = "alberta.foragax_open_development_screen_aggregate.v4"
ATTEMPT_SCHEMA: Final = "alberta.foragax_open_development_screen_attempt.v4"
SCORING_SCHEMA: Final = "alberta.foragax_open_screen_scoring.v2"
INPUT_SNAPSHOT_SCHEMA: Final = "alberta.foragax_open_screen_inputs.v3"
HARNESS_VERSION: Final = 4
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONTAINER_SOURCE_ROOT: Final = "/opt/foragax-agents"
_CONTAINER_PROTOCOL_ROOT: Final = "/protocol"
_PROBE_CONTAINER_PATH: Final = "/harness/preflight.py"
_SCORER_CONTAINER_PATH: Final = "/harness/scorer.py"
_REFERENCE_SCORER_CONTAINER_PATH: Final = "/harness/reference_scorer.py"
_EXPECTED_IMAGE_ENTRYPOINT: Final = [
    "/opt/foragax-agents/.venv/bin/python",
    "-I",
]
_EXPECTED_IMAGE_WORKDIR: Final = "/opt/foragax-agents"
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_KNOWN_PINNED_OUTPUTS: Final = (
    "outputs/ftl_decision",
    "outputs/continual_ia",
    "outputs/recurring_feature",
    "outputs/scale_robust_feature",
    "outputs/continual_multiagent",
)
_KNOWN_PROTOCOL_ROOTS: Final = (
    "outputs/forager/fov_baseline_screening_v1",
    "outputs/forager/fov_stateful_baseline_screening_v1",
    "outputs/forager/fov_baseline_screening_cpu_v2",
    "outputs/forager/fov_stateful_baseline_screening_cpu_v2",
    "outputs/forager/fov_baseline_screening_cpu_v3",
    "outputs/forager/fov_stateful_baseline_screening_cpu_v3",
)
_MAX_NPZ_BYTES: Final = 64 * 1024**2
_MAX_ZIP_MEMBERS: Final = 512
_MAX_UNCOMPRESSED_BYTES: Final = 64 * 1024**2
_MAX_REWARD_MEMBER_BYTES: Final = 4 * 1024**2
# Container environment pinned for determinism: force JAX/XLA onto CPU and hide every
# GPU path (empty CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES=void), disable XLA client
# preallocation, and fix PYTHONHASHSEED so hash-order-dependent iteration in candidate
# code cannot vary between runs.
_CPU_V2_ENVIRONMENT: Final[dict[str, str]] = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORM_NAME": "cpu",
    "JAX_PLATFORMS": "cpu",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "PYTHONHASHSEED": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
# v3 additionally pins the matplotlib and numba cache directories inside the writable
# tmpfs, so library caches land in the ephemeral mount instead of attempting writes to
# the read-only root; the preflight contract below requires both directories to exist.
_CPU_V3_ENVIRONMENT: Final[dict[str, str]] = {
    **_CPU_V2_ENVIRONMENT,
    "MPLCONFIGDIR": "/tmp/alberta-matplotlib-cache",
    "NUMBA_CACHE_DIR": "/tmp/alberta-numba-cache",
}
_CPU_V3_PREFLIGHT_CONTRACT: Final[dict[str, Any]] = {
    "mode": "exact_entrypoint_help_before_experiment_construction",
    "arguments": ["--help"],
    "entrypoints_from_frozen_configurations": True,
    "environment_or_transition_construction_allowed": False,
    "required_cache_directories": {
        "MPLCONFIGDIR": "/tmp/alberta-matplotlib-cache",
        "NUMBA_CACHE_DIR": "/tmp/alberta-numba-cache",
    },
}
# Sandbox contract recorded in every plan and receipt: no network, read-only root,
# all capabilities dropped, no-new-privileges, non-root uid/gid 65532 (the conventional
# distroless "nonroot" user), a pids limit as a fork-bomb guard, and one
# noexec/nosuid/nodev tmpfs as the only writable path.
_CPU_PROTOCOL_SANDBOX: Final[dict[str, Any]] = {
    "capabilities": "all_dropped",
    "container_user": "65532:65532",
    "host_devices": [],
    "network": "none",
    "no_new_privileges": True,
    "pids_limit": 512,
    "root_filesystem": "read_only",
    "tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=8g,mode=1777",
}


def _cpu_environment(schema: str) -> dict[str, str]:
    if schema in CPU_V3_SCHEMAS:
        return dict(_CPU_V3_ENVIRONMENT)
    if schema in CPU_V2_SCHEMAS:
        return dict(_CPU_V2_ENVIRONMENT)
    raise ScreenError(f"schema has no CPU environment contract: {schema}")


def _sandbox_contract(protocol: FrozenProtocol) -> dict[str, Any]:
    if protocol.schema not in CPU_SCHEMAS:
        raise ScreenError("only CPU overlay protocols have a sandbox contract")
    return {
        "backend": "cpu",
        "network": "none",
        "root_filesystem": "read_only",
        "container_user": "65532:65532",
        "capabilities": "all_dropped",
        "no_new_privileges": True,
        "pids_limit": 512,
        "tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=8g,mode=1777",
        "host_devices": [],
        "protocol_mount": "read_only:/protocol",
        "result_mount": "one candidate-specific writable bind:/run-output",
        "environment": _cpu_environment(protocol.schema),
    }


class ScreenError(RuntimeError):
    """Raised when an input or artifact violates the frozen screen contract."""


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ScreenError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise ScreenError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class FrozenConfiguration:
    """One hash-bound candidate from a frozen protocol."""

    path: str
    sha256: str
    agent: str
    entrypoint: str

    def __post_init__(self) -> None:
        _require_string(self.path, "path")
        _require_sha256(self.sha256, "sha256")
        _require_string(self.agent, "agent")
        _require_string(self.entrypoint, "entrypoint")

    @property
    def run_id(self) -> str:
        stem = PurePosixPath(self.path).stem
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
        if not safe:
            raise ScreenError(f"configuration path has no safe stem: {self.path}")
        return f"{safe}-{self.sha256[:12]}"


@dataclass(frozen=True)
class FrozenProtocol:
    """Validated protocol fields needed by the execution harness."""

    root: Path
    configuration_root: Path
    raw: dict[str, Any]
    raw_bytes: bytes
    sha256: str
    schema: str
    horizon: int
    seeds: tuple[int, ...]
    index_argument: str
    image_reference: str
    image_id: str
    backend: str
    scorer_path: Path | None
    scorer_sha256: str | None
    reference_scorer_path: Path | None
    reference_scorer_sha256: str | None
    base_protocol_sha256: str | None
    predecessor_protocol_root: Path | None
    predecessor_protocol_sha256: str | None
    harness_snapshot_path: Path | None
    probe_snapshot_path: Path | None
    source_files: tuple[tuple[str, str], ...]
    configurations: tuple[FrozenConfiguration, ...]
    metric: dict[str, Any]
    selection_rule: dict[str, Any]
    advance_count: int


@dataclass(frozen=True)
class ProtocolSnapshot:
    """Read-only execution copy of every mutable host protocol input."""

    protocol: FrozenProtocol
    inventory: tuple[dict[str, Any], ...]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if type(self.protocol) is not FrozenProtocol:
            raise ScreenError("protocol must be a FrozenProtocol")
        if type(self.inventory) is not tuple:
            raise ScreenError("inventory must be a tuple")
        if any(type(item) is not dict for item in self.inventory):
            raise ScreenError("inventory must contain exact dictionaries")
        _require_sha256(self.inventory_sha256, "inventory_sha256")


@dataclass(frozen=True)
class ProcessCapture:
    """Captured result from a small host-side subprocess."""

    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise ScreenError("returncode must be an integer")
        if type(self.stdout) is not bytes:
            raise ScreenError("stdout must be bytes")
        if type(self.stderr) is not bytes:
            raise ScreenError("stderr must be bytes")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScreenError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> NoReturn:
    raise ScreenError(f"JSON contains non-finite number {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ScreenError(f"JSON contains non-finite number {value!r}")
    return parsed


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_number,
            parse_float=_parse_finite_json_float,
        )
    except ScreenError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError(f"{label} is not valid duplicate-free UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ScreenError(f"{label} must contain a JSON object")
    return cast(dict[str, Any], value)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ScreenError("manifest contains a non-canonical JSON value") from error
    return (encoded + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_file_record(
    path: Path,
    label: str,
    *,
    maximum: int | None = None,
) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScreenError(f"cannot safely open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (maximum is not None and before.st_size > maximum)
        ):
            raise ScreenError(f"{label} must be a bounded single-link regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise ScreenError(f"{label} exceeds its byte bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ScreenError(f"{label} changed while it was hashed")
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ScreenError(f"missing {label}: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ScreenError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_stable_regular_file(
    path: Path,
    label: str,
    *,
    maximum: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScreenError(f"cannot open {label} without following links: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (maximum is not None and before.st_size > maximum)
        ):
            raise ScreenError(f"{label} is not a bounded single-link regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise ScreenError(f"{label} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ScreenError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _normalized_relative_path(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ScreenError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ScreenError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise ScreenError(f"{label} must use normalized POSIX separators")
    return value


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ScreenError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _require_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ScreenError(f"{label} must be an array")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ScreenError(f"{label} must be a positive integer")
    return value


def _source_records(protocol: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    value = protocol.get("source_files")
    if value is None:
        value = _require_dict(protocol.get("implementation"), "implementation").get("source_files")
    records = _require_list(value, "source_files")
    if not records:
        raise ScreenError("source_files must not be empty")
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _require_dict(raw_record, f"source_files[{index}]")
        relative = _normalized_relative_path(record.get("path"), f"source_files[{index}].path")
        digest = _require_sha256(record.get("sha256"), f"source_files[{index}].sha256")
        if relative in seen:
            raise ScreenError(f"duplicate source file record: {relative}")
        seen.add(relative)
        resolved.append((relative, digest))
    return tuple(resolved)


def _runtime_identity(protocol: dict[str, Any], schema: str) -> tuple[str, str]:
    runtime = _require_dict(protocol.get("runtime"), "runtime")
    if schema == BASELINE_SCHEMA:
        reference_key = "intended_image_reference"
        id_key = "intended_image_id"
    else:
        reference_key = "development_image_reference"
        id_key = "development_image_id"
    reference = _require_string(runtime.get(reference_key), f"runtime.{reference_key}")
    image_id = _require_string(runtime.get(id_key), f"runtime.{id_key}")
    if _IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ScreenError(f"runtime.{id_key} must be an exact lowercase image ID")
    if runtime.get("qualified_production_image") is not False:
        raise ScreenError("this harness only accepts an explicitly unqualified development image")
    return reference, image_id


def _validate_metric(raw: dict[str, Any], horizon: int, schema: str) -> dict[str, Any]:
    metric = _require_dict(raw.get("metric"), "metric")
    expected_pairs: dict[str, Any] = {
        "name": "fov_last_10pct_ema_auc",
        "ema_decay": 0.999,
        "ema_initial_value": 0.0,
        "bias_correction": False,
        "subsample_every_steps": 100,
        "direction": "maximize",
    }
    for key, expected in expected_pairs.items():
        if metric.get(key) != expected:
            raise ScreenError(f"metric.{key} is not the supported frozen value {expected!r}")
    if metric.get("subsample_first_reward") is not True:
        raise ScreenError("metric.subsample_first_reward must be true")
    tail = metric.get("tail_fraction", metric.get("tail_fraction_of_sampled_curve"))
    if tail != 0.1:
        raise ScreenError("metric tail fraction must be exactly 0.1")
    sample_count = (horizon + 99) // 100
    tail_start = int(0.9 * sample_count)
    if schema == STATEFUL_SCHEMA:
        expected_shape = metric.get("reward_trace_shape")
        if expected_shape != [horizon]:
            raise ScreenError("stateful metric reward_trace_shape drift")
        if metric.get("sample_count") != sample_count:
            raise ScreenError("stateful metric sample_count drift")
        if metric.get("tail_start_index") != tail_start:
            raise ScreenError("stateful metric tail_start_index drift")
        if metric.get("tail_sample_count") != sample_count - tail_start:
            raise ScreenError("stateful metric tail_sample_count drift")
        implementation = _require_dict(metric.get("implementation"), "metric.implementation")
        implementation_path = _normalized_relative_path(
            implementation.get("path"), "metric.implementation.path"
        )
        implementation_file = _regular_file(
            Path(cast(str, raw["__protocol_root"])) / implementation_path,
            "metric implementation",
        )
        expected_hash = _require_sha256(
            implementation.get("sha256"), "metric.implementation.sha256"
        )
        if _stable_file_record(implementation_file, "metric implementation")[0] != expected_hash:
            raise ScreenError("metric implementation hash drift")
    return metric


def _validate_selection_rule(raw: dict[str, Any], schema: str) -> tuple[dict[str, Any], int]:
    rule = _require_dict(raw.get("selection_rule"), "selection_rule")
    if rule.get("advance_count") != 3:
        raise ScreenError("selection_rule.advance_count must remain exactly three")
    if schema == BASELINE_SCHEMA:
        if rule.get("statistic") != "mean_over_two_open_development_seeds":
            raise ScreenError("baseline selection statistic drift")
        if rule.get("tie_break") != "config_filename_ascending":
            raise ScreenError("baseline tie-break drift")
    else:
        if rule.get("aggregate_statistic") != (
            "arithmetic mean over the exact two open development seeds"
        ):
            raise ScreenError("stateful aggregate statistic drift")
        if rule.get("ranking") != "descending aggregate statistic":
            raise ScreenError("stateful ranking direction drift")
        if rule.get("tie_break") != ("configuration path ascending by Unicode code point"):
            raise ScreenError("stateful tie-break drift")
        if rule.get("frozen_before_reward_execution") is not True:
            raise ScreenError("stateful selection rule was not frozen before execution")
    return rule, 3


def _validate_config_payload(
    path: Path,
    expected_hash: str,
    relative: str,
    task: dict[str, Any],
    horizon: int,
    schema: str,
) -> tuple[str, str]:
    config_file = _regular_file(path, f"configuration {relative}")
    raw = _read_stable_regular_file(config_file, f"configuration {relative}")
    if _sha256_bytes(raw) != expected_hash:
        raise ScreenError(f"configuration hash drift: {relative}")
    config = _load_json_bytes(raw, f"configuration {relative}")
    if config.get("problem") != "Foragax":
        raise ScreenError(f"configuration problem drift: {relative}")
    if config.get("total_steps") != horizon:
        raise ScreenError(f"configuration horizon drift: {relative}")
    agent = _require_string(config.get("agent"), f"configuration {relative}.agent")
    hypers = _require_dict(config.get("metaParameters"), f"configuration {relative}.metaParameters")
    environment = _require_dict(
        hypers.get("environment"), f"configuration {relative}.metaParameters.environment"
    )
    if environment.get("env_id") != task.get("env_id"):
        raise ScreenError(f"configuration environment ID drift: {relative}")
    if environment.get("aperture_size") != task.get("aperture_size"):
        raise ScreenError(f"configuration aperture drift: {relative}")
    if schema == STATEFUL_SCHEMA and environment.get("observation_type") != "color":
        raise ScreenError(f"stateful configuration observation type drift: {relative}")
    experiment = _require_dict(
        hypers.get("experiment"), f"configuration {relative}.metaParameters.experiment"
    )
    if experiment.get("seed_offset", 0) != 0:
        raise ScreenError(f"configuration seed offset drift: {relative}")
    if agent.startswith("PPO"):
        rollout = hypers.get("rollout_steps")
        updates = hypers.get("num_updates")
        if type(rollout) is not int or not 1 <= rollout <= horizon:
            raise ScreenError(f"PPO rollout_steps is not explicit: {relative}")
        if type(updates) is not int or not 1 <= updates <= horizon:
            raise ScreenError(f"PPO num_updates is not explicit: {relative}")
        if rollout * updates != horizon:
            raise ScreenError(f"PPO schedule does not match horizon: {relative}")
        entrypoint = "src/rtu_ppo.py"
    else:
        entrypoint = "src/continuing_main.py"
    return agent, entrypoint


def _repository_path(value: Any, label: str, *, directory: bool) -> Path:
    relative = _normalized_relative_path(value, label)
    unresolved = _REPOSITORY_ROOT / relative
    if unresolved.is_symlink():
        raise ScreenError(f"{label} must not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(_REPOSITORY_ROOT)
    except (FileNotFoundError, ValueError) as error:
        raise ScreenError(f"{label} escapes or is absent from the repository") from error
    if directory:
        if not resolved.is_dir():
            raise ScreenError(f"{label} must be a directory")
    else:
        _regular_file(resolved, label)
    return resolved


def _load_cpu_protocol(
    *,
    root: Path,
    raw: dict[str, Any],
    raw_bytes: bytes,
    protocol_sha256: str,
    schema: str,
) -> FrozenProtocol:
    """Validate one frozen CPU overlay protocol against its lineage and contracts.

    A CPU overlay must chain to its frozen GPU-era base protocol by schema and
    digest; a v3 protocol must additionally supersede its exact v2 predecessor
    (and a v2 must not claim one).  The runtime block has to reproduce the
    module's frozen environment/sandbox contracts verbatim, the scoring block
    must pin the in-image scorer and the exact EMA arithmetic string, and the
    protocol must declare itself nonpromoting with SOTA claims forbidden —
    any drift fails closed before a plan can be built.
    """
    if raw.get("status") != "configuration_frozen_execution_pending":
        raise ScreenError("CPU protocol is not execution-pending")
    if raw.get("evidence_class") != "open_development":
        raise ScreenError("CPU protocol is not open-development")
    if (
        raw.get("scientific_promotion_allowed") is not False
        or raw.get("sota_claim_allowed") is not False
    ):
        raise ScreenError("CPU protocol must forbid promotion and SOTA claims")
    base_record = _require_dict(raw.get("base_protocol"), "base_protocol")
    base_root = _repository_path(
        base_record.get("repository_path"),
        "base_protocol.repository_path",
        directory=True,
    )
    base = load_frozen_protocol(base_root)
    expected_base_schema = (
        BASELINE_SCHEMA if schema in BASELINE_CPU_SCHEMAS else STATEFUL_SCHEMA
    )
    if (
        base.schema != expected_base_schema
        or base_record.get("schema_version") != expected_base_schema
        or base_record.get("sha256") != base.sha256
    ):
        raise ScreenError("CPU protocol base identity drift")

    predecessor_root: Path | None = None
    predecessor_sha256: str | None = None
    if schema in CPU_V3_SCHEMAS:
        if raw.get("preflight") != _CPU_V3_PREFLIGHT_CONTRACT:
            raise ScreenError("CPU v3 zero-transition preflight contract drift")
        predecessor_record = _require_dict(
            raw.get("supersedes_protocol"), "supersedes_protocol"
        )
        predecessor_root = _repository_path(
            predecessor_record.get("repository_path"),
            "supersedes_protocol.repository_path",
            directory=True,
        )
        predecessor = load_frozen_protocol(predecessor_root)
        expected_predecessor_schema = (
            BASELINE_CPU_SCHEMA
            if schema == BASELINE_CPU_V3_SCHEMA
            else STATEFUL_CPU_SCHEMA
        )
        if (
            predecessor.schema != expected_predecessor_schema
            or predecessor_record.get("schema_version") != expected_predecessor_schema
            or predecessor_record.get("sha256") != predecessor.sha256
        ):
            raise ScreenError("CPU v3 predecessor identity drift")
        predecessor_sha256 = predecessor.sha256
    elif "supersedes_protocol" in raw:
        raise ScreenError("CPU v2 protocol unexpectedly declares a predecessor")

    runtime = _require_dict(raw.get("runtime"), "runtime")
    if (
        runtime.get("backend") != "cpu"
        or runtime.get("qualified_production_image") is not False
        or runtime.get("environment") != _cpu_environment(schema)
        or runtime.get("sandbox") != _CPU_PROTOCOL_SANDBOX
    ):
        raise ScreenError("CPU protocol runtime/sandbox contract drift")
    image_reference = _require_string(
        runtime.get("development_image_reference"),
        "runtime.development_image_reference",
    )
    image_id = _require_string(
        runtime.get("development_image_id"),
        "runtime.development_image_id",
    )
    if _IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ScreenError("CPU protocol image ID is invalid")

    scoring = _require_dict(raw.get("scoring"), "scoring")
    if (
        scoring.get("runtime") != "exact_development_image"
        or scoring.get("arithmetic")
        != "EMA_DECAY * ema + (1.0 - EMA_DECAY) * reward"
    ):
        raise ScreenError("CPU protocol scoring runtime/arithmetic drift")
    scorer_path = _repository_path(
        scoring.get("implementation_path"),
        "scoring.implementation_path",
        directory=False,
    )
    scorer_sha256 = _require_sha256(
        scoring.get("implementation_sha256"),
        "scoring.implementation_sha256",
    )
    if _stable_file_record(scorer_path, "CPU protocol scorer")[0] != scorer_sha256:
        raise ScreenError("CPU protocol scorer hash drift")
    reference_path: Path | None = None
    reference_sha256: str | None = None
    if schema in STATEFUL_CPU_SCHEMAS:
        reference_path = _repository_path(
            scoring.get("reference_implementation_path"),
            "scoring.reference_implementation_path",
            directory=False,
        )
        reference_sha256 = _require_sha256(
            scoring.get("reference_implementation_sha256"),
            "scoring.reference_implementation_sha256",
        )
        if (
            _stable_file_record(reference_path, "CPU protocol reference scorer")[0]
            != reference_sha256
        ):
            raise ScreenError("CPU protocol reference scorer hash drift")

    return FrozenProtocol(
        root=root,
        configuration_root=base.root,
        raw=raw,
        raw_bytes=raw_bytes,
        sha256=protocol_sha256,
        schema=schema,
        horizon=base.horizon,
        seeds=base.seeds,
        index_argument=base.index_argument,
        image_reference=image_reference,
        image_id=image_id,
        backend="cpu",
        scorer_path=scorer_path,
        scorer_sha256=scorer_sha256,
        reference_scorer_path=reference_path,
        reference_scorer_sha256=reference_sha256,
        base_protocol_sha256=base.sha256,
        predecessor_protocol_root=predecessor_root,
        predecessor_protocol_sha256=predecessor_sha256,
        harness_snapshot_path=None,
        probe_snapshot_path=None,
        source_files=base.source_files,
        configurations=base.configurations,
        metric=base.metric,
        selection_rule=base.selection_rule,
        advance_count=base.advance_count,
    )


def load_frozen_protocol(protocol_dir: Path) -> FrozenProtocol:
    """Load and strictly validate one supported frozen protocol directory."""

    if protocol_dir.is_symlink():
        raise ScreenError("protocol directory must not be a symlink")
    try:
        root = protocol_dir.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScreenError(f"protocol directory does not exist: {protocol_dir}") from error
    if not root.is_dir():
        raise ScreenError("protocol path must be a directory")
    protocol_path = _regular_file(root / "PROTOCOL.json", "PROTOCOL.json")
    raw_bytes = _read_stable_regular_file(protocol_path, "PROTOCOL.json")
    raw = _load_json_bytes(raw_bytes, "PROTOCOL.json")
    schema = _require_string(raw.get("schema_version"), "schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        raise ScreenError(f"unsupported frozen protocol schema: {schema}")
    protocol_sha256 = _sha256_bytes(raw_bytes)
    if protocol_sha256 != _FROZEN_PROTOCOL_SHA256[schema]:
        raise ScreenError("PROTOCOL.json bytes do not match the registered frozen protocol SHA-256")
    if schema in CPU_SCHEMAS:
        return _load_cpu_protocol(
            root=root,
            raw=raw,
            raw_bytes=raw_bytes,
            protocol_sha256=protocol_sha256,
            schema=schema,
        )
    if raw.get("status") != "configuration_frozen_execution_pending":
        raise ScreenError("protocol is not in frozen, execution-pending state")
    if raw.get("evidence_class") != "open_development":
        raise ScreenError("protocol is not explicitly open-development")
    if raw.get("scientific_promotion_allowed") is not False:
        raise ScreenError("protocol does not explicitly forbid scientific promotion")
    if schema == STATEFUL_SCHEMA and raw.get("sota_claim_allowed") is not False:
        raise ScreenError("stateful protocol does not explicitly forbid SOTA claims")

    task = _require_dict(raw.get("task"), "task")
    if task.get("foragax_distribution") != "continual-foragax":
        raise ScreenError("unsupported Foragax distribution")
    if task.get("foragax_version") != "0.55.0":
        raise ScreenError("unsupported Foragax version")
    if task.get("env_id") != "ForagaxTwoBiomeLarge-v1" or task.get("aperture_size") != 9:
        raise ScreenError("unsupported Foragax FOV task")
    horizon_key = "steps" if schema == BASELINE_SCHEMA else "steps_per_seed"
    horizon = _require_positive_int(task.get(horizon_key), f"task.{horizon_key}")
    raw_seeds = _require_list(task.get("seeds"), "task.seeds")
    if len(raw_seeds) != 2:
        raise ScreenError("frozen screen must contain exactly two seeds")
    seeds: list[int] = []
    for index, value in enumerate(raw_seeds):
        if type(value) is not int or value < 0:
            raise ScreenError(f"task.seeds[{index}] must be a non-negative integer")
        seeds.append(value)
    if seeds != sorted(set(seeds)) or seeds[1] != seeds[0] + 1:
        raise ScreenError("frozen seeds must be unique, ordered, and contiguous")
    index_argument = f"{seeds[0]}:{seeds[-1] + 1}"
    if schema == STATEFUL_SCHEMA:
        transport = _require_dict(task.get("seed_transport"), "task.seed_transport")
        if transport.get("pyexputils_indices") != index_argument:
            raise ScreenError("stateful PyExpUtils index transport drift")
        if transport.get("nested_seed_offset") != 0:
            raise ScreenError("stateful nested seed offset drift")
        if transport.get("expected_stored_seeds") != seeds:
            raise ScreenError("stateful stored seed declaration drift")
        if transport.get("expected_effective_seeds") != seeds:
            raise ScreenError("stateful effective seed declaration drift")

    image_reference, image_id = _runtime_identity(raw, schema)
    source_files = _source_records(raw)
    raw["__protocol_root"] = root.as_posix()
    metric = _validate_metric(raw, horizon, schema)
    raw.pop("__protocol_root")
    selection_rule, advance_count = _validate_selection_rule(raw, schema)

    raw_configurations = _require_list(raw.get("configurations"), "configurations")
    if not raw_configurations:
        raise ScreenError("configurations must not be empty")
    configurations: list[FrozenConfiguration] = []
    seen_paths: set[str] = set()
    seen_run_ids: set[str] = set()
    for index, raw_record in enumerate(raw_configurations):
        record = _require_dict(raw_record, f"configurations[{index}]")
        relative = _normalized_relative_path(record.get("path"), f"configurations[{index}].path")
        if not relative.startswith("configs/"):
            raise ScreenError("configuration paths must remain under configs/")
        expected_hash = _require_sha256(record.get("sha256"), f"configurations[{index}].sha256")
        if relative in seen_paths:
            raise ScreenError(f"duplicate configuration path: {relative}")
        seen_paths.add(relative)
        agent, entrypoint = _validate_config_payload(
            root / relative, expected_hash, relative, task, horizon, schema
        )
        config = FrozenConfiguration(relative, expected_hash, agent, entrypoint)
        if config.run_id in seen_run_ids:
            raise ScreenError(f"configuration run ID collision: {config.run_id}")
        seen_run_ids.add(config.run_id)
        configurations.append(config)

    config_dir = root / "configs"
    discovered = {
        path.relative_to(root).as_posix()
        for path in config_dir.rglob("*.json")
        if path.is_file() and not path.is_symlink()
    }
    if discovered != seen_paths:
        missing = sorted(seen_paths - discovered)
        extra = sorted(discovered - seen_paths)
        raise ScreenError(f"configuration inventory mismatch; missing={missing}, extra={extra}")

    return FrozenProtocol(
        root=root,
        configuration_root=root,
        raw=raw,
        raw_bytes=raw_bytes,
        sha256=protocol_sha256,
        schema=schema,
        horizon=horizon,
        seeds=tuple(seeds),
        index_argument=index_argument,
        image_reference=image_reference,
        image_id=image_id,
        backend=cast(
            str,
            _require_dict(raw.get("runtime"), "runtime").get("backend", "unspecified"),
        ),
        scorer_path=(
            root / "score_raw_rewards.py" if schema == STATEFUL_SCHEMA else None
        ),
        scorer_sha256=(
            cast(str, metric["implementation"]["sha256"])
            if schema == STATEFUL_SCHEMA
            else None
        ),
        reference_scorer_path=None,
        reference_scorer_sha256=None,
        base_protocol_sha256=None,
        predecessor_protocol_root=None,
        predecessor_protocol_sha256=None,
        harness_snapshot_path=None,
        probe_snapshot_path=None,
        source_files=source_files,
        configurations=tuple(configurations),
        metric=metric,
        selection_rule=selection_rule,
        advance_count=advance_count,
    )


def _harness_path() -> Path:
    return Path(__file__).resolve(strict=True)


def _probe_path() -> Path:
    return Path(__file__).with_name("_foragax_open_screen_probe.py").resolve(strict=True)


def _harness_identity(protocol: FrozenProtocol) -> dict[str, Any]:
    module = protocol.harness_snapshot_path
    probe = protocol.probe_snapshot_path
    expected_root = protocol.root.parent / "execution"
    if module != expected_root / "harness.py" or probe != expected_root / "probe.py":
        raise ScreenError("protocol does not reference the exact snapshotted harness and probe")
    module_sha256, module_size = _stable_file_record(module, "snapshotted open-screen harness")
    probe_sha256, probe_size = _stable_file_record(probe, "snapshotted open-screen probe")
    return {
        "version": HARNESS_VERSION,
        "identity_source": "pre_preflight_input_snapshot",
        "module_snapshot_path": "inputs/execution/harness.py",
        "module_sha256": module_sha256,
        "module_size_bytes": module_size,
        "probe_snapshot_path": "inputs/execution/probe.py",
        "probe_sha256": probe_sha256,
        "probe_size_bytes": probe_size,
    }


def _bound_harness_identity(snapshot: ProtocolSnapshot) -> dict[str, Any]:
    """Return the harness identity recorded before any preflight execution."""

    records = {cast(str, record["path"]): record for record in snapshot.inventory}
    try:
        module = records["execution/harness.py"]
        probe = records["execution/probe.py"]
    except KeyError as error:
        raise ScreenError(
            "input snapshot omits the executing harness or preflight probe"
        ) from error
    identity = {
        "version": HARNESS_VERSION,
        "identity_source": "pre_preflight_input_snapshot",
        "module_snapshot_path": "inputs/execution/harness.py",
        "module_sha256": module.get("sha256"),
        "module_size_bytes": module.get("size_bytes"),
        "probe_snapshot_path": "inputs/execution/probe.py",
        "probe_sha256": probe.get("sha256"),
        "probe_size_bytes": probe.get("size_bytes"),
    }
    if identity != _harness_identity(snapshot.protocol):
        raise ScreenError("pre-read harness identity differs from snapshotted bytes")
    return identity


def _capture_process(command: Sequence[str]) -> ProcessCapture:
    completed = subprocess.run(list(command), check=False, capture_output=True)
    return ProcessCapture(completed.returncode, completed.stdout, completed.stderr)


def _docker_identity(docker: str) -> dict[str, Any]:
    resolved_text = shutil.which(docker)
    if resolved_text is None and "/" in docker:
        resolved_text = Path(docker).resolve(strict=False).as_posix()
    if resolved_text is None:
        raise ScreenError(f"cannot resolve Docker executable: {docker}")
    resolved = _regular_file(Path(resolved_text).resolve(strict=True), "Docker executable")
    capture = _capture_process(
        [resolved.as_posix(), "version", "--format", "{{json .}}"]
    )
    if capture.returncode != 0 or capture.stderr:
        raise ScreenError("Docker version identity command failed or wrote stderr")
    version = _load_json_bytes(capture.stdout, "Docker version identity")
    digest, size = _stable_file_record(resolved, "Docker executable")
    return {
        "requested_command": docker,
        "executable_path": resolved.as_posix(),
        "executable_sha256": digest,
        "executable_size_bytes": size,
        "version": version,
    }


def _host_runtime_identity() -> dict[str, Any]:
    executable = _regular_file(Path(sys.executable).resolve(strict=True), "host Python")
    digest, size = _stable_file_record(executable, "host Python")
    return {
        "implementation": sys.implementation.name,
        "python_version": sys.version,
        "byteorder": sys.byteorder,
        "executable_sha256": digest,
        "executable_size_bytes": size,
    }


def _run_process_to_files(command: Sequence[str], stdout: BinaryIO, stderr: BinaryIO) -> int:
    completed = subprocess.run(list(command), check=False, stdout=stdout, stderr=stderr)
    return completed.returncode


def _inspect_image(docker: str, image_id: str) -> dict[str, Any]:
    capture = _capture_process([docker, "image", "inspect", image_id])
    if capture.returncode != 0:
        detail = capture.stderr.decode("utf-8", errors="replace").strip()
        raise ScreenError(f"cannot inspect exact OCI image {image_id}: {detail}")
    try:
        value = json.loads(
            capture.stdout.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_number,
            parse_float=_parse_finite_json_float,
        )
    except ScreenError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError("docker image inspect returned invalid JSON") from error
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ScreenError("docker image inspect must resolve exactly one image")
    inspection = cast(dict[str, Any], value[0])
    if inspection.get("Id") != image_id:
        raise ScreenError("docker resolved a different image ID")
    config = _require_dict(inspection.get("Config"), "docker image Config")
    if config.get("Entrypoint") != _EXPECTED_IMAGE_ENTRYPOINT:
        raise ScreenError("development image entrypoint drift")
    if config.get("WorkingDir") != _EXPECTED_IMAGE_WORKDIR:
        raise ScreenError("development image working directory drift")
    return {
        "id": image_id,
        "entrypoint": _EXPECTED_IMAGE_ENTRYPOINT,
        "working_dir": _EXPECTED_IMAGE_WORKDIR,
    }


def _reject_docker_mount_unsafe_path(path: Path, label: str) -> None:
    text = path.as_posix()
    if "," in text or "\n" in text or "\r" in text:
        raise ScreenError(f"{label} contains a character unsafe for Docker --mount")


def _sandbox_prefix(protocol: FrozenProtocol, docker: str) -> list[str]:
    command = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        "65532:65532",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=8g,mode=1777",
        "--workdir",
        _EXPECTED_IMAGE_WORKDIR,
    ]
    for key, value in sorted(_cpu_environment(protocol.schema).items()):
        command.extend(["--env", f"{key}={value}"])
    return command


def build_preflight_command(protocol: FrozenProtocol, docker: str) -> list[str]:
    """Return the exact no-reward OCI preflight command."""

    if protocol.schema not in EXECUTABLE_CPU_SCHEMAS or protocol.backend != "cpu":
        raise ScreenError("execution requires an explicitly frozen CPU v3 protocol")
    probe = protocol.probe_snapshot_path
    if probe is None:
        raise ScreenError("CPU v3 protocol has no pre-preflight probe snapshot")
    _reject_docker_mount_unsafe_path(protocol.root, "protocol path")
    _reject_docker_mount_unsafe_path(protocol.configuration_root, "configuration path")
    _reject_docker_mount_unsafe_path(probe, "probe path")
    if protocol.scorer_path is None:
        raise ScreenError("CPU protocol has no bound scorer")
    _reject_docker_mount_unsafe_path(protocol.scorer_path, "scorer path")
    command = [
        *_sandbox_prefix(protocol, docker),
        "--mount",
        f"type=bind,src={protocol.root},dst={_CONTAINER_PROTOCOL_ROOT},readonly",
        "--mount",
        f"type=bind,src={protocol.configuration_root},dst=/protocol-input,readonly",
        "--mount",
        f"type=bind,src={probe},dst={_PROBE_CONTAINER_PATH},readonly",
        "--mount",
        f"type=bind,src={protocol.scorer_path},dst={_SCORER_CONTAINER_PATH},readonly",
    ]
    if protocol.reference_scorer_path is not None:
        _reject_docker_mount_unsafe_path(
            protocol.reference_scorer_path,
            "reference scorer path",
        )
        command.extend(
            [
                "--mount",
                "type=bind,"
                f"src={protocol.reference_scorer_path},"
                f"dst={_REFERENCE_SCORER_CONTAINER_PATH},readonly",
            ]
        )
    if protocol.predecessor_protocol_root is None:
        raise ScreenError("CPU v3 protocol has no hash-bound predecessor")
    _reject_docker_mount_unsafe_path(
        protocol.predecessor_protocol_root,
        "predecessor protocol path",
    )
    command.extend(
        [
            "--mount",
            "type=bind,"
            f"src={protocol.predecessor_protocol_root},"
            "dst=/predecessor-protocol,readonly",
        ]
    )
    command.extend([protocol.image_id, _PROBE_CONTAINER_PATH])
    return command


def build_candidate_command(
    protocol: FrozenProtocol,
    config: FrozenConfiguration,
    payload_dir: Path,
    docker: str,
) -> list[str]:
    """Return the exact reward command for one two-lane candidate."""

    if protocol.schema not in EXECUTABLE_CPU_SCHEMAS or protocol.backend != "cpu":
        raise ScreenError("execution requires an explicitly frozen CPU v3 protocol")
    _reject_docker_mount_unsafe_path(protocol.root, "protocol path")
    _reject_docker_mount_unsafe_path(protocol.configuration_root, "configuration path")
    _reject_docker_mount_unsafe_path(payload_dir, "candidate output path")
    config_container_path = f"/protocol-input/{config.path}"
    entrypoint = f"{_CONTAINER_SOURCE_ROOT}/{config.entrypoint}"
    return [
        *_sandbox_prefix(protocol, docker),
        "--mount",
        f"type=bind,src={protocol.root},dst={_CONTAINER_PROTOCOL_ROOT},readonly",
        "--mount",
        f"type=bind,src={protocol.configuration_root},dst=/protocol-input,readonly",
        "--mount",
        f"type=bind,src={payload_dir},dst=/run-output",
        protocol.image_id,
        entrypoint,
        "--exp",
        config_container_path,
        "--idxs",
        protocol.index_argument,
        "--save_path",
        "/run-output/results",
        "--checkpoint_path",
        "/tmp/checkpoints",
        "--silent",
    ]


def _run_preflight(protocol: FrozenProtocol, docker: str) -> tuple[dict[str, Any], bytes, bytes]:
    """Run the no-reward OCI preflight and cross-check its probe against the protocol.

    The in-container probe echoes back every identity it can observe —
    protocol/base/predecessor/scorer digests, the full source-file and
    configuration inventories, per-configuration seeds and entrypoints, and
    (for stateful protocols) the scorer-equivalence cases — and each echo
    must match the frozen protocol exactly.  The probe only invokes
    entrypoint ``--help``; it must attest that no environment or transition
    construction happened, keeping the preflight reward-blind.
    """
    command = build_preflight_command(protocol, docker)
    capture = _capture_process(command)
    if capture.returncode != 0:
        detail = capture.stderr.decode("utf-8", errors="replace").strip()
        raise ScreenError(f"no-reward OCI preflight failed: {detail}")
    probe = _load_json_bytes(capture.stdout, "OCI preflight stdout")
    if probe.get("schema_version") != "alberta.foragax_open_development_preflight.v2":
        raise ScreenError("OCI preflight schema mismatch")
    if probe.get("status") != "passed":
        raise ScreenError("OCI preflight did not report passed status")
    if probe.get("protocol_schema") != protocol.schema:
        raise ScreenError("OCI preflight protocol schema mismatch")
    if probe.get("protocol_sha256") != protocol.sha256:
        raise ScreenError("OCI preflight protocol hash mismatch")
    if probe.get("base_protocol_sha256") != protocol.base_protocol_sha256:
        raise ScreenError("OCI preflight base protocol hash mismatch")
    expected_base_schema = (
        BASELINE_SCHEMA if protocol.schema in BASELINE_CPU_SCHEMAS else STATEFUL_SCHEMA
    )
    if probe.get("base_protocol_schema") != expected_base_schema:
        raise ScreenError("OCI preflight base protocol schema mismatch")
    if probe.get("scorer_sha256") != protocol.scorer_sha256:
        raise ScreenError("OCI preflight scorer hash mismatch")
    if probe.get("predecessor_protocol_sha256") != protocol.predecessor_protocol_sha256:
        raise ScreenError("OCI preflight predecessor protocol hash mismatch")
    equivalence = _require_dict(
        probe.get("scorer_equivalence"),
        "preflight.scorer_equivalence",
    )
    if protocol.schema in STATEFUL_CPU_SCHEMAS:
        cases = _require_list(
            equivalence.get("cases"),
            "preflight.scorer_equivalence.cases",
        )
        if (
            equivalence.get("status") != "passed"
            or equivalence.get("reference_scorer_sha256")
            != protocol.reference_scorer_sha256
            or len(cases) != 2
        ):
            raise ScreenError("OCI preflight scorer equivalence mismatch")
    elif equivalence != {
        "status": "not_applicable",
        "reason": "base protocol has no bound scorer",
    }:
        raise ScreenError("OCI preflight baseline scorer declaration mismatch")
    expected_sources = [{"path": path, "sha256": digest} for path, digest in protocol.source_files]
    if probe.get("source_files") != expected_sources:
        raise ScreenError("OCI preflight source identity mismatch")
    probe_configs = _require_list(probe.get("configurations"), "preflight.configurations")
    if len(probe_configs) != len(protocol.configurations) or any(
        not isinstance(item, dict) for item in probe_configs
    ):
        raise ScreenError("OCI preflight configuration inventory is malformed")
    if [cast(dict[str, Any], item).get("path") for item in probe_configs] != [
        config.path for config in protocol.configurations
    ]:
        raise ScreenError("OCI preflight configuration order or inventory mismatch")
    for config, raw_item in zip(protocol.configurations, probe_configs, strict=True):
        item = cast(dict[str, Any], raw_item)
        if (
            item.get("sha256") != config.sha256
            or item.get("agent") != config.agent
            or item.get("entrypoint") != config.entrypoint
            or item.get("num_permutations") != 1
            or item.get("stored_seeds") != list(protocol.seeds)
            or item.get("effective_seeds") != list(protocol.seeds)
            or type(item.get("result_root")) is not str
            or not isinstance(item.get("metadata_contract"), dict)
        ):
            raise ScreenError(f"OCI preflight configuration binding drift: {config.path}")
    entrypoints = list(dict.fromkeys(config.entrypoint for config in protocol.configurations))
    expected_executable_preflight = {
        "status": "passed",
        "transition_operations_invoked": False,
        "entrypoints": [
            {
                "path": entrypoint,
                "argv": [
                    "/opt/foragax-agents/.venv/bin/python",
                    "-I",
                    f"/opt/foragax-agents/{entrypoint}",
                    "--help",
                ],
                "returncode": 0,
                "help_marker_present": True,
                "forbidden_cache_diagnostics_absent": True,
            }
            for entrypoint in entrypoints
        ],
        "cache_directories": [
            {
                "environment_variable": "MPLCONFIGDIR",
                "path": "/tmp/alberta-matplotlib-cache",
                "directory": True,
                "owner_uid": 65532,
                "owner_gid": 65532,
                "writable": True,
            },
            {
                "environment_variable": "NUMBA_CACHE_DIR",
                "path": "/tmp/alberta-numba-cache",
                "directory": True,
                "owner_uid": 65532,
                "owner_gid": 65532,
                "writable": True,
            },
        ],
    }
    if probe.get("executable_import_preflight") != expected_executable_preflight:
        raise ScreenError("OCI executable/import/cache preflight contract mismatch")
    runtime = _require_dict(probe.get("runtime"), "preflight.runtime")
    devices = _require_list(runtime.get("jax_devices"), "preflight.runtime.jax_devices")
    if (
        runtime.get("uid") != 65532
        or runtime.get("gid") != 65532
        or runtime.get("nonroot") is not True
        or runtime.get("root_filesystem_read_only") is not True
        or runtime.get("network_interfaces") != ["lo"]
        or runtime.get("nvidia_device_glob") != []
        or runtime.get("continual_foragax_version") != "0.55.0"
        or runtime.get("jax_default_backend") != "cpu"
        or not devices
        or any(type(device) is not str or not device.startswith("cpu:") for device in devices)
        or runtime.get("jax_platform_name") != "cpu"
        or runtime.get("jax_platforms") != "cpu"
        or runtime.get("nvidia_visible_devices") != "void"
        or runtime.get("cuda_visible_devices") != ""
        or runtime.get("pythonhashseed") != "0"
        or runtime.get("mplconfigdir") != "/tmp/alberta-matplotlib-cache"
        or runtime.get("numba_cache_dir") != "/tmp/alberta-numba-cache"
    ):
        raise ScreenError("OCI preflight runtime/backend contract mismatch")
    return probe, capture.stdout, capture.stderr


def _ensure_exclusive_file(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_manifest_pair(path: Path, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json(payload)
    _ensure_exclusive_file(path, raw)
    _ensure_exclusive_file(
        path.with_suffix(path.suffix + ".sha256"), (_sha256_bytes(raw) + "\n").encode()
    )


def _validate_manifest_pair(path: Path, schema: str) -> dict[str, Any]:
    manifest_path = path
    sidecar_path = path.with_suffix(path.suffix + ".sha256")
    raw = _read_stable_regular_file(
        manifest_path,
        path.name,
        maximum=64 * 1024**2,
    )
    payload = _load_json_bytes(raw, path.name)
    if _canonical_json(payload) != raw:
        raise ScreenError(f"{path.name} is not canonical JSON")
    expected_sidecar = (_sha256_bytes(raw) + "\n").encode("ascii")
    if _read_stable_regular_file(
        sidecar_path,
        f"{path.name}.sha256",
        maximum=1024,
    ) != expected_sidecar:
        raise ScreenError(f"{path.name} SHA-256 sidecar mismatch")
    if payload.get("schema_version") != schema:
        raise ScreenError(f"{path.name} schema mismatch")
    return payload


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _reject_known_pinned_output(output: Path) -> None:
    for relative in _KNOWN_PINNED_OUTPUTS:
        protected = (_REPOSITORY_ROOT / relative).resolve(strict=False)
        if _paths_overlap(output, protected):
            raise ScreenError(f"screen output must not overlap pinned evidence path {relative}")
    for relative in _KNOWN_PROTOCOL_ROOTS:
        protected = (_REPOSITORY_ROOT / relative).resolve(strict=False)
        if _paths_overlap(output, protected):
            raise ScreenError(f"screen output must not overlap protocol path {relative}")


def _resolve_output_root(path: Path, protocol: FrozenProtocol) -> Path:
    if path.is_symlink():
        raise ScreenError("output directory must not be a symlink")
    output = path.resolve(strict=False)
    if _paths_overlap(output, protocol.root):
        raise ScreenError("output directory must not overlap the frozen protocol directory")
    _reject_known_pinned_output(output)
    if output.exists() and not output.is_dir():
        raise ScreenError("output path exists but is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ScreenError("resolved output directory must not be a symlink")
    return output.resolve(strict=True)


def _precheck_output_before_lock(output: Path) -> None:
    names = {path.name for path in output.iterdir()}
    if not names:
        return
    if ".screen.lock" not in names:
        raise ScreenError("refusing a non-empty output root before creating its lock")
    allowed = {
        ".screen.lock",
        ".incomplete",
        "aggregate.json",
        "aggregate.json.sha256",
        "inputs",
        "preflight.stderr.log",
        "preflight.stdout.log",
        "runs",
        "screen_plan.json",
        "screen_plan.json.sha256",
    }
    extras = names - allowed
    if extras:
        raise ScreenError(f"screen output contains unexpected root artifacts: {sorted(extras)}")
    lock = _regular_file(output / ".screen.lock", ".screen.lock")
    if lock.stat().st_size != 0:
        raise ScreenError("screen lock file must remain empty")


def _open_output_lock(output: Path, *, create: bool = True) -> BinaryIO:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(output / ".screen.lock", flags, 0o600)
    except OSError as error:
        raise ScreenError("cannot safely open the screen output lock") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != 0
    ):
        os.close(descriptor)
        raise ScreenError("screen lock must be an empty single-link regular file")
    return os.fdopen(descriptor, "r+b")


def _snapshot_source_files(protocol: FrozenProtocol) -> dict[str, bytes]:
    if protocol.schema not in EXECUTABLE_CPU_SCHEMAS or protocol.scorer_path is None:
        raise ScreenError("only CPU v3 protocols can be snapshotted for execution")
    if (
        protocol.predecessor_protocol_root is None
        or protocol.predecessor_protocol_sha256 is None
    ):
        raise ScreenError("CPU v3 protocol has no hash-bound predecessor")
    files = {
        "execution/harness.py": _read_stable_regular_file(
            _harness_path(), "live open-screen harness"
        ),
        "execution/probe.py": _read_stable_regular_file(
            _probe_path(), "live open-screen probe"
        ),
        "protocol/PROTOCOL.json": _read_stable_regular_file(
            protocol.root / "PROTOCOL.json", "CPU protocol"
        ),
        "base/PROTOCOL.json": _read_stable_regular_file(
            protocol.configuration_root / "PROTOCOL.json", "base protocol"
        ),
        "scorer.py": _read_stable_regular_file(protocol.scorer_path, "bound scorer"),
        "predecessor/PROTOCOL.json": _read_stable_regular_file(
            protocol.predecessor_protocol_root / "PROTOCOL.json",
            "predecessor CPU protocol",
        ),
    }
    for config in protocol.configurations:
        files[f"base/{config.path}"] = _read_stable_regular_file(
            protocol.configuration_root / config.path,
            f"configuration {config.path}",
        )
    if protocol.reference_scorer_path is not None:
        files["reference_scorer.py"] = _read_stable_regular_file(
            protocol.reference_scorer_path,
            "reference scorer",
        )
    expected_hashes = {
        "protocol/PROTOCOL.json": protocol.sha256,
        "base/PROTOCOL.json": cast(str, protocol.base_protocol_sha256),
        "scorer.py": cast(str, protocol.scorer_sha256),
        "predecessor/PROTOCOL.json": protocol.predecessor_protocol_sha256,
        **{f"base/{config.path}": config.sha256 for config in protocol.configurations},
    }
    if protocol.reference_scorer_sha256 is not None:
        expected_hashes["reference_scorer.py"] = protocol.reference_scorer_sha256
    for relative, expected in expected_hashes.items():
        if _sha256_bytes(files[relative]) != expected:
            raise ScreenError(f"snapshot source hash drift: {relative}")
    return files


def _snapshot_inventory(files: Mapping[str, bytes]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "path": relative,
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
        }
        for relative, raw in sorted(files.items())
    )


def _prepare_protocol_snapshot(
    protocol: FrozenProtocol,
    output: Path,
    *,
    allow_create: bool = True,
) -> ProtocolSnapshot:
    files = _snapshot_source_files(protocol)
    inventory = _snapshot_inventory(files)
    directories = _parent_directories(files)
    inventory_sha256 = _sha256_bytes(
        _canonical_json({"directories": directories, "files": list(inventory)})
    )
    root = output / "inputs"
    manifest_payload = {
        "schema_version": INPUT_SNAPSHOT_SCHEMA,
        "directories": directories,
        "files": list(inventory),
        "inventory_sha256": inventory_sha256,
    }
    if root.is_symlink():
        raise ScreenError("input snapshot must not be a symlink")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ScreenError("input snapshot must be a non-symlink directory")
        persisted = _validate_manifest_pair(
            root / "snapshot.json", INPUT_SNAPSHOT_SCHEMA
        )
        if persisted != manifest_payload:
            raise ScreenError("input snapshot manifest differs from current frozen inputs")
        discovered = {
            path.relative_to(root).as_posix()
            for path in _payload_files(root)
            if path.relative_to(root).as_posix()
            not in {"snapshot.json", "snapshot.json.sha256"}
        }
        if discovered != set(files):
            raise ScreenError("input snapshot file inventory drift")
        if _directory_inventory(root) != directories:
            raise ScreenError("input snapshot directory inventory drift")
        for relative, raw in files.items():
            if _read_stable_regular_file(root / relative, f"snapshot {relative}") != raw:
                raise ScreenError(f"input snapshot bytes drift: {relative}")
    else:
        if not allow_create:
            raise ScreenError("input snapshot is missing from a completed screen")
        root.mkdir(mode=0o700)
        for relative, raw in sorted(files.items()):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ensure_exclusive_file(destination, raw)
        _write_manifest_pair(root / "snapshot.json", manifest_payload)
    if _directory_inventory(root) != directories:
        raise ScreenError("input snapshot directory inventory changed")
    snapped = replace(
        protocol,
        root=root / "protocol",
        configuration_root=root / "base",
        scorer_path=root / "scorer.py",
        predecessor_protocol_root=root / "predecessor",
        harness_snapshot_path=root / "execution/harness.py",
        probe_snapshot_path=root / "execution/probe.py",
        reference_scorer_path=(
            root / "reference_scorer.py"
            if protocol.reference_scorer_path is not None
            else None
        ),
    )
    return ProtocolSnapshot(snapped, inventory, inventory_sha256)


def _verify_live_sources_match_snapshot(
    original_protocol: FrozenProtocol,
    expected: ProtocolSnapshot,
    output: Path,
    stage: str,
) -> ProtocolSnapshot:
    """Re-read every live local input and require the pre-preflight snapshot."""

    try:
        current = _prepare_protocol_snapshot(
            original_protocol,
            output,
            allow_create=False,
        )
    except ScreenError as error:
        raise ScreenError(f"execution-critical local input changed {stage}") from error
    if (
        current.inventory_sha256 != expected.inventory_sha256
        or current.inventory != expected.inventory
        or _harness_identity(current.protocol) != _harness_identity(expected.protocol)
    ):
        raise ScreenError(f"execution-critical local input identity changed {stage}")
    return current


def _plan_payload(
    input_snapshot: ProtocolSnapshot,
    image: dict[str, Any],
    docker_identity: dict[str, Any],
    host_runtime: dict[str, Any],
    probe: dict[str, Any],
    preflight_stdout: bytes,
    preflight_stderr: bytes,
) -> dict[str, Any]:
    protocol = input_snapshot.protocol
    snapshot_manifest = _validate_manifest_pair(
        protocol.root.parent / "snapshot.json",
        INPUT_SNAPSHOT_SCHEMA,
    )
    if (
        snapshot_manifest.get("inventory_sha256") != input_snapshot.inventory_sha256
        or snapshot_manifest.get("files") != list(input_snapshot.inventory)
    ):
        raise ScreenError("pre-read input snapshot identity drift before plan binding")
    return {
        "schema_version": PLAN_SCHEMA,
        "classification": "open_development_nonpromoting",
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "reward_informed_early_stopping_allowed": False,
        "collector_summaries_used": False,
        "execution_backend": "cpu",
        "protocol": {
            "schema_version": protocol.schema,
            "sha256": protocol.sha256,
            "horizon_per_seed": protocol.horizon,
            "seeds": list(protocol.seeds),
            "indices_argument": protocol.index_argument,
            "configuration_order": [config.path for config in protocol.configurations],
            "selection_rule": protocol.selection_rule,
        },
        "development_image": {
            "reference_for_information_only": protocol.image_reference,
            **image,
            "qualified_production_image": False,
        },
        "sandbox": _sandbox_contract(protocol),
        "harness": _bound_harness_identity(input_snapshot),
        "input_snapshot": snapshot_manifest,
        "host_runtime": host_runtime,
        "docker_runtime": docker_identity,
        "preflight": {
            "command": build_preflight_command(
                protocol, cast(str, docker_identity["executable_path"])
            ),
            "result": probe,
            "stdout_sha256": _sha256_bytes(preflight_stdout),
            "stdout_size_bytes": len(preflight_stdout),
            "initial_diagnostic_stderr_sha256": _sha256_bytes(preflight_stderr),
            "initial_diagnostic_stderr_size_bytes": len(preflight_stderr),
        },
    }


def _prepare_plan(
    original_protocol: FrozenProtocol,
    snapshot: ProtocolSnapshot,
    output: Path,
    docker: str,
) -> dict[str, Any]:
    protocol = snapshot.protocol
    _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "before preflight runtime inspection",
    )
    docker_identity = _docker_identity(docker)
    docker_executable = _require_string(
        docker_identity.get("executable_path"),
        "Docker executable path",
    )
    image = _inspect_image(docker_executable, protocol.image_id)
    host_runtime = _host_runtime_identity()
    _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "immediately before preflight",
    )
    probe, probe_stdout, probe_stderr = _run_preflight(protocol, docker_executable)
    _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "during preflight",
    )
    current_docker = _docker_identity(docker_executable)
    current_docker["requested_command"] = docker_identity["requested_command"]
    if current_docker != docker_identity:
        raise ScreenError("Docker runtime identity changed during preflight")
    if _host_runtime_identity() != host_runtime:
        raise ScreenError("host Python runtime identity changed during preflight")
    if _inspect_image(docker_executable, protocol.image_id) != image:
        raise ScreenError("development image identity changed during preflight")
    expected = _plan_payload(
        snapshot,
        image,
        docker_identity,
        host_runtime,
        probe,
        probe_stdout,
        probe_stderr,
    )
    _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "between preflight and plan persistence",
    )
    plan_path = output / "screen_plan.json"
    stdout_path = output / "preflight.stdout.log"
    stderr_path = output / "preflight.stderr.log"
    if plan_path.exists():
        existing = _validate_manifest_pair(plan_path, PLAN_SCHEMA)
        existing_preflight = _require_dict(
            existing.get("preflight"), "existing screen plan preflight"
        )
        expected_preflight = _require_dict(
            expected.get("preflight"), "current screen plan preflight"
        )
        for key in (
            "initial_diagnostic_stderr_sha256",
            "initial_diagnostic_stderr_size_bytes",
        ):
            expected_preflight[key] = existing_preflight.get(key)
        if existing != expected:
            raise ScreenError("existing screen plan is not byte-identical to current inputs")
        if _read_stable_regular_file(stdout_path, stdout_path.name) != probe_stdout:
            raise ScreenError("preserved preflight stdout differs from current exact preflight")
        preserved_stderr = _read_stable_regular_file(
            stderr_path, stderr_path.name
        )
        if (
            len(preserved_stderr)
            != existing_preflight.get("initial_diagnostic_stderr_size_bytes")
            or _sha256_bytes(preserved_stderr)
            != existing_preflight.get("initial_diagnostic_stderr_sha256")
        ):
            raise ScreenError("preserved initial preflight diagnostics do not verify")
        _verify_live_sources_match_snapshot(
            original_protocol,
            snapshot,
            output,
            "while resuming the screen plan",
        )
        return existing

    allowed_before_plan = {".screen.lock", "inputs"}
    extras = {path.name for path in output.iterdir()} - allowed_before_plan
    if extras:
        raise ScreenError(f"refusing non-empty output without a screen plan: {sorted(extras)}")
    _ensure_exclusive_file(stdout_path, probe_stdout)
    _ensure_exclusive_file(stderr_path, probe_stderr)
    _write_manifest_pair(plan_path, expected)
    _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "while persisting the screen plan",
    )
    return expected


def _verify_runtime_identity(
    original_protocol: FrozenProtocol,
    snapshot: ProtocolSnapshot,
    output: Path,
    plan: Mapping[str, Any],
    docker: str,
) -> None:
    protocol = snapshot.protocol
    current = _verify_live_sources_match_snapshot(
        original_protocol,
        snapshot,
        output,
        "after planning",
    )
    if (
        _harness_identity(current.protocol) != _harness_identity(snapshot.protocol)
    ):
        raise ScreenError("snapshotted harness or preflight probe identity changed after planning")
    _verify_bound_runtime_identity(protocol, plan, docker)
    planned_snapshot = _require_dict(plan.get("input_snapshot"), "screen_plan.input_snapshot")
    if (
        planned_snapshot.get("inventory_sha256") != snapshot.inventory_sha256
        or planned_snapshot.get("files") != list(snapshot.inventory)
    ):
        raise ScreenError("screen plan input snapshot identity drift")


def _verify_bound_runtime_identity(
    protocol: FrozenProtocol,
    plan: Mapping[str, Any],
    docker: str,
) -> None:
    if plan.get("harness") != _harness_identity(protocol):
        raise ScreenError("snapshotted harness or preflight probe identity changed after planning")
    planned_snapshot = _require_dict(plan.get("input_snapshot"), "screen_plan.input_snapshot")
    persisted_snapshot = _validate_manifest_pair(
        protocol.root.parent / "snapshot.json",
        INPUT_SNAPSHOT_SCHEMA,
    )
    if planned_snapshot != persisted_snapshot:
        raise ScreenError("screen plan input snapshot identity drift")
    expected_files = [
        _require_dict(record, "input snapshot file")
        for record in _require_list(planned_snapshot.get("files"), "input snapshot files")
    ]
    expected_directories = _require_list(
        planned_snapshot.get("directories"),
        "input snapshot directories",
    )
    if (
        _sha256_bytes(
            _canonical_json(
                {"directories": expected_directories, "files": expected_files}
            )
        )
        != planned_snapshot.get("inventory_sha256")
    ):
        raise ScreenError("screen plan input snapshot inventory digest drift")
    root = protocol.root.parent
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in _payload_files(root)
        if path.relative_to(root).as_posix()
        not in {"snapshot.json", "snapshot.json.sha256"}
    }
    expected_paths = {cast(str, record.get("path")) for record in expected_files}
    if actual_paths != expected_paths or _directory_inventory(root) != expected_directories:
        raise ScreenError("snapshotted execution input filesystem inventory drift")
    for record in expected_files:
        relative = _require_string(record.get("path"), "input snapshot file path")
        digest, size = _stable_file_record(root / relative, f"snapshotted input {relative}")
        if digest != record.get("sha256") or size != record.get("size_bytes"):
            raise ScreenError(f"snapshotted execution input bytes changed: {relative}")
    planned_docker = _require_dict(plan.get("docker_runtime"), "screen_plan.docker_runtime")
    current_docker = _docker_identity(docker)
    current_docker["requested_command"] = planned_docker.get("requested_command")
    if planned_docker != current_docker:
        raise ScreenError("Docker runtime identity changed after preflight")
    if plan.get("host_runtime") != _host_runtime_identity():
        raise ScreenError("host Python runtime identity changed after preflight")
    development_image = _require_dict(
        plan.get("development_image"),
        "screen_plan.development_image",
    )
    docker_executable = _require_string(
        planned_docker.get("executable_path"),
        "screen_plan.docker_runtime.executable_path",
    )
    if docker != docker_executable:
        raise ScreenError("execution did not use the plan-bound Docker executable path")
    current_image = _inspect_image(docker_executable, protocol.image_id)
    if any(development_image.get(key) != value for key, value in current_image.items()):
        raise ScreenError("development image identity changed after preflight")


def _validate_output_tree(
    protocol: FrozenProtocol,
    output: Path,
    plan: Mapping[str, Any],
    *,
    require_complete_runs: bool,
) -> None:
    if output.is_symlink() or not output.is_dir():
        raise ScreenError("screen output root must be a non-symlink directory")
    _reject_known_pinned_output(output)
    allowed = {
        ".screen.lock",
        ".incomplete",
        "aggregate.json",
        "aggregate.json.sha256",
        "inputs",
        "preflight.stderr.log",
        "preflight.stdout.log",
        "runs",
        "screen_plan.json",
        "screen_plan.json.sha256",
    }
    discovered = {path.name for path in output.iterdir()}
    extras = discovered - allowed
    if extras:
        raise ScreenError(f"screen output contains unexpected root artifacts: {sorted(extras)}")
    lock = _regular_file(output / ".screen.lock", ".screen.lock")
    if lock.stat().st_size != 0:
        raise ScreenError("screen lock file must remain empty")

    preflight = _require_dict(plan.get("preflight"), "screen_plan.preflight")
    for filename, field in (
        ("preflight.stdout.log", "stdout"),
        ("preflight.stderr.log", "initial_diagnostic_stderr"),
    ):
        digest, size = _stable_file_record(output / filename, filename)
        if size != preflight.get(f"{field}_size_bytes"):
            raise ScreenError(f"{filename} size does not match the screen plan")
        if digest != preflight.get(f"{field}_sha256"):
            raise ScreenError(f"{filename} hash does not match the screen plan")

    aggregate_pair = {"aggregate.json", "aggregate.json.sha256"} & discovered
    if aggregate_pair and aggregate_pair != {"aggregate.json", "aggregate.json.sha256"}:
        raise ScreenError("aggregate manifest and SHA-256 sidecar must appear together")

    expected_ids = {config.run_id for config in protocol.configurations}
    runs = output / "runs"
    discovered_ids: set[str] = set()
    if runs.exists():
        if runs.is_symlink() or not runs.is_dir():
            raise ScreenError("runs must be a non-symlink directory")
        for child in runs.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise ScreenError(f"runs contains a non-directory entry: {child.name}")
            discovered_ids.add(child.name)
    if not discovered_ids <= expected_ids:
        raise ScreenError(
            f"runs contains unknown candidate IDs: {sorted(discovered_ids - expected_ids)}"
        )
    if require_complete_runs and discovered_ids != expected_ids:
        raise ScreenError(
            f"completed run directory set mismatch; missing={sorted(expected_ids - discovered_ids)}"
        )
    if aggregate_pair and discovered_ids != expected_ids:
        raise ScreenError("aggregate artifacts are forbidden while the run set is incomplete")

    incomplete = output / ".incomplete"
    if incomplete.exists():
        if incomplete.is_symlink() or not incomplete.is_dir():
            raise ScreenError(".incomplete must be a non-symlink directory")
        entries = sorted(path.name for path in incomplete.iterdir())
        if entries:
            raise ScreenError(
                f"an interrupted attempt cannot be resumed in place; incomplete entries={entries}"
            )


def _metric_contract(protocol: FrozenProtocol) -> dict[str, Any]:
    sample_count = (protocol.horizon + 99) // 100
    tail_start = int(0.9 * sample_count)
    return {
        "name": "fov_last_10pct_ema_auc",
        "horizon": protocol.horizon,
        "ema_decay": 0.999,
        "ema_initial_value": 0.0,
        "bias_correction": False,
        "subsample_every_steps": 100,
        "subsample_first_reward": True,
        "subsample_index_origin": 0,
        "sample_count": sample_count,
        "tail_fraction_of_sampled_curve": 0.1,
        "tail_start_index": tail_start,
        "tail_sample_count": sample_count - tail_start,
        "direction": "maximize",
        "collector_summaries_used": False,
    }


def score_raw_rewards(rewards: npt.NDArray[np.generic], horizon: int) -> dict[str, Any]:
    """Validate and compute the exact frozen unadjusted FOV tail-EMA statistic.

    This is the Forager paper's field-of-view metric (arXiv:2605.01131; see
    ``FORAGER_BENCHMARK.md``): the mean over the final 10% of the unadjusted
    reward-EMA curve — decay 0.999, zero initial value, no bias correction —
    sampled every 100 steps starting at step 0, matching the upstream
    stored-EMA convention (frames 0, 100, ...).  The constants are frozen
    protocol values (see ``_metric_contract``/``_validate_metric``), not
    tunables.  Returns the score together with the reward-trace digest,
    dtype/shape, and tail-boundary bookkeeping used for drift checks against
    the in-image scorer.
    """

    if rewards.shape != (horizon,):
        raise ScreenError(f"raw rewards must have exact shape ({horizon},), got {rewards.shape}")
    if rewards.dtype.kind not in {"i", "u", "f"}:
        raise ScreenError(f"raw rewards have unsupported dtype {rewards.dtype}")
    if not bool(np.all(np.isfinite(rewards))):
        raise ScreenError("raw rewards contain non-finite values")
    rewards64 = rewards.astype(np.float64, copy=False)
    ema = 0.0
    samples: list[float] = []
    for index, reward in enumerate(rewards64):
        ema = 0.999 * ema + (1.0 - 0.999) * float(reward)
        if index % 100 == 0:
            samples.append(ema)
    expected_count = (horizon + 99) // 100
    if len(samples) != expected_count:
        raise ScreenError("internal FOV EMA sample count mismatch")
    tail_start = int(0.9 * expected_count)
    tail = np.asarray(samples[tail_start:], dtype=np.float64)
    if tail.size != expected_count - tail_start or tail.size == 0:
        raise ScreenError("internal FOV EMA tail boundary mismatch")
    contiguous = np.ascontiguousarray(rewards)
    with np.errstate(over="ignore", invalid="ignore"):
        reward_sum = float(np.sum(rewards64, dtype=np.float64))
        fov_score = float(np.mean(tail, dtype=np.float64))
    if not all(math.isfinite(value) for value in (reward_sum, fov_score, ema)):
        raise ScreenError("raw rewards produce a non-finite aggregate or EMA")
    return {
        "reward_dtype": rewards.dtype.str,
        "reward_shape": [horizon],
        "reward_trace_sha256": _sha256_bytes(contiguous.tobytes(order="C")),
        "reward_sum_float64": reward_sum,
        "fov_last_10pct_ema_auc": fov_score,
        "final_unadjusted_ema": float(ema),
        "ema_sample_count": expected_count,
        "ema_tail_start_index": tail_start,
        "ema_tail_sample_count": int(tail.size),
    }


def _validate_reward_member_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    horizon: int,
    path: Path,
) -> None:
    try:
        with archive.open(info, "r") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    member,
                    max_header_size=10_000,
                )
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    member,
                    max_header_size=10_000,
                )
            else:
                raise ScreenError(f"rewards.npy uses an unsupported NPY version: {path}")
            expected_data_bytes = horizon * dtype.itemsize
            if (
                shape != (horizon,)
                or dtype.kind not in {"i", "u", "f"}
                or dtype.hasobject
                or expected_data_bytes < 0
                or expected_data_bytes > _MAX_REWARD_MEMBER_BYTES
                or info.file_size - member.tell() != expected_data_bytes
            ):
                raise ScreenError(f"rewards.npy header/size contract drift: {path}")
    except (EOFError, OSError, ValueError, NotImplementedError) as error:
        raise ScreenError(f"cannot safely parse rewards.npy header: {path}") from error


def _validate_zip_structure(path: Path, horizon: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScreenError(f"cannot safely open reward NPZ: {path}") from error
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > _MAX_NPZ_BYTES
            ):
                raise ScreenError(f"NPZ size or link count is outside its bound: {path}")
            with zipfile.ZipFile(stream, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if (
                    not infos
                    or len(infos) > _MAX_ZIP_MEMBERS
                    or len(names) != len(set(names))
                    or names.count("rewards.npy") != 1
                ):
                    raise ScreenError(f"NPZ has an invalid member inventory: {path}")
                total = 0
                for info in infos:
                    member = PurePosixPath(info.filename)
                    if (
                        info.is_dir()
                        or member.is_absolute()
                        or member.name != info.filename
                        or member.suffix != ".npy"
                        or "." in member.parts
                        or ".." in member.parts
                        or info.flag_bits & 0x1
                        or info.file_size < 0
                        or info.file_size > _MAX_UNCOMPRESSED_BYTES
                        or (
                            info.filename == "rewards.npy"
                            and info.file_size > _MAX_REWARD_MEMBER_BYTES
                        )
                        or (info.compress_size == 0 and info.file_size != 0)
                    ):
                        raise ScreenError(f"NPZ has unsafe or encrypted member: {path}")
                    total += info.file_size
                    if total > _MAX_UNCOMPRESSED_BYTES:
                        raise ScreenError(f"NPZ expands beyond its bound: {path}")
                    if info.filename == "rewards.npy":
                        _validate_reward_member_header(archive, info, horizon, path)
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ScreenError(f"NPZ CRC failure in member {bad_member!r}: {path}")
            after = os.fstat(stream.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ScreenError(f"NPZ changed while its ZIP structure was validated: {path}")
    except (OSError, zipfile.BadZipFile, NotImplementedError) as error:
        raise ScreenError(f"invalid NPZ archive: {path}") from error
    finally:
        os.close(descriptor)


def _payload_files(payload: Path) -> list[Path]:
    files: list[Path] = []
    identities: set[tuple[int, int]] = set()
    for root, directories, filenames in os.walk(payload, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            child = root_path / directory
            if child.is_symlink():
                raise ScreenError(f"payload contains symlink directory: {child}")
        for filename in filenames:
            child = root_path / filename
            metadata = _regular_file(child, "payload artifact").stat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in identities:
                raise ScreenError(f"payload contains an inode alias: {child}")
            identities.add(identity)
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(payload).as_posix())


def _directory_inventory(root: Path) -> list[str]:
    directories: list[str] = []
    for current, names, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in names:
            child = current_path / name
            try:
                metadata = child.lstat()
            except FileNotFoundError as error:
                raise ScreenError(f"directory disappeared during inventory: {child}") from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ScreenError(f"tree contains a non-directory or symlink: {child}")
            directories.append(child.relative_to(root).as_posix())
    return sorted(directories)


def _parent_directories(paths: Iterable[str]) -> list[str]:
    directories: set[str] = set()
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return sorted(directories)


def _tagged_sqlite_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        return {"type": "real", "value": value.hex()}
    if type(value) is str:
        return {"type": "text", "value": value}
    if type(value) is bytes:
        return {"type": "blob", "value": value.hex()}
    raise ScreenError("results.db contains an unsupported SQLite value")


def _canonical_results_database(contents: bytes) -> dict[str, Any]:
    """Reduce a candidate ``results.db`` to a canonical, comparison-safe JSON form.

    The bytes are deserialized into a fresh in-memory connection with
    ``trusted_schema`` off and ``query_only`` on, must pass
    ``integrity_check``, and may contain only plain tables and indexes —
    views, triggers, and virtual tables are rejected, since those can run
    attacker-chosen SQL when the database is queried.  Exactly one
    ``_metadata_`` table is required; its schema row, column list, and rows
    (ordered by id then seed) are returned with every value tagged by SQLite
    storage class, floats and blobs rendered as hex, so the later equality
    check against the preflight contract is exact rather than subject to
    float formatting or column affinity.
    """
    if not contents or len(contents) > 64 * 1024**2:
        raise ScreenError("results.db is empty or exceeds its byte bound")
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(contents)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ScreenError("results.db failed SQLite integrity_check")
        schema = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        if any(
            row[0] not in {"index", "table"}
            or (
                row[0] == "table"
                and type(row[3]) is str
                and "VIRTUAL TABLE" in row[3].upper()
            )
            for row in schema
        ):
            raise ScreenError("results.db contains a trigger, view, or unsupported schema object")
        metadata_schema = [
            row for row in schema if row[0] == "table" and row[1] == "_metadata_"
        ]
        if len(metadata_schema) != 1:
            raise ScreenError("results.db must contain exactly one _metadata_ table")
        columns = connection.execute('PRAGMA table_info("_metadata_")').fetchall()
        names = [str(row[1]) for row in columns]
        if "seed" not in names or "id" not in names:
            raise ScreenError("results.db omits seed/id metadata columns")
        quoted = ", ".join('"' + name.replace('"', '""') + '"' for name in names)
        rows = connection.execute(
            f'SELECT {quoted} FROM "_metadata_" ORDER BY "id", "seed"'
        ).fetchall()
        return {
            "schema": [list(row) for row in metadata_schema],
            "columns": [list(row) for row in columns],
            "rows": [
                [_tagged_sqlite_value(value) for value in row]
                for row in rows
            ],
        }
    except sqlite3.DatabaseError as error:
        raise ScreenError("results.db is not a valid SQLite metadata database") from error
    finally:
        connection.close()


def _result_root(value: Any) -> str:
    relative = _normalized_relative_path(value, "result root")
    path = PurePosixPath(relative)
    if len(path.parts) < 2 or path.parts[0] != "results":
        raise ScreenError("result root must be nested below results/")
    return relative


def _validate_payload_components(
    payload: Path,
    seeds: Sequence[int],
    horizon: int,
    *,
    entrypoint: str,
    result_root: str,
    metadata_contract: Mapping[str, Any],
) -> list[Path]:
    if type(entrypoint) is not str or entrypoint not in {
        "src/continuing_main.py",
        "src/rtu_ppo.py",
    }:
        raise ScreenError("payload validation entrypoint is unsupported")
    files = _payload_files(payload)
    canonical_root = _result_root(result_root)
    expected_npz = [f"{canonical_root}/data/{seed}.npz" for seed in seeds]
    expected_database = f"{canonical_root}/results.db"
    discovered = [path.relative_to(payload).as_posix() for path in files]
    expected_files = sorted([*expected_npz, expected_database])
    if discovered != expected_files:
        raise ScreenError(
            "payload must contain only the ordered seed NPZs and one sibling results.db"
        )
    expected_directories = _parent_directories(expected_files)
    if entrypoint == "src/rtu_ppo.py":
        expected_directories = sorted(
            [*expected_directories, f"{canonical_root}/videos"]
        )
    if _directory_inventory(payload) != expected_directories:
        raise ScreenError("payload contains an unexpected or missing result directory")
    archives = [payload / relative for relative in expected_npz]
    for path in archives:
        _validate_zip_structure(path, horizon)
    database = payload / expected_database
    actual_metadata = _canonical_results_database(
        _read_stable_regular_file(
            database,
            "results.db",
            maximum=64 * 1024**2,
        )
    )
    if actual_metadata != metadata_contract:
        raise ScreenError("results.db rows/config metadata differ from preflight")
    return archives


def validate_reward_archives(
    payload: Path,
    seeds: Sequence[int],
    horizon: int,
    *,
    entrypoint: str,
    result_root: str | None = None,
    metadata_contract: Mapping[str, Any] | None = None,
    scorer_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate exact upstream artifacts against pinned-image scorer records."""

    if scorer_records is None:
        raise ScreenError("reward eligibility requires records from the pinned-image scorer")
    if metadata_contract is None:
        raise ScreenError("reward eligibility requires preflight-bound results.db metadata")
    files = _payload_files(payload)
    if result_root is None:
        archives = [path for path in files if path.suffix == ".npz"]
        roots = {path.parent.parent.relative_to(payload).as_posix() for path in archives}
        if len(roots) != 1:
            raise ScreenError("raw NPZ archives do not derive one result root")
        result_root = next(iter(roots))
    canonical_root = _result_root(result_root)
    expected_npz = [f"{canonical_root}/data/{seed}.npz" for seed in seeds]
    _validate_payload_components(
        payload,
        seeds,
        horizon,
        entrypoint=entrypoint,
        result_root=canonical_root,
        metadata_contract=metadata_contract,
    )

    if len(scorer_records) != len(seeds):
        raise ScreenError("pinned scorer record count drift")
    expected_record_keys = {
        "archive_path",
        "ema_sample_count",
        "ema_tail_sample_count",
        "ema_tail_start_index",
        "final_unadjusted_ema",
        "fov_last_10pct_ema_auc",
        "npz_sha256",
        "npz_size_bytes",
        "reward_dtype",
        "reward_shape",
        "reward_sum_float64",
        "reward_trace_sha256",
        "seed",
    }
    sample_count = (horizon + 99) // 100
    tail_start = int((1.0 - 0.1) * sample_count)
    results: list[dict[str, Any]] = []
    for position, seed in enumerate(seeds):
        path = payload / expected_npz[position]
        record = dict(scorer_records[position])
        digest, size = _stable_file_record(
            path,
            f"reward NPZ for seed {seed}",
            maximum=_MAX_NPZ_BYTES,
        )
        numeric_fields = (
            "reward_sum_float64",
            "fov_last_10pct_ema_auc",
            "final_unadjusted_ema",
        )
        if (
            set(record) != expected_record_keys
            or record.get("seed") != seed
            or record.get("archive_path")
            != (PurePosixPath("payload") / expected_npz[position]).as_posix()
            or record.get("npz_sha256") != digest
            or record.get("npz_size_bytes") != size
            or record.get("reward_shape") != [horizon]
            or record.get("ema_sample_count") != sample_count
            or record.get("ema_tail_start_index") != tail_start
            or record.get("ema_tail_sample_count") != sample_count - tail_start
            or not all(
                type(record.get(field)) is float
                and math.isfinite(cast(float, record[field]))
                for field in numeric_fields
            )
            or _SHA256_RE.fullmatch(cast(str, record.get("reward_trace_sha256", "")))
            is None
            or type(record.get("reward_dtype")) is not str
        ):
            raise ScreenError("pinned scorer record does not bind the exact NPZ and metric")
        results.append(record)
    return results


def _artifact_inventory(run_root: Path) -> list[dict[str, Any]]:
    excluded = {"run_manifest.json", "run_manifest.json.sha256"}
    records: list[dict[str, Any]] = []
    for path in _payload_files(run_root):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        digest, size = _stable_file_record(path, f"run artifact {relative}")
        records.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return records


def _container_argv(protocol: FrozenProtocol, config: FrozenConfiguration) -> list[str]:
    return [
        f"{_CONTAINER_SOURCE_ROOT}/{config.entrypoint}",
        "--exp",
        f"/protocol-input/{config.path}",
        "--idxs",
        protocol.index_argument,
        "--save_path",
        "/run-output/results",
        "--checkpoint_path",
        "/tmp/checkpoints",
        "--silent",
    ]


def _scorer_container_argv(
    protocol: FrozenProtocol,
    result_root: str,
) -> list[str]:
    command = [
        _SCORER_CONTAINER_PATH,
        "--payload-root",
        "/run-output",
        "--result-root",
        _result_root(result_root),
        "--horizon",
        str(protocol.horizon),
    ]
    for seed in protocol.seeds:
        command.extend(["--seed", str(seed)])
    return command


def build_scorer_command(
    protocol: FrozenProtocol,
    payload_dir: Path,
    result_root: str,
    docker: str,
) -> list[str]:
    """Return the exact pinned-image raw-reward scoring command."""

    if protocol.schema not in EXECUTABLE_CPU_SCHEMAS or protocol.backend != "cpu":
        raise ScreenError("scoring requires an explicitly frozen CPU v3 protocol")
    if protocol.scorer_path is None or protocol.scorer_sha256 is None:
        raise ScreenError("CPU protocol has no hash-bound scorer")
    _reject_docker_mount_unsafe_path(protocol.scorer_path, "scorer path")
    _reject_docker_mount_unsafe_path(payload_dir, "candidate payload path")
    return [
        *_sandbox_prefix(protocol, docker),
        "--mount",
        f"type=bind,src={payload_dir},dst=/run-output,readonly",
        "--mount",
        f"type=bind,src={protocol.scorer_path},dst={_SCORER_CONTAINER_PATH},readonly",
        protocol.image_id,
        *_scorer_container_argv(protocol, result_root),
    ]


def _preflight_configuration(
    protocol: FrozenProtocol,
    plan: Mapping[str, Any],
    config: FrozenConfiguration,
) -> dict[str, Any]:
    preflight = _require_dict(plan.get("preflight"), "screen_plan.preflight")
    result = _require_dict(preflight.get("result"), "screen_plan.preflight.result")
    records = _require_list(result.get("configurations"), "preflight.configurations")
    expected_paths = [candidate.path for candidate in protocol.configurations]
    actual_paths = [
        _require_dict(record, "preflight configuration").get("path") for record in records
    ]
    if actual_paths != expected_paths:
        raise ScreenError("screen plan preflight configuration inventory drift")
    position = expected_paths.index(config.path)
    record = _require_dict(records[position], f"preflight configuration {config.path}")
    expected_keys = {
        "agent",
        "effective_seeds",
        "entrypoint",
        "metadata_contract",
        "num_permutations",
        "num_updates",
        "path",
        "result_root",
        "rollout_steps",
        "sha256",
        "stored_seeds",
    }
    if set(record) != expected_keys:
        raise ScreenError(f"preflight configuration field inventory drift: {config.path}")
    if (
        record.get("path") != config.path
        or record.get("sha256") != config.sha256
        or record.get("agent") != config.agent
        or record.get("entrypoint") != config.entrypoint
        or record.get("num_permutations") != 1
        or record.get("stored_seeds") != list(protocol.seeds)
        or record.get("effective_seeds") != list(protocol.seeds)
    ):
        raise ScreenError(f"preflight configuration binding drift: {config.path}")
    _result_root(record.get("result_root"))
    _require_dict(record.get("metadata_contract"), "preflight metadata_contract")
    return record


def _parse_scorer_capture(
    protocol: FrozenProtocol,
    result_root: str,
    capture: ProcessCapture,
) -> dict[str, Any]:
    if capture.returncode != 0:
        detail = capture.stderr.decode("utf-8", errors="replace").strip()
        raise ScreenError(f"pinned-image scorer failed with exit {capture.returncode}: {detail}")
    payload = _load_json_bytes(capture.stdout, "pinned-image scorer stdout")
    if _canonical_json(payload) != capture.stdout:
        raise ScreenError("pinned-image scorer stdout is not canonical JSON")
    if set(payload) != {"horizon", "records", "result_root", "schema_version", "seeds"}:
        raise ScreenError("pinned-image scorer result field inventory drift")
    if (
        payload.get("schema_version") != SCORING_SCHEMA
        or payload.get("horizon") != protocol.horizon
        or payload.get("seeds") != list(protocol.seeds)
        or payload.get("result_root") != _result_root(result_root)
    ):
        raise ScreenError("pinned-image scorer result contract drift")
    records = _require_list(payload.get("records"), "pinned-image scorer records")
    if len(records) != len(protocol.seeds) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ScreenError("pinned-image scorer record inventory drift")
    return payload


def _recompute_scoring_outcome(
    protocol: FrozenProtocol,
    payload: Path,
    result_root: str,
    entrypoint: str,
    metadata_contract: Mapping[str, Any],
    capture: ProcessCapture,
) -> tuple[dict[str, Any] | None, str | None]:
    scored: dict[str, Any] | None = None
    try:
        scored = _parse_scorer_capture(protocol, result_root, capture)
        records = _require_list(scored["records"], "scorer records")
        validate_reward_archives(
            payload,
            protocol.seeds,
            protocol.horizon,
            entrypoint=entrypoint,
            result_root=result_root,
            metadata_contract=metadata_contract,
            scorer_records=[
                _require_dict(record, "scorer record") for record in records
            ],
        )
    except ScreenError as error:
        return scored, str(error)
    assert scored is not None
    return scored, None


def _input_contract(
    protocol: FrozenProtocol,
    config: FrozenConfiguration,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    preflight_config = _preflight_configuration(protocol, plan, config)
    return {
        "protocol_schema": protocol.schema,
        "protocol_sha256": protocol.sha256,
        "base_protocol_sha256": protocol.base_protocol_sha256,
        "predecessor_protocol_sha256": protocol.predecessor_protocol_sha256,
        "configuration_path": config.path,
        "configuration_sha256": config.sha256,
        "agent": config.agent,
        "entrypoint": config.entrypoint,
        "horizon_per_seed": protocol.horizon,
        "seeds": list(protocol.seeds),
        "indices_argument": protocol.index_argument,
        "image_id": protocol.image_id,
        "source_files": [
            {"path": path, "sha256": digest} for path, digest in protocol.source_files
        ],
        "screen_plan_sha256": _sha256_bytes(_canonical_json(plan)),
        "input_snapshot": plan.get("input_snapshot"),
        "development_image": plan.get("development_image"),
        "docker_runtime": plan.get("docker_runtime"),
        "host_runtime": plan.get("host_runtime"),
        "sandbox": _sandbox_contract(protocol),
        "container_argv": _container_argv(protocol, config),
        "scoring": {
            "implementation_sha256": protocol.scorer_sha256,
            "runtime": "exact_development_image",
            "container_argv": _scorer_container_argv(
                protocol,
                cast(str, preflight_config["result_root"]),
            ),
        },
        "result_contract": preflight_config,
        "metric_contract": _metric_contract(protocol),
        "harness": plan.get("harness"),
    }


def _input_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(contract))


def _attempt_payload(
    protocol: FrozenProtocol,
    config: FrozenConfiguration,
    contract: dict[str, Any],
    command: Sequence[str],
    *,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "classification": "open_development_nonpromoting",
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "started_at_utc": _utc_now() if started_at_utc is None else started_at_utc,
        "configuration": config.path,
        "input_contract": contract,
        "input_contract_sha256": _input_contract_sha256(contract),
        "host_command": list(command),
        "reward_horizon_per_seed": protocol.horizon,
    }


def _execute_candidate(
    protocol: FrozenProtocol,
    config: FrozenConfiguration,
    output: Path,
    docker: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one candidate under the staged-attempt discipline.

    Everything happens under ``.incomplete/<run_id>`` — exclusive attempt
    record, process logs, payload validation, in-image scoring — and only a
    fully manifested attempt is atomically renamed into ``runs/``.  A crash
    leaves the staging directory behind as a non-resumable failure.  A
    candidate that exits nonzero or fails validation is not an error: it
    becomes a ``completed_ineligible`` run with its failures recorded, so one
    bad candidate cannot block the panel.  The bound runtime identity is
    re-verified after the candidate process and again before scoring.
    """
    incomplete_parent = output / ".incomplete"
    runs_parent = output / "runs"
    incomplete_parent.mkdir(exist_ok=True)
    runs_parent.mkdir(exist_ok=True)
    for parent, label in (
        (incomplete_parent, ".incomplete"),
        (runs_parent, "runs"),
    ):
        if parent.is_symlink() or not parent.is_dir():
            raise ScreenError(f"{label} must be a non-symlink directory")
    staging = incomplete_parent / config.run_id
    final = runs_parent / config.run_id
    if staging.exists() or staging.is_symlink():
        raise ScreenError(f"incomplete prior attempt exists and cannot be resumed: {staging}")
    if final.exists() or final.is_symlink():
        raise ScreenError(f"completed run already exists unexpectedly: {final}")
    staging.mkdir(mode=0o755)
    payload = staging / "payload"
    payload.mkdir(mode=0o777)
    payload.chmod(0o777)
    command = build_candidate_command(protocol, config, payload.resolve(), docker)
    contract = _input_contract(protocol, config, plan)
    attempt = _attempt_payload(protocol, config, contract, command)
    _ensure_exclusive_file(staging / "attempt.json", _canonical_json(attempt))

    stdout_path = staging / "stdout.log"
    stderr_path = staging / "stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        returncode = _run_process_to_files(command, stdout, stderr)
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    _verify_bound_runtime_identity(protocol, plan, docker)

    failures: list[str] = []
    rewards: list[dict[str, Any]] = []
    scoring: dict[str, Any] = {
        "executed": False,
        "implementation_sha256": protocol.scorer_sha256,
        "runtime": "exact_development_image",
    }
    if returncode != 0:
        failures.append(f"process_exit_code:{returncode}")
    else:
        preflight_config = _preflight_configuration(protocol, plan, config)
        result_root = cast(str, preflight_config["result_root"])
        metadata_contract = _require_dict(
            preflight_config["metadata_contract"],
            "preflight metadata_contract",
        )
        try:
            _validate_payload_components(
                payload,
                protocol.seeds,
                protocol.horizon,
                entrypoint=config.entrypoint,
                result_root=result_root,
                metadata_contract=metadata_contract,
            )
        except ScreenError as error:
            failures.append(f"raw_reward_validation:{error}")
        if not failures:
            _verify_bound_runtime_identity(protocol, plan, docker)
            scoring_command = build_scorer_command(
                protocol,
                payload.resolve(),
                result_root,
                docker,
            )
            scorer_capture = _capture_process(scoring_command)
            _ensure_exclusive_file(staging / "scorer.stdout.log", scorer_capture.stdout)
            _ensure_exclusive_file(staging / "scorer.stderr.log", scorer_capture.stderr)
            scoring = {
                "executed": True,
                "implementation_sha256": protocol.scorer_sha256,
                "runtime": "exact_development_image",
                "command": scoring_command,
                "exit_code": scorer_capture.returncode,
                "stdout_path": "scorer.stdout.log",
                "stderr_path": "scorer.stderr.log",
                "result": None,
            }
            try:
                scored = _parse_scorer_capture(protocol, result_root, scorer_capture)
                scoring["result"] = scored
                scorer_records = _require_list(scored["records"], "scorer records")
                rewards = validate_reward_archives(
                    payload,
                    protocol.seeds,
                    protocol.horizon,
                    entrypoint=config.entrypoint,
                    result_root=result_root,
                    metadata_contract=metadata_contract,
                    scorer_records=[
                        _require_dict(record, "scorer record") for record in scorer_records
                    ],
                )
            except ScreenError as error:
                failures.append(f"pinned_image_scoring:{error}")

    status_value = "completed_eligible" if not failures else "completed_ineligible"
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "classification": "open_development_nonpromoting",
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "status": status_value,
        "configuration": {
            "path": config.path,
            "sha256": config.sha256,
            "agent": config.agent,
            "entrypoint": config.entrypoint,
        },
        "input_contract": contract,
        "input_contract_sha256": _input_contract_sha256(contract),
        "process": {
            "exit_code": returncode,
            "completed_at_utc": _utc_now(),
            "stdout_path": "stdout.log",
            "stderr_path": "stderr.log",
        },
        "eligibility_failures": failures,
        "scoring": scoring,
        "raw_reward_validation": rewards,
        "metric_contract": _metric_contract(protocol),
        "collector_summaries_used": False,
    }
    manifest["directories"] = _directory_inventory(staging)
    manifest["artifacts"] = _artifact_inventory(staging)
    _write_manifest_pair(staging / "run_manifest.json", manifest)
    os.rename(staging, final)
    return manifest


def _validate_completed_run(
    protocol: FrozenProtocol,
    config: FrozenConfiguration,
    run_root: Path,
    plan: Mapping[str, Any],
    docker: str,
) -> dict[str, Any]:
    """Re-validate a previously completed run before a resumed harness accepts it.

    Resume never re-executes: an existing run is either byte-exact —
    manifest keys and classification, input contract and its digest,
    artifact/directory inventories, canonical ``attempt.json``, and the
    recorded scoring all re-verified against the current protocol and plan —
    or the whole resume fails.  This is what makes completed attempts
    immutable: any tampering or contract drift is indistinguishable from
    corruption and rejected the same way.
    """
    if run_root.is_symlink() or not run_root.is_dir():
        raise ScreenError(f"completed run must be a non-symlink directory: {run_root}")
    manifest = _validate_manifest_pair(run_root / "run_manifest.json", RUN_SCHEMA)
    expected_manifest_keys = {
        "artifacts",
        "classification",
        "collector_summaries_used",
        "configuration",
        "directories",
        "eligibility_failures",
        "input_contract",
        "input_contract_sha256",
        "metric_contract",
        "process",
        "raw_reward_validation",
        "schema_version",
        "scientific_promotion_allowed",
        "scoring",
        "sota_claim_allowed",
        "status",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("classification") != "open_development_nonpromoting"
        or manifest.get("scientific_promotion_allowed") is not False
        or manifest.get("sota_claim_allowed") is not False
    ):
        raise ScreenError(f"completed run manifest field/classification drift: {config.path}")
    configuration = _require_dict(manifest.get("configuration"), "run.configuration")
    expected_configuration = {
        "path": config.path,
        "sha256": config.sha256,
        "agent": config.agent,
        "entrypoint": config.entrypoint,
    }
    if configuration != expected_configuration:
        raise ScreenError(f"completed run configuration drift: {config.path}")
    contract = _input_contract(protocol, config, plan)
    if manifest.get("input_contract") != contract:
        raise ScreenError(f"completed run input contract drift: {config.path}")
    if manifest.get("input_contract_sha256") != _input_contract_sha256(contract):
        raise ScreenError(f"completed run contract hash drift: {config.path}")
    status_value = manifest.get("status")
    if status_value not in {"completed_eligible", "completed_ineligible"}:
        raise ScreenError(f"completed run status is invalid: {config.path}")
    if manifest.get("artifacts") != _artifact_inventory(run_root):
        raise ScreenError(f"completed run artifacts are not byte-identical: {config.path}")
    if manifest.get("directories") != _directory_inventory(run_root):
        raise ScreenError(f"completed run directory inventory drift: {config.path}")
    attempt_raw = _read_stable_regular_file(
        run_root / "attempt.json",
        "attempt.json",
        maximum=16 * 1024**2,
    )
    attempt = _load_json_bytes(attempt_raw, "attempt.json")
    if _canonical_json(attempt) != attempt_raw:
        raise ScreenError(f"attempt.json is not canonical: {config.path}")
    started_at = attempt.get("started_at_utc")
    if (
        type(started_at) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", started_at) is None
    ):
        raise ScreenError(f"attempt timestamp is invalid: {config.path}")
    output = run_root.parent.parent
    execution_payload = output / ".incomplete" / config.run_id / "payload"
    expected_command = build_candidate_command(protocol, config, execution_payload, docker)
    expected_attempt = _attempt_payload(
        protocol,
        config,
        contract,
        expected_command,
        started_at_utc=started_at,
    )
    if attempt != expected_attempt:
        raise ScreenError(f"attempt projection or host command drift: {config.path}")
    process = _require_dict(manifest.get("process"), "run.process")
    completed_at = process.get("completed_at_utc")
    if (
        set(process) != {"completed_at_utc", "exit_code", "stderr_path", "stdout_path"}
        or process.get("stdout_path") != "stdout.log"
        or process.get("stderr_path") != "stderr.log"
        or type(completed_at) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", completed_at) is None
    ):
        raise ScreenError(f"completed run log path contract drift: {config.path}")
    failures = _require_list(manifest.get("eligibility_failures"), "run.eligibility_failures")
    rewards = _require_list(manifest.get("raw_reward_validation"), "run.raw_reward_validation")
    scoring = _require_dict(manifest.get("scoring"), "run.scoring")
    if manifest.get("metric_contract") != _metric_contract(protocol):
        raise ScreenError(f"completed run metric contract drift: {config.path}")
    if manifest.get("collector_summaries_used") is not False:
        raise ScreenError(f"completed run used collector summaries: {config.path}")
    if status_value == "completed_eligible":
        if type(process.get("exit_code")) is not int or process.get("exit_code") != 0 or failures:
            raise ScreenError(f"eligible run has process or eligibility failures: {config.path}")
        preflight_config = _preflight_configuration(protocol, plan, config)
        result_root = cast(str, preflight_config["result_root"])
        metadata_contract = _require_dict(
            preflight_config["metadata_contract"],
            "preflight metadata_contract",
        )
        expected_scoring_command = build_scorer_command(
            protocol,
            execution_payload,
            result_root,
            docker,
        )
        if (
            set(scoring)
            != {
                "command",
                "executed",
                "exit_code",
                "implementation_sha256",
                "result",
                "runtime",
                "stderr_path",
                "stdout_path",
            }
            or scoring.get("executed") is not True
            or scoring.get("implementation_sha256") != protocol.scorer_sha256
            or scoring.get("runtime") != "exact_development_image"
            or scoring.get("command") != expected_scoring_command
            or scoring.get("exit_code") != 0
            or scoring.get("stdout_path") != "scorer.stdout.log"
            or scoring.get("stderr_path") != "scorer.stderr.log"
        ):
            raise ScreenError(f"eligible run scorer execution drift: {config.path}")
        stored_capture = ProcessCapture(
            0,
            _read_stable_regular_file(run_root / "scorer.stdout.log", "scorer stdout"),
            _read_stable_regular_file(run_root / "scorer.stderr.log", "scorer stderr"),
        )
        stored_scoring = _parse_scorer_capture(protocol, result_root, stored_capture)
        if scoring.get("result") != stored_scoring:
            raise ScreenError(f"eligible run scorer result/log drift: {config.path}")
        current_command = build_scorer_command(
            protocol,
            (run_root / "payload").resolve(strict=True),
            result_root,
            docker,
        )
        _verify_bound_runtime_identity(protocol, plan, docker)
        current_scoring = _parse_scorer_capture(
            protocol,
            result_root,
            _capture_process(current_command),
        )
        if current_scoring != stored_scoring:
            raise ScreenError(f"eligible run pinned-image rescoring drift: {config.path}")
        current_records = _require_list(current_scoring["records"], "scorer records")
        recomputed = validate_reward_archives(
            run_root / "payload",
            protocol.seeds,
            protocol.horizon,
            entrypoint=config.entrypoint,
            result_root=result_root,
            metadata_contract=metadata_contract,
            scorer_records=[
                _require_dict(record, "scorer record") for record in current_records
            ],
        )
        if rewards != recomputed:
            raise ScreenError(f"eligible run metric does not match raw rewards: {config.path}")
    else:
        if not failures:
            raise ScreenError(f"ineligible run does not declare a failure: {config.path}")
        if rewards:
            raise ScreenError(
                f"ineligible run unexpectedly records eligible rewards: {config.path}"
            )
        exit_code = process.get("exit_code")
        if type(exit_code) is not int:
            raise ScreenError(f"ineligible run exit code is invalid: {config.path}")
        if exit_code != 0 and failures != [f"process_exit_code:{exit_code}"]:
            raise ScreenError(
                f"ineligible run does not exactly bind its process failure: {config.path}"
            )
        allowed_prefixes = ("raw_reward_validation:", "pinned_image_scoring:")
        if exit_code == 0 and (
            len(failures) != 1
            or type(failures[0]) is not str
            or not failures[0].startswith(allowed_prefixes)
        ):
            raise ScreenError(f"ineligible run has an invalid trace failure: {config.path}")
        preflight_config = _preflight_configuration(protocol, plan, config)
        result_root = cast(str, preflight_config["result_root"])
        metadata_contract = _require_dict(
            preflight_config["metadata_contract"],
            "preflight metadata_contract",
        )
        if scoring.get("executed") is False:
            if scoring != {
                "executed": False,
                "implementation_sha256": protocol.scorer_sha256,
                "runtime": "exact_development_image",
            }:
                raise ScreenError(f"ineligible run nonexecution scorer drift: {config.path}")
        elif scoring.get("executed") is True:
            expected_scoring_command = build_scorer_command(
                protocol,
                execution_payload,
                result_root,
                docker,
            )
            if (
                set(scoring)
                != {
                    "command",
                    "executed",
                    "exit_code",
                    "implementation_sha256",
                    "result",
                    "runtime",
                    "stderr_path",
                    "stdout_path",
                }
                or scoring.get("implementation_sha256") != protocol.scorer_sha256
                or scoring.get("runtime") != "exact_development_image"
                or scoring.get("command") != expected_scoring_command
                or type(scoring.get("exit_code")) is not int
                or scoring.get("stdout_path") != "scorer.stdout.log"
                or scoring.get("stderr_path") != "scorer.stderr.log"
            ):
                raise ScreenError(f"ineligible run scorer projection drift: {config.path}")
        else:
            raise ScreenError(f"ineligible run scorer execution flag drift: {config.path}")
        if exit_code == 0:
            layout_error: ScreenError | None = None
            try:
                _validate_payload_components(
                    run_root / "payload",
                    protocol.seeds,
                    protocol.horizon,
                    entrypoint=config.entrypoint,
                    result_root=result_root,
                    metadata_contract=metadata_contract,
                )
            except ScreenError as error:
                layout_error = error
            failure = cast(str, failures[0])
            if failure.startswith("raw_reward_validation:"):
                if scoring.get("executed") is not False or layout_error is None:
                    raise ScreenError(
                        f"ineligible raw-validation outcome is stale: {config.path}"
                    )
            else:
                if scoring.get("executed") is not True or layout_error is not None:
                    raise ScreenError(
                        f"ineligible scorer outcome no longer has valid raw inputs: {config.path}"
                    )
                stored_capture = ProcessCapture(
                    cast(int, scoring["exit_code"]),
                    _read_stable_regular_file(
                        run_root / "scorer.stdout.log",
                        "scorer stdout",
                    ),
                    _read_stable_regular_file(
                        run_root / "scorer.stderr.log",
                        "scorer stderr",
                    ),
                )
                stored_result, stored_failure = _recompute_scoring_outcome(
                    protocol,
                    run_root / "payload",
                    result_root,
                    config.entrypoint,
                    metadata_contract,
                    stored_capture,
                )
                if (
                    stored_failure is None
                    or failure != f"pinned_image_scoring:{stored_failure}"
                    or scoring.get("result") != stored_result
                ):
                    raise ScreenError(
                        f"persisted ineligible scorer outcome drift: {config.path}"
                    )
                current_command = build_scorer_command(
                    protocol,
                    (run_root / "payload").resolve(strict=True),
                    result_root,
                    docker,
                )
                _verify_bound_runtime_identity(protocol, plan, docker)
                current_capture = _capture_process(current_command)
                _, current_failure = _recompute_scoring_outcome(
                    protocol,
                    run_root / "payload",
                    result_root,
                    config.entrypoint,
                    metadata_contract,
                    current_capture,
                )
                if (
                    current_failure is None
                    or current_capture.returncode != stored_capture.returncode
                    or current_capture.stdout != stored_capture.stdout
                ):
                    raise ScreenError(
                        f"ineligible pinned-image scorer outcome is stale: {config.path}"
                    )
    return manifest


def _all_run_manifests(
    protocol: FrozenProtocol,
    output: Path,
    plan: Mapping[str, Any],
    docker: str,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for config in protocol.configurations:
        run_root = output / "runs" / config.run_id
        if not run_root.exists():
            raise ScreenError(
                "aggregate is forbidden until every frozen configuration has one "
                "completed attempt; "
                f"missing {config.path}"
            )
        manifests.append(_validate_completed_run(protocol, config, run_root, plan, docker))
    return manifests


def _aggregate_payload(
    protocol: FrozenProtocol, manifests: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for manifest in manifests:
        configuration = _require_dict(manifest["configuration"], "run.configuration")
        path = cast(str, configuration["path"])
        run_manifest_sha256 = _sha256_bytes(_canonical_json(manifest))
        if manifest["status"] == "completed_eligible":
            reward_records = _require_list(
                manifest["raw_reward_validation"], "run.raw_reward_validation"
            )
            seed_scores = [
                float(_require_dict(record, "raw reward record")["fov_last_10pct_ema_auc"])
                for record in reward_records
            ]
            if len(seed_scores) != len(protocol.seeds) or not all(
                math.isfinite(value) for value in seed_scores
            ):
                raise ScreenError(f"eligible candidate has an invalid seed score set: {path}")
            eligible.append(
                {
                    "configuration": path,
                    "configuration_sha256": configuration["sha256"],
                    "run_manifest_sha256": run_manifest_sha256,
                    "seed_scores": [
                        {"seed": seed, "fov_last_10pct_ema_auc": value}
                        for seed, value in zip(protocol.seeds, seed_scores, strict=True)
                    ],
                    "aggregate_mean": math.fsum(seed_scores) / len(seed_scores),
                }
            )
        else:
            ineligible.append(
                {
                    "configuration": path,
                    "configuration_sha256": configuration["sha256"],
                    "run_manifest_sha256": run_manifest_sha256,
                    "eligibility_failures": manifest["eligibility_failures"],
                }
            )
    eligible.sort(
        key=lambda item: (-cast(float, item["aggregate_mean"]), cast(str, item["configuration"]))
    )
    ineligible.sort(key=lambda item: cast(str, item["configuration"]))
    ranked = [{"rank": index, **item} for index, item in enumerate(eligible, start=1)]
    advance_count = min(protocol.advance_count, len(ranked))
    advanced = [cast(str, record["configuration"]) for record in ranked[:advance_count]]
    limitations = [
        "The exact two seeds are open, consumed development seeds.",
        "The exact runtime is an unqualified development image executed on CPU.",
        "This operational screen cannot promote evidence or support a SOTA, "
        "superiority, or official claim.",
        "Candidate compute, state, replay, and optimizer budgets are not necessarily matched.",
    ]
    if protocol.schema in STATEFUL_CPU_SCHEMAS:
        limitations.append(
            "The upstream RTU-PPO path reuses one derived RNG for action sampling and "
            "environment stepping; this paired-comparison confound is source-bound but "
            "not removed."
        )
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "classification": "open_development_nonpromoting",
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "superiority_claim_allowed": False,
        "complete_frozen_candidate_set": True,
        "reward_informed_early_stopping_used": False,
        "collector_summaries_used": False,
        "protocol": {
            "schema_version": protocol.schema,
            "sha256": protocol.sha256,
            "configuration_count": len(protocol.configurations),
            "horizon_per_seed": protocol.horizon,
            "seeds": list(protocol.seeds),
        },
        "metric_contract": _metric_contract(protocol),
        "selection_rule": protocol.selection_rule,
        "eligible_ranking": ranked,
        "ineligible_candidates_rank_after_eligible": ineligible,
        "advanced_for_later_open_development_only": advanced,
        "advanced_count": advance_count,
        "limitations": limitations,
    }


def aggregate_screen(
    protocol: FrozenProtocol,
    output: Path,
    plan: Mapping[str, Any],
    docker: str,
) -> dict[str, Any]:
    """Validate all completed attempts and materialize the frozen aggregate."""

    plan_protocol = _require_dict(plan.get("protocol"), "screen_plan.protocol")
    if plan_protocol.get("sha256") != protocol.sha256:
        raise ScreenError("screen plan protocol drift")
    _validate_output_tree(protocol, output, plan, require_complete_runs=True)
    manifests = _all_run_manifests(protocol, output, plan, docker)
    aggregate = _aggregate_payload(protocol, manifests)
    path = output / "aggregate.json"
    if path.exists():
        existing = _validate_manifest_pair(path, AGGREGATE_SCHEMA)
        if existing != aggregate:
            raise ScreenError("existing aggregate is not byte-identical to recomputation")
    else:
        _write_manifest_pair(path, aggregate)
    return aggregate


def run_screen(
    protocol_dir: Path,
    output_dir: Path,
    image_id: str,
    *,
    docker: str = "docker",
) -> dict[str, Any]:
    """Execute or strictly resume every candidate, then aggregate exactly once."""

    original_protocol = load_frozen_protocol(protocol_dir)
    if (
        original_protocol.schema not in EXECUTABLE_CPU_SCHEMAS
        or original_protocol.backend != "cpu"
    ):
        raise ScreenError("execution rejects legacy, CPU v2, or backend-mismatched protocols")
    if _IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ScreenError("--image-id must be an explicit lowercase sha256:<64-hex> ID")
    if image_id != original_protocol.image_id:
        raise ScreenError("explicit --image-id does not equal the frozen protocol image ID")
    output = _resolve_output_root(output_dir, original_protocol)
    _precheck_output_before_lock(output)
    with _open_output_lock(output) as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ScreenError("another process holds the screen output lock") from error
        _precheck_output_before_lock(output)
        initial_snapshot = _prepare_protocol_snapshot(original_protocol, output)
        protocol = initial_snapshot.protocol
        plan = _prepare_plan(original_protocol, initial_snapshot, output, docker)
        execution_docker = _require_string(
            _require_dict(plan["docker_runtime"], "screen_plan.docker_runtime").get(
                "executable_path"
            ),
            "screen_plan.docker_runtime.executable_path",
        )
        _validate_output_tree(protocol, output, plan, require_complete_runs=False)
        for config in protocol.configurations:
            current_snapshot = _prepare_protocol_snapshot(original_protocol, output)
            if current_snapshot.inventory_sha256 != initial_snapshot.inventory_sha256:
                raise ScreenError("frozen inputs changed before candidate execution")
            _verify_runtime_identity(
                original_protocol,
                initial_snapshot,
                output,
                plan,
                execution_docker,
            )
            final = output / "runs" / config.run_id
            incomplete = output / ".incomplete" / config.run_id
            if incomplete.exists() or incomplete.is_symlink():
                raise ScreenError(
                    f"cannot resume an incomplete attempt; use a new output root: {incomplete}"
                )
            if final.exists() or final.is_symlink():
                _validate_completed_run(protocol, config, final, plan, execution_docker)
            else:
                _execute_candidate(protocol, config, output, execution_docker, plan)
            _verify_live_sources_match_snapshot(
                original_protocol,
                initial_snapshot,
                output,
                f"while processing candidate {config.path}",
            )
        final_snapshot = _prepare_protocol_snapshot(original_protocol, output)
        if final_snapshot.inventory_sha256 != initial_snapshot.inventory_sha256:
            raise ScreenError("frozen inputs changed before screen completion")
        _verify_runtime_identity(
            original_protocol,
            initial_snapshot,
            output,
            plan,
            execution_docker,
        )
        aggregate = aggregate_screen(protocol, output, plan, execution_docker)
        post_aggregate_snapshot = _prepare_protocol_snapshot(original_protocol, output)
        if post_aggregate_snapshot.inventory_sha256 != initial_snapshot.inventory_sha256:
            raise ScreenError("frozen inputs changed during final aggregation")
        _verify_runtime_identity(
            original_protocol,
            initial_snapshot,
            output,
            plan,
            execution_docker,
        )
        return aggregate


def validate_screen(
    protocol_dir: Path,
    output_dir: Path,
    *,
    docker: str = "docker",
) -> dict[str, Any]:
    """Read-only validation and recomputation of an already completed screen."""

    original_protocol = load_frozen_protocol(protocol_dir)
    if (
        original_protocol.schema not in EXECUTABLE_CPU_SCHEMAS
        or original_protocol.backend != "cpu"
    ):
        raise ScreenError("result validation requires the exact CPU v3 protocol")
    if output_dir.is_symlink():
        raise ScreenError("screen output root must not be a symlink")
    output = output_dir.resolve(strict=True)
    _reject_known_pinned_output(output)
    _precheck_output_before_lock(output)
    with _open_output_lock(output, create=False) as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ScreenError("another process holds the screen output lock") from error
        _precheck_output_before_lock(output)
        _validate_manifest_pair(output / "screen_plan.json", PLAN_SCHEMA)
        snapshot = _prepare_protocol_snapshot(
            original_protocol,
            output,
            allow_create=False,
        )
        protocol = snapshot.protocol
        plan = _prepare_plan(original_protocol, snapshot, output, docker)
        execution_docker = _require_string(
            _require_dict(plan["docker_runtime"], "screen_plan.docker_runtime").get(
                "executable_path"
            ),
            "screen_plan.docker_runtime.executable_path",
        )
        _validate_output_tree(protocol, output, plan, require_complete_runs=True)
        manifests = _all_run_manifests(protocol, output, plan, execution_docker)
        aggregate = _aggregate_payload(protocol, manifests)
        persisted = _validate_manifest_pair(output / "aggregate.json", AGGREGATE_SCHEMA)
        if persisted != aggregate:
            raise ScreenError("persisted aggregate is not byte-identical to recomputation")
        final_snapshot = _prepare_protocol_snapshot(
            original_protocol,
            output,
            allow_create=False,
        )
        if final_snapshot.inventory_sha256 != snapshot.inventory_sha256:
            raise ScreenError("frozen inputs changed during result validation")
        _verify_runtime_identity(
            original_protocol,
            snapshot,
            output,
            plan,
            execution_docker,
        )
        return {
            "status": "valid",
            "protocol_sha256": protocol.sha256,
            "configuration_count": len(protocol.configurations),
            "eligible_count": len(aggregate["eligible_ranking"]),
            "ineligible_count": len(aggregate["ineligible_candidates_rank_after_eligible"]),
            "aggregate_sha256": _stable_file_record(
                output / "aggregate.json",
                "aggregate.json",
            )[0],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict, resumable, nonpromoting CPU OCI harness for the two frozen current-Foragax "
            "FOV development screens."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute/resume every candidate and aggregate")
    run.add_argument("--protocol-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--image-id", required=True)
    run.add_argument("--docker", default="docker")
    validate_protocol = subparsers.add_parser(
        "validate-protocol", help="validate frozen protocol/configuration bytes without execution"
    )
    validate_protocol.add_argument("--protocol-dir", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate-results", help="read-only validation and exact aggregate recomputation"
    )
    validate.add_argument("--protocol-dir", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--docker", default="docker")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "run":
            aggregate = run_screen(
                args.protocol_dir,
                args.output_dir,
                args.image_id,
                docker=args.docker,
            )
            payload = {
                "status": "completed",
                "classification": "open_development_nonpromoting",
                "configuration_count": len(aggregate["eligible_ranking"])
                + len(aggregate["ineligible_candidates_rank_after_eligible"]),
                "eligible_count": len(aggregate["eligible_ranking"]),
                "ineligible_count": len(aggregate["ineligible_candidates_rank_after_eligible"]),
                "aggregate_path": (args.output_dir / "aggregate.json").resolve().as_posix(),
            }
        elif args.command == "validate-protocol":
            protocol = load_frozen_protocol(args.protocol_dir)
            payload = {
                "status": "valid",
                "classification": "open_development_nonpromoting",
                "schema_version": protocol.schema,
                "protocol_sha256": protocol.sha256,
                "configuration_count": len(protocol.configurations),
                "horizon_per_seed": protocol.horizon,
                "seeds": list(protocol.seeds),
                "image_id": protocol.image_id,
            }
        else:
            payload = validate_screen(
                args.protocol_dir,
                args.output_dir,
                docker=args.docker,
            )
    except (OSError, ScreenError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    if payload.get("ineligible_count", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
