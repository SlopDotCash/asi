"""Offline, non-executing readiness preflight for the matched Forager rerun.

This module never pulls, builds, loads, or runs an OCI image and never creates
benchmark output.  It records the distinction between Docker's image config
digest and a registry manifest digest, validates the frozen open-development
schedule, and reports the remaining launch blockers without authorizing a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from alberta_framework.benchmarks.forager_matched_candidate_universe import (
    MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
)
from alberta_framework.benchmarks.forager_matched_open_protocol import (
    MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS,
    MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS,
    MATCHED_CURRENT_HORIZON,
    MATCHED_CURRENT_REQUIRED_IMAGE_SHA256,
    MATCHED_CURRENT_TUNING_SEEDS,
)

SCHEMA: Final = "asi.forager_scientific_rerun_preflight.v1"
PLAN_SCHEMA: Final = "asi.forager_scientific_rerun_plan.v1"
REQUIRED_IMAGE_ID: Final = f"sha256:{MATCHED_CURRENT_REQUIRED_IMAGE_SHA256}"
SELECTION_CANDIDATES: Final = (
    MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS + MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS
)
CELL_COUNT: Final = len(SELECTION_CANDIDATES) * len(MATCHED_CURRENT_TUNING_SEEDS)
TOTAL_ENVIRONMENT_TRANSITIONS: Final = CELL_COUNT * MATCHED_CURRENT_HORIZON
TOTAL_OBSERVATION_DELIVERIES: Final = TOTAL_ENVIRONMENT_TRANSITIONS + CELL_COUNT

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REGISTRY_REFERENCE_RE = re.compile(
    r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+@sha256:[0-9a-f]{64}\Z"
)
_MAX_RECORD_BYTES: Final = 16 * 1024 * 1024
_MAX_INSPECT_BYTES: Final = 2 * 1024 * 1024
_INSPECT_TIMEOUT_SECONDS: Final = 20

# These are the immutable records that bind the frozen image for the matched
# lane.  They contain the Docker image ID, but no registry repository/manifest
# identity.  Adding a location is a protocol change and must be reviewed.
PINNED_IDENTITY_RECORDS: Final = (
    "outputs/forager/official_cpu_qualification_5eca_2000001_v1/receipt.v1.json",
    "outputs/forager/official_cpu_qualification_5eca_2000001_v1/qualification.json",
    "outputs/forager/rng_parity_live_qualification_v1_execution/receipt.json",
)
PINNED_IDENTITY_RECORD_SHA256: Final = (
    "0950c32ef4fe498bbfc7176394f7954dfbaec1ecfdfd41fdaa7ef057bdea2851",
    "0700d0cc5f884733b0bdc847290b173872915f2a939f00f9b4b9ff4aa3ed4ba6",
    "3d67fabd0d9357087c4d856c0598cf77a066847891157fd776ed43943b7641bb",
)

REQUIRED_DYNAMIC_RESOURCE_FIELDS: Final = (
    "environment_reset_count",
    "environment_transition_count",
    "observation_delivery_count",
    "agent_action_query_count",
    "model_query_count",
    "optimizer_update_count",
    "replay_insert_count",
    "replay_sample_count",
    "persistent_numeric_bytes_initial",
    "persistent_numeric_bytes_final",
    "persistent_numeric_bytes_peak",
    "raw_result_bytes",
    "elapsed_ns_telemetry_only",
)

LocalStatus = Literal["exact_present", "image_absent", "runtime_unavailable", "inspection_failed"]


class ForagerScientificRerunPreflightError(ValueError):
    """A rerun preflight input or derived record is invalid."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an exact integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise TypeError("process streams must be exact bytes")
        if len(self.stdout) > _MAX_INSPECT_BYTES or len(self.stderr) > _MAX_INSPECT_BYTES:
            raise ForagerScientificRerunPreflightError("OCI inspection output exceeds its bound")


ProcessRunner = Callable[[Sequence[str]], ProcessResult]


@dataclass(frozen=True, slots=True)
class PinnedIdentityRecord:
    path: str
    sha256: str
    size_bytes: int
    required_image_id_occurrences: int
    registry_references: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ForagerScientificRerunPreflightError("record path must be an exact string")
        _require_digest(self.sha256, "record sha256", prefix=False)
        _require_int(self.size_bytes, "record size", 1, _MAX_RECORD_BYTES)
        _require_int(
            self.required_image_id_occurrences,
            "required image occurrence count",
            1,
            1_000_000,
        )
        if type(self.registry_references) is not tuple:
            raise ForagerScientificRerunPreflightError(
                "registry references must be an exact tuple"
            )
        if any(
            type(value) is not str or _REGISTRY_REFERENCE_RE.fullmatch(value) is None
            for value in self.registry_references
        ):
            raise ForagerScientificRerunPreflightError("invalid registry reference")


@dataclass(frozen=True, slots=True)
class LocalImageInspection:
    status: LocalStatus
    runtime_executable: str | None
    observed_image_id: str | None
    detail: str

    def __post_init__(self) -> None:
        allowed = {"exact_present", "image_absent", "runtime_unavailable", "inspection_failed"}
        if type(self.status) is not str or self.status not in allowed:
            raise ForagerScientificRerunPreflightError("invalid local image status")
        if self.runtime_executable is not None and (
            type(self.runtime_executable) is not str or not self.runtime_executable
        ):
            raise ForagerScientificRerunPreflightError("invalid runtime executable")
        if (self.status == "runtime_unavailable") != (self.runtime_executable is None):
            raise ForagerScientificRerunPreflightError(
                "runtime availability contradicts the executable identity"
            )
        if self.observed_image_id is not None:
            _require_digest(self.observed_image_id, "observed image ID")
        if self.status in {"runtime_unavailable", "image_absent"} and (
            self.observed_image_id is not None
        ):
            raise ForagerScientificRerunPreflightError(
                "unobserved image status cannot carry an image ID"
            )
        if type(self.detail) is not str or not self.detail or len(self.detail) > 1_024:
            raise ForagerScientificRerunPreflightError("invalid local inspection detail")
        if (self.status == "exact_present") != (self.observed_image_id == REQUIRED_IMAGE_ID):
            raise ForagerScientificRerunPreflightError(
                "exact_present must bind the required Docker image ID"
            )


@dataclass(frozen=True, slots=True)
class ForagerScientificRerunPlan:
    schema: str
    candidate_universe_sha256: str
    candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    horizon_per_cell: int
    cell_count: int
    total_environment_resets: int
    total_environment_transitions: int
    total_observation_deliveries: int
    total_agent_action_queries: int
    required_dynamic_resource_fields: tuple[str, ...]
    timing_is_telemetry_only: bool
    development_only: bool
    promotion_authorized: bool
    negative_outcomes_append_only: bool

    def __post_init__(self) -> None:
        validate_run_plan(self)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema: str
    preflight_source_sha256: str
    image_identity_kind: str
    required_image_id: str
    registry_reference: str | None
    pinned_records: tuple[PinnedIdentityRecord, ...]
    local_image: LocalImageInspection
    plan: ForagerScientificRerunPlan
    current_source_qualification_sha256: str | None
    exact_dynamic_resource_contract_satisfied: bool
    fresh_output_namespace: str | None
    blockers: tuple[str, ...]
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        validate_preflight_report(self)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _plain(dataclasses.asdict(self)))


def _require_digest(value: object, name: str, *, prefix: bool = True) -> str:
    if type(value) is not str:
        raise ForagerScientificRerunPreflightError(f"{name} must be an exact string")
    candidate = value if prefix else f"sha256:{value}"
    if _SHA256_RE.fullmatch(candidate) is None:
        raise ForagerScientificRerunPreflightError(f"{name} must be a lowercase SHA-256")
    return value


def _preflight_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerScientificRerunPreflightError(
            f"{name} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _plain(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is list:
        return [_plain(item) for item in cast(list[object], value)]
    if type(value) is tuple:
        return [_plain(item) for item in cast(tuple[object, ...], value)]
    if type(value) is dict:
        return {
            key: _plain(item)
            for key, item in cast(dict[str, object], value).items()
        }
    raise ForagerScientificRerunPreflightError("record contains a non-JSON value")


def build_run_plan() -> ForagerScientificRerunPlan:
    """Return the exact nonpromoting 210-cell open-tuning schedule contract."""
    return ForagerScientificRerunPlan(
        schema=PLAN_SCHEMA,
        candidate_universe_sha256=MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
        candidate_ids=SELECTION_CANDIDATES,
        seeds=MATCHED_CURRENT_TUNING_SEEDS,
        horizon_per_cell=MATCHED_CURRENT_HORIZON,
        cell_count=CELL_COUNT,
        total_environment_resets=CELL_COUNT,
        total_environment_transitions=TOTAL_ENVIRONMENT_TRANSITIONS,
        total_observation_deliveries=TOTAL_OBSERVATION_DELIVERIES,
        total_agent_action_queries=TOTAL_ENVIRONMENT_TRANSITIONS,
        required_dynamic_resource_fields=REQUIRED_DYNAMIC_RESOURCE_FIELDS,
        timing_is_telemetry_only=True,
        development_only=True,
        promotion_authorized=False,
        negative_outcomes_append_only=True,
    )


def validate_run_plan(value: object) -> ForagerScientificRerunPlan:
    if type(value) is not ForagerScientificRerunPlan:
        raise ForagerScientificRerunPreflightError(
            "run plan must be an exact ForagerScientificRerunPlan"
        )
    expected = (
        PLAN_SCHEMA,
        MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
        SELECTION_CANDIDATES,
        MATCHED_CURRENT_TUNING_SEEDS,
        MATCHED_CURRENT_HORIZON,
        CELL_COUNT,
        CELL_COUNT,
        TOTAL_ENVIRONMENT_TRANSITIONS,
        TOTAL_OBSERVATION_DELIVERIES,
        TOTAL_ENVIRONMENT_TRANSITIONS,
        REQUIRED_DYNAMIC_RESOURCE_FIELDS,
        True,
        True,
        False,
        True,
    )
    observed = tuple(getattr(value, field.name) for field in dataclasses.fields(value))
    if observed != expected:
        raise ForagerScientificRerunPreflightError("run plan differs from the frozen contract")
    return value


def _read_record(root: Path, relative: str) -> PinnedIdentityRecord:
    path = root / relative
    descriptor = -1
    try:
        root_resolved = root.resolve(strict=True)
        path.parent.resolve(strict=True).relative_to(root_resolved)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_RECORD_BYTES
        ):
            raise ForagerScientificRerunPreflightError(
                f"pinned identity record {relative} is not a bounded single-link file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerScientificRerunPreflightError(
                    f"pinned identity record {relative} ended during its bounded read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerScientificRerunPreflightError(
                f"pinned identity record {relative} grew during its bounded read"
            )
        after = os.fstat(descriptor)
        current = os.lstat(path)
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ForagerScientificRerunPreflightError(
                f"pinned identity record {relative} changed during its bounded read"
            )
        raw = b"".join(chunks)
    except (OSError, ValueError) as exc:
        raise ForagerScientificRerunPreflightError(
            f"cannot read pinned identity record {relative}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    occurrences = raw.count(REQUIRED_IMAGE_ID.encode("ascii"))
    if occurrences == 0:
        raise ForagerScientificRerunPreflightError(
            f"pinned identity record {relative} does not bind the required image"
        )
    references = tuple(
        sorted(
            {
                token.decode("ascii")
                for token in re.findall(
                    rb"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+@sha256:[0-9a-f]{64}",
                    raw,
                )
            }
        )
    )
    digest = hashlib.sha256(raw).hexdigest()
    expected_digest = PINNED_IDENTITY_RECORD_SHA256[PINNED_IDENTITY_RECORDS.index(relative)]
    if digest != expected_digest:
        raise ForagerScientificRerunPreflightError(
            f"pinned identity record {relative} differs from its frozen digest"
        )
    return PinnedIdentityRecord(
        path=relative,
        sha256=digest,
        size_bytes=len(raw),
        required_image_id_occurrences=occurrences,
        registry_references=references,
    )


def audit_pinned_identity_records(project_root: Path) -> tuple[PinnedIdentityRecord, ...]:
    """Audit the declared immutable records without scanning reward payloads."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a Path")
    records = tuple(_read_record(project_root, relative) for relative in PINNED_IDENTITY_RECORDS)
    references = {reference for record in records for reference in record.registry_references}
    if references:
        raise ForagerScientificRerunPreflightError(
            "a registry reference appeared; review and version the resolver contract"
        )
    return records


def _default_runner(command: Sequence[str]) -> ProcessResult:
    try:
        completed = subprocess.run(
            tuple(command),
            check=False,
            capture_output=True,
            timeout=_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForagerScientificRerunPreflightError("local OCI inspection failed") from exc
    return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


def inspect_local_image(
    runtime: str | Path = "docker", *, runner: ProcessRunner = _default_runner
) -> LocalImageInspection:
    """Inspect only the local image store; this command cannot pull or run an image."""
    requested = runtime.as_posix() if isinstance(runtime, Path) else runtime
    if type(requested) is not str or not requested:
        raise TypeError("runtime must be a non-empty str or Path")
    executable = shutil.which(requested)
    if executable is None:
        return LocalImageInspection("runtime_unavailable", None, None, "OCI runtime not found")
    result = runner(
        (
            executable,
            "image",
            "inspect",
            "--format={{json .Id}}",
            REQUIRED_IMAGE_ID,
        )
    )
    if type(result) is not ProcessResult:
        raise ForagerScientificRerunPreflightError(
            "local OCI inspection runner returned the wrong exact type"
        )
    if result.returncode != 0:
        return LocalImageInspection(
            "image_absent", executable, None, "required image is absent from the local store"
        )
    if result.stderr:
        return LocalImageInspection(
            "inspection_failed", executable, None, "OCI inspection unexpectedly wrote stderr"
        )
    try:
        observed = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerScientificRerunPreflightError("OCI inspection returned invalid JSON") from exc
    if type(observed) is not str or _SHA256_RE.fullmatch(observed) is None:
        raise ForagerScientificRerunPreflightError("OCI inspection returned an invalid image ID")
    if observed != REQUIRED_IMAGE_ID:
        return LocalImageInspection(
            "inspection_failed", executable, observed, "OCI runtime resolved a different image ID"
        )
    return LocalImageInspection("exact_present", executable, observed, "exact local image present")


def build_preflight_report(
    project_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: ProcessRunner = _default_runner,
) -> PreflightReport:
    records = audit_pinned_identity_records(project_root)
    local = inspect_local_image(runtime, runner=runner)
    blockers: list[str] = []
    if local.status != "exact_present":
        blockers.append("exact Docker image config digest is absent from the local OCI store")
        blockers.append(
            "no pinned registry repository@manifest-digest resolves the required "
            "image config digest"
        )
    blockers.extend(
        (
            "fresh current-source v2 qualification manifest is missing",
            "current qualification lacks the exact dynamic byte/query receipt contract",
            "fresh append-only nonpromoting output namespace is not declared",
        )
    )
    return PreflightReport(
        schema=SCHEMA,
        preflight_source_sha256=_preflight_source_sha256(),
        image_identity_kind="docker_image_config_digest_not_registry_manifest_digest",
        required_image_id=REQUIRED_IMAGE_ID,
        registry_reference=None,
        pinned_records=records,
        local_image=local,
        plan=build_run_plan(),
        current_source_qualification_sha256=None,
        exact_dynamic_resource_contract_satisfied=False,
        fresh_output_namespace=None,
        blockers=tuple(blockers),
    )


def validate_preflight_report(value: object) -> PreflightReport:
    if type(value) is not PreflightReport:
        raise ForagerScientificRerunPreflightError("preflight report has the wrong exact type")
    if type(value.pinned_records) is not tuple or any(
        type(record) is not PinnedIdentityRecord for record in value.pinned_records
    ):
        raise ForagerScientificRerunPreflightError(
            "pinned records must contain exact PinnedIdentityRecord values"
        )
    if (
        value.schema != SCHEMA
        or value.preflight_source_sha256 != _preflight_source_sha256()
        or value.image_identity_kind
        != "docker_image_config_digest_not_registry_manifest_digest"
        or value.required_image_id != REQUIRED_IMAGE_ID
        or value.registry_reference is not None
        or tuple(record.path for record in value.pinned_records) != PINNED_IDENTITY_RECORDS
        or type(value.local_image) is not LocalImageInspection
        or type(value.plan) is not ForagerScientificRerunPlan
        or type(value.blockers) is not tuple
        or value.current_source_qualification_sha256 is not None
        or type(value.exact_dynamic_resource_contract_satisfied) is not bool
        or value.exact_dynamic_resource_contract_satisfied
        or value.fresh_output_namespace is not None
        or type(value.launch_authorized) is not bool
        or value.launch_authorized
    ):
        raise ForagerScientificRerunPreflightError("preflight fail-closed contract mismatch")
    for record in value.pinned_records:
        if type(record) is not PinnedIdentityRecord:
            raise ForagerScientificRerunPreflightError("pinned record has the wrong exact type")
        PinnedIdentityRecord.__post_init__(record)
    if tuple(record.sha256 for record in value.pinned_records) != PINNED_IDENTITY_RECORD_SHA256:
        raise ForagerScientificRerunPreflightError("pinned record digest closure mismatch")
    LocalImageInspection.__post_init__(value.local_image)
    validate_run_plan(value.plan)
    expected_blockers: list[str] = []
    if value.local_image.status != "exact_present":
        expected_blockers.extend(
            (
                "exact Docker image config digest is absent from the local OCI store",
                "no pinned registry repository@manifest-digest resolves the required "
                "image config digest",
            )
        )
    expected_blockers.extend(
        (
            "fresh current-source v2 qualification manifest is missing",
            "current qualification lacks the exact dynamic byte/query receipt contract",
            "fresh append-only nonpromoting output namespace is not declared",
        )
    )
    if value.blockers != tuple(expected_blockers):
        raise ForagerScientificRerunPreflightError("preflight blocker closure mismatch")
    return value


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--runtime", default="docker")
    arguments = parser.parse_args(argv)
    report = build_preflight_report(arguments.project_root, runtime=arguments.runtime)
    sys.stdout.write(
        json.dumps(report.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Print the offline readiness report; blocked readiness exits two."""
    try:
        return _cli(tuple(sys.argv[1:] if argv is None else argv))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Forager scientific rerun preflight: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_COUNT",
    "PINNED_IDENTITY_RECORDS",
    "PINNED_IDENTITY_RECORD_SHA256",
    "REQUIRED_DYNAMIC_RESOURCE_FIELDS",
    "REQUIRED_IMAGE_ID",
    "SELECTION_CANDIDATES",
    "TOTAL_ENVIRONMENT_TRANSITIONS",
    "TOTAL_OBSERVATION_DELIVERIES",
    "ForagerScientificRerunPlan",
    "ForagerScientificRerunPreflightError",
    "LocalImageInspection",
    "PinnedIdentityRecord",
    "PreflightReport",
    "ProcessResult",
    "audit_pinned_identity_records",
    "build_preflight_report",
    "build_run_plan",
    "inspect_local_image",
    "main",
    "validate_preflight_report",
    "validate_run_plan",
]
