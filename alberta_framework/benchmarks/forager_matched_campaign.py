"""Resumable, content-only execution of the matched-current open-tuning block.

The campaign deliberately stops at an unresolved verification request.  It cannot authenticate
its own qualification or execution, select candidates, seal a protocol, run held-out evaluation,
or authorize promotion.  Raw OCI exports stay opaque on the host: this module only checks their
regular-file identity, byte length, and SHA-256 before delegating scoring to the frozen in-image
scorer.

Every invocation reconstructs the protocol and the 21-candidate selection plan from a verified
qualification bundle, requalifies the live OCI runtime, and replays all persisted bindings.  The
candidate panel and its two descriptive-only arms are frozen in
:mod:`~alberta_framework.benchmarks.forager_matched_open_protocol`, assembled from the provenance
in :mod:`~alberta_framework.benchmarks.forager_matched_candidate_universe`.  The two descriptive
candidates remain in the canonical universe and open protocol but are never in the tuning
execution plan or schedule.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks.forager_matched_evidence import MatchedScoreEvidence
from alberta_framework.benchmarks.forager_matched_protocol import ForagerMatchedProtocol

MATCHED_OPEN_CAMPAIGN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_open_tuning_campaign.v2"
)
MATCHED_OPEN_SCHEDULE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_open_tuning_schedule.v1"
)
MATCHED_RAW_BINDING_SCHEMA_VERSION: Final = "alberta.forager_matched_raw_binding.v1"
MATCHED_COMPLETION_POINTER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_cell_completion_pointer.v1"
)
MATCHED_ATTEMPT_FAILURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_attempt_failure.v1"
)
MATCHED_OPEN_COMPLETION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_open_tuning_completion.v2"
)

_LOGGER = logging.getLogger(__name__)
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_MAX_RAW_BYTES: Final = 512 * 1024 * 1024
_MAX_ATTEMPTS_PER_CELL: Final = 128
_MAX_FAILURE_RECORDS_PER_ATTEMPT: Final = 128
_MAX_RETAINED_RAW_BYTES_PER_CELL: Final = 1024 * 1024 * 1024
_MAX_RETAINED_RAW_BYTES_PER_CAMPAIGN: Final = 8 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_RE = re.compile(r"attempt-([0-9]{6})\Z")
_FAILURE_RE = re.compile(r"failure-([0-9]{6})\.json\Z")

_IMMUTABLE_ARTIFACTS: Final = (
    "open-protocol.json",
    "candidate-universe.json",
    "execution-plan.json",
    "source-manifest.json",
    "executor-manifest.json",
    "qualification-manifest.json",
    "execution-schedule.json",
    "live-runtime.json",
    "campaign.json",
)
_FINAL_ARTIFACTS: Final = (
    "execution-receipt-index.json",
    "score-evidence.json",
    "verification-request.json",
    "completion-summary.json",
)


class ForagerMatchedCampaignError(ValueError):
    """A campaign root, runtime binding, attempt, or completion failed closed."""


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    """Derived status; no mutable progress counter is trusted or persisted."""

    output_root: Path
    state: str
    completed_cells: int
    total_cells: int
    next_candidate_id: str | None
    next_seed: int | None
    protocol_sha256: str
    qualification_manifest_sha256: str
    plan_sha256: str
    live_runtime_identity_sha256: str
    score_evidence_sha256: str | None
    verification_subject_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "alberta.forager_matched_campaign_status.v2",
            "classification": "content_only_unendorsed_nonpromoting",
            "state": self.state,
            "output_root": self.output_root.as_posix(),
            "completed_cells": self.completed_cells,
            "total_cells": self.total_cells,
            "next_candidate_id": self.next_candidate_id,
            "next_seed": self.next_seed,
            "protocol_sha256": self.protocol_sha256,
            "qualification_manifest_sha256": self.qualification_manifest_sha256,
            "plan_sha256": self.plan_sha256,
            "live_runtime_identity_sha256": self.live_runtime_identity_sha256,
            "score_evidence_sha256": self.score_evidence_sha256,
            "verification_subject_sha256": self.verification_subject_sha256,
            "promotion_authorized": False,
            "external_verification_required": True,
        }


@dataclass(frozen=True, slots=True)
class CompletedCampaignBundle:
    """Exact completed content closure, without any authentication or promotion claim.

    Instances returned by :func:`load_completed_open_tuning_campaign` are rebuilt from the
    qualification inputs and every completed cell, then compared with all four persisted final
    artifacts.  The verification request remains unresolved: this value is not an
    :class:`~alberta_framework.benchmarks.forager_matched_evidence.AuthenticatedEvidenceBindings`
    and does not confer external trust.
    """

    output_root: Path
    protocol: ForagerMatchedProtocol
    plan: executor.MatchedExecutionPlan
    live_runtime: executor.LiveRuntimeIdentity
    candidate_ids: tuple[str, ...]
    active_seeds: tuple[int, ...]
    schedule: Mapping[str, Any]
    seed_artifacts: Mapping[str, tuple[executor.SeedExecutionArtifacts, ...]]
    execution_receipt_index: executor.MatchedExecutionReceiptIndex
    score_evidence: MatchedScoreEvidence
    verification_request: executor.VerificationRequest
    completion_summary: Mapping[str, Any]
    final_file_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _RebuiltInputs:
    bundle: qualification.MatchedCurrentQualificationBundle
    protocol: ForagerMatchedProtocol
    plan: executor.MatchedExecutionPlan
    candidate_ids: tuple[str, ...]
    assets: dict[str, executor.CandidateExecutionAssets]
    schedule: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CampaignContext:
    root: Path
    rebuilt: _RebuiltInputs
    live_runtime: executor.LiveRuntimeIdentity


_CompletionSummaryBuilder = Callable[
    [
        _CampaignContext,
        executor.MatchedExecutionReceiptIndex,
        MatchedScoreEvidence,
        executor.VerificationRequest,
    ],
    Mapping[str, Any],
]
_CompletionSummaryValidator = Callable[
    [
        _CampaignContext,
        executor.MatchedExecutionReceiptIndex,
        MatchedScoreEvidence,
        executor.VerificationRequest,
        Mapping[str, Any],
    ],
    None,
]


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the campaign's compact canonical ASCII JSON encoding."""
    try:
        return json.dumps(
            _plain(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedCampaignError("value is not canonical JSON") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    if type(value) is tuple:
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedCampaignError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedCampaignError(f"non-finite JSON number {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ForagerMatchedCampaignError(f"non-finite JSON number {value!r}")
    return parsed


def _decode_canonical(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_JSON_BYTES or raw.startswith(b"\xef\xbb\xbf"):
        raise ForagerMatchedCampaignError(f"{label} violates the JSON byte contract")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_json_float,
        )
    except ForagerMatchedCampaignError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedCampaignError(f"{label} is not strict JSON") from exc
    if type(decoded) is not dict:
        raise ForagerMatchedCampaignError(f"{label} must be a JSON object")
    result = cast(dict[str, Any], decoded)
    if raw != canonical_json_bytes(result):
        raise ForagerMatchedCampaignError(f"{label} is not canonical JSON")
    return result


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ForagerMatchedCampaignError(f"{label} is not a regular directory")
    return path.resolve()


def _stable_file_hash(path: Path, label: str, *, maximum: int) -> tuple[str, int]:
    try:
        path_before = os.lstat(path)
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} is missing") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} cannot be opened safely") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum
            or (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ForagerMatchedCampaignError(f"{label} is not a bounded single-link file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or total != before.st_size:
        raise ForagerMatchedCampaignError(f"{label} changed while being hashed")
    current = os.lstat(path)
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_nlink,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if current_identity != after_identity:
        raise ForagerMatchedCampaignError(f"{label} path changed while being hashed")
    return digest.hexdigest(), total


def _read_stable(path: Path, label: str, *, maximum: int = _MAX_JSON_BYTES) -> bytes:
    try:
        path_before = os.lstat(path)
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} is missing") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum
            or (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ForagerMatchedCampaignError(f"{label} is not a bounded single-link file")
        raw = bytearray()
        while len(raw) < before.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, before.st_size - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ForagerMatchedCampaignError(f"{label} changed while being read")
    current = os.lstat(path)
    if (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_nlink,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ForagerMatchedCampaignError(f"{label} path changed while being read")
    return bytes(raw)


def _bounded_entry_names(path: Path, label: str, *, maximum: int) -> set[str]:
    if type(maximum) is not int or maximum < 1:
        raise AssertionError("directory-entry bound must be a positive integer")
    names: set[str] = set()
    try:
        iterator = os.scandir(path)
    except OSError as exc:
        raise ForagerMatchedCampaignError(f"{label} cannot be enumerated") from exc
    with iterator:
        for entry in iterator:
            if len(names) >= maximum:
                raise ForagerMatchedCampaignError(f"{label} contains too many entries")
            if entry.name in names:
                raise ForagerMatchedCampaignError(f"{label} repeats an entry name")
            names.add(entry.name)
    return names


def _rename_no_replace(source: Path, destination: Path) -> None:
    parent = _regular_directory(destination.parent, "artifact parent")
    if source.parent.resolve() != parent or destination.parent.resolve() != parent:
        raise ForagerMatchedCampaignError("atomic publication paths do not share one parent")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ForagerMatchedCampaignError("renameat2 is required for exclusive publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        result = renameat2(
            parent_fd,
            os.fsencode(source.name),
            parent_fd,
            os.fsencode(destination.name),
            1,  # RENAME_NOREPLACE: fail with EEXIST rather than replace the target
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise ForagerMatchedCampaignError(
                    f"artifact {destination.name!r} already exists"
                )
            raise ForagerMatchedCampaignError(
                f"exclusive publication failed with errno {error}"
            )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _link_anonymous_no_replace(
    descriptor: int,
    parent: Path,
    destination_name: str,
) -> None:
    """Give a name to an anonymous ``O_TMPFILE`` inode without ever replacing a file.

    ``linkat(fd, "", parent_fd, name, AT_EMPTY_PATH)`` links the open inode
    itself; unlike ``rename`` it fails with ``EEXIST`` when the name already
    exists, preserving write-once artifact semantics.  Python's ``os.link``
    does not expose ``AT_EMPTY_PATH``, hence the direct ``libc`` call.
    """
    if (
        not destination_name
        or destination_name in {".", ".."}
        or "/" in destination_name
        or "\x00" in destination_name
    ):
        raise ForagerMatchedCampaignError("anonymous publication name is unsafe")
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise ForagerMatchedCampaignError("linkat is required for anonymous publication")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        result = linkat(
            descriptor,
            b"",
            parent_fd,
            os.fsencode(destination_name),
            0x1000,  # AT_EMPTY_PATH: link the O_TMPFILE inode itself, not a path under it
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise ForagerMatchedCampaignError(
                    f"artifact {destination_name!r} already exists"
                )
            raise ForagerMatchedCampaignError(
                f"anonymous artifact publication failed with errno {error}"
            )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _publish_bytes(path: Path, raw: bytes) -> str:
    """Publish ``raw`` crash-safely: fill an anonymous ``O_TMPFILE`` inode, then link it in.

    The inode has no directory entry until :func:`_link_anonymous_no_replace`
    succeeds, so a crash mid-write can never leave a partial artifact visible,
    and an existing artifact is never replaced.
    """
    if not raw:
        raise ForagerMatchedCampaignError("empty artifacts are forbidden")
    parent = _regular_directory(path.parent, "artifact parent")
    flags = (
        os.O_WRONLY
        | getattr(os, "O_TMPFILE", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if not getattr(os, "O_TMPFILE", 0):
        raise ForagerMatchedCampaignError("O_TMPFILE is required for crash-safe publication")
    try:
        descriptor = os.open(parent, flags, 0o400)
    except OSError as exc:
        raise ForagerMatchedCampaignError(
            "campaign filesystem does not support anonymous temporary files"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset : offset + 1024 * 1024])
            if written <= 0:
                raise ForagerMatchedCampaignError("artifact write made no progress")
            offset += written
        os.fsync(descriptor)
        _link_anonymous_no_replace(descriptor, parent, path.name)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _campaign_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    root = _regular_directory(root, "campaign output root")
    lock_path = root / "campaign.json"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise ForagerMatchedCampaignError("campaign lock cannot be opened safely") from exc
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > _MAX_JSON_BYTES
        ):
            raise ForagerMatchedCampaignError("campaign lock inode identity drifted")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ForagerMatchedCampaignError(
                "campaign is already locked by another invocation"
            ) from exc
        locked = True
        current = os.lstat(lock_path)
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_size,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
        ):
            raise ForagerMatchedCampaignError("campaign lock path changed during acquisition")
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _publish_json_pair(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    _publish_bytes(path, raw)
    try:
        _publish_bytes(path.with_name(path.name + ".sha256"), f"{digest}\n".encode("ascii"))
    except BaseException:
        # The canonical payload remains immutable and recoverable, but publication did not
        # complete.  Verification fails closed until an operator explicitly repairs the pair.
        raise
    return digest


def _publish_exact_json_pair(path: Path, raw: bytes, expected_sha256: str) -> None:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ForagerMatchedCampaignError(
            "exact qualification manifest bytes differ from their loader digest"
        )
    _publish_bytes(path, raw)
    _publish_bytes(path.with_name(path.name + ".sha256"), f"{digest}\n".encode("ascii"))


def _load_json_pair(
    path: Path,
    label: str,
    *,
    repair_missing_sidecar: bool = False,
) -> tuple[dict[str, Any], str]:
    raw = _read_stable(path, label)
    payload = _decode_canonical(raw, label)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_path = path.with_name(path.name + ".sha256")
    if not sidecar_path.exists() and repair_missing_sidecar:
        _publish_bytes(sidecar_path, f"{digest}\n".encode("ascii"))
    sidecar = _read_stable(
        sidecar_path,
        f"{label} digest sidecar",
        maximum=128,
    )
    if sidecar != f"{digest}\n".encode("ascii"):
        raise ForagerMatchedCampaignError(f"{label} digest sidecar differs")
    return payload, digest


def selection_candidate_ids(protocol: ForagerMatchedProtocol) -> tuple[str, ...]:
    """Return the exact frozen selection-group order, excluding descriptive arms."""
    if not isinstance(protocol, ForagerMatchedProtocol):
        raise TypeError("protocol must be a ForagerMatchedProtocol")
    groups = protocol.selection_plan.groups
    if tuple(group.selection_group for group in groups) != ("alberta", "external"):
        raise ForagerMatchedCampaignError("open protocol selection-group order drifted")
    candidate_ids = tuple(
        candidate_id for group in groups for candidate_id in group.candidate_ids
    )
    expected = (
        open_protocol.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS
        + open_protocol.MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS
    )
    if candidate_ids != expected or len(candidate_ids) != 21 or len(set(candidate_ids)) != 21:
        raise ForagerMatchedCampaignError("open protocol selection panel drifted")
    if set(candidate_ids) & set(open_protocol.MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS):
        raise ForagerMatchedCampaignError("descriptive candidate entered open tuning")
    for candidate_id in candidate_ids:
        candidate = protocol.candidate_index[candidate_id]
        if not candidate.pairing.eligible:
            raise ForagerMatchedCampaignError("selection candidate is not pairing eligible")
    return candidate_ids


def build_seed_major_schedule(protocol: ForagerMatchedProtocol) -> dict[str, Any]:
    """Freeze a deterministic seed-major schedule over only the 21 selection candidates."""
    candidate_ids = selection_candidate_ids(protocol)
    if protocol.stage != "open_tuning" or protocol.active_seeds != (
        open_protocol.MATCHED_CURRENT_TUNING_SEEDS
    ):
        raise ForagerMatchedCampaignError("campaign requires the exact open tuning seed block")
    unsigned: dict[str, Any] = {
        "schema_version": MATCHED_OPEN_SCHEDULE_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "stage": "open_tuning",
        "protocol_sha256": protocol.protocol_sha256,
        "ordering": "seed_major_then_frozen_selection_group_candidate_order",
        "candidate_order": list(candidate_ids),
        "active_seeds": list(protocol.active_seeds),
        "cells": [
            {"ordinal": ordinal, "candidate_id": candidate_id, "seed": seed}
            for ordinal, (seed, candidate_id) in enumerate(
                (seed, candidate_id)
                for seed in protocol.active_seeds
                for candidate_id in candidate_ids
            )
        ],
        "promotion_authorized": False,
        "external_verification_required": True,
    }
    return {**unsigned, "schedule_sha256": _canonical_sha256(unsigned)}


def _rebuild_inputs(qualification_root: Path) -> _RebuiltInputs:
    bundle = qualification.load_matched_current_qualification_bundle(qualification_root)
    if (
        hashlib.sha256(bundle.manifest_bytes).hexdigest() != bundle.manifest_sha256
        or qualification._canonical_json_bytes(bundle.manifest) != bundle.manifest_bytes
    ):
        raise ForagerMatchedCampaignError(
            "qualification manifest exact bytes changed after loader verification"
        )
    protocol = open_protocol.build_forager_matched_open_protocol(
        runtime=bundle.runtime_qualification,
        candidate_qualifications=bundle.candidate_qualifications,
    )
    candidate_ids = selection_candidate_ids(protocol)
    if tuple(bundle.candidate_assets) != open_protocol.MATCHED_CURRENT_CANDIDATE_IDS:
        raise ForagerMatchedCampaignError("qualification candidate asset order drifted")
    assets = {
        candidate_id: cast(
            executor.CandidateExecutionAssets,
            bundle.candidate_assets[candidate_id],
        )
        for candidate_id in candidate_ids
    }
    try:
        cpu_root = bundle.cpu_qualification_root
        rng_root = bundle.rng_parity_qualification_root
    except AttributeError as exc:
        raise ForagerMatchedCampaignError(
            "qualification bundle does not carry its executor qualification roots"
        ) from exc
    plan = executor.build_execution_plan(
        protocol,
        assets,
        qualification_manifest_sha256=bundle.manifest_sha256,
        candidate_ids=candidate_ids,
        cpu_qualification_root=cpu_root,
        rng_parity_qualification_root=rng_root,
    )
    if tuple(item.candidate.candidate_id for item in plan.candidates) != candidate_ids:
        raise ForagerMatchedCampaignError("execution plan candidate order drifted")
    return _RebuiltInputs(
        bundle=bundle,
        protocol=protocol,
        plan=plan,
        candidate_ids=candidate_ids,
        assets=assets,
        schedule=build_seed_major_schedule(protocol),
    )


def _qualify_live(
    plan: executor.MatchedExecutionPlan,
    runtime: str | Path,
    runner: executor.ProcessRunner | None,
) -> executor.LiveRuntimeIdentity:
    if runner is None:
        return executor.qualify_live_runtime(plan, runtime=runtime)
    return executor.qualify_live_runtime(plan, runtime=runtime, runner=runner)


def _campaign_manifest(
    rebuilt: _RebuiltInputs,
    live: executor.LiveRuntimeIdentity,
) -> dict[str, Any]:
    qualification_manifest_sha = rebuilt.bundle.manifest_sha256
    universe_sha = universe.MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256
    if rebuilt.protocol.selection_plan.candidate_universe_sha256 != universe_sha:
        raise ForagerMatchedCampaignError("protocol candidate-universe binding drifted")
    if (
        rebuilt.plan.qualification_manifest_sha256 != qualification_manifest_sha
        or rebuilt.plan.payload.get("qualification_manifest_sha256")
        != qualification_manifest_sha
        or rebuilt.plan.executor_manifest.get("qualification_manifest_sha256")
        != qualification_manifest_sha
    ):
        raise ForagerMatchedCampaignError("qualification manifest plan binding drifted")
    return {
        "schema_version": MATCHED_OPEN_CAMPAIGN_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "prepared_open_tuning_external_verification_required",
        "stage": "open_tuning",
        "authority": {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "protocol_sha256": rebuilt.protocol.protocol_sha256,
        "candidate_universe_sha256": universe_sha,
        "execution_plan_sha256": rebuilt.plan.plan_sha256,
        "source_manifest_sha256": rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": rebuilt.plan.executor_manifest_sha256,
        "qualification_manifest_sha256": qualification_manifest_sha,
        "live_runtime_identity_sha256": live.identity_sha256,
        "execution_schedule_sha256": cast(str, rebuilt.schedule["schedule_sha256"]),
        "candidate_order": list(rebuilt.candidate_ids),
        "active_seeds": list(rebuilt.protocol.active_seeds),
        "cell_count": len(rebuilt.candidate_ids) * len(rebuilt.protocol.active_seeds),
        "descriptive_candidates_excluded": list(
            open_protocol.MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS
        ),
        "host_reward_array_access": "forbidden",
        "retention_bounds": {
            "max_attempts_per_cell": _MAX_ATTEMPTS_PER_CELL,
            "max_failure_records_per_attempt": _MAX_FAILURE_RECORDS_PER_ATTEMPT,
            "max_raw_archive_bytes": _MAX_RAW_BYTES,
            "max_retained_raw_bytes_per_cell": _MAX_RETAINED_RAW_BYTES_PER_CELL,
            "max_retained_raw_bytes_per_campaign": (
                _MAX_RETAINED_RAW_BYTES_PER_CAMPAIGN
            ),
        },
        "completion_boundary": (
            "score_evidence_and_unresolved_verification_request_only"
        ),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _canonical_campaign_destination(
    requested_output: Path,
    prospective_output: Path,
    qualified_root: Path,
) -> Path:
    """Resolve the created parent and reject any post-check path redirection."""
    if not requested_output.name or requested_output.name in {".", ".."}:
        raise ForagerMatchedCampaignError("campaign output name is unsafe")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    canonical_parent = _regular_directory(
        requested_output.parent,
        "campaign output parent",
    )
    actual_output = canonical_parent / requested_output.name
    if actual_output != prospective_output:
        raise ForagerMatchedCampaignError(
            "campaign output parent was redirected after its prospective path check"
        )
    if _paths_overlap(actual_output, qualified_root):
        raise ForagerMatchedCampaignError(
            "campaign and qualification output roots overlap after parent resolution"
        )
    if actual_output.exists() or actual_output.is_symlink():
        raise ForagerMatchedCampaignError("campaign output root already exists")
    return actual_output


def _publish_initial_root(
    root: Path,
    rebuilt: _RebuiltInputs,
    live: executor.LiveRuntimeIdentity,
) -> None:
    parent = _regular_directory(root.parent, "campaign parent")
    if root != parent / root.name:
        raise ForagerMatchedCampaignError(
            "canonical campaign destination parent changed before publication"
        )
    if root.exists() or root.is_symlink():
        raise ForagerMatchedCampaignError("campaign output root already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=parent))
    try:
        (temporary / "runs").mkdir(mode=0o700)
        (temporary / "completions").mkdir(mode=0o700)
        initial: tuple[tuple[str, Mapping[str, Any]], ...] = (
            ("open-protocol.json", rebuilt.protocol.to_dict()),
            (
                "candidate-universe.json",
                universe.matched_current_candidate_universe_descriptor(),
            ),
            ("execution-plan.json", rebuilt.plan.to_dict()),
            ("source-manifest.json", rebuilt.plan.source_manifest),
            ("executor-manifest.json", rebuilt.plan.executor_manifest),
            ("execution-schedule.json", rebuilt.schedule),
            ("live-runtime.json", live.unsigned_dict),
            ("campaign.json", _campaign_manifest(rebuilt, live)),
        )
        for name, payload in initial:
            _publish_json_pair(temporary / name, payload)
        _publish_exact_json_pair(
            temporary / "qualification-manifest.json",
            rebuilt.bundle.manifest_bytes,
            rebuilt.bundle.manifest_sha256,
        )
        _rename_no_replace(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_open_tuning_campaign(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> CampaignStatus:
    """Atomically create a new campaign root without executing a benchmark cell."""
    if not isinstance(qualification_root, Path) or not isinstance(output_root, Path):
        raise TypeError("qualification_root and output_root must be Paths")
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedCampaignError("campaign output root already exists")
    qualified = _regular_directory(qualification_root, "qualification output root")
    try:
        resolved_output = output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedCampaignError("campaign output path cannot be resolved") from exc
    if _paths_overlap(resolved_output, qualified):
        raise ForagerMatchedCampaignError(
            "campaign and qualification output roots overlap"
        )
    canonical_output = _canonical_campaign_destination(
        output_root,
        resolved_output,
        qualified,
    )
    rebuilt = _rebuild_inputs(qualified)
    verification = universe.verify_matched_current_candidate_universe_sources(
        _REPOSITORY_ROOT
    )
    if verification.candidate_universe_sha256 != (
        rebuilt.protocol.selection_plan.candidate_universe_sha256
    ):
        raise ForagerMatchedCampaignError("candidate-universe source verification drifted")
    live = _qualify_live(rebuilt.plan, runtime, runner)
    _publish_initial_root(canonical_output, rebuilt, live)
    with _campaign_lock(canonical_output, exclusive=False):
        return _load_context_and_status(
            qualified,
            canonical_output,
            runtime=runtime,
            runner=runner,
        )[1]


def _expected_root_names() -> set[str]:
    artifacts = _IMMUTABLE_ARTIFACTS + _FINAL_ARTIFACTS
    return {
        "runs",
        "completions",
        *(name for name in artifacts),
        *(f"{name}.sha256" for name in artifacts),
    }


def _validate_root_shape(root: Path) -> None:
    actual = _bounded_entry_names(
        root,
        "campaign root",
        maximum=len(_expected_root_names()),
    )
    unknown = actual - _expected_root_names()
    if unknown:
        raise ForagerMatchedCampaignError(
            f"campaign root contains unknown artifacts: {sorted(unknown)}"
        )
    required = {
        "runs",
        "completions",
        *(_IMMUTABLE_ARTIFACTS),
        *(f"{name}.sha256" for name in _IMMUTABLE_ARTIFACTS),
    }
    missing = required - actual
    if missing:
        raise ForagerMatchedCampaignError(
            f"campaign root is missing required artifacts: {sorted(missing)}"
        )
    _regular_directory(root / "runs", "campaign runs directory")
    _regular_directory(root / "completions", "campaign completions directory")


def _expect_artifact(
    root: Path,
    name: str,
    expected: Mapping[str, Any],
    *,
    repair_missing_sidecar: bool = False,
) -> str:
    payload, digest = _load_json_pair(
        root / name,
        name,
        repair_missing_sidecar=repair_missing_sidecar,
    )
    if payload != _plain(expected):
        raise ForagerMatchedCampaignError(f"persisted {name} differs from rebuilt inputs")
    return digest


def _expect_qualification_artifact(
    root: Path,
    bundle: qualification.MatchedCurrentQualificationBundle,
) -> str:
    path = root / "qualification-manifest.json"
    raw = _read_stable(path, "qualification manifest copy")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _read_stable(
        path.with_name(path.name + ".sha256"),
        "qualification manifest copy digest sidecar",
        maximum=128,
    )
    if (
        raw != bundle.manifest_bytes
        or digest != bundle.manifest_sha256
        or sidecar != f"{digest}\n".encode("ascii")
    ):
        raise ForagerMatchedCampaignError(
            "persisted qualification manifest differs from exact loader-verified bytes"
        )
    return digest


def _load_context(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path,
    runner: executor.ProcessRunner | None,
) -> _CampaignContext:
    root = _regular_directory(output_root, "campaign output root")
    rebuilt = _rebuild_inputs(qualification_root)
    verification = universe.verify_matched_current_candidate_universe_sources(
        _REPOSITORY_ROOT
    )
    if verification.candidate_universe_sha256 != (
        rebuilt.protocol.selection_plan.candidate_universe_sha256
    ):
        raise ForagerMatchedCampaignError("candidate-universe source verification drifted")
    live = _qualify_live(rebuilt.plan, runtime, runner)
    _validate_root_shape(root)
    _expect_artifact(root, "open-protocol.json", rebuilt.protocol.to_dict())
    _expect_artifact(
        root,
        "candidate-universe.json",
        universe.matched_current_candidate_universe_descriptor(),
    )
    persisted_plan, plan_file_sha = _load_json_pair(
        root / "execution-plan.json",
        "execution plan",
    )
    if plan_file_sha != rebuilt.plan.plan_sha256 or persisted_plan != rebuilt.plan.to_dict():
        raise ForagerMatchedCampaignError("persisted execution plan differs from rebuilt plan")
    replayed = executor.parse_execution_plan(
        persisted_plan,
        protocol=rebuilt.protocol,
        assets=rebuilt.assets,
        expected_plan_sha256=rebuilt.plan.plan_sha256,
        expected_qualification_manifest_sha256=rebuilt.bundle.manifest_sha256,
        cpu_qualification_root=rebuilt.bundle.cpu_qualification_root,
        rng_parity_qualification_root=rebuilt.bundle.rng_parity_qualification_root,
    )
    if replayed.plan_sha256 != rebuilt.plan.plan_sha256:
        raise ForagerMatchedCampaignError("execution plan replay digest drifted")
    _expect_artifact(root, "source-manifest.json", rebuilt.plan.source_manifest)
    _expect_artifact(root, "executor-manifest.json", rebuilt.plan.executor_manifest)
    qualification_copy_sha = _expect_qualification_artifact(root, rebuilt.bundle)
    if qualification_copy_sha != rebuilt.bundle.manifest_sha256:
        raise ForagerMatchedCampaignError(
            "persisted qualification manifest differs from its loader-verified digest"
        )
    _expect_artifact(root, "execution-schedule.json", rebuilt.schedule)
    live_payload, live_file_sha = _load_json_pair(
        root / "live-runtime.json",
        "live runtime identity",
    )
    if (
        live_payload != live.unsigned_dict
        or live_file_sha != live.identity_sha256
        or _canonical_sha256(live_payload) != live.identity_sha256
    ):
        raise ForagerMatchedCampaignError("live runtime identity drifted across invocations")
    _expect_artifact(root, "campaign.json", _campaign_manifest(rebuilt, live))
    return _CampaignContext(root=root, rebuilt=rebuilt, live_runtime=live)


def _cell_paths(root: Path, candidate_id: str, seed: int) -> tuple[Path, Path]:
    run_cell = root / "runs" / candidate_id / f"seed-{seed}"
    completion = root / "completions" / candidate_id / f"seed-{seed}.json"
    return run_cell, completion


def _validate_directory_members(
    path: Path,
    allowed: set[str],
    label: str,
) -> None:
    actual = _bounded_entry_names(path, label, maximum=len(allowed))
    unknown = actual - allowed
    if unknown:
        raise ForagerMatchedCampaignError(f"{label} contains unknown entries: {sorted(unknown)}")


def _raw_binding(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    attempt_name: str,
    raw_sha256: str,
    raw_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": MATCHED_RAW_BINDING_SCHEMA_VERSION,
        "classification": "opaque_raw_content_binding_nonpromoting",
        "stage": context.rebuilt.protocol.stage,
        "candidate_id": candidate_id,
        "seed": seed,
        "attempt": attempt_name,
        "protocol_sha256": context.rebuilt.protocol.protocol_sha256,
        "execution_plan_sha256": context.rebuilt.plan.plan_sha256,
        "source_manifest_sha256": context.rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": context.rebuilt.plan.executor_manifest_sha256,
        "live_runtime_identity_sha256": context.live_runtime.identity_sha256,
        "raw_archive": {
            "path": "raw-output.tar",
            "sha256": raw_sha256,
            "size_bytes": raw_size,
            "host_access": "opaque_byte_hash_and_size_only_reward_arrays_not_opened",
        },
        "promotion_authorized": False,
        "external_verification_required": True,
    }


def _persist_raw_binding(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    attempt: Path,
) -> tuple[dict[str, Any], str]:
    raw_sha, raw_size = _stable_file_hash(
        attempt / "raw-output.tar",
        "opaque raw OCI archive",
        maximum=_MAX_RAW_BYTES,
    )
    binding = _raw_binding(
        context,
        candidate_id,
        seed,
        attempt.name,
        raw_sha,
        raw_size,
    )
    digest = _publish_json_pair(attempt / "raw-binding.json", binding)
    return binding, digest


def _validate_raw_binding(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    attempt: Path,
    *,
    repair_pairs: bool = False,
) -> tuple[dict[str, Any], str]:
    payload, digest = _load_json_pair(
        attempt / "raw-binding.json",
        "raw archive binding",
        repair_missing_sidecar=repair_pairs,
    )
    raw_sha, raw_size = _stable_file_hash(
        attempt / "raw-output.tar",
        "bound opaque raw OCI archive",
        maximum=_MAX_RAW_BYTES,
    )
    expected = _raw_binding(
        context,
        candidate_id,
        seed,
        attempt.name,
        raw_sha,
        raw_size,
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(expected):
        raise ForagerMatchedCampaignError("raw archive binding differs from actual opaque file")
    return payload, digest


def _validate_bundle(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    attempt: Path,
    binding: Mapping[str, Any],
    *,
    repair_pairs: bool = False,
) -> tuple[executor.SeedExecutionArtifacts, str]:
    payload, digest = _load_json_pair(
        attempt / "bundle.json",
        "seed artifact bundle",
        repair_missing_sidecar=repair_pairs,
    )
    artifact = executor.parse_seed_artifact_bundle(payload, plan=context.rebuilt.plan)
    if artifact.candidate_id != candidate_id or artifact.seed != seed:
        raise ForagerMatchedCampaignError("seed artifact bundle belongs to another cell")
    raw = cast(Mapping[str, Any], binding["raw_archive"])
    if (
        artifact.live_runtime_identity_sha256 != context.live_runtime.identity_sha256
        or artifact.raw_artifact["container_export_sha256"] != raw["sha256"]
        or artifact.raw_artifact["container_export_size_bytes"] != raw["size_bytes"]
    ):
        raise ForagerMatchedCampaignError("seed bundle differs from raw/runtime binding")
    return artifact, digest


def _completion_pointer(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    attempt: Path,
    raw_binding_sha256: str,
    bundle_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": MATCHED_COMPLETION_POINTER_SCHEMA_VERSION,
        "classification": "write_once_content_completion_pointer",
        "stage": context.rebuilt.protocol.stage,
        "candidate_id": candidate_id,
        "seed": seed,
        "attempt": attempt.name,
        "raw_binding_sha256": raw_binding_sha256,
        "seed_artifact_bundle_sha256": bundle_sha256,
        "protocol_sha256": context.rebuilt.protocol.protocol_sha256,
        "execution_plan_sha256": context.rebuilt.plan.plan_sha256,
        "live_runtime_identity_sha256": context.live_runtime.identity_sha256,
        "promotion_authorized": False,
        "external_verification_required": True,
    }


@dataclass(frozen=True, slots=True)
class _CellScan:
    artifact: executor.SeedExecutionArtifacts | None
    completed_attempt: Path | None
    raw_binding_sha256: str | None
    bundle_sha256: str | None
    resumable_attempt: Path | None
    resumable_binding: Mapping[str, Any] | None
    next_attempt_number: int
    pointer_present: bool
    retained_raw_bytes: int


def _validate_failures(
    attempt: Path,
    candidate_id: str,
    seed: int,
    *,
    repair_pairs: bool = False,
) -> None:
    failures = attempt / "failures"
    if not failures.exists():
        return
    _regular_directory(failures, "attempt failures directory")
    names = _bounded_entry_names(
        failures,
        "attempt failure ledger",
        maximum=2 * _MAX_FAILURE_RECORDS_PER_ATTEMPT,
    )
    json_names = sorted(name for name in names if _FAILURE_RE.fullmatch(name))
    if len(json_names) > _MAX_FAILURE_RECORDS_PER_ATTEMPT:
        raise ForagerMatchedCampaignError("attempt contains too many failure records")
    if repair_pairs:
        for name in json_names:
            sidecar_name = f"{name}.sha256"
            if sidecar_name not in names:
                _load_json_pair(
                    failures / name,
                    "attempt failure record",
                    repair_missing_sidecar=True,
                )
                names.add(sidecar_name)
    expected_names = {
        *(json_names),
        *(f"{name}.sha256" for name in json_names),
    }
    if names != expected_names:
        raise ForagerMatchedCampaignError(
            "attempt failure ledger contains unknown/incomplete files"
        )
    for ordinal, name in enumerate(json_names, start=1):
        match = _FAILURE_RE.fullmatch(name)
        assert match is not None
        if int(match.group(1)) != ordinal:
            raise ForagerMatchedCampaignError("attempt failure ledger is not contiguous")
        payload, _digest = _load_json_pair(
            failures / name,
            "attempt failure record",
            repair_missing_sidecar=repair_pairs,
        )
        required = {
            "schema_version": MATCHED_ATTEMPT_FAILURE_SCHEMA_VERSION,
            "classification": "append_only_nonpromoting_failure_record",
            "candidate_id": candidate_id,
            "seed": seed,
            "attempt": attempt.name,
            "failure_ordinal": ordinal,
            "promotion_authorized": False,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise ForagerMatchedCampaignError("attempt failure record closure drifted")
        if set(payload) != {
            *required,
            "phase",
            "exception_type",
            "exception_message_sha256",
            "raw_binding_present",
        }:
            raise ForagerMatchedCampaignError("attempt failure record fields drifted")
        if (
            payload["phase"] not in {"candidate_and_scorer", "scorer_recovery"}
            or type(payload["exception_type"]) is not str
            or not payload["exception_type"]
            or type(payload["raw_binding_present"]) is not bool
            or type(payload["exception_message_sha256"]) is not str
            or _SHA256_RE.fullmatch(payload["exception_message_sha256"]) is None
        ):
            raise ForagerMatchedCampaignError("attempt failure record value drifted")


def _scan_cell(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    *,
    repair_pairs: bool = False,
) -> _CellScan:
    run_cell, completion_path = _cell_paths(context.root, candidate_id, seed)
    attempts: list[Path] = []
    if run_cell.exists():
        _regular_directory(run_cell, "cell run directory")
        names = sorted(
            _bounded_entry_names(
                run_cell,
                "cell run directory",
                maximum=_MAX_ATTEMPTS_PER_CELL,
            )
        )
        numbers: list[int] = []
        for name in names:
            match = _ATTEMPT_RE.fullmatch(name)
            if match is None:
                raise ForagerMatchedCampaignError("cell run directory contains unknown entries")
            number = int(match.group(1))
            if number < 1:
                raise ForagerMatchedCampaignError("attempt numbering starts at one")
            numbers.append(number)
            attempts.append(_regular_directory(run_cell / name, "attempt directory"))
        if numbers != list(range(1, len(numbers) + 1)):
            raise ForagerMatchedCampaignError("attempt directories are not contiguous")
    completed: list[
        tuple[Path, executor.SeedExecutionArtifacts, str, str]
    ] = []
    resumable: list[tuple[Path, Mapping[str, Any]]] = []
    retained_raw_bytes = 0
    for attempt in attempts:
        allowed = {
            "raw-output.tar",
            "raw-binding.json",
            "raw-binding.json.sha256",
            "bundle.json",
            "bundle.json.sha256",
            "failures",
        }
        _validate_directory_members(attempt, allowed, "attempt directory")
        _validate_failures(
            attempt,
            candidate_id,
            seed,
            repair_pairs=repair_pairs,
        )
        raw_exists = (attempt / "raw-output.tar").exists()
        if raw_exists:
            try:
                raw_metadata = (attempt / "raw-output.tar").lstat()
            except OSError as exc:
                raise ForagerMatchedCampaignError("opaque raw archive disappeared") from exc
            if (
                not stat.S_ISREG(raw_metadata.st_mode)
                or raw_metadata.st_nlink != 1
                or raw_metadata.st_size < 1
                or raw_metadata.st_size > _MAX_RAW_BYTES
            ):
                raise ForagerMatchedCampaignError(
                    "opaque raw archive is not a bounded single-link file"
                )
            retained_raw_bytes += raw_metadata.st_size
            if retained_raw_bytes > _MAX_RETAINED_RAW_BYTES_PER_CELL:
                raise ForagerMatchedCampaignError(
                    "cell exceeds its retained raw-byte bound"
                )
        binding_parts = (
            (attempt / "raw-binding.json").exists(),
            (attempt / "raw-binding.json.sha256").exists(),
        )
        bundle_parts = (
            (attempt / "bundle.json").exists(),
            (attempt / "bundle.json.sha256").exists(),
        )
        if binding_parts == (True, False) and repair_pairs:
            _load_json_pair(
                attempt / "raw-binding.json",
                "raw archive binding",
                repair_missing_sidecar=True,
            )
            binding_parts = (True, True)
        if bundle_parts == (True, False) and repair_pairs:
            _load_json_pair(
                attempt / "bundle.json",
                "seed artifact bundle",
                repair_missing_sidecar=True,
            )
            bundle_parts = (True, True)
        if binding_parts[0] != binding_parts[1] or bundle_parts[0] != bundle_parts[1]:
            raise ForagerMatchedCampaignError("attempt contains an incomplete canonical pair")
        if not any(binding_parts):
            if any(bundle_parts):
                raise ForagerMatchedCampaignError("bundle exists without a raw binding")
            # Empty or raw-only attempts are preserved but never resumed: there is no durable
            # execution binding proving which invocation produced their bytes.
            if raw_exists:
                _stable_file_hash(
                    attempt / "raw-output.tar",
                    "unbound opaque raw OCI archive",
                    maximum=_MAX_RAW_BYTES,
                )
            continue
        if not raw_exists:
            raise ForagerMatchedCampaignError("raw binding exists without its opaque archive")
        binding, binding_sha = _validate_raw_binding(
            context,
            candidate_id,
            seed,
            attempt,
            repair_pairs=repair_pairs,
        )
        if any(bundle_parts):
            artifact, bundle_sha = _validate_bundle(
                context,
                candidate_id,
                seed,
                attempt,
                binding,
                repair_pairs=repair_pairs,
            )
            completed.append((attempt, artifact, binding_sha, bundle_sha))
        else:
            resumable.append((attempt, binding))
    if len(completed) > 1 or (completed and resumable) or len(resumable) > 1:
        raise ForagerMatchedCampaignError("cell contains ambiguous completed/resumable attempts")

    completion_parts = (completion_path.exists(), completion_path.with_name(
        completion_path.name + ".sha256"
    ).exists())
    if completion_parts == (True, False) and repair_pairs:
        _load_json_pair(
            completion_path,
            "cell completion pointer",
            repair_missing_sidecar=True,
        )
        completion_parts = (True, True)
    if completion_parts[0] != completion_parts[1]:
        raise ForagerMatchedCampaignError("completion pointer pair is incomplete")
    pointer_present = all(completion_parts)
    if pointer_present:
        if len(completed) != 1:
            raise ForagerMatchedCampaignError("completion pointer has no unique completed bundle")
        attempt, artifact, binding_sha, bundle_sha = completed[0]
        pointer, _pointer_sha = _load_json_pair(
            completion_path,
            "cell completion pointer",
            repair_missing_sidecar=repair_pairs,
        )
        expected_pointer = _completion_pointer(
            context,
            candidate_id,
            seed,
            attempt,
            binding_sha,
            bundle_sha,
        )
        if pointer != expected_pointer:
            raise ForagerMatchedCampaignError("cell completion pointer drifted")
        return _CellScan(
            artifact,
            attempt,
            binding_sha,
            bundle_sha,
            None,
            None,
            len(attempts) + 1,
            True,
            retained_raw_bytes,
        )
    if completed:
        attempt, artifact, binding_sha, bundle_sha = completed[0]
        return _CellScan(
            artifact,
            attempt,
            binding_sha,
            bundle_sha,
            None,
            None,
            len(attempts) + 1,
            False,
            retained_raw_bytes,
        )
    resumable_attempt = resumable[0][0] if resumable else None
    resumable_binding = resumable[0][1] if resumable else None
    return _CellScan(
        None,
        None,
        None,
        None,
        resumable_attempt,
        resumable_binding,
        len(attempts) + 1,
        False,
        retained_raw_bytes,
    )


def _validate_dynamic_roots(context: _CampaignContext) -> None:
    candidate_ids = set(context.rebuilt.candidate_ids)
    seed_names = {f"seed-{seed}" for seed in context.rebuilt.protocol.active_seeds}
    completion_names = {
        *(f"seed-{seed}.json" for seed in context.rebuilt.protocol.active_seeds),
        *(f"seed-{seed}.json.sha256" for seed in context.rebuilt.protocol.active_seeds),
    }
    for root_name in ("runs", "completions"):
        dynamic_root = _regular_directory(
            context.root / root_name,
            f"campaign {root_name} directory",
        )
        names = _bounded_entry_names(
            dynamic_root,
            f"campaign {root_name} directory",
            maximum=len(candidate_ids),
        )
        if not names <= candidate_ids:
            raise ForagerMatchedCampaignError(
                f"{root_name} contains unknown or descriptive candidate directories"
            )
        for candidate_id in names:
            candidate_root = _regular_directory(
                dynamic_root / candidate_id,
                f"{root_name} candidate directory",
            )
            allowed = seed_names if root_name == "runs" else completion_names
            actual = _bounded_entry_names(
                candidate_root,
                f"{root_name} candidate directory",
                maximum=len(allowed),
            )
            if not actual <= allowed:
                raise ForagerMatchedCampaignError(
                    f"{root_name} candidate directory contains an unknown seed"
                )


def _scan_all_cells(
    context: _CampaignContext,
    *,
    repair_pairs: bool = False,
) -> dict[tuple[str, int], _CellScan]:
    _validate_dynamic_roots(context)
    scans: dict[tuple[str, int], _CellScan] = {}
    retained_raw_bytes = 0
    for candidate_id in context.rebuilt.candidate_ids:
        for seed in context.rebuilt.protocol.active_seeds:
            scans[(candidate_id, seed)] = _scan_cell(
                context,
                candidate_id,
                seed,
                repair_pairs=repair_pairs,
            )
            retained_raw_bytes += scans[(candidate_id, seed)].retained_raw_bytes
            if retained_raw_bytes > _MAX_RETAINED_RAW_BYTES_PER_CAMPAIGN:
                raise ForagerMatchedCampaignError(
                    "campaign exceeds its retained raw-byte bound"
                )
    return scans


def _ensure_regular_directory(path: Path, label: str) -> Path:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        return _regular_directory(path, label)
    return _regular_directory(path, label)


def _publish_completion_pointer(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    scan: _CellScan,
) -> None:
    if (
        scan.artifact is None
        or scan.completed_attempt is None
        or scan.raw_binding_sha256 is None
        or scan.bundle_sha256 is None
        or scan.pointer_present
    ):
        raise ForagerMatchedCampaignError("cannot publish completion for this cell state")
    _ensure_regular_directory(
        context.root / "completions" / candidate_id,
        "completion candidate directory",
    )
    _run_cell, completion_path = _cell_paths(context.root, candidate_id, seed)
    _publish_json_pair(
        completion_path,
        _completion_pointer(
            context,
            candidate_id,
            seed,
            scan.completed_attempt,
            scan.raw_binding_sha256,
            scan.bundle_sha256,
        ),
    )


def _persist_failure(
    attempt: Path,
    candidate_id: str,
    seed: int,
    *,
    phase: str,
    error: Exception,
    raw_binding_present: bool,
) -> None:
    failures = _ensure_regular_directory(attempt / "failures", "attempt failures directory")
    names = _bounded_entry_names(
        failures,
        "attempt failure ledger",
        maximum=2 * _MAX_FAILURE_RECORDS_PER_ATTEMPT,
    )
    existing_json = sorted(name for name in names if _FAILURE_RE.fullmatch(name))
    ordinal = len(existing_json) + 1
    if ordinal > _MAX_FAILURE_RECORDS_PER_ATTEMPT:
        raise ForagerMatchedCampaignError("attempt failure-record bound is exhausted")
    value = {
        "schema_version": MATCHED_ATTEMPT_FAILURE_SCHEMA_VERSION,
        "classification": "append_only_nonpromoting_failure_record",
        "candidate_id": candidate_id,
        "seed": seed,
        "attempt": attempt.name,
        "failure_ordinal": ordinal,
        "phase": phase,
        "exception_type": type(error).__name__,
        "exception_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "raw_binding_present": raw_binding_present,
        "promotion_authorized": False,
    }
    _publish_json_pair(failures / f"failure-{ordinal:06d}.json", value)


def _execute_with_optional_runner(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    raw_path: Path,
    runner: executor.ProcessRunner | None,
) -> executor.SeedExecutionArtifacts:
    if runner is None:
        return executor.execute_seed(
            context.rebuilt.plan,
            candidate_id,
            seed,
            raw_path,
            context.live_runtime,
        )
    return executor.execute_seed(
        context.rebuilt.plan,
        candidate_id,
        seed,
        raw_path,
        context.live_runtime,
        runner=runner,
    )


def _score_with_optional_runner(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    raw_path: Path,
    binding: Mapping[str, Any],
    runner: executor.ProcessRunner | None,
) -> executor.SeedExecutionArtifacts:
    raw = cast(Mapping[str, Any], binding["raw_archive"])
    raw_sha256 = cast(str, raw["sha256"])
    raw_size = cast(int, raw["size_bytes"])
    if runner is None:
        return executor.score_seed_archive(
            context.rebuilt.plan,
            candidate_id,
            seed,
            raw_path,
            context.live_runtime,
            expected_raw_archive_sha256=raw_sha256,
            expected_raw_archive_size=raw_size,
        )
    return executor.score_seed_archive(
        context.rebuilt.plan,
        candidate_id,
        seed,
        raw_path,
        context.live_runtime,
        expected_raw_archive_sha256=raw_sha256,
        expected_raw_archive_size=raw_size,
        runner=runner,
    )


def _run_one_cell(
    context: _CampaignContext,
    candidate_id: str,
    seed: int,
    scan: _CellScan,
    runner: executor.ProcessRunner | None,
) -> None:
    if scan.artifact is not None:
        if not scan.pointer_present:
            _publish_completion_pointer(context, candidate_id, seed, scan)
        return
    if scan.resumable_attempt is not None:
        if scan.resumable_binding is None:
            raise ForagerMatchedCampaignError("resumable attempt has no raw binding")
        attempt = scan.resumable_attempt
        resume_binding = scan.resumable_binding
        phase = "scorer_recovery"
        try:
            artifact = _score_with_optional_runner(
                context,
                candidate_id,
                seed,
                attempt / "raw-output.tar",
                resume_binding,
                runner,
            )
        except Exception as exc:
            _persist_failure(
                attempt,
                candidate_id,
                seed,
                phase=phase,
                error=exc,
                raw_binding_present=True,
            )
            raise ForagerMatchedCampaignError(
                f"scorer-only recovery failed for {candidate_id} seed {seed}"
            ) from exc
        _publish_json_pair(attempt / "bundle.json", artifact.to_dict())
    else:
        run_cell, _completion = _cell_paths(context.root, candidate_id, seed)
        _ensure_regular_directory(run_cell.parent, "run candidate directory")
        _ensure_regular_directory(run_cell, "cell run directory")
        if scan.next_attempt_number > _MAX_ATTEMPTS_PER_CELL:
            raise ForagerMatchedCampaignError("attempt bound is exhausted")
        attempt = run_cell / f"attempt-{scan.next_attempt_number:06d}"
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ForagerMatchedCampaignError("attempt was created concurrently") from exc
        execution_binding: Mapping[str, Any] | None = None
        phase = "candidate_and_scorer"
        try:
            artifact = _execute_with_optional_runner(
                context,
                candidate_id,
                seed,
                attempt / "raw-output.tar",
                runner,
            )
            execution_binding, _binding_sha = _persist_raw_binding(
                context,
                candidate_id,
                seed,
                attempt,
            )
            raw = cast(Mapping[str, Any], execution_binding["raw_archive"])
            if (
                artifact.raw_artifact["container_export_sha256"] != raw["sha256"]
                or artifact.raw_artifact["container_export_size_bytes"] != raw["size_bytes"]
            ):
                raise ForagerMatchedCampaignError(
                    "executor result differs from its persisted opaque raw archive"
                )
        except Exception as exc:
            raw_path = attempt / "raw-output.tar"
            if execution_binding is None and raw_path.exists():
                try:
                    execution_binding, _binding_sha = _persist_raw_binding(
                        context,
                        candidate_id,
                        seed,
                        attempt,
                    )
                except Exception as binding_exc:
                    _persist_failure(
                        attempt,
                        candidate_id,
                        seed,
                        phase=phase,
                        error=binding_exc,
                        raw_binding_present=False,
                    )
                    raise ForagerMatchedCampaignError(
                        f"raw binding recovery failed for {candidate_id} seed {seed}"
                    ) from binding_exc
            _persist_failure(
                attempt,
                candidate_id,
                seed,
                phase=phase,
                error=exc,
                raw_binding_present=execution_binding is not None,
            )
            raise ForagerMatchedCampaignError(
                f"matched execution failed for {candidate_id} seed {seed}"
            ) from exc
        _publish_json_pair(attempt / "bundle.json", artifact.to_dict())
    completed_scan = _scan_cell(context, candidate_id, seed)
    if completed_scan.artifact is None or completed_scan.pointer_present:
        raise ForagerMatchedCampaignError("completed attempt did not replay before publication")
    _publish_completion_pointer(context, candidate_id, seed, completed_scan)


def _ordered_artifacts(
    context: _CampaignContext,
    scans: Mapping[tuple[str, int], _CellScan],
) -> dict[str, tuple[executor.SeedExecutionArtifacts, ...]]:
    result: dict[str, tuple[executor.SeedExecutionArtifacts, ...]] = {}
    for candidate_id in context.rebuilt.candidate_ids:
        records: list[executor.SeedExecutionArtifacts] = []
        for seed in context.rebuilt.protocol.active_seeds:
            scan = scans[(candidate_id, seed)]
            if scan.artifact is None or not scan.pointer_present:
                raise ForagerMatchedCampaignError("score evidence requires exact completed cells")
            records.append(scan.artifact)
        result[candidate_id] = tuple(records)
    return result


def _completion_summary(
    context: _CampaignContext,
    receipt_index: executor.MatchedExecutionReceiptIndex,
    score_evidence: Any,
    request: executor.VerificationRequest,
) -> dict[str, Any]:
    return {
        "schema_version": MATCHED_OPEN_COMPLETION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": "open_tuning",
        "protocol_sha256": context.rebuilt.protocol.protocol_sha256,
        "qualification_manifest_sha256": context.rebuilt.bundle.manifest_sha256,
        "execution_plan_sha256": context.rebuilt.plan.plan_sha256,
        "source_manifest_sha256": context.rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": context.rebuilt.plan.executor_manifest_sha256,
        "live_runtime_identity_sha256": context.live_runtime.identity_sha256,
        "candidate_count": len(context.rebuilt.candidate_ids),
        "seed_count": len(context.rebuilt.protocol.active_seeds),
        "completed_cell_count": (
            len(context.rebuilt.candidate_ids)
            * len(context.rebuilt.protocol.active_seeds)
        ),
        "execution_receipt_index_payload_sha256": receipt_index.payload_sha256,
        "score_evidence_sha256": score_evidence.payload_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
        "verification_authentication_state": "unresolved_external_verifier_required",
        "selection_created": False,
        "sealed_protocol_created": False,
        "evaluation_artifacts_created": False,
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "host_reward_array_access": "forbidden_not_performed",
    }


def _validate_completion_summary_common(
    context: _CampaignContext,
    receipt_index: executor.MatchedExecutionReceiptIndex,
    score_evidence: MatchedScoreEvidence,
    request: executor.VerificationRequest,
    summary: Mapping[str, Any],
) -> None:
    if not isinstance(summary, Mapping):
        raise ForagerMatchedCampaignError("completion summary builder returned a non-mapping")
    required: dict[str, Any] = {
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": context.rebuilt.protocol.stage,
        "protocol_sha256": context.rebuilt.protocol.protocol_sha256,
        "qualification_manifest_sha256": context.rebuilt.bundle.manifest_sha256,
        "execution_plan_sha256": context.rebuilt.plan.plan_sha256,
        "source_manifest_sha256": context.rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": context.rebuilt.plan.executor_manifest_sha256,
        "live_runtime_identity_sha256": context.live_runtime.identity_sha256,
        "candidate_count": len(context.rebuilt.candidate_ids),
        "seed_count": len(context.rebuilt.protocol.active_seeds),
        "completed_cell_count": (
            len(context.rebuilt.candidate_ids)
            * len(context.rebuilt.protocol.active_seeds)
        ),
        "execution_receipt_index_payload_sha256": receipt_index.payload_sha256,
        "score_evidence_sha256": score_evidence.payload_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "host_reward_array_access": "forbidden_not_performed",
    }
    drifted = [name for name, expected in required.items() if summary.get(name) != expected]
    if drifted:
        raise ForagerMatchedCampaignError(
            "completion summary violates common closure/authority invariants: "
            + ", ".join(drifted)
        )


def _validate_open_completion_summary(
    context: _CampaignContext,
    receipt_index: executor.MatchedExecutionReceiptIndex,
    score_evidence: MatchedScoreEvidence,
    request: executor.VerificationRequest,
    summary: Mapping[str, Any],
) -> None:
    _validate_completion_summary_common(
        context,
        receipt_index,
        score_evidence,
        request,
        summary,
    )
    campaign_summary = _plain(summary)
    expected = _completion_summary(context, receipt_index, score_evidence, request)
    if campaign_summary != expected:
        raise ForagerMatchedCampaignError(
            "open completion summary differs from its exact v2 schema"
        )


def _build_completed_campaign_bundle(
    context: _CampaignContext,
    scans: Mapping[tuple[str, int], _CellScan],
    *,
    create: bool,
    completion_summary_builder: _CompletionSummaryBuilder = _completion_summary,
    completion_summary_validator: _CompletionSummaryValidator = (
        _validate_open_completion_summary
    ),
) -> CompletedCampaignBundle:
    artifacts = _ordered_artifacts(context, scans)
    receipt_index = executor.build_execution_receipt_index(
        context.rebuilt.plan,
        artifacts,
    )
    scores = executor.build_score_evidence(context.rebuilt.plan, artifacts)
    if tuple(
        item.execution_receipt_sha256 for item in receipt_index.execution_receipts
    ) != tuple(item.execution_receipt_sha256 for item in scores.candidate_scores):
        raise ForagerMatchedCampaignError(
            "receipt index and score evidence execution-receipt hashes differ"
        )
    request = executor.build_verification_request(context.rebuilt.plan, scores)
    completion_summary = completion_summary_builder(
        context,
        receipt_index,
        scores,
        request,
    )
    completion_summary_validator(
        context,
        receipt_index,
        scores,
        request,
        completion_summary,
    )
    values: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("execution-receipt-index.json", receipt_index.to_dict()),
        ("score-evidence.json", scores.to_dict()),
        ("verification-request.json", request.to_dict()),
        ("completion-summary.json", completion_summary),
    )
    present = {
        name: ((context.root / name).exists(), (context.root / f"{name}.sha256").exists())
        for name, _value in values
    }
    final_file_sha256: dict[str, str] = {}
    if create:
        for name, value in values:
            parts = present[name]
            if any(parts):
                if parts == (True, False):
                    final_file_sha256[name] = _expect_artifact(
                        context.root,
                        name,
                        value,
                        repair_missing_sidecar=True,
                    )
                elif not all(parts):
                    raise ForagerMatchedCampaignError("final artifact pair is incomplete")
                else:
                    final_file_sha256[name] = _expect_artifact(context.root, name, value)
            else:
                final_file_sha256[name] = _publish_json_pair(context.root / name, value)
    else:
        if any(any(parts) for parts in present.values()):
            if not all(all(parts) for parts in present.values()):
                raise ForagerMatchedCampaignError("final artifact set is incomplete")
            for name, value in values:
                final_file_sha256[name] = _expect_artifact(context.root, name, value)
        else:
            raise ForagerMatchedCampaignError("complete block is missing final artifacts")
    loaded_index = executor.load_execution_receipt_index(
        context.root / "execution-receipt-index.json",
        plan=context.rebuilt.plan,
        artifacts=artifacts,
        expected_payload_sha256=receipt_index.payload_sha256,
    )
    if loaded_index != receipt_index:
        raise ForagerMatchedCampaignError("persisted execution receipt index replay drifted")
    return CompletedCampaignBundle(
        output_root=context.root,
        protocol=context.rebuilt.protocol,
        plan=context.rebuilt.plan,
        live_runtime=context.live_runtime,
        candidate_ids=context.rebuilt.candidate_ids,
        active_seeds=context.rebuilt.protocol.active_seeds,
        schedule=cast(Mapping[str, Any], _freeze_json(context.rebuilt.schedule)),
        seed_artifacts=MappingProxyType(
            {candidate_id: tuple(records) for candidate_id, records in artifacts.items()}
        ),
        execution_receipt_index=receipt_index,
        score_evidence=scores,
        verification_request=request,
        completion_summary=cast(Mapping[str, Any], _freeze_json(completion_summary)),
        final_file_sha256=MappingProxyType(final_file_sha256),
    )


def _finalize_or_validate(
    context: _CampaignContext,
    scans: Mapping[tuple[str, int], _CellScan],
    *,
    create: bool,
    completion_summary_builder: _CompletionSummaryBuilder = _completion_summary,
    completion_summary_validator: _CompletionSummaryValidator = (
        _validate_open_completion_summary
    ),
) -> tuple[str, str]:
    bundle = _build_completed_campaign_bundle(
        context,
        scans,
        create=create,
        completion_summary_builder=completion_summary_builder,
        completion_summary_validator=completion_summary_validator,
    )
    return (
        bundle.score_evidence.payload_sha256,
        bundle.verification_request.verification_subject_sha256,
    )


def _derive_status(
    context: _CampaignContext,
    scans: Mapping[tuple[str, int], _CellScan],
    *,
    completion_summary_builder: _CompletionSummaryBuilder = _completion_summary,
    completion_summary_validator: _CompletionSummaryValidator = (
        _validate_open_completion_summary
    ),
) -> CampaignStatus:
    ordered_cells = [
        (cast(str, cell["candidate_id"]), cast(int, cell["seed"]))
        for cell in cast(Sequence[Mapping[str, Any]], context.rebuilt.schedule["cells"])
    ]
    completed = sum(
        scans[cell].artifact is not None and scans[cell].pointer_present for cell in ordered_cells
    )
    recoverable = next(
        (
            cell
            for cell in ordered_cells
            if (
                (scans[cell].artifact is not None and not scans[cell].pointer_present)
                or scans[cell].resumable_attempt is not None
            )
        ),
        None,
    )
    next_cell = next(
        (
            cell
            for cell in ordered_cells
            if scans[cell].artifact is None or not scans[cell].pointer_present
        ),
        None,
    )
    score_sha: str | None = None
    subject_sha: str | None = None
    total = len(ordered_cells)
    if completed == total:
        score_sha, subject_sha = _finalize_or_validate(
            context,
            scans,
            create=False,
            completion_summary_builder=completion_summary_builder,
            completion_summary_validator=completion_summary_validator,
        )
        state = "complete_content_only_external_verification_unresolved"
    else:
        if any(
            (context.root / name).exists()
            or (context.root / f"{name}.sha256").exists()
            for name in _FINAL_ARTIFACTS
        ):
            raise ForagerMatchedCampaignError("final evidence exists before exact block completion")
        state = "recovery_required" if recoverable is not None else "in_progress"
    return CampaignStatus(
        output_root=context.root,
        state=state,
        completed_cells=completed,
        total_cells=total,
        next_candidate_id=None if next_cell is None else next_cell[0],
        next_seed=None if next_cell is None else next_cell[1],
        protocol_sha256=context.rebuilt.protocol.protocol_sha256,
        qualification_manifest_sha256=context.rebuilt.bundle.manifest_sha256,
        plan_sha256=context.rebuilt.plan.plan_sha256,
        live_runtime_identity_sha256=context.live_runtime.identity_sha256,
        score_evidence_sha256=score_sha,
        verification_subject_sha256=subject_sha,
    )


def _load_context_and_status(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path,
    runner: executor.ProcessRunner | None,
) -> tuple[_CampaignContext, CampaignStatus]:
    context = _load_context(
        qualification_root,
        output_root,
        runtime=runtime,
        runner=runner,
    )
    scans = _scan_all_cells(context)
    return context, _derive_status(context, scans)


def _run_open_tuning_campaign_locked(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
    max_cells: int | None = None,
) -> CampaignStatus:
    """Resume sequential execution; a failure preserves all attempts and stops immediately."""
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ForagerMatchedCampaignError("max_cells must be a positive integer or None")
    context = _load_context(
        qualification_root,
        output_root,
        runtime=runtime,
        runner=runner,
    )
    return _run_resumable_context_locked(
        context,
        runner=runner,
        max_cells=max_cells,
    )


def _run_resumable_context_locked(
    context: _CampaignContext,
    *,
    runner: executor.ProcessRunner | None,
    max_cells: int | None,
    completion_summary_builder: _CompletionSummaryBuilder = _completion_summary,
    completion_summary_validator: _CompletionSummaryValidator = (
        _validate_open_completion_summary
    ),
    mutation_guard: Callable[[], None] | None = None,
) -> CampaignStatus:
    """Resume a replayed context while the caller holds its exclusive writer lock.

    Authority-bearing callers must also authenticate before entry.  ``mutation_guard`` may
    recheck a held root identity immediately before each append-only mutation boundary.
    """
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ForagerMatchedCampaignError("max_cells must be a positive integer or None")
    if mutation_guard is not None:
        mutation_guard()
    scans = _scan_all_cells(context, repair_pairs=True)
    if mutation_guard is not None:
        mutation_guard()
    executed = 0
    for cell in cast(Sequence[Mapping[str, Any]], context.rebuilt.schedule["cells"]):
        candidate_id = cast(str, cell["candidate_id"])
        seed = cast(int, cell["seed"])
        scan = scans[(candidate_id, seed)]
        if scan.artifact is not None and scan.pointer_present:
            continue
        if mutation_guard is not None:
            mutation_guard()
        _run_one_cell(context, candidate_id, seed, scan, runner)
        if mutation_guard is not None:
            mutation_guard()
        executed += 1
        scans[(candidate_id, seed)] = _scan_cell(context, candidate_id, seed)
        if max_cells is not None and executed >= max_cells:
            break
    scans = _scan_all_cells(context)
    if all(scan.artifact is not None and scan.pointer_present for scan in scans.values()):
        if mutation_guard is not None:
            mutation_guard()
        _finalize_or_validate(
            context,
            scans,
            create=True,
            completion_summary_builder=completion_summary_builder,
            completion_summary_validator=completion_summary_validator,
        )
        if mutation_guard is not None:
            mutation_guard()
    return _derive_status(
        context,
        scans,
        completion_summary_builder=completion_summary_builder,
        completion_summary_validator=completion_summary_validator,
    )


def run_open_tuning_campaign(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
    max_cells: int | None = None,
) -> CampaignStatus:
    """Acquire the single-writer lease and resume the sequential tuning campaign."""
    with _campaign_lock(output_root, exclusive=True):
        return _run_open_tuning_campaign_locked(
            qualification_root,
            output_root,
            runtime=runtime,
            runner=runner,
            max_cells=max_cells,
        )


def verify_open_tuning_campaign(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> CampaignStatus:
    """Rebuild and validate all immutable, attempt, raw, and completion bindings."""
    with _campaign_lock(output_root, exclusive=False):
        return _load_context_and_status(
            qualification_root,
            output_root,
            runtime=runtime,
            runner=runner,
        )[1]


def load_completed_open_tuning_campaign(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> CompletedCampaignBundle:
    """Load the exact completed content bundle without resolving external trust.

    The shared campaign lock is held while the qualification, immutable inputs, every cell, and
    all four final artifacts are rebuilt and compared.  This loader never repairs artifacts and
    the returned verification request remains externally unresolved.
    """
    with _campaign_lock(output_root, exclusive=False):
        context = _load_context(
            qualification_root,
            output_root,
            runtime=runtime,
            runner=runner,
        )
        scans = _scan_all_cells(context)
        if not all(
            scan.artifact is not None and scan.pointer_present for scan in scans.values()
        ):
            if any(
                (context.root / name).exists()
                or (context.root / f"{name}.sha256").exists()
                for name in _FINAL_ARTIFACTS
            ):
                raise ForagerMatchedCampaignError(
                    "final evidence exists before exact block completion"
                )
            raise ForagerMatchedCampaignError("open tuning campaign is not complete")
        return _build_completed_campaign_bundle(context, scans, create=False)


def campaign_status(
    qualification_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> CampaignStatus:
    """Return fully revalidated derived status (including live-runtime requalification)."""
    return verify_open_tuning_campaign(
        qualification_root,
        output_root,
        runtime=runtime,
        runner=runner,
    )


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "run", "verify", "status"):
        command = subparsers.add_parser(operation, allow_abbrev=False)
        command.add_argument("--qualification-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--runtime", default="docker")
        if operation == "run":
            command.add_argument("--max-cells", type=int)
    arguments = parser.parse_args(argv)
    if arguments.operation == "prepare":
        status = prepare_open_tuning_campaign(
            arguments.qualification_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    elif arguments.operation == "run":
        status = run_open_tuning_campaign(
            arguments.qualification_root,
            arguments.output_root,
            runtime=arguments.runtime,
            max_cells=arguments.max_cells,
        )
    elif arguments.operation == "verify":
        status = verify_open_tuning_campaign(
            arguments.qualification_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    else:
        status = campaign_status(
            arguments.qualification_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    sys.stdout.buffer.write(canonical_json_bytes(status.to_dict()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prepare/run/verify/status CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return _cli(tuple(argv) if argv is not None else tuple(sys.argv[1:]))
    except (OSError, ValueError) as exc:
        _LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CampaignStatus",
    "CompletedCampaignBundle",
    "ForagerMatchedCampaignError",
    "MATCHED_ATTEMPT_FAILURE_SCHEMA_VERSION",
    "MATCHED_COMPLETION_POINTER_SCHEMA_VERSION",
    "MATCHED_OPEN_CAMPAIGN_SCHEMA_VERSION",
    "MATCHED_OPEN_COMPLETION_SCHEMA_VERSION",
    "MATCHED_OPEN_SCHEDULE_SCHEMA_VERSION",
    "MATCHED_RAW_BINDING_SCHEMA_VERSION",
    "build_seed_major_schedule",
    "campaign_status",
    "canonical_json_bytes",
    "load_completed_open_tuning_campaign",
    "main",
    "prepare_open_tuning_campaign",
    "run_open_tuning_campaign",
    "selection_candidate_ids",
    "verify_open_tuning_campaign",
]
