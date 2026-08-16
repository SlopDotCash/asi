"""Atomic, content-replayable sealing for the matched-current Forager protocol.

The seal bundle is the boundary between completed open tuning and held-out
evaluation.  It snapshots the compact completed-campaign closure, asks the
configured external resolver to authenticate that exact closure, computes the
frozen selection, and mechanically seals the protocol.  It never runs an
evaluation cell and never grants promotion authority.

Persisted :class:`AuthenticatedEvidenceBindings` bytes are only a cache of the
resolver result used at seal time.  :func:`load_forager_matched_seal_bundle_content`
checks canonical content and deterministic replay without treating that cache
as authentication.  Consumers needing authority must additionally call
:func:`authenticate_forager_matched_seal_bundle`, which invokes the resolver
again and requires an exact cache match.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol

MATCHED_SEAL_BUNDLE_SCHEMA_VERSION: Final = "alberta.forager_matched_seal_bundle.v2"
MATCHED_SEALED_TRANSITION_SCHEMA_VERSION: Final = (
    evaluation.MATCHED_SEALED_TRANSITION_SCHEMA_VERSION
)

_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_NODES: Final = 1_000_000
_MAX_JSON_DEPTH: Final = 64
_MAX_ROOT_ENTRIES: Final = 32
_MAX_MATCHED_CANDIDATES: Final = 23
_MAX_MATCHED_TUNING_SEEDS: Final = 10
_MAX_MATCHED_EVALUATION_SEEDS: Final = 30
_MAX_MATCHED_SELECTION_GROUPS: Final = 2
_MAX_MATCHED_SELECTION_GROUP_SIZE: Final = 14
_MAX_MATCHED_SELECTION_BOOTSTRAP_RESAMPLES: Final = 10_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_ARTIFACT_PATHS: Final = MappingProxyType(
    {
        "open_protocol": "open-protocol.json",
        "open_execution_plan": "open-execution-plan.json",
        "open_live_runtime": "open-live-runtime.json",
        "open_execution_receipt_index": "open-execution-receipt-index.json",
        "open_score_evidence": "open-score-evidence.json",
        "open_verification_request": "open-verification-request.json",
        "open_completion_summary": "open-completion-summary.json",
        "open_authenticated_bindings_cache": "open-authenticated-bindings-cache.json",
        "selection_result": "selection-result.json",
        "selection_report": "selection-report.json",
        "sealed_protocol": "sealed-protocol.json",
    }
)
_MANIFEST_NAME: Final = "seal.json"


class ForagerMatchedSealError(ValueError):
    """A seal input, bundle, publication, or trust resolution failed closed."""


class PublishedSealUncertainError(ForagerMatchedSealError):
    """Publication occurred, but its durability or final verification is uncertain."""

    def __init__(self, destination: Path, detail: str) -> None:
        self.destination = destination
        super().__init__(f"seal published at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class ContentVerifiedSealBundle:
    """A deterministically replayed seal whose external trust is unchecked.

    ``recorded_bindings_cache`` is immutable plain content, not an
    :class:`~alberta_framework.benchmarks.forager_matched_evidence.AuthenticatedEvidenceBindings`
    authority object.  Only a fresh resolver call can authenticate the receipt
    it names.
    """

    output_root: Path
    manifest: Mapping[str, Any]
    open_protocol: protocol.ForagerMatchedProtocol
    open_score_evidence: evidence.MatchedScoreEvidence
    open_verification_request: executor.VerificationRequest
    recorded_bindings_cache: Mapping[str, Any]
    selection_result: protocol.ForagerMatchedSelectionResult
    selection_report: Mapping[str, Any]
    sealed_protocol: protocol.ForagerMatchedProtocol
    sealed_transition: Mapping[str, Any]
    sealed_transition_sha256: str


@dataclass(frozen=True, slots=True)
class _OpenDirectory:
    """A directory kept open so all operations stay bound to one inode."""

    path: Path
    descriptor: int
    inode_identity: tuple[int, int, int]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ForagerMatchedSealError("canonical mappings require string keys")
            result[key] = _plain(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ForagerMatchedSealError(
        f"canonical content contains unsupported {type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return compact canonical ASCII JSON for seal-owned artifacts."""
    try:
        return json.dumps(
            _plain(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedSealError("value is not canonical JSON") from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedSealError(f"{label} must be a plain object")
    return cast(dict[str, Any], value)


def _require_array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedSealError(f"{label} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ForagerMatchedSealError(
            f"{label} fields differ (missing={missing}, extra={extra})"
        )


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedSealError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ForagerMatchedSealError(f"{label} must be a nonempty bounded identifier")
    if not all(character.isalnum() or character in "._-" for character in value):
        raise ForagerMatchedSealError(f"{label} is not a portable identifier")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ForagerMatchedSealError(f"{label} must be an integer >= {minimum}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedSealError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedSealError(f"non-finite JSON number {value!r} is forbidden")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ForagerMatchedSealError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _validate_json_complexity(value: Any, label: str) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedSealError(f"{label} exceeds the JSON node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedSealError(f"{label} exceeds the JSON depth bound")
        if type(item) is dict:
            pending.extend(
                (child, depth + 1)
                for child in cast(dict[str, Any], item).values()
            )
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)


def _decode_canonical(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedSealError(f"{label} violates the JSON byte bound")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_json_float,
        )
    except ForagerMatchedSealError:
        raise
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedSealError(f"{label} is not strict UTF-8 JSON") from exc
    _validate_json_complexity(decoded, label)
    payload = _require_object(decoded, label)
    if canonical_json_bytes(payload) != raw:
        raise ForagerMatchedSealError(f"{label} is not canonical JSON")
    return payload


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


def _inode_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ForagerMatchedSealError(f"{label} is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ForagerMatchedSealError(f"{label} must be a non-symlink directory")
    return path.resolve()


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _open_stable_directory(path: Path, label: str) -> _OpenDirectory:
    try:
        path_metadata = path.lstat()
        canonical = path.resolve(strict=True)
        descriptor = os.open(canonical, _directory_open_flags())
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedSealError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(canonical)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or path.is_symlink()
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
        ):
            raise ForagerMatchedSealError(f"{label} changed while being opened")
        return _OpenDirectory(
            path=canonical,
            descriptor=descriptor,
            inode_identity=identity,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _open_stable_directory_at(
    parent: _OpenDirectory,
    name: str,
    path: Path,
    label: str,
) -> _OpenDirectory:
    try:
        path_metadata = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent.descriptor)
    except OSError as exc:
        raise ForagerMatchedSealError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
        ):
            raise ForagerMatchedSealError(f"{label} changed while being opened")
        return _OpenDirectory(path=path, descriptor=descriptor, inode_identity=identity)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_open_directory_path(opened: _OpenDirectory, label: str) -> None:
    try:
        descriptor_metadata = os.fstat(opened.descriptor)
        path_metadata = os.lstat(opened.path)
    except OSError as exc:
        raise ForagerMatchedSealError(f"{label} is no longer reachable") from exc
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or opened.path.is_symlink()
        or _inode_identity(descriptor_metadata) != opened.inode_identity
        or _inode_identity(path_metadata) != opened.inode_identity
    ):
        raise ForagerMatchedSealError(f"{label} no longer names the opened inode")


def _read_stable_regular_at(
    root: _OpenDirectory,
    name: str,
    label: str,
    *,
    maximum: int = _MAX_JSON_BYTES,
) -> bytes:
    try:
        path_metadata = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _file_open_flags(), dir_fd=root.descriptor)
    except OSError as exc:
        raise ForagerMatchedSealError(f"cannot safely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum
            or _stat_identity(path_metadata) != _stat_identity(before)
        ):
            raise ForagerMatchedSealError(f"{label} is not a bounded single-link file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedSealError(f"{label} ended while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedSealError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise ForagerMatchedSealError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _expected_root_names() -> set[str]:
    names = {*_ARTIFACT_PATHS.values(), _MANIFEST_NAME}
    return names | {f"{name}.sha256" for name in names}


def _root_inventory(root: _OpenDirectory) -> dict[str, tuple[int, ...]]:
    names: set[str] = set()
    inventory: dict[str, tuple[int, ...]] = {}
    try:
        iterator = os.scandir(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedSealError("cannot enumerate seal bundle root") from exc
    with iterator:
        for entry in iterator:
            if len(names) >= _MAX_ROOT_ENTRIES:
                raise ForagerMatchedSealError("seal bundle root exceeds its entry bound")
            if entry.name in names:
                raise ForagerMatchedSealError("seal bundle root repeats an entry")
            names.add(entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedSealError("cannot inspect seal bundle entry") from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ForagerMatchedSealError(
                    "seal bundle contains a link, hardlink, directory, or special file"
                )
            inventory[entry.name] = _stat_identity(metadata)
    expected = _expected_root_names()
    if names != expected:
        raise ForagerMatchedSealError(
            "seal bundle inventory differs "
            f"(missing={sorted(expected - names)}, extra={sorted(names - expected)})"
        )
    return inventory


def _load_pair_at(root: _OpenDirectory, name: str, label: str) -> tuple[bytes, str]:
    raw = _read_stable_regular_at(root, name, label)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _read_stable_regular_at(
        root,
        f"{name}.sha256",
        f"{label} SHA-256 sidecar",
        maximum=128,
    )
    if sidecar != f"{digest}\n".encode("ascii"):
        raise ForagerMatchedSealError(f"{label} SHA-256 sidecar differs")
    return raw, digest


def _parse_recorded_bindings(
    value: Mapping[str, Any],
) -> evidence.AuthenticatedEvidenceBindings:
    payload = dict(value)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "stage",
            "protocol_sha256",
            "score_evidence_sha256",
            "source_manifest_sha256",
            "executor_manifest_sha256",
            "qualification_manifest_sha256",
            "execution_closure_sha256",
            "trust_anchor_identity",
            "verification_subject_sha256",
            "verification_receipt_sha256",
        },
        "recorded authenticated-bindings cache",
    )
    if (
        payload["schema_version"]
        != evidence.AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION
        or payload["stage"] != "open_tuning"
    ):
        raise ForagerMatchedSealError("recorded bindings cache schema/stage drifted")
    try:
        return evidence.AuthenticatedEvidenceBindings(
            stage="open_tuning",
            protocol_sha256=_require_sha256(payload["protocol_sha256"], "bindings protocol"),
            score_evidence_sha256=_require_sha256(
                payload["score_evidence_sha256"], "bindings score evidence"
            ),
            source_manifest_sha256=_require_sha256(
                payload["source_manifest_sha256"], "bindings source manifest"
            ),
            executor_manifest_sha256=_require_sha256(
                payload["executor_manifest_sha256"], "bindings executor manifest"
            ),
            qualification_manifest_sha256=_require_sha256(
                payload["qualification_manifest_sha256"],
                "bindings qualification manifest",
            ),
            execution_closure_sha256=_require_sha256(
                payload["execution_closure_sha256"], "bindings execution closure"
            ),
            trust_anchor_identity=_require_identifier(
                payload["trust_anchor_identity"], "bindings trust anchor"
            ),
            verification_subject_sha256=_require_sha256(
                payload["verification_subject_sha256"], "bindings verification subject"
            ),
            verification_receipt_sha256=_require_sha256(
                payload["verification_receipt_sha256"], "bindings verification receipt"
            ),
        )
    except evidence.ForagerMatchedEvidenceError as exc:
        raise ForagerMatchedSealError(f"recorded bindings cache is invalid: {exc}") from exc


def _validate_request_content(
    request: executor.VerificationRequest,
    open_protocol: protocol.ForagerMatchedProtocol,
    scores: evidence.MatchedScoreEvidence,
) -> None:
    expected = {
        "stage": scores.stage,
        "protocol_sha256": scores.protocol_sha256,
        "score_evidence_sha256": scores.payload_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "qualification_manifest_sha256": scores.qualification_manifest_sha256,
        "execution_closure_sha256": evidence.matched_execution_closure_sha256(
            open_protocol, scores
        ),
        "trust_anchor_identity": open_protocol.runtime.qualification_trust_anchor_identity,
        "verification_subject_sha256": evidence.matched_verification_subject_sha256(
            open_protocol, scores
        ),
    }
    actual = request.to_dict()
    drifted = [name for name, value in expected.items() if actual[name] != value]
    if drifted:
        raise ForagerMatchedSealError(
            "open verification request differs from score/protocol content: "
            + ", ".join(drifted)
        )


def _validate_execution_plan_snapshot(
    payload: Mapping[str, Any],
    open_protocol: protocol.ForagerMatchedProtocol,
    scores: evidence.MatchedScoreEvidence,
) -> str:
    value = dict(payload)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "classification",
            "promotion_authorized",
            "external_verification_required",
            "stage",
            "protocol_sha256",
            "qualification_manifest_sha256",
            "active_seeds",
            "horizon",
            "candidate_order",
            "source_manifest",
            "source_manifest_sha256",
            "executor_manifest",
            "executor_manifest_sha256",
            "candidate_command_templates",
            "scoring_boundary",
        },
        "open execution plan",
    )
    if (
        value["schema_version"] != executor.MATCHED_EXECUTION_PLAN_SCHEMA_VERSION
        or value["classification"] != "matched_current_execution_candidate"
        or value["promotion_authorized"] is not False
        or value["external_verification_required"] is not True
        or value["stage"] != "open_tuning"
    ):
        raise ForagerMatchedSealError("open execution plan schema/state drifted")
    active_seeds = tuple(
        _require_int(item, "execution plan seed")
        for item in _require_array(value["active_seeds"], "execution plan active seeds")
    )
    horizon = _require_int(value["horizon"], "execution plan horizon", minimum=1)
    candidate_order = tuple(
        _require_identifier(item, "execution plan candidate")
        for item in _require_array(value["candidate_order"], "execution plan candidate order")
    )
    source_manifest = _require_object(value["source_manifest"], "plan source manifest")
    executor_manifest = _require_object(
        value["executor_manifest"], "plan executor manifest"
    )
    source_digest = _require_sha256(
        value["source_manifest_sha256"], "plan source manifest"
    )
    executor_digest = _require_sha256(
        value["executor_manifest_sha256"], "plan executor manifest"
    )
    qualification_digest = _require_sha256(
        value["qualification_manifest_sha256"], "plan qualification manifest"
    )
    if (
        value["protocol_sha256"] != open_protocol.protocol_sha256
        or qualification_digest != scores.qualification_manifest_sha256
        or active_seeds != open_protocol.active_seeds
        or horizon != open_protocol.horizon
        or candidate_order
        != tuple(item.candidate_id for item in scores.candidate_scores)
        or _canonical_sha256(source_manifest) != source_digest
        or source_digest != scores.source_evidence_sha256
        or _canonical_sha256(executor_manifest) != executor_digest
        or executor_digest != scores.executor_evidence_sha256
        or executor_manifest.get("schema_version")
        != executor.MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION
        or executor_manifest.get("protocol_sha256") != open_protocol.protocol_sha256
        or executor_manifest.get("qualification_manifest_sha256")
        != qualification_digest
    ):
        raise ForagerMatchedSealError("open execution plan closure drifted")
    _require_array(
        value["candidate_command_templates"],
        "execution plan candidate command templates",
    )
    _require_object(value["scoring_boundary"], "execution plan scoring boundary")
    return _canonical_sha256(value)


def _validate_live_runtime_snapshot(
    payload: Mapping[str, Any],
    scores: evidence.MatchedScoreEvidence,
) -> str:
    value = dict(payload)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "executable_sha256",
            "version",
            "image_inspection",
            "executor_manifest_sha256",
        },
        "open live runtime",
    )
    if value["schema_version"] != executor.MATCHED_LIVE_RUNTIME_SCHEMA_VERSION:
        raise ForagerMatchedSealError("open live runtime schema drifted")
    _require_sha256(value["executable_sha256"], "live runtime executable")
    _require_object(value["version"], "live runtime version")
    _require_object(value["image_inspection"], "live runtime image inspection")
    executor_digest = _require_sha256(
        value["executor_manifest_sha256"],
        "live runtime executor manifest",
    )
    if executor_digest != scores.executor_evidence_sha256:
        raise ForagerMatchedSealError("open live runtime executor manifest drifted")
    return _canonical_sha256(value)


def _validate_receipt_index(
    payload: Mapping[str, Any],
    open_protocol: protocol.ForagerMatchedProtocol,
    scores: evidence.MatchedScoreEvidence,
    *,
    expected_plan_sha256: str,
    expected_live_runtime_identity_sha256: str,
) -> None:
    """Replay the open execution receipt index against the score evidence.

    Checks, in order: the fail-closed header (schema, nonpromoting authority
    flags, ``open_tuning`` stage), the index's self-declared payload digest, the
    campaign closure (protocol/plan/source/executor/live-runtime digests, seed
    block, candidate order), and finally every per-candidate receipt preimage —
    each must hash to the receipt digest recorded in the score evidence and equal
    the expected field set exactly, including per-seed artifact digests.  Any
    drift raises :class:`ForagerMatchedSealError`.
    """
    value = dict(payload)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "classification",
            "authentication_state",
            "promotion_authorized",
            "external_verification_required",
            "stage",
            "protocol_sha256",
            "plan_sha256",
            "source_manifest_sha256",
            "executor_manifest_sha256",
            "live_runtime_identity_sha256",
            "active_seeds",
            "horizon",
            "candidate_order",
            "execution_receipts",
            "payload_sha256",
        },
        "open execution receipt index",
    )
    if (
        value["schema_version"] != executor.MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION
        or value["classification"] != "content_complete_execution_receipt_preimages"
        or value["authentication_state"]
        != "content_only_unendorsed_external_verifier_required"
        or value["promotion_authorized"] is not False
        or value["external_verification_required"] is not True
        or value["stage"] != "open_tuning"
    ):
        raise ForagerMatchedSealError("open execution receipt index schema/state drifted")
    declared = _require_sha256(value["payload_sha256"], "receipt index payload")
    plan_sha256 = _require_sha256(value["plan_sha256"], "receipt index plan")
    live_runtime_sha256 = _require_sha256(
        value["live_runtime_identity_sha256"],
        "receipt index live runtime identity",
    )
    unsigned = dict(value)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared:
        raise ForagerMatchedSealError("open execution receipt index payload digest differs")
    score_ids = tuple(item.candidate_id for item in scores.candidate_scores)
    candidate_order = tuple(
        _require_identifier(item, "receipt index candidate")
        for item in _require_array(value["candidate_order"], "receipt index candidate order")
    )
    active_seeds = tuple(
        _require_int(item, "receipt index seed")
        for item in _require_array(value["active_seeds"], "receipt index active seeds")
    )
    expected_header = {
        "protocol_sha256": open_protocol.protocol_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "plan_sha256": expected_plan_sha256,
        "live_runtime_identity_sha256": expected_live_runtime_identity_sha256,
        "horizon": open_protocol.horizon,
    }
    drifted = [name for name, expected in expected_header.items() if value[name] != expected]
    if (
        drifted
        or active_seeds != open_protocol.active_seeds
        or candidate_order != score_ids
    ):
        raise ForagerMatchedSealError("open execution receipt index closure drifted")
    receipts = _require_array(value["execution_receipts"], "execution receipts")
    if len(receipts) != len(scores.candidate_scores):
        raise ForagerMatchedSealError("execution receipts do not cover the score panel")
    for index, (raw_receipt, score) in enumerate(
        zip(receipts, scores.candidate_scores, strict=True)
    ):
        item = _require_object(raw_receipt, f"execution_receipts[{index}]")
        _require_exact_keys(
            item,
            {"candidate_id", "execution_receipt_sha256", "receipt_payload"},
            f"execution_receipts[{index}]",
        )
        receipt_payload = _require_object(
            item["receipt_payload"], f"execution_receipts[{index}].receipt_payload"
        )
        receipt_sha = _require_sha256(
            item["execution_receipt_sha256"],
            f"execution_receipts[{index}].execution_receipt_sha256",
        )
        _require_exact_keys(
            receipt_payload,
            {
                "schema_version",
                "candidate_id",
                "stage",
                "protocol_sha256",
                "plan_sha256",
                "source_manifest_sha256",
                "executor_manifest_sha256",
                "capability_descriptor_sha256",
                "capability_qualification_receipt_sha256",
                "live_runtime_identity_sha256",
                "seed_artifacts",
                "authentication_state",
            },
            f"execution_receipts[{index}].receipt_payload",
        )
        expected_seed_artifacts = [
            {
                "seed": record.seed,
                "raw_artifact_sha256": record.raw_artifact_sha256,
                "reward_trace_sha256": record.reward_trace_sha256,
                "scoring_record_sha256": record.scoring_record_sha256,
            }
            for record in score.records
        ]
        expected_receipt_fields = {
            "schema_version": executor.MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "candidate_id": score.candidate_id,
            "stage": "open_tuning",
            "protocol_sha256": open_protocol.protocol_sha256,
            "plan_sha256": plan_sha256,
            "source_manifest_sha256": scores.source_evidence_sha256,
            "executor_manifest_sha256": scores.executor_evidence_sha256,
            "capability_descriptor_sha256": score.capability_descriptor_sha256,
            "capability_qualification_receipt_sha256": (
                score.capability_qualification_receipt_sha256
            ),
            "live_runtime_identity_sha256": live_runtime_sha256,
            "seed_artifacts": expected_seed_artifacts,
            "authentication_state": "content_complete_external_verifier_required",
        }
        if (
            item["candidate_id"] != score.candidate_id
            or receipt_sha != score.execution_receipt_sha256
            or _canonical_sha256(receipt_payload) != receipt_sha
            or receipt_payload != expected_receipt_fields
        ):
            raise ForagerMatchedSealError("execution receipt preimage differs from score evidence")


def _validate_completion_summary(
    summary: Mapping[str, Any],
    receipt_index: Mapping[str, Any],
    scores: evidence.MatchedScoreEvidence,
    request: executor.VerificationRequest,
) -> None:
    required = {
        "schema_version",
        "classification",
        "status",
        "stage",
        "protocol_sha256",
        "qualification_manifest_sha256",
        "execution_plan_sha256",
        "source_manifest_sha256",
        "executor_manifest_sha256",
        "live_runtime_identity_sha256",
        "candidate_count",
        "seed_count",
        "completed_cell_count",
        "execution_receipt_index_payload_sha256",
        "score_evidence_sha256",
        "verification_subject_sha256",
        "verification_authentication_state",
        "selection_created",
        "sealed_protocol_created",
        "evaluation_artifacts_created",
        "promotion_authorized",
        "performance_claim",
        "external_verification_required",
        "host_reward_array_access",
    }
    _require_exact_keys(summary, required, "open completion summary")
    if (
        summary["schema_version"] != campaign.MATCHED_OPEN_COMPLETION_SCHEMA_VERSION
        or summary["classification"] != "content_only_unendorsed_nonpromoting"
        or summary["status"]
        != "complete_content_only_external_verification_unresolved"
        or summary["stage"] != "open_tuning"
        or summary["verification_authentication_state"]
        != "unresolved_external_verifier_required"
        or summary["selection_created"] is not False
        or summary["sealed_protocol_created"] is not False
        or summary["evaluation_artifacts_created"] is not False
        or summary["promotion_authorized"] is not False
        or summary["performance_claim"] is not False
        or summary["external_verification_required"] is not True
        or summary["host_reward_array_access"] != "forbidden_not_performed"
    ):
        raise ForagerMatchedSealError("open completion summary schema/state drifted")
    expected = {
        "protocol_sha256": scores.protocol_sha256,
        "qualification_manifest_sha256": scores.qualification_manifest_sha256,
        "execution_plan_sha256": receipt_index["plan_sha256"],
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "live_runtime_identity_sha256": receipt_index["live_runtime_identity_sha256"],
        "candidate_count": len(scores.candidate_scores),
        "seed_count": len(scores.active_seeds),
        "completed_cell_count": len(scores.candidate_scores) * len(scores.active_seeds),
        "execution_receipt_index_payload_sha256": receipt_index["payload_sha256"],
        "score_evidence_sha256": scores.payload_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
    }
    drifted = [name for name, value in expected.items() if summary[name] != value]
    if drifted:
        raise ForagerMatchedSealError(
            "open completion summary closure drifted: " + ", ".join(drifted)
        )


def _artifact_ref(raw: bytes, path: str, payload_sha256: str | None) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": payload_sha256,
    }


def _build_manifest(
    *,
    artifact_bytes: Mapping[str, bytes],
    open_protocol: protocol.ForagerMatchedProtocol,
    receipt_index: Mapping[str, Any],
    scores: evidence.MatchedScoreEvidence,
    request: executor.VerificationRequest,
    completion_summary: Mapping[str, Any],
    bindings: evidence.AuthenticatedEvidenceBindings,
    selection: evidence.SelectionComputation,
    sealed_protocol: protocol.ForagerMatchedProtocol,
    transition: Mapping[str, Any],
    transition_sha256: str,
) -> dict[str, Any]:
    """Assemble the self-digested seal manifest.

    Fields: frozen schema/classification/stage plus the authority-boundary flags
    (persisted bindings are cache-only, resolver revalidation required,
    self-authentication forbidden); ``artifacts`` maps each bundled role to its
    path, file digest, and canonical payload digest where one exists;
    ``open_campaign`` records the completed open-tuning closure (protocol, plan,
    source/executor manifests, live runtime, receipt index, score evidence,
    verification subject, candidate order, active seeds, cell count);
    ``selection`` binds the plan/result/report digests and the recorded resolver
    cache; ``sealed_transition`` carries the transition descriptor, its digest,
    and the sealed protocol digest.  The unsigned body is closed with
    ``payload_sha256``, which :func:`_validate_manifest_shape` re-derives on load.
    """
    payload_digests: dict[str, str | None] = {
        "open_protocol": None,
        "open_execution_plan": cast(
            str,
            completion_summary["execution_plan_sha256"],
        ),
        "open_live_runtime": cast(
            str,
            completion_summary["live_runtime_identity_sha256"],
        ),
        "open_execution_receipt_index": cast(str, receipt_index["payload_sha256"]),
        "open_score_evidence": scores.payload_sha256,
        "open_verification_request": None,
        "open_completion_summary": None,
        "open_authenticated_bindings_cache": None,
        "selection_result": None,
        "selection_report": selection.report_sha256,
        "sealed_protocol": None,
    }
    artifacts = {
        role: _artifact_ref(
            artifact_bytes[role],
            path,
            payload_digests[role],
        )
        for role, path in _ARTIFACT_PATHS.items()
    }
    transition_body = dict(transition)
    unsigned: dict[str, Any] = {
        "schema_version": MATCHED_SEAL_BUNDLE_SCHEMA_VERSION,
        "classification": "content_only_external_trust_cache_nonpromoting",
        "stage": "sealed_evaluation",
        "authority_boundary": {
            "persisted_bindings_are_cache_only": True,
            "external_resolver_revalidation_required": True,
            "self_authentication_forbidden": True,
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "evaluation_executed": False,
        "artifacts": artifacts,
        "open_campaign": {
            "protocol_sha256": open_protocol.protocol_sha256,
            "qualification_manifest_sha256": scores.qualification_manifest_sha256,
            "execution_plan_sha256": completion_summary["execution_plan_sha256"],
            "source_manifest_sha256": scores.source_evidence_sha256,
            "executor_manifest_sha256": scores.executor_evidence_sha256,
            "live_runtime_identity_sha256": receipt_index[
                "live_runtime_identity_sha256"
            ],
            "execution_receipt_index_payload_sha256": receipt_index["payload_sha256"],
            "score_evidence_payload_sha256": scores.payload_sha256,
            "verification_subject_sha256": request.verification_subject_sha256,
            "candidate_order": [item.candidate_id for item in scores.candidate_scores],
            "active_seeds": list(scores.active_seeds),
            "completed_cell_count": completion_summary["completed_cell_count"],
        },
        "selection": {
            "selection_plan_sha256": open_protocol.selection_plan.plan_sha256,
            "selection_result_sha256": selection.selection_result.selection_result_sha256,
            "selection_report_payload_sha256": selection.report_sha256,
            "recorded_bindings_cache_sha256": bindings.bindings_sha256,
            "external_verification_subject_sha256": (
                bindings.verification_subject_sha256
            ),
            "external_verification_receipt_sha256": (
                bindings.verification_receipt_sha256
            ),
        },
        "sealed_transition": {
            "descriptor": transition_body,
            "descriptor_sha256": transition_sha256,
            "sealed_protocol_sha256": sealed_protocol.protocol_sha256,
        },
    }
    return {**unsigned, "payload_sha256": _canonical_sha256(unsigned)}


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "classification",
            "stage",
            "authority_boundary",
            "promotion_authorized",
            "performance_claim",
            "evaluation_executed",
            "artifacts",
            "open_campaign",
            "selection",
            "sealed_transition",
            "payload_sha256",
        },
        "seal manifest",
    )
    if (
        manifest["schema_version"] != MATCHED_SEAL_BUNDLE_SCHEMA_VERSION
        or manifest["classification"]
        != "content_only_external_trust_cache_nonpromoting"
        or manifest["stage"] != "sealed_evaluation"
        or manifest["promotion_authorized"] is not False
        or manifest["performance_claim"] is not False
        or manifest["evaluation_executed"] is not False
    ):
        raise ForagerMatchedSealError("seal manifest schema/authority state drifted")
    authority = _require_object(manifest["authority_boundary"], "authority boundary")
    if authority != {
        "persisted_bindings_are_cache_only": True,
        "external_resolver_revalidation_required": True,
        "self_authentication_forbidden": True,
    }:
        raise ForagerMatchedSealError("seal manifest authority boundary drifted")
    declared = _require_sha256(manifest["payload_sha256"], "seal manifest payload")
    unsigned = dict(manifest)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared:
        raise ForagerMatchedSealError("seal manifest payload digest differs")


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in cast(dict[str, Any], value).items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_selection_replay_budget(
    open_protocol: protocol.ForagerMatchedProtocol,
) -> None:
    """Bound replay work before the loader recomputes the frozen selection.

    Loading a seal bundle replays the open-tuning ranking (a bootstrap over
    candidates x seeds x resamples), so an oversized artifact could otherwise buy
    unbounded compute from every verifier.  Each ``_MAX_MATCHED_*`` cap equals the
    frozen matched-current campaign dimension (23 registered candidates, 10 tuning
    and 30 evaluation seeds, 2 selection groups of at most 14 candidates, 10_000
    bootstrap resamples); a protocol claiming more is rejected before any replay.
    """
    groups = open_protocol.selection_plan.groups
    if (
        len(open_protocol.candidates) > _MAX_MATCHED_CANDIDATES
        or len(open_protocol.tuning_seeds) > _MAX_MATCHED_TUNING_SEEDS
        or len(open_protocol.evaluation_seeds) > _MAX_MATCHED_EVALUATION_SEEDS
        or len(groups) > _MAX_MATCHED_SELECTION_GROUPS
        or any(
            len(group.candidate_ids) > _MAX_MATCHED_SELECTION_GROUP_SIZE
            for group in groups
        )
        or open_protocol.selection_plan.bootstrap_resamples
        > _MAX_MATCHED_SELECTION_BOOTSTRAP_RESAMPLES
    ):
        raise ForagerMatchedSealError(
            "seal selection replay exceeds the frozen matched-current resource envelope"
        )


def _load_forager_matched_seal_bundle_from_open_root(
    root: _OpenDirectory,
    initial_inventory: Mapping[str, tuple[int, ...]],
) -> ContentVerifiedSealBundle:
    manifest_raw, _manifest_file_sha = _load_pair_at(
        root, _MANIFEST_NAME, "seal manifest"
    )
    manifest = _decode_canonical(manifest_raw, "seal manifest")
    _validate_manifest_shape(manifest)
    artifact_refs = _require_object(manifest["artifacts"], "seal artifact references")
    if set(artifact_refs) != set(_ARTIFACT_PATHS):
        raise ForagerMatchedSealError("seal artifact roles differ")
    artifact_bytes: dict[str, bytes] = {}
    for role, expected_path in _ARTIFACT_PATHS.items():
        reference = _require_object(artifact_refs[role], f"artifact reference {role}")
        _require_exact_keys(
            reference,
            {"path", "file_sha256", "payload_sha256"},
            f"artifact reference {role}",
        )
        if reference["path"] != expected_path:
            raise ForagerMatchedSealError(f"artifact role {role} uses the wrong path")
        raw, digest = _load_pair_at(root, expected_path, role.replace("_", " "))
        if digest != _require_sha256(reference["file_sha256"], f"{role} file digest"):
            raise ForagerMatchedSealError(f"artifact role {role} file digest differs")
        artifact_bytes[role] = raw

    try:
        open_protocol = protocol.parse_forager_matched_protocol(
            artifact_bytes["open_protocol"]
        )
        scores = evidence.parse_matched_score_evidence(
            artifact_bytes["open_score_evidence"]
        )
        selection_result = protocol.parse_forager_matched_selection_result(
            artifact_bytes["selection_result"]
        )
        sealed_protocol = protocol.parse_forager_matched_protocol(
            artifact_bytes["sealed_protocol"]
        )
    except (protocol.ForagerMatchedProtocolError, evidence.ForagerMatchedEvidenceError) as exc:
        raise ForagerMatchedSealError(f"seal scientific artifact is invalid: {exc}") from exc
    if open_protocol.stage != "open_tuning" or sealed_protocol.stage != "sealed_evaluation":
        raise ForagerMatchedSealError("seal protocol stages are invalid")
    _validate_selection_replay_budget(open_protocol)
    canonical_scientific_bytes = {
        "open_protocol": open_protocol.canonical_bytes,
        "open_score_evidence": scores.canonical_bytes,
        "selection_result": selection_result.canonical_bytes,
        "sealed_protocol": sealed_protocol.canonical_bytes,
    }
    drifted_canonical = [
        role
        for role, expected in canonical_scientific_bytes.items()
        if artifact_bytes[role] != expected
    ]
    if drifted_canonical:
        raise ForagerMatchedSealError(
            "seal scientific artifacts are not canonical: "
            + ", ".join(drifted_canonical)
        )

    request_payload = _decode_canonical(
        artifact_bytes["open_verification_request"], "open verification request"
    )
    try:
        request = executor.parse_verification_request(
            artifact_bytes["open_verification_request"]
        )
    except executor.ForagerMatchedExecutorError as exc:
        raise ForagerMatchedSealError(f"open verification request is invalid: {exc}") from exc
    if request.stage != "open_tuning":
        raise ForagerMatchedSealError("seal requires an open-tuning verification request")
    bindings_payload = _decode_canonical(
        artifact_bytes["open_authenticated_bindings_cache"],
        "recorded authenticated-bindings cache",
    )
    bindings = _parse_recorded_bindings(bindings_payload)
    _validate_request_content(request, open_protocol, scores)
    if request.to_dict() != {
        key: _plain(value) for key, value in request_payload.items()
    }:
        raise ForagerMatchedSealError("open verification request normalization drifted")
    if bindings.to_dict() != {
        key: _plain(value) for key, value in bindings_payload.items()
    }:
        raise ForagerMatchedSealError("recorded bindings cache normalization drifted")
    try:
        replayed_selection = evidence.compute_open_selection(
            open_protocol,
            scores,
            authenticated_bindings=bindings,
        )
    except evidence.ForagerMatchedEvidenceError as exc:
        raise ForagerMatchedSealError(f"selection content replay failed: {exc}") from exc
    if replayed_selection.selection_result.to_dict() != selection_result.to_dict():
        raise ForagerMatchedSealError("persisted selection result does not replay")
    try:
        selection_report = evidence.parse_matched_selection_report(
            artifact_bytes["selection_report"],
            open_protocol=open_protocol,
            open_evidence=scores,
            authenticated_bindings=bindings,
            selection_result=selection_result,
            expected_payload_sha256=replayed_selection.report_sha256,
        )
        expected_sealed = protocol.seal_forager_matched_protocol(
            open_protocol, selection_result
        )
        validation = protocol.validate_sealed_protocol_transition(
            open_protocol,
            sealed_protocol,
            selection_result,
            selection_result.selection_result_sha256,
        )
    except (protocol.ForagerMatchedProtocolError, evidence.ForagerMatchedEvidenceError) as exc:
        raise ForagerMatchedSealError(f"sealed transition replay failed: {exc}") from exc
    if expected_sealed.to_dict() != sealed_protocol.to_dict():
        raise ForagerMatchedSealError("persisted sealed protocol does not replay")
    try:
        transition = evaluation.build_sealed_transition_descriptor(
            sealed_protocol,
            validation,
        )
        transition_sha256 = evaluation.canonical_sealed_transition_descriptor_sha256(
            sealed_protocol,
            validation,
        )
    except evaluation.ForagerMatchedEvaluationCampaignError as exc:
        raise ForagerMatchedSealError(
            f"sealed transition descriptor replay failed: {exc}"
        ) from exc

    plan_snapshot = _decode_canonical(
        artifact_bytes["open_execution_plan"],
        "open execution plan",
    )
    plan_sha256 = _validate_execution_plan_snapshot(
        plan_snapshot,
        open_protocol,
        scores,
    )
    live_runtime_snapshot = _decode_canonical(
        artifact_bytes["open_live_runtime"],
        "open live runtime",
    )
    live_runtime_sha256 = _validate_live_runtime_snapshot(
        live_runtime_snapshot,
        scores,
    )
    receipt_index = _decode_canonical(
        artifact_bytes["open_execution_receipt_index"],
        "open execution receipt index",
    )
    _validate_receipt_index(
        receipt_index,
        open_protocol,
        scores,
        expected_plan_sha256=plan_sha256,
        expected_live_runtime_identity_sha256=live_runtime_sha256,
    )
    completion_summary = _decode_canonical(
        artifact_bytes["open_completion_summary"], "open completion summary"
    )
    _validate_completion_summary(completion_summary, receipt_index, scores, request)

    expected_manifest = _build_manifest(
        artifact_bytes=artifact_bytes,
        open_protocol=open_protocol,
        receipt_index=receipt_index,
        scores=scores,
        request=request,
        completion_summary=completion_summary,
        bindings=bindings,
        selection=replayed_selection,
        sealed_protocol=sealed_protocol,
        transition=transition,
        transition_sha256=transition_sha256,
    )
    if expected_manifest != manifest:
        raise ForagerMatchedSealError("seal manifest differs from replayed content")
    final_inventory = _root_inventory(root)
    if final_inventory != dict(initial_inventory):
        raise ForagerMatchedSealError("seal bundle inventory changed during replay")
    _assert_open_directory_path(root, "seal bundle root")
    return ContentVerifiedSealBundle(
        output_root=root.path,
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        open_protocol=open_protocol,
        open_score_evidence=scores,
        open_verification_request=request,
        recorded_bindings_cache=cast(
            Mapping[str, Any],
            _freeze_json(bindings_payload),
        ),
        selection_result=selection_result,
        selection_report=selection_report,
        sealed_protocol=sealed_protocol,
        sealed_transition=cast(Mapping[str, Any], _freeze_json(transition)),
        sealed_transition_sha256=transition_sha256,
    )


def load_forager_matched_seal_bundle_content(root: Path) -> ContentVerifiedSealBundle:
    """Load and replay a seal while holding its root directory inode open."""
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    opened = _open_stable_directory(root, "seal bundle root")
    try:
        initial_inventory = _root_inventory(opened)
        return _load_forager_matched_seal_bundle_from_open_root(
            opened,
            initial_inventory,
        )
    finally:
        os.close(opened.descriptor)


def _validate_expected_authority(
    request: executor.VerificationRequest,
    *,
    expected_trust_anchor_identity: str,
    expected_verification_subject_sha256: str | None,
) -> None:
    expected_anchor = _require_identifier(
        expected_trust_anchor_identity,
        "expected trust anchor identity",
    )
    if request.trust_anchor_identity != expected_anchor:
        raise ForagerMatchedSealError(
            "seal trust anchor differs from the caller-pinned identity"
        )
    if expected_verification_subject_sha256 is not None:
        expected_subject = _require_sha256(
            expected_verification_subject_sha256,
            "expected verification subject",
        )
        if request.verification_subject_sha256 != expected_subject:
            raise ForagerMatchedSealError(
                "seal verification subject differs from the caller-pinned digest"
            )


def _validate_expected_manifest(
    manifest: Mapping[str, Any],
    expected_seal_manifest_sha256: str | None,
) -> None:
    if expected_seal_manifest_sha256 is None:
        return
    expected = _require_sha256(
        expected_seal_manifest_sha256,
        "expected seal manifest",
    )
    if manifest.get("payload_sha256") != expected:
        raise ForagerMatchedSealError(
            "seal manifest differs from the caller-pinned digest"
        )


def authenticate_forager_matched_seal_bundle(
    value: ContentVerifiedSealBundle | Path,
    *,
    resolver: executor.TrustResolver,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_sha256: str | None = None,
    expected_verification_subject_sha256: str | None = None,
) -> evidence.AuthenticatedEvidenceBindings:
    """Freshly resolve the caller-pinned subject and return the resolver result.

    The returned dataclass is convenient validated data, not an authority
    capability: Python callers can construct the same type directly.  Every
    authority-bearing consumer must call this function itself immediately
    before acting and must supply its own out-of-band pins.
    """
    if isinstance(value, Path):
        content = load_forager_matched_seal_bundle_content(value)
    elif type(value) is ContentVerifiedSealBundle:
        content = load_forager_matched_seal_bundle_content(value.output_root)
        if content.manifest != value.manifest:
            raise ForagerMatchedSealError(
                "content-verified seal object differs from its persisted root"
            )
    else:
        raise TypeError("value must be a ContentVerifiedSealBundle or Path")
    _validate_expected_authority(
        content.open_verification_request,
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_verification_subject_sha256=(
            expected_verification_subject_sha256
        ),
    )
    _validate_expected_manifest(
        content.manifest,
        expected_seal_manifest_sha256,
    )
    try:
        resolved = executor.resolve_authenticated_bindings(
            content.open_verification_request,
            resolver,
        )
    except Exception as exc:
        raise ForagerMatchedSealError(f"external trust resolution failed: {exc}") from exc
    if resolved.to_dict() != _plain(content.recorded_bindings_cache):
        raise ForagerMatchedSealError(
            "external resolver result differs from the recorded bindings cache"
        )
    return resolved


def _write_exclusive_at(root: _OpenDirectory, name: str, raw: bytes) -> None:
    if not raw:
        raise ForagerMatchedSealError("empty seal artifacts are forbidden")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=root.descriptor)
    except OSError as exc:
        raise ForagerMatchedSealError(f"cannot stage seal artifact {name!r}") from exc
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise ForagerMatchedSealError(f"cannot completely stage artifact {name!r}")
            remaining = remaining[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedSealError(f"staged artifact {name!r} changed while writing")
    except OSError as exc:
        raise ForagerMatchedSealError(f"cannot stage seal artifact {name!r}") from exc
    finally:
        os.close(descriptor)


def _write_pair_at(root: _OpenDirectory, name: str, raw: bytes) -> None:
    digest = hashlib.sha256(raw).hexdigest()
    _write_exclusive_at(root, name, raw)
    _write_exclusive_at(root, f"{name}.sha256", f"{digest}\n".encode("ascii"))


def _durably_sync_open_tree(root: _OpenDirectory) -> None:
    """Fsync every verified regular file and then the staging directory."""
    _assert_open_directory_path(root, "seal staging directory")
    initial_inventory = _root_inventory(root)
    try:
        for name in sorted(_expected_root_names(), key=lambda item: item.encode("utf-8")):
            expected = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
                raise ForagerMatchedSealError("seal staging tree contains an unsafe file")
            try:
                descriptor = os.open(name, _file_open_flags(), dir_fd=root.descriptor)
            except OSError as exc:
                raise ForagerMatchedSealError("cannot safely open staged seal file") from exc
            try:
                opened = os.fstat(descriptor)
                if _stat_identity(expected) != _stat_identity(opened):
                    raise ForagerMatchedSealError("staged seal file changed before fsync")
                os.fsync(descriptor)
                current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
                if (
                    _stat_identity(opened) != _stat_identity(os.fstat(descriptor))
                    or _stat_identity(opened) != _stat_identity(current)
                ):
                    raise ForagerMatchedSealError("staged seal file changed during fsync")
            except OSError as exc:
                raise ForagerMatchedSealError("cannot durably sync staged seal file") from exc
            finally:
                os.close(descriptor)
        os.fsync(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedSealError("cannot durably sync seal staging directory") from exc
    if _root_inventory(root) != initial_inventory:
        raise ForagerMatchedSealError("seal staging inventory changed during fsync")
    _assert_open_directory_path(root, "seal staging directory")


def _parent_entry_matches_open_directory(
    parent: _OpenDirectory,
    name: str,
    opened: _OpenDirectory,
) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.fstat(opened.descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry.st_mode)
        and _inode_identity(entry) == opened.inode_identity
        and _inode_identity(descriptor) == opened.inode_identity
    )


def _sync_publication_parent(parent: _OpenDirectory) -> None:
    os.fsync(parent.descriptor)


def _publish_verified_no_replace(
    parent: _OpenDirectory,
    staging: _OpenDirectory,
    source_name: str,
    destination_name: str,
    destination: Path,
) -> None:
    _assert_open_directory_path(parent, "seal output parent")
    if not _parent_entry_matches_open_directory(parent, source_name, staging):
        raise ForagerMatchedSealError(
            "seal staging name no longer refers to the verified staging inode"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ForagerMatchedSealError("renameat2 is required for seal publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent.descriptor,
        os.fsencode(source_name),
        parent.descriptor,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ForagerMatchedSealError("seal output was created concurrently")
        raise ForagerMatchedSealError(
            f"exclusive seal publication failed with errno {error}"
        )

    try:
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedSealError(
                "published seal does not refer to the verified staging inode"
            )
        try:
            os.stat(source_name, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForagerMatchedSealError("staging name survived exclusive publication")
        _sync_publication_parent(parent)
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedSealError("published seal changed during parent fsync")
        _assert_open_directory_path(parent, "seal output parent")
    except BaseException as exc:
        raise PublishedSealUncertainError(
            destination,
            "publication durability or inode verification failed",
        ) from exc


def _cleanup_owned_staging(
    parent: _OpenDirectory,
    name: str,
    staging: _OpenDirectory,
) -> None:
    """Best-effort cleanup restricted to the exact staging inode we created."""
    if not _parent_entry_matches_open_directory(parent, name, staging):
        return
    try:
        with os.scandir(staging.descriptor) as iterator:
            entries = [(entry.name, entry.stat(follow_symlinks=False)) for entry in iterator]
        if any(
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            for _, metadata in entries
        ):
            return
        for entry_name, metadata in entries:
            current = os.stat(
                entry_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(current) != _stat_identity(metadata):
                return
        for entry_name, _ in entries:
            os.unlink(entry_name, dir_fd=staging.descriptor)
        if not _parent_entry_matches_open_directory(parent, name, staging):
            return
        os.rmdir(name, dir_fd=parent.descriptor)
    except OSError:
        return


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _prospective_output(
    qualification_root: Path,
    campaign_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    if not output_root.name or output_root.name in {".", ".."}:
        raise ForagerMatchedSealError("seal output name is unsafe")
    qualified = _regular_directory(qualification_root, "qualification root")
    completed = _regular_directory(campaign_root, "open campaign root")
    try:
        prospective = output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedSealError("seal output path cannot be resolved") from exc
    if _paths_overlap(prospective, qualified) or _paths_overlap(prospective, completed):
        raise ForagerMatchedSealError("seal output overlaps an input root")
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedSealError("seal output root already exists")
    return qualified, completed, prospective


def _open_destination_parent(
    requested: Path,
    prospective: Path,
) -> tuple[_OpenDirectory, Path]:
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = _open_stable_directory(requested.parent, "seal output parent")
    try:
        destination = parent.path / requested.name
        if destination != prospective:
            raise ForagerMatchedSealError(
                "seal output parent was redirected before publication"
            )
        try:
            os.stat(requested.name, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ForagerMatchedSealError("cannot inspect seal output destination") from exc
        else:
            raise ForagerMatchedSealError("seal output root already exists")
        return parent, destination
    except BaseException:
        os.close(parent.descriptor)
        raise


def _create_owned_staging(
    parent: _OpenDirectory,
    destination: Path,
) -> tuple[str, _OpenDirectory]:
    for _ in range(64):
        name = f".seal-partial-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ForagerMatchedSealError("cannot create seal staging directory") from exc
        staging_path = parent.path / name
        try:
            staging = _open_stable_directory_at(
                parent,
                name,
                staging_path,
                "seal staging directory",
            )
        except BaseException:
            # Do not remove a name that could have been substituted after mkdir.
            raise
        if destination.parent != parent.path:
            os.close(staging.descriptor)
            raise ForagerMatchedSealError("seal destination escaped its opened parent")
        return name, staging
    raise ForagerMatchedSealError("cannot allocate a unique seal staging directory")


def _completed_artifact_bytes(completed: Any) -> dict[str, bytes]:
    return {
        "open_protocol": completed.protocol.canonical_bytes,
        "open_execution_plan": completed.plan.canonical_bytes,
        "open_live_runtime": executor.canonical_json_bytes(
            completed.live_runtime.unsigned_dict
        ),
        "open_execution_receipt_index": completed.execution_receipt_index.canonical_bytes,
        "open_score_evidence": completed.score_evidence.canonical_bytes,
        "open_verification_request": completed.verification_request.canonical_bytes,
        "open_completion_summary": canonical_json_bytes(completed.completion_summary),
    }


def _validate_completed_file_hashes(completed: Any, artifact_bytes: Mapping[str, bytes]) -> None:
    expected_by_role = {
        "open_execution_receipt_index": "execution-receipt-index.json",
        "open_score_evidence": "score-evidence.json",
        "open_verification_request": "verification-request.json",
        "open_completion_summary": "completion-summary.json",
    }
    persisted = completed.final_file_sha256
    for role, name in expected_by_role.items():
        if persisted.get(name) != hashlib.sha256(artifact_bytes[role]).hexdigest():
            raise ForagerMatchedSealError(
                f"completed campaign file digest differs for {name}"
            )


def _verify_staged_bundle(root: _OpenDirectory) -> ContentVerifiedSealBundle:
    return _load_forager_matched_seal_bundle_from_open_root(
        root,
        _root_inventory(root),
    )


def create_forager_matched_seal_bundle(
    qualification_root: Path,
    campaign_root: Path,
    output_root: Path,
    *,
    resolver: executor.TrustResolver,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_sha256: str | None = None,
    expected_verification_subject_sha256: str | None = None,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> ContentVerifiedSealBundle:
    """Authenticate tuning and publish content; return no reusable authority proof."""
    if not all(
        isinstance(path, Path) for path in (qualification_root, campaign_root, output_root)
    ):
        raise TypeError("qualification_root, campaign_root, and output_root must be Paths")
    qualified, completed_root, prospective = _prospective_output(
        qualification_root,
        campaign_root,
        output_root,
    )
    try:
        completed = campaign.load_completed_open_tuning_campaign(
            qualified,
            completed_root,
            runtime=runtime,
            runner=runner,
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedSealError(f"completed open campaign is invalid: {exc}") from exc
    if completed.protocol.stage != "open_tuning":
        raise ForagerMatchedSealError("completed campaign is not open tuning")
    _validate_selection_replay_budget(completed.protocol)
    _validate_expected_authority(
        completed.verification_request,
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_verification_subject_sha256=(
            expected_verification_subject_sha256
        ),
    )
    artifact_bytes = _completed_artifact_bytes(completed)
    _validate_completed_file_hashes(completed, artifact_bytes)
    plan_payload = _decode_canonical(
        artifact_bytes["open_execution_plan"],
        "completed execution plan",
    )
    plan_sha256 = _validate_execution_plan_snapshot(
        plan_payload,
        completed.protocol,
        completed.score_evidence,
    )
    live_runtime_payload = _decode_canonical(
        artifact_bytes["open_live_runtime"],
        "completed live runtime",
    )
    live_runtime_sha256 = _validate_live_runtime_snapshot(
        live_runtime_payload,
        completed.score_evidence,
    )
    receipt_payload = _decode_canonical(
        artifact_bytes["open_execution_receipt_index"],
        "completed execution receipt index",
    )
    _validate_receipt_index(
        receipt_payload,
        completed.protocol,
        completed.score_evidence,
        expected_plan_sha256=plan_sha256,
        expected_live_runtime_identity_sha256=live_runtime_sha256,
    )
    try:
        authenticated = executor.resolve_authenticated_bindings(
            completed.verification_request,
            resolver,
        )
        selection = evidence.compute_open_selection(
            completed.protocol,
            completed.score_evidence,
            authenticated_bindings=authenticated,
        )
        sealed = protocol.seal_forager_matched_protocol(
            completed.protocol,
            selection.selection_result,
        )
        validation = protocol.validate_sealed_protocol_transition(
            completed.protocol,
            sealed,
            selection.selection_result,
            selection.selection_result.selection_result_sha256,
        )
        transition = evaluation.build_sealed_transition_descriptor(
            sealed,
            validation,
        )
        transition_sha256 = evaluation.canonical_sealed_transition_descriptor_sha256(
            sealed,
            validation,
        )
    except Exception as exc:
        raise ForagerMatchedSealError(
            f"seal computation or trust resolution failed: {exc}"
        ) from exc
    artifact_bytes.update(
        {
            "open_authenticated_bindings_cache": executor.canonical_json_bytes(
                authenticated.to_dict()
            ),
            "selection_result": selection.selection_result.canonical_bytes,
            "selection_report": selection.canonical_report_bytes,
            "sealed_protocol": sealed.canonical_bytes,
        }
    )
    manifest = _build_manifest(
        artifact_bytes=artifact_bytes,
        open_protocol=completed.protocol,
        receipt_index=receipt_payload,
        scores=completed.score_evidence,
        request=completed.verification_request,
        completion_summary=completed.completion_summary,
        bindings=authenticated,
        selection=selection,
        sealed_protocol=sealed,
        transition=transition,
        transition_sha256=transition_sha256,
    )
    _validate_expected_manifest(manifest, expected_seal_manifest_sha256)

    parent, destination = _open_destination_parent(output_root, prospective)
    staging_name: str | None = None
    staging: _OpenDirectory | None = None
    published = False
    try:
        staging_name, staging = _create_owned_staging(parent, destination)
        for role, name in _ARTIFACT_PATHS.items():
            _write_pair_at(staging, name, artifact_bytes[role])
        _write_pair_at(staging, _MANIFEST_NAME, canonical_json_bytes(manifest))
        staged = _verify_staged_bundle(staging)
        if staged.manifest != _freeze_json(manifest):
            raise ForagerMatchedSealError("staged seal manifest replay drifted")
        _durably_sync_open_tree(staging)
        _publish_verified_no_replace(
            parent,
            staging,
            staging_name,
            destination.name,
            destination,
        )
        published = True
        published_root = _OpenDirectory(
            path=destination,
            descriptor=staging.descriptor,
            inode_identity=staging.inode_identity,
        )
        content = _load_forager_matched_seal_bundle_from_open_root(
            published_root,
            _root_inventory(published_root),
        )
        if _plain(content.recorded_bindings_cache) != authenticated.to_dict():
            raise ForagerMatchedSealError("published seal bindings cache drifted")
        return content
    except PublishedSealUncertainError:
        raise
    except BaseException as exc:
        if staging is not None and _parent_entry_matches_open_directory(
            parent,
            destination.name,
            staging,
        ):
            published = True
        if published:
            raise PublishedSealUncertainError(
                destination,
                "final content verification failed",
            ) from exc
        if staging is not None and staging_name is not None:
            _cleanup_owned_staging(parent, staging_name, staging)
        raise
    finally:
        if staging is not None:
            os.close(staging.descriptor)
        os.close(parent.descriptor)


__all__ = [
    "ContentVerifiedSealBundle",
    "ForagerMatchedSealError",
    "MATCHED_SEAL_BUNDLE_SCHEMA_VERSION",
    "MATCHED_SEALED_TRANSITION_SCHEMA_VERSION",
    "PublishedSealUncertainError",
    "authenticate_forager_matched_seal_bundle",
    "canonical_json_bytes",
    "create_forager_matched_seal_bundle",
    "load_forager_matched_seal_bundle_content",
]
