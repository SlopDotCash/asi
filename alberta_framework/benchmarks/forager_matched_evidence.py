"""Strict score-evidence bridge for matched-current Forager comparisons.

The protocol module freezes scientific intent and the statistics module accepts
already validated score vectors.  This module is the deliberately narrow bridge
between them.  It:

* parses one canonical, self-hashed score-evidence bundle;
* checks every candidate, seed, metric, task, RNG, runtime, and capability
  subject against a frozen protocol;
* deterministically ranks open-tuning candidates without embedding raw scores
  in the selection report; and
* constructs the exact statistics-v3 contract for a validated open-to-sealed
  protocol transition.

Digests in this module are content identities, not signatures or trust grants.
The source and executor evidence manifests must be authenticated by the
external verifier named by the matched protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, cast

import numpy as np

from alberta_framework.benchmarks.forager_matched_protocol import (
    FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
    ForagerMatchedProtocol,
    ForagerMatchedProtocolError,
    ForagerMatchedSelectionResult,
    RankedSelectionGroup,
    SealedProtocolValidation,
    candidate_capability_descriptor_sha256,
    canonical_selection_result_bytes,
    parse_forager_matched_protocol,
    parse_forager_matched_selection_result,
    validate_sealed_protocol_transition,
)
from alberta_framework.benchmarks.forager_matched_statistics import (
    PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
    BootstrapSpec,
    ComparisonSpec,
    DescriptiveDiagnosticScores,
    EvidenceBinding,
    LearningMethodScores,
    MatchedComparisonContract,
    MatchedStatisticsError,
    PermutationSpec,
)

MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION: Final = "alberta.forager_matched_score_evidence.v2"
MATCHED_SELECTION_REPORT_SCHEMA_VERSION: Final = "alberta.forager_matched_selection_report.v1"
MATCHED_SELECTION_GROUP_EVIDENCE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_selection_group_evidence.v1"
)
MATCHED_METRIC_BINDING_SCHEMA_VERSION: Final = "alberta.forager_matched_metric_binding.v1"
MATCHED_EXECUTION_CLOSURE_SCHEMA_VERSION: Final = "alberta.forager_matched_execution_closure.v2"
MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_verification_subject.v2"
)
AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION: Final = (
    "alberta.forager_authenticated_evidence_bindings.v2"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_JSON_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_NODES: Final = 1_000_000
_MAX_JSON_DEPTH: Final = 64
_MAX_CANDIDATES: Final = 256
_MAX_SEEDS: Final = 4_096
_MAX_SEED: Final = 2**31 - 1
_BOOTSTRAP_CHUNK_ELEMENTS: Final = 2_000_000
_SELECTION_COMPUTATION_FACTORY_TOKEN: Final = object()

Stage = Literal["open_tuning", "sealed_evaluation"]


class ForagerMatchedEvidenceError(ValueError):
    """A score bundle or protocol-to-statistics adaptation failed closed."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedEvidenceError("value is not canonical-JSON encodable") from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_DESCRIPTOR: Final = {
    "schema_version": ("alberta.forager_matched_selection_bootstrap_rng_implementation.v1"),
    "generator": "numpy.random.Generator",
    "bit_generator": "numpy.random.PCG64",
    "seed_domain": "nonnegative_int31",
    "candidate_stream_policy": "reset_identical_seed_per_candidate",
    "draw": "integers_low_0_high_n_exclusive",
    "draw_dtype": "numpy.int64",
    "draw_order": "row_major_candidate_seed_block",
    "chunking": "stream_preserving",
    "numpy_implementation": "runtime_profile_bound",
}
MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256: Final = _canonical_sha256(
    _SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_DESCRIPTOR
)
_SELECTION_STATISTIC_IMPLEMENTATION_DESCRIPTOR: Final = {
    "schema_version": ("alberta.forager_matched_selection_statistic_implementation.v1"),
    "input": "ordered_complete_candidate_seed_score_vector",
    "score_dtype": "numpy.float64",
    "point_estimate": "numpy_mean_float64",
    "bootstrap_rng_implementation_sha256": (MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256),
    "bootstrap_statistic": "numpy_mean_float64",
    "bootstrap_interval": "two_sided_equal_tail_percentile",
    "conservative_endpoint": "lower",
    "endpoint_quantile": "(1-confidence)/2",
    "quantile_method": "linear",
    "singleton_interval": "point_mass_at_mean",
    "ranking_direction": "maximize",
    "tie_break": "candidate_id_ascending",
    "numpy_implementation": "runtime_profile_bound",
}
MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256: Final = _canonical_sha256(
    _SELECTION_STATISTIC_IMPLEMENTATION_DESCRIPTOR
)


def matched_selection_bootstrap_rng_implementation_descriptor() -> dict[str, Any]:
    """Return the canonical semantics behind the frozen RNG digest.

    The digest is embedded in every open protocol's selection plan.
    :func:`compute_open_selection` verifies it twice per replay: the descriptor is
    re-hashed against the module constant (drift fails closed), and the plan's
    recorded digest must name exactly this implementation.
    """
    return dict(_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_DESCRIPTOR)


def matched_selection_statistic_implementation_descriptor() -> dict[str, Any]:
    """Return the canonical semantics behind the frozen statistic digest.

    Verified exactly like the RNG descriptor: re-hashed on every selection replay
    and matched against the digest frozen into the protocol's selection plan.
    """
    return dict(_SELECTION_STATISTIC_IMPLEMENTATION_DESCRIPTOR)


def _verify_selection_implementation_descriptors() -> None:
    if (
        _canonical_sha256(_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_DESCRIPTOR)
        != MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
        or _canonical_sha256(_SELECTION_STATISTIC_IMPLEMENTATION_DESCRIPTOR)
        != MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
    ):
        raise ForagerMatchedEvidenceError(
            "selection implementation descriptors drifted from their frozen digests"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedEvidenceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedEvidenceError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json_float(value: str) -> float:
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatchedEvidenceError(f"invalid JSON number {value!r}") from exc
    if not math.isfinite(parsed):
        raise ForagerMatchedEvidenceError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _validate_json_complexity(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedEvidenceError("score evidence exceeds the JSON node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedEvidenceError("score evidence exceeds the JSON nesting bound")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def decode_strict_json(raw: bytes | str) -> Any:
    """Decode bounded duplicate-free JSON with finite numeric values."""
    if type(raw) is bytes:
        if len(raw) > _MAX_JSON_BYTES:
            raise ForagerMatchedEvidenceError("score evidence exceeds the byte bound")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ForagerMatchedEvidenceError("score evidence is not UTF-8") from exc
    elif type(raw) is str:
        try:
            encoded = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ForagerMatchedEvidenceError("score evidence contains invalid Unicode") from exc
        if len(encoded) > _MAX_JSON_BYTES:
            raise ForagerMatchedEvidenceError("score evidence exceeds the byte bound")
        text = raw
    else:
        raise TypeError("raw score evidence must be bytes or str")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_json_float,
        )
    except ForagerMatchedEvidenceError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ForagerMatchedEvidenceError("score evidence is not strict JSON") from exc
    _validate_json_complexity(value)
    return value


def _read_stable_regular_file(path: str | Path, *, label: str) -> bytes:
    source = Path(path)
    try:
        path_before = os.lstat(source)
    except OSError as exc:
        raise ForagerMatchedEvidenceError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise ForagerMatchedEvidenceError(
            f"{label} must be a single-link regular file and must not be a symlink"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ForagerMatchedEvidenceError(
            f"{label} is not an openable non-symlink regular file: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ForagerMatchedEvidenceError(
                f"{label} must be a single-link regular file"
            )
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise ForagerMatchedEvidenceError(f"{label} changed before it was opened")
        if before.st_size < 0 or before.st_size > _MAX_JSON_BYTES:
            raise ForagerMatchedEvidenceError(f"{label} exceeds the byte bound")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedEvidenceError(f"could not read {label}: {exc}") from exc
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise ForagerMatchedEvidenceError(f"{label} changed while it was being read")
    try:
        path_after = os.lstat(source)
    except OSError as exc:
        raise ForagerMatchedEvidenceError(f"{label} disappeared while being read") from exc
    if (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_nlink,
        path_after.st_uid,
        path_after.st_gid,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    ) != identity_after:
        raise ForagerMatchedEvidenceError(f"{label} path changed while being read")
    return raw


def _object(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedEvidenceError(f"{path} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedEvidenceError(f"{path} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedEvidenceError(
            f"{path} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, path: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ForagerMatchedEvidenceError(f"{path} must be a bounded non-empty string")
    return value


def _identifier(value: Any, path: str) -> str:
    result = _string(value, path, maximum=128)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ForagerMatchedEvidenceError(f"{path} must be a portable identifier")
    return result


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path, maximum=64)
    if _SHA256_RE.fullmatch(result) is None:
        raise ForagerMatchedEvidenceError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedEvidenceError(f"{path} must be an integer in [{minimum}, {maximum}]")
    return value


def _score_from_hex(value: Any, path: str) -> float:
    """Parse a score transported as ``float.hex()`` text.

    Hex transport round-trips IEEE-754 doubles exactly and, with the enforced
    ``score.hex() == text`` identity, gives each value one canonical spelling —
    score digests cannot be perturbed by decimal formatting.
    """
    text = _string(value, path, maximum=32)
    try:
        score = float.fromhex(text)
    except (OverflowError, ValueError) as exc:
        raise ForagerMatchedEvidenceError(f"{path} is not a hexadecimal float") from exc
    if not math.isfinite(score) or score.hex() != text:
        raise ForagerMatchedEvidenceError(f"{path} is not a canonical finite hexadecimal float")
    return score


def _finite_float(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ForagerMatchedEvidenceError(f"{path} must be a finite built-in float")
    return value


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in cast(dict[str, Any], value).items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ForagerMatchedEvidenceError(
                    "selection report mappings must contain only string keys"
                )
            result[key] = _thaw_json(item)
        return result
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SeedScoreRecord:
    """One scalar score and the raw/scorer identities from which it came."""

    seed: int
    score: float
    raw_artifact_sha256: str
    reward_trace_sha256: str
    scoring_record_sha256: str

    def __post_init__(self) -> None:
        _integer(self.seed, "seed record seed", minimum=0, maximum=_MAX_SEED)
        _finite_float(self.score, "seed record score")
        _sha256(
            self.raw_artifact_sha256,
            "seed record raw_artifact_sha256",
        )
        _sha256(
            self.reward_trace_sha256,
            "seed record reward_trace_sha256",
        )
        _sha256(
            self.scoring_record_sha256,
            "seed record scoring_record_sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "score_hex": self.score.hex(),
            "raw_artifact_sha256": self.raw_artifact_sha256,
            "reward_trace_sha256": self.reward_trace_sha256,
            "scoring_record_sha256": self.scoring_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateScoreEvidence:
    """Complete active-seed block for one protocol candidate."""

    candidate_id: str
    capability_descriptor_sha256: str
    capability_qualification_receipt_sha256: str
    execution_receipt_sha256: str
    records: tuple[SeedScoreRecord, ...]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate score candidate_id")
        _sha256(
            self.capability_descriptor_sha256,
            "candidate score capability_descriptor_sha256",
        )
        _sha256(
            self.capability_qualification_receipt_sha256,
            "candidate score capability_qualification_receipt_sha256",
        )
        _sha256(
            self.execution_receipt_sha256,
            "candidate score execution_receipt_sha256",
        )
        if type(self.records) is not tuple or not 1 <= len(self.records) <= _MAX_SEEDS:
            raise ForagerMatchedEvidenceError(
                f"candidate score records must be a tuple with 1..{_MAX_SEEDS} entries"
            )
        if any(type(record) is not SeedScoreRecord for record in self.records):
            raise ForagerMatchedEvidenceError("candidate score records contain an invalid object")
        if len(set(self.seeds)) != len(self.seeds):
            raise ForagerMatchedEvidenceError("candidate score records repeat a seed")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(record.seed for record in self.records)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(record.score for record in self.records)

    @property
    def score_vector_sha256(self) -> str:
        return _canonical_sha256(
            {
                "candidate_id": self.candidate_id,
                "schema_version": ("alberta.forager_matched_score_vector.v1"),
                "scores_hex": [record.score.hex() for record in self.records],
                "seeds": [record.seed for record in self.records],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "capability_descriptor_sha256": self.capability_descriptor_sha256,
            "capability_qualification_receipt_sha256": (
                self.capability_qualification_receipt_sha256
            ),
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class MatchedScoreEvidence:
    """Canonical score bundle shared by selection and sealed analysis."""

    schema_version: Literal["alberta.forager_matched_score_evidence.v2"]
    stage: Stage
    protocol_sha256: str
    active_seeds: tuple[int, ...]
    horizon: int
    metric: str
    metric_implementation_sha256: str
    task_identity_sha256: str
    environment_rng_schedule_sha256: str
    runtime_profile_sha256: str
    source_evidence_sha256: str
    executor_evidence_sha256: str
    qualification_manifest_sha256: str
    candidate_scores: tuple[CandidateScoreEvidence, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION:
            raise ForagerMatchedEvidenceError("score evidence schema_version is unsupported")
        if self.stage not in {"open_tuning", "sealed_evaluation"}:
            raise ForagerMatchedEvidenceError("score evidence stage is unsupported")
        _sha256(self.protocol_sha256, "score evidence protocol_sha256")
        if type(self.active_seeds) is not tuple or not 1 <= len(self.active_seeds) <= _MAX_SEEDS:
            raise ForagerMatchedEvidenceError(
                f"score evidence active_seeds must be a tuple with 1..{_MAX_SEEDS} entries"
            )
        checked_seeds = tuple(
            _integer(
                seed,
                f"score evidence active_seeds[{index}]",
                minimum=0,
                maximum=_MAX_SEED,
            )
            for index, seed in enumerate(self.active_seeds)
        )
        if len(set(checked_seeds)) != len(checked_seeds):
            raise ForagerMatchedEvidenceError("score evidence active_seeds repeat a seed")
        _integer(
            self.horizon,
            "score evidence horizon",
            minimum=1,
            maximum=2**31 - 1,
        )
        _identifier(self.metric, "score evidence metric")
        _sha256(
            self.metric_implementation_sha256,
            "score evidence metric_implementation_sha256",
        )
        _sha256(
            self.task_identity_sha256,
            "score evidence task_identity_sha256",
        )
        _sha256(
            self.environment_rng_schedule_sha256,
            "score evidence environment_rng_schedule_sha256",
        )
        _sha256(
            self.runtime_profile_sha256,
            "score evidence runtime_profile_sha256",
        )
        _sha256(
            self.source_evidence_sha256,
            "score evidence source_evidence_sha256",
        )
        _sha256(
            self.executor_evidence_sha256,
            "score evidence executor_evidence_sha256",
        )
        _sha256(
            self.qualification_manifest_sha256,
            "score evidence qualification_manifest_sha256",
        )
        if (
            type(self.candidate_scores) is not tuple
            or not 1 <= len(self.candidate_scores) <= _MAX_CANDIDATES
        ):
            raise ForagerMatchedEvidenceError(
                "score evidence candidate_scores must be a bounded tuple"
            )
        if any(
            type(candidate) is not CandidateScoreEvidence for candidate in self.candidate_scores
        ):
            raise ForagerMatchedEvidenceError(
                "score evidence candidate_scores contain an invalid object"
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidate_scores)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ForagerMatchedEvidenceError("score evidence candidate_scores repeat a candidate")
        declared = _sha256(
            self.payload_sha256,
            "score evidence payload_sha256",
        )
        if _canonical_sha256(self.unsigned_dict()) != declared:
            raise ForagerMatchedEvidenceError("score evidence payload_sha256 does not verify")

    @property
    def candidate_index(self) -> Mapping[str, CandidateScoreEvidence]:
        return {candidate.candidate_id: candidate for candidate in self.candidate_scores}

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "protocol_sha256": self.protocol_sha256,
            "active_seeds": list(self.active_seeds),
            "horizon": self.horizon,
            "metric": self.metric,
            "metric_implementation_sha256": self.metric_implementation_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "environment_rng_schedule_sha256": (self.environment_rng_schedule_sha256),
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "executor_evidence_sha256": self.executor_evidence_sha256,
            "qualification_manifest_sha256": self.qualification_manifest_sha256,
            "candidate_scores": [candidate.to_dict() for candidate in self.candidate_scores],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "payload_sha256": self.payload_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def _verification_subject_sha256(
    *,
    stage: Stage,
    protocol_sha256: str,
    score_evidence_sha256: str,
    source_manifest_sha256: str,
    executor_manifest_sha256: str,
    qualification_manifest_sha256: str,
    execution_closure_sha256: str,
    trust_anchor_identity: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
            "stage": stage,
            "protocol_sha256": protocol_sha256,
            "score_evidence_sha256": score_evidence_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "executor_manifest_sha256": executor_manifest_sha256,
            "qualification_manifest_sha256": qualification_manifest_sha256,
            "execution_closure_sha256": execution_closure_sha256,
            "trust_anchor_identity": trust_anchor_identity,
        }
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedEvidenceBindings:
    """Out-of-band verifier resolution for one exact score bundle.

    Constructing this value does not itself authenticate anything.  The caller
    must obtain every field from the trust resolver identified by
    ``trust_anchor_identity``.  Requiring this separate object prevents the
    score bundle from nominating its own expected identities.  The resolver
    must authenticate that the receipt artifact names this exact
    ``verification_subject_sha256``; a digest alone is not a signature.
    """

    stage: Stage
    protocol_sha256: str
    score_evidence_sha256: str
    source_manifest_sha256: str
    executor_manifest_sha256: str
    qualification_manifest_sha256: str
    execution_closure_sha256: str
    trust_anchor_identity: str
    verification_subject_sha256: str
    verification_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.stage not in {"open_tuning", "sealed_evaluation"}:
            raise ForagerMatchedEvidenceError("authenticated evidence stage is unsupported")
        for name, value in (
            ("protocol_sha256", self.protocol_sha256),
            ("score_evidence_sha256", self.score_evidence_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("executor_manifest_sha256", self.executor_manifest_sha256),
            ("qualification_manifest_sha256", self.qualification_manifest_sha256),
            ("execution_closure_sha256", self.execution_closure_sha256),
            (
                "verification_subject_sha256",
                self.verification_subject_sha256,
            ),
            (
                "verification_receipt_sha256",
                self.verification_receipt_sha256,
            ),
        ):
            _sha256(value, f"authenticated evidence {name}")
        _identifier(
            self.trust_anchor_identity,
            "authenticated evidence trust_anchor_identity",
        )
        expected_subject = _verification_subject_sha256(
            stage=self.stage,
            protocol_sha256=self.protocol_sha256,
            score_evidence_sha256=self.score_evidence_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            executor_manifest_sha256=self.executor_manifest_sha256,
            qualification_manifest_sha256=self.qualification_manifest_sha256,
            execution_closure_sha256=self.execution_closure_sha256,
            trust_anchor_identity=self.trust_anchor_identity,
        )
        if self.verification_subject_sha256 != expected_subject:
            raise ForagerMatchedEvidenceError(
                "authenticated evidence verification subject does not bind "
                "the exact evidence closure"
            )
        domain_digests = (
            self.score_evidence_sha256,
            self.source_manifest_sha256,
            self.executor_manifest_sha256,
            self.qualification_manifest_sha256,
            self.execution_closure_sha256,
            self.verification_subject_sha256,
            self.verification_receipt_sha256,
        )
        if len(set(domain_digests)) != len(domain_digests):
            raise ForagerMatchedEvidenceError(
                "authenticated evidence reuses a digest across distinct evidence domains"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION),
            "stage": self.stage,
            "protocol_sha256": self.protocol_sha256,
            "score_evidence_sha256": self.score_evidence_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "executor_manifest_sha256": self.executor_manifest_sha256,
            "qualification_manifest_sha256": self.qualification_manifest_sha256,
            "execution_closure_sha256": self.execution_closure_sha256,
            "trust_anchor_identity": self.trust_anchor_identity,
            "verification_subject_sha256": (self.verification_subject_sha256),
            "verification_receipt_sha256": (self.verification_receipt_sha256),
        }

    @property
    def bindings_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SelectionComputation:
    """Opaque-score selection result plus a replayable, score-free report."""

    selection_result: ForagerMatchedSelectionResult
    report: Mapping[str, Any]
    _factory_token: InitVar[object]

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SELECTION_COMPUTATION_FACTORY_TOKEN:
            raise ForagerMatchedEvidenceError(
                "SelectionComputation instances must come from authenticated selection replay"
            )
        try:
            result = parse_forager_matched_selection_result(self.selection_result.to_dict())
        except ForagerMatchedProtocolError as exc:
            raise ForagerMatchedEvidenceError(f"selection result is invalid: {exc}") from exc
        report = _parse_matched_selection_report_structure(
            self.report,
            selection_result=result,
        )
        object.__setattr__(self, "selection_result", result)
        object.__setattr__(self, "report", report)

    @property
    def report_sha256(self) -> str:
        return _sha256(self.report["payload_sha256"], "selection report payload_sha256")

    @property
    def canonical_report_bytes(self) -> bytes:
        return _canonical_json_bytes(cast(dict[str, Any], _thaw_json(self.report)))


def _parse_matched_selection_report_structure(
    value: Mapping[str, Any] | bytes | str,
    *,
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any],
    expected_payload_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Parse a canonical score-free report and bind it to its exact result."""
    try:
        result = (
            parse_forager_matched_selection_result(selection_result.to_dict())
            if isinstance(selection_result, ForagerMatchedSelectionResult)
            else parse_forager_matched_selection_result(selection_result)
        )
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvidenceError(f"selection result is invalid: {exc}") from exc
    if type(value) in (bytes, str):
        decoded = decode_strict_json(value)
        if type(value) is bytes:
            input_bytes = value
        else:
            try:
                input_bytes = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedEvidenceError(
                    "selection report contains invalid Unicode"
                ) from exc
        canonical_input = True
    elif isinstance(value, Mapping):
        try:
            thawed = cast(dict[str, Any], _thaw_json(value))
        except ForagerMatchedEvidenceError:
            raise
        except (RecursionError, TypeError, ValueError) as exc:
            raise ForagerMatchedEvidenceError(
                "selection report mapping is not a finite JSON tree"
            ) from exc
        decoded = decode_strict_json(_canonical_json_bytes(thawed))
        input_bytes = b""
        canonical_input = False
    else:
        raise TypeError("selection report must be a mapping, bytes, or str")
    payload = _object(decoded, "selection_report")
    _exact_keys(
        payload,
        {
            "schema_version",
            "open_protocol_sha256",
            "score_evidence_sha256",
            "authenticated_evidence_bindings_sha256",
            "external_verification_subject_sha256",
            "external_verification_receipt_sha256",
            "selection_plan_sha256",
            "selection_result_sha256",
            "raw_scores_embedded",
            "groups",
            "payload_sha256",
        },
        "selection_report",
    )
    if payload["schema_version"] != MATCHED_SELECTION_REPORT_SCHEMA_VERSION:
        raise ForagerMatchedEvidenceError("selection report schema_version is unsupported")
    if payload["raw_scores_embedded"] is not False:
        raise ForagerMatchedEvidenceError("selection report must not embed raw scores")
    open_sha256 = _sha256(
        payload["open_protocol_sha256"],
        "selection_report.open_protocol_sha256",
    )
    score_sha256 = _sha256(
        payload["score_evidence_sha256"],
        "selection_report.score_evidence_sha256",
    )
    authenticated_bindings_sha256 = _sha256(
        payload["authenticated_evidence_bindings_sha256"],
        "selection_report.authenticated_evidence_bindings_sha256",
    )
    verification_subject_sha256 = _sha256(
        payload["external_verification_subject_sha256"],
        "selection_report.external_verification_subject_sha256",
    )
    verification_receipt_sha256 = _sha256(
        payload["external_verification_receipt_sha256"],
        "selection_report.external_verification_receipt_sha256",
    )
    plan_sha256 = _sha256(
        payload["selection_plan_sha256"],
        "selection_report.selection_plan_sha256",
    )
    result_sha256 = _sha256(
        payload["selection_result_sha256"],
        "selection_report.selection_result_sha256",
    )
    if (
        open_sha256 != result.open_protocol_sha256
        or plan_sha256 != result.selection_plan_sha256
        or result_sha256 != result.selection_result_sha256
    ):
        raise ForagerMatchedEvidenceError(
            "selection report does not bind its exact selection result"
        )
    raw_groups = _array(payload["groups"], "selection_report.groups")
    if not 1 <= len(raw_groups) <= _MAX_CANDIDATES:
        raise ForagerMatchedEvidenceError("selection report groups have invalid length")
    parsed_group_rows: list[tuple[str, tuple[str, ...], str]] = []
    for group_index, raw_group in enumerate(raw_groups):
        path = f"selection_report.groups[{group_index}]"
        group = _object(raw_group, path)
        _exact_keys(
            group,
            {
                "schema_version",
                "open_protocol_sha256",
                "score_evidence_sha256",
                "authenticated_evidence_bindings_sha256",
                "external_verification_subject_sha256",
                "external_verification_receipt_sha256",
                "selection_plan_sha256",
                "selection_group",
                "ranked_candidate_ids",
                "candidate_statistics",
                "ranking_evidence_sha256",
            },
            path,
        )
        if group["schema_version"] != MATCHED_SELECTION_GROUP_EVIDENCE_SCHEMA_VERSION:
            raise ForagerMatchedEvidenceError(f"{path}.schema_version is unsupported")
        if (
            group["open_protocol_sha256"] != open_sha256
            or group["score_evidence_sha256"] != score_sha256
            or group["authenticated_evidence_bindings_sha256"] != authenticated_bindings_sha256
            or group["external_verification_subject_sha256"] != verification_subject_sha256
            or group["external_verification_receipt_sha256"] != verification_receipt_sha256
            or group["selection_plan_sha256"] != plan_sha256
        ):
            raise ForagerMatchedEvidenceError(
                f"{path} differs from the report-wide evidence bindings"
            )
        group_id = _identifier(
            group["selection_group"],
            f"{path}.selection_group",
        )
        raw_ranked = _array(
            group["ranked_candidate_ids"],
            f"{path}.ranked_candidate_ids",
        )
        if not 1 <= len(raw_ranked) <= _MAX_CANDIDATES:
            raise ForagerMatchedEvidenceError(f"{path}.ranked_candidate_ids has invalid length")
        ranked = tuple(
            _identifier(candidate_id, f"{path}.ranked_candidate_ids[{index}]")
            for index, candidate_id in enumerate(raw_ranked)
        )
        if len(set(ranked)) != len(ranked):
            raise ForagerMatchedEvidenceError(f"{path}.ranked_candidate_ids repeats a candidate")
        raw_statistics = _array(
            group["candidate_statistics"],
            f"{path}.candidate_statistics",
        )
        if len(raw_statistics) != len(ranked):
            raise ForagerMatchedEvidenceError(
                f"{path}.candidate_statistics does not cover the ranked set"
            )
        statistic_ids: list[str] = []
        for statistic_index, raw_statistic in enumerate(raw_statistics):
            statistic_path = f"{path}.candidate_statistics[{statistic_index}]"
            statistic = _object(raw_statistic, statistic_path)
            _exact_keys(
                statistic,
                {
                    "candidate_id",
                    "score_vector_sha256",
                    "mean_hex",
                    "selection_statistic_hex",
                },
                statistic_path,
            )
            statistic_ids.append(
                _identifier(
                    statistic["candidate_id"],
                    f"{statistic_path}.candidate_id",
                )
            )
            _sha256(
                statistic["score_vector_sha256"],
                f"{statistic_path}.score_vector_sha256",
            )
            _score_from_hex(
                statistic["mean_hex"],
                f"{statistic_path}.mean_hex",
            )
            _score_from_hex(
                statistic["selection_statistic_hex"],
                f"{statistic_path}.selection_statistic_hex",
            )
        if tuple(statistic_ids) != tuple(sorted(statistic_ids)) or set(statistic_ids) != set(
            ranked
        ):
            raise ForagerMatchedEvidenceError(
                f"{path}.candidate_statistics must be the exact ranked set in candidate-ID order"
            )
        declared_ranking_sha256 = _sha256(
            group["ranking_evidence_sha256"],
            f"{path}.ranking_evidence_sha256",
        )
        group_body = dict(group)
        del group_body["ranking_evidence_sha256"]
        if _canonical_sha256(group_body) != declared_ranking_sha256:
            raise ForagerMatchedEvidenceError(f"{path}.ranking_evidence_sha256 does not verify")
        parsed_group_rows.append((group_id, ranked, declared_ranking_sha256))
    expected_group_rows = tuple(
        (
            group.selection_group,
            group.ranked_candidate_ids,
            group.ranking_evidence_sha256,
        )
        for group in result.ranked_groups
    )
    if tuple(parsed_group_rows) != expected_group_rows:
        raise ForagerMatchedEvidenceError(
            "selection report group evidence differs from the selection result"
        )
    declared_payload_sha256 = _sha256(
        payload["payload_sha256"],
        "selection_report.payload_sha256",
    )
    unsigned = dict(payload)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared_payload_sha256:
        raise ForagerMatchedEvidenceError("selection report payload_sha256 does not verify")
    if expected_payload_sha256 is not None and declared_payload_sha256 != _sha256(
        expected_payload_sha256,
        "expected_payload_sha256",
    ):
        raise ForagerMatchedEvidenceError(
            "selection report does not match the externally expected digest"
        )
    canonical = _canonical_json_bytes(payload)
    if canonical_input and input_bytes != canonical:
        raise ForagerMatchedEvidenceError("selection report bytes are not canonical")
    return cast(Mapping[str, Any], _freeze_json(payload))


def load_matched_selection_report(
    path: str | Path,
    *,
    open_protocol: ForagerMatchedProtocol | Mapping[str, Any],
    open_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    authenticated_bindings: AuthenticatedEvidenceBindings,
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any],
    expected_payload_sha256: str,
) -> Mapping[str, Any]:
    """Load one stable canonical report with an out-of-band digest."""
    raw = _read_stable_regular_file(path, label="selection report")
    return parse_matched_selection_report(
        raw,
        open_protocol=open_protocol,
        open_evidence=open_evidence,
        authenticated_bindings=authenticated_bindings,
        selection_result=selection_result,
        expected_payload_sha256=expected_payload_sha256,
    )


def _parse_seed_record(value: Any, path: str) -> SeedScoreRecord:
    payload = _object(value, path)
    _exact_keys(
        payload,
        {
            "seed",
            "score_hex",
            "raw_artifact_sha256",
            "reward_trace_sha256",
            "scoring_record_sha256",
        },
        path,
    )
    return SeedScoreRecord(
        seed=_integer(
            payload["seed"],
            f"{path}.seed",
            minimum=0,
            maximum=_MAX_SEED,
        ),
        score=_score_from_hex(payload["score_hex"], f"{path}.score_hex"),
        raw_artifact_sha256=_sha256(
            payload["raw_artifact_sha256"],
            f"{path}.raw_artifact_sha256",
        ),
        reward_trace_sha256=_sha256(
            payload["reward_trace_sha256"],
            f"{path}.reward_trace_sha256",
        ),
        scoring_record_sha256=_sha256(
            payload["scoring_record_sha256"],
            f"{path}.scoring_record_sha256",
        ),
    )


def _parse_candidate_scores(
    value: Any,
    path: str,
) -> CandidateScoreEvidence:
    payload = _object(value, path)
    _exact_keys(
        payload,
        {
            "candidate_id",
            "capability_descriptor_sha256",
            "capability_qualification_receipt_sha256",
            "execution_receipt_sha256",
            "records",
        },
        path,
    )
    raw_records = _array(payload["records"], f"{path}.records")
    if not 1 <= len(raw_records) <= _MAX_SEEDS:
        raise ForagerMatchedEvidenceError(f"{path}.records must contain 1..{_MAX_SEEDS} entries")
    records = tuple(
        _parse_seed_record(record, f"{path}.records[{index}]")
        for index, record in enumerate(raw_records)
    )
    seeds = tuple(record.seed for record in records)
    if len(set(seeds)) != len(seeds):
        raise ForagerMatchedEvidenceError(f"{path}.records repeats a seed")
    return CandidateScoreEvidence(
        candidate_id=_identifier(
            payload["candidate_id"],
            f"{path}.candidate_id",
        ),
        capability_descriptor_sha256=_sha256(
            payload["capability_descriptor_sha256"],
            f"{path}.capability_descriptor_sha256",
        ),
        capability_qualification_receipt_sha256=_sha256(
            payload["capability_qualification_receipt_sha256"],
            f"{path}.capability_qualification_receipt_sha256",
        ),
        execution_receipt_sha256=_sha256(
            payload["execution_receipt_sha256"],
            f"{path}.execution_receipt_sha256",
        ),
        records=records,
    )


def parse_matched_score_evidence(
    value: Mapping[str, Any] | bytes | str,
    *,
    expected_payload_sha256: str | None = None,
) -> MatchedScoreEvidence:
    """Parse and self-verify one canonical score-evidence bundle."""
    if type(value) in (bytes, str):
        raw = value
        decoded = decode_strict_json(raw)
        input_bytes = raw if type(raw) is bytes else raw.encode("utf-8")
        canonical_input = True
    elif isinstance(value, Mapping):
        decoded = decode_strict_json(_canonical_json_bytes(dict(value)))
        canonical_input = False
        input_bytes = b""
    else:
        raise TypeError("score evidence must be a mapping, bytes, or str")
    payload = _object(decoded, "score_evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "stage",
            "protocol_sha256",
            "active_seeds",
            "horizon",
            "metric",
            "metric_implementation_sha256",
            "task_identity_sha256",
            "environment_rng_schedule_sha256",
            "runtime_profile_sha256",
            "source_evidence_sha256",
            "executor_evidence_sha256",
            "qualification_manifest_sha256",
            "candidate_scores",
            "payload_sha256",
        },
        "score_evidence",
    )
    if payload["schema_version"] != MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION:
        raise ForagerMatchedEvidenceError("score_evidence.schema_version is unsupported")
    stage = _string(payload["stage"], "score_evidence.stage")
    if stage not in {"open_tuning", "sealed_evaluation"}:
        raise ForagerMatchedEvidenceError("score_evidence.stage is unsupported")
    raw_seeds = _array(
        payload["active_seeds"],
        "score_evidence.active_seeds",
    )
    if not 1 <= len(raw_seeds) <= _MAX_SEEDS:
        raise ForagerMatchedEvidenceError("score_evidence.active_seeds has invalid length")
    seeds = tuple(
        _integer(
            seed,
            f"score_evidence.active_seeds[{index}]",
            minimum=0,
            maximum=_MAX_SEED,
        )
        for index, seed in enumerate(raw_seeds)
    )
    if len(set(seeds)) != len(seeds):
        raise ForagerMatchedEvidenceError("score_evidence.active_seeds repeats a seed")
    raw_candidates = _array(
        payload["candidate_scores"],
        "score_evidence.candidate_scores",
    )
    if not 1 <= len(raw_candidates) <= _MAX_CANDIDATES:
        raise ForagerMatchedEvidenceError("score_evidence.candidate_scores has invalid length")
    candidates = tuple(
        _parse_candidate_scores(
            candidate,
            f"score_evidence.candidate_scores[{index}]",
        )
        for index, candidate in enumerate(raw_candidates)
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ForagerMatchedEvidenceError("score_evidence.candidate_scores repeats a candidate")
    declared_sha256 = _sha256(
        payload["payload_sha256"],
        "score_evidence.payload_sha256",
    )
    unsigned = dict(payload)
    del unsigned["payload_sha256"]
    if _canonical_sha256(unsigned) != declared_sha256:
        raise ForagerMatchedEvidenceError("score_evidence.payload_sha256 does not verify")
    if expected_payload_sha256 is not None and declared_sha256 != _sha256(
        expected_payload_sha256,
        "expected_payload_sha256",
    ):
        raise ForagerMatchedEvidenceError(
            "score evidence does not match the externally expected digest"
        )
    if canonical_input and input_bytes != _canonical_json_bytes(payload):
        raise ForagerMatchedEvidenceError("score evidence bytes are not canonical")
    return MatchedScoreEvidence(
        schema_version=MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION,
        stage=cast(Stage, stage),
        protocol_sha256=_sha256(
            payload["protocol_sha256"],
            "score_evidence.protocol_sha256",
        ),
        active_seeds=seeds,
        horizon=_integer(
            payload["horizon"],
            "score_evidence.horizon",
            minimum=1,
            maximum=2**31 - 1,
        ),
        metric=_identifier(payload["metric"], "score_evidence.metric"),
        metric_implementation_sha256=_sha256(
            payload["metric_implementation_sha256"],
            "score_evidence.metric_implementation_sha256",
        ),
        task_identity_sha256=_sha256(
            payload["task_identity_sha256"],
            "score_evidence.task_identity_sha256",
        ),
        environment_rng_schedule_sha256=_sha256(
            payload["environment_rng_schedule_sha256"],
            "score_evidence.environment_rng_schedule_sha256",
        ),
        runtime_profile_sha256=_sha256(
            payload["runtime_profile_sha256"],
            "score_evidence.runtime_profile_sha256",
        ),
        source_evidence_sha256=_sha256(
            payload["source_evidence_sha256"],
            "score_evidence.source_evidence_sha256",
        ),
        executor_evidence_sha256=_sha256(
            payload["executor_evidence_sha256"],
            "score_evidence.executor_evidence_sha256",
        ),
        qualification_manifest_sha256=_sha256(
            payload["qualification_manifest_sha256"],
            "score_evidence.qualification_manifest_sha256",
        ),
        candidate_scores=candidates,
        payload_sha256=declared_sha256,
    )


def load_matched_score_evidence(
    path: str | Path,
    *,
    expected_payload_sha256: str,
) -> MatchedScoreEvidence:
    """Load one stable canonical score bundle with an out-of-band digest."""
    raw = _read_stable_regular_file(path, label="score evidence")
    return parse_matched_score_evidence(
        raw,
        expected_payload_sha256=expected_payload_sha256,
    )


def load_forager_matched_selection_result(
    path: str | Path,
    *,
    expected_selection_result_sha256: str,
) -> ForagerMatchedSelectionResult:
    """Load one stable canonical selection result with an out-of-band digest."""
    expected_digest = _sha256(
        expected_selection_result_sha256,
        "expected_selection_result_sha256",
    )
    raw = _read_stable_regular_file(path, label="selection result")
    try:
        result = parse_forager_matched_selection_result(raw)
        canonical = canonical_selection_result_bytes(result)
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvidenceError(f"selection result is invalid: {exc}") from exc
    if raw != canonical:
        raise ForagerMatchedEvidenceError("selection result bytes are not canonical")
    if result.selection_result_sha256 != expected_digest:
        raise ForagerMatchedEvidenceError(
            "selection result does not match the externally expected digest"
        )
    return result


def _protocol_instance(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
) -> ForagerMatchedProtocol:
    try:
        if isinstance(protocol, ForagerMatchedProtocol):
            return parse_forager_matched_protocol(protocol.to_dict())
        return parse_forager_matched_protocol(protocol)
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvidenceError(f"matched protocol is invalid: {exc}") from exc


def matched_execution_closure_sha256(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    score_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
) -> str:
    """Hash the complete runtime/candidate/record subject a verifier attests."""
    frozen = _protocol_instance(protocol)
    scores = (
        parse_matched_score_evidence(score_evidence.to_dict())
        if isinstance(score_evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(score_evidence)
    )
    execution_receipts = tuple(
        candidate.execution_receipt_sha256 for candidate in scores.candidate_scores
    )
    raw_artifacts = tuple(
        record.raw_artifact_sha256
        for candidate in scores.candidate_scores
        for record in candidate.records
    )
    scoring_records = tuple(
        record.scoring_record_sha256
        for candidate in scores.candidate_scores
        for record in candidate.records
    )
    reward_traces = tuple(
        record.reward_trace_sha256
        for candidate in scores.candidate_scores
        for record in candidate.records
    )
    for label, values in (
        ("candidate execution receipts", execution_receipts),
        ("per-seed raw artifacts", raw_artifacts),
        ("per-seed scoring records", scoring_records),
    ):
        if len(set(values)) != len(values):
            raise ForagerMatchedEvidenceError(f"execution closure reuses {label}")
    artifact_domains = (
        ("score evidence", {scores.payload_sha256}),
        ("source manifest", {scores.source_evidence_sha256}),
        ("executor manifest", {scores.executor_evidence_sha256}),
        ("qualification manifest", {scores.qualification_manifest_sha256}),
        (
            "capability descriptors",
            {candidate.capability_descriptor_sha256 for candidate in scores.candidate_scores},
        ),
        (
            "capability qualification receipts",
            {
                candidate.capability_qualification_receipt_sha256
                for candidate in scores.candidate_scores
            },
        ),
        ("candidate execution receipts", set(execution_receipts)),
        ("per-seed raw artifacts", set(raw_artifacts)),
        ("per-seed reward traces", set(reward_traces)),
        ("per-seed scoring records", set(scoring_records)),
    )
    for left_index, (left_label, left_values) in enumerate(artifact_domains):
        for right_label, right_values in artifact_domains[left_index + 1 :]:
            if left_values & right_values:
                raise ForagerMatchedEvidenceError(
                    "execution closure reuses a digest across distinct artifact "
                    f"domains: {left_label} and {right_label}"
                )
    return _canonical_sha256(
        {
            "schema_version": MATCHED_EXECUTION_CLOSURE_SCHEMA_VERSION,
            "stage": scores.stage,
            "protocol_sha256": scores.protocol_sha256,
            "active_seeds": list(scores.active_seeds),
            "horizon": scores.horizon,
            "metric": scores.metric,
            "metric_implementation_sha256": (scores.metric_implementation_sha256),
            "task_identity_sha256": scores.task_identity_sha256,
            "environment_rng_schedule_sha256": (scores.environment_rng_schedule_sha256),
            "runtime": {
                "image_sha256": frozen.runtime.image_sha256,
                "runtime_profile_sha256": scores.runtime_profile_sha256,
                "executor_qualification_receipt_sha256": (
                    frozen.runtime.executor_qualification_receipt_sha256
                ),
                "qualification_trust_anchor_identity": (
                    frozen.runtime.qualification_trust_anchor_identity
                ),
            },
            "source_manifest_sha256": scores.source_evidence_sha256,
            "executor_manifest_sha256": scores.executor_evidence_sha256,
            "qualification_manifest_sha256": scores.qualification_manifest_sha256,
            "candidates": [candidate.to_dict() for candidate in scores.candidate_scores],
        }
    )


def matched_verification_subject_sha256(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    score_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
) -> str:
    """Hash the exact non-circular subject an external receipt must attest."""
    frozen = _protocol_instance(protocol)
    scores = (
        parse_matched_score_evidence(score_evidence.to_dict())
        if isinstance(score_evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(score_evidence)
    )
    return _verification_subject_sha256(
        stage=scores.stage,
        protocol_sha256=scores.protocol_sha256,
        score_evidence_sha256=scores.payload_sha256,
        source_manifest_sha256=scores.source_evidence_sha256,
        executor_manifest_sha256=scores.executor_evidence_sha256,
        qualification_manifest_sha256=scores.qualification_manifest_sha256,
        execution_closure_sha256=matched_execution_closure_sha256(
            frozen,
            scores,
        ),
        trust_anchor_identity=(frozen.runtime.qualification_trust_anchor_identity),
    )


def _validate_authenticated_bindings(
    protocol: ForagerMatchedProtocol,
    scores: MatchedScoreEvidence,
    bindings: AuthenticatedEvidenceBindings,
) -> None:
    if type(bindings) is not AuthenticatedEvidenceBindings:
        raise ForagerMatchedEvidenceError(
            "authenticated_bindings must be an AuthenticatedEvidenceBindings"
        )
    actual = {
        "stage": scores.stage,
        "protocol_sha256": scores.protocol_sha256,
        "score_evidence_sha256": scores.payload_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "qualification_manifest_sha256": scores.qualification_manifest_sha256,
        "execution_closure_sha256": matched_execution_closure_sha256(
            protocol,
            scores,
        ),
        "trust_anchor_identity": (protocol.runtime.qualification_trust_anchor_identity),
        "verification_subject_sha256": matched_verification_subject_sha256(
            protocol,
            scores,
        ),
    }
    expected = {
        "stage": bindings.stage,
        "protocol_sha256": bindings.protocol_sha256,
        "score_evidence_sha256": bindings.score_evidence_sha256,
        "source_manifest_sha256": bindings.source_manifest_sha256,
        "executor_manifest_sha256": bindings.executor_manifest_sha256,
        "qualification_manifest_sha256": bindings.qualification_manifest_sha256,
        "execution_closure_sha256": bindings.execution_closure_sha256,
        "trust_anchor_identity": bindings.trust_anchor_identity,
        "verification_subject_sha256": bindings.verification_subject_sha256,
    }
    drifted = [name for name, value in actual.items() if value != expected[name]]
    if drifted:
        raise ForagerMatchedEvidenceError(
            "score evidence differs from externally authenticated bindings: " + ", ".join(drifted)
        )


def _expected_tuning_candidate_ids(
    protocol: ForagerMatchedProtocol,
) -> tuple[str, ...]:
    return tuple(
        candidate_id
        for group in protocol.selection_plan.groups
        for candidate_id in group.candidate_ids
    )


def validate_score_evidence_against_protocol(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    *,
    authenticated_bindings: AuthenticatedEvidenceBindings,
    expected_candidate_ids: Sequence[str] | None = None,
) -> tuple[ForagerMatchedProtocol, MatchedScoreEvidence]:
    """Validate the full score bundle against one exact protocol stage."""
    frozen = _protocol_instance(protocol)
    scores = (
        parse_matched_score_evidence(evidence.to_dict())
        if isinstance(evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(evidence)
    )
    if expected_candidate_ids is None:
        expected_ids = (
            _expected_tuning_candidate_ids(frozen) if frozen.stage == "open_tuning" else ()
        )
    else:
        if type(expected_candidate_ids) not in {list, tuple}:
            raise ForagerMatchedEvidenceError("expected candidate IDs must be a list or tuple")
        if len(expected_candidate_ids) > _MAX_CANDIDATES:
            raise ForagerMatchedEvidenceError("expected candidate IDs exceed the candidate bound")
        expected_ids = tuple(expected_candidate_ids)
    if not expected_ids:
        raise ForagerMatchedEvidenceError(
            "expected candidate IDs must be explicit for sealed evidence"
        )
    expected_ids = tuple(
        _identifier(
            candidate_id,
            f"expected_candidate_ids[{index}]",
        )
        for index, candidate_id in enumerate(expected_ids)
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise ForagerMatchedEvidenceError("expected candidate IDs contain duplicates")
    unknown_expected = tuple(
        candidate_id for candidate_id in expected_ids if candidate_id not in frozen.candidate_index
    )
    if unknown_expected:
        raise ForagerMatchedEvidenceError(
            "expected candidate IDs are absent from the protocol: " + ", ".join(unknown_expected)
        )
    actual_ids = tuple(candidate.candidate_id for candidate in scores.candidate_scores)
    expected_common = {
        "stage": frozen.stage,
        "protocol_sha256": frozen.protocol_sha256,
        "active_seeds": frozen.active_seeds,
        "horizon": frozen.horizon,
        "metric": frozen.analysis_plan.metric,
        "metric_implementation_sha256": (frozen.analysis_plan.metric_implementation_sha256),
        "task_identity_sha256": frozen.task.task_identity_sha256,
        "environment_rng_schedule_sha256": (frozen.task.environment_rng_schedule_sha256),
        "runtime_profile_sha256": frozen.runtime.runtime_profile_sha256,
        "candidate_ids": expected_ids,
    }
    actual_common = {
        "stage": scores.stage,
        "protocol_sha256": scores.protocol_sha256,
        "active_seeds": scores.active_seeds,
        "horizon": scores.horizon,
        "metric": scores.metric,
        "metric_implementation_sha256": scores.metric_implementation_sha256,
        "task_identity_sha256": scores.task_identity_sha256,
        "environment_rng_schedule_sha256": (scores.environment_rng_schedule_sha256),
        "runtime_profile_sha256": scores.runtime_profile_sha256,
        "candidate_ids": actual_ids,
    }
    drifted = [
        name for name, expected in expected_common.items() if actual_common[name] != expected
    ]
    if drifted:
        raise ForagerMatchedEvidenceError(
            "score evidence differs from the frozen protocol: " + ", ".join(drifted)
        )
    for candidate_scores in scores.candidate_scores:
        candidate = frozen.candidate_index.get(candidate_scores.candidate_id)
        if candidate is None:
            raise ForagerMatchedEvidenceError(
                f"score evidence names unknown candidate {candidate_scores.candidate_id!r}"
            )
        if candidate_scores.seeds != frozen.active_seeds:
            raise ForagerMatchedEvidenceError(
                f"candidate {candidate.candidate_id!r} does not contain the exact active seed block"
            )
        expected_subject = candidate_capability_descriptor_sha256(candidate)
        binding = candidate.runtime_binding
        if (
            candidate_scores.capability_descriptor_sha256 != expected_subject
            or binding.qualified_capability_descriptor_sha256 != expected_subject
            or candidate_scores.capability_qualification_receipt_sha256
            != binding.capability_qualification_receipt_sha256
        ):
            raise ForagerMatchedEvidenceError(
                f"candidate {candidate.candidate_id!r} capability evidence "
                "does not bind its frozen semantics"
            )
    _validate_authenticated_bindings(
        frozen,
        scores,
        authenticated_bindings,
    )
    return frozen, scores


def _bootstrap_lower_endpoint(
    values: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Return ``(mean, lower)`` for the ``conservative_ci_endpoint`` ranking statistic.

    ``lower`` is the ``(1 - confidence) / 2`` quantile (``numpy.quantile`` with
    ``method="linear"``) of the bootstrap distribution of the mean — the lower
    endpoint of a two-sided equal-tail percentile interval.  Per-seed scores are
    resampled with replacement via ``numpy.random.Generator(PCG64(seed))``; the
    frozen plan supplies the same seed for every candidate, so ranks replay
    deterministically.  A singleton vector degenerates to a point mass at its mean.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ForagerMatchedEvidenceError("selection values must be a non-empty finite vector")
    mean = float(np.mean(array, dtype=np.float64))
    if array.size == 1:
        return mean, mean
    means = np.empty(resamples, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    rows_per_chunk = max(
        1,
        _BOOTSTRAP_CHUNK_ELEMENTS // int(array.size),
    )
    offset = 0
    while offset < resamples:
        count = min(rows_per_chunk, resamples - offset)
        indices = rng.integers(
            0,
            array.size,
            size=(count, array.size),
            dtype=np.int64,
        )
        means[offset : offset + count] = np.mean(
            array[indices],
            axis=1,
            dtype=np.float64,
        )
        offset += count
    quantile = (1.0 - confidence) / 2.0
    lower = float(np.quantile(means, quantile, method="linear"))
    if not math.isfinite(mean) or not math.isfinite(lower):
        raise ForagerMatchedEvidenceError("selection bootstrap produced a non-finite statistic")
    return mean, lower


def compute_open_selection(
    protocol: ForagerMatchedProtocol | Mapping[str, Any],
    evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    *,
    authenticated_bindings: AuthenticatedEvidenceBindings,
) -> SelectionComputation:
    """Replay the frozen open-tuning ranking and return its canonical result."""
    _verify_selection_implementation_descriptors()
    frozen, scores = validate_score_evidence_against_protocol(
        protocol,
        evidence,
        authenticated_bindings=authenticated_bindings,
    )
    if frozen.stage != "open_tuning":
        raise ForagerMatchedEvidenceError("selection requires an open_tuning protocol")
    if (
        frozen.selection_plan.statistic_implementation_sha256
        != MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
        or frozen.selection_plan.bootstrap_rng_implementation_sha256
        != MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
    ):
        raise ForagerMatchedEvidenceError(
            "selection plan implementation digests do not identify this replay implementation"
        )
    score_index = scores.candidate_index
    result_groups: list[RankedSelectionGroup] = []
    report_groups: list[dict[str, Any]] = []
    for group in frozen.selection_plan.groups:
        statistics: list[tuple[str, float, float, str]] = []
        for candidate_id in group.candidate_ids:
            candidate_scores = score_index[candidate_id]
            if frozen.selection_plan.statistic == "mean":
                array = np.asarray(candidate_scores.scores, dtype=np.float64)
                if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
                    raise ForagerMatchedEvidenceError(
                        "selection values must be a non-empty finite vector"
                    )
                mean = float(np.mean(array, dtype=np.float64))
                if not math.isfinite(mean):
                    raise ForagerMatchedEvidenceError(
                        "selection mean produced a non-finite statistic"
                    )
                selection_value = mean
            else:
                mean, selection_value = _bootstrap_lower_endpoint(
                    candidate_scores.scores,
                    resamples=frozen.selection_plan.bootstrap_resamples,
                    seed=frozen.selection_plan.bootstrap_seed,
                    confidence=frozen.selection_plan.confidence,
                )
            statistics.append(
                (
                    candidate_id,
                    selection_value,
                    mean,
                    candidate_scores.score_vector_sha256,
                )
            )
        ordered = sorted(
            statistics,
            key=lambda item: (-item[1], item[0]),
        )
        group_body = {
            "schema_version": (MATCHED_SELECTION_GROUP_EVIDENCE_SCHEMA_VERSION),
            "open_protocol_sha256": frozen.protocol_sha256,
            "score_evidence_sha256": scores.payload_sha256,
            "authenticated_evidence_bindings_sha256": (authenticated_bindings.bindings_sha256),
            "external_verification_subject_sha256": (
                authenticated_bindings.verification_subject_sha256
            ),
            "external_verification_receipt_sha256": (
                authenticated_bindings.verification_receipt_sha256
            ),
            "selection_plan_sha256": frozen.selection_plan.plan_sha256,
            "selection_group": group.selection_group,
            "ranked_candidate_ids": [item[0] for item in ordered],
            "candidate_statistics": [
                {
                    "candidate_id": candidate_id,
                    "score_vector_sha256": vector_sha256,
                    "mean_hex": mean.hex(),
                    "selection_statistic_hex": selection_value.hex(),
                }
                for candidate_id, selection_value, mean, vector_sha256 in sorted(
                    statistics,
                    key=lambda item: item[0],
                )
            ],
        }
        ranking_sha256 = _canonical_sha256(group_body)
        result_groups.append(
            RankedSelectionGroup(
                selection_group=group.selection_group,
                ranked_candidate_ids=tuple(item[0] for item in ordered),
                ranking_evidence_sha256=ranking_sha256,
            )
        )
        report_groups.append(
            {
                **group_body,
                "ranking_evidence_sha256": ranking_sha256,
            }
        )
    result = parse_forager_matched_selection_result(
        {
            "schema_version": (FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION),
            "open_protocol_sha256": frozen.protocol_sha256,
            "selection_plan_sha256": frozen.selection_plan.plan_sha256,
            "tuning_seeds": list(frozen.tuning_seeds),
            "ranked_groups": [group.to_dict() for group in result_groups],
        }
    )
    report_body = {
        "schema_version": MATCHED_SELECTION_REPORT_SCHEMA_VERSION,
        "open_protocol_sha256": frozen.protocol_sha256,
        "score_evidence_sha256": scores.payload_sha256,
        "authenticated_evidence_bindings_sha256": (authenticated_bindings.bindings_sha256),
        "external_verification_subject_sha256": (
            authenticated_bindings.verification_subject_sha256
        ),
        "external_verification_receipt_sha256": (
            authenticated_bindings.verification_receipt_sha256
        ),
        "selection_plan_sha256": frozen.selection_plan.plan_sha256,
        "selection_result_sha256": result.selection_result_sha256,
        "raw_scores_embedded": False,
        "groups": report_groups,
    }
    report = {
        **report_body,
        "payload_sha256": _canonical_sha256(report_body),
    }
    return SelectionComputation(
        selection_result=result,
        report=report,
        _factory_token=_SELECTION_COMPUTATION_FACTORY_TOKEN,
    )


def parse_matched_selection_report(
    value: Mapping[str, Any] | bytes | str,
    *,
    open_protocol: ForagerMatchedProtocol | Mapping[str, Any],
    open_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    authenticated_bindings: AuthenticatedEvidenceBindings,
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any],
    expected_payload_sha256: str,
) -> Mapping[str, Any]:
    """Validate a report by replaying its authenticated open score evidence."""
    replay = compute_open_selection(
        open_protocol,
        open_evidence,
        authenticated_bindings=authenticated_bindings,
    )
    try:
        supplied_result = (
            parse_forager_matched_selection_result(selection_result.to_dict())
            if isinstance(
                selection_result,
                ForagerMatchedSelectionResult,
            )
            else parse_forager_matched_selection_result(selection_result)
        )
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvidenceError(f"selection result is invalid: {exc}") from exc
    if (
        supplied_result.selection_result_sha256 != replay.selection_result.selection_result_sha256
        or supplied_result.to_dict() != replay.selection_result.to_dict()
    ):
        raise ForagerMatchedEvidenceError(
            "selection result does not replay from authenticated open evidence"
        )
    parsed = _parse_matched_selection_report_structure(
        value,
        selection_result=supplied_result,
        expected_payload_sha256=expected_payload_sha256,
    )
    if (
        _canonical_json_bytes(cast(dict[str, Any], _thaw_json(parsed)))
        != replay.canonical_report_bytes
    ):
        raise ForagerMatchedEvidenceError(
            "selection report does not replay from authenticated open evidence"
        )
    return parsed


def _metric_binding_sha256(
    protocol: ForagerMatchedProtocol,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": MATCHED_METRIC_BINDING_SCHEMA_VERSION,
            "metric": protocol.analysis_plan.metric,
            "metric_implementation_sha256": (protocol.analysis_plan.metric_implementation_sha256),
            "direction": protocol.analysis_plan.metric_direction,
            "horizon": protocol.horizon,
        }
    )


def build_statistics_contract(
    open_protocol: ForagerMatchedProtocol | Mapping[str, Any],
    sealed_protocol: ForagerMatchedProtocol | Mapping[str, Any],
    selection_result: ForagerMatchedSelectionResult | Mapping[str, Any],
    selection_report: Mapping[str, Any] | bytes | str,
    open_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    evaluation_evidence: MatchedScoreEvidence | Mapping[str, Any] | bytes | str,
    *,
    open_authenticated_bindings: AuthenticatedEvidenceBindings,
    evaluation_authenticated_bindings: AuthenticatedEvidenceBindings,
    expected_selection_report_sha256: str,
) -> tuple[
    MatchedComparisonContract,
    SealedProtocolValidation,
    MatchedScoreEvidence,
]:
    """Build the exact statistics-v3 contract for one sealed transition."""
    open_value = _protocol_instance(open_protocol)
    sealed_value = _protocol_instance(sealed_protocol)
    open_score_value = (
        parse_matched_score_evidence(open_evidence.to_dict())
        if isinstance(open_evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(open_evidence)
    )
    evaluation_score_value = (
        parse_matched_score_evidence(evaluation_evidence.to_dict())
        if isinstance(evaluation_evidence, MatchedScoreEvidence)
        else parse_matched_score_evidence(evaluation_evidence)
    )
    replayed_selection = compute_open_selection(
        open_value,
        open_score_value,
        authenticated_bindings=open_authenticated_bindings,
    )
    try:
        result_value = (
            parse_forager_matched_selection_result(selection_result.to_dict())
            if isinstance(selection_result, ForagerMatchedSelectionResult)
            else parse_forager_matched_selection_result(selection_result)
        )
        if result_value.to_dict() != replayed_selection.selection_result.to_dict():
            raise ForagerMatchedEvidenceError(
                "selection result does not replay from authenticated open evidence"
            )
        validated_report = parse_matched_selection_report(
            selection_report,
            open_protocol=open_value,
            open_evidence=open_evidence,
            authenticated_bindings=open_authenticated_bindings,
            selection_result=result_value,
            expected_payload_sha256=expected_selection_report_sha256,
        )
        transition = validate_sealed_protocol_transition(
            open_value,
            sealed_value,
            result_value,
            replayed_selection.selection_result.selection_result_sha256,
        )
    except ForagerMatchedEvidenceError:
        raise
    except ForagerMatchedProtocolError as exc:
        raise ForagerMatchedEvidenceError(f"sealed protocol transition is invalid: {exc}") from exc
    _, evidence = validate_score_evidence_against_protocol(
        sealed_value,
        evaluation_score_value,
        authenticated_bindings=evaluation_authenticated_bindings,
        expected_candidate_ids=transition.evaluation_candidate_ids,
    )
    qualification_digests = {
        open_score_value.qualification_manifest_sha256,
        evidence.qualification_manifest_sha256,
        open_authenticated_bindings.qualification_manifest_sha256,
        evaluation_authenticated_bindings.qualification_manifest_sha256,
    }
    if len(qualification_digests) != 1:
        raise ForagerMatchedEvidenceError(
            "open and evaluation evidence do not share one exact qualification manifest"
        )
    selection_report_sha256 = _sha256(
        validated_report["payload_sha256"],
        "validated selection report payload_sha256",
    )
    score_index = evidence.candidate_index
    inferential_ids = tuple(
        candidate_id
        for candidate_id in transition.evaluation_candidate_ids
        if sealed_value.candidate_index[candidate_id].pairing.eligible
        and sealed_value.candidate_index[candidate_id].pairing.analysis_role == "inferential"
    )
    resolved = transition.resolved_hypotheses
    if not resolved:
        raise ForagerMatchedEvidenceError("sealed transition resolved no hypotheses")
    comparisons = tuple(
        ComparisonSpec(
            hypothesis_id=hypothesis.hypothesis_id,
            intervention_id=hypothesis.intervention_candidate_id,
            comparator_id=hypothesis.comparator_candidate_id,
        )
        for hypothesis in resolved
    )
    plan = sealed_value.analysis_plan
    if (
        plan.primary.implementation_sha256 != PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
        or plan.secondary.implementation_sha256 != SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
    ):
        raise ForagerMatchedEvidenceError(
            "analysis plan implementation digests do not identify the "
            "statistics-v3 replay implementations"
        )
    try:
        common_evidence = EvidenceBinding(
            horizon=sealed_value.horizon,
            metric_sha256=_metric_binding_sha256(sealed_value),
            environment_sha256=sealed_value.task.task_identity_sha256,
            rng_schedule_sha256=(sealed_value.task.environment_rng_schedule_sha256),
            runtime_profile_sha256=(sealed_value.runtime.runtime_profile_sha256),
            source_evidence_sha256=evidence.source_evidence_sha256,
            executor_evidence_sha256=evidence.executor_evidence_sha256,
            score_evidence_sha256=evidence.payload_sha256,
            execution_closure_sha256=(evaluation_authenticated_bindings.execution_closure_sha256),
            authenticated_bindings_sha256=(evaluation_authenticated_bindings.bindings_sha256),
            external_verification_subject_sha256=(
                evaluation_authenticated_bindings.verification_subject_sha256
            ),
            external_verification_receipt_sha256=(
                evaluation_authenticated_bindings.verification_receipt_sha256
            ),
            sealed_protocol_sha256=sealed_value.protocol_sha256,
            selection_result_sha256=result_value.selection_result_sha256,
            selection_report_sha256=selection_report_sha256,
        )
        methods = tuple(
            LearningMethodScores(
                method_id=candidate_id,
                seeds=sealed_value.evaluation_seeds,
                scores=score_index[candidate_id].scores,
                evidence=common_evidence,
                preregistered=True,
            )
            for candidate_id in inferential_ids
        )
        diagnostics = tuple(
            DescriptiveDiagnosticScores(
                candidate_id=candidate_id,
                seeds=sealed_value.evaluation_seeds,
                scores=score_index[candidate_id].scores,
                exclusion_reasons=(
                    sealed_value.candidate_index[candidate_id].pairing.exclusion_reasons
                ),
            )
            for candidate_id in (sealed_value.evaluation_panel.fixed_descriptive_candidate_ids)
        )
        contract = MatchedComparisonContract(
            methods=methods,
            primary_comparison=comparisons[0],
            secondary_comparisons=comparisons[1:],
            fixed_descriptive_diagnostics=diagnostics,
            bootstrap=BootstrapSpec(
                resamples=plan.primary.resamples,
                seed=plan.primary.seed,
                confidence=plan.primary.confidence,
            ),
            permutation=PermutationSpec(
                monte_carlo_resamples=(plan.secondary.monte_carlo_resamples),
                seed=plan.secondary.seed,
                familywise_alpha=(plan.secondary.familywise_alpha),
            ),
            primary_margin=plan.primary.primary_margin,
            primary_analysis_implementation_sha256=(plan.primary.implementation_sha256),
            secondary_analysis_implementation_sha256=(plan.secondary.implementation_sha256),
            metric_direction=plan.metric_direction,
        )
    except MatchedStatisticsError as exc:
        raise ForagerMatchedEvidenceError(
            f"protocol/evidence cannot form a statistics contract: {exc}"
        ) from exc
    return contract, transition, evidence


__all__ = [
    "AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION",
    "AuthenticatedEvidenceBindings",
    "CandidateScoreEvidence",
    "ForagerMatchedEvidenceError",
    "MATCHED_EXECUTION_CLOSURE_SCHEMA_VERSION",
    "MATCHED_METRIC_BINDING_SCHEMA_VERSION",
    "MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION",
    "MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256",
    "MATCHED_SELECTION_GROUP_EVIDENCE_SCHEMA_VERSION",
    "MATCHED_SELECTION_REPORT_SCHEMA_VERSION",
    "MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256",
    "MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION",
    "MatchedScoreEvidence",
    "SeedScoreRecord",
    "SelectionComputation",
    "build_statistics_contract",
    "compute_open_selection",
    "decode_strict_json",
    "load_forager_matched_selection_result",
    "load_matched_score_evidence",
    "load_matched_selection_report",
    "matched_execution_closure_sha256",
    "matched_selection_bootstrap_rng_implementation_descriptor",
    "matched_selection_statistic_implementation_descriptor",
    "matched_verification_subject_sha256",
    "parse_matched_score_evidence",
    "parse_matched_selection_report",
    "validate_score_evidence_against_protocol",
]
