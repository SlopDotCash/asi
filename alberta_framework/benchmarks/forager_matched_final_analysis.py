"""Authenticated final statistics closure for the matched Forager campaign.

Creation fully replays the exact seal and completed 6-by-30 evaluation around two caller-supplied
resolvers.  The open subject is resolved first; the evaluation resolver is the last external
callback, after which an internal content replay and source/runtime capture precede deterministic
statistics and atomic publication.  Subject equality does not establish resolver legitimacy.
Persisted bindings are cache content only: loading this bundle never authenticates them and never
confers promotion, SOTA, or unrestricted performance-claim authority.

The published tree is campaign-root-independent for scalar-statistics and digest-closure replay
only under the matching live enumerated source set and runtime.  It contains an exact seal subtree,
an exact snapshot of every immutable and final evaluation artifact, and the statistics
contract/result.  It intentionally does not duplicate raw cell archives, reward traces,
qualification source archives, or executable source bytes; recomputing workloads or scores still
requires the original qualified campaign roots.  Its interpretation is limited to the four
tuning-selected inferential arms plus two fixed descriptive arms and the three frozen
Alberta-vs-external calculations.  It defines no ranking among those six arms or the full
registered universe.  Fixed descriptive arms never enter inferential tests, and the secondary
sign-flip/Holm output is nonconfirmatory sensitivity analysis because the frozen protocol does not
state sign exchangeability.  The primary percentile-bootstrap endpoint is likewise a frozen
resampling summary, not an established population confidence bound: the protocol states no seed-
superpopulation model or bootstrap regularity assumptions.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

import numpy as np

import alberta_framework as alberta_package
import alberta_framework.benchmarks as benchmarks_package
import alberta_framework.core as core_package
from alberta_framework.benchmarks import (
    causal_map_forager,
    forager,
    forager_rng_parity,
    forager_rtu_ppo_rng_isolation,
)
from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from alberta_framework.benchmarks import forager_matched_statistics as statistics
from alberta_framework.core import horde as core_horde
from alberta_framework.core import horde_actor_critic as core_horde_actor_critic
from alberta_framework.core import initializers as core_initializers
from alberta_framework.core import multi_head_learner as core_multi_head_learner
from alberta_framework.core import normalizers as core_normalizers
from alberta_framework.core import optimizers as core_optimizers
from alberta_framework.core import recurrent_trace_actor_critic
from alberta_framework.core import types as core_types

MATCHED_FINAL_ANALYSIS_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_final_analysis_manifest.v3"
)
MATCHED_FINAL_ANALYSIS_RUNTIME_SOURCE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_final_analysis_runtime_source.v2"
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_CANDIDATES: Final = 6
_EXPECTED_SEEDS: Final = 30
_EXPECTED_CELLS: Final = 180
_EXPECTED_HYPOTHESIS_IDS: Final = (
    "alberta_vs_external",
    "alberta_vs_external_rank2",
    "alberta_vs_external_rank3",
)
_MAX_ANALYSIS_SOURCE_BYTES: Final = 4 * 1024 * 1024
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_FINALIZER_SOURCE_PATH: Final = Path(__file__).resolve()
_STATISTICS_SOURCE_PATH: Final = Path(statistics.__file__).resolve()
_ANALYSIS_SOURCE_PATHS: Final = (
    (
        "alberta_package_initializer",
        Path(alberta_package.__file__).resolve(),
        "alberta_framework/__init__.py",
    ),
    (
        "benchmarks_package_initializer",
        Path(benchmarks_package.__file__).resolve(),
        "alberta_framework/benchmarks/__init__.py",
    ),
    (
        "core_package_initializer",
        Path(core_package.__file__).resolve(),
        "alberta_framework/core/__init__.py",
    ),
    (
        "finalizer",
        _FINALIZER_SOURCE_PATH,
        "alberta_framework/benchmarks/forager_matched_final_analysis.py",
    ),
    (
        "statistics",
        _STATISTICS_SOURCE_PATH,
        "alberta_framework/benchmarks/forager_matched_statistics.py",
    ),
    (
        "evidence",
        Path(evidence.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_evidence.py",
    ),
    (
        "protocol",
        Path(protocol.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_protocol.py",
    ),
    (
        "seal",
        Path(seal.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_seal.py",
    ),
    (
        "evaluation",
        Path(evaluation.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_evaluation_campaign.py",
    ),
    (
        "campaign",
        Path(campaign.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_campaign.py",
    ),
    (
        "executor",
        Path(executor.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_executor.py",
    ),
    (
        "qualification",
        Path(qualification.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_qualification.py",
    ),
    (
        "candidate_universe",
        Path(universe.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_candidate_universe.py",
    ),
    (
        "open_protocol",
        Path(open_protocol.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_open_protocol.py",
    ),
    (
        "sealed_campaign",
        Path(sealed_campaign.__file__).resolve(),
        "alberta_framework/benchmarks/forager_matched_sealed_evaluation_campaign.py",
    ),
    (
        "forager_rng_parity",
        Path(forager_rng_parity.__file__).resolve(),
        "alberta_framework/benchmarks/forager_rng_parity.py",
    ),
    (
        "forager_rtu_ppo_rng_isolation",
        Path(forager_rtu_ppo_rng_isolation.__file__).resolve(),
        "alberta_framework/benchmarks/forager_rtu_ppo_rng_isolation.py",
    ),
    (
        "causal_map_forager",
        Path(causal_map_forager.__file__).resolve(),
        "alberta_framework/benchmarks/causal_map_forager.py",
    ),
    (
        "forager",
        Path(forager.__file__).resolve(),
        "alberta_framework/benchmarks/forager.py",
    ),
    (
        "recurrent_trace_actor_critic",
        Path(recurrent_trace_actor_critic.__file__).resolve(),
        "alberta_framework/core/recurrent_trace_actor_critic.py",
    ),
    (
        "core_initializers",
        Path(core_initializers.__file__).resolve(),
        "alberta_framework/core/initializers.py",
    ),
    (
        "core_horde",
        Path(core_horde.__file__).resolve(),
        "alberta_framework/core/horde.py",
    ),
    (
        "core_horde_actor_critic",
        Path(core_horde_actor_critic.__file__).resolve(),
        "alberta_framework/core/horde_actor_critic.py",
    ),
    (
        "core_optimizers",
        Path(core_optimizers.__file__).resolve(),
        "alberta_framework/core/optimizers.py",
    ),
    (
        "core_types",
        Path(core_types.__file__).resolve(),
        "alberta_framework/core/types.py",
    ),
    (
        "core_normalizers",
        Path(core_normalizers.__file__).resolve(),
        "alberta_framework/core/normalizers.py",
    ),
    (
        "core_multi_head_learner",
        Path(core_multi_head_learner.__file__).resolve(),
        "alberta_framework/core/multi_head_learner.py",
    ),
)

_SEAL_ARTIFACTS: Final = (
    "open-protocol.json",
    "open-execution-plan.json",
    "open-live-runtime.json",
    "open-execution-receipt-index.json",
    "open-score-evidence.json",
    "open-verification-request.json",
    "open-completion-summary.json",
    "open-authenticated-bindings-cache.json",
    "selection-result.json",
    "selection-report.json",
    "sealed-protocol.json",
    "seal.json",
)
_EVALUATION_ARTIFACTS: Final = (
    *sealed_campaign._IMMUTABLE_ARTIFACTS,
    *campaign._FINAL_ARTIFACTS,
)
_ANALYSIS_ARTIFACTS: Final = (
    "analysis-runtime-source.json",
    "evaluation-authenticated-bindings-cache.json",
    "statistics-contract.json",
    "statistics-result.json",
)
_ROOT_FILES: Final = ("manifest.json",)


class ForagerMatchedFinalAnalysisError(ValueError):
    """Final-analysis inputs, content, authority pins, or publication failed closed."""


class PublishedFinalAnalysisUncertainError(ForagerMatchedFinalAnalysisError):
    """Publication occurred, but durability or final replay could not be established."""

    def __init__(self, destination: Path, detail: str) -> None:
        self.destination = destination
        super().__init__(f"final analysis published at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class ContentVerifiedFinalAnalysisBundle:
    """Strictly replayed content with no fresh-authentication authority."""

    output_root: Path
    manifest: Mapping[str, Any]
    seal_content: seal.ContentVerifiedSealBundle
    evaluation_score_evidence: evidence.MatchedScoreEvidence
    evaluation_verification_request: executor.VerificationRequest
    open_bindings_cache: evidence.AuthenticatedEvidenceBindings
    evaluation_bindings_cache: evidence.AuthenticatedEvidenceBindings
    analysis_runtime_source: Mapping[str, Any]
    contract: statistics.MatchedComparisonContract
    result: statistics.MatchedComparisonResult

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise ForagerMatchedFinalAnalysisError("output_root must be a Path")
        if not isinstance(self.manifest, Mapping):
            raise ForagerMatchedFinalAnalysisError("manifest must be a Mapping")
        if not isinstance(self.seal_content, seal.ContentVerifiedSealBundle):
            raise ForagerMatchedFinalAnalysisError(
                "seal_content must be a ContentVerifiedSealBundle"
            )
        if not isinstance(self.evaluation_score_evidence, evidence.MatchedScoreEvidence):
            raise ForagerMatchedFinalAnalysisError(
                "evaluation_score_evidence must be a MatchedScoreEvidence"
            )
        if not isinstance(
            self.evaluation_verification_request, executor.VerificationRequest
        ):
            raise ForagerMatchedFinalAnalysisError(
                "evaluation_verification_request must be a VerificationRequest"
            )
        if not isinstance(
            self.open_bindings_cache, evidence.AuthenticatedEvidenceBindings
        ):
            raise ForagerMatchedFinalAnalysisError(
                "open_bindings_cache must be an AuthenticatedEvidenceBindings"
            )
        if not isinstance(
            self.evaluation_bindings_cache, evidence.AuthenticatedEvidenceBindings
        ):
            raise ForagerMatchedFinalAnalysisError(
                "evaluation_bindings_cache must be an AuthenticatedEvidenceBindings"
            )
        if not isinstance(self.analysis_runtime_source, Mapping):
            raise ForagerMatchedFinalAnalysisError(
                "analysis_runtime_source must be a Mapping"
            )
        if not isinstance(self.contract, statistics.MatchedComparisonContract):
            raise ForagerMatchedFinalAnalysisError(
                "contract must be a MatchedComparisonContract"
            )
        if not isinstance(self.result, statistics.MatchedComparisonResult):
            raise ForagerMatchedFinalAnalysisError(
                "result must be a MatchedComparisonResult"
            )


@dataclass(frozen=True, slots=True)
class FreshFinalAnalysisBindings:
    """Fresh resolver results as plain data, not a reusable authority capability."""

    open_bindings: evidence.AuthenticatedEvidenceBindings
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings

    def __post_init__(self) -> None:
        if not isinstance(self.open_bindings, evidence.AuthenticatedEvidenceBindings):
            raise ForagerMatchedFinalAnalysisError(
                "open_bindings must be an AuthenticatedEvidenceBindings"
            )
        if not isinstance(
            self.evaluation_bindings, evidence.AuthenticatedEvidenceBindings
        ):
            raise ForagerMatchedFinalAnalysisError(
                "evaluation_bindings must be an AuthenticatedEvidenceBindings"
            )


@dataclass(frozen=True, slots=True)
class _CreationInputs:
    seal_content: seal.ContentVerifiedSealBundle
    seal_artifacts: Mapping[str, bytes]
    completed: campaign.CompletedCampaignBundle
    evaluation_artifacts: Mapping[str, bytes]


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedFinalAnalysisError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ForagerMatchedFinalAnalysisError(f"{label} must be a non-empty string")
    return value


def _require_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        bound = f"..{maximum}" if maximum is not None else " or greater"
        raise ForagerMatchedFinalAnalysisError(
            f"{label} must be an integer in {minimum}{bound}"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ForagerMatchedFinalAnalysisError(f"{label} fields drifted")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return seal.canonical_json_bytes(value)


def _json_exact_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical({"value": left}) == _canonical({"value": right})
    except (OverflowError, RecursionError, TypeError, ValueError):
        return False


def _analysis_source_records() -> dict[str, dict[str, Any]]:
    roles = tuple(role for role, _path, _relative in _ANALYSIS_SOURCE_PATHS)
    paths = tuple(path for _role, path, _relative in _ANALYSIS_SOURCE_PATHS)
    if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
        raise ForagerMatchedFinalAnalysisError(
            "analysis source set repeats a role or resolved path"
        )
    records: dict[str, dict[str, Any]] = {}
    for role, path, relative in _ANALYSIS_SOURCE_PATHS:
        try:
            expected_path = (_REPOSITORY_ROOT / relative).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ForagerMatchedFinalAnalysisError(
                f"cannot resolve repository source path for {role}"
            ) from exc
        if path != expected_path:
            raise ForagerMatchedFinalAnalysisError(
                f"imported {role} module is outside its labeled repository path"
            )
        try:
            raw = executor._read_stable_file(
                path,
                f"{role} analysis source",
                maximum=_MAX_ANALYSIS_SOURCE_BYTES,
            )
        except (OSError, ValueError) as exc:
            raise ForagerMatchedFinalAnalysisError(
                f"cannot capture exact {role} analysis source"
            ) from exc
        records[role] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    return records


_ANALYSIS_SOURCE_RECORDS_AT_IMPORT: Final = _analysis_source_records()


def _analysis_runtime_source_identity() -> dict[str, Any]:
    sources = _analysis_source_records()
    if sources != _ANALYSIS_SOURCE_RECORDS_AT_IMPORT:
        drifted = sorted(
            set(sources) | set(_ANALYSIS_SOURCE_RECORDS_AT_IMPORT)
        )
        drifted = [
            role
            for role in drifted
            if sources.get(role) != _ANALYSIS_SOURCE_RECORDS_AT_IMPORT.get(role)
        ]
        raise ForagerMatchedFinalAnalysisError(
            "analysis source changed after the finalizer module was imported: "
            + ", ".join(drifted)
        )
    try:
        uname = os.uname()
        platform_identity = {
            "machine": uname.machine,
            "release": uname.release,
            "system": uname.sysname,
        }
    except AttributeError:
        platform_identity = {
            "machine": "unavailable",
            "release": "unavailable",
            "system": os.name,
        }
    body: dict[str, Any] = {
        "schema_version": MATCHED_FINAL_ANALYSIS_RUNTIME_SOURCE_SCHEMA_VERSION,
        "classification": (
            "explicit_final_analysis_source_set_and_versioned_runtime_identity"
        ),
        "authentication_state": "content_only_unendorsed_reproducibility_identity",
        "hash_algorithm": "sha256",
        "hash_scope": "exact_raw_single_link_regular_file_bytes",
        "source_set_scope": (
            "manually_enumerated_replay_related_files_and_selected_package_initializers_"
            "completeness_not_established"
        ),
        "mechanically_complete_transitive_import_closure": False,
        "package_wide_source_closure_captured": False,
        "promotion_authorized": False,
        "sources": sources,
        "runtime": {
            "identity_scope": (
                "python_numpy_versions_and_platform_not_binary_or_shared_library_closure"
            ),
            "bit_exact_runtime_closure": False,
            "native_shared_library_closure_captured": False,
            "python": {
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
                "hexversion": sys.hexversion,
                "cache_tag": sys.implementation.cache_tag,
                "byteorder": sys.byteorder,
            },
            "numpy": {"version": np.__version__},
            "platform": {
                "os_name": os.name,
                "sys_platform": sys.platform,
                **platform_identity,
            },
        },
        "statistics_contract_schema": statistics.CONTRACT_SCHEMA,
        "statistics_result_schema": statistics.RESULT_SCHEMA,
        "primary_implementation_sha256": (statistics.PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256),
        "secondary_implementation_sha256": (
            statistics.SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
        ),
    }
    return {**body, "payload_sha256": campaign._canonical_sha256(body)}


def _parse_analysis_runtime_source(raw: bytes) -> Mapping[str, Any]:
    payload = seal._decode_canonical(raw, "analysis runtime/source identity")
    expected = _analysis_runtime_source_identity()
    if not _json_exact_equal(payload, expected):
        raise ForagerMatchedFinalAnalysisError(
            "analysis runtime/source identity differs from the enumerated source set or runtime"
        )
    return cast(Mapping[str, Any], seal._freeze_json(payload))


def _pair_names(names: tuple[str, ...]) -> set[str]:
    return {*(name for name in names), *(f"{name}.sha256" for name in names)}


def _directory_inventory(
    opened: seal._OpenDirectory,
    *,
    expected_files: tuple[str, ...],
    expected_directories: tuple[str, ...] = (),
    label: str,
) -> dict[str, tuple[int, ...]]:
    expected = _pair_names(expected_files) | set(expected_directories)
    inventory: dict[str, tuple[int, ...]] = {}
    try:
        iterator = os.scandir(opened.descriptor)
    except OSError as exc:
        raise ForagerMatchedFinalAnalysisError(f"cannot enumerate {label}") from exc
    with iterator:
        for entry in iterator:
            if len(inventory) >= len(expected):
                raise ForagerMatchedFinalAnalysisError(f"{label} exceeds its entry bound")
            if entry.name in inventory:
                raise ForagerMatchedFinalAnalysisError(f"{label} repeats an entry")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedFinalAnalysisError(f"cannot inspect {label} entry") from exc
            if entry.name in expected_directories:
                safe = stat.S_ISDIR(metadata.st_mode)
            else:
                safe = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            if not safe:
                raise ForagerMatchedFinalAnalysisError(f"{label} contains an unsafe entry")
            inventory[entry.name] = seal._stat_identity(metadata)
    if set(inventory) != expected:
        raise ForagerMatchedFinalAnalysisError(
            f"{label} inventory differs "
            f"(missing={sorted(expected - set(inventory))}, "
            f"extra={sorted(set(inventory) - expected)})"
        )
    return inventory


def _read_pairs(
    opened: seal._OpenDirectory,
    names: tuple[str, ...],
    label: str,
) -> tuple[dict[str, bytes], dict[str, str]]:
    raw_by_name: dict[str, bytes] = {}
    digest_by_name: dict[str, str] = {}
    for name in names:
        raw, digest = seal._load_pair_at(opened, name, f"{label} {name}")
        raw_by_name[name] = raw
        digest_by_name[name] = digest
    return raw_by_name, digest_by_name


def _parse_bindings(
    raw: bytes,
    *,
    expected_stage: Literal["open_tuning", "sealed_evaluation"],
) -> evidence.AuthenticatedEvidenceBindings:
    try:
        value = seal._decode_canonical(raw, f"{expected_stage} bindings cache")
    except seal.ForagerMatchedSealError as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"{expected_stage} bindings cache is invalid: {exc}"
        ) from exc
    expected_keys = {
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
    }
    if set(value) != expected_keys:
        raise ForagerMatchedFinalAnalysisError("bindings cache shape drifted")
    if (
        value["schema_version"] != evidence.AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION
        or value["stage"] != expected_stage
    ):
        raise ForagerMatchedFinalAnalysisError("bindings cache schema/stage drifted")
    try:
        result = evidence.AuthenticatedEvidenceBindings(
            stage=expected_stage,
            protocol_sha256=_require_sha256(
                value["protocol_sha256"],
                "bindings protocol",
            ),
            score_evidence_sha256=_require_sha256(
                value["score_evidence_sha256"], "bindings score evidence"
            ),
            source_manifest_sha256=_require_sha256(
                value["source_manifest_sha256"], "bindings source manifest"
            ),
            executor_manifest_sha256=_require_sha256(
                value["executor_manifest_sha256"], "bindings executor manifest"
            ),
            qualification_manifest_sha256=_require_sha256(
                value["qualification_manifest_sha256"],
                "bindings qualification manifest",
            ),
            execution_closure_sha256=_require_sha256(
                value["execution_closure_sha256"], "bindings execution closure"
            ),
            trust_anchor_identity=_require_identifier(
                value["trust_anchor_identity"], "bindings trust anchor"
            ),
            verification_subject_sha256=_require_sha256(
                value["verification_subject_sha256"], "bindings verification subject"
            ),
            verification_receipt_sha256=_require_sha256(
                value["verification_receipt_sha256"], "bindings verification receipt"
            ),
        )
    except evidence.ForagerMatchedEvidenceError as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"{expected_stage} bindings cache evidence closure is invalid: {exc}"
        ) from exc
    if result.to_dict() != value:
        raise ForagerMatchedFinalAnalysisError("bindings cache canonical replay drifted")
    return result


def _expected_qualification_authority_boundary() -> dict[str, Any]:
    return {
        "endorsement_created": False,
        "endorsements_at_seal": 0,
        "gpu_qualified": False,
        "performance_claim": False,
        "seed_class": "open_development",
        "trust_profile_created": False,
        "trust_profiles_at_seal": 0,
    }


def _assert_request_authority_boundary(
    request: executor.VerificationRequest,
    *,
    label: str,
) -> None:
    if not _json_exact_equal(
        seal._plain(request.qualification_authority_boundary),
        _expected_qualification_authority_boundary(),
    ):
        raise ForagerMatchedFinalAnalysisError(
            f"{label} verification request qualification authority boundary drifted"
        )


def _assert_request_bindings(
    request: executor.VerificationRequest,
    bindings: evidence.AuthenticatedEvidenceBindings,
) -> None:
    _assert_request_authority_boundary(request, label=request.stage)
    request_value = request.to_dict()
    binding_value = bindings.to_dict()
    fields = (
        "stage",
        "protocol_sha256",
        "score_evidence_sha256",
        "source_manifest_sha256",
        "executor_manifest_sha256",
        "qualification_manifest_sha256",
        "execution_closure_sha256",
        "trust_anchor_identity",
        "verification_subject_sha256",
    )
    drifted = [name for name in fields if request_value[name] != binding_value[name]]
    if drifted:
        raise ForagerMatchedFinalAnalysisError(
            "verification request differs from bindings cache: " + ", ".join(drifted)
        )


def _validate_request_pin(
    request: executor.VerificationRequest,
    *,
    expected_trust_anchor_identity: str,
    expected_verification_subject_sha256: str,
    label: str,
) -> None:
    _assert_request_authority_boundary(request, label=label)
    anchor = _require_identifier(
        expected_trust_anchor_identity,
        f"expected {label} trust anchor identity",
    )
    subject = _require_sha256(
        expected_verification_subject_sha256,
        f"expected {label} verification subject",
    )
    if request.trust_anchor_identity != anchor:
        raise ForagerMatchedFinalAnalysisError(
            f"{label} request differs from caller-pinned trust anchor"
        )
    if request.verification_subject_sha256 != subject:
        raise ForagerMatchedFinalAnalysisError(
            f"{label} request differs from caller-pinned verification subject"
        )


def _load_seal_snapshot(
    seal_root: Path,
) -> tuple[seal.ContentVerifiedSealBundle, dict[str, bytes]]:
    opened = seal._open_stable_directory(seal_root, "final-analysis seal input")
    try:
        initial = _directory_inventory(
            opened,
            expected_files=_SEAL_ARTIFACTS,
            label="final-analysis seal input",
        )
        content = seal._load_forager_matched_seal_bundle_from_open_root(
            opened,
            initial,
        )
        artifacts, _digests = _read_pairs(
            opened,
            _SEAL_ARTIFACTS,
            "final-analysis seal input",
        )
        if (
            _directory_inventory(
                opened,
                expected_files=_SEAL_ARTIFACTS,
                label="final-analysis seal input",
            )
            != initial
        ):
            raise ForagerMatchedFinalAnalysisError("seal input changed during snapshot")
        seal._assert_open_directory_path(opened, "final-analysis seal input")
        return content, artifacts
    finally:
        os.close(opened.descriptor)


def _snapshot_evaluation_root(root: Path) -> dict[str, bytes]:
    opened = seal._open_stable_directory(root, "completed evaluation input")
    try:
        initial = _directory_inventory(
            opened,
            expected_files=_EVALUATION_ARTIFACTS,
            expected_directories=("runs", "completions"),
            label="completed evaluation input",
        )
        artifacts, _digests = _read_pairs(
            opened,
            _EVALUATION_ARTIFACTS,
            "completed evaluation input",
        )
        if (
            _directory_inventory(
                opened,
                expected_files=_EVALUATION_ARTIFACTS,
                expected_directories=("runs", "completions"),
                label="completed evaluation input",
            )
            != initial
        ):
            raise ForagerMatchedFinalAnalysisError(
                "completed evaluation input changed during snapshot"
            )
        seal._assert_open_directory_path(opened, "completed evaluation input")
        return artifacts
    finally:
        os.close(opened.descriptor)


def _validate_exact_completed_panel(
    seal_content: seal.ContentVerifiedSealBundle,
    completed: campaign.CompletedCampaignBundle,
) -> None:
    transition = protocol.validate_sealed_protocol_transition(
        seal_content.open_protocol,
        seal_content.sealed_protocol,
        seal_content.selection_result,
        seal_content.selection_result.selection_result_sha256,
    )
    seal_open_campaign = seal_content.manifest.get("open_campaign")
    if not isinstance(seal_open_campaign, Mapping):
        raise ForagerMatchedFinalAnalysisError(
            "seal manifest open-campaign closure must be an object"
        )
    qualification_digests = {
        seal_content.open_score_evidence.qualification_manifest_sha256,
        seal_content.open_verification_request.qualification_manifest_sha256,
        seal_open_campaign.get("qualification_manifest_sha256"),
        completed.plan.qualification_manifest_sha256,
        completed.plan.executor_manifest.get("qualification_manifest_sha256"),
        completed.score_evidence.qualification_manifest_sha256,
        completed.verification_request.qualification_manifest_sha256,
        completed.completion_summary.get("qualification_manifest_sha256"),
    }
    if (
        len(qualification_digests) != 1
        or
        completed.protocol != seal_content.sealed_protocol
        or completed.protocol.stage != "sealed_evaluation"
        or completed.candidate_ids != transition.evaluation_candidate_ids
        or len(completed.candidate_ids) != _EXPECTED_CANDIDATES
        or completed.active_seeds != completed.protocol.evaluation_seeds
        or len(completed.active_seeds) != _EXPECTED_SEEDS
        or len(completed.seed_artifacts) != _EXPECTED_CANDIDATES
        or sum(len(records) for records in completed.seed_artifacts.values()) != _EXPECTED_CELLS
        or completed.verification_request.stage != "sealed_evaluation"
        or completed.completion_summary.get("promotion_authorized") is not False
        or completed.completion_summary.get("performance_claim") is not False
    ):
        raise ForagerMatchedFinalAnalysisError(
            "completed evaluation is not the exact nonpromoting 6-by-30 sealed block"
        )
    if set(completed.final_file_sha256) != set(campaign._FINAL_ARTIFACTS):
        raise ForagerMatchedFinalAnalysisError(
            "completed evaluation final artifact closure is incomplete"
        )


def _assert_snapshot_matches_completed(
    inputs: _CreationInputs,
) -> None:
    completed = inputs.completed
    evaluation_artifacts = inputs.evaluation_artifacts
    exact: dict[str, bytes] = {
        "sealed-protocol.json": completed.protocol.canonical_bytes,
        "sealed-transition.json": _canonical(inputs.seal_content.sealed_transition),
        "seal-manifest.json": _canonical(inputs.seal_content.manifest),
        "candidate-universe.json": _canonical(
            universe.matched_current_candidate_universe_descriptor()
        ),
        "execution-plan.json": completed.plan.canonical_bytes,
        "source-manifest.json": executor.canonical_json_bytes(completed.plan.source_manifest),
        "executor-manifest.json": executor.canonical_json_bytes(completed.plan.executor_manifest),
        "execution-schedule.json": executor.canonical_json_bytes(completed.schedule),
        "live-runtime.json": executor.canonical_json_bytes(completed.live_runtime.unsigned_dict),
        "execution-receipt-index.json": (completed.execution_receipt_index.canonical_bytes),
        "score-evidence.json": completed.score_evidence.canonical_bytes,
        "verification-request.json": completed.verification_request.canonical_bytes,
        "completion-summary.json": _canonical(completed.completion_summary),
    }
    drifted = [name for name, expected in exact.items() if evaluation_artifacts[name] != expected]
    if drifted:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation snapshot differs from replayed completion: " + ", ".join(drifted)
        )
    for name, digest in completed.final_file_sha256.items():
        if hashlib.sha256(evaluation_artifacts[name]).hexdigest() != digest:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation snapshot final digest differs for {name}"
            )


def _load_creation_inputs(
    qualification_root: Path,
    seal_root: Path,
    evaluation_campaign_root: Path,
    *,
    runtime: str | Path,
    runner: executor.ProcessRunner | None,
) -> _CreationInputs:
    if not all(
        isinstance(path, Path) for path in (qualification_root, seal_root, evaluation_campaign_root)
    ):
        raise TypeError("qualification, seal, and evaluation roots must be Paths")
    seal_content, seal_artifacts = _load_seal_snapshot(seal_root)
    try:
        completed = sealed_campaign.load_completed_sealed_evaluation_campaign_content(
            qualification_root,
            seal_root,
            evaluation_campaign_root,
            runtime=runtime,
            runner=runner,
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"completed sealed evaluation is invalid: {exc}"
        ) from exc
    _validate_exact_completed_panel(seal_content, completed)
    evaluation_artifacts = _snapshot_evaluation_root(completed.output_root)
    inputs = _CreationInputs(
        seal_content=seal_content,
        seal_artifacts=seal_artifacts,
        completed=completed,
        evaluation_artifacts=evaluation_artifacts,
    )
    _assert_snapshot_matches_completed(inputs)
    return inputs


def _frozen_live_runtime_replay_runner(
    live: executor.LiveRuntimeIdentity,
) -> executor.ProcessRunner:
    executable = live.executable.as_posix()
    version_command = (executable, "version", "--format={{json .}}")
    inspection_command = (
        executable,
        "image",
        "inspect",
        "--format={{json .}}",
        f"sha256:{executor.QUALIFIED_IMAGE_SHA256}",
    )
    version_bytes = executor.canonical_json_bytes(live.version)
    inspection_bytes = executor.canonical_json_bytes(live.image_inspection)

    def replay(command: Sequence[str]) -> executor.ProcessResult:
        exact = tuple(command)
        if exact == version_command:
            return executor.ProcessResult(0, version_bytes, b"")
        if exact == inspection_command:
            return executor.ProcessResult(0, inspection_bytes, b"")
        raise ForagerMatchedFinalAnalysisError(
            "internal live-runtime replay received an unexpected command"
        )

    return replay


def _assert_same_creation_inputs(
    expected: _CreationInputs,
    current: _CreationInputs,
    label: str,
) -> None:
    if (
        dict(current.seal_artifacts) != dict(expected.seal_artifacts)
        or dict(current.evaluation_artifacts) != dict(expected.evaluation_artifacts)
        or current.seal_content.manifest != expected.seal_content.manifest
        or current.completed.protocol != expected.completed.protocol
        or current.completed.candidate_ids != expected.completed.candidate_ids
        or current.completed.active_seeds != expected.completed.active_seeds
        or current.completed.final_file_sha256 != expected.completed.final_file_sha256
        or current.completed.verification_request != expected.completed.verification_request
    ):
        raise ForagerMatchedFinalAnalysisError(f"final-analysis inputs changed {label}")


def _validate_evaluation_receipt_index(
    payload: Mapping[str, Any],
    sealed_protocol: protocol.ForagerMatchedProtocol,
    scores: evidence.MatchedScoreEvidence,
    *,
    plan_sha256: str,
    live_runtime_sha256: str,
) -> str:
    declared = _require_sha256(
        payload.get("payload_sha256"),
        "evaluation receipt-index payload",
    )
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    if campaign._canonical_sha256(unsigned) != declared:
        raise ForagerMatchedFinalAnalysisError("evaluation receipt-index self-hash differs")
    candidate_order = tuple(item.candidate_id for item in scores.candidate_scores)
    required = {
        "schema_version": executor.MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION,
        "classification": "content_complete_execution_receipt_preimages",
        "authentication_state": ("content_only_unendorsed_external_verifier_required"),
        "promotion_authorized": False,
        "external_verification_required": True,
        "stage": "sealed_evaluation",
        "protocol_sha256": sealed_protocol.protocol_sha256,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "live_runtime_identity_sha256": live_runtime_sha256,
        "active_seeds": list(sealed_protocol.active_seeds),
        "horizon": sealed_protocol.horizon,
        "candidate_order": list(candidate_order),
    }
    _require_exact_keys(
        payload,
        {*required, "execution_receipts", "payload_sha256"},
        "evaluation receipt-index",
    )
    drifted = [
        name
        for name, expected in required.items()
        if not _json_exact_equal(payload.get(name), expected)
    ]
    raw_receipts = payload.get("execution_receipts")
    if drifted or type(raw_receipts) is not list or len(raw_receipts) != len(candidate_order):
        raise ForagerMatchedFinalAnalysisError("evaluation receipt-index header closure drifted")
    for index, (raw_item, score) in enumerate(
        zip(raw_receipts, scores.candidate_scores, strict=True)
    ):
        if type(raw_item) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                "evaluation receipt-index contains a non-object receipt"
            )
        item = cast(dict[str, Any], raw_item)
        _require_exact_keys(
            item,
            {"candidate_id", "execution_receipt_sha256", "receipt_payload"},
            f"evaluation receipt-index item {index}",
        )
        receipt = item.get("receipt_payload")
        if type(receipt) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                "evaluation receipt-index omits a receipt preimage"
            )
        receipt_value = cast(dict[str, Any], receipt)
        expected_seed_artifacts = [
            {
                "seed": record.seed,
                "raw_artifact_sha256": record.raw_artifact_sha256,
                "reward_trace_sha256": record.reward_trace_sha256,
                "scoring_record_sha256": record.scoring_record_sha256,
            }
            for record in score.records
        ]
        expected_receipt = {
            "schema_version": executor.MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "candidate_id": score.candidate_id,
            "stage": "sealed_evaluation",
            "protocol_sha256": sealed_protocol.protocol_sha256,
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
        receipt_sha256 = campaign._canonical_sha256(receipt_value)
        if (
            item.get("candidate_id") != score.candidate_id
            or item.get("execution_receipt_sha256") != receipt_sha256
            or score.execution_receipt_sha256 != receipt_sha256
            or not _json_exact_equal(receipt_value, expected_receipt)
        ):
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation receipt {index} differs from score evidence"
            )
    return declared


def _validate_inventory_payload(payload: Any, *, label: str) -> str:
    if type(payload) is not dict:
        raise ForagerMatchedFinalAnalysisError(f"{label} must be an object")
    value = cast(dict[str, Any], payload)
    _require_exact_keys(value, {"schema_version", "files"}, label)
    files = value.get("files")
    if (
        value.get("schema_version") != executor.MATCHED_SOURCE_INVENTORY_SCHEMA_VERSION
        or type(files) is not list
        or not 0 < len(files) <= executor._MAX_SOURCE_FILES
    ):
        raise ForagerMatchedFinalAnalysisError(f"{label} header drifted")
    paths: list[str] = []
    total_size = 0
    for index, raw_record in enumerate(files):
        if type(raw_record) is not dict:
            raise ForagerMatchedFinalAnalysisError(f"{label} file {index} must be an object")
        record = cast(dict[str, Any], raw_record)
        _require_exact_keys(
            record,
            {"path", "sha256", "size_bytes", "mode"},
            f"{label} file {index}",
        )
        path = record.get("path")
        if type(path) is not str or not 0 < len(path) <= 512:
            raise ForagerMatchedFinalAnalysisError(
                f"{label} file {index} path must be a bounded string"
            )
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or not relative.parts
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != path
            or "\x00" in path
        ):
            raise ForagerMatchedFinalAnalysisError(f"{label} file {index} path is unsafe")
        _require_sha256(record.get("sha256"), f"{label} file {index} digest")
        size = _require_exact_int(
            record.get("size_bytes"),
            f"{label} file {index} size",
            maximum=executor._MAX_SOURCE_BYTES,
        )
        total_size += size
        if total_size > executor._MAX_SOURCE_BYTES:
            raise ForagerMatchedFinalAnalysisError(f"{label} exceeds its total byte bound")
        _require_exact_int(
            record.get("mode"),
            f"{label} file {index} mode",
            maximum=0o7777,
        )
        paths.append(path)
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")) or len(set(paths)) != len(
        paths
    ):
        raise ForagerMatchedFinalAnalysisError(f"{label} paths are not unique canonical order")
    return qualification._canonical_sha256(value)


def _expected_entrypoint_binding(candidate_id: str) -> dict[str, Any]:
    if type(candidate_id) is not str:
        raise ForagerMatchedFinalAnalysisError("candidate_id must be an exact string")
    if candidate_id in open_protocol.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS:
        source_key = "alberta"
        entrypoint_path = "alberta_framework/benchmarks/_forager_matched_alberta_worker.py"
        invocation_style = "alberta_single_seed_v1"
        result_root = "results"
    else:
        try:
            raw_source, entrypoint_path, invocation_style, result_root, _agent = (
                qualification._EXTERNAL_EXECUTION[candidate_id]
            )
        except KeyError as exc:
            raise ForagerMatchedFinalAnalysisError(
                f"candidate {candidate_id!r} has no frozen entrypoint binding"
            ) from exc
        source_key = str(raw_source)
    return {
        "source_key": source_key,
        "path": entrypoint_path,
        "python_import_root": "." if source_key == "alberta" else "src",
        "invocation_style": invocation_style,
        "result_root": result_root,
        "rng_isolation_patch_sha256": (
            qualification._QUALIFIED_RNG_PATCH_SHA256
            if source_key == "upstream_rng_isolated"
            else None
        ),
    }


def _expected_candidate_command_template(
    sealed_protocol: protocol.ForagerMatchedProtocol,
    candidate: protocol.MatchedCandidate,
    entrypoint: Mapping[str, Any],
) -> list[str]:
    container_import_root = executor._container_source_path(
        cast(str, entrypoint["python_import_root"])
    )
    return [
        "<OCI_RUNTIME>",
        "run",
        *executor._sandbox_options(sealed_protocol),
        "--mount=type=bind,source=<SOURCE_ROOT>,destination=/inputs/source,readonly",
        "--mount=type=bind,source=<CONFIG>,destination=/inputs/configuration.json,readonly",
        "--mount=type=bind,source=<HELPER>,destination=/harness/matched_container.py,readonly",
        "--workdir=/inputs/source",
        f"sha256:{sealed_protocol.runtime.image_sha256}",
        executor.QUALIFIED_PYTHON,
        "-I",
        "-B",
        "-c",
        executor._VERIFIED_SCRIPT_LAUNCHER,
        executor.CONTAINER_HELPER,
        "<HELPER_SHA256>",
        f"--contract={executor.CONTAINER_CONTRACT}",
        "run",
        f"--python={executor.QUALIFIED_PYTHON}",
        f"--source-root={executor.CONTAINER_SOURCE_ROOT}",
        f"--entrypoint={executor.CONTAINER_SOURCE_ROOT}/{entrypoint['path']}",
        f"--python-import-root={container_import_root}",
        f"--config={executor.CONTAINER_CONFIG}",
        f"--source-inventory-sha256={candidate.source.inventory_sha256}",
        f"--configuration-sha256={candidate.configuration.derived_sha256}",
        f"--invocation-style={entrypoint['invocation_style']}",
        f"--result-root={entrypoint['result_root']}",
        "--seed=<ACTIVE_SEED>",
        f"--horizon={sealed_protocol.horizon}",
    ]


def _validate_resource_supplement(
    payload: Any,
    *,
    candidate_id: str,
    horizon: int,
    label: str,
) -> None:
    if type(payload) is not dict:
        raise ForagerMatchedFinalAnalysisError(f"{label} must be an object")
    value = cast(dict[str, Any], payload)
    _require_exact_keys(
        value,
        {
            "fixed_substrate_parameter_count",
            "non_gradient_operations",
            "target_snapshot_parameter_count",
        },
        label,
    )
    _require_exact_int(
        value.get("fixed_substrate_parameter_count"),
        f"{label} fixed substrate count",
    )
    _require_exact_int(
        value.get("target_snapshot_parameter_count"),
        f"{label} target snapshot count",
    )
    operations = value.get("non_gradient_operations")
    if type(operations) is not dict:
        raise ForagerMatchedFinalAnalysisError(f"{label} operations must be an object")
    operation_value = cast(dict[str, Any], operations)
    _require_exact_keys(
        operation_value,
        {
            "causal_nonparametric_transition_updates",
            "redo_recycles",
            "target_snapshot_refreshes",
        },
        f"{label} operations",
    )
    for name, raw_value in operation_value.items():
        _require_exact_int(raw_value, f"{label} {name}")
    expected_causal_updates = (
        horizon if candidate_id in open_protocol.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS else 0
    )
    if operation_value["causal_nonparametric_transition_updates"] != expected_causal_updates:
        raise ForagerMatchedFinalAnalysisError(
            f"{label} causal nonparametric update count drifted"
        )


def _validate_qualification_manifest(
    raw: bytes,
    seal_content: seal.ContentVerifiedSealBundle,
) -> tuple[str, Mapping[str, str]]:
    try:
        decoded = qualification._decode_json(raw, "evaluation qualification manifest")
    except qualification.ForagerMatchedQualificationError as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"evaluation qualification manifest is invalid: {exc}"
        ) from exc
    if type(decoded) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification manifest must be a JSON object"
        )
    payload = cast(dict[str, Any], decoded)
    try:
        canonical = qualification._canonical_json_bytes(payload)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification manifest is not canonical JSON"
        ) from exc
    if canonical != raw:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification manifest differs from its exact UTF-8 canonical bytes"
        )
    qualification_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    _require_exact_keys(
        payload,
        {
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
        },
        "evaluation qualification manifest",
    )
    authority = payload.get("authority")
    reward_boundary = payload.get("reward_blind_boundary")
    if (
        payload.get("schema_version") != qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION
        or payload.get("classification") != "content_only_unendorsed_nonpromoting"
        or payload.get("status") != "structurally_qualified_external_trust_resolution_required"
        or payload.get("promotion_authorized") is not False
        or payload.get("performance_claim") is not False
        or payload.get("external_verification_required") is not True
        or not _json_exact_equal(
            authority,
            {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
            },
        )
        or not _json_exact_equal(
            reward_boundary,
            {
            "qualification_seed": qualification.PUBLIC_QUALIFICATION_SEED,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "tuning_seeds_used": [],
            "evaluation_seeds_used": [],
            "environment_resets": len(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS),
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
            },
        )
        or payload.get("candidate_order") != list(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS)
        or payload.get("open_protocol_sha256") != seal_content.open_protocol.protocol_sha256
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification authority/reward closure drifted"
        )
    frozen_runtime = seal_content.open_protocol.runtime
    expected_runtime_qualification = {
        "image_sha256": frozen_runtime.image_sha256,
        "runtime_profile_sha256": frozen_runtime.runtime_profile_sha256,
        "executor_qualification_receipt_sha256": (
            frozen_runtime.executor_qualification_receipt_sha256
        ),
        "qualification_trust_anchor_identity": (
            frozen_runtime.qualification_trust_anchor_identity
        ),
    }
    if not _json_exact_equal(
        payload.get("runtime_qualification"),
        expected_runtime_qualification,
    ) or not _json_exact_equal(
        expected_runtime_qualification,
        asdict(qualification._runtime_qualification()),
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification runtime identity drifted"
        )
    if not _json_exact_equal(
        payload.get("resource_accounting_semantics"),
        qualification._plain_json(qualification._RESOURCE_ACCOUNTING_SEMANTICS),
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification resource semantics drifted"
        )
    qualification_probe = payload.get("qualification_probe")
    if type(qualification_probe) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification probe binding must be an object"
        )
    probe_value = cast(dict[str, Any], qualification_probe)
    _require_exact_keys(
        probe_value,
        {"source_key", "path", "sha256"},
        "evaluation qualification probe binding",
    )
    if (
        probe_value.get("source_key") != "alberta"
        or probe_value.get("path")
        != "alberta_framework/benchmarks/forager_matched_qualification.py"
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification probe source binding drifted"
        )
    _require_sha256(
        probe_value.get("sha256"),
        "evaluation qualification probe source",
    )
    qualification_inventory_expectations = _validate_executor_qualification_artifacts(
        payload.get("frozen_executor_qualification_artifacts"),
        label="evaluation qualification manifest frozen executor artifacts",
    )

    executor_roots = payload.get("executor_qualification_roots")
    if type(executor_roots) is not dict or set(executor_roots) != {"cpu", "rng_parity"}:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification executor-root set drifted"
        )
    for key, expected_path in (
        ("cpu", "executor-qualification/cpu"),
        ("rng_parity", "executor-qualification/rng-parity"),
    ):
        raw_record = cast(dict[str, Any], executor_roots)[key]
        if type(raw_record) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification executor root {key} must be an object"
            )
        record = cast(dict[str, Any], raw_record)
        _require_exact_keys(
            record,
            {"path", "inventory", "inventory_sha256"},
            f"evaluation qualification executor root {key}",
        )
        inventory_sha256 = _validate_inventory_payload(
            record.get("inventory"),
            label=f"evaluation qualification executor root {key} inventory",
        )
        if (
            record.get("path") != expected_path
            or record.get("inventory_sha256") != inventory_sha256
        ):
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification executor root {key} closure drifted"
            )
        inventory_files = cast(dict[str, Any], record["inventory"])["files"]
        inventory_index = {
            cast(str, item["path"]): item
            for item in cast(list[dict[str, Any]], inventory_files)
        }
        for path, (expected_sha256, expected_size) in qualification_inventory_expectations[
            key
        ].items():
            item = inventory_index.get(path)
            if (
                item is None
                or item.get("sha256") != expected_sha256
                or item.get("size_bytes") != expected_size
            ):
                raise ForagerMatchedFinalAnalysisError(
                    f"evaluation qualification executor root {key} critical file drifted"
                )

    raw_sources = payload.get("sources")
    if type(raw_sources) is not dict or set(raw_sources) != {
        "alberta",
        "upstream",
        "upstream_rng_isolated",
    }:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification source set drifted"
        )
    source_values = cast(dict[str, Any], raw_sources)
    expected_source_bindings: dict[str, Mapping[str, Any]] = {}
    for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS:
        source_key = qualification._source_key_for_candidate(candidate_id)
        binding = seal_content.open_protocol.candidate_index[candidate_id].source.to_dict()
        previous = expected_source_bindings.setdefault(source_key, binding)
        if previous != binding:
            raise ForagerMatchedFinalAnalysisError(
                f"frozen candidates disagree about source binding {source_key}"
            )
    inventory_sha256_by_source: dict[str, str] = {}
    for source_key in ("alberta", "upstream", "upstream_rng_isolated"):
        raw_record = source_values[source_key]
        if type(raw_record) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification source {source_key} must be an object"
            )
        record = cast(dict[str, Any], raw_record)
        _require_exact_keys(
            record,
            {
                "binding",
                "root",
                "archive",
                "inventory",
                "snapshot_descriptor_path",
                "patch_path",
            },
            f"evaluation qualification source {source_key}",
        )
        archive = record.get("archive")
        inventory = record.get("inventory")
        if type(archive) is not dict or type(inventory) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification source {source_key} records must be objects"
            )
        archive_value = cast(dict[str, Any], archive)
        inventory_value = cast(dict[str, Any], inventory)
        _require_exact_keys(
            archive_value,
            {"path", "sha256", "size_bytes"},
            f"evaluation qualification source {source_key} archive",
        )
        _require_exact_keys(
            inventory_value,
            {"path", "canonical_sha256"},
            f"evaluation qualification source {source_key} inventory",
        )
        expected_binding = expected_source_bindings[source_key]
        expected_descriptor = (
            None
            if expected_binding["snapshot_descriptor_sha256"] is None
            else f"sources/{source_key}/snapshot-descriptor.json"
        )
        expected_patch = (
            "sources/upstream_rng_isolated/rng-isolation.patch"
            if source_key == "upstream_rng_isolated"
            else None
        )
        inventory_digest = _require_sha256(
            inventory_value.get("canonical_sha256"),
            f"evaluation qualification source {source_key} detailed inventory",
        )
        if (
            not _json_exact_equal(record.get("binding"), expected_binding)
            or record.get("root") != f"sources/{source_key}/source"
            or archive_value.get("path") != f"sources/{source_key}/source.tar"
            or archive_value.get("sha256") != expected_binding["archive_sha256"]
            or inventory_value.get("path") != f"sources/{source_key}/inventory.json"
            or record.get("snapshot_descriptor_path") != expected_descriptor
            or record.get("patch_path") != expected_patch
        ):
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification source {source_key} closure drifted"
            )
        _require_exact_int(
            archive_value.get("size_bytes"),
            f"evaluation qualification source {source_key} archive size",
            minimum=1,
            maximum=qualification._MAX_SOURCE_BYTES,
        )
        inventory_sha256_by_source[source_key] = inventory_digest

    raw_candidates = payload.get("candidates")
    if type(raw_candidates) is not dict or set(raw_candidates) != set(
        open_protocol.MATCHED_CURRENT_CANDIDATE_IDS
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation qualification candidate set drifted"
        )
    candidate_values = cast(dict[str, Any], raw_candidates)
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS:
        raw_record = candidate_values[candidate_id]
        if type(raw_record) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification candidate {candidate_id} must be an object"
            )
        record = cast(dict[str, Any], raw_record)
        _require_exact_keys(
            record,
            {
                "source_key",
                "configuration",
                "probe",
                "effective_seed_proof_sha256",
                "resources",
                "resource_supplement",
                "capability_receipt",
                "entrypoint",
            },
            f"evaluation qualification candidate {candidate_id}",
        )
        frozen_candidate = seal_content.open_protocol.candidate_index[candidate_id]
        expected_entrypoint = _expected_entrypoint_binding(candidate_id)
        configuration = record.get("configuration")
        probe = record.get("probe")
        receipt = record.get("capability_receipt")
        entrypoint = record.get("entrypoint")
        if not all(type(value) is dict for value in (configuration, probe, receipt, entrypoint)):
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification candidate {candidate_id} nested records drifted"
            )
        configuration_value = cast(dict[str, Any], configuration)
        probe_record = cast(dict[str, Any], probe)
        receipt_record = cast(dict[str, Any], receipt)
        entrypoint_record = cast(dict[str, Any], entrypoint)
        _require_exact_keys(
            configuration_value,
            {"binding", "original_path", "derived_path"},
            f"evaluation qualification candidate {candidate_id} configuration",
        )
        _require_exact_keys(
            probe_record,
            {"path", "sha256", "stderr_sha256"},
            f"evaluation qualification candidate {candidate_id} probe",
        )
        _require_exact_keys(
            receipt_record,
            {"path", "sha256"},
            f"evaluation qualification candidate {candidate_id} receipt",
        )
        _require_exact_keys(
            entrypoint_record,
            {"path", "sha256", "python_import_root", "invocation_style", "result_root"},
            f"evaluation qualification candidate {candidate_id} entrypoint",
        )
        if (
            record.get("source_key") != expected_entrypoint["source_key"]
            or not _json_exact_equal(
                configuration_value.get("binding"),
                frozen_candidate.configuration.to_dict(),
            )
            or configuration_value.get("original_path")
            != f"configurations/{candidate_id}/original.json"
            or configuration_value.get("derived_path")
            != f"configurations/{candidate_id}/derived.json"
            or record.get("effective_seed_proof_sha256")
            != frozen_candidate.seed_contract.effective_seed_proof_sha256
            or not _json_exact_equal(
                record.get("resources"),
                frozen_candidate.resources.to_dict(),
            )
            or probe_record.get("path") != f"probes/{candidate_id}.json"
            or probe_record.get("stderr_sha256") != empty_sha256
            or receipt_record.get("path") != f"receipts/{candidate_id}.json"
            or receipt_record.get("sha256")
            != frozen_candidate.runtime_binding.capability_qualification_receipt_sha256
            or entrypoint_record.get("path") != expected_entrypoint["path"]
            or entrypoint_record.get("python_import_root")
            != expected_entrypoint["python_import_root"]
            or entrypoint_record.get("invocation_style")
            != expected_entrypoint["invocation_style"]
            or entrypoint_record.get("result_root") != expected_entrypoint["result_root"]
        ):
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation qualification candidate {candidate_id} closure drifted"
            )
        _require_sha256(
            probe_record.get("sha256"),
            f"evaluation qualification candidate {candidate_id} probe",
        )
        _require_sha256(
            entrypoint_record.get("sha256"),
            f"evaluation qualification candidate {candidate_id} entrypoint",
        )
        _validate_resource_supplement(
            record.get("resource_supplement"),
            candidate_id=candidate_id,
            horizon=seal_content.open_protocol.horizon,
            label=f"evaluation qualification candidate {candidate_id} resource supplement",
        )
    return qualification_manifest_sha256, inventory_sha256_by_source


def _validate_executor_qualification_artifacts(
    raw_payload: Any,
    *,
    label: str,
) -> Mapping[str, Mapping[str, tuple[str, int]]]:
    """Replay the non-authority boundary embedded in both qualification copies."""
    if type(raw_payload) is not dict:
        raise ForagerMatchedFinalAnalysisError(f"{label} must be an object")
    payload = cast(dict[str, Any], raw_payload)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "classification",
            "cpu_qualification",
            "rng_parity_qualification",
        },
        label,
    )
    cpu = payload.get("cpu_qualification")
    rng = payload.get("rng_parity_qualification")
    if type(cpu) is not dict or type(rng) is not dict:
        raise ForagerMatchedFinalAnalysisError(f"{label} qualification records must be objects")
    cpu_value = cast(dict[str, Any], cpu)
    rng_value = cast(dict[str, Any], rng)
    _require_exact_keys(
        cpu_value,
        {"receipt", "qualification", "environment_profile", "authority_boundary"},
        f"{label} CPU qualification",
    )
    _require_exact_keys(
        rng_value,
        {
            "plan",
            "receipt",
            "status",
            "external_executor_receipt_requires_trust_resolver",
            "promotion_authorized",
            "environment_rng_schedule_sha256",
            "rng_parity_contract_sha256",
        },
        f"{label} RNG parity qualification",
    )
    receipt = cpu_value.get("receipt")
    cpu_qualification = cpu_value.get("qualification")
    environment = cpu_value.get("environment_profile")
    rng_plan = rng_value.get("plan")
    rng_receipt = rng_value.get("receipt")
    records = (
        (receipt, {"path", "file_sha256", "size_bytes"}, f"{label} CPU receipt"),
        (
            cpu_qualification,
            {"path", "file_sha256", "size_bytes", "qualification_sha256"},
            f"{label} CPU qualification artifact",
        ),
        (
            environment,
            {"path", "file_sha256", "size_bytes", "canonical_payload_sha256"},
            f"{label} CPU environment profile",
        ),
        (rng_plan, {"path", "file_sha256", "size_bytes"}, f"{label} RNG plan"),
        (rng_receipt, {"path", "file_sha256", "size_bytes"}, f"{label} RNG receipt"),
    )
    for raw_record, keys, record_label in records:
        if type(raw_record) is not dict:
            raise ForagerMatchedFinalAnalysisError(f"{record_label} must be an object")
        record = cast(dict[str, Any], raw_record)
        _require_exact_keys(record, keys, record_label)
        if type(record["size_bytes"]) is not int or record["size_bytes"] <= 0:
            raise ForagerMatchedFinalAnalysisError(
                f"{record_label} size must be a positive integer"
            )
    receipt_value = cast(dict[str, Any], receipt)
    qualification_value = cast(dict[str, Any], cpu_qualification)
    environment_value = cast(dict[str, Any], environment)
    rng_plan_value = cast(dict[str, Any], rng_plan)
    rng_receipt_value = cast(dict[str, Any], rng_receipt)
    expected_authority = _expected_qualification_authority_boundary()
    if (
        payload["schema_version"] != "alberta.forager_matched_qualification_artifacts.v1"
        or payload["classification"] != "content_identity_only_external_verification_required"
        or not _json_exact_equal(cpu_value["authority_boundary"], expected_authority)
        or receipt_value["path"] != "official_cpu_qualification_5eca_2000001_v1/receipt.v1.json"
        or receipt_value["file_sha256"] != executor.CPU_QUALIFICATION_RECEIPT_FILE_SHA256
        or qualification_value["path"]
        != "official_cpu_qualification_5eca_2000001_v1/qualification.json"
        or qualification_value["file_sha256"] != executor.CPU_QUALIFICATION_FILE_SHA256
        or qualification_value["qualification_sha256"] != executor.QUALIFIED_EXECUTOR_RECEIPT_SHA256
        or environment_value["path"]
        != "official_cpu_qualification_5eca_2000001_v1/environment-profile.json"
        or environment_value["file_sha256"] != executor.CPU_ENVIRONMENT_PROFILE_FILE_SHA256
        or environment_value["canonical_payload_sha256"]
        != executor.QUALIFIED_RUNTIME_PROFILE_SHA256
        or rng_plan_value["path"] != "rng_parity_live_qualification_v1_execution/plan.json"
        or rng_plan_value["file_sha256"] != executor.RNG_PARITY_PLAN_FILE_SHA256
        or rng_receipt_value["path"] != "rng_parity_live_qualification_v1_execution/receipt.json"
        or rng_receipt_value["file_sha256"] != executor.RNG_PARITY_RECEIPT_FILE_SHA256
        or rng_value["status"] != "content_complete_external_executor_receipt_unverified"
        or rng_value["external_executor_receipt_requires_trust_resolver"] is not True
        or rng_value["promotion_authorized"] is not False
        or rng_value["environment_rng_schedule_sha256"]
        != executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
        or rng_value["rng_parity_contract_sha256"] != executor.RNG_PARITY_CONTRACT_SHA256
    ):
        raise ForagerMatchedFinalAnalysisError(
            f"{label} authority and qualification closure drifted"
        )
    return {
        "cpu": {
            "receipt.v1.json": (
                cast(str, receipt_value["file_sha256"]),
                cast(int, receipt_value["size_bytes"]),
            ),
            "qualification.json": (
                cast(str, qualification_value["file_sha256"]),
                cast(int, qualification_value["size_bytes"]),
            ),
            "environment-profile.json": (
                cast(str, environment_value["file_sha256"]),
                cast(int, environment_value["size_bytes"]),
            ),
        },
        "rng_parity": {
            "plan.json": (
                cast(str, rng_plan_value["file_sha256"]),
                cast(int, rng_plan_value["size_bytes"]),
            ),
            "receipt.json": (
                cast(str, rng_receipt_value["file_sha256"]),
                cast(int, rng_receipt_value["size_bytes"]),
            ),
        },
    }


def _validate_plan_and_manifests(
    plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    executor_manifest: Mapping[str, Any],
    sealed_protocol: protocol.ForagerMatchedProtocol,
    candidate_order: tuple[str, ...],
    qualification_source_inventory_sha256: Mapping[str, str],
    qualification_manifest_sha256: str,
) -> None:
    qualification_digest = _require_sha256(
        qualification_manifest_sha256,
        "evaluation qualification manifest",
    )
    _require_exact_keys(
        plan,
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
        "evaluation execution plan",
    )
    _require_exact_keys(
        source_manifest,
        {"schema_version", "stage", "protocol_sha256", "candidates"},
        "evaluation source manifest",
    )
    raw_candidates = source_manifest.get("candidates")
    if type(raw_candidates) is not list or len(raw_candidates) != len(candidate_order):
        raise ForagerMatchedFinalAnalysisError("evaluation source manifest candidate block drifted")
    source_keys = {
        "candidate_id",
        "capability_descriptor_sha256",
        "capability_qualification_receipt_sha256",
        "source",
        "source_inventory_hash_scheme",
        "executor_inventory_sha256",
        "configuration",
        "entrypoint_family",
        "entrypoint_path",
        "python_import_root",
        "invocation_style",
        "result_root",
        "rng_isolation_patch_sha256",
    }
    for index, (raw_candidate, candidate_id) in enumerate(
        zip(raw_candidates, candidate_order, strict=True)
    ):
        if type(raw_candidate) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                "evaluation source manifest contains a non-object candidate"
            )
        candidate = cast(dict[str, Any], raw_candidate)
        _require_exact_keys(
            candidate,
            source_keys,
            f"evaluation source manifest candidate {index}",
        )
        if candidate_id not in sealed_protocol.candidate_index:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation source manifest names unknown candidate {candidate_id!r}"
            )
        frozen_candidate = sealed_protocol.candidate_index[candidate_id]
        expected_entrypoint = _expected_entrypoint_binding(candidate_id)
        source_key = cast(str, expected_entrypoint["source_key"])
        if (
            candidate["candidate_id"] != candidate_id
            or not _json_exact_equal(
                candidate["source"], frozen_candidate.source.to_dict()
            )
            or not _json_exact_equal(
                candidate["configuration"], frozen_candidate.configuration.to_dict()
            )
            or candidate["entrypoint_family"] != frozen_candidate.entrypoint_family
            or candidate["capability_descriptor_sha256"]
            != protocol.candidate_capability_descriptor_sha256(frozen_candidate)
            or candidate["capability_qualification_receipt_sha256"]
            != frozen_candidate.runtime_binding.capability_qualification_receipt_sha256
            or candidate["source_inventory_hash_scheme"]
            != forager_rng_parity.REQUIRED_SOURCE_TREE_HASH_SCHEME
            or candidate["executor_inventory_sha256"]
            != qualification_source_inventory_sha256[source_key]
            or candidate["entrypoint_path"] != expected_entrypoint["path"]
            or candidate["python_import_root"]
            != expected_entrypoint["python_import_root"]
            or candidate["invocation_style"] != expected_entrypoint["invocation_style"]
            or candidate["result_root"] != expected_entrypoint["result_root"]
            or candidate["rng_isolation_patch_sha256"]
            != expected_entrypoint["rng_isolation_patch_sha256"]
        ):
            raise ForagerMatchedFinalAnalysisError(
                "evaluation source manifest differs from the sealed candidate"
            )

    _require_exact_keys(
        executor_manifest,
        {
            "schema_version",
            "authentication_state",
            "protocol_sha256",
            "qualification_manifest_sha256",
            "runtime",
            "qualified_lock",
            "container_helper",
            "scorer",
            "sandbox",
            "resource_limits",
            "qualification_artifacts",
        },
        "evaluation executor manifest",
    )
    qualified_lock = executor_manifest.get("qualified_lock")
    container_helper = executor_manifest.get("container_helper")
    scorer = executor_manifest.get("scorer")
    resource_limits = executor_manifest.get("resource_limits")
    qualification_artifacts = executor_manifest.get("qualification_artifacts")
    scoring_boundary = plan.get("scoring_boundary")
    if type(container_helper) is not dict or type(scorer) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation executor helper and scorer must be objects"
        )
    helper_value = cast(dict[str, Any], container_helper)
    scorer_value = cast(dict[str, Any], scorer)
    _require_exact_keys(
        helper_value,
        {"path", "sha256", "size_bytes", "contract"},
        "evaluation executor container helper",
    )
    _require_exact_keys(
        scorer_value,
        {"path", "sha256", "size_bytes", "execution_boundary"},
        "evaluation executor scorer",
    )
    _validate_executor_qualification_artifacts(
        qualification_artifacts,
        label="evaluation executor qualification artifacts",
    )
    _require_sha256(
        helper_value.get("sha256"),
        "evaluation executor container helper",
    )
    if (
        source_manifest.get("schema_version") != executor.MATCHED_SOURCE_MANIFEST_SCHEMA_VERSION
        or source_manifest.get("stage") != "sealed_evaluation"
        or source_manifest.get("protocol_sha256") != sealed_protocol.protocol_sha256
        or executor_manifest.get("schema_version")
        != executor.MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION
        or executor_manifest.get("authentication_state")
        != "unendorsed_external_trust_resolution_required"
        or executor_manifest.get("protocol_sha256") != sealed_protocol.protocol_sha256
        or executor_manifest.get("qualification_manifest_sha256")
        != qualification_digest
        or plan.get("qualification_manifest_sha256") != qualification_digest
        or not _json_exact_equal(
            executor_manifest.get("runtime"), sealed_protocol.runtime.to_dict()
        )
        or not _json_exact_equal(
            executor_manifest.get("sandbox"), sealed_protocol.runtime.sandbox.to_dict()
        )
        or not _json_exact_equal(
            qualified_lock,
            {
            "image_sha256": executor.QUALIFIED_IMAGE_SHA256,
            "runtime_profile_sha256": executor.QUALIFIED_RUNTIME_PROFILE_SHA256,
            "executor_qualification_receipt_sha256": (executor.QUALIFIED_EXECUTOR_RECEIPT_SHA256),
            "environment_rng_schedule_sha256": (executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256),
            "rng_parity_contract_sha256": executor.RNG_PARITY_CONTRACT_SHA256,
            "metric_semantics_sha256": executor.MATCHED_METRIC_SEMANTICS_SHA256,
            },
        )
        or helper_value["path"] != "alberta_framework/benchmarks/_forager_matched_container.py"
        or helper_value["contract"] != executor.CONTAINER_CONTRACT
        or type(helper_value["size_bytes"]) is not int
        or helper_value["size_bytes"] <= 0
        or scorer_value["path"] != "alberta_framework/benchmarks/_foragax_open_screen_scorer_v3.py"
        or type(scorer_value["size_bytes"]) is not int
        or scorer_value["size_bytes"] <= 0
        or scorer_value["execution_boundary"]
        != "qualified_oci_only_host_must_not_load_reward_arrays"
        or scorer_value["sha256"] != sealed_protocol.analysis_plan.metric_implementation_sha256
        or not _json_exact_equal(
            resource_limits,
            {
            "cpu_quota": executor._CONTAINER_CPU_QUOTA,
            "memory": executor._CONTAINER_MEMORY_LIMIT,
            "memory_swap": executor._CONTAINER_MEMORY_LIMIT,
            "pids": 512,
            "execution_timeout_seconds": executor._PROCESS_TIMEOUT_SECONDS,
            },
        )
        or not _json_exact_equal(
            scoring_boundary,
            {
            "host_reward_array_access": "forbidden",
            "scorer_runtime": "same_exact_qualified_oci_image",
            "scorer_source_sha256": scorer_value["sha256"],
            "scorer_output": "canonical_hashes_and_scalar_score_only",
            },
        )
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation execution source/executor/reward closure drifted"
        )
    templates = plan.get("candidate_command_templates")
    if type(templates) is not list or len(templates) != len(candidate_order):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation candidate command template block drifted"
        )
    for index, (raw_template, candidate_id) in enumerate(
        zip(templates, candidate_order, strict=True)
    ):
        if type(raw_template) is not dict:
            raise ForagerMatchedFinalAnalysisError(
                "evaluation candidate command template is not an object"
            )
        template = cast(dict[str, Any], raw_template)
        _require_exact_keys(
            template,
            {"candidate_id", "argv"},
            f"evaluation candidate command template {index}",
        )
        if candidate_id not in sealed_protocol.candidate_index:
            raise ForagerMatchedFinalAnalysisError(
                f"evaluation command template names unknown candidate {candidate_id!r}"
            )
        expected_argv = _expected_candidate_command_template(
            sealed_protocol,
            sealed_protocol.candidate_index[candidate_id],
            _expected_entrypoint_binding(candidate_id),
        )
        if (
            template["candidate_id"] != candidate_id
            or template["argv"] != expected_argv
        ):
            raise ForagerMatchedFinalAnalysisError("evaluation candidate command template drifted")


def _expected_campaign_manifest(
    *,
    seal_content: seal.ContentVerifiedSealBundle,
    sealed_protocol: protocol.ForagerMatchedProtocol,
    candidate_order: tuple[str, ...],
    plan_sha256: str,
    scores: evidence.MatchedScoreEvidence,
    qualification_manifest_sha256: str,
    live_runtime_sha256: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    selected = candidate_order[:4]
    fixed = candidate_order[4:]
    return {
        "schema_version": (sealed_campaign.MATCHED_SEALED_EVALUATION_CAMPAIGN_SCHEMA_VERSION),
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "prepared_sealed_evaluation_fresh_authentication_required",
        "stage": "sealed_evaluation",
        "authority": {
            "content_only": True,
            "cached_bindings_accepted": False,
            "fresh_seal_authentication_required_before_mutation": True,
        },
        "content_capture_threat_boundary": {
            "campaign_tree_writers": "cooperative_lock_participants_trusted",
            "root_guard_scope": "top_level_inode_and_path_swap_detection_only",
            "noncooperative_same_uid_writers": "out_of_scope",
            "claim_authority": "independent_final_subject_authentication_required",
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "seal_manifest_payload_sha256": seal_content.manifest["payload_sha256"],
        "open_verification_subject_sha256": (
            seal_content.open_verification_request.verification_subject_sha256
        ),
        "sealed_transition_sha256": seal_content.sealed_transition_sha256,
        "protocol_sha256": sealed_protocol.protocol_sha256,
        "candidate_universe_sha256": (sealed_protocol.selection_plan.candidate_universe_sha256),
        "execution_plan_sha256": plan_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "qualification_manifest_sha256": qualification_manifest_sha256,
        "live_runtime_identity_sha256": live_runtime_sha256,
        "execution_schedule_sha256": schedule_sha256,
        "candidate_order": list(candidate_order),
        "selected_candidate_ids": list(selected),
        "fixed_descriptive_candidate_ids": list(fixed),
        "active_seeds": list(sealed_protocol.active_seeds),
        "cell_count": _EXPECTED_CELLS,
        "host_reward_array_access": "forbidden",
        "retention_bounds": {
            "max_attempts_per_cell": campaign._MAX_ATTEMPTS_PER_CELL,
            "max_failure_records_per_attempt": campaign._MAX_FAILURE_RECORDS_PER_ATTEMPT,
            "max_raw_archive_bytes": campaign._MAX_RAW_BYTES,
            "max_retained_raw_bytes_per_cell": (campaign._MAX_RETAINED_RAW_BYTES_PER_CELL),
            "max_retained_raw_bytes_per_campaign": (campaign._MAX_RETAINED_RAW_BYTES_PER_CAMPAIGN),
        },
        "completion_boundary": (
            "evaluation_score_evidence_and_unresolved_verification_request_only"
        ),
    }


def _validate_evaluation_snapshot(
    artifacts: Mapping[str, bytes],
    seal_content: seal.ContentVerifiedSealBundle,
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings,
) -> tuple[
    evidence.MatchedScoreEvidence,
    executor.VerificationRequest,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    sealed_protocol = protocol.parse_forager_matched_protocol(
        seal._decode_canonical(
            artifacts["sealed-protocol.json"],
            "evaluation sealed protocol",
        )
    )
    if sealed_protocol != seal_content.sealed_protocol:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation sealed protocol differs from seal closure"
        )
    transition = protocol.validate_sealed_protocol_transition(
        seal_content.open_protocol,
        sealed_protocol,
        seal_content.selection_result,
        seal_content.selection_result.selection_result_sha256,
    )
    transition_payload = seal._decode_canonical(
        artifacts["sealed-transition.json"],
        "evaluation sealed transition",
    )
    expected_transition = evaluation.build_sealed_transition_descriptor(
        sealed_protocol,
        transition,
    )
    if not _json_exact_equal(
        transition_payload,
        expected_transition,
    ) or not _json_exact_equal(
        transition_payload,
        seal._plain(seal_content.sealed_transition),
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation sealed transition differs from typed replay"
        )
    if not _json_exact_equal(
        seal._decode_canonical(
            artifacts["seal-manifest.json"],
            "evaluation seal manifest",
        ),
        seal._plain(seal_content.manifest),
    ):
        raise ForagerMatchedFinalAnalysisError("evaluation seal manifest differs from seal closure")
    if not _json_exact_equal(
        seal._decode_canonical(
            artifacts["candidate-universe.json"],
            "evaluation candidate universe",
        ),
        universe.matched_current_candidate_universe_descriptor(),
    ):
        raise ForagerMatchedFinalAnalysisError("evaluation candidate universe snapshot drifted")
    (
        qualification_manifest_sha256,
        qualification_source_inventory_sha256,
    ) = _validate_qualification_manifest(
        artifacts["qualification-manifest.json"],
        seal_content,
    )
    qualification_manifest = cast(
        dict[str, Any],
        qualification._decode_json(
            artifacts["qualification-manifest.json"],
            "evaluation qualification manifest",
        ),
    )

    plan = seal._decode_canonical(
        artifacts["execution-plan.json"],
        "evaluation execution plan",
    )
    source_manifest = seal._decode_canonical(
        artifacts["source-manifest.json"],
        "evaluation source manifest",
    )
    executor_manifest = seal._decode_canonical(
        artifacts["executor-manifest.json"],
        "evaluation executor manifest",
    )
    try:
        request_payload = seal._decode_canonical(
            artifacts["verification-request.json"],
            "evaluation verification request",
        )
    except seal.ForagerMatchedSealError as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"evaluation verification request is invalid: {exc}"
        ) from exc
    if not _json_exact_equal(
        request_payload.get("qualification_authority_boundary"),
        _expected_qualification_authority_boundary(),
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation verification request qualification authority boundary drifted"
        )
    try:
        scores = evidence.parse_matched_score_evidence(artifacts["score-evidence.json"])
        request = executor.parse_verification_request(
            artifacts["verification-request.json"]
        )
    except (
        evidence.ForagerMatchedEvidenceError,
        executor.ForagerMatchedExecutorError,
    ) as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"evaluation score/request artifact is invalid: {exc}"
        ) from exc
    _assert_request_bindings(request, evaluation_bindings)
    candidate_order = tuple(item.candidate_id for item in scores.candidate_scores)
    _validate_plan_and_manifests(
        plan,
        source_manifest,
        executor_manifest,
        sealed_protocol,
        candidate_order,
        qualification_source_inventory_sha256,
        qualification_manifest_sha256,
    )
    seal_open_campaign = seal_content.manifest.get("open_campaign")
    if not isinstance(seal_open_campaign, Mapping):
        raise ForagerMatchedFinalAnalysisError(
            "seal manifest open-campaign closure must be an object"
        )
    qualification_digests = {
        qualification_manifest_sha256,
        seal_content.open_score_evidence.qualification_manifest_sha256,
        seal_content.open_verification_request.qualification_manifest_sha256,
        seal_open_campaign.get(
            "qualification_manifest_sha256"
        ),
        scores.qualification_manifest_sha256,
        request.qualification_manifest_sha256,
        evaluation_bindings.qualification_manifest_sha256,
    }
    plan_active_seeds = plan.get("active_seeds")
    plan_candidate_order = plan.get("candidate_order")
    if (
        len(qualification_digests) != 1
        or
        not _json_exact_equal(
            qualification_manifest.get("frozen_executor_qualification_artifacts"),
            executor_manifest.get("qualification_artifacts"),
        )
        or plan.get("schema_version") != executor.MATCHED_EXECUTION_PLAN_SCHEMA_VERSION
        or plan.get("classification") != "matched_current_execution_candidate"
        or plan.get("stage") != "sealed_evaluation"
        or plan.get("promotion_authorized") is not False
        or plan.get("external_verification_required") is not True
        or plan.get("protocol_sha256") != sealed_protocol.protocol_sha256
        or plan.get("qualification_manifest_sha256")
        != qualification_manifest_sha256
        or executor_manifest.get("qualification_manifest_sha256")
        != qualification_manifest_sha256
        or not _json_exact_equal(plan.get("horizon"), sealed_protocol.horizon)
        or type(plan_active_seeds) is not list
        or not _json_exact_equal(plan_active_seeds, list(sealed_protocol.active_seeds))
        or type(plan_candidate_order) is not list
        or not _json_exact_equal(plan_candidate_order, list(candidate_order))
        or not _json_exact_equal(plan.get("source_manifest"), source_manifest)
        or not _json_exact_equal(plan.get("executor_manifest"), executor_manifest)
        or plan.get("source_manifest_sha256") != scores.source_evidence_sha256
        or plan.get("executor_manifest_sha256") != scores.executor_evidence_sha256
        or campaign._canonical_sha256(source_manifest) != scores.source_evidence_sha256
        or campaign._canonical_sha256(executor_manifest) != scores.executor_evidence_sha256
    ):
        raise ForagerMatchedFinalAnalysisError("evaluation execution-plan closure drifted")
    live_runtime = seal._decode_canonical(
        artifacts["live-runtime.json"],
        "evaluation live runtime",
    )
    _require_exact_keys(
        live_runtime,
        {
            "schema_version",
            "executable_sha256",
            "version",
            "image_inspection",
            "executor_manifest_sha256",
        },
        "evaluation live runtime",
    )
    image_inspection = live_runtime.get("image_inspection")
    version = live_runtime.get("version")
    _require_sha256(
        live_runtime.get("executable_sha256"),
        "evaluation live-runtime executable",
    )
    if type(version) is not dict or type(image_inspection) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation live-runtime version and image inspection must be objects"
        )
    image_value = cast(dict[str, Any], image_inspection)
    image_config = image_value.get("Config")
    if type(image_config) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation live-runtime image Config must be an object"
        )
    labels = cast(dict[str, Any], image_config).get("Labels")
    if (
        live_runtime.get("schema_version") != executor.MATCHED_LIVE_RUNTIME_SCHEMA_VERSION
        or live_runtime.get("executor_manifest_sha256") != scores.executor_evidence_sha256
        or image_value.get("Id") != f"sha256:{sealed_protocol.runtime.image_sha256}"
        or type(labels) is not dict
        or cast(dict[str, Any], labels).get("io.elizaos.alberta.foragax.launcher-contract")
        != "oci-read-only-stdout-tar-v4"
    ):
        raise ForagerMatchedFinalAnalysisError("evaluation live-runtime closure drifted")
    live_runtime_sha256 = campaign._canonical_sha256(live_runtime)
    receipt_index = seal._decode_canonical(
        artifacts["execution-receipt-index.json"],
        "evaluation execution receipt index",
    )
    receipt_payload_sha256 = _validate_evaluation_receipt_index(
        receipt_index,
        sealed_protocol,
        scores,
        plan_sha256=campaign._canonical_sha256(plan),
        live_runtime_sha256=live_runtime_sha256,
    )
    schedule_payload = seal._decode_canonical(
        artifacts["execution-schedule.json"],
        "evaluation execution schedule",
    )
    if type(schedule_payload) is not dict:
        raise ForagerMatchedFinalAnalysisError(
            "evaluation execution schedule must be an object"
        )
    schedule_sha256 = _require_sha256(
        schedule_payload.get("schedule_sha256"),
        "evaluation execution schedule SHA-256",
    )
    parsed_schedule = evaluation.parse_sealed_evaluation_schedule(
        artifacts["execution-schedule.json"],
        sealed_protocol=sealed_protocol,
        transition=transition,
        expected_schedule_sha256=schedule_sha256,
    )
    expected_schedule = evaluation.build_sealed_evaluation_schedule(
        sealed_protocol,
        transition,
    )
    if not _json_exact_equal(
        schedule_payload,
        expected_schedule,
    ) or not _json_exact_equal(
        seal._plain(parsed_schedule),
        expected_schedule,
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation schedule differs by JSON type from exact replay"
        )
    if (
        len(candidate_order) != _EXPECTED_CANDIDATES
        or len(scores.active_seeds) != _EXPECTED_SEEDS
        or candidate_order != transition.evaluation_candidate_ids
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation score evidence is not the exact 6-by-30 panel"
        )
    expected_selected = transition.evaluation_candidate_ids[:4]
    expected_fixed = transition.evaluation_candidate_ids[4:]
    if (
        candidate_order != (*expected_selected, *expected_fixed)
        or len(expected_selected) != 4
        or len(expected_fixed) != 2
        or set(expected_selected) & set(expected_fixed)
        or expected_fixed != sealed_protocol.evaluation_panel.fixed_descriptive_candidate_ids
    ):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation selected/fixed six-candidate partition drifted"
        )
    completion = seal._decode_canonical(
        artifacts["completion-summary.json"],
        "evaluation completion summary",
    )
    expected_completion = {
        "schema_version": (sealed_campaign.MATCHED_SEALED_EVALUATION_COMPLETION_SCHEMA_VERSION),
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": "sealed_evaluation",
        "protocol_sha256": sealed_protocol.protocol_sha256,
        "qualification_manifest_sha256": qualification_manifest_sha256,
        "execution_plan_sha256": campaign._canonical_sha256(plan),
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "live_runtime_identity_sha256": live_runtime_sha256,
        "execution_schedule_sha256": schedule_sha256,
        "seal_manifest_payload_sha256": seal_content.manifest["payload_sha256"],
        "open_verification_subject_sha256": (
            seal_content.open_verification_request.verification_subject_sha256
        ),
        "sealed_transition_sha256": seal_content.sealed_transition_sha256,
        "candidate_count": _EXPECTED_CANDIDATES,
        "seed_count": _EXPECTED_SEEDS,
        "completed_cell_count": _EXPECTED_CELLS,
        "execution_receipt_index_payload_sha256": receipt_payload_sha256,
        "score_evidence_sha256": scores.payload_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
        "evaluation_verification_authentication_state": ("unresolved_external_verifier_required"),
        "selection_inherited_from_seal": True,
        "sealed_protocol_inherited_from_seal": True,
        "evaluation_artifacts_created": True,
        "cached_bindings_accepted_as_authority": False,
        "content_capture_threat_boundary": {
            "campaign_tree_writers": "cooperative_lock_participants_trusted",
            "root_guard_scope": "top_level_inode_and_path_swap_detection_only",
            "noncooperative_same_uid_writers": "out_of_scope",
            "claim_authority": "independent_final_subject_authentication_required",
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "host_reward_array_access": "forbidden_not_performed",
    }
    if not _json_exact_equal(completion, expected_completion):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation completion exact authority/threat/reward closure drifted"
        )
    campaign_manifest = seal._decode_canonical(
        artifacts["campaign.json"],
        "evaluation campaign manifest",
    )
    expected_campaign = _expected_campaign_manifest(
        seal_content=seal_content,
        sealed_protocol=sealed_protocol,
        candidate_order=candidate_order,
        plan_sha256=campaign._canonical_sha256(plan),
        scores=scores,
        qualification_manifest_sha256=qualification_manifest_sha256,
        live_runtime_sha256=live_runtime_sha256,
        schedule_sha256=schedule_sha256,
    )
    if not _json_exact_equal(campaign_manifest, expected_campaign):
        raise ForagerMatchedFinalAnalysisError(
            "evaluation campaign exact authority/threat/reward closure drifted"
        )
    return scores, request, completion, schedule_payload


def _build_contract_and_result(
    seal_content: seal.ContentVerifiedSealBundle,
    evaluation_scores: evidence.MatchedScoreEvidence,
    *,
    open_bindings: evidence.AuthenticatedEvidenceBindings,
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings,
) -> tuple[
    statistics.MatchedComparisonContract,
    statistics.MatchedComparisonResult,
]:
    selection_report_sha256 = _require_sha256(
        seal_content.selection_report.get("payload_sha256"),
        "selection report payload",
    )
    contract, transition, parsed_scores = evidence.build_statistics_contract(
        seal_content.open_protocol,
        seal_content.sealed_protocol,
        seal_content.selection_result,
        seal_content.selection_report,
        seal_content.open_score_evidence,
        evaluation_scores,
        open_authenticated_bindings=open_bindings,
        evaluation_authenticated_bindings=evaluation_bindings,
        expected_selection_report_sha256=selection_report_sha256,
    )
    expected_transition = evaluation.build_sealed_transition_descriptor(
        seal_content.sealed_protocol,
        transition,
    )
    if (
        parsed_scores != evaluation_scores
        or expected_transition != seal._plain(seal_content.sealed_transition)
        or evaluation.canonical_sealed_transition_descriptor_sha256(
            seal_content.sealed_protocol,
            transition,
        )
        != seal_content.sealed_transition_sha256
    ):
        raise ForagerMatchedFinalAnalysisError(
            "statistics contract differs from the sealed evaluation transition"
        )
    fixed_ids = seal_content.sealed_protocol.evaluation_panel.fixed_descriptive_candidate_ids
    method_ids = tuple(method.method_id for method in contract.methods)
    diagnostic_ids = tuple(item.candidate_id for item in contract.fixed_descriptive_diagnostics)
    comparisons = (
        contract.primary_comparison,
        *contract.secondary_comparisons,
    )
    actual_hypotheses = tuple(
        (
            comparison.hypothesis_id,
            comparison.intervention_id,
            comparison.comparator_id,
        )
        for comparison in comparisons
    )
    expected_hypotheses = tuple(
        (
            hypothesis_id,
            method_ids[0] if len(method_ids) == 4 else "",
            comparator_id,
        )
        for hypothesis_id, comparator_id in zip(
            _EXPECTED_HYPOTHESIS_IDS,
            method_ids[1:],
            strict=True,
        )
    ) if len(method_ids) == 4 else ()
    resolved_hypotheses = tuple(
        (
            hypothesis.hypothesis_id,
            hypothesis.intervention_candidate_id,
            hypothesis.comparator_candidate_id,
        )
        for hypothesis in transition.resolved_hypotheses
    )
    if (
        len(method_ids) != 4
        or len(contract.secondary_comparisons) != 2
        or actual_hypotheses != expected_hypotheses
        or resolved_hypotheses != expected_hypotheses
    ):
        raise ForagerMatchedFinalAnalysisError(
            "statistics contract is not the exact ordered three-contrast "
            "Alberta-vs-external family"
        )
    compared_ids = {
        endpoint
        for comparison in comparisons
        for endpoint in (comparison.intervention_id, comparison.comparator_id)
    }
    if (
        diagnostic_ids != fixed_ids
        or set(fixed_ids) & set(method_ids)
        or set(fixed_ids) & compared_ids
    ):
        raise ForagerMatchedFinalAnalysisError(
            "fixed descriptive candidates entered inferential statistics"
        )
    result = statistics.analyze_matched_scores(contract)
    replayed = statistics.load_canonical_result(result.canonical_json(), contract)
    if replayed != result:
        raise ForagerMatchedFinalAnalysisError(
            "statistics result failed immediate canonical replay"
        )
    result_payload = result.to_payload()
    if (
        result_payload.get("no_promotion_authority") is not True
        or result_payload.get("evidence_digests_require_external_validation") is not True
    ):
        raise ForagerMatchedFinalAnalysisError(
            "statistics result lost its no-promotion authority boundary"
        )
    return contract, result


def _artifact_digest_map(
    seal_artifacts: Mapping[str, bytes],
    evaluation_artifacts: Mapping[str, bytes],
    analysis_artifacts: Mapping[str, bytes],
) -> dict[str, str]:
    values = {
        **{f"seal/{name}": hashlib.sha256(raw).hexdigest() for name, raw in seal_artifacts.items()},
        **{
            f"evaluation/{name}": hashlib.sha256(raw).hexdigest()
            for name, raw in evaluation_artifacts.items()
        },
        **{
            f"analysis/{name}": hashlib.sha256(raw).hexdigest()
            for name, raw in analysis_artifacts.items()
        },
    }
    expected = {
        *(f"seal/{name}" for name in _SEAL_ARTIFACTS),
        *(f"evaluation/{name}" for name in _EVALUATION_ARTIFACTS),
        *(f"analysis/{name}" for name in _ANALYSIS_ARTIFACTS),
    }
    if set(values) != expected:
        raise ForagerMatchedFinalAnalysisError("final-analysis artifact digest inventory drifted")
    return values


def _build_manifest(
    *,
    seal_content: seal.ContentVerifiedSealBundle,
    evaluation_scores: evidence.MatchedScoreEvidence,
    evaluation_request: executor.VerificationRequest,
    evaluation_completion: Mapping[str, Any],
    evaluation_schedule: Mapping[str, Any],
    open_bindings: evidence.AuthenticatedEvidenceBindings,
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings,
    analysis_runtime_source: Mapping[str, Any],
    contract: statistics.MatchedComparisonContract,
    result: statistics.MatchedComparisonResult,
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    fixed_ids = tuple(item.candidate_id for item in contract.fixed_descriptive_diagnostics)
    candidate_order = tuple(item.candidate_id for item in evaluation_scores.candidate_scores)
    selected_ids = tuple(method.method_id for method in contract.methods)
    seal_open_campaign = seal_content.manifest.get("open_campaign")
    if not isinstance(seal_open_campaign, Mapping):
        raise ForagerMatchedFinalAnalysisError(
            "seal manifest open-campaign closure must be an object"
        )
    qualification_digests = {
        seal_content.open_score_evidence.qualification_manifest_sha256,
        seal_content.open_verification_request.qualification_manifest_sha256,
        seal_open_campaign.get("qualification_manifest_sha256"),
        open_bindings.qualification_manifest_sha256,
        evaluation_scores.qualification_manifest_sha256,
        evaluation_request.qualification_manifest_sha256,
        evaluation_bindings.qualification_manifest_sha256,
        evaluation_completion.get("qualification_manifest_sha256"),
    }
    if len(qualification_digests) != 1:
        raise ForagerMatchedFinalAnalysisError(
            "final analysis does not share one exact qualification manifest"
        )
    qualification_digest = _require_sha256(
        next(iter(qualification_digests)),
        "final-analysis qualification manifest",
    )
    if (
        len(candidate_order) != _EXPECTED_CANDIDATES
        or len(selected_ids) != 4
        or len(fixed_ids) != 2
        or candidate_order != (*selected_ids, *fixed_ids)
    ):
        raise ForagerMatchedFinalAnalysisError(
            "analysis claim boundary is not the exact four-selected-plus-two-fixed partition"
        )
    comparisons = (contract.primary_comparison, *contract.secondary_comparisons)
    contrast_records = [
        {
            "hypothesis_id": comparison.hypothesis_id,
            "intervention_candidate_id": comparison.intervention_id,
            "comparator_candidate_id": comparison.comparator_id,
            "role": "primary" if index == 0 else "secondary_sensitivity",
        }
        for index, comparison in enumerate(comparisons)
    ]
    if (
        tuple(item["hypothesis_id"] for item in contrast_records)
        != _EXPECTED_HYPOTHESIS_IDS
        or tuple(item["intervention_candidate_id"] for item in contrast_records)
        != (selected_ids[0],) * 3
        or tuple(item["comparator_candidate_id"] for item in contrast_records)
        != selected_ids[1:]
    ):
        raise ForagerMatchedFinalAnalysisError(
            "analysis manifest is not the exact ordered three-contrast family"
        )
    secondary_sensitivity_records = [
        {
            "hypothesis_id": item.comparison.hypothesis_id,
            "computed_holm_reject": item.holm.reject,
            "interpretation": "nonconfirmatory_sensitivity_only",
        }
        for item in result.secondary
    ]
    runtime_source = seal._plain(analysis_runtime_source)
    sources = cast(dict[str, Any], runtime_source["sources"])
    runtime = cast(dict[str, Any], runtime_source["runtime"])
    python_runtime = cast(dict[str, Any], runtime["python"])
    numpy_runtime = cast(dict[str, Any], runtime["numpy"])
    body: dict[str, Any] = {
        "schema_version": MATCHED_FINAL_ANALYSIS_MANIFEST_SCHEMA_VERSION,
        "classification": (
            "resolver_supplied_subject_bound_bindings_at_creation_cache_only_nonpromoting"
        ),
        "status": "calculation_complete_four_tuning_selected_plus_two_fixed_nonpromoting",
        "stage": "sealed_evaluation_analysis",
        "qualification_manifest_sha256": qualification_digest,
        "authority_boundary": {
            "persisted_bindings_are_cache_only": True,
            "content_loader_authenticates_nothing": True,
            "fresh_external_resolver_subject_resolution_required": True,
            "resolver_supplied_bindings_are_subject_bound": True,
            "resolver_legitimacy_not_established_by_bundle": True,
            "qualification_remains_content_only_unendorsed": True,
            "fresh_authentication_claim_authorized": False,
            "self_authentication_forbidden": True,
        },
        "claim_boundary": {
            "scope": (
                "three_preregistered_alberta_vs_selected_external_contrasts_"
                "within_six_executed_arms"
            ),
            "heldout_executed_candidate_ids": list(candidate_order),
            "tuning_selected_inferential_candidate_ids": list(selected_ids),
            "fixed_descriptive_candidate_ids": list(fixed_ids),
            "ordered_alberta_vs_external_contrasts": contrast_records,
            "contrast_count": 3,
            "six_arm_ranking_authorized": False,
            "full_registered_universe_best_claim_authorized": False,
            "candidate_universe_v2_contrast_specific_scope_enforced": True,
            "registered_panel_ranking_identified_by_design": False,
            "tuning_selection_endpoint_interpretation": (
                "frozen_ranking_statistic_not_population_confidence_bound"
            ),
            "tuning_selection_group_best_claim_authorized": False,
            "primary_gate_pass_never_authorizes_promotion": True,
            "primary_bootstrap_interpretation": (
                "frozen_resampling_summary_not_established_population_confidence_bound"
            ),
            "primary_seed_superpopulation_model": "unstated_in_frozen_protocol",
            "primary_sampling_exchangeability_assumption": "unstated_in_frozen_protocol",
            "primary_bootstrap_regularity_assumptions": "unstated_in_frozen_protocol",
            "primary_population_inferential_claim_authorized": False,
            "secondary_sign_flip_interpretation": ("nonconfirmatory_sensitivity_only"),
            "secondary_sign_exchangeability_assumption": ("unstated_in_frozen_protocol"),
            "secondary_inferential_claim_authorized": False,
            "secondary_reject_flags_are_claim_gates": False,
            "sota_claim_authorized": False,
            "unrestricted_performance_claim_authorized": False,
            "promotion_authorized": False,
            "statistics_result_standalone_claim_interpretation_forbidden": True,
            "primary_superiority_field_is_frozen_gate_name_only": True,
            "secondary_reject_fields_are_nonconfirmatory_calculation_outputs": True,
        },
        "bundle_replay_scope": {
            "scope": (
                "campaign_root_independent_scalar_statistics_and_digest_closure_"
                "under_matching_live_code_and_runtime"
            ),
            "self_contained_without_live_checkout": False,
            "self_contained_without_external_runtime": False,
            "raw_execution_archives_copied": False,
            "reward_trace_payloads_copied": False,
            "qualification_source_archives_copied": False,
            "executable_source_bytes_copied": False,
            "qualified_execution_source_available_by_digest_only": True,
            "analysis_replay_requires_matching_explicit_source_set_and_runtime": True,
            "source_set_is_not_a_mechanically_complete_transitive_import_closure": True,
            "workload_or_score_recomputation_requires_original_qualified_roots": True,
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "sota_claim_authorized": False,
        "external_verification_required": True,
        "analysis_executed": True,
        "raw_scores_or_differences_embedded_in_statistics_artifacts": False,
        "artifact_sha256": dict(artifact_sha256),
        "analysis_execution": {
            "runtime_source_payload_sha256": runtime_source["payload_sha256"],
            "finalizer_source_sha256": cast(dict[str, Any], sources["finalizer"])["sha256"],
            "statistics_source_sha256": cast(dict[str, Any], sources["statistics"])["sha256"],
            "python_implementation": python_runtime["implementation"],
            "python_version": python_runtime["version"],
            "numpy_version": numpy_runtime["version"],
            "source_hashes_and_runtime_identity_are_integrity_metadata_not_authentication": (True),
        },
        "seal": {
            "seal_manifest_payload_sha256": _require_sha256(
                seal_content.manifest.get("payload_sha256"),
                "seal manifest payload",
            ),
            "open_protocol_sha256": seal_content.open_protocol.protocol_sha256,
            "qualification_manifest_sha256": qualification_digest,
            "open_score_evidence_sha256": (seal_content.open_score_evidence.payload_sha256),
            "open_verification_subject_sha256": (
                seal_content.open_verification_request.verification_subject_sha256
            ),
            "open_verification_receipt_sha256": (open_bindings.verification_receipt_sha256),
            "open_bindings_cache_sha256": open_bindings.bindings_sha256,
            "selection_result_sha256": (seal_content.selection_result.selection_result_sha256),
            "selection_report_sha256": _require_sha256(
                seal_content.selection_report.get("payload_sha256"),
                "selection report payload",
            ),
            "sealed_protocol_sha256": seal_content.sealed_protocol.protocol_sha256,
            "sealed_transition_sha256": seal_content.sealed_transition_sha256,
        },
        "evaluation": {
            "protocol_sha256": evaluation_scores.protocol_sha256,
            "qualification_manifest_sha256": qualification_digest,
            "score_evidence_sha256": evaluation_scores.payload_sha256,
            "verification_subject_sha256": (evaluation_request.verification_subject_sha256),
            "verification_receipt_sha256": (evaluation_bindings.verification_receipt_sha256),
            "bindings_cache_sha256": evaluation_bindings.bindings_sha256,
            "execution_receipt_index_payload_sha256": (
                evaluation_completion["execution_receipt_index_payload_sha256"]
            ),
            "execution_plan_sha256": evaluation_completion["execution_plan_sha256"],
            "execution_schedule_sha256": evaluation_schedule["schedule_sha256"],
            "candidate_order": list(candidate_order),
            "active_seeds": list(evaluation_scores.active_seeds),
            "completed_cell_count": _EXPECTED_CELLS,
        },
        "statistics": {
            "contract_schema": statistics.CONTRACT_SCHEMA,
            "contract_payload_sha256": contract.payload_sha256,
            "result_schema": statistics.RESULT_SCHEMA,
            "result_payload_sha256": result.payload_sha256,
            "standalone_result_claim_interpretation_forbidden": True,
            "primary_superiority_field_is_frozen_gate_name_only": True,
            "secondary_reject_fields_are_nonconfirmatory_calculation_outputs": True,
            "primary_implementation_sha256": (contract.primary_analysis_implementation_sha256),
            "secondary_implementation_sha256": (contract.secondary_analysis_implementation_sha256),
            "primary_hypothesis_id": (contract.primary_comparison.hypothesis_id),
            "primary_frozen_resampling_gate_passed": result.primary.superiority_passed,
            "secondary_sensitivity_records": secondary_sensitivity_records,
            "ordered_contrast_records": contrast_records,
            "contrast_candidate_ids": list(selected_ids),
            "primary_calculation_candidate_ids": [
                contract.primary_comparison.intervention_id,
                contract.primary_comparison.comparator_id,
            ],
            "primary_population_inferential_claim_authorized": False,
            "secondary_inferential_claim_authorized": False,
            "fixed_descriptive_candidate_ids": list(fixed_ids),
            "fixed_descriptive_candidates_excluded_from_inference": True,
        },
    }
    return {**body, "payload_sha256": campaign._canonical_sha256(body)}


def _open_child(
    parent: seal._OpenDirectory,
    name: str,
    label: str,
) -> seal._OpenDirectory:
    return seal._open_stable_directory_at(
        parent,
        name,
        parent.path / name,
        label,
    )


def _load_from_open_root(
    opened: seal._OpenDirectory,
) -> ContentVerifiedFinalAnalysisBundle:
    root_inventory = _directory_inventory(
        opened,
        expected_files=_ROOT_FILES,
        expected_directories=("seal", "evaluation", "analysis"),
        label="final-analysis bundle root",
    )
    opened_children: list[seal._OpenDirectory] = []
    try:
        seal_directory = _open_child(
            opened,
            "seal",
            "final-analysis seal subtree",
        )
        opened_children.append(seal_directory)
        evaluation_directory = _open_child(
            opened,
            "evaluation",
            "final-analysis evaluation subtree",
        )
        opened_children.append(evaluation_directory)
        analysis_directory = _open_child(
            opened,
            "analysis",
            "final-analysis analysis subtree",
        )
        opened_children.append(analysis_directory)
        seal_inventory = _directory_inventory(
            seal_directory,
            expected_files=_SEAL_ARTIFACTS,
            label="final-analysis seal subtree",
        )
        seal_content = seal._load_forager_matched_seal_bundle_from_open_root(
            seal_directory,
            seal_inventory,
        )
        seal_artifacts, _seal_digests = _read_pairs(
            seal_directory,
            _SEAL_ARTIFACTS,
            "final-analysis seal subtree",
        )
        evaluation_inventory = _directory_inventory(
            evaluation_directory,
            expected_files=_EVALUATION_ARTIFACTS,
            label="final-analysis evaluation subtree",
        )
        evaluation_artifacts, _evaluation_digests = _read_pairs(
            evaluation_directory,
            _EVALUATION_ARTIFACTS,
            "final-analysis evaluation subtree",
        )
        analysis_inventory = _directory_inventory(
            analysis_directory,
            expected_files=_ANALYSIS_ARTIFACTS,
            label="final-analysis analysis subtree",
        )
        analysis_artifacts, _analysis_digests = _read_pairs(
            analysis_directory,
            _ANALYSIS_ARTIFACTS,
            "final-analysis analysis subtree",
        )
        analysis_runtime_source = _parse_analysis_runtime_source(
            analysis_artifacts["analysis-runtime-source.json"]
        )
        manifest_raw, _manifest_digest = seal._load_pair_at(
            opened,
            "manifest.json",
            "final-analysis manifest",
        )

        open_bindings = _parse_bindings(
            seal_artifacts["open-authenticated-bindings-cache.json"],
            expected_stage="open_tuning",
        )
        _assert_request_bindings(
            seal_content.open_verification_request,
            open_bindings,
        )
        evaluation_bindings = _parse_bindings(
            analysis_artifacts["evaluation-authenticated-bindings-cache.json"],
            expected_stage="sealed_evaluation",
        )
        (
            evaluation_scores,
            evaluation_request,
            evaluation_completion,
            evaluation_schedule,
        ) = _validate_evaluation_snapshot(
            evaluation_artifacts,
            seal_content,
            evaluation_bindings,
        )
        contract, result = _build_contract_and_result(
            seal_content,
            evaluation_scores,
            open_bindings=open_bindings,
            evaluation_bindings=evaluation_bindings,
        )
        if analysis_artifacts["statistics-contract.json"] != _canonical(
            cast(Mapping[str, Any], contract.to_payload())
        ):
            raise ForagerMatchedFinalAnalysisError(
                "persisted statistics contract differs from current enumerated-source replay"
            )
        replayed_result = statistics.load_canonical_result(
            analysis_artifacts["statistics-result.json"],
            contract,
        )
        if replayed_result != result:
            raise ForagerMatchedFinalAnalysisError(
                "persisted statistics result differs from current enumerated-source replay"
            )
        if not _json_exact_equal(
            seal._plain(analysis_runtime_source),
            _analysis_runtime_source_identity(),
        ):
            raise ForagerMatchedFinalAnalysisError(
                "analysis source or runtime changed during statistics replay"
            )
        artifact_sha256 = _artifact_digest_map(
            seal_artifacts,
            evaluation_artifacts,
            analysis_artifacts,
        )
        expected_manifest = _build_manifest(
            seal_content=seal_content,
            evaluation_scores=evaluation_scores,
            evaluation_request=evaluation_request,
            evaluation_completion=evaluation_completion,
            evaluation_schedule=evaluation_schedule,
            open_bindings=open_bindings,
            evaluation_bindings=evaluation_bindings,
            analysis_runtime_source=analysis_runtime_source,
            contract=contract,
            result=result,
            artifact_sha256=artifact_sha256,
        )
        if manifest_raw != _canonical(expected_manifest):
            raise ForagerMatchedFinalAnalysisError(
                "final-analysis manifest differs from exact replay"
            )
        if (
            _directory_inventory(
                seal_directory,
                expected_files=_SEAL_ARTIFACTS,
                label="final-analysis seal subtree",
            )
            != seal_inventory
            or _directory_inventory(
                evaluation_directory,
                expected_files=_EVALUATION_ARTIFACTS,
                label="final-analysis evaluation subtree",
            )
            != evaluation_inventory
            or _directory_inventory(
                analysis_directory,
                expected_files=_ANALYSIS_ARTIFACTS,
                label="final-analysis analysis subtree",
            )
            != analysis_inventory
            or _directory_inventory(
                opened,
                expected_files=_ROOT_FILES,
                expected_directories=("seal", "evaluation", "analysis"),
                label="final-analysis bundle root",
            )
            != root_inventory
        ):
            raise ForagerMatchedFinalAnalysisError("final-analysis bundle changed during replay")
        for directory, label in (
            (seal_directory, "final-analysis seal subtree"),
            (evaluation_directory, "final-analysis evaluation subtree"),
            (analysis_directory, "final-analysis analysis subtree"),
            (opened, "final-analysis bundle root"),
        ):
            seal._assert_open_directory_path(directory, label)
        return ContentVerifiedFinalAnalysisBundle(
            output_root=opened.path,
            manifest=cast(
                Mapping[str, Any],
                seal._freeze_json(expected_manifest),
            ),
            seal_content=seal_content,
            evaluation_score_evidence=evaluation_scores,
            evaluation_verification_request=evaluation_request,
            open_bindings_cache=open_bindings,
            evaluation_bindings_cache=evaluation_bindings,
            analysis_runtime_source=analysis_runtime_source,
            contract=contract,
            result=result,
        )
    finally:
        for child in reversed(opened_children):
            os.close(child.descriptor)


def load_final_analysis_content(
    output_root: Path,
) -> ContentVerifiedFinalAnalysisBundle:
    """Replay campaign-root-independent content under the matching live code/runtime."""
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    opened = seal._open_stable_directory(output_root, "final-analysis bundle root")
    try:
        return _load_from_open_root(opened)
    finally:
        os.close(opened.descriptor)


def _prospective_output(
    inputs: _CreationInputs,
    qualification_root: Path,
    evaluation_campaign_root: Path,
    output_root: Path,
) -> Path:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not output_root.name or output_root.name in {".", ".."}:
        raise ForagerMatchedFinalAnalysisError("final-analysis output name is unsafe")
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedFinalAnalysisError("final-analysis output root already exists")
    try:
        prospective = output_root.resolve(strict=False)
        protected = (
            qualification_root.resolve(strict=True),
            inputs.seal_content.output_root.resolve(strict=True),
            evaluation_campaign_root.resolve(strict=True),
        )
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedFinalAnalysisError(
            "final-analysis output or input path cannot be resolved"
        ) from exc
    if any(campaign._paths_overlap(prospective, path) for path in protected):
        raise ForagerMatchedFinalAnalysisError("final-analysis output overlaps an input root")
    return prospective


def _sync_files(
    opened: seal._OpenDirectory,
    names: tuple[str, ...],
    label: str,
) -> None:
    initial = _directory_inventory(
        opened,
        expected_files=names,
        label=label,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(initial, key=lambda value: value.encode("utf-8")):
        descriptor = os.open(name, flags, dir_fd=opened.descriptor)
        try:
            expected = initial[name]
            before = os.fstat(descriptor)
            if seal._stat_identity(before) != expected:
                raise ForagerMatchedFinalAnalysisError(f"{label} entry changed before fsync")
            os.fsync(descriptor)
            current = os.stat(name, dir_fd=opened.descriptor, follow_symlinks=False)
            if (
                seal._stat_identity(os.fstat(descriptor)) != expected
                or seal._stat_identity(current) != expected
            ):
                raise ForagerMatchedFinalAnalysisError(f"{label} entry changed during fsync")
        finally:
            os.close(descriptor)
    os.fsync(opened.descriptor)
    if _directory_inventory(opened, expected_files=names, label=label) != initial:
        raise ForagerMatchedFinalAnalysisError(f"{label} changed during fsync")
    seal._assert_open_directory_path(opened, label)


def _sync_staged_tree(staging: seal._OpenDirectory) -> None:
    children = (
        ("seal", _SEAL_ARTIFACTS),
        ("evaluation", _EVALUATION_ARTIFACTS),
        ("analysis", _ANALYSIS_ARTIFACTS),
    )
    for name, artifacts in children:
        opened = _open_child(staging, name, f"staged final-analysis {name} subtree")
        try:
            _sync_files(
                opened,
                artifacts,
                f"staged final-analysis {name} subtree",
            )
        finally:
            os.close(opened.descriptor)
    root_inventory = _directory_inventory(
        staging,
        expected_files=_ROOT_FILES,
        expected_directories=("seal", "evaluation", "analysis"),
        label="staged final-analysis root",
    )
    for name in _pair_names(_ROOT_FILES):
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=staging.descriptor)
        try:
            os.fsync(descriptor)
            if seal._stat_identity(os.fstat(descriptor)) != root_inventory[name]:
                raise ForagerMatchedFinalAnalysisError(
                    "staged final-analysis root file changed during fsync"
                )
        finally:
            os.close(descriptor)
    os.fsync(staging.descriptor)
    if (
        _directory_inventory(
            staging,
            expected_files=_ROOT_FILES,
            expected_directories=("seal", "evaluation", "analysis"),
            label="staged final-analysis root",
        )
        != root_inventory
    ):
        raise ForagerMatchedFinalAnalysisError("staged final-analysis root changed during fsync")
    seal._assert_open_directory_path(staging, "staged final-analysis root")


def _cleanup_owned_staging(
    parent: seal._OpenDirectory,
    name: str,
    staging: seal._OpenDirectory,
) -> None:
    """Best-effort subset cleanup restricted to the exact staging inode."""
    if not seal._parent_entry_matches_open_directory(parent, name, staging):
        return

    def snapshot_allowed(
        opened: seal._OpenDirectory,
        *,
        allowed_files: set[str],
        allowed_directories: set[str],
    ) -> dict[str, os.stat_result]:
        with os.scandir(opened.descriptor) as iterator:
            entries = {
                entry.name: entry.stat(follow_symlinks=False)
                for entry in iterator
            }
        for entry_name, metadata in entries.items():
            if entry_name in allowed_directories:
                safe = stat.S_ISDIR(metadata.st_mode)
            elif entry_name in allowed_files:
                safe = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            else:
                safe = False
            if not safe:
                raise ForagerMatchedFinalAnalysisError(
                    "owned cleanup staging tree contains an unexpected entry"
                )
        for entry_name, metadata in entries.items():
            current = os.stat(
                entry_name,
                dir_fd=opened.descriptor,
                follow_symlinks=False,
            )
            if seal._stat_identity(current) != seal._stat_identity(metadata):
                raise ForagerMatchedFinalAnalysisError(
                    "owned cleanup staging entry changed before removal"
                )
        return entries

    try:
        children = (
            ("seal", _SEAL_ARTIFACTS),
            ("evaluation", _EVALUATION_ARTIFACTS),
            ("analysis", _ANALYSIS_ARTIFACTS),
        )
        root_entries = snapshot_allowed(
            staging,
            allowed_files=_pair_names(_ROOT_FILES),
            allowed_directories={child_name for child_name, _artifacts in children},
        )
        for child_name, artifacts in children:
            child_metadata = root_entries.get(child_name)
            if child_metadata is None:
                continue
            child = seal._open_stable_directory_at(
                staging,
                child_name,
                staging.path / child_name,
                "owned cleanup subtree",
            )
            try:
                if child.inode_identity != seal._inode_identity(child_metadata):
                    raise ForagerMatchedFinalAnalysisError(
                        "owned cleanup child inode changed before removal"
                    )
                child_entries = snapshot_allowed(
                    child,
                    allowed_files=_pair_names(artifacts),
                    allowed_directories=set(),
                )
                for entry_name in sorted(
                    child_entries,
                    key=lambda value: value.encode("utf-8"),
                ):
                    os.unlink(entry_name, dir_fd=child.descriptor)
                with os.scandir(child.descriptor) as iterator:
                    if next(iterator, None) is not None:
                        raise ForagerMatchedFinalAnalysisError(
                            "owned cleanup child is not empty after removal"
                        )
                if not seal._parent_entry_matches_open_directory(
                    staging,
                    child_name,
                    child,
                ):
                    raise ForagerMatchedFinalAnalysisError(
                        "owned cleanup child name changed before rmdir"
                    )
                os.rmdir(child_name, dir_fd=staging.descriptor)
            finally:
                os.close(child.descriptor)
        for entry_name in sorted(
            set(root_entries) & _pair_names(_ROOT_FILES),
            key=lambda value: value.encode("utf-8"),
        ):
            current = os.stat(
                entry_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if seal._stat_identity(current) != seal._stat_identity(root_entries[entry_name]):
                raise ForagerMatchedFinalAnalysisError(
                    "owned cleanup root file changed before removal"
                )
            os.unlink(entry_name, dir_fd=staging.descriptor)
        with os.scandir(staging.descriptor) as iterator:
            if next(iterator, None) is not None:
                raise ForagerMatchedFinalAnalysisError(
                    "owned cleanup staging root is not empty after removal"
                )
        if seal._parent_entry_matches_open_directory(parent, name, staging):
            os.rmdir(name, dir_fd=parent.descriptor)
    except (OSError, ValueError):
        return


def _publish_bundle(
    *,
    inputs: _CreationInputs,
    analysis_artifacts: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    qualification_root: Path,
    requested: Path,
    prospective: Path,
) -> ContentVerifiedFinalAnalysisBundle:
    parent, destination = seal._open_destination_parent(requested, prospective)
    staging_name: str | None = None
    staging: seal._OpenDirectory | None = None
    published = False
    try:
        if any(
            campaign._paths_overlap(destination, protected)
            for protected in (
                qualification_root.resolve(strict=True),
                inputs.seal_content.output_root,
                inputs.completed.output_root,
            )
        ):
            raise ForagerMatchedFinalAnalysisError(
                "final-analysis destination overlaps an input after parent resolution"
            )
        staging_name, staging = seal._create_owned_staging(parent, destination)
        subtrees: tuple[tuple[str, Mapping[str, bytes]], ...] = (
            ("seal", inputs.seal_artifacts),
            ("evaluation", inputs.evaluation_artifacts),
            ("analysis", analysis_artifacts),
        )
        for child_name, artifacts in subtrees:
            os.mkdir(child_name, 0o700, dir_fd=staging.descriptor)
            child = _open_child(staging, child_name, f"staged {child_name} subtree")
            try:
                for artifact_name, raw in artifacts.items():
                    seal._write_pair_at(child, artifact_name, raw)
            finally:
                os.close(child.descriptor)
        seal._write_pair_at(staging, "manifest.json", _canonical(manifest))
        verified = _load_from_open_root(staging)
        statistics_manifest = cast(Mapping[str, Any], manifest["statistics"])
        if (
            verified.contract.payload_sha256 != statistics_manifest["contract_payload_sha256"]
            or verified.result.payload_sha256 != statistics_manifest["result_payload_sha256"]
        ):
            raise ForagerMatchedFinalAnalysisError("staged final-analysis typed replay drifted")
        _sync_staged_tree(staging)
        seal._publish_verified_no_replace(
            parent,
            staging,
            staging_name,
            destination.name,
            destination,
        )
        published = True
        if not seal._parent_entry_matches_open_directory(
            parent,
            destination.name,
            staging,
        ):
            raise ForagerMatchedFinalAnalysisError(
                "published final-analysis destination no longer names the verified inode"
            )
        published_root = seal._OpenDirectory(
            path=destination,
            descriptor=staging.descriptor,
            inode_identity=staging.inode_identity,
        )
        published_content = _load_from_open_root(published_root)
        if (
            published_content.manifest != verified.manifest
            or published_content.contract != verified.contract
            or published_content.result != verified.result
            or not seal._parent_entry_matches_open_directory(
                parent,
                destination.name,
                staging,
            )
        ):
            raise ForagerMatchedFinalAnalysisError(
                "published final-analysis content or inode differs from staging replay"
            )
        seal._assert_open_directory_path(parent, "final-analysis output parent")
        return published_content
    except seal.PublishedSealUncertainError as exc:
        raise PublishedFinalAnalysisUncertainError(
            destination,
            "publication durability or inode verification failed",
        ) from exc
    except BaseException as exc:
        source_still_names_staging = (
            staging_name is not None
            and staging is not None
            and seal._parent_entry_matches_open_directory(
                parent,
                staging_name,
                staging,
            )
        )
        destination_names_staging = (
            staging is not None
            and seal._parent_entry_matches_open_directory(
                parent,
                destination.name,
                staging,
            )
        )
        if published or destination_names_staging or (
            staging is not None and not source_still_names_staging
        ):
            raise PublishedFinalAnalysisUncertainError(
                destination,
                "post-publication content or inode replay failed",
            ) from exc
        if staging_name is not None and staging is not None:
            _cleanup_owned_staging(parent, staging_name, staging)
        raise
    finally:
        if staging is not None:
            os.close(staging.descriptor)
        os.close(parent.descriptor)


def _resolve_evaluation(
    request: executor.VerificationRequest,
    resolver: executor.TrustResolver,
    *,
    expected_trust_anchor_identity: str,
    expected_verification_subject_sha256: str,
) -> evidence.AuthenticatedEvidenceBindings:
    _validate_request_pin(
        request,
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_verification_subject_sha256),
        label="evaluation",
    )
    try:
        return executor.resolve_authenticated_bindings(request, resolver)
    except Exception as exc:
        raise ForagerMatchedFinalAnalysisError(
            f"evaluation trust resolution failed: {exc}"
        ) from exc


def create_forager_matched_final_analysis_bundle(
    qualification_root: Path,
    seal_root: Path,
    evaluation_campaign_root: Path,
    output_root: Path,
    *,
    open_resolver: executor.TrustResolver,
    evaluation_resolver: executor.TrustResolver,
    expected_open_trust_anchor_identity: str,
    expected_open_verification_subject_sha256: str,
    expected_evaluation_trust_anchor_identity: str,
    expected_evaluation_verification_subject_sha256: str,
    expected_seal_manifest_payload_sha256: str,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> ContentVerifiedFinalAnalysisBundle:
    """Resolve two subjects, replay content, analyze, and atomically publish."""
    if not all(
        isinstance(path, Path)
        for path in (
            qualification_root,
            seal_root,
            evaluation_campaign_root,
            output_root,
        )
    ):
        raise TypeError("all final-analysis roots must be Paths")
    inputs = _load_creation_inputs(
        qualification_root,
        seal_root,
        evaluation_campaign_root,
        runtime=runtime,
        runner=runner,
    )
    open_request = inputs.seal_content.open_verification_request
    evaluation_request = inputs.completed.verification_request
    _validate_request_pin(
        open_request,
        expected_trust_anchor_identity=expected_open_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_open_verification_subject_sha256),
        label="open",
    )
    _validate_request_pin(
        evaluation_request,
        expected_trust_anchor_identity=expected_evaluation_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_evaluation_verification_subject_sha256),
        label="evaluation",
    )
    expected_manifest = _require_sha256(
        expected_seal_manifest_payload_sha256,
        "expected seal manifest payload",
    )
    if inputs.seal_content.manifest.get("payload_sha256") != expected_manifest:
        raise ForagerMatchedFinalAnalysisError("seal manifest differs from caller-pinned payload")
    prospective = _prospective_output(
        inputs,
        qualification_root,
        evaluation_campaign_root,
        output_root,
    )

    try:
        open_bindings = seal.authenticate_forager_matched_seal_bundle(
            inputs.seal_content,
            resolver=open_resolver,
            expected_trust_anchor_identity=expected_open_trust_anchor_identity,
            expected_seal_manifest_sha256=expected_manifest,
            expected_verification_subject_sha256=(expected_open_verification_subject_sha256),
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedFinalAnalysisError(f"open trust resolution failed: {exc}") from exc
    replay_live = inputs.completed.live_runtime
    replayed_after_open = _load_creation_inputs(
        qualification_root,
        seal_root,
        evaluation_campaign_root,
        runtime=replay_live.executable,
        runner=_frozen_live_runtime_replay_runner(replay_live),
    )
    _assert_same_creation_inputs(
        inputs,
        replayed_after_open,
        "during open trust resolution",
    )
    inputs = replayed_after_open
    evaluation_request = inputs.completed.verification_request
    # The evaluation resolver is intentionally the final external callback before computation.
    evaluation_bindings = _resolve_evaluation(
        evaluation_request,
        evaluation_resolver,
        expected_trust_anchor_identity=expected_evaluation_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_evaluation_verification_subject_sha256),
    )
    replay_live = inputs.completed.live_runtime
    replayed_after_evaluation = _load_creation_inputs(
        qualification_root,
        seal_root,
        evaluation_campaign_root,
        runtime=replay_live.executable,
        runner=_frozen_live_runtime_replay_runner(replay_live),
    )
    _assert_same_creation_inputs(
        inputs,
        replayed_after_evaluation,
        "during evaluation trust resolution",
    )
    inputs = replayed_after_evaluation
    analysis_runtime_source = _analysis_runtime_source_identity()
    evaluation_request = inputs.completed.verification_request
    (
        evaluation_scores,
        replayed_evaluation_request,
        evaluation_completion,
        evaluation_schedule,
    ) = _validate_evaluation_snapshot(
        inputs.evaluation_artifacts,
        inputs.seal_content,
        evaluation_bindings,
    )
    if replayed_evaluation_request != evaluation_request:
        raise ForagerMatchedFinalAnalysisError(
            "captured evaluation request differs from completed replay"
        )
    contract, result = _build_contract_and_result(
        inputs.seal_content,
        evaluation_scores,
        open_bindings=open_bindings,
        evaluation_bindings=evaluation_bindings,
    )
    if not _json_exact_equal(
        _analysis_runtime_source_identity(),
        analysis_runtime_source,
    ):
        raise ForagerMatchedFinalAnalysisError(
            "analysis source or runtime changed during final statistics"
        )
    analysis_artifacts: dict[str, bytes] = {
        "analysis-runtime-source.json": _canonical(analysis_runtime_source),
        "evaluation-authenticated-bindings-cache.json": executor.canonical_json_bytes(
            evaluation_bindings.to_dict()
        ),
        "statistics-contract.json": _canonical(cast(Mapping[str, Any], contract.to_payload())),
        "statistics-result.json": result.canonical_json(),
    }
    artifact_sha256 = _artifact_digest_map(
        inputs.seal_artifacts,
        inputs.evaluation_artifacts,
        analysis_artifacts,
    )
    manifest = _build_manifest(
        seal_content=inputs.seal_content,
        evaluation_scores=evaluation_scores,
        evaluation_request=evaluation_request,
        evaluation_completion=evaluation_completion,
        evaluation_schedule=evaluation_schedule,
        open_bindings=open_bindings,
        evaluation_bindings=evaluation_bindings,
        analysis_runtime_source=analysis_runtime_source,
        contract=contract,
        result=result,
        artifact_sha256=artifact_sha256,
    )
    return _publish_bundle(
        inputs=inputs,
        analysis_artifacts=analysis_artifacts,
        manifest=manifest,
        qualification_root=qualification_root,
        requested=output_root,
        prospective=prospective,
    )


def publish_authenticated_final_analysis(
    qualification_root: Path,
    seal_root: Path,
    evaluation_campaign_root: Path,
    output_root: Path,
    *,
    open_resolver: executor.TrustResolver,
    evaluation_resolver: executor.TrustResolver,
    expected_open_trust_anchor_identity: str,
    expected_open_verification_subject_sha256: str,
    expected_evaluation_trust_anchor_identity: str,
    expected_evaluation_verification_subject_sha256: str,
    expected_seal_manifest_payload_sha256: str,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> ContentVerifiedFinalAnalysisBundle:
    """Alias with an action-oriented name for the atomic creation workflow."""
    return create_forager_matched_final_analysis_bundle(
        qualification_root,
        seal_root,
        evaluation_campaign_root,
        output_root,
        open_resolver=open_resolver,
        evaluation_resolver=evaluation_resolver,
        expected_open_trust_anchor_identity=expected_open_trust_anchor_identity,
        expected_open_verification_subject_sha256=(expected_open_verification_subject_sha256),
        expected_evaluation_trust_anchor_identity=(expected_evaluation_trust_anchor_identity),
        expected_evaluation_verification_subject_sha256=(
            expected_evaluation_verification_subject_sha256
        ),
        expected_seal_manifest_payload_sha256=(expected_seal_manifest_payload_sha256),
        runtime=runtime,
        runner=runner,
    )


def authenticate_final_analysis_content(
    value: ContentVerifiedFinalAnalysisBundle | Path,
    *,
    open_resolver: executor.TrustResolver,
    evaluation_resolver: executor.TrustResolver,
    expected_open_trust_anchor_identity: str,
    expected_open_verification_subject_sha256: str,
    expected_evaluation_trust_anchor_identity: str,
    expected_evaluation_verification_subject_sha256: str,
    expected_seal_manifest_payload_sha256: str,
    expected_analysis_manifest_payload_sha256: str,
) -> FreshFinalAnalysisBindings:
    """Freshly resolve both copied subjects; returned bindings remain plain data."""
    if isinstance(value, ContentVerifiedFinalAnalysisBundle):
        content_root = value.output_root
    elif isinstance(value, Path):
        content_root = value
    else:
        raise TypeError("value must be a ContentVerifiedFinalAnalysisBundle or Path")
    content = load_final_analysis_content(content_root)
    if isinstance(value, ContentVerifiedFinalAnalysisBundle) and (
        content.manifest != value.manifest
        or content.result != value.result
        or content.contract != value.contract
    ):
        raise ForagerMatchedFinalAnalysisError(
            "supplied final-analysis object differs from persisted content"
        )
    analysis_manifest_pin = _require_sha256(
        expected_analysis_manifest_payload_sha256,
        "expected analysis manifest payload",
    )
    if content.manifest.get("payload_sha256") != analysis_manifest_pin:
        raise ForagerMatchedFinalAnalysisError(
            "analysis manifest differs from caller-pinned payload"
        )
    seal_manifest_pin = _require_sha256(
        expected_seal_manifest_payload_sha256,
        "expected seal manifest payload",
    )
    if content.seal_content.manifest.get("payload_sha256") != seal_manifest_pin:
        raise ForagerMatchedFinalAnalysisError("seal manifest differs from caller-pinned payload")
    _validate_request_pin(
        content.seal_content.open_verification_request,
        expected_trust_anchor_identity=expected_open_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_open_verification_subject_sha256),
        label="open",
    )
    _validate_request_pin(
        content.evaluation_verification_request,
        expected_trust_anchor_identity=expected_evaluation_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_evaluation_verification_subject_sha256),
        label="evaluation",
    )
    try:
        open_bindings = seal.authenticate_forager_matched_seal_bundle(
            content.output_root / "seal",
            resolver=open_resolver,
            expected_trust_anchor_identity=expected_open_trust_anchor_identity,
            expected_seal_manifest_sha256=seal_manifest_pin,
            expected_verification_subject_sha256=(expected_open_verification_subject_sha256),
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedFinalAnalysisError(f"open trust resolution failed: {exc}") from exc
    evaluation_bindings = _resolve_evaluation(
        content.evaluation_verification_request,
        evaluation_resolver,
        expected_trust_anchor_identity=expected_evaluation_trust_anchor_identity,
        expected_verification_subject_sha256=(expected_evaluation_verification_subject_sha256),
    )
    if (
        open_bindings != content.open_bindings_cache
        or evaluation_bindings != content.evaluation_bindings_cache
    ):
        raise ForagerMatchedFinalAnalysisError(
            "fresh resolver result differs from persisted cache content"
        )
    replayed = load_final_analysis_content(content.output_root)
    if replayed.manifest != content.manifest:
        raise ForagerMatchedFinalAnalysisError(
            "final-analysis content changed during fresh authentication"
        )
    return FreshFinalAnalysisBindings(
        open_bindings=open_bindings,
        evaluation_bindings=evaluation_bindings,
    )


__all__ = [
    "ContentVerifiedFinalAnalysisBundle",
    "ForagerMatchedFinalAnalysisError",
    "FreshFinalAnalysisBindings",
    "MATCHED_FINAL_ANALYSIS_MANIFEST_SCHEMA_VERSION",
    "MATCHED_FINAL_ANALYSIS_RUNTIME_SOURCE_SCHEMA_VERSION",
    "PublishedFinalAnalysisUncertainError",
    "authenticate_final_analysis_content",
    "create_forager_matched_final_analysis_bundle",
    "load_final_analysis_content",
    "publish_authenticated_final_analysis",
]
