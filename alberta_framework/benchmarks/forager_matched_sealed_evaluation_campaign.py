"""Resumable content-only execution of the matched-current sealed evaluation block.

The sealed block is the held-out stage of the matched-current Forager
protocol: the exact 6-candidate by 30-seed (180-cell) grid fixed by the seal
bundle from :mod:`alberta_framework.benchmarks.forager_matched_seal` — the
boundary between completed open tuning and held-out evaluation.  The six
candidates are the four the frozen selection advanced plus two fixed
descriptive references; the 30 seeds are the held-out evaluation seeds of
:mod:`alberta_framework.benchmarks.forager_matched_open_protocol`, disjoint
from the open-stage tuning seeds.

Mutating entry points freshly authenticate the caller-pinned seal immediately before
publication or execution.  Persisted seal bindings are cache content only and are never
accepted as authority.  Read-only verification, status, and completed-loading entry points
replay content without making an authentication or promotion claim.

The host keeps raw OCI exports opaque.  Execution and scoring reuse the append-only campaign
engine; only the frozen scorer inside the qualified OCI image may inspect reward arrays.
The cooperative campaign lock and top-level inode guard protect normal writers and path swaps;
they do not claim recursive tamper-proofing against a noncooperative same-UID local writer.
Independent authentication of the final verification subject remains the cryptographic authority
boundary for any performance or promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as schedule
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import forager_matched_trust as trust
from alberta_framework.benchmarks.forager_matched_evidence import MatchedScoreEvidence

MATCHED_SEALED_EVALUATION_CAMPAIGN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_evaluation_campaign.v2"
)
MATCHED_SEALED_EVALUATION_COMPLETION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_evaluation_completion.v2"
)

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_CANDIDATES: Final = 6
_EXPECTED_SEEDS: Final = 30
_EXPECTED_CELLS: Final = 180
_IMMUTABLE_ARTIFACTS: Final = (
    "sealed-protocol.json",
    "sealed-transition.json",
    "seal-manifest.json",
    "candidate-universe.json",
    "execution-plan.json",
    "source-manifest.json",
    "executor-manifest.json",
    "qualification-manifest.json",
    "execution-schedule.json",
    "live-runtime.json",
    "campaign.json",
)
_FINAL_ARTIFACTS: Final = campaign._FINAL_ARTIFACTS


class ForagerMatchedSealedEvaluationCampaignError(ValueError):
    """A sealed-evaluation input, root, transition, or completion failed closed."""


class PublishedSealedEvaluationCampaignUncertainError(
    ForagerMatchedSealedEvaluationCampaignError
):
    """Publication occurred, but final replay could not establish a safe handoff."""

    def __init__(self, destination: Path, detail: str) -> None:
        self.destination = destination
        super().__init__(f"campaign published at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class _SealedInputs:
    rebuilt: campaign._RebuiltInputs
    seal_content: seal.ContentVerifiedSealBundle
    transition: protocol.SealedProtocolValidation
    seal_manifest_payload_sha256: str
    open_verification_subject_sha256: str


@dataclass(frozen=True, slots=True)
class _SealedContext:
    engine: campaign._CampaignContext
    inputs: _SealedInputs


@dataclass(frozen=True, slots=True)
class _CampaignRootGuard:
    opened: seal._OpenDirectory
    campaign_file_identity: tuple[int, ...]
    runs_identity: tuple[int, ...]
    completions_identity: tuple[int, ...]

    def check(self) -> None:
        seal._assert_open_directory_path(
            self.opened,
            "sealed evaluation campaign root",
        )
        try:
            inside = os.stat(
                "campaign.json",
                dir_fd=self.opened.descriptor,
                follow_symlinks=False,
            )
            outside = os.lstat(self.opened.path / "campaign.json")
        except OSError as exc:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation campaign lock path changed"
            ) from exc
        if (
            not stat.S_ISREG(inside.st_mode)
            or inside.st_nlink != 1
            or _file_identity(inside) != self.campaign_file_identity
            or _file_identity(outside) != self.campaign_file_identity
        ):
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation campaign lock inode changed"
            )
        for name, expected_identity in (
            ("runs", self.runs_identity),
            ("completions", self.completions_identity),
        ):
            try:
                inside_directory = os.stat(
                    name,
                    dir_fd=self.opened.descriptor,
                    follow_symlinks=False,
                )
                outside_directory = os.lstat(self.opened.path / name)
            except OSError as exc:
                raise ForagerMatchedSealedEvaluationCampaignError(
                    f"sealed evaluation {name} root changed"
                ) from exc
            if (
                not stat.S_ISDIR(inside_directory.st_mode)
                or _directory_identity(inside_directory) != expected_identity
                or _directory_identity(outside_directory) != expected_identity
            ):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    f"sealed evaluation {name} root inode changed"
                )


_SummaryBuilder = Callable[
    [
        campaign._CampaignContext,
        executor.MatchedExecutionReceiptIndex,
        MatchedScoreEvidence,
        executor.VerificationRequest,
    ],
    Mapping[str, Any],
]
_SummaryValidator = Callable[
    [
        campaign._CampaignContext,
        executor.MatchedExecutionReceiptIndex,
        MatchedScoreEvidence,
        executor.VerificationRequest,
        Mapping[str, Any],
    ],
    None,
]


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


@contextmanager
def _locked_campaign_root(
    root: Path,
    *,
    exclusive: bool,
) -> Iterator[Callable[[], None]]:
    opened = seal._open_stable_directory(root, "sealed evaluation campaign root")
    try:
        try:
            campaign_file = os.stat(
                "campaign.json",
                dir_fd=opened.descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation campaign lock is missing"
            ) from exc
        if not stat.S_ISREG(campaign_file.st_mode) or campaign_file.st_nlink != 1:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation campaign lock is unsafe"
            )
        try:
            runs = os.stat("runs", dir_fd=opened.descriptor, follow_symlinks=False)
            completions = os.stat(
                "completions",
                dir_fd=opened.descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation dynamic roots are missing"
            ) from exc
        if not stat.S_ISDIR(runs.st_mode) or not stat.S_ISDIR(completions.st_mode):
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation dynamic roots are unsafe"
            )
        guard = _CampaignRootGuard(
            opened,
            _file_identity(campaign_file),
            _directory_identity(runs),
            _directory_identity(completions),
        )
        with campaign._campaign_lock(opened.path, exclusive=exclusive):
            guard.check()
            try:
                yield guard.check
            except BaseException as body_error:
                try:
                    guard.check()
                except BaseException as guard_error:
                    raise guard_error from body_error
                raise
            else:
                guard.check()
    finally:
        os.close(opened.descriptor)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedSealedEvaluationCampaignError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _qualification_bundle_manifest_sha256(
    bundle: qualification.MatchedCurrentQualificationBundle,
) -> str:
    """Return the loader-preserved digest after replaying its exact byte identity."""
    digest = _require_sha256(
        getattr(bundle, "manifest_sha256", None),
        "qualification bundle manifest_sha256",
    )
    raw = getattr(bundle, "manifest_bytes", None)
    manifest = getattr(bundle, "manifest", None)
    if type(raw) is not bytes or not isinstance(manifest, Mapping):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "qualification bundle does not preserve its exact manifest bytes"
        )
    try:
        canonical = qualification._canonical_json_bytes(manifest)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "qualification bundle manifest is not canonical JSON"
        ) from exc
    if raw != canonical or hashlib.sha256(raw).hexdigest() != digest:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "qualification bundle manifest bytes differ from their loader digest"
        )
    return digest


def _open_qualification_manifest_sha256(
    bundle: qualification.MatchedCurrentQualificationBundle,
    content: seal.ContentVerifiedSealBundle,
) -> str:
    """Require one qualification identity across the root and authenticated open seal."""
    bundle_digest = _qualification_bundle_manifest_sha256(bundle)
    open_campaign = content.manifest.get("open_campaign")
    if not isinstance(open_campaign, Mapping):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "seal manifest omits its open campaign qualification binding"
        )
    recorded_bindings = content.recorded_bindings_cache
    if not isinstance(recorded_bindings, Mapping):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "seal recorded bindings cache is not an object"
        )
    digests = (
        bundle_digest,
        _require_sha256(
            open_campaign.get("qualification_manifest_sha256"),
            "seal manifest qualification_manifest_sha256",
        ),
        _require_sha256(
            getattr(content.open_score_evidence, "qualification_manifest_sha256", None),
            "open score-evidence qualification_manifest_sha256",
        ),
        _require_sha256(
            getattr(content.open_verification_request, "qualification_manifest_sha256", None),
            "open verification-request qualification_manifest_sha256",
        ),
        _require_sha256(
            recorded_bindings.get("qualification_manifest_sha256"),
            "open authenticated-bindings qualification_manifest_sha256",
        ),
    )
    if any(value != bundle_digest for value in digests[1:]):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation qualification manifest differs from authenticated open closure"
        )
    return bundle_digest


def _sealed_input_qualification_manifest_sha256(inputs: _SealedInputs) -> str:
    """Require the sealed execution plan to carry the exact authenticated open digest."""
    digest = _open_qualification_manifest_sha256(
        inputs.rebuilt.bundle,
        inputs.seal_content,
    )
    plan = inputs.rebuilt.plan
    plan_payload = plan.payload
    executor_manifest = plan.executor_manifest
    if (
        plan.qualification_manifest_sha256 != digest
        or plan_payload.get("qualification_manifest_sha256") != digest
        or executor_manifest.get("qualification_manifest_sha256") != digest
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed execution plan qualification-manifest closure drifted"
        )
    return digest


def _completed_qualification_manifest_sha256(
    inputs: _SealedInputs,
    context: campaign._CampaignContext,
    score_evidence: MatchedScoreEvidence,
    request: executor.VerificationRequest,
) -> str:
    """Require the evaluation score/request closure to retain the sealed-plan digest."""
    digest = _sealed_input_qualification_manifest_sha256(inputs)
    context_plan = context.rebuilt.plan
    values = (
        _qualification_bundle_manifest_sha256(context.rebuilt.bundle),
        getattr(context_plan, "qualification_manifest_sha256", None),
        context_plan.payload.get("qualification_manifest_sha256"),
        context_plan.executor_manifest.get("qualification_manifest_sha256"),
        getattr(score_evidence, "qualification_manifest_sha256", None),
        getattr(request, "qualification_manifest_sha256", None),
    )
    if any(value != digest for value in values):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation score/request qualification-manifest closure drifted"
        )
    return digest


def _validate_mutation_pins(
    *,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_payload_sha256: str,
    expected_open_verification_subject_sha256: str,
) -> None:
    if type(expected_trust_anchor_identity) is not str or not (
        expected_trust_anchor_identity
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "expected_trust_anchor_identity must be a non-empty string"
        )
    _require_sha256(
        expected_seal_manifest_payload_sha256,
        "expected_seal_manifest_payload_sha256",
    )
    _require_sha256(
        expected_open_verification_subject_sha256,
        "expected_open_verification_subject_sha256",
    )


def _seal_manifest_payload_sha256(
    content: seal.ContentVerifiedSealBundle,
) -> str:
    return _require_sha256(
        content.manifest.get("payload_sha256"),
        "seal manifest payload_sha256",
    )


def _validate_input_roots(
    qualification_root: Path,
    seal_root: Path,
) -> tuple[Path, Path]:
    if not isinstance(qualification_root, Path) or not isinstance(seal_root, Path):
        raise TypeError("qualification_root and seal_root must be Paths")
    qualified = campaign._regular_directory(
        qualification_root,
        "qualification output root",
    )
    sealed = campaign._regular_directory(seal_root, "seal bundle root")
    if campaign._paths_overlap(qualified, sealed):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "qualification and seal roots overlap"
        )
    return qualified, sealed


def _rebuild_sealed_inputs(
    qualification_root: Path,
    seal_root: Path,
) -> _SealedInputs:
    qualified, sealed_root = _validate_input_roots(qualification_root, seal_root)
    bundle = qualification.load_matched_current_qualification_bundle(qualified)
    content = seal.load_forager_matched_seal_bundle_content(sealed_root)
    rebuilt_open = open_protocol.build_forager_matched_open_protocol(
        runtime=bundle.runtime_qualification,
        candidate_qualifications=bundle.candidate_qualifications,
    )
    if rebuilt_open.to_dict() != content.open_protocol.to_dict():
        raise ForagerMatchedSealedEvaluationCampaignError(
            "seal open protocol differs from the qualification reconstruction"
        )
    qualification_digest = _open_qualification_manifest_sha256(bundle, content)
    if tuple(bundle.candidate_assets) != open_protocol.MATCHED_CURRENT_CANDIDATE_IDS:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "qualification candidate asset order drifted"
        )
    verification = universe.verify_matched_current_candidate_universe_sources(
        _REPOSITORY_ROOT
    )
    if verification.candidate_universe_sha256 != (
        content.sealed_protocol.selection_plan.candidate_universe_sha256
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "candidate-universe source verification drifted"
        )
    try:
        transition = protocol.validate_sealed_protocol_transition(
            rebuilt_open,
            content.sealed_protocol,
            content.selection_result,
            content.selection_result.selection_result_sha256,
        )
        descriptor = schedule.build_sealed_transition_descriptor(
            content.sealed_protocol,
            transition,
        )
        descriptor_sha256 = schedule.canonical_sealed_transition_descriptor_sha256(
            content.sealed_protocol,
            transition,
        )
    except (
        protocol.ForagerMatchedProtocolError,
        schedule.ForagerMatchedEvaluationCampaignError,
    ) as exc:
        raise ForagerMatchedSealedEvaluationCampaignError(
            f"sealed transition reconstruction failed: {exc}"
        ) from exc
    if (
        campaign.canonical_json_bytes(descriptor)
        != campaign.canonical_json_bytes(content.sealed_transition)
        or descriptor_sha256 != content.sealed_transition_sha256
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "seal transition descriptor differs from typed replay"
        )
    evaluation_schedule = schedule.build_sealed_evaluation_schedule(
        content.sealed_protocol,
        transition,
    )
    schedule.parse_sealed_evaluation_schedule(
        campaign.canonical_json_bytes(evaluation_schedule),
        sealed_protocol=content.sealed_protocol,
        transition=transition,
        expected_schedule_sha256=cast(str, evaluation_schedule["schedule_sha256"]),
    )
    candidate_ids = transition.evaluation_candidate_ids
    if (
        len(candidate_ids) != _EXPECTED_CANDIDATES
        or len(content.sealed_protocol.active_seeds) != _EXPECTED_SEEDS
        or len(cast(list[Any], evaluation_schedule["cells"])) != _EXPECTED_CELLS
        or tuple(evaluation_schedule["candidate_order"]) != candidate_ids
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation is not the exact 6-by-30 block"
        )
    assets = {
        candidate_id: cast(
            executor.CandidateExecutionAssets,
            bundle.candidate_assets[candidate_id],
        )
        for candidate_id in candidate_ids
    }
    plan = executor.build_execution_plan(
        content.sealed_protocol,
        assets,
        qualification_manifest_sha256=qualification_digest,
        candidate_ids=candidate_ids,
        cpu_qualification_root=bundle.cpu_qualification_root,
        rng_parity_qualification_root=bundle.rng_parity_qualification_root,
    )
    if tuple(item.candidate.candidate_id for item in plan.candidates) != candidate_ids:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed execution plan candidate order drifted"
        )
    rebuilt = campaign._RebuiltInputs(
        bundle=bundle,
        protocol=content.sealed_protocol,
        plan=plan,
        candidate_ids=candidate_ids,
        assets=assets,
        schedule=evaluation_schedule,
    )
    inputs = _SealedInputs(
        rebuilt=rebuilt,
        seal_content=content,
        transition=transition,
        seal_manifest_payload_sha256=_seal_manifest_payload_sha256(content),
        open_verification_subject_sha256=(
            content.open_verification_request.verification_subject_sha256
        ),
    )
    _sealed_input_qualification_manifest_sha256(inputs)
    return inputs


def _campaign_manifest(
    inputs: _SealedInputs,
    live: executor.LiveRuntimeIdentity,
) -> dict[str, Any]:
    rebuilt = inputs.rebuilt
    qualification_digest = _sealed_input_qualification_manifest_sha256(inputs)
    selected = inputs.transition.evaluation_candidate_ids[:4]
    fixed = inputs.transition.evaluation_candidate_ids[4:]
    return {
        "schema_version": MATCHED_SEALED_EVALUATION_CAMPAIGN_SCHEMA_VERSION,
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
            "claim_authority": (
                "independent_final_subject_authentication_required"
            ),
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "seal_manifest_payload_sha256": inputs.seal_manifest_payload_sha256,
        "open_verification_subject_sha256": (
            inputs.open_verification_subject_sha256
        ),
        "sealed_transition_sha256": inputs.seal_content.sealed_transition_sha256,
        "protocol_sha256": rebuilt.protocol.protocol_sha256,
        "candidate_universe_sha256": (
            rebuilt.protocol.selection_plan.candidate_universe_sha256
        ),
        "execution_plan_sha256": rebuilt.plan.plan_sha256,
        "source_manifest_sha256": rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": rebuilt.plan.executor_manifest_sha256,
        "qualification_manifest_sha256": qualification_digest,
        "live_runtime_identity_sha256": live.identity_sha256,
        "execution_schedule_sha256": cast(
            str,
            rebuilt.schedule["schedule_sha256"],
        ),
        "candidate_order": list(rebuilt.candidate_ids),
        "selected_candidate_ids": list(selected),
        "fixed_descriptive_candidate_ids": list(fixed),
        "active_seeds": list(rebuilt.protocol.active_seeds),
        "cell_count": len(rebuilt.candidate_ids) * len(rebuilt.protocol.active_seeds),
        "host_reward_array_access": "forbidden",
        "retention_bounds": {
            "max_attempts_per_cell": campaign._MAX_ATTEMPTS_PER_CELL,
            "max_failure_records_per_attempt": (
                campaign._MAX_FAILURE_RECORDS_PER_ATTEMPT
            ),
            "max_raw_archive_bytes": campaign._MAX_RAW_BYTES,
            "max_retained_raw_bytes_per_cell": (
                campaign._MAX_RETAINED_RAW_BYTES_PER_CELL
            ),
            "max_retained_raw_bytes_per_campaign": (
                campaign._MAX_RETAINED_RAW_BYTES_PER_CAMPAIGN
            ),
        },
        "completion_boundary": (
            "evaluation_score_evidence_and_unresolved_verification_request_only"
        ),
    }


def _completion_summary(
    inputs: _SealedInputs,
    context: campaign._CampaignContext,
    receipt_index: executor.MatchedExecutionReceiptIndex,
    score_evidence: MatchedScoreEvidence,
    request: executor.VerificationRequest,
) -> dict[str, Any]:
    qualification_digest = _completed_qualification_manifest_sha256(
        inputs,
        context,
        score_evidence,
        request,
    )
    return {
        "schema_version": MATCHED_SEALED_EVALUATION_COMPLETION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": "sealed_evaluation",
        "seal_manifest_payload_sha256": inputs.seal_manifest_payload_sha256,
        "open_verification_subject_sha256": (
            inputs.open_verification_subject_sha256
        ),
        "sealed_transition_sha256": inputs.seal_content.sealed_transition_sha256,
        "protocol_sha256": context.rebuilt.protocol.protocol_sha256,
        "qualification_manifest_sha256": qualification_digest,
        "execution_plan_sha256": context.rebuilt.plan.plan_sha256,
        "source_manifest_sha256": context.rebuilt.plan.source_manifest_sha256,
        "executor_manifest_sha256": context.rebuilt.plan.executor_manifest_sha256,
        "live_runtime_identity_sha256": context.live_runtime.identity_sha256,
        "execution_schedule_sha256": cast(
            str,
            context.rebuilt.schedule["schedule_sha256"],
        ),
        "candidate_count": len(context.rebuilt.candidate_ids),
        "seed_count": len(context.rebuilt.protocol.active_seeds),
        "completed_cell_count": (
            len(context.rebuilt.candidate_ids)
            * len(context.rebuilt.protocol.active_seeds)
        ),
        "execution_receipt_index_payload_sha256": receipt_index.payload_sha256,
        "score_evidence_sha256": score_evidence.payload_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
        "evaluation_verification_authentication_state": (
            "unresolved_external_verifier_required"
        ),
        "selection_inherited_from_seal": True,
        "sealed_protocol_inherited_from_seal": True,
        "evaluation_artifacts_created": True,
        "cached_bindings_accepted_as_authority": False,
        "content_capture_threat_boundary": {
            "campaign_tree_writers": "cooperative_lock_participants_trusted",
            "root_guard_scope": "top_level_inode_and_path_swap_detection_only",
            "noncooperative_same_uid_writers": "out_of_scope",
            "claim_authority": (
                "independent_final_subject_authentication_required"
            ),
        },
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "host_reward_array_access": "forbidden_not_performed",
    }


def _summary_builder(inputs: _SealedInputs) -> _SummaryBuilder:
    def build(
        context: campaign._CampaignContext,
        receipt_index: executor.MatchedExecutionReceiptIndex,
        score_evidence: MatchedScoreEvidence,
        request: executor.VerificationRequest,
    ) -> Mapping[str, Any]:
        return _completion_summary(
            inputs,
            context,
            receipt_index,
            score_evidence,
            request,
        )

    return build


def _summary_validator(inputs: _SealedInputs) -> _SummaryValidator:
    def validate(
        context: campaign._CampaignContext,
        receipt_index: executor.MatchedExecutionReceiptIndex,
        score_evidence: MatchedScoreEvidence,
        request: executor.VerificationRequest,
        summary: Mapping[str, Any],
    ) -> None:
        campaign._validate_completion_summary_common(
            context,
            receipt_index,
            score_evidence,
            request,
            summary,
        )
        expected = _completion_summary(
            inputs,
            context,
            receipt_index,
            score_evidence,
            request,
        )
        if campaign.canonical_json_bytes(campaign._plain(summary)) != campaign.canonical_json_bytes(
            expected
        ):
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed completion summary differs from its exact v2 schema"
            )

    return validate


def _expected_root_names() -> set[str]:
    artifacts = _IMMUTABLE_ARTIFACTS + _FINAL_ARTIFACTS
    return {
        "runs",
        "completions",
        *(name for name in artifacts),
        *(f"{name}.sha256" for name in artifacts),
    }


def _validate_root_shape(root: Path) -> None:
    actual = campaign._bounded_entry_names(
        root,
        "sealed evaluation campaign root",
        maximum=len(_expected_root_names()),
    )
    unknown = actual - _expected_root_names()
    if unknown:
        raise ForagerMatchedSealedEvaluationCampaignError(
            f"sealed evaluation root contains unknown artifacts: {sorted(unknown)}"
        )
    required = {
        "runs",
        "completions",
        *_IMMUTABLE_ARTIFACTS,
        *(f"{name}.sha256" for name in _IMMUTABLE_ARTIFACTS),
    }
    missing = required - actual
    if missing:
        raise ForagerMatchedSealedEvaluationCampaignError(
            f"sealed evaluation root is missing artifacts: {sorted(missing)}"
        )
    campaign._regular_directory(root / "runs", "sealed evaluation runs directory")
    campaign._regular_directory(
        root / "completions",
        "sealed evaluation completions directory",
    )


def _prospective_output(
    inputs: _SealedInputs,
    output_root: Path,
) -> Path:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not output_root.name or output_root.name in {".", ".."}:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation output name is unsafe"
        )
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation output root already exists"
        )
    try:
        prospective = output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation output path cannot be resolved"
        ) from exc
    protected = (
        inputs.rebuilt.bundle.output_root,
        inputs.seal_content.output_root,
    )
    if any(campaign._paths_overlap(prospective, root) for root in protected):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation output overlaps a protected input root"
        )
    return prospective


def _initial_payloads(
    inputs: _SealedInputs,
    live: executor.LiveRuntimeIdentity,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    rebuilt = inputs.rebuilt
    return (
        ("sealed-protocol.json", rebuilt.protocol.to_dict()),
        ("sealed-transition.json", inputs.seal_content.sealed_transition),
        ("seal-manifest.json", inputs.seal_content.manifest),
        (
            "candidate-universe.json",
            universe.matched_current_candidate_universe_descriptor(),
        ),
        ("execution-plan.json", rebuilt.plan.to_dict()),
        ("source-manifest.json", rebuilt.plan.source_manifest),
        ("executor-manifest.json", rebuilt.plan.executor_manifest),
        ("qualification-manifest.json", rebuilt.bundle.manifest),
        ("execution-schedule.json", rebuilt.schedule),
        ("live-runtime.json", live.unsigned_dict),
        ("campaign.json", _campaign_manifest(inputs, live)),
    )


def _sync_staging_tree(
    staging: seal._OpenDirectory,
    expected_files: set[str],
) -> None:
    seal._assert_open_directory_path(staging, "sealed evaluation staging directory")
    expected_names = {"runs", "completions", *expected_files}
    with os.scandir(staging.descriptor) as iterator:
        actual_names = {entry.name for entry in iterator}
    if actual_names != expected_names:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation staging inventory drifted"
        )
    initial_identities: dict[str, tuple[int, ...]] = {}
    for name in expected_names:
        metadata = os.stat(
            name,
            dir_fd=staging.descriptor,
            follow_symlinks=False,
        )
        if name in {"runs", "completions"}:
            safe = stat.S_ISDIR(metadata.st_mode)
        else:
            safe = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        if not safe:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation staging entry is unsafe"
            )
        initial_identities[name] = _file_identity(metadata)
    for name in ("runs", "completions"):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        descriptor = os.open(name, flags, dir_fd=staging.descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    "sealed evaluation staging directory changed"
                )
            with os.scandir(descriptor) as iterator:
                if next(iterator, None) is not None:
                    raise ForagerMatchedSealedEvaluationCampaignError(
                        "sealed evaluation dynamic staging directory is not empty"
                    )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(expected_files, key=lambda item: item.encode("utf-8")):
        expected = os.stat(name, dir_fd=staging.descriptor, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=staging.descriptor)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _file_identity(opened) != _file_identity(expected)
            ):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    "sealed evaluation staged file changed"
                )
            os.fsync(descriptor)
            current = os.stat(
                name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if _file_identity(opened) != _file_identity(current):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    "sealed evaluation staged file changed during fsync"
                )
        finally:
            os.close(descriptor)
    os.fsync(staging.descriptor)
    seal._assert_open_directory_path(staging, "sealed evaluation staging directory")
    with os.scandir(staging.descriptor) as iterator:
        final_names = {entry.name for entry in iterator}
    if final_names != expected_names:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation staging inventory changed during synchronization"
        )
    for name, expected_identity in initial_identities.items():
        current = os.stat(
            name,
            dir_fd=staging.descriptor,
            follow_symlinks=False,
        )
        if _file_identity(current) != expected_identity:
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation staging entry changed during synchronization"
            )


def _cleanup_owned_staging(
    parent: seal._OpenDirectory,
    name: str,
    staging: seal._OpenDirectory,
) -> None:
    if not seal._parent_entry_matches_open_directory(parent, name, staging):
        return
    try:
        with os.scandir(staging.descriptor) as iterator:
            entries = [
                (entry.name, entry.stat(follow_symlinks=False)) for entry in iterator
            ]
        if any(
            entry_name not in _expected_root_names()
            or (
                entry_name in {"runs", "completions"}
                and not stat.S_ISDIR(metadata.st_mode)
            )
            or (
                entry_name not in {"runs", "completions"}
                and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1)
            )
            for entry_name, metadata in entries
        ):
            return
        for entry_name, metadata in entries:
            current = os.stat(
                entry_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if _file_identity(current) != _file_identity(metadata):
                return
        for directory_name in ("runs", "completions"):
            if any(entry_name == directory_name for entry_name, _metadata in entries):
                os.rmdir(directory_name, dir_fd=staging.descriptor)
        for entry_name, _metadata in entries:
            if entry_name not in {"runs", "completions"}:
                os.unlink(entry_name, dir_fd=staging.descriptor)
        if seal._parent_entry_matches_open_directory(parent, name, staging):
            os.rmdir(name, dir_fd=parent.descriptor)
    except OSError:
        return


def _publish_initial_root(
    inputs: _SealedInputs,
    live: executor.LiveRuntimeIdentity,
    requested: Path,
    prospective: Path,
) -> Path:
    qualification_digest = _sealed_input_qualification_manifest_sha256(inputs)
    parent, destination = seal._open_destination_parent(requested, prospective)
    staging_name: str | None = None
    staging: seal._OpenDirectory | None = None
    try:
        protected = (
            inputs.rebuilt.bundle.output_root,
            inputs.seal_content.output_root,
        )
        if any(campaign._paths_overlap(destination, root) for root in protected):
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation output overlaps an input after parent resolution"
            )
        staging_name, staging = seal._create_owned_staging(parent, destination)
        for name in ("runs", "completions"):
            os.mkdir(name, 0o700, dir_fd=staging.descriptor)
        expected_files: set[str] = set()
        payloads = _initial_payloads(inputs, live)
        payload_names = tuple(name for name, _payload in payloads)
        if payload_names != _IMMUTABLE_ARTIFACTS or len(set(payload_names)) != len(
            payload_names
        ):
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation initial artifact inventory is inconsistent"
            )
        for name, payload in payloads:
            raw = (
                inputs.rebuilt.bundle.manifest_bytes
                if name == "qualification-manifest.json"
                else campaign.canonical_json_bytes(payload)
            )
            if name == "qualification-manifest.json" and (
                hashlib.sha256(raw).hexdigest() != qualification_digest
            ):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    "qualification manifest bytes changed before sealed publication"
                )
            seal._write_pair_at(staging, name, raw)
            expected_files.update((name, f"{name}.sha256"))
        _sync_staging_tree(staging, expected_files)
        seal._publish_verified_no_replace(
            parent,
            staging,
            staging_name,
            destination.name,
            destination,
        )
    except seal.PublishedSealUncertainError as exc:
        raise PublishedSealedEvaluationCampaignUncertainError(
            destination,
            "publication durability or inode verification failed",
        ) from exc
    except BaseException:
        if staging_name is not None and staging is not None:
            _cleanup_owned_staging(parent, staging_name, staging)
        raise
    finally:
        if staging is not None:
            os.close(staging.descriptor)
        os.close(parent.descriptor)
    return destination


def _load_context(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    runtime: str | Path,
    runner: executor.ProcessRunner | None,
) -> _SealedContext:
    root = campaign._regular_directory(
        output_root,
        "sealed evaluation campaign root",
    )
    inputs = _rebuild_sealed_inputs(qualification_root, seal_root)
    if any(
        campaign._paths_overlap(root, protected)
        for protected in (
            inputs.rebuilt.bundle.output_root,
            inputs.seal_content.output_root,
        )
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation root overlaps a protected input root"
        )
    live = campaign._qualify_live(inputs.rebuilt.plan, runtime, runner)
    _validate_root_shape(root)
    campaign._expect_artifact(
        root,
        "sealed-protocol.json",
        inputs.rebuilt.protocol.to_dict(),
    )
    campaign._expect_artifact(
        root,
        "sealed-transition.json",
        inputs.seal_content.sealed_transition,
    )
    campaign._expect_artifact(
        root,
        "seal-manifest.json",
        inputs.seal_content.manifest,
    )
    campaign._expect_artifact(
        root,
        "candidate-universe.json",
        universe.matched_current_candidate_universe_descriptor(),
    )
    persisted_plan, plan_file_sha256 = campaign._load_json_pair(
        root / "execution-plan.json",
        "sealed evaluation execution plan",
    )
    if (
        plan_file_sha256 != inputs.rebuilt.plan.plan_sha256
        or persisted_plan != inputs.rebuilt.plan.to_dict()
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "persisted sealed execution plan differs from rebuilt plan"
        )
    replayed = executor.parse_execution_plan(
        persisted_plan,
        protocol=inputs.rebuilt.protocol,
        assets=inputs.rebuilt.assets,
        expected_plan_sha256=inputs.rebuilt.plan.plan_sha256,
        expected_qualification_manifest_sha256=(
            inputs.rebuilt.bundle.manifest_sha256
        ),
        cpu_qualification_root=inputs.rebuilt.bundle.cpu_qualification_root,
        rng_parity_qualification_root=(
            inputs.rebuilt.bundle.rng_parity_qualification_root
        ),
    )
    if replayed.plan_sha256 != inputs.rebuilt.plan.plan_sha256:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed execution plan replay digest drifted"
        )
    campaign._expect_artifact(
        root,
        "source-manifest.json",
        inputs.rebuilt.plan.source_manifest,
    )
    campaign._expect_artifact(
        root,
        "executor-manifest.json",
        inputs.rebuilt.plan.executor_manifest,
    )
    qualification_copy_sha = campaign._expect_qualification_artifact(
        root,
        inputs.rebuilt.bundle,
    )
    if qualification_copy_sha != inputs.rebuilt.bundle.manifest_sha256:
        raise ForagerMatchedSealedEvaluationCampaignError(
            "sealed qualification manifest copy differs from authenticated digest"
        )
    persisted_schedule, _schedule_file_sha256 = campaign._load_json_pair(
        root / "execution-schedule.json",
        "sealed evaluation execution schedule",
    )
    parsed_schedule = schedule.parse_sealed_evaluation_schedule(
        campaign.canonical_json_bytes(persisted_schedule),
        sealed_protocol=inputs.rebuilt.protocol,
        transition=inputs.transition,
        expected_schedule_sha256=cast(
            str,
            inputs.rebuilt.schedule["schedule_sha256"],
        ),
    )
    if campaign.canonical_json_bytes(parsed_schedule) != campaign.canonical_json_bytes(
        inputs.rebuilt.schedule
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "persisted sealed schedule differs from rebuilt schedule"
        )
    live_payload, live_file_sha256 = campaign._load_json_pair(
        root / "live-runtime.json",
        "sealed evaluation live runtime identity",
    )
    if (
        live_payload != live.unsigned_dict
        or live_file_sha256 != live.identity_sha256
        or campaign._canonical_sha256(live_payload) != live.identity_sha256
    ):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "live runtime identity drifted across sealed evaluation invocations"
        )
    campaign._expect_artifact(
        root,
        "campaign.json",
        _campaign_manifest(inputs, live),
    )
    engine = campaign._CampaignContext(
        root=root,
        rebuilt=inputs.rebuilt,
        live_runtime=live,
    )
    return _SealedContext(engine=engine, inputs=inputs)


def _authenticate_for_mutation(
    content: seal.ContentVerifiedSealBundle,
    *,
    resolver: executor.TrustResolver,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_payload_sha256: str,
    expected_open_verification_subject_sha256: str,
) -> None:
    _validate_mutation_pins(
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_seal_manifest_payload_sha256=(
            expected_seal_manifest_payload_sha256
        ),
        expected_open_verification_subject_sha256=(
            expected_open_verification_subject_sha256
        ),
    )
    authenticated = seal.authenticate_forager_matched_seal_bundle(
        content,
        resolver=resolver,
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_seal_manifest_sha256=expected_seal_manifest_payload_sha256,
        expected_verification_subject_sha256=(
            expected_open_verification_subject_sha256
        ),
    )
    del authenticated


def _derive_content_status(context: _SealedContext) -> campaign.CampaignStatus:
    scans = campaign._scan_all_cells(context.engine)
    return campaign._derive_status(
        context.engine,
        scans,
        completion_summary_builder=_summary_builder(context.inputs),
        completion_summary_validator=_summary_validator(context.inputs),
    )


def prepare_sealed_evaluation_campaign(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    resolver: executor.TrustResolver,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_payload_sha256: str,
    expected_open_verification_subject_sha256: str,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> campaign.CampaignStatus:
    """Authenticate the exact caller-pinned seal, then atomically prepare evaluation."""
    _validate_mutation_pins(
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_seal_manifest_payload_sha256=(
            expected_seal_manifest_payload_sha256
        ),
        expected_open_verification_subject_sha256=(
            expected_open_verification_subject_sha256
        ),
    )
    inputs = _rebuild_sealed_inputs(qualification_root, seal_root)
    prospective = _prospective_output(inputs, output_root)
    live = campaign._qualify_live(inputs.rebuilt.plan, runtime, runner)
    _authenticate_for_mutation(
        inputs.seal_content,
        resolver=resolver,
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_seal_manifest_payload_sha256=(
            expected_seal_manifest_payload_sha256
        ),
        expected_open_verification_subject_sha256=(
            expected_open_verification_subject_sha256
        ),
    )
    prospective = _prospective_output(inputs, output_root)
    published = _publish_initial_root(inputs, live, output_root, prospective)
    try:
        return verify_sealed_evaluation_campaign_content(
            qualification_root,
            seal_root,
            published,
            runtime=runtime,
            runner=runner,
        )
    except BaseException as exc:
        raise PublishedSealedEvaluationCampaignUncertainError(
            published,
            "post-publication content replay failed",
        ) from exc


def run_sealed_evaluation_campaign(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    resolver: executor.TrustResolver,
    expected_trust_anchor_identity: str,
    expected_seal_manifest_payload_sha256: str,
    expected_open_verification_subject_sha256: str,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
    max_cells: int | None = None,
) -> campaign.CampaignStatus:
    """Freshly authenticate the pinned seal, then resume append-only evaluation."""
    _validate_mutation_pins(
        expected_trust_anchor_identity=expected_trust_anchor_identity,
        expected_seal_manifest_payload_sha256=(
            expected_seal_manifest_payload_sha256
        ),
        expected_open_verification_subject_sha256=(
            expected_open_verification_subject_sha256
        ),
    )
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ForagerMatchedSealedEvaluationCampaignError(
            "max_cells must be a positive integer or None"
        )
    with _locked_campaign_root(output_root, exclusive=True) as recheck_root:
        context = _load_context(
            qualification_root,
            seal_root,
            output_root,
            runtime=runtime,
            runner=runner,
        )
        recheck_root()
        _authenticate_for_mutation(
            context.inputs.seal_content,
            resolver=resolver,
            expected_trust_anchor_identity=expected_trust_anchor_identity,
            expected_seal_manifest_payload_sha256=(
                expected_seal_manifest_payload_sha256
            ),
            expected_open_verification_subject_sha256=(
                expected_open_verification_subject_sha256
            ),
        )
        recheck_root()
        return campaign._run_resumable_context_locked(
            context.engine,
            runner=runner,
            max_cells=max_cells,
            completion_summary_builder=_summary_builder(context.inputs),
            completion_summary_validator=_summary_validator(context.inputs),
            mutation_guard=recheck_root,
        )


def verify_sealed_evaluation_campaign_content(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> campaign.CampaignStatus:
    """Replay campaign content only; this does not authenticate or authorize execution."""
    with _locked_campaign_root(output_root, exclusive=False) as recheck_root:
        context = _load_context(
            qualification_root,
            seal_root,
            output_root,
            runtime=runtime,
            runner=runner,
        )
        recheck_root()
        return _derive_content_status(context)


def sealed_evaluation_campaign_content_status(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> campaign.CampaignStatus:
    """Return fully replayed content status without resolving external authority."""
    return verify_sealed_evaluation_campaign_content(
        qualification_root,
        seal_root,
        output_root,
        runtime=runtime,
        runner=runner,
    )


def load_completed_sealed_evaluation_campaign_content(
    qualification_root: Path,
    seal_root: Path,
    output_root: Path,
    *,
    runtime: str | Path = "docker",
    runner: executor.ProcessRunner | None = None,
) -> campaign.CompletedCampaignBundle:
    """Load the completed evaluation content closure without authenticating it.

    The returned verification request remains unresolved and cannot authorize promotion.
    This function never repairs artifacts and never accepts cached seal bindings as authority.
    """
    with _locked_campaign_root(output_root, exclusive=False) as recheck_root:
        context = _load_context(
            qualification_root,
            seal_root,
            output_root,
            runtime=runtime,
            runner=runner,
        )
        recheck_root()
        scans = campaign._scan_all_cells(context.engine)
        recheck_root()
        if not all(
            scan.artifact is not None and scan.pointer_present
            for scan in scans.values()
        ):
            if any(
                (context.engine.root / name).exists()
                or (context.engine.root / f"{name}.sha256").exists()
                for name in _FINAL_ARTIFACTS
            ):
                raise ForagerMatchedSealedEvaluationCampaignError(
                    "final evaluation evidence exists before exact block completion"
                )
            raise ForagerMatchedSealedEvaluationCampaignError(
                "sealed evaluation campaign is not complete"
            )
        return campaign._build_completed_campaign_bundle(
            context.engine,
            scans,
            create=False,
            completion_summary_builder=_summary_builder(context.inputs),
            completion_summary_validator=_summary_validator(context.inputs),
        )


_CLI_ERROR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_evaluation_campaign_cli_error.v1"
)
_CLI_LOAD_RESULT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_sealed_evaluation_campaign_cli_load_result.v1"
)
_EXIT_FAILED_CLOSED: Final = 2
_EXIT_PUBLISHED_UNCERTAIN: Final = 3


def _add_mutation_arguments(command: argparse.ArgumentParser) -> None:
    """Declare the fresh-authentication material every mutating operation must pin.

    There is deliberately no bypass, force, or cached-bindings flag: the reference
    resolver and all three expected pins are required, and every fail-closed refusal
    from the underlying entry points surfaces unchanged.
    """
    command.add_argument("--trust-anchor", type=Path, required=True)
    command.add_argument("--trust-key", type=Path, required=True)
    command.add_argument("--trust-receipt-directory", type=Path, required=True)
    command.add_argument("--expected-trust-anchor-identity", required=True)
    command.add_argument("--expected-seal-manifest-payload-sha256", required=True)
    command.add_argument("--expected-open-verification-subject-sha256", required=True)


def _cli_resolver(arguments: argparse.Namespace) -> trust.SignedReceiptTrustResolver:
    """Build the reference signed-receipt resolver from explicit ceremony paths."""
    return trust.SignedReceiptTrustResolver(
        anchor_path=arguments.trust_anchor,
        key_path=arguments.trust_key,
        receipt_directory=arguments.trust_receipt_directory,
    )


def _cli_load_result(bundle: campaign.CompletedCampaignBundle) -> dict[str, Any]:
    """Summarize a completed content closure without any authentication claim."""
    return {
        "schema_version": _CLI_LOAD_RESULT_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": "sealed_evaluation",
        "output_root": bundle.output_root.as_posix(),
        "candidate_count": len(bundle.candidate_ids),
        "candidate_order": list(bundle.candidate_ids),
        "seed_count": len(bundle.active_seeds),
        "completed_cell_count": len(bundle.candidate_ids) * len(bundle.active_seeds),
        "protocol_sha256": bundle.protocol.protocol_sha256,
        "execution_plan_sha256": bundle.plan.plan_sha256,
        "live_runtime_identity_sha256": bundle.live_runtime.identity_sha256,
        "execution_receipt_index_payload_sha256": (
            bundle.execution_receipt_index.payload_sha256
        ),
        "score_evidence_sha256": bundle.score_evidence.payload_sha256,
        "verification_subject_sha256": (
            bundle.verification_request.verification_subject_sha256
        ),
        "final_file_sha256": dict(bundle.final_file_sha256),
        "promotion_authorized": False,
        "external_verification_required": True,
    }


def _write_cli_error(
    status_label: str,
    error: BaseException,
    *,
    destination: Path | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": _CLI_ERROR_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": status_label,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "promotion_authorized": False,
        "external_verification_required": True,
    }
    if destination is not None:
        payload["destination"] = destination.as_posix()
    sys.stderr.buffer.write(campaign.canonical_json_bytes(payload) + b"\n")


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "run", "verify", "status", "load"):
        command = subparsers.add_parser(operation, allow_abbrev=False)
        command.add_argument("--qualification-root", type=Path, required=True)
        command.add_argument("--seal-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--runtime", default="docker")
        if operation in {"prepare", "run"}:
            _add_mutation_arguments(command)
        if operation == "run":
            command.add_argument("--max-cells", type=int)
    arguments = parser.parse_args(argv)
    if arguments.operation == "load":
        bundle = load_completed_sealed_evaluation_campaign_content(
            arguments.qualification_root,
            arguments.seal_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
        sys.stdout.buffer.write(campaign.canonical_json_bytes(_cli_load_result(bundle)))
        return 0
    if arguments.operation == "prepare":
        status = prepare_sealed_evaluation_campaign(
            arguments.qualification_root,
            arguments.seal_root,
            arguments.output_root,
            resolver=_cli_resolver(arguments),
            expected_trust_anchor_identity=arguments.expected_trust_anchor_identity,
            expected_seal_manifest_payload_sha256=(
                arguments.expected_seal_manifest_payload_sha256
            ),
            expected_open_verification_subject_sha256=(
                arguments.expected_open_verification_subject_sha256
            ),
            runtime=arguments.runtime,
        )
    elif arguments.operation == "run":
        status = run_sealed_evaluation_campaign(
            arguments.qualification_root,
            arguments.seal_root,
            arguments.output_root,
            resolver=_cli_resolver(arguments),
            expected_trust_anchor_identity=arguments.expected_trust_anchor_identity,
            expected_seal_manifest_payload_sha256=(
                arguments.expected_seal_manifest_payload_sha256
            ),
            expected_open_verification_subject_sha256=(
                arguments.expected_open_verification_subject_sha256
            ),
            runtime=arguments.runtime,
            max_cells=arguments.max_cells,
        )
    elif arguments.operation == "verify":
        status = verify_sealed_evaluation_campaign_content(
            arguments.qualification_root,
            arguments.seal_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    else:
        status = sealed_evaluation_campaign_content_status(
            arguments.qualification_root,
            arguments.seal_root,
            arguments.output_root,
            runtime=arguments.runtime,
        )
    sys.stdout.buffer.write(campaign.canonical_json_bytes(status.to_dict()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prepare/run/verify/status/load sealed-evaluation CLI.

    Success writes one canonical JSON document to stdout and returns ``0``.
    Failures write one canonical JSON error payload to stderr and exit
    nonzero: ``3`` when a root was published but its safe handoff is
    uncertain (:class:`PublishedSealedEvaluationCampaignUncertainError`),
    ``2`` for every other fail-closed refusal (``OSError``/``ValueError``,
    surfaced unchanged from the underlying entry points).  Argument errors
    keep argparse's usage behavior (exit ``2``); ``--help`` exits ``0``.
    There are no bypass, force, or repair flags.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return _cli(tuple(argv) if argv is not None else tuple(sys.argv[1:]))
    except PublishedSealedEvaluationCampaignUncertainError as exc:
        _write_cli_error("published_uncertain", exc, destination=exc.destination)
        return _EXIT_PUBLISHED_UNCERTAIN
    except (OSError, ValueError) as exc:
        _write_cli_error("failed_closed", exc)
        return _EXIT_FAILED_CLOSED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ForagerMatchedSealedEvaluationCampaignError",
    "MATCHED_SEALED_EVALUATION_CAMPAIGN_SCHEMA_VERSION",
    "MATCHED_SEALED_EVALUATION_COMPLETION_SCHEMA_VERSION",
    "PublishedSealedEvaluationCampaignUncertainError",
    "load_completed_sealed_evaluation_campaign_content",
    "main",
    "prepare_sealed_evaluation_campaign",
    "run_sealed_evaluation_campaign",
    "sealed_evaluation_campaign_content_status",
    "verify_sealed_evaluation_campaign_content",
]
