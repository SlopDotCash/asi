"""End-to-end contracts for
:mod:`alberta_framework.benchmarks.forager_matched_final_analysis`.

Final analysis is the last phase of the matched-current pipeline: it replays
the exact seal and the completed 6x30 sealed evaluation around two external
resolvers (open subject first, evaluation resolver as the last external
callback), then computes the frozen statistics and atomically publishes a
campaign-root-independent bundle.  The statistics being finalized are those
frozen in :mod:`forager_matched_statistics`: the paired percentile bootstrap
as the primary endpoint and the sign-flip permutation test with Holm
correction as nonconfirmatory secondary sensitivity analysis, over the three
Alberta-vs-external contrasts.

The suite drives a full synthetic campaign fixture (``_fixture``: real
on-disk qualification, campaign, seal, and evaluation trees built from the
shared executor/protocol fixtures) and enforces, fail-closed: resolver
ordering and zero-write failure paths, cross-stage qualification digest
binding, exact JSON schemas (numeric aliases and v1 outer schemas rejected),
publication atomicity under ABA/parent-swap races, reward-tamper detection
via semantic replay, and rejection of any incomplete three-contrast family
or false promotion claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as evaluation
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_final_analysis as final_analysis
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from alberta_framework.benchmarks import forager_matched_statistics as statistics
from tests import test_forager_matched_executor as executor_fixtures
from tests import test_forager_matched_open_protocol as protocol_fixtures
from tests._forager_matched_platform import requires_renameat2

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_renameat2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _raw_bytes(label: str, candidate_id: str, seed: int) -> bytes:
    return f"opaque:{label}:{candidate_id}:{seed}".encode("ascii")


def _execution_ready_protocol(
    tmp_path: Path,
    frozen: protocol.ForagerMatchedProtocol,
) -> tuple[
    protocol.ForagerMatchedProtocol,
    dict[str, executor.CandidateExecutionAssets],
    str,
    int,
]:
    """Materialize one real source/config closure for the synthetic protocol."""
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    entrypoints = {
        cast(str, final_analysis._expected_entrypoint_binding(candidate_id)["path"])
        for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS
    }
    for relative in sorted(entrypoints, key=lambda item: item.encode("utf-8")):
        path = source_root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise SystemExit('synthetic fixture only')\n", encoding="utf-8")
        path.chmod(0o644)
    inventory = executor.source_inventory(source_root)
    inventory_sha256 = executor.source_inventory_sha256(source_root)
    executor_inventory_sha256 = campaign._canonical_sha256(inventory)
    source_archive = tmp_path / "source.tar"
    source_archive.write_bytes(b"synthetic-final-analysis-source")
    archive_sha256 = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    configuration = tmp_path / "configuration.json"
    configuration.write_bytes(b"{}")
    configuration_sha256 = hashlib.sha256(b"{}").hexdigest()

    candidates: list[protocol.MatchedCandidate] = []
    receipts: dict[str, dict[str, Any]] = {}
    for original in frozen.candidates:
        entrypoint = final_analysis._expected_entrypoint_binding(original.candidate_id)
        invocation_style = cast(str, entrypoint["invocation_style"])
        patch_sha256 = cast(str | None, entrypoint["rng_isolation_patch_sha256"])
        if patch_sha256 is not None:
            implementation_kind = original.implementation_kind
        elif invocation_style == "official_foragax_ppo_frozen_updates_v1":
            implementation_kind = "fixture_ppo"
        elif invocation_style == "official_foragax_continuing_main_v4":
            implementation_kind = "fixture_external"
        else:
            implementation_kind = "fixture_alberta"
        source = replace(
            original.source,
            provenance_kind="reviewed_snapshot",
            tree_git_sha1=None,
            archive_sha256=archive_sha256,
            inventory_sha256=inventory_sha256,
            snapshot_descriptor_sha256=_sha(
                f"fixture-snapshot:{entrypoint['source_key']}"
            ),
        )
        candidate = replace(
            original,
            implementation_kind=implementation_kind,
            source=source,
            configuration=replace(
                original.configuration,
                original_sha256=configuration_sha256,
                derived_sha256=configuration_sha256,
                allowed_transforms=(),
            ),
        )
        descriptor = protocol.candidate_capability_descriptor_sha256(candidate)
        candidate = replace(
            candidate,
            runtime_binding=replace(
                candidate.runtime_binding,
                qualified_capability_descriptor_sha256=descriptor,
            ),
        )
        receipt = executor_fixtures._receipt(  # noqa: SLF001
            candidate.to_dict(),
            entrypoint=cast(str, entrypoint["path"]),
            python_import_root=cast(str, entrypoint["python_import_root"]),
            invocation_style=invocation_style,
            result_root=cast(str, entrypoint["result_root"]),
            patch_sha256=patch_sha256,
        )
        receipt_sha256 = hashlib.sha256(
            executor.canonical_json_bytes(receipt)
        ).hexdigest()
        candidate = replace(
            candidate,
            runtime_binding=replace(
                candidate.runtime_binding,
                capability_qualification_receipt_sha256=receipt_sha256,
            ),
        )
        candidates.append(candidate)
        receipts[candidate.candidate_id] = receipt

    candidate_tuple = tuple(candidates)
    rebound = replace(
        frozen,
        candidates=candidate_tuple,
        candidate_index=MappingProxyType(
            {candidate.candidate_id: candidate for candidate in candidate_tuple}
        ),
    )
    assets = {
        candidate.candidate_id: executor.CandidateExecutionAssets(
            candidate_id=candidate.candidate_id,
            source_root=source_root,
            source_archive=source_archive,
            source_inventory=inventory,
            original_configuration=configuration,
            configuration=configuration,
            capability_receipt=receipts[candidate.candidate_id],
        )
        for candidate in candidate_tuple
    }
    return rebound, assets, executor_inventory_sha256, source_archive.stat().st_size


def _qualification_manifest(
    frozen: protocol.ForagerMatchedProtocol,
    qualification_artifacts: Mapping[str, Any],
    *,
    executor_inventory_sha256: str,
    source_archive_size: int,
) -> dict[str, Any]:
    artifacts = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(qualification_artifacts)),
    )

    def inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": executor.MATCHED_SOURCE_INVENTORY_SCHEMA_VERSION,
            "files": sorted(records, key=lambda item: cast(str, item["path"]).encode()),
        }

    cpu = cast(dict[str, Any], artifacts["cpu_qualification"])
    rng = cast(dict[str, Any], artifacts["rng_parity_qualification"])
    cpu_files = inventory(
        [
            {
                "path": path,
                "mode": 0o644,
                "size_bytes": source["size_bytes"],
                "sha256": source["file_sha256"],
            }
            for path, source in (
                ("receipt.v1.json", cast(dict[str, Any], cpu["receipt"])),
                ("qualification.json", cast(dict[str, Any], cpu["qualification"])),
                (
                    "environment-profile.json",
                    cast(dict[str, Any], cpu["environment_profile"]),
                ),
            )
        ]
    )
    rng_files = inventory(
        [
            {
                "path": path,
                "mode": 0o644,
                "size_bytes": source["size_bytes"],
                "sha256": source["file_sha256"],
            }
            for path, source in (
                ("plan.json", cast(dict[str, Any], rng["plan"])),
                ("receipt.json", cast(dict[str, Any], rng["receipt"])),
            )
        ]
    )
    sources: dict[str, Any] = {}
    for source_key in ("alberta", "upstream", "upstream_rng_isolated"):
        candidate_id = next(
            candidate_id
            for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS
            if qualification._source_key_for_candidate(candidate_id) == source_key
        )
        binding = frozen.candidate_index[candidate_id].source.to_dict()
        sources[source_key] = {
            "binding": binding,
            "root": f"sources/{source_key}/source",
            "archive": {
                "path": f"sources/{source_key}/source.tar",
                "sha256": binding["archive_sha256"],
                "size_bytes": source_archive_size,
            },
            "inventory": {
                "path": f"sources/{source_key}/inventory.json",
                "canonical_sha256": executor_inventory_sha256,
            },
            "snapshot_descriptor_path": (
                None
                if binding["snapshot_descriptor_sha256"] is None
                else f"sources/{source_key}/snapshot-descriptor.json"
            ),
            "patch_path": (
                "sources/upstream_rng_isolated/rng-isolation.patch"
                if source_key == "upstream_rng_isolated"
                else None
            ),
        }
    candidates: dict[str, Any] = {}
    for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS:
        candidate = frozen.candidate_index[candidate_id]
        entrypoint = final_analysis._expected_entrypoint_binding(candidate_id)
        candidates[candidate_id] = {
            "source_key": entrypoint["source_key"],
            "configuration": {
                "binding": candidate.configuration.to_dict(),
                "original_path": f"configurations/{candidate_id}/original.json",
                "derived_path": f"configurations/{candidate_id}/derived.json",
            },
            "probe": {
                "path": f"probes/{candidate_id}.json",
                "sha256": _sha(f"probe:{candidate_id}"),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            },
            "effective_seed_proof_sha256": candidate.seed_contract.effective_seed_proof_sha256,
            "resources": candidate.resources.to_dict(),
            "resource_supplement": {
                "fixed_substrate_parameter_count": 0,
                "non_gradient_operations": {
                    "causal_nonparametric_transition_updates": (
                        frozen.horizon
                        if candidate_id in open_protocol.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
                        else 0
                    ),
                    "redo_recycles": 0,
                    "target_snapshot_refreshes": 0,
                },
                "target_snapshot_parameter_count": 0,
            },
            "capability_receipt": {
                "path": f"receipts/{candidate_id}.json",
                "sha256": (
                    candidate.runtime_binding.capability_qualification_receipt_sha256
                ),
            },
            "entrypoint": {
                "path": entrypoint["path"],
                "sha256": _sha(f"entrypoint:{candidate_id}"),
                "python_import_root": entrypoint["python_import_root"],
                "invocation_style": entrypoint["invocation_style"],
                "result_root": entrypoint["result_root"],
            },
        }
    return {
        "schema_version": qualification.MATCHED_CURRENT_QUALIFICATION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "structurally_qualified_external_trust_resolution_required",
        "promotion_authorized": False,
        "performance_claim": False,
        "external_verification_required": True,
        "authority": {
            "identity": qualification.MATCHED_CURRENT_AUTHORITY_IDENTITY,
            "content_only": True,
            "externally_endorsed": False,
            "external_signature_created": False,
            "trust_profile_created": False,
        },
        "reward_blind_boundary": {
            "qualification_seed": qualification.PUBLIC_QUALIFICATION_SEED,
            "qualification_seed_class": "public_nonbenchmark_seed",
            "tuning_seeds_used": [],
            "evaluation_seeds_used": [],
            "environment_resets": len(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS),
            "environment_transitions": 0,
            "reward_arrays_read": 0,
            "result_archives_opened": 0,
        },
        "runtime_qualification": asdict(qualification._runtime_qualification()),
        "qualification_probe": {
            "source_key": "alberta",
            "path": "alberta_framework/benchmarks/forager_matched_qualification.py",
            "sha256": _sha("qualification-probe"),
        },
        "resource_accounting_semantics": qualification._plain_json(
            qualification._RESOURCE_ACCOUNTING_SEMANTICS
        ),
        "executor_qualification_roots": {
            "cpu": {
                "path": "executor-qualification/cpu",
                "inventory": cpu_files,
                "inventory_sha256": qualification._canonical_sha256(cpu_files),
            },
            "rng_parity": {
                "path": "executor-qualification/rng-parity",
                "inventory": rng_files,
                "inventory_sha256": qualification._canonical_sha256(rng_files),
            },
        },
        "frozen_executor_qualification_artifacts": artifacts,
        "candidate_order": list(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS),
        "sources": sources,
        "candidates": candidates,
        "open_protocol_sha256": frozen.protocol_sha256,
    }


def _bindings(
    request: executor.VerificationRequest,
    label: str,
) -> evidence.AuthenticatedEvidenceBindings:
    return evidence.AuthenticatedEvidenceBindings(
        stage=request.stage,
        protocol_sha256=request.protocol_sha256,
        score_evidence_sha256=request.score_evidence_sha256,
        source_manifest_sha256=request.source_manifest_sha256,
        executor_manifest_sha256=request.executor_manifest_sha256,
        qualification_manifest_sha256=request.qualification_manifest_sha256,
        execution_closure_sha256=request.execution_closure_sha256,
        trust_anchor_identity=request.trust_anchor_identity,
        verification_subject_sha256=request.verification_subject_sha256,
        verification_receipt_sha256=_sha(f"verification-receipt:{label}"),
    )


def _plan(
    frozen: protocol.ForagerMatchedProtocol,
    candidate_ids: tuple[str, ...],
    base: executor.MatchedExecutionPlan,
    qualification_manifest_sha256: str,
    assets: Mapping[str, executor.CandidateExecutionAssets],
) -> executor.MatchedExecutionPlan:
    selected_assets = {candidate_id: assets[candidate_id] for candidate_id in candidate_ids}
    return executor.build_execution_plan(
        frozen,
        selected_assets,
        qualification_manifest_sha256=qualification_manifest_sha256,
        cpu_qualification_root=base.cpu_qualification_root,
        rng_parity_qualification_root=base.rng_parity_qualification_root,
        candidate_ids=candidate_ids,
    )


def _panel(
    tmp_path: Path,
    frozen: protocol.ForagerMatchedProtocol,
    candidate_ids: tuple[str, ...],
    base: executor.MatchedExecutionPlan,
    assets: Mapping[str, executor.CandidateExecutionAssets],
    *,
    label: str,
    qualification_manifest_sha256: str,
) -> tuple[
    executor.MatchedExecutionPlan,
    executor.LiveRuntimeIdentity,
    dict[str, tuple[executor.SeedExecutionArtifacts, ...]],
    executor.MatchedExecutionReceiptIndex,
    evidence.MatchedScoreEvidence,
    executor.VerificationRequest,
    evidence.AuthenticatedEvidenceBindings,
]:
    plan = _plan(
        frozen,
        candidate_ids,
        base,
        qualification_manifest_sha256,
        assets,
    )
    live = executor_fixtures._runtime(tmp_path / f"runtime-{label}", plan)
    artifacts: dict[str, tuple[executor.SeedExecutionArtifacts, ...]] = {}
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate = plan.candidate_index[candidate_id]
        records: list[executor.SeedExecutionArtifacts] = []
        for seed_index, seed in enumerate(frozen.active_seeds):
            raw = _raw_bytes(label, candidate_id, seed)
            records.append(
                executor._artifact_mappings(
                    plan=plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=hashlib.sha256(raw).hexdigest(),
                    raw_archive_size=len(raw),
                    live_runtime=live,
                    scorer_record={
                        "fov_last_10pct_ema_auc": float(candidate_index + 1) + seed_index / 100.0,
                        "npz_sha256": _sha(f"npz:{label}:{candidate_id}:{seed}"),
                        "npz_size_bytes": 4096 + seed_index,
                        "reward_trace_sha256": _sha(f"trace:{label}:{candidate_id}:{seed}"),
                        "reward_dtype": "<f4",
                        "reward_shape": [frozen.horizon],
                    },
                )
            )
        artifacts[candidate_id] = tuple(records)
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    scores = executor.build_score_evidence(plan, artifacts)
    request = executor.build_verification_request(plan, scores)
    return (
        plan,
        live,
        artifacts,
        receipt_index,
        scores,
        request,
        _bindings(request, label),
    )


@dataclass(frozen=True, slots=True)
class _Fixture:
    qualification_root: Path
    qualification_manifest: Mapping[str, Any]
    qualification_manifest_bytes: bytes
    qualification_manifest_sha256: str
    seal_root: Path
    evaluation_root: Path
    open_bindings: evidence.AuthenticatedEvidenceBindings
    evaluation_bindings: evidence.AuthenticatedEvidenceBindings
    completed: campaign.CompletedCampaignBundle
    evaluation_context: campaign._CampaignContext
    sealed_inputs: sealed_campaign._SealedInputs


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    install_completed_loader: bool = True,
) -> _Fixture:
    base_root = tmp_path / "base"
    base_root.mkdir()
    base = executor_fixtures._plan(base_root)
    original_open = open_protocol.build_forager_matched_open_protocol(
        runtime=qualification._runtime_qualification(),
        candidate_qualifications=protocol_fixtures._qualifications(),
    )
    analysis_plan = replace(
        original_open.analysis_plan,
        primary=replace(original_open.analysis_plan.primary, resamples=31),
        secondary=replace(
            original_open.analysis_plan.secondary,
            monte_carlo_resamples=64,
        ),
    )
    open_frozen = replace(original_open, analysis_plan=analysis_plan)
    (
        open_frozen,
        execution_assets,
        executor_inventory_sha256,
        source_archive_size,
    ) = _execution_ready_protocol(tmp_path / "execution-assets", open_frozen)
    qualification_payload = _qualification_manifest(
        open_frozen,
        cast(
            Mapping[str, Any],
            base.executor_manifest["qualification_artifacts"],
        ),
        executor_inventory_sha256=executor_inventory_sha256,
        source_archive_size=source_archive_size,
    )
    qualification_bytes = qualification._canonical_json_bytes(qualification_payload)
    qualification_manifest_sha256 = hashlib.sha256(qualification_bytes).hexdigest()
    tuning_ids = tuple(
        candidate_id
        for group in open_frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    (
        open_plan,
        open_live,
        open_artifacts,
        open_receipts,
        open_scores,
        open_request,
        open_bindings,
    ) = _panel(
        tmp_path,
        open_frozen,
        tuning_ids,
        base,
        execution_assets,
        label="open",
        qualification_manifest_sha256=qualification_manifest_sha256,
    )
    open_root = tmp_path / "open-campaign"
    open_root.mkdir()
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    open_rebuilt = campaign._RebuiltInputs(
        bundle=cast(
            Any,
            SimpleNamespace(
                manifest=qualification_payload,
                manifest_bytes=qualification_bytes,
                manifest_sha256=qualification_manifest_sha256,
            ),
        ),
        protocol=open_frozen,
        plan=open_plan,
        candidate_ids=tuning_ids,
        assets={},
        schedule={},
    )
    open_context = campaign._CampaignContext(open_root, open_rebuilt, open_live)
    open_summary = campaign._completion_summary(
        open_context,
        open_receipts,
        open_scores,
        open_request,
    )
    open_final_raw = {
        "execution-receipt-index.json": open_receipts.canonical_bytes,
        "score-evidence.json": open_scores.canonical_bytes,
        "verification-request.json": open_request.canonical_bytes,
        "completion-summary.json": seal.canonical_json_bytes(open_summary),
    }
    open_completed = campaign.CompletedCampaignBundle(
        output_root=open_root,
        protocol=open_frozen,
        plan=open_plan,
        live_runtime=open_live,
        candidate_ids=tuning_ids,
        active_seeds=open_frozen.active_seeds,
        schedule={},
        seed_artifacts=open_artifacts,
        execution_receipt_index=open_receipts,
        score_evidence=open_scores,
        verification_request=open_request,
        completion_summary=open_summary,
        final_file_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in open_final_raw.items()
        },
    )
    monkeypatch.setattr(
        campaign,
        "load_completed_open_tuning_campaign",
        lambda *_args, **_kwargs: open_completed,
    )
    seal_root = tmp_path / "seal"
    seal_content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        open_root,
        seal_root,
        resolver=lambda _request: open_bindings,
        expected_trust_anchor_identity=open_bindings.trust_anchor_identity,
        expected_verification_subject_sha256=(open_bindings.verification_subject_sha256),
    )

    transition = protocol.validate_sealed_protocol_transition(
        seal_content.open_protocol,
        seal_content.sealed_protocol,
        seal_content.selection_result,
        seal_content.selection_result.selection_result_sha256,
    )
    evaluation_ids = transition.evaluation_candidate_ids
    (
        evaluation_plan,
        evaluation_live,
        evaluation_artifacts,
        evaluation_receipts,
        evaluation_scores,
        evaluation_request,
        evaluation_bindings,
    ) = _panel(
        tmp_path,
        seal_content.sealed_protocol,
        evaluation_ids,
        base,
        execution_assets,
        label="evaluation",
        qualification_manifest_sha256=qualification_manifest_sha256,
    )
    evaluation_schedule = evaluation.build_sealed_evaluation_schedule(
        seal_content.sealed_protocol,
        transition,
    )
    qualification_bundle = cast(
        Any,
        SimpleNamespace(
            output_root=qualification_root,
            manifest=qualification_payload,
            manifest_bytes=qualification_bytes,
            manifest_sha256=qualification_manifest_sha256,
            cpu_qualification_root=base.cpu_qualification_root,
            rng_parity_qualification_root=base.rng_parity_qualification_root,
        ),
    )
    evaluation_rebuilt = campaign._RebuiltInputs(
        bundle=qualification_bundle,
        protocol=seal_content.sealed_protocol,
        plan=evaluation_plan,
        candidate_ids=evaluation_ids,
        assets={},
        schedule=evaluation_schedule,
    )
    sealed_inputs = sealed_campaign._SealedInputs(
        rebuilt=evaluation_rebuilt,
        seal_content=seal_content,
        transition=transition,
        seal_manifest_payload_sha256=cast(
            str,
            seal_content.manifest["payload_sha256"],
        ),
        open_verification_subject_sha256=(
            seal_content.open_verification_request.verification_subject_sha256
        ),
    )
    evaluation_root = tmp_path / "evaluation"
    evaluation_context = campaign._CampaignContext(
        evaluation_root,
        evaluation_rebuilt,
        evaluation_live,
    )
    prospective = sealed_campaign._prospective_output(
        sealed_inputs,
        evaluation_root,
    )
    sealed_campaign._publish_initial_root(
        sealed_inputs,
        evaluation_live,
        evaluation_root,
        prospective,
    )
    evaluation_summary = sealed_campaign._completion_summary(
        sealed_inputs,
        evaluation_context,
        evaluation_receipts,
        evaluation_scores,
        evaluation_request,
    )
    final_values = {
        "execution-receipt-index.json": evaluation_receipts.to_dict(),
        "score-evidence.json": evaluation_scores.to_dict(),
        "verification-request.json": evaluation_request.to_dict(),
        "completion-summary.json": evaluation_summary,
    }
    final_hashes = {
        name: campaign._publish_json_pair(evaluation_root / name, value)
        for name, value in final_values.items()
    }
    completed = campaign.CompletedCampaignBundle(
        output_root=evaluation_root,
        protocol=seal_content.sealed_protocol,
        plan=evaluation_plan,
        live_runtime=evaluation_live,
        candidate_ids=evaluation_ids,
        active_seeds=seal_content.sealed_protocol.active_seeds,
        schedule=evaluation_schedule,
        seed_artifacts=evaluation_artifacts,
        execution_receipt_index=evaluation_receipts,
        score_evidence=evaluation_scores,
        verification_request=evaluation_request,
        completion_summary=evaluation_summary,
        final_file_sha256=final_hashes,
    )
    if install_completed_loader:
        monkeypatch.setattr(
            sealed_campaign,
            "load_completed_sealed_evaluation_campaign_content",
            lambda *_args, **_kwargs: completed,
        )
    return _Fixture(
        qualification_root=qualification_root,
        qualification_manifest=qualification_payload,
        qualification_manifest_bytes=qualification_bytes,
        qualification_manifest_sha256=qualification_manifest_sha256,
        seal_root=seal_root,
        evaluation_root=evaluation_root,
        open_bindings=open_bindings,
        evaluation_bindings=evaluation_bindings,
        completed=completed,
        evaluation_context=evaluation_context,
        sealed_inputs=sealed_inputs,
    )


def _materialize_completed_cells(value: _Fixture) -> None:
    context = value.evaluation_context
    for candidate_id, artifacts in value.completed.seed_artifacts.items():
        for artifact in artifacts:
            run_cell, completion_path = campaign._cell_paths(
                context.root,
                candidate_id,
                artifact.seed,
            )
            attempt = run_cell / "attempt-000001"
            attempt.mkdir(parents=True)
            raw = _raw_bytes("evaluation", candidate_id, artifact.seed)
            (attempt / "raw-output.tar").write_bytes(raw)
            binding = campaign._raw_binding(
                context,
                candidate_id,
                artifact.seed,
                attempt.name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
            )
            binding_sha256 = campaign._publish_json_pair(
                attempt / "raw-binding.json",
                binding,
            )
            bundle_sha256 = campaign._publish_json_pair(
                attempt / "bundle.json",
                artifact.to_dict(),
            )
            completion_path.parent.mkdir(parents=True, exist_ok=True)
            campaign._publish_json_pair(
                completion_path,
                campaign._completion_pointer(
                    context,
                    candidate_id,
                    artifact.seed,
                    attempt,
                    binding_sha256,
                    bundle_sha256,
                ),
            )


def _creation_kwargs(value: _Fixture) -> dict[str, Any]:
    seal_content = seal.load_forager_matched_seal_bundle_content(value.seal_root)
    return {
        "open_resolver": cast(Any, lambda _request: value.open_bindings),
        "evaluation_resolver": cast(
            Any,
            lambda _request: value.evaluation_bindings,
        ),
        "expected_open_trust_anchor_identity": (value.open_bindings.trust_anchor_identity),
        "expected_open_verification_subject_sha256": (
            value.open_bindings.verification_subject_sha256
        ),
        "expected_evaluation_trust_anchor_identity": (
            value.evaluation_bindings.trust_anchor_identity
        ),
        "expected_evaluation_verification_subject_sha256": (
            value.evaluation_bindings.verification_subject_sha256
        ),
        "expected_seal_manifest_payload_sha256": seal_content.manifest["payload_sha256"],
    }


def _expected_pair_inventory(names: tuple[str, ...]) -> set[str]:
    return {*(name for name in names), *(f"{name}.sha256" for name in names)}


def _rewrite_json_pair(path: Path, payload: Mapping[str, Any]) -> None:
    raw = seal.canonical_json_bytes(payload)
    sidecar = path.with_name(f"{path.name}.sha256")
    path.chmod(0o600)
    sidecar.chmod(0o600)
    path.write_bytes(raw)
    sidecar.write_bytes(f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"))


def _rewrite_copied_artifact_and_manifest(
    output: Path,
    subtree: str,
    name: str,
    payload: Mapping[str, Any],
) -> None:
    artifact_path = output / subtree / name
    _rewrite_json_pair(artifact_path, payload)
    raw = artifact_path.read_bytes()
    manifest_path = output / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    artifact_sha256 = cast(dict[str, Any], manifest["artifact_sha256"])
    artifact_sha256[f"{subtree}/{name}"] = hashlib.sha256(raw).hexdigest()
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)


def _install_golden_analysis_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class GoldenOS:
        name = "posix"

        @staticmethod
        def uname() -> SimpleNamespace:
            return SimpleNamespace(
                machine="x86_64",
                release="fixture-release",
                sysname="Linux",
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(os, name)

    sources = {
        role: {
            "path": f"fixtures/{role}.py",
            "sha256": _sha(f"fixed-{role}-source"),
            "size_bytes": 123 + index,
        }
        for index, role in enumerate(("finalizer", "statistics"))
    }
    monkeypatch.setattr(
        final_analysis,
        "_ANALYSIS_SOURCE_RECORDS_AT_IMPORT",
        sources,
    )
    monkeypatch.setattr(
        final_analysis,
        "_analysis_source_records",
        lambda: json.loads(json.dumps(sources)),
    )
    monkeypatch.setattr(
        final_analysis,
        "sys",
        SimpleNamespace(
            implementation=SimpleNamespace(name="cpython", cache_tag="fixture-312"),
            version_info=(3, 12, 0),
            hexversion=0x030C00F0,
            byteorder="little",
            platform="linux",
        ),
    )
    monkeypatch.setattr(
        final_analysis,
        "np",
        SimpleNamespace(__version__="fixture-numpy"),
    )
    monkeypatch.setattr(final_analysis, "os", GoldenOS())


def test_publish_orders_authentication_before_analysis_and_replays_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _install_golden_analysis_runtime(monkeypatch)
    events: list[str] = []
    original_analyze = statistics.analyze_matched_scores

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open-resolver")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation-resolver")
        return fixture.evaluation_bindings

    def analyze(
        contract: statistics.MatchedComparisonContract,
    ) -> statistics.MatchedComparisonResult:
        events.append("analysis")
        return original_analyze(contract)

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)
    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = evaluation_resolver
    output = tmp_path / "published" / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **kwargs,
    )
    # The authenticated analysis closure carries the source-derived protocol,
    # executor, score, and selection identities from both campaign stages.
    golden_payload_sha256 = {
        "manifest.json": (
            "2cd3ca64f3e7f59e9a463d9a91c9ec80332c9c2815d11c9e98f9a992181bc9b1"
        ),
        "analysis/analysis-runtime-source.json": (
            "ca6656e3d1e12aeab5c660921cc3a9ce00611e9c41a612e646a2ca458f70f52b"
        ),
        "analysis/evaluation-authenticated-bindings-cache.json": (
            "16113cdaed428b31d579646d35d0d96cade9358b36c75cbd7287bece606168d7"
        ),
        "analysis/statistics-contract.json": (
            "2444a37a4e77d6d9697d80a19f374208d0d3f2a17380eea650744fd5aff9412a"
        ),
        "analysis/statistics-result.json": (
            "cd3e6972c1364c976180923c8e5035b180ddf8966a9fa4110601c8fd34d12492"
        ),
    }
    for relative, expected_sha256 in golden_payload_sha256.items():
        path = output / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        assert path.with_name(f"{path.name}.sha256").read_bytes() == (
            f"{expected_sha256}\n".encode("ascii")
        )
    assert content.manifest["payload_sha256"] == (
        "7997b8e9b47db22d172b36864cb2ae740791ee2e5300fe70e0e2806bcb7f85bc"
    )
    assert content.analysis_runtime_source["payload_sha256"] == (
        "a23e65272f0785d4a2b3843610800088a5f2ca8352ef858d9ab4dfeed3caafd6"
    )

    assert events[:3] == ["open-resolver", "evaluation-resolver", "analysis"]
    assert events.count("open-resolver") == 1
    assert events.count("evaluation-resolver") == 1
    qualification_digest = fixture.qualification_manifest_sha256
    assert content.manifest["schema_version"] == (
        "alberta.forager_matched_final_analysis_manifest.v3"
    )
    assert content.manifest["qualification_manifest_sha256"] == qualification_digest
    assert content.seal_content.open_score_evidence.qualification_manifest_sha256 == (
        qualification_digest
    )
    assert content.seal_content.open_verification_request.qualification_manifest_sha256 == (
        qualification_digest
    )
    assert content.open_bindings_cache.qualification_manifest_sha256 == qualification_digest
    assert content.evaluation_score_evidence.qualification_manifest_sha256 == (
        qualification_digest
    )
    assert content.evaluation_verification_request.qualification_manifest_sha256 == (
        qualification_digest
    )
    assert content.evaluation_bindings_cache.qualification_manifest_sha256 == (
        qualification_digest
    )
    seal_manifest = cast(Mapping[str, Any], content.manifest["seal"])
    evaluation_manifest = cast(Mapping[str, Any], content.manifest["evaluation"])
    assert seal_manifest["qualification_manifest_sha256"] == qualification_digest
    assert evaluation_manifest["qualification_manifest_sha256"] == qualification_digest
    assert (fixture.evaluation_root / "qualification-manifest.json").read_bytes() == (
        fixture.qualification_manifest_bytes
    )
    assert hashlib.sha256(fixture.qualification_manifest_bytes).hexdigest() == (
        qualification_digest
    )
    evaluation_campaign = cast(
        dict[str, Any],
        json.loads((fixture.evaluation_root / "campaign.json").read_bytes()),
    )
    evaluation_completion = cast(
        dict[str, Any],
        json.loads((fixture.evaluation_root / "completion-summary.json").read_bytes()),
    )
    assert evaluation_campaign["schema_version"] == (
        "alberta.forager_matched_sealed_evaluation_campaign.v2"
    )
    assert evaluation_completion["schema_version"] == (
        "alberta.forager_matched_sealed_evaluation_completion.v2"
    )
    assert evaluation_campaign["qualification_manifest_sha256"] == qualification_digest
    assert evaluation_completion["qualification_manifest_sha256"] == qualification_digest
    assert content.analysis_runtime_source["schema_version"] == (
        "alberta.forager_matched_final_analysis_runtime_source.v2"
    )
    assert len(content.evaluation_score_evidence.candidate_scores) == 6
    assert all(
        len(candidate.records) == 30
        for candidate in content.evaluation_score_evidence.candidate_scores
    )
    expected_descriptive = ("exact_ppo", "search_oracle")
    assert (
        tuple(item.candidate_id for item in content.contract.fixed_descriptive_diagnostics)
        == expected_descriptive
    )
    assert (
        tuple(item.candidate_id for item in content.result.fixed_descriptive_exclusions)
        == expected_descriptive
    )
    inferential = {method.method_id for method in content.contract.methods}
    assert inferential.isdisjoint(expected_descriptive)

    assert set(path.name for path in output.iterdir()) == {
        "seal",
        "evaluation",
        "analysis",
        "manifest.json",
        "manifest.json.sha256",
    }
    assert set(path.name for path in (output / "seal").iterdir()) == (
        _expected_pair_inventory(final_analysis._SEAL_ARTIFACTS)
    )
    assert set(path.name for path in (output / "evaluation").iterdir()) == (
        _expected_pair_inventory(final_analysis._EVALUATION_ARTIFACTS)
    )
    assert set(path.name for path in (output / "analysis").iterdir()) == (
        _expected_pair_inventory(final_analysis._ANALYSIS_ARTIFACTS)
    )

    claim = cast(Mapping[str, Any], content.manifest["claim_boundary"])
    scope = cast(Mapping[str, Any], content.manifest["bundle_replay_scope"])
    assert claim["promotion_authorized"] is False
    assert claim["sota_claim_authorized"] is False
    assert claim["secondary_sign_flip_interpretation"] == ("nonconfirmatory_sensitivity_only")
    assert claim["secondary_sign_exchangeability_assumption"] == ("unstated_in_frozen_protocol")
    assert claim["secondary_inferential_claim_authorized"] is False
    assert claim["secondary_reject_flags_are_claim_gates"] is False
    assert claim["primary_population_inferential_claim_authorized"] is False
    assert claim["primary_seed_superpopulation_model"] == "unstated_in_frozen_protocol"
    assert claim["primary_sampling_exchangeability_assumption"] == (
        "unstated_in_frozen_protocol"
    )
    assert claim["primary_bootstrap_regularity_assumptions"] == (
        "unstated_in_frozen_protocol"
    )
    assert claim["six_arm_ranking_authorized"] is False
    assert claim["full_registered_universe_best_claim_authorized"] is False
    assert claim["scope"] == (
        "three_preregistered_alberta_vs_selected_external_contrasts_within_six_executed_arms"
    )
    assert claim["candidate_universe_v2_contrast_specific_scope_enforced"] is True
    assert claim["registered_panel_ranking_identified_by_design"] is False
    assert claim["tuning_selection_endpoint_interpretation"] == (
        "frozen_ranking_statistic_not_population_confidence_bound"
    )
    assert claim["tuning_selection_group_best_claim_authorized"] is False
    assert claim["statistics_result_standalone_claim_interpretation_forbidden"] is True
    assert claim["primary_superiority_field_is_frozen_gate_name_only"] is True
    assert claim["secondary_reject_fields_are_nonconfirmatory_calculation_outputs"] is True
    assert tuple(claim["heldout_executed_candidate_ids"]) == tuple(
        candidate.candidate_id for candidate in content.evaluation_score_evidence.candidate_scores
    )
    assert len(claim["tuning_selected_inferential_candidate_ids"]) == 4
    contrasts = cast(list[Mapping[str, Any]], claim["ordered_alberta_vs_external_contrasts"])
    assert tuple(item["hypothesis_id"] for item in contrasts) == (
        "alberta_vs_external",
        "alberta_vs_external_rank2",
        "alberta_vs_external_rank3",
    )
    assert len({item["intervention_candidate_id"] for item in contrasts}) == 1
    assert tuple(item["comparator_candidate_id"] for item in contrasts) == tuple(
        claim["tuning_selected_inferential_candidate_ids"][1:]
    )
    assert tuple(claim["fixed_descriptive_candidate_ids"]) == expected_descriptive
    assert content.manifest["promotion_authorized"] is False
    assert content.manifest["sota_claim_authorized"] is False
    assert scope["scope"] == (
        "campaign_root_independent_scalar_statistics_and_digest_closure_"
        "under_matching_live_code_and_runtime"
    )
    assert scope["self_contained_without_live_checkout"] is False
    assert scope["self_contained_without_external_runtime"] is False
    assert scope["raw_execution_archives_copied"] is False
    assert scope["reward_trace_payloads_copied"] is False
    assert scope["source_set_is_not_a_mechanically_complete_transitive_import_closure"] is True
    assert scope["workload_or_score_recomputation_requires_original_qualified_roots"] is True
    authority = cast(Mapping[str, Any], content.manifest["authority_boundary"])
    assert authority["resolver_legitimacy_not_established_by_bundle"] is True
    assert authority["qualification_remains_content_only_unendorsed"] is True
    assert authority["fresh_authentication_claim_authorized"] is False

    fixture.qualification_root.rename(tmp_path / "qualification-moved")
    fixture.seal_root.rename(tmp_path / "seal-moved")
    fixture.evaluation_root.rename(tmp_path / "evaluation-moved")
    offline = final_analysis.load_final_analysis_content(output)
    assert offline.contract == content.contract
    assert offline.result == content.result

    for legacy_version in ("v1", "v2"):
        legacy_output = tmp_path / f"legacy-final-manifest-{legacy_version}"
        shutil.copytree(output, legacy_output)
        legacy_manifest_path = legacy_output / "manifest.json"
        legacy_manifest = cast(
            dict[str, Any],
            json.loads(legacy_manifest_path.read_bytes()),
        )
        legacy_manifest["schema_version"] = (
            f"alberta.forager_matched_final_analysis_manifest.{legacy_version}"
        )
        unsigned_legacy_manifest = dict(legacy_manifest)
        unsigned_legacy_manifest.pop("payload_sha256")
        legacy_manifest["payload_sha256"] = campaign._canonical_sha256(
            unsigned_legacy_manifest
        )
        _rewrite_json_pair(legacy_manifest_path, legacy_manifest)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="manifest differs from exact replay",
        ):
            final_analysis.load_final_analysis_content(legacy_output)


def test_pin_failures_and_resolver_failures_write_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "never-created" / "final-analysis"
    calls: list[str] = []
    kwargs = _creation_kwargs(fixture)

    def should_not_resolve_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def should_not_resolve_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("evaluation")
        return fixture.evaluation_bindings

    kwargs.update(
        {
            "open_resolver": should_not_resolve_open,
            "evaluation_resolver": should_not_resolve_evaluation,
            "expected_open_verification_subject_sha256": _sha("wrong-open-subject"),
        }
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="caller-pinned verification subject",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **kwargs,
        )
    assert calls == []
    assert not output.parent.exists()

    kwargs = _creation_kwargs(fixture)

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def fail_evaluation(_request: executor.VerificationRequest) -> Any:
        calls.append("evaluation")
        raise RuntimeError("authority unavailable")

    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = fail_evaluation
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="evaluation trust resolution failed",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **kwargs,
        )
    assert calls == ["open", "evaluation"]
    assert not output.parent.exists()


def test_bindings_cache_parser_rejects_schema_drift() -> None:
    payload = {
        "schema_version": "alberta.forager_authenticated_evidence_bindings.v1",
        "stage": "sealed_evaluation",
        "protocol_sha256": _sha("protocol"),
        "score_evidence_sha256": _sha("scores"),
        "source_manifest_sha256": _sha("sources"),
        "executor_manifest_sha256": _sha("executor"),
        "qualification_manifest_sha256": _sha("qualification"),
        "execution_closure_sha256": _sha("closure"),
        "trust_anchor_identity": "test-anchor",
        "verification_subject_sha256": _sha("subject"),
        "verification_receipt_sha256": _sha("receipt"),
    }
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="schema/stage drifted",
    ):
        final_analysis._parse_bindings(
            executor.canonical_json_bytes(payload),
            expected_stage="sealed_evaluation",
        )


def test_bindings_cache_parser_normalizes_decode_and_subject_errors() -> None:
    payload = {
        "schema_version": evidence.AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION,
        "stage": "sealed_evaluation",
        "protocol_sha256": _sha("protocol"),
        "score_evidence_sha256": _sha("scores"),
        "source_manifest_sha256": _sha("sources"),
        "executor_manifest_sha256": _sha("executor"),
        "qualification_manifest_sha256": _sha("qualification"),
        "execution_closure_sha256": _sha("closure"),
        "trust_anchor_identity": "test-anchor",
        "verification_subject_sha256": _sha("inconsistent-subject"),
        "verification_receipt_sha256": _sha("receipt"),
    }
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="bindings cache evidence closure is invalid",
    ):
        final_analysis._parse_bindings(
            executor.canonical_json_bytes(payload),
            expected_stage="sealed_evaluation",
        )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="bindings cache is invalid",
    ):
        final_analysis._parse_bindings(
            b"{",
            expected_stage="sealed_evaluation",
        )


def test_qualification_manifest_exact_bytes_and_unicode_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    copied = (fixture.evaluation_root / "qualification-manifest.json").read_bytes()
    sidecar = (
        fixture.evaluation_root / "qualification-manifest.json.sha256"
    ).read_text(encoding="ascii")
    assert copied == fixture.qualification_manifest_bytes
    assert hashlib.sha256(copied).hexdigest() == fixture.qualification_manifest_sha256
    assert sidecar == f"{fixture.qualification_manifest_sha256}\n"

    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    escaped_key = copied.replace(
        b'"classification"',
        b'"cl\\u0061ssification"',
        1,
    )
    assert qualification._decode_json(escaped_key, "escaped test") == (
        qualification._decode_json(copied, "canonical test")
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact UTF-8 canonical bytes",
    ):
        final_analysis._validate_qualification_manifest(escaped_key, seal_content)

    unicode_payload = {"label": "Alberta café — exact UTF-8"}
    unicode_raw = qualification._canonical_json_bytes(unicode_payload)
    unicode_digest = hashlib.sha256(unicode_raw).hexdigest()
    unicode_path = tmp_path / "unicode-qualification.json"
    campaign._publish_exact_json_pair(unicode_path, unicode_raw, unicode_digest)
    assert unicode_path.read_bytes() == unicode_raw
    assert b"caf\xc3\xa9" in unicode_raw
    assert b"\\u00e9" not in unicode_raw
    assert unicode_path.with_name(f"{unicode_path.name}.sha256").read_bytes() == (
        f"{unicode_digest}\n".encode("ascii")
    )


def test_cross_stage_qualification_digest_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    changed_digest = _sha("different-cross-stage-qualification")
    original_digest = fixture.qualification_manifest_sha256
    assert {
        seal_content.open_score_evidence.qualification_manifest_sha256,
        seal_content.open_verification_request.qualification_manifest_sha256,
        fixture.completed.plan.qualification_manifest_sha256,
        fixture.completed.score_evidence.qualification_manifest_sha256,
        fixture.completed.verification_request.qualification_manifest_sha256,
        fixture.open_bindings.qualification_manifest_sha256,
        fixture.evaluation_bindings.qualification_manifest_sha256,
        fixture.completed.completion_summary["qualification_manifest_sha256"],
    } == {original_digest}

    changed_score_payload = fixture.completed.score_evidence.to_dict()
    changed_score_payload["qualification_manifest_sha256"] = changed_digest
    unsigned_score = dict(changed_score_payload)
    unsigned_score.pop("payload_sha256")
    changed_score_payload["payload_sha256"] = campaign._canonical_sha256(unsigned_score)
    changed_scores = evidence.parse_matched_score_evidence(changed_score_payload)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            seal_content,
            replace(fixture.completed, score_evidence=changed_scores),
        )

    changed_open_score_payload = seal_content.open_score_evidence.to_dict()
    changed_open_score_payload["qualification_manifest_sha256"] = changed_digest
    unsigned_open_score = dict(changed_open_score_payload)
    unsigned_open_score.pop("payload_sha256")
    changed_open_score_payload["payload_sha256"] = campaign._canonical_sha256(
        unsigned_open_score
    )
    changed_open_scores = evidence.parse_matched_score_evidence(
        changed_open_score_payload
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            replace(seal_content, open_score_evidence=changed_open_scores),
            fixture.completed,
        )

    def request_with_changed_qualification(
        request: executor.VerificationRequest,
    ) -> executor.VerificationRequest:
        subject_sha256 = campaign._canonical_sha256(
            {
                "schema_version": evidence.MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
                "stage": request.stage,
                "protocol_sha256": request.protocol_sha256,
                "qualification_manifest_sha256": changed_digest,
                "score_evidence_sha256": request.score_evidence_sha256,
                "source_manifest_sha256": request.source_manifest_sha256,
                "executor_manifest_sha256": request.executor_manifest_sha256,
                "execution_closure_sha256": request.execution_closure_sha256,
                "trust_anchor_identity": request.trust_anchor_identity,
            }
        )
        return replace(
            request,
            qualification_manifest_sha256=changed_digest,
            verification_subject_sha256=subject_sha256,
        )

    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            replace(
                seal_content,
                open_verification_request=request_with_changed_qualification(
                    seal_content.open_verification_request
                ),
            ),
            fixture.completed,
        )

    changed_seal_manifest = cast(
        dict[str, Any],
        json.loads(seal.canonical_json_bytes(seal_content.manifest)),
    )
    cast(dict[str, Any], changed_seal_manifest["open_campaign"])[
        "qualification_manifest_sha256"
    ] = changed_digest
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            replace(seal_content, manifest=changed_seal_manifest),
            fixture.completed,
        )

    changed_executor_manifest = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(fixture.completed.plan.executor_manifest)),
    )
    changed_executor_manifest["qualification_manifest_sha256"] = changed_digest
    changed_plan_payload = fixture.completed.plan.to_dict()
    changed_plan_payload["qualification_manifest_sha256"] = changed_digest
    changed_plan_payload["executor_manifest"] = changed_executor_manifest
    changed_plan_payload["executor_manifest_sha256"] = campaign._canonical_sha256(
        changed_executor_manifest
    )
    changed_plan = replace(
        fixture.completed.plan,
        qualification_manifest_sha256=changed_digest,
        executor_manifest=changed_executor_manifest,
        payload=changed_plan_payload,
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            seal_content,
            replace(fixture.completed, plan=changed_plan),
        )

    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            seal_content,
            replace(
                fixture.completed,
                verification_request=request_with_changed_qualification(
                    fixture.completed.verification_request
                ),
            ),
        )

    changed_completion = dict(fixture.completed.completion_summary)
    changed_completion["qualification_manifest_sha256"] = changed_digest
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact nonpromoting 6-by-30 sealed block",
    ):
        final_analysis._validate_exact_completed_panel(
            seal_content,
            replace(fixture.completed, completion_summary=changed_completion),
        )

    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    qualification_payload = cast(
        dict[str, Any],
        qualification._decode_json(
            artifacts["qualification-manifest.json"],
            "test qualification manifest",
        ),
    )
    probe = cast(dict[str, Any], qualification_payload["qualification_probe"])
    probe["sha256"] = _sha("changed-probe-source")
    artifacts["qualification-manifest.json"] = qualification._canonical_json_bytes(
        qualification_payload
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="qualification manifest|execution-plan closure|source/executor/reward closure",
    ):
        final_analysis._validate_evaluation_snapshot(
            artifacts,
            seal_content,
            fixture.evaluation_bindings,
        )


def test_literal_v1_evaluation_outer_schemas_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    legacy_cases = (
        ("execution-plan.json", "alberta.forager_matched_execution_plan.v1"),
        ("score-evidence.json", "alberta.forager_matched_score_evidence.v1"),
        (
            "verification-request.json",
            "alberta.forager_matched_verification_request.v1",
        ),
        (
            "campaign.json",
            "alberta.forager_matched_sealed_evaluation_campaign.v1",
        ),
        (
            "completion-summary.json",
            "alberta.forager_matched_sealed_evaluation_completion.v1",
        ),
    )
    for name, legacy_schema in legacy_cases:
        changed = dict(artifacts)
        payload = cast(dict[str, Any], json.loads(changed[name]))
        payload["schema_version"] = legacy_schema
        changed[name] = seal.canonical_json_bytes(payload)
        with pytest.raises(
            (
                final_analysis.ForagerMatchedFinalAnalysisError,
                evidence.ForagerMatchedEvidenceError,
                executor.ForagerMatchedExecutorError,
            )
        ):
            final_analysis._validate_evaluation_snapshot(
                changed,
                seal_content,
                fixture.evaluation_bindings,
            )


def test_content_loading_is_resolver_free_and_fresh_auth_checks_both_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("content loading must not invoke a resolver")

    with monkeypatch.context() as guarded:
        guarded.setattr(
            seal,
            "authenticate_forager_matched_seal_bundle",
            unexpected,
        )
        guarded.setattr(executor, "resolve_authenticated_bindings", unexpected)
        replayed = final_analysis.load_final_analysis_content(output)
    assert replayed.result == content.result
    authority = cast(Mapping[str, Any], replayed.manifest["authority_boundary"])
    assert authority["persisted_bindings_are_cache_only"] is True
    assert authority["content_loader_authenticates_nothing"] is True

    calls: list[str] = []

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("open")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        calls.append("evaluation")
        return fixture.evaluation_bindings

    auth_kwargs = {
        **_creation_kwargs(fixture),
        "open_resolver": open_resolver,
        "evaluation_resolver": evaluation_resolver,
        "expected_analysis_manifest_payload_sha256": content.manifest["payload_sha256"],
    }
    authenticated = final_analysis.authenticate_final_analysis_content(
        content,
        **auth_kwargs,
    )
    assert calls == ["open", "evaluation"]
    assert authenticated.open_bindings == fixture.open_bindings
    assert authenticated.evaluation_bindings == fixture.evaluation_bindings

    changed_evaluation = replace(
        fixture.evaluation_bindings,
        verification_receipt_sha256=_sha("different-evaluation-receipt"),
    )
    auth_kwargs["evaluation_resolver"] = lambda _request: changed_evaluation
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="differs from persisted cache",
    ):
        final_analysis.authenticate_final_analysis_content(output, **auth_kwargs)


def test_loader_rejects_false_promotion_and_partial_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )

    false_promotion = tmp_path / "false-promotion"
    shutil.copytree(output, false_promotion)
    manifest_path = false_promotion / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    claim = cast(dict[str, Any], manifest["claim_boundary"])
    claim["promotion_authorized"] = True
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="manifest differs from exact replay",
    ):
        final_analysis.load_final_analysis_content(false_promotion)

    false_secondary_inference = tmp_path / "false-secondary-inference"
    shutil.copytree(output, false_secondary_inference)
    manifest_path = false_secondary_inference / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_bytes()))
    claim = cast(dict[str, Any], manifest["claim_boundary"])
    claim["secondary_inferential_claim_authorized"] = True
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256")
    manifest["payload_sha256"] = campaign._canonical_sha256(unsigned)
    _rewrite_json_pair(manifest_path, manifest)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="manifest differs from exact replay",
    ):
        final_analysis.load_final_analysis_content(false_secondary_inference)

    partial = tmp_path / "partial"
    shutil.copytree(output, partial)
    (partial / "analysis" / "statistics-result.json.sha256").unlink()
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="inventory differs",
    ):
        final_analysis.load_final_analysis_content(partial)


def test_post_publish_failure_is_uncertain_and_preserves_published_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"

    real_load = final_analysis._load_from_open_root
    replay_calls = 0

    def fail_post_publish(
        opened: seal._OpenDirectory,
    ) -> final_analysis.ContentVerifiedFinalAnalysisBundle:
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 2:
            raise RuntimeError("injected final replay failure")
        return real_load(opened)

    with monkeypatch.context() as injected:
        injected.setattr(
            final_analysis,
            "_load_from_open_root",
            fail_post_publish,
        )
        with pytest.raises(
            final_analysis.PublishedFinalAnalysisUncertainError,
            match="post-publication content or inode replay failed",
        ) as caught:
            final_analysis.create_forager_matched_final_analysis_bundle(
                fixture.qualification_root,
                fixture.seal_root,
                fixture.evaluation_root,
                output,
                **_creation_kwargs(fixture),
            )
    assert caught.value.destination == output
    assert output.is_dir()
    assert final_analysis.load_final_analysis_content(output).output_root == output


def test_concurrent_destination_is_preserved_and_owned_staging_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    real_publish = seal._publish_verified_no_replace

    def concurrent_publish(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        destination.mkdir()
        (destination / "owner").write_text("concurrent", encoding="utf-8")
        real_publish(
            parent,
            staging,
            source_name,
            destination_name,
            destination,
        )

    monkeypatch.setattr(seal, "_publish_verified_no_replace", concurrent_publish)
    with pytest.raises(ValueError, match="created concurrently"):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert (output / "owner").read_text(encoding="utf-8") == "concurrent"
    assert not tuple(output.parent.glob(".seal-partial-*"))


def test_each_resolver_is_followed_by_full_input_replay_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    completion_path = fixture.evaluation_root / "completion-summary.json"
    original = cast(dict[str, Any], json.loads(completion_path.read_bytes()))
    mutated = dict(original)
    mutated["status"] = "mutated-during-resolver"
    events: list[str] = []

    def analyze(_contract: statistics.MatchedComparisonContract) -> Any:
        events.append("analysis")
        raise AssertionError("statistics must not run after input mutation")

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)

    def mutate_during_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open")
        _rewrite_json_pair(completion_path, mutated)
        return fixture.open_bindings

    def record_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation")
        return fixture.evaluation_bindings

    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = mutate_during_open
    kwargs["evaluation_resolver"] = record_evaluation
    first_output = tmp_path / "first-output" / "analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="snapshot differs from replayed completion",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            first_output,
            **kwargs,
        )
    assert events == ["open"]
    assert not first_output.parent.exists()

    _rewrite_json_pair(completion_path, original)
    events.clear()

    def stable_open(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open")
        return fixture.open_bindings

    def mutate_during_evaluation(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("evaluation")
        _rewrite_json_pair(completion_path, mutated)
        return fixture.evaluation_bindings

    kwargs["open_resolver"] = stable_open
    kwargs["evaluation_resolver"] = mutate_during_evaluation
    second_output = tmp_path / "second-output" / "analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="snapshot differs from replayed completion",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            second_output,
            **kwargs,
        )
    assert events == ["open", "evaluation"]
    assert not second_output.parent.exists()


def test_publication_parent_aba_is_rejected_without_touching_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_path = tmp_path / "publication"
    moved_parent = tmp_path / "publication-moved"
    output = parent_path / "final-analysis"
    real_sync = final_analysis._sync_staged_tree

    def sync_then_substitute_parent(staging: seal._OpenDirectory) -> None:
        real_sync(staging)
        parent_path.rename(moved_parent)
        parent_path.mkdir()
        (parent_path / "replacement-owner").write_text(
            "must-survive",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        final_analysis,
        "_sync_staged_tree",
        sync_then_substitute_parent,
    )
    with pytest.raises(ValueError, match="output parent"):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert not output.exists()
    assert (parent_path / "replacement-owner").read_text(encoding="utf-8") == ("must-survive")
    assert not tuple(moved_parent.glob(".seal-partial-*"))


def test_evaluation_resolver_is_last_external_callback_at_real_loader_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        install_completed_loader=False,
    )
    _materialize_completed_cells(fixture)
    events: list[str] = []
    evaluation_resolved = False

    def caller_runner(_command: Sequence[str]) -> executor.ProcessResult:
        events.append("caller-runner")
        raise AssertionError("the context stub must not execute the runner")

    def context_boundary(
        qualification_root: Path,
        seal_root: Path,
        output_root: Path,
        *,
        runtime: str | Path,
        runner: executor.ProcessRunner | None,
    ) -> sealed_campaign._SealedContext:
        del qualification_root, seal_root, output_root, runtime
        events.append("post-evaluation-replay" if evaluation_resolved else "replay")
        if evaluation_resolved and runner is caller_runner:
            raise AssertionError("caller runner reached after evaluation resolver")
        return sealed_campaign._SealedContext(
            engine=fixture.evaluation_context,
            inputs=fixture.sealed_inputs,
        )

    monkeypatch.setattr(sealed_campaign, "_load_context", context_boundary)

    def open_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        events.append("open-resolver")
        return fixture.open_bindings

    def evaluation_resolver(
        _request: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal evaluation_resolved
        events.append("evaluation-resolver")
        evaluation_resolved = True
        return fixture.evaluation_bindings

    real_analyze = statistics.analyze_matched_scores
    real_identity = final_analysis._analysis_runtime_source_identity

    def identity() -> dict[str, Any]:
        events.append("analysis-identity")
        return real_identity()

    def analyze(
        contract: statistics.MatchedComparisonContract,
    ) -> statistics.MatchedComparisonResult:
        events.append("analysis")
        return real_analyze(contract)

    monkeypatch.setattr(statistics, "analyze_matched_scores", analyze)
    monkeypatch.setattr(final_analysis, "_analysis_runtime_source_identity", identity)
    kwargs = _creation_kwargs(fixture)
    kwargs["open_resolver"] = open_resolver
    kwargs["evaluation_resolver"] = evaluation_resolver
    kwargs["runner"] = caller_runner
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        tmp_path / "final-analysis",
        **kwargs,
    )
    assert events[:7] == [
        "replay",
        "open-resolver",
        "replay",
        "evaluation-resolver",
        "post-evaluation-replay",
        "analysis-identity",
        "analysis",
    ]
    assert "caller-runner" not in events


def test_post_publish_byte_identical_destination_aba_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    moved = tmp_path / "published-original-inode"
    real_publish = seal._publish_verified_no_replace

    def publish_then_swap(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        real_publish(
            parent,
            staging,
            source_name,
            destination_name,
            destination,
        )
        destination.rename(moved)
        shutil.copytree(moved, destination)

    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish_then_swap)
    with pytest.raises(final_analysis.PublishedFinalAnalysisUncertainError):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert output.is_dir()
    assert moved.is_dir()
    assert output.stat().st_ino != moved.stat().st_ino


@pytest.mark.parametrize("failure_ordinal", [2, 3])
def test_partial_child_open_failure_closes_every_prior_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_ordinal: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )
    real_open_child = final_analysis._open_child
    opened_descriptors: list[int] = []
    calls = 0

    def fail_partial_open(
        parent: seal._OpenDirectory,
        name: str,
        label: str,
    ) -> seal._OpenDirectory:
        nonlocal calls
        calls += 1
        if calls == failure_ordinal:
            raise final_analysis.ForagerMatchedFinalAnalysisError("injected child open failure")
        opened = real_open_child(parent, name, label)
        opened_descriptors.append(opened.descriptor)
        return opened

    monkeypatch.setattr(final_analysis, "_open_child", fail_partial_open)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="injected child open failure",
    ):
        final_analysis.load_final_analysis_content(output)
    leaked: list[int] = []
    for descriptor in opened_descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        leaked.append(descriptor)
        os.close(descriptor)
    assert leaked == []


def test_analysis_runtime_source_artifact_records_enumerated_source_set_and_scoped_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "final-analysis"
    content = final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        output,
        **_creation_kwargs(fixture),
    )
    assert "analysis-runtime-source.json" in final_analysis._ANALYSIS_ARTIFACTS
    artifact = cast(
        dict[str, Any],
        json.loads((output / "analysis" / "analysis-runtime-source.json").read_bytes()),
    )
    sources = cast(dict[str, Any], artifact["sources"])
    assert artifact["schema_version"] == (
        final_analysis.MATCHED_FINAL_ANALYSIS_RUNTIME_SOURCE_SCHEMA_VERSION
    )
    assert artifact["classification"] == (
        "explicit_final_analysis_source_set_and_versioned_runtime_identity"
    )
    assert artifact["mechanically_complete_transitive_import_closure"] is False
    assert artifact["package_wide_source_closure_captured"] is False
    assert set(sources) == {
        role for role, _path, _relative in final_analysis._ANALYSIS_SOURCE_PATHS
    }
    assert len(sources) == 27
    current_records = final_analysis._analysis_source_records()
    for role in (
        "forager_rng_parity",
        "forager_rtu_ppo_rng_isolation",
        "causal_map_forager",
        "forager",
        "recurrent_trace_actor_critic",
        "core_horde",
        "core_multi_head_learner",
        "core_normalizers",
        "core_optimizers",
        "core_types",
        "core_initializers",
    ):
        assert sources[role]["sha256"] == current_records[role]["sha256"]
    finalizer_raw = Path(final_analysis.__file__).resolve().read_bytes()
    statistics_raw = Path(statistics.__file__).resolve().read_bytes()
    assert sources["finalizer"] == {
        "path": "alberta_framework/benchmarks/forager_matched_final_analysis.py",
        "sha256": hashlib.sha256(finalizer_raw).hexdigest(),
        "size_bytes": len(finalizer_raw),
    }
    assert sources["statistics"] == {
        "path": "alberta_framework/benchmarks/forager_matched_statistics.py",
        "sha256": hashlib.sha256(statistics_raw).hexdigest(),
        "size_bytes": len(statistics_raw),
    }
    runtime = cast(dict[str, Any], artifact["runtime"])
    python = cast(dict[str, Any], runtime["python"])
    numpy = cast(dict[str, Any], runtime["numpy"])
    assert python["implementation"] == sys.implementation.name
    assert python["version"] == list(sys.version_info[:3])
    assert python["cache_tag"] == sys.implementation.cache_tag
    assert numpy["version"] == np.__version__
    assert (
        cast(Mapping[str, Any], content.manifest["analysis_execution"])[
            "runtime_source_payload_sha256"
        ]
        == artifact["payload_sha256"]
    )

    baseline = final_analysis._analysis_source_records()

    def drifted_sources() -> dict[str, dict[str, Any]]:
        drifted = cast(
            dict[str, dict[str, Any]],
            json.loads(json.dumps(baseline)),
        )
        drifted["finalizer"]["sha256"] = _sha("offline-finalizer-drift")
        return drifted

    with monkeypatch.context() as drifted:
        drifted.setattr(final_analysis, "_analysis_source_records", drifted_sources)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="analysis source changed",
        ):
            final_analysis.load_final_analysis_content(output)


def test_analysis_source_drift_during_statistics_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    baseline = final_analysis._analysis_source_records()
    captures = 0

    def drifting_sources() -> dict[str, dict[str, Any]]:
        nonlocal captures
        captures += 1
        copied = cast(
            dict[str, dict[str, Any]],
            json.loads(json.dumps(baseline)),
        )
        if captures >= 2:
            copied["statistics"]["sha256"] = _sha("changed-statistics-source")
        return copied

    monkeypatch.setattr(final_analysis, "_analysis_source_records", drifting_sources)
    output = tmp_path / "never-created" / "final-analysis"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="analysis source changed",
    ):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert captures == 2
    assert not output.parent.exists()


def test_nested_authority_threat_and_reward_tampering_fails_semantic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    pristine = tmp_path / "pristine"
    final_analysis.create_forager_matched_final_analysis_bundle(
        fixture.qualification_root,
        fixture.seal_root,
        fixture.evaluation_root,
        pristine,
        **_creation_kwargs(fixture),
    )

    mutations: tuple[tuple[str, Any], ...] = (
        (
            "qualification-manifest.json",
            lambda payload: cast(dict[str, Any], payload["reward_blind_boundary"]).__setitem__(
                "reward_arrays_read", 1
            ),
        ),
        (
            "campaign.json",
            lambda payload: cast(
                dict[str, Any], payload["content_capture_threat_boundary"]
            ).__setitem__("noncooperative_same_uid_writers", "trusted"),
        ),
        (
            "completion-summary.json",
            lambda payload: payload.__setitem__("cached_bindings_accepted_as_authority", True),
        ),
    )
    for name, mutate in mutations:
        tampered = tmp_path / f"tampered-{name.removesuffix('.json')}"
        shutil.copytree(pristine, tampered)
        artifact_path = tampered / "evaluation" / name
        payload = cast(dict[str, Any], json.loads(artifact_path.read_bytes()))
        mutate(payload)
        _rewrite_copied_artifact_and_manifest(
            tampered,
            "evaluation",
            name,
            payload,
        )
        with pytest.raises(final_analysis.ForagerMatchedFinalAnalysisError):
            final_analysis.load_final_analysis_content(tampered)


def test_plan_source_executor_live_and_receipt_shapes_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    final_analysis._validate_evaluation_snapshot(
        artifacts,
        seal_content,
        fixture.evaluation_bindings,
    )
    for name in (
        "execution-plan.json",
        "source-manifest.json",
        "executor-manifest.json",
        "live-runtime.json",
        "execution-receipt-index.json",
    ):
        changed = dict(artifacts)
        payload = cast(dict[str, Any], json.loads(changed[name]))
        payload["unexpected_authority_field"] = False
        if name == "execution-receipt-index.json":
            unsigned = dict(payload)
            unsigned.pop("payload_sha256")
            payload["payload_sha256"] = campaign._canonical_sha256(unsigned)
        changed[name] = seal.canonical_json_bytes(payload)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="fields drifted",
        ):
            final_analysis._validate_evaluation_snapshot(
                changed,
                seal_content,
                fixture.evaluation_bindings,
            )

    changed = dict(artifacts)
    executor_payload = cast(
        dict[str, Any],
        json.loads(changed["executor-manifest.json"]),
    )
    qualification_artifacts = cast(dict[str, Any], executor_payload["qualification_artifacts"])
    cpu_qualification = cast(dict[str, Any], qualification_artifacts["cpu_qualification"])
    authority = cast(dict[str, Any], cpu_qualification["authority_boundary"])
    authority["performance_claim"] = True
    changed["executor-manifest.json"] = seal.canonical_json_bytes(executor_payload)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="authority and qualification closure drifted",
    ):
        final_analysis._validate_evaluation_snapshot(
            changed,
            seal_content,
            fixture.evaluation_bindings,
        )


def test_finalizer_rejects_incomplete_three_contrast_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    original_builder = evidence.build_statistics_contract

    def omit_rank_three(*args: Any, **kwargs: Any) -> Any:
        contract, transition, scores = original_builder(*args, **kwargs)
        return (
            replace(
                contract,
                secondary_comparisons=contract.secondary_comparisons[:1],
            ),
            transition,
            scores,
        )

    monkeypatch.setattr(evidence, "build_statistics_contract", omit_rank_three)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="exact ordered three-contrast",
    ):
        final_analysis._build_contract_and_result(
            seal_content,
            fixture.completed.score_evidence,
            open_bindings=fixture.open_bindings,
            evaluation_bindings=fixture.evaluation_bindings,
        )


def test_nested_qualification_projection_rejects_semantic_and_type_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }

    def mutate_critical_inventory(payload: dict[str, Any]) -> None:
        roots = cast(dict[str, Any], payload["executor_qualification_roots"])
        cpu = cast(dict[str, Any], roots["cpu"])
        inventory = cast(dict[str, Any], cpu["inventory"])
        files = cast(list[dict[str, Any]], inventory["files"])
        next(item for item in files if item["path"] == "receipt.v1.json")[
            "sha256"
        ] = _sha("changed-critical-receipt")
        cpu["inventory_sha256"] = qualification._canonical_sha256(inventory)

    def mutate_source_binding(payload: dict[str, Any]) -> None:
        sources = cast(dict[str, Any], payload["sources"])
        alberta = cast(dict[str, Any], sources["alberta"])
        binding = cast(dict[str, Any], alberta["binding"])
        archive = cast(dict[str, Any], alberta["archive"])
        binding["archive_sha256"] = _sha("changed-source-archive")
        archive["sha256"] = binding["archive_sha256"]

    def candidate_record(payload: dict[str, Any]) -> dict[str, Any]:
        candidates = cast(dict[str, Any], payload["candidates"])
        return cast(dict[str, Any], candidates["causal_e025_q050"])

    mutations: tuple[tuple[str, Any], ...] = (
        (
            "boolean authority alias",
            lambda value: cast(dict[str, Any], value["authority"]).__setitem__(
                "content_only", 1
            ),
        ),
        (
            "runtime identity",
            lambda value: cast(dict[str, Any], value["runtime_qualification"]).__setitem__(
                "image_sha256", _sha("wrong-runtime")
            ),
        ),
        (
            "probe schema",
            lambda value: cast(dict[str, Any], value["qualification_probe"]).__setitem__(
                "extra", False
            ),
        ),
        (
            "resource semantics",
            lambda value: cast(
                dict[str, Any], value["resource_accounting_semantics"]
            ).__setitem__("scope_limitation", "tampered"),
        ),
        ("critical executor inventory", mutate_critical_inventory),
        ("source binding", mutate_source_binding),
        (
            "candidate configuration",
            lambda value: cast(
                dict[str, Any], candidate_record(value)["configuration"]
            ).__setitem__("binding", {}),
        ),
        (
            "candidate resources",
            lambda value: cast(dict[str, Any], candidate_record(value)["resources"]).__setitem__(
                "optimizer_update_count", False
            ),
        ),
        (
            "candidate receipt",
            lambda value: cast(
                dict[str, Any], candidate_record(value)["capability_receipt"]
            ).__setitem__("sha256", _sha("wrong-receipt")),
        ),
        (
            "candidate entrypoint",
            lambda value: cast(
                dict[str, Any], candidate_record(value)["entrypoint"]
            ).__setitem__("result_root", "wrong"),
        ),
        (
            "candidate resource supplement",
            lambda value: cast(
                dict[str, Any],
                cast(
                    dict[str, Any],
                    candidate_record(value)["resource_supplement"],
                )["non_gradient_operations"],
            ).__setitem__("causal_nonparametric_transition_updates", 0),
        ),
    )
    for _label, mutate in mutations:
        changed = dict(artifacts)
        payload = cast(
            dict[str, Any],
            json.loads(changed["qualification-manifest.json"]),
        )
        mutate(payload)
        changed["qualification-manifest.json"] = seal.canonical_json_bytes(payload)
        with pytest.raises(final_analysis.ForagerMatchedFinalAnalysisError):
            final_analysis._validate_evaluation_snapshot(
                changed,
                seal_content,
                fixture.evaluation_bindings,
            )


def test_source_projection_and_command_templates_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    _qualification_digest, inventory_sha256 = final_analysis._validate_qualification_manifest(
        artifacts["qualification-manifest.json"],
        seal_content,
    )
    plan = seal._decode_canonical(artifacts["execution-plan.json"], "test plan")
    source = seal._decode_canonical(artifacts["source-manifest.json"], "test source")
    executor_manifest = seal._decode_canonical(
        artifacts["executor-manifest.json"],
        "test executor",
    )
    scores = evidence.parse_matched_score_evidence(artifacts["score-evidence.json"])
    candidate_order = tuple(item.candidate_id for item in scores.candidate_scores)

    def validate(changed_plan: dict[str, Any], changed_source: dict[str, Any]) -> None:
        final_analysis._validate_plan_and_manifests(
            changed_plan,
            changed_source,
            executor_manifest,
            seal_content.sealed_protocol,
            candidate_order,
            inventory_sha256,
            _qualification_digest,
        )

    for field, replacement in (
        ("source_inventory_hash_scheme", "wrong"),
        ("executor_inventory_sha256", _sha("wrong-inventory")),
        ("entrypoint_path", "wrong.py"),
        ("python_import_root", "wrong"),
        ("invocation_style", "wrong"),
        ("result_root", "wrong"),
        ("rng_isolation_patch_sha256", _sha("unexpected-patch")),
    ):
        changed_source = cast(dict[str, Any], json.loads(json.dumps(source)))
        first = cast(list[dict[str, Any]], changed_source["candidates"])[0]
        first[field] = replacement
        with pytest.raises(final_analysis.ForagerMatchedFinalAnalysisError):
            validate(plan, changed_source)

    changed_plan = cast(dict[str, Any], json.loads(json.dumps(plan)))
    templates = cast(list[dict[str, Any]], changed_plan["candidate_command_templates"])
    cast(list[str], templates[0]["argv"])[-1] = "--horizon=1"
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="command template drifted",
    ):
        validate(changed_plan, source)

    changed_executor = cast(dict[str, Any], json.loads(json.dumps(executor_manifest)))
    helper = cast(dict[str, Any], changed_executor["container_helper"])
    helper["sha256"] = int("1" * 64)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="container helper must be a lowercase SHA-256",
    ):
        final_analysis._validate_plan_and_manifests(
            plan,
            source,
            changed_executor,
            seal_content.sealed_protocol,
            candidate_order,
            inventory_sha256,
            _qualification_digest,
        )

    malformed_plan_cases: tuple[tuple[str, Any], ...] = (
        ("horizon", float(seal_content.sealed_protocol.horizon)),
        ("active_seeds", None),
        (
            "active_seeds",
            [
                float(seal_content.sealed_protocol.active_seeds[0]),
                *seal_content.sealed_protocol.active_seeds[1:],
            ],
        ),
        ("candidate_order", None),
    )
    for field, replacement_value in malformed_plan_cases:
        changed_plan = cast(dict[str, Any], json.loads(json.dumps(plan)))
        changed_plan[field] = replacement_value
        changed_artifacts = dict(artifacts)
        changed_artifacts["execution-plan.json"] = seal.canonical_json_bytes(
            changed_plan
        )
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="execution-plan closure drifted",
        ):
            final_analysis._validate_evaluation_snapshot(
                changed_artifacts,
                seal_content,
                fixture.evaluation_bindings,
            )

    unknown_id = "unknown_evaluation_candidate"
    unknown_order = (unknown_id, *candidate_order[1:])
    changed_source = cast(dict[str, Any], json.loads(json.dumps(source)))
    cast(list[dict[str, Any]], changed_source["candidates"])[0]["candidate_id"] = (
        unknown_id
    )
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="unknown candidate",
    ):
        final_analysis._validate_plan_and_manifests(
            plan,
            changed_source,
            executor_manifest,
            seal_content.sealed_protocol,
            unknown_order,
            inventory_sha256,
            _qualification_digest,
        )


def test_live_runtime_projection_rejects_malformed_core_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }

    def wrong_config(value: dict[str, Any]) -> None:
        inspection = cast(dict[str, Any], value["image_inspection"])
        inspection["Config"] = []

    def wrong_labels(value: dict[str, Any]) -> None:
        inspection = cast(dict[str, Any], value["image_inspection"])
        config = cast(dict[str, Any], inspection["Config"])
        config["Labels"] = []

    mutations: tuple[Any, ...] = (
        lambda value: value.__setitem__("executable_sha256", "invalid"),
        lambda value: value.__setitem__("version", []),
        lambda value: cast(dict[str, Any], value["image_inspection"]).__setitem__(
            "Id", f"sha256:{_sha('wrong-image')}"
        ),
        wrong_config,
        wrong_labels,
        lambda value: cast(
            dict[str, Any],
            cast(dict[str, Any], value["image_inspection"])["Config"],
        )["Labels"].__setitem__(
            "io.elizaos.alberta.foragax.launcher-contract",
            "wrong",
        ),
    )
    for mutate in mutations:
        changed = dict(artifacts)
        payload = cast(dict[str, Any], json.loads(changed["live-runtime.json"]))
        mutate(payload)
        changed["live-runtime.json"] = seal.canonical_json_bytes(payload)
        with pytest.raises(final_analysis.ForagerMatchedFinalAnalysisError):
            final_analysis._validate_evaluation_snapshot(
                changed,
                seal_content,
                fixture.evaluation_bindings,
            )

    changed = dict(artifacts)
    schedule_payload = cast(
        dict[str, Any],
        json.loads(changed["execution-schedule.json"]),
    )
    schedule_payload.pop("schedule_sha256")
    changed["execution-schedule.json"] = seal.canonical_json_bytes(schedule_payload)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="execution schedule SHA-256",
    ):
        final_analysis._validate_evaluation_snapshot(
            changed,
            seal_content,
            fixture.evaluation_bindings,
        )


def test_analysis_runtime_source_replay_rejects_json_numeric_aliases() -> None:
    expected = final_analysis._analysis_runtime_source_identity()
    mutations: tuple[Any, ...] = (
        lambda value: value.__setitem__(
            "schema_version",
            "alberta.forager_matched_final_analysis_runtime_source.v1",
        ),
        lambda value: value.__setitem__("promotion_authorized", 0),
        lambda value: cast(
            dict[str, Any],
            cast(dict[str, Any], value["sources"])["finalizer"],
        ).__setitem__(
            "size_bytes",
            float(
                cast(
                    dict[str, Any],
                    cast(dict[str, Any], value["sources"])["finalizer"],
                )["size_bytes"]
            ),
        ),
    )
    for mutate in mutations:
        aliased = cast(dict[str, Any], json.loads(json.dumps(expected)))
        mutate(aliased)
        with pytest.raises(
            final_analysis.ForagerMatchedFinalAnalysisError,
            match="runtime/source identity differs",
        ):
            final_analysis._parse_analysis_runtime_source(
                seal.canonical_json_bytes(aliased)
            )


def test_verification_request_authority_boundary_is_exact_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    seal_content = seal.load_forager_matched_seal_bundle_content(fixture.seal_root)
    artifacts = {
        name: (fixture.evaluation_root / name).read_bytes()
        for name in final_analysis._EVALUATION_ARTIFACTS
    }
    request = cast(
        dict[str, Any],
        json.loads(artifacts["verification-request.json"]),
    )
    boundary = cast(dict[str, Any], request["qualification_authority_boundary"])
    boundary["performance_claim"] = 0
    artifacts["verification-request.json"] = seal.canonical_json_bytes(request)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="qualification authority boundary drift",
    ):
        final_analysis._validate_evaluation_snapshot(
            artifacts,
            seal_content,
            fixture.evaluation_bindings,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("promotion_authorized", 0), ("horizon", 499_712.0)),
)
def test_receipt_index_header_rejects_json_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    sealed_protocol = fixture.completed.protocol
    scores = fixture.completed.score_evidence
    plan = fixture.completed.plan
    live = fixture.completed.live_runtime
    payload = cast(
        dict[str, Any],
        json.loads((fixture.evaluation_root / "execution-receipt-index.json").read_bytes()),
    )
    payload[field] = replacement
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    payload["payload_sha256"] = campaign._canonical_sha256(unsigned)
    with pytest.raises(
        final_analysis.ForagerMatchedFinalAnalysisError,
        match="receipt-index header closure drifted",
    ):
        final_analysis._validate_evaluation_receipt_index(
            payload,
            sealed_protocol,
            scores,
            plan_sha256=campaign._canonical_sha256(plan.to_dict()),
            live_runtime_sha256=campaign._canonical_sha256(live.unsigned_dict),
        )


@pytest.mark.parametrize("failure_ordinal", (1, 2))
def test_partial_pair_write_failure_removes_owned_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_ordinal: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    real_write_pair = seal._write_pair_at
    calls = 0

    def fail_partial_pair(
        opened: seal._OpenDirectory,
        name: str,
        raw: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_ordinal:
            raise OSError("injected partial pair write failure")
        real_write_pair(opened, name, raw)

    monkeypatch.setattr(seal, "_write_pair_at", fail_partial_pair)
    with pytest.raises(OSError, match="injected partial pair write failure"):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert output.parent.is_dir()
    assert list(output.parent.iterdir()) == []


def test_publish_then_base_exception_reports_uncertain_and_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "publication" / "final-analysis"
    real_publish = seal._publish_verified_no_replace

    def publish_then_interrupt(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        real_publish(
            parent,
            staging,
            source_name,
            destination_name,
            destination,
        )
        raise KeyboardInterrupt("injected post-rename interruption")

    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish_then_interrupt)
    with pytest.raises(final_analysis.PublishedFinalAnalysisUncertainError):
        final_analysis.create_forager_matched_final_analysis_bundle(
            fixture.qualification_root,
            fixture.seal_root,
            fixture.evaluation_root,
            output,
            **_creation_kwargs(fixture),
        )
    assert output.is_dir()


def test_final_command_template_matches_executor_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    plan = fixture.completed.plan
    templates = cast(list[dict[str, Any]], plan.to_dict()["candidate_command_templates"])
    for template, prepared in zip(templates, plan.candidates, strict=True):
        entrypoint = final_analysis._expected_entrypoint_binding(
            prepared.candidate.candidate_id
        )
        assert template["argv"] == executor._normalized_candidate_template(
            plan.protocol,
            prepared,
        )
        assert template["argv"] == final_analysis._expected_candidate_command_template(
            plan.protocol,
            prepared.candidate,
            entrypoint,
        )
