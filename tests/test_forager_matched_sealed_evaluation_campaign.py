"""Lifecycle tests for
:mod:`alberta_framework.benchmarks.forager_matched_sealed_evaluation_campaign`.

The sealed-evaluation engine executes the held-out 6-candidate x 30-seed
block after the seal: it re-authenticates the caller-pinned seal before any
mutating step, reuses the append-only campaign engine, and finalizes into
stage-specific unresolved content (never an authority claim).  This suite
covers the engine lifecycle on a synthetic in-memory sealed context
(``_sealed_context``): exact 6x30 manifest and inventory, authenticate-
before-publish event ordering, read-only verification, bound-raw recovery
and completion-pointer repair, and rejection of false-authority completion
summaries.

Deeper tamper and authentication attacks around a *real* seal bundle live in
the twin suite ``test_forager_matched_sealed_evaluation_campaign_adversarial``.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evaluation_campaign as schedule
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from alberta_framework.benchmarks import forager_matched_seal as seal
from alberta_framework.benchmarks import (
    forager_matched_sealed_evaluation_campaign as sealed_campaign,
)
from alberta_framework.benchmarks import forager_matched_trust as trust
from tests import test_forager_matched_executor as executor_fixtures
from tests import test_forager_matched_final_analysis as final_analysis_fixtures
from tests import test_forager_matched_open_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _qualification_material(label: str) -> tuple[dict[str, Any], bytes, str]:
    manifest = {
        "schema_version": "test.qualification.v2",
        "label": f"café-{label}",
    }
    raw = qualification._canonical_json_bytes(manifest)
    return manifest, raw, hashlib.sha256(raw).hexdigest()


def _execution_ready_protocol(
    frozen: protocol.ForagerMatchedProtocol,
    base: executor.PreparedCandidate,
) -> tuple[protocol.ForagerMatchedProtocol, dict[str, dict[str, Any]]]:
    """Rebind synthetic candidates to one real, fully verifiable test source."""
    candidates: list[protocol.MatchedCandidate] = []
    receipts: dict[str, dict[str, Any]] = {}
    for original in frozen.candidates:
        candidate = replace(
            original,
            implementation_kind=base.candidate.implementation_kind,
            entrypoint_family=base.candidate.entrypoint_family,
            source=base.candidate.source,
            configuration=base.candidate.configuration,
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
            entrypoint=base.entrypoint_path,
            invocation_style=base.invocation_style,
            result_root=base.result_root,
            patch_sha256=base.rng_isolation_patch_sha256,
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
    return (
        replace(
            frozen,
            candidates=candidate_tuple,
            candidate_index=MappingProxyType(
                {item.candidate_id: item for item in candidate_tuple}
            ),
        ),
        receipts,
    )


def _sealed_context(tmp_path: Path) -> sealed_campaign._SealedContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base_plan_root = tmp_path / "base-plan"
    base_plan_root.mkdir()
    base_plan = executor_fixtures._plan(base_plan_root)
    base_candidate = base_plan.candidates[0]
    open_value = protocol_fixtures._build()
    open_value, receipts = _execution_ready_protocol(open_value, base_candidate)
    result_payload = {
        "schema_version": protocol.FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
        "open_protocol_sha256": open_value.protocol_sha256,
        "selection_plan_sha256": open_value.selection_plan.plan_sha256,
        "tuning_seeds": list(open_value.tuning_seeds),
        "ranked_groups": [
            {
                "selection_group": group.selection_group,
                "ranked_candidate_ids": list(reversed(group.candidate_ids)),
                "ranking_evidence_sha256": _sha(
                    f"sealed-ranking:{group.selection_group}"
                ),
            }
            for group in open_value.selection_plan.groups
        ],
    }
    selection = protocol.parse_forager_matched_selection_result(result_payload)
    sealed_protocol = protocol.seal_forager_matched_protocol(open_value, selection)
    transition = protocol.validate_sealed_protocol_transition(
        open_value,
        sealed_protocol,
        selection,
        selection.selection_result_sha256,
    )
    evaluation_schedule = schedule.build_sealed_evaluation_schedule(
        sealed_protocol,
        transition,
    )

    qualification_manifest, qualification_bytes, qualification_digest = (
        _qualification_material("sealed-context")
    )
    assets = {
        candidate_id: executor.CandidateExecutionAssets(
            candidate_id=candidate_id,
            source_root=base_candidate.source_root,
            source_archive=base_candidate.source_archive,
            source_inventory=base_candidate.source_inventory,
            original_configuration=base_candidate.original_configuration,
            configuration=base_candidate.configuration,
            capability_receipt=receipts[candidate_id],
        )
        for candidate_id in transition.evaluation_candidate_ids
    }
    (
        plan,
        live,
        _seed_artifacts,
        _receipt_index,
        sealed_scores,
        _score_request,
        _score_bindings,
    ) = final_analysis_fixtures._panel(
        tmp_path,
        sealed_protocol,
        transition.evaluation_candidate_ids,
        base_plan,
        assets,
        label="sealed-context",
        qualification_manifest_sha256=qualification_digest,
    )

    qualification_root = tmp_path / "qualification"
    seal_root = tmp_path / "seal"
    qualification_root.mkdir()
    seal_root.mkdir()
    bundle = cast(
        Any,
        SimpleNamespace(
            output_root=qualification_root,
            manifest=qualification_manifest,
            manifest_bytes=qualification_bytes,
            manifest_sha256=qualification_digest,
            cpu_qualification_root=base_plan.cpu_qualification_root,
            rng_parity_qualification_root=base_plan.rng_parity_qualification_root,
        ),
    )
    rebuilt = campaign._RebuiltInputs(
        bundle=bundle,
        protocol=sealed_protocol,
        plan=plan,
        candidate_ids=transition.evaluation_candidate_ids,
        assets={},
        schedule=evaluation_schedule,
    )
    open_request_values = {
        "stage": "open_tuning",
        "protocol_sha256": open_value.protocol_sha256,
        "qualification_manifest_sha256": qualification_digest,
        "score_evidence_sha256": _sha("open-score"),
        "source_manifest_sha256": _sha("open-source"),
        "executor_manifest_sha256": _sha("open-executor"),
        "execution_closure_sha256": _sha("open-closure"),
        "trust_anchor_identity": (
            open_value.runtime.qualification_trust_anchor_identity
        ),
    }
    open_subject = campaign._canonical_sha256(
        {
            "schema_version": evidence.MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
            **open_request_values,
        }
    )
    open_request = executor.VerificationRequest(
        **cast(Any, open_request_values),
        verification_subject_sha256=open_subject,
        qualification_authority_boundary={
            "endorsement_created": False,
            "endorsements_at_seal": 0,
            "gpu_qualified": False,
            "performance_claim": False,
            "seed_class": "open_development",
            "trust_profile_created": False,
            "trust_profiles_at_seal": 0,
        },
        rng_parity_qualification_status=(
            "content_complete_external_executor_receipt_unverified"
        ),
    )
    transition_descriptor = schedule.build_sealed_transition_descriptor(
        sealed_protocol,
        transition,
    )
    transition_sha256 = schedule.canonical_sealed_transition_descriptor_sha256(
        sealed_protocol,
        transition,
    )
    seal_content = seal.ContentVerifiedSealBundle(
        output_root=seal_root,
        manifest={
            "payload_sha256": _sha("seal-manifest"),
            "open_campaign": {
                "qualification_manifest_sha256": qualification_digest
            },
        },
        open_protocol=open_value,
        open_score_evidence=sealed_scores,
        open_verification_request=open_request,
        recorded_bindings_cache={
            "qualification_manifest_sha256": qualification_digest,
        },
        selection_result=selection,
        selection_report={},
        sealed_protocol=sealed_protocol,
        sealed_transition=transition_descriptor,
        sealed_transition_sha256=transition_sha256,
    )
    inputs = sealed_campaign._SealedInputs(
        rebuilt=rebuilt,
        seal_content=seal_content,
        transition=transition,
        seal_manifest_payload_sha256=_sha("seal-manifest"),
        open_verification_subject_sha256=open_subject,
    )
    engine = campaign._CampaignContext(
        root=tmp_path / "campaign",
        rebuilt=rebuilt,
        live_runtime=live,
    )
    return sealed_campaign._SealedContext(engine=engine, inputs=inputs)


def _status(context: sealed_campaign._SealedContext) -> campaign.CampaignStatus:
    return campaign.CampaignStatus(
        output_root=context.engine.root,
        state="in_progress",
        completed_cells=0,
        total_cells=180,
        next_candidate_id=context.engine.rebuilt.candidate_ids[0],
        next_seed=context.engine.rebuilt.protocol.active_seeds[0],
        protocol_sha256=context.engine.rebuilt.protocol.protocol_sha256,
        qualification_manifest_sha256=(
            context.engine.rebuilt.bundle.manifest_sha256
        ),
        plan_sha256=context.engine.rebuilt.plan.plan_sha256,
        live_runtime_identity_sha256=context.engine.live_runtime.identity_sha256,
        score_evidence_sha256=None,
        verification_subject_sha256=None,
    )


def _publish_context(context: sealed_campaign._SealedContext) -> None:
    prospective = sealed_campaign._prospective_output(
        context.inputs,
        context.engine.root,
    )
    sealed_campaign._publish_initial_root(
        context.inputs,
        context.engine.live_runtime,
        context.engine.root,
        prospective,
    )


def _artifact(
    context: sealed_campaign._SealedContext,
    candidate_id: str,
    seed: int,
    raw_sha256: str,
    raw_size: int,
) -> executor.SeedExecutionArtifacts:
    return executor._artifact_mappings(
        plan=context.engine.rebuilt.plan,
        candidate=context.engine.rebuilt.plan.candidate_index[candidate_id],
        seed=seed,
        raw_archive_sha256=raw_sha256,
        raw_archive_size=raw_size,
        live_runtime=context.engine.live_runtime,
        scorer_record={
            "fov_last_10pct_ema_auc": 0.5,
            "npz_sha256": _sha(f"npz:{candidate_id}:{seed}"),
            "npz_size_bytes": 4096,
            "reward_trace_sha256": _sha(f"trace:{candidate_id}:{seed}"),
            "reward_dtype": "<f4",
            "reward_shape": [context.engine.rebuilt.protocol.horizon],
        },
    )


def _pins(context: sealed_campaign._SealedContext) -> dict[str, Any]:
    return {
        "resolver": cast(Any, lambda _request: None),
        "expected_trust_anchor_identity": (
            context.inputs.seal_content.open_verification_request.trust_anchor_identity
        ),
        "expected_seal_manifest_payload_sha256": (
            context.inputs.seal_manifest_payload_sha256
        ),
        "expected_open_verification_subject_sha256": (
            context.inputs.open_verification_subject_sha256
        ),
    }


def test_exact_six_by_thirty_manifest_and_initial_inventory(tmp_path: Path) -> None:
    context = _sealed_context(tmp_path)
    manifest = sealed_campaign._campaign_manifest(
        context.inputs,
        context.engine.live_runtime,
    )

    assert len(context.engine.rebuilt.candidate_ids) == 6
    assert len(context.engine.rebuilt.protocol.active_seeds) == 30
    assert len(context.engine.rebuilt.schedule["cells"]) == 180
    assert manifest["candidate_order"] == list(context.engine.rebuilt.candidate_ids)
    assert manifest["selected_candidate_ids"] == list(
        context.engine.rebuilt.candidate_ids[:4]
    )
    assert manifest["fixed_descriptive_candidate_ids"] == list(
        context.engine.rebuilt.candidate_ids[4:]
    )
    assert manifest["schema_version"] == (
        sealed_campaign.MATCHED_SEALED_EVALUATION_CAMPAIGN_SCHEMA_VERSION
    )
    assert manifest["qualification_manifest_sha256"] == (
        context.engine.rebuilt.bundle.manifest_sha256
    )
    assert manifest["content_capture_threat_boundary"] == {
        "campaign_tree_writers": "cooperative_lock_participants_trusted",
        "root_guard_scope": "top_level_inode_and_path_swap_detection_only",
        "noncooperative_same_uid_writers": "out_of_scope",
        "claim_authority": "independent_final_subject_authentication_required",
    }
    # These source-derived identities close transitively over the qualified
    # container helper through the executor manifest, plan, and live runtime.
    golden_payload_sha256 = {
        "sealed-protocol.json": (
            "cc7a06b9fdc006de5fb20c1fe2806ce45eaa5e068de9fcb9a302236ce0405b59"
        ),
        "sealed-transition.json": (
            "d7b44946802c0ee08445dcd3e794fab3b30f9c779fdb3097df923a0ccb73c1be"
        ),
        "seal-manifest.json": (
            "aad445405166220d829a425c269fe92cb5f73cbb298fb08efafab2990f13b355"
        ),
        "candidate-universe.json": (
            "6a9315cb996fe5698e4c1580d30da9b0524e9875ce085d1399bb975cc5b510a8"
        ),
        "execution-plan.json": (
            "29517b5739c150efff9dfabe09720686520f38ec9f9715cc6b5c50ebf95e26a3"
        ),
        "source-manifest.json": (
            "a1d727703b24ef6f96dc93b4fc306b9426ad2cfd054ba371aa84ac4c14dd7a68"
        ),
        "executor-manifest.json": (
            "e27cf3e16832c903435ef4da15e7e32afa834cab950faaceb6c08d2f9cf15462"
        ),
        "qualification-manifest.json": (
            "d54fb71f96641d8cbe1eb3b78a3ac9aef3e680b4ea4efba14181dd74a5c1332f"
        ),
        "execution-schedule.json": (
            "2859d85fbc9d44960e1b37fedf26589e2f3c5155b9356083ba7b5a427443f58f"
        ),
        "live-runtime.json": (
            "bd88fbfa40b57023836d05525df429601864686be7446dc8b5116c88594c0421"
        ),
        "campaign.json": (
            "c563582c1c40176af9eadf3a98475e99ea2851a36a8aad6ab14571f5198b1cd1"
        ),
    }
    _publish_context(context)
    expected = {
        "runs",
        "completions",
        *sealed_campaign._IMMUTABLE_ARTIFACTS,
        *(f"{name}.sha256" for name in sealed_campaign._IMMUTABLE_ARTIFACTS),
    }
    assert {path.name for path in context.engine.root.iterdir()} == expected
    assert (context.engine.root / "qualification-manifest.json").read_bytes() == (
        context.engine.rebuilt.bundle.manifest_bytes
    )
    for name, expected_sha256 in golden_payload_sha256.items():
        assert hashlib.sha256((context.engine.root / name).read_bytes()).hexdigest() == (
            expected_sha256
        )
        assert (context.engine.root / f"{name}.sha256").read_bytes() == (
            f"{expected_sha256}\n".encode("ascii")
        )


def test_prepare_auth_failure_creates_no_output_or_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "uncreated" / "campaign"
    monkeypatch.setattr(
        sealed_campaign,
        "_rebuild_sealed_inputs",
        lambda *_args: context.inputs,
    )
    monkeypatch.setattr(
        campaign,
        "_qualify_live",
        lambda *_args: context.engine.live_runtime,
    )

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("authority rejected")

    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", reject)
    with pytest.raises(RuntimeError, match="authority rejected"):
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            output,
            **_pins(context),
        )
    assert not output.parent.exists()


def test_prepare_event_order_authenticates_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "campaign"
    events: list[str] = []

    def rebuild(*_args: Any) -> sealed_campaign._SealedInputs:
        events.append("replay")
        return context.inputs

    def qualify(*_args: Any) -> executor.LiveRuntimeIdentity:
        events.append("qualify")
        return context.engine.live_runtime

    def authenticate(*_args: Any, **kwargs: Any) -> None:
        events.append("authenticate")
        assert kwargs["expected_seal_manifest_payload_sha256"] == _sha(
            "seal-manifest"
        )

    def publish(*_args: Any) -> Path:
        events.append("publish")
        return output

    def verify(*_args: Any, **_kwargs: Any) -> campaign.CampaignStatus:
        events.append("verify")
        return _status(context)

    monkeypatch.setattr(sealed_campaign, "_rebuild_sealed_inputs", rebuild)
    monkeypatch.setattr(campaign, "_qualify_live", qualify)
    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", authenticate)
    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        verify,
    )

    sealed_campaign.prepare_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        output,
        **_pins(context),
    )
    assert events == ["replay", "qualify", "authenticate", "publish", "verify"]


def test_postpublication_verify_failure_is_uncertain_and_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path / "fixture")
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        sealed_campaign,
        "_rebuild_sealed_inputs",
        lambda *_args: context.inputs,
    )
    monkeypatch.setattr(
        campaign,
        "_qualify_live",
        lambda *_args: context.engine.live_runtime,
    )
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: None,
    )

    def publish(*_args: Any) -> Path:
        output.mkdir()
        (output / "preserved").write_bytes(b"published")
        return output

    monkeypatch.setattr(sealed_campaign, "_publish_initial_root", publish)
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("replay failed")),
    )
    with pytest.raises(
        sealed_campaign.PublishedSealedEvaluationCampaignUncertainError
    ) as captured:
        sealed_campaign.prepare_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            output,
            **_pins(context),
        )
    assert captured.value.destination == output
    assert (output / "preserved").read_bytes() == b"published"


def test_read_only_verification_never_authenticates_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    monkeypatch.setattr(
        sealed_campaign,
        "_load_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: pytest.fail("read-only API resolved authority"),
    )
    before = {
        path.relative_to(context.engine.root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in context.engine.root.rglob("*")
        if path.is_file()
    }
    status = sealed_campaign.verify_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
    )
    after = {
        path.relative_to(context.engine.root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in context.engine.root.rglob("*")
        if path.is_file()
    }
    assert status.total_cells == 180
    assert status.completed_cells == 0
    assert after == before


def test_run_recovers_bound_raw_then_repairs_missing_completion_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    monkeypatch.setattr(
        sealed_campaign,
        "_load_context",
        lambda *_args, **_kwargs: context,
    )
    auth_calls: list[str] = []
    monkeypatch.setattr(
        sealed_campaign,
        "_authenticate_for_mutation",
        lambda *_args, **_kwargs: auth_calls.append("auth"),
    )
    execute_calls: list[tuple[str, int]] = []
    score_calls: list[tuple[str, int]] = []

    def fail_after_raw(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        raw_path: Path,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        execute_calls.append((candidate_id, seed))
        raw_path.write_bytes(f"opaque:{candidate_id}:{seed}".encode("ascii"))
        raise RuntimeError("injected scorer failure")

    def score_raw(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        _raw_path: Path,
        binding: Any,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        score_calls.append((candidate_id, seed))
        raw = binding["raw_archive"]
        return _artifact(
            context,
            candidate_id,
            seed,
            cast(str, raw["sha256"]),
            cast(int, raw["size_bytes"]),
        )

    monkeypatch.setattr(campaign, "_execute_with_optional_runner", fail_after_raw)
    monkeypatch.setattr(campaign, "_score_with_optional_runner", score_raw)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="matched execution"):
        sealed_campaign.run_sealed_evaluation_campaign(
            tmp_path / "qualification",
            tmp_path / "seal",
            context.engine.root,
            max_cells=1,
            **_pins(context),
        )
    status = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        max_cells=1,
        **_pins(context),
    )
    assert status.completed_cells == 1
    assert len(execute_calls) == 1
    assert len(score_calls) == 1

    candidate_id, seed = execute_calls[0]
    _run, pointer = campaign._cell_paths(context.engine.root, candidate_id, seed)
    pointer.unlink()
    pointer.with_name(f"{pointer.name}.sha256").unlink()
    repaired = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        max_cells=1,
        **_pins(context),
    )
    assert repaired.completed_cells == 1
    assert pointer.is_file()
    assert len(execute_calls) == 1
    assert len(score_calls) == 1
    assert len(auth_calls) == 3


@pytest.mark.slow
def test_full_evaluation_finalizes_stage_specific_unresolved_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sealed_context(tmp_path)
    _publish_context(context)
    events: list[str] = []

    def load(*_args: Any, **_kwargs: Any) -> sealed_campaign._SealedContext:
        events.append("replay")
        return context

    def authenticate(*_args: Any, **_kwargs: Any) -> None:
        events.append("authenticate")

    def execute(
        _current: campaign._CampaignContext,
        candidate_id: str,
        seed: int,
        raw_path: Path,
        _runner: executor.ProcessRunner | None,
    ) -> executor.SeedExecutionArtifacts:
        if not any(event == "execute" for event in events):
            events.append("execute")
        raw = f"opaque:{candidate_id}:{seed}".encode("ascii")
        raw_path.write_bytes(raw)
        return _artifact(
            context,
            candidate_id,
            seed,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        )

    monkeypatch.setattr(sealed_campaign, "_load_context", load)
    monkeypatch.setattr(sealed_campaign, "_authenticate_for_mutation", authenticate)
    monkeypatch.setattr(campaign, "_execute_with_optional_runner", execute)

    status = sealed_campaign.run_sealed_evaluation_campaign(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
        **_pins(context),
    )
    assert events[:3] == ["replay", "authenticate", "execute"]
    assert status.state == "complete_content_only_external_verification_unresolved"
    assert status.completed_cells == status.total_cells == 180
    bundle = sealed_campaign.load_completed_sealed_evaluation_campaign_content(
        tmp_path / "qualification",
        tmp_path / "seal",
        context.engine.root,
    )
    assert bundle.candidate_ids == context.inputs.transition.evaluation_candidate_ids
    assert bundle.completion_summary["stage"] == "sealed_evaluation"
    assert bundle.completion_summary["selection_inherited_from_seal"] is True
    assert bundle.completion_summary["promotion_authorized"] is False
    assert bundle.completion_summary[
        "evaluation_verification_authentication_state"
    ] == "unresolved_external_verifier_required"
    assert bundle.completion_summary["sealed_transition_sha256"] == (
        context.inputs.seal_content.sealed_transition_sha256
    )
    # Per-cell bundles bind the same live-runtime identity, so this closure
    # changes deterministically when the qualified helper bytes change.
    golden_final_sha256 = {
        "execution-receipt-index.json": (
            "39642af77c69525d6f669c94648d15d57d54b51f17ef0145e0efad672ea781fd"
        ),
        "score-evidence.json": (
            "41d0fbbdb00e63745d0dc1415529147f843b0dc04d81bf1e7ac956b3269613e3"
        ),
        "verification-request.json": (
            "b7c092fba2bb4b5cbb6b74440f8a6e31860ed649e12365437a912e18f14dae9c"
        ),
        "completion-summary.json": (
            "a3e52dfcef4982579b44444ed71934257cff7f94adf1f17d7669bae69e87237f"
        ),
    }
    for name, expected_sha256 in golden_final_sha256.items():
        assert hashlib.sha256((context.engine.root / name).read_bytes()).hexdigest() == (
            expected_sha256
        )
        assert (context.engine.root / f"{name}.sha256").read_bytes() == (
            f"{expected_sha256}\n".encode("ascii")
        )


def test_false_authority_completion_summary_is_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    context = _sealed_context(tmp_path)
    receipt = cast(Any, SimpleNamespace(payload_sha256=_sha("receipt")))
    qualification_digest = context.engine.rebuilt.bundle.manifest_sha256
    scores = cast(
        Any,
        SimpleNamespace(
            payload_sha256=_sha("scores"),
            qualification_manifest_sha256=qualification_digest,
        ),
    )
    request = cast(
        Any,
        SimpleNamespace(
            verification_subject_sha256=_sha("final-subject"),
            qualification_manifest_sha256=qualification_digest,
        ),
    )
    summary = sealed_campaign._completion_summary(
        context.inputs,
        context.engine,
        receipt,
        scores,
        request,
    )
    assert summary["schema_version"] == (
        sealed_campaign.MATCHED_SEALED_EVALUATION_COMPLETION_SCHEMA_VERSION
    )
    assert summary["qualification_manifest_sha256"] == qualification_digest
    assert hashlib.sha256(campaign.canonical_json_bytes(summary)).hexdigest() == (
        "23a9c7a3e936fcaaf87cc40cea11a05c236a078022541f057f2ab1a8b05514ae"
    )
    legacy = dict(summary)
    legacy["schema_version"] = "alberta.forager_matched_sealed_evaluation_completion.v1"
    with pytest.raises(ValueError, match="exact v2 schema"):
        sealed_campaign._summary_validator(context.inputs)(
            context.engine,
            receipt,
            scores,
            request,
            legacy,
        )
    summary["promotion_authorized"] = True
    summary["protocol_sha256"] = _sha("wrong-protocol")

    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="common closure/authority invariants",
    ):
        sealed_campaign._summary_validator(context.inputs)(
            context.engine,
            receipt,
            scores,
            request,
            summary,
        )
    assert not any(
        (context.engine.root / name).exists()
        for name in campaign._FINAL_ARTIFACTS
    )


# --- CLI surface (``main``): canonical JSON out, fail-closed JSON errors ----


def _cli_status_stub(tmp_path: Path) -> campaign.CampaignStatus:
    return campaign.CampaignStatus(
        output_root=tmp_path / "campaign",
        state="in_progress",
        completed_cells=0,
        total_cells=180,
        next_candidate_id="candidate",
        next_seed=3_000_001,
        protocol_sha256="1" * 64,
        qualification_manifest_sha256="2" * 64,
        plan_sha256="3" * 64,
        live_runtime_identity_sha256="4" * 64,
        score_evidence_sha256=None,
        verification_subject_sha256=None,
    )


def _cli_roots(tmp_path: Path) -> list[str]:
    return [
        "--qualification-root",
        str(tmp_path / "qualification"),
        "--seal-root",
        str(tmp_path / "seal"),
        "--output-root",
        str(tmp_path / "campaign"),
    ]


def _cli_mutation_pins(tmp_path: Path) -> list[str]:
    return [
        "--trust-anchor",
        str(tmp_path / "anchor.json"),
        "--trust-key",
        str(tmp_path / "ceremony.key"),
        "--trust-receipt-directory",
        str(tmp_path / "receipts"),
        "--expected-trust-anchor-identity",
        "ceremony_anchor_v1",
        "--expected-seal-manifest-payload-sha256",
        "5" * 64,
        "--expected-open-verification-subject-sha256",
        "6" * 64,
    ]


def test_cli_help_exits_zero_and_names_all_five_operations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        sealed_campaign.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for operation in ("prepare", "run", "verify", "status", "load"):
        assert operation in help_text
        with pytest.raises(SystemExit) as operation_exit:
            sealed_campaign.main([operation, "--help"])
        assert operation_exit.value.code == 0
        capsys.readouterr()


def test_cli_usage_errors_bypass_flags_and_missing_pins_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as no_operation:
        sealed_campaign.main([])
    assert no_operation.value.code == 2
    run_arguments = ["run", *_cli_roots(tmp_path), *_cli_mutation_pins(tmp_path)]
    for bypass in (
        "--force",
        "--skip-seal-authentication",
        "--accept-cached-bindings",
        "--repair",
        "--promote",
    ):
        with pytest.raises(SystemExit) as rejected:
            sealed_campaign.main([*run_arguments, bypass])
        assert rejected.value.code == 2
    # Read-only operations expose no authentication material at all.
    with pytest.raises(SystemExit) as verify_pins:
        sealed_campaign.main(
            [
                "verify",
                *_cli_roots(tmp_path),
                "--expected-trust-anchor-identity",
                "anchor",
            ]
        )
    assert verify_pins.value.code == 2
    # Mutating operations refuse to parse without the full pin set.
    partial = [
        argument
        for argument in _cli_mutation_pins(tmp_path)
        if argument != "--expected-seal-manifest-payload-sha256" and argument != "5" * 64
    ]
    with pytest.raises(SystemExit) as missing_pin:
        sealed_campaign.main(["prepare", *_cli_roots(tmp_path), *partial])
    assert missing_pin.value.code == 2
    capsys.readouterr()


def test_cli_status_and_verify_write_canonical_status_json_and_forward_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _cli_status_stub(tmp_path)
    calls: list[tuple[str, Path, Path, Path, dict[str, Any]]] = []

    def record(name: str) -> Any:
        def entry(
            qualification_root: Path,
            seal_root: Path,
            output_root: Path,
            **kwargs: Any,
        ) -> campaign.CampaignStatus:
            calls.append((name, qualification_root, seal_root, output_root, kwargs))
            return status

        return entry

    monkeypatch.setattr(
        sealed_campaign,
        "sealed_evaluation_campaign_content_status",
        record("status"),
    )
    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        record("verify"),
    )
    for operation in ("status", "verify"):
        assert sealed_campaign.main(
            [operation, *_cli_roots(tmp_path), "--runtime", "podman"]
        ) == 0
        captured = capsys.readouterr()
        assert captured.out.encode("ascii") == campaign.canonical_json_bytes(
            status.to_dict()
        )
        assert captured.err == ""
    assert calls == [
        (
            operation,
            tmp_path / "qualification",
            tmp_path / "seal",
            tmp_path / "campaign",
            {"runtime": "podman"},
        )
        for operation in ("status", "verify")
    ]


def test_cli_prepare_and_run_build_reference_resolver_and_forward_exact_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _cli_status_stub(tmp_path)
    calls: list[tuple[str, Path, Path, Path, dict[str, Any]]] = []

    def record(name: str) -> Any:
        def entry(
            qualification_root: Path,
            seal_root: Path,
            output_root: Path,
            **kwargs: Any,
        ) -> campaign.CampaignStatus:
            calls.append((name, qualification_root, seal_root, output_root, kwargs))
            return status

        return entry

    monkeypatch.setattr(
        sealed_campaign,
        "prepare_sealed_evaluation_campaign",
        record("prepare"),
    )
    monkeypatch.setattr(
        sealed_campaign,
        "run_sealed_evaluation_campaign",
        record("run"),
    )
    assert sealed_campaign.main(
        ["prepare", *_cli_roots(tmp_path), *_cli_mutation_pins(tmp_path)]
    ) == 0
    assert sealed_campaign.main(
        [
            "run",
            *_cli_roots(tmp_path),
            *_cli_mutation_pins(tmp_path),
            "--max-cells",
            "7",
        ]
    ) == 0
    capsys.readouterr()
    assert [call[0] for call in calls] == ["prepare", "run"]
    for name, qualification_root, seal_root, output_root, kwargs in calls:
        assert qualification_root == tmp_path / "qualification"
        assert seal_root == tmp_path / "seal"
        assert output_root == tmp_path / "campaign"
        resolver = kwargs["resolver"]
        assert isinstance(resolver, trust.SignedReceiptTrustResolver)
        assert resolver.anchor_path == tmp_path / "anchor.json"
        assert resolver.key_path == tmp_path / "ceremony.key"
        assert resolver.receipt_directory == tmp_path / "receipts"
        assert kwargs["expected_trust_anchor_identity"] == "ceremony_anchor_v1"
        assert kwargs["expected_seal_manifest_payload_sha256"] == "5" * 64
        assert kwargs["expected_open_verification_subject_sha256"] == "6" * 64
        assert kwargs["runtime"] == "docker"
        if name == "run":
            assert kwargs["max_cells"] == 7
        else:
            assert "max_cells" not in kwargs


def test_cli_load_writes_canonical_completed_content_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = cast(
        Any,
        SimpleNamespace(
            output_root=tmp_path / "campaign",
            protocol=SimpleNamespace(protocol_sha256=_sha("protocol")),
            plan=SimpleNamespace(plan_sha256=_sha("plan")),
            live_runtime=SimpleNamespace(identity_sha256=_sha("live")),
            candidate_ids=("alpha", "beta"),
            active_seeds=(11, 12, 13),
            execution_receipt_index=SimpleNamespace(payload_sha256=_sha("receipts")),
            score_evidence=SimpleNamespace(payload_sha256=_sha("scores")),
            verification_request=SimpleNamespace(
                verification_subject_sha256=_sha("subject")
            ),
            final_file_sha256={"completion-summary.json": _sha("summary")},
        ),
    )
    monkeypatch.setattr(
        sealed_campaign,
        "load_completed_sealed_evaluation_campaign_content",
        lambda *_args, **_kwargs: bundle,
    )
    assert sealed_campaign.main(["load", *_cli_roots(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.encode("ascii") == campaign.canonical_json_bytes(
        sealed_campaign._cli_load_result(bundle)
    )
    payload = json.loads(captured.out)
    assert payload["schema_version"] == (
        "alberta.forager_matched_sealed_evaluation_campaign_cli_load_result.v1"
    )
    assert payload["stage"] == "sealed_evaluation"
    assert payload["candidate_order"] == ["alpha", "beta"]
    assert payload["completed_cell_count"] == 6
    assert payload["verification_subject_sha256"] == _sha("subject")
    assert payload["promotion_authorized"] is False
    assert payload["external_verification_required"] is True


def test_cli_maps_fail_closed_refusals_to_exit_two_with_json_error_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> campaign.CampaignStatus:
        raise sealed_campaign.ForagerMatchedSealedEvaluationCampaignError(
            "sealed evaluation campaign is not complete"
        )

    monkeypatch.setattr(
        sealed_campaign,
        "verify_sealed_evaluation_campaign_content",
        refuse,
    )
    assert sealed_campaign.main(["verify", *_cli_roots(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "schema_version": (
            "alberta.forager_matched_sealed_evaluation_campaign_cli_error.v1"
        ),
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "failed_closed",
        "error_type": "ForagerMatchedSealedEvaluationCampaignError",
        "error_message": "sealed evaluation campaign is not complete",
        "promotion_authorized": False,
        "external_verification_required": True,
    }

    def disk_gone(*_args: Any, **_kwargs: Any) -> campaign.CampaignStatus:
        raise OSError("campaign root device disappeared")

    monkeypatch.setattr(
        sealed_campaign,
        "sealed_evaluation_campaign_content_status",
        disk_gone,
    )
    assert sealed_campaign.main(["status", *_cli_roots(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error_payload = json.loads(captured.err)
    assert error_payload["status"] == "failed_closed"
    assert error_payload["error_type"] == "OSError"
    assert "Traceback" not in captured.err


def test_cli_maps_published_uncertain_to_exit_three_with_destination_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "campaign"

    def uncertain(*_args: Any, **_kwargs: Any) -> campaign.CampaignStatus:
        raise sealed_campaign.PublishedSealedEvaluationCampaignUncertainError(
            destination,
            "post-publication content replay failed",
        )

    monkeypatch.setattr(
        sealed_campaign,
        "prepare_sealed_evaluation_campaign",
        uncertain,
    )
    assert sealed_campaign.main(
        ["prepare", *_cli_roots(tmp_path), *_cli_mutation_pins(tmp_path)]
    ) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["status"] == "published_uncertain"
    assert payload["error_type"] == "PublishedSealedEvaluationCampaignUncertainError"
    assert payload["destination"] == destination.as_posix()
    assert f"campaign published at {destination}" in payload["error_message"]
    assert payload["promotion_authorized"] is False
    assert payload["external_verification_required"] is True


def test_cli_console_script_targets_module_main() -> None:
    project_root = Path(sealed_campaign.__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]
    assert scripts["alberta-forager-matched-sealed-evaluation"] == (
        "alberta_framework.benchmarks."
        "forager_matched_sealed_evaluation_campaign:main"
    )
    assert callable(sealed_campaign.main)
