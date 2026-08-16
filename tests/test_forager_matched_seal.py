"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_seal`.

The seal bundle is the boundary between completed open tuning and held-out
evaluation: it snapshots the completed-campaign closure, has the external
resolver authenticate that exact closure, computes the frozen selection, and
mechanically flips the protocol stage to ``sealed_evaluation``.  What is
sealed is content — the campaign closure, selection result, and transition
bytes; what is *not* conferred is authority: persisted resolver bindings are
cache only, and consumers must re-authenticate via the resolver.

The threat model exercised here is a same-UID local adversary attacking the
seal around its creation: cross-carrier qualification-manifest tampering,
self-authenticating caches, staging-name and parent-inode substitution races,
inventory mutation after artifact reads, path swaps under the loader, and
fsync/publication failures that must never leave a partially published or
silently replaced bundle.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_seal as seal
from tests import test_forager_matched_evidence as evidence_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_QUALIFICATION_MANIFEST_SHA256 = _sha("matched-current-qualification-manifest")


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(executor.canonical_json_bytes(value)).hexdigest()


def _completed_campaign(tmp_path: Path) -> tuple[Any, evidence.AuthenticatedEvidenceBindings]:
    open_payload, toy_protocol, _toy_scores = evidence_fixtures._open_fixture()
    open_payload["evaluation_seeds"] = list(range(2_200_001, 2_200_031))
    open_payload["selection_plan"]["groups"] = [
        {
            "selection_group": "alberta",
            "candidate_ids": ["alberta_causal", "alberta_route"],
            "advance_count": 1,
        },
        {
            "selection_group": "external",
            "candidate_ids": ["external_dqn", "isolated_rtu"],
            "advance_count": 2,
        },
    ]
    for candidate in open_payload["candidates"]:
        if candidate["candidate_id"] == "isolated_rtu":
            candidate["selection_group"] = "external"
            original = next(
                item
                for item in toy_protocol.candidates
                if item.candidate_id == "isolated_rtu"
            )
            candidate["runtime_binding"][
                "qualified_capability_descriptor_sha256"
            ] = protocol.candidate_capability_descriptor_sha256(
                replace(original, selection_group="external")
            )
    open_payload["evaluation_panel"]["selection_slots"] = [
        {"selection_group": "alberta", "rank": 1},
        {"selection_group": "external", "rank": 1},
        {"selection_group": "external", "rank": 2},
    ]
    open_payload["secondary_hypotheses"][0]["intervention_slot"] = {
        "selection_group": "external",
        "rank": 2,
    }
    open_protocol = protocol.parse_forager_matched_protocol(open_payload)
    candidate_ids = tuple(
        candidate_id
        for group in open_protocol.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    raw_scores = evidence_fixtures._score_payload(
        open_protocol,
        candidate_ids,
        score_by_candidate={
            "alberta_causal": (1.0, 1.0),
            "alberta_route": (3.0, 3.0),
            "external_dqn": (2.0, 2.0),
            "isolated_rtu": (2.5, 2.5),
        },
    )
    score_payload = {key: value for key, value in raw_scores.items()}
    score_payload["qualification_manifest_sha256"] = _QUALIFICATION_MANIFEST_SHA256
    source_manifest = {"schema_version": "test.source-manifest.v1"}
    executor_manifest = {
        "schema_version": executor.MATCHED_EXECUTOR_MANIFEST_SCHEMA_VERSION,
        "protocol_sha256": open_protocol.protocol_sha256,
        "qualification_manifest_sha256": _QUALIFICATION_MANIFEST_SHA256,
    }
    score_payload["source_evidence_sha256"] = _canonical_sha(source_manifest)
    score_payload["executor_evidence_sha256"] = _canonical_sha(executor_manifest)
    score_rows = [dict(item) for item in score_payload["candidate_scores"]]
    candidate_order = tuple(cast(str, item["candidate_id"]) for item in score_rows)
    plan_payload: dict[str, Any] = {
        "schema_version": executor.MATCHED_EXECUTION_PLAN_SCHEMA_VERSION,
        "classification": "matched_current_execution_candidate",
        "promotion_authorized": False,
        "external_verification_required": True,
        "stage": "open_tuning",
        "protocol_sha256": open_protocol.protocol_sha256,
        "qualification_manifest_sha256": _QUALIFICATION_MANIFEST_SHA256,
        "active_seeds": list(open_protocol.active_seeds),
        "horizon": open_protocol.horizon,
        "candidate_order": list(candidate_order),
        "source_manifest": source_manifest,
        "source_manifest_sha256": score_payload["source_evidence_sha256"],
        "executor_manifest": executor_manifest,
        "executor_manifest_sha256": score_payload["executor_evidence_sha256"],
        "candidate_command_templates": [],
        "scoring_boundary": {},
    }
    plan_sha256 = _canonical_sha(plan_payload)
    live_payload: dict[str, Any] = {
        "schema_version": executor.MATCHED_LIVE_RUNTIME_SCHEMA_VERSION,
        "executable_sha256": _sha("runtime-executable"),
        "version": {"client": "test"},
        "image_inspection": {"digest": "test"},
        "executor_manifest_sha256": score_payload["executor_evidence_sha256"],
    }
    live_sha256 = _canonical_sha(live_payload)
    indexed_receipts: list[executor.IndexedExecutionReceipt] = []
    for score_row in score_rows:
        records = [dict(item) for item in score_row["records"]]
        receipt_payload: dict[str, Any] = {
            "schema_version": executor.MATCHED_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "candidate_id": score_row["candidate_id"],
            "stage": "open_tuning",
            "protocol_sha256": open_protocol.protocol_sha256,
            "plan_sha256": plan_sha256,
            "source_manifest_sha256": score_payload["source_evidence_sha256"],
            "executor_manifest_sha256": score_payload["executor_evidence_sha256"],
            "capability_descriptor_sha256": score_row["capability_descriptor_sha256"],
            "capability_qualification_receipt_sha256": (
                score_row["capability_qualification_receipt_sha256"]
            ),
            "live_runtime_identity_sha256": live_sha256,
            "seed_artifacts": [
                {
                    "seed": item["seed"],
                    "raw_artifact_sha256": item["raw_artifact_sha256"],
                    "reward_trace_sha256": item["reward_trace_sha256"],
                    "scoring_record_sha256": item["scoring_record_sha256"],
                }
                for item in records
            ],
            "authentication_state": "content_complete_external_verifier_required",
        }
        receipt_sha256 = hashlib.sha256(
            executor.canonical_json_bytes(receipt_payload)
        ).hexdigest()
        score_row["execution_receipt_sha256"] = receipt_sha256
        indexed_receipts.append(
            executor.IndexedExecutionReceipt(
                candidate_id=cast(str, score_row["candidate_id"]),
                execution_receipt_sha256=receipt_sha256,
                receipt_payload=receipt_payload,
            )
        )
    score_payload["candidate_scores"] = score_rows
    score_payload = evidence_fixtures._rehash(score_payload)
    scores = evidence.parse_matched_score_evidence(score_payload)
    candidate_order = tuple(item.candidate_id for item in scores.candidate_scores)
    receipt_unsigned: dict[str, Any] = {
        "schema_version": executor.MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION,
        "classification": "content_complete_execution_receipt_preimages",
        "authentication_state": "content_only_unendorsed_external_verifier_required",
        "promotion_authorized": False,
        "external_verification_required": True,
        "stage": "open_tuning",
        "protocol_sha256": open_protocol.protocol_sha256,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "live_runtime_identity_sha256": live_sha256,
        "active_seeds": list(open_protocol.active_seeds),
        "horizon": open_protocol.horizon,
        "candidate_order": list(candidate_order),
        "execution_receipts": [item.to_dict() for item in indexed_receipts],
    }
    receipt_index = executor.MatchedExecutionReceiptIndex(
        schema_version=executor.MATCHED_EXECUTION_RECEIPT_INDEX_SCHEMA_VERSION,
        stage="open_tuning",
        protocol_sha256=open_protocol.protocol_sha256,
        plan_sha256=plan_sha256,
        source_manifest_sha256=scores.source_evidence_sha256,
        executor_manifest_sha256=scores.executor_evidence_sha256,
        live_runtime_identity_sha256=live_sha256,
        active_seeds=open_protocol.active_seeds,
        horizon=open_protocol.horizon,
        candidate_order=candidate_order,
        execution_receipts=tuple(indexed_receipts),
        payload_sha256=hashlib.sha256(
            executor.canonical_json_bytes(receipt_unsigned)
        ).hexdigest(),
    )
    request = executor.VerificationRequest(
        stage="open_tuning",
        protocol_sha256=open_protocol.protocol_sha256,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        score_evidence_sha256=scores.payload_sha256,
        source_manifest_sha256=scores.source_evidence_sha256,
        executor_manifest_sha256=scores.executor_evidence_sha256,
        execution_closure_sha256=evidence.matched_execution_closure_sha256(
            open_protocol, scores
        ),
        trust_anchor_identity=open_protocol.runtime.qualification_trust_anchor_identity,
        verification_subject_sha256=evidence.matched_verification_subject_sha256(
            open_protocol, scores
        ),
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
    bindings = evidence.AuthenticatedEvidenceBindings(
        stage="open_tuning",
        protocol_sha256=request.protocol_sha256,
        score_evidence_sha256=request.score_evidence_sha256,
        source_manifest_sha256=request.source_manifest_sha256,
        executor_manifest_sha256=request.executor_manifest_sha256,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        execution_closure_sha256=request.execution_closure_sha256,
        trust_anchor_identity=request.trust_anchor_identity,
        verification_subject_sha256=request.verification_subject_sha256,
        verification_receipt_sha256=_sha("open-external-verification-receipt"),
    )
    completion_summary: dict[str, Any] = {
        "schema_version": campaign.MATCHED_OPEN_COMPLETION_SCHEMA_VERSION,
        "classification": "content_only_unendorsed_nonpromoting",
        "status": "complete_content_only_external_verification_unresolved",
        "stage": "open_tuning",
        "protocol_sha256": open_protocol.protocol_sha256,
        "qualification_manifest_sha256": _QUALIFICATION_MANIFEST_SHA256,
        "execution_plan_sha256": plan_sha256,
        "source_manifest_sha256": scores.source_evidence_sha256,
        "executor_manifest_sha256": scores.executor_evidence_sha256,
        "live_runtime_identity_sha256": live_sha256,
        "candidate_count": len(candidate_order),
        "seed_count": len(open_protocol.active_seeds),
        "completed_cell_count": len(candidate_order) * len(open_protocol.active_seeds),
        "execution_receipt_index_payload_sha256": receipt_index.payload_sha256,
        "score_evidence_sha256": scores.payload_sha256,
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
    final_raw = {
        "execution-receipt-index.json": receipt_index.canonical_bytes,
        "score-evidence.json": scores.canonical_bytes,
        "verification-request.json": executor.canonical_json_bytes(request.to_dict()),
        "completion-summary.json": seal.canonical_json_bytes(completion_summary),
    }
    completed = SimpleNamespace(
        output_root=tmp_path / "open-campaign",
        protocol=open_protocol,
        plan=SimpleNamespace(
            plan_sha256=plan_sha256,
            canonical_bytes=executor.canonical_json_bytes(plan_payload),
        ),
        live_runtime=SimpleNamespace(
            identity_sha256=live_sha256,
            unsigned_dict=live_payload,
        ),
        candidate_ids=candidate_order,
        active_seeds=open_protocol.active_seeds,
        schedule={},
        seed_artifacts={},
        execution_receipt_index=receipt_index,
        score_evidence=scores,
        verification_request=request,
        completion_summary=completion_summary,
        final_file_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in final_raw.items()
        },
    )
    return completed, bindings


def _install_completed_loader(
    monkeypatch: pytest.MonkeyPatch,
    completed: Any,
) -> None:
    def load(
        _qualification_root: Path,
        _campaign_root: Path,
        *,
        runtime: str | Path = "docker",
        runner: executor.ProcessRunner | None = None,
    ) -> Any:
        del runtime, runner
        return completed

    monkeypatch.setattr(campaign, "load_completed_open_tuning_campaign", load)


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    qualification_root = tmp_path / "qualification"
    campaign_root = tmp_path / "open-campaign"
    qualification_root.mkdir()
    campaign_root.mkdir()
    return qualification_root, campaign_root, tmp_path / "seal"


def test_atomic_seal_round_trip_keeps_content_and_external_trust_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    resolver_calls: list[executor.VerificationRequest] = []

    def resolver(request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        resolver_calls.append(request)
        return bindings

    created = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=resolver,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )

    assert len(resolver_calls) == 1
    assert created.output_root == output_root
    assert created.sealed_protocol.stage == "sealed_evaluation"
    assert created.sealed_protocol.active_seeds == created.open_protocol.evaluation_seeds
    assert created.manifest["schema_version"] == "alberta.forager_matched_seal_bundle.v2"
    assert created.manifest["promotion_authorized"] is False
    assert created.manifest["evaluation_executed"] is False
    open_campaign = cast(dict[str, Any], dict(created.manifest["open_campaign"]))
    assert set(open_campaign) == {
        "protocol_sha256",
        "qualification_manifest_sha256",
        "execution_plan_sha256",
        "source_manifest_sha256",
        "executor_manifest_sha256",
        "live_runtime_identity_sha256",
        "execution_receipt_index_payload_sha256",
        "score_evidence_payload_sha256",
        "verification_subject_sha256",
        "candidate_order",
        "active_seeds",
        "completed_cell_count",
    }
    assert open_campaign["qualification_manifest_sha256"] == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert (
        created.manifest["sealed_transition"]["descriptor_sha256"]
        == created.sealed_transition_sha256
    )
    assert created.manifest["authority_boundary"] == {
        "persisted_bindings_are_cache_only": True,
        "external_resolver_revalidation_required": True,
        "self_authentication_forbidden": True,
    }
    golden_payload_sha256 = {
        "open-protocol.json": (
            "b11df9dc3ba36b18d963176334a1bb2fa475d970fe27131f56f58fd5af524fa6"
        ),
        "open-execution-plan.json": (
            "6a3fb963c78b3e93046a2ef219915f5df380d615cc53681028aa236284c61804"
        ),
        "open-live-runtime.json": (
            "0f8c71ed1924a8910a68c06cc40bac7ad0db3cb54a8b0542f34464cf7379c22c"
        ),
        "open-execution-receipt-index.json": (
            "2a80de622714094238cf5df2a3cf007e78171a14721a32d1e31039cafad27d3d"
        ),
        "open-score-evidence.json": (
            "bf491e9afabb27b9d4853c8bcc1cebc28b911dd735aef0be3f7b7678bb8d1721"
        ),
        "open-verification-request.json": (
            "dc71ffb09e59239b489ac52d8d1e33a6aafdb0a85e2e516703886ba22691c8b3"
        ),
        "open-completion-summary.json": (
            "9f1d6e4ce1063c732a78b55af177d66585e8f527e9aa6e44046edd80d5e0f09e"
        ),
        "open-authenticated-bindings-cache.json": (
            "ee4227a0d07adb11f6d65ca5491c61add60b6d57ddff9f1b87e40a13a9de30bb"
        ),
        "selection-result.json": (
            "dc7599abf1a4a282b40ffad7389a55388310095b9ef0c8a2920793e7629019b0"
        ),
        "selection-report.json": (
            "441fb17f029aec5b396938a9ef75236663b31266bb83d47f8d0ee7a34df5892f"
        ),
        "sealed-protocol.json": (
            "f8b1dd99afc8dc4324455af347343e090b6dfb90cbcba49cf2cc64d32d27eda8"
        ),
        "seal.json": (
            "4e49e71b1b25f14b49bf9a89156d86c3f1a51f2d69da8e348373962a5e359651"
        ),
    }
    for name, expected_sha256 in golden_payload_sha256.items():
        assert hashlib.sha256((output_root / name).read_bytes()).hexdigest() == (
            expected_sha256
        )
        assert (output_root / f"{name}.sha256").read_bytes() == (
            f"{expected_sha256}\n".encode("ascii")
        )
    assert created.manifest["payload_sha256"] == (
        "ada732ee000d894dd262aa47310edfd22f97afc1d61314571f231db0b65cf20d"
    )

    content_only = seal.load_forager_matched_seal_bundle_content(output_root)
    assert len(resolver_calls) == 1
    assert not isinstance(
        content_only.recorded_bindings_cache,
        evidence.AuthenticatedEvidenceBindings,
    )
    assert content_only.selection_result == created.selection_result
    assert content_only.open_score_evidence.qualification_manifest_sha256 == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert content_only.open_verification_request.qualification_manifest_sha256 == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert content_only.recorded_bindings_cache["qualification_manifest_sha256"] == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    authenticated = seal.authenticate_forager_matched_seal_bundle(
        content_only,
        resolver=resolver,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    assert len(resolver_calls) == 2
    assert authenticated == bindings


def test_qualification_manifest_cross_carrier_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    changed_digest = _sha("different-qualification-manifest")
    plan_payload = seal._decode_canonical(
        completed.plan.canonical_bytes,
        "test execution plan",
    )
    receipt_index = seal._decode_canonical(
        completed.execution_receipt_index.canonical_bytes,
        "test receipt index",
    )

    assert plan_payload["qualification_manifest_sha256"] == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert completed.score_evidence.qualification_manifest_sha256 == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert completed.verification_request.qualification_manifest_sha256 == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert bindings.qualification_manifest_sha256 == _QUALIFICATION_MANIFEST_SHA256
    assert completed.completion_summary["qualification_manifest_sha256"] == (
        _QUALIFICATION_MANIFEST_SHA256
    )

    changed_score_payload = completed.score_evidence.to_dict()
    changed_score_payload["qualification_manifest_sha256"] = changed_digest
    unsigned_score = dict(changed_score_payload)
    del unsigned_score["payload_sha256"]
    changed_score_payload["payload_sha256"] = _canonical_sha(unsigned_score)
    changed_scores = evidence.parse_matched_score_evidence(changed_score_payload)
    with pytest.raises(seal.ForagerMatchedSealError, match="plan closure drifted"):
        seal._validate_execution_plan_snapshot(
            plan_payload,
            completed.protocol,
            changed_scores,
        )

    changed_plan = dict(plan_payload)
    changed_plan["qualification_manifest_sha256"] = changed_digest
    with pytest.raises(seal.ForagerMatchedSealError, match="plan closure drifted"):
        seal._validate_execution_plan_snapshot(
            changed_plan,
            completed.protocol,
            completed.score_evidence,
        )

    changed_request = completed.verification_request.to_dict()
    changed_request["qualification_manifest_sha256"] = changed_digest
    with pytest.raises(executor.ForagerMatchedExecutorError, match="subject"):
        executor.parse_verification_request(changed_request)

    changed_bindings = bindings.to_dict()
    changed_bindings["qualification_manifest_sha256"] = changed_digest
    with pytest.raises(seal.ForagerMatchedSealError, match="bindings cache is invalid"):
        seal._parse_recorded_bindings(changed_bindings)

    changed_completion = dict(completed.completion_summary)
    changed_completion["qualification_manifest_sha256"] = changed_digest
    with pytest.raises(seal.ForagerMatchedSealError, match="completion summary closure"):
        seal._validate_completion_summary(
            changed_completion,
            receipt_index,
            completed.score_evidence,
            completed.verification_request,
        )


@pytest.mark.parametrize("invalid_horizon", [499_712.0, True])
def test_execution_plan_rejects_non_integer_horizon(
    tmp_path: Path,
    invalid_horizon: object,
) -> None:
    completed, _ = _completed_campaign(tmp_path)
    plan_payload = seal._decode_canonical(
        completed.plan.canonical_bytes,
        "test execution plan",
    )
    plan_payload["horizon"] = invalid_horizon

    with pytest.raises(seal.ForagerMatchedSealError, match="horizon must be an integer"):
        seal._validate_execution_plan_snapshot(
            plan_payload,
            completed.protocol,
            completed.score_evidence,
        )


def test_literal_v1_qualification_carriers_and_seal_manifest_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    plan_payload = seal._decode_canonical(
        completed.plan.canonical_bytes,
        "test execution plan",
    )
    legacy_plan = dict(plan_payload)
    legacy_plan["schema_version"] = "alberta.forager_matched_execution_plan.v1"
    with pytest.raises(seal.ForagerMatchedSealError, match="plan schema/state"):
        seal._validate_execution_plan_snapshot(
            legacy_plan,
            completed.protocol,
            completed.score_evidence,
        )

    legacy_scores = completed.score_evidence.to_dict()
    legacy_scores["schema_version"] = "alberta.forager_matched_score_evidence.v1"
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="unsupported"):
        evidence.parse_matched_score_evidence(legacy_scores)

    legacy_request = completed.verification_request.to_dict()
    legacy_request["schema_version"] = "alberta.forager_matched_verification_request.v1"
    with pytest.raises(executor.ForagerMatchedExecutorError, match="schema/authentication"):
        executor.parse_verification_request(legacy_request)

    legacy_bindings = bindings.to_dict()
    legacy_bindings["schema_version"] = "alberta.forager_authenticated_evidence_bindings.v1"
    with pytest.raises(seal.ForagerMatchedSealError, match="schema/stage"):
        seal._parse_recorded_bindings(legacy_bindings)

    receipt_index = seal._decode_canonical(
        completed.execution_receipt_index.canonical_bytes,
        "test receipt index",
    )
    legacy_completion = dict(completed.completion_summary)
    legacy_completion["schema_version"] = (
        "alberta.forager_matched_open_tuning_completion.v1"
    )
    with pytest.raises(seal.ForagerMatchedSealError, match="completion summary schema/state"):
        seal._validate_completion_summary(
            legacy_completion,
            receipt_index,
            completed.score_evidence,
            completed.verification_request,
        )

    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    legacy_manifest = dict(content.manifest)
    legacy_manifest["schema_version"] = "alberta.forager_matched_seal_bundle.v1"
    with pytest.raises(seal.ForagerMatchedSealError, match="schema/authority"):
        seal._validate_manifest_shape(legacy_manifest)


def test_content_valid_cache_cannot_authenticate_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    content = seal.load_forager_matched_seal_bundle_content(output_root)
    changed = replace(bindings, verification_receipt_sha256=_sha("different-receipt"))

    with pytest.raises(
        seal.ForagerMatchedSealError,
        match="differs from the recorded bindings cache",
    ):
        seal.authenticate_forager_matched_seal_bundle(
            content,
            resolver=lambda _request: changed,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )


def test_no_authenticated_wrapper_is_exposed_and_authentication_returns_plain_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )

    assert not hasattr(seal, "ExternallyAuthenticatedSealBundle")
    assert seal.authenticate_forager_matched_seal_bundle(
        content,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    ) == bindings


def test_create_rejects_anchor_and_subject_pins_before_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    resolver_calls = 0

    def resolver(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal resolver_calls
        resolver_calls += 1
        return bindings

    with pytest.raises(seal.ForagerMatchedSealError, match="caller-pinned identity"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=resolver,
            expected_trust_anchor_identity="different-anchor",
        )
    with pytest.raises(seal.ForagerMatchedSealError, match="caller-pinned digest"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            tmp_path / "other-seal",
            resolver=resolver,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
            expected_verification_subject_sha256=_sha("different-subject"),
        )

    assert resolver_calls == 0
    assert not output_root.exists()


def test_authentication_rejects_all_pin_mismatches_before_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    resolver_calls = 0

    def resolver(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal resolver_calls
        resolver_calls += 1
        return bindings

    mismatches = (
        {
            "expected_trust_anchor_identity": "different-anchor",
        },
        {
            "expected_trust_anchor_identity": bindings.trust_anchor_identity,
            "expected_verification_subject_sha256": _sha("different-subject"),
        },
        {
            "expected_trust_anchor_identity": bindings.trust_anchor_identity,
            "expected_seal_manifest_sha256": _sha("different-manifest"),
        },
    )
    for pins in mismatches:
        with pytest.raises(seal.ForagerMatchedSealError, match="caller-pinned"):
            seal.authenticate_forager_matched_seal_bundle(
                content,
                resolver=resolver,
                **pins,
            )
    assert resolver_calls == 0

    assert seal.authenticate_forager_matched_seal_bundle(
        content,
        resolver=resolver,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
        expected_verification_subject_sha256=bindings.verification_subject_sha256,
        expected_seal_manifest_sha256=cast(str, content.manifest["payload_sha256"]),
    ) == bindings
    assert resolver_calls == 1


def test_create_manifest_pin_mismatch_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    resolver_calls = 0

    def resolver(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal resolver_calls
        resolver_calls += 1
        return bindings

    with pytest.raises(seal.ForagerMatchedSealError, match="caller-pinned digest"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=resolver,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
            expected_seal_manifest_sha256=_sha("different-manifest"),
        )

    assert resolver_calls == 1
    assert not output_root.exists()
    assert not tuple(tmp_path.glob(".seal-partial-*"))


def test_resolver_failure_creates_no_output_or_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, _bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root = tmp_path / "qualification"
    campaign_root = tmp_path / "open-campaign"
    qualification_root.mkdir()
    campaign_root.mkdir()
    output_root = tmp_path / "uncreated" / "seal"

    def reject(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        raise RuntimeError("external verifier rejected subject")

    with pytest.raises(seal.ForagerMatchedSealError, match="trust resolution"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=reject,
            expected_trust_anchor_identity=completed.verification_request.trust_anchor_identity,
        )
    assert not output_root.parent.exists()


def test_create_replays_then_fsyncs_then_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    events: list[str] = []
    real_verify = seal._verify_staged_bundle
    real_sync = seal._durably_sync_open_tree
    real_publish = seal._publish_verified_no_replace

    def verify(root: seal._OpenDirectory) -> seal.ContentVerifiedSealBundle:
        events.append("replay")
        return real_verify(root)

    def sync(root: seal._OpenDirectory) -> None:
        events.append("fsync")
        real_sync(root)

    def publish(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        events.append("publish")
        real_publish(parent, staging, source_name, destination_name, destination)

    monkeypatch.setattr(seal, "_verify_staged_bundle", verify)
    monkeypatch.setattr(seal, "_durably_sync_open_tree", sync)
    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish)
    seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )

    assert events == ["replay", "fsync", "publish"]


def test_fsync_failure_removes_staging_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    staged: list[Path] = []

    def fail(root: seal._OpenDirectory) -> None:
        staged.append(root.path)
        raise seal.ForagerMatchedSealError("injected fsync failure")

    def publish(*_args: Any) -> None:
        raise AssertionError("publication must not run after fsync failure")

    monkeypatch.setattr(seal, "_durably_sync_open_tree", fail)
    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish)
    with pytest.raises(seal.ForagerMatchedSealError, match="injected fsync failure"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )
    assert len(staged) == 1
    assert not staged[0].exists()
    assert not output_root.exists()


def test_existing_output_is_never_replaced_or_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    output_root.mkdir()
    marker = output_root / "owner"
    marker.write_text("existing", encoding="utf-8")
    called = False

    def resolver(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal called
        called = True
        return bindings

    with pytest.raises(seal.ForagerMatchedSealError, match="already exists"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=resolver,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )
    assert called is False
    assert marker.read_text(encoding="utf-8") == "existing"


def test_publish_loses_a_concurrent_create_without_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    real_publish = seal._publish_verified_no_replace

    def publish(
        parent: seal._OpenDirectory,
        staging: seal._OpenDirectory,
        source_name: str,
        destination_name: str,
        destination: Path,
    ) -> None:
        destination.mkdir()
        (destination / "owner").write_text("concurrent", encoding="utf-8")
        real_publish(parent, staging, source_name, destination_name, destination)

    monkeypatch.setattr(seal, "_publish_verified_no_replace", publish)
    with pytest.raises(seal.ForagerMatchedSealError, match="created concurrently"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert (output_root / "owner").read_text(encoding="utf-8") == "concurrent"
    assert not tuple(output_root.parent.glob(".seal-partial-*"))


def test_staging_name_substitution_is_rejected_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    real_sync = seal._durably_sync_open_tree
    moved_staging: list[Path] = []
    replacement_markers: list[Path] = []

    def sync_then_substitute(root: seal._OpenDirectory) -> None:
        real_sync(root)
        moved = root.path.with_name(f"{root.path.name}-moved")
        root.path.rename(moved)
        root.path.mkdir()
        marker = root.path / "replacement-owner"
        marker.write_text("must-survive", encoding="utf-8")
        moved_staging.append(moved)
        replacement_markers.append(marker)

    monkeypatch.setattr(seal, "_durably_sync_open_tree", sync_then_substitute)
    with pytest.raises(seal.ForagerMatchedSealError, match="verified staging inode"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert not output_root.exists()
    assert len(moved_staging) == 1
    assert (moved_staging[0] / "seal.json").is_file()
    assert replacement_markers[0].read_text(encoding="utf-8") == "must-survive"


def test_parent_inode_substitution_is_rejected_without_touching_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root = tmp_path / "qualification"
    campaign_root = tmp_path / "open-campaign"
    qualification_root.mkdir()
    campaign_root.mkdir()
    parent_path = tmp_path / "publication"
    output_root = parent_path / "seal"
    moved_parent = tmp_path / "publication-moved"
    real_sync = seal._durably_sync_open_tree

    def sync_then_substitute_parent(root: seal._OpenDirectory) -> None:
        real_sync(root)
        parent_path.rename(moved_parent)
        parent_path.mkdir()
        (parent_path / "replacement-owner").write_text(
            "must-survive",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        seal,
        "_durably_sync_open_tree",
        sync_then_substitute_parent,
    )
    with pytest.raises(seal.ForagerMatchedSealError, match="output parent"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert not output_root.exists()
    assert (parent_path / "replacement-owner").read_text(encoding="utf-8") == "must-survive"
    assert not tuple(moved_parent.glob(".seal-partial-*"))


def test_parent_fsync_failure_reports_published_destination_as_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)

    def fail_parent_sync(_parent: seal._OpenDirectory) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(seal, "_sync_publication_parent", fail_parent_sync)
    with pytest.raises(seal.PublishedSealUncertainError) as caught:
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert caught.value.destination == output_root
    assert output_root.is_dir()
    assert seal.load_forager_matched_seal_bundle_content(output_root).output_root == output_root


def test_post_publish_replay_failure_reports_destination_as_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    real_load = seal._load_forager_matched_seal_bundle_from_open_root
    load_calls = 0

    def fail_second_replay(
        root: seal._OpenDirectory,
        inventory: Any,
    ) -> seal.ContentVerifiedSealBundle:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 2:
            raise seal.ForagerMatchedSealError("injected final replay failure")
        return real_load(root, inventory)

    monkeypatch.setattr(
        seal,
        "_load_forager_matched_seal_bundle_from_open_root",
        fail_second_replay,
    )
    with pytest.raises(seal.PublishedSealUncertainError) as caught:
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=lambda _request: bindings,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert load_calls == 2
    assert caught.value.destination == output_root
    assert (output_root / "seal.json").is_file()


def test_loader_rejects_unknown_links_and_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )

    unexpected = output_root / "unexpected"
    unexpected.symlink_to("open-protocol.json")
    with pytest.raises(seal.ForagerMatchedSealError, match="link|inventory"):
        seal.load_forager_matched_seal_bundle_content(output_root)
    unexpected.unlink()

    result_path = output_root / "selection-result.json"
    result_path.chmod(0o600)
    result_path.write_bytes(result_path.read_bytes() + b" ")
    sidecar = output_root / "selection-result.json.sha256"
    sidecar.chmod(0o600)
    sidecar.write_text(hashlib.sha256(result_path.read_bytes()).hexdigest() + "\n")
    with pytest.raises(seal.ForagerMatchedSealError, match="file digest differs"):
        seal.load_forager_matched_seal_bundle_content(output_root)


def test_seal_closes_over_canonical_plan_and_live_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    content = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )

    artifacts = cast(dict[str, Any], dict(content.manifest["artifacts"]))
    assert artifacts["open_execution_plan"]["path"] == "open-execution-plan.json"
    assert artifacts["open_execution_plan"]["payload_sha256"] == completed.plan.plan_sha256
    assert artifacts["open_live_runtime"]["path"] == "open-live-runtime.json"
    assert (
        artifacts["open_live_runtime"]["payload_sha256"]
        == completed.live_runtime.identity_sha256
    )
    assert (output_root / "open-execution-plan.json").read_bytes() == (
        completed.plan.canonical_bytes
    )
    assert (output_root / "open-live-runtime.json").read_bytes() == (
        executor.canonical_json_bytes(completed.live_runtime.unsigned_dict)
    )
    receipt = seal._decode_canonical(
        (output_root / "open-execution-receipt-index.json").read_bytes(),
        "test receipt index",
    )
    assert receipt["plan_sha256"] == completed.plan.plan_sha256
    assert receipt["live_runtime_identity_sha256"] == completed.live_runtime.identity_sha256


@pytest.mark.parametrize(
    ("field", "malformed"),
    (("plan_sha256", False), ("live_runtime_identity_sha256", {"sha": "not-a-string"})),
)
def test_receipt_index_rejects_malformed_plan_and_live_runtime_sha_fields(
    tmp_path: Path,
    field: str,
    malformed: Any,
) -> None:
    completed, _bindings = _completed_campaign(tmp_path)
    receipt = seal._decode_canonical(
        completed.execution_receipt_index.canonical_bytes,
        "test receipt index",
    )
    receipt[field] = malformed

    with pytest.raises(seal.ForagerMatchedSealError, match="lowercase SHA-256"):
        seal._validate_receipt_index(
            receipt,
            completed.protocol,
            completed.score_evidence,
            expected_plan_sha256=completed.plan.plan_sha256,
            expected_live_runtime_identity_sha256=(
                completed.live_runtime.identity_sha256
            ),
        )


def test_canonical_json_replay_enforces_node_and_depth_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seal, "_MAX_JSON_NODES", 5)
    with pytest.raises(seal.ForagerMatchedSealError, match="node bound"):
        seal._decode_canonical(b'{"items":[1,2,3,4]}', "oversized JSON")

    monkeypatch.setattr(seal, "_MAX_JSON_NODES", 100)
    monkeypatch.setattr(seal, "_MAX_JSON_DEPTH", 1)
    with pytest.raises(seal.ForagerMatchedSealError, match="depth bound"):
        seal._decode_canonical(b'{"items":[1]}', "overdeep JSON")


def test_frozen_selection_replay_cap_rejects_before_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    assert (
        seal._MAX_MATCHED_CANDIDATES,
        seal._MAX_MATCHED_TUNING_SEEDS,
        seal._MAX_MATCHED_EVALUATION_SEEDS,
        seal._MAX_MATCHED_SELECTION_BOOTSTRAP_RESAMPLES,
        seal._MAX_MATCHED_SELECTION_GROUPS,
        seal._MAX_MATCHED_SELECTION_GROUP_SIZE,
    ) == (23, 10, 30, 10_000, 2, 14)
    completed.protocol = replace(
        completed.protocol,
        selection_plan=replace(
            completed.protocol.selection_plan,
            bootstrap_resamples=10_001,
        ),
    )
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    resolver_calls = 0

    def resolver(_request: executor.VerificationRequest) -> evidence.AuthenticatedEvidenceBindings:
        nonlocal resolver_calls
        resolver_calls += 1
        return bindings

    with pytest.raises(seal.ForagerMatchedSealError, match="resource envelope"):
        seal.create_forager_matched_seal_bundle(
            qualification_root,
            campaign_root,
            output_root,
            resolver=resolver,
            expected_trust_anchor_identity=bindings.trust_anchor_identity,
        )

    assert resolver_calls == 0
    assert not output_root.exists()


def test_loader_detects_inventory_mutation_after_all_artifact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    real_load_pair = seal._load_pair_at
    injected = False

    def load_pair(
        root: seal._OpenDirectory,
        name: str,
        label: str,
    ) -> tuple[bytes, str]:
        nonlocal injected
        result = real_load_pair(root, name, label)
        if name == "sealed-protocol.json" and not injected:
            descriptor = os.open(
                "unexpected",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root.descriptor,
            )
            os.close(descriptor)
            injected = True
        return result

    monkeypatch.setattr(seal, "_load_pair_at", load_pair)
    with pytest.raises(seal.ForagerMatchedSealError, match="inventory"):
        seal.load_forager_matched_seal_bundle_content(output_root)
    assert injected is True


def test_loader_holds_original_root_inode_and_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    moved_root = tmp_path / "seal-original-inode"
    real_load_pair = seal._load_pair_at
    swapped = False

    def load_pair(
        root: seal._OpenDirectory,
        name: str,
        label: str,
    ) -> tuple[bytes, str]:
        nonlocal swapped
        result = real_load_pair(root, name, label)
        if name == "seal.json" and not swapped:
            output_root.rename(moved_root)
            output_root.mkdir()
            swapped = True
        return result

    monkeypatch.setattr(seal, "_load_pair_at", load_pair)
    with pytest.raises(seal.ForagerMatchedSealError, match="opened inode"):
        seal.load_forager_matched_seal_bundle_content(output_root)

    assert swapped is True
    assert (moved_root / "seal.json").is_file()
    assert not (output_root / "seal.json").exists()


def test_fsync_walk_visits_all_files_before_root_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, bindings = _completed_campaign(tmp_path)
    _install_completed_loader(monkeypatch, completed)
    qualification_root, campaign_root, output_root = _roots(tmp_path)
    created = seal.create_forager_matched_seal_bundle(
        qualification_root,
        campaign_root,
        output_root,
        resolver=lambda _request: bindings,
        expected_trust_anchor_identity=bindings.trust_anchor_identity,
    )
    assert created.output_root == output_root
    events: list[Path] = []

    def record(descriptor: int) -> None:
        events.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))

    monkeypatch.setattr(os, "fsync", record)
    opened = seal._open_stable_directory(output_root, "test seal root")
    try:
        seal._durably_sync_open_tree(opened)
    finally:
        os.close(opened.descriptor)
    expected_files = {output_root / name for name in seal._expected_root_names()}
    assert set(events[:-1]) == expected_files
    assert events[-1] == output_root
