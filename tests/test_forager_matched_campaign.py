"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_campaign`.

The campaign engine executes the open-tuning block of the matched-current
pipeline, whose phases are: capability qualification -> open-tuning campaign
(21 candidates x 10 tuning seeds, this module) -> external authentication +
selection/seal -> sealed 30-seed evaluation -> final analysis.  The engine is
resumable and content-only: it cannot authenticate its own qualification,
select winners, or seal.  Tests cover the golden byte contract for every
persisted artifact, seed-major scheduling that excludes the two
descriptive-only arms, crash/resume repair paths, and fail-closed rejection
of tampered, symlinked, hard-linked, or concurrently-written state.

``_context`` assembles a minimal single-candidate campaign around the shared
executor fixtures, with a stubbed qualification bundle and a fake qualified
runtime — no OCI runtime or real candidate ever executes.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_campaign as campaign
from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol
from alberta_framework.benchmarks import forager_matched_qualification as qualification
from tests import test_forager_matched_executor as executor_fixtures
from tests import test_forager_matched_open_protocol as protocol_fixtures

pytestmark = pytest.mark.integration
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# A synthetic ``test...v1`` manifest suffices because the campaign binds the
# qualification manifest by SHA-256 only and never interprets or authenticates
# its contents — authentication belongs to the external trust resolver.  Any
# canonical bytes with a self-consistent digest triple exercise the binding.
_QUALIFICATION_MANIFEST = {"schema_version": "test.matched_current_qualification.v1"}
_QUALIFICATION_MANIFEST_BYTES = campaign.canonical_json_bytes(_QUALIFICATION_MANIFEST)
_QUALIFICATION_MANIFEST_SHA256 = hashlib.sha256(_QUALIFICATION_MANIFEST_BYTES).hexdigest()


def _context(tmp_path: Path) -> campaign._CampaignContext:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True)
    _payload, protocol, assets = executor_fixtures._fixture(fixture_root)
    candidate_id = next(iter(assets))
    plan = executor.build_execution_plan(
        protocol,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=(candidate_id,),
    )
    live = executor_fixtures._runtime(tmp_path / "runtime", plan)
    root = tmp_path / "campaign-root"
    root.mkdir()
    (root / "runs").mkdir()
    (root / "completions").mkdir()
    schedule: dict[str, Any] = {
        "cells": [
            {"ordinal": ordinal, "candidate_id": candidate_id, "seed": seed}
            for ordinal, seed in enumerate(plan.protocol.active_seeds)
        ]
    }
    schedule["schedule_sha256"] = hashlib.sha256(
        campaign.canonical_json_bytes(schedule)
    ).hexdigest()
    rebuilt = campaign._RebuiltInputs(
        bundle=cast(
            Any,
            SimpleNamespace(
                manifest=_QUALIFICATION_MANIFEST,
                manifest_bytes=_QUALIFICATION_MANIFEST_BYTES,
                manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
                cpu_qualification_root=tmp_path,
                rng_parity_qualification_root=tmp_path,
            ),
        ),
        protocol=plan.protocol,
        plan=plan,
        candidate_ids=(candidate_id,),
        assets={},
        schedule=schedule,
    )
    return campaign._CampaignContext(root, rebuilt, live)


def _runner(
    context: campaign._CampaignContext,
    calls: list[str],
    *,
    fail_score: bool = False,
) -> executor.ProcessRunner:
    seed = context.rebuilt.protocol.active_seeds[0]

    def run(command: list[str] | tuple[str, ...]) -> executor.ProcessResult:
        reinspection = executor_fixtures._runtime_reinspection_result(
            command,
            context.live_runtime,
        )
        if reinspection is not None:
            return reinspection
        if "score" in command:
            calls.append("score")
            if fail_score:
                return executor.ProcessResult(3, b"", b"")
            seed_argument = next(item for item in command if item.startswith("--seed="))
            active_seed = int(seed_argument.split("=")[1])
            return executor.ProcessResult(
                0,
                executor_fixtures._scoring_output(context.rebuilt.plan, active_seed),
                b"",
            )
        calls.append("candidate")
        return executor.ProcessResult(
            0,
            f"opaque-raw-{seed}-{len(calls)}".encode("ascii"),
            b"",
        )

    return cast(executor.ProcessRunner, run)


def _complete_loader_context(context: campaign._CampaignContext) -> None:
    calls: list[str] = []
    candidate_id = context.rebuilt.candidate_ids[0]
    for seed in context.rebuilt.protocol.active_seeds:
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls),
        )
    campaign._finalize_or_validate(
        context,
        campaign._scan_all_cells(context),
        create=True,
    )
    campaign._publish_json_pair(
        context.root / "campaign.json",
        {"schema_version": "test.completed_campaign_lock.v1"},
    )


def _use_loader_context(
    monkeypatch: pytest.MonkeyPatch,
    context: campaign._CampaignContext,
) -> None:
    monkeypatch.setattr(
        campaign,
        "_load_context",
        lambda *_args, **_kwargs: context,
    )


def _published_context_for_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[campaign._CampaignContext, Path]:
    context = _context(tmp_path / "fixture")
    rebuilt = context.rebuilt
    fixture_universe_sha = rebuilt.protocol.selection_plan.candidate_universe_sha256
    monkeypatch.setattr(
        universe,
        "MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256",
        fixture_universe_sha,
    )
    output_root = tmp_path / "published-campaign"
    campaign._publish_initial_root(output_root, rebuilt, context.live_runtime)
    monkeypatch.setattr(campaign, "_rebuild_inputs", lambda _root: rebuilt)
    monkeypatch.setattr(
        universe,
        "verify_matched_current_candidate_universe_sources",
        lambda _root: SimpleNamespace(candidate_universe_sha256=fixture_universe_sha),
    )
    monkeypatch.setattr(
        campaign,
        "_qualify_live",
        lambda *_args, **_kwargs: context.live_runtime,
    )
    monkeypatch.setattr(
        executor,
        "parse_execution_plan",
        lambda *_args, **_kwargs: rebuilt.plan,
    )
    return context, output_root


def test_selection_schedule_excludes_both_descriptive_candidates() -> None:
    protocol = protocol_fixtures._build()

    candidate_ids = campaign.selection_candidate_ids(protocol)
    schedule = campaign.build_seed_major_schedule(protocol)

    assert candidate_ids == (
        open_protocol.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS
        + open_protocol.MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS
    )
    assert len(candidate_ids) == 21
    assert set(candidate_ids).isdisjoint(open_protocol.MATCHED_CURRENT_DESCRIPTIVE_CANDIDATE_IDS)
    assert len(schedule["cells"]) == 210
    assert [cell["candidate_id"] for cell in schedule["cells"][:21]] == list(
        candidate_ids
    )
    assert {cell["seed"] for cell in schedule["cells"][:21]} == {
        open_protocol.MATCHED_CURRENT_TUNING_SEEDS[0]
    }
    assert schedule["schedule_sha256"] == hashlib.sha256(
        campaign.canonical_json_bytes(
            {key: value for key, value in schedule.items() if key != "schedule_sha256"}
        )
    ).hexdigest()


def test_rebuild_inputs_forwards_exact_qualification_manifest_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = protocol_fixtures._build()
    manifest = {
        "schema_version": "test.matched_current_qualification.v1",
        "label": "Alberta café",
    }
    manifest_bytes = qualification._canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    candidate_assets = {
        candidate_id: object()
        for candidate_id in open_protocol.MATCHED_CURRENT_CANDIDATE_IDS
    }
    bundle = SimpleNamespace(
        runtime_qualification=object(),
        candidate_qualifications={},
        candidate_assets=candidate_assets,
        cpu_qualification_root=tmp_path / "cpu",
        rng_parity_qualification_root=tmp_path / "rng-parity",
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )
    expected_candidate_ids = campaign.selection_candidate_ids(frozen)
    fake_plan = SimpleNamespace(
        candidates=tuple(
            SimpleNamespace(candidate=SimpleNamespace(candidate_id=candidate_id))
            for candidate_id in expected_candidate_ids
        )
    )
    plan_calls: list[dict[str, Any]] = []

    def build_plan(
        protocol: Any,
        assets: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        plan_calls.append({"protocol": protocol, "assets": assets, **kwargs})
        return fake_plan

    monkeypatch.setattr(
        campaign.qualification,
        "load_matched_current_qualification_bundle",
        lambda _root: bundle,
    )
    monkeypatch.setattr(
        open_protocol,
        "build_forager_matched_open_protocol",
        lambda **_kwargs: frozen,
    )
    monkeypatch.setattr(executor, "build_execution_plan", build_plan)

    rebuilt = campaign._rebuild_inputs(tmp_path / "qualification")

    assert rebuilt.bundle.manifest_bytes == manifest_bytes
    assert rebuilt.bundle.manifest_sha256 == manifest_sha256
    assert len(plan_calls) == 1
    assert plan_calls[0]["protocol"] is frozen
    assert tuple(plan_calls[0]["assets"]) == expected_candidate_ids
    assert plan_calls[0]["qualification_manifest_sha256"] == manifest_sha256
    assert plan_calls[0]["candidate_ids"] == expected_candidate_ids
    assert plan_calls[0]["cpu_qualification_root"] == tmp_path / "cpu"
    assert plan_calls[0]["rng_parity_qualification_root"] == tmp_path / "rng-parity"

    changed_bytes = manifest_bytes + b"\n"
    changed_bundle = SimpleNamespace(
        **{
            **vars(bundle),
            "manifest_bytes": changed_bytes,
            "manifest_sha256": hashlib.sha256(changed_bytes).hexdigest(),
        }
    )
    monkeypatch.setattr(
        campaign.qualification,
        "load_matched_current_qualification_bundle",
        lambda _root: changed_bundle,
    )
    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="exact bytes changed",
    ):
        campaign._rebuild_inputs(tmp_path / "qualification")
    assert len(plan_calls) == 1


@pytest.mark.parametrize("carrier", ("plan", "payload", "executor_manifest"))
def test_campaign_manifest_rejects_each_qualification_digest_carrier_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    carrier: str,
) -> None:
    context = _context(tmp_path)
    plan = context.rebuilt.plan
    wrong_digest = hashlib.sha256(f"wrong-{carrier}".encode("ascii")).hexdigest()
    qualification_digest = plan.qualification_manifest_sha256
    payload = dict(plan.payload)
    executor_manifest = dict(plan.executor_manifest)
    plan_digest = qualification_digest
    if carrier == "plan":
        plan_digest = wrong_digest
    elif carrier == "payload":
        payload["qualification_manifest_sha256"] = wrong_digest
    else:
        executor_manifest["qualification_manifest_sha256"] = wrong_digest
    changed_plan = SimpleNamespace(
        qualification_manifest_sha256=plan_digest,
        payload=payload,
        executor_manifest=executor_manifest,
        plan_sha256=plan.plan_sha256,
        source_manifest_sha256=plan.source_manifest_sha256,
        executor_manifest_sha256=plan.executor_manifest_sha256,
    )
    fixture_universe_sha = context.rebuilt.protocol.selection_plan.candidate_universe_sha256
    monkeypatch.setattr(
        universe,
        "MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256",
        fixture_universe_sha,
    )

    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="qualification manifest plan binding drifted",
    ):
        campaign._campaign_manifest(
            replace(context.rebuilt, plan=cast(Any, changed_plan)),
            context.live_runtime,
        )


@pytest.mark.parametrize("carrier", ("payload", "sidecar"))
def test_load_context_rejects_persisted_qualification_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    carrier: str,
) -> None:
    context, output_root = _published_context_for_replay(tmp_path, monkeypatch)
    manifest_path = output_root / "qualification-manifest.json"
    sidecar_path = output_root / "qualification-manifest.json.sha256"
    if carrier == "payload":
        changed = campaign.canonical_json_bytes(
            {
                **_QUALIFICATION_MANIFEST,
                "tampered": True,
            }
        )
        manifest_path.chmod(0o600)
        manifest_path.write_bytes(changed)
        manifest_path.chmod(0o400)
        sidecar_path.chmod(0o600)
        sidecar_path.write_bytes(
            f"{hashlib.sha256(changed).hexdigest()}\n".encode("ascii")
        )
        sidecar_path.chmod(0o400)
    else:
        sidecar_path.chmod(0o600)
        sidecar_path.write_bytes(f"{'0' * 64}\n".encode("ascii"))
        sidecar_path.chmod(0o400)

    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="persisted qualification manifest differs",
    ):
        campaign._load_context(
            tmp_path / "qualification",
            output_root,
            runtime="docker",
            runner=None,
        )
    assert context.rebuilt.bundle.manifest_sha256 == _QUALIFICATION_MANIFEST_SHA256


def test_load_context_rejects_literal_v1_open_campaign_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context_value, output_root = _published_context_for_replay(tmp_path, monkeypatch)
    path = output_root / "campaign.json"
    payload, _digest = campaign._load_json_pair(path, "campaign")
    payload["schema_version"] = "alberta.forager_matched_open_tuning_campaign.v1"
    raw = campaign.canonical_json_bytes(payload)
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.chmod(0o600)
    sidecar.write_bytes(f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"))
    sidecar.chmod(0o400)

    with pytest.raises(
        campaign.ForagerMatchedCampaignError,
        match="persisted campaign.json differs",
    ):
        campaign._load_context(
            tmp_path / "qualification",
            output_root,
            runtime="docker",
            runner=None,
        )


def test_open_campaign_v2_golden_byte_contract(tmp_path: Path) -> None:
    """Keep phase-neutral engine refactors byte-identical for open tuning."""

    def digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    context = _context(tmp_path / "cell")
    frozen_protocol = protocol_fixtures._build()
    frozen_schedule = campaign.build_seed_major_schedule(frozen_protocol)
    manifest_inputs = replace(
        context.rebuilt,
        protocol=frozen_protocol,
        candidate_ids=campaign.selection_candidate_ids(frozen_protocol),
        schedule=frozen_schedule,
    )
    immutable_bytes = {
        "open-protocol.json": frozen_protocol.canonical_bytes,
        "candidate-universe.json": campaign.canonical_json_bytes(
            universe.matched_current_candidate_universe_descriptor()
        ),
        "execution-plan.json": context.rebuilt.plan.canonical_bytes,
        "source-manifest.json": campaign.canonical_json_bytes(
            context.rebuilt.plan.source_manifest
        ),
        "executor-manifest.json": campaign.canonical_json_bytes(
            context.rebuilt.plan.executor_manifest
        ),
        "qualification-manifest.json": context.rebuilt.bundle.manifest_bytes,
        "execution-schedule.json": campaign.canonical_json_bytes(frozen_schedule),
        "live-runtime.json": campaign.canonical_json_bytes(
            context.live_runtime.unsigned_dict
        ),
        "campaign.json": campaign.canonical_json_bytes(
            campaign._campaign_manifest(manifest_inputs, context.live_runtime)
        ),
    }
    assert {name: digest(raw) for name, raw in immutable_bytes.items()} == {
        "open-protocol.json": "448f2eab78f6ffbb390a9bbe053a5ff379a3e91b77961d6e7acd96609cb668ca",
        "candidate-universe.json": (
            "6a9315cb996fe5698e4c1580d30da9b0524e9875ce085d1399bb975cc5b510a8"
        ),
        "execution-plan.json": (
            "da881ac9300f6d3bb2668855ab0ccd931fc3e162758867f387657cd07721abbb"
        ),
        "source-manifest.json": (
            "9aa2a792a6e48438ed91d3ac180e87348ce3f4781fd6a87a987634593b604f23"
        ),
        "executor-manifest.json": (
            "162c5037803e2bd25c2d63705a107fbdc67bfb8bb9acff3eeac31c6abba59df3"
        ),
        "qualification-manifest.json": (
            "0ac448b2686c7f7da8cc3f2a489bc764f76e215193de67cd62308ac21d83d24a"
        ),
        "execution-schedule.json": (
            "06cbe35d345bc3e4fdc79d7628a181a0ac126f929e0c0b70c2b25d9ef250d7d5"
        ),
        "live-runtime.json": (
            "d3ea32b2091b84146ff276b34d72b00e9fa2084fe00a50810277f69a35e35be1"
        ),
        "campaign.json": (
            "9cb3fca20ffd4dbb1950da8da88fe59cf6ebf3aaabe9b4c1cc15c116e912a3dc"
        ),
    }
    assert frozen_schedule["schedule_sha256"] == (
        "0c5aa63a5a8e7c1482bf26f165b3c5277075360c9ff96c58bc82bd93463992f4"
    )

    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    run_cell, _completion = campaign._cell_paths(context.root, candidate_id, seed)
    attempt = run_cell / "attempt-000001"
    attempt.mkdir(parents=True)
    (attempt / "raw-output.tar").write_bytes(b"golden-opaque-raw")
    binding, binding_sha256 = campaign._persist_raw_binding(
        context,
        candidate_id,
        seed,
        attempt,
    )
    assert binding_sha256 == (
        "890d4c8b8c6d21803c1aa7efd78c27286aaab9ccf1976acbece4309eb0e9e0ca"
    )
    artifact = executor.score_seed_archive(
        context.rebuilt.plan,
        candidate_id,
        seed,
        attempt / "raw-output.tar",
        context.live_runtime,
        expected_raw_archive_sha256=cast(
            str, cast(dict[str, Any], binding["raw_archive"])["sha256"]
        ),
        expected_raw_archive_size=cast(
            int, cast(dict[str, Any], binding["raw_archive"])["size_bytes"]
        ),
        runner=_runner(context, []),
    )
    bundle_sha256 = campaign._publish_json_pair(
        attempt / "bundle.json",
        artifact.to_dict(),
    )
    assert bundle_sha256 == (
        "1a1130d342bf689e8fed389e45bb26205bf246f49b920691d8998c6e2c10f679"
    )
    pointer = campaign._completion_pointer(
        context,
        candidate_id,
        seed,
        attempt,
        binding_sha256,
        bundle_sha256,
    )
    assert digest(campaign.canonical_json_bytes(pointer)) == (
        "a9fa7c0057215eb0689587973b2dd8fa560ac64cae6d0e3326c07488df540388"
    )
    campaign._persist_failure(
        attempt,
        candidate_id,
        seed,
        phase="scorer_recovery",
        error=ValueError("golden failure"),
        raw_binding_present=True,
    )
    assert digest((attempt / "failures" / "failure-000001.json").read_bytes()) == (
        "be9b503d81d1f2ff568376c0cb5baca7e9160247328c4b41e9218a2dedf63f10"
    )

    completed_context = _context(tmp_path / "completed")
    _complete_loader_context(completed_context)
    expected_final = {
        "execution-receipt-index.json": (
            "a8dcafdb050d2c0f32d2ef6743750d187d0cdec5cac6783315dd2342796a7691"
        ),
        "score-evidence.json": (
            "6fdfe584ac705b62da1042ba35d035c29e1fd574a93381a5b536d91d78df2266"
        ),
        "verification-request.json": (
            "91567672900044aa8d7a5f10983e3dac423d11985993dc6f27672bfa3da0c591"
        ),
        "completion-summary.json": (
            "9553ec581ca9660d5975b38e227dd30116a15ff7a63d92f5e1d57a27e3f9f9eb"
        ),
    }
    for name, expected_digest in expected_final.items():
        assert digest((completed_context.root / name).read_bytes()) == expected_digest
        assert (completed_context.root / f"{name}.sha256").read_bytes() == (
            f"{expected_digest}\n".encode("ascii")
        )

    status = campaign._derive_status(
        completed_context,
        campaign._scan_all_cells(completed_context),
    )
    normalized_status = replace(status, output_root=Path("/golden/open-campaign"))
    assert digest(campaign.canonical_json_bytes(normalized_status.to_dict())) == (
        "817f793086abb911dd7b21bd46114aa9fbab21f72365030ad92428007146e970"
    )


def test_cell_bindings_derive_stage_from_context_protocol(tmp_path: Path) -> None:
    context = _context(tmp_path)
    protocol_stub = SimpleNamespace(
        stage="sealed_evaluation",
        protocol_sha256=context.rebuilt.protocol.protocol_sha256,
        active_seeds=context.rebuilt.protocol.active_seeds,
    )
    sealed_context = replace(
        context,
        rebuilt=replace(context.rebuilt, protocol=cast(Any, protocol_stub)),
    )
    candidate_id = sealed_context.rebuilt.candidate_ids[0]
    seed = sealed_context.rebuilt.protocol.active_seeds[0]
    binding = campaign._raw_binding(
        sealed_context,
        candidate_id,
        seed,
        "attempt-000001",
        "1" * 64,
        123,
    )
    pointer = campaign._completion_pointer(
        sealed_context,
        candidate_id,
        seed,
        Path("attempt-000001"),
        "2" * 64,
        "3" * 64,
    )

    assert binding["stage"] == "sealed_evaluation"
    assert pointer["stage"] == "sealed_evaluation"


def test_custom_completion_summary_builder_flows_through_resume_and_replay(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    builder_calls: list[str] = []

    def build_summary(
        current: campaign._CampaignContext,
        receipt_index: executor.MatchedExecutionReceiptIndex,
        score_evidence: Any,
        request: executor.VerificationRequest,
    ) -> dict[str, Any]:
        builder_calls.append("summary")
        value = campaign._completion_summary(
            current,
            receipt_index,
            score_evidence,
            request,
        )
        return {
            **value,
            "schema_version": "test.injected_completion_summary.v1",
            "campaign_kind": "sealed_evaluation_test",
        }

    def validate_summary(
        current: campaign._CampaignContext,
        receipt_index: executor.MatchedExecutionReceiptIndex,
        score_evidence: Any,
        request: executor.VerificationRequest,
        summary: Any,
    ) -> None:
        campaign._validate_completion_summary_common(
            current,
            receipt_index,
            score_evidence,
            request,
            summary,
        )
        assert summary["schema_version"] == "test.injected_completion_summary.v1"
        assert summary["campaign_kind"] == "sealed_evaluation_test"

    status = campaign._run_resumable_context_locked(
        context,
        runner=_runner(context, calls),
        max_cells=None,
        completion_summary_builder=build_summary,
        completion_summary_validator=validate_summary,
    )

    assert status.state == "complete_content_only_external_verification_unresolved"
    persisted, _digest = campaign._load_json_pair(
        context.root / "completion-summary.json",
        "completion summary",
    )
    assert persisted["schema_version"] == "test.injected_completion_summary.v1"
    assert persisted["campaign_kind"] == "sealed_evaluation_test"
    scans = campaign._scan_all_cells(context)
    bundle = campaign._build_completed_campaign_bundle(
        context,
        scans,
        create=False,
        completion_summary_builder=build_summary,
        completion_summary_validator=validate_summary,
    )
    assert bundle.completion_summary["campaign_kind"] == "sealed_evaluation_test"
    assert len(builder_calls) >= 3
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="differs from rebuilt"):
        campaign._derive_status(context, scans)


def test_completed_campaign_loader_returns_exact_immutable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _complete_loader_context(context)
    _use_loader_context(monkeypatch, context)

    bundle = campaign.load_completed_open_tuning_campaign(
        tmp_path / "qualification",
        context.root,
        runner=cast(Any, lambda _command: None),
    )

    assert bundle.output_root == context.root
    assert bundle.protocol == context.rebuilt.protocol
    assert bundle.plan == context.rebuilt.plan
    assert bundle.live_runtime == context.live_runtime
    assert bundle.score_evidence.payload_sha256 == bundle.completion_summary[
        "score_evidence_sha256"
    ]
    assert bundle.verification_request.verification_subject_sha256 == (
        bundle.completion_summary["verification_subject_sha256"]
    )
    assert bundle.completion_summary["verification_authentication_state"] == (
        "unresolved_external_verifier_required"
    )
    assert bundle.completion_summary["promotion_authorized"] is False
    with pytest.raises(TypeError):
        cast(dict[str, Any], bundle.schedule)["mutated"] = True
    with pytest.raises(TypeError):
        cast(dict[str, Any], bundle.seed_artifacts)["mutated"] = ()
    with pytest.raises(TypeError):
        cast(dict[str, Any], bundle.completion_summary)["mutated"] = True
    with pytest.raises(TypeError):
        cast(dict[str, str], bundle.final_file_sha256)["mutated"] = "0" * 64


def test_completed_campaign_loader_rejects_incomplete_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    campaign._publish_json_pair(
        context.root / "campaign.json",
        {"schema_version": "test.incomplete_campaign_lock.v1"},
    )
    _use_loader_context(monkeypatch, context)

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="not complete"):
        campaign.load_completed_open_tuning_campaign(
            tmp_path / "qualification",
            context.root,
            runner=cast(Any, lambda _command: None),
        )


@pytest.mark.parametrize("artifact_name", campaign._FINAL_ARTIFACTS)
def test_completed_campaign_loader_rejects_each_self_consistent_final_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    context = _context(tmp_path)
    _complete_loader_context(context)
    _use_loader_context(monkeypatch, context)
    path = context.root / artifact_name
    payload, _digest = campaign._load_json_pair(path, artifact_name)
    payload["tampered"] = True
    raw = campaign.canonical_json_bytes(payload)
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.chmod(0o600)
    sidecar.write_bytes(f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"))
    sidecar.chmod(0o400)

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="differs from rebuilt"):
        campaign.load_completed_open_tuning_campaign(
            tmp_path / "qualification",
            context.root,
            runner=cast(Any, lambda _command: None),
        )


def test_completed_campaign_loader_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _complete_loader_context(context)
    _use_loader_context(monkeypatch, context)

    def snapshot() -> dict[str, tuple[bytes, int, int, int]]:
        return {
            path.relative_to(context.root).as_posix(): (
                path.read_bytes(),
                path.stat().st_mode,
                path.stat().st_mtime_ns,
                path.stat().st_ino,
            )
            for path in sorted(context.root.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    campaign.load_completed_open_tuning_campaign(
        tmp_path / "qualification",
        context.root,
        runner=cast(Any, lambda _command: None),
    )
    assert snapshot() == before


def test_completed_campaign_bundle_preserves_plan_candidate_seed_and_final_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _complete_loader_context(context)
    _use_loader_context(monkeypatch, context)

    bundle = campaign.load_completed_open_tuning_campaign(
        tmp_path / "qualification",
        context.root,
        runner=cast(Any, lambda _command: None),
    )

    expected_candidates = tuple(
        candidate.candidate.candidate_id for candidate in context.rebuilt.plan.candidates
    )
    assert bundle.candidate_ids == expected_candidates
    assert tuple(bundle.seed_artifacts) == expected_candidates
    assert tuple(
        record.seed
        for candidate_id in expected_candidates
        for record in bundle.seed_artifacts[candidate_id]
    ) == tuple(
        seed
        for _candidate_id in expected_candidates
        for seed in context.rebuilt.protocol.active_seeds
    )
    assert bundle.execution_receipt_index.candidate_order == expected_candidates
    assert tuple(item.candidate_id for item in bundle.score_evidence.candidate_scores) == (
        expected_candidates
    )
    assert tuple(bundle.final_file_sha256) == campaign._FINAL_ARTIFACTS


def test_prepare_rejects_an_existing_output_root(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="already exists"):
        campaign.prepare_open_tuning_campaign(
            tmp_path / "qualification",
            root,
        )


def test_scorer_failure_resumes_bound_raw_without_rerunning_candidate(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    initial = campaign._scan_cell(context, candidate_id, seed)
    first_calls: list[str] = []

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="execution failed"):
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            initial,
            _runner(context, first_calls, fail_score=True),
        )

    failed = campaign._scan_cell(context, candidate_id, seed)
    assert failed.resumable_attempt is not None
    assert campaign._derive_status(
        context,
        campaign._scan_all_cells(context),
    ).state == "recovery_required"
    assert first_calls == ["candidate", "score"]
    resumed_calls: list[str] = []
    campaign._run_one_cell(
        context,
        candidate_id,
        seed,
        failed,
        _runner(context, resumed_calls),
    )
    completed = campaign._scan_cell(context, candidate_id, seed)

    assert resumed_calls == ["score"]
    assert completed.artifact is not None
    assert completed.pointer_present is True
    assert (failed.resumable_attempt / "failures" / "failure-000001.json").is_file()


def test_raw_only_orphan_starts_a_new_attempt_and_is_never_resumed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    run_cell, _completion = campaign._cell_paths(context.root, candidate_id, seed)
    run_cell.mkdir(parents=True)
    orphan = run_cell / "attempt-000001"
    orphan.mkdir()
    orphan_raw = orphan / "raw-output.tar"
    orphan_raw.write_bytes(b"unbound-opaque-orphan")
    scan = campaign._scan_cell(context, candidate_id, seed)

    assert scan.resumable_attempt is None
    assert scan.next_attempt_number == 2
    calls: list[str] = []
    campaign._run_one_cell(
        context,
        candidate_id,
        seed,
        scan,
        _runner(context, calls),
    )

    assert calls == ["candidate", "score"]
    assert orphan_raw.read_bytes() == b"unbound-opaque-orphan"
    assert (run_cell / "attempt-000002" / "bundle.json").is_file()


def test_bound_raw_corruption_fails_before_any_oci_call(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    calls: list[str] = []
    with pytest.raises(campaign.ForagerMatchedCampaignError):
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls, fail_score=True),
        )
    failed = campaign._scan_cell(context, candidate_id, seed)
    assert failed.resumable_attempt is not None
    raw = failed.resumable_attempt / "raw-output.tar"
    raw.chmod(0o600)
    raw.write_bytes(b"replaced-tampered-opaque-bytes")
    oci_called = False

    def forbidden(_command: list[str]) -> executor.ProcessResult:
        nonlocal oci_called
        oci_called = True
        raise AssertionError("OCI must not be reached")

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="raw archive binding"):
        # Validation happens before the scorer-only runner can be selected.
        campaign._scan_cell(context, candidate_id, seed)
    assert oci_called is False


def test_raw_binding_rejects_equal_valued_float_size_alias(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    run_cell, _completion = campaign._cell_paths(context.root, candidate_id, seed)
    attempt = run_cell / "attempt-000001"
    attempt.mkdir(parents=True)
    raw_path = attempt / "raw-output.tar"
    raw_path.write_bytes(b"opaque-raw-content")
    raw_sha256, raw_size = campaign._stable_file_hash(
        raw_path,
        "test opaque raw archive",
        maximum=1024,
    )
    binding = campaign._raw_binding(
        context,
        candidate_id,
        seed,
        attempt.name,
        raw_sha256,
        raw_size,
    )
    cast(dict[str, Any], binding["raw_archive"])["size_bytes"] = float(raw_size)
    binding_bytes = campaign.canonical_json_bytes(binding)
    (attempt / "raw-binding.json").write_bytes(binding_bytes)
    (attempt / "raw-binding.json.sha256").write_text(
        hashlib.sha256(binding_bytes).hexdigest() + "\n"
    )

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="raw archive binding"):
        campaign._validate_raw_binding(context, candidate_id, seed, attempt)


def test_completed_bundle_without_pointer_repairs_only_the_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    calls: list[str] = []
    original = campaign._publish_completion_pointer

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise campaign.ForagerMatchedCampaignError("simulated pointer interruption")

    monkeypatch.setattr(campaign, "_publish_completion_pointer", interrupted)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="pointer interruption"):
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls),
        )
    recoverable = campaign._scan_cell(context, candidate_id, seed)
    assert recoverable.artifact is not None
    assert recoverable.pointer_present is False

    monkeypatch.setattr(campaign, "_publish_completion_pointer", original)
    campaign._run_one_cell(
        context,
        candidate_id,
        seed,
        recoverable,
        _runner(context, calls),
    )
    assert calls == ["candidate", "score"]
    assert campaign._scan_cell(context, candidate_id, seed).pointer_present is True


def test_duplicate_completed_and_resumable_attempts_fail_closed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    calls: list[str] = []
    campaign._run_one_cell(
        context,
        candidate_id,
        seed,
        campaign._scan_cell(context, candidate_id, seed),
        _runner(context, calls),
    )
    run_cell, _completion = campaign._cell_paths(context.root, candidate_id, seed)
    duplicate = run_cell / "attempt-000002"
    duplicate.mkdir()
    (duplicate / "raw-output.tar").write_bytes(b"second-opaque-output")
    campaign._persist_raw_binding(context, candidate_id, seed, duplicate)

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="ambiguous"):
        campaign._scan_cell(context, candidate_id, seed)


def test_unknown_symlink_and_hardlinked_dynamic_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    unknown_context = _context(tmp_path / "unknown")
    candidate_id = unknown_context.rebuilt.candidate_ids[0]
    seed = unknown_context.rebuilt.protocol.active_seeds[0]
    run_cell, _completion = campaign._cell_paths(
        unknown_context.root,
        candidate_id,
        seed,
    )
    attempt = run_cell / "attempt-000001"
    attempt.mkdir(parents=True)
    (attempt / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="unknown"):
        campaign._scan_cell(unknown_context, candidate_id, seed)

    symlink_context = _context(tmp_path / "symlink")
    os.symlink(
        tmp_path,
        symlink_context.root / "runs" / symlink_context.rebuilt.candidate_ids[0],
    )
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="regular directory"):
        campaign._validate_dynamic_roots(symlink_context)

    hardlink_context = _context(tmp_path / "hardlink")
    hard_candidate = hardlink_context.rebuilt.candidate_ids[0]
    hard_seed = hardlink_context.rebuilt.protocol.active_seeds[0]
    hard_calls: list[str] = []
    with pytest.raises(campaign.ForagerMatchedCampaignError):
        campaign._run_one_cell(
            hardlink_context,
            hard_candidate,
            hard_seed,
            campaign._scan_cell(hardlink_context, hard_candidate, hard_seed),
            _runner(hardlink_context, hard_calls, fail_score=True),
        )
    resumable = campaign._scan_cell(hardlink_context, hard_candidate, hard_seed)
    assert resumable.resumable_attempt is not None
    os.link(
        resumable.resumable_attempt / "raw-output.tar",
        hardlink_context.root / "hardlink-copy.tar",
    )
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="single-link"):
        campaign._scan_cell(hardlink_context, hard_candidate, hard_seed)


def test_final_artifacts_are_forbidden_early_and_exact_after_complete_block(
    tmp_path: Path,
) -> None:
    early = _context(tmp_path / "early")
    (early.root / "score-evidence.json.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="before exact block"):
        campaign._derive_status(early, campaign._scan_all_cells(early))

    context = _context(tmp_path / "complete")
    calls: list[str] = []
    for seed in context.rebuilt.protocol.active_seeds:
        candidate_id = context.rebuilt.candidate_ids[0]
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls),
        )
    scans = campaign._scan_all_cells(context)
    score_sha, subject_sha = campaign._finalize_or_validate(context, scans, create=True)
    status = campaign._derive_status(context, scans)

    assert status.state == "complete_content_only_external_verification_unresolved"
    assert status.score_evidence_sha256 == score_sha
    assert status.verification_subject_sha256 == subject_sha
    summary, _digest = campaign._load_json_pair(
        context.root / "completion-summary.json",
        "completion summary",
    )
    assert summary["selection_created"] is False
    assert summary["sealed_protocol_created"] is False
    assert summary["evaluation_artifacts_created"] is False
    assert summary["promotion_authorized"] is False
    artifact_blocks = campaign._ordered_artifacts(context, scans)
    receipt_index = executor.load_execution_receipt_index(
        context.root / "execution-receipt-index.json",
        plan=context.rebuilt.plan,
        artifacts=artifact_blocks,
        expected_payload_sha256=summary[
            "execution_receipt_index_payload_sha256"
        ],
    )
    score_payload, _score_file_sha = campaign._load_json_pair(
        context.root / "score-evidence.json",
        "score evidence",
    )
    assert [
        item.execution_receipt_sha256 for item in receipt_index.execution_receipts
    ] == [
        item["execution_receipt_sha256"] for item in score_payload["candidate_scores"]
    ]
    campaign._finalize_or_validate(context, scans, create=False)


def test_self_consistent_execution_receipt_index_tamper_fails_replay(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    candidate_id = context.rebuilt.candidate_ids[0]
    for seed in context.rebuilt.protocol.active_seeds:
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls),
        )
    scans = campaign._scan_all_cells(context)
    campaign._finalize_or_validate(context, scans, create=True)
    index_path = context.root / "execution-receipt-index.json"
    payload, _file_sha = campaign._load_json_pair(index_path, "execution receipt index")
    indexed = payload["execution_receipts"][0]
    indexed["receipt_payload"]["seed_artifacts"][0]["raw_artifact_sha256"] = "f" * 64
    indexed["execution_receipt_sha256"] = hashlib.sha256(
        executor.canonical_json_bytes(indexed["receipt_payload"])
    ).hexdigest()
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(
        executor.canonical_json_bytes(unsigned)
    ).hexdigest()
    raw = campaign.canonical_json_bytes(payload)
    index_path.chmod(0o600)
    index_path.write_bytes(raw)
    sidecar = index_path.with_name(index_path.name + ".sha256")
    sidecar.chmod(0o600)
    sidecar.write_bytes(f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii"))

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="differs from rebuilt"):
        campaign._finalize_or_validate(context, scans, create=False)


def test_run_mode_repairs_payload_first_sidecar_interruption(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_id = context.rebuilt.candidate_ids[0]
    seed = context.rebuilt.protocol.active_seeds[0]
    calls: list[str] = []
    with pytest.raises(campaign.ForagerMatchedCampaignError):
        campaign._run_one_cell(
            context,
            candidate_id,
            seed,
            campaign._scan_cell(context, candidate_id, seed),
            _runner(context, calls, fail_score=True),
        )
    failed = campaign._scan_cell(context, candidate_id, seed)
    assert failed.resumable_attempt is not None
    sidecar = failed.resumable_attempt / "raw-binding.json.sha256"
    sidecar.unlink()

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="incomplete canonical pair"):
        campaign._scan_cell(context, candidate_id, seed)
    repaired = campaign._scan_cell(context, candidate_id, seed, repair_pairs=True)

    assert repaired.resumable_attempt == failed.resumable_attempt
    assert sidecar.is_file()


def test_persisted_live_runtime_identity_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    rebuilt = context.rebuilt
    # The lightweight executor fixture uses a synthetic universe identity; bind the module's
    # frozen identity to it only inside this persistence-boundary test.
    fixture_universe_sha = rebuilt.protocol.selection_plan.candidate_universe_sha256
    monkeypatch.setattr(
        universe,
        "MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256",
        fixture_universe_sha,
    )
    output_root = tmp_path / "persisted-runtime"
    campaign._publish_initial_root(output_root, rebuilt, context.live_runtime)
    assert (output_root / "qualification-manifest.json").read_bytes() == (
        context.rebuilt.bundle.manifest_bytes
    )
    assert (output_root / "qualification-manifest.json.sha256").read_bytes() == (
        f"{context.rebuilt.bundle.manifest_sha256}\n".encode("ascii")
    )
    drifted = replace(
        context.live_runtime,
        version={"Client": {"Version": "runtime-drift"}},
    )
    monkeypatch.setattr(campaign, "_rebuild_inputs", lambda _root: rebuilt)
    monkeypatch.setattr(
        universe,
        "verify_matched_current_candidate_universe_sources",
        lambda _root: SimpleNamespace(candidate_universe_sha256=fixture_universe_sha),
    )
    monkeypatch.setattr(campaign, "_qualify_live", lambda *_args: drifted)
    monkeypatch.setattr(
        executor,
        "parse_execution_plan",
        lambda *_args, **_kwargs: rebuilt.plan,
    )

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="runtime identity drifted"):
        campaign._load_context(
            tmp_path / "qualification",
            output_root,
            runtime="docker",
            runner=cast(Any, lambda _command: None),
        )


def test_prepare_rejects_output_nested_under_qualification_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    output_root = qualification_root / "new-parent" / "campaign"
    rebuild_called = False

    def forbidden_rebuild(_root: Path) -> Any:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("qualification must not be loaded after overlap is known")

    monkeypatch.setattr(campaign, "_rebuild_inputs", forbidden_rebuild)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="overlap"):
        campaign.prepare_open_tuning_campaign(qualification_root, output_root)

    assert rebuild_called is False
    assert not output_root.parent.exists()


def test_prepare_rejects_parent_symlink_redirection_before_qualification_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    (qualification_root / "nested-parent").mkdir()
    safe_parent = tmp_path / "safe-parent"
    (safe_parent / "nested-parent").mkdir(parents=True)
    routed_parent = tmp_path / "routed-parent"
    routed_parent.symlink_to(safe_parent, target_is_directory=True)
    output_root = routed_parent / "nested-parent" / "campaign"
    rebuild_called = False

    original = campaign._canonical_campaign_destination

    def redirect_then_resolve(
        requested: Path,
        prospective: Path,
        qualified: Path,
    ) -> Path:
        routed_parent.unlink()
        routed_parent.symlink_to(qualification_root, target_is_directory=True)
        return original(requested, prospective, qualified)

    def forbidden_rebuild(_root: Path) -> Any:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("redirected destination must fail before qualification rebuild")

    monkeypatch.setattr(
        campaign,
        "_canonical_campaign_destination",
        redirect_then_resolve,
    )
    monkeypatch.setattr(campaign, "_rebuild_inputs", forbidden_rebuild)

    with pytest.raises(campaign.ForagerMatchedCampaignError, match="redirected"):
        campaign.prepare_open_tuning_campaign(qualification_root, output_root)

    assert rebuild_called is False
    assert not (qualification_root / "nested-parent" / "campaign").exists()
    assert not (safe_parent / "nested-parent" / "campaign").exists()


def test_anonymous_publication_failure_leaves_no_visible_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise campaign.ForagerMatchedCampaignError("simulated link interruption")

    monkeypatch.setattr(campaign, "_link_anonymous_no_replace", interrupted)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="link interruption"):
        campaign._publish_bytes(tmp_path / "artifact.json", b"{}")

    assert list(tmp_path.iterdir()) == []


def test_campaign_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    root = tmp_path / "locked-campaign"
    root.mkdir()
    campaign._publish_bytes(root / "campaign.json", b"{}")

    with campaign._campaign_lock(root, exclusive=True):
        with pytest.raises(campaign.ForagerMatchedCampaignError, match="already locked"):
            with campaign._campaign_lock(root, exclusive=True):
                raise AssertionError("unreachable")


def test_attempt_count_and_retained_raw_bytes_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts_context = _context(tmp_path / "attempts")
    candidate_id = attempts_context.rebuilt.candidate_ids[0]
    seed = attempts_context.rebuilt.protocol.active_seeds[0]
    run_cell, _completion = campaign._cell_paths(attempts_context.root, candidate_id, seed)
    run_cell.mkdir(parents=True)
    monkeypatch.setattr(campaign, "_MAX_ATTEMPTS_PER_CELL", 2)
    for number in range(1, 4):
        (run_cell / f"attempt-{number:06d}").mkdir()
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="too many entries"):
        campaign._scan_cell(attempts_context, candidate_id, seed)

    raw_context = _context(tmp_path / "raw")
    raw_candidate = raw_context.rebuilt.candidate_ids[0]
    raw_seed = raw_context.rebuilt.protocol.active_seeds[0]
    raw_cell, _completion = campaign._cell_paths(raw_context.root, raw_candidate, raw_seed)
    orphan = raw_cell / "attempt-000001"
    orphan.mkdir(parents=True)
    (orphan / "raw-output.tar").write_bytes(b"five!")
    monkeypatch.setattr(campaign, "_MAX_RETAINED_RAW_BYTES_PER_CELL", 4)
    with pytest.raises(campaign.ForagerMatchedCampaignError, match="retained raw-byte bound"):
        campaign._scan_cell(raw_context, raw_candidate, raw_seed)


def test_cli_writes_canonical_status_to_stdout_and_main_maps_validation_to_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = campaign.CampaignStatus(
        output_root=tmp_path,
        state="in_progress",
        completed_cells=0,
        total_cells=210,
        next_candidate_id="causal_e025_q050",
        next_seed=2_300_001,
        protocol_sha256="1" * 64,
        qualification_manifest_sha256="4" * 64,
        plan_sha256="2" * 64,
        live_runtime_identity_sha256="3" * 64,
        score_evidence_sha256=None,
        verification_subject_sha256=None,
    )
    monkeypatch.setattr(campaign, "campaign_status", lambda *_args, **_kwargs: status)
    assert campaign._cli(
        [
            "status",
            "--qualification-root",
            str(tmp_path / "qualification"),
            "--output-root",
            str(tmp_path / "campaign"),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out.encode("ascii") == campaign.canonical_json_bytes(status.to_dict())
    assert captured.err == ""

    monkeypatch.setattr(
        campaign,
        "_cli",
        lambda _argv: (_ for _ in ()).throw(
            campaign.ForagerMatchedCampaignError("expected validation failure")
        ),
    )
    assert campaign.main([]) == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_campaign_and_qualification_console_scripts_are_registered() -> None:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["alberta-forager-matched-campaign"] == (
        "alberta_framework.benchmarks.forager_matched_campaign:main"
    )
    assert scripts["alberta-forager-matched-qualification"] == (
        "alberta_framework.benchmarks.forager_matched_qualification:main"
    )
