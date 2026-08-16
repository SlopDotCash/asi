"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_executor`.

The executor is the strict CPU boundary between a parsed matched-current
protocol and live OCI execution: plan construction is nonexecuting and
content-addressed, live runs go through small injected-runner primitives, and
the host hashes the opaque result archive without ever opening reward bytes
(only the frozen scorer inside the qualified image does).  The suite is
adversarial throughout: manifest/receipt/digest drift, rename swaps, raw
archive mutation mid-scoring, symlink and FIFO source trees, hostile USTAR
members, and self-attested receipts must all fail closed, while the
plan -> execute -> score -> receipt-index path replays deterministically.

No real container runtime is used: ``_runtime`` fabricates a qualified
runtime identity from stub binaries and canned inspection payloads.  The
helpers here (``_fixture``, ``_plan``, ``_runtime``,
``_runtime_reinspection_result``, ``_scoring_output``) are imported by the
campaign, sealed-evaluation-campaign, and final-analysis suites as
``executor_fixtures``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import tarfile
import tempfile
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn, cast

import pytest

from alberta_framework.benchmarks import _forager_matched_container as container_helper
from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_rng_parity as parity
from tests import test_forager_matched_open_protocol as open_protocol_fixtures
from tests import test_forager_matched_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_QUALIFICATION_MANIFEST_SHA256 = _sha("matched-current-qualification-manifest")


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(executor.canonical_json_bytes(value)).hexdigest()


def _receipt(
    candidate: dict[str, Any],
    *,
    entrypoint: str,
    python_import_root: str = "src",
    invocation_style: str,
    result_root: str,
    patch_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": executor.MATCHED_CAPABILITY_RECEIPT_SCHEMA_VERSION,
        "status": "qualified",
        "candidate_id": candidate["candidate_id"],
        "capability_descriptor_sha256": protocol_fixtures._capability_descriptor_sha256(
            candidate
        ),
        "qualification_trust_anchor_identity": candidate["runtime_binding"][
            "qualification_trust_anchor_identity"
        ],
        "source": copy.deepcopy(candidate["source"]),
        "configuration_sha256": candidate["configuration"]["derived_sha256"],
        "image_sha256": candidate["runtime_binding"]["image_sha256"],
        "runtime_profile_sha256": candidate["runtime_binding"]["runtime_profile_sha256"],
        "task_identity_sha256": candidate["runtime_binding"]["task_identity_sha256"],
        "environment_rng_schedule_sha256": candidate["environment_rng"]["schedule_sha256"],
        "rng_parity_contract_sha256": executor.RNG_PARITY_CONTRACT_SHA256,
        "entrypoint_family": candidate["entrypoint_family"],
        "entrypoint_path": entrypoint,
        "python_import_root": python_import_root,
        "invocation_style": invocation_style,
        "result_root": result_root,
        "agent_rng_identity": candidate["agent_rng"]["identity"],
        "environment_key_shared": candidate["agent_rng"]["environment_key_shared"],
        "rng_isolation_patch_sha256": patch_sha256,
    }


def _fixture(
    tmp_path: Path,
    *,
    candidate_id: str = "alberta_causal",
) -> tuple[
    dict[str, Any],
    protocol.ForagerMatchedProtocol,
    dict[str, executor.CandidateExecutionAssets],
]:
    """Build a plan-ready single-candidate fixture on real temporary files.

    Rebinds the generic protocol payload to the executor's qualified
    constants (image/profile/scorer/task digests), materializes a minimal
    on-disk source tree whose entrypoint raises if ever executed, and
    returns ``(payload, parsed protocol, execution assets)`` for
    ``candidate_id``.  Shared with sibling suites via ``executor_fixtures``.
    """
    payload = protocol_fixtures._payload()
    if candidate_id == "isolated_ppo":
        def replace_isolated_rtu(value: Any) -> Any:
            if type(value) is dict:
                return {key: replace_isolated_rtu(item) for key, item in value.items()}
            if type(value) is list:
                return [replace_isolated_rtu(item) for item in value]
            return "isolated_ppo" if value == "isolated_rtu" else value

        payload = cast(dict[str, Any], replace_isolated_rtu(payload))
        isolated = next(
            item for item in payload["candidates"] if item["candidate_id"] == "isolated_ppo"
        )
        isolated["implementation_kind"] = "upstream_ppo_isolated_rng"
    task = parity.task_descriptor()
    payload["task"].update(
        {
            "preset": task["preset"],
            "environment_id": task["environment_id"],
            "foragax_distribution": task["foragax_distribution"],
            "foragax_version": task["foragax_version"],
            "observation_type": task["observation_type"],
            "aperture_size": task["aperture_size"],
            "task_identity_sha256": task["task_sha256"],
            "environment_rng_schedule_sha256": (
                executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
            ),
        }
    )
    payload["analysis_plan"]["metric_implementation_sha256"] = (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    payload["selection_plan"]["metric_implementation_sha256"] = (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    payload["runtime"].update(
        {
            "image_sha256": executor.QUALIFIED_IMAGE_SHA256,
            "runtime_profile_sha256": executor.QUALIFIED_RUNTIME_PROFILE_SHA256,
            "executor_qualification_receipt_sha256": (
                executor.QUALIFIED_EXECUTOR_RECEIPT_SHA256
            ),
        }
    )
    for candidate in payload["candidates"]:
        candidate["runtime_binding"].update(
            {
                "image_sha256": executor.QUALIFIED_IMAGE_SHA256,
                "runtime_profile_sha256": executor.QUALIFIED_RUNTIME_PROFILE_SHA256,
                "task_identity_sha256": task["task_sha256"],
            }
        )
        if not candidate["agent_rng"]["environment_key_shared"]:
            candidate["environment_rng"]["schedule_sha256"] = (
                executor.MATCHED_ENVIRONMENT_RNG_SCHEDULE_SHA256
            )

    candidate = next(item for item in payload["candidates"] if item["candidate_id"] == candidate_id)
    source_root = tmp_path / f"{candidate_id}-source"
    source_root.mkdir()
    source_entrypoint = (
        "src/worker.py" if candidate_id == "alberta_causal" else "src/rtu_ppo.py"
    )
    entrypoint = source_root / source_entrypoint
    entrypoint.parent.mkdir()
    entrypoint.write_text("raise SystemExit('not executed by tests')\n", encoding="utf-8")
    entrypoint.chmod(0o644)
    inventory = executor.source_inventory(source_root)
    source_archive = tmp_path / f"{candidate_id}.tar"
    source_archive.write_bytes(f"archive:{candidate_id}".encode("ascii"))
    configuration = tmp_path / f"{candidate_id}.json"
    configuration.write_bytes(b"{}")
    source_update: dict[str, Any] = {
        "archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "inventory_sha256": executor.source_inventory_sha256(source_root),
    }
    candidate["source"].pop("tree_sha256", None)
    if candidate_id in {"alberta_causal", "isolated_ppo", "isolated_rtu"}:
        source_update.update(
            {
                "provenance_kind": "reviewed_snapshot",
                "tree_git_sha1": None,
                "snapshot_descriptor_sha256": _sha(f"snapshot:{candidate_id}"),
            }
        )
    else:
        source_update["snapshot_descriptor_sha256"] = None
    candidate["source"].update(source_update)
    candidate["configuration"].update(
        {
            "original_sha256": hashlib.sha256(b"{}").hexdigest(),
            "derived_sha256": hashlib.sha256(b"{}").hexdigest(),
            "allowed_transforms": [],
        }
    )
    patch_sha256 = (
        executor.QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256
        if candidate_id in {"isolated_ppo", "isolated_rtu"}
        else None
    )
    receipt = _receipt(
        candidate,
        entrypoint=source_entrypoint,
        invocation_style=(
            "alberta_single_seed_v1"
            if candidate_id == "alberta_causal"
            else "official_foragax_ppo_frozen_updates_v1"
        ),
        result_root=("results/alberta" if candidate_id == "alberta_causal" else "results/rtu"),
        patch_sha256=patch_sha256,
    )
    candidate["runtime_binding"]["capability_qualification_receipt_sha256"] = _canonical_sha(
        receipt
    )
    for item in payload["candidates"]:
        item["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            protocol_fixtures._capability_descriptor_sha256(item)
        )
    frozen = protocol.parse_forager_matched_protocol(payload)
    assets = {
        candidate_id: executor.CandidateExecutionAssets(
            candidate_id=candidate_id,
            source_root=source_root,
            source_archive=source_archive,
            source_inventory=inventory,
            original_configuration=configuration,
            configuration=configuration,
            capability_receipt=receipt,
        )
    }
    return payload, frozen, assets


def _plan(
    tmp_path: Path,
    *,
    candidate_id: str = "alberta_causal",
) -> executor.MatchedExecutionPlan:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id=candidate_id)
    return executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=(candidate_id,),
    )


def _runtime_version_payload() -> dict[str, Any]:
    return {"Client": {"Version": "qualified-test-runtime"}}


def _runtime_inspection_payload() -> dict[str, Any]:
    return {
        "Id": f"sha256:{executor.QUALIFIED_IMAGE_SHA256}",
        "Config": {
            "Labels": {
                "io.elizaos.alberta.foragax.launcher-contract": (
                    "oci-read-only-stdout-tar-v4"
                )
            }
        },
    }


def _runtime_reinspection_result(
    command: Sequence[str],
    live: executor.LiveRuntimeIdentity,
) -> executor.ProcessResult | None:
    if len(command) >= 2 and command[1] == "version":
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(live.version),
            b"",
        )
    if len(command) >= 3 and command[1:3] == ("image", "inspect"):
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(live.image_inspection),
            b"",
        )
    return None


def _runtime(
    tmp_path: Path,
    plan: executor.MatchedExecutionPlan,
) -> executor.LiveRuntimeIdentity:
    """Qualify a fake live runtime: a stub ``docker`` binary (which exits 99
    if actually invoked) plus canned version/inspection payloads routed
    through the injected runner.  No container runtime is touched."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "docker"
    runtime.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    runtime.chmod(0o755)

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        if "version" in command:
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(_runtime_version_payload()),
                b"",
            )
        return executor.ProcessResult(
            0,
            executor.canonical_json_bytes(_runtime_inspection_payload()),
            b"",
        )

    return executor.qualify_live_runtime(plan, runtime=runtime, runner=runner)


def _scoring_output(plan: executor.MatchedExecutionPlan, seed: int) -> bytes:
    candidate = plan.candidates[0]
    sample_count = (plan.protocol.horizon + 99) // 100
    tail_start = int(0.9 * sample_count)
    return executor.canonical_json_bytes(
        {
            "schema_version": executor.SCORER_OUTPUT_SCHEMA,
            "horizon": plan.protocol.horizon,
            "seeds": [seed],
            "result_root": candidate.result_root,
            "records": [
                {
                    "archive_path": (
                        f"payload/{candidate.result_root}/data/{seed}.npz"
                    ),
                    "ema_sample_count": sample_count,
                    "ema_tail_sample_count": sample_count - tail_start,
                    "ema_tail_start_index": tail_start,
                    "final_unadjusted_ema": 0.25,
                    "fov_last_10pct_ema_auc": 0.5,
                    "npz_sha256": _sha(f"npz:{seed}"),
                    "npz_size_bytes": 4096,
                    "reward_dtype": "<f4",
                    "reward_shape": [plan.protocol.horizon],
                    "reward_sum_float64": 10.0,
                    "reward_trace_sha256": _sha(f"trace-content:{seed}"),
                    "seed": seed,
                }
            ],
        }
    )


def _execute_fixture(
    tmp_path: Path,
) -> tuple[
    executor.MatchedExecutionPlan,
    executor.SeedExecutionArtifacts,
]:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        calls.append(tuple(command))
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(0, _scoring_output(plan, seed), b"")
        return executor.ProcessResult(0, b"opaque-ustar-never-opened-on-host", b"")

    artifacts = executor.execute_seed(
        plan,
        "alberta_causal",
        seed,
        tmp_path / "raw.tar",
        live,
        runner=runner,
    )
    assert sum("run" in command for command in calls) == 2
    assert sum(len(command) >= 2 and command[1] == "version" for command in calls) == 2
    assert sum(command[1:3] == ("image", "inspect") for command in calls) == 2
    return plan, artifacts


def _complete_execution_artifacts(
    plan: executor.MatchedExecutionPlan,
    live: executor.LiveRuntimeIdentity,
) -> dict[str, list[executor.SeedExecutionArtifacts]]:
    blocks: dict[str, list[executor.SeedExecutionArtifacts]] = {}
    for candidate in plan.candidates:
        candidate_id = candidate.candidate.candidate_id
        records: list[executor.SeedExecutionArtifacts] = []
        for seed in plan.protocol.active_seeds:
            records.append(
                executor._artifact_mappings(
                    plan=plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=_sha(f"archive:{candidate_id}:{seed}"),
                    raw_archive_size=1024,
                    live_runtime=live,
                    scorer_record={
                        "fov_last_10pct_ema_auc": 0.5,
                        "npz_sha256": _sha(f"npz:{candidate_id}:{seed}"),
                        "npz_size_bytes": 4096,
                        "reward_trace_sha256": _sha(
                            f"trace-content:{candidate_id}:{seed}"
                        ),
                        "reward_dtype": "<f4",
                        "reward_shape": [plan.protocol.horizon],
                    },
                )
            )
        blocks[candidate_id] = records
    return blocks


def _two_candidate_plan(tmp_path: Path) -> executor.MatchedExecutionPlan:
    causal_root = tmp_path / "causal"
    rtu_root = tmp_path / "rtu"
    causal_root.mkdir()
    rtu_root.mkdir()
    causal_payload, _causal_protocol, causal_assets = _fixture(
        causal_root,
        candidate_id="alberta_causal",
    )
    rtu_payload, _rtu_protocol, rtu_assets = _fixture(
        rtu_root,
        candidate_id="isolated_rtu",
    )
    rtu_candidate = next(
        item for item in rtu_payload["candidates"] if item["candidate_id"] == "isolated_rtu"
    )
    for index, candidate in enumerate(causal_payload["candidates"]):
        if candidate["candidate_id"] == "isolated_rtu":
            causal_payload["candidates"][index] = copy.deepcopy(rtu_candidate)
            break
    frozen = protocol.parse_forager_matched_protocol(causal_payload)
    assets = {
        "alberta_causal": causal_assets["alberta_causal"],
        "isolated_rtu": rtu_assets["isolated_rtu"],
    }
    return executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=tuple(assets),
    )


def _refresh_receipt_index_digest(payload: dict[str, Any]) -> None:
    unsigned = copy.deepcopy(payload)
    del unsigned["payload_sha256"]
    payload["payload_sha256"] = _canonical_sha(unsigned)


def _direct_plan(
    template: executor.MatchedExecutionPlan,
    *,
    source_manifest: Mapping[str, Any],
    executor_manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    candidate_index: Mapping[str, executor.PreparedCandidate],
    candidates: tuple[executor.PreparedCandidate, ...] | None = None,
    protocol_value: protocol.ForagerMatchedProtocol | None = None,
) -> executor.MatchedExecutionPlan:
    return executor.MatchedExecutionPlan(
        protocol=template.protocol if protocol_value is None else protocol_value,
        qualification_manifest_sha256=template.qualification_manifest_sha256,
        candidates=template.candidates if candidates is None else candidates,
        source_manifest=source_manifest,
        executor_manifest=executor_manifest,
        payload=payload,
        candidate_index=candidate_index,
        cpu_qualification_root=template.cpu_qualification_root,
        rng_parity_qualification_root=template.rng_parity_qualification_root,
    )


def test_plan_is_content_addressed_nonexecuting_and_replayable(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("alberta_causal",),
    )

    assert plan.payload["promotion_authorized"] is False
    assert plan.payload["external_verification_required"] is True
    assert plan.source_manifest_sha256 == plan.payload["source_manifest_sha256"]
    assert plan.executor_manifest_sha256 == plan.payload["executor_manifest_sha256"]
    assert plan.executor_manifest["scorer"]["sha256"] == (
        executor.QUALIFIED_SCORER_SOURCE_SHA256
    )
    qualification = plan.executor_manifest["qualification_artifacts"]
    assert qualification["cpu_qualification"]["authority_boundary"] == {
        "endorsement_created": False,
        "endorsements_at_seal": 0,
        "gpu_qualified": False,
        "performance_claim": False,
        "seed_class": "open_development",
        "trust_profile_created": False,
        "trust_profiles_at_seal": 0,
    }
    assert qualification["rng_parity_qualification"]["status"] == (
        "content_complete_external_executor_receipt_unverified"
    )
    assert qualification["rng_parity_qualification"]["promotion_authorized"] is False
    assert b"<ACTIVE_SEED>" in plan.canonical_bytes
    assert str(tmp_path).encode() not in plan.canonical_bytes
    for mapping, key in (
        (plan.payload, "stage"),
        (plan.source_manifest, "stage"),
        (plan.executor_manifest, "protocol_sha256"),
        (plan.candidate_index, "replacement"),
    ):
        with pytest.raises(TypeError):
            cast(Any, mapping)[key] = "mutated"
    with pytest.raises(TypeError):
        cast(Any, plan.executor_manifest["qualified_lock"])["image_sha256"] = "0" * 64

    replayed = executor.parse_execution_plan(
        plan.canonical_bytes,
        protocol=frozen,
        assets=assets,
        expected_plan_sha256=plan.plan_sha256,
        expected_qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
    )
    assert replayed.to_dict() == plan.to_dict()


def test_direct_plan_rejects_legacy_plan_schema_despite_exact_manifest_closure(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    payload = template.to_dict()
    source_manifest = cast(dict[str, Any], payload["source_manifest"])
    executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
    payload["schema_version"] = "alberta.forager_matched_execution_plan.v1"

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="object schema_version is unsupported",
    ):
        _direct_plan(
            template,
            source_manifest=source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_legacy_executor_schema_despite_rehashed_closure(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    payload = template.to_dict()
    source_manifest = cast(dict[str, Any], payload["source_manifest"])
    executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
    executor_manifest["schema_version"] = "alberta.forager_matched_executor_manifest.v1"
    payload["executor_manifest_sha256"] = _canonical_sha(executor_manifest)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="executor-manifest schema_version is unsupported",
    ):
        _direct_plan(
            template,
            source_manifest=source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_protocol_outside_qualified_lock(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    changed_protocol = replace(
        template.protocol,
        runtime=replace(
            template.protocol.runtime,
            executor_qualification_receipt_sha256=_sha("unqualified-executor"),
        ),
    )
    source_manifest = cast(dict[str, Any], template.to_dict()["source_manifest"])
    source_manifest["protocol_sha256"] = changed_protocol.protocol_sha256
    executor_manifest = cast(dict[str, Any], template.to_dict()["executor_manifest"])
    executor_manifest["protocol_sha256"] = changed_protocol.protocol_sha256
    executor_manifest["runtime"] = changed_protocol.runtime.to_dict()
    payload = template.to_dict()
    payload.update(
        {
            "protocol_sha256": changed_protocol.protocol_sha256,
            "source_manifest": source_manifest,
            "source_manifest_sha256": _canonical_sha(source_manifest),
            "executor_manifest": executor_manifest,
            "executor_manifest_sha256": _canonical_sha(executor_manifest),
        }
    )

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="qualified matched-current lock",
    ):
        _direct_plan(
            template,
            protocol_value=changed_protocol,
            source_manifest=source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_rehashed_executor_manifest_semantic_forgery(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    payload = template.to_dict()
    executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
    resource_limits = cast(dict[str, Any], executor_manifest["resource_limits"])
    resource_limits["pids"] = 999
    payload["executor_manifest_sha256"] = _canonical_sha(executor_manifest)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="exact qualified reconstruction",
    ):
        _direct_plan(
            template,
            source_manifest=template.source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_protocol_candidate_index_outside_hashed_tuple(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    original = template.candidates[0]
    forged_candidate = replace(
        original.candidate,
        resources=replace(
            original.candidate.resources,
            parameter_count=original.candidate.resources.parameter_count + 1,
        ),
    )
    forged = replace(original, candidate=forged_candidate)
    changed_protocol = replace(
        template.protocol,
        candidate_index=MappingProxyType(
            {forged_candidate.candidate_id: forged_candidate}
        ),
    )
    source_manifest = executor._source_manifest_for_prepared(  # noqa: SLF001
        changed_protocol,
        (forged,),
    )
    payload = template.to_dict()
    payload["source_manifest"] = source_manifest
    payload["source_manifest_sha256"] = _canonical_sha(source_manifest)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="candidate index differs from its canonical candidate tuple",
    ):
        _direct_plan(
            template,
            protocol_value=changed_protocol,
            source_manifest=source_manifest,
            executor_manifest=template.executor_manifest,
            payload=payload,
            candidate_index={forged_candidate.candidate_id: forged},
            candidates=(forged,),
        )


def test_direct_plan_rejects_noncanonical_outer_semantics_and_source_schema(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    semantic_cases: tuple[tuple[str, Any], ...] = (
        ("classification", "promoting"),
        ("promotion_authorized", True),
        ("external_verification_required", False),
        ("stage", "sealed_evaluation"),
        ("active_seeds", [float(template.protocol.active_seeds[0])]),
        ("horizon", float(template.protocol.horizon)),
        ("candidate_command_templates", []),
        ("scoring_boundary", {}),
    )
    for field, changed_value in semantic_cases:
        payload = template.to_dict()
        source_manifest = cast(dict[str, Any], payload["source_manifest"])
        executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
        payload[field] = changed_value
        with pytest.raises(executor.ForagerMatchedExecutorError):
            _direct_plan(
                template,
                source_manifest=source_manifest,
                executor_manifest=executor_manifest,
                payload=payload,
                candidate_index=dict(template.candidate_index),
            )

    payload = template.to_dict()
    payload.pop("candidate_order")
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="execution plan payload keys differ",
    ):
        _direct_plan(
            template,
            source_manifest=template.source_manifest,
            executor_manifest=template.executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )

    payload = template.to_dict()
    source_manifest = cast(dict[str, Any], payload["source_manifest"])
    executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
    source_manifest["schema_version"] = "alberta.forager_matched_source_manifest.v0"
    payload["source_manifest_sha256"] = _canonical_sha(source_manifest)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="exact manifest closure",
    ):
        _direct_plan(
            template,
            source_manifest=source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )

    payload = template.to_dict()
    source_manifest = cast(dict[str, Any], payload["source_manifest"])
    executor_manifest = cast(dict[str, Any], payload["executor_manifest"])
    source_manifest["candidates"] = []
    payload["source_manifest_sha256"] = _canonical_sha(source_manifest)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="typed protocol/candidate closure",
    ):
        _direct_plan(
            template,
            source_manifest=source_manifest,
            executor_manifest=executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_unparsed_prepared_candidate_receipt(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    receipt = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(template.candidates[0].capability_receipt)),
    )
    receipt["status"] = "forged"
    forged = replace(template.candidates[0], capability_receipt=receipt)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="capability receipt",
    ):
        _direct_plan(
            template,
            source_manifest=template.source_manifest,
            executor_manifest=template.executor_manifest,
            payload=template.payload,
            candidate_index={forged.candidate.candidate_id: forged},
            candidates=(forged,),
        )


def test_direct_plan_rejects_prepared_candidate_source_changed_after_preparation(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    prepared = template.candidates[0]
    prepared.source_root.joinpath(*PurePosixPath(prepared.entrypoint_path).parts).write_bytes(
        b"changed after preparation\n"
    )

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="source root bytes differ",
    ):
        _direct_plan(
            template,
            source_manifest=template.source_manifest,
            executor_manifest=template.executor_manifest,
            payload=template.payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_rejects_candidate_with_same_id_but_outside_protocol(
    tmp_path: Path,
) -> None:
    template = _plan(tmp_path)
    original = template.candidates[0]
    changed_candidate = replace(
        original.candidate,
        resources=replace(
            original.candidate.resources,
            parameter_count=original.candidate.resources.parameter_count + 1,
        ),
    )
    forged = replace(original, candidate=changed_candidate)
    source_manifest = executor._source_manifest_for_prepared(  # noqa: SLF001
        template.protocol,
        (forged,),
    )
    payload = template.to_dict()
    payload["source_manifest"] = source_manifest
    payload["source_manifest_sha256"] = _canonical_sha(source_manifest)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="typed protocol/candidate closure",
    ):
        _direct_plan(
            template,
            source_manifest=source_manifest,
            executor_manifest=template.executor_manifest,
            payload=payload,
            candidate_index={forged.candidate.candidate_id: forged},
            candidates=(forged,),
        )


def test_direct_plan_normalizes_cyclic_outer_mapping_error(tmp_path: Path) -> None:
    template = _plan(tmp_path)
    payload = template.to_dict()
    payload["source_manifest"] = payload

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="canonical JSON",
    ):
        _direct_plan(
            template,
            source_manifest=template.source_manifest,
            executor_manifest=template.executor_manifest,
            payload=payload,
            candidate_index=dict(template.candidate_index),
        )


def test_direct_plan_snapshots_all_retained_caller_mappings(tmp_path: Path) -> None:
    template = _plan(tmp_path)
    serialized = template.to_dict()
    source_manifest = copy.deepcopy(
        cast(dict[str, Any], serialized["source_manifest"])
    )
    executor_manifest = copy.deepcopy(
        cast(dict[str, Any], serialized["executor_manifest"])
    )
    payload = copy.deepcopy(serialized)
    capability_receipt = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(template.candidates[0].capability_receipt)),
    )
    source_inventory = cast(
        dict[str, Any],
        json.loads(executor.canonical_json_bytes(template.candidates[0].source_inventory)),
    )
    caller_candidate = replace(
        template.candidates[0],
        capability_receipt=capability_receipt,
        source_inventory=source_inventory,
    )
    caller_candidates = (caller_candidate,)
    candidate_index = {caller_candidate.candidate.candidate_id: caller_candidate}
    expected_payload = copy.deepcopy(payload)
    expected_source_sha256 = _canonical_sha(source_manifest)
    expected_executor_sha256 = _canonical_sha(executor_manifest)
    expected_candidate_ids = tuple(candidate_index)

    direct = _direct_plan(
        template,
        source_manifest=source_manifest,
        executor_manifest=executor_manifest,
        payload=payload,
        candidate_index=candidate_index,
        candidates=caller_candidates,
    )

    source_manifest["stage"] = "mutated_source"
    cast(list[dict[str, Any]], source_manifest["candidates"])[0][
        "candidate_id"
    ] = "mutated_source_candidate"
    executor_manifest["authentication_state"] = "mutated_executor"
    cast(dict[str, Any], executor_manifest["qualified_lock"])["image_sha256"] = "0" * 64
    payload["classification"] = "mutated_payload"
    cast(dict[str, Any], payload["source_manifest"])["stage"] = "mutated_payload_source"
    cast(dict[str, Any], payload["executor_manifest"])[
        "authentication_state"
    ] = "mutated_payload_executor"
    candidate_index.clear()
    candidate_index["replacement"] = template.candidates[0]
    capability_receipt["status"] = "mutated_receipt"
    cast(list[dict[str, Any]], source_inventory["files"])[0]["sha256"] = "0" * 64

    assert direct.to_dict() == expected_payload
    assert direct.source_manifest_sha256 == expected_source_sha256
    assert direct.executor_manifest_sha256 == expected_executor_sha256
    assert tuple(direct.candidate_index) == expected_candidate_ids
    assert direct.candidates[0].capability_receipt["status"] == "qualified"
    assert direct.candidates[0].source_inventory["files"][0]["sha256"] != "0" * 64


def test_plan_qualification_manifest_binding_is_hard_v2_and_hash_sensitive(
    tmp_path: Path,
) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    changed_qualification_sha256 = _sha("changed-qualification-manifest")
    first = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("alberta_causal",),
    )
    changed = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=changed_qualification_sha256,
        candidate_ids=("alberta_causal",),
    )

    assert first.payload["schema_version"] == "alberta.forager_matched_execution_plan.v2"
    assert first.executor_manifest["schema_version"] == (
        "alberta.forager_matched_executor_manifest.v2"
    )
    assert first.qualification_manifest_sha256 == _QUALIFICATION_MANIFEST_SHA256
    assert first.payload["qualification_manifest_sha256"] == _QUALIFICATION_MANIFEST_SHA256
    assert first.executor_manifest["qualification_manifest_sha256"] == (
        _QUALIFICATION_MANIFEST_SHA256
    )
    assert first.source_manifest_sha256 == changed.source_manifest_sha256
    assert first.executor_manifest_sha256 != changed.executor_manifest_sha256
    assert first.plan_sha256 != changed.plan_sha256

    with pytest.raises(executor.ForagerMatchedExecutorError, match="lowercase SHA-256"):
        executor.build_execution_plan(
            frozen,
            assets,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256.upper(),
            candidate_ids=("alberta_causal",),
        )
    with pytest.raises(executor.ForagerMatchedExecutorError, match="classification/schema"):
        executor.parse_execution_plan(
            first.canonical_bytes,
            protocol=frozen,
            assets=assets,
            expected_plan_sha256=first.plan_sha256,
            expected_qualification_manifest_sha256=changed_qualification_sha256,
        )

    legacy = first.to_dict()
    legacy["schema_version"] = "alberta.forager_matched_execution_plan.v1"
    with pytest.raises(executor.ForagerMatchedExecutorError, match="classification/schema"):
        executor.parse_execution_plan(
            legacy,
            protocol=frozen,
            assets=assets,
            expected_plan_sha256=first.plan_sha256,
            expected_qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    ("qualification_kind", "filename"),
    (("cpu", "receipt.v1.json"), ("rng", "receipt.json")),
)
def test_qualification_loader_rejects_artifact_drift_despite_matching_constants(
    tmp_path: Path,
    qualification_kind: str,
    filename: str,
) -> None:
    cpu_root = tmp_path / "cpu"
    rng_root = tmp_path / "rng"
    cpu_root.mkdir()
    rng_root.mkdir()
    for name in ("receipt.v1.json", "qualification.json", "environment-profile.json"):
        shutil.copy2(executor.DEFAULT_CPU_QUALIFICATION_ROOT / name, cpu_root / name)
    for name in ("plan.json", "receipt.json"):
        shutil.copy2(executor.DEFAULT_RNG_PARITY_QUALIFICATION_ROOT / name, rng_root / name)
    target = (cpu_root if qualification_kind == "cpu" else rng_root) / filename
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="frozen file digest"):
        executor.load_executor_qualification_artifacts(
            cpu_root=cpu_root,
            rng_parity_root=rng_root,
        )


def test_replay_uses_frozen_open_builder_transforms_including_search_oracle() -> None:
    frozen = open_protocol_fixtures._build()
    for candidate_id, requirement in open_protocol_fixtures._UPSTREAM_CONFIGS.items():
        relative_path, _original_sha256, derived_sha256, _transforms = requirement
        if candidate_id == "search_oracle":
            path = open_protocol_fixtures._SEARCH_ORACLE_FIXTURE
        elif candidate_id == "external_dqn_plain":
            path = Path(__file__).parent / "fixtures/forager_matched/DQN.json"
        else:
            path = open_protocol_fixtures._PINNED_CONFIG_ROOT / relative_path
        original = path.read_bytes()
        if candidate_id == "search_oracle":
            original = original.removesuffix(b"\n")
        replayed = executor._replay_configuration_transforms(
            frozen.candidate_index[candidate_id],
            original,
        )
        assert hashlib.sha256(replayed).hexdigest() == derived_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_sha256", "0" * 64),
        ("runtime_profile_sha256", "0" * 64),
        ("executor_qualification_receipt_sha256", "0" * 64),
    ],
)
def test_plan_rejects_unqualified_runtime_lock(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    payload, _frozen, assets = _fixture(tmp_path)
    payload["runtime"][field] = value
    for candidate in payload["candidates"]:
        if field in candidate["runtime_binding"]:
            candidate["runtime_binding"][field] = value
        candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            protocol_fixtures._capability_descriptor_sha256(candidate)
        )

    with pytest.raises(executor.ForagerMatchedExecutorError, match="qualified matched-current"):
        executor.build_execution_plan(
            protocol.parse_forager_matched_protocol(payload),
            assets,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
            candidate_ids=("alberta_causal",),
        )


def test_plan_rejects_horizon_task_scorer_and_source_drift(tmp_path: Path) -> None:
    payload, frozen, assets = _fixture(tmp_path)
    mutators: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.__setitem__("horizon", executor.MATCHED_HORIZON - 1),
        lambda value: value["task"].__setitem__("task_identity_sha256", "0" * 64),
        lambda value: value["analysis_plan"].__setitem__(
            "metric_implementation_sha256", "0" * 64
        ),
    )
    for mutator in mutators:
        changed = copy.deepcopy(payload)
        mutator(changed)
        if changed["horizon"] != executor.MATCHED_HORIZON:
            for candidate in changed["candidates"]:
                candidate["execution_semantics"] = {
                    "rollout_steps": None,
                    "num_rollouts": None,
                    "update_semantics": "environment_step_counted",
                }
        if changed["analysis_plan"]["metric_implementation_sha256"] == "0" * 64:
            changed["selection_plan"]["metric_implementation_sha256"] = "0" * 64
            with pytest.raises(executor.ForagerMatchedExecutorError):
                executor.build_execution_plan(
                    changed,
                    assets,
                    qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
                    candidate_ids=("alberta_causal",),
                )

    assets["alberta_causal"].source_archive.write_bytes(b"drift")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="source archive"):
        executor.build_execution_plan(
            frozen,
            assets,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
            candidate_ids=("alberta_causal",),
        )


def test_receipt_is_exact_canonical_and_not_a_self_attested_boolean(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["alberta_causal"].capability_receipt))
    )
    receipt["self_attested_trusted"] = True
    changed_assets = {
        "alberta_causal": executor.CandidateExecutionAssets(
            candidate_id="alberta_causal",
            source_root=assets["alberta_causal"].source_root,
            source_archive=assets["alberta_causal"].source_archive,
            source_inventory=assets["alberta_causal"].source_inventory,
            original_configuration=assets["alberta_causal"].original_configuration,
            configuration=assets["alberta_causal"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.build_execution_plan(
            frozen,
            changed_assets,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
            candidate_ids=("alberta_causal",),
        )


def test_rtu_requires_separately_bound_isolated_rng_patch(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="isolated_rtu")
    plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("isolated_rtu",),
    )
    assert plan.candidates[0].rng_isolation_patch_sha256 == (
        executor.QUALIFIED_RTU_RNG_ISOLATION_PATCH_SHA256
    )

    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["isolated_rtu"].capability_receipt))
    )
    receipt["rng_isolation_patch_sha256"] = None
    changed = {
        "isolated_rtu": executor.CandidateExecutionAssets(
            candidate_id="isolated_rtu",
            source_root=assets["isolated_rtu"].source_root,
            source_archive=assets["isolated_rtu"].source_archive,
            source_inventory=assets["isolated_rtu"].source_inventory,
            original_configuration=assets["isolated_rtu"].original_configuration,
            configuration=assets["isolated_rtu"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="source-bound RNG patch"):
        executor.build_execution_plan(
            frozen,
            changed,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
            candidate_ids=("isolated_rtu",),
        )


def test_isolated_ppo_rejects_any_unreviewed_rng_patch(tmp_path: Path) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="isolated_ppo")
    receipt = copy.deepcopy(
        dict(cast(Mapping[str, Any], assets["isolated_ppo"].capability_receipt))
    )
    receipt["rng_isolation_patch_sha256"] = _sha("unreviewed-isolated-ppo-patch")
    frozen_payload = frozen.to_dict()
    candidate = next(
        item for item in frozen_payload["candidates"] if item["candidate_id"] == "isolated_ppo"
    )
    candidate["runtime_binding"]["capability_qualification_receipt_sha256"] = _canonical_sha(
        receipt
    )
    candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        protocol_fixtures._capability_descriptor_sha256(candidate)
    )
    changed_assets = {
        "isolated_ppo": executor.CandidateExecutionAssets(
            candidate_id="isolated_ppo",
            source_root=assets["isolated_ppo"].source_root,
            source_archive=assets["isolated_ppo"].source_archive,
            source_inventory=assets["isolated_ppo"].source_inventory,
            original_configuration=assets["isolated_ppo"].original_configuration,
            configuration=assets["isolated_ppo"].configuration,
            capability_receipt=receipt,
        )
    }
    with pytest.raises(executor.ForagerMatchedExecutorError, match="reviewed RNG patch"):
        executor.build_execution_plan(
            frozen_payload,
            changed_assets,
            qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
            candidate_ids=("isolated_ppo",),
        )


def test_fixed_descriptive_shared_rng_ppo_remains_labeled_noninferential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _payload, frozen, assets = _fixture(tmp_path, candidate_id="exact_ppo")
    monkeypatch.setattr(
        executor,
        "QUALIFIED_UPSTREAM_SOURCE_ARCHIVE_SHA256",
        frozen.candidate_index["exact_ppo"].source.archive_sha256,
    )
    plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("exact_ppo",),
    )
    candidate = plan.candidates[0].candidate
    assert candidate.pairing.analysis_role == "descriptive_only"
    assert candidate.pairing.eligible is False
    assert candidate.pairing.exclusion_reasons == ("shared_agent_environment_rng",)


def test_live_runtime_and_commands_pin_exact_cpu_sandbox_seed_and_horizon(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    command = executor.build_candidate_command(plan, "alberta_causal", seed, live)

    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--cpus=4.0" in command
    assert "--memory=16g" in command
    assert "--memory-swap=16g" in command
    assert f"sha256:{executor.QUALIFIED_IMAGE_SHA256}" in command
    assert f"--seed={seed}" in command
    assert f"--horizon={executor.MATCHED_HORIZON}" in command
    assert not any("--device" in item for item in command)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="active protocol seed"):
        executor.build_candidate_command(plan, "alberta_causal", seed + 10_000, live)


@pytest.mark.parametrize(
    ("failure_phase", "cleanup_error", "expected_type", "expected_message"),
    (
        (
            "kill",
            OSError("injected kill failure"),
            executor.ForagerMatchedExecutorError,
            "OCI process could not be terminated cleanly",
        ),
        (
            "kill",
            ProcessLookupError("injected exited-child race"),
            executor._BoundedProcessOutputError,  # noqa: SLF001
            "active byte limit",
        ),
        (
            "wait",
            subprocess.TimeoutExpired(("unused",), 10),
            executor.ForagerMatchedExecutorError,
            "OCI process could not be reaped after termination",
        ),
        (
            "wait",
            OSError("injected wait failure"),
            executor.ForagerMatchedExecutorError,
            "OCI process could not be inspected after termination",
        ),
        (
            "selector_close",
            OSError("injected selector close failure"),
            executor.ForagerMatchedExecutorError,
            "OCI process resources could not be closed cleanly",
        ),
        (
            "stdout_close",
            OSError("injected stdout close failure"),
            executor.ForagerMatchedExecutorError,
            "OCI process resources could not be closed cleanly",
        ),
        (
            "stderr_close",
            OSError("injected stderr close failure"),
            executor.ForagerMatchedExecutorError,
            "OCI process resources could not be closed cleanly",
        ),
    ),
    ids=(
        "kill-error",
        "kill-process-gone",
        "reap-timeout",
        "reap-error",
        "selector-close-error",
        "stdout-close-error",
        "stderr-close-error",
    ),
)
def test_bounded_process_normalizes_cleanup_failures_and_attempts_every_close(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    cleanup_error: BaseException,
    expected_type: type[BaseException],
    expected_message: str,
) -> None:
    class CloseBuffer(BytesIO):
        def __init__(self, failure_name: str) -> None:
            super().__init__()
            self.failure_name = failure_name
            self.close_called = False

        def close(self) -> None:
            self.close_called = True
            super().close()
            if failure_phase == self.failure_name:
                raise cleanup_error

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = CloseBuffer("stdout_close")
            self.stderr = CloseBuffer("stderr_close")
            self.kill_called = False
            self.waited = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_called = True
            if failure_phase == "kill":
                raise cleanup_error

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            if failure_phase == "wait":
                raise cleanup_error
            return -9

    class SelectorKey:
        data = "stdout"
        fd = 101

    class OverflowSelector:
        def __init__(self) -> None:
            self.closed = False

        def register(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def get_map(self) -> dict[str, bool]:
            return {"active": True}

        def select(self, _timeout: float) -> list[tuple[SelectorKey, int]]:
            return [(SelectorKey(), selectors.EVENT_READ)]

        def close(self) -> None:
            self.closed = True
            if failure_phase == "selector_close":
                raise cleanup_error

    process = FakeProcess()
    selector = OverflowSelector()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"xx")

    with pytest.raises(expected_type, match=expected_message) as caught:
        executor._run_bounded_process(  # noqa: SLF001
            ("unused",),
            timeout=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )

    assert process.kill_called is True
    assert process.waited is True
    assert selector.closed is True
    assert process.stdout.close_called is True
    assert process.stderr.close_called is True
    if isinstance(cleanup_error, ProcessLookupError):
        assert caught.value.__cause__ is None
    else:
        assert caught.value.__cause__ is cleanup_error


@pytest.mark.parametrize("failure_kind", ["timeout", "runner_error"])
def test_default_runner_cidfile_force_removes_interrupted_container(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def fail_bounded_run(command: Sequence[str], **_kwargs: Any) -> NoReturn:
        materialized = tuple(command)
        calls.append(materialized)
        cid_argument = next(item for item in materialized if item.startswith("--cidfile="))
        Path(cid_argument.split("=", 1)[1]).write_text(
            container_id + "\n",
            encoding="ascii",
        )
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(materialized, timeout=1)
        raise OSError("synthetic runner failure")

    def fake_run(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        assert materialized == ("/usr/bin/docker", "rm", "--force", container_id)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(executor, "_run_bounded_process", fail_bounded_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cleanup=force_removed",
    ):
        executor._default_runner(("/usr/bin/docker", "run", "qualified-image"))

    assert len(calls) == 2


def test_default_runner_force_removes_named_container_before_cidfile_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    observed_name = ""

    def fail_bounded_run(command: Sequence[str], **_kwargs: Any) -> NoReturn:
        nonlocal observed_name
        materialized = tuple(command)
        calls.append(materialized)
        name_argument = next(item for item in materialized if item.startswith("--name="))
        observed_name = name_argument.split("=", 1)[1]
        assert re.fullmatch(r"alberta-matched-executor-[0-9a-f]{32}", observed_name)
        raise subprocess.TimeoutExpired(materialized, timeout=1)

    def fake_run(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        assert materialized == ("/usr/bin/docker", "rm", "--force", observed_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(executor, "_run_bounded_process", fail_bounded_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cleanup=force_removed_by_name",
    ):
        executor._default_runner(("/usr/bin/docker", "run", "qualified-image"))

    assert len(calls) == 2


def test_executor_cleanup_uses_exact_name_for_a_partial_cidfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "container.cid"
    cidfile.write_bytes(b"partial")
    container_name = "alberta-matched-executor-" + "b" * 32
    observed: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        observed.append(materialized)
        assert materialized == ("/usr/bin/docker", "rm", "--force", container_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cidfile contract failed after cleanup=force_removed_by_name",
    ):
        executor._cleanup_interrupted_container(  # noqa: SLF001
            ("/usr/bin/docker", "run"),
            cidfile,
            container_name,
        )
    assert len(observed) == 1


def test_executor_cleanup_wraps_cidfile_read_oserror_after_exact_name_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "container.cid"
    cidfile.write_bytes(b"b" * 64)
    container_name = "alberta-matched-executor-" + "c" * 32
    read_error = FileNotFoundError("injected cidfile disappearance")
    observed: list[tuple[str, ...]] = []

    def fail_read(*_args: Any, **_kwargs: Any) -> Any:
        raise read_error

    def cleanup(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        observed.append(materialized)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(executor, "_read_stable_file", fail_read)
    monkeypatch.setattr(subprocess, "run", cleanup)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cidfile contract failed after cleanup=force_removed_by_name",
    ) as caught:
        executor._cleanup_interrupted_container(  # noqa: SLF001
            ("/usr/bin/docker", "run"),
            cidfile,
            container_name,
        )
    assert caught.value.__cause__ is read_error
    assert observed == [("/usr/bin/docker", "rm", "--force", container_name)]


def test_default_runner_cleans_a_completed_nonzero_container_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    observed_name = ""

    def completed(command: Sequence[str], **_kwargs: Any) -> Any:
        nonlocal observed_name
        materialized = tuple(command)
        calls.append(materialized)
        observed_name = next(
            item.split("=", 1)[1]
            for item in materialized
            if item.startswith("--name=")
        )
        return executor.ProcessResult(125, b"", b"failed")

    def cleanup(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        calls.append(materialized)
        assert materialized == ("/usr/bin/docker", "rm", "--force", observed_name)
        return subprocess.CompletedProcess(materialized, 0)

    monkeypatch.setattr(executor, "_run_bounded_process", completed)
    monkeypatch.setattr(subprocess, "run", cleanup)
    result = executor._default_runner(
        ("/usr/bin/docker", "run", "qualified-image")
    )
    assert result.returncode == 125
    assert len(calls) == 2


@pytest.mark.parametrize(
    "fault",
    (
        FileNotFoundError("injected runner launch failure"),
        subprocess.TimeoutExpired(("docker", "run"), 1),
        executor._BoundedProcessOutputError(  # noqa: SLF001
            "injected runner output overflow"
        ),
    ),
    ids=("oserror", "timeout", "overflow"),
)
def test_injected_executor_runner_failures_use_the_public_error(
    fault: BaseException,
) -> None:
    def runner(_command: Sequence[str]) -> Any:
        raise fault

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="candidate runner failed",
    ) as caught:
        executor._runner_result(  # noqa: SLF001
            runner,
            ("docker", "run", "image"),
            "candidate",
        )
    assert caught.value.__cause__ is fault


def test_executor_cleanup_accepts_bounded_proof_that_name_is_already_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-executor-" + "d" * 32
    commands: list[tuple[str, ...]] = []

    def missing(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        commands.append(materialized)
        return subprocess.CompletedProcess(materialized, 1)

    def inspect(command: Sequence[str], **_kwargs: Any) -> Any:
        materialized = tuple(command)
        commands.append(materialized)
        return executor.ProcessResult(0, b"", b"")

    monkeypatch.setattr(subprocess, "run", missing)
    monkeypatch.setattr(executor, "_run_bounded_process", inspect)
    state = executor._cleanup_interrupted_container(  # noqa: SLF001
        ("/usr/bin/docker", "run"),
        tmp_path / "missing.cid",
        container_name,
    )
    assert state == "already_absent_by_name"
    assert commands == [
        ("/usr/bin/docker", "rm", "--force", container_name),
        (
            "/usr/bin/docker",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            f"--filter=name=^/{container_name}$",
        ),
    ]


def test_executor_cleanup_wraps_bounded_absence_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-executor-" + "f" * 32
    overflow = executor._BoundedProcessOutputError(  # noqa: SLF001
        "injected cleanup inspection overflow"
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )

    def inspect(*_args: Any, **_kwargs: Any) -> Any:
        raise overflow

    monkeypatch.setattr(executor, "_run_bounded_process", inspect)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="could not prove the exact name absent",
    ) as caught:
        executor._cleanup_interrupted_container(  # noqa: SLF001
            ("/usr/bin/docker", "run"),
            tmp_path / "missing.cid",
            container_name,
        )
    assert caught.value.__cause__ is overflow


@pytest.mark.parametrize(
    "caller_option",
    ("--name=caller-owned", "--cidfile=/tmp/caller-owned.cid"),
)
def test_default_runner_rejects_caller_owned_cleanup_identifiers(
    caller_option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def run(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("rejected commands must not run")

    monkeypatch.setattr(executor, "_run_bounded_process", run)
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="already contains a name or cidfile",
    ):
        executor._default_runner(
            ("/usr/bin/docker", "run", caller_option, "qualified-image")
        )
    assert called is False


def test_executor_cleanup_fails_if_the_exact_name_still_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-matched-executor-" + "e" * 32

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )
    monkeypatch.setattr(
        executor,
        "_run_bounded_process",
        lambda *_args, **_kwargs: executor.ProcessResult(
            0,
            b"f" * 64 + b"\n",
            b"",
        ),
    )
    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="did not remove or prove absent",
    ):
        executor._cleanup_interrupted_container(  # noqa: SLF001
            ("/usr/bin/docker", "run"),
            tmp_path / "missing.cid",
            container_name,
        )


@pytest.mark.parametrize(
    ("stream_name", "overflow"),
    (
        ("stdout", False),
        ("stdout", True),
        ("stderr", False),
        ("stderr", True),
    ),
    ids=("stdout-exact", "stdout-plus-one", "stderr-exact", "stderr-plus-one"),
)
def test_default_runner_enforces_active_per_stream_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    overflow: bool,
) -> None:
    stdout_limit = 37
    stderr_limit = 19
    monkeypatch.setattr(executor, "_MAX_RAW_ARCHIVE_BYTES", stdout_limit)
    monkeypatch.setattr(executor, "_MAX_PROCESS_STDERR_BYTES", stderr_limit)
    real_temporary_file = tempfile.TemporaryFile
    closed_sizes: dict[int, int] = {}
    created = 0

    class TrackingTemporaryFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal created
            self.index = created
            created += 1
            self.handle = real_temporary_file(*args, **kwargs)

        def __enter__(self) -> TrackingTemporaryFile:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def close(self) -> None:
            if not self.handle.closed:
                self.handle.flush()
                closed_sizes[self.index] = os.fstat(self.handle.fileno()).st_size
                self.handle.close()

    monkeypatch.setattr(tempfile, "TemporaryFile", TrackingTemporaryFile)
    selected_limit = stdout_limit if stream_name == "stdout" else stderr_limit
    descriptor = 1 if stream_name == "stdout" else 2
    output_size = selected_limit + int(overflow)
    command = (
        sys.executable,
        "-c",
        f"import os; os.write({descriptor}, b'x' * {output_size})",
    )

    if overflow:
        with pytest.raises(
            executor.ForagerMatchedExecutorError,
            match="output exceeded its byte bound",
        ):
            executor._default_runner(command)
    else:
        result = executor._default_runner(command)
        assert result.returncode == 0
        assert len(result.stdout if stream_name == "stdout" else result.stderr) == selected_limit

    selected_sink = 0 if stream_name == "stdout" else 1
    other_sink = 1 - selected_sink
    assert closed_sizes[selected_sink] == selected_limit
    assert closed_sizes[other_sink] == 0


def test_default_runner_drains_stdout_and_stderr_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout_size = 256 * 1024 + 17
    stderr_size = 192 * 1024 + 11
    monkeypatch.setattr(executor, "_MAX_RAW_ARCHIVE_BYTES", stdout_size)
    monkeypatch.setattr(executor, "_MAX_PROCESS_STDERR_BYTES", stderr_size)
    script = f"""
import os
import threading

def emit(descriptor, value, total):
    chunk = value * 8192
    while total:
        current = chunk[:total]
        os.write(descriptor, current)
        total -= len(current)

threads = (
    threading.Thread(target=emit, args=(1, b'o', {stdout_size})),
    threading.Thread(target=emit, args=(2, b'e', {stderr_size})),
)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""

    result = executor._default_runner((sys.executable, "-c", script))

    assert result.returncode == 0
    assert result.stdout == b"o" * stdout_size
    assert result.stderr == b"e" * stderr_size


def test_default_runner_output_overflow_force_removes_cidfile_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "b" * 64
    cidfile_payload = container_id + "\n"
    cleanup_log = tmp_path / "cleanup.log"
    runtime = tmp_path / "fake-runtime"
    runtime.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import os",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"cleanup_log = Path({cleanup_log.as_posix()!r})",
                "if sys.argv[1] == 'run':",
                "    cidfile = next(",
                "        item.split('=', 1)[1]",
                "        for item in sys.argv",
                "        if item.startswith('--cidfile=')",
                "    )",
                f"    Path(cidfile).write_text({cidfile_payload!r}, encoding='ascii')",
                "    os.write(1, b'x' * 9)",
                "    time.sleep(60)",
                "elif sys.argv[1:3] == ['rm', '--force']:",
                "    cleanup_log.write_text(' '.join(sys.argv[1:]), encoding='ascii')",
                "else:",
                "    raise SystemExit(2)",
                "",
            )
        ),
        encoding="utf-8",
    )
    runtime.chmod(0o755)
    monkeypatch.setattr(executor, "_MAX_RAW_ARCHIVE_BYTES", 8)
    monkeypatch.setattr(executor, "_MAX_PROCESS_STDERR_BYTES", 8)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="cleanup=force_removed",
    ):
        executor._default_runner((runtime.as_posix(), "run", "qualified-image"))

    assert cleanup_log.read_text(encoding="ascii") == f"rm --force {container_id}"


def test_bounded_process_timeout_kills_and_reaps_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    script = (
        "import os,time; "
        f"open({pid_path.as_posix()!r}, 'w', encoding='ascii').write(str(os.getpid())); "
        "time.sleep(60)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        executor._run_bounded_process(  # noqa: SLF001
            (sys.executable, "-c", script),
            timeout=1.0,
            maximum_stdout_bytes=16,
            maximum_stderr_bytes=16,
        )

    child_pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_default_runner_preserves_asymmetric_stream_limits() -> None:
    assert executor._MAX_RAW_ARCHIVE_BYTES == 512 * 1024 * 1024  # noqa: SLF001
    assert executor._MAX_PROCESS_STDERR_BYTES == 16 * 1024 * 1024  # noqa: SLF001


def test_execute_seed_rebinds_daemon_and_image_before_launch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    container_run_called = False

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        nonlocal container_run_called
        if len(command) >= 2 and command[1] == "version":
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(
                    {"Client": {"Version": "drifted-runtime"}}
                ),
                b"",
            )
        if len(command) >= 3 and command[1:3] == ("image", "inspect"):
            return executor.ProcessResult(
                0,
                executor.canonical_json_bytes(live.image_inspection),
                b"",
            )
        container_run_called = True
        return executor.ProcessResult(0, b"must-not-run", b"")

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="daemon version or image identity changed",
    ):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            tmp_path / "raw-runtime-drift.tar",
            live,
            runner=runner,
        )

    assert container_run_called is False


def test_execute_seed_keeps_raw_archive_opaque_and_emits_hash_only_artifacts(
    tmp_path: Path,
) -> None:
    plan, artifacts = _execute_fixture(tmp_path)
    raw_bytes = (tmp_path / "raw.tar").read_bytes()

    assert raw_bytes == b"opaque-ustar-never-opened-on-host"
    assert artifacts.score == 0.5
    assert artifacts.raw_artifact["container_export_sha256"] == hashlib.sha256(
        raw_bytes
    ).hexdigest()
    assert artifacts.trace_artifact["host_reward_array_access"] == "forbidden_not_performed"
    serialized = executor.canonical_json_bytes(artifacts.to_dict())
    assert b"reward_sum_float64" not in serialized
    assert b"final_unadjusted_ema" not in serialized
    assert b"rewards" not in serialized

    parsed = executor.parse_seed_artifact_bundle(artifacts.to_dict(), plan=plan)
    assert parsed == artifacts


def test_score_seed_archive_resumes_after_scorer_failure_without_rerunning_candidate(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-resume.tar"
    raw_bytes = b"opaque-candidate-output-preserved-for-resume"

    def failing_runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(3, b"", b"")
        return executor.ProcessResult(0, raw_bytes, b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="scoring failed"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            runner=failing_runner,
        )
    assert raw_path.read_bytes() == raw_bytes

    resumed_commands: list[tuple[str, ...]] = []

    def resumed_runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        resumed_commands.append(tuple(command))
        assert "score" in command
        return executor.ProcessResult(0, _scoring_output(plan, seed), b"")

    artifacts = executor.score_seed_archive(
        plan,
        "alberta_causal",
        seed,
        raw_path,
        live,
        expected_raw_archive_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        expected_raw_archive_size=len(raw_bytes),
        runner=resumed_runner,
    )

    assert len(resumed_commands) == 1
    assert artifacts.raw_artifact["container_export_sha256"] == hashlib.sha256(
        raw_bytes
    ).hexdigest()
    assert raw_path.read_bytes() == raw_bytes


def test_score_seed_archive_rejects_expected_binding_mismatch_before_oci(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-binding-mismatch.tar"
    raw_path.write_bytes(b"opaque")
    runner_called = False

    def runner(_command: Sequence[str]) -> executor.ProcessResult:
        nonlocal runner_called
        runner_called = True
        raise AssertionError("OCI must not run for a mismatched archive binding")

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="expected execution binding",
    ):
        executor.score_seed_archive(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            expected_raw_archive_sha256=_sha("different archive"),
            expected_raw_archive_size=raw_path.stat().st_size,
            runner=runner,
        )

    assert runner_called is False


def test_scorer_output_rejects_extra_reward_values_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    payload = executor.decode_strict_json(_scoring_output(plan, seed))
    payload["records"][0]["reward_values"] = [0.0]

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            return executor.ProcessResult(0, executor.canonical_json_bytes(payload), b"")
        return executor.ProcessResult(0, b"opaque", b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            tmp_path / "raw-extra.tar",
            live,
            runner=runner,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("horizon",), 499_712.0),
        (("seeds", 0), float(2_300_001)),
        (("records", 0, "seed"), float(2_300_001)),
        (("records", 0, "reward_shape", 0), 499_712.0),
        (("records", 0, "ema_sample_count"), 4_998.0),
        (("records", 0, "ema_tail_start_index"), 4_498.0),
        (("records", 0, "ema_tail_sample_count"), 500.0),
        (("records", 0, "ema_tail_sample_count"), True),
    ],
)
def test_scorer_output_rejects_numeric_aliases_for_integer_fields(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: float | bool,
) -> None:
    plan = _plan(tmp_path)
    seed = plan.protocol.active_seeds[0]
    payload = executor.decode_strict_json(_scoring_output(plan, seed))
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(executor.ForagerMatchedExecutorError, match="must be an integer"):
        executor._parse_scorer_output(
            executor.canonical_json_bytes(payload),
            plan=plan,
            candidate=plan.candidates[0],
            seed=seed,
        )


def test_execute_seed_rejects_raw_archive_mutation_during_scoring(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-mutation.tar"

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        if "score" in command:
            raw_path.unlink()
            raw_path.write_bytes(b"mutated-after-scoring-command-construction")
            return executor.ProcessResult(0, _scoring_output(plan, seed), b"")
        return executor.ProcessResult(0, b"original-opaque-ustar", b"")

    with pytest.raises(executor.ForagerMatchedExecutorError, match="changed during scoring"):
        executor.execute_seed(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            runner=runner,
        )


def test_score_seed_archive_rejects_rename_swap_even_when_path_is_restored(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    seed = plan.protocol.active_seeds[0]
    raw_path = tmp_path / "raw-swap.tar"
    replacement_path = tmp_path / "raw-swap-attacker.tar"
    held_path = tmp_path / "raw-swap-original-held.tar"
    original = b"opaque-original-archive"
    replacement = b"opaque-attacker-archive"
    assert len(original) == len(replacement)
    raw_path.write_bytes(original)
    replacement_path.write_bytes(replacement)
    expected_digest = hashlib.sha256(original).hexdigest()
    declared_digests: list[str] = []
    scorer_rejected = False
    swap_performed = False

    def runner(command: Sequence[str]) -> executor.ProcessResult:
        nonlocal scorer_rejected, swap_performed
        if len(command) >= 2 and command[1] == "version" and not swap_performed:
            os.replace(raw_path, held_path)
            os.replace(replacement_path, raw_path)
            swap_performed = True
        reinspection = _runtime_reinspection_result(command, live)
        if reinspection is not None:
            return reinspection
        assert "score" in command
        declared_digest = next(
            item.split("=", 1)[1]
            for item in command
            if item.startswith("--raw-archive-sha256=")
        )
        declared_digests.append(declared_digest)
        mounted_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        os.replace(raw_path, replacement_path)
        os.replace(held_path, raw_path)
        if mounted_digest != declared_digest:
            scorer_rejected = True
            return executor.ProcessResult(2, b"", b"raw archive digest mismatch")
        return executor.ProcessResult(0, _scoring_output(plan, seed), b"")

    artifacts: executor.SeedExecutionArtifacts | None = None
    with pytest.raises(executor.ForagerMatchedExecutorError, match="scoring failed"):
        artifacts = executor.score_seed_archive(
            plan,
            "alberta_causal",
            seed,
            raw_path,
            live,
            expected_raw_archive_sha256=expected_digest,
            expected_raw_archive_size=len(original),
            runner=runner,
        )

    assert artifacts is None
    assert scorer_rejected is True
    assert declared_digests == [expected_digest]
    assert raw_path.read_bytes() == original


def test_build_score_evidence_is_bridge_compatible_and_requires_external_resolver(
    tmp_path: Path,
) -> None:
    plan, first = _execute_fixture(tmp_path)
    records = []
    for position, seed in enumerate(plan.protocol.active_seeds):
        if position == 0:
            records.append(first)
            continue
        scorer_record = executor.decode_strict_json(_scoring_output(plan, seed))["records"][0]
        records.append(
            executor._artifact_mappings(
                plan=plan,
                candidate=plan.candidates[0],
                seed=seed,
                raw_archive_sha256=_sha(f"archive:{seed}"),
                raw_archive_size=1024,
                live_runtime=_runtime(tmp_path / f"runtime-{seed}", plan),
                scorer_record=scorer_record,
            )
        )
    # Evidence requires one live runtime identity for the complete candidate
    # block; replay the second hash-only artifact under the first identity.
    records[1] = executor.SeedExecutionArtifacts(
        candidate_id=records[1].candidate_id,
        seed=records[1].seed,
        score=records[1].score,
        live_runtime_identity_sha256=first.live_runtime_identity_sha256,
        raw_artifact=records[1].raw_artifact,
        trace_artifact=records[1].trace_artifact,
        scoring_record={
            **dict(records[1].scoring_record),
            "live_runtime_identity_sha256": first.live_runtime_identity_sha256,
        },
    )
    scores = executor.build_score_evidence(plan, {"alberta_causal": records})
    request = executor.build_verification_request(plan, scores)

    assert request.verification_subject_sha256 == evidence.matched_verification_subject_sha256(
        plan.protocol, scores
    )
    assert request.to_dict()["authentication_state"] == (
        "unresolved_external_verifier_required"
    )
    assert request.to_dict()["qualification_authority_boundary"][
        "endorsement_created"
    ] is False
    assert request.to_dict()["qualification_authority_boundary"][
        "trust_profile_created"
    ] is False
    assert request.to_dict()["qualification_authority_boundary"][
        "performance_claim"
    ] is False
    assert request.to_dict()["rng_parity_qualification_status"] == (
        "content_complete_external_executor_receipt_unverified"
    )
    assert request.to_dict()["qualification_promotion_authorized"] is False

    def resolver(
        subject: executor.VerificationRequest,
    ) -> evidence.AuthenticatedEvidenceBindings:
        return evidence.AuthenticatedEvidenceBindings(
            stage=subject.stage,
            protocol_sha256=subject.protocol_sha256,
            score_evidence_sha256=subject.score_evidence_sha256,
            source_manifest_sha256=subject.source_manifest_sha256,
            executor_manifest_sha256=subject.executor_manifest_sha256,
            qualification_manifest_sha256=subject.qualification_manifest_sha256,
            execution_closure_sha256=subject.execution_closure_sha256,
            trust_anchor_identity=subject.trust_anchor_identity,
            verification_subject_sha256=subject.verification_subject_sha256,
            verification_receipt_sha256=_sha("external-verifier-receipt"),
        )

    bindings = executor.resolve_authenticated_bindings(request, resolver)
    _frozen, validated = evidence.validate_score_evidence_against_protocol(
        plan.protocol,
        scores,
        authenticated_bindings=bindings,
        expected_candidate_ids=("alberta_causal",),
    )
    assert validated == scores


def test_verification_request_round_trip_loader_and_digest_domains(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    scores = executor.build_score_evidence(plan, artifacts)
    request = executor.build_verification_request(plan, scores)
    raw = request.canonical_bytes

    assert request.request_sha256 == hashlib.sha256(raw).hexdigest()
    assert request.request_sha256 != request.verification_subject_sha256
    assert executor.canonical_verification_request_bytes(request) == raw
    assert executor.canonical_verification_request_sha256(request) == request.request_sha256
    assert executor.parse_verification_request(raw) == request
    assert executor.parse_verification_request(raw.decode("ascii")) == request
    assert executor.parse_verification_request(request.to_dict()) == request

    path = tmp_path / "verification-request.json"
    path.write_bytes(raw)
    assert executor.load_verification_request(
        path,
        expected_request_sha256=request.request_sha256,
    ) == request

    with pytest.raises(executor.ForagerMatchedExecutorError, match="expected digest"):
        executor.load_verification_request(
            path,
            expected_request_sha256="0" * 64,
        )
    with pytest.raises(executor.ForagerMatchedExecutorError, match="canonical"):
        executor.parse_verification_request(json.dumps(request.to_dict(), indent=2))

    symlink = tmp_path / "verification-request-link.json"
    os.symlink(path, symlink)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="single-link regular"):
        executor.load_verification_request(
            symlink,
            expected_request_sha256=request.request_sha256,
        )


def test_score_and_verification_request_bind_exact_qualification_manifest_v2(
    tmp_path: Path,
) -> None:
    _payload, frozen, assets = _fixture(tmp_path)
    changed_qualification_sha256 = _sha("changed-request-qualification-manifest")
    plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("alberta_causal",),
    )
    changed_plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=changed_qualification_sha256,
        candidate_ids=("alberta_causal",),
    )
    scores = executor.build_score_evidence(
        plan,
        _complete_execution_artifacts(
            plan,
            _runtime(tmp_path / "original-runtime", plan),
        ),
    )
    changed_scores = executor.build_score_evidence(
        changed_plan,
        _complete_execution_artifacts(
            changed_plan,
            _runtime(tmp_path / "changed-runtime", changed_plan),
        ),
    )
    request = executor.build_verification_request(plan, scores)
    changed_request = executor.build_verification_request(changed_plan, changed_scores)

    assert scores.schema_version == "alberta.forager_matched_score_evidence.v2"
    assert scores.qualification_manifest_sha256 == _QUALIFICATION_MANIFEST_SHA256
    assert request.to_dict()["schema_version"] == (
        "alberta.forager_matched_verification_request.v2"
    )
    assert request.qualification_manifest_sha256 == _QUALIFICATION_MANIFEST_SHA256
    assert request.execution_closure_sha256 != changed_request.execution_closure_sha256
    assert request.verification_subject_sha256 != changed_request.verification_subject_sha256
    assert request.request_sha256 != changed_request.request_sha256

    mismatched_score_payload = scores.to_dict()
    mismatched_score_payload["qualification_manifest_sha256"] = changed_qualification_sha256
    unsigned_score_payload = copy.deepcopy(mismatched_score_payload)
    del unsigned_score_payload["payload_sha256"]
    mismatched_score_payload["payload_sha256"] = _canonical_sha(unsigned_score_payload)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="execution plan manifests"):
        executor.build_verification_request(plan, mismatched_score_payload)

    legacy_request = request.to_dict()
    legacy_request["schema_version"] = "alberta.forager_matched_verification_request.v1"
    with pytest.raises(executor.ForagerMatchedExecutorError, match="schema/authentication"):
        executor.parse_verification_request(legacy_request)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("authentication_state", "authenticated", "authentication boundary"),
        ("qualification_promotion_authorized", True, "authentication boundary"),
        ("qualification_manifest_sha256", "0" * 64, "subject"),
        ("score_evidence_sha256", "0" * 64, "subject"),
        (
            "rng_parity_qualification_status",
            "verified",
            "RNG parity qualification status",
        ),
    ],
)
def test_verification_request_rejects_boundary_and_subject_drift(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    request = executor.build_verification_request(
        plan,
        executor.build_score_evidence(plan, artifacts),
    )
    payload = request.to_dict()
    payload[field] = value

    with pytest.raises(executor.ForagerMatchedExecutorError, match=error):
        executor.parse_verification_request(payload)


def test_execution_receipt_index_exposes_exact_score_evidence_preimages(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    live = _runtime(tmp_path, plan)
    artifacts = _complete_execution_artifacts(plan, live)

    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    scores = executor.build_score_evidence(plan, artifacts)
    score_receipt_hashes = {
        candidate.candidate_id: candidate.execution_receipt_sha256
        for candidate in scores.candidate_scores
    }

    assert receipt_index.protocol_sha256 == plan.protocol.protocol_sha256
    assert receipt_index.plan_sha256 == plan.plan_sha256
    assert receipt_index.source_manifest_sha256 == plan.source_manifest_sha256
    assert receipt_index.executor_manifest_sha256 == plan.executor_manifest_sha256
    assert receipt_index.live_runtime_identity_sha256 == live.identity_sha256
    assert receipt_index.candidate_order == ("alberta_causal",)
    assert receipt_index.to_dict()["promotion_authorized"] is False
    assert receipt_index.payload_sha256 == _canonical_sha(receipt_index.unsigned_dict())
    for item in receipt_index.execution_receipts:
        assert item.execution_receipt_sha256 == _canonical_sha(
            dict(item.receipt_payload)
        )
        assert item.execution_receipt_sha256 == score_receipt_hashes[item.candidate_id]
        assert tuple(
            seed_artifact["seed"]
            for seed_artifact in item.receipt_payload["seed_artifacts"]
        ) == plan.protocol.active_seeds

    parsed = executor.parse_execution_receipt_index(
        receipt_index.canonical_bytes,
        plan=plan,
        artifacts=artifacts,
        expected_payload_sha256=receipt_index.payload_sha256,
    )
    assert parsed == receipt_index
    index_path = tmp_path / "execution-receipt-index.json"
    index_path.write_bytes(receipt_index.canonical_bytes)
    loaded = executor.load_execution_receipt_index(
        index_path,
        plan=plan,
        artifacts=artifacts,
        expected_payload_sha256=receipt_index.payload_sha256,
    )
    assert loaded == receipt_index


def test_execution_receipt_index_rejects_self_consistent_receipt_tamper(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    payload = receipt_index.to_dict()
    entry = payload["execution_receipts"][0]
    receipt_payload = entry["receipt_payload"]
    receipt_payload["seed_artifacts"][0]["raw_artifact_sha256"] = "0" * 64
    entry["execution_receipt_sha256"] = _canonical_sha(receipt_payload)
    _refresh_receipt_index_digest(payload)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="differs from exact plan/artifacts",
    ):
        executor.parse_execution_receipt_index(
            payload,
            plan=plan,
            artifacts=artifacts,
        )


def test_execution_receipt_index_direct_construction_is_strict(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    original = receipt_index.execution_receipts[0]

    extra_field_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    extra_field_payload["unexpected"] = False
    with pytest.raises(executor.ForagerMatchedExecutorError, match="keys differ"):
        executor.IndexedExecutionReceipt(
            candidate_id=original.candidate_id,
            execution_receipt_sha256=_canonical_sha(extra_field_payload),
            receipt_payload=extra_field_payload,
        )

    reordered_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    reordered_payload["seed_artifacts"].reverse()
    reordered_receipt = executor.IndexedExecutionReceipt(
        candidate_id=original.candidate_id,
        execution_receipt_sha256=_canonical_sha(reordered_payload),
        receipt_payload=reordered_payload,
    )
    unsigned = executor._execution_receipt_index_unsigned_dict(
        schema_version=receipt_index.schema_version,
        stage=receipt_index.stage,
        protocol_sha256=receipt_index.protocol_sha256,
        plan_sha256=receipt_index.plan_sha256,
        source_manifest_sha256=receipt_index.source_manifest_sha256,
        executor_manifest_sha256=receipt_index.executor_manifest_sha256,
        live_runtime_identity_sha256=receipt_index.live_runtime_identity_sha256,
        active_seeds=receipt_index.active_seeds,
        horizon=receipt_index.horizon,
        candidate_order=receipt_index.candidate_order,
        execution_receipts=(reordered_receipt,),
    )
    with pytest.raises(executor.ForagerMatchedExecutorError, match="seed order drift"):
        executor.MatchedExecutionReceiptIndex(
            schema_version=receipt_index.schema_version,
            stage=receipt_index.stage,
            protocol_sha256=receipt_index.protocol_sha256,
            plan_sha256=receipt_index.plan_sha256,
            source_manifest_sha256=receipt_index.source_manifest_sha256,
            executor_manifest_sha256=receipt_index.executor_manifest_sha256,
            live_runtime_identity_sha256=receipt_index.live_runtime_identity_sha256,
            active_seeds=receipt_index.active_seeds,
            horizon=receipt_index.horizon,
            candidate_order=receipt_index.candidate_order,
            execution_receipts=(reordered_receipt,),
            payload_sha256=_canonical_sha(unsigned),
        )


def test_indexed_execution_receipt_freezes_caller_owned_payload(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    original = executor.build_execution_receipt_index(
        plan,
        artifacts,
    ).execution_receipts[0]
    caller_payload = copy.deepcopy(original.to_dict()["receipt_payload"])
    receipt = executor.IndexedExecutionReceipt(
        candidate_id=original.candidate_id,
        execution_receipt_sha256=original.execution_receipt_sha256,
        receipt_payload=caller_payload,
    )

    caller_payload["seed_artifacts"][0]["raw_artifact_sha256"] = "0" * 64

    assert receipt.to_dict() == original.to_dict()


def test_execution_receipt_index_binds_candidate_order(tmp_path: Path) -> None:
    plan = _two_candidate_plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    receipt_index = executor.build_execution_receipt_index(plan, artifacts)
    payload = receipt_index.to_dict()
    payload["candidate_order"].reverse()
    payload["execution_receipts"].reverse()
    _refresh_receipt_index_digest(payload)

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="differs from exact plan/artifacts",
    ):
        executor.parse_execution_receipt_index(
            payload,
            plan=plan,
            artifacts=artifacts,
        )


def test_execution_receipt_index_rejects_live_runtime_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    artifacts = _complete_execution_artifacts(plan, _runtime(tmp_path, plan))
    records = artifacts["alberta_causal"]
    different_runtime_sha256 = _sha("different-live-runtime")
    second = records[1]
    records[1] = executor.SeedExecutionArtifacts(
        candidate_id=second.candidate_id,
        seed=second.seed,
        score=second.score,
        live_runtime_identity_sha256=different_runtime_sha256,
        raw_artifact=second.raw_artifact,
        trace_artifact=second.trace_artifact,
        scoring_record={
            **dict(second.scoring_record),
            "live_runtime_identity_sha256": different_runtime_sha256,
        },
    )

    with pytest.raises(
        executor.ForagerMatchedExecutorError,
        match="multiple live runtime identities",
    ):
        executor.build_execution_receipt_index(plan, artifacts)


def test_score_evidence_requires_one_live_runtime_identity_across_candidates(
    tmp_path: Path,
) -> None:
    panel_plan = _two_candidate_plan(tmp_path)
    first_runtime = _runtime(tmp_path / "runtime-a", panel_plan)
    second_runtime = replace(first_runtime, executable_sha256=_sha("different-runtime"))
    artifact_blocks: dict[str, list[executor.SeedExecutionArtifacts]] = {}
    for candidate, runtime in zip(
        panel_plan.candidates,
        (first_runtime, second_runtime),
        strict=True,
    ):
        records: list[executor.SeedExecutionArtifacts] = []
        for seed in panel_plan.protocol.active_seeds:
            scorer_record = executor.decode_strict_json(
                _scoring_output(panel_plan, seed)
            )["records"][0]
            records.append(
                executor._artifact_mappings(
                    plan=panel_plan,
                    candidate=candidate,
                    seed=seed,
                    raw_archive_sha256=_sha(f"archive:{candidate.candidate.candidate_id}:{seed}"),
                    raw_archive_size=1024,
                    live_runtime=runtime,
                    scorer_record=scorer_record,
                )
            )
        artifact_blocks[candidate.candidate.candidate_id] = records

    with pytest.raises(executor.ForagerMatchedExecutorError, match="candidate panel used multiple"):
        executor.build_score_evidence(panel_plan, artifact_blocks)


def test_plan_and_artifact_loaders_require_external_digests(tmp_path: Path) -> None:
    payload, frozen, assets = _fixture(tmp_path)
    del payload
    plan = executor.build_execution_plan(
        frozen,
        assets,
        qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        candidate_ids=("alberta_causal",),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(plan.canonical_bytes)
    assert executor.load_execution_plan(
        plan_path,
        protocol=frozen,
        assets=assets,
        expected_plan_sha256=plan.plan_sha256,
        expected_qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
    ).plan_sha256 == plan.plan_sha256
    with pytest.raises(executor.ForagerMatchedExecutorError, match="external expected"):
        executor.load_execution_plan(
            plan_path,
            protocol=frozen,
            assets=assets,
            expected_plan_sha256="0" * 64,
            expected_qualification_manifest_sha256=_QUALIFICATION_MANIFEST_SHA256,
        )


def test_source_inventory_rejects_symlink_and_inode_alias(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = root / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    (root / "link.py").symlink_to(target)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="symlink"):
        executor.source_inventory(root)
    (root / "link.py").unlink()
    os.link(target, root / "hard.py")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="hardlink"):
        executor.source_inventory(root)


def test_source_inventory_rejects_excessive_directory_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        (root / f"empty-{index}").mkdir()
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 2)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 10)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="directory bound"):
        executor.source_inventory(root)


def test_source_inventory_rejects_excessive_recursion_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    current = root
    for index in range(3):
        current = current / f"level-{index}"
        current.mkdir()
    monkeypatch.setattr(executor, "_MAX_SOURCE_DEPTH", 2)
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 10)

    with pytest.raises(executor.ForagerMatchedExecutorError, match="recursion-depth bound"):
        executor.source_inventory(root)


def test_source_inventory_bounds_single_directory_before_eager_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        (root / f"entry-{index}.py").write_bytes(b"pass\n")
    monkeypatch.setattr(executor, "_MAX_SOURCE_FILES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_DIRECTORIES", 10)
    monkeypatch.setattr(executor, "_MAX_SOURCE_ENTRIES", 2)

    def forbid_listdir(_path: object) -> NoReturn:
        raise AssertionError("source inventory must accumulate through bounded scandir")

    monkeypatch.setattr(os, "listdir", forbid_listdir)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="total-entry bound"):
        executor.source_inventory(root)


def test_stable_file_and_source_inventory_reject_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    with pytest.raises(executor.ForagerMatchedExecutorError, match="regular file"):
        executor._read_stable_file(fifo, "FIFO", maximum=1024)

    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "source.fifo")
    with pytest.raises(executor.ForagerMatchedExecutorError, match="non-regular"):
        executor.source_inventory(source)


def test_container_safe_extract_opens_root_once_and_binds_raw_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "raw.tar"
    payloads = {
        "stdout.log": b"",
        "stderr.log": b"warning archived, not emitted by helper",
        "results/rtu/data/7.npz": b"synthetic-not-a-protected-result",
    }
    with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, BytesIO(raw))
    extraction_root = tmp_path / "extract"
    monkeypatch.setattr(container_helper, "EXTRACT_ROOT", extraction_root)

    container_helper._safe_extract(
        archive_path,
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )

    assert (extraction_root / "results/rtu/data/7.npz").read_bytes() == payloads[
        "results/rtu/data/7.npz"
    ]


def test_container_ustar_caps_include_all_header_payload_and_record_padding() -> None:
    member_sizes = [0, 1, 511, 512, 513]
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for index, size in enumerate(member_sizes):
            info = tarfile.TarInfo(f"member-{index}")
            info.size = size
            archive.addfile(info, BytesIO(b"x" * size))

    assert len(stream.getvalue()) == container_helper._ustar_stream_size(member_sizes)
    assert container_helper._maximum_ustar_stream_size(
        payload_bytes=container_helper.MAX_OUTPUT_PAYLOAD_BYTES,
        member_count=container_helper.MAX_MEMBERS,
    ) <= container_helper.MAX_RAW_ARCHIVE_BYTES
    assert container_helper.MAX_RAW_ARCHIVE_BYTES == executor._MAX_RAW_ARCHIVE_BYTES

    descriptor = os.open("/dev/null", os.O_RDONLY)
    oversized = container_helper._OutputMember(
        name="oversized",
        descriptor=descriptor,
        size=container_helper.MAX_RAW_ARCHIVE_BYTES,
        identity=(),
    )
    with pytest.raises(container_helper.ContainerError, match="USTAR stream exceeds"):
        container_helper._write_ustar([oversized])
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_container_ppo_dispatch_omits_max_steps_and_archives_workload_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    entrypoint = source / "src/rtu_ppo.py"
    entrypoint.parent.mkdir(parents=True)
    relative_input = source / "runtime-input.txt"
    relative_input.write_text("snapshot-value", encoding="utf-8")
    entrypoint.write_text(
        f"""from pathlib import Path
import sys
args = sys.argv[1:]
assert '--max_steps' not in args
seed = args[args.index('-i') + 1]
save = Path(args[args.index('--save_path') + 1])
target = save / 'rtu' / 'data' / f'{{seed}}.npz'
target.parent.mkdir(parents=True)
Path({relative_input.as_posix()!r}).write_text('mutated-host-mount')
relative_value = Path('runtime-input.txt').read_text()
target.write_bytes((' '.join(args) + ' relative=' + relative_value).encode())
print('captured stdout')
sys.stderr.write('captured upstream warning')
""",
        encoding="utf-8",
    )
    configuration = tmp_path / "configuration.json"
    configuration.write_bytes(b"{}")
    output_root = tmp_path / "output"
    monkeypatch.setattr(container_helper, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(container_helper, "EXECUTION_SOURCE_ROOT", tmp_path / "snapshot")
    monkeypatch.setattr(container_helper, "EXECUTION_CONFIG", tmp_path / "snapshot.json")
    captured: dict[str, bytes] = {}

    def capture_members(members: list[container_helper._OutputMember]) -> None:
        try:
            for member in members:
                captured[member.name] = os.pread(member.descriptor, member.size, 0)
        finally:
            for member in members:
                os.close(member.descriptor)

    monkeypatch.setattr(container_helper, "_write_ustar", capture_members)
    container_helper._run(
        Namespace(
            source_root=source.resolve().as_posix(),
            entrypoint=entrypoint.resolve().as_posix(),
            python_import_root=(source / "src").resolve().as_posix(),
            config=configuration.resolve().as_posix(),
            python=Path(sys.executable).resolve().as_posix(),
            source_inventory_sha256=executor.source_inventory_sha256(source),
            configuration_sha256=hashlib.sha256(b"{}").hexdigest(),
            invocation_style="official_foragax_ppo_frozen_updates_v1",
            result_root="results/rtu",
            seed=7,
            horizon=executor.MATCHED_HORIZON,
        )
    )

    assert b"--max_steps" not in captured["results/rtu/data/7.npz"]
    assert b"relative=snapshot-value" in captured["results/rtu/data/7.npz"]
    assert captured["stdout.log"] == b"captured stdout\n"
    assert captured["stderr.log"] == b"captured upstream warning"
