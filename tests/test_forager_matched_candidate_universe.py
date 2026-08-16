"""Contract tests for
:mod:`alberta_framework.benchmarks.forager_matched_candidate_universe`.

The candidate universe is the frozen provenance record explaining how the
matched-current panel was assembled from historical development screens —
screens that used consumed seeds and nonmatched horizons and therefore can
never be scientific evidence themselves.  Tests pin the descriptor's exact
content digest, require every screened arm's inclusion or exclusion to be
accounted for, keep every screen classified nonpromoting (no superiority or
SOTA claims), require the registered panel to match the open-protocol
builder, and exercise the bound-JSON source verifier fail-closed (tampered
bytes, symlinks, ambiguous or non-finite JSON, duplicate rank rows, hash
drift).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_open_protocol as open_protocol

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Frozen content digest of the canonical universe descriptor bytes; any
# change to the descriptor is deliberate and must re-pin this constant.
_EXPECTED_DIGEST = "6a9315cb996fe5698e4c1580d30da9b0524e9875ce085d1399bb975cc5b510a8"


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rehash_payload(payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = _canonical_sha256(unsigned)


@pytest.mark.unit
def test_descriptor_is_canonical_detached_and_content_addressed() -> None:
    raw = universe.canonical_matched_current_candidate_universe_bytes()

    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DIGEST
    assert universe.MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256 == _EXPECTED_DIGEST
    assert universe.parse_matched_current_candidate_universe_artifact(raw) == (
        universe.matched_current_candidate_universe_descriptor()
    )
    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="does not match the frozen digest",
    ):
        universe.parse_matched_current_candidate_universe_artifact(raw + b"\n")

    detached = universe.matched_current_candidate_universe_descriptor()
    detached["limitations"].append("mutation")
    assert "mutation" not in universe.matched_current_candidate_universe_descriptor()[
        "limitations"
    ]


@pytest.mark.unit
def test_descriptor_accounts_for_all_complete_nonpromoting_sources() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    screens = {item["screen_id"]: item for item in descriptor["source_screens"]}
    local_sources = {
        item["screen_id"]: item
        for item in descriptor["local_candidate_generation_sources"]
    }
    arms = descriptor["screened_arms"]

    assert set(screens) == {"dqn_common_control_v3", "stateful_corrected_v4"}
    assert set(local_sources) == {
        "horde_fov_tuning_v2",
        "rtu_schema23_screening_v1",
    }
    assert len(arms) == 29
    for screen in screens.values():
        assert screen["classification"] == "open_development_nonpromoting"
        assert screen["evidence_use"] == "candidate_generation_provenance_only"
        assert screen["seeds"] == [2_000_001, 2_000_002]
        assert screen["scientific_promotion_allowed"] is False
        assert screen["superiority_claim_allowed"] is False
        assert screen["sota_claim_allowed"] is False
    for source in local_sources.values():
        assert source["classification"] == "open_development_nonpromoting"
        assert source["evidence_use"] == "candidate_generation_provenance_only"
        assert source["scientific_promotion_allowed"] is False
        assert source["superiority_claim_allowed"] is False
        assert source["sota_claim_allowed"] is False
        assert source["historical_source_authorizes_current_execution"] is False

    dqn = [item for item in arms if item["screen_id"] == "dqn_common_control_v3"]
    stateful = [item for item in arms if item["screen_id"] == "stateful_corrected_v4"]
    horde = [item for item in arms if item["screen_id"] == "horde_fov_tuning_v2"]
    local_rtu = [
        item for item in arms if item["screen_id"] == "rtu_schema23_screening_v1"
    ]
    assert sorted(item["open_development_rank"] for item in dqn) == list(range(1, 12))
    assert sorted(item["open_development_rank"] for item in stateful) == list(range(1, 9))
    assert sorted(item["open_development_rank"] for item in horde) == list(range(1, 5))
    assert sorted(item["open_development_rank"] for item in local_rtu) == list(range(1, 7))
    assert all(item["screen_result_is_scientific_evidence"] is False for item in arms)
    assert all(item["screen_result_transfers_to_derived_candidate"] is False for item in arms)


@pytest.mark.unit
def test_registered_panel_matches_open_protocol_and_explains_rng_roles() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    assert tuple(
        item["candidate_id"] for item in descriptor["registered_panel"]
    ) == open_protocol.MATCHED_CURRENT_CANDIDATE_IDS
    panel = {item["candidate_id"]: item for item in descriptor["registered_panel"]}

    assert set(panel) == set(open_protocol.MATCHED_CURRENT_CANDIDATE_IDS)
    assert len(panel) == 23
    alberta_ids = set(open_protocol.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS)
    assert len(alberta_ids) == 14
    assert all(
        panel[candidate_id]["selection_group"] == "alberta"
        for candidate_id in alberta_ids
    )
    assert all(
        panel[candidate_id]["analysis_role"] == "inferential"
        for candidate_id in alberta_ids
    )
    assert all(
        panel[candidate_id]["pairing_eligible"] is True
        for candidate_id in alberta_ids
    )
    external_ids = set(open_protocol.MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS)
    assert len(external_ids) == 7
    assert all(
        panel[candidate_id]["selection_group"] == "external"
        and panel[candidate_id]["analysis_role"] == "inferential"
        and panel[candidate_id]["pairing_eligible"] is True
        for candidate_id in external_ids
    )
    assert panel["external_dqn_crelu"]["implementation_relationship"] == (
        "matched_task_common_control_crelu_with_bound_horizon_transform"
    )
    assert panel["external_dqn_plain"]["implementation_relationship"] == (
        "official_repository_fov9_config_with_bound_horizon_and_diagnostic_transforms"
    )
    for candidate_id in ("isolated_ppo", "isolated_rtu"):
        assert panel[candidate_id]["analysis_role"] == "inferential"
        assert panel[candidate_id]["pairing_eligible"] is True
        assert panel[candidate_id]["rng_relationship"] == (
            "isolated_agent_and_environment_streams"
        )
        assert panel[candidate_id]["implementation_relationship"] == (
            "reviewed_rng_isolation_derivative_requiring_independent_qualification"
        )

    assert panel["exact_ppo"]["analysis_role"] == "descriptive_only"
    assert panel["exact_ppo"]["pairing_eligible"] is False
    assert panel["exact_ppo"]["rng_relationship"] == "shared_agent_and_environment_rng"
    assert panel["search_oracle"]["analysis_role"] == "descriptive_only"
    assert panel["search_oracle"]["observation_access"] == (
        "privileged_global_objects_and_known_reward_priority"
    )
    assert descriptor["unregistered_references"] == [
        {
            "reference_id": "exact_upstream_rtu_ppo",
            "source_screen_id": "stateful_corrected_v4",
            "source_configuration": "configs/PPO-RTU_LN_128_1_relu.json",
            "registered": False,
            "reason": descriptor["unregistered_references"][0]["reason"],
        }
    ]


@pytest.mark.unit
def test_screen_dispositions_cover_every_inclusion_and_exclusion() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    arms = {
        (item["screen_id"], item["configuration"]): item
        for item in descriptor["screened_arms"]
    }

    dqn_ln = arms[("dqn_common_control_v3", "configs/DQN_LN-common-control.json")]
    dqn_crelu = arms[
        ("dqn_common_control_v3", "configs/DQN_CReLU-common-control.json")
    ]
    dqn_plain = arms[("dqn_common_control_v3", "configs/DQN-common-control.json")]
    assert dqn_ln["registered_candidate_ids"] == ["external_dqn_ln"]
    assert dqn_ln["disposition"] == "registered_horizon_transform"
    assert dqn_crelu["registered_candidate_ids"] == ["external_dqn_crelu"]
    assert dqn_crelu["disposition"] == "registered_horizon_transform"
    assert dqn_plain["registered_candidate_ids"] == ["external_dqn_plain"]
    assert dqn_plain["disposition"] == (
        "registered_horizon_and_diagnostic_transform"
    )
    assert sum(
        item["disposition"] == "excluded_lower_rank_same_family"
        for item in descriptor["screened_arms"]
        if item["screen_id"] == "dqn_common_control_v3"
    ) == 8

    rtu = arms[
        ("stateful_corrected_v4", "configs/PPO-RTU_LN_128_1_relu.json")
    ]
    ppo = arms[("stateful_corrected_v4", "configs/PPO_2048_relu.json")]
    redo = arms[("stateful_corrected_v4", "configs/DQN_ReDo_PostLNScore.json")]
    drqn = arms[("stateful_corrected_v4", "configs/DRQN-paper-v1.json")]
    assert rtu["registered_candidate_ids"] == ["isolated_rtu"]
    assert rtu["disposition"] == "registered_rng_isolated_derivative"
    assert ppo["registered_candidate_ids"] == ["isolated_ppo", "exact_ppo"]
    assert ppo["disposition"] == (
        "registered_exact_orientation_and_rng_isolated_derivative"
    )
    assert redo["registered_candidate_ids"] == ["external_dqn_redo"]
    assert drqn["registered_candidate_ids"] == ["external_drqn_paper"]

    horde = {
        item["configuration"]: item
        for item in descriptor["screened_arms"]
        if item["screen_id"] == "horde_fov_tuning_v2"
    }
    assert set(horde) == {
        "variants/default/config",
        "variants/eps05/config",
        "variants/recurrent64/config",
        "variants/step3e3/config",
    }
    assert all(
        item["disposition"] == "registered_matched_worker_transform"
        and len(item["registered_candidate_ids"]) == 1
        for item in horde.values()
    )

    local_rtu = {
        item["configuration"]: item
        for item in descriptor["screened_arms"]
        if item["screen_id"] == "rtu_schema23_screening_v1"
    }
    selected = local_rtu["variants/rtu_h08_taylor/config"]
    assert selected["registered_candidate_ids"] == ["alberta_rtu_h08_taylor"]
    assert selected["disposition"] == "registered_preselected_family_representative"
    assert sum(
        item["disposition"] == "excluded_by_frozen_family_selection"
        for item in local_rtu.values()
    ) == 5


@pytest.mark.unit
def test_local_configuration_and_historical_provenance_are_exactly_bound() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    arms = {
        (item["screen_id"], item["configuration"]): item
        for item in descriptor["screened_arms"]
    }
    default = arms[("horde_fov_tuning_v2", "variants/default/config")]
    recurrent = arms[("horde_fov_tuning_v2", "variants/recurrent64/config")]
    local_rtu = arms[
        ("rtu_schema23_screening_v1", "variants/rtu_h08_taylor/config")
    ]
    assert default["configuration_sha256"] == (
        "8194d90048677a88bfa6954d78cdbf66e86f94d5607e4fec4d961c83846267c7"
    )
    assert default["worker_configuration_sha256"] == (
        "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a"
    )
    assert recurrent["historical_descriptor_sha256"] == (
        "55e20d903803a0ce312019209232925a4c19505918a3698583ce7b4ad648780c"
    )
    assert local_rtu["configuration_sha256"] == (
        "f1571d16ed0ff39a8383336b95420f912402e48bee95f063d7984c56b776d4d7"
    )
    assert local_rtu["worker_configuration_sha256"] == (
        "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
    )
    assert local_rtu["historical_descriptor_sha256"] == (
        "ead4297b65fab08408625e4a842c71a3f03f64323579d6e1787cc082014d1be8"
    )
    worker_fingerprints = open_protocol.matched_current_alberta_configuration_fingerprints()
    for arm in descriptor["screened_arms"]:
        if arm["screen_id"] not in {
            "horde_fov_tuning_v2",
            "rtu_schema23_screening_v1",
        } or not arm["registered_candidate_ids"]:
            continue
        candidate_id = arm["registered_candidate_ids"][0]
        assert arm["worker_configuration_sha256"] == worker_fingerprints[candidate_id]

    panel = {item["candidate_id"]: item for item in descriptor["registered_panel"]}
    assert panel["alberta_rtu_h08_taylor"]["source_screen_id"] == (
        "rtu_schema23_screening_v1"
    )
    assert panel["isolated_rtu"]["source_screen_id"] == "stateful_corrected_v4"
    assert panel["alberta_rtu_h08_taylor"]["implementation_relationship"] != (
        panel["isolated_rtu"]["implementation_relationship"]
    )


@pytest.mark.unit
def test_claim_scope_is_narrow_and_never_universal() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    scope = descriptor["scope"]
    boundaries = descriptor["claim_boundaries"]

    assert descriptor["schema_version"] == "alberta.forager_matched_candidate_universe.v2"
    assert scope["registered_panel_complete"] is True
    assert scope["registered_candidate_count"] == 23
    assert scope["alberta_inferential_candidate_count"] == 14
    assert scope["external_inferential_candidate_count"] == 7
    assert scope["descriptive_candidate_count"] == 2
    assert scope["bound_json_file_count"] == 14
    assert scope["research_literature_exhaustive"] is False
    assert scope["scientific_promotion_allowed"] is False
    assert scope["superiority_claim_allowed"] is False
    assert scope["sota_claim_allowed"] is False
    assert boundaries["descriptor_alone_supports_performance_claim"] is False
    assert boundaries["screens_support_matched_candidate_ranking"] is False
    assert boundaries["screens_support_derived_rng_isolated_performance"] is False
    assert boundaries["historical_sources_authorize_current_execution"] is False
    assert boundaries["eventual_claim_requires_sealed_matched_evaluation"] is True
    assert boundaries["registered_panel_ranking_identified_by_design"] is False
    assert "contrast-specific" in boundaries["narrowest_permitted_eventual_scope"]
    assert "best among" not in boundaries["narrowest_permitted_eventual_scope"]
    assert "best member of the registered panel" in boundaries["forbidden_scope"]
    assert "universal state of the art" in boundaries["forbidden_scope"]


@pytest.mark.integration
def test_exact_json_only_screen_bindings_verify() -> None:
    paths = universe.matched_current_screening_json_paths()
    assert len(paths) == 14
    assert all(path.endswith(".json") for path in paths)
    assert all(".npz" not in path and "/reward-traces/" not in path for path in paths)

    verification = universe.verify_matched_current_candidate_universe_sources(
        _REPOSITORY_ROOT
    )
    assert verification.candidate_universe_sha256 == _EXPECTED_DIGEST
    assert verification.verified_json_paths == paths
    assert verification.to_dict()["reward_array_files_read"] == 0


@pytest.mark.integration
def test_source_verifier_rejects_a_tampered_bound_json(tmp_path: Path) -> None:
    for relative_path in universe.matched_current_screening_json_paths():
        source = _REPOSITORY_ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    aggregate = (
        tmp_path
        / "outputs/forager/fov_baseline_screening_cpu_v3_execution/aggregate.json"
    )
    aggregate.write_bytes(
        aggregate.read_bytes().replace(
            b'"sota_claim_allowed":false', b'"sota_claim_allowed":true'
        )
    )

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="bound JSON digest mismatch",
    ):
        universe.verify_matched_current_candidate_universe_sources(tmp_path)


@pytest.mark.integration
def test_source_verifier_rejects_symlinked_bindings(tmp_path: Path) -> None:
    paths = universe.matched_current_screening_json_paths()
    for relative_path in paths:
        source = _REPOSITORY_ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == paths[-1]:
            destination.symlink_to(source)
        else:
            shutil.copyfile(source, destination)

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="may not contain symlinks",
    ):
        universe.verify_matched_current_candidate_universe_sources(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"outer":{"value":1,"value":2}}', "duplicate JSON object key"),
        (b'{"value":1e999}', "non-finite JSON number"),
    ),
)
def test_bound_json_reader_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path,
    raw: bytes,
    message: str,
) -> None:
    path = tmp_path / "bound.json"
    path.write_bytes(raw)

    with pytest.raises(universe.ForagerMatchedCandidateUniverseError, match=message):
        universe._read_bound_json(tmp_path, path.name, hashlib.sha256(raw).hexdigest())


@pytest.mark.unit
def test_external_screen_validator_rejects_duplicate_rank_rows() -> None:
    binding = universe._SCREEN_BINDINGS[0]
    protocol = json.loads((_REPOSITORY_ROOT / binding.protocol_path).read_bytes())
    screen_plan = json.loads((_REPOSITORY_ROOT / binding.screen_plan_path).read_bytes())
    aggregate = json.loads((_REPOSITORY_ROOT / binding.aggregate_path).read_bytes())
    aggregate["eligible_ranking"] = [
        copy.deepcopy(aggregate["eligible_ranking"][0])
        for _ in aggregate["eligible_ranking"]
    ]

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="must contain every rank exactly once in order",
    ):
        universe._verify_one_screen(binding, protocol, screen_plan, aggregate)


@pytest.mark.unit
def test_rtu_source_validator_rejects_float_alias_for_integer_aperture() -> None:
    binding = next(
        item
        for item in universe._LOCAL_CANDIDATE_GENERATION_BINDINGS
        if item.screen_id == "rtu_schema23_screening_v1"
    )
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in binding.artifacts
    }
    artifacts["protocol"]["task"]["aperture_size"] = 9.0

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match=r"task\.aperture_size must be an integer",
    ):
        universe._verify_rtu_candidate_generation(binding, artifacts)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("artifact_role", "path", "match"),
    [
        ("protocol", ("task", "steps"), r"task\.steps must be an integer"),
        ("protocol", ("task", "seeds", 0), r"task\.seeds\[0\] must be an integer"),
        ("matrix", ("steps",), r"matrix\.steps must be an integer"),
        ("matrix", ("seeds", 0), r"matrix\.seeds\[0\] must be an integer"),
        ("matrix", ("tuning_seeds", 0), r"matrix\.tuning_seeds\[0\] must be an integer"),
        (
            "matrix",
            ("evaluation_seeds", 0),
            r"matrix\.evaluation_seeds\[0\] must be an integer",
        ),
        (
            "protocol",
            ("future_evaluation", "seeds", 0),
            r"future_evaluation\.seeds\[0\] must be an integer",
        ),
        (
            "protocol",
            ("selection_rule", "advance_count"),
            r"selection_rule\.advance_count must be an integer",
        ),
    ],
)
def test_rtu_source_validator_rejects_float_aliases_across_full_integer_contract(
    artifact_role: str, path: tuple[object, ...], match: str
) -> None:
    """Reviewer-found gap on PR #461: the exact-int policy applied to
    aperture_size alone left every adjacent integer attestation (task/matrix
    steps and seeds, future_evaluation seeds, selection_rule advance_count)
    still fail-open to a numeric alias via bare equality."""
    binding = next(
        item
        for item in universe._LOCAL_CANDIDATE_GENERATION_BINDINGS
        if item.screen_id == "rtu_schema23_screening_v1"
    )
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in binding.artifacts
    }
    target: Any = artifacts[artifact_role]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = float(target[path[-1]])

    with pytest.raises(universe.ForagerMatchedCandidateUniverseError, match=match):
        universe._verify_rtu_candidate_generation(binding, artifacts)


@pytest.mark.unit
def test_external_screen_validator_reconciles_snapshot_configuration_hashes() -> None:
    binding = universe._SCREEN_BINDINGS[0]
    protocol = json.loads((_REPOSITORY_ROOT / binding.protocol_path).read_bytes())
    screen_plan = json.loads((_REPOSITORY_ROOT / binding.screen_plan_path).read_bytes())
    aggregate = json.loads((_REPOSITORY_ROOT / binding.aggregate_path).read_bytes())
    snapshot = screen_plan["input_snapshot"]
    record = next(
        item
        for item in snapshot["files"]
        if item["path"] == "base/configs/DQN_LN-common-control.json"
    )
    record["sha256"] = "0" * 64
    snapshot_inventory = {
        "directories": snapshot["directories"],
        "files": snapshot["files"],
    }
    canonical_inventory = json.dumps(
        snapshot_inventory,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    snapshot["inventory_sha256"] = hashlib.sha256(
        canonical_inventory + b"\n"
    ).hexdigest()

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="input snapshot configuration hashes are invalid",
    ):
        universe._verify_one_screen(binding, protocol, screen_plan, aggregate)


@pytest.mark.unit
def test_local_screen_validator_rejects_duplicate_rank_rows() -> None:
    binding = universe._LOCAL_CANDIDATE_GENERATION_BINDINGS[0]
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in binding.artifacts
    }
    report = copy.deepcopy(artifacts["report"])
    ranking = report["selection_results"]["groups"]["policy"]["ranked_variants"]
    report["selection_results"]["groups"]["policy"]["ranked_variants"] = [
        copy.deepcopy(ranking[0]) for _ in ranking
    ]
    _rehash_payload(report)

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="must contain every rank exactly once in order",
    ):
        universe._verify_local_execution_pair(
            binding,
            artifacts["execution_manifest"],
            report,
            schema_version="2.0",
            selection_group="policy",
            implementation_kind="alberta_horde_ac",
            selected_variant_id="step3e3",
        )


@pytest.mark.unit
def test_internal_descriptor_validator_rejects_registered_panel_reordering() -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    descriptor["registered_panel"] = list(reversed(descriptor["registered_panel"]))

    with pytest.raises(AssertionError, match="exact frozen order"):
        universe._validate_internal_descriptor(descriptor)


@pytest.mark.unit
def test_local_screen_validator_recomputes_normalized_matrix_hash() -> None:
    binding = universe._LOCAL_CANDIDATE_GENERATION_BINDINGS[0]
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in binding.artifacts
    }
    execution_manifest = copy.deepcopy(artifacts["execution_manifest"])
    report = copy.deepcopy(artifacts["report"])
    execution_manifest["matrix_config"]["unhashed_tamper"] = "must fail closed"
    report["matrix_config"] = copy.deepcopy(execution_manifest["matrix_config"])
    _rehash_payload(execution_manifest)
    report["execution_manifest_sha256"] = execution_manifest["payload_sha256"]
    _rehash_payload(report)

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="normalized matrix configuration hash does not verify",
    ):
        universe._verify_local_execution_pair(
            binding,
            execution_manifest,
            report,
            schema_version="2.0",
            selection_group="policy",
            implementation_kind="alberta_horde_ac",
            selected_variant_id="step3e3",
        )


@pytest.mark.unit
def test_local_screen_validator_recomputes_manifest_payload_hash() -> None:
    binding = universe._LOCAL_CANDIDATE_GENERATION_BINDINGS[0]
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in binding.artifacts
    }
    execution_manifest = copy.deepcopy(artifacts["execution_manifest"])
    report = copy.deepcopy(artifacts["report"])
    execution_manifest["payload_sha256"] = "0" * 64
    report["execution_manifest_sha256"] = "0" * 64
    _rehash_payload(report)

    with pytest.raises(
        universe.ForagerMatchedCandidateUniverseError,
        match="execution_manifest payload_sha256 does not verify",
    ):
        universe._verify_local_execution_pair(
            binding,
            execution_manifest,
            report,
            schema_version="2.0",
            selection_group="policy",
            implementation_kind="alberta_horde_ac",
            selected_variant_id="step3e3",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("config", "canonical configuration hash does not verify"),
        ("descriptor", "canonical variant descriptor hash does not verify"),
    ),
)
def test_local_screen_validator_recomputes_variant_content_hashes(
    mutation: str,
    message: str,
) -> None:
    original_binding = universe._LOCAL_CANDIDATE_GENERATION_BINDINGS[0]
    artifacts = {
        artifact.role: json.loads((_REPOSITORY_ROOT / artifact.path).read_bytes())
        for artifact in original_binding.artifacts
    }
    execution_manifest = copy.deepcopy(artifacts["execution_manifest"])
    report = copy.deepcopy(artifacts["report"])
    matrix_variant = execution_manifest["matrix_config"]["variants"]["default"]
    if mutation == "config":
        matrix_variant["config"]["unhashed_tamper"] = "must fail closed"
        report["variants"]["default"]["config"] = copy.deepcopy(
            matrix_variant["config"]
        )
    else:
        matrix_variant["unhashed_tamper"] = "must fail closed"
    report["matrix_config"] = copy.deepcopy(execution_manifest["matrix_config"])
    normalized_matrix_sha256 = _canonical_sha256(execution_manifest["matrix_config"])
    binding = replace(
        original_binding,
        normalized_matrix_sha256=normalized_matrix_sha256,
    )
    execution_manifest["matrix_config_sha256"] = normalized_matrix_sha256
    report["matrix_config_sha256"] = normalized_matrix_sha256
    _rehash_payload(execution_manifest)
    report["execution_manifest_sha256"] = execution_manifest["payload_sha256"]
    _rehash_payload(report)

    with pytest.raises(universe.ForagerMatchedCandidateUniverseError, match=message):
        universe._verify_local_execution_pair(
            binding,
            execution_manifest,
            report,
            schema_version="2.0",
            selection_group="policy",
            implementation_kind="alberta_horde_ac",
            selected_variant_id="step3e3",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selection_group", "bogus"),
        ("analysis_role", "descriptive_only"),
        ("pairing_eligible", False),
    ),
)
def test_internal_descriptor_validator_rejects_external_group_role_drift(
    field: str,
    value: str | bool,
) -> None:
    descriptor = universe.matched_current_candidate_universe_descriptor()
    candidate = next(
        item
        for item in descriptor["registered_panel"]
        if item["candidate_id"] == "external_dqn_ln"
    )
    candidate[field] = value

    with pytest.raises(AssertionError, match="selection group|role/pairing"):
        universe._validate_internal_descriptor(descriptor)
