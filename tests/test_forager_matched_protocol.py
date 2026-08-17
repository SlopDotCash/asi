"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_protocol`.

The module under test is the strict, reward-agnostic parser for the
matched-current Forager comparison protocol: it validates scientific intent
and execution identity (candidate strata, seed contracts, runtime bindings,
the frozen statistics plan) without executing agents or reading results.
This suite exercises that contract fail-closed: mutations of a valid payload
— digest tampering, stage confusion, seed-set overlap, relabelled strata,
non-canonical or hostile JSON — must be rejected, while the one legal
transition (``open_tuning`` -> ``sealed_evaluation`` via a bound selection
result) must replay deterministically.

The synthetic-payload builders here (``_candidate``, ``_payload``,
``_sealed_payload``) double as a fixture library: sibling suites import this
module as ``protocol_fixtures`` (``test_forager_matched_evidence``,
``test_forager_matched_executor``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_protocol as matched

# Mirrors the executor's MATCHED_HORIZON: a ~500k-step horizon that power-of-two
# rollout lengths divide exactly (499_712 = 244 * 2_048 = 122 * 4_096); the
# parser requires ``rollout_steps`` to divide the horizon with no remainder.
HORIZON = 499_712
# Placeholder 64-hex digests — the parser validates digest format and
# cross-references between sections, never the referenced content.
IMAGE_SHA = "1" * 64
PROFILE_SHA = "2" * 64
TASK_SHA = "3" * 64
SCHEDULE_SHA = "4" * 64
# Qualification trust-anchor identity: names the authority that issued each
# candidate's capability receipt.  The parser requires every candidate's
# runtime binding to cite the same anchor as the protocol's runtime block.
TRUST_ANCHOR = "alberta_protocol_qualification_anchor_v1"


def _capability_descriptor_sha256(candidate: dict[str, Any]) -> str:
    runtime = candidate["runtime_binding"]
    payload = {
        "agent_rng": candidate["agent_rng"],
        "candidate_id": candidate["candidate_id"],
        "configuration": candidate["configuration"],
        "entrypoint_family": candidate["entrypoint_family"],
        "environment_rng": candidate["environment_rng"],
        "execution_semantics": candidate["execution_semantics"],
        "implementation_kind": candidate["implementation_kind"],
        "observation_access": candidate["observation_access"],
        "pairing": candidate["pairing"],
        "resources": candidate["resources"],
        "runtime_identity": {
            "image_sha256": runtime["image_sha256"],
            "runtime_profile_sha256": runtime["runtime_profile_sha256"],
            "task_identity_sha256": runtime["task_identity_sha256"],
        },
        "schema_version": "alberta.forager_candidate_capability_descriptor.v1",
        "seed_contract": candidate["seed_contract"],
        "selection_group": candidate["selection_group"],
        "source": candidate["source"],
        "stratum": candidate["stratum"],
    }
    raw = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _candidate(
    candidate_id: str,
    *,
    stratum: str,
    implementation_kind: str,
    entrypoint_family: str,
    selection_group: str,
    eligible: bool,
    analysis_role: str,
    exclusion_reasons: list[str],
    agent_rng_identity: str = "isolated_agent_rng_v1",
    environment_key_shared: bool = False,
    environment_rng_identity: str = "dedicated_environment_split_chain_v1",
    access_mode: str = "partial_observation",
    privileged_fields: list[str] | None = None,
    aperture_size: int = 9,
    rollout_steps: int | None = None,
) -> dict[str, Any]:
    num_rollouts = HORIZON // rollout_steps if rollout_steps is not None else None
    candidate: dict[str, Any] = {
        "candidate_id": candidate_id,
        "selection_group": selection_group,
        "stratum": stratum,
        "implementation_kind": implementation_kind,
        "entrypoint_family": entrypoint_family,
        "source": {
            "provenance_kind": (
                "reviewed_snapshot" if stratum == "alberta_learning" else "git_tree"
            ),
            "repository": "https://github.com/example/forager-agent",
            "base_commit": "a" * 40,
            "tree_git_sha1": None if stratum == "alberta_learning" else "5" * 40,
            "archive_sha256": "6" * 64,
            "inventory_sha256": "a" * 64,
            "snapshot_descriptor_sha256": "b" * 64 if stratum == "alberta_learning" else None,
        },
        "configuration": {
            "original_path": f"configs/{candidate_id}.json",
            "original_sha256": "7" * 64,
            "derived_sha256": "7" * 64,
            "allowed_transforms": [],
        },
        "seed_contract": {
            "transport": "adapter_injected",
            "offset": 0,
            "effective_seed_expression": "active_seed",
            "effective_seed_proof_sha256": "8" * 64,
        },
        "execution_semantics": {
            "rollout_steps": rollout_steps,
            "num_rollouts": num_rollouts,
            "update_semantics": (
                "rollout_minibatch_updates"
                if rollout_steps is not None
                else "environment_step_counted"
            ),
        },
        "observation_access": {
            "access_mode": access_mode,
            "observation_type": "color",
            "aperture_size": aperture_size,
            "privileged_fields": privileged_fields or [],
        },
        "environment_rng": {
            "identity": environment_rng_identity,
            "schedule_sha256": SCHEDULE_SHA,
        },
        "agent_rng": {
            "identity": agent_rng_identity,
            "environment_key_shared": environment_key_shared,
        },
        "runtime_binding": {
            "image_sha256": IMAGE_SHA,
            "runtime_profile_sha256": PROFILE_SHA,
            "task_identity_sha256": TASK_SHA,
            "qualified_capability_descriptor_sha256": "0" * 64,
            "capability_qualification_receipt_sha256": hashlib.sha256(
                f"capability:{candidate_id}".encode()
            ).hexdigest(),
            "qualification_trust_anchor_identity": TRUST_ANCHOR,
        },
        "resources": {
            "parameter_count": 1_024,
            "optimizer_update_count": HORIZON // 4,
            "replay_capacity_transitions": 10_000,
            "recurrent_state_elements": 0,
        },
        "pairing": {
            "analysis_role": analysis_role,
            "eligible": eligible,
            "exclusion_reasons": exclusion_reasons,
        },
    }
    candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        _capability_descriptor_sha256(candidate)
    )
    return candidate


def _slot(selection_group: str, rank: int = 1) -> dict[str, Any]:
    return {"selection_group": selection_group, "rank": rank}


def _selection_result_payload(
    open_protocol: matched.ForagerMatchedProtocol,
) -> dict[str, Any]:
    return {
        "schema_version": matched.FORAGER_MATCHED_SELECTION_RESULT_SCHEMA_VERSION,
        "open_protocol_sha256": open_protocol.protocol_sha256,
        "selection_plan_sha256": open_protocol.selection_plan.plan_sha256,
        "tuning_seeds": list(open_protocol.tuning_seeds),
        "ranked_groups": [
            {
                "selection_group": group.selection_group,
                "ranked_candidate_ids": list(group.candidate_ids),
                "ranking_evidence_sha256": hashlib.sha256(
                    f"ranking:{group.selection_group}".encode()
                ).hexdigest(),
            }
            for group in open_protocol.selection_plan.groups
        ],
    }


def _sealed_payload(
    open_payload: dict[str, Any],
    selection_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    open_protocol = matched.parse_forager_matched_protocol(open_payload)
    result = (
        _selection_result_payload(open_protocol)
        if selection_result is None
        else copy.deepcopy(selection_result)
    )
    sealed = copy.deepcopy(open_payload)
    sealed["stage"] = "sealed_evaluation"
    sealed["active_seeds"] = list(open_protocol.evaluation_seeds)
    advance_by_group = {
        group.selection_group: group.advance_count for group in open_protocol.selection_plan.groups
    }
    sealed["selection_outcome"] = {
        "status": "resolved",
        "open_protocol_sha256": open_protocol.protocol_sha256,
        "selection_result_sha256": matched.canonical_selection_result_sha256(result),
        "resolved_slots": [
            {
                "selection_group": group["selection_group"],
                "rank": rank,
                "candidate_id": group["ranked_candidate_ids"][rank - 1],
            }
            for group in result["ranked_groups"]
            for rank in range(1, advance_by_group[group["selection_group"]] + 1)
        ],
    }
    return sealed, result


def test_seal_protocol_builds_the_exact_validated_transition() -> None:
    open_payload = _payload()
    open_protocol = matched.parse_forager_matched_protocol(open_payload)
    result_payload = _selection_result_payload(open_protocol)

    sealed = matched.seal_forager_matched_protocol(open_protocol, result_payload)
    expected_payload, _ = _sealed_payload(open_payload, result_payload)

    assert sealed.to_dict() == expected_payload
    validation = matched.validate_sealed_protocol_transition(
        open_protocol,
        sealed,
        result_payload,
    )
    assert validation.open_protocol_sha256 == open_protocol.protocol_sha256
    assert sealed.active_seeds == open_protocol.evaluation_seeds


def test_seal_protocol_rejects_unbound_or_incomplete_selection() -> None:
    open_protocol = matched.parse_forager_matched_protocol(_payload())
    result_payload = _selection_result_payload(open_protocol)
    result_payload["open_protocol_sha256"] = "f" * 64
    with pytest.raises(matched.ForagerMatchedProtocolError, match="canonical open protocol"):
        matched.seal_forager_matched_protocol(open_protocol, result_payload)

    result_payload = _selection_result_payload(open_protocol)
    result_payload["ranked_groups"][0]["ranked_candidate_ids"][0] = "external_dqn"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="rank every"):
        matched.seal_forager_matched_protocol(open_protocol, result_payload)


def _payload(*, stage: str = "open_tuning") -> dict[str, Any]:
    candidates = [
        _candidate(
            "alberta_causal",
            stratum="alberta_learning",
            implementation_kind="alberta_causal_map",
            entrypoint_family="alberta_single_seed_worker",
            selection_group="alberta",
            eligible=True,
            analysis_role="inferential",
            exclusion_reasons=[],
        ),
        _candidate(
            "external_dqn",
            stratum="external_learning",
            implementation_kind="upstream_dqn",
            entrypoint_family="continuing_main",
            selection_group="external",
            eligible=True,
            analysis_role="inferential",
            exclusion_reasons=[],
        ),
        _candidate(
            "isolated_rtu",
            stratum="external_learning",
            implementation_kind="upstream_rtu_ppo_isolated_rng",
            entrypoint_family="rtu_ppo",
            selection_group="recurrent",
            eligible=True,
            analysis_role="inferential",
            exclusion_reasons=[],
            rollout_steps=128,
        ),
        _candidate(
            "exact_ppo",
            stratum="external_learning",
            implementation_kind="upstream_ppo",
            entrypoint_family="rtu_ppo",
            selection_group="exact_orientation",
            eligible=False,
            analysis_role="descriptive_only",
            exclusion_reasons=["shared_agent_environment_rng"],
            agent_rng_identity="shared_agent_environment_rng_v1",
            environment_key_shared=True,
            environment_rng_identity="shared_agent_environment_rng_v1",
            rollout_steps=2_048,
        ),
        _candidate(
            "search_oracle",
            stratum="privileged_context",
            implementation_kind="upstream_search_oracle",
            entrypoint_family="continuing_main",
            selection_group="privileged",
            eligible=False,
            analysis_role="descriptive_only",
            exclusion_reasons=["privileged_observation_access"],
            access_mode="privileged_reward_grid",
            privileged_fields=["global_objects", "reward_grid"],
            aperture_size=-1,
        ),
        _candidate(
            "paper_dqn",
            stratum="historical_orientation",
            implementation_kind="historical_dqn",
            entrypoint_family="historical_legacy",
            selection_group="historical",
            eligible=False,
            analysis_role="descriptive_only",
            exclusion_reasons=["historical_runtime_mismatch"],
            access_mode="historical_legacy",
        ),
    ]
    tuning_seeds = [2_300_001, 2_300_002]
    evaluation_seeds = [2_200_001, 2_200_002]
    payload: dict[str, Any] = {
        "schema_version": matched.FORAGER_MATCHED_PROTOCOL_SCHEMA_VERSION,
        "stage": "open_tuning",
        "task": {
            "task_id": "forager_fov9_current",
            "preset": "field_of_view",
            "environment_id": "ForagaxTwoBiomeLarge-v1",
            "foragax_distribution": "continual-foragax",
            "foragax_version": "0.55.0",
            "observation_type": "color",
            "aperture_size": 9,
            "task_identity_sha256": TASK_SHA,
            "environment_rng_schedule_sha256": SCHEDULE_SHA,
        },
        "horizon": HORIZON,
        "tuning_seeds": tuning_seeds,
        "evaluation_seeds": evaluation_seeds,
        "active_seeds": tuning_seeds,
        "candidates": candidates,
        "selection_plan": {
            "metric": "fov_last_10pct_ema_auc",
            "metric_implementation_sha256": "c" * 64,
            "candidate_universe_sha256": "b" * 64,
            "direction": "maximize",
            "statistic": "conservative_ci_endpoint",
            "statistic_implementation_sha256": "d" * 64,
            "confidence": 0.95,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 2_300_000,
            "bootstrap_rng_identity": "numpy_generator_pcg64",
            "bootstrap_rng_implementation_sha256": "e" * 64,
            "resampling_unit": "candidate_seed_block",
            "quantile_method": "linear",
            "bootstrap_interval": "two_sided_equal_tail",
            "conservative_endpoint": "lower",
            "endpoint_quantile": "(1-confidence)/2",
            "tie_break": "candidate_id_ascending",
            "groups": [
                {
                    "selection_group": "alberta",
                    "candidate_ids": ["alberta_causal"],
                    "advance_count": 1,
                },
                {
                    "selection_group": "external",
                    "candidate_ids": ["external_dqn"],
                    "advance_count": 1,
                },
                {
                    "selection_group": "recurrent",
                    "candidate_ids": ["isolated_rtu"],
                    "advance_count": 1,
                },
            ],
        },
        "selection_outcome": {
            "status": "pending",
            "open_protocol_sha256": None,
            "selection_result_sha256": None,
            "resolved_slots": [],
        },
        "analysis_plan": {
            "metric": "fov_last_10pct_ema_auc",
            "metric_implementation_sha256": "c" * 64,
            "metric_direction": "maximize",
            "primary": {
                "method": "paired_percentile_bootstrap_lower_bound",
                "resamples": 10_000,
                "seed": 2_400_000,
                "confidence": 0.95,
                "primary_margin": 0.0,
                "rng_algorithm": "PCG64",
                "quantile_method": "linear",
                "implementation_sha256": "f" * 64,
                "gate": "lower_bound_strictly_greater_than_margin",
            },
            "secondary": {
                "method": "paired_sign_flip",
                "monte_carlo_resamples": 100_000,
                "seed": 2_400_001,
                "exact_max_pairs": 20,
                "rng_algorithm": "PCG64",
                "implementation_sha256": "0" * 64,
                "alternative": "greater",
                "multiplicity_method": "holm",
                "familywise_alpha": 0.05,
            },
        },
        "evaluation_panel": {
            "selection_slots": [
                _slot("alberta"),
                _slot("external"),
                _slot("recurrent"),
            ],
            "fixed_descriptive_candidate_ids": ["exact_ppo", "search_oracle"],
            "alberta_primary_slot": _slot("alberta"),
            "primary_nonprivileged_external_baseline_slot": _slot("external"),
            "require_complete_blocks": True,
            "pairing_failure_policy": "fail_closed",
        },
        "primary_hypothesis": {
            "hypothesis_id": "causal_vs_external",
            "intervention_slot": _slot("alberta"),
            "comparator_slot": _slot("external"),
            "estimand": "paired_mean_difference",
            "method": "paired_percentile_bootstrap_lower_bound",
            "alternative": "greater",
            "difference_order": "intervention_minus_comparator",
            "paired": True,
        },
        "secondary_hypotheses": [
            {
                "hypothesis_id": "rtu_vs_external",
                "intervention_slot": _slot("recurrent"),
                "comparator_slot": _slot("external"),
                "estimand": "paired_mean_difference",
                "method": "paired_sign_flip",
                "alternative": "greater",
                "difference_order": "intervention_minus_comparator",
                "paired": True,
            }
        ],
        "multiplicity_policy": {
            "method": "holm",
            "alpha": 0.05,
            "hypothesis_ids": ["rtu_vs_external"],
            "primary_excluded": True,
        },
        "privileged_context": {
            "candidate_ids": ["search_oracle"],
            "analysis_role": "descriptive_only",
            "selection_eligible": False,
            "pairing_eligible": False,
        },
        "historical_orientation": {
            "candidate_ids": ["paper_dqn"],
            "analysis_role": "descriptive_only",
            "selection_eligible": False,
            "pairing_eligible": False,
        },
        "runtime": {
            "executor_kind": "oci",
            "image_sha256": IMAGE_SHA,
            "runtime_profile_sha256": PROFILE_SHA,
            "executor_qualification_receipt_sha256": "9" * 64,
            "qualification_trust_anchor_identity": TRUST_ANCHOR,
            "source_mount_mode": "read_only_content_addressed_mount",
            "default_prng": "threefry2x32",
            "threefry_partitionable": True,
            "platform": "cpu",
            "sandbox": {
                "network": "none",
                "root_filesystem": "read_only",
                "capabilities": "all_dropped",
                "no_new_privileges": True,
                "container_user": "65532:65532",
                "host_devices": [],
                "writable_tmpfs_only": True,
            },
        },
    }
    if stage == "sealed_evaluation":
        payload, _ = _sealed_payload(payload)
    return payload


def _candidate_by_id(payload: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in payload["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return cast(dict[str, Any], candidate)
    raise AssertionError(candidate_id)


def test_parse_normalizes_freezes_and_hashes_protocol() -> None:
    protocol = matched.parse_forager_matched_protocol(_payload())

    assert protocol.horizon == 499_712
    assert protocol.stage == "open_tuning"
    assert protocol.active_seeds == protocol.tuning_seeds
    assert protocol.candidate_index["isolated_rtu"].execution_semantics.num_rollouts == 3_904
    assert protocol.to_dict() == _payload()
    assert protocol.protocol_sha256 == hashlib.sha256(protocol.canonical_bytes).hexdigest()
    assert protocol.protocol_sha256 == matched.canonical_json_sha256(protocol)
    assert protocol.canonical_bytes == matched.canonical_json_bytes(protocol.to_dict())
    assert b"timestamp" not in protocol.canonical_bytes

    with pytest.raises(FrozenInstanceError):
        protocol.horizon = 512_000  # type: ignore[misc]
    with pytest.raises(TypeError):
        protocol.candidate_index["new"] = protocol.candidates[0]  # type: ignore[index]

    invalid = replace(protocol, horizon=0)
    with pytest.raises(matched.ForagerMatchedProtocolError, match="horizon"):
        matched.canonical_json_bytes(invalid)


def test_normalized_protocol_round_trips_and_preserves_order() -> None:
    payload = _payload()
    normalized = matched.normalize_forager_matched_protocol(payload)
    reparsed = matched.parse_forager_matched_protocol(normalized)

    assert reparsed.to_dict() == normalized
    assert [item.candidate_id for item in reparsed.candidates] == [
        item["candidate_id"] for item in payload["candidates"]
    ]
    assert [item.hypothesis_id for item in reparsed.secondary_hypotheses] == ["rtu_vs_external"]


def test_parser_accepts_raw_strict_json_and_rejects_raw_duplicates() -> None:
    raw = json.dumps(_payload(), separators=(",", ":"))
    assert matched.parse_forager_matched_protocol(raw).to_dict() == _payload()

    with pytest.raises(matched.ForagerMatchedProtocolError, match="duplicate JSON"):
        matched.parse_forager_matched_protocol('{"schema_version":"a","schema_version":"b"}')


def test_sealed_evaluation_uses_only_evaluation_seed_set() -> None:
    payload = _payload(stage="sealed_evaluation")
    protocol = matched.parse_forager_matched_protocol(payload)

    assert protocol.stage == "sealed_evaluation"
    assert protocol.active_seeds == (2_200_001, 2_200_002)
    assert protocol.selection_outcome.status == "resolved"


def test_open_plan_uses_pending_outcome_and_stage_invariant_slots() -> None:
    protocol = matched.parse_forager_matched_protocol(_payload())

    assert protocol.selection_outcome.status == "pending"
    assert protocol.selection_outcome.open_protocol_sha256 is None
    assert protocol.selection_outcome.selection_result_sha256 is None
    assert protocol.selection_outcome.resolved_slots == ()
    assert protocol.selection_plan.slots == protocol.evaluation_panel.selection_slots
    assert [slot.to_dict() for slot in protocol.selection_plan.slots] == [
        _slot("alberta"),
        _slot("external"),
        _slot("recurrent"),
    ]


def test_analysis_plan_exactly_freezes_statistics_v2_methods() -> None:
    protocol = matched.parse_forager_matched_protocol(_payload())

    primary = protocol.analysis_plan.primary
    secondary = protocol.analysis_plan.secondary
    assert primary.method == "paired_percentile_bootstrap_lower_bound"
    assert primary.rng_algorithm == "PCG64"
    assert primary.quantile_method == "linear"
    assert primary.gate == "lower_bound_strictly_greater_than_margin"
    assert primary.primary_margin == 0.0
    assert secondary.method == "paired_sign_flip"
    assert secondary.exact_max_pairs == 20
    assert secondary.rng_algorithm == "PCG64"
    assert secondary.multiplicity_method == "holm"
    assert secondary.familywise_alpha == protocol.multiplicity_policy.alpha
    assert protocol.primary_hypothesis.method == primary.method
    assert all(
        hypothesis.method == secondary.method for hypothesis in protocol.secondary_hypotheses
    )

    selection = protocol.selection_plan
    assert selection.bootstrap_interval == "two_sided_equal_tail"
    assert selection.conservative_endpoint == "lower"
    assert selection.endpoint_quantile == "(1-confidence)/2"

    for path, value in (
        (("primary", "rng_algorithm"), "MT19937"),
        (("primary", "quantile_method"), "nearest"),
        (("primary", "primary_margin"), -0.0 - 0.1),
        (("secondary", "exact_max_pairs"), 21),
        (("secondary", "multiplicity_method"), "bonferroni"),
    ):
        payload = _payload()
        payload["analysis_plan"][path[0]][path[1]] = value
        with pytest.raises(matched.ForagerMatchedProtocolError):
            matched.parse_forager_matched_protocol(payload)

    for field, value in (
        ("bootstrap_interval", "one_sided"),
        ("conservative_endpoint", "upper"),
        ("endpoint_quantile", "1-confidence"),
    ):
        payload = _payload()
        payload["selection_plan"][field] = value
        with pytest.raises(matched.ForagerMatchedProtocolError):
            matched.parse_forager_matched_protocol(payload)


def test_sealed_transition_replays_canonical_result_and_exact_resolution() -> None:
    open_payload = _payload()
    open_protocol = matched.parse_forager_matched_protocol(open_payload)
    sealed_payload, result_payload = _sealed_payload(open_payload)

    validation = matched.validate_sealed_protocol_transition(
        open_protocol,
        sealed_payload,
        result_payload,
        matched.canonical_selection_result_sha256(result_payload),
    )

    assert validation.open_protocol_sha256 == open_protocol.protocol_sha256
    assert validation.evaluation_candidate_ids == (
        "alberta_causal",
        "external_dqn",
        "isolated_rtu",
        "exact_ppo",
        "search_oracle",
    )
    assert validation.primary_intervention_candidate_id == "alberta_causal"
    assert validation.primary_comparator_candidate_id == "external_dqn"
    assert validation.resolved_hypotheses[1].intervention_candidate_id == "isolated_rtu"


@pytest.mark.parametrize("mutation", ["metric", "configuration", "evaluation_seeds"])
def test_sealed_transition_rejects_every_nonstage_mutation(mutation: str) -> None:
    open_payload = _payload()
    sealed_payload, result_payload = _sealed_payload(open_payload)
    if mutation == "metric":
        sealed_payload["selection_plan"]["metric"] = "posthoc_metric"
        sealed_payload["analysis_plan"]["metric"] = "posthoc_metric"
    elif mutation == "configuration":
        candidate = _candidate_by_id(sealed_payload, "external_dqn")
        configuration = candidate["configuration"]
        configuration["original_sha256"] = "f" * 64
        configuration["derived_sha256"] = "f" * 64
        candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
            _capability_descriptor_sha256(candidate)
        )
    else:
        sealed_payload["evaluation_seeds"] = [2_000_000_000, 2_000_000_001]
        sealed_payload["active_seeds"] = list(sealed_payload["evaluation_seeds"])

    with pytest.raises(matched.ForagerMatchedProtocolError, match="changed a field"):
        matched.validate_sealed_protocol_transition(open_payload, sealed_payload, result_payload)


def test_transition_rejects_open_and_selection_result_digest_tampering() -> None:
    open_payload = _payload()
    sealed_payload, result_payload = _sealed_payload(open_payload)
    sealed_payload["selection_outcome"]["open_protocol_sha256"] = "f" * 64
    with pytest.raises(matched.ForagerMatchedProtocolError, match="canonical open"):
        matched.validate_sealed_protocol_transition(open_payload, sealed_payload, result_payload)

    sealed_payload, result_payload = _sealed_payload(open_payload)
    with pytest.raises(matched.ForagerMatchedProtocolError, match="supplied canonical digest"):
        matched.validate_sealed_protocol_transition(
            open_payload,
            sealed_payload,
            result_payload,
            "f" * 64,
        )

    sealed_payload, result_payload = _sealed_payload(open_payload)
    result_payload["open_protocol_sha256"] = "f" * 64
    sealed_payload["selection_outcome"]["selection_result_sha256"] = (
        matched.canonical_selection_result_sha256(result_payload)
    )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="not bound"):
        matched.validate_sealed_protocol_transition(open_payload, sealed_payload, result_payload)


def test_selection_result_can_resolve_a_nontrivial_group_winner() -> None:
    open_payload = _payload()
    alternative = _candidate(
        "external_alt",
        stratum="external_learning",
        implementation_kind="upstream_dqn_alternative",
        entrypoint_family="continuing_main_alternative",
        selection_group="external",
        eligible=True,
        analysis_role="inferential",
        exclusion_reasons=[],
    )
    open_payload["candidates"].insert(2, alternative)
    open_payload["selection_plan"]["groups"][1]["candidate_ids"].append("external_alt")
    open_protocol = matched.parse_forager_matched_protocol(open_payload)
    result_payload = _selection_result_payload(open_protocol)
    result_payload["ranked_groups"][1]["ranked_candidate_ids"] = [
        "external_alt",
        "external_dqn",
    ]
    sealed_payload, result_payload = _sealed_payload(open_payload, result_payload)

    validation = matched.validate_sealed_protocol_transition(
        open_payload, sealed_payload, result_payload
    )

    assert validation.primary_comparator_candidate_id == "external_alt"
    assert validation.evaluation_candidate_ids[1] == "external_alt"

    sealed_payload["selection_outcome"]["resolved_slots"][1]["candidate_id"] = "external_dqn"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="do not match"):
        matched.validate_sealed_protocol_transition(open_payload, sealed_payload, result_payload)


def test_selection_result_canonicalization_revalidates_dataclass_and_full_rankings() -> None:
    open_payload = _payload()
    open_protocol = matched.parse_forager_matched_protocol(open_payload)
    result_payload = _selection_result_payload(open_protocol)
    result = matched.parse_forager_matched_selection_result(result_payload)
    assert result.selection_result_sha256 == matched.canonical_selection_result_sha256(
        result_payload
    )

    invalid = replace(result, tuning_seeds=(2_300_001, 2_300_001))
    with pytest.raises(matched.ForagerMatchedProtocolError, match="duplicate seeds"):
        matched.canonical_selection_result_bytes(invalid)

    sealed_payload, _ = _sealed_payload(open_payload, result_payload)
    result_payload["ranked_groups"][1]["ranked_candidate_ids"] = ["search_oracle"]
    sealed_payload["selection_outcome"]["selection_result_sha256"] = (
        matched.canonical_selection_result_sha256(result_payload)
    )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="rank every"):
        matched.validate_sealed_protocol_transition(open_payload, sealed_payload, result_payload)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "2.5", "schema_version"),
        ("stage", "evaluation", "stage"),
        ("horizon", 0, "horizon"),
        ("horizon", True, "integer"),
    ],
)
def test_top_level_schema_stage_and_horizon_are_strict(
    field: str,
    value: Any,
    error: str,
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(matched.ForagerMatchedProtocolError, match=error):
        matched.parse_forager_matched_protocol(payload)


def test_top_level_keys_are_exact_and_host_metadata_is_impossible() -> None:
    payload = _payload()
    payload["created_at_utc"] = "2026-01-01T00:00:00Z"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="unknown keys"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    del payload["historical_orientation"]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="missing required"):
        matched.parse_forager_matched_protocol(payload)


@pytest.mark.parametrize(
    "seeds_update",
    [
        {"tuning_seeds": [1, 1], "active_seeds": [1, 1]},
        {"tuning_seeds": [True], "active_seeds": [True]},
        {"tuning_seeds": [-1], "active_seeds": [-1]},
        {"tuning_seeds": [2**31], "active_seeds": [2**31]},
        {"tuning_seeds": [], "active_seeds": []},
    ],
)
def test_seed_lists_reject_duplicates_bools_bounds_and_empty(
    seeds_update: dict[str, Any],
) -> None:
    payload = _payload()
    payload.update(seeds_update)
    with pytest.raises(matched.ForagerMatchedProtocolError):
        matched.parse_forager_matched_protocol(payload)


def test_seed_sets_are_disjoint_and_active_set_is_exact_for_stage() -> None:
    payload = _payload()
    payload["evaluation_seeds"] = [2_300_001]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="overlap"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["active_seeds"] = list(reversed(payload["tuning_seeds"]))
    with pytest.raises(matched.ForagerMatchedProtocolError, match="exactly match"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload(stage="sealed_evaluation")
    payload["active_seeds"] = payload["tuning_seeds"]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="exactly match"):
        matched.parse_forager_matched_protocol(payload)


def test_rollout_must_divide_supplied_horizon_without_hard_coded_value() -> None:
    protocol = matched.parse_forager_matched_protocol(_payload())
    assert protocol.horizon == 244 * 2_048

    payload = _payload()
    payload["horizon"] = 500_000
    with pytest.raises(matched.ForagerMatchedProtocolError, match="must divide"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    _candidate_by_id(payload, "exact_ppo")["execution_semantics"]["num_rollouts"] = 243
    with pytest.raises(matched.ForagerMatchedProtocolError, match="num_rollouts"):
        matched.parse_forager_matched_protocol(payload)


def test_runtime_requires_exact_qualified_cpu_oci_sandbox() -> None:
    mutations: list[tuple[list[str], Any]] = [
        (["executor_kind"], "host"),
        (["source_mount_mode"], "live_tree"),
        (["default_prng"], "rbg"),
        (["threefry_partitionable"], False),
        (["platform"], "gpu"),
        (["sandbox", "network"], "host"),
        (["sandbox", "root_filesystem"], "read_write"),
        (["sandbox", "capabilities"], "default"),
        (["sandbox", "no_new_privileges"], False),
        (["sandbox", "container_user"], "0:0"),
        (["sandbox", "host_devices"], ["/dev/nvidia0"]),
        (["sandbox", "writable_tmpfs_only"], False),
    ]
    for keys, value in mutations:
        payload = _payload()
        target = payload["runtime"]
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        with pytest.raises(matched.ForagerMatchedProtocolError):
            matched.parse_forager_matched_protocol(payload)


@pytest.mark.parametrize(
    ("binding_field", "value", "error"),
    [
        ("image_sha256", "a" * 64, "runtime image"),
        ("runtime_profile_sha256", "b" * 64, "runtime profile"),
        ("task_identity_sha256", "c" * 64, "task identity"),
    ],
)
def test_pairable_candidate_runtime_and_task_identity_must_match(
    binding_field: str,
    value: str,
    error: str,
) -> None:
    payload = _payload()
    _candidate_by_id(payload, "external_dqn")["runtime_binding"][binding_field] = value
    with pytest.raises(matched.ForagerMatchedProtocolError, match=error):
        matched.parse_forager_matched_protocol(payload)


def test_historical_orientation_may_retain_noncurrent_runtime_bindings() -> None:
    payload = _payload()
    historical = _candidate_by_id(payload, "paper_dqn")
    historical["runtime_binding"] = {
        "image_sha256": "a" * 64,
        "runtime_profile_sha256": "b" * 64,
        "task_identity_sha256": "c" * 64,
        "qualified_capability_descriptor_sha256": "0" * 64,
        "capability_qualification_receipt_sha256": "d" * 64,
        "qualification_trust_anchor_identity": "historical_archive_anchor_v1",
    }
    historical["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        _capability_descriptor_sha256(historical)
    )

    protocol = matched.parse_forager_matched_protocol(payload)

    assert protocol.candidate_index["paper_dqn"].pairing.eligible is False


def test_pairable_candidates_require_dedicated_common_environment_rng() -> None:
    payload = _payload()
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["environment_rng"]["identity"] = "other_schedule"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="common dedicated"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["environment_rng"]["schedule_sha256"] = "a" * 64
    with pytest.raises(matched.ForagerMatchedProtocolError, match="common dedicated"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["agent_rng"]["environment_key_shared"] = True
    with pytest.raises(matched.ForagerMatchedProtocolError, match="declaration disagree"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["agent_rng"]["identity"] = "renamed_rng_claim"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="isolated agent RNG"):
        matched.parse_forager_matched_protocol(payload)


def test_exact_upstream_ppo_shared_rng_is_descriptive_and_ineligible() -> None:
    matched.parse_forager_matched_protocol(_payload())

    payload = _payload()
    ppo = _candidate_by_id(payload, "exact_ppo")
    ppo["pairing"] = {
        "analysis_role": "inferential",
        "eligible": True,
        "exclusion_reasons": [],
    }
    with pytest.raises(matched.ForagerMatchedProtocolError, match="exact upstream"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    ppo = _candidate_by_id(payload, "exact_ppo")
    ppo["implementation_kind"] = "upstream_ppo_isolated_rng"
    ppo["agent_rng"] = {
        "identity": "isolated_agent_rng_v1",
        "environment_key_shared": False,
    }
    ppo["environment_rng"]["identity"] = "dedicated_environment_split_chain_v1"
    ppo["pairing"] = {
        "analysis_role": "inferential",
        "eligible": True,
        "exclusion_reasons": [],
    }
    payload["selection_plan"]["groups"].append(
        {
            "selection_group": "exact_orientation",
            "candidate_ids": ["exact_ppo"],
            "advance_count": 1,
        }
    )
    payload["evaluation_panel"]["selection_slots"].append(_slot("exact_orientation"))
    payload["evaluation_panel"]["fixed_descriptive_candidate_ids"].remove("exact_ppo")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="candidate semantics"):
        matched.parse_forager_matched_protocol(payload)

    ppo["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        _capability_descriptor_sha256(ppo)
    )
    ppo["runtime_binding"]["capability_qualification_receipt_sha256"] = hashlib.sha256(
        b"capability:exact_ppo:isolated_rng"
    ).hexdigest()
    protocol = matched.parse_forager_matched_protocol(payload)
    assert protocol.candidate_index["exact_ppo"].pairing.eligible is True

    payload = _payload()
    ppo = _candidate_by_id(payload, "exact_ppo")
    ppo["environment_rng"]["identity"] = "dedicated_environment_split_chain_v1"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="exact upstream"):
        matched.parse_forager_matched_protocol(payload)


def test_privileged_and_historical_candidates_can_never_be_pairable() -> None:
    for candidate_id in ("search_oracle", "paper_dqn"):
        payload = _payload()
        candidate = _candidate_by_id(payload, candidate_id)
        candidate["pairing"] = {
            "analysis_role": "inferential",
            "eligible": True,
            "exclusion_reasons": [],
        }
        with pytest.raises(matched.ForagerMatchedProtocolError):
            matched.parse_forager_matched_protocol(payload)


def test_observation_modes_cannot_escape_their_context_strata() -> None:
    payload = _payload()
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["observation_access"] = {
        "access_mode": "privileged_reward_grid",
        "observation_type": "color",
        "aperture_size": -1,
        "privileged_fields": ["reward_grid"],
    }
    with pytest.raises(matched.ForagerMatchedProtocolError, match="privileged stratum"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    candidate = _candidate_by_id(payload, "search_oracle")
    candidate["observation_access"] = {
        "access_mode": "partial_observation",
        "observation_type": "color",
        "aperture_size": 9,
        "privileged_fields": [],
    }
    with pytest.raises(matched.ForagerMatchedProtocolError, match="known privileged"):
        matched.parse_forager_matched_protocol(payload)


def test_context_membership_and_order_exactly_match_candidate_strata() -> None:
    payload = _payload()
    payload["privileged_context"]["candidate_ids"] = []
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs/order"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["historical_orientation"]["candidate_ids"] = ["paper_dqn", "search_oracle"]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs/order"):
        matched.parse_forager_matched_protocol(payload)


def test_evaluation_panel_has_one_pairable_alberta_and_external_primary() -> None:
    payload = _payload()
    payload["evaluation_panel"]["alberta_primary_slot"] = _slot("recurrent")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="Alberta primary"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["evaluation_panel"]["primary_nonprivileged_external_baseline_slot"] = _slot("alberta")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="primary slots"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["evaluation_panel"]["selection_slots"].append(_slot("external"))
    with pytest.raises(matched.ForagerMatchedProtocolError, match="duplicate selection slots"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["evaluation_panel"]["fixed_descriptive_candidate_ids"].append("paper_dqn")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="fixed descriptive"):
        matched.parse_forager_matched_protocol(payload)


def test_primary_ids_and_fail_closed_paired_contract_are_exact() -> None:
    payload = _payload()
    payload["primary_hypothesis"]["comparator_slot"] = _slot("recurrent")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="exactly match"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["primary_hypothesis"]["paired"] = False
    with pytest.raises(matched.ForagerMatchedProtocolError, match="paired must be true"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["evaluation_panel"]["pairing_failure_policy"] = "fall_back_unpaired"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="fail_closed"):
        matched.parse_forager_matched_protocol(payload)


def test_secondary_order_must_exactly_match_holm_family() -> None:
    payload = _payload()
    payload["multiplicity_policy"]["hypothesis_ids"] = []
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs/order"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["secondary_hypotheses"][0]["intervention_slot"] = _slot("privileged")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="outside the panel"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["secondary_hypotheses"][0]["hypothesis_id"] = "causal_vs_external"
    payload["multiplicity_policy"]["hypothesis_ids"] = ["causal_vs_external"]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs must be unique"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["secondary_hypotheses"][0]["intervention_slot"] = _slot("external")
    payload["secondary_hypotheses"][0]["comparator_slot"] = _slot("alberta")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="reverses"):
        matched.parse_forager_matched_protocol(payload)


def test_selection_groups_must_exist_and_cannot_over_advance() -> None:
    payload = _payload()
    payload["selection_plan"]["groups"][0]["selection_group"] = "missing"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs/order"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["selection_plan"]["groups"][0]["advance_count"] = 2
    with pytest.raises(matched.ForagerMatchedProtocolError, match="may not exceed"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["selection_plan"]["groups"].pop()
    with pytest.raises(matched.ForagerMatchedProtocolError, match="IDs/order"):
        matched.parse_forager_matched_protocol(payload)


def test_candidate_ids_and_configuration_paths_are_safe_and_unique() -> None:
    payload = _payload()
    payload["candidates"][1]["candidate_id"] = "alberta_causal"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="duplicate candidate"):
        matched.parse_forager_matched_protocol(payload)

    for unsafe in (
        "/etc/passwd",
        "../config.json",
        "C:\\config.json",
        "~/config.json",
        "configs//config.json",
        "configs/./config.json",
        "configs/config.json/",
    ):
        payload = _payload()
        _candidate_by_id(payload, "external_dqn")["configuration"]["original_path"] = unsafe
        with pytest.raises(matched.ForagerMatchedProtocolError, match="relative POSIX"):
            matched.parse_forager_matched_protocol(payload)


def test_source_provenance_requires_https_full_commit_and_lowercase_digests() -> None:
    payload = _payload()
    source = _candidate_by_id(payload, "external_dqn")["source"]
    source["repository"] = "file:///tmp/repo"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="HTTPS"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    source = _candidate_by_id(payload, "external_dqn")["source"]
    source["base_commit"] = "abc123"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="full lowercase"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    source = _candidate_by_id(payload, "external_dqn")["source"]
    source["tree_git_sha1"] = "A" * 40
    with pytest.raises(matched.ForagerMatchedProtocolError, match="lowercase"):
        matched.parse_forager_matched_protocol(payload)

    for malformed_url in (
        "https://example.com:not-a-port/repository",
        "https://example.com/repository\n",
        "https://example.com////",
    ):
        payload = _payload()
        source = _candidate_by_id(payload, "external_dqn")["source"]
        source["repository"] = malformed_url
        with pytest.raises(matched.ForagerMatchedProtocolError, match="HTTPS"):
            matched.parse_forager_matched_protocol(payload)


def test_reviewed_snapshot_provenance_does_not_claim_base_commit_tree() -> None:
    protocol = matched.parse_forager_matched_protocol(_payload())
    source = protocol.candidate_index["alberta_causal"].source
    assert source.provenance_kind == "reviewed_snapshot"
    assert source.tree_git_sha1 is None
    assert source.base_commit == "a" * 40
    assert len(source.archive_sha256) == 64
    assert len(source.inventory_sha256) == 64
    assert source.snapshot_descriptor_sha256 is not None
    assert len(source.snapshot_descriptor_sha256) == 64

    payload = _payload()
    _candidate_by_id(payload, "alberta_causal")["source"]["tree_git_sha1"] = "5" * 40
    with pytest.raises(matched.ForagerMatchedProtocolError, match="must be null"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    _candidate_by_id(payload, "external_dqn")["source"]["tree_git_sha1"] = None
    with pytest.raises(matched.ForagerMatchedProtocolError, match="tree_git_sha1"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    _candidate_by_id(payload, "external_dqn")["source"]["snapshot_descriptor_sha256"] = (
        "b" * 64
    )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="must be null"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    _candidate_by_id(payload, "alberta_causal")["source"]["snapshot_descriptor_sha256"] = None
    with pytest.raises(
        matched.ForagerMatchedProtocolError, match="snapshot_descriptor_sha256"
    ):
        matched.parse_forager_matched_protocol(payload)


def test_qualification_receipts_and_trust_anchor_are_mandatory_bindings() -> None:
    payload = _payload()
    del payload["runtime"]["executor_qualification_receipt_sha256"]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="missing required"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    binding = _candidate_by_id(payload, "external_dqn")["runtime_binding"]
    binding["qualification_trust_anchor_identity"] = "different_anchor"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="trust anchor"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    binding = _candidate_by_id(payload, "alberta_causal")["runtime_binding"]
    binding["qualified_capability_descriptor_sha256"] = "f" * 64
    with pytest.raises(matched.ForagerMatchedProtocolError, match="candidate semantics"):
        matched.parse_forager_matched_protocol(payload)

    protocol = matched.parse_forager_matched_protocol(_payload())
    candidate = protocol.candidate_index["alberta_causal"]
    assert (
        candidate.runtime_binding.qualified_capability_descriptor_sha256
        == matched.candidate_capability_descriptor_sha256(candidate)
    )


def test_known_privileged_and_historical_implementations_cannot_be_relabelled() -> None:
    payload = _payload()
    oracle = _candidate_by_id(payload, "search_oracle")
    oracle["stratum"] = "external_learning"
    oracle["selection_group"] = "external"
    oracle["observation_access"] = {
        "access_mode": "partial_observation",
        "observation_type": "color",
        "aperture_size": 9,
        "privileged_fields": [],
    }
    oracle["pairing"] = {"analysis_role": "inferential", "eligible": True, "exclusion_reasons": []}
    payload["privileged_context"]["candidate_ids"] = []
    payload["selection_plan"]["groups"][1]["candidate_ids"].append("search_oracle")
    payload["evaluation_panel"]["fixed_descriptive_candidate_ids"].remove("search_oracle")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="known privileged"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    historical = _candidate_by_id(payload, "paper_dqn")
    historical["stratum"] = "external_learning"
    historical["selection_group"] = "external"
    historical["observation_access"] = {
        "access_mode": "partial_observation",
        "observation_type": "color",
        "aperture_size": 9,
        "privileged_fields": [],
    }
    historical["pairing"] = {
        "analysis_role": "inferential",
        "eligible": True,
        "exclusion_reasons": [],
    }
    payload["historical_orientation"]["candidate_ids"] = []
    payload["selection_plan"]["groups"][1]["candidate_ids"].append("paper_dqn")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="known historical"):
        matched.parse_forager_matched_protocol(payload)


def test_configuration_transforms_are_typed_unique_and_hash_explaining() -> None:
    payload = _payload()
    config = _candidate_by_id(payload, "external_dqn")["configuration"]
    config["derived_sha256"] = "a" * 64
    with pytest.raises(matched.ForagerMatchedProtocolError, match="must explain"):
        matched.parse_forager_matched_protocol(payload)

    transform = {
        "transform_type": "set_horizon",
        "target": "total_steps",
        "value_type": "integer",
        "value": HORIZON,
    }
    payload = _payload()
    config = _candidate_by_id(payload, "external_dqn")["configuration"]
    config["derived_sha256"] = "a" * 64
    config["allowed_transforms"] = [transform]
    candidate = _candidate_by_id(payload, "external_dqn")
    candidate["runtime_binding"]["qualified_capability_descriptor_sha256"] = (
        _capability_descriptor_sha256(candidate)
    )
    parsed = matched.parse_forager_matched_protocol(payload)
    assert parsed.candidate_index["external_dqn"].configuration.allowed_transforms[0].value == (
        HORIZON
    )

    payload = _payload()
    config = _candidate_by_id(payload, "external_dqn")["configuration"]
    config["derived_sha256"] = "a" * 64
    invalid = copy.deepcopy(transform)
    invalid["value"] = True
    config["allowed_transforms"] = [invalid]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="must be an integer"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    config = _candidate_by_id(payload, "external_dqn")["configuration"]
    config["derived_sha256"] = "a" * 64
    config["allowed_transforms"] = [transform, copy.deepcopy(transform)]
    with pytest.raises(matched.ForagerMatchedProtocolError, match="repeats a target"):
        matched.parse_forager_matched_protocol(payload)


def test_pairable_seed_transport_must_preserve_active_seed() -> None:
    payload = _payload()
    seed_contract = _candidate_by_id(payload, "external_dqn")["seed_contract"]
    seed_contract["offset"] = 1
    seed_contract["effective_seed_expression"] = "active_seed_plus_offset"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="preserve the active seed"):
        matched.parse_forager_matched_protocol(payload)


def test_strict_loader_rejects_duplicate_keys_nonfinite_json_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="duplicate JSON"):
        matched.load_forager_matched_protocol(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"horizon":NaN}', encoding="utf-8")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="non-finite"):
        matched.load_forager_matched_protocol(nonfinite)

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_payload()), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(valid)
    with pytest.raises(matched.ForagerMatchedProtocolError, match="non-symlink"):
        matched.load_forager_matched_protocol(link)


def test_strict_decoder_rejects_overflow_numbers_and_invalid_unicode() -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="non-finite"):
        matched.decode_strict_json('{"number":1e10000}')

    with pytest.raises(matched.ForagerMatchedProtocolError, match="Unicode"):
        matched.decode_strict_json('{"text":"\\ud800"}')


def test_direct_values_and_canonicalization_are_validated_fail_closed() -> None:
    payload = _payload()
    payload["selection_plan"]["confidence"] = float("nan")
    with pytest.raises(matched.ForagerMatchedProtocolError, match="non-finite"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["selection_plan"]["confidence"] = 10**400
    with pytest.raises(matched.ForagerMatchedProtocolError, match="finite number"):
        matched.parse_forager_matched_protocol(payload)

    payload = _payload()
    payload["host_path"] = "/tmp/results"
    with pytest.raises(matched.ForagerMatchedProtocolError, match="unknown keys"):
        matched.canonical_json_bytes(payload)


def test_loader_rejects_noncanonical_and_accepts_exact_canonical_json(tmp_path: Path) -> None:
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")

    with pytest.raises(matched.ForagerMatchedProtocolError, match="exact canonical"):
        matched.load_forager_matched_protocol(source)

    canonical = matched.canonical_json_bytes(_payload())
    source.write_bytes(canonical)
    protocol = matched.load_forager_matched_protocol(source)
    assert protocol.canonical_bytes == source.read_bytes()


def test_matched_protocol_records_reject_leftover_identities() -> None:
    """Public protocol records must not keep leftover bool/int identities."""

    with pytest.raises(matched.ForagerMatchedProtocolError, match="aperture_size"):
        matched.MatchedTask(
            task_id="task",
            preset="field_of_view",
            environment_id="env",
            foragax_distribution="dist",
            foragax_version="0.1.0",
            observation_type="rgb",
            aperture_size=True,
            task_identity_sha256=TASK_SHA,
            environment_rng_schedule_sha256=SCHEDULE_SHA,
        )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="environment_key_shared"):
        matched.AgentRNGContract(identity="threefry", environment_key_shared=1)
    with pytest.raises(matched.ForagerMatchedProtocolError, match="parameter_count"):
        matched.ResourceAccounting(
            parameter_count=True,
            optimizer_update_count=0,
            replay_capacity_transitions=0,
            recurrent_state_elements=0,
        )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="eligible"):
        matched.PairingEligibility(
            analysis_role="inferential", eligible=1, exclusion_reasons=()
        )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="rank"):
        matched.SelectionSlot(selection_group="alberta", rank=True)
    with pytest.raises(matched.ForagerMatchedProtocolError, match="advance_count"):
        matched.SelectionGroup(selection_group="alberta", candidate_ids=("a",), advance_count=True)

    legal_task = matched.MatchedTask(
        task_id="task",
        preset="field_of_view",
        environment_id="env",
        foragax_distribution="dist",
        foragax_version="0.1.0",
        observation_type="rgb",
        aperture_size=15,
        task_identity_sha256=TASK_SHA,
        environment_rng_schedule_sha256=SCHEDULE_SHA,
    )
    legal_rng = matched.AgentRNGContract(identity="threefry", environment_key_shared=False)
    legal_resources = matched.ResourceAccounting(0, 0, 0, 0)
    legal_pairing = matched.PairingEligibility(
        analysis_role="inferential", eligible=True, exclusion_reasons=()
    )
    legal_slot = matched.SelectionSlot(selection_group="alberta", rank=1)
    legal_group = matched.SelectionGroup(
        selection_group="alberta", candidate_ids=("a",), advance_count=1
    )
    dumped = json.dumps(
        {
            "aperture_size": legal_task.aperture_size,
            "environment_key_shared": legal_rng.environment_key_shared,
            "parameter_count": legal_resources.parameter_count,
            "eligible": legal_pairing.eligible,
            "rank": legal_slot.rank,
            "advance_count": legal_group.advance_count,
        },
        allow_nan=False,
    )
    assert '"aperture_size": 15' in dumped
    assert '"environment_key_shared": false' in dumped
    assert '"parameter_count": 0' in dumped
    assert '"eligible": true' in dumped
    assert '"rank": 1' in dumped
    assert '"advance_count": 1' in dumped
    assert '"aperture_size": true' not in dumped
    assert '"environment_key_shared": 1' not in dumped
    assert '"parameter_count": true' not in dumped
    assert '"eligible": 1' not in dumped
    assert '"rank": true' not in dumped
    assert '"advance_count": true' not in dumped


def test_matched_protocol_record_scalars_reject_hostile_exact_type_facades() -> None:
    class IntFacade(int):
        pass

    with pytest.raises(matched.ForagerMatchedProtocolError, match="aperture_size"):
        matched.MatchedTask(
            task_id="task",
            preset="field_of_view",
            environment_id="env",
            foragax_distribution="dist",
            foragax_version="0.1.0",
            observation_type="rgb",
            aperture_size=IntFacade(15),
            task_identity_sha256=TASK_SHA,
            environment_rng_schedule_sha256=SCHEDULE_SHA,
        )
    with pytest.raises(matched.ForagerMatchedProtocolError, match="rank"):
        matched.SelectionSlot(selection_group="alberta", rank=IntFacade(1))


@pytest.mark.parametrize("aperture_size", (0, 2, 65_536))
def test_matched_task_direct_construction_enforces_parser_aperture_bounds(
    aperture_size: int,
) -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="aperture_size"):
        matched.MatchedTask(
            task_id="task",
            preset="field_of_view",
            environment_id="env",
            foragax_distribution="dist",
            foragax_version="0.1.0",
            observation_type="rgb",
            aperture_size=aperture_size,
            task_identity_sha256=TASK_SHA,
            environment_rng_schedule_sha256=SCHEDULE_SHA,
        )


@pytest.mark.parametrize("rank", (0, 257))
def test_selection_slot_direct_construction_enforces_rank_bounds(rank: int) -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="rank"):
        matched.SelectionSlot(selection_group="alberta", rank=rank)


@pytest.mark.parametrize("advance_count", (0, 2, 257))
def test_selection_group_direct_construction_enforces_count_invariants(
    advance_count: int,
) -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="advance_count"):
        matched.SelectionGroup(
            selection_group="alberta",
            candidate_ids=("candidate",),
            advance_count=advance_count,
        )


@pytest.mark.parametrize(
    ("analysis_role", "eligible", "exclusion_reasons"),
    (
        ("descriptive_only", True, ()),
        ("inferential", True, ("reason",)),
        ("inferential", False, ("reason",)),
        ("descriptive_only", False, ()),
    ),
)
def test_pairing_direct_construction_enforces_parser_cross_field_invariants(
    analysis_role: matched.AnalysisRole,
    eligible: bool,
    exclusion_reasons: tuple[str, ...],
) -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="pairing"):
        matched.PairingEligibility(
            analysis_role=analysis_role,
            eligible=eligible,
            exclusion_reasons=exclusion_reasons,
        )


@pytest.mark.parametrize("value", (-1, 2**63))
def test_resource_record_direct_construction_enforces_exact_bounds(value: int) -> None:
    with pytest.raises(matched.ForagerMatchedProtocolError, match="parameter_count"):
        matched.ResourceAccounting(value, 0, 0, 0)
