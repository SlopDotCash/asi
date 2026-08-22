"""Fail-closed contracts for external benchmark qualification plans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alberta_framework.benchmarks.external_qualification import (
    COMMON_GATES,
    EXTERNAL_QUALIFICATION_PLANS,
    ExternalCodeRevision,
    ExternalQualificationPlan,
    qualification_plan,
)

pytestmark = pytest.mark.unit


def test_registry_exactly_covers_research_wave_and_is_not_run_ready() -> None:
    assert tuple(plan.issue for plan in EXTERNAL_QUALIFICATION_PLANS) == tuple(range(1574, 1584))
    assert len({plan.lane_id for plan in EXTERNAL_QUALIFICATION_PLANS}) == 10
    for plan in EXTERNAL_QUALIFICATION_PLANS:
        assert plan.blockers == plan.required_gates
        with pytest.raises(RuntimeError, match=f"external lane {plan.lane_id} is not qualified"):
            plan.require_ready()


def test_clear_uses_official_curation_source_without_claiming_asset_readiness() -> None:
    plan = qualification_plan(1579)
    assert plan.code_revisions == (
        ExternalCodeRevision(
            "https://github.com/linzhiqiu/continual-learning.git",
            "620cab4a7d99921fde73b67b53879470533cb39a",
        ),
        ExternalCodeRevision(
            "https://github.com/ElvishElvis/CLEAR-Continual_Learning_Benchmark.git",
            "75d5d2e7d412a787e0decf0417a4868c56691252",
        ),
        ExternalCodeRevision(
            "https://github.com/ContinualAI/avalanche.git",
            "eb075be393e1f458b2c352514ff6c17b5a2c0f4e",
        ),
    )
    assert "assets_checksums_and_storage_approved" in plan.blockers
    assert "external_code_available_and_license_reviewed" in plan.blockers
    assert "provider_archive_revision_and_checksums_disclosed" in plan.blockers


def test_action_conditioned_lane_pins_official_dreamer_cdp_source() -> None:
    plan = qualification_plan(1575)
    assert plan.code_revisions[0] == ExternalCodeRevision(
        "https://github.com/fmi-basel/Dreamer-CDP.git",
        "a851fa3e3d70b624b094ee1810ad4bb602346092",
    )
    assert "isolated_runtime_locked" in plan.blockers


def test_ftl_lane_pins_official_continual_bench_source_without_claiming_readiness() -> None:
    plan = qualification_plan(1574)
    assert plan.code_revisions == (
        ExternalCodeRevision(
            "https://github.com/sail-sg/ContinualBench.git",
            "a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad",
        ),
    )
    assert plan.blockers == plan.required_gates
    assert "external_code_available_and_license_reviewed" in plan.blockers
    assert "external_execution_separately_authorized" in plan.blockers


def test_ftl_source_manifest_content_binds_official_revision_without_execution_claims() -> None:
    payload = json.loads(
        Path("external_runtimes/continual_bench/qualification-plan.json").read_bytes()
    )
    assert payload["schema"] == "asi.continual_bench_external_source_qualification.v1"
    assert payload["source"] == {
        "repository": "https://github.com/sail-sg/ContinualBench.git",
        "commit": "a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad",
        "git_tree": "ebf540dbac186f13858f97dfe12eb0b3c823ec43",
        "source_archive_sha256": (
            "7726bc3badd6ad8752845b50a98e84e8d19c549c49bacf7bda84cd3933aa6e04"
        ),
        "license": "MIT",
        "license_path": "LICENSE.txt",
        "license_sha256": (
            "854b88f1dd8df45fc717efc3926da5d10efb6b1122b47ddbea639eb2637a867f"
        ),
    }
    assert payload["claims"] == {
        "source_identity_content_bound": True,
        "runtime_qualified": False,
        "assets_qualified": False,
        "external_execution_authorized": False,
        "paper_parity_claimed": False,
        "scientific_promotion_allowed": False,
    }


def test_loss_of_plasticity_lane_preserves_protocol_and_cost_blockers() -> None:
    plan = qualification_plan(1583)
    assert "input_permutation_not_misreported_as_random_labels" in plan.blockers
    assert "postpublication_code_changes_and_rl_step_mismatch_reviewed" in plan.blockers
    assert "costly_imagenet_and_rl_lanes_separately_registered" in plan.blockers


def test_completed_r0_plan_still_cannot_authorize_external_execution() -> None:
    revision = ExternalCodeRevision("https://github.com/org/repo.git", "a" * 40)
    plan = ExternalQualificationPlan(
        issue=1,
        lane_id="fixture",
        paper_revisions=("paper-v1",),
        code_revisions=(revision,),
        required_gates=("first", "second"),
        completed_gates=("second", "first"),
    )
    assert plan.blockers == ()
    with pytest.raises(RuntimeError, match="cannot authorize external execution"):
        plan.require_ready()


def test_missing_official_code_remains_a_blocker_even_if_claimed_complete() -> None:
    plan = ExternalQualificationPlan(
        issue=1,
        lane_id="fixture",
        paper_revisions=("paper-v1",),
        code_revisions=(),
        required_gates=COMMON_GATES,
        completed_gates=COMMON_GATES,
    )
    assert "external_code_available_and_license_reviewed" in plan.blockers
    with pytest.raises(RuntimeError, match="external_code_available_and_license_reviewed"):
        plan.require_ready()


def test_common_gates_require_provenance_and_separate_launch_authorization() -> None:
    assert "paper_code_and_asset_provenance_verified" in COMMON_GATES
    assert "external_execution_separately_authorized" in COMMON_GATES


@pytest.mark.parametrize("issue", [True, 1574.0, "1574"])
def test_lookup_rejects_scalar_aliases(issue: object) -> None:
    with pytest.raises(ValueError, match="exact integer"):
        qualification_plan(issue)


def test_plan_rejects_hostile_container_and_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="lane_id"):
        ExternalQualificationPlan(1, StringSubclass("lane"), ("paper",), (), COMMON_GATES)
    with pytest.raises(ValueError, match="paper_revisions"):
        ExternalQualificationPlan(1, "lane", ["paper"], (), COMMON_GATES)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entries"):
        ExternalQualificationPlan(1, "lane", (StringSubclass("paper"),), (), COMMON_GATES)


def test_code_revision_requires_credential_free_url_and_full_commit() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        ExternalCodeRevision("https://token@github.com/org/repo.git", "a" * 40)
    with pytest.raises(ValueError, match="full lowercase"):
        ExternalCodeRevision("https://github.com/org/repo.git", "A" * 40)


def test_completed_gates_must_be_known_and_unique() -> None:
    with pytest.raises(ValueError, match="subset"):
        ExternalQualificationPlan(1, "lane", ("paper",), (), ("known",), ("unknown",))
    with pytest.raises(ValueError, match="duplicates"):
        ExternalQualificationPlan(1, "lane", ("paper",), (), ("known", "known"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("paper_revisions", tuple(f"paper-{index}" for index in range(17))),
        ("code_revisions", tuple([object()] * 17)),
        ("required_gates", tuple(f"gate-{index}" for index in range(33))),
        ("completed_gates", tuple(f"gate-{index}" for index in range(33))),
    ],
)
def test_plan_rejects_oversized_tuples_before_entry_validation(
    field: str, value: tuple[object, ...]
) -> None:
    values: dict[str, object] = {
        "issue": 1,
        "lane_id": "lane",
        "paper_revisions": ("paper",),
        "code_revisions": (),
        "required_gates": ("gate",),
        "completed_gates": (),
    }
    values[field] = value
    with pytest.raises(ValueError, match=f"{field} contains too many entries"):
        ExternalQualificationPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["x" * 257, "invalid-\ud800"])
def test_plan_rejects_oversized_or_invalid_unicode_strings(value: str) -> None:
    with pytest.raises(ValueError, match="UTF-8 byte limit|valid Unicode"):
        ExternalQualificationPlan(1, "lane", (value,), (), ("gate",))


def test_plan_rejects_aggregate_payload_before_set_construction() -> None:
    gates = tuple(f"{index:02d}-" + "x" * 180 for index in range(32))
    with pytest.raises(ValueError, match="aggregate UTF-8 byte limit"):
        ExternalQualificationPlan(1, "lane", ("paper",), (), gates, gates)


def test_plan_revalidates_forged_nested_revision() -> None:
    revision = object.__new__(ExternalCodeRevision)
    object.__setattr__(revision, "repository", "https://github.com/org/repo.git")
    object.__setattr__(revision, "commit", "invalid-\ud800")
    with pytest.raises(ValueError, match="valid Unicode"):
        ExternalQualificationPlan(1, "lane", ("paper",), (revision,), ("gate",))
