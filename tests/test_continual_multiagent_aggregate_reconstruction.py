"""Fail-closed reconstruction regressions for continual-multiagent evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import alberta_framework.evaluation.continual_multiagent_artifact as artifact_module

pytestmark = pytest.mark.unit

_PINNED_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "outputs/continual_multiagent/evidence.json"
)


def _source_projected_scratch() -> dict[str, object]:
    """Return a nonpromoting scratch copy bound to the current validator source."""

    artifact = json.loads(_PINNED_ARTIFACT.read_text(encoding="utf-8"))
    content = artifact["content"]
    content["source_provenance"] = artifact_module._source_provenance()
    artifact["content_digest"]["sha256"] = artifact_module.scientific_content_sha256(content)
    assert artifact_module.validate_evidence_artifact(artifact).accepted
    return artifact


def _rehash(artifact: dict[str, object]) -> None:
    content = artifact["content"]
    artifact["content_digest"]["sha256"] = artifact_module.scientific_content_sha256(content)


def test_rehashed_primitive_summary_corruption_fails_closed() -> None:
    fabricated = copy.deepcopy(_source_projected_scratch())
    content = fabricated["content"]
    summaries = content["seed_summaries"]
    for seed in summaries:
        joint = seed["conditions"]["joint_adaptive"]
        joint["phase_mean_rewards"][0] = -999.0
        joint["read_only_probe_matrix"][2][0] = -999.0
        joint["mean_forgetting"] = 999.0
        joint["maximum_forgetting"] = 999.0
        joint["interference_forgetting"] = 999.0
        joint["mean_stability_gap"] = -999.0
        joint["controller_budget"] = {
            "state_scalars": 999,
            "state_bytes": 999,
            "action_scalars_per_step": 999,
        }
    _rehash(fabricated)

    validation = artifact_module.validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert any("inconsistent with seed summaries" in error for error in validation.errors)


def test_rehashed_coordinated_aggregate_forgery_fails_closed() -> None:
    fabricated = copy.deepcopy(_source_projected_scratch())
    content = fabricated["content"]
    aggregate = content["aggregate"]
    aggregate.update(
        {
            "recurrent_a_probe_reward": 0.95,
            "mean_forgetting": 0.04,
            "maximum_forgetting": 999.0,
            "mean_stability_gap": 0.19,
        }
    )
    aggregate["resource_budget"] = {
        "state_scalars": 999,
        "state_bytes": 999,
        "action_scalars_per_step": 999,
        "identical_across_conditions": True,
    }
    forged_actuals = {
        "recurrent_a_probe_reward": 0.95,
        "mean_forgetting": 0.04,
        "mean_stability_gap": 0.19,
        "budgets_identical": 1.0,
    }
    for check in content["acceptance"]["checks"]:
        if check["name"] in forged_actuals:
            check["actual"] = forged_actuals[check["name"]]
    _rehash(fabricated)

    validation = artifact_module.validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert any("inconsistent with seed summaries" in error for error in validation.errors)


def test_rehashed_finiteness_and_budget_declarations_are_reconstructed() -> None:
    fabricated = copy.deepcopy(_source_projected_scratch())
    content = fabricated["content"]
    aggregate = content["aggregate"]
    aggregate["all_scientific_and_timing_values_finite"] = False
    aggregate["resource_budget"]["identical_across_conditions"] = False
    for check in content["acceptance"]["checks"]:
        if check["name"] in {"all_values_finite", "budgets_identical"}:
            check["actual"] = 0.0
            check["passed"] = False
    content["acceptance"]["passed"] = False
    fabricated["operational_diagnostics"]["overall_acceptance_passed"] = False
    _rehash(fabricated)

    validation = artifact_module.validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert any("finiteness" in error or "resource budget" in error for error in validation.errors)


def test_operational_finiteness_is_reconstructed_from_condition_timings() -> None:
    fabricated = copy.deepcopy(_source_projected_scratch())
    timings = fabricated["operational_diagnostics"]["condition_timings"]
    timings[0]["p95_update_latency_ms"] = None

    validation = artifact_module.validate_evidence_artifact(fabricated)

    assert not validation.valid
    assert not validation.accepted
    assert any("must be finite and nonnegative" in error for error in validation.errors)
