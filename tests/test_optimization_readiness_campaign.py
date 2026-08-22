from __future__ import annotations

from copy import deepcopy

import pytest

from alberta_framework.evaluation.optimization_readiness_campaign import (
    optimization_readiness_campaign_plan,
    require_optimization_readiness_execution,
    validate_optimization_readiness_campaign_plan,
)


def test_campaign_plan_freezes_prospective_matched_comparison() -> None:
    plan = validate_optimization_readiness_campaign_plan(
        optimization_readiness_campaign_plan()
    )

    assert plan["paper"]["revision"] == "arXiv:2605.09044v1"
    assert plan["paper"]["official_code_revision"] == (
        "none-cited-in-arxiv-v1-as-of-2026-08-17"
    )
    assert plan["sampling"]["future_gain_horizons"] == [1, 10, 100]
    assert len(plan["sampling"]["seeds"]) == 10
    assert len(plan["sampling"]["checkpoint_task_pairs"]) == 12
    assert plan["comparison"]["predictors"] == [
        "optimization_readiness",
        "representation_energy_rank_0_99",
        "parameter_norm",
        "gradient_norm",
        "curvature_energy_rank_0_99",
    ]
    assert plan["policy"]["development_only"] is True
    assert plan["policy"]["scientific_promotion_allowed"] is False
    assert plan["authorization"]["execution_authorized"] is False
    assert plan["resources"]["per_horizon_charges"] == [
        {"horizon": 1, "parameter_updates": 128, "observations": 1_291_024},
        {"horizon": 10, "parameter_updates": 1_280, "observations": 1_295_632},
        {"horizon": 100, "parameter_updates": 12_800, "observations": 1_341_712},
    ]


def test_campaign_execution_fails_before_accepting_a_plan() -> None:
    class Hostile(dict):
        def __iter__(self):
            raise AssertionError("unauthorized execution inspected its input")

    with pytest.raises(PermissionError, match="not authorized"):
        require_optimization_readiness_execution(Hostile())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sampling", "future_gain_horizons", [1, 10]),
        ("sampling", "seeds", [0] * 10),
        ("validation", "diagnostic_batch_count", 127),
        ("resources", "timing_is_telemetry_only", False),
        ("policy", "retain_negative_outcomes", False),
        ("authorization", "execution_authorized", True),
    ],
)
def test_campaign_plan_rejects_protocol_drift(
    section: str, field: str, value: object
) -> None:
    payload = deepcopy(optimization_readiness_campaign_plan())
    payload[section][field] = value
    with pytest.raises(ValueError, match="frozen optimization-readiness campaign plan"):
        validate_optimization_readiness_campaign_plan(payload)


def test_campaign_plan_binds_current_prerequisite_source() -> None:
    payload = deepcopy(optimization_readiness_campaign_plan())
    payload["identity"]["source_sha256"][
        "alberta_framework/evaluation/optimization_readiness.py"
    ] = "0" * 64
    with pytest.raises(ValueError, match="frozen optimization-readiness campaign plan"):
        validate_optimization_readiness_campaign_plan(payload)
