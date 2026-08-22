"""Frozen pre-run contract for the issue #1568 prospective comparison.

This module deliberately cannot execute the campaign.  It reserves fresh
development inputs and binds the prerequisite implementation so review and
explicit authorization can precede any observations or seed consumption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

PLAN_SCHEMA: Final = "asi.optimization-readiness.campaign-plan.v1"
RESULT_SCHEMA: Final = "asi.optimization-readiness.campaign-result.v1"
EXECUTION_AUTHORIZED: Final = False
_ROOT = Path(__file__).resolve().parents[2]
_PREREQUISITE = Path("alberta_framework/evaluation/optimization_readiness.py")
_SEEDS: Final = (191738449, 400137777, 629555293, 1066941361, 1521016777,
                 2112317971, 2550319013, 3034091879, 3589119781, 4107369503)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs() -> list[dict[str, object]]:
    return [
        {"checkpoint_after_task": checkpoint, "evaluation_task_offset": offset}
        for checkpoint in (1, 3, 5, 7)
        for offset in (0, 1, 2)
    ]


def optimization_readiness_campaign_plan() -> dict[str, Any]:
    """Return a new copy of the literal, permanently nonpromoting plan."""
    return {
        "schema": PLAN_SCHEMA,
        "issue": "SlopDotCash/asi#1568",
        "paper": {
            "revision": "arXiv:2605.09044v1",
            "revision_date": "2026-05-09",
            "official_code_revision": "none-cited-in-arxiv-v1-as-of-2026-08-17",
            "reference_implementation": "equations-implemented-in-asi-prerequisite",
            "protocol_differences": [
                "ASI uses caller-supplied gradients under a source-bound implementation",
                "ASI evaluates IPMNIST task-boundary checkpoints instead of paper models",
                "all outcomes are development-only and permanently nonpromoting",
            ],
        },
        "identity": {
            "source_sha256": {str(_PREREQUISITE): _sha256(_ROOT / _PREREQUISITE)},
            "result_schema": RESULT_SCHEMA,
            "output_namespace": "outputs/optimization_readiness/prospective_v1",
        },
        "sampling": {
            "workload": "ipmnist",
            "seeds": list(_SEEDS),
            "checkpoint_task_pairs": _pairs(),
            "future_gain_horizons": [1, 10, 100],
            "future_gain_rollouts_per_horizon": 128,
            "future_gain_step_size": 0.001,
            "checkpoint_sampling": "after-task checkpoints 1,3,5,7",
            "task_sampling": "current and next two deterministic task offsets",
        },
        "validation": {
            "full_validation_observations": 10_000,
            "diagnostic_batch_size": 4,
            "diagnostic_batch_count": 128,
            "sampling": "independent-with-replacement Threefry subkeys",
            "parity_test": "paper empirical equations",
            "mechanism_off_test": "zero gradient strength or reliability gives zero readiness",
        },
        "comparison": {
            "target": "future_relative_loss_reduction_after_matched_updates",
            "primary_metric": "pairwise_checkpoint_ranking_accuracy",
            "predictors": [
                "optimization_readiness",
                "representation_energy_rank_0_99",
                "parameter_norm",
                "gradient_norm",
                "curvature_energy_rank_0_99",
            ],
            "matched_axes": [
                "seed", "checkpoint", "task", "updates", "observations",
                "validation_batches", "allowed_information", "resource_ceilings",
            ],
            "allowed_boundary_information": ["task_start"],
            "allowed_task_information": [
                "current_validation_inputs", "current_validation_labels"
            ],
        },
        "resources": {
            "persistent_bytes_ceiling": 268_435_456,
            "peak_working_set_bytes_ceiling": 536_870_912,
            "charged_fields": [
                "persistent_bytes", "peak_working_set_bytes", "environment_steps",
                "data_steps", "model_queries", "parameter_updates", "timing_seconds",
            ],
            "timing_is_telemetry_only": True,
            "matched_resource_receipts_required": True,
            "per_horizon_charges": [
                {"horizon": 1, "parameter_updates": 128, "observations": 1_291_024},
                {"horizon": 10, "parameter_updates": 1_280, "observations": 1_295_632},
                {"horizon": 100, "parameter_updates": 12_800, "observations": 1_341_712},
            ],
        },
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "retain_negative_outcomes": True,
            "fresh_seeds_reserved_not_consumed": True,
            "completed_result_exists": False,
        },
        "authorization": {
            "execution_authorized": False,
            "maintainer_review_required": True,
            "authorization_change_requires_separate_review": True,
        },
    }


def validate_optimization_readiness_campaign_plan(payload: object) -> dict[str, Any]:
    """Accept only the current literal plan, including its live source binding."""
    expected = optimization_readiness_campaign_plan()
    if type(payload) is not dict or payload != expected:
        raise ValueError("payload does not match the frozen optimization-readiness campaign plan")
    # Return a fresh canonical value so callers cannot mutate module-owned state.
    return expected


def require_optimization_readiness_execution(payload: object) -> None:
    """Fail before inspecting caller data while execution remains unauthorized."""
    if not EXECUTION_AUTHORIZED:
        raise PermissionError(
            "optimization-readiness campaign execution is not authorized in this revision"
        )
    validate_optimization_readiness_campaign_plan(payload)
