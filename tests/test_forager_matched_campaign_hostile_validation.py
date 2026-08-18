"""Hostile input and boundary validation for Forager matched campaign records."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_campaign import (
    CampaignStatus,
    CompletedCampaignBundle,
    ForagerMatchedCampaignError,
)


def test_campaign_status_valid_construction() -> None:
    status = CampaignStatus(
        output_root=Path("campaign/output"),
        state="running",
        completed_cells=5,
        total_cells=10,
        next_candidate_id="cand_1",
        next_seed=12345,
        protocol_sha256="a" * 64,
        qualification_manifest_sha256="b" * 64,
        plan_sha256="c" * 64,
        live_runtime_identity_sha256="d" * 64,
        score_evidence_sha256="e" * 64,
        verification_subject_sha256="f" * 64,
    )
    assert status.state == "running"
    assert status.completed_cells == 5
    assert status.total_cells == 10


def test_campaign_status_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatchedCampaignError, match="output_root must be a Path"):
        CampaignStatus(
            output_root="campaign/output",  # type: ignore[arg-type]
            state="running",
            completed_cells=5,
            total_cells=10,
            next_candidate_id=None,
            next_seed=None,
            protocol_sha256="a" * 64,
            qualification_manifest_sha256="b" * 64,
            plan_sha256="c" * 64,
            live_runtime_identity_sha256="d" * 64,
            score_evidence_sha256=None,
            verification_subject_sha256=None,
        )

    with pytest.raises(
        ForagerMatchedCampaignError, match="completed_cells cannot exceed total_cells"
    ):
        CampaignStatus(
            output_root=Path("campaign/output"),
            state="running",
            completed_cells=15,
            total_cells=10,
            next_candidate_id=None,
            next_seed=None,
            protocol_sha256="a" * 64,
            qualification_manifest_sha256="b" * 64,
            plan_sha256="c" * 64,
            live_runtime_identity_sha256="d" * 64,
            score_evidence_sha256=None,
            verification_subject_sha256=None,
        )

    with pytest.raises(ForagerMatchedCampaignError, match="protocol_sha256 must be a 64-character"):
        CampaignStatus(
            output_root=Path("campaign/output"),
            state="running",
            completed_cells=5,
            total_cells=10,
            next_candidate_id=None,
            next_seed=None,
            protocol_sha256="invalid",
            qualification_manifest_sha256="b" * 64,
            plan_sha256="c" * 64,
            live_runtime_identity_sha256="d" * 64,
            score_evidence_sha256=None,
            verification_subject_sha256=None,
        )


def test_completed_campaign_bundle_validation() -> None:
    with pytest.raises(ForagerMatchedCampaignError, match="output_root must be a Path"):
        CompletedCampaignBundle(
            output_root="invalid/path",  # type: ignore[arg-type]
            protocol=None,  # type: ignore[arg-type]
            plan=None,  # type: ignore[arg-type]
            live_runtime=None,  # type: ignore[arg-type]
            candidate_ids=("cand_1",),
            active_seeds=(1, 2),
            schedule={},
            seed_artifacts={},
            execution_receipt_index=None,  # type: ignore[arg-type]
            score_evidence=None,  # type: ignore[arg-type]
            verification_request=None,  # type: ignore[arg-type]
            completion_summary={},
            final_file_sha256={},
        )
