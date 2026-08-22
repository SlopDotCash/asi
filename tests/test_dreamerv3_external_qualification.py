"""Contracts for the read-only DreamerV3 source qualification."""

from __future__ import annotations

import dataclasses

import pytest

from alberta_framework.benchmarks.dreamerv3_external_qualification import (
    SOURCE_QUALIFICATION,
)
from alberta_framework.benchmarks.external_qualification import qualification_plan

pytestmark = pytest.mark.unit


def test_source_qualification_binds_catalog_and_content_identities() -> None:
    plan = qualification_plan(1576)
    qualification = SOURCE_QUALIFICATION
    assert qualification.repository == plan.code_revisions[0].repository
    assert qualification.commit == plan.code_revisions[0].commit
    assert qualification.git_tree == "a6611dd5cca395eebcd387ebcad2685bb2d9dbdf"
    assert qualification.source_archive_sha256 == (
        "bf7a237bd345e200f895943145b33e0296d40a8b90b2b7144c57985bd30698f4"
    )
    assert qualification.source_archive_bytes == 6_312_430
    assert qualification.license_spdx == "MIT"
    assert qualification.license_sha256 == (
        "9a0db563b71a42110ce6e52c066ec957ca908dd2fbff91e85df09d81a43076d2"
    )


def test_state_observation_slice_is_prospective_and_fail_closed() -> None:
    qualification = SOURCE_QUALIFICATION
    assert qualification.config_overlays == ("dmc_proprio", "debug")
    assert qualification.observation_mode == "proprioceptive_state"
    assert qualification.completed_gates == (
        "external_code_available_and_license_reviewed",
    )
    assert "isolated_runtime_locked" in qualification.blockers
    assert "assets_checksums_and_storage_approved" in qualification.blockers
    assert "external_execution_separately_authorized" in qualification.blockers
    assert qualification.source_acquired_read_only
    assert not qualification.runtime_built
    assert not qualification.workload_executed
    assert not qualification.paper_parity_claimed
    assert not qualification.scientific_promotion_allowed
    with pytest.raises(RuntimeError, match="source-qualified only"):
        qualification.require_execution_ready()


def test_qualification_rejects_claim_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="commit"):
        dataclasses.replace(SOURCE_QUALIFICATION, commit="0" * 40)
    with pytest.raises(ValueError, match="promotion"):
        dataclasses.replace(SOURCE_QUALIFICATION, scientific_promotion_allowed=True)
    with pytest.raises(ValueError, match="unresolved dependencies"):
        dataclasses.replace(SOURCE_QUALIFICATION, unresolved_dependencies=())
