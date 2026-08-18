"""Hostile input and boundary validation for open protocol qualifications."""

from __future__ import annotations

import hashlib

import pytest

from alberta_framework.benchmarks.forager_matched_open_protocol import (
    ForagerMatchedOpenProtocolBuildError,
    MatchedCurrentCandidateQualification,
    MatchedCurrentRuntimeQualification,
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_runtime_qualification_valid_construction() -> None:
    qual = MatchedCurrentRuntimeQualification(
        image_sha256=_digest(b"image"),
        runtime_profile_sha256=_digest(b"profile"),
        executor_qualification_receipt_sha256=_digest(b"executor"),
        qualification_trust_anchor_identity="anchor_id",
    )
    assert qual.qualification_trust_anchor_identity == "anchor_id"


def test_runtime_qualification_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedOpenProtocolBuildError, match="must be a lowercase SHA-256"
    ):
        MatchedCurrentRuntimeQualification(
            image_sha256="invalid",
            runtime_profile_sha256=_digest(b"profile"),
            executor_qualification_receipt_sha256=_digest(b"executor"),
            qualification_trust_anchor_identity="anchor_id",
        )

    with pytest.raises(
        ForagerMatchedOpenProtocolBuildError, match="is a placeholder, not a real content binding"
    ):
        MatchedCurrentRuntimeQualification(
            image_sha256="a" * 64,
            runtime_profile_sha256=_digest(b"profile"),
            executor_qualification_receipt_sha256=_digest(b"executor"),
            qualification_trust_anchor_identity="anchor_id",
        )

    with pytest.raises(
        ForagerMatchedOpenProtocolBuildError, match="must be a non-empty string"
    ):
        MatchedCurrentRuntimeQualification(
            image_sha256=_digest(b"image"),
            runtime_profile_sha256=_digest(b"profile"),
            executor_qualification_receipt_sha256=_digest(b"executor"),
            qualification_trust_anchor_identity="",
        )


def test_candidate_qualification_rejects_invalid_types() -> None:
    with pytest.raises(
        ForagerMatchedOpenProtocolBuildError, match="candidate source must be a SourceBinding"
    ):
        MatchedCurrentCandidateQualification(
            source=None,  # type: ignore[arg-type]
            configuration=None,  # type: ignore[arg-type]
            effective_seed_proof_sha256=_digest(b"seed_proof"),
            capability_qualification_receipt_sha256=_digest(b"receipt"),
            resources=None,  # type: ignore[arg-type]
        )
