"""Hostile string gate for scale robust operational git_head and digest before len."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    CANONICALIZATION,
    DIGEST_ALGORITHM,
    DIGEST_SCOPE,
    SCHEMA_VERSION,
    validate_evidence_artifact,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _artifact_with_git_hostile(hostile: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": {
            "protocol": {},
            "configuration": {},
            "thresholds": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {},
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": "a" * 40,
                "source_sha256": {},
                "interpretation": "test",
            },
        },
        "content_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": "a" * 64,
        },
        "operational_metadata": {
            "git_head": hostile,
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "wall_time_seconds_by_condition": {},
            "interpretation": "test",
        },
    }


def _artifact_with_digest_hostile(hostile: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": {
            "protocol": {},
            "configuration": {},
            "thresholds": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {},
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": "a" * 40,
                "source_sha256": {},
                "interpretation": "test",
            },
        },
        "content_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": hostile,
        },
        "operational_metadata": {
            "git_head": "a" * 40,
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "wall_time_seconds_by_condition": {},
            "interpretation": "test",
        },
    }


def test_scale_operational_git_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 40)
    _HostileStr.calls = 0
    artifact = _artifact_with_git_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_scale_operational_git_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _artifact_with_git_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_scale_digest_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_scale_digest_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_scale_benign_still_validates() -> None:
    artifact_bad = _artifact_with_git_hostile("short")  # type: ignore[dict-item]
    validation_bad = validate_evidence_artifact(artifact_bad)  # type: ignore[arg-type]
    assert validation_bad.valid is False
    artifact_bad2 = _artifact_with_digest_hostile("short")  # type: ignore[dict-item]
    validation_bad2 = validate_evidence_artifact(artifact_bad2)  # type: ignore[arg-type]
    assert validation_bad2.valid is False
    assert _HostileStr.calls == 0
