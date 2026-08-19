"""Hostile string gate for continual_ia digest before len."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import (
    CANONICALIZATION,
    DIGEST_SCOPE,
    SCHEMA_VERSION,
    validate_ia_evidence_artifact,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile len must not run")

    def __str__(self) -> str:
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _artifact_with_digest_hostile(hostile: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "content": {
            "protocol": {},
            "configuration": {},
            "thresholds": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {},
            "provenance": {},
        },
        "content_digest": {
            "algorithm": "sha256",
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": hostile,
        },
        "operational_diagnostics": {
            "digest_exclusion_reason": "test",
            "environment": "test",
            "condition_timings": {},
            "maximum_update_latency_ms": 0,
            "checks": [],
            "passed": False,
            "overall_acceptance_passed": False,
        },
    }


def test_continual_ia_digest_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)
    validation = validate_ia_evidence_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_continual_ia_digest_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)
    validation = validate_ia_evidence_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_continual_ia_digest_benign_still_validates() -> None:
    artifact_bad = _artifact_with_digest_hostile("short")
    validation_bad = validate_ia_evidence_artifact(artifact_bad)
    assert validation_bad.valid is False
    assert _HostileStr.calls == 0


def test_continual_ia_digest_non_string_types_rejected() -> None:
    for bad in [123, None, True, b"bytes", ["a" * 64]]:
        artifact = _artifact_with_digest_hostile(bad)
        validation = validate_ia_evidence_artifact(artifact)
        assert validation.valid is False
