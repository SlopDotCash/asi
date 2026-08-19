"""Hostile string gate for continual_multiagent check name before bool."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    CANONICALIZATION,
    DIGEST_ALGORITHM,
    DIGEST_SCOPE,
    SCHEMA_VERSION,
    validate_evidence_artifact,
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


def _artifact_with_name_hostile(hostile: object) -> dict[str, object]:
    hostile_check = {
        "name": hostile,
        "passed": True,
        "actual": 1.0,
        "comparator": ">=",
        "threshold": 1.0,
        "detail": "dummy",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "content": {
            "protocol": {},
            "configuration": {},
            "thresholds": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {},
            "source_provenance": {},
        },
        "content_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": "a" * 64,
        },
        "operational_diagnostics": {
            "digest_exclusion_reason": "test",
            "environment": "test",
            "condition_timings": {},
            "maximum_update_latency_ms": 0,
            "checks": [hostile_check],
            "passed": False,
            "overall_acceptance_passed": False,
        },
    }


def test_multiagent_name_rejects_hostile_before_bool() -> None:
    hostile = _HostileStr("test")
    _HostileStr.calls = 0
    artifact = _artifact_with_name_hostile(hostile)
    validation = validate_evidence_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_multiagent_name_rejects_hostile_empty_before_bool() -> None:
    hostile = _HostileStr("")
    _HostileStr.calls = 0
    artifact = _artifact_with_name_hostile(hostile)
    validation = validate_evidence_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_multiagent_name_benign_still_validates() -> None:
    artifact_bad = _artifact_with_name_hostile("")
    validation_bad = validate_evidence_artifact(artifact_bad)
    assert validation_bad.valid is False
    assert _HostileStr.calls == 0


def test_multiagent_name_non_string_rejected() -> None:
    for bad in [123, None, True, b"bytes", ["test"]]:
        artifact = _artifact_with_name_hostile(bad)
        validation = validate_evidence_artifact(artifact)
        assert validation.valid is False
