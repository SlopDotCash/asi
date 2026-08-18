"""Hostile comparator gate for recurring feature artifact before hash."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import (
    CANONICALIZATION,
    DIGEST_ALGORITHM,
    DIGEST_SCOPE,
    SCHEMA_VERSION,
    validate_recurring_feature_artifact,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
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

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")


def _artifact_with_hostile_comparator(hostile: object) -> dict[str, object]:
    # Minimal artifact that reaches comparator validation
    # Keep top-level and scientific_payload keys to ensure _validate_checks is called
    hostile_check = {
        "name": "canonical_protocol",
        "passed": True,
        "actual": 1.0,
        "comparator": hostile,
        "threshold": 1.0,
        "detail": "dummy detail for test",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": {
            "protocol": {},
            "configuration": {},
            "criteria": {},
            "bootstrap": {},
            "memory_budget": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {
                "passed": True,
                "checks": [hostile_check],
                "upstream_summary": "test",
                "upstream_failures": [],
            },
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": "a" * 40,
                "source_sha256": {},
                "interpretation": "test",
            },
        },
        "scientific_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": "a" * 64,
        },
        "operational_metadata": {
            "gate_wall_seconds": 1.0,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "git_worktree": {"head": "a" * 40, "dirty": False},
            "runtime": {"python": "3.12", "platform": "test"},
            "protocol": {},
            "gate_result": {},
        },
    }


def test_recurring_comparator_rejects_hostile_before_hash() -> None:
    hostile = _HostileStr(">=")
    _HostileStr.calls = 0
    artifact = _artifact_with_hostile_comparator(hostile)  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    # Should not have called hostile hash/eq/bool/len
    assert _HostileStr.calls == 0
    assert validation.valid is False
    # ensure error contains generic message without repr dispatch
    for err in validation.errors:
        assert "_HostileStr" not in err
        assert "hostile" not in err.lower()


def test_recurring_comparator_rejects_hostile_unknown_before_hash() -> None:
    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    artifact = _artifact_with_hostile_comparator(hostile)  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_recurring_comparator_benign_still_works() -> None:
    # benign valid comparator should be processed without error about hash
    artifact = _artifact_with_hostile_comparator(">=")  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    # Will be invalid due to dummy payload, but should not crash and hostile count remains 0
    assert validation.valid is False
    # also test invalid comparator string
    artifact2 = _artifact_with_hostile_comparator("invalid_comparator")  # type: ignore[dict-item]
    validation2 = validate_recurring_feature_artifact(artifact2)  # type: ignore[arg-type]
    assert validation2.valid is False
    assert _HostileStr.calls == 0


def test_recurring_comparator_bool_subclass_rejected() -> None:
    artifact = _artifact_with_hostile_comparator(True)  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
