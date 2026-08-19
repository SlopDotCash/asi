"""Hostile string gate for recurring feature artifact name/detail before bool."""

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


def _artifact_with_name_hostile(hostile: object) -> dict[str, object]:
    hostile_check = {
        "name": hostile,
        "passed": True,
        "actual": 1.0,
        "comparator": ">=",
        "threshold": 1.0,
        "detail": "dummy detail",
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


def _artifact_with_detail_hostile(hostile: object) -> dict[str, object]:
    hostile_check = {
        "name": "canonical_protocol",
        "passed": True,
        "actual": 1.0,
        "comparator": ">=",
        "threshold": 1.0,
        "detail": hostile,
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


def test_recurring_name_rejects_hostile_before_bool() -> None:
    hostile = _HostileStr("test_name")
    _HostileStr.calls = 0
    artifact = _artifact_with_name_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_recurring_detail_rejects_hostile_before_bool() -> None:
    hostile = _HostileStr("test detail")
    _HostileStr.calls = 0
    artifact = _artifact_with_detail_hostile(hostile)  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_recurring_name_detail_empty_string_still_rejected() -> None:
    artifact = _artifact_with_name_hostile("")  # type: ignore[dict-item]
    validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    artifact2 = _artifact_with_detail_hostile("")  # type: ignore[dict-item]
    validation2 = validate_recurring_feature_artifact(artifact2)  # type: ignore[arg-type]
    assert validation2.valid is False
    assert _HostileStr.calls == 0


def test_recurring_name_detail_non_string_rejected() -> None:
    for bad in [123, None, True, b"bytes"]:
        artifact = _artifact_with_name_hostile(bad)  # type: ignore[dict-item]
        validation = validate_recurring_feature_artifact(artifact)  # type: ignore[arg-type]
        assert validation.valid is False
        artifact2 = _artifact_with_detail_hostile(bad)  # type: ignore[dict-item]
        validation2 = validate_recurring_feature_artifact(artifact2)  # type: ignore[arg-type]
        assert validation2.valid is False
