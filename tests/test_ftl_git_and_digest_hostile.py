"""Hostile string gate for ftl git_worktree and digest before len."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.ftl_decision_artifact import (
    CANONICALIZATION,
    DIGEST_ALGORITHM,
    DIGEST_SCOPE,
    SCHEMA_VERSION,
    validate_ftl_decision_artifact,
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
        "scientific_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": "a" * 64,
        },
        "operational_metadata": {
            "gate_wall_seconds": 1.0,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "git_worktree": {"head": hostile, "dirty": False},
            "runtime": {"python": "3.12", "platform": "test"},
            "protocol": {},
            "gate_result": {},
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
        "scientific_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": hostile,
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


def test_ftl_git_worktree_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 40)
    _HostileStr.calls = 0
    artifact = _artifact_with_git_hostile(hostile)
    validation = validate_ftl_decision_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_ftl_git_worktree_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _artifact_with_git_hostile(hostile)
    validation = validate_ftl_decision_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_ftl_digest_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)
    validation = validate_ftl_decision_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_ftl_digest_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _artifact_with_digest_hostile(hostile)
    validation = validate_ftl_decision_artifact(artifact)
    assert _HostileStr.calls == 0
    assert validation.valid is False


def test_ftl_benign_still_validates() -> None:
    artifact_bad = _artifact_with_git_hostile("short")
    validation_bad = validate_ftl_decision_artifact(artifact_bad)
    assert validation_bad.valid is False
    artifact_bad2 = _artifact_with_digest_hostile("short")
    validation_bad2 = validate_ftl_decision_artifact(artifact_bad2)
    assert validation_bad2.valid is False
    assert _HostileStr.calls == 0


def test_ftl_non_string_types_rejected() -> None:
    for bad in [123, None, True, b"bytes", ["a" * 40]]:
        artifact = _artifact_with_git_hostile(bad)
        validation = validate_ftl_decision_artifact(artifact)
        assert validation.valid is False
        artifact2 = _artifact_with_digest_hostile(bad)
        validation2 = validate_ftl_decision_artifact(artifact2)
        assert validation2.valid is False
