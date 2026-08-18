"""Hostile string gate for FTL generation_head before len."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.ftl_decision_artifact import (
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


def _minimal_ftl_artifact_with_git_head(hostile: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": {
            "protocol": {"kind": "test"},
            "configuration": {},
            "thresholds": {},
            "bootstrap": {},
            "memory": {},
            "seed_summaries": [],
            "aggregate": {},
            "acceptance": {"passed": False, "checks": []},
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": hostile,
                "source_sha256": {},
                "interpretation": "test",
            },
        },
        "scientific_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": "utf8-json-sort-keys-compact-no-nan",
            "sha256": "a" * 64,
        },
        "operational_metadata": {
            "gate_wall_seconds": 1.0,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "git_worktree": {"head": "a" * 40, "dirty": False},
            "runtime": {"python": "3.12"},
            "protocol": {},
        },
    }


def test_ftl_git_head_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 40)
    _HostileStr.calls = 0
    artifact = _minimal_ftl_artifact_with_git_head(hostile)  # type: ignore[dict-item]
    validation = validate_ftl_decision_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0
    for err in validation.errors:
        assert "_HostileStr" not in err


def test_ftl_git_head_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _minimal_ftl_artifact_with_git_head(hostile)  # type: ignore[dict-item]
    validation = validate_ftl_decision_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_ftl_git_head_benign_still_validates() -> None:
    artifact = _minimal_ftl_artifact_with_git_head("b" * 40)  # type: ignore[dict-item]
    validation = validate_ftl_decision_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_ftl_git_head_non_string_rejected() -> None:
    for bad in [123, None, True]:
        artifact = _minimal_ftl_artifact_with_git_head(bad)  # type: ignore[dict-item]
        validation = validate_ftl_decision_artifact(artifact)  # type: ignore[arg-type]
        assert validation.valid is False


def test_ftl_git_head_hostile_wrong_hex_before_len() -> None:
    # hostile subclass with valid length but invalid hex should still not call len
    hostile = _HostileStr("z" * 40)
    _HostileStr.calls = 0
    artifact = _minimal_ftl_artifact_with_git_head(hostile)  # type: ignore[dict-item]
    validation = validate_ftl_decision_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0
