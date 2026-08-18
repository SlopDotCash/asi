"""Hostile string gate for scale robust git_head and digest before len."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
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


def _minimal_artifact_with_git_head(hostile: object) -> dict[str, object]:
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
            "acceptance": {"passed": False, "checks": [], "retention_ablation_interpretation": ""},
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": hostile,
                "source_sha256": {},
                "interpretation": "test",
            },
        },
        "content_digest": {
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
            "gate_result": {},
        },
    }


def _minimal_artifact_with_digest(hostile: object) -> dict[str, object]:

    # need at least one valid source path key to trigger digest loop
    # Use a dummy path that is in _SOURCE_PATHS; we can inspect it
    # But easiest: provide a mapping with one entry that will be validated
    # The hashes validation iterates over hashes.items() regardless of expected_paths,
    # so we can provide any dict with hostile value
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
            "acceptance": {"passed": False, "checks": [], "retention_ablation_interpretation": ""},
            "source_provenance": {
                "repository_subtree": "research/alberta",
                "git_head": "a" * 40,
                "source_sha256": {
                    "alberta_framework/evaluation/scale_robust_feature_artifact.py": hostile
                },
                "interpretation": "test",
            },
        },
        "content_digest": {
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
            "gate_result": {},
        },
    }


def test_scale_git_head_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 40)
    _HostileStr.calls = 0
    artifact = _minimal_artifact_with_git_head(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0
    for err in validation.errors:
        assert "_HostileStr" not in err


def test_scale_git_head_rejects_hostile_short_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _minimal_artifact_with_git_head(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_scale_digest_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("b" * 64)
    _HostileStr.calls = 0
    artifact = _minimal_artifact_with_digest(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_scale_benign_still_validates() -> None:
    artifact = _minimal_artifact_with_git_head("c" * 40)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False  # still invalid due to dummy payload but not crash
    artifact2 = _minimal_artifact_with_digest("d" * 64)  # type: ignore[dict-item]
    validation2 = validate_evidence_artifact(artifact2)  # type: ignore[arg-type]
    assert validation2.valid is False
    assert _HostileStr.calls == 0


def test_scale_non_string_rejected() -> None:
    for bad in [123, None, True, 12.3]:
        artifact = _minimal_artifact_with_git_head(bad)  # type: ignore[dict-item]
        validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
        assert validation.valid is False
