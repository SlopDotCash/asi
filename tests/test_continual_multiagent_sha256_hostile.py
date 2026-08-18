"""Hostile string gate for continual multiagent digest sha256 before len."""

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


def _minimal_artifact_with_digest(hostile_sha: object) -> dict[str, object]:
    # Minimal artifact that will reach digest validation.
    # Provide required top-level keys to avoid early returns, but content can be minimal.
    # The digest sha check is independent of content validity; we just need digest to be hostile.
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
            "sha256": hostile_sha,
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


def test_continual_digest_rejects_hostile_before_len() -> None:
    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    artifact = _minimal_artifact_with_digest(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert any(
        "content_digest.sha256 must be a 64-character string" in e
        for e in validation.errors
    ) or any("sha256" in e for e in validation.errors)
    assert _HostileStr.calls == 0
    # ensure no repr dispatch in errors
    for err in validation.errors:
        assert "_HostileStr" not in err
        assert "hostile" not in err.lower()


def test_continual_digest_rejects_hostile_wrong_length_before_len() -> None:
    hostile = _HostileStr("short")
    _HostileStr.calls = 0
    artifact = _minimal_artifact_with_digest(hostile)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_continual_digest_accepts_exact_str_still_validates() -> None:
    # benign exact string with wrong length should be caught without hostile
    artifact_bad = _minimal_artifact_with_digest("short")  # type: ignore[dict-item]
    validation_bad = validate_evidence_artifact(artifact_bad)  # type: ignore[arg-type]
    assert validation_bad.valid is False
    # benign exact 64-char still goes to digest mismatch path but not crash
    artifact_ok_len = _minimal_artifact_with_digest("a" * 64)  # type: ignore[dict-item]
    validation_ok = validate_evidence_artifact(artifact_ok_len)  # type: ignore[arg-type]
    # It will still be invalid due to other content errors, but must not have crashed on sha check
    assert validation_ok.valid is False
    assert _HostileStr.calls == 0


def test_continual_digest_rejects_bool_subclass_before_len() -> None:
    # bool is subclass of int, but for str leaf, passing bool should be rejected without len
    artifact_bool = _minimal_artifact_with_digest(True)  # type: ignore[dict-item]
    validation = validate_evidence_artifact(artifact_bool)  # type: ignore[arg-type]
    assert validation.valid is False
    assert _HostileStr.calls == 0


def test_continual_digest_non_string_types_rejected() -> None:
    for bad in [123, None, 12.34, b"bytes", ["a"]]:
        artifact = _minimal_artifact_with_digest(bad)  # type: ignore[dict-item]
        validation = validate_evidence_artifact(artifact)  # type: ignore[arg-type]
        assert validation.valid is False
