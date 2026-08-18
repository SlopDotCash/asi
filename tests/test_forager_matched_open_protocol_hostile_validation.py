"""Hostile input and boundary validation for open protocol qualifications."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from alberta_framework.benchmarks.forager_matched_open_protocol import (
    _CANDIDATE_SPECS,
    MATCHED_CURRENT_HORIZON,
    ForagerMatchedOpenProtocolBuildError,
    MatchedCurrentCandidateQualification,
    MatchedCurrentRuntimeQualification,
    _CandidateSpec,
    _require_real_sha256,
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


def test_real_digest_rejects_string_subclass_before_hooks() -> None:
    calls = 0

    class HostileDigest(str):
        def __bool__(self) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("truth hook reached")

        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("comparison hook reached")

        __hash__ = str.__hash__

    with pytest.raises(ForagerMatchedOpenProtocolBuildError, match="SHA-256"):
        _require_real_sha256(HostileDigest(_digest(b"real")), "digest")
    assert calls == 0


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


def _legal_spec(candidate_id: str = "isolated_ppo") -> _CandidateSpec:
    return next(spec for spec in _CANDIDATE_SPECS if spec.candidate_id == candidate_id)


def test_frozen_candidate_specs_remain_legal() -> None:
    assert len(_CANDIDATE_SPECS) == 23
    isolated_ppo = _legal_spec("isolated_ppo")
    isolated_rtu = _legal_spec("isolated_rtu")
    oracle = _legal_spec("search_oracle")
    causal = _legal_spec("causal_e025_q050")
    assert isolated_ppo.rollout_steps == 2_048
    assert MATCHED_CURRENT_HORIZON // isolated_ppo.rollout_steps == 244
    assert isolated_rtu.rollout_steps == 128
    assert MATCHED_CURRENT_HORIZON // isolated_rtu.rollout_steps == 3_904
    assert causal.rollout_steps is None
    assert isolated_ppo.aperture_size == 9
    assert oracle.aperture_size == -1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_steps", True),
        ("rollout_steps", False),
        ("rollout_steps", 0),
        ("aperture_size", True),
        ("aperture_size", False),
        ("aperture_size", 0),
        ("aperture_size", 8),
        ("environment_key_shared", 1),
        ("pairing_eligible", 1),
        ("seed_transport", "not_a_transport"),
        ("candidate_id", ""),
    ],
)
def test_candidate_spec_rejects_bool_horizon_identities(
    field: str, value: object
) -> None:
    with pytest.raises(ForagerMatchedOpenProtocolBuildError, match=field):
        replace(_legal_spec(), **{field: value})


def test_candidate_spec_bool_rollout_would_collapse_horizon_identity() -> None:
    """On origin/main, True constructed and MATCHED_CURRENT_HORIZON // True == 499712."""

    legal = _legal_spec("isolated_ppo")
    assert MATCHED_CURRENT_HORIZON // legal.rollout_steps == 244
    assert MATCHED_CURRENT_HORIZON // True == MATCHED_CURRENT_HORIZON
    with pytest.raises(ForagerMatchedOpenProtocolBuildError, match="rollout_steps"):
        replace(legal, rollout_steps=True)
    with pytest.raises(ForagerMatchedOpenProtocolBuildError, match="aperture_size"):
        replace(legal, aperture_size=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"rollout_steps": 127},
        {"rollout_steps": None},
        {"source_base_commit": "A" * 40},
        {"environment_key_shared": True},
        {"privileged_fields": ("global_objects",)},
        {"pairing_eligible": False},
    ],
)
def test_candidate_spec_rejects_cross_field_and_provenance_drift(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ForagerMatchedOpenProtocolBuildError):
        replace(_legal_spec(), **updates)
