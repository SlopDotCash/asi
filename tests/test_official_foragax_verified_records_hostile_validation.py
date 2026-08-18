"""Hostile input and boundary validation for official Foragax verification records."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.official_foragax import (
    OfficialForagaxBatchRun,
    OfficialForagaxRun,
    OfficialForagaxValidationError,
    VerifiedOfficialForagaxEvidence,
)


def _make_evidence() -> VerifiedOfficialForagaxEvidence:
    return VerifiedOfficialForagaxEvidence(
        manifest_path=Path("output/manifest.json"),
        manifest_sha256="a" * 64,
        manifest_kind="official_foragax_single",
        trust_descriptor_id="trust.v1",
        trust_descriptor_sha256="b" * 64,
        profile_id="profile.v1",
        profile_sha256="c" * 64,
        artifact_identities_sha256="d" * 64,
        endorsement_descriptor_id="endorsement.v1",
        endorsement_descriptor_sha256="e" * 64,
        endorsement_sha256="f" * 64,
    )


def test_verified_official_foragax_evidence_validation() -> None:
    ev = _make_evidence()
    assert ev.manifest_kind == "official_foragax_single"

    with pytest.raises(OfficialForagaxValidationError, match="manifest_path must be a Path"):
        VerifiedOfficialForagaxEvidence(
            manifest_path="output/manifest.json",  # type: ignore[arg-type]
            manifest_sha256="a" * 64,
            manifest_kind="official_foragax_single",
            trust_descriptor_id="trust.v1",
            trust_descriptor_sha256="b" * 64,
            profile_id="profile.v1",
            profile_sha256="c" * 64,
            artifact_identities_sha256="d" * 64,
            endorsement_descriptor_id="endorsement.v1",
            endorsement_descriptor_sha256="e" * 64,
            endorsement_sha256="f" * 64,
        )

    with pytest.raises(OfficialForagaxValidationError, match="manifest_kind must be"):
        VerifiedOfficialForagaxEvidence(
            manifest_path=Path("output/manifest.json"),
            manifest_sha256="a" * 64,
            manifest_kind="invalid_kind",  # type: ignore[arg-type]
            trust_descriptor_id="trust.v1",
            trust_descriptor_sha256="b" * 64,
            profile_id="profile.v1",
            profile_sha256="c" * 64,
            artifact_identities_sha256="d" * 64,
            endorsement_descriptor_id="endorsement.v1",
            endorsement_descriptor_sha256="e" * 64,
            endorsement_sha256="f" * 64,
        )


def test_official_foragax_run_validation() -> None:
    run = OfficialForagaxRun(
        manifest_path=Path("manifest.json"),
        artifact_path=Path("artifact.json"),
        manifest={"key": "val"},
        resumed=False,
    )
    assert run.resumed is False

    with pytest.raises(OfficialForagaxValidationError, match="resumed must be an exact boolean"):
        OfficialForagaxRun(
            manifest_path=Path("manifest.json"),
            artifact_path=Path("artifact.json"),
            manifest={"key": "val"},
            resumed=0,  # type: ignore[arg-type]
        )


def test_official_foragax_batch_run_validation() -> None:
    batch = OfficialForagaxBatchRun(
        manifest_path=Path("manifest.json"),
        artifact_paths=(Path("artifact_0.json"), Path("artifact_1.json")),
        manifest={"key": "val"},
        resumed=True,
    )
    assert batch.resumed is True

    with pytest.raises(
        OfficialForagaxValidationError, match="artifact_paths must be a tuple of Path instances"
    ):
        OfficialForagaxBatchRun(
            manifest_path=Path("manifest.json"),
            artifact_paths=[Path("artifact_0.json")],  # type: ignore[arg-type]
            manifest={"key": "val"},
            resumed=True,
        )
