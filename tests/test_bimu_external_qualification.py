"""Fail-closed coverage for the official BiMU source qualification."""

from __future__ import annotations

import copy

import pytest

from alberta_framework.benchmarks.bimu_external_qualification import (
    BiMUExternalQualification,
    qualification_payload,
    validate_qualification_payload,
)

pytestmark = pytest.mark.unit


def test_official_source_and_license_are_content_bound() -> None:
    qualification = BiMUExternalQualification()
    qualification.validate()
    assert qualification.commit == "1b8a1a1fb892fbe89401390b3ff9611d7f3a5168"
    assert qualification.tree == "cbeeb50cdd3421fc046e7a2b73e26147419227e9"
    assert qualification.archive_sha256 == (
        "452a5b573160de80b3c3a73e6ef875c702f4560581b358c0758e2857886ff87b"
    )
    assert qualification.license_id == "CC-BY-4.0"
    assert qualification.license_sha256 == (
        "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"
    )


def test_paper_configuration_and_implementation_files_are_bound() -> None:
    qualification = BiMUExternalQualification()
    files = {item.path: item.sha256 for item in qualification.protocol_files}
    assert files["configurations/main-pmnist-1000tasks-100neurons/bimu.json"] == (
        "30eef43939443099fea396c8258de8e7f7336ccb5fd84e4118af2921314b3211"
    )
    assert files["optimizers/bimu.py"] == (
        "c0a247e341bdbf53e82fad88ed0fcaea3fe72edeb1b4c6e9bf6ef7c3bf6f7f7f"
    )
    assert qualification.official_environment_fully_locked is False
    assert qualification.external_execution_authorized is False
    assert qualification.paper_parity_allowed is False


def test_qualification_payload_is_strict_and_self_bound() -> None:
    payload = qualification_payload()
    assert validate_qualification_payload(copy.deepcopy(payload)) == payload
    forged = copy.deepcopy(payload)
    forged["external_execution_authorized"] = True
    with pytest.raises(ValueError, match="qualification"):
        validate_qualification_payload(forged)

