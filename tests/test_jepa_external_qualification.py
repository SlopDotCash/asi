"""Fail-closed tests for the external JEPA source/checkpoint inventory."""

from __future__ import annotations

import json

import pytest

from alberta_framework.benchmarks import jepa_transfer_feasibility as lane
from alberta_framework.benchmarks.jepa_external_qualification import (
    JEPACheckpointPin,
    JEPAExternalQualification,
    JEPASourcePin,
    catalog_payload,
    validate_catalog_payload,
)

pytestmark = pytest.mark.unit


def test_sources_bind_exact_trees_archives_and_distinct_licenses() -> None:
    qualification = JEPAExternalQualification()
    qualification.validate()
    jepa_wm, vjepa2 = qualification.sources
    assert isinstance(jepa_wm, JEPASourcePin)
    assert jepa_wm.tree == "23f381d7a8a934b006d7cdfc5620a8af29fd20a4"
    assert jepa_wm.license_id == "CC-BY-NC-4.0"
    assert vjepa2.tree == "dd6cfc1e792158510b983d827cb2e84f47fd5706"
    assert vjepa2.license_id == "MIT"
    assert vjepa2.additional_license_id == "Apache-2.0"


def test_relevant_checkpoint_objects_are_content_bound_without_download() -> None:
    qualification = JEPAExternalQualification()
    assert tuple(pin.filename for pin in qualification.checkpoints) == (
        "jepa_wm_droid.pth.tar",
        "vjepa2_ac_droid.pth.tar",
        "vjepa2_ac_oss.pth.tar",
    )
    assert all(isinstance(pin, JEPACheckpointPin) for pin in qualification.checkpoints)
    assert qualification.checkpoints[0].sha256 == (
        "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"
    )
    assert qualification.total_checkpoint_bytes == 11_678_963_913


def test_license_asset_and_execution_gates_remain_closed() -> None:
    qualification = JEPAExternalQualification()
    assert qualification.model_repository_license == "CC-BY-NC-4.0"
    assert qualification.checkpoints_downloaded is False
    assert qualification.dataset_inventory_content_bound is False
    assert qualification.droid_rights_review_complete is False
    assert qualification.external_execution_authorized is False
    assert qualification.physical_execution_authorized is False
    assert qualification.paper_parity_allowed is False


def test_catalog_round_trips_as_strict_primitive_json() -> None:
    payload = catalog_payload()
    decoded = json.loads(json.dumps(payload))
    assert validate_catalog_payload(decoded) == decoded
    decoded["external_execution_authorized"] = True
    with pytest.raises(ValueError, match="catalog"):
        validate_catalog_payload(decoded)
    with pytest.raises(ValueError, match="string"):
        validate_catalog_payload({"hostile": "x" * 100_001})
    with pytest.raises(ValueError, match="finite"):
        validate_catalog_payload({"hostile": float("nan")})


def test_existing_cli_emits_external_inventory_without_running_native_lane(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lane.main(("--external-catalog",)) == 0
    payload = json.loads(capsys.readouterr().out)
    validate_catalog_payload(payload)
    assert payload["checkpoints_downloaded"] is False
