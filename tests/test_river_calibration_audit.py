"""The RiverSwim audit must fail closed on missing source provenance."""

import importlib.util
import json
from pathlib import Path

import pytest


def _audit_module():
    path = (
        Path(__file__).parents[1]
        / "outputs/reference_life_scorecard_development"
        / "river-calibration-audit-2026-10-14/analyze.py"
    )
    spec = importlib.util.spec_from_file_location("river_calibration_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load RiverSwim audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_records_rejects_missing_source_identity(tmp_path: Path) -> None:
    audit = _audit_module()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    payload = {
        "schema": "asi.reference_life_scorecard.run.v1",
        "status": "completed",
        "failure": None,
        "environment_kind": "riverswim",
        "arm": "random",
        "seed": 1,
        "plan_sha256": audit.EXPECTED_PLAN_SHA256,
        "evidence_policy": {"permanently_nonpromoting": True},
        "outcome": {"accepted_events": 20000},
    }
    path = shard_dir / "riverswim__random__1.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid source identity"):
        audit.load_records(run=tmp_path, arms=("random",), seeds=(1,))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reproduction_verifies_retained_reports_without_writing() -> None:
    audit = _audit_module()
    before = (_sha256(audit.RETAINED_ANALYSIS), _sha256(audit.RETAINED_README))
    result = audit.verify_retained()
    after = (_sha256(audit.RETAINED_ANALYSIS), _sha256(audit.RETAINED_README))
    assert result["plan_sha256"] == audit.EXPECTED_PLAN_SHA256
    assert before == after


def test_write_output_refuses_occupied_and_retained_destinations(tmp_path: Path) -> None:
    audit = _audit_module()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "analysis.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        audit.write_output(occupied)
    with pytest.raises(ValueError, match="refusing to overwrite the retained"):
        audit.write_output(audit.OUT)


def test_write_output_to_fresh_dir_matches_retained(tmp_path: Path) -> None:
    audit = _audit_module()
    fresh = tmp_path / "fresh"
    audit.write_output(fresh)
    assert (fresh / "analysis.json").read_bytes() == audit.RETAINED_ANALYSIS.read_bytes()
    assert (fresh / "README.md").read_bytes() == audit.RETAINED_README.read_bytes()
