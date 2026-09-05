"""Fail-closed contracts for the non-executing Forager rerun preflight."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from alberta_framework.benchmarks import forager_scientific_rerun_preflight as preflight

pytestmark = pytest.mark.integration
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runner(
    returncode: int, stdout: bytes = b"", stderr: bytes = b""
) -> preflight.ProcessRunner:
    def run(command: Sequence[str]) -> preflight.ProcessResult:
        assert "image" in command
        assert "inspect" in command
        assert "run" not in command
        assert "pull" not in command
        return preflight.ProcessResult(returncode, stdout, stderr)

    return run


def test_exact_frozen_schedule_and_whole_campaign_accounting() -> None:
    plan = preflight.build_run_plan()
    assert len(plan.candidate_ids) == 21
    assert len(plan.seeds) == 10
    assert plan.cell_count == 210
    assert plan.horizon_per_cell == 499_712
    assert plan.total_environment_resets == 210
    assert plan.total_environment_transitions == 104_939_520
    assert plan.total_agent_action_queries == 104_939_520
    assert plan.total_observation_deliveries == 104_939_730
    assert plan.timing_is_telemetry_only
    assert plan.development_only and not plan.promotion_authorized
    assert plan.negative_outcomes_append_only
    assert "persistent_numeric_bytes_peak" in plan.required_dynamic_resource_fields
    assert "model_query_count" in plan.required_dynamic_resource_fields


def test_pinned_records_contain_config_id_but_no_registry_resolver() -> None:
    records = preflight.audit_pinned_identity_records(_PROJECT_ROOT)
    assert tuple(record.path for record in records) == preflight.PINNED_IDENTITY_RECORDS
    assert all(record.required_image_id_occurrences > 0 for record in records)
    assert all(not record.registry_references for record in records)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_pinned_record_audit_rejects_links(tmp_path: Path, link_kind: str) -> None:
    relative = Path(preflight.PINNED_IDENTITY_RECORDS[0])
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    source = _PROJECT_ROOT / relative
    backing = tmp_path / "identity-record.json"
    backing.write_bytes(source.read_bytes())
    if link_kind == "symlink":
        destination.symlink_to(backing)
        with pytest.raises(preflight.ForagerScientificRerunPreflightError):
            preflight.audit_pinned_identity_records(tmp_path)
    else:
        os.link(backing, destination)
        try:
            with pytest.raises(preflight.ForagerScientificRerunPreflightError):
                preflight.audit_pinned_identity_records(tmp_path)
        finally:
            destination.unlink()


def test_current_checkout_is_precisely_blocked_without_runtime_or_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight.shutil.which",
        lambda _runtime: None,
    )
    report = preflight.build_preflight_report(_PROJECT_ROOT)
    assert report.local_image.status == "runtime_unavailable"
    assert len(report.preflight_source_sha256) == 64
    assert report.registry_reference is None
    assert not report.launch_authorized
    assert any("registry" in blocker for blocker in report.blockers)
    assert any("current-source" in blocker for blocker in report.blockers)
    assert any("byte/query" in blocker for blocker in report.blockers)
    assert any("output namespace" in blocker for blocker in report.blockers)


def test_local_inspection_uses_only_image_inspect_and_accepts_exact_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight.shutil.which",
        lambda _runtime: "/usr/bin/docker",
    )
    inspection = preflight.inspect_local_image(
        runner=_runner(0, json.dumps(preflight.REQUIRED_IMAGE_ID).encode("ascii"))
    )
    assert inspection.status == "exact_present"
    assert inspection.observed_image_id == preflight.REQUIRED_IMAGE_ID

    report = preflight.build_preflight_report(
        _PROJECT_ROOT,
        runner=_runner(0, json.dumps(preflight.REQUIRED_IMAGE_ID).encode("ascii")),
    )
    assert not any("registry" in blocker for blocker in report.blockers)
    assert len(report.blockers) == 3


def test_local_inspection_reports_absent_and_rejects_wrong_or_hostile_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight.shutil.which",
        lambda _runtime: "/usr/bin/docker",
    )
    assert preflight.inspect_local_image(runner=_runner(1, stderr=b"not found")).status == (
        "image_absent"
    )
    wrong = "sha256:" + "1" * 64
    inspection = preflight.inspect_local_image(
        runner=_runner(0, json.dumps(wrong).encode("ascii"))
    )
    assert inspection.status == "inspection_failed"
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="invalid JSON"):
        preflight.inspect_local_image(runner=_runner(0, b"not-json"))
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="wrong exact type"):
        preflight.inspect_local_image(runner=lambda _command: object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "runtime_executable", "observed_image_id"),
    [
        ("exact_present", None, preflight.REQUIRED_IMAGE_ID),
        ("runtime_unavailable", "/usr/bin/docker", None),
        ("image_absent", "/usr/bin/docker", "sha256:" + "1" * 64),
    ],
)
def test_local_image_record_rejects_impossible_runtime_and_observation_states(
    status: preflight.LocalStatus,
    runtime_executable: str | None,
    observed_image_id: str | None,
) -> None:
    with pytest.raises(preflight.ForagerScientificRerunPreflightError):
        preflight.LocalImageInspection(
            status=status,
            runtime_executable=runtime_executable,
            observed_image_id=observed_image_id,
            detail="invalid cross-field state",
        )


def test_hostile_plan_and_report_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = preflight.build_run_plan()
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="frozen contract"):
        preflight.validate_run_plan(dataclasses.replace(plan, cell_count=209))
    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight.shutil.which",
        lambda _runtime: None,
    )
    report = preflight.build_preflight_report(_PROJECT_ROOT)
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="fail-closed"):
        preflight.validate_preflight_report(dataclasses.replace(report, launch_authorized=True))
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="fail-closed"):
        preflight.validate_preflight_report(
            dataclasses.replace(report, preflight_source_sha256="1" * 64)
        )
    forged_record = dataclasses.replace(report.pinned_records[0], sha256="1" * 64)
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="digest closure"):
        preflight.validate_preflight_report(
            dataclasses.replace(report, pinned_records=(forged_record, *report.pinned_records[1:]))
        )
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="blocker closure"):
        preflight.validate_preflight_report(dataclasses.replace(report, blockers=("forged",)))
    forged = object.__new__(preflight.PreflightReport)
    for field in dataclasses.fields(report):
        object.__setattr__(forged, field.name, getattr(report, field.name))
    object.__setattr__(forged, "pinned_records", (object(),))
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="pinned records"):
        preflight.validate_preflight_report(forged)
    with pytest.raises(preflight.ForagerScientificRerunPreflightError, match="exact"):
        preflight.validate_run_plan(dataclasses.asdict(plan))


def test_cli_emits_machine_readable_blocker_and_never_authorizes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "alberta_framework.benchmarks.forager_scientific_rerun_preflight.shutil.which",
        lambda _runtime: None,
    )
    assert preflight.main(["--project-root", str(_PROJECT_ROOT)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["launch_authorized"] is False
    assert payload["plan"]["cell_count"] == 210
    assert payload["registry_reference"] is None
