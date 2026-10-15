"""Real CLI regressions for the review's consistently-invalid-shard counterexample."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from resource_policy import EXPECTED_SETTINGS, FROZEN_INPUTS, ResourcePolicyError, require_settings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


@pytest.mark.parametrize("key", EXPECTED_SETTINGS)
@pytest.mark.parametrize("bad", [None, "", "8", True, 1])
def test_policy_rejects_missing_noncanonical_or_nonstring(key, bad):
    settings = dict(EXPECTED_SETTINGS)
    if bad is None:
        settings.pop(key)
    else:
        settings[key] = bad
    with pytest.raises(ResourcePolicyError, match=key):
        require_settings(settings, context="fixture")


@pytest.mark.parametrize("key", EXPECTED_SETTINGS)
@pytest.mark.parametrize("missing", [False, True])
def test_run_cli_rejects_before_dispatch_or_output(tmp_path, key, missing):
    env = dict(os.environ, **EXPECTED_SETTINGS)
    if missing:
        env.pop(key)
    else:
        env[key] = "true" if key == "XLA_PYTHON_CLIENT_PREALLOCATE" else "8"
    output = tmp_path / "must-not-exist.json"
    result = subprocess.run([
        sys.executable, str(HERE / "run.py"), "--seed", "101001", "--arm", "gamma_090",
        "--out", str(output),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2 and key in result.stderr
    assert not output.exists()
    assert not output.with_suffix(".failure.json").exists()


@pytest.mark.parametrize("key", EXPECTED_SETTINGS)
@pytest.mark.parametrize("missing", [False, True])
def test_validation_rejects_consistently_invalid_six_shards(tmp_path, key, missing):
    records = tmp_path / "fixtures"
    records.mkdir()
    for path in sorted(FROZEN_INPUTS.glob("*-seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if missing:
            record["runtime"]["thread_settings"].pop(key)
        else:
            record["runtime"]["thread_settings"][key] = (
                "true" if key == "XLA_PYTHON_CLIENT_PREALLOCATE" else "8"
            )
        (records / path.name).write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "must-not-exist.json"
    result = subprocess.run([
        sys.executable, str(HERE / "validate.py"), "--records", str(records),
        "--preregistration", str(FROZEN_INPUTS / "preregistration.json"),
        "--source-root", str(ROOT), "--out", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2 and key in result.stderr
    assert not output.exists()


def test_compliant_retained_records_reproduce_original_summary(tmp_path):
    output = tmp_path / "summary.json"
    report = tmp_path / "resource-audit.json"
    result = subprocess.run([
        sys.executable, str(HERE / "validate.py"), "--records", str(FROZEN_INPUTS),
        "--preregistration", str(FROZEN_INPUTS / "preregistration.json"),
        "--source-root", str(ROOT), "--out", str(output), "--audit-report", str(report),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == (FROZEN_INPUTS / "summary.json").read_bytes()
    audit = json.loads(report.read_text())
    assert audit["compliant_comparison_records"] == 6
    assert audit["compliant_preflight_records"] == 2
    assert {item["path"] for item in audit["records"]} >= {
        "preflight/preflight-gamma_090.json",
        "preflight/preflight-gamma_099.json",
    }


def test_validation_rejects_noncompliant_nested_preflight(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    preflight = records / "preflight"
    preflight.mkdir()
    for path in sorted(FROZEN_INPUTS.glob("*-seed*.json")):
        shutil.copy2(path, records / path.name)
    for path in sorted((FROZEN_INPUTS / "preflight").glob("preflight-*.json")):
        shutil.copy2(path, preflight / path.name)
    bad = preflight / "preflight-gamma_090.json"
    record = json.loads(bad.read_text(encoding="utf-8"))
    record["runtime"]["thread_settings"]["OMP_NUM_THREADS"] = "8"
    bad.write_text(json.dumps(record), encoding="utf-8")

    output = tmp_path / "must-not-exist.json"
    result = subprocess.run([
        sys.executable, str(HERE / "validate.py"), "--records", str(records),
        "--preregistration", str(FROZEN_INPUTS / "preregistration.json"),
        "--source-root", str(ROOT), "--out", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2 and "preflight/preflight-gamma_090.json" in result.stderr
    assert not output.exists()


def test_compliant_run_help_forwards_without_running_a_life():
    result = subprocess.run([
        sys.executable, str(HERE / "run.py"), "--help",
    ], cwd=ROOT, env=dict(os.environ, **EXPECTED_SETTINGS), capture_output=True,
        text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "--preflight" in result.stdout
