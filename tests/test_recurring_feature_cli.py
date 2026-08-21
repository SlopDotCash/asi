"""Unit coverage for alberta_framework.evaluation.recurring_feature_cli.

Exercises the immutable-output guard (_resolved_new_output) and the verify
path's error classification (_verify), with the heavy gate/artifact
dependencies stubbed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recurring_feature_cli",
        str(repo_root / "alberta_framework/evaluation/recurring_feature_cli.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["recurring_feature_cli"] = module

    # Stub artifact + gate dependencies.
    fake_artifact = types.ModuleType(
        "alberta_framework.evaluation.recurring_feature_artifact"
    )
    fake_artifact.load_recurring_feature_artifact = lambda path: {
        "schema": "recurring-feature.v1"
    }
    fake_artifact.validate_recurring_feature_artifact = lambda a: types.SimpleNamespace(
        accepted=True, valid=True, errors=[]
    )
    fake_artifact.write_recurring_feature_artifact = lambda *a, **k: {
        "scientific_digest": {"sha256": "abc123"}
    }
    sys.modules[
        "alberta_framework.evaluation.recurring_feature_artifact"
    ] = fake_artifact

    fake_gate = types.ModuleType("alberta_framework.recurring_feature_gate")
    fake_gate.run_recurring_feature_gate = lambda: types.SimpleNamespace()
    fake_gate.RecurringFeatureGateResult = type("RecurringFeatureGateResult", (), {})
    sys.modules["alberta_framework.recurring_feature_gate"] = fake_gate

    spec.loader.exec_module(module)
    return module


mod = load_module()


def test_resolved_new_output_accepts_new_path(tmp_path: Path) -> None:
    target = tmp_path / "new" / "evidence.json"
    assert mod._resolved_new_output(target) == target.resolve()


def test_resolved_new_output_rejects_canonical(tmp_path: Path) -> None:
    canonical = mod.DEFAULT_OUTPUT.expanduser().resolve()
    with pytest.raises(FileExistsError, match="canonical"):
        mod._resolved_new_output(canonical)


def test_resolved_new_output_rejects_existing(tmp_path: Path) -> None:
    existing = tmp_path / "evidence.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        mod._resolved_new_output(existing)


def test_verify_emits_accepted(tmp_path: Path, capsys) -> None:
    f = tmp_path / "evidence.json"
    f.write_text("{}", encoding="utf-8")
    code = mod._verify(f)
    assert code == 0
    out = capsys.readouterr().out
    assert '"accepted": true' in out
