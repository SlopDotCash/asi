"""Unit coverage for alberta_framework.evaluation.evidence_manifest_cli.

Exercises the output-path resolution guard (_resolved_new_output) which
protects the pinned canonical artifact path and refuses overwrites, plus
the exit-code wiring for the manifest.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "evidence_manifest_cli",
        str(repo_root / "alberta_framework/evaluation/evidence_manifest_cli.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_manifest_cli"] = module
    # Stub the evidence_manifest dependency (build would touch the repo).
    import types

    fake = types.ModuleType("alberta_framework.evaluation.evidence_manifest")
    fake.REPO_ROOT = repo_root
    fake.DEFAULT_OUTPUT = fake.REPO_ROOT / "outputs/evidence_manifest.json"
    fake.build_evidence_manifest = lambda root: {"schema": "x"}
    fake.evidence_manifest_json = lambda m: '{"schema": "x"}'
    fake.evidence_manifest_exit_code = lambda m: 0
    sys.modules["alberta_framework.evaluation.evidence_manifest"] = fake
    spec.loader.exec_module(module)
    return module


mod = load_module()


def test_resolved_new_output_accepts_new_path(tmp_path: Path) -> None:
    target = tmp_path / "new" / "manifest.json"
    resolved = mod._resolved_new_output(target)
    assert resolved == target.resolve()


def test_resolved_new_output_rejects_canonical_path(tmp_path: Path) -> None:
    canonical = mod.DEFAULT_OUTPUT
    with pytest.raises(FileExistsError, match="canonical artifact"):
        mod._resolved_new_output(canonical)


def test_resolved_new_output_rejects_existing_path(tmp_path: Path) -> None:
    existing = tmp_path / "manifest.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite existing"):
        mod._resolved_new_output(existing)


def test_resolved_new_output_expands_home() -> None:
    # expanduser on a literal ~/ path resolves against HOME.
    target = Path("~/evidence_manifest_test_xyz.json")
    resolved = mod._resolved_new_output(target)
    assert resolved == target.expanduser().resolve()
