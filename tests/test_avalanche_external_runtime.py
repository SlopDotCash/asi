from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from alberta_framework.benchmarks.external_qualification import qualification_plan
from alberta_framework.benchmarks.native_supervised_suite import (
    AVALANCHE_QUALIFICATION_PLAN_SHA256,
)

pytestmark = pytest.mark.unit

ROOT = Path("external_runtimes/avalanche_native_suite")


class _HookMapping(dict[object, object]):
    calls = 0

    def keys(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook ran")


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "avalanche_runtime_verifier_test", ROOT / "verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan() -> dict[str, object]:
    value = json.loads((ROOT / "qualification-plan.json").read_bytes())
    assert type(value) is dict
    return value


def test_plan_binds_official_source_license_and_every_runtime_input() -> None:
    assert hashlib.sha256((ROOT / "qualification-plan.json").read_bytes()).hexdigest() == (
        AVALANCHE_QUALIFICATION_PLAN_SHA256
    )
    plan = _plan()
    authority = plan["authority"]
    inputs = plan["qualification_inputs"]
    assert type(authority) is dict and type(inputs) is dict
    revision = qualification_plan(1578).code_revisions[0]
    assert authority["paper_revision"] == qualification_plan(1578).paper_revisions[0]
    assert authority["repository"] == revision.repository
    assert authority["commit"] == revision.commit
    assert authority["git_tree"] == "fdfe9d9b4578587bf83a3970eaaf9701bb3db2a6"
    assert authority["license"] == "MIT"
    files = {
        "dockerfile_sha256": "Dockerfile",
        "requirements_in_sha256": "requirements.in",
        "requirements_lock_sha256": "requirements.lock",
        "fetch_source_sha256": "fetch_source.py",
        "verify_runtime_sha256": "verify_runtime.py",
    }
    for field, relative in files.items():
        assert inputs[field] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_runtime_is_hash_locked_data_free_and_never_authorizes_execution() -> None:
    plan = _plan()
    runtime = plan["runtime"]
    diagnostic = plan["prospective_diagnostic"]
    claims = plan["claims"]
    assert type(runtime) is dict and type(diagnostic) is dict and type(claims) is dict
    assert runtime["torch"] == "2.2.2+cpu"
    assert runtime["numpy"] == "1.26.4"
    assert runtime["avalanche"] == "0.6.0a"
    assert runtime["uid"] == 65_532 and runtime["gid"] == 65_532
    assert runtime["home"] == "/tmp/asi-runtime-home"
    assert runtime["pip"] == "23.0.1"
    assert runtime["wheel"] == "0.44.0"
    source_build_exceptions = runtime["source_build_exceptions"]
    assert type(source_build_exceptions) is list and len(source_build_exceptions) == 1
    assert source_build_exceptions[0]["distribution"] == "gputil"
    assert source_build_exceptions[0]["source_sha256"] == (
        "099e52c65e512cdfa8c8763fca67f5a5c2afb63469602d5dcb4d296b3661efb9"
    )
    assert diagnostic["dataset_in_image"] is False
    assert diagnostic["dataset_downloaded"] is False
    assert diagnostic["workload_executed"] is False
    assert claims and all(value is False for value in claims.values())
    assert len(plan["blockers"]) == 10  # type: ignore[arg-type]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "@sha256:45360d9eb0ff" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--only-binary=:all:" in dockerfile
    assert "--no-binary=gputil" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "HOME=/tmp/asi-runtime-home" in dockerfile
    assert "USER 65532:65532\nRUN --network=none python verify_runtime.py" in dockerfile
    assert "MNIST" not in dockerfile and "CIFAR" not in dockerfile


def test_verifier_accepts_exact_plan_and_rejects_weakened_future_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _module()
    for source, destination in (
        ("Dockerfile", "Dockerfile.source"),
        ("requirements.in", "requirements.in"),
        ("requirements.lock", "requirements.lock"),
        ("fetch_source.py", "fetch_source.py"),
        ("verify_runtime.py", "verify_runtime.py"),
    ):
        (tmp_path / destination).write_bytes((ROOT / source).read_bytes())
    monkeypatch.setattr(verifier, "QUALIFICATION_ROOT", tmp_path)
    plan = _plan()
    verifier._preflight(plan)
    verifier._validate_plan(plan)
    assert len(verifier._lock_versions()) == 84

    promoted = copy.deepcopy(plan)
    promoted["claims"]["external_execution_authorized"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="claims exceed"):
        verifier._validate_plan(promoted)
    weakened = copy.deepcopy(plan)
    weakened["runtime"]["future_invocation_requirements"][2] = (  # type: ignore[index]
        "network preferred"
    )
    with pytest.raises(ValueError, match="runtime plan"):
        verifier._validate_plan(weakened)
    removed = copy.deepcopy(plan)
    assert type(removed["blockers"]) is list
    removed["blockers"].pop()
    with pytest.raises(ValueError, match="ten exact blockers"):
        verifier._validate_plan(removed)


@pytest.mark.parametrize(
    ("section", "field", "alias", "message"),
    [
        ("claims", "external_execution_authorized", 0, "claims exceed"),
        ("claims", "scientific_promotion_allowed", 0, "claims exceed"),
        ("prospective_diagnostic", "dataset_in_image", 0, "diagnostic differs"),
        (
            "prospective_diagnostic",
            "avalanche_scenario_construction_only",
            1,
            "diagnostic differs",
        ),
    ],
)
def test_verifier_rejects_boolean_integer_aliases(
    section: str,
    field: str,
    alias: int,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _module()
    for source, destination in (
        ("Dockerfile", "Dockerfile.source"),
        ("requirements.in", "requirements.in"),
        ("requirements.lock", "requirements.lock"),
        ("fetch_source.py", "fetch_source.py"),
        ("verify_runtime.py", "verify_runtime.py"),
    ):
        (tmp_path / destination).write_bytes((ROOT / source).read_bytes())
    monkeypatch.setattr(verifier, "QUALIFICATION_ROOT", tmp_path)
    plan = _plan()
    nested = plan[section]
    assert type(nested) is dict
    nested[field] = alias
    with pytest.raises(ValueError, match=message):
        verifier._validate_plan(plan)


def test_verifier_rejects_numeric_alias_for_issue_identity() -> None:
    verifier = _module()
    plan = _plan()
    plan["qualification_issue"] = 1578.0
    with pytest.raises(ValueError, match="plan issue differs"):
        verifier._validate_plan(plan)


def test_verifier_rejects_hostile_or_unbounded_plan_before_hooks() -> None:
    verifier = _module()
    hostile = _HookMapping({"schema": "x"})
    _HookMapping.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        verifier._preflight(hostile)
    assert _HookMapping.calls == 0

    deep: object = 0
    for _ in range(34):
        deep = [deep]
    with pytest.raises(ValueError, match="depth"):
        verifier._preflight(deep)
    with pytest.raises(ValueError, match="item limit"):
        verifier._preflight([0] * 2049)


def test_lock_and_fetcher_are_bounded_and_parse_as_python310() -> None:
    for name in ("fetch_source.py", "verify_runtime.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name, feature_version=(3, 10))
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert len(re.findall(r"^[A-Za-z0-9][A-Za-z0-9._-]*==", lock, flags=re.MULTILINE)) == 82
    assert "torch-2.2.2%2Bcpu-cp310-cp310-linux_x86_64.whl" in lock
    assert "torchvision-0.17.2%2Bcpu-cp310-cp310-linux_x86_64.whl" in lock
    fetch = (ROOT / "fetch_source.py").read_text(encoding="utf-8")
    assert "_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024" in fetch
    assert "_MAX_EXPANDED_BYTES = 64 * 1024 * 1024" in fetch
    assert "member.isdir() or member.isreg()" in fetch
    assert "SOURCE_ARCHIVE_SHA256" in fetch
