"""The RiverSwim decay diagnostic must bind records to retained and committed bytes."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ARTIFACT = (
    Path(__file__).parents[1]
    / "outputs/reference_life_control_calibration"
    / "discounted-sarsa-decay-2026-10-15"
)


def _provenance_module():
    path = ARTIFACT / "provenance.py"
    spec = importlib.util.spec_from_file_location("riverswim_decay_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load RiverSwim provenance module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _make_source_repo(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    repo = tmp_path / "repo"
    package = repo / "alberta_framework"
    package.mkdir(parents=True)
    content = b"VALUE = 1\n"
    (package / "initialization.py").write_bytes(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "source",
        ],
        cwd=repo,
        check=True,
    )
    commit = _git(repo, "rev-parse", "HEAD")
    sources = {"alberta_framework/initialization.py": hashlib.sha256(content).hexdigest()}
    return repo, commit, sources


def _record(
    commit: str,
    sources: dict[str, str],
    plan_sha256: str,
    script_sha256: str,
    seed: int,
    decay: int,
) -> dict[str, object]:
    left, right = 0.004, 1.0
    return {
        "schema": "asi.riverswim_sarsa_decay_diagnostic.shard.v2",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "plan_sha256": plan_sha256,
        "script_sha256": script_sha256,
        "source_commit": commit,
        "source_sha256": dict(sources),
        "seed": seed,
        "epsilon_decay_steps": decay,
        "config": {"alpha": 0.1, "epsilon_decay_steps": decay},
        "environment": {"reward_left": left, "reward_right": right},
        "accepted_events": 20000,
        "executed_events": 20000,
        "dispatch_attempts": 20000,
        "environment_rng_cursor": 20000,
        "mean_reward": left,
        "late_mean_reward": left,
        "rightmost_state_visits": 0,
        "action_counts": [20000, 0],
        "resources": {"initial": {"bytes": 220}, "final": {"bytes": 220}},
        "runtime": {"python": "3.12.0"},
        "rewards": [left] * 20000,
        "actions": [0] * 20000,
        "positions": [0] * 20000,
    }


def _build_case(tmp_path: Path) -> tuple[Path, Path]:
    """Build a source repository and a case directory with its raw records."""

    repo, commit, sources = _make_source_repo(tmp_path)
    case = tmp_path / "case"
    case.mkdir()
    for name in ("summarize.py", "provenance.py"):
        (case / name).write_bytes((ARTIFACT / name).read_bytes())
    plan_bytes = json.dumps(
        {"horizon": 20000, "scientific_promotion_allowed": False, "source_commit": commit},
        sort_keys=True,
    ).encode()
    (case / "canonical-calibration-plan.json").write_bytes(plan_bytes)
    script_bytes = b'"""fixture runner"""\n'
    (case / "experiment.py").write_bytes(script_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    script_sha256 = hashlib.sha256(script_bytes).hexdigest()
    for seed in (100201, 100202, 100203):
        for decay, prefix in ((2500, "baseline"), (10000, "slow-decay")):
            record = _record(commit, sources, plan_sha256, script_sha256, seed, decay)
            (case / f"{prefix}-seed{seed}.json").write_text(
                json.dumps(record, sort_keys=True), encoding="utf-8"
            )
    return repo, case


def _tamper(case: Path, name: str, mutate) -> None:
    path = case / name
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def _run_summarize(
    repo: Path, case: Path, out_name: str = "out.json"
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(case / "summarize.py"), str(case), str(case / out_name)],
        cwd=repo,
        capture_output=True,
    )


def test_summarize_accepts_consistent_records(tmp_path: Path) -> None:
    repo, case = _build_case(tmp_path)

    completed = _run_summarize(repo, case)

    assert completed.returncode == 0, completed.stderr.decode()
    summary = json.loads((case / "out.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "bounded_hypothesis_not_supported"
    assert summary["n_paired_seeds"] == 3


def test_summarize_rejects_source_digest_tamper(tmp_path: Path) -> None:
    repo, case = _build_case(tmp_path)

    def mutate(record):
        first = sorted(record["source_sha256"])[0]
        record["source_sha256"][first] = "0" * 64

    _tamper(case, "baseline-seed100201.json", mutate)

    completed = _run_summarize(repo, case)

    assert completed.returncode != 0
    assert b"source_sha256 differs from commit blobs" in completed.stderr


def test_summarize_rejects_commit_drift(tmp_path: Path) -> None:
    repo, case = _build_case(tmp_path)

    _tamper(case, "slow-decay-seed100202.json", lambda r: r.update(source_commit="0" * 40))

    completed = _run_summarize(repo, case)

    assert completed.returncode != 0
    assert b"does not match pinned plan commit" in completed.stderr


def test_summarize_rejects_plan_digest_tamper(tmp_path: Path) -> None:
    repo, case = _build_case(tmp_path)

    _tamper(case, "baseline-seed100201.json", lambda r: r.update(plan_sha256="0" * 64))

    completed = _run_summarize(repo, case)

    assert completed.returncode != 0
    assert b"plan_sha256" in completed.stderr
    assert b"does not match retained bytes" in completed.stderr


def test_summarize_rejects_script_digest_tamper(tmp_path: Path) -> None:
    repo, case = _build_case(tmp_path)

    _tamper(case, "slow-decay-seed100201.json", lambda r: r.update(script_sha256="0" * 64))

    completed = _run_summarize(repo, case)

    assert completed.returncode != 0
    assert b"script_sha256" in completed.stderr
    assert b"does not match retained bytes" in completed.stderr


def test_source_commit_guard_rejects_mismatch() -> None:
    provenance = _provenance_module()

    with pytest.raises(ValueError, match="does not match pinned plan commit"):
        provenance.require_record_source_commit("actual", "expected", "drifted.json")
    provenance.require_record_source_commit("expected", "expected", "matched.json")


def test_digest_guard_rejects_mismatch() -> None:
    provenance = _provenance_module()

    with pytest.raises(ValueError, match="1.json plan_sha256"):
        provenance.require_record_digest("actual", "expected", "1.json", "plan_sha256")
    provenance.require_record_digest("expected", "expected", "1.json", "plan_sha256")


def test_source_guard_rejects_tamper() -> None:
    provenance = _provenance_module()

    with pytest.raises(ValueError, match="differs from commit blobs"):
        provenance.require_record_sources(
            {"source_sha256": {"alberta_framework/__init__.py": "1"}},
            "r.json",
            {"alberta_framework/__init__.py": "2"},
        )
    provenance.require_record_sources(
        {"source_sha256": {"alberta_framework/__init__.py": "2"}},
        "r.json",
        {"alberta_framework/__init__.py": "2"},
    )


def test_source_blob_digests_read_committed_bytes(tmp_path: Path) -> None:
    repo, commit, sources = _make_source_repo(tmp_path)
    provenance = _provenance_module()

    assert provenance.source_blob_digests(repo, commit, sorted(sources)) == sources
    with pytest.raises(ValueError, match="is missing from commit"):
        provenance.source_blob_digests(repo, commit, ["alberta_framework/absent.py"])
