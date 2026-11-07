"""Tests for the Switching Prototype base-step probe, step 2 (0.20), and its analysis."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as sc

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "outputs/reference_life_control_calibration/switching-prototype-step020-2026-11-02"
PARENT = ROOT / "outputs/reference_life_control_calibration/switching-prototype-step-2026-11-01"
ENV = "switching_two_state"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"sp20_{name}", DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


run = _load("run")
analyze = _load("analyze")
envcheck = _load("envcheck")


def test_seeds_and_reused_arms_are_the_parent_probes() -> None:
    assert run.SEEDS == tuple(range(93610, 93622))
    packed = json.loads((PARENT / "evidence/seed93610.json").read_text())
    assert tuple(x["arm"] for x in packed["lives"]) == run.PARENT_ARMS
    assert run.RUN_ARMS == (run.PROBE_ARM,) and run.PROBE_STEP == 0.20


def test_launcher_runs_only_the_probe_on_the_fixed_roster() -> None:
    text = (DIR / "launch.sh").read_text()
    assert f"seq {run.SEEDS[0]} {run.SEEDS[-1]}" in text
    assert re.search(r'REPLAY_ARMS="([^"]+)"', text).group(1).split() == list(run.PARENT_ARMS)
    assert f"--seed {run.REPLAY_SEED}" in text
    for case in run.ENVCHECK_CASES:
        assert case in text


@pytest.mark.parametrize("arm", ["prototype", "prototype_frozen"])
def test_default_prototype_configs_are_the_scorecard_configs(arm: str) -> None:
    plan = sc.build_development_plan()
    assert run.prototype_config(arm) == sc._prototype_agent_config(
        plan, environment_kind=ENV, arm=arm
    )


def test_probe_changes_only_the_base_step() -> None:
    default, probe = run.prototype_config("prototype"), run.prototype_config(run.PROBE_ARM)
    assert probe.oak.stomp.base_step_size == 0.20
    assert default.oak.stomp.base_step_size == 0.05
    restored = dataclasses.replace(
        probe,
        oak=dataclasses.replace(
            probe.oak, stomp=dataclasses.replace(probe.oak.stomp, base_step_size=0.05)
        ),
    )
    assert restored == default


def _fake_probe_dir(tmp_path: Path, step: float = 0.20) -> Path:
    """The #3656 0.10 lives relabelled as the probe: analysis must reproduce #3656's gate."""
    out = tmp_path / "evidence"
    out.mkdir()
    for seed in run.SEEDS:
        packed = json.loads((PARENT / f"evidence/seed{seed}.json").read_text())
        life = copy.deepcopy(packed["lives"][0])
        life["arm"], life["prototype_base_step_size"] = run.PROBE_ARM, step
        (out / f"seed{seed}.json").write_text(json.dumps({"seed": seed, "lives": [life]}))
    return out


def test_analysis_reproduces_the_parent_gate_on_relabelled_lives(tmp_path: Path) -> None:
    verdict, summary = analyze.analyze(analyze.load(_fake_probe_dir(tmp_path)))
    parent = json.loads((PARENT / "ANALYSIS.json").read_text())
    assert verdict == parent["verdict"] == "PASS"
    assert summary["probe_gate"] == parent["probe_gate"]
    assert summary["arms"]["prototype_step010"] == parent["arms"]["prototype_step010"]
    assert summary["probe_minus_step010_descriptive"]["normalized_mean"] == 0.0


def test_analysis_voids_on_wrong_probe_step(tmp_path: Path) -> None:
    verdict, summary = analyze.analyze(analyze.load(_fake_probe_dir(tmp_path, step=0.10)))
    assert verdict == "VOID" and summary["void_reasons"]


def test_replay_check_matches_committed_lives_and_flags_drift() -> None:
    packed = json.loads((PARENT / "evidence/seed93610.json").read_text())
    for life in packed["lives"]:
        assert envcheck.check_replay(life["arm"], life) == []
    bad = copy.deepcopy(packed["lives"][3])
    bad["outcome"]["reward_sum"] += 1.0
    assert envcheck.check_replay(bad["arm"], bad)
