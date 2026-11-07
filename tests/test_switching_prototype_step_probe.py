"""Tests for the Switching Prototype base-step development probe runner and analysis."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as sc

DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs/reference_life_control_calibration/switching-prototype-step-2026-11-01"
)


def _load(name: str, plan_run: ModuleType | None = None):
    spec = importlib.util.spec_from_file_location(f"sps_{name}", DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The retained scripts import their sibling run.py under a bare name.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "path", sys.path.copy())
        if plan_run is not None:
            patch.setitem(sys.modules, "run", plan_run)
        spec.loader.exec_module(module)
    return module


run = _load("run")
analyze = _load("analyze", run)
envcheck = _load("envcheck", run)
ENV = "switching_two_state"


def _scorecard_spec(arm: str):
    plan = sc.build_development_plan()
    return plan, next(
        s
        for s in sc.iter_run_specs(plan)
        if s.environment_kind == ENV and s.arm == arm and s.seed == 70_000
    )


# --- seeds ---------------------------------------------------------------------------


def test_seeds_are_fresh_and_disjoint_from_consumed_rosters() -> None:
    consumed = (
        set(sc.SEED_ROSTER)
        | set(range(47310, 47322))
        | set(range(91510, 91522))
        | set(range(81010, 81016))
        | {100101, 100102, 100103, 100201, 100202, 100203}
    )
    assert len(run.SEEDS) == len(set(run.SEEDS)) == len(sc.SEED_ROSTER) == 12
    assert not set(run.SEEDS) & consumed


def test_launcher_uses_the_fixed_roster_and_arms() -> None:
    text = (DIR / "launch.sh").read_text()
    assert f"seq {run.SEEDS[0]} {run.SEEDS[-1]}" in text
    assert re.search(r'ARMS="([^"]+)"', text).group(1).split() == list(run.ARMS)
    for case in run.ENVCHECK_CASES:
        assert case in text


# --- the default stays bit-identical; only the step changes in the probe arm ---------


@pytest.mark.parametrize("arm", ["prototype", "prototype_frozen"])
def test_default_prototype_configs_are_the_scorecard_configs(arm: str) -> None:
    plan = sc.build_development_plan()
    assert run.prototype_config(arm) == sc._prototype_agent_config(
        plan, environment_kind=ENV, arm=arm
    )


@pytest.mark.parametrize("arm", run.STORED_ARMS)
def test_default_arm_manifests_and_life_configs_match_the_scorecard(arm: str) -> None:
    plan, spec = _scorecard_spec(arm)
    want = sc.build_scorecard_runner(plan, spec)
    got = run.build_runner(arm, 70_000, horizon=run.HORIZON, life_id=spec.lifecycle_id)
    assert got.agent_adapter.manifest.manifest_id == want.agent_adapter.manifest.manifest_id
    assert got.config.config_sha256 == want.config.config_sha256
    assert (
        got.environment_adapter.manifest.descriptor()
        == want.environment_adapter.manifest.descriptor()
    )


def test_probe_changes_only_the_base_step_size() -> None:
    default = run.prototype_config("prototype")
    probe = run.prototype_config(run.PROBE_ARM)
    assert default.oak.stomp.base_step_size == 0.05
    assert probe.oak.stomp.base_step_size == run.PROBE_STEP == 0.10
    restored = dataclasses.replace(
        probe,
        oak=dataclasses.replace(
            probe.oak, stomp=dataclasses.replace(probe.oak.stomp, base_step_size=0.05)
        ),
    )
    assert restored == default


def test_probe_manifest_differs_from_default_only_in_base_step() -> None:
    def descriptor(arm: str) -> dict:
        life_id = run.lifecycle_id(arm, 0)
        runner = run.build_runner(arm, 70_000, horizon=run.HORIZON, life_id=life_id)
        return sc._agent_manifest_descriptor(runner.agent_adapter.manifest)

    diff = envcheck.config_diff(
        descriptor("prototype")["config"], descriptor(run.PROBE_ARM)["config"]
    )
    assert len(diff) == 1
    ((key, (old, new)),) = diff.items()
    assert key.endswith("base_step_size") and (old, new) == (0.05, 0.10)


def test_probe_inherits_the_prototype_parameter_change_rule() -> None:
    assert run.scorecard_arm(run.PROBE_ARM) == "prototype"
    assert sc.parameter_change_check("prototype", 0)["passed"] is False


# --- gates ---------------------------------------------------------------------------


def _life(arm: str, seed: int, reward: float, *, status: str = "completed") -> dict:
    step = {run.PROBE_ARM: run.PROBE_STEP, "prototype": 0.05, "prototype_frozen": 0.0}.get(arm)
    learns = sc.parameter_change_check(run.scorecard_arm(arm), 1)["passed"]
    return {
        "arm": arm,
        "seed": seed,
        "horizon": run.HORIZON,
        "prototype_base_step_size": step,
        "outcome": {
            "status": status,
            "reward_sum": reward,
            "parameter_change_check": {"passed": True, "learns": learns},
        },
    }


def _lives(**rewards: float) -> dict:
    base = {
        run.PROBE_ARM: 3100.0,
        "prototype": 3000.0,
        "prototype_frozen": 2000.0,
        "differential_sarsa": 3200.0,
        "sarsa": 2900.0,
        "random": 2000.0,
        "privileged_oracle": 4000.0,
    } | rewards
    return {arm: {s: _life(arm, s, base[arm]) for s in run.SEEDS} for arm in run.ARMS}


def test_thresholds_are_the_scorecards() -> None:
    assert (analyze.BEATS_FROZEN_LCB, analyze.NONINFERIORITY_LCB) == (0.05, -0.05)
    assert (sc.CONTROL_GATE_T_CRITICAL, sc.CONTROL_GATE_LCB_THRESHOLD) == (2.201, 0.10)


def test_pass_at_the_inclusive_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    # scale 2000; pin each threshold to the exact LCB the data produce: equality must pass
    monkeypatch.setattr(analyze, "BEATS_FROZEN_LCB", sc._paired_lcb([1100 / 2000] * 12))
    monkeypatch.setattr(analyze, "NONINFERIORITY_LCB", sc._paired_lcb([-100 / 2000] * 12))
    lives = _lives(**{run.PROBE_ARM: 3100.0, "prototype_frozen": 2000.0})
    verdict, summary = analyze.analyze(lives)
    assert verdict == "PASS"
    assert summary["probe_gate"]["best_sarsa_control"] == "differential_sarsa"
    monkeypatch.setattr(analyze, "NONINFERIORITY_LCB", sc._paired_lcb([-99 / 2000] * 12))
    assert analyze.analyze(lives)[0] == "FAIL"


def test_fail_when_noninferiority_misses() -> None:
    verdict, summary = analyze.analyze(_lives(**{run.PROBE_ARM: 3080.0}))
    assert verdict == "FAIL" and summary["probe_gate"]["beats_frozen"]
    assert not summary["probe_gate"]["sarsa_noninferior"]


def test_fail_when_frozen_is_not_beaten() -> None:
    verdict, summary = analyze.analyze(
        _lives(**{run.PROBE_ARM: 2090.0, "differential_sarsa": 2300.0, "sarsa": 2290.0})
    )
    assert verdict in ("FAIL", "VOID")
    assert verdict == "FAIL" and not summary["probe_gate"]["beats_frozen"]


def test_best_control_is_the_higher_qualified_mean() -> None:
    verdict, summary = analyze.analyze(_lives(sarsa=3300.0, **{run.PROBE_ARM: 3250.0}))
    assert summary["probe_gate"]["best_sarsa_control"] == "sarsa" and verdict == "PASS"


def test_void_when_no_control_qualifies_at_the_strict_threshold() -> None:
    # normalized 0.10 exactly is not > 0.10
    verdict, _ = analyze.analyze(_lives(differential_sarsa=2200.0, sarsa=2200.0))
    assert verdict == "VOID"


def test_void_on_failed_life_or_wrong_step() -> None:
    lives = _lives()
    lives["sarsa"][run.SEEDS[3]]["outcome"]["status"] = "failed"
    assert analyze.analyze(lives)[0] == "VOID"
    lives = _lives()
    lives[run.PROBE_ARM][run.SEEDS[0]]["prototype_base_step_size"] = 0.05
    assert analyze.analyze(lives)[0] == "VOID"
    lives = _lives()
    lives["prototype"][run.SEEDS[0]]["prototype_base_step_size"] = 0.10
    assert analyze.analyze(lives)[0] == "VOID"


def test_void_on_invalid_scale() -> None:
    assert analyze.analyze(_lives(privileged_oracle=2000.0))[0] == "VOID"


# --- environment check logic ---------------------------------------------------------


def _probe_pair() -> tuple[dict, dict]:
    default = {
        "agent_manifest": {"config": {"oak": {"stomp": {"base_step_size": 0.05, "x": 1}}}},
        "outcome": {"transcript_sha256": "a"},
    }
    probe = copy.deepcopy(default)
    probe["agent_manifest"]["config"]["oak"]["stomp"]["base_step_size"] = 0.10
    probe["prototype_base_step_size"] = 0.10
    probe["outcome"] = {"status": "completed", "transcript_sha256": "b",
                        "parameter_change_check": {"passed": True}}
    return probe, default


def test_envcheck_probe_accepts_a_step_only_change() -> None:
    probe, default = _probe_pair()
    assert envcheck.check_probe(probe, default) == []


def test_envcheck_probe_rejects_extra_drift_and_unexercised_step() -> None:
    probe, default = _probe_pair()
    probe["agent_manifest"]["config"]["oak"]["stomp"]["x"] = 2
    assert envcheck.check_probe(probe, default)
    probe, default = _probe_pair()
    probe["outcome"]["transcript_sha256"] = "a"
    assert envcheck.check_probe(probe, default)
