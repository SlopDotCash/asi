"""Tests for the RiverSwim Q0 full-calibration development runner and analysis."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as sc

DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs/reference_life_control_calibration/riverswim-q0-full-2026-11-01"
)


def _load(name: str, plan_run: ModuleType | None = None):
    spec = importlib.util.spec_from_file_location(f"rq0_{name}", DIR / f"{name}.py")
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


def _life(arm: str, seed: int, reward: float, *, changes: int | None = None) -> dict:
    learning = arm in ("sarsa", "sarsa_q0")
    return {
        "arm": arm,
        "seed": seed,
        "horizon": run.HORIZON,
        "source_identity": {"git_head": "0" * 40, "files": {}},
        "runtime_identity": {"python": "x"},
        "outcome": {
            "status": "completed",
            "accepted_events": run.HORIZON,
            "parameter_change_events": (run.HORIZON if learning else 0)
            if changes is None
            else changes,
            "reward_sum": reward,
            "oracle_reward_sum": 17000.0,
            "regret_sum": 1.0,
            "high_end_visit_count": 0,
            "late_reward_sum": 0.0,
            "persistent_bytes_initial": 220,
            "persistent_bytes_final": 220,
        },
    }


def _packed(q0: list[float], default: float = 100.0) -> list[dict]:
    return [
        {
            "seed": seed,
            "lives": [
                _life("sarsa", seed, default),
                _life("sarsa_q0", seed, q0[i]),
                _life("random", seed, 60.0),
                _life("privileged_oracle", seed, 17060.0),
            ],
        }
        for i, seed in enumerate(run.SEEDS)
    ]


def test_seeds_are_fresh_and_disjoint_from_consumed_rosters() -> None:
    consumed = (
        set(sc.SEED_ROSTER)
        | set(range(81010, 81016))
        | {100101, 100102, 100103, 100201, 100202, 100203}
    )
    assert len(run.SEEDS) == 12 and not consumed & set(run.SEEDS)


def test_horizon_and_gate_are_the_scorecard_constants() -> None:
    assert run.HORIZON == 20_000 == sc.RIVERSWIM_HORIZON
    assert analyze.T_CRITICAL == sc.CONTROL_GATE_T_CRITICAL == 2.201
    assert analyze.THRESHOLD == sc.CONTROL_GATE_LCB_THRESHOLD == 0.10


def test_q0_arm_differs_from_scorecard_sarsa_only_in_initial_q_value() -> None:
    default = run.build_adapter("sarsa", run.HORIZON).config.to_config()
    optimistic = run.build_adapter("sarsa_q0", run.HORIZON).config.to_config()
    differing = {
        k for k in default.keys() | optimistic.keys() if default.get(k) != optimistic.get(k)
    }
    assert differing == {"initial_q_value"}
    assert optimistic["initial_q_value"] == 10.0


def test_lifecycle_ids_are_unique_and_scorecard_shaped() -> None:
    ids = {run.lifecycle_id(arm, seed) for arm in run.ARMS for seed in run.SEEDS}
    assert len(ids) == 48 and all(sc._is_scorecard_lifecycle_id(value) for value in ids)


def test_pass_when_normalized_lcb_exceeds_threshold() -> None:
    result = analyze.analyze(_packed([9000.0 + 10 * i for i in range(12)]))
    assert result["verdict"] == "PASS"
    assert result["normalization_scale"] == 17000.0
    assert result["gate"]["sarsa_q0"]["paired_t_lcb_95"] > 0.1


def test_fail_when_lcb_is_at_or_below_threshold() -> None:
    bimodal = [16000.0 if i < 3 else 60.0 for i in range(12)]
    result = analyze.analyze(_packed(bimodal))
    assert result["verdict"] == "FAIL"
    mean, sd, lcb = analyze._lcb([(v - 60.0) / 17000.0 for v in bimodal])
    assert math.isclose(result["gate"]["sarsa_q0"]["paired_t_lcb_95"], lcb)
    assert lcb <= 0.1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p[0]["lives"].pop(1),
        lambda p: p[0]["lives"][1]["outcome"].update(status="failed"),
        lambda p: p[0]["lives"][1]["outcome"].update(reward_sum=float("nan")),
        lambda p: p[0]["lives"][1]["outcome"].update(parameter_change_events=0),
        lambda p: p[0]["lives"][2]["outcome"].update(parameter_change_events=5),
        lambda p: p[0]["lives"][1]["outcome"].update(persistent_bytes_final=224),
        lambda p: p[0]["lives"][1].update(runtime_identity={"python": "y"}),
        lambda p: p[0]["lives"][1]["outcome"].update(accepted_events=19_999),
    ],
)
def test_void_on_any_invalid_life(mutate) -> None:
    packed = _packed([9000.0] * 12)
    mutate(packed)
    assert analyze.analyze(packed)["verdict"] == "VOID"


def test_void_on_nonpositive_scale() -> None:
    packed = _packed([9000.0] * 12)
    for entry in packed:
        entry["lives"][3]["outcome"]["reward_sum"] = 60.0
    assert analyze.analyze(packed)["verdict"] == "VOID"


envcheck = _load("envcheck", run)


def test_oracle_sum_difference_is_accumulation_order_not_drift() -> None:
    gain = envcheck.riverswim_gain()
    assert envcheck.sequential_sum(gain, 20_000) == 17143.002837739168  # stored scorecard
    assert float(20_000) * gain == 17143.002837745098  # stationary metrics closed form


def test_q0_reference_check_still_requires_every_field() -> None:
    assert set(envcheck.ALL_FIELDS) >= {"oracle_reward_sum", "regret_sum", "transcript_sha256"}
    record = envcheck.json.loads(run.Q0_REFERENCE.read_text())
    life = {
        "agent_manifest_id": record["agent_manifest_id"],
        "outcome": {f: record["result"][f] for f in envcheck.ALL_FIELDS},
    }
    assert envcheck.check_q0(life) == []
    life["outcome"]["transcript_sha256"] = "0" * 64
    assert envcheck.check_q0(life)


def test_scorecard_check_rejects_a_reward_change() -> None:
    gain = envcheck.riverswim_gain()
    shard = run.SCORECARD_SHARDS / "riverswim__random__70000.json"
    stored = envcheck.json.loads(shard.read_text())
    outcome = {k: stored["outcome"][k] for k in envcheck.DECISION_FIELDS}
    outcome |= {
        "oracle_reward_sum": float(20_000) * gain,
        "regret_sum": 0.0,
        "transcript_sha256": stored["outcome"]["transcript_sha256"],
        "persistent_bytes_initial": 220,
        "persistent_bytes_final": 220,
    }
    life = {
        "agent_manifest_id": stored["resolved"]["agent_manifest"]["manifest_id"],
        "outcome": outcome,
    }
    base = envcheck.check_scorecard("random", 70000, life, gain)
    life["outcome"]["reward_sum"] += 1e-9
    assert len(envcheck.check_scorecard("random", 70000, life, gain)) == len(base) + 1
