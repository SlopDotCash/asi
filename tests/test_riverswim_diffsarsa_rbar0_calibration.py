"""Tests for the RiverSwim differential-SARSA R-bar0 development runner and analysis."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as sc
from alberta_framework.streams.closed_loop import RiverSwimMDP

DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01"
)


def _load(name: str):
    sys.path.insert(0, str(DIR))
    spec = importlib.util.spec_from_file_location(f"rr0_{name}", DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


run = _load("run")
analyze = _load("analyze")
envcheck = _load("envcheck")
LEARNING = ("differential_sarsa", "differential_sarsa_rbar0")


def _life(arm: str, seed: int, reward: float, *, changes: int | None = None) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "horizon": run.HORIZON,
        "source_identity": {"git_head": "0" * 40, "files": {}},
        "runtime_identity": {"python": "x"},
        "outcome": {
            "status": "completed",
            "accepted_events": run.HORIZON,
            "parameter_change_events": (run.HORIZON if arm in LEARNING else 0)
            if changes is None
            else changes,
            "reward_sum": reward,
            "oracle_reward_sum": 17000.0,
            "regret_sum": 1.0,
            "high_end_visit_count": 0,
            "late_reward_sum": 0.0,
            "final_average_reward": 0.5,
            "persistent_bytes_initial": 204,
            "persistent_bytes_final": 204,
        },
    }


def _packed(rbar0: list[float], default: float = 100.0) -> list[dict]:
    return [
        {
            "seed": seed,
            "lives": [
                _life("differential_sarsa", seed, default),
                _life("differential_sarsa_rbar0", seed, rbar0[i]),
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
        | set(range(47310, 47322))
    )
    assert run.SEEDS == tuple(range(91510, 91522))
    assert len(run.SEEDS) == 12 and not consumed & set(run.SEEDS)


def test_horizon_and_gate_are_the_scorecard_constants() -> None:
    assert run.HORIZON == 20_000 == sc.RIVERSWIM_HORIZON
    assert analyze.T_CRITICAL == sc.CONTROL_GATE_T_CRITICAL == 2.201
    assert analyze.THRESHOLD == sc.CONTROL_GATE_LCB_THRESHOLD == 0.10


def test_rbar0_is_the_largest_step_reward_and_bounds_the_optimal_gain() -> None:
    river = sc._river_environment_config(sc.build_development_plan())
    mdp = RiverSwimMDP(river)
    assert run.RBAR0 == 1.0 == float(river.reward_right)
    assert float(river.reward_left) < run.RBAR0
    assert run.RBAR0 >= float(mdp.optimal_average_reward()) == 0.8571501418872549


def test_rbar0_arm_differs_from_scorecard_arm_only_in_initial_average_reward() -> None:
    default = run.build_adapter("differential_sarsa", run.HORIZON)
    optimistic = run.build_adapter("differential_sarsa_rbar0", run.HORIZON)
    a, b = default.config.to_config(), optimistic.config.to_config()
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {"initial_average_reward"}
    assert b["initial_average_reward"] == 1.0
    stored = envcheck.json.loads(
        (run.SCORECARD_SHARDS / "riverswim__differential_sarsa__70000.json").read_text()
    )
    assert default.manifest.manifest_id == stored["resolved"]["agent_manifest"]["manifest_id"]


def test_lifecycle_ids_are_unique_and_scorecard_shaped() -> None:
    ids = {run.lifecycle_id(arm, seed) for arm in run.ARMS for seed in run.SEEDS}
    assert len(ids) == 48 and all(sc._is_scorecard_lifecycle_id(value) for value in ids)


def test_pass_when_normalized_lcb_exceeds_threshold() -> None:
    result = analyze.analyze(_packed([9000.0 + 10 * i for i in range(12)]))
    assert result["verdict"] == "PASS"
    assert result["normalization_scale"] == 17000.0
    assert result["gate"]["differential_sarsa_rbar0"]["paired_t_lcb_95"] > 0.1


def test_fail_when_lcb_is_at_or_below_threshold() -> None:
    bimodal = [16000.0 if i < 3 else 60.0 for i in range(12)]
    result = analyze.analyze(_packed(bimodal))
    assert result["verdict"] == "FAIL"
    mean, sd, lcb = analyze._lcb([(v - 60.0) / 17000.0 for v in bimodal])
    assert math.isclose(result["gate"]["differential_sarsa_rbar0"]["paired_t_lcb_95"], lcb)
    assert lcb <= 0.1


def test_gate_reads_the_rbar0_arm_not_the_default_arm() -> None:
    packed = _packed([60.0] * 12, default=16000.0)
    assert analyze.analyze(packed)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p[0]["lives"].pop(1),
        lambda p: p[0]["lives"][1]["outcome"].update(status="failed"),
        lambda p: p[0]["lives"][1]["outcome"].update(reward_sum=float("nan")),
        lambda p: p[0]["lives"][1]["outcome"].update(parameter_change_events=0),
        lambda p: p[0]["lives"][0]["outcome"].update(parameter_change_events=0),
        lambda p: p[0]["lives"][2]["outcome"].update(parameter_change_events=5),
        lambda p: p[0]["lives"][1]["outcome"].update(persistent_bytes_final=208),
        lambda p: p[0]["lives"][1]["outcome"].update(
            persistent_bytes_initial=208, persistent_bytes_final=208
        ),
        lambda p: p[0]["lives"][1].update(runtime_identity={"python": "y"}),
        lambda p: p[0]["lives"][1]["outcome"].update(accepted_events=19_999),
        lambda p: p[1]["lives"].append(dict(p[1]["lives"][0])),
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


def test_oracle_sum_difference_is_accumulation_order_not_drift() -> None:
    gain = envcheck.riverswim_gain()
    assert envcheck.sequential_sum(gain, 20_000) == 17143.002837739168  # stored scorecard
    assert float(20_000) * gain == 17143.002837745098  # stationary metrics closed form


def test_q0_reference_check_still_requires_every_field() -> None:
    record = envcheck.json.loads(run.Q0_REFERENCE.read_text())
    life = {
        "agent_manifest_id": record["agent_manifest_id"],
        "outcome": {f: record["result"][f] for f in envcheck.ALL_FIELDS},
    }
    assert envcheck.check_q0(life) == []
    life["outcome"]["transcript_sha256"] = "0" * 64
    assert envcheck.check_q0(life)


def _stored_life(arm: str) -> dict:
    gain = envcheck.riverswim_gain()
    stored = envcheck.json.loads(
        (run.SCORECARD_SHARDS / f"riverswim__{arm}__70000.json").read_text()
    )
    out = stored["outcome"]
    oracle = float(20_000) * gain
    outcome = {k: out[k] for k in envcheck.DECISION_FIELDS} | {
        "oracle_reward_sum": oracle,
        "regret_sum": oracle - out["reward_sum"],
        "transcript_sha256": out["transcript_sha256"],
        "persistent_bytes_initial": out["resources"]["initial"]["persistent_numeric_bytes_total"],
        "persistent_bytes_final": out["resources"]["final"]["persistent_numeric_bytes_total"],
    }
    return {
        "agent_manifest_id": stored["resolved"]["agent_manifest"]["manifest_id"],
        "outcome": outcome,
    }


@pytest.mark.parametrize("arm", ["differential_sarsa", "random"])
def test_scorecard_check_accepts_the_stored_life_and_rejects_any_change(arm: str) -> None:
    gain = envcheck.riverswim_gain()
    assert envcheck.check_scorecard(arm, 70000, _stored_life(arm), gain) == []
    for field, change in (
        ("reward_sum", 1e-9),
        ("parameter_change_events", 1),
        ("persistent_bytes_final", 4),
    ):
        life = _stored_life(arm)
        life["outcome"][field] += change
        assert envcheck.check_scorecard(arm, 70000, life, gain)
    life = _stored_life(arm)
    life["outcome"]["transcript_sha256"] = "0" * 64
    assert envcheck.check_scorecard(arm, 70000, life, gain)
    life = _stored_life(arm)
    life["agent_manifest_id"] = "0" * 64
    assert envcheck.check_scorecard(arm, 70000, life, gain)


def test_envcheck_covers_the_default_differential_life_and_3308() -> None:
    assert run.ENVCHECK_CASES["scorecard_differential_70000"] == (
        "differential_sarsa",
        70_000,
        20_000,
        "scorecard",
    )
    assert run.ENVCHECK_CASES["pr3308_q0_81010"][:3] == ("sarsa_q0", 81_010, 5_000)
    record = envcheck.json.loads(run.Q0_REFERENCE.read_text())
    assert run.build_adapter("sarsa_q0", 5_000).manifest.manifest_id == record["agent_manifest_id"]


def test_launcher_arms_and_seeds_match_the_runner() -> None:
    text = (DIR / "launch.sh").read_text()
    assert "seq 91510 91521" in text
    assert "for arm in " + " ".join(run.ARMS) + "; do" in text
    for case in run.ENVCHECK_CASES:
        assert case in text
    assert "riverswim-diffsarsa-rbar0-2026-11-01/run.py" in (DIR / "memguard.sh").read_text()
