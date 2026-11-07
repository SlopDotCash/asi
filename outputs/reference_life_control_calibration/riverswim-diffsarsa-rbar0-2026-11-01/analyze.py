"""Apply the frozen scorecard control gate to the differential-SARSA R-bar0 lives.

Verdicts: PASS (differential_sarsa_rbar0 normalized t LCB > 0.1), FAIL (valid run,
LCB <= 0.1), VOID (incomplete roster, failed/non-finite life, parameter-change or
resource violation, invalid scale, or mixed source/runtime identity). Exit 0/1/3.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from alberta_framework.benchmarks import reference_life_scorecard as sc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as plan_run  # noqa: E402

T_CRITICAL = sc.CONTROL_GATE_T_CRITICAL
THRESHOLD = sc.CONTROL_GATE_LCB_THRESHOLD
GATED = "differential_sarsa_rbar0"
DEFAULT = "differential_sarsa"
LEARNING_EXPECTATION = {
    DEFAULT: "differential_sarsa",
    GATED: "differential_sarsa",
    "random": "random",
    "privileged_oracle": "privileged_oracle",
}


def _lcb(values: list[float]) -> tuple[float, float, float]:
    mean = sc._mean(values)
    sd = sc._sample_sd(values)
    return mean, sd, mean - T_CRITICAL * sd / math.sqrt(len(values))


def analyze(packed: list[dict[str, Any]]) -> dict[str, Any]:
    void: list[str] = []
    lives: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in packed:
        for life in entry.get("lives", []):
            key = (life.get("arm"), life.get("seed"))
            if key in lives:
                void.append(f"duplicate life {key}")
            lives[key] = life
    expected = {(arm, seed) for arm in plan_run.ARMS for seed in plan_run.SEEDS}
    if set(lives) != expected:
        void.append("lives do not exactly cover the fixed 4-arm x 12-seed roster")
    identities = set()
    for key in sorted(expected & set(lives)):
        life = lives[key]
        out = life.get("outcome", {})
        identities.add(
            json.dumps([life.get("source_identity"), life.get("runtime_identity")], sort_keys=True)
        )
        if (
            life.get("horizon") != plan_run.HORIZON
            or out.get("accepted_events") != plan_run.HORIZON
        ):
            void.append(f"{key}: horizon or accepted events differ from {plan_run.HORIZON}")
        if out.get("status") != "completed":
            void.append(f"{key}: life did not complete ({out.get('error', out.get('phase'))})")
            continue
        for name in ("reward_sum", "oracle_reward_sum", "regret_sum"):
            value = out.get(name)
            if not isinstance(value, float) or not math.isfinite(value):
                void.append(f"{key}: non-finite {name}")
        check = sc.parameter_change_check(
            LEARNING_EXPECTATION[key[0]], out.get("parameter_change_events", -1)
        )
        if not check["passed"]:
            void.append(f"{key}: parameter-change check failed")
        if out.get("persistent_bytes_initial") != out.get("persistent_bytes_final"):
            void.append(f"{key}: persistent numeric bytes grew")
    if len(identities) > 1:
        void.append("lives do not share one source/runtime identity")
    for seed in plan_run.SEEDS:
        a, b = lives.get((DEFAULT, seed)), lives.get((GATED, seed))
        if (
            a
            and b
            and a["outcome"].get("persistent_bytes_initial")
            != b["outcome"].get("persistent_bytes_initial")
        ):
            void.append(f"seed {seed}: arms differ in persistent numeric bytes")
    result: dict[str, Any] = {
        "protocol": plan_run.PROTOCOL,
        "seeds": list(plan_run.SEEDS),
        "horizon": plan_run.HORIZON,
        "initial_average_reward": plan_run.RBAR0,
        "t_critical": T_CRITICAL,
        "lcb_threshold_exclusive": THRESHOLD,
        "void_reasons": void,
    }
    if void:
        result["verdict"] = "VOID"
        return result
    reward = {k: v["outcome"]["reward_sum"] for k, v in lives.items()}
    seeds = plan_run.SEEDS
    scale = sc._mean([reward[("privileged_oracle", s)] - reward[("random", s)] for s in seeds])
    if not (math.isfinite(scale) and scale > 0.0):
        result.update(
            verdict="VOID", void_reasons=["normalization scale is not finite and positive"]
        )
        return result
    norm = {
        arm: [(reward[(arm, s)] - reward[("random", s)]) / scale for s in seeds]
        for arm in (DEFAULT, GATED)
    }
    paired = [norm[GATED][i] - norm[DEFAULT][i] for i in range(len(seeds))]
    gate = {}
    for name, values in (
        (DEFAULT, norm[DEFAULT]),
        (GATED, norm[GATED]),
        ("rbar0_minus_default", paired),
    ):
        mean, sd, lcb = _lcb(values)
        gate[name] = {"mean": mean, "sample_sd": sd, "paired_t_lcb_95": lcb}
    result["normalization_scale"] = scale
    result["gate"] = gate
    result["per_seed"] = [
        {
            "seed": s,
            "reward_sum": {arm: reward[(arm, s)] for arm in plan_run.ARMS},
            "normalized_differential_sarsa": norm[DEFAULT][i],
            "normalized_differential_sarsa_rbar0": norm[GATED][i],
            "normalized_rbar0_minus_default": paired[i],
            "high_end_visits": {
                arm: lives[(arm, s)]["outcome"]["high_end_visit_count"] for arm in (DEFAULT, GATED)
            },
            "late_reward_sum": {
                arm: lives[(arm, s)]["outcome"]["late_reward_sum"] for arm in (DEFAULT, GATED)
            },
            "final_average_reward": {
                arm: lives[(arm, s)]["outcome"].get("final_average_reward")
                for arm in (DEFAULT, GATED)
            },
        }
        for i, s in enumerate(seeds)
    ]
    result["verdict"] = "PASS" if gate[GATED]["paired_t_lcb_95"] > THRESHOLD else "FAIL"
    return result


def main(argv: list[str]) -> int:
    evidence = Path(argv[1]) if len(argv) > 1 else HERE / "evidence"
    packed = [json.loads(p.read_text()) for p in sorted(evidence.glob("seed*.json"))]
    result = analyze(packed)
    if result["verdict"] != "VOID":
        current = plan_run.source_identity()["files"]
        recorded = packed[0]["lives"][0]["source_identity"]["files"]
        if recorded != current:
            result.update(
                verdict="VOID", void_reasons=["recorded source digests differ from this checkout"]
            )
    print(json.dumps(result, indent=1, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "VOID": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
