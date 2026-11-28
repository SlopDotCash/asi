"""Reproduce the six-seed matched RiverSwim comparison from its retained lives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ARMS = ("sarsa_q0", "differential_sarsa_rbar0", "random", "privileged_oracle")
SEEDS = range(612670, 612676)
PLAN_HEAD = "06702c614899f6ae5417b77dc05b922e3dc766ab"
COMMON = {
    "alberta_framework/reference_life_controls.py",
    "alberta_framework/reference_life.py",
    "alberta_framework/streams/closed_loop.py",
    "alberta_framework/benchmarks/reference_life_scorecard.py",
}
ROOT = "outputs/reference_life_control_calibration/"
Q_RUNNER = ROOT + "riverswim-q0-full-2026-11-01/run.py"
D_RUNNER = ROOT + "riverswim-diffsarsa-rbar0-2026-11-01/run.py"
POLICY = {
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(raw: Path, repo: Path) -> dict:
    rows = {}
    manifests = {}
    source_hashes = {}
    raw_hashes = {}
    runtime = None
    expected = {f"{arm}-{seed}.json" for arm in ARMS for seed in SEEDS}
    check({f.name for f in raw.glob("*.json")} == expected, "incomplete or extra roster")
    for seed in SEEDS:
        life_hash = hashlib.sha256(f"asi.riverswim.matched.{seed}".encode("ascii")).hexdigest()
        for arm in ARMS:
            name = f"{arm}-{seed}.json"
            encoded = (raw / name).read_bytes()
            row = json.loads(encoded)
            context = f"{arm}/{seed}"
            check((row["seed"], row["arm"], row["horizon"]) == (seed, arm, 20000), context)
            check(row["lifecycle_id"] == f"prototype.{life_hash[:16]}", f"lifecycle: {context}")
            check(row["source_identity"]["git_head"] == PLAN_HEAD, f"wrong plan source: {context}")
            check(row["evidence_policy"] == POLICY, f"evidence policy: {context}")
            manifest = row["agent_manifest_id"]
            check(manifests.setdefault(arm, manifest) == manifest, f"changed adapter: {context}")
            is_diff = arm == "differential_sarsa_rbar0"
            if is_diff:
                check(row["initial_average_reward"] == 1, context)
                fields = COMMON | {"alberta_framework/core/average_reward.py", D_RUNNER}
            else:
                check(row["initial_q_value"] == (10 if arm == "sarsa_q0" else None), context)
                fields = COMMON | {Q_RUNNER}
            source = row["source_identity"]["files"]
            check(set(source) == fields, f"source list: {context}")
            for path in fields:
                if path not in source_hashes:
                    source_hashes[path] = hashlib.sha256((repo / path).read_bytes()).hexdigest()
                check(source[path] == source_hashes[path], f"source digest: {path}, {context}")
            if runtime is None:
                runtime = row["runtime_identity"]
            check(row["runtime_identity"] == runtime, f"runtime: {context}")
            outcome = row["outcome"]
            check(
                (outcome["status"], outcome["phase"], outcome["accepted_events"])
                == ("completed", "completed", 20000),
                f"incomplete life: {context}",
            )
            check(math.isfinite(outcome["reward_sum"]), f"nonfinite reward: {context}")
            check(0 <= outcome["high_end_visit_count"] <= 20000, context)
            check(len(bytes.fromhex(outcome["transcript_sha256"])) == 32, context)
            check(row["wall_clock_seconds_telemetry_only"] > 0, context)
            rows[seed, arm] = outcome
            raw_hashes[name] = hashlib.sha256(encoded).hexdigest()

    def reward(seed: int, arm: str) -> float:
        return rows[seed, arm]["reward_sum"]

    denominator = mean(reward(s, "privileged_oracle") - reward(s, "random") for s in SEEDS)
    check(math.isfinite(denominator) and denominator > 0, "invalid common denominator")
    paired = []
    for seed in SEEDS:
        rewards = {arm: reward(seed, arm) for arm in ARMS}
        normalized = {arm: (rewards[arm] - rewards["random"]) / denominator for arm in ARMS[:2]}
        paired.append(
            {
                "seed": seed,
                "rewards": rewards,
                "normalized": normalized,
                "difference": (rewards[ARMS[0]] - rewards[ARMS[1]]) / denominator,
                "high_end_visits": {a: rows[seed, a]["high_end_visit_count"] for a in ARMS},
            }
        )
    differences = [p["difference"] for p in paired]
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    decision = (
        ARMS[0] if positive == len(SEEDS) else ARMS[1] if negative == len(SEEDS) else "retain_both"
    )
    return {
        "common_oracle_minus_random": denominator,
        "paired": paired,
        "mean_normalized_return": {a: mean(p["normalized"][a] for p in paired) for a in ARMS[:2]},
        "mean_paired_difference": mean(differences),
        "standard_error_of_paired_mean": stdev(differences) / math.sqrt(len(differences)),
        "positive": positive,
        "negative": negative,
        "ties": len(SEEDS) - positive - negative,
        "development_decision": decision,
        "runtime": runtime,
        "manifests": manifests,
        "source_sha256": source_hashes,
        "run_source_commit": PLAN_HEAD,
        "raw_sha256": raw_hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(summarize(args.raw, args.repo), indent=2, sort_keys=True, allow_nan=False))
