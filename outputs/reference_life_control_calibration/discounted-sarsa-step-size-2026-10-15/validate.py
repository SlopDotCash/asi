"""Independently validate and summarize all preregistered SARSA step-size factorial traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from datetime import datetime
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.inputs / "plan.json").read_text(encoding="utf-8"))
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8-sig"))
    prereg_time = datetime.fromisoformat(prereg["created_at"])
    horizon = plan["horizon"]
    records = []
    by_pair = {}
    for path in sorted(args.records.glob("*-seed*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        require(r["schema"] == "asi.riverswim_sarsa_step_size_diagnostic.shard.v1", "schema")
        for field, expected in (
            ("development_only", True),
            ("scientific_promotion_allowed", False),
            ("reference_dev_population_allowed", False),
            ("worktree_clean_at_start", True),
            ("comparison_eligible", True),
        ):
            require(r[field] is expected, field)
        require(r["plan_sha256"] == digest(args.inputs / "plan.json"), "plan digest")
        require(r["script_sha256"] == digest(args.inputs / "run.py"), "script digest")
        for field in (
            "accepted_events",
            "executed_events",
            "dispatch_attempts",
            "environment_rng_cursor",
        ):
            require(type(r[field]) is int and r[field] == horizon, field)
        require(r["seed"] in plan["seeds"] and r["arm"] in plan["arms"], "roster")
        require(r["life_config"]["seed"] == r["seed"], "life seed")
        require(r["life_config"]["max_accepted_events"] == horizon, "life horizon")
        require(0 <= r["parameter_change_events"] <= horizon, "parameter change count")
        require(r["source_commit"] == prereg["head"], "frozen preregistration commit")
        start, end = (datetime.fromisoformat(r[k]) for k in ("started_at_utc", "completed_at_utc"))
        require(prereg_time <= start <= end, "UTC preregistration/collection order")
        require(
            r["life_config"]["lifecycle_id"] == f"step-size-diagnostic.{r['seed']}.{r['arm']}",
            "lifecycle identity",
        )
        key = r["seed"], r["arm"]
        require(key not in by_pair, "duplicate arm/seed")
        actions, rewards, positions = r["actions"], r["rewards"], r["positions"]
        require(len(actions) == len(rewards) == len(positions) == horizon, "trace length")
        env = r["environment"]
        prior = env["initial_state"]
        first_upstream = None
        visits = 0
        for index, (action, reward, position) in enumerate(
            zip(actions, rewards, positions, strict=True)
        ):
            require(type(action) is int and action in (0, 1), "action")
            require(type(position) is int and 0 <= position < env["n_states"], "position")
            require(type(reward) in (float, int) and math.isfinite(reward), "reward")
            expected = (
                env["reward_left"]
                if prior == 0 and action == 0
                else env["reward_right"]
                if prior == env["n_states"] - 1 and action == 1
                else 0.0
            )
            require(reward == expected, "state/action reward lattice")
            require(
                position == max(0, prior - 1) if action == 0 else abs(position - prior) <= 1,
                "transition support",
            )
            if first_upstream is None and reward == env["reward_right"]:
                first_upstream = index + 1
            visits += position == env["n_states"] - 1
            prior = position
        mean = statistics.mean(rewards)
        late = statistics.mean(rewards[-2000:])
        require(math.isclose(mean, r["mean_reward"], abs_tol=1e-15, rel_tol=0), "mean")
        require(math.isclose(late, r["late_mean_reward"], abs_tol=1e-15, rel_tol=0), "late mean")
        require(visits == r["rightmost_state_visits"], "visits")
        require(first_upstream == r["first_upstream_reward_event"], "first upstream reward")
        require([actions.count(0), actions.count(1)] == r["action_counts"], "action counts")
        require(
            [s["accepted_events"] for s in r["snapshots"]] == plan["fixed_checkpoints"],
            "checkpoints",
        )
        for s in r["snapshots"]:
            q = s["q_by_state_action"]
            require(len(q) == env["n_states"] and all(len(row) == 2 for row in q), "q shape")
            require(all(math.isfinite(v) for row in q for v in row), "q finite")
            for metric in ("eligibility_l2", "eligibility_max_abs", "parameter_l2"):
                require(
                    type(s[metric]) in (float, int) and math.isfinite(s[metric]) and s[metric] >= 0,
                    "snapshot " + metric,
                )
        require(r["resources"]["initial"] == r["resources"]["final"], "endpoint resource drift")
        r["input_path"], r["input_sha256"] = path.name, digest(path)
        records.append(r)
        by_pair[key] = r
    require(
        set(by_pair) == {(s, a) for s in plan["seeds"] for a in plan["arms"]}, "complete roster"
    )
    first = records[0]
    for relative, expected_digest in first["source_sha256"].items():
        require(
            digest(args.source_root / relative) == expected_digest, "current source " + relative
        )
    for r in records:
        for field in ("source_commit", "source_sha256", "runtime", "environment", "resources"):
            require(r[field] == first[field], "cross-shard " + field)
    previous_end = prereg_time
    intervals = []
    for seed, arm_a, arm_b in plan["schedule"]:
        batch = [by_pair[seed, arm_a], by_pair[seed, arm_b]]
        starts = [datetime.fromisoformat(r["started_at_utc"]) for r in batch]
        ends = [datetime.fromisoformat(r["completed_at_utc"]) for r in batch]
        require(previous_end <= min(starts), "declared pair schedule order")
        previous_end = max(ends)
        intervals.extend((start, 1) for start in starts)
        intervals.extend((end, -1) for end in ends)
    running = peak = 0
    for _, change in sorted(intervals):
        running += change
        peak = max(peak, running)
    require(peak <= 2 and running == 0, "recorded collection interval concurrency")
    def canonical_float32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]
    common_config = None
    for r in records:
        expected = plan["arms"][r["arm"]]
        remaining = dict(r["config"])
        for field in ("lamda", "step_size"):
            require(
                remaining.pop(field) == canonical_float32(expected[field]), "arm config " + field
            )
        if common_config is None:
            common_config = remaining
        require(remaining == common_config, "unplanned config difference")
    pairs = []
    for seed in plan["seeds"]:
        arms = {a: by_pair[seed, a] for a in plan["arms"]}
        require(
            all(r["snapshots"][0] == arms["trace_original"]["snapshots"][0] for r in arms.values()),
            "initial diagnostics pairing",
        )
        means = {a: r["mean_reward"] for a, r in arms.items()}
        trace_delta = means["trace_scaled"] - means["trace_original"]
        no_trace_delta = means["no_trace_scaled"] - means["no_trace_original"]
        pairs.append(
            {
                "seed": seed,
                "arm_mean_rewards": means,
                "primary_trace_step_delta": trace_delta,
                "no_trace_step_delta": no_trace_delta,
                "interaction_delta": trace_delta - no_trace_delta,
                "trace_scaled_vs_no_trace_original": means["trace_scaled"]
                - means["no_trace_original"],
                "trace_scaled_vs_no_trace_scaled": means["trace_scaled"] - means["no_trace_scaled"],
                "arm_diagnostics": {
                    a: {
                        k: r[k]
                        for k in (
                            "late_mean_reward",
                            "rightmost_state_visits",
                            "first_upstream_reward_event",
                            "action_counts",
                            "parameter_change_events",
                        )
                    }
                    for a, r in arms.items()
                },
            }
        )

    def moments(values: list[float]) -> dict[str, float]:
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}

    primary = [p["primary_trace_step_delta"] for p in pairs]
    interaction = [p["interaction_delta"] for p in pairs]
    primary_pass = all(d > 0 for d in primary) and statistics.mean(primary) > 0.01
    interaction_pass = (
        primary_pass and all(d > 0 for d in interaction) and statistics.mean(interaction) > 0.01
    )
    decision = (
        "bounded_trace_step_reduction_supported"
        if primary_pass
        else "bounded_trace_step_reduction_negative"
        if statistics.mean(primary) <= 0
        else "bounded_trace_step_reduction_inconclusive"
    )
    summary = {
        "schema": "asi.riverswim_sarsa_step_size_diagnostic.summary.v1",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "source_commit": first["source_commit"],
        "plan_sha256": first["plan_sha256"],
        "script_sha256": first["script_sha256"],
        "runtime": first["runtime"],
        "n_paired_seeds": len(pairs),
        "accepted_events": horizon * len(records),
        "arm_statistics": {
            a: moments([p["arm_mean_rewards"][a] for p in pairs]) for a in plan["arms"]
        },
        "primary_trace_step_delta": moments(primary),
        "no_trace_step_delta": moments([p["no_trace_step_delta"] for p in pairs]),
        "interaction_delta": moments(interaction),
        "trace_scaled_vs_no_trace_original": moments(
            [p["trace_scaled_vs_no_trace_original"] for p in pairs]
        ),
        "trace_scaled_vs_no_trace_scaled": moments(
            [p["trace_scaled_vs_no_trace_scaled"] for p in pairs]
        ),
        "primary_rule_passed": primary_pass,
        "interaction_rule_passed": interaction_pass,
        "decision": decision,
        "paired_seeds": pairs,
        "resources": first["resources"],
        "preregistration": prereg,
        "utc_schedule_order_verified": True,
        "maximum_recorded_collection_concurrency": peak,
        "inputs": [{"path": r["input_path"], "sha256": r["input_sha256"]} for r in records],
        "limits": [
            plan["scope_limit"],
            "Reported spreads are sample standard deviations, not confidence bounds.",
            "Raw-reward aggregates and transition support are independently reconstructed; "
            "diagnostic snapshot values are checked for finite shape, "
            "not independently replayed learner states.",
            "UTC times and hashes are consistency data, not execution attestation.",
        ],
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "arm_statistics",
                    "primary_trace_step_delta",
                    "no_trace_step_delta",
                    "interaction_delta",
                    "decision",
                    "primary_rule_passed",
                    "interaction_rule_passed",
                    "paired_seeds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
