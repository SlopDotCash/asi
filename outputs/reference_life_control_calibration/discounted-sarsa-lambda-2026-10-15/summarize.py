"""Independently validate and summarize all preregistered SARSA(lambda) traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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
    args = parser.parse_args()
    plan = json.loads((args.inputs / "plan.json").read_text(encoding="utf-8"))
    horizon = plan["horizon"]
    records = []
    by_pair = {}
    for path in sorted(args.records.glob("*-seed*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        require(r["schema"] == "asi.riverswim_sarsa_lambda_diagnostic.shard.v1", "schema")
        for field, expected in (
            ("development_only", True),
            ("scientific_promotion_allowed", False),
            ("reference_dev_population_allowed", False),
            ("worktree_clean_at_start", True),
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
    pairs = []
    for seed in plan["seeds"]:
        base, trace = by_pair[seed, "baseline"], by_pair[seed, "trace"]
        a, b = dict(base["config"]), dict(trace["config"])
        require(a.pop("lamda") == 0.0 and abs(b.pop("lamda") - 0.8) < 1e-7, "lambda values")
        require(a == b, "more than one changed config variable")
        require(base["snapshots"][0] == trace["snapshots"][0], "initial Q pairing")
        pairs.append(
            {
                "seed": seed,
                "baseline_mean_reward": base["mean_reward"],
                "trace_mean_reward": trace["mean_reward"],
                "paired_delta": trace["mean_reward"] - base["mean_reward"],
                "baseline_late_mean_reward": base["late_mean_reward"],
                "trace_late_mean_reward": trace["late_mean_reward"],
                "baseline_rightmost_visits": base["rightmost_state_visits"],
                "trace_rightmost_visits": trace["rightmost_state_visits"],
                "baseline_first_upstream_reward_event": base["first_upstream_reward_event"],
                "trace_first_upstream_reward_event": trace["first_upstream_reward_event"],
                "baseline_parameter_change_events": base["parameter_change_events"],
                "trace_parameter_change_events": trace["parameter_change_events"],
            }
        )

    def moments(values: list[float]) -> dict[str, float]:
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}

    deltas = [p["paired_delta"] for p in pairs]
    all_positive = all(d > 0 for d in deltas)
    supported = all_positive and statistics.mean(deltas) > 0.01
    decision = (
        "bounded_hypothesis_supported"
        if supported
        else "bounded_hypothesis_negative"
        if statistics.mean(deltas) <= 0
        else "bounded_hypothesis_inconclusive"
    )
    summary = {
        "schema": "asi.riverswim_sarsa_lambda_diagnostic.summary.v1",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reference_dev_population_allowed": False,
        "plan_sha256": first["plan_sha256"],
        "script_sha256": first["script_sha256"],
        "source_commit": first["source_commit"],
        "runtime": first["runtime"],
        "n_paired_seeds": len(pairs),
        "accepted_events": horizon * len(records),
        "baseline": moments([p["baseline_mean_reward"] for p in pairs]),
        "trace": moments([p["trace_mean_reward"] for p in pairs]),
        "paired_delta": moments(deltas),
        "all_seeds_positive": all_positive,
        "decision": decision,
        "paired_seeds": pairs,
        "resources": first["resources"],
        "inputs": [{"path": r["input_path"], "sha256": r["input_sha256"]} for r in records],
        "limits": [
            plan["scope_limit"],
            "Numeric state bytes exclude total process/native peak memory.",
            "Independent checks reconstruct metrics, reward lattice and transition support; "
            "they are not authenticated execution attestation.",
            "No confidence interval or claim of general superiority from three development seeds.",
        ],
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("baseline", "trace", "paired_delta", "decision", "paired_seeds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
