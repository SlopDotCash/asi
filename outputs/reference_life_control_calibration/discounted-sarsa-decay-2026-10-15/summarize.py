"""Re-derive the paired diagnostic from full reference-life raw traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from provenance import (
    require_record_digest,
    require_record_source_commit,
    require_record_sources,
    source_blob_digests,
)

root = Path(sys.argv[1])
out = Path(sys.argv[2])
retained = Path(__file__).resolve().parent
plan_path = retained / "canonical-calibration-plan.json"
script_path = retained / "experiment.py"
plan = json.loads(plan_path.read_text(encoding="utf-8"))
assert plan["horizon"] == 20000 and plan["scientific_promotion_allowed"] is False
expected_source_commit = plan["source_commit"]
plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
labelled = []
for path in sorted(root.glob("*-seed*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    require_record_source_commit(record.get("source_commit"), expected_source_commit, path.name)
    require_record_digest(record.get("plan_sha256"), plan_sha256, path.name, "plan_sha256")
    require_record_digest(record.get("script_sha256"), script_sha256, path.name, "script_sha256")
    labelled.append((path, record))
if not labelled:
    raise SystemExit(f"no raw records found in {root}")
source_paths = labelled[0][1].get("source_sha256")
if not isinstance(source_paths, dict) or not source_paths:
    raise ValueError(f"{labelled[0][0].name} source_sha256 is missing")
digest_by_path = source_blob_digests(Path.cwd(), expected_source_commit, sorted(source_paths))
records = []
for path, record in labelled:
    require_record_sources(record, path.name, digest_by_path)
    assert record["schema"] == "asi.riverswim_sarsa_decay_diagnostic.shard.v2"
    assert record["development_only"] is True and record["scientific_promotion_allowed"] is False
    assert all(
        record[k] == 20000
        for k in (
            "accepted_events",
            "executed_events",
            "dispatch_attempts",
            "environment_rng_cursor",
        )
    )
    rewards = record["rewards"]
    actions = record["actions"]
    positions = record["positions"]
    assert len(rewards) == len(actions) == len(positions) == 20000
    left = record["environment"]["reward_left"]
    right = record["environment"]["reward_right"]
    prior = 0
    for reward, action, position in zip(rewards, actions, positions, strict=True):
        assert type(reward) in (int, float) and math.isfinite(reward)
        assert type(action) is int and action in (0, 1)
        assert type(position) is int and 0 <= position <= 5
        expected = (
            left if prior == 0 and action == 0 else right if prior == 5 and action == 1 else 0.0
        )
        assert reward == expected, "Reward does not match the observed state/action lattice"
        if action == 0:
            assert position == max(0, prior - 1)
        else:
            assert abs(position - prior) <= 1
        prior = position
    assert math.isclose(statistics.mean(rewards), record["mean_reward"], rel_tol=0, abs_tol=1e-15)
    assert math.isclose(
        statistics.mean(rewards[-2000:]), record["late_mean_reward"], rel_tol=0, abs_tol=1e-15
    )
    assert sum(p == 5 for p in positions) == record["rightmost_state_visits"]
    assert [actions.count(0), actions.count(1)] == record["action_counts"]
    record["input_path"] = path.name
    record["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append(record)
assert len(records) == 6 and {(r["seed"], r["epsilon_decay_steps"]) for r in records} == {
    (s, d) for s in (100201, 100202, 100203) for d in (2500, 10000)
}
first = records[0]
for r in records:
    for field in (
        "plan_sha256",
        "script_sha256",
        "source_commit",
        "source_sha256",
        "runtime",
        "environment",
    ):
        assert r[field] == first[field], "Shards differ in " + field
    assert r["resources"]["initial"] == r["resources"]["final"], (
        "Numeric payload changed within a shard"
    )
    assert r["resources"] == first["resources"], "Numeric payload differs between arms"
by = {(r["seed"], r["epsilon_decay_steps"]): r for r in records}
paired = []
for seed in (100201, 100202, 100203):
    baseline, candidate = by[seed, 2500], by[seed, 10000]
    a, b = dict(baseline["config"]), dict(candidate["config"])
    a.pop("epsilon_decay_steps")
    b.pop("epsilon_decay_steps")
    assert a == b
    paired.append(
        {
            "seed": seed,
            "baseline_mean_reward": baseline["mean_reward"],
            "slow_decay_mean_reward": candidate["mean_reward"],
            "paired_delta": candidate["mean_reward"] - baseline["mean_reward"],
            "baseline_late_mean_reward": baseline["late_mean_reward"],
            "slow_decay_late_mean_reward": candidate["late_mean_reward"],
            "baseline_rightmost_visits": baseline["rightmost_state_visits"],
            "slow_decay_rightmost_visits": candidate["rightmost_state_visits"],
        }
    )
base = [p["baseline_mean_reward"] for p in paired]
cand = [p["slow_decay_mean_reward"] for p in paired]
delta = [p["paired_delta"] for p in paired]
passed = all(d > 0 for d in delta) and statistics.mean(delta) > 0.01
result = {
    "schema": "asi.riverswim_sarsa_decay_diagnostic.summary.v2",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
    "source_commit": first["source_commit"],
    "plan_sha256": first["plan_sha256"],
    "script_sha256": first["script_sha256"],
    "runtime": first["runtime"],
    "n_paired_seeds": 3,
    "accepted_events": 120000,
    "baseline_mean_reward": statistics.mean(base),
    "baseline_sample_std": statistics.stdev(base),
    "slow_decay_mean_reward": statistics.mean(cand),
    "slow_decay_sample_std": statistics.stdev(cand),
    "paired_mean_delta": statistics.mean(delta),
    "paired_sample_std": statistics.stdev(delta),
    "decision": "bounded_hypothesis_supported" if passed else "bounded_hypothesis_not_supported",
    "all_seeds_positive": all(d > 0 for d in delta),
    "paired_seeds": paired,
    "resources": first["resources"],
    "inputs": [{"path": r["input_path"], "sha256": r["input_sha256"]} for r in records],
    "limits": [
        "Three development seeds only; no scientific confidence or general superiority claim.",
        "The frozen scorecard plan, gates and retained artifacts are unchanged.",
        "Timing is telemetry only; native peak memory and checkpoint safety are not qualified.",
        "Hashes are consistency identifiers, not authenticated execution attestation.",
        "The failed accelerated v1 attempt is excluded entirely.",
    ],
}
with out.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in (
                "baseline_mean_reward",
                "baseline_sample_std",
                "slow_decay_mean_reward",
                "slow_decay_sample_std",
                "paired_mean_delta",
                "paired_sample_std",
                "decision",
            )
        },
        indent=2,
    )
)
