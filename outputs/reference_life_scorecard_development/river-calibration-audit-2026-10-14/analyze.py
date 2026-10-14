"""Descriptively audit retained RiverSwim scorecard shards (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/reference_life_scorecard_development/run2-2026-10-09"
OUT = Path(__file__).resolve().parent
ARMS = (
    "random",
    "sarsa",
    "differential_sarsa",
    "prototype",
    "prototype_frozen",
    "privileged_oracle",
)
SEEDS = tuple(range(70000, 70012))
EXPECTED_PLAN_SHA256 = "4c4396848da3abebc53fb61c7b241866566ad2b44fa83ee3395a5a389a869b86"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_records() -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    source_digests: set[str] = set()
    for arm in ARMS:
        for seed in SEEDS:
            path = RUN / "shards" / f"riverswim__{arm}__{seed}.json"
            payload = json.loads(path.read_text())
            if payload.get("schema") != "asi.reference_life_scorecard.run.v1":
                raise ValueError(f"unexpected shard schema: {path}")
            if payload.get("status") != "completed" or payload.get("failure") is not None:
                raise ValueError(f"shard is not complete: {path}")
            if payload.get("environment_kind") != "riverswim":
                raise ValueError(f"wrong environment: {path}")
            if payload.get("arm") != arm or payload.get("seed") != seed:
                raise ValueError(f"arm/seed identity mismatch: {path}")
            if payload.get("plan_sha256") != EXPECTED_PLAN_SHA256:
                raise ValueError(f"plan identity mismatch: {path}")
            if payload.get("evidence_policy", {}).get("permanently_nonpromoting") is not True:
                raise ValueError(f"unexpected evidence policy: {path}")
            outcome = payload.get("outcome")
            if not isinstance(outcome, dict) or outcome.get("accepted_events") != 20000:
                raise ValueError(f"invalid outcome: {path}")
            source_digests.add(canonical_sha256(payload.get("source_identity")))
            records[arm][seed] = outcome
    if len(source_digests) != 1:
        raise ValueError("source identities differ across audited shards")
    return records


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return None if denom == 0 else sum(x * y for x, y in zip(dx, dy)) / denom


def summarize(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for arm in ARMS:
        rows = records[arm]
        rewards = [float(rows[seed]["reward_sum"]) for seed in SEEDS]
        visits = [int(rows[seed]["high_end_visit_count"]) for seed in SEEDS]
        arms[arm] = {
            "n": len(rewards),
            "reward_sum": {
                "mean": statistics.mean(rewards),
                "sample_sd": statistics.stdev(rewards),
                "min": min(rewards),
                "max": max(rewards),
            },
            "high_end_visit_count": {
                "mean": statistics.mean(visits),
                "min": min(visits),
                "max": max(visits),
                "zero_seed_count": sum(value == 0 for value in visits),
            },
            "early_mean_reward": statistics.mean(
                float(rows[seed]["windows"]["early"]["mean_reward"]) for seed in SEEDS
            ),
            "late_mean_reward": statistics.mean(
                float(rows[seed]["windows"]["late"]["mean_reward"]) for seed in SEEDS
            ),
            "seed_values": [
                {
                    "seed": seed,
                    "reward_sum": float(rows[seed]["reward_sum"]),
                    "high_end_visit_count": int(rows[seed]["high_end_visit_count"]),
                    "early_mean_reward": float(rows[seed]["windows"]["early"]["mean_reward"]),
                    "late_mean_reward": float(rows[seed]["windows"]["late"]["mean_reward"]),
                }
                for seed in SEEDS
            ],
        }
        if arm in ("sarsa", "differential_sarsa", "prototype"):
            paired[arm] = {
                "pearson_reward_vs_high_end_visits": pearson(
                    rewards, [float(value) for value in visits]
                ),
                "interpretation": "descriptive association across the 12 retained seeds only",
            }
    return {
        "schema": "asi.reference_life_riverswim_shard_audit.v1",
        "source_run": "run2-2026-10-09",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "seed_roster": list(SEEDS),
        "environment": "riverswim",
        "horizon_per_shard": 20000,
        "claim_scope": "descriptive_secondary_analysis_permanently_nonpromoting",
        "arms": arms,
        "paired_diagnostics": paired,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RiverSwim control diagnostic",
        "",
        "This is a read-only secondary analysis of the 72 retained RiverSwim shards from",
        "`run2-2026-10-09`; it is permanently nonpromoting and consumes no new seeds.",
        "The analysis checks complete shard status, the frozen plan digest, the 12-seed",
        "roster, and one shared source identity before reporting descriptive summaries.",
        "",
        "| arm | reward sum mean ± sample SD | reward range | high-end visits mean (range) "
        "| zero-visit seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        data = result["arms"][arm]
        reward = data["reward_sum"]
        visits = data["high_end_visit_count"]
        lines.append(
            f"| `{arm}` | {reward['mean']:.3f} ± {reward['sample_sd']:.3f} "
            f"| {reward['min']:.3f}–{reward['max']:.3f} "
            f"| {visits['mean']:.1f} ({visits['min']}–{visits['max']}) "
            f"| {visits['zero_seed_count']} / 12 |"
        )
    lines.extend(
        [
            "",
            "## Seed-level pattern",
            "",
            "Discounted SARSA reward totals range from "
            f"{result['arms']['sarsa']['reward_sum']['min']:.3f} to "
            f"{result['arms']['sarsa']['reward_sum']['max']:.3f}; its high-end visit counts "
            f"range from {result['arms']['sarsa']['high_end_visit_count']['min']} to "
            f"{result['arms']['sarsa']['high_end_visit_count']['max']}. Prototype has the same "
            "qualitative spread. Differential SARSA records zero high-end visits on every "
            "seed and remains near random reward, while the privileged oracle is included "
            "only as a scale/control reference.",
            "",
            "Across these same 12 seeds, Pearson correlations between reward total and "
            "high-end visits are:",
            "",
        ]
    )
    for arm, diag in result["paired_diagnostics"].items():
        corr = diag["pearson_reward_vs_high_end_visits"]
        value = "undefined (zero visit variance)" if corr is None else f"{corr:.6f}"
        lines.append(f"- `{arm}`: {value}.")
    lines.extend(
        [
            "",
            "## Interpretation and next bounded question",
            "",
            "The retained records are consistent with an exploration/reachability bottleneck:",
            "successful discounted-SARSA and Prototype lives visit the rewarding end many",
            "times, while low-return lives often do not. This is a descriptive association,",
            "not a causal finding; high-end visits are an outcome telemetry field, and the",
            "same consumed seeds cannot be used to select or confirm a new policy.",
            "",
            "A fresh, separately frozen development pilot should compare the existing",
            "discounted-SARSA control with a slower epsilon-decay schedule while retaining",
            "matched random and finite-horizon-oracle scale controls. It should use new seed",
            "IDs, report the complete per-seed returns and high-end visits, and stop after a",
            "small screening roster unless the preregistered continuation rule is met. No",
            "threshold or result from the 2026-10-09 scorecard should be changed.",
            "",
            "Reproduce the secondary analysis with:",
            "",
            "```bash",
            "python3 outputs/reference_life_scorecard_development/"
            "river-calibration-audit-2026-10-14/analyze.py",
            "```",
            "",
            "The JSON output beside this report contains all per-seed values and the exact",
            "plan digest. This analysis does not validate execution beyond the retained shard",
            "consistency fields and does not establish benchmark or scientific performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = summarize(load_records())
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(render_markdown(result))
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
