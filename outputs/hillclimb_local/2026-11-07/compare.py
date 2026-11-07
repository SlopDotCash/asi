"""Recompute descriptive local ASI comparisons from the saved shard records."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

IP_PREFIX = "rls_head_resid_identmap50_r_idet003_"
IP_ARMS = (
    "s02_anneal_f1t1000_wd0",
    "s005625_anneal_f1t1000_wd0",
    "s005625_anneal_f2t1000_wd0",
)
REF_ARMS = {
    "switching": ("prototype_step020", "prototype_step010", "prototype", "differential_sarsa"),
    "riverswim": ("sarsa_q0", "differential_sarsa_rbar0", "sarsa", "differential_sarsa"),
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_pairs(candidate: list[float], control: list[float]) -> dict[str, Any]:
    gains = [x - y for x, y in zip(candidate, control, strict=True)]
    return {
        "mean_gain": statistics.mean(gains),
        "gains_in_seed_order": gains,
        "improved_seeds": sum(gain > 0 for gain in gains),
        "paired_seeds": len(gains),
    }


def compare_ip(root: Path, *, n_tasks: int) -> dict[str, Any]:
    seeds = [311001, 311002, 311003] if n_tasks == 12 else [311011]
    arms = IP_ARMS if n_tasks == 12 else IP_ARMS[:2]
    receipt: dict[str, Any] = {}
    curves: dict[str, list[list[float]]] = {}
    for arm in arms:
        curves[arm] = []
        rows = []
        for seed in seeds:
            shard = load(root / f"{arm}.{seed}.json")
            assert shard["seed"] == seed
            assert shard["config_name"] == f"{IP_PREFIX}{arm}"
            config = shard["config"]
            assert config == {
                "hidden1": 300, "hidden2": 150, "input_dim": 784,
                "n_classes": 10, "n_tasks": n_tasks, "task_length": 5000,
            }
            identity = {k: shard[k] for k in ("source_provenance", "dataset_provenance")}
            if not receipt:
                receipt.update(identity)
            assert all(receipt[k] == value for k, value in identity.items())
            assert shard["hyperparameters"]["weight_decay"] == 0.0
            assert shard["noise_mode"] == "step"
            curve = shard["per_task_accuracy"]
            assert len(curve) == n_tasks and all(0 <= x <= 1 for x in curve)
            assert len(shard["per_task_loss"]) == n_tasks
            assert all(math.isfinite(x) for x in shard["per_task_loss"])
            curves[arm].append(curve)
            rows.append(statistics.mean(curve))
        receipt[f"{arm}_mean"] = statistics.mean(rows)
        receipt[f"{arm}_seed_means"] = rows
        if arm != arms[0]:
            receipt[f"s02_vs_{arm}"] = summarize_pairs(receipt[f"{arms[0]}_seed_means"], rows)
            if n_tasks == 200:
                receipt[f"s02_vs_{arm}_50_task_blocks"] = [
                    statistics.mean(curves[arms[0]][0][i:i + 50])
                    - statistics.mean(curves[arm][0][i:i + 50])
                    for i in range(0, n_tasks, 50)
                ]
    receipt["seeds"] = seeds
    receipt["n_tasks"] = n_tasks
    receipt["task_length"] = 5000
    return receipt


def compare_ref(root: Path, environment: str) -> dict[str, Any]:
    seeds = [118331, 118332, 118333]
    arms = REF_ARMS[environment]
    horizon = 4000 if environment == "switching" else 20000
    arm_outcomes: dict[str, Any] = {}
    identities: dict[int, str] = {}
    source_heads: set[str] = set()
    for arm in arms:
        rows = []
        for seed in seeds:
            shard = load(root / f"{environment}.{arm}.{seed}.json")
            assert (shard["arm"], shard["seed"], shard["horizon"]) == (arm, seed, horizon)
            assert identities.setdefault(seed, shard["lifecycle_id"]) == shard["lifecycle_id"]
            source_heads.add(shard["source_identity"]["git_head"])
            row = shard["outcome"]
            assert row["status"] == "completed", row
            assert row["accepted_events"] == horizon
            assert math.isfinite(row["reward_sum"])
            assert row["parameter_change_events"] > 0
            rows.append(row)
        rewards = [row["reward_sum"] for row in rows]
        result = {
            "mean_reward": statistics.mean(rewards),
            "reward_by_seed": rewards,
            "parameter_changes_by_seed": [r["parameter_change_events"] for r in rows],
        }
        if environment == "switching":
            for stage in ("initial", "final"):
                result[f"persistent_bytes_{stage}_by_seed"] = [
                    r["resources"][stage]["persistent_numeric_bytes_total"] for r in rows
                ]
        else:
            for key in (
                "high_end_visit_count", "persistent_bytes_initial", "persistent_bytes_final",
            ):
                result[f"{key}_by_seed"] = [r[key] for r in rows]
        arm_outcomes[arm] = result
    assert len(source_heads) == 1
    pairs = (
        [(arms[0], control) for control in arms[1:]] if environment == "switching"
        else [(arms[0], arms[2]), (arms[1], arms[3]), (arms[0], arms[1])]
    )
    return {
        "source_head": source_heads.pop(),
        "seeds": seeds,
        "horizon": horizon,
        "arms": arm_outcomes,
        "pairs": {
            f"{a}_vs_{b}": summarize_pairs(
                arm_outcomes[a]["reward_by_seed"], arm_outcomes[b]["reward_by_seed"]
            )
            for a, b in pairs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane", choices=("ipmnist", "ipmnist-200", "switching", "riverswim")
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.lane.startswith("ipmnist"):
        report = compare_ip(args.root / args.lane, n_tasks=12 if args.lane == "ipmnist" else 200)
    else:
        report = compare_ref(args.root / "reference", args.lane)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
