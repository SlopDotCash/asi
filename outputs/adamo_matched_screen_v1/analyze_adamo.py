"""Frozen analysis for the AdamO matched bounded-development screen.

Implements exactly the analysis frozen in the pre-registration on issue
#1560: per-arm/seed means and trajectories, final-task Jacobian isometry
diagnostics, resource aggregation, and the two paired comparisons. Reads
only the retained receipts; writes analysis_summary.json.
"""

import hashlib
import json
from pathlib import Path

RECEIPT_DIR = Path(__file__).parent / "receipts"
SEEDS = [15600, 15601, 15602, 15603]
ARMS = ["adamw_control", "adamo_inert", "adamo_l1e3", "adam_iso_joint_l1e3"]
PAIRS = [("adamo_l1e3", "adamo_inert"), ("adamo_l1e3", "adam_iso_joint_l1e3")]


def mean(values):
    return sum(values) / len(values)


def paired_delta(arm_records, cand, base, field):
    per_seed = []
    for seed in SEEDS:
        av = arm_records[(cand, seed)][field]
        bv = arm_records[(base, seed)][field]
        per_seed.append(mean(av) - mean(bv))
    return {
        "per_seed": per_seed,
        "mean": mean(per_seed),
        "all_seeds_positive": all(v > 0 for v in per_seed),
        "all_seeds_negative": all(v < 0 for v in per_seed),
    }


def main() -> None:
    receipts = {}
    manifest = {}
    for seed in SEEDS:
        path = RECEIPT_DIR / f"adamo_bounded-development_s{seed}.json"
        receipts[seed_key := seed] = json.loads(path.read_text())
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    per_seed_receipts = {
        (arm, seed): next(
            a for a in receipts[seed]["arms"] if a["arm"] == arm
        )
        for arm in ARMS
        for seed in SEEDS
    }

    summary = {
        "schema": "asi.adamo_matched_screen_analysis.v1",
        "preregistration": (
            "https://github.com/SlopDotCash/asi/issues/1560#issuecomment-5473394736"
        ),
        "development_only": True,
        "promotion_allowed": False,
        "receipt_sha256": manifest,
        "per_arm": {
            arm: {
                "mean_accuracy_per_seed": [
                    mean(per_seed_receipts[(arm, s)]["per_task_accuracy"]) for s in SEEDS
                ],
                "mean_accuracy": mean(
                    [mean(per_seed_receipts[(arm, s)]["per_task_accuracy"]) for s in SEEDS]
                ),
                "mean_loss": mean(
                    [mean(per_seed_receipts[(arm, s)]["per_task_loss"]) for s in SEEDS]
                ),
                "mean_plasticity": mean(
                    [mean(per_seed_receipts[(arm, s)]["per_task_plasticity"]) for s in SEEDS]
                ),
                "final_task_jacobian": {
                    field: [
                        per_seed_receipts[(arm, s)]["post_task_diagnostics"][-1][field]
                        for s in SEEDS
                    ]
                    for field in (
                        "jacobian_min_singular_value",
                        "jacobian_max_singular_value",
                        "jacobian_mean_singular_value",
                        "jacobian_rms_distance_from_one",
                        "jacobian_condition_number_clipped_1e12",
                        "weight_gram_penalty",
                    )
                },
                "resources": per_seed_receipts[(arm, SEEDS[0])]["resources"],
            }
            for arm in ARMS
        },
        "paired_comparisons": {
            f"{cand}_minus_{base}": {
                "accuracy": paired_delta(per_seed_receipts, cand, base, "per_task_accuracy"),
                "loss": paired_delta(per_seed_receipts, cand, base, "per_task_loss"),
                "plasticity": paired_delta(per_seed_receipts, cand, base, "per_task_plasticity"),
            }
            for cand, base in PAIRS
        },
    }

    # inert reduction check: bit-exact equality of adamo_inert vs adamw_control
    reduction = []
    for seed in SEEDS:
        inert = next(a for a in receipts[seed]["arms"] if a["arm"] == "adamo_inert")
        ctrl = next(a for a in receipts[seed]["arms"] if a["arm"] == "adamw_control")
        last_i = inert["post_task_diagnostics"][-1]
        last_c = ctrl["post_task_diagnostics"][-1]
        reduction.append(
            last_i["parameter_sha256"] == last_c["parameter_sha256"]
            and inert["per_task_accuracy"] == ctrl["per_task_accuracy"]
        )
    summary["inert_reduction_bit_exact_per_seed"] = reduction

    out = Path(__file__).parent / "analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["paired_comparisons"], indent=2))
    print("inert bit-exact:", reduction)


if __name__ == "__main__":
    main()
