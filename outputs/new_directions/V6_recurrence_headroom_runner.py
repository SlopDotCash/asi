"""V6 driver — is recurrence unexploited? headroom for direction D.

Pre-registered in elizaOS/asi#1875. Development diagnostic, permanently
nonpromoting. Micro suite only; touches no IPMNIST lane, no pinned artifact,
no registered source, no promotion seed.

Measures the paired gap ``D = M4 - M1`` for every shipped ladder arm. No
mechanism is proposed or implemented: this measures the existing ladder only,
so a future direction-D proposal starts from a number.

Per the pre-registration the family-separation control runs first and voids the
comparison if M1 and M4 do not differ in the property under test.

Usage:
    python V6_recurrence_headroom_runner.py --out V6_recurrence_headroom.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from alberta_framework.benchmarks import micro_continual as micro

SCHEMA = "alberta.new_directions.v6_recurrence_headroom.v1"
SEEDS = (0, 1, 2)
FAMILIES = ("input_permutation", "recurrence")
ARMS = micro.LADDER_ARMS


def family_separation_control(seed: int) -> dict[str, Any]:
    """Confirm M1 and M4 differ in the property under test, before any accuracy."""
    out: dict[str, Any] = {}
    for family in FAMILIES:
        config = micro.MicroStreamConfig(family=family)
        stream = micro.generate_stream(config, seed)
        permutations = np.asarray(stream.permutations)
        distinct = len({tuple(int(v) for v in row) for row in permutations})
        out[family] = {
            "n_regimes": int(config.n_regimes),
            "distinct_permutations": distinct,
            "recurrence_pool": int(config.recurrence_pool),
        }
    m1, m4 = out["input_permutation"], out["recurrence"]
    out["separated"] = bool(
        m1["distinct_permutations"] == m1["n_regimes"]
        and m4["distinct_permutations"] == m4["recurrence_pool"]
        and m4["distinct_permutations"] < m1["distinct_permutations"]
    )
    return out


def run_one(family: str, arm: str, seed: int) -> float:
    """One shipped micro-suite run; returns its overall online accuracy."""
    config = micro.MicroStreamConfig(family=family)
    result = micro.run_micro_arm(config, arm, seed)
    for field in ("overall_accuracy", "average_online_accuracy", "mean_accuracy"):
        value = getattr(result, field, None)
        if value is not None:
            return float(value)
    raise AttributeError(f"no accuracy field on {type(result).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    started = time.time()

    # Control first, per the pre-registration.
    control = family_separation_control(args.seeds[0])
    if not control["separated"]:
        artifact = {
            "schema": SCHEMA,
            "void": True,
            "reason": "family-separation control failed; M1 and M4 do not differ",
            "control": control,
            "wall_clock_seconds": round(time.time() - started, 1),
        }
        args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
        print("VOID: families are not separated; no accuracy reported")
        return 1

    runs: list[dict[str, Any]] = []
    for arm in ARMS:
        for family in FAMILIES:
            for seed in args.seeds:
                accuracy = run_one(family, arm, seed)
                runs.append(
                    {"arm": arm, "family": family, "seed": seed, "accuracy": accuracy}
                )
                print(f"  {family:18s} {arm:12s} seed={seed} acc={accuracy:.4f}", flush=True)

    def value(arm: str, family: str, seed: int) -> float:
        return next(
            r["accuracy"]
            for r in runs
            if r["arm"] == arm and r["family"] == family and r["seed"] == seed
        )

    gaps: list[dict[str, Any]] = []
    for arm in ARMS:
        per_seed = [
            value(arm, "recurrence", s) - value(arm, "input_permutation", s)
            for s in args.seeds
        ]
        array = np.asarray(per_seed, dtype=np.float64)
        gaps.append(
            {
                "arm": arm,
                "per_seed_gap": [round(float(v), 6) for v in per_seed],
                "mean_gap": float(array.mean()),
                "all_seeds_positive": bool((array > 0.0).all()),
                "exploits_recurrence": bool(array.mean() > 0.0 and (array > 0.0).all()),
                "m1_mean": float(
                    np.mean([value(arm, "input_permutation", s) for s in args.seeds])
                ),
                "m4_mean": float(
                    np.mean([value(arm, "recurrence", s) for s in args.seeds])
                ),
            }
        )

    exploiting = [g["arm"] for g in gaps if g["exploits_recurrence"]]
    best_m4 = max(gaps, key=lambda g: g["m4_mean"])

    artifact = {
        "schema": SCHEMA,
        "void": False,
        "control": control,
        "runs": runs,
        "paired_gaps": gaps,
        "protocol": {
            "runner": "alberta_framework.benchmarks.micro_continual (shipped)",
            "families": list(FAMILIES),
            "arms": list(ARMS),
            "seeds": list(args.seeds),
            "recurrence_pool": control["recurrence"]["recurrence_pool"],
            "metric": "overall online accuracy",
        },
        "outcome": {
            "criterion": "mean paired gap > 0 with all seeds positive",
            "arms_exploiting_recurrence": exploiting,
            "any_arm_exploits_recurrence": bool(exploiting),
            "best_m4_arm": best_m4["arm"],
            "best_m4_mean": best_m4["m4_mean"],
        },
        "evidence_policy": (
            "development_screening_diagnostic; permanently nonpromoting. Micro "
            "suite only; no IPMNIST lane, pinned artifact, or registered source."
        ),
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {args.out} in {artifact['wall_clock_seconds']}s")
    print(f"arms exploiting recurrence: {exploiting or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
