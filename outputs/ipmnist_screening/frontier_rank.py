"""Rank sigma0_* frontier arms vs the upgd_ema_norm_sigma0 base (paired by seed).

Development screening diagnostic only — never promotes scientific claims.

Modes:
  (default)       human-readable ranked table (60-task screen + any 200-task confirms)
  --winners-only  newline-separated arm names whose paired 60-task delta > 0.002
  --json          machine-readable ranked results
"""
import glob
import json
import statistics
import sys
import time

BASE = "upgd_ema_norm_sigma0"
ARMS = [
    "sigma0_hidden_norm", "sigma0_localgate",
    "sigma0_ndecay099", "sigma0_ndecay09999",
    "sigma0_eps1e6", "sigma0_eps1e4",
    "sigma0_gate_beta05", "sigma0_gate_beta2",
]
THRESHOLD = 0.002


def seed_means(cfg: str, root: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for path in glob.glob(f"{root}/{cfg}_seed*.json"):
        with open(path) as fh:
            d = json.load(fh)
        out[int(d["seed"])] = statistics.mean(d["per_task_accuracy"])
    return out


def build() -> dict:
    screen_base = seed_means(BASE, "outputs/ipmnist_screening/shards")
    confirm_base = seed_means(BASE, "outputs/ipmnist_screening/confirm_full")
    rows = []
    for arm in ARMS:
        screen = seed_means(arm, "outputs/ipmnist_screening/shards")
        confirm = seed_means(arm, "outputs/ipmnist_screening/confirm_full")
        shared = sorted(set(screen) & set(screen_base))
        row: dict = {"config_name": arm, "n_screen_seeds": len(shared)}
        if shared:
            row["screen_mean"] = statistics.mean(screen[s] for s in shared)
            row["screen_paired_delta_vs_base"] = statistics.mean(
                screen[s] - screen_base[s] for s in shared
            )
            row["screen_per_seed_delta"] = [
                round(screen[s] - screen_base[s], 6) for s in shared
            ]
            row["confirmation_candidate"] = (
                row["screen_paired_delta_vs_base"] > THRESHOLD
            )
        cshared = sorted(set(confirm) & set(confirm_base))
        if cshared:
            row["confirm_mean"] = statistics.mean(confirm[s] for s in cshared)
            row["n_confirm_seeds"] = len(cshared)
            row["confirm_paired_delta_vs_base"] = statistics.mean(
                confirm[s] - confirm_base[s] for s in cshared
            )
        elif confirm:
            row["n_confirm_seeds"] = 0
            row["confirm_unpaired_seeds"] = sorted(confirm)
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("screen_mean") or 0.0))
    return {
        "base": BASE,
        "base_screen_mean": (
            statistics.mean(screen_base.values()) if screen_base else None
        ),
        "base_confirm_mean": (
            statistics.mean(confirm_base.values()) if confirm_base else None
        ),
        "confirmation_threshold": THRESHOLD,
        "created_unix": time.time(),
        "evidence_policy": {
            "development_only": True,
            "evidence_class": "development_screening_diagnostic",
            "scientific_promotion_allowed": False,
        },
        "results": rows,
    }


def main() -> None:
    data = build()
    if "--winners-only" in sys.argv:
        for row in data["results"]:
            if row.get("confirmation_candidate"):
                print(row["config_name"])
        return
    if "--json" in sys.argv:
        json.dump(data, sys.stdout, indent=1, sort_keys=True)
        print()
        return
    print(f"base {data['base']}: screen={data['base_screen_mean']}"
          f" confirm={data['base_confirm_mean']}")
    for row in data["results"]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
