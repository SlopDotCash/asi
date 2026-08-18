"""Assemble the rule-discovery real-protocol promotion summary (nonpromoting).

Reads the published screening shards (60-task promotions + diagnostics and
the 200-task confirmation) and writes one immutable JSON summary next to the
search artifact. Development screening diagnostic — never promotable
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import _atomic_write_json
from alberta_framework.benchmarks.rule_discovery import NONPROMOTING_POLICY

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "ipmnist_screening"
OUT = REPO_ROOT / "outputs" / "rule_discovery" / "real_screen_v1.json"
SEEDS = (0, 1, 2)


def _arm(directory: Path, name: str) -> dict[str, object]:
    values = []
    for seed in SEEDS:
        payload = json.loads((directory / f"{name}_seed{seed}.json").read_text())
        values.append(float(np.mean(payload["per_task_accuracy"])))
    return {"per_seed": values, "mean": float(np.mean(values))}


def main() -> int:
    shards = BASE / "shards"
    confirm = BASE / "confirm_full"
    screen = {
        name: _arm(shards, name)
        for name in (
            "disc_r1", "disc_r2", "disc_r3",
            "disc_r1_pscale", "disc_r1_pscale_norms",
            "sigma0_shiftnorm_d099", "sgd_ema_norm_d099", "upgd_w_control",
        )
    }
    full = {
        name: _arm(confirm, name)
        for name in ("disc_r1_pscale_norms", "sigma0_shiftnorm_d099")
        if all((confirm / f"{name}_seed{s}.json").exists() for s in SEEDS)
    }
    champion = np.asarray(screen["sigma0_shiftnorm_d099"]["per_seed"])
    paired = {
        name: {
            "per_seed": [
                float(v - c) for v, c in zip(screen[name]["per_seed"], champion)
            ],
            "mean": float(np.mean(screen[name]["per_seed"]) - np.mean(champion)),
        }
        for name in (
            "disc_r1", "disc_r2", "disc_r3",
            "disc_r1_pscale", "disc_r1_pscale_norms",
        )
    }
    payload = {
        "schema": "alberta.rule_discovery.real_screen.v1",
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "bar": 0.8640,
        "seeds": list(SEEDS),
        "screen_60_task": screen,
        "paired_vs_champion_60_task": paired,
        "confirm_200_task": full,
        "verdicts": {
            "disc_r1_verbatim": (
                "below bar (0.78372); beats published UPGD-W control +0.006, no gate"
            ),
            "disc_r2_verbatim": "below bar (0.72313)",
            "disc_r3_verbatim": "below bar (0.77339)",
            "disc_r1_pscale": (
                "hidden RMS at protocol scale costs ~-0.051 (transfer-killer isolated)"
            ),
            "disc_r1_pscale_norms": (
                "discovered structure (surprise budget replaces utility gate) at champion "
                "constants beats the champion on the 60-task screen (+0.00173, 3/3 seeds) "
                "and ties-to-slightly-beats at 200 tasks (+0.00066 paired, 2/3 seeds; "
                "screening claims nothing by itself)"
            ),
        },
    }
    _atomic_write_json(OUT, payload)
    print(json.dumps(payload["screen_60_task"], indent=1))
    print(json.dumps(payload.get("confirm_200_task", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
