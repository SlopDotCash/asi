"""Maintained rule-discovery summary builder for explicit campaign inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import alberta_framework.benchmarks.ipmnist_screening as ipmnist_screening_module
import alberta_framework.benchmarks.rule_discovery as rule_discovery_module
from alberta_framework._seed_validation import require_jax_seed, require_unique_jax_seeds
from alberta_framework.benchmarks.ipmnist_provenance import analysis_provenance
from alberta_framework.benchmarks.rule_discovery import NONPROMOTING_POLICY

SCREEN_STAGE_N_TASKS = {
    "screen": 60,
    "confirmation": 200,
}
SCREEN_N_TASKS = SCREEN_STAGE_N_TASKS["screen"]
CONFIRMATION_N_TASKS = SCREEN_STAGE_N_TASKS["confirmation"]
RULE_DISCOVERY_TASK_LENGTH = 5_000
RULE_DISCOVERY_NOISE_MODE = "step"

SCREEN_ARMS = (
    "disc_r1",
    "disc_r2",
    "disc_r3",
    "disc_r1_pscale",
    "disc_r1_pscale_norms",
    "sigma0_shiftnorm_d099",
    "sgd_ema_norm_d099",
    "upgd_w_control",
)
DISCOVERY_ARMS = (
    "disc_r1",
    "disc_r2",
    "disc_r3",
    "disc_r1_pscale",
    "disc_r1_pscale_norms",
)
CHAMPION = "sigma0_shiftnorm_d099"


def _arm(
    directory: Path,
    name: str,
    seeds: Sequence[int],
    *,
    expected_n_tasks: int,
) -> dict[str, Any]:
    values = []
    for seed in seeds:
        path = directory / f"{name}_seed{seed}.json"
        if not path.exists():
            raise ValueError(f"{name} is missing seed {seed} in {directory}")
        payload = ipmnist_screening_module.load_shard(path)
        payload_name = payload["config_name"]
        if payload_name != name:
            raise ValueError(
                f"{path}: config_name {payload_name!r} does not match "
                f"expected arm {name!r}"
            )
        config = payload["config"]
        if config["n_tasks"] != expected_n_tasks:
            raise ValueError(
                f"{path}: n_tasks {config['n_tasks']} does not match expected "
                f"{expected_n_tasks}"
            )
        if config["task_length"] != RULE_DISCOVERY_TASK_LENGTH:
            raise ValueError(f"{path}: task_length must be {RULE_DISCOVERY_TASK_LENGTH}")
        if payload["noise_mode"] != RULE_DISCOVERY_NOISE_MODE:
            raise ValueError(f"{path}: noise_mode must be {RULE_DISCOVERY_NOISE_MODE!r}")
        payload_seed = require_jax_seed(payload.get("seed"), name=f"{path} seed")
        if payload_seed != seed:
            raise ValueError(f"{path} seed does not match requested seed {seed}")
        accuracy = np.asarray(payload["per_task_accuracy"], dtype=np.float64)
        values.append(float(np.mean(accuracy)))
    return {"per_seed": values, "mean": float(np.mean(values))}


def build_legacy_rule_discovery_summary(
    screen_dir: Path,
    confirm_dir: Path,
    *,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Reconstruct the exact legacy v1 payload for compatibility checks."""
    seeds = require_unique_jax_seeds(seeds)
    screen = {
        name: _arm(screen_dir, name, seeds, expected_n_tasks=SCREEN_N_TASKS)
        for name in SCREEN_ARMS
    }
    confirm_names = ("disc_r1_pscale_norms", CHAMPION)
    present = [
        (confirm_dir / f"{name}_seed{seed}.json").exists()
        for name in confirm_names
        for seed in seeds
    ]
    if any(present) and not all(present):
        raise ValueError("rule-discovery confirmation seeds are incomplete")
    full = (
        {
            name: _arm(confirm_dir, name, seeds, expected_n_tasks=CONFIRMATION_N_TASKS)
            for name in confirm_names
        }
        if all(present)
        else {}
    )
    champion = np.asarray(screen[CHAMPION]["per_seed"], dtype=np.float64)
    paired: dict[str, Any] = {}
    for name in DISCOVERY_ARMS:
        values = np.asarray(screen[name]["per_seed"], dtype=np.float64)
        differences = values - champion
        paired[name] = {
            "per_seed": [float(value) for value in differences],
            "mean": float(np.mean(values) - np.mean(champion)),
        }
    return {
        "schema": "alberta.rule_discovery.real_screen.v1",
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "bar": 0.8640,
        "seeds": list(seeds),
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


def build_rule_discovery_summary(
    screen_dir: Path,
    confirm_dir: Path,
    *,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Build the maintained v2 summary with explicit legacy compatibility."""
    seeds = require_unique_jax_seeds(seeds)
    legacy = build_legacy_rule_discovery_summary(
        screen_dir, confirm_dir, seeds=seeds
    )
    legacy_canonical = json.dumps(
        legacy, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    maintained = dict(legacy)
    maintained["schema"] = "asi.rule_discovery.real_screen.v2"
    maintained["legacy_compatibility"] = {
        "schema": legacy["schema"],
        "canonical_sha256": hashlib.sha256(legacy_canonical).hexdigest(),
    }
    maintained["provenance"] = analysis_provenance(
        command="rule-discovery-summary",
        input_paths=[
            directory / f"{name}_seed{seed}.json"
            for directory, names in (
                (screen_dir, SCREEN_ARMS),
                (confirm_dir, tuple(legacy["confirm_200_task"])),
            )
            for name in names
            for seed in seeds
        ],
        sources={
            "ipmnist_screening": Path(ipmnist_screening_module.__file__),
            "rule_discovery": Path(rule_discovery_module.__file__),
            "rule_discovery_summary": Path(__file__),
        },
        repository_root=Path(__file__).resolve().parents[2],
    )
    return maintained


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--confirm-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit a nonpromoting summary to stdout without modifying artifacts."""
    args = _parser().parse_args(argv)
    result = build_rule_discovery_summary(
        args.screen_dir, args.confirm_dir, seeds=args.seeds
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
