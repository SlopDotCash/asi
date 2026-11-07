"""Render checked plots for the frozen RiverSwim development panel."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

SCHEMA = "asi.riverswim_sarsa_discount_horizon.shard.v1"
ARMS = ("gamma_090", "gamma_099")
SEEDS = (101001, 101002, 101003)
HORIZON = 20000
FROZEN_ROOT = Path(__file__).resolve().parent.parent


def sample_indices(length: int, stride: int = 100) -> list[int]:
    if length < 1 or stride < 1:
        raise ValueError("length and stride must be positive")
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def load_records(folder: Path) -> dict[tuple[int, str], list[float]]:
    records = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = folder / f"{arm}-seed{seed}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("schema") != SCHEMA:
                raise ValueError(f"{path.name}: unexpected schema")
            if type(record.get("seed")) is not int or record["seed"] != seed:
                raise ValueError(f"{path.name}: unexpected seed")
            if record.get("arm") != arm:
                raise ValueError(f"{path.name}: unexpected arm")
            if (
                type(record.get("accepted_events")) is not int
                or record["accepted_events"] != HORIZON
            ):
                raise ValueError(f"{path.name}: invalid accepted event count")
            config = record.get("config")
            expected = {
                "gamma": 0.9 if arm == "gamma_090" else 0.99,
                "lamda": 0.8,
                "step_size": 0.014,
            }
            if not isinstance(config, dict) or any(
                config.get(key) != struct.unpack("f", struct.pack("f", value))[0]
                for key, value in expected.items()
            ):
                raise ValueError(f"{path.name}: unexpected configuration")
            for name in ("actions", "positions", "rewards"):
                values = record.get(name)
                if not isinstance(values, list) or len(values) != HORIZON:
                    raise ValueError(f"{path.name}: invalid {name} length")
            rewards = record["rewards"]
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in rewards
            ):
                raise ValueError(f"{path.name}: invalid rewards")
            records[(seed, arm)] = rewards
    return records


def output_paths(folder: Path, records: Path) -> tuple[Path, Path]:
    out = folder.resolve()
    for protected in (FROZEN_ROOT, records.resolve()):
        if out == protected or out.is_relative_to(protected):
            raise ValueError("output must be outside the frozen and input directories")
    paths = (out / "reward-prefixes.png", out / "reward-prefixes.svg")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("output files must not already exist")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.records)
    png, svg = output_paths(args.out, args.records)
    png.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"svg.hashsalt": "riverswim-discount-horizon-2026-10-15", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), sharey=True)
    indices = sample_indices(HORIZON)
    x = np.arange(1, HORIZON + 1)
    for ax, seed in zip(axes, SEEDS, strict=True):
        for arm, color, style, label in (
            ("gamma_090", "#D9484E", "--", "gamma 0.90"),
            ("gamma_099", "#188C2B", "-", "gamma 0.99"),
        ):
            rewards = np.asarray(records[(seed, arm)], dtype=np.float64)
            ax.plot(
                x[indices],
                (np.cumsum(rewards) / x)[indices],
                style,
                color=color,
                label=label,
                linewidth=1.8,
            )
        ax.set_title(f"Development seed {seed}")
        ax.set_xlabel("Accepted environment events")
        ax.set_xlim(0, HORIZON)
        ax.set_ylim(0, 1)
        ax.set_xticks((0, 10000, 20000), ("0", "10k", "20k"))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cumulative mean reward")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("RiverSwim: discount probe at lambda 0.8, learning rate 0.014", y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
