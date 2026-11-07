"""Plot descriptive cumulative rewards; no additional selection metric."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
plt.rcParams.update({"svg.hashsalt": "riverswim-discount-horizon-2026-10-15", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), sharey=True)
for ax, seed in zip(axes, (101001, 101002, 101003), strict=True):
    for arm, color, style, label in (
        ("gamma_090", "#D9484E", "--", "gamma 0.90"),
        ("gamma_099", "#188C2B", "-", "gamma 0.99"),
    ):
        record = json.loads((args.records / f"{arm}-seed{seed}.json").read_text())
        rewards = np.asarray(record["rewards"], dtype=np.float64)
        x = np.arange(1, len(rewards) + 1)
        indices = np.r_[np.arange(0, len(rewards), 100), len(rewards) - 1]
        ax.plot(x[indices], (np.cumsum(rewards) / x)[indices], style,
                color=color, label=label, linewidth=1.8)
    ax.set_title(f"Development seed {seed}")
    ax.set_xlabel("Accepted environment events")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 1)
    ax.set_xticks((0, 10000, 20000), ("0", "10k", "20k"))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Cumulative mean reward")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("RiverSwim: discount probe at lambda 0.8, learning rate 0.014", y=1.02)
fig.tight_layout(rect=(0, 0.07, 1, 1))
args.out.mkdir(exist_ok=True)
fig.savefig(args.out / "reward-prefixes.png", dpi=160, bbox_inches="tight")
svg = args.out / "reward-prefixes.svg"
fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
plt.close(fig)
svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8", newline="\n",
)
print("Descriptive plots saved; matplotlib", matplotlib.__version__)
