"""Descriptive reward-prefix curves; never an additional selection metric."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
styles = {
    "trace_original": ("#c44e52", "--", "lambda 0.8, step 0.05"),
    "trace_scaled": ("#228833", "-", "lambda 0.8, step 0.014"),
    "no_trace_original": ("#4477aa", "--", "lambda 0, step 0.05"),
    "no_trace_scaled": ("#66aadd", "-", "lambda 0, step 0.014"),
}
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "asi-step-size-probe-2026-10-15"})
fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharex=True, sharey=True)
for ax, seed in zip(axes, (100601, 100602, 100603), strict=True):
    for arm, (color, linestyle, label) in styles.items():
        record = json.loads((args.records / f"{arm}-seed{seed}.json").read_text(encoding="utf-8"))
        rewards = np.asarray(record["rewards"], dtype=np.float64)
        assert len(rewards) == 20000 and np.isfinite(rewards).all()
        prefix = np.cumsum(rewards) / np.arange(1, 20001)
        indices = np.unique(np.r_[0, np.arange(99, 20000, 100), 19999])
        ax.plot(indices + 1, prefix[indices], color=color, linestyle=linestyle, lw=1.8, label=label)
    ax.set(title=f"Development seed {seed}", xlabel="Accepted environment events")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 1)
    ax.set_xticks((0, 10000, 20000), ("0", "10k", "20k"))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Cumulative mean reward")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("RiverSwim: fixed eligibility-trace / step-size factorial", y=1.02, fontsize=13)
fig.tight_layout(rect=(0, 0.1, 1, 1))
args.out.mkdir(exist_ok=True)
fig.savefig(args.out / "reward-prefixes.png", dpi=160, bbox_inches="tight")
fig.savefig(args.out / "reward-prefixes.svg", bbox_inches="tight", metadata={"Date": None})
plt.close(fig)
svg = args.out / "reward-prefixes.svg"
svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8", newline="\n",
)
print("Saved descriptive reward-prefix plots; plotting runtime matplotlib", matplotlib.__version__)
