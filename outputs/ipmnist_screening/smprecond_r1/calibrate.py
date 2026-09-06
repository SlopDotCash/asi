"""smprecond_r1 step-size calibration — 3 tasks, seed 0 ONLY (tuning).

Gross-viability grid per PREREGISTRATION.md: divergence/NaN/dead-arm
exclusion, not ranking (negative results #1, #20). Writes one JSON per arm
under calibration/ (new paths, append-only).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    IPMNISTConfig,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    default_openml_data_home,
    load_mnist_train,
)

OUT = Path(__file__).parent / "calibration"
OUT.mkdir(exist_ok=True)

GRID = (0.0003, 0.001, 0.003, 0.01)
SEED = 0
CONFIG = IPMNISTConfig(n_tasks=3, task_length=5000)


def main() -> None:
    data_x, data_y = load_mnist_train(default_openml_data_home())
    base = screening_spec("rls_head_resid_identmap50_r")
    arms = [("incumbent_ref", base)]
    for sm_step in GRID:
        tag = f"sm{sm_step:g}"
        arms.append(
            (
                tag,
                replace(
                    base,
                    name=base.name + "_" + tag,
                    hyperparameters={
                        **base.hyperparameters,
                        "body_sm_decay": 0.999,
                        "body_sm_step": sm_step,
                        "body_sm_eps": 1e-8,
                    },
                ),
            )
        )
    for tag, spec in arms:
        out = OUT / f"{tag}_seed{SEED}.json"
        if out.exists():
            print(f"skip {tag} (exists)")
            continue
        result = run_screening_config(data_x, data_y, spec, SEED, CONFIG)
        acc = np.asarray(result.per_task_accuracy, dtype=np.float64)
        payload = {
            "arm": tag,
            "spec_name": spec.name,
            "seed": SEED,
            "n_tasks": CONFIG.n_tasks,
            "task_length": CONFIG.task_length,
            "per_task_accuracy": [float(a) for a in acc],
            "mean_accuracy": float(acc.mean()),
            "finite": bool(np.all(np.isfinite(acc))),
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(tag, payload["per_task_accuracy"], payload["mean_accuracy"])


if __name__ == "__main__":
    main()
