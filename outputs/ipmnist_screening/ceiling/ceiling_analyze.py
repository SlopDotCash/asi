"""Analyze ceiling-run outputs: oracle ceilings + champion error budget."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/shaw/milady/research/alberta/outputs/ipmnist_screening/ceiling")
CONFIRM = Path("/home/shaw/milady/research/alberta/outputs/ipmnist_screening/confirm_full")

LATE_LO, LATE_HI = 4000, 5000  # within-task plateau window
BUCKETS = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 1000),
           (1000, 2000), (2000, 3500), (3500, 5000)]


def load_runs(prefix: str) -> dict[int, dict]:
    runs = {}
    for f in sorted(glob.glob(str(OUT / f"{prefix}_seed*.json"))):
        d = json.loads(Path(f).read_text())
        d["per_step"] = np.load(OUT / f"{d['tag']}_seed{d['seed']}_per_step.npy")
        runs[d["seed"]] = d
    return runs


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def across_seed_spread(values: object) -> float:
    """Sample standard deviation across seeds, matching the campaign convention.

    Issue #33 unified across-seed spread on the sample estimator (``ddof=1``),
    which the screening merge and ``compute_statistics`` both use. ``np.std``
    defaults to the population estimator, understating the spread of a seed
    sample by ``sqrt(n / (n - 1))`` — 22% at n=3, 41% at n=2. A single seed has
    no spread rather than a zero-width interval implying one.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))


def main() -> None:
    report: dict = {}

    # ---- (1a) stationary oracle --------------------------------------------
    print("=" * 70)
    print("(1a) STATIONARY ORACLE — fresh net, unpermuted MNIST, 5000 online steps")
    for spec in ("sigma0_ndecay099", "adamw_control", "sgd_ema_norm"):
        runs = load_runs(f"stationary_{spec}")
        if not runs:
            continue
        means = [r["mean_accuracy"] for r in runs.values()]
        curves = np.stack([r["per_step"][0].astype(np.float64) for r in runs.values()])
        mc = curves.mean(axis=0)
        late = mc[LATE_LO:LATE_HI].mean()
        print(f"  {spec:20s} avg-online={np.mean(means):.5f} (seeds {sorted(runs)}, "
              f"sd {across_seed_spread(means):.5f})  late-window(4000-5000)={late:.5f}")
        bucket_acc = {f"{a}-{b}": round(float(mc[a:b].mean()), 5) for a, b in BUCKETS}
        print(f"      within-task buckets: {bucket_acc}")
        report[f"stationary_{spec}"] = {
            "avg_online_mean": float(np.mean(means)),
            "avg_online_per_seed": means,
            "late_window": float(late),
            "buckets": bucket_acc,
        }

    # ---- (1b) carried oracle ----------------------------------------------
    print("=" * 70)
    print("(1b) CARRIED ORACLE — same permutation every task (no non-stationarity)")
    for spec in ("sigma0_ndecay099", "adamw_control"):
        runs = load_runs(f"carried_{spec}")
        if not runs:
            continue
        pt = np.stack([np.asarray(r["per_task_accuracy"]) for r in runs.values()])
        n_tasks = pt.shape[1]
        late_tasks = pt[:, -10:].mean(axis=1)  # per-seed mean of last 10 tasks
        # per-step plateau in the last 10 tasks
        plate = np.stack([
            r["per_step"][-10:, :].astype(np.float64).mean() for r in runs.values()
        ])
        last_task_late = np.stack([
            r["per_step"][-10:, LATE_LO:LATE_HI].astype(np.float64).mean()
            for r in runs.values()
        ])
        print(f"  {spec:20s} seeds {sorted(runs)}  n_tasks={n_tasks}")
        print(f"      task-avg curve: t1={pt[:, 0].mean():.4f} t5={pt[:, 4].mean():.4f} "
              f"t10={pt[:, 9].mean():.4f} t20={pt[:, 19].mean():.4f} "
              f"t40={pt[:, 39].mean():.4f} t60={pt[:, -1].mean():.4f}")
        print(f"      late-task (last 10 tasks) avg-online = {late_tasks.mean():.5f} "
              f"(sd {across_seed_spread(late_tasks):.5f})")
        print(f"      late-task late-window (steps 4000-5000) = {last_task_late.mean():.5f}")
        report[f"carried_{spec}"] = {
            "per_task_mean_curve": [round(float(v), 5) for v in pt.mean(axis=0)],
            "late_task_avg_online": float(late_tasks.mean()),
            "late_task_late_window": float(last_task_late.mean()),
        }

    # ---- (1c) batch reference ---------------------------------------------
    print("=" * 70)
    print("(1c) BATCH REFERENCE — converged minibatch-Adam 300x150 on MNIST test")
    batch = []
    for f in sorted(glob.glob(str(OUT / "batch_reference_seed*.json"))):
        d = json.loads(Path(f).read_text())
        batch.append(d)
        print(f"  seed {d['seed']}: final={d['test_accuracy_final']} "
              f"best={d['test_accuracy_best']} train20k={d['train_accuracy_20k']}")
    if batch:
        report["batch_reference"] = {
            "test_best_mean": float(np.mean([d["test_accuracy_best"] for d in batch])),
            "per_seed": [d["test_accuracy_best"] for d in batch],
        }

    # ---- (2) error budget of the champion ---------------------------------
    print("=" * 70)
    print("(2) ERROR BUDGET — full-protocol champion (sigma0_ndecay099, 200 tasks)")
    runs = load_runs("full_sigma0_ndecay099")
    if runs:
        # cross-check per-task means against confirm_full shards
        for seed, r in sorted(runs.items()):
            ref = json.loads((CONFIRM / f"sigma0_ndecay099_seed{seed}.json").read_text())
            diff = np.max(np.abs(
                np.asarray(r["per_task_accuracy"]) - np.asarray(ref["per_task_accuracy"])
            ))
            print(f"  seed {seed}: overall={r['mean_accuracy']:.5f}  "
                  f"max|per-task diff vs confirm_full shard|={diff:.2e}")
        A = np.stack([runs[s]["per_step"].astype(np.float64) for s in sorted(runs)])
        # A: [n_seeds, 200, 5000]
        overall = A.mean()
        plateau_t = A[:, :, LATE_LO:LATE_HI].mean(axis=2)          # [seeds, tasks]
        asym_err = (1.0 - plateau_t).mean()
        E = 1.0 - overall
        transient = E - asym_err
        print(f"\n  overall online accuracy = {overall:.5f}  -> total error E = {E:.5f}")
        print(f"  (i)+(iii) asymptotic error (1 - late-window acc, avg over tasks) "
              f"= {asym_err:.5f}  ({100 * asym_err / E:.1f}% of E)")
        print(f"  (ii) within-task transient excess = {transient:.5f}  "
              f"({100 * transient / E:.1f}% of E)")

        # transient bucket contributions (excess error vs each task's own plateau)
        curve = A.mean(axis=(0, 1))                                # [5000]
        plat = plateau_t.mean()
        print(f"\n  mean within-task curve (all 200 tasks, 3 seeds): plateau={plat:.5f}")
        print(f"  {'bucket':>12s} {'acc':>8s} {'excess-err':>10s} {'contrib to E':>12s} {'% of E':>7s}")
        bucket_rows = []
        for a, b in BUCKETS:
            acc = curve[a:b].mean()
            # excess vs per-task plateau, computed exactly:
            excess = (plateau_t[:, :, None] - A[:, :, a:b]).mean()
            contrib = excess * (b - a) / 5000.0
            bucket_rows.append((f"{a}-{b}", round(float(acc), 5),
                                round(float(excess), 5), round(float(contrib), 5)))
            print(f"  {a:>5d}-{b:<6d} {acc:>8.5f} {excess:>10.5f} {contrib:>12.5f} "
                  f"{100 * contrib / E:>6.1f}%")

        # first-500 vs late-window (task-requested metric)
        first500 = A[:, :, :500].mean()
        print(f"\n  first-500-step accuracy (avg over tasks) = {first500:.5f} "
              f"vs late-window = {plat:.5f}  (gap {plat - first500:.5f})")

        # (iii) late-life drift: plateau_t trend across tasks
        pt_mean = plateau_t.mean(axis=0)  # [200]
        t = np.arange(200)
        mask = t >= 20
        slope = np.polyfit(t[mask], pt_mean[mask], 1)[0]
        e20_60 = pt_mean[20:60].mean()
        e160_200 = pt_mean[160:200].mean()
        avg_online_t = A.mean(axis=(0, 2))
        print(f"\n  late-life drift: plateau tasks 20-60 = {e20_60:.5f}, "
              f"tasks 160-200 = {e160_200:.5f}  (delta {e160_200 - e20_60:+.5f})")
        print(f"  linear slope (tasks 20-200) = {slope:+.2e}/task "
              f"-> {slope * 180:+.5f} over 180 tasks")
        print(f"  avg-online tasks 20-60 = {avg_online_t[20:60].mean():.5f}, "
              f"tasks 160-200 = {avg_online_t[160:200].mean():.5f}")
        # early-life warmup contribution: tasks 0-19 vs 20+ average
        early_tasks = avg_online_t[:20].mean()
        late_avg = avg_online_t[20:].mean()
        print(f"  early-life (tasks 0-19 avg-online) = {early_tasks:.5f} vs "
              f"tasks 20-199 = {late_avg:.5f} -> first-20-task warmup costs "
              f"{(late_avg - early_tasks) * 20 / 200:.5f} of the overall score")

        report["error_budget"] = {
            "overall": float(overall),
            "total_error": float(E),
            "asymptotic_error": float(asym_err),
            "transient_excess": float(transient),
            "plateau_mean": float(plat),
            "first500": float(first500),
            "buckets": bucket_rows,
            "plateau_tasks_20_60": float(e20_60),
            "plateau_tasks_160_200": float(e160_200),
            "drift_slope_per_task": float(slope),
        }

    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=1))
    print("\nWROTE", OUT / "analysis_summary.json")


if __name__ == "__main__":
    main()
