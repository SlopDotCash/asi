# Polyak body iterate averaging on anneal_f2t1000 — RESULTS (negative)

Preregistration: `PREREGISTRATION.md` (commit 451b6729, before any code or run).
Arm/tests: 2947bac3. Wave infra: aa879362. Branch from 3c38b025; origin/main fd25c476 (unmoved).
Status: `development_screening_diagnostic`, nonpromoting.

## Parent reuse (all three checks pass)
1. Traced code: the SMALL-config jaxprs of `..._anneal_f2t1000` (sha256 bfb2f7ed…) and `..._idet003_s005625` (ec000d32…) are byte-identical before and after the patch.
2. Environment: `uv sync --frozen --extra dev`, empty freeze diff; the shard `environment` and `dataset_provenance` blocks equal the stored ones.
3. Exact match: the seed-0 60-task re-run (`parent_check/`) equals the stored anneal-wave shard bit-for-bit (per-task accuracy and loss). The 30-task tune parent equals its 30-task prefix.
So pairing uses the stored `../wave_identmap50_r_idet003_s005625_anneal/shards/` parent, seeds 0-2.

## Tune (seed 0, 30 tasks; parent 0.921887)
| cell | Δ1-30 | Δ1-3 | Δ1-15 | Δ16-30 |
|---|---|---|---|---|
| w100h0 | **-0.000353** | -0.001533 | -0.000787 | +0.000080 |
| w100h1 | -0.000607 | -0.002600 | -0.001053 | -0.000160 |
| w1000h0 | -0.000733 | -0.007867 | -0.002040 | +0.000573 |
| w1000h1 | -0.001713 | -0.013200 | -0.003320 | -0.000107 |

All four cells are below the parent. Under the preregistered rule, the best one, w100h0, advances. Averaging into the RLS head (h1) is worse than h0 at both windows.

## Screen (seeds 0-2, 60 tasks x 5000), w100h0 vs parent
| seed | parent (stored, exact-match verified) | polyak w100h0 | paired Δ |
|---|---|---|---|
| 0 | 0.936323 | 0.936273 | -0.000050 |
| 1 | 0.937073 | 0.937007 | -0.000067 |
| 2 | 0.937107 | 0.937427 | +0.000320 |

Paired mean **+0.000068**, 95% t-CI [-0.000475, +0.000611] (df=2), **1/3 seeds improved**.

Horizon breakdown (mean paired Δ): tasks 1-15 -0.000498; 16-30 -0.000107; 31-45 +0.000191; 46-60 +0.000684.
The late-life trend is positive but is about 3x below the bar. It is not run further (as preregistered, there is no extension off a failed screen).

Bar (≥ +0.002, ≥ 2/3 improved): **FAILS**. Seeds 3-9 were not run, and 10-19 are untouched.

Control `upgd_w_control` (re-measured, seeds 0-2): 0.777477 / 0.778560 / 0.777240, bit-exact equal to the anneal-wave control shards on all 3 seeds.

## Reproduce
`bash launch.sh jobs_tune.txt`, then `bash launch.sh jobs_screen.txt` (from the repo root). Analysis: `python3 analyze.py` from `outputs/ipmnist_screening/`.
