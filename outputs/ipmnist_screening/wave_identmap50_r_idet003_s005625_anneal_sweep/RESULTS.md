# Anneal schedule sweep on anneal_f2t1000: RESULTS (both arms negative)

Preregistration: `PREREGISTRATION.md` (e4a51694, committed before any code or run). Registry entry and tests: cece144a. Wave infra: a783d4ae.
Branch from 3c38b025; origin/main fd25c476 (unmoved as of 20:10 GMT). Lane status: development_screening_diagnostic, nonpromoting.

## Parent reuse (all three checks pass)
1. Jaxpr of the parent `..._anneal_f2t1000` (sha256 bfb2f7ed…) and of `..._anneal_f4t1000` (fc09ccde…) are byte-identical before and after the registry change.
2. Environment: the frozen uv sync has the same package set. The shard `environment` and `dataset_provenance` blocks equal the stored ones.
3. The fresh seed-0 60-task parent re-run (`parent_check/`) equals the stored anneal-wave shard bit-for-bit, in both accuracy and loss.

Pairing therefore uses `../wave_identmap50_r_idet003_s005625_anneal/shards/` for seeds 0-9.

Control `upgd_w_control` re-measured on seeds 0-2: 0.777477, 0.778560 and 0.777240. All three are bit-exact against the stored control shards.

## Screen: seeds 0-9, 60 tasks × 5000

### f2t2000 vs parent f2t1000
| seed | parent | f2t2000 | paired Δ |
|---|---|---|---|
| 0 | 0.936323 | 0.936290 | -0.000033 |
| 1 | 0.937073 | 0.936673 | -0.000400 |
| 2 | 0.937107 | 0.936777 | -0.000330 |
| 3 | 0.936637 | 0.936367 | -0.000270 |
| 4 | 0.937797 | 0.937617 | -0.000180 |
| 5 | 0.937553 | 0.937477 | -0.000077 |
| 6 | 0.937383 | 0.937087 | -0.000297 |
| 7 | 0.937333 | 0.936957 | -0.000377 |
| 8 | 0.937380 | 0.937033 | -0.000347 |
| 9 | 0.937170 | 0.936920 | -0.000250 |

mean -0.000256; 95% CI [-0.000345, -0.000167]; Bonferroni 97.5% CI [-0.000361, -0.000151]; improved 0/10 -> FAILS
horizon: 1-15 +0.001273; 16-30 -0.000117; 31-45 -0.000920; 46-60 -0.001260

### f4t1000 vs parent f2t1000
| seed | parent | f4t1000 | paired Δ |
|---|---|---|---|
| 0 | 0.936323 | 0.936263 | -0.000060 |
| 1 | 0.937073 | 0.936690 | -0.000383 |
| 2 | 0.937107 | 0.937153 | +0.000047 |
| 3 | 0.936637 | 0.936460 | -0.000177 |
| 4 | 0.937797 | 0.937783 | -0.000013 |
| 5 | 0.937553 | 0.937600 | +0.000047 |
| 6 | 0.937383 | 0.937323 | -0.000060 |
| 7 | 0.937333 | 0.937223 | -0.000110 |
| 8 | 0.937380 | 0.937357 | -0.000023 |
| 9 | 0.937170 | 0.937130 | -0.000040 |

mean -0.000077; 95% CI [-0.000168, +0.000013]; Bonferroni 97.5% CI [-0.000185, +0.000030]; improved 2/10 -> FAILS
horizon: 1-15 -0.000459; 16-30 -0.000076; 31-45 -0.000000; 46-60 +0.000225

### Secondary f4t1000 − f2t2000: mean +0.000179; 95% CI [+0.000086, +0.000271]; f4t1000 better on 9/10
horizon: 1-15 -0.001732; 16-30 +0.000041; 31-45 +0.000920; 46-60 +0.001485

## Decision (preregistered rule applied to each arm on its own)
- **f2t2000 (gentler): FAILS.** Paired -0.000256, 0/10 seeds improved, and the CI lies entirely below 0.
- **f4t1000 (stronger): FAILS.** Paired -0.000077, 2/10 seeds improved, and the CI straddles 0.
- No arm is promoted. Seeds 10-19 are untouched.

## Reading
- The horizon pattern holds for both arms.
  - The gentler arm wins early, at tasks 1-15 (+0.0013), and loses late, at tasks 46-60 (-0.0013).
  - The stronger arm loses early (-0.00046) and gains late (+0.00023).
- The secondary comparison (f4t1000 − f2t2000, positive on 9/10 seeds, CI above 0, rising to +0.0015 by tasks 46-60) shows that more annealing helps later in life.
- Within 60 tasks, though, the parent's floor/tau sits close to the optimum on this axis. Neither move comes near +0.002.
- The late-life trend of f4t1000 is the only lead. It would need a 200-task test, which is out of scope here and not preregistered.

## Reproduce
From the repo root, run `bash <wave>/launch.sh <wave>/jobs.txt`. Then, from `outputs/ipmnist_screening/`, run `python3 <wave>/analyze.py`.
