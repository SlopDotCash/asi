# Development plan: RLS head ridge (0.3 / 0.1 / 0.03) on the s02 wd0 incumbent

Written and committed before any run (2026-10-31, GMT). Base commit 484546ad (head of #3646, where
`rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (constant body step 0.02, weight_decay 0)
beat `..._s005625_anneal_f2t1000_wd0` over the full 200 tasks, +0.007665, 20/20). Line citations are to
`alberta_framework/benchmarks/ipmnist_screening.py` on this branch.

## Question
The s02 RLS head starts and restarts its inverse covariance at `P = I / rls_ridge_init` with the default
ridge 1.0. On a sister chain (gated, body step 0.0031640625, weight_decay 0.01), lowering this ridge to
0.1 gave +0.0104 at 60 tasks (#3122) and 0.03 a further +0.0006 (#3123, axis flattening). Neither cell
has been measured on the weight_decay 0 lineage. Does a smaller ridge, meaning a larger RLS gain right
after each reset, help s02?

## Mechanism (no new code path: registry cells only)
- `ridge_init = hp["rls_ridge_init"]` (L4887) sets the initial `P = eye / ridge_init` (L4979, L4998,
  L5011) and the value P is reset to when the learner's own shift detector fires
  (`mean(shifted) >= rls_reset_frac` = 0.05, L5273-5276; `wout` is kept).
- The update is Sherman-Morrison RLS with `rls_lambda = 1.0` (L5266-5270), so between resets P is
  nonincreasing and bounded by `I / ridge`. The wind-up collapse recorded for small ridges at lambda
  0.999 (`docs/research/ipmnist-theory.md`, section 3) cannot occur here.
- Boundary signal: the reset is driven by the learner's own detector on its input stream. No oracle task
  boundary is used.
- Literature: in Haykin, *Adaptive Filter Theory*, the RLS recursion is initialised with
  `P(0) = delta^-1 I`, and a small delta suits high-SNR data. Covariance resetting to a fixed
  `P = I/ridge` on a detected change is the adaptive-control remedy for gain loss (Goodwin and Sin,
  *Adaptive Filtering, Prediction and Control*, 1984). This plan re-tunes that reset value.
- New code: three registry cells (L8835-8855). The learner is unchanged.

## Arms
- Incumbent: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (unchanged registry entry, ridge 1.0).
- New: `..._wd0_r03`, `..._wd0_r01`, `..._wd0_r003` (ridge 0.3 / 0.1 / 0.03). Each is identical to the
  incumbent except for `rls_ridge_init`. 0.1 and 0.03 are the sister-chain cells; 0.3 is the midpoint
  towards the parent.

## Seeds (unseen; no shard name with seed >= 155 below 15780 on any local or PR ref, 2026-10-31 19:31 GMT;
154 is used by the weight-clipping tune, and 155-164 are reserved by that plan)
- Tune (development) seed: 165. Screen seeds: 166-175. All at 60 tasks, task length 5000.

## Procedure
0. **Cache pre-flight:** the OpenML cache must match `openml_cache.sha256` (5 files, same count) or the
   launcher stops before any job (STATUS=PREFLIGHT_FAIL).
1. **Env check (VOID-stop):** `upgd_w_control` seed 0 (60 tasks) must match the committed shard at
   3c38b025, and the stored s02 shard seed 113 (60 tasks) must match `wd0_const_step_extend/screen/`
   at ef85fff8, bit for bit. Any failure or mismatch: STATUS=VOID, and nothing further runs.
2. **Tune gate (seed 165):** the paired mean per-task accuracy gain of each ridge cell over s02. Diverged
   cells are excluded. STOP unless the best eligible cell beats s02 by more than +0.0005. Ties
   (within 1e-9) go to the larger ridge (closer to the parent).
3. **Screen (seeds 166-175):** winner vs s02, paired. PASS needs all of: mean gain >= +0.002, at least
   7/10 seeds improved, CI95 lower bound > 0 (t = 2.262, df 9), **and no diverged winner seed**.
   10-task block means are reported.
4. **Edge rule:** if ridge 0.03 (the smallest) wins, the result carries an EDGE FLAG (a smaller ridge may
   do better); the verdict is unchanged.

## Divergence rules (fixed in advance; enforced mechanically by `analyze.py`)
A shard is **DIVERGED** if any of:
- it is missing after one retry (crash);
- any per-task accuracy or per-task loss is non-finite (NaN or Inf);
- any per-task accuracy is outside [0, 1];
- any single-task accuracy is below the floor **0.5**.

Justification for 0.5: chance is 0.1. The lowest single-task accuracy of the 11 committed s02 60-task
shards (`wd0_const_step_extend/`) is 0.787, and of the 20 s02 200-task shards (`wd0_const_step_200task/`)
it is 0.7752. Small-ridge failures recorded at lambda 0.999 were collapses, so they would trip this floor.

Consequences: a diverged new cell is excluded at the tune (STOP_ALL_DIVERGED if all are). Any diverged
screen seed of the winner makes the verdict FAIL regardless of the mean (it is scored -1.0, not
improved). A diverged incumbent shard voids the run (VOID). One automatic retry per missing shard; no
other reruns.

## Runtime and resources
26 shards at 60 tasks (2 env check, 4 tune, 20 screen), at most 4 one-thread workers, 3 GB
MemAvailable floor (`memguard.sh` pauses the newest worker; pausing does not change results). Estimate
about 30 min (the same-shape weight-clipping run did the env check and tune in 12 min).
