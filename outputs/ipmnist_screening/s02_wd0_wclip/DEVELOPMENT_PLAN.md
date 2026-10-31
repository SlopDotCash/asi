# Development plan: residual-body weight clipping (kappa 2 / 4 / 8) on the s02 wd0 incumbent

Written and committed before any run (2026-10-31, GMT). Base commit 484546ad (head of #3646, where
`rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (constant body step 0.02, weight_decay 0) beat `..._s005625_anneal_f2t1000_wd0` over the full
200 tasks, +0.007665, 20/20). Line citations are to `alberta_framework/benchmarks/ipmnist_screening.py`
on this branch.

## Question
At weight_decay 0 nothing bounds the body weights any more (`param_decay = 1 - step_size*weight_decay`
is exactly 1, L4885). Does per-layer weight clipping of the residual body, the Weight Clipping
mechanism already used by `upgd_w_wclip_*` and `wclip_ema_norm`, help that incumbent?

## Mechanism (new code: one opt-in key)
- `body_wclip_kappa` in `_make_rls_head_learner` (def L4849; validation L4897-4910). After each
  residual-body update (L5267-5270), `w1`, `b1`, `w2`, `b2` are clipped to `[-kappa*s_l, +kappa*s_l]`,
  `s_l = 1/sqrt(fan_in)` (`_wclip_bound`, L3150, the existing helper). The RLS readout (`wout`, `P`) and
  the unused `w3`/`b3` are never clipped. The identmap learner passes hp straight to the inner learner
  (L5477), so the anneal and identifier paths are unchanged.
- Key absent: no clip is traced, so every existing cell is unchanged (the identmap registry test still pins
  every arm). A bound far above any weight (kappa 1e30) is pinned bit-exact to the parent trajectory.
- Prior result: `wclip_ema_norm` (kappa 2, wd 0) was -0.01130 vs its control in FINAL_REPORT.md
  (an earlier, different lane, without an RLS head). It has never been tested in the RLS-head lane.

## Arms
- Incumbent: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (unchanged registry entry).
- New: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0_wclip2`, `_wclip4`, `_wclip8` (kappa 2.0 / 4.0 / 8.0), identical to the incumbent
  except for `body_wclip_kappa`. kappa 2 is the paper's value; 4 and 8 are looser bounds, closer to the parent.

## Seeds (unseen; 0 hits for seeds 154-179 in shard names across 3126 unique commits on all local and
PR refs, 2026-10-31 18:35 GMT; seeds 0-153 are used)
- Tune (development) seed: 154. Screen seeds: 155-164. All at 60 tasks, task length 5000.
- Seeds 165-179 are not reserved here; a 200-task confirmation would need its own plan and check.

## Procedure
0. **Cache pre-flight:** the OpenML cache must match `openml_cache.sha256` (5 files, same count) or the
   launcher stops before any job (STATUS=PREFLIGHT_FAIL).
1. **Env check (VOID-stop):** `upgd_w_control` seed 0 (60 tasks) must match the committed shard at
   3c38b025, and the stored s02 shard `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` seed 113 (60 tasks) must match `wd0_const_step_extend/screen/`
   at ef85fff8, bit for bit. Any failure or mismatch: STATUS=VOID, nothing further runs.
2. **Tune gate (seed 154):** the paired mean per-task accuracy gain of each kappa cell over s02. Diverged
   cells are excluded. STOP unless the best eligible cell beats s02 by more than +0.0005. Ties
   (within 1e-9) go to the larger kappa (the looser bound, closer to the parent).
3. **Screen (seeds 155-164):** winner vs s02, paired. PASS needs all of: mean gain >= +0.002, at least
   7/10 seeds improved, CI95 lower bound > 0 (t = 2.262, df 9), **and no diverged winner seed**. 10-task
   block means are reported.
4. **Edge rule:** if kappa 2 (the tightest) wins, the result carries an EDGE FLAG (a tighter bound may do
   better); the verdict is unchanged.

## Divergence rules (fixed in advance; enforced mechanically by `analyze.py`)
A shard is **DIVERGED** if any of:
- it is missing after one retry (crash);
- any per-task accuracy or per-task loss is non-finite (NaN or Inf);
- any per-task accuracy is outside [0, 1];
- any single-task accuracy is below the floor **0.5**.

Justification for 0.5: chance is 0.1. The lowest single-task accuracy of the 11 committed s02 60-task
shards (`wd0_const_step_extend/`) is 0.787, and of the 20 s02 200-task shards (`wd0_const_step_200task/`)
it is 0.7752. Earlier wd0 constant-step shards never went below 0.7104.

Consequences: a diverged new cell is excluded at the tune (STOP_ALL_DIVERGED if all are). Any diverged
screen seed of the winner makes the verdict FAIL regardless of the mean (it is scored -1.0, not
improved). A diverged incumbent shard voids the run (VOID). One automatic retry per missing shard; no
other reruns.

## Runtime and resources
26 shards at 60 tasks (2 env check, 4 tune, 20 screen), at most 4 one-thread workers, 3 GB
MemAvailable floor (`memguard.sh` pauses the newest worker; pausing does not change results). Estimate
about 30 min (the same-shape extend2 run took 26 min).
