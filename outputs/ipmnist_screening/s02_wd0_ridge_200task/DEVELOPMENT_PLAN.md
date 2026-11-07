# Development plan: s02 with RLS-head ridge 0.1 (r01) vs s02 over the full 200-task life

Written and committed before any run (2026-10-31, GMT). Base commit da36e02d (head of our own 60-task ridge screen on s02),
which is stacked on our own #3578, #3610, #3626, #3632, #3636, #3638 and #3646 and is behind `main` (#3648 is a sibling branch, not in this stack).
Line citations are to `alberta_framework/benchmarks/ipmnist_screening.py` at da36e02d. Development screening diagnostic; nonpromoting.

## Credit and lineage (post hoc; disclosed)
The ridge idea is a port of lumix5's #3122 (lowering `rls_ridge_init` 1.0 -> 0.1 on the gated s0031640625 chain) and #3123
(0.1 -> 0.03, a small further increment) onto our s02 lineage. The cell was chosen by a chain of our own screens, each tuned
on the outcome of the last: #3610 (weight_decay 0) -> #3626 (wd0 over 200 tasks) -> #3632 (constant step) -> #3636 (step 0.01)
-> #3638 (step 0.02) -> #3646 (s02 over 200 tasks, PASS) -> #3648 (weight clipping on s02, negative at the tune gate)
-> our 60-task ridge screen at da36e02d (tune seed 165 picked ridge 0.1 over 0.3 and 0.03; screen seeds 166-175 PASS,
+0.002451, CI95 [+0.002244, +0.002657], 10/10). This confirmation checks whether that 60-task gain holds over 200 tasks.
Seeds are fresh, so the confirmation itself is not post hoc, but the arm selection is. Seeds 155-164 (reserved by the
weight-clipping development plan, which stopped at its tune gate) were never run and are released; they are not used here.

## Arms (both registered at da36e02d)
- Candidate: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0_r01` (identical to s02 except `rls_ridge_init` 0.1).
- Incumbent: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (s02; `rls_ridge_init` default 1.0, L4792).
The ridge enters only as the initial head inverse covariance P = I/ridge (read at L4887; L4979, L4998, L5011) and as the
value the head's own shift detector resets P to (L5271-5276). The forgetting factor is 1.0 in both arms, so P does not wind up.

## Seeds
20 confirmation seeds 176-195 at 200 tasks, paired per seed. Unseen: 0 hits for seed176-seed195 in shard names across
3286 refs at 20:13:34 GMT (positive control: seed166 found at da36e02d). No tune: the cell is fixed.

## Procedure
0. **Cache pre-flight:** the OpenML cache must match `openml_cache.sha256` (5 files, same count) or the launcher stops before any job (STATUS=PREFLIGHT_FAIL).
1. **Env check (VOID-stop), bit for bit on all 60 per-task accuracies, at 60 tasks:**
   - `upgd_w_control` seed 0 vs `wave_identmap50_r_idet003_s005625_anneal/shards/` at 3c38b025;
   - incumbent s02 seed 113 vs `wd0_const_step_extend/screen/` at ef85fff8;
   - candidate r01 seed 166 vs `s02_wd0_ridge/screen/` at da36e02d.
   Any mismatch or failed job: VOID, stop.
2. **Confirm:** per seed, the gain is the mean over 200 tasks of (candidate - incumbent) per-task accuracy.
   **PASS** needs all of: mean gain >= +0.002; at least 14/20 seeds improved; CI95 lower bound > 0 (t = 2.093, df 19);
   and no diverged candidate seed. Otherwise FAIL.
3. **Reporting:** per-seed gains; 50-task block gains (tasks 1-50, 51-100, 101-150, 151-200), mean over valid seeds;
   the late block (151-200) is flagged if negative (disclosure only, not a bar).

## Divergence rules (fixed in advance; enforced by the pure functions in `analyze.py`)
A shard is **DIVERGED** if it is missing after one retry, has any non-finite (NaN or Inf) per-task accuracy or loss,
has an accuracy outside [0, 1], or has any single-task accuracy below **0.5**.
- Floor check: chance is 0.1. Across the 200-task s02 shards of #3646 (20 seeds) and every 60-task shard of the ridge screen,
  the lowest single-task accuracy is 0.7752 (200-task s02, #3646) and 0.7736 (60-task ridge screen, s02). A healthy run sits well above the floor.
- Accuracy is `argmax(logits) == y`; a NaN collapse can still give finite accuracies, hence the loss check as well as the floor.
- Consequences: a diverged candidate seed is a FAIL for that seed (scored -1.0, not improved) and is flagged, and **any
  diverged candidate seed makes the verdict FAIL regardless of the mean**. A diverged incumbent shard: **VOID**.

## Resources
40 shards at 200 tasks (about 485-585 s each for this learner family, measured in `wd0_const_step_200task/`) plus 3 env-check
shards; at most 4 one-thread workers, so about 1 h 30 min to 1 h 50 min. 3 GB MemAvailable floor at job start and an in-run
memguard. No commits while it runs. All times GMT.
