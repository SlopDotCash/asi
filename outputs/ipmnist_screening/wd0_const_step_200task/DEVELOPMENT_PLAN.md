# Development plan: wd0 constant step 0.02 vs anneal_f2t1000_wd0 over the full 200-task life

Written and committed before any run (2026-10-31, GMT). Base commit ef85fff8 (head of our own #3638),
which is behind `main` (11 commits at 13:19 GMT; it does not contain main's head). Line citations are to
`alberta_framework/benchmarks/ipmnist_screening.py` at ef85fff8. Development screening diagnostic; nonpromoting.

## Lineage (post hoc; disclosed)
This cell was chosen by a chain of our own screens, each tuned on the outcome of the last:
#3610 (weight_decay 0 on anneal_f2t1000) -> #3626 (wd0 over 200 tasks) -> #3632 (constant step 0.005625 at wd0)
-> #3634 (that constant step over 200 tasks) -> #3636 (step 0.005625 -> 0.01) -> #3638 (0.01 -> 0.02, at the grid edge)
-> an unpublished follow-up screen of 0.03/0.04/0.05 vs 0.02 that FAILED (0.05: +0.001561, below the +0.002 bar),
so 0.02 is kept. A #3636 reviewer asked for a full-life confirmation of the tuned step against the annealed wd0 baseline;
this is it. Seeds are fresh, so the confirmation itself is not post hoc, but the arm selection is.

## Arms (both registered at ef85fff8)
- Candidate: `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` (step_size 0.02 == anneal_floor 0.02, tau 1000, weight_decay 0): constant step (L5517-5521).
- Incumbent: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0` (step_size 0.005625, anneal_floor 0.0028125, tau 1000, weight_decay 0).
At wd0 `param_decay = 1 - step_size*weight_decay` is exactly 1 (L4885, used at L5054-5058).

## Seeds
20 confirmation seeds 134-153 at 200 tasks, paired per seed. Unseen: 0 hits for seed134-seed153 in shard names across
3220 refs at 13:20:06 GMT (positive control: seed113 found at ef85fff8). No tune: the cell is fixed.

## Procedure
0. **Cache pre-flight:** the OpenML cache must match `openml_cache.sha256` (5 files, same count) or the launcher stops before any job (STATUS=PREFLIGHT_FAIL).
1. **Env check (VOID-stop), bit for bit on all 60 per-task accuracies, at 60 tasks:**
   - `upgd_w_control` seed 0 vs `wave_identmap50_r_idet003_s005625_anneal/shards/` at 3c38b025;
   - `..._s02_anneal_f1t1000_wd0` seed 113 vs `wd0_const_step_extend/screen/` at ef85fff8;
   - incumbent `..._s005625_anneal_f2t1000_wd0` seed 40 vs `wd0_confirm200/envcheck/` at ef85fff8 (added in 5332cb22).
   Any mismatch or failed job: VOID, stop.
2. **Confirm:** per seed, the gain is the mean over 200 tasks of (candidate - incumbent) per-task accuracy.
   **PASS** needs all of: mean gain >= +0.002; at least 14/20 seeds improved; CI95 lower bound > 0 (t = 2.093, df 19);
   and no diverged candidate seed. Otherwise FAIL.
3. **Reporting:** per-seed gains; 50-task block gains (tasks 1-50, 51-100, 101-150, 151-200), mean over valid seeds;
   the late block (151-200) is flagged if negative (disclosure only, not a bar).

## Divergence rules (fixed in advance; enforced by the pure functions in `analyze.py`)
A shard is **DIVERGED** if it is missing after one retry, has any non-finite (NaN or Inf) per-task accuracy or loss,
has an accuracy outside [0, 1], or has any single-task accuracy below **0.5**.
- Floor check: chance is 0.1. Across every 200-task shard in the tree at ef85fff8 (all learners, 38 groups), the lowest
  single-task accuracy is 0.6774 (`upgd_w_wd0005`), with no non-finite accuracy or loss; for the incumbent cell itself
  (`wd0_confirm200/`, 20 seeds) it is 0.6958. A healthy 200-task run therefore sits well above 0.5.
- The identmap learner has no finite-state guard (L4849-5530) and accuracy is `argmax(logits) == y` (L5264); `jnp.argmax`
  ranks NaN as the maximum, so a NaN collapse can still give finite accuracies; hence the loss check as well as the floor.
- Consequences: a diverged candidate seed is a FAIL for that seed (scored -1.0, not improved) and is flagged, and **any
  diverged candidate seed makes the verdict FAIL regardless of the mean**. A diverged incumbent shard: **VOID**.

## Resources
40 shards at 200 tasks (about 485-585 s each, measured in `wd0_confirm200/`) plus 3 env-check shards; at most 4 one-thread
workers, so about 1 h 30 min to 1 h 45 min. 3 GB MemAvailable floor at job start and an in-run memguard. No commits while it runs.
