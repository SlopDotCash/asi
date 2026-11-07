# Preregistration: anneal_f4t1000 vs anneal_f2t1000 over the full 200-task horizon (seeds 0-19)

Committed before any run or code change in this directory. Branch `confirm/identmap50-r-idet003-s005625-anneal-f4t1000-200task`, cut from 3c38b025 (main fd25c476, fetched from the main clone). No source changes: both configs are already in the registry at 3c38b025.

## Question
Does more within-task decay of the body step pay over a full 200-task life? This is a test of a mechanism, not a promotion of a past winner.
- Arm: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000`, schedule max(0.00140625, 0.005625·(1+s/1000)^-½). The step keeps decaying through each 5000-step task.
- Parent (off): `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`, schedule max(0.0028125, 0.005625·(1+s/1000)^-½). This is the confirmed anneal arm (+0.006576 over its own parent at 200 tasks, 20/20 seeds, commit 612491db), so it is the new incumbent.

## Disclosure of prior information
- **f4t1000 FAILED at 60 tasks.** In the anneal sweep (0240ff25), f4t1000 minus f2t1000 was −0.000077 paired on seeds 0-9, with 2/10 seeds improved. Its gap by task block was −0.00046 (tasks 1-15), −0.00008 (16-30), 0.00000 (31-45) and +0.00023 (46-60). That rising trend is the only reason for this test.
- **Seeds 0-9 have been seen at 60 tasks** in that sweep, and those first 60 tasks are prefix-identical to this run. Seeds 10-19 have never been run for f4t1000. Tasks 61-200 have not been run for f4t1000 on any seed.
- The parent's 200-task shards for seeds 0-19 (612491db) have been inspected; that is the published confirmation.
- **Prior estimate:** the gain over 200 tasks is probably **+0.0005 to +0.0015**, below the +0.002 bar. This comes from a linear extrapolation of the per-block trend. **A negative result is the expected outcome, as planned**, and will be recorded as a negative result that closes this axis of the anneal schedule.

## Design
- Seeds 0-19 (same as the confirmation lane); 200 tasks × 5000 steps. Each worker is single-threaded; at most 4 workers run in parallel.
- Primary statistic: per seed, Δ = mean over 200 tasks of per_task_accuracy(arm) − same(parent), paired by seed.

## Bars (ALL required; this is the only comparison, so no multiplicity adjustment)
1. Mean paired Δ ≥ **+0.002**.
2. At least **14/20** seeds improved (Δ > 0).
3. 95% CI lower bound **> 0**: mean − t·SE, with t(0.975, df=19) = 2.093024.

If any bar fails: FAILS, recorded as a negative result, and no promotion.

## Environment check (runs first; ALL must match bit-for-bit, or the run is VOID)
Following the #3585 review, the check covers the anneal path and both configs under test:
1. `upgd_w_control` seed 0, 60 tasks, vs stored `wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json` (at 3c38b025).
2. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000` seed 0, 200 tasks, vs stored `anneal_f2t1000_confirm_r1/shards/rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed0.json` (at 612491db).
3. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000` seed 0, 60 tasks, vs stored `wave_identmap50_r_idet003_s005625_anneal_sweep/shards/rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000_seed0.json` (at 0240ff25).

"Bit-for-bit" means the per_task_accuracy lists are exactly equal (same length, every float identical). If any check fails, the launcher stops before running any arm job. The run is then VOID, not a result, and is reported as such.

## Parent reuse rule
The 20 parent shards at 200 tasks from 612491db are reused **only if check 2 reproduces exactly**. They are copied verbatim, with SHA256 sums recorded. If check 2 fails, the run is void anyway (see above). An explicit re-run of all 20 parents is then needed under a fresh decision; nothing is mixed.

## Reporting
- A per-seed table, mean Δ, the 95% CI, the improved count and the verdict.
- Gains by 50-task block (1-50, 51-100, 101-150, 151-200): reported only, not bars.
- The three environment-check outcomes.

## Stop rules
- Do not start without the heavy slot, i.e. no other heavy benchmark job running. Hold at most 4 workers and keep at least 3 GB of RAM available.
- Stop on any environment-check mismatch (VOID).
- If any arm job fails or crashes, re-run that seed once with the same command. If it fails again, the run is VOID, with no partial verdict.
- No looking at partial results to stop early. The verdict is computed once, after all 20 arm shards exist.
- No extra seeds, horizons or configs after the fact. Stored results are never edited.
