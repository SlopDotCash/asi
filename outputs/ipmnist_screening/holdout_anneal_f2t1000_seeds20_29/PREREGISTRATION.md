# Preregistration: holdout of anneal_f2t1000 on unseen seeds 20-29 (60 tasks)

Committed before any run. Branch `holdout/identmap50-r-idet003-s005625-anneal-seeds20-29` is cut from 3c38b025. origin/main is fd25c476, fetched from the main local clone at 20:43 GMT, unmoved.
Lane status: development_screening_diagnostic, nonpromoting.

## Why seeds 20-29 rather than the preregistered 10-19 holdout
The anneal screen held out seeds 10-19. The 200-task, 20-seed confirmation (612491db) has since run seeds 10-19 for both this arm and its parent. For seeds 0-9 the 60-task runs were shown to be bit-identical to the first 60 tasks of the 200-task runs, and the same code path applies to seeds 10-19. So a 60-task run on 10-19 would be fixed by existing shards: it is not unseen evidence. No 10-19 run was launched, and the 10-19 first-60 numbers were not inspected for this decision.

Seeds 20-29 have never been run for this arm or for its parent on any branch, so they are genuinely unseen.

## Comparison
- Candidate: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`. Its body step within each detected task is `max(0.005625/2, 0.005625*(1+since_shift/1000)^-1/2)`.
- Parent: `rls_head_resid_identmap50_r_idet003_s005625`, run **fresh** on seeds 20-29. Nothing is cached for these seeds, so nothing is reused.
- Seeds 20-29 at 60 tasks × 5000, paired per seed.
- Control `upgd_w_control`:
  - re-measured on seed 20, reported only (there is no stored value);
  - re-run on seed 0, which must match the stored anneal-wave control shard bit-for-bit to prove the environment.

## Bar (same as the original anneal screen's 10-seed decision)
- Mean paired Δ (candidate − parent) ≥ +0.002.
- At least 7/10 seeds improved. This is the screen's 2-of-3 rule scaled to 10 seeds, as already used in the anneal sweep.
- 95% t-CI lower bound > 0 (df=9, t=2.262157).
- All three conditions are required. A miss gets a negative-results entry.

## Reported
- per-seed table, mean, 95% CI, seeds improved;
- horizon breakdown over tasks 1-15, 16-30, 31-45, 46-60;
- the control values;
- for context only, alongside the prior screen (seeds 0-9, 60 tasks) and the 200-task confirmation (+0.006576, 20/20).

Single comparison, so no multiplicity adjustment.
