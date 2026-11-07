# wave_identmap50_r_idet003_s005625_anneal_polyak

Polyak-style body iterate averaging (window-capped running mean, reset at each detected shift and at every remap, prediction-only) on the parent `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`.
- Read `PREREGISTRATION.md` first, then `RESULTS.md`.
- Result: **negative** (+0.000068, 1/3 seeds improved; the bar is +0.002 with 2/3).
- `tune/` holds seed 0 at 30 tasks. `parent_check/` holds the seed-0 60-task parent exact-match re-run. `shards/` holds the screen arm and the control. `logs/` holds the per-run logs.
