# wave_identmap50_r_idet003_s005625_anneal_sweep
Two preregistered schedule variants of the parent `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`:
- `f2t2000` (gentler: longer tau);
- `f4t1000` (stronger: lower floor).

Both were run on seeds 0-9 at 60 tasks. **Both are negative** (-0.000256 and -0.000077). See `PREREGISTRATION.md` and `RESULTS.md`.

Folders:
- `parent_check/`: the parent exact-match re-run;
- `shards/`: the arms and the control;
- `logs/`: per-run logs.
