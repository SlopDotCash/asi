# PREREGISTRATION: anneal-floor retune on anneal_f2t1000_wd0

Written 31 Oct 2026, 08:10 GMT, before any shard of this experiment was run. Branch `screen/identmap50-r-idet003-s005625-anneal-wd0-floor-retune`, cut from e7ea7062 (our own #3626 head).

## Lineage (post hoc, disclosed)
This arm was chosen after we saw our own #3610 and our own 200-task wd=0 confirmation (#3626). In #3610, dropping the body weight_decay to 0 on anneal_f2t1000 gave +0.013667 paired on seeds 40-49. In #3626 it gave +0.020114 on seeds 50-69, 20/20.

Every earlier anneal tune ran at weight_decay 0.01: our own #3584, #3586 (f4t1000, negative) and #3587 (higher peak, negative). There, the step also set the shrinkage, so those results were confounded. This experiment asks whether the floor choice changes once the decay is gone, including whether the anneal is needed at all.

## Code facts (line numbers verified with `git show <sha>:alberta_framework/benchmarks/ipmnist_screening.py | sed -n`)
- 3c38b025 L5050-5054: `decay = jnp.where(lr == float32(step_size), float32(param_decay), 1.0 - lr * weight_decay)`. This is the step/decay coupling.
- e7ea7062 L5053-5057: the same expression with `dl` (lr, or decay_lr when given) in place of lr.
- e7ea7062 L4885, in `_make_rls_head_learner` (def at L4849): `param_decay = 1.0 - step_size * weight_decay`. With weight_decay 0, both branches of the coupling equal 1.0 exactly, so the step schedule no longer touches the weights' scale.
- e7ea7062 L5419-5428: the anneal keys `anneal_floor` / `anneal_tau` / `step_size` (the peak). Validation requires 0 < floor <= step_size and tau > 0. The comment at L5418 pins `floor == step` as bit-exactly the un-annealed parent.
- e7ea7062 L5517-5521: `lr = max(floor, step_size * rsqrt(1 + s/tau))`.

## Arms (one axis: the floor; all else equal to the wd0 parent)
- Parent/control: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0`, floor 0.0028125, tau 1000, peak 0.005625, wd 0.
- Tune cells:
  - `..._anneal_f1t1000_wd0`: floor 0.005625 = peak, a constant step (no anneal).
  - `..._anneal_f4t1000_wd0`: floor 0.00140625.
  - `..._anneal_f8t1000_wd0`: floor 0.000703125.

## Protocol
IPMNIST, task length 5000. Paired gain = mean over tasks of (cell minus parent) per-task accuracy, per seed.

1. **Env check (VOID-stop).** Rerun two stored configs on this branch at 60 tasks:
   - `upgd_w_control` seed 0, against 3c38b025's `wave_identmap50_r_idet003_s005625_anneal/shards/`;
   - `..._f2t1000_wd0` seed 40, against 396d6d3c's `wd_sweep_screen/screen/`.

   All 60 per-task accuracies must match bit-for-bit. If any mismatch or job failure occurs, the run is VOID and stops, and nothing is analyzed.
2. **Tune (seed 70, 60 tasks).** Run the parent and the 3 cells.
   - The winner is the cell with the largest paired gain, but only if that gain is > +0.0005. Otherwise the current floor (f2) wins: STOP, a negative, nonpromoting result.
   - Ties between cells (equal to 1e-9) go to the cell nearest the current floor in log2(floor/0.0028125), then to the higher floor.
3. **Screen = confirm (seeds 71-80, 60 tasks).** Run the chosen cell vs the parent, paired.
   - PASS needs all of: mean ≥ +0.002, ≥ 7/10 seeds > 0, and a 95% CI lower bound > 0 (t = 2.262, df 9).
   - All 10 seeds are reported and none is dropped. 50-task blocks don't apply at 60 tasks, so per-seed values and the first/second 30-task halves are reported.
4. Seeds 81-89 are reserved for any later 200-task follow-up and are not used here.

## Seed freshness
`git log --all` over 3151 refs (after fetch, 08:04 GMT) found 0 IPMNIST shards with seeds 70-89 for any config. See `duplicate_check.txt`.

## Resources
- At most 4 one-thread workers (OMP/BLAS/XLA single-thread).
- A 3 GB MemAvailable floor, enforced both at job start and during the run: a guard pauses (SIGSTOP) the newest worker below 3 GB and resumes it at ≥ 3.5 GB. Pausing doesn't change results, and every pause/resume is logged.
- One heavy job at a time; launched detached. Every log's first line is `HEAD <sha>`. All times are GMT.

Development screening diagnostic; nonpromoting unless the bars pass.
