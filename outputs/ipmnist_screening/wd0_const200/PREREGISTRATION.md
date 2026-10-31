# Preregistration: constant step (f1t1000_wd0) vs anneal_f2t1000_wd0 over the full 200-task life (unseen seeds 81-100)

Committed before any run in this directory (UK time, GMT, 2026-10-31, written 09:12 GMT). Branch `confirm/identmap50-r-idet003-s005625-wd0-const-200task`, cut from c8028d3a (our own wd0 floor re-tune evidence head), whose ancestry is e7ea7062 (our own #3626) → 396d6d3c (our own #3610) → base 3c38b025. `main` is at ea6eee9d; this branch is behind main and not rebased. There are no source changes: both configs are already registered at c8028d3a.

## Question
Does the 60-task gain from removing the within-task step anneal at weight_decay 0 hold over a full 200-task life, on seeds never run for any IPMNIST config?
- Arm: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0`. Its floor equals the step size (0.005625), so the step is constant 0.005625 with no anneal, at weight_decay 0. It was registered at c8028d3a, `alberta_framework/benchmarks/ipmnist_screening.py` L8777-8796 (the floor tuple is at L8790-8795).
- Parent (off): `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0`. Its step is max(0.0028125, 0.005625·(1+s/1000)^-½) at weight_decay 0, registered at c8028d3a L8755-8775 (name at L8757-8758, tag "0" at L8774).

## Disclosure of prior information (post hoc lineage)
- **Post hoc.** This test follows our own #3610 (wd 0 frozen at 60 tasks), our own #3626 (wd 0 holds at 200 tasks, +0.020114 paired, seeds 50-69) and our own floor re-tune at c8028d3a. That re-tune froze f1t1000 at a tune gate on seed 70, then screened +0.005648 paired on seeds 71-80 at 60 tasks: SD 0.000291, CI95 [+0.005440, +0.005856], 10/10 improved and 598/600 tasks positive.
- **The decaying gain motivates this run.** In that screen the gain shrank by 10-task block: +0.009862 / +0.006398 / +0.005360 / +0.004716 / +0.004078 / +0.003476. Tasks 61-200 have never been run for f1t1000_wd0. If the trend continues, the late-life gain could reach zero or turn negative.
- **f4/f8 never bind.** In the floor re-tune, the f4 (0.00140625) and f8 (0.000703125) floors never applied: within a 5000-step task the annealed step bottoms out at about 0.0022966, so those two cells were bitwise-identical. Only f1 vs f2 was a real comparison. This was a design oversight there and does not affect this run.
- **Step/decay coupling.** At c8028d3a, `ipmnist_screening.py` L5054-5058 reads `decay = where(dl == step_size, param_decay, 1 - dl*weight_decay)`, and L4885 (in `_make_rls_head_learner`, defined at L4849) sets `param_decay = 1 - step_size*weight_decay`. At weight_decay 0 the decay is exactly 1 on both branches, for both arms. So the only difference between the arms is the step schedule (L5419-5428 validates the anneal keys; L5517-5521 computes `lr = max(floor, step*rsqrt(1+s/tau))`, and L5418 pins floor == step as bit-exact with the un-annealed path).
- Seeds 81-100 are unseen. Across 3218 refs checked at 09:10:25 GMT (`git log --all --name-only`), no file whose path contains "ipmnist" ends in `_seed81.json` … `_seed100.json`. The only seed-100 shards on any ref are in `outputs/upgd_label_emnist/`, a different lane.
- **Prior estimate:** about 55% that all three bars pass. The 60-task gain is about 2.8× the bar and is consistent across seeds, but it is falling roughly geometrically, and a 200-task mean gives the late tasks 70% of the weight.

## Design
- Seeds 81-100, at 200 tasks × 5000 steps, with both the arm and the parent run fresh (40 shards).
- Each worker is single-threaded, with at most 4 in parallel. A 3 GB MemAvailable floor is enforced at job start (wait_slot) and during the run (memguard.sh pauses the newest worker with SIGSTOP; pausing cannot change a deterministic single-process result).
- Primary statistic, per seed: Δ = mean over 200 tasks of per_task_accuracy(arm) − the same for the parent, paired by seed.
- **Runtime estimate:** about 2 h. Our own #3626 ran 40 shards of this family at 200 tasks with 4 workers in 116 minutes (05:47 to 07:43 GMT), plus about 5 minutes of 60-task env checks here.

## Bars (ALL required; this is the only comparison)
1. Mean paired Δ ≥ **+0.002**.
2. At least **14/20** seeds improved (Δ > 0).
3. 95% CI lower bound **> 0**: mean − t·SE, with t(0.975, df=19) = 2.093.

If any bar fails, the result FAILS and is recorded as a negative result.

## Decay by 50-task block (descriptive only; not a bar)
- For tasks 1-50, 51-100, 101-150 and 151-200: the mean over seeds of the per-seed block Δ, with a 95% CI (t = 2.093, df = 19) and the improved count.
- **Pre-stated flag:** if the late-block mean (tasks 151-200) is ≤ 0, the report flags it prominently ("gain gone or reversed late in life"). The flag does not change the verdict, which depends only on the three bars above.

## Environment check (runs first; ALL must match bit for bit, or the run is VOID)
1. `upgd_w_control` seed 0, 60 tasks, vs `3c38b025:outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json`.
2. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0` seed 71, 60 tasks, vs `c8028d3a:outputs/ipmnist_screening/wd0_floor_retune/screen/rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0_seed71.json`.
3. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0` seed 71, 60 tasks, vs `c8028d3a:outputs/ipmnist_screening/wd0_floor_retune/screen/rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0_seed71.json`.

"Bit for bit" means the per_task_accuracy lists are exactly equal. On any mismatch or failed job, the launcher stops before any 200-task job, the run is VOID, and it is reported as such. There is no automatic re-run.

## Reporting
- A per-seed table, mean Δ, SD, SE, the 95% CI, the improved count and the verdict.
- The 50-task block table above, with the late-block flag.
- The three env-check outcomes and every memguard pause.

## Stop rules
- Do not start without the heavy slot.
- Stop (VOID) on any env-check mismatch.
- If a 200-task job crashes, re-run that shard once with the same command. A second failure makes the run VOID.
- No early peeking. The verdict is computed once, after all 40 shards exist.
- No extra seeds, horizons or configs afterwards. Stored results are never edited.
- **Nonpromoting.** This is a development confirmation; it does not by itself promote the arm. The base is behind `main`.
