# Preregistration: weight_decay = 0 vs anneal_f2t1000 over the full 200-task horizon (unseen seeds 50-69)

Committed before any run in this directory (UK time, GMT, 2026-10-31). Branch `confirm/identmap50-r-idet003-s005625-anneal-wd0-200task`, cut from 396d6d3c (our own #3610 head), whose base is 3c38b025. `main` is at ee9df14c; this branch is not rebased. There are no source changes: both configs are already registered at 396d6d3c.

## Question
Does the 60-task gain from dropping the body weight_decay to 0 hold over a full 200-task life, on seeds never run for this family?
- Arm: `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0`, frozen in our own #3610 by its preregistered tune and tie rule. It uses the same anneal step schedule as the parent with weight_decay 0.
- Parent (off): `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`, step max(0.0028125, 0.005625·(1+s/1000)^-½) with weight_decay 0.01 (the untuned UPGD-W default).

## Disclosure of prior information (post hoc lineage)
- This test follows a chain of our own post hoc screens on the same lane: #3589 (partial reset), #3593 (the gain is decay-driven), #3596 (it is amount-only; 0.0084 and 0.005 help) and #3610 (sweep; wd 0 frozen).
- #3610 measured +0.013667 paired at 60 tasks on seeds 40-49, with CI95 [+0.013334, +0.014000] and 10/10 improved. Its gain by 15-task block was +0.010457 / +0.013001 / +0.014797 / +0.016412. In its tie-break, wd 0.001 and wd 0 were within 0.000008 of each other.
- Seeds 50-69 are unseen. No `identmap50_r_idet003*_seed5x/6x` shard exists on any of 3149 refs checked at 05:40 GMT. Tasks 61-200 have never been run for the wd 0 arm on any seed.
- **Step/decay coupling.** ipmnist_screening.py lines 5049-5053 read `decay = where(lr == step_size, param_decay, 1 - lr*weight_decay)`. With weight_decay = 0, the decay is exactly 1 on every step where lr ≠ step_size (the annealed steps), so this arm applies no weight decay off-step. It removes decay rather than retuning a rate. Any plasticity loss from unchecked weight growth would show up only over a long horizon, which is why this 200-task test matters.
- **Prior estimate:** about 85% that all three bars pass. The 60-task gain is about 7× the bar and still rising by block. The main risk is late-life weight growth eroding it.

## Design
- Seeds 50-69, at 200 tasks × 5000 steps. Both the arm and the parent are run fresh, because no stored 200-task parent exists for these seeds.
- Each worker is single-threaded, with at most 4 in parallel and at least 3 GB of RAM available.
- Primary statistic, per seed: Δ = mean over 200 tasks of per_task_accuracy(arm) − the same for the parent, paired by seed.
- **Runtime estimate:** about 95 minutes. Our own #3586 took about 500 s per 200-task shard at 4 workers. Here there are 40 shards in 10 waves (about 85 minutes), plus about 4 minutes of 60-task env checks.

## Bars (ALL required; this is the only comparison)
1. Mean paired Δ ≥ **+0.002**.
2. At least **14/20** seeds improved (Δ > 0).
3. 95% CI lower bound **> 0**: mean − t·SE, with t(0.975, df=19) = 2.093.

If any bar fails, the result FAILS and is recorded as a negative result.

## Environment check (runs first; ALL must match bit for bit, or the run is VOID)
1. `upgd_w_control` seed 0, 60 tasks, vs `3c38b025:outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json`.
2. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000` seed 40, 60 tasks, vs `396d6d3c:outputs/ipmnist_screening/wd_sweep_screen/screen/rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed40.json` (our own #3610 evidence).
3. `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0` seed 40, 60 tasks, vs `396d6d3c:outputs/ipmnist_screening/wd_sweep_screen/screen/rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0_seed40.json`.

"Bit for bit" means the per_task_accuracy lists are exactly equal. On any mismatch or failed job, the launcher stops before any 200-task job, the run is VOID, and it is reported as such. There is no automatic re-run.

## Reporting
- A per-seed table, mean Δ, the 95% CI, the improved count and the verdict.
- Gains by 50-task block (reported only, not bars).
- The three env-check outcomes.

## Stop rules
- Do not start without the heavy slot.
- Stop (VOID) on any env-check mismatch.
- If a 200-task job crashes, re-run that seed once with the same command. A second failure makes the run VOID.
- No early peeking. The verdict is computed once, after all 40 shards exist.
- No extra seeds, horizons or configs afterwards. Stored results are never edited.
- **Nonpromoting.** This is a development confirmation; it does not by itself promote the arm.
