# Preregistration: constant body step-size retune at weight_decay 0 (IPMNIST, 60 tasks)

Written 2026-10-31 11:09 GMT, before any run of this experiment. Base commit ab572898 (our own #3632 branch head). Development screening diagnostic; nonpromoting.

## Question
The confirmed constant-step arm `rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0` (step 0.005625, floor == step, weight_decay 0) beat anneal_f2t1000_wd0 by +0.005648 at 60 tasks (our own #3632) and +0.003587 at 200 tasks (our own 200-task confirmation). Its body step 0.005625 was inherited from wave-era tuning in which decay was coupled to the step. Is a different constant step better now that the step is decoupled from shrinkage?

## Code facts (pinned at ab572898; `alberta_framework/benchmarks/ipmnist_screening.py`)
- L5054-5058: per-step body decay is `jnp.where(dl == float32(step_size), param_decay, 1.0 - dl*weight_decay)`; L4885: `param_decay = 1 - step_size*weight_decay`. At weight_decay 0 both branches are exactly 1, so the step size changes only the gradient step.
- L5418-5428: anneal keys and validation (`0 < anneal_floor <= step_size`); L5517-5521: `lr = max(floor, step*rsqrt(1+s/tau))`. With floor == step the schedule is the constant `step_size`.
- L8755-8796: parent and floor-variant registry entries; the new cells are appended right after (L8797-8815 on this branch).

## Arms (new registry entries; only step_size and anneal_floor change, kept equal)
| cell | step |
|---|---|
| ..._s004_anneal_f1t1000_wd0 | 0.004 |
| ..._s005625_anneal_f1t1000_wd0 (incumbent, existing) | 0.005625 |
| ..._s0075_anneal_f1t1000_wd0 | 0.0075 |
| ..._s01_anneal_f1t1000_wd0 | 0.01 |

All other hp identical to the incumbent (rls_lambda 1.0, rls_reset_frac 0.05, head_resid 1.0, ident_match_at 50, ident_match2 200, ident_match3 2000, ident_reset_frac 0.03, anneal_tau 1000, weight_decay 0).

**Behavioural difference** is established by code reasoning, not a smoke run (no compute was used before this commit): lr equals step_size at every step (L5517-5521 with floor == step), the four steps are distinct in float32 (tested), and lr multiplies every body update, so trajectories diverge from the first update. This avoids the non-binding-floor failure of the earlier f4/f8 cells, which were bit-identical to their parent for that reason.

## Seeds
Tune: seed 101. Screen: seeds 102-111. All unseen (0 IPMNIST shards for seeds 101-129 across all refs, checked 2026-10-31 11:06 GMT).

## Procedure (launch.sh in this directory)
1. **Env check (VOID-stop):** `upgd_w_control` seed 0 must match `3c38b025:outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json` bit for bit, and the incumbent at seed 71, 60 tasks, must match `ab572898:outputs/ipmnist_screening/wd0_floor_retune/screen/rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0_seed71.json`. Any failure or mismatch: VOID, stop.
2. **Tune gate (seed 101, 60 tasks):** mean per-task gain of each new step over the incumbent. The best cell must exceed +0.0005, otherwise STOP (current step kept; ties go to the current step). Ties between new cells go to the step closest to 0.005625 in log ratio. 10-task blocks are reported.
3. **Screen (seeds 102-111, 60 tasks):** winner vs incumbent, paired. PASS requires all of: mean gain >= +0.002; >= 7/10 seeds improved; CI95 lower bound > 0 (t = 2.262, df 9). Per-seed gains, 10-task block means, mean, SE and CI are reported for every seed.

At most 4 one-thread workers; a 3 GB MemAvailable floor at job start and an in-run memguard (SIGSTOP the newest worker below 3072 MB, resume at 3584 MB). Each shard log starts with a HEAD line. Shards resume idempotently; failed tune/screen shards are retried once.

## Disclosures
- Post hoc: chosen after seeing our own #3610, #3626, #3632 and the 200-task confirmation, all of which motivate this arm.
- Prior: our own #3587 found that raising the anneal start step hurt (peaks 0.0075/0.01/0.015 at seed 0: -0.001323/-0.003407/-0.008810, helping early and hurting later), but at weight_decay 0.01, where step and shrinkage are coupled. That is confounded, so the direction here is open; the grid brackets the incumbent on both sides. The 200-task gain of the constant step decays over blocks, which also suggests the step may be off-optimum.
- Base ab572898 is behind main (ea6eee9d); the screening module is unchanged between c8028d3a and ab572898.
- A single-seed tune gate is noisy; the screen on 10 fresh seeds is the only confirmatory test.
- Nonpromoting development diagnostic at 60 tasks; it says nothing beyond 60 tasks.
- Duplicate check: duplicate_check.txt (no duplicate found).
