# Preregistration: extended constant body step grid at weight_decay 0 (IPMNIST, 60 tasks)

Written 2026-10-31 12:05 GMT, before any run of this experiment. Base commit f3c7d4a2 (the wd0 constant-step retune evidence). Development screening diagnostic; nonpromoting.

## Question
The wd0 constant-step retune (`outputs/ipmnist_screening/wd0_const_step_retune/`, f3c7d4a2) found step 0.01 beats 0.005625 by +0.004211 paired (10/10, CI95 [+0.004066, +0.004357]). 0.01 was the top of that grid, and tune gains rose monotonically with step (-0.003050 / +0.002197 / +0.004013 for 0.004 / 0.0075 / 0.01). Does a still larger constant step help?

## Code facts (pinned at f3c7d4a2; `alberta_framework/benchmarks/ipmnist_screening.py`)
- L5054-5058 and L4885: body decay is `1 - step*weight_decay` on both branches; exactly 1 at weight_decay 0, so step only scales the gradient step.
- L5418-5428: anneal keys and validation (`0 < anneal_floor <= step_size`); L5517-5521: `lr = max(floor, step*rsqrt(1+s/tau))`; with floor == step the step is constant.
- L8797-8815: the 0.004 / 0.0075 / 0.01 cells; the new cells are appended right after (L8816-8834 on this branch).
- **Non-finite values:** the identmap learner path (L4849-5530) has no isfinite/NaN guards, and the runner does not reject non-finite learner state for this factory (the finite-state guards at L7390-7394 and L7453-7457 belong to the Adamo-family learners only). Online accuracy is `argmax(logits) == y` (L5264), which is always a finite 0/1 value; `jax.numpy.argmax` ranks NaN as the maximum (checked: `argmax([0, nan, 1]) == 1`). So a diverged learner would not crash or emit NaN accuracy; it would show up as an accuracy collapse.

## Arms (new cells; only step_size and anneal_floor change, kept equal)
| cell | step |
|---|---|
| ..._s01_anneal_f1t1000_wd0 (incumbent, existing at f3c7d4a2) | 0.01 |
| ..._s0125_anneal_f1t1000_wd0 | 0.0125 |
| ..._s015_anneal_f1t1000_wd0 | 0.015 |
| ..._s02_anneal_f1t1000_wd0 | 0.02 |

All other hyperparameters identical (rls_lambda 1.0, rls_reset_frac 0.05, head_resid 1.0, ident_match_at 50, ident_match2 200, ident_match3 2000, ident_reset_frac 0.03, anneal_tau 1000, weight_decay 0). **Behavioural difference** is established by code reasoning, not a smoke run: lr equals step_size at every step, the four steps are distinct in float32 (tested), and lr multiplies every body update.

## Seeds
Tune: seed 112. Screen: seeds 113-122. All unseen (0 IPMNIST shards for seeds 112-129 across 3223 refs, checked 2026-10-31 12:01 GMT).

## Procedure (launch.sh in this directory)
0. **Cache pre-flight (fail fast):** the gitignored OpenML cache `outputs/upgd_ipmnist/openml_cache` must be present and match `openml_cache.sha256` (5 files, by name and sha256, and no extra files). Otherwise the launcher writes PREFLIGHT_FAIL and starts nothing.
1. **Env check (VOID-stop):** `upgd_w_control` seed 0 must match `3c38b025:outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json`, and the incumbent at seed 102, 60 tasks, must match `f3c7d4a2:outputs/ipmnist_screening/wd0_const_step_retune/screen/rls_head_resid_identmap50_r_idet003_s01_anneal_f1t1000_wd0_seed102.json`, both bit for bit. Any failure or mismatch: VOID, no automatic re-run.
2. **Tune gate (seed 112, 60 tasks):** mean per-task gain of each new step over 0.01. The best must exceed +0.0005, otherwise STOP (0.01 kept). Ties go to the smaller step. 10-task blocks are reported.
3. **Screen (seeds 113-122, 60 tasks):** winner vs 0.01, paired. PASS requires all of: mean gain >= +0.002; >= 7/10 seeds improved; CI95 lower bound > 0 (t = 2.262, df 9). Per-seed gains, 10-task block means, mean, SE and CI are reported.

**Divergence rule (pre-stated):** a candidate shard that is still missing after one retry (a crash), or that holds any non-finite or out-of-[0,1] accuracy, counts as a loss: it is ineligible at the tune gate, and in the screen it enters with gain -1.0 (not improved). Any task with accuracy < 0.5 is printed as a divergence warning. A failed incumbent shard voids the run (the incumbent is verified by the env check).

**Edge rule (pre-stated):** if 0.02 wins the tune, the report flags that the optimum may lie above 0.02. The verdict is not changed by the flag.

At most 4 one-thread workers; a 3 GB MemAvailable floor at job start and an in-run memguard (SIGSTOP the newest worker below 3072 MB, resume at 3584 MB). Each shard log starts with a HEAD line.

## Disclosures
- Post hoc: chosen because the previous grid's winner sat at its top edge.
- Prior: our own #3587 found larger starting steps hurt at weight_decay 0.01 (peak 0.015: -0.008810), but that is confounded with shrinkage. In the previous retune the gain was front-loaded (10-task blocks +0.009320 ... +0.002104), so larger steps may help early and hurt later; the 60-task horizon cannot rule out later harm.
- Single tune seed; the 10 fresh screen seeds are the confirmatory test.
- The base commit carries the unpublished wd0 constant-step retune; base is behind main.
- Nonpromoting development diagnostic at 60 tasks.
- Duplicate check: duplicate_check.txt (no duplicate found).
