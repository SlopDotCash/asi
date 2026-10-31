# Preregistration: decay-only on fresh seeds vs a plain lower weight_decay (development screen, nonpromoting)

This was written before any run on this branch, which is cut from 3c38b025. It cherry-picks:
- the partial-reset code and tests: 3cdadff8 (from c0efe4c8) and 4b6bbf15 (from 054d1504);
- the wd-decoupled code and tests: a5c9ea2d (from 080d4cbf) and 975f9385 (from e4de69e3).

## Why (post hoc, disclosed)
In the wd-decoupled control (#3593), the decay-only arm was descriptive. On seeds 4-13 it gained +0.001656 vs the parent, with CI [+0.001565, +0.001748] and 10/10 seeds improved.
- Decay-only means the parent step with the pr50 decay clock (f_d = 0.5).
- That arm was post hoc, and seeds 4-13 had been seen many times (#3589, #3593).

This screen asks two questions:
1. Does decay-only replicate on fresh, never-run seeds?
2. Is the gain specifically from *shift-timed* decay relief, or from plain lower decay?

**The coupling** (ipmnist_screening.py, `_gated_sgd`, lines 5049-5053 at 3c38b025):
`decay = jnp.where(lr == step_size, param_decay, 1 - lr*weight_decay)`, with weight_decay = 0.01.

## Weight-decay history (searched before this prereg; the plan file has the full output)
- **The parent's 0.01** is the UPGD-W published default. `_upgd_hp` and the RLS-head defaults `_rls_head_hp` both set `"weight_decay": 0.01`.
- **No rls_head / identmap / anneal arm** has ever overridden it. There is no wd sweep in this lane.
- **Ancestors only:**
  - `upgd_w_wd0005`: 0.78431 vs the UPGD-W reproduction at 0.779147 (10 seeds, 200 tasks).
  - `upgd_w_wd002`: screen only.
  - `upgd_ema_norm_wd0005`: 0.84745 vs `upgd_ema_norm` at 0.85362, so lower decay hurt under the EMA normalizer (3 seeds).
  - `upgd_w_wclip_*_wd0`: weight clipping replaces decay.
- No PR or issue tests a lower constant decay on this parent.

## Fresh seeds
Seeds **30-39**. Across every local branch, all 3042 fetched PR head refs, and all worktrees, this parent family (`rls_head_resid_identmap50_r_idet003*`) has shards only for seeds 0-29 (185 files). No shard exists for seed ≥ 30.

## Arms (60 tasks × 5000 steps, seeds 30-39, no tuning on these seeds)
- **Parent:** `…_anneal_f2t1000`. It has **never been run on seeds 30-39, so it is run here** and paired per seed.
- **A:** `…_pr50_decayonly`, f_s = 0 and f_d = 0.5, exactly as in wd-decoupled.
- **B (primary control):** `…_wd00084`, a plain constant weight_decay of **0.0084** on the parent step.
  - The value is chosen analytically from existing code, with no runs: it matches A's mean effective per-step decay rate over the 60-task stream.
  - Calculation: for one detected shift per task, A's decay clock settles near the floor. Its mean decay-lr is 0.0028220, against the parent's 0.0033747 (ratio 0.8362). So 0.01 × 0.8362 = 0.00836, rounded to 0.0084.
  - The result does not depend on detection delay over 0-1000 steps.
  - A and B have the same mean decay; they differ only in timing. B keeps the parent's post-shift decay spike and A removes it.
- **B2 (descriptive):** `…_wd0005`, weight_decay 0.005, the ancestor value. It is a "much lower decay overall" reference and does not enter the rule.

## Bars and decision rule (preregistered)
Comparisons are paired per seed: mean over tasks, then mean, SE and 95% t CI over seeds (df=9, t=2.262).

**A vs parent.** A PASSES only if all three hold:
- the mean is ≥ +0.002;
- at least 7/10 seeds improve;
- the CI lower bound is > 0.

Otherwise A FAILS. A PASS is a screen pass only; it stays nonpromoting until a separate 200-task confirmation.

**Timing read**, with D = A − B (paired) and G = A's gain vs the parent:
- **TIMING-SPECIFIC:** G > 0, the CI lower bound of D is > 0, and D ≥ 0.5·G. The gain is shift-timed decay relief.
- **AMOUNT-ONLY:** G ≤ 0, or the CI lower bound of D is ≤ 0, or D < 0.25·G. Plain lower decay explains it, so the next step would be a weight_decay sweep, not the clock mechanism.
- Otherwise **MIXED**.

Report:
- every seed;
- B vs parent, B2 vs parent and A − B2;
- the count of seeds improved;
- gains by 15-task block.

## Env check (first; VOID-stop on any mismatch, no automatic re-run)
- `upgd_w_control` seed 0 and the parent seed 0, against the 3c38b025 shards.
- The stored decay-only seed 4, against the wd-decoupled branch e9f6125c.
- All must match bit for bit.

## Prior
- P(A passes the bars on fresh seeds) ≈ 25%. The +0.002 bar sits above the +0.00166 seen on seeds 4-13.
- P(A gain > 0 with CI lower bound > 0) ≈ 75%.
- P(TIMING-SPECIFIC) ≈ 35%, P(MIXED) ≈ 25%, P(AMOUNT-ONLY) ≈ 40%.

## Runtime
About 35 min with 4 one-thread workers and a 3 GB RAM floor:
- the env check takes about 9 min, limited by the control run;
- then 40 runs (4 arms × 10 seeds) at about 2.4 min each take about 24 min.

## Disclosures
- **Post hoc:** this follows the wd-decoupled result (#3593), which followed partial reset (#3589).
- The step/decay coupling is quoted above.
- Arm B's value was computed analytically from the schedule, not from data, and assumes one detected shift per task.
- The ancestor wd0005 evidence is mixed: it helped raw UPGD-W and hurt under the EMA normalizer.
- A non-negative weight_decay guard was added to the identmap learner, for the invalid-value pin. It does not change any valid run.
- Development screen only; nonpromoting until confirmed.
