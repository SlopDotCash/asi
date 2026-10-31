# RESULTS: anneal peak on anneal_f2t1000: negative at the tuning gate

Preregistration: b9deafcd (the deliberate-break clarification is in 062cf365). Run on 31 Oct 2026, launched 00:10 GMT; the tune ended at 00:21:52 GMT. Development screening diagnostic: nonpromoting.

## Environment check: PASS (bit-for-bit, 60 tasks, vs 3c38b025)
- MATCH upgd_w_control seed 0
- MATCH rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000 seed 0

## Tune (seed 0 only; reported outside the screen and the bars)
The parent is the stored #3580 (612491db) seed-0 run, `per_task_accuracy[:60]`, with a mean of 0.936323. This prefix is bit-identical to the env-check re-run.

| peak | arm mean | paired gain | tasks 1-15 | 16-30 | 31-45 | 46-60 | tasks improved |
|---|---|---|---|---|---|---|---|
| 0.0075 (p75) | 0.935000 | −0.001323 | +0.001107 | −0.001253 | −0.002227 | −0.002920 | 10/60 |
| 0.01 (p10) | 0.932917 | −0.003407 | +0.002627 | −0.003093 | −0.006053 | −0.007107 | 8/60 |
| 0.015 (p15) | 0.927513 | −0.008810 | +0.002933 | −0.008133 | −0.013920 | −0.016120 | 4/60 |

## Verdict: STOP at the preregistered tune gate (no peak beats the parent on seed 0). Negative.
- The screen (seeds 1-3) and the confirmation (seeds 4-13) were never run, so the confirmation bars (mean ≥ +0.002, at least 7/10 improved, CI lower bound > 0 with df = 9) were **not evaluated**. No seed 4-13 result exists.
- The pattern is monotone in the peak. A larger starting step helps only in the first 15 tasks, then hurts more and more in each later block. The higher step pays early but costs more as tasks accumulate. This matches the gated-chain step axis, which improved downward to 0.0031640625 (#3121, #3122).
- Single-seed tuning data. This is not a multi-seed estimate.

## Disclosures
- The arm was chosen after the f4t1000 negative (#3586).
- The parent results were reused from the stored #3580 200-task results (prefix [:60]; SHA256SUMS in parent_reused/). Seeds 4-13 were copied but never used.
- Prior: about 10% PASS.
