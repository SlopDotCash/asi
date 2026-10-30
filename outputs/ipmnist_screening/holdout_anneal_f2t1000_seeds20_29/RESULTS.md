# RESULTS: anneal_f2t1000 vs parent s005625 on unseen seeds 20-29 (60 tasks)

Preregistration: commit 5d533018 (made before any run). Infrastructure: commit 48225702.
Bars (all required): paired mean ≥ +0.002, at least 7/10 seeds improved, and 95% CI lower bound > 0 (t, df=9, 2.2622). Single comparison.
Parent `rls_head_resid_identmap50_r_idet003_s005625` was run fresh on these seeds; no stored results were reused.

| seed | parent s005625 | anneal_f2t1000 | paired Δ |
|---|---|---|---|
| 20 | 0.933973 | 0.935923 | +0.001950 |
| 21 | 0.935150 | 0.937567 | +0.002417 |
| 22 | 0.934463 | 0.936617 | +0.002153 |
| 23 | 0.935977 | 0.937777 | +0.001800 |
| 24 | 0.935683 | 0.937623 | +0.001940 |
| 25 | 0.935403 | 0.937373 | +0.001970 |
| 26 | 0.935550 | 0.937573 | +0.002023 |
| 27 | 0.934800 | 0.937100 | +0.002300 |
| 28 | 0.934883 | 0.937093 | +0.002210 |
| 29 | 0.935267 | 0.937593 | +0.002327 |

mean +0.002109; 95% CI [+0.001965, +0.002253]; improved 10/10 -> PASSES
horizon: 1-15 -0.003116; 16-30 +0.001864; 31-45 +0.004301; 46-60 +0.005387

**Verdict: PASSES all three preregistered bars.** The margin over the +0.002 mean bar is small (+0.000109). Seeds 20, 23, 24 and 25 are individually below +0.002, but the bar applies to the mean.
The horizon pattern matches the screen and the 200-task confirmation: the arm is worse early (tasks 1-15), because it starts with a large step, and better from task 16 on, with the gain growing over the horizon.

Environment checks:
- `upgd_w_control` seed 0 (60 tasks) re-run: bit-exact against the stored `wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json`. All 60 per-task values are identical (mean 0.777477).
- `upgd_w_control` seed 20 (reported only): mean 0.777733.
- Package set identical to the confirm-200 environment (FREEZE_SAME).
