# Results: full-length RiverSwim calibration, discounted SARSA with Q0 = 10

Development-only and permanently nonpromoting. The protocol is in `DEVELOPMENT_PLAN.md`, including amendments 1 and 2.

**Verdict: PASS.** Gate statistic: normalized `sarsa_q0`. Its paired t LCB (t = 2.201, n = 12) is 0.8874, against the strict threshold > 0.10. Mean 0.8910, sd 0.0057. No VOID condition was hit.

Normalization scale (mean oracle minus random reward): 17040.2304. Horizon: 20,000 events. Seeds: 47310-47321.

Descriptive, not part of the gate:

- Default `sarsa`: mean 0.2298, LCB -0.0202.
- Paired `sarsa_q0 - sarsa` normalized difference: mean 0.6612, sd 0.3924, LCB 0.4119.

| seed | random | oracle | sarsa | sarsa_q0 | norm sarsa | norm sarsa_q0 | paired diff | high-end visits (sarsa / q0) |
|---|---|---|---|---|---|---|---|---|
| 47310 | 59.27 | 17119 | 94.07 | 15239.50 | 0.0020 | 0.8908 | 0.8888 | 1 / 15473 |
| 47311 | 67.83 | 17080 | 95.53 | 15319.40 | 0.0016 | 0.8950 | 0.8934 | 0 / 15577 |
| 47312 | 50.56 | 17303 | 93.85 | 15313.52 | 0.0025 | 0.8957 | 0.8932 | 6 / 15563 |
| 47313 | 56.22 | 16901 | 15139.62 | 15186.50 | 0.8852 | 0.8879 | 0.0028 | 15324 / 15378 |
| 47314 | 60.07 | 17040 | 94.26 | 15223.39 | 0.0020 | 0.8899 | 0.8878 | 0 / 15474 |
| 47315 | 64.49 | 17056 | 14988.54 | 15219.43 | 0.8758 | 0.8894 | 0.0135 | 15223 / 15481 |
| 47316 | 47.62 | 17144 | 95.94 | 15202.47 | 0.0028 | 0.8894 | 0.8865 | 0 / 15452 |
| 47317 | 58.12 | 17093 | 1856.43 | 15398.49 | 0.1055 | 0.9002 | 0.7947 | 2116 / 15620 |
| 47318 | 81.36 | 16960 | 94.48 | 15106.55 | 0.0008 | 0.8817 | 0.8810 | 0 / 15342 |
| 47319 | 56.25 | 17208 | 0.47 | 15144.74 | -0.0033 | 0.8855 | 0.8887 | 2 / 15370 |
| 47320 | 58.99 | 17029 | 99.90 | 15163.43 | 0.0024 | 0.8864 | 0.8840 | 6 / 15420 |
| 47321 | 55.41 | 17266 | 15047.58 | 15387.46 | 0.8798 | 0.8998 | 0.0199 | 15243 / 15648 |

Every seed is kept. Default SARSA found the high-reward end on 4 of 12 seeds (47313, 47315, 47321 fully, 47317 partly); Q0 = 10 found it on all 12. Where default SARSA succeeded, the paired gain is small (0.003-0.020).

## Environment check (amended)

PASS. The #3308 Q0 = 10 life on seed 81010 matched every field bit-exactly. Scorecard seed-70000 sarsa, random and oracle matched accepted events, parameter changes and reward sums bit-exactly (persistent bytes too, for sarsa and random). The known `oracle_reward_sum` accumulation-order difference and the sarsa manifest/transcript change from #3308 were checked exactly, as Amendment 1 describes.

## Attempts

1. Commit b3fb21bd: VOID at the environment check. The cause is isolated in Amendment 1. No calibration life ran.
2. Commit 33cc00bd: the environment check passed, then VOID from a launcher variable-shadowing bug. All 48 life calls were rejected at argument parsing, so no life ran (Amendment 2).
3. Commit 9038cab0: completed, the result above.

## Limits

- One environment (the scorecard's six-state RiverSwim) and one horizon. Q0 = 10 = 1/(1-gamma) is not tuned, and no other value was tried.
- Q0 = 10 is optimistic for this reward scale. The result is about exploration on RiverSwim, not a general claim.
- The random and oracle normalizers were rerun on the fresh seeds with the scorecard's pinned arms. The scorecard's own retained shards are untouched.
- CPU, Python 3.13.5, jax 0.11.0, numpy 2.5.1, one thread per process. Wall clock is telemetry only: 48 lives, 97-174 s each, 7,009 s in total, 34 min elapsed on 4 workers.
