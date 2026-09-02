# smprecond_confirm_r1 — 200-task confirmation results (2026-09-02)

Development confirmation; permanently nonpromoting. Measurement commit
`03a574bb` (source bytes identical to the screen's `0bf003c8`; the
intervening commit is outputs-only). 40 shards (2 arms x seeds 0-19,
200 tasks x 5000 steps, OMP_NUM_THREADS=1), merged at the same commit.
Summary: `summary.json`.

## Null check — PASSED

Remeasured `rls_head_resid_identmap50_r` 20-seed mean 0.9165689 vs the
standing 0.91657 (identmap_star_confirm_r1): |diff| = 1.1e-6 <= 0.001.

## Verdict — FAILED CONFIRMATION (valid rejection at the frozen bar)

| arm | 20-seed mean | paired mean vs control | seeds positive | bar |
|---|---|---|---|---|
| `rls_head_resid_sm3e4_i50r` | 0.9176932 | **+0.0011243** (se 0.000109) | 20/20 (min +0.000489, max +0.002063) | MISSES +0.002 |

- The effect is real (10.3x stderr, every seed positive) and too small:
  the preregistered CONFIRMED-WIN bar required paired mean >= +0.002.
- 60t -> 200t decay factor: 0.005236 / 0.0011243 = **4.66x** — steeper
  than negative result #21's 3.5x. The preregistered prediction that a
  within-task-convergence mechanism should decay LESS than an early-life
  transient mechanism is refuted; the gain concentrates in early life
  here too (consistent with the body's features maturing over life, after
  which per-weight step adaptation has little left to speed up).
- No degradation signature: candidate late-window slope -1.25e-4 vs
  control -1.03e-4; plasticity 0.00853 vs 0.00453.

## Disposition (preregistered)

No new standing best — 0.91657 (`rls_head_resid_identmap50_r`) stands.
Both smprecond arms are deregistered; the mechanism implementation and
its bit-exact reduction pins are retained (lane precedent, negative
results #19-#21). Ledger entry #23 records the closure. No retuning: a
smaller step size or a boundary-reset variant is not licensed as a rescue
of this round.
