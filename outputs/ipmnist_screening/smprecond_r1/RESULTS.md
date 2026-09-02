# smprecond_r1 — 60-task screen results (2026-09-01)

Development screening diagnostic; permanently nonpromoting.
Measurement commit `0bf003c81bf10160e6dc552f75823ad302e564da`; 9 shards
(3 arms x seeds 0-2, 60 tasks x 5000 steps, OMP_NUM_THREADS=1, merged with
the standard `merge` command at the same commit). Summary: `summary.json`.

## Null check — PASSED

Remeasured `rls_head_resid_identmap50_r` per-seed online accuracy
[0.916020, 0.897083, 0.921170] reproduces the star2 remeasurement exactly
(|diff| = 0 <= 0.001 per seed). The screen is valid.

## Paired results vs the remeasured incumbent

| arm | mean | per-seed diff | paired mean | verdict (frozen bars) |
|---|---|---|---|---|
| `rls_head_resid_sm3e4_i50r` | 0.916660 | +0.00529 / +0.00652 / +0.00389 | **+0.005236** (se 0.00076) | WIN >= +0.002, all seeds improve -> escalate |
| `rls_head_resid_sm1e3_i50r` | 0.906626 | -0.00517 / -0.00468 / -0.00455 | -0.004799 (se 0.00019) | LOSS — valid rejection |

- The step-size cliff is sharp: 0.0003 wins on all seeds, 0.001 loses on
  all seeds, and 0.01 diverged to chance in calibration. The winning arm's
  plasticity roughly doubles (0.01223 vs 0.00562) with a slightly more
  positive late-window slope (+3.74e-4 vs +3.35e-4) — no degradation
  signature (preregistered failure condition B did not fire at 0.0003; the
  elevated plasticity at 0.001 with negative accuracy suggests it is the
  onset of exactly that noise-amplification failure).
- Preregistered failure condition A (boundary staleness) did not dominate:
  the win is on every seed.

## Decision (preregistered)

`rls_head_resid_sm3e4_i50r` escalates to a 200-task, 20-seed paired
confirmation at the same frozen bar (+0.002, all seeds improve), new path
`../smprecond_confirm_r1/`. Negative result #21 (60t overstates 200t
effects ~3.5x in this family) is acknowledged in advance: +0.005236 / 3.5
= +0.0015 would MISS the 200t bar, so the honest prior is that this
confirmation is a coin flip at best; the confirmation decides, not the
screen.
