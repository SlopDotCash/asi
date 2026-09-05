# Confirmation — FAILED. The standing best does not move.

200 tasks x 5000 steps, seeds 0-19, 40 shards, one runner, source commit
`12dbc136`. Paired baseline: `rls_head_resid_l1_preset005`, remeasured here.
Bars frozen in `../precond_r2/PREREGISTRATION.md`.
Development screening diagnostic; permanently nonpromoting.

## Verdict

`rls_head_resid_l1_preset005_tp_nogate` measured **+0.000791 +/- 0.000094**
paired against the remeasured incumbent, with **19 of 20** seeds improving
(seed 16: -0.000122).

The frozen confirmation bar was **>= +0.002 with all seeds improving**. Both
conditions failed. **No new standing best. The standing best remains
0.87114** (`summary_rls_head_confirm.json`).

| arm | avg online acc (n=20, 200 tasks) | paired vs incumbent | seeds+ |
|---|---|---|---|
| `..._tp_nogate` | 0.8718691 +/- 0.0001145 | +0.000791 +/- 0.000094 | 19/20 |
| `rls_head_resid_l1_preset005` | 0.8710777 +/- 0.0001037 | — | — |

## The finding: this mechanism class DECAYS WITH HORIZON, ~3.5x

| horizon | n | paired effect |
|---|---|---|
| 60 tasks | 10 | +0.002737 +/- 0.000178 |
| 200 tasks | 20 | **+0.000791 +/- 0.000094** |

A **3.46x decay**. The 60-task screen did not merely overstate the
significance of this arm, it overstated the *effect size* by a factor of
three and a half. Both measurements are internally sound — 10/10 and 19/20
seeds positive, tight standard errors — they simply measure different
things, because the gain concentrates in early life and is then diluted
across a horizon three times longer.

This extends negative result #9 ("do not use two-task rank as a 60-task
selector") one rung up: **for this mechanism class, 60-task paired effects
do not transfer to the 200-task horizon either.** The screening proxy is a
filter, not an estimator.

Direct implication for a result already on record: issue #1937's gate
ablation measured +0.001712 at 60 tasks in `gate_ablation_r2/` and was
correctly held back from confirmation under the ambiguous-band rule. Since
gate removal is one of the two components combined here, this result is
evidence that its own 200-task value would also be well under its 60-task
figure. The ambiguous-band rule earned its keep.

## The instrument is trustworthy

The remeasured incumbent reproduces the standing number almost exactly,
across different hardware:

- measured here: 0.8710777 +/- 0.0001037
- standing (`summary_rls_head_confirm.json`): 0.8711435 +/- 0.0001025
- difference: **-0.0000658**, well inside one standard error

So the failure is a property of the arm, not of the runner. (Negative
result #8 still applies to bitwise prefix claims; this is an agreement of
20-seed means, not a bitwise claim.)

## Not a collapse

Both arms remain healthy at the long horizon: late-window slopes are
positive (`tp_nogate` 8.03e-05, incumbent 1.29e-04), and plasticity is
slightly *higher* for the candidate (0.00745 vs 0.00448). The arm is not
frozen and not degrading — it is simply worth much less than the screen
suggested.

## What stands

- Standing best: **0.87114**, unchanged.
- The 60-task composition result (+0.002737, 10/10 seeds) stands as a
  correct measurement *at that horizon*, and is now also the clearest
  example in this lane of a 60-task effect that does not survive to 200.
- Nothing is promoted. No threshold was moved. The confirmation bar was
  frozen before the screen was run and it was not met.
