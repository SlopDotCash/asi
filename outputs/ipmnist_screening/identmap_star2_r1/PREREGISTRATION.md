# Preregistration — match-time star, round 2

**Written 2026-08-29 local (2026-08-30Z), BEFORE any 60-task run.**
Development screening diagnostic; permanently nonpromoting. New path.

## Arms, paired vs the confirmed `rls_head_resid_identmap50_r`

| arm | schedule | rationale |
|---|---|---|
| `rls_head_resid_identmap25_r` | 25/200/2000 | round-1 star monotone toward earlier; V1-interpolated accuracy ~0.10 at N=25. The accuracy floor did not bite at ~0.20; this probes whether it exists at all. |
| `rls_head_resid_identmap50_fast` | 50/100/500 | the timing-dominates-accuracy principle applied to the SECOND and THIRD matches; trades the late 2000-sample match for an early correction. |

Control: `rls_head_resid_identmap50_r`, remeasured in this screen (same
runner, same source commit). Seeds 0-2, 60 tasks x 5000 steps.

## Bars — FROZEN (lane precedent, unchanged)

WIN >= +0.002 paired vs the remeasured identmap50_r with all 3 seeds
improving -> escalate the single best arm to a 200-task, 20-seed
confirmation at the same bar. TIE +0.0005..+0.002. LOSS < +0.0005 -> valid
rejection, recorded. A failed gate is a valid rejection; no retuning.

## Predictions

1. `identmap50_fast` WINS (~+0.002-0.005): the 100-sample correction
   arrives inside the window where the crude 50-sample map still costs
   accuracy.
2. `identmap25_r` is the coin-flip: monotonicity says win, but ~0.10
   accuracy approaches chance and the floor must exist somewhere.
3. Failure condition for both: if the round-1 monotonicity was driven by
   the 200-sample REFINE (not the early first match), the fast-refine arm
   wins and the 25 arm loses — that dissociation is itself the finding.

No smoke was run before this file.
