# Preregistration — identifier match-time star

**Written 2026-08-29, BEFORE any 60-task run of the candidate arms.**
Development screening diagnostic; permanently nonpromoting. New path.

## Hypothesis

V8 measured that timing dominates accuracy, and V7 that utility is concave
in identification accuracy. The confirmed arm waits 200 samples for its
first match. An earlier, cruder first match (N=100 at ~0.40 accuracy, N=50
at ~0.20 — interpolated from V1's measured 0.197@50 / 0.619@200), refined
at 200 and 2000, may capture part of the 0-200-step window the current arm
concedes. Failure mode, stated in advance: below some accuracy floor an
early wrong remap is WORSE than none (it scrambles already-normalized
statistics); V7 measured p=0 harmless only as a full re-scramble at step 0,
not mid-task.

## Arms

`rls_head_resid_identmap100_r` (100/200/2000) and
`rls_head_resid_identmap50_r` (50/200/2000), paired against the NEW
incumbent `rls_head_resid_identmap200_r`, remeasured in this screen.
Seeds 0-2, 60 tasks x 5000 steps, one runner, one source commit.

## Bars — FROZEN (lane precedent, unchanged)

WIN >= +0.002 paired vs the remeasured identmap200_r with all 3 seeds
improving -> escalate to the 200-task, 20-seed confirmation (same bar).
TIE +0.0005..+0.002 -> inconclusive, not escalated. LOSS < +0.0005 ->
valid rejection, recorded. A failed gate is a valid rejection; no retuning.

No smoke was run before this file; the arms first execute in the screen.
