# Preregistration — second-moment body preconditioning under identmap, round 1

**Written 2026-09-01 local, BEFORE any diagnostic or 60-task run.**
Development screening diagnostic; permanently nonpromoting. New path.

## Hypothesis and mechanism (named before running)

The remaining 0.016 to the 0.933 family asymptote is within-task convergence
(CEILING_ANALYSIS.md: the 0.904 -> 0.933 gap is optimization speed; the
mechanism named there is "per-weight preconditioning that stays continually
stable"). Every prior adaptive-optimizer negative (adamw_cbp* family at
0.801, guarded_cbp_adam, the seven hybrid negatives) predates the identmap
mechanism: those arms faced a full input permutation every 5,000 steps, which
invalidates per-weight curvature statistics and collapses transfer. The
confirmed identifier (`rls_head_resid_identmap50_r`, 0.9166 at 200t) remaps
inputs into a fixed reference frame within ~50-2000 post-shift samples, so
first-layer weights — and therefore per-weight second moments — stay aligned
across task boundaries. Waves #19-#21 preconditioned the *residual signal
direction* at fixed step geometry; no wave has changed the body's per-weight
step geometry itself, and none ran under identmap.

**One variable:** the residual body update gains RMSProp-class per-weight
second-moment preconditioning. `v = b2*v + (1-b2)*g^2` per body tensor,
bias-corrected by the gate clock, update
`param*decay - sm_step * g/(sqrt(v_hat)+eps) * (1 - sigmoid(gate))`.
Utility gate, EMA input norm, RLS head, detector P reset, and the identmap
schedule (50/200/2000) are all byte-identical to the incumbent. The inert
endpoint `body_sm_decay = 0` reduces bit-exactly to the incumbent (pinned by
a suite test, failing-test-first).

Frozen non-tuned constants: `body_sm_decay = 0.999`, `body_sm_eps = 1e-8`.
No boundary reset of `v` (the stable-frame hypothesis IS the claim; a reset
variant is a different arm and is not run in this round).

## Step-size calibration (tuning-only, negative result #1 discipline)

The preconditioned geometry cannot reuse the champion's `step_size = 0.01`
(negative result #1: rates do not transfer across update geometries).
Calibration: 3-task diagnostic, seed 0 ONLY, grid
`sm_step in {0.0003, 0.001, 0.003, 0.01}`. Purpose is gross viability
(divergence/NaN/dead-arm exclusion), not ranking (negative result #20:
3-task rank inverts by 60 tasks). The two largest surviving step sizes
adjacent to the diagnostic best enter the screen. Calibration outputs are
recorded under `calibration/` in this directory.

## Screen

Arms: at most two `sm_step` values of
`rls_head_resid_sm{tag}_i50r`, paired vs the remeasured
`rls_head_resid_identmap50_r` control. Seeds 0-2, 60 tasks x 5000 steps,
one measurement commit, OMP_NUM_THREADS=1, worker/jobs/xargs pattern.

## Bars — FROZEN (lane precedent, unchanged)

- WIN >= +0.002 paired mean vs the remeasured incumbent with all 3 seeds
  improving -> escalate the single best arm to a 200-task, 20-seed
  confirmation at the same bar (+0.002, all seeds). Negative result #21
  (60t overstates 200t ~3.5x) is acknowledged: the confirmation is the
  arbiter regardless of the 60t margin.
- TIE +0.0005..+0.002: recorded, no escalation, no retuning.
- LOSS < +0.0005: valid rejection, recorded in the ledger.

## Null check (honored if it fires)

The remeasured incumbent's per-seed 60t online accuracy must match the
star2 remeasurement ([0.916020, 0.897083, 0.921170], mean 0.9114244) within
|diff| <= 0.001 per seed. If it does not, the screen is VOID (runner or
source drift, not a mechanism result) and the failure is investigated
before any candidate number is read.

## Predictions and failure conditions

1. If the stable-frame hypothesis is right, faster within-task convergence
   shows up as a paired gain concentrated in mid-task steps (500-4000),
   predicted +0.002..+0.010 at 60t.
2. Failure condition A — boundary staleness: in the 50-2000 sample window
   the remap is partially wrong, so `v` briefly misconditions exactly when
   gradients are largest; if this dominates, the arm loses on early-task
   windows and the net effect is <= 0.
3. Failure condition B — gate interaction: quiet weights have tiny `v`, so
   preconditioning amplifies noise steps on low-utility weights faster than
   the sigmoid gate suppresses them; shows up as reduced late-window slope
   or plasticity collapse.
4. If both step sizes lose, the recorded conclusion is that per-weight
   second-moment preconditioning does not pay even in the identmap frame,
   closing the "conditioning statistics were invalidated by permutation"
   explanation of the historical Adam negatives at this composition point.

If the screen loses, the loss is reported with numbers; nothing is retuned.

No smoke was run before this file.
