# Preregistration — smprecond 200-task confirmation, round 1

**Written 2026-09-01 local, BEFORE any 200-task run. After (and conditioned
only on) the smprecond_r1 60-task screen decision.**
Development confirmation; permanently nonpromoting. New path.

## Arms

- Candidate: `rls_head_resid_sm3e4_i50r` (the single 60t screen winner;
  +0.005236 paired, all seeds).
- Control: `rls_head_resid_identmap50_r`, remeasured in this run (same
  runner, same measurement commit).

Seeds 0-19 (the lane's standard 20-seed confirmation set; the screen
consumed 0-2 for selection, 3-19 remain selection-untouched for this
mechanism). 200 tasks x 5000 steps, OMP_NUM_THREADS=1, one measurement
commit, worker/jobs/xargs pattern, standard merge with
`--control-name rls_head_resid_identmap50_r`.

## Bars — FROZEN (lane precedent, unchanged)

- CONFIRMED WIN: paired mean >= +0.002 vs the remeasured control with all
  20 seeds improving -> new standing development best; registry keeps the
  arm; ledger and provenance updated.
- FAILED CONFIRMATION: anything else — a valid rejection (negative result
  #21 pattern); the arm is deregistered, the mechanism outcome recorded in
  the ledger either way. No retuning, no threshold move, no re-run.

## Null check (honored if it fires)

The remeasured control's 20-seed mean must lie within +/-0.001 of the
standing identmap50_r confirmation mean 0.91657
(identmap_star_confirm_r1). If not, the run is VOID and investigated
before any candidate number is read.

## Prior stated in advance

Negative result #21 measured a ~3.5x 60t->200t effect decay for a
*different* mechanism in this family (gains concentrated in early life).
Applied naively, +0.005236 / 3.5 = +0.0015 misses the bar. The smprecond
mechanism targets *within-task* convergence on every task rather than
early-life transients, so the decay factor should be smaller — that is
exactly what this confirmation tests. Predicted outcome if the mechanism
is what the calibration diagnostic suggests (per-task convergence
speedup): paired mean +0.003..+0.005 with, at 200t stderr levels, all
seeds positive. Failure mode: early-life concentration (decay >= 3.5x),
landing +0.001..+0.002 and failing the all-seeds condition.
