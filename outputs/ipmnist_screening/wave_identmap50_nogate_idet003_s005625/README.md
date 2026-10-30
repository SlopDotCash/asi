# Wave: `rls_head_resid_identmap50_r_nogate_idet003_s005625` first screen

New arm (this PR): `rls_head_resid_identmap50_r_nogate_idet003_s005625` —
the standing ungated identmap composition at the 0.03 identifier
trigger (`rls_head_resid_identmap50_r_nogate_idet003`, PR #3109;
identmap50_r schedule 50/200/2000, `gate_scale` 0,
`ident_reset_frac` 0.03, head P reset `rls_reset_frac` 0.05) with the
residual body's plain-SGD step size lowered to 0.005625 from the
standing best's 0.0075 (`rls_head_resid_identmap50_r_nogate_idet003_s0075`,
PR #3110). Every other constant is verbatim the parent.

Lowering the step from the champion's gated-calibrated 0.01 to
0.0075 gained tightly on every seed (+0.00645 ± 0.00008 paired vs
the idet003 parent), the mirror image of the uniform −0.00611 loss
measured at 0.0125 — so at 0.0075 the response had not yet turned:
step-noise reduction was still winning over within-task
convergence speed. This wave screens the next −25% bracket,
0.005625, which was never screened on any ungated composition.
Reduction pins (`TestRLSHeadIdentMapNogateIdet003S005625`, 5 tests):
step restored (0.0075) follows the s0075 composition bit-for-bit,
gate restored follows the gated s005625 composition bit-for-bit,
and identifier disabled (`ident_match_at` 0) follows
`rls_head_resid_l1_preset005_nogate` at 0.005625 bit-for-bit.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed),
run sequentially. Before the wave, a 3-task `upgd_w_control` seed-0
reproduction shard matched the pinned `shards/upgd_w_control_seed0.json`
per-task accuracies bit-exactly on the shared prefix (0.69019997,
0.77059996, 0.78119999), verifying this environment against the
pinned baseline. The wave was run once, on this PR branch (based on
main directly); the retained shards bind this branch's arm commit.

The new shards are schema v2 while the pinned reference shards are
v1, and the merge tool refuses to mix schemas — so the shards live
outside the v1 `shards/*.json` glob, `summary.json` here merges only
the candidate shards (candidate named as its own control), and the
paired differences below are computed directly from the shards'
`per_task_accuracy` vectors (the same per-seed means the merge
reports). The parent column reuses the retained
`wave_identmap50_nogate_idet003_s0075/` shards from PR #3110's
screen.

## Results

Mean online accuracy (60 tasks):

| seed | s005625 combo | upgd_w_control | diff vs control | s0075 parent | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.93927 | 0.77748 | +0.16180 | 0.93412 | +0.00515 |
| 1 | 0.93908 | 0.77856 | +0.16052 | 0.93423 | +0.00485 |
| 2 | 0.93928 | 0.77724 | +0.16204 | 0.93399 | +0.00529 |
| mean | 0.93921 | 0.77776 | **+0.16145** | 0.93411 | **+0.00510** |

- Improves on the `upgd_w_control` paired reference on **every
  seed** (mean +0.16145, above the runbook's +0.005
  confirmation-candidate threshold) and on the standing-best s0075
  parent on **every seed** (mean +0.00510, paired stderr 0.00013).
  The per-seed spread stays collapsed (candidate per-seed stderr
  0.00007). The step response is still monotone at this bracket —
  the gain over the parent (+0.00510) is somewhat smaller than the
  0.01→0.0075 step's gain (+0.00645), consistent with approaching
  the step-size optimum from above.
- Per the family's calibration warning (negative result #21),
  60-task paired effects in this family have overstated 200-task
  effects (~3.5x) before; this is a screen result, not a confirmed
  gain.
- Recorded `wall_clock_seconds` per shard: 94.5 / 93.5 / 95.5
  (sequential runs).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It
claims no promotion; confirmation would require the frozen full
protocol under a maintainer-run lane.
