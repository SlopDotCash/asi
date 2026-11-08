# Wave: `rls_head_resid_identmap50_r_nogate_idet003_s0075` first screen

New arm (this PR): `rls_head_resid_identmap50_r_nogate_idet003_s0075` —
the standing ungated identmap composition at the 0.03 identifier
trigger (`rls_head_resid_identmap50_r_nogate_idet003`, PR #3109;
identmap50_r schedule 50/200/2000, `gate_scale` 0,
`ident_reset_frac` 0.03, head P reset `rls_reset_frac` 0.05) with the
residual body's plain-SGD step size lowered to 0.0075 from the
champion's 0.01. Every other constant is verbatim the parent.

The 0.01 step size was calibrated for the utility-gated update,
whose gate attenuates most steps, and was never re-probed for the
ungated body. A preceding unsubmitted screen raised the step to
0.0125 on this same composition and lost tightly on every seed
(0.92156 mean, -0.00611 ± 0.00007 paired vs the idet003 parent), so
step-noise amplification already dominates at 0.01 and the optimum
may lie below the gated-calibrated constant. This wave screens the
symmetric -25% bracket, 0.0075. Reduction pins
(`TestRLSHeadIdentMapNogateIdet003S0075`, 5 tests): step restored
(0.01) follows the idet003 composition bit-for-bit, gate restored
follows the gated s0075 composition bit-for-bit, and identifier
disabled (`ident_match_at` 0) follows
`rls_head_resid_l1_preset005_nogate` at 0.0075 bit-for-bit.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed),
run sequentially. Before the wave, a 3-task `upgd_w_control` seed-0
reproduction shard matched the pinned `shards/upgd_w_control_seed0.json`
per-task accuracies bit-exactly on the shared prefix (0.69019997,
0.77059996, 0.78119999), verifying this environment against the
pinned baseline. The wave was run twice (once on the design branch,
once on this PR branch); the retained shards are the PR-branch run
and are bitwise identical to the first run on every seed.

The new shards are schema v2 while the pinned reference shards are
v1, and the merge tool refuses to mix schemas — so the shards live
outside the v1 `shards/*.json` glob, `summary.json` here merges only
the candidate shards (candidate named as its own control), and the
paired differences below are computed directly from the shards'
`per_task_accuracy` vectors (the same per-seed means the merge
reports). The parent column reuses the retained
`wave_identmap50_nogate_idet003/` shards from that arm's first
screen (PR #3109).

## Results

Mean online accuracy (60 tasks):

| seed | s0075 combo | upgd_w_control | diff vs control | idet003 parent | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.93412 | 0.77748 | +0.15664 | 0.92753 | +0.00659 |
| 1 | 0.93423 | 0.77856 | +0.15567 | 0.92781 | +0.00643 |
| 2 | 0.93399 | 0.77724 | +0.15675 | 0.92767 | +0.00632 |
| mean | 0.93412 | 0.77776 | **+0.15636** | 0.92767 | **+0.00645** |

- Improves on the `upgd_w_control` paired reference on **every
  seed** (mean +0.15636, above the runbook's +0.005
  confirmation-candidate threshold) and on the standing-best
  idet003 parent on **every seed** (mean +0.00645, paired stderr
  0.00008). The per-seed spread stays collapsed (candidate per-seed
  stderr 0.00007), and the gain over the parent is uniformly
  distributed rather than concentrated in single tasks — consistent
  with a pure step-noise reduction on an already-identified frame,
  the mirror image of the uniform -0.00611 loss at 0.0125.
- Per the family's calibration warning (negative result #21),
  60-task paired effects in this family have overstated 200-task
  effects (~3.5x) before; this is a screen result, not a confirmed
  gain.
- Recorded `wall_clock_seconds` per shard: 99.2 / 96.7 / 96.1
  (sequential runs on an uncontended box).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It
claims no promotion; confirmation would require the frozen full
protocol under a maintainer-run lane.
