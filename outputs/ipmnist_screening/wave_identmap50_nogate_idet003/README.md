# Wave: `rls_head_resid_identmap50_r_nogate_idet003` first screen

New arm (this PR): `rls_head_resid_identmap50_r_nogate_idet003` — the
standing ungated identmap composition
(`rls_head_resid_identmap50_r_nogate`, the identmap50_r schedule
50/200/2000 with `gate_scale` 0) with the identifier's own
shift-detector trigger at 0.03 instead of the 0.05 default
(`ident_reset_frac` 0.03; the head's P reset `rls_reset_frac` stays
0.05, and every other constant is verbatim the parent).

The detector calibration behind the family puts the within-task
shifted fraction at most 0.018 and a typical boundary at 0.061, so a
boundary whose shifted fraction lands in [0.03, 0.05) never triggers
a remap at the 0.05 default — the identifier simply does not fire for
that task. The 0.03 fraction (1.7x margin over the within-task
maximum, the same margin the registry cites for the preset003 head
reset) was only ever probed on the head's P reset, where it lost
(-0.00037 paired on this same ungated parent); the identifier's
trigger itself was never varied on any identifier arm. Reduction
pins (`TestRLSHeadIdentMapNogateIdet003`, 5 tests): trigger restored
(0.05) follows the ungated identmap composition bit-for-bit, gate
restored follows the gated idet003 composition bit-for-bit, and
identifier disabled (`ident_match_at` 0) follows
`rls_head_resid_l1_preset005_nogate` bit-for-bit.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed).
Before the wave, a 3-task `upgd_w_control` seed-0 reproduction shard
matched the pinned `shards/upgd_w_control_seed0.json` per-task
accuracies bit-exactly on the shared prefix (0.69019997, 0.77059996,
0.78119999), verifying this environment against the pinned baseline.

The new shards are schema v2 while the pinned reference shards are v1,
and the merge tool refuses to mix schemas — so the shards live outside
the v1 `shards/*.json` glob, `summary.json` here merges only the
candidate shards (candidate named as its own control), and the paired
differences below are computed directly from the shards'
`per_task_accuracy` vectors (the same per-seed means the merge
reports). The parent column reuses the retained
`wave_identmap50_nogate/` shards from that arm's first screen.

## Results

Mean online accuracy (60 tasks):

| seed | idet003 combo | upgd_w_control | diff vs control | identmap50_r_nogate (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.92753 | 0.77748 | +0.15005 | 0.91678 | +0.01075 |
| 1 | 0.92781 | 0.77856 | +0.14925 | 0.89892 | +0.02889 |
| 2 | 0.92767 | 0.77724 | +0.15043 | 0.92239 | +0.00528 |
| mean | 0.92767 | 0.77776 | **+0.14991** | 0.91270 | **+0.01497** |

- Improves on the `upgd_w_control` paired reference on **every seed**
  (+0.14991 mean, above the runbook's +0.005 confirmation-candidate
  threshold) and on its ungated parent on **every seed** (+0.01497
  mean, paired stderr 0.00713).
- The per-seed spread collapses (candidate per-seed stderr 0.00008
  vs the parent's 0.00721): per-task diffs are concentrated in
  individual tasks (up to +0.069 on a single task, with the first
  task bitwise identical), consistent with boundaries the 0.05
  trigger missed now firing a remap. Seed 1 — the parent's weakest
  seed — has no negative task at all.
- Per the family's calibration warning (negative result #21),
  60-task paired effects in this family have overstated 200-task
  effects (~3.5x) before; this is a screen result, not a confirmed
  gain, and the parent's own 60-task advantage over its gated parent
  was far smaller (+0.00127), so the size here should be read with
  that discount in mind.
- Recorded `wall_clock_seconds` per shard: 234.9 / 239.8 / 244.4 s
  (three seeds ran concurrently on a contended box).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol
under a maintainer-run lane.
