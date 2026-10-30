# Wave: `rls_head_l0999_preset005_pcap` first screen

New arm (this PR): `rls_head_l0999_preset005_pcap` — the champion body
(shift-adaptive EMA-norm d099 + utility-gated sigma-0 SGD) with a
streaming-RLS readout at forgetting 0.999, with BOTH wave-2 wind-up
salvages composed:

- detector-driven P reset at the calibrated 0.05 shifted fraction
  (`rls_head_l0999_preset005`, screened 0.86484), and
- the P trace cap 1e4 (`rls_head_l0999_pcap`, screened 0.86608).

Each guard was screened only in isolation. They address the same
failure (covariance wind-up along unexcited feature directions under
forgetting) from different sides — the reset discards stale covariance
at detected shifts, the cap bounds its trace continuously — so whether
the reset makes the cap redundant, or the two compose, was unmeasured.
All constants are verbatim the parent arms; the only new
hyperparameters are the two guard settings. Reduction pins
(`TestRLSHeadCombo`, 5 tests): cap off follows the preset parent
bit-for-bit, reset untriggerable follows the pcap parent bit-for-bit,
and both inert follows the plain `rls_head_l0999` head bit-for-bit.

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
reports).

## Results

Mean online accuracy (60 tasks):

| seed | combo | upgd_w_control | diff vs control | preset005 (parent) | diff | pcap (parent) | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.86518 | 0.77748 | +0.08770 | 0.86466 | +0.00052 | 0.86473 | +0.00044 |
| 1 | 0.86696 | 0.77856 | +0.08840 | 0.86328 | +0.00369 | 0.86751 | −0.00055 |
| 2 | 0.86687 | 0.77724 | +0.08963 | 0.86660 | +0.00027 | 0.86598 | +0.00089 |
| mean | 0.86634 (stderr 0.00058) | 0.77776 | **+0.08858** | 0.86484 | **+0.00149** | 0.86608 | **+0.00026** |

- Improves on the `upgd_w_control` paired reference on **every seed**
  (+0.08858 mean, above the runbook's +0.005 confirmation-candidate
  threshold) and on the preset parent on every seed (+0.00149 mean).
- Against the stronger single-guard parent (`pcap`) the composition is
  a mean-level wash: +0.00026 with seed 1 negative (−0.00055). The two
  guards are largely redundant — the cap already bounds the wind-up
  the reset discards — and the composition does not regress the family
  (it is the family's best mean by a hair), but it does not establish
  a uniform gain over the cap alone.
- Recorded `wall_clock_seconds` per shard: 514.6 / 506.0 / 514.9 s
  (three seeds ran concurrently on a 2-core box).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol
under a maintainer-run lane.
