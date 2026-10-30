# Wave: `sigma0_shiftnorm_d099_gated_l2init` screening (development only)

This wave screens a new arm added with this wave: the gated-L2-init pull
term first registered for `sigma0_ndecay099_gated_l2init` (screened in
`wave_gated_l2init/`), transplanted onto the `sigma0_shiftnorm_d099`
champion body — the shift-triggered re-conditioning normalizer, utility
gate, and decoupled decay are unchanged, and the only new term is
`step_size * l2init_pull_scale * (1 - gate) * (w - w_0)` with
`l2init_pull_scale=0.01`. Reduction pin (`TestShiftnormGatedL2Init`):
`l2init_pull_scale=0` delegates to the champion factory's own step, so
the trajectory collapses bit-exactly to `sigma0_shiftnorm_d099`.

## Protocol

- Standard screening scale: 60 tasks x 5,000 steps, seeds 0-2,
  `OMP_NUM_THREADS=1`, exact per-step noise path (`noise_mode=step`;
  the arm itself is sigma=0). One process per (arm, seed).
- Environment/baseline verification before the wave: a 3-task
  `upgd_w_control` seed-0 reproduction shard matched the pinned
  `shards/upgd_w_control_seed0.json` per-task accuracies **bit-exactly**
  on the shared prefix (0.69019997, 0.77059996, 0.78119999), so the
  existing control shards are used as the paired reference instead of
  re-running ~3 h of duplicate control compute.
- The new shards are schema `alberta.ipmnist_screening.shard.v2` while
  the pinned reference shards are v1; the merge tool refuses to mix
  schemas, so the shards live in this separate directory (the v1
  `shards/*.json` merge glob is untouched) and the paired differences
  below are computed directly from the shards' `per_task_accuracy`
  vectors (the same per-seed means the merge reports).
- `summary.json` here is the merge output for the three new shards
  only (merge requires `OMP_NUM_THREADS=1` in the environment to match
  the runtime recorded by the v2 shards).

## Results (mean online accuracy, 60 tasks)

| seed | gated_l2init | upgd_w_control (paired) | diff vs control | sigma0_shiftnorm_d099 (paired) | diff vs base |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8650 | 0.7775 | +0.0875 | 0.8636 | +0.0015 |
| 1 | 0.8653 | 0.7786 | +0.0867 | 0.8642 | +0.0011 |
| 2 | 0.8656 | 0.7772 | +0.0884 | 0.8642 | +0.0014 |
| mean | 0.86531 (stderr 0.00017) | 0.77776 | **+0.0875** | 0.86396 | **+0.0013** |

The arm improves on the `upgd_w_control` paired reference on every
seed (mean +0.0875, above the runbook's +0.005 confirmation-candidate
threshold) and on its own `sigma0_shiftnorm_d099` champion base on
every seed (mean +0.0013). The gated pull's marginal contribution is
smaller on the shiftnorm body than it was on the `sigma0_ndecay099`
body (+0.0037) but remains uniformly positive. Recorded
`wall_clock_seconds` per shard: 605.2 / 606.8 / 620.6 s (the box was
heavily contended during the wave; the arm's uncontended cost class is
the champion's, ~160 s per shard).

## Evidence class

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion: any confirmation would need the frozen full protocol
(200 tasks, held-out seeds) under a maintainer-run lane.
