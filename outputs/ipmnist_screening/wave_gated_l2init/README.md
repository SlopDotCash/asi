# Wave: `sigma0_ndecay099_gated_l2init` screening (development only)

This wave screens a registered arm that had a registry entry and
reduction-pin tests (`TestGatedL2Init`: `l2init_pull_scale=0` collapses
bit-exactly to `sigma0_ndecay099`) but no screening shards:
`sigma0_ndecay099_gated_l2init` — the `sigma0_ndecay099` champion plus an
additive, utility-gated pull toward the initial weights
(`l2init_pull_scale=0.01`).

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

| seed | gated_l2init | upgd_w_control (paired) | diff vs control | sigma0_ndecay099 (paired) | diff vs base |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8654 | 0.7775 | +0.0879 | 0.8615 | +0.0039 |
| 1 | 0.8651 | 0.7786 | +0.0865 | 0.8615 | +0.0036 |
| 2 | 0.8653 | 0.7772 | +0.0881 | 0.8618 | +0.0035 |
| mean | 0.86528 (stderr 0.00008) | 0.77778 | **+0.0875** | 0.86158 | **+0.0037** |

The arm improves on the `upgd_w_control` paired reference on every
seed (mean +0.0875, above the runbook's +0.005 confirmation-candidate
threshold) and on its own `sigma0_ndecay099` base on every seed
(mean +0.0037). Total wall clock for the three shards: 663.75 s.

## Evidence class

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion: any confirmation would need the frozen full protocol
(200 tasks, held-out seeds) under a maintainer-run lane.
