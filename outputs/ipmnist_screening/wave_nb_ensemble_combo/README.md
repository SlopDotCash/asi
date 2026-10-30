# Wave: `nb_ensemble_nbreset_rls3` first screen

New arm (this PR): `nb_ensemble_nbreset_rls3` — the transient-attack
ensemble (`nb_ensemble_champion`: shift-adaptive-normalized sigma-0 champion
MLP + streaming naive-Bayes tracker, accuracy-weighted probability mixture)
with BOTH single-flag additions composed:

- probe (b) `ens_nb_reset=1`: detector-driven NB anneal-clock resets, and
- probe (c) `ens_use_rls=1`: the third member, linear RLS over normalized
  pixels (lin_rls pipeline verbatim).

Each addition was screened alone over the base ensemble — nbreset 0.86671
(+0.0018), rls3 0.86500 (+0.0001) — but the composition was never registered
or measured. All member constants are verbatim their source arms; the only
new hyperparameters are the two flags. Reduction pin: the combo factory fed
the registered hyperparameters with both flags at 0 follows
`nb_ensemble_champion` bit-for-bit (`TestNBEnsembleCombo`, 4 tests, including
the registry-shape and both-mechanisms-carried checks).

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed).
Before the wave, a 3-task `upgd_w_control` seed-0 reproduction shard
matched the pinned `shards/upgd_w_control_seed0.json` per-task accuracies
bit-exactly on the shared prefix (0.69019997, 0.77059996, 0.78119999),
verifying this environment against the pinned baseline.

The new shards are schema v2 while the pinned reference shards are v1, and
the merge tool refuses to mix schemas — so the shards live outside the v1
`shards/*.json` glob, `summary.json` here merges only the candidate shards
(candidate named as its own control), and the paired differences below are
computed directly from the shards' `per_task_accuracy` vectors (the same
per-seed means the merge reports).

## Results

Mean online accuracy (60 tasks):

| seed | combo | upgd_w_control | diff vs control | nb_ensemble_champion (base) | diff vs base | nb_ensemble_nbreset | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.86633 | 0.77748 | +0.08886 | 0.86456 | +0.00177 | 0.86627 | +0.00006 |
| 1 | 0.86691 | 0.77856 | +0.08835 | 0.86504 | +0.00187 | 0.86679 | +0.00012 |
| 2 | 0.86713 | 0.77724 | +0.08989 | 0.86516 | +0.00197 | 0.86707 | +0.00006 |
| mean | 0.86679 (stderr 0.00024) | 0.77776 | **+0.08903** | 0.86492 | **+0.00187** | 0.86671 | **+0.00008** |

- Improves on the `upgd_w_control` paired reference on **every seed**
  (+0.08903 mean, above the runbook's +0.005 confirmation-candidate
  threshold) and on its own base ensemble on every seed (+0.00187 mean).
- Against the stronger single-flag arm (`nb_ensemble_nbreset`) the RLS
  member's marginal is small but uniformly positive (+0.00008 mean, no
  seed negative) — the composition does not regress either parent.
- Recorded `wall_clock_seconds` per shard: 883.6 / 878.2 / 479.4 s
  (seeds 0–1 ran under 2-way box contention; seed 2 uncontended).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims no
promotion; confirmation would require the frozen full protocol under a
maintainer-run lane.
