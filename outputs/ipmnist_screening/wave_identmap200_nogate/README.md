# Wave: `rls_head_resid_identmap200_r_nogate` first screen

New arm (this PR): `rls_head_resid_identmap200_r_nogate` — the confirmed
`rls_head_resid_identmap200_r` arm (residual RLS head behind the online
permutation identifier, matches at 200/500/2000 post-shift samples,
200-task confirmed at 0.9091) with the residual body's utility gate
removed (`gate_scale` 0; plain decayed SGD on the body).

The gate ablation on the non-identmap incumbent
(`rls_head_resid_l1_preset005_nogate`, negative result #18) measured
gate removal at +0.001712 +/- 0.000174 (n=10, all seeds positive),
below its frozen +0.002 escalation bar, at ~2.9x lower body cost. The
identmap50_r composition measured the transfer on the refining
schedule (+0.00127 at 60 tasks, every seed positive); whether the
effect also transfers under this arm's later first match (200 samples,
so the body trains longer on unremapped inputs after each shift) was
unmeasured, and no arm combined the two. Every other constant is
verbatim the parent arm. Reduction pins
(`TestRLSHeadIdentMap200Nogate`, 4 tests): gate restored (1.0) follows
`rls_head_resid_identmap200_r` bit-for-bit, and identifier disabled
(`ident_match_at` 0) follows `rls_head_resid_l1_preset005_nogate`
bit-for-bit.

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
reports). The parent arm's per-seed values are its retained 60-task
screen shards in `identmap_r1/shards/`.

## Results

Mean online accuracy (60 tasks):

| seed | nogate combo | upgd_w_control | diff vs control | identmap200_r (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.90861 | 0.77748 | +0.13113 | 0.90831 | +0.00030 |
| 1 | 0.89418 | 0.77856 | +0.11562 | 0.89286 | +0.00132 |
| 2 | 0.91367 | 0.77724 | +0.13643 | 0.91295 | +0.00073 |
| mean | 0.90549 | 0.77776 | **+0.12773** | 0.90470 | **+0.00078** |

- Improves on the `upgd_w_control` paired reference on **every seed**
  (+0.12773 mean, above the runbook's +0.005 confirmation-candidate
  threshold) and on its gated parent on **every seed** (+0.00078 mean,
  paired stderr 0.00030) — the gate-removal effect measured on the
  non-identmap incumbent (+0.0017, n=10) transfers to this schedule's
  remapped body as well, at roughly half the 50-schedule size, with no
  seed negative.
- Per the family's calibration warning (negative result #21), 60-task
  paired effects in this family have overstated 200-task effects
  (~3.5x) before; +0.00078 at 60 tasks is a screen result, not a
  confirmed gain.
- Recorded `wall_clock_seconds` per shard: 181.3 / 191.3 / 186.1 s
  (three seeds ran concurrently on a 2-core box).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol
under a maintainer-run lane.
