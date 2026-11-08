# Wave: `rls_head_resid_identmap50_r_idet002` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet002` — the confirmed
gated identmap composition (`rls_head_resid_identmap50_r`, schedule
50/200/2000, utility gate kept) with the identifier's own shift-detector
trigger at 0.02 instead of the 0.05 default (`ident_reset_frac` 0.02;
the head's P reset `rls_reset_frac` stays 0.05, and every other constant
is verbatim the parent).

This is the far-side bracket cell of the gated idet003 trigger screen
(PR #3115, +0.01490 paired at 0.03 on every seed with the per-seed
spread collapsed). The bracket closes the trigger axis: does the
equalization keep improving below 0.03, or break as remaps start firing
on noise? PR #3109's chain walks the ungated step axis, so this cell is
perpendicular to it, not a duplicate.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed),
sequential. Before the wave, a 3-task `upgd_w_control` seed-0
reproduction shard (`repro_control_seed0_3task.json`) matched the pinned
`shards/upgd_w_control_seed0.json` per-task accuracies bit-exactly on
the shared prefix (0.69019997, 0.77059996, 0.78119999), verifying this
environment against the pinned baseline.

The candidate shards are schema v2 while the pinned reference shards are
v1, and the merge tool refuses to mix schemas — so the shards live
outside the v1 `shards/*.json` glob, the wave `summary.json` merges only
the candidate shards (candidate named as its own control), and the
paired differences below are computed directly from the shards'
`per_task_accuracy` vectors. The parent column reuses the retained
200-task confirmation shards' 60-task prefix
(`smprecond_confirm_r1/shards/`), the same reference the idet003 wave
used.

Runner-version note: 3-task parent shards rerun on this commit
(`repro_parent_seed{0,1,2}_3task.json`) are value-identical to the
idet003 wave's repro shards (the same runner code path; both commits are
main plus a registry entry only). Against the retained parent shards the
full 3-task prefix drifts by up to 0.001 per task (seed 1, task 2;
cross-version float reassociation, the documented within-runner version
drift class — task 0 stays within ±0.0002). The paired table below
therefore carries a ≤0.001 version offset, still an order below the
measured effect.

Pre-boundary identity check: with no remap fired, task 0 must be
bit-identical to the same-runner parent (the trigger change does nothing
before the first detected boundary). Seeds 0 and 2 match bitwise
(0.73219997, 0.75180000). Seed 1 diverges in task 0 (0.74219996 vs
0.74619997): the 0.02 trigger's first remap fires within task 0 on that
seed, so the identity check applies to seeds 0 and 2 only. This is the
mechanism working as designed at the more sensitive trigger — remaps now
start earlier than any 0.03/0.05 arm fired them.

## Results

Mean online accuracy (60 tasks):

| seed | idet002 (gated) | upgd_w_control | diff vs control | identmap50_r (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.92442 | 0.77748 | +0.14694 | 0.91602 | +0.00840 |
| 1 | 0.92355 | 0.77856 | +0.14499 | 0.89708 | +0.02647 |
| 2 | 0.92364 | 0.77724 | +0.14640 | 0.92117 | +0.00247 |
| mean | 0.92387 | 0.77776 | **+0.14611** | 0.91142 | **+0.01245** |

- Improves on the gated parent on **every seed** (mean +0.01245, paired
  stderr 0.00722, 3/3 seeds positive).
- **The trigger axis peaks at 0.03**: idet002 (0.92387) sits 0.00246
  below idet003 (0.92633, PR #3115). Lowering the trigger further still
  equalizes — the candidate's per-seed spread stays collapsed (per-seed
  stderr 0.00028 vs the parent's 0.00732) — but the mean gives back part
  of the 0.03 gain, consistent with remaps starting to fire on noise
  (seed 1's task-0 remap is the visible case).
- Read together with PR #3109's ungated twin, the 0.03 trigger value is
  the bracket's center on both gate settings; the axis is now measured
  at three points on the gated side (0.02/0.03/0.05).
- Above the runbook's +0.005 confirmation-candidate threshold vs the
  control, but per the family's calibration warning (negative results
  #21/#23) 60-task paired effects in this class have decayed 3.5–4.7x at
  the 200-task horizon before; this is a screen result, not a confirmed
  gain.
- Recorded `wall_clock_seconds` per shard: 71.98 / 71.45 / 74.35 s
  (sequential).

Unmeasured next cells (nobody's declared axis): the 0.04 trigger
bracket, and the gate + trigger + step-size triple on the gated chain.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
