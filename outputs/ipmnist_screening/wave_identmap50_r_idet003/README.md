# Wave: `rls_head_resid_identmap50_r_idet003` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003` — the confirmed
gated identmap composition (`rls_head_resid_identmap50_r`, schedule
50/200/2000, utility gate kept) with the identifier's own shift-detector
trigger at 0.03 instead of the 0.05 default (`ident_reset_frac` 0.03;
the head's P reset `rls_reset_frac` stays 0.05, and every other constant
is verbatim the parent).

This is the gate-kept twin of PR #3109's `nogate_idet003` screen. The
identifier's trigger was never varied on any gated identifier arm: the
0.03 fraction was probed on the head's P reset (`rls_reset_frac`, lost
-0.00037 paired on this parent) and on the ungated composition (+0.01497,
PR #3109). Measuring the trigger under the gate isolates the trigger
effect from the gate axis — the family's standard single-axis
dissection — and tells whether a future best-composition arm can keep
the gate (the gate is 2.9x cheaper per shard and never cleared the
escalation bar when removed, negative result #18).

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
(`smprecond_confirm_r1/shards/`), the same reference the mom9/mom99
waves used.

Runner-version note: 3-task parent shards rerun on this commit
(`repro_parent_seed{0,1,2}_3task.json`) drift from the retained parent
shards by up to 0.001 per task (seed 1, task 2; cross-version float
reassociation, the documented within-runner version drift class —
task 0 stays within ±0.0002); task 0 of the
candidate is bitwise identical to the same-runner parent on all three
seeds (0.73219997, 0.74619997, 0.7518), as the mechanism requires —
before the first detected boundary the trigger change does nothing.
The paired table below therefore carries a ≤0.001 version offset,
an order below the measured effect.

## Results

Mean online accuracy (60 tasks):

| seed | idet003 (gated) | upgd_w_control | diff vs control | identmap50_r (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.92617 | 0.77748 | +0.14869 | 0.91602 | +0.01015 |
| 1 | 0.92648 | 0.77856 | +0.14792 | 0.89708 | +0.02940 |
| 2 | 0.92633 | 0.77724 | +0.14909 | 0.92117 | +0.00516 |
| mean | 0.92633 | 0.77776 | **+0.14857** | 0.91142 | **+0.01490** |

- Improves on the gated parent on **every seed** (mean +0.01490, paired
  stderr 0.00739; the stderr is large only because seed 1's parent was
  the weak seed — the trigger gains +0.0294 exactly there).
- The candidate's per-seed spread collapses (per-seed stderr 0.00009 vs
  the parent's 0.00732): the same equalization signature as the ungated
  idet003 screen, consistent with boundaries the 0.05 trigger missed now
  firing a remap.
- The trigger effect is **gate-independent at this resolution**: +0.01490
  under the gate vs +0.01497 ungated (PR #3109), and gated idet003
  (0.92633) sits 0.0013 below the ungated nogate_idet003 (0.92767) —
  the gate is again approximately neutral, its known cost profile.
- Above the runbook's +0.005 confirmation-candidate threshold vs the
  control, but per the family's calibration warning (negative results
  #21/#23) 60-task paired effects in this class have decayed 3.5–4.7x at
  the 200-task horizon before; this is a screen result, not a confirmed
  gain.
- Recorded `wall_clock_seconds` per shard: 69.29 / 68.15 / 67.88 s
  (sequential).

Unmeasured next cells (nobody's declared axis): gated idet003 under the
step bracket (the gate + trigger + step triple), and trigger brackets
0.02/0.04.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
