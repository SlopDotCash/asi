# Wave: `rls_head_resid_identmap50_r_idet003_s005625` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s005625` — the
gated idet003 composition (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03 — the gated trigger peak, PR #3115) with the residual body's step
size lowered to 0.005625 from the champion-calibrated 0.01. Every other
constant is verbatim the parent.

This is the gate + trigger + step-size triple on the gated chain, named
as an unmeasured next cell in the idet002 wave README (PR #3116). The
two axes composing it were each measured separately: the trigger axis
peaks at 0.03 on the gated chain (PR #3115, +0.01490 paired), and the
step axis improved monotonically from 0.01 through 0.005625 on the
ungated chain (PRs #3109–#3111, +0.01155 cumulative). The gated chain
had never screened any step value below 0.01, and no composition had
measured both winning settings together.

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
`per_task_accuracy` vectors. Three parent references are reported: the
registered gated parent `rls_head_resid_identmap50_r` (retained
200-task confirmation shards' 60-task prefix in `smprecond_confirm_r1/`),
the gated idet003 peak (retained `wave_identmap50_r_idet003/` shards
from PR #3115's screen, fetched from that PR's branch — not committed
here; identified by the per-shard `sha256` values in that wave's
`summary.json` — candidate shards `08ae9801…`, `d475a0c3…`,
`236da7f2…`), and the pinned `upgd_w_control` shards.

Runner-version note: 3-task parent shards rerun on this commit
(`repro_parent_seed{0,1,2}_3task.json`) are **value-identical to the
idet003 wave's repro shards** on every seed (both commits are main plus
a registry entry only — the same runner code path, now verified rather
than assumed). The idet003 pairing column therefore carries no
version-offset caveat; the gated-parent column retains the same
≤0.001/task cross-version offset versus the retained confirmation
shards documented in the idet002 wave (PR #3116), an order below the
measured effects.

Mechanism note: unlike the trigger-only idet002 cell, no pre-boundary
task-0 bit-identity holds here — `step_size` changes every body update
from step 0. The mechanism evidence is the reduction pins in
`TestRLSHeadIdentMapIdet003S005625` (step restored follows the gated
idet003 composition, trigger restored follows the gated identmap50_r
composition at the same step, identifier disabled follows the gated
non-identifier incumbent at the same step — all bit-for-bit).

## Results

Mean online accuracy (60 tasks):

| seed | triple (gated) | upgd_w_control | diff vs control | idet003 (parent) | diff | identmap50_r | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.93446 | 0.77748 | +0.15698 | 0.92617 | +0.00829 | 0.91602 | +0.01844 |
| 1 | 0.93467 | 0.77856 | +0.15611 | 0.92648 | +0.00819 | 0.89708 | +0.03759 |
| 2 | 0.93495 | 0.77724 | +0.15771 | 0.92633 | +0.00861 | 0.92117 | +0.01378 |
| mean | 0.93469 | 0.77776 | **+0.15693** | 0.92633 | **+0.00837** | 0.91142 | **+0.02327** |

- Improves on the gated idet003 parent on **every seed** (mean
  +0.00837, paired stderr 0.00013, 3/3 seeds positive) — a larger
  step-bracket gain than the ungated chain measured at the same step
  (+0.00510 over its s0075 parent, PR #3111). The gated chain takes
  the step-size axis too.
- The candidate's per-seed spread stays collapsed (per-seed stderr
  0.00014 vs the gated parent's 0.00732 — about 52x smaller stderr,
  about 2700x smaller seed variance), continuing the equalization
  signature: the gated chain is now measured at 0.93469 mean.
- This is the gated chain's best measured result (vs identmap50_r
  0.9166 confirmed / 0.91142 60-task prefix, idet003 0.92633, idet002
  0.92387). The ungated s005625 chain remains higher on this screen
  (0.93921, PR #3111) on the ungated side of the gate ablation.
- Above the runbook's +0.005 confirmation-candidate threshold vs the
  control, but per the family's calibration warning (negative results
  #21/#23) 60-task paired effects in this class have decayed 3.5–4.7x
  at the 200-task horizon before; this is a screen result, not a
  confirmed gain.
- Recorded `wall_clock_seconds` per shard: 73.53 / 70.00 / 74.99 s
  (sequential).

Unmeasured next cells (nobody's declared axis): the 0.04 trigger
bracket on the gated chain, and the gated chain's next step bracket
(0.00421875, the ungated chain's monotone direction below 0.005625).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
