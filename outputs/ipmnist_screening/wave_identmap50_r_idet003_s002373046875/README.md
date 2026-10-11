# Wave: `rls_head_resid_identmap50_r_idet003_s002373046875` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s002373046875` — the
gated idet003_s0031640625 composition (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, residual body step size 0.0031640625 — the gated chain's best measured
screen, PR #3120) with the body step size lowered one more -25% bracket to
0.002373046875 = 0.0031640625 × 0.75. Every other constant is verbatim the
parent.

This is the gated chain's next step bracket, named as the unmeasured next
cell in the s0031640625 wave README (PR #3120), with that README's explicit
caveat: "the halving increments suggest the axis is close to its floor; the
bracket after that is unlikely to be worth a wave unless 0.002373046875
still improves 3/3."

**Outcome: rejected.** The candidate loses to its direct parent on 2/3
seeds (mean paired −0.00009, paired stderr 0.00011 — a wash). The step
axis no longer improves: the increment sequence on the gated chain is
+0.00837 (0.01 → 0.005625, PR #3117), +0.00191 (→ 0.00421875, PR #3118),
+0.00096 (→ 0.0031640625, PR #3120), and now −0.00009 (this wave). The
gated chain's step axis is closed at 0.0031640625, joining the trigger
axis closed at 0.03 (PR #3119): both open directions of the gated chain
are now measured to their peaks.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed),
sequential. Before the wave, a 3-task `upgd_w_control` seed-0
reproduction shard (`repro_control_seed0_3task.json`) matched the pinned
control shard's per-task accuracies bit-exactly on the shared prefix
(0.69019997, 0.77059996, 0.78119999), verifying this environment against
the pinned baseline.

The candidate shards are schema v2 while the pinned reference shards are
v1, and the merge tool refuses to mix schemas — so the shards live
outside the v1 `shards/*.json` glob, the wave `summary.json` merges only
the candidate shards (candidate named as its own control), and the paired
differences below are computed directly from the shards'
`per_task_accuracy` vectors. Three parent references are reported: the
direct parent composition
`rls_head_resid_identmap50_r_idet003_s0031640625` (retained
`wave_identmap50_r_idet003_s0031640625/` shards from PR #3120's screen,
read from that PR's branch — not committed here), the gated idet003 peak
at step 0.01 (retained `wave_identmap50_r_idet003/` shards from PR
#3115's screen, read from that PR's branch — not committed here), and the
pinned `upgd_w_control` shards. The registered gated parent
`rls_head_resid_identmap50_r` is carried from the parent waves' tables.

Runner-version note: 3-task `rls_head_resid_identmap50_r` shards rerun on
this commit (`repro_parent_seed{0,1,2}_3task.json`) are **value-identical
to the s0031640625 wave's repro shards** on every seed (both commits are
main plus a registry entry only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat; the idet003 column retains the same ≤0.001/task
cross-version offset versus the retained confirmation shards documented
in the idet002 wave (PR #3116), an order below the measured effects.

Mechanism note: unlike the trigger-only waves, a body-step change rescales
every update from step 1, so task 0 is not bit-identical to the parent
(seed 0: 0.66839999 vs 0.68059999; seed 1: 0.69439995 vs 0.70400000;
seed 2: 0.70239997 vs 0.71020001) — the whole trajectory shifts, which is
expected for this axis.

## Results

Mean online accuracy (60 tasks):

| seed | step 0.002373046875 (candidate) | s0031640625 (parent) | diff | idet003 (step 0.01) | diff | control | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.93706 | 0.93721 | −0.00015 | 0.92617 | +0.01089 | 0.77748 | +0.15958 |
| 1 | 0.93722 | 0.93746 | −0.00024 | 0.92648 | +0.01074 | 0.77856 | +0.15866 |
| 2 | 0.93818 | 0.93805 | +0.00013 | 0.92633 | +0.01185 | 0.77724 | +0.16094 |
| mean | 0.93749 | 0.93757 | **−0.00009** | 0.92633 | **+0.01116** | 0.77776 | **+0.15973** |

- **Loses to the direct parent on 2/3 seeds** (mean paired −0.00009,
  paired stderr 0.00011) — the −25% bracket crosses zero and the step
  axis stops improving. The gated chain's step axis is **closed at
  0.0031640625**; the bracket below it is not worth a wave.
- Still above the idet003 step-0.01 peak on every seed (+0.01116 paired,
  3/3) and far above the pinned control (+0.15973 paired, 3/3) — the
  rejection is a local step-axis wash, not a family regression.
- Candidate per-seed spread (per-seed stderr 0.00035) stays at the
  parent's tightened scale (0.00025); both remain two orders below the
  pre-identification arms.
- Screen-scale caution: the paired deficit (−0.00009) is far below the
  0.005 confirmation threshold recorded in the summary; this is a 60-task
  × 3 seed development screen, and confirmation would require the frozen
  full protocol under a maintainer-run lane.
- Recorded `wall_clock_seconds` per shard: 65.07 / 65.04 / 65.33 s
  (sequential).

## Next cell

None on this axis. Both open directions of the gated chain are now
measured to their peaks: the identifier trigger axis closed at 0.03
(PR #3119) and the body-step axis closes here at 0.0031640625. The gated
chain's next movement needs a new axis (or a new mechanism composition),
not another bracket on a measured-out direction.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
