# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625` — the
gated idet003_s00421875 composition (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, residual body step size 0.00421875 — the gated chain's best measured
screen, PR #3118) with the body step size lowered one more -25% bracket to
0.0031640625 = 0.00421875 × 0.75. Every other constant is verbatim the
parent.

This is the gated chain's next step bracket, named as the unmeasured next
cell in the s00421875 wave README (PR #3119): with the trigger axis closed
at 0.03 (PR #3119), the step axis is the chain's only remaining open
direction, with the README's caveat that the last increment (+0.00191) was
already 4.4x smaller than the one before (+0.00837).

**Outcome: improves on every seed.** The candidate beats its direct parent
on 3/3 seeds (mean +0.00096, paired stderr 0.00011). The step axis has now
improved monotonically through every measured bracket on the gated chain
(0.01 → 0.005625, +0.00837, PR #3117; 0.005625 → 0.00421875, +0.00191,
PR #3118; 0.00421875 → 0.0031640625, +0.00096, this wave) — still
positive, with the increment roughly halving each bracket.

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
`rls_head_resid_identmap50_r_idet003_s00421875` (retained
`wave_identmap50_r_idet003_s00421875/` shards from PR #3118's screen,
read from that PR's branch — not committed here), the gated idet003 peak
at step 0.01 (retained `wave_identmap50_r_idet003/` shards from PR
#3115's screen, read from that PR's branch — not committed here), and
the pinned `upgd_w_control` shards. The registered gated parent
`rls_head_resid_identmap50_r` is carried from the parent waves' tables.

Runner-version note: 3-task `rls_head_resid_identmap50_r` shards rerun on
this commit (`repro_parent_seed{0,1,2}_3task.json`) are **value-identical
to the s00421875 wave's repro shards** on every seed (both commits are
main plus a registry entry only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat; the idet003 column retains the same ≤0.001/task
cross-version offset versus the retained confirmation shards documented
in the idet002 wave (PR #3116), an order below the measured effects.

Mechanism note: unlike the trigger-only waves, a body-step change rescales
every update from step 1, so task 0 is not bit-identical to the parent
(seed 0: 0.68059999 vs 0.69479996; seed 1: 0.70400000 vs 0.71399999;
seed 2: 0.71020001 vs 0.71840000) — the whole trajectory shifts, which is
expected for this axis.

## Results

Mean online accuracy (60 tasks):

| seed | step 0.0031640625 (candidate) | s00421875 (parent) | diff | idet003 (step 0.01) | diff | control | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.93721 | 0.93638 | +0.00083 | 0.92617 | +0.01104 | 0.77748 | +0.15973 |
| 1 | 0.93746 | 0.93657 | +0.00089 | 0.92648 | +0.01097 | 0.77856 | +0.15890 |
| 2 | 0.93805 | 0.93687 | +0.00117 | 0.92633 | +0.01171 | 0.77724 | +0.16081 |
| mean | 0.93757 | 0.93661 | **+0.00096** | 0.92633 | **+0.01124** | 0.77776 | **+0.15981** |

- **Beats the direct parent on 3/3 seeds** (mean +0.00096, paired stderr
  0.00011) — the step axis remains monotone-improving on the gated chain.
- The increment roughly halved again (+0.00096 vs +0.00191), so the axis
  is converging; the next bracket (0.002373046875) is the named
  unmeasured cell below.
- The candidate's per-seed spread stays at the parent's tightened scale
  (per-seed stderr 0.00025 vs the parent's 0.00014, both two orders below
  the pre-identification arms) — the equalization signature survives the
  smaller step.
- Still above the idet003 step-0.01 peak on every seed (+0.01124 paired,
  3/3) and far above the pinned control (+0.15981 paired, 3/3).
- Screen-scale caution: the paired gain (+0.00096) is below the 0.005
  confirmation threshold recorded in the summary; this is a 60-task × 3
  seed development screen, and confirmation would require the frozen full
  protocol under a maintainer-run lane.
- Recorded `wall_clock_seconds` per shard: 64.87 / 64.82 / 64.71 s
  (sequential).

## Next cell

The gated chain's step axis continues to improve: the next -25% bracket,
0.002373046875 = 0.0031640625 × 0.75, is the named unmeasured next cell
(nobody's declared axis). The halving increments suggest the axis is
close to its floor; the bracket after that is unlikely to be worth a wave
unless 0.002373046875 still improves 3/3.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
