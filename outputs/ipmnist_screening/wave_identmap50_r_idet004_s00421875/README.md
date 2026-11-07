# Wave: `rls_head_resid_identmap50_r_idet004_s00421875` first screen — REJECTED

New arm (this PR): `rls_head_resid_identmap50_r_idet004_s00421875` — the
gated idet003_s00421875 composition (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, body step size 0.00421875 — the
gated chain's best measured screen, PR #3118) with the identifier
trigger raised one bracket, `ident_reset_frac` 0.03 → 0.04. Every other
constant is verbatim the parent.

This is the gated chain's next trigger bracket, named as an unmeasured
next cell in the s00421875 wave README (PR #3118). The trigger axis had
improved monotonically at body step 0.01 (0.02 → 0.92387, PR #3116;
0.03 → 0.92633, PR #3115), and 0.04 had never been screened on any
chain.

**Outcome: rejected.** The candidate loses to its direct parent on 2 of
3 seeds (mean −0.00246, paired stderr 0.00192). The trigger axis peaks
at 0.03 on the gated chain at the best body step, mirroring the 0.03
peak already measured at step 0.01 (PR #3116).

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
the candidate shards (candidate named as its own control), and the
paired differences below are computed directly from the shards'
`per_task_accuracy` vectors. Three parent references are reported: the
direct parent composition
`rls_head_resid_identmap50_r_idet003_s00421875` (retained
`wave_identmap50_r_idet003_s00421875/` shards from PR #3118's screen,
read from that PR's branch — not committed here), the gated idet003 peak
at step 0.01 (retained `wave_identmap50_r_idet003/` shards from PR
#3115's screen, read from that PR's branch — not committed here), and
the pinned `upgd_w_control` shards. The registered gated parent
`rls_head_resid_identmap50_r` is carried from the parent waves' tables.

Runner-version note: 3-task parent shards rerun on this commit
(`repro_parent_seed{0,1,2}_3task.json`) are **value-identical to the
s00421875 wave's repro shards** on every seed (both commits are main
plus a registry entry only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat; the idet003 column retains the same ≤0.001/task
cross-version offset versus the retained confirmation shards documented
in the idet002 wave (PR #3116), an order below the measured effects.

Mechanism note: unlike the step-size waves, a trigger-only change does
nothing before the first detected boundary, so task 0 must be
bit-identical to the same-runner parent — and it is, on **all three
seeds** (0.69479996, 0.71399999, 0.71840000, exactly the parent's task-0
values). The divergence is confined to post-boundary remap timing: the
less sensitive 0.04 trigger fires later (or never, where 0.03 fired
spuriously), which is the mechanism working as designed — and measuring
worse.

## Results

Mean online accuracy (60 tasks):

| seed | trigger 0.04 (candidate) | s00421875 (parent) | diff | idet003 (step 0.01) | diff | control | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.93516 | 0.93638 | −0.00122 | 0.92617 | +0.00899 | 0.77748 | +0.15768 |
| 1 | 0.93035 | 0.93657 | −0.00622 | 0.92648 | +0.00387 | 0.77856 | +0.15179 |
| 2 | 0.93695 | 0.93687 | +0.00007 | 0.92633 | +0.01061 | 0.77724 | +0.15971 |
| mean | 0.93415 | 0.93661 | **−0.00246** | 0.92633 | **+0.00782** | 0.77776 | **+0.15639** |

- **Loses to the direct parent on 2/3 seeds** (mean −0.00246, paired
  stderr 0.00192). The single positive seed is +0.00007, inside one
  paired stderr — a rejection, not a split decision.
- Still above the idet003 step-0.01 peak on every seed (+0.00782
  paired, 3/3) — the rejection is attributable to the trigger bracket,
  not the step composition.
- The candidate's per-seed spread **reopens** (per-seed stderr 0.00197
  vs the parent's 0.00014): the equalization signature that the
  identifier chain had been tightening breaks at the less sensitive
  trigger, consistent with later/missed remaps leaving some seeds
  uncorrected longer.
- Do-not-retry scope: `ident_reset_frac` 0.04 on the gated
  identmap50_r chain at body step 0.00421875 is refuted at screen scale;
  higher triggers (0.05 was the identmap50_r default) are strictly less
  sensitive and are unmeasured only in composition with the smaller
  step, which this direction makes unpromising. Lower triggers (0.02)
  are already measured worse at step 0.01 (PR #3116). The trigger axis
  is closed on this chain unless a mechanism-level reason to reopen
  appears. The arm stays registered with its reduction pins
  (`TestRLSHeadIdentMapIdet004S00421875`); the shards bind this
  commit. No `docs/evidence/negative-results.md` entry is added in this
  PR — entry #24 is claimed by open PR #3113, and this bracket
  rejection follows the #3114 convention of recording the refutation in
  the wave README; fold it into the ledger at family close if wanted.
- Recorded `wall_clock_seconds` per shard: 66.23 / 65.08 / 65.19 s
  (sequential).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.

Unmeasured next cell (nobody's declared axis): the gated chain's next
body-step bracket, 0.0031640625 = 0.00421875 × 0.75 — the step axis is
the only remaining open direction on the gated chain, with the caveat
that its last increment (+0.00191) was already 4.4x smaller than the one
before (+0.00837), so diminishing returns are visible.
