# Wave: `rls_head_resid_identmap50_r_mom9` first screen

New arm (this PR): `rls_head_resid_identmap50_r_mom9` — the confirmed
`rls_head_resid_identmap50_r` arm (residual RLS head behind the online
permutation identifier, refining matches at 50/200/2000 post-shift
samples) with classical bias-corrected EMA momentum (`body_momentum`
0.9) on the gated residual body. Every other constant is verbatim the
parent arm.

Motivation: the family's named remaining headroom is within-task
convergence (negative result #22), and the live step-size brackets show
noise reduction still pays on this head — but the reduction so far comes
from the RLS protocol head and the utility gate, not from averaging the
body's descent direction. Momentum has never been measured on this
family (the `sgd_momentum_gate` loss was on the pre-identmap
cross-entropy family). Mechanism precedent: the `sgd_momentum_gate` EMA
form, which keeps `m_hat` at raw-gradient scale. Gate arithmetic is
byte-identical to `_gated_sgd` (including the closed-EMA skip guard);
only the descent direction changes. `body_momentum` 0 never builds the
momentum branch (incumbent path byte-for-byte, pinned). Reduction pins
(`TestRLSHeadResidualMomentum`, 5 tests): reduction, live+finite,
fail-closed composition (no `resid_whiten`/`resid_newton`/`body_sm`/
L2-Init), identmap composition, and registry shape.

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
reports). The parent reference is the retained 200-task confirmation
shards' 60-task prefix (`smprecond_confirm_r1/shards/`), whose
first-60 per-task means reproduce the parent's published 60-task screen
values exactly.

## Results

Mean online accuracy (60 tasks):

| seed | mom9 | upgd_w_control | diff vs control | identmap50_r (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.91553 | 0.77748 | +0.13805 | 0.91602 | −0.00049 |
| 1 | 0.89664 | 0.77856 | +0.11808 | 0.89708 | −0.00045 |
| 2 | 0.92121 | 0.77724 | +0.14397 | 0.92117 | +0.00004 |
| mean | 0.91113 | 0.77776 | **+0.13337** | 0.91142 | **−0.00030** |

- **Rejected vs parent:** momentum at µ 0.9 does not improve on the
  gated-SGD parent — mean paired delta −0.00030 with paired stderr
  0.00017, and 2 of 3 seeds negative. The only positive seed is
  +0.00004, within noise of zero.
- Vs the `upgd_w_control` paired reference the arm is +0.13337 mean
  (all seeds positive), as every arm in this family is; that gap is the
  parent composition's, not momentum's.
- Consistent with the family precedent: this head's noise reduction is
  already carried by the RLS protocol head and the utility gate; EMA
  averaging of the body's gradient direction adds nothing measurable
  and slightly overshoots. Recorded so the idea is not retried at this
  composition; higher µ values are unmeasured and unclaimed.
- Recorded `wall_clock_seconds` per shard: 83.2 / 75.4 / 74.1 s (seeds
  ran sequentially on this host).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol
under a maintainer-run lane.
