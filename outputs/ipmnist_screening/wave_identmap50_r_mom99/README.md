# Wave: `rls_head_resid_identmap50_r_mom99` first screen

New arm (this PR): `rls_head_resid_identmap50_r_mom99` — the confirmed
`rls_head_resid_identmap50_r` arm (residual RLS head behind the online
permutation identifier, refining matches at 50/200/2000 post-shift
samples) with classical bias-corrected EMA momentum (`body_momentum`
0.99) on the gated residual body. Every other constant is verbatim the
parent arm.

Motivation: negative result #24 measured the µ 0.9 end of the momentum
bracket at this composition (null-to-negative, −0.00030 paired) and
explicitly left higher µ unmeasured and unclaimed. µ 0.99 stretches the
body averaging window to roughly 100 steps — long enough to span the
identifier's 50-sample first match and 200-sample refinement — so this
wave separates "averaging per se is neutral" from "window length across
shift adaptation is the cost". Mechanism, validation, and reduction pins
are the retained mom9 machinery (generic in `body_momentum`); the only
new code is the registry entry and its registry-shape pin
(`TestRLSHeadIdentMap.test_registry_arms`).

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, one process per (arm, seed).
Before the wave, a 3-task `upgd_w_control` seed-0 reproduction shard
(`repro_upgd_w_control_seed0.json`, retained here) matched the pinned
`shards/upgd_w_control_seed0.json` per-task accuracies bit-exactly on
the 3-task prefix (0.69019997, 0.77059996, 0.78119999), verifying this
environment against the pinned baseline.

The shards are schema v2 and the pinned parent reference shards are the
retained 200-task confirmation shards' 60-task prefix
(`smprecond_confirm_r1/shards/`), whose first-60 per-task means
reproduce the parent's published 60-task screen values exactly. The
`summary.json` here merges only the candidate shards (candidate named
as its own control; the merge tool refuses to mix schemas), and the
paired differences below are computed directly from the shards'
`per_task_accuracy` vectors (the same per-seed means the merge
reports).

## Results

Mean online accuracy (60 tasks):

| seed | mom99 | upgd_w_control | diff vs control | identmap50_r (parent) | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.91261 | 0.77748 | +0.13513 | 0.91602 | −0.00341 |
| 1 | 0.89115 | 0.77856 | +0.11259 | 0.89708 | −0.00594 |
| 2 | 0.91921 | 0.77724 | +0.14197 | 0.92117 | −0.00196 |
| mean | 0.90766 | 0.77776 | **+0.12990** | 0.91142 | **−0.00377** |

- **Rejected vs parent:** momentum at µ 0.99 loses to the gated-SGD
  parent — mean paired delta −0.00377 with paired stderr 0.00116, and
  3 of 3 seeds negative (the per-seed diffs are −0.00341, −0.00594,
  −0.00196). The loss is ~12x the µ 0.9 delta (−0.00030) and ~3.3x its
  own stderr.
- Vs the `upgd_w_control` paired reference the arm is +0.12990 mean
  (all seeds positive), as every arm in this family is; that gap is the
  parent composition's, not momentum's.
- **The µ bracket is now closed on both sides.** µ 0.9 ≈ null
  (−0.00030); µ 0.99 clearly negative. Reading: the head's noise
  reduction is already carried by the RLS protocol head and the utility
  gate, and a ~100-step body averaging window spans the identifier's
  post-shift re-adaptation phase (first match at 50 samples,
  refinement at 200), dragging stale pre-shift descent direction
  through the window the parent spends re-adapting. Averaging window
  length across shift adaptation, not averaging per se, is the cost.
- The mechanism is closed at this composition for the measured bracket;
  do not retry `body_momentum` on this head without a mechanism change
  (e.g. window reset at detected shifts, which would be a new
  mechanism, not this one).
- Recorded `wall_clock_seconds` per shard: 70.7 / 74.2 / 71.6 s (seeds
  ran sequentially on this host).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol
under a maintainer-run lane.
