# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_lowrank` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_lowrank` — the
gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac` 0.03, body step
0.0031640625, RLS head ridge 0.03 — the gated chain's best measured screen, PR #3123)
with the head's dense Sherman-Morrison `P` replaced by a diagonal accumulator plus a
sliding window of the 8 most recent exact downdate directions: with `rls_lambda` 1.0
every step is still the exact symmetric downdate
`P <- P - outer(v, v)`, `v = P phi / sqrt(1 + phi^T P phi)`; the diagonal `p` absorbs
every downdate's diagonal footprint and the window carries the cross-direction
(off-diagonal) structure that negative result #29 measured to be the head's value —
`O(8*m)` state and work per step instead of `O(m^2)`.

This is the cost cell negative result #29 leaves open and explicitly names ("a
low-rank *matrix* factorization of `P`, not a scalar denominator normalization").
#27/#28/#29 closed diagonal, block-local, and scalar-pooled cells; the full head
holds +0.00115 over the best cheap cell. The arm has **one new pinned constant** —
the window size `rls_head_rank = 8` (`m = h2+1 = 151`, so 8*151 stored floats vs
151^2); `rls_lambda`, ridge, reset, gate, body step, and the residual feedback are
the incumbent's verbatim constants (the detector-driven reset restores the whole `P`
— `p` to `1/ridge_init`, window cleared — and keeps `wout`; the trace cap is rejected
at build time as in the parent). Nothing is dropped before step 8, so an un-dropped
window is mathematically the full head (pinned by test up to float reduction order);
the approximation is exactly the forgetting of dropped directions' cross coupling.
On a single-active-direction `phi` the window terms cancel at the active coordinate
and the active gain reduces to the diagonal head's (pinned by test).

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes. Before the wave:

- 3-task parent reproduction shards (`repro_parent_seed{0,1,2}_3task.json`) matched
  the retained r02 parent shards' 3-task prefixes **bit-exactly on all three seeds**
  (0.8444/0.9100/0.9210, 0.8526/0.9108/0.9152, 0.8526/0.9032/0.9150) — the
  registration adds a registry entry plus a build-time branch only, and the flag-off
  pin holds bitwise, so the parent pairing carries no version-offset caveat;
- a 3-task `upgd_w_control` seed-0 reproduction shard
  (`repro_control_seed0_3task.json`) matched the pinned control's per-task accuracies
  bit-exactly on the shared prefix (0.69019997, 0.77059996, 0.78119999).

The wave `summary.json` merges only the candidate shards (candidate named as its own
control), and the paired differences below are computed directly from the shards'
`per_task_accuracy` vectors. The parent reference is the retained
`wave_identmap50_r_idet003_s0031640625_r02/` parent shards.

## Result: consistent small loss that lands exactly at the scalar-pooling level — bounded-rank curvature sketches do not capture the head's coupling value

Mean online accuracy (60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes):

| seed | low-rank window R=8 (candidate) | r02 (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.94742 | 0.94896 | −0.00154 |
| 1 | 0.94769 | 0.94867 | −0.00098 |
| 2 | 0.94723 | 0.94814 | −0.00091 |
| mean | 0.94745 | 0.94859 | **−0.00114** (paired stderr 0.00020, 3/3 seeds negative) |

- **A uniform small loss, and dead even with the pooled-denominator cell.**
  Per-10-task block diffs (seed 0): −0.00176/−0.00042/−0.00154/−0.00196/−0.00200/
  −0.00154; (seed 1): −0.00142/+0.00012/−0.00148/−0.00236/−0.00054/−0.00018;
  (seed 2): −0.00176/−0.00034/−0.00152/−0.00020/−0.00068/−0.00094. Early (tasks
  0–29) vs late (30–59): −0.00124/−0.00183 (seed 0), −0.00093/−0.00103 (seed 1),
  −0.00121/−0.00061 (seed 2). Task 0 — before any detected boundary — measures
  0.8344/0.8418/0.8412 vs the parent's 0.8444/0.8526/0.8526. The secondary
  candidate comparison against #29's pooled-denominator arm is a wash
  (+0.000003 mean, 2/3 seeds positive): keeping the last 8 exact downdate
  directions' cross coupling recovers none of the full head's margin over the
  no-mixing cells.
- **The head ladder is measured at five cells and is flat below the full head:**
  full Sherman-Morrison 0.94859 > block G=4 0.94827 > diagonal 0.94763 >
  pooled-denominator diagonal 0.94744 ≈ low-rank window (R=8) 0.94745. Whatever
  carries the full head's +0.00115 over the cheap cells, a bounded window of
  recent-curvature directions does not hold it: the load-bearing cross-direction
  structure is either older than the window or denser than any rank-8 sketch of
  this exact-downdate form.
- **Plasticity is unchanged** (mean 0.714/0.712/0.717 vs the parent's
  0.700/0.691/0.701), **every loss and accuracy is finite and in range**, and
  wall-clock per shard does **not** drop (63.7–65.7 s vs the parent's 64.8–65.6 s,
  sequential) — at `m = 151` the factored matvec saves nothing against the dense
  matmul, so this cell is not even a cost win, just 0.00114 worse.
- The candidate is still **+0.16969 above the plain `upgd_w_control`** (0.77776).

Conclusions: negative result #29's named escape class — a low-rank matrix
factorization of `P` via the exact downdate directions themselves — is measured and
does not recover the full head's level; it lands exactly at the scalar-pooling cell.
The four cheap cells (diagonal, block-local, scalar-pooled, rank-8 downdate window)
are now all closed on this chain. Do not re-probe sliding-window rank sketches on
this chain without a mechanism that identifies which cross-direction structure is
load-bearing (e.g. measured direction importance, not recency). The full
Sherman-Morrison head stands as load-bearing at `O(m^2)` cost on this incumbent.

## Records

- `rls_head_resid_identmap50_r_idet003_s0031640625_r02_lowrank_seed{0,1,2}.json`
  (schema `alberta.ipmnist_screening.shard.v2`, candidate binding commit
  `d068ddbf`, `worktree_clean: true`), `repro_parent_seed{0,1,2}_3task.json`,
  `repro_control_seed0_3task.json`, `summary.json`, per-shard logs under `logs/`.
- Negative result #30 appended to `docs/evidence/negative-results.md`.
- Development-only (`development_only` lane, permanently nonpromoting): this
  screen selects/rejects development candidates and creates no performance or
  scientific evidence, populates no `reference-dev` configuration, and consumes no
  frozen evidence seeds.
