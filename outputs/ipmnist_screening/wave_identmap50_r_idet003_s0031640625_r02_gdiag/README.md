# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_gdiag` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_gdiag` — the
gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac` 0.03, body step
0.0031640625, RLS head ridge 0.03 — the gated chain's best measured screen, PR #3123)
with the head's Sherman-Morrison `P` replaced by a diagonal per-direction gain
accumulator whose per-step gain denominator is the diagonal-trace form of the full
head's global excitation: `K_j = p_j*phi_j / (lambda + sum_k p_k*phi_k^2)`,
`p_j <- (p_j - K_j*phi_j*p_j)/lambda`. The off-diagonal `P` entries never exist
(O(m) state and work per step), but every active direction pools its excitation into
one shared denominator, so simultaneously active features again share estimation
gain — the exact mechanism negative result #27 hypothesized the diagonal cell's
deficit comes from losing.

This is the coupling cell the head ladder leaves open after negative results
#27/#28: #27 closed scalar-step and plain-diagonal heads, #28 closed block-local
retries ("not a different block partition or count") and named the one open escape
— global coupling structure at sub-O(d^2) cost. The pooled cell is the minimal such
structure: global (every pair of active directions is coupled through the shared
denominator) at O(m) state. The arm has **no new hyperparameter and no calibration
bracket**: `rls_lambda` 1.0, `rls_ridge_init` 0.03, `rls_reset_frac` 0.05, gate,
body step, and the residual feedback wiring are all the incumbent's verbatim
constants (the detector-driven reset keeps its parent meaning — restore
per-direction gain, keep `wout` — and the P-trace cap is rejected at build time as
in the parent). On a single-active-direction `phi` the pooled denominator reduces to
the per-coordinate one and the step is bitwise the diagonal head's (pinned by test).

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

## Result: consistent small loss that lands BELOW the plain diagonal — scalar excitation pooling is not the head coupling's value

Mean online accuracy (60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes):

| seed | pooled-denominator diagonal (candidate) | r02 (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.94777 | 0.94896 | −0.00119 |
| 1 | 0.94761 | 0.94867 | −0.00106 |
| 2 | 0.94695 | 0.94814 | −0.00119 |
| mean | 0.94744 | 0.94859 | **−0.00115** (paired stderr 0.00004, 3/3 seeds negative) |

- **A uniform small loss, slightly larger early — and below the no-pooling
  diagonal.** Per-10-task block diffs (seed 0): −0.00204/−0.00094/−0.00080/
  −0.00118/−0.00200/−0.00018; (seed 1): −0.00242/−0.00050/−0.00128/−0.00274/
  −0.00052/+0.00108; (seed 2): −0.00206/−0.00022/−0.00226/−0.00062/−0.00092/
  −0.00106. Early (tasks 0–29) vs late (30–59): −0.00126/−0.00112 (seed 0),
  −0.00140/−0.00073 (seed 1), −0.00151/−0.00087 (seed 2). Task 0 — before any
  detected boundary — measures 0.8310/0.8364/0.8392 vs the parent's
  0.8444/0.8526/0.8526: with many simultaneously active hidden directions, pooling
  all of their excitation into one shared denominator shrinks every coordinate's
  gain by the total activity, behaving like an input-dependent ridge inflation that
  is largest exactly where the full head's coupling helps most (the earliest fit).
- **The head ladder is now measured at four cells and is not monotone in "amount of
  coupling":** full Sherman-Morrison 0.94859 > block G=4 0.94827 > diagonal 0.94763
  > pooled-denominator diagonal 0.94744. Restoring only the scalar pooling while
  keeping the diagonal numerator lands 0.00019 below the plain diagonal on the
  paired candidate comparison (2/3 seeds negative; secondary, both arms measured on
  the same protocol) — the scalar pooling is not merely insufficient to explain the
  diagonal's deficit, it is a small net cost on this feature geometry. The margin
  the full head holds over the diagonal must therefore live in the coupling
  structure the pooled cell does not model: the off-diagonal numerator/accumulator
  mixing that moves cross-direction information through `P`, not the global
  excitation normalization.
- **Plasticity is unchanged** (mean 0.715/0.715/0.722 vs the parent's
  0.700/0.691/0.701), **every loss and accuracy is finite and in range**, and
  wall-clock per shard **drops** (60.8–61.2 s vs the parent's 64.8–65.6 s,
  sequential) — the O(m) cell is measurably cheaper, just 0.00115 worse here.
- The candidate is still **+0.16968 above the plain `upgd_w_control`** (0.77776).

Conclusions: negative result #27's stated deficit hypothesis ("the coupling pools
excitation across simultaneously active directions") is refuted in its scalar
form — restoring exactly that pooling at O(m) cost recovers none of the full head's
margin and lands below the no-pooling diagonal. The head ladder's value is carried
by directional (cross-feature) mixing in the Sherman-Morrison numerator and
accumulator. Any future sub-O(d^2) head replacement must carry cross-feature
directional structure (e.g. a low-rank *matrix* factorization of `P`, not a scalar
denominator normalization); scalar-pooled and block-local denominator retries on
this chain are closed. Do not re-probe scalar pooled denominators on this chain
without a mechanism that changes the numerator's cross-direction information flow.

## Records

- `rls_head_resid_identmap50_r_idet003_s0031640625_r02_gdiag_seed{0,1,2}.json`
  (schema `alberta.ipmnist_screening.shard.v2`, candidate binding commit
  `4d648e0b`, `worktree_clean: true`), `repro_parent_seed{0,1,2}_3task.json`,
  `repro_control_seed0_3task.json`, `summary.json`, per-shard logs under `logs/`.
- Negative result #29 appended to `docs/evidence/negative-results.md`.
- Development-only (`development_only` lane, permanently nonpromoting): this
  screen selects/rejects development candidates and creates no performance or
  scientific evidence, populates no `reference-dev` configuration, and consumes no
  frozen evidence seeds.
