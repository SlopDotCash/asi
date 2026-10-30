# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_diaghead` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_diaghead` — the
gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac` 0.03, body step
0.0031640625, RLS head ridge 0.03 — the gated chain's best measured screen, PR #3123)
with the head's full Sherman-Morrison `P` replaced by a diagonal per-direction gain
accumulator: each coordinate runs its own scalar RLS on the shared one-hot squared-loss
residual, `K_j = p_j*phi_j / (lambda + p_j*phi_j^2)`, `p_j` nonincreasing at
`lambda = 1`, O(m) state and work per step instead of O(m^2).

This is negative result #26's named escape class ("a second-order or per-direction
adaptive head") and ipmnist-theory §6.6's removability question, one cell further
than #26: the SGD-head control removed the head's adaptive gain entirely and
collapsed (0.45005, negative #26); this cell keeps per-direction second-order
adaptation and removes only the off-diagonal cross-direction coupling `phi^T P phi`.
On a single-active-direction `phi` the diagonal gain is bitwise the full head's gain,
so the removed mechanism is exactly the coupling that pairs of simultaneously active
features use to share estimation gain. The arm has **no new hyperparameter and no
calibration bracket**: `rls_lambda` 1.0, `rls_ridge_init` 0.03, `rls_reset_frac` 0.05,
gate, body step, and the residual feedback wiring are all the incumbent's verbatim
constants (the detector-driven reset keeps its parent meaning — restore per-direction
gain, keep `wout` — and the P-trace cap is rejected at build time as in the parent).

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

## Result: consistent small loss — the cross-direction coupling is load-bearing at the margin

Mean online accuracy (60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes):

| seed | diagonal head (candidate) | r02 (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.94811 | 0.94896 | −0.00085 |
| 1 | 0.94752 | 0.94867 | −0.00115 |
| 2 | 0.94727 | 0.94814 | −0.00087 |
| mean | 0.94763 | 0.94859 | **−0.00096** (paired stderr 0.00010, 3/3 seeds negative) |

- **A uniform small loss, not a transient and not a collapse.** Per-10/20/30-task
  block diffs sit at −0.00094/−0.00091/−0.00079 (seed 0), −0.00166/−0.00114/−0.00099
  (seed 1), −0.00124/−0.00126/−0.00048 (seed 2): the deficit is present in every
  block of every seed, slightly larger early and never reversing. The candidate is
  still **+0.16987 above the plain `upgd_w_control`** (0.77776) and **0.50 above the
  SGD-head control** (0.45005, negative #26).
- **The within-task fit itself is slightly slower.** Task 0 — before any detected
  boundary — measures 0.8358/0.8434/0.8436 vs the parent's 0.8444/0.8526/0.8526:
  removing the cross-direction denominator coupling slows the earliest
  convergence the way a small ridge increase would, consistent with the ridge
  axis's saturation picture (PR #3123) — the coupling effectively pools excitation
  across simultaneously active directions, and dropping it costs a fixed fraction
  of the earliest fit everywhere rather than opening a late-life gap.
- **Plasticity is unchanged** (mean 0.716/0.713/0.721 vs the parent's
  0.700/0.691/0.701) and **no instability artifact is present**: every loss is
  finite, and wall-clock per shard **drops** (61.7–61.9 s vs 64.8–65.6 s,
  sequential) — the O(m) head is measurably cheaper, it is just 0.001 worse here.
- **Conclusions:** the per-direction second-order structure — not the residual's
  geometry alone — carries essentially all of the head's within-task fit (#26's
  collapse is fully explained by the missing adaptive gain), but the
  cross-direction coupling `phi^T P phi` is worth a small, uniform, 3/3-seed
  **−0.00096 ± 0.00010** on the identified chain. The O(d^2) head is therefore
  *not freely* removable at O(d) cost (§6.6): a diagonal second-order head retains
  ~99.9% of the incumbent's level and loses to it on every seed. Recorded as
  **negative result #27** in `docs/evidence/negative-results.md`. The incumbent
  stands at `..._idet003_s0031640625_r02` (0.94859); this arm claims no promotion.

## Axis status

Head-family ladder on the identified gated chain: full Sherman-Morrison 0.94859
(incumbent) → per-direction diagonal 0.94763 (this wave, −0.00096) → plain SGD
0.45005 (negative #26). The named escape class from #26 ("a second-order or
per-direction adaptive head") is now measured: per-direction adaptation without
coupling loses a small but uniform margin, so any future head replacement must keep
cross-direction coupling (e.g. a factored/low-rank curvature) to be competitive —
a scalar-step or diagonal-head family retry without new coupling structure is
closed.

## Validation

- **Reduction pins (new `TestRLSHeadIdentMapIdet003S0031640625R02Diaghead`, 10
  tests):** `rls_head_diag=0.0` (explicit) follows the registered r02 parent
  **bit-for-bit**; the infinite-ridge degenerate freezes the head at `wout = 0`
  (accuracy = class-0 indicator, loss 0.5/step, the documented measurable
  degenerate); with the reset at its untriggerable sentinel the first update equals
  the exact per-direction recursion `K_j = p_j*phi_j/(1+p_j*phi_j^2)`,
  `wout_1 = outer(K, onehot)`, `p_1 = p_0 − K*(p_0*phi)` (1e-6 tolerance: the
  elementwise chain and the reset-out compilation re-fuse XLA multiplies by 1–2
  float32 ulp; the flag-off and registry pins stay bitwise); a
  single-active-direction `phi` gives a **bitwise-equal** gain to the full
  Sherman-Morrison head (the documented sparsity reduction); the accumulator is
  nonincreasing, nonnegative, and finite across a boundary-heavy stream (no wind-up
  at lambda 1); registry shape (parent keys + `rls_head_diag`); composition guards
  reject untested envelopes (lambda != 1, non-residual body, ungated, whitened,
  Newton, SGD-head compose, SM body preconditioner, P-trace cap, invalid flag).
  `test_registry_arms` updated.
- **Focused checks on `e562e60b`:** `pytest tests/test_ipmnist_screening.py -k
  "RLSHead"` → 67 passed; `ruff check` clean on both touched files; `mypy
  alberta_framework/benchmarks/ipmnist_screening.py` → no issues.
- **Environment identity:** the control and all three parent repro shards reproduce
  the pinned references bit-exactly (see Protocol).
- **Retained artifacts:**
  `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_diaghead/`
  (README, 3 candidate shards schema v2 binding commit `e562e60b`, control + 3
  parent repro shards, `summary.json`, `worktree_clean: true`, per-shard logs).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`,
`evidence_class: development_screening_diagnostic`). It claims no promotion;
confirmation would require the frozen full protocol under a maintainer-run lane.

Landing order: `test_registry_arms` asserts exact set equality, so this conflicts
with the other open screen PRs by construction, same as the rest of the chain.
