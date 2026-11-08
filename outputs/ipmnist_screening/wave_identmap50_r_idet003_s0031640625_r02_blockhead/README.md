# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_blockhead` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_blockhead`
— the gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, residual body step size 0.0031640625, RLS head ridge 0.03 — the gated
chain's best measured screen, PR #3123) with the head's full Sherman-Morrison
`P` restricted to `G = 4` contiguous blocks of the `m = h2+1 = 151` head
coordinates (blocks of 38/38/38/37 real coordinates, the tail zero-padded to
equal sizes). Each block runs its own exact Sherman-Morrison RLS on the shared
one-hot squared-loss residual: `K_b = P_b phi_b / (lambda + phi_b^T P_b
phi_b)`, `P_b <- (P_b - outer(K_b, P_b phi_b))/lambda` with the parent's
per-update symmetrization and the parent's detector-driven reset meaning
(restore every block's gain, keep `wout`). Every other constant is verbatim
the parent, and there is no new tuned constant: the block count `G` is the
measured axis itself (`G = 1` is bitwise the full head, pinned by test;
`G = m` is exactly the diagonal head, negative result #27).

This is the interior cell of negative result #27's named escape class
("coupling at sub-O(d^2) cost", e.g. "factored/low-rank curvature"): the
block restriction keeps all within-block cross-direction coupling exactly and
removes only the cross-block coupling — 8,550 of 11,325 coordinate pairs
(~75.5%) are decoupled, at `O(m^2/G) = 4·38^2 = 5776` state entries (~25% of
the full head's 22,801) and measurably lower wall-clock. Negative #27 closed
the endpoints (`G = 1` full 0.94859, `G = m` diagonal 0.94763) and named
exactly this mechanism class as the only remaining competitive route; this
wave measures the interior instead of retrying a closed endpoint.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, sequential fresh processes. Before
the wave:

- 3-task parent reproduction shards (`repro_parent_seed{0,1,2}_3task.json`)
  matched the diaghead wave's retained repro shards' 3-task prefixes
  **bit-exactly on all three seeds** (0.8444/0.9100/0.9210,
  0.8526/0.9108/0.9152, 0.8526/0.9032/0.9150) — the registration adds a
  registry entry plus a build-time branch only, and the `G = 1` pin holds
  bitwise, so the parent pairing carries no version-offset caveat;
- a 3-task `upgd_w_control` seed-0 reproduction shard
  (`repro_control_seed0_3task.json`) matched the pinned control's per-task
  accuracies bit-exactly on the shared prefix (0.69019997, 0.77059996,
  0.78119999).

The wave `summary.json` merges only the candidate shards (candidate named as
its own control), and the paired differences below are computed directly from
the shards' `per_task_accuracy` vectors. The parent reference is the retained
`wave_identmap50_r_idet003_s0031640625_r02/` parent shards.

Mechanism note: the block restriction touches only the head's `P` structure,
so trajectories diverge from the parent inside the first task. Task 0 is
bit-identical on seed 0 (0.8444) and slightly different on seeds 1–2
(0.8538 vs 0.8526, 0.8506 vs 0.8526) — unlike the diagonal cell, the earliest
within-task fit is *not* uniformly slower.

## Results

Mean online accuracy (60 tasks):

| seed | block head G=4 (candidate) | r02 full head (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.94848 | 0.94896 | −0.00048 |
| 1 | 0.94840 | 0.94867 | −0.00027 |
| 2 | 0.94794 | 0.94814 | −0.00020 |
| mean | 0.94827 | 0.94859 | **−0.00032** (paired stderr 0.00008, 3/3 seeds negative) |

- **A consistent small loss, one third of the diagonal cell's deficit.**
  Head ladder on the identified gated chain: full `G = 1` 0.94859 (incumbent)
  → block `G = 4` 0.94827 (this wave, −0.00032) → diagonal `G = m` 0.94763
  (negative #27, −0.00096) → plain SGD 0.45005 (negative #26). The candidate
  is still **+0.17051 above the plain `upgd_w_control`** (0.77776).
- **The deficit is a steady-state margin, not a slower earliest fit.** Task
  blocks 1–10 are mixed (+0.00078/+0.00022/−0.00000), while blocks 11–30
  (−0.00028/−0.00059/−0.00035) and 31–60 (−0.00062/−0.00057/−0.00024) are
  negative on every seed. Within-block coupling carries essentially all of
  the head's rapid within-task fit; the removed cross-block coupling is worth
  a small uniform late-life margin.
- **Plasticity is unchanged** (mean 0.708/0.706/0.712 vs the parent's
  0.700/0.691/0.701) and **no instability artifact is present**: every loss
  is finite, and wall-clock per shard **drops** (62.4–62.5 s vs 64.8–65.6 s,
  sequential) — the O(m^2/4) head is measurably cheaper, it is just 0.0003
  worse here.
- Screen-scale caution: negative results #21/#23 measured that this family's
  60-task effects decay 3.5–4.7x at 200 tasks. The effect here is ~4σ paired
  at the screen but ~1/3 of an already-small cell; this is a development
  screen (`development_only`, `scientific_promotion_allowed: false`), not a
  confirmation candidate. The incumbent stands at `..._r02` (0.94859); this
  arm claims no promotion.

## Axis status: the G ladder is measured at its endpoints and interior

Block-factored coupling at O(m^2/G) cost recovers ~2/3 of the diagonal
cell's deficit while decoupling ~3/4 of the coordinate pairs, and still loses
on every seed. The loss shrinks smoothly with `G` (G = m: −0.00096 → G = 4:
−0.00032 → G = 1: 0), so on this feature geometry the cross-block coupling is
load-bearing at the margin and its value is roughly proportional to the
fraction of coupled pairs. Negative #27's escape class is now measured: a
block-local exact-RLS replacement does not retain the incumbent's level. Any
future sub-O(d^2) head replacement must pool excitation *globally* (e.g. a
low-rank/factored global curvature), not block-locally; scalar-step
(#26), diagonal (#27), and block-local (this wave) retries are closed. The
incumbent stands at `..._idet003_s0031640625_r02` (0.94859).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.

## Validation

New `TestRLSHeadIdentMapIdet003S0031640625R02Blockhead` (9 tests): the
registry shape binds the parent's constants plus `rls_head_block = 4.0` only;
explicit `rls_head_block = 1.0` follows the registered r02 parent
**bit-for-bit** (the `G = 1` endpoint is the unchanged full-P graph); with
the reset at its untriggerable sentinel the first update equals the exact
per-block recursion (1e-5 tolerance: XLA fuses the einsum chains) and the
zero-padded tail coordinates stay inert at the initial ridge value with zero
coupling to real coordinates; `G = m` reproduces the registered diagonal
head's screen trajectory (1e-6); the harness run is finite, in range, and off
the incumbent's trajectory; the identmap factory delegates bitwise at
`ident_match_at = 0` with a rank-3 `P` state; a block count above the head
width fails at init time; and invalid values and untested compositions
(non-integer, λ < 1, parallel head, ungated, whitened/Newton/SGD/SM bodies,
diag composition, trace cap) fail at build time.

Records: `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_blockhead/`
(3 candidate shards binding commit `5c96b398`, control + 3 parent repro
shards bit-exact against the retained prior-wave prefixes, `summary.json`,
`README.md`).
