# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02` — the
gated `idet003_s0031640625_r01` composition (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, residual body step size 0.0031640625, RLS head ridge 0.1 — the gated
chain's best measured screen, PR #3122) with the RLS head's initial/reset
ridge `rls_ridge_init` lowered 0.1 → 0.03. Every other constant is verbatim
the parent.

This is the second cell of the gated chain's new head axis, opened by PR
#3122 (+0.01040 on every seed, above the 0.005 confirmation bar). The wave-3
ridge star (1.0/0.3/0.1/0.03/0.01) ran **only at `rls_lambda=0.999`**
(`summary_rls_head.json`), so its small-ridge cells are confounded with
forgetting wind-up (negative result #10); at λ=1.0 the Sherman-Morrison P
matrix is nonincreasing PSD (P_t ≤ P0 = I/0.03 ≈ 33·I), so the smallest named
ladder cell is genuinely unmeasured and structurally safe.
The P-trace cap stays disabled (`rls_p_trace_cap = 0.0`) exactly as in the
parent. Negative #20's closure covers forgetting (`λ < 1`), which this arm
does not touch.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, sequential fresh processes. Before
the wave, a 3-task `upgd_w_control` seed-0 reproduction shard
(`repro_control_seed0_3task.json`) matched the pinned control shard's
per-task accuracies bit-exactly on the shared prefix (0.69019997, 0.77059996,
0.78119999), verifying this environment against the pinned baseline.

The candidate shards are schema v2 while the pinned reference shards are v1,
and the merge tool refuses to mix schemas — so the shards live outside the
v1 `shards/*.json` glob, the wave `summary.json` merges only the candidate
shards (candidate named as its own control), and the paired differences
below are computed directly from the shards' `per_task_accuracy` vectors.
Four parent references are reported: the direct parent composition
`rls_head_resid_identmap50_r_idet003_s0031640625_r01` (retained
`wave_identmap50_r_idet003_s0031640625_r01/` shards from PR #3122's screen,
read from that PR's branch — not committed here), the chain's ridge-1.0
origin `rls_head_resid_identmap50_r_idet003_s0031640625` (PR #3120's
branch), the gated idet003 peak at step 0.01 (PR #3115's branch), and the
pinned `upgd_w_control` shards.

Runner-version note: 3-task `rls_head_resid_identmap50_r` shards rerun on
this commit (`repro_parent_seed{0,1,2}_3task.json`) are **value-identical to
the r01 wave's repro shards** on every seed (both commits are main plus a
registry entry and tests only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat; the idet003 and s0031640625 columns retain the same
≤0.001/task cross-version offset versus the retained confirmation shards
documented in the idet002 wave (PR #3116), an order below the measured
effects.

Mechanism note: the ridge change touches only the head's P initialization
and P reset, so the trajectories diverge from the parent from the first task
(seed 0 task 0: 0.84439999 vs 0.81919998 — the faster head improves the very
first task, as in the r01 wave), expected for a head axis change.

## Results

Mean online accuracy (60 tasks):

| seed | ridge 0.03 (candidate) | r01 ridge 0.1 (parent) | diff | s0031640625 (ridge 1.0) | diff | control | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.94896 | 0.94802 | +0.00094 | 0.93721 | +0.01175 | 0.77748 | +0.17148 |
| 1 | 0.94867 | 0.94797 | +0.00070 | 0.93746 | +0.01121 | 0.77856 | +0.17011 |
| 2 | 0.94814 | 0.94794 | +0.00020 | 0.93805 | +0.01009 | 0.77724 | +0.17090 |
| mean | 0.94859 | 0.94797 | **+0.00062** | 0.93757 | **+0.01102** | 0.77776 | **+0.17083** |

- **Beats the direct parent on 3/3 seeds** (mean +0.00062, paired stderr
  0.00028, ~2.2σ) — a real but small increment, ~1/17 of the ridge 0.1 cell
  (+0.01040). The ridge axis stays monotone in the improving direction but
  is flattening sharply toward saturation near 0.948–0.949.
- **The gain is early-run convergence, not a higher asymptote.** Split by
  task block (per-seed mean diffs vs the parent): tasks 1–10
  +0.00658/+0.00676/+0.00580, tasks 11–30 +0.00078/+0.00079/+0.00052,
  tasks 31–60 −0.00083/−0.00138/−0.00187. The smaller ridge converges every
  task's head faster (including before any detected boundary), and at the
  plateau the larger early/post-reset gains cost a slightly noisier
  steady state (last-15 diffs −0.00088/−0.00209/−0.00195).
- Late-window check (last 15 tasks): candidate slopes −0.000071/−0.000053/
  −0.000206 per task vs the parent's −0.000018/+0.000031/−0.000274 — both
  plateau; no within-run collapse (every seed's full-horizon mean stays
  +0.0002 above the parent).
- Per-task spread is unchanged from the parent (candidate per-seed stderr
  0.0020–0.0021 vs the parent's 0.0024–0.0026); candidate per-seed means
  span 0.00082 vs the parent's 0.00008.
- Mean plasticity rises further (0.691–0.701 per seed vs the parent's
  0.609–0.625): larger head gains feed larger residuals to the gated body.
  Plasticity is telemetry here, not a resource cost; the wall-clock per
  shard is unchanged (64.8/65.0/65.6 s vs the parent's 65.4/65.0/65.1 s,
  sequential).
- Screen-scale caution: negative results #21/#23 measured that this
  family's 60-task effects decay 3.5–4.7x at 200 tasks — and this cell is
  already ~2σ, so it is **not a confirmation candidate** (below the 0.005
  threshold by ~9x). This is a development screen (`development_only`,
  `scientific_promotion_allowed: false`).

## Axis status: measured to saturation

Ridge ladder on the gated chain at λ=1.0 (whole-run mean diffs vs the
immediate parent, 3/3 seeds each): 1.0 → 0.1 **+0.01040** (PR #3122);
0.1 → 0.03 **+0.00062** (this wave). The increment collapsed ~17x while the
mechanism shifted from "faster convergence raises the whole trajectory" to
"early gain trades against a slightly lower tail". The remaining named
ladder cell (0.01, P0 = 100·I) would extrapolate the same trade with less
headroom; screening it is not expected to clear noise, so the ridge axis is
declared **measured to saturation** at 0.03 unless a maintainer wants the
last cell for completeness. The gated chain's incumbent stands at
`..._r02` (0.94859) — a +0.00062 revision of the head — and the next
chain-level headroom named by negative result #22 (within-task convergence)
is now largely consumed on this axis; new movement needs a new mechanism,
not a smaller ridge.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
