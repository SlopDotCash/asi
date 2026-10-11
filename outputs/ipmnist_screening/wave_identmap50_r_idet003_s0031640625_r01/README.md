# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r01` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r01` — the
gated `idet003_s0031640625` composition (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, residual body step size 0.0031640625 — the gated chain's best measured
screen, PR #3120) with the RLS head's initial/reset ridge `rls_ridge_init`
lowered 1.0 → 0.1. Every other constant is verbatim the parent.

This is the gated chain's first cell on a **new head axis**. Both bracket
axes are measured to their peaks and closed (identifier trigger 0.03, PR
#3119; body step 0.0031640625, PR #3121), so the next movement needed a new
axis or mechanism. The ridge is the head's convergence-speed knob: the head
P matrix initializes and resets to `I / rls_ridge_init`, so ridge 0.1 makes
P0 = 10·I and raises earliest and post-reset gains ~10x, letting the head
converge faster to each task's least-squares solution — the within-task
convergence residue that negative result #22 names as the family's remaining
headroom ("the 0.016 to the 0.933 asymptote is within-task convergence,
untouched by identification").

The cell is genuinely unmeasured, not a retry of negative result #10. The
wave-3 ridge star (1.0/0.3/0.1/0.03/0.01) ran **only at `rls_lambda=0.999`**
(`summary_rls_head.json`); its small-ridge collapse at 60 tasks (0.81348 at
ridge 0.1) is confounded with forgetting wind-up — P grows as (1/λ)^t along
unexcited directions, and a ridge-boosted P0 amplifies exactly that growth.
At λ=1.0 the Sherman-Morrison P matrix is nonincreasing PSD
(P_t ≤ P0 = 10·I, trace ≤ n/ridge = 1500, far below the 1e4 cap), so the
#10 collapse mechanism is structurally absent at this cell. Negative #20's
closure covers forgetting (`λ < 1`), which this arm does not touch.

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
direct parent composition `rls_head_resid_identmap50_r_idet003_s0031640625`
(retained `wave_identmap50_r_idet003_s0031640625/` shards from PR #3120's
screen, read from that PR's branch — not committed here), the gated idet003
peak at step 0.01 (retained `wave_identmap50_r_idet003/` shards from PR
#3115's screen, read from that PR's branch — not committed here), and the
pinned `upgd_w_control` shards.

Runner-version note: 3-task `rls_head_resid_identmap50_r` shards rerun on
this commit (`repro_parent_seed{0,1,2}_3task.json`) are **value-identical
to the s0031640625 wave's repro shards** on every seed (both commits are
main plus a registry entry only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat; the idet003 column retains the same ≤0.001/task
cross-version offset versus the retained confirmation shards documented
in the idet002 wave (PR #3116), an order below the measured effects.

Mechanism note: the ridge change touches only the head's P initialization
and P reset, so — unlike a body-step change — task 0 (before any detected
boundary) is expected to stay close to the parent until the first P reset
or fast head convergence shifts the body's error signal; in practice
the trajectories diverge from the parent from the first task (seed 0
task 0: 0.81919998 vs 0.68059999 — the faster head improves the very
first task), so the whole trajectory shifts, expected for a head axis
change.

## Results

Mean online accuracy (60 tasks):

| seed | ridge 0.1 (candidate) | s0031640625 (parent) | diff | idet003 (step 0.01) | diff | control | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.94802 | 0.93721 | +0.01081 | 0.92617 | +0.02185 | 0.77748 | +0.17054 |
| 1 | 0.94797 | 0.93746 | +0.01051 | 0.92648 | +0.02149 | 0.77856 | +0.16941 |
| 2 | 0.94794 | 0.93805 | +0.00989 | 0.92633 | +0.02160 | 0.77724 | +0.17070 |
| mean | 0.94797 | 0.93757 | **+0.01040** | 0.92633 | **+0.02165** | 0.77776 | **+0.17022** |

- **Beats the direct parent on 3/3 seeds** (mean +0.01040, paired stderr
  0.00027). This is the gated chain's largest single-cell increment since
  the identifier itself (+0.01124), ~11x the last step bracket (+0.00096,
  PR #3120), and it **exceeds the 0.005 full-protocol confirmation
  threshold** recorded in the summary.
- The mechanism-level signature is extremely tight across seeds: the three
  candidate per-seed means span 0.00008 (0.94794/0.94797/0.94802), versus
  the parent's 0.00084 span. The per-task spread widens (candidate
  per-seed stderr 0.0024–0.0026 vs the parent's 0.00025) — faster head
  gains raise per-task variance — but the arm-level effect reproduces
  almost exactly.
- The candidate's mean plasticity roughly doubles (0.608–0.625 per seed vs
  the parent's 0.350–0.355): larger head gains feed larger residuals to
  the gated body. Plasticity is telemetry here, not a resource cost; the
  wall-clock per shard is unchanged (65.4/65.1/65.1 s vs the parent's
  64.9 s scale, sequential).
- Late-window check (last 15 tasks): candidate slopes −0.000018/+0.000031/
  −0.000274 per task vs the parent's +0.000236/+0.000189/−0.000023 — the
  faster head converges to its asymptote and plateaus (seed 2 mildly
  negative). No within-run collapse: every seed's full-horizon mean stays
  +0.0099 above the parent.
- Screen-scale caution: this is a 60-task × 3 seed development screen
  (`development_only`, `scientific_promotion_allowed: false`); negative
  results #21/#23 measured that this family's 60-task effects can decay
  3.5–4.7x at 200 tasks, so the confirmation call belongs to the frozen
  full protocol under a maintainer-run lane, not to this screen. The
  candidate is a confirmation candidate by the summary's own threshold
  bookkeeping.

## Next cell

The ridge axis is now open in the improving direction: the named unmeasured
next cell is ridge 0.03 (= 0.1 × 0.3, the wave-3 star's ladder spacing,
P0 ≈ 33·I), with the wave-3 caveat that per-task variance widens as the
ridge falls and the late-window plateau needs watching. The upward
direction (ridge 0.3) is the fallback bracket if 0.03 loses.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
