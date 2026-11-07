# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_l2init` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_l2init` — the
gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac` 0.03, body step
0.0031640625, RLS head ridge 0.03 — the gated chain's best measured screen, PR #3123)
with decoupled L2-to-initialization on the four residual body tensors `w1/b1/w2/b2`
only (the issue-#14 endpoint at its frozen `decay_to_init=1.0`). The decay term of the
gated SGD step pulls toward each tensor's initial value at the unchanged
`weight_decay` rate instead of toward zero; the gate, step size, RLS recursion,
`w3/b3`, and every identifier constant are verbatim the parent.

This is the first measured L2-Init composition anywhere in the `rls_head` family: the
endpoint's own registry entry `rls_head_resid_l1_preset005_l2init` (PR #183) is
code-only, "no result artifact and does not authorize execution", and no other arm has
ever combined decay-toward-init with the RLS residual head or the identifier. The cell
is the chain's first new-mechanism-class probe after negative result #24 closed the
last named single-knob axis. Motivation: both recent axis closures localize their costs
in the late-life tail (#23 preconditioner, #24 reset), and decay-toward-init is the
published mechanism that targets exactly late-life plasticity loss (Kumar et al.);
the sigma0-family gated-L2-Init screens (PRs #3102/#3103) measured small uniformly
positive marginals on the no-RLS-head bodies.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes. Before the wave, a 3-task
`upgd_w_control` seed-0 reproduction shard (`repro_control_seed0_3task.json`) matched
the pinned control shard's per-task accuracies bit-exactly on the shared prefix
(0.69019997, 0.77059996, 0.78119999), verifying this environment against the pinned
baseline.

The candidate shards are schema v2 while the pinned reference shards are v1, and the
merge tool refuses to mix schemas — so the shards live outside the v1 `shards/*.json`
glob, the wave `summary.json` merges only the candidate shards (candidate named as its
own control), and the paired differences below are computed directly from the shards'
`per_task_accuracy` vectors. The parent reference is the retained
`wave_identmap50_r_idet003_s0031640625_r02/` shards from PR #3123's screen (read from
that PR's branch — not committed here). Both commits are main plus a registry entry
and tests only (the same runner code path, now verified rather than assumed), so the
parent pairing carries no version-offset caveat.

Mechanism note: the decay-target change touches the body's four tensors from the first
step, so the trajectories diverge from the parent from the first task (expected for a
body-mechanism change; task 0 is nearly unchanged — 0.84380/0.85380/0.85340 vs the
parent's 0.84440/0.85260/0.85260 — the from-scratch task is dominated by learning, not
by the decay target).

## Results

Mean online accuracy (60 tasks):

| seed | L2-Init (candidate) | r02 (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.93775 | 0.94896 | −0.01121 |
| 1 | 0.93752 | 0.94867 | −0.01115 |
| 2 | 0.93699 | 0.94814 | −0.01115 |
| mean | 0.93742 | 0.94859 | **−0.01117** |

- **Loses to the direct parent on 3/3 seeds** (mean −0.01117, paired stderr 0.000018,
  ~600σ). Valid rejection at screen resolution by an order of magnitude beyond any
  calibration question: no retuning, no second cell in this direction, no
  confirmation-seed touch.
- **The loss is uniform across the whole life, not a tail trade.** Per-seed diffs by
  task block — tasks 1–10 −0.00816/−0.00672/−0.00798, tasks 11–30
  −0.01244/−0.01255/−0.01242, tasks 31–60 −0.01177/−0.01214/−0.01179. L2-Init's
  premise (pay early diversity for late-life retention) does not appear: there is no
  early phase where the candidate is competitive and no late phase where it recovers.
- **Mean plasticity drops 0.700/0.691/0.701 → 0.597/0.589/0.594.** The decay-target
  swap replaces an ungated pull toward zero with an equally ungated pull toward the
  random task-0 draw; on this chain — identified inputs, gated small-step body, RLS
  head re-solved to the current features — that drag erodes the matured feature map
  every step, and the head cannot recover information the features no longer carry.
- No collapse: per-seed minima are the candidates' own task-0 values
  (0.84380/0.85380/0.85340); per-task spread tightens slightly (per-seed stderr
  0.00176–0.00186 vs the parent's 0.00203–0.00210); last-15-task slopes −0.00007
  vs the parent's −0.00011 — both plateau. Wall-clock per shard is unchanged
  (64.2–64.5 s vs the parent's 64.9–65.6 s, sequential).
- Screen-scale caution is moot at this effect size: the family's measured 60→200-task
  decay (3.5–4.7x, negative results #21/#23) cannot rescue a −0.011 loss.

## Axis status: L2-Init measured closed on the identified chain

The chain's named single-knob axes were already measured closed (negative result #24);
this wave measures the first new-mechanism-class candidate and rejects it at the
frozen endpoint strength. The uniform-drag failure mode gives the same closure shape
as #23: do not re-probe decay-toward-init on this chain with a smaller pull strength
or a gate-modulated pull without reasoning that changes the uniform erosion of matured
features. Recorded as negative result #25
(`docs/evidence/negative-results.md`).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`,
`evidence_class: development_screening_diagnostic`); it claims no promotion.
