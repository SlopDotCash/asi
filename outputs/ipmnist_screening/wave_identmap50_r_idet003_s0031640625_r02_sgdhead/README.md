# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_sgdhead` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_sgdhead` — the
gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r` schedule
50/200/2000, utility gate kept, identifier trigger `ident_reset_frac` 0.03, body step
0.0031640625, RLS head ridge 0.03 — the gated chain's best measured screen, PR #3123)
with the Sherman-Morrison head update replaced by plain decayed SGD on the same one-hot
squared loss. The residual `err = y − ŷ` enters the head update in the RLS gain's sign,
so both rules descend the same loss; the body still trains on the head's own pre-update
residual gradient, so only the head's estimation rule changes. The detector-driven P
reset and trace cap compile out (P is never touched), and `rls_ridge_init` stays 0.03
verbatim though it no longer affects dynamics.

This is the matched-gradient control that ipmnist-theory §6.3 (open hypothesis 3,
error-signal identity) and §6.6 (open hypothesis 6, protocol-pure translation) ask for:
if the body lift survives an SGD head, the residual's geometry — not the stable head's
rapid fit — carries it and the O(d²) RLS head is removable; if it collapses, the
convergence-shortfall account is confirmed on the identified chain. No `rls_head` arm
has ever run a non-RLS head.

## Head-step calibration (seed-0, 3-task probes — not retained shards)

The registration draft ported the parallel-head family's constants
(`rls_head_sgd_step` 0.01, `weight_decay` 0.01). Those constants are mis-scaled for
this head's input (the body's hidden features, not EMA-normalized inputs), and the
first calibration probes also caught a pre-publication sign defect in the draft: the
residual entered as `− step · outer(phi, err)`, i.e. gradient ascent on the loss —
3-task probes measured chance-level accuracy (~0.09, below the 0.10 floor the head
never beats) with non-finite loss at steps ≥ 0.003. The sign was fixed
(`+ step · outer(phi, err)`, the RLS gain's sign) and the registration amended before
any 60-task shard was retained; every number below is post-fix.

Post-fix bracket (3 tasks, seed 0, mean online accuracy; all cells finite):

| step | 3e-5 | 1e-4 | 3e-4 | 0.001 | 0.003 | 0.005 | 0.01 | 0.02 | 0.05 | **0.1** | 0.2 | 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc | 0.1083 | 0.1084 | 0.1094 | 0.1114 | 0.1159 | 0.1227 | 0.1364 | 0.1553 | 0.1709 | **0.1691** | 0.1577 | 0.1257 |

Accuracy rises monotonically to a flat optimum at 0.05–0.1 and declines at 0.2/0.4;
the registered endpoint takes the bracket's best region cell **0.1** with
`weight_decay` 0.01 verbatim. The conclusion below is invariant across the whole
stable bracket: even the best 3-task cell (0.171) is nowhere near the parent's 3-task
prefix (0.844–0.853).

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2, `OMP_NUM_THREADS=1`,
`noise_mode=step`, sequential fresh processes. Before the wave:

- a 3-task `upgd_w_control` seed-0 reproduction shard
  (`repro_control_seed0_3task.json`) matched the pinned control shard's per-task
  accuracies bit-exactly on the shared prefix (0.69019997, 0.77059996, 0.78119999);
- 3-task parent reproduction shards (`repro_parent_seed{0,1,2}_3task.json`) matched
  the retained r02 parent shards' 3-task prefixes **bit-exactly on all three seeds**
  — the registration adds a registry entry plus a build-time branch only, and the
  flag-off pin holds bitwise, so the parent pairing carries no version-offset caveat.

The wave `summary.json` merges only the candidate shards (candidate named as its own
control), and the paired differences below are computed directly from the shards'
`per_task_accuracy` vectors. The parent reference is the retained
`wave_identmap50_r_idet003_s0031640625_r02/` shards from PR #3123's screen.

## Results

Mean online accuracy (60 tasks):

| seed | SGD head 0.1 (candidate) | r02 (parent) | diff |
| --- | --- | --- | --- |
| 0 | 0.44358 | 0.94896 | −0.50538 |
| 1 | 0.45230 | 0.94867 | −0.49637 |
| 2 | 0.45426 | 0.94814 | −0.49388 |
| mean | 0.45005 | 0.94859 | **−0.49854** (paired stderr 0.00349, 3/3 seeds negative) |

- **Catastrophic, decisive rejection.** The candidate lands **0.33 below even the
  plain `upgd_w_control`** (0.77776), not merely below the parent. The deficit is a
  lower asymptote, not a transient: per-10-task blocks sit at 0.29–0.33 (1–10),
  0.47–0.48 (11–30), and 0.475–0.48 (31–60) vs the parent's 0.925 → 0.954.
- **The within-task fit itself collapses.** Task 0 — before any detected boundary, so
  identical inputs and schedule — measures 0.1174/0.1178/0.1248 vs the parent's
  0.8444/0.8526/0.8526. Plain SGD at its calibrated-best step cannot match exact
  second-order (Sherman-Morrison) within-task convergence in 5,000 steps, and the body
  then trains on the resulting poor residual: the stable, non-overflowing analogue of
  negative result #11's body/head feedback failure. Mean plasticity drops
  0.700/0.691/0.701 → 0.213/0.216/0.216.
- **No instability artifact.** Every recorded loss is finite, per-seed minima are the
  candidates' own task-0 values, and wall-clock per shard is unchanged (62.5–63.4 s vs
  the parent's 64.8–65.6 s, sequential). Late-15-task slopes (−0.000298/+0.000020/
  −0.000259) sit at the same plateau shape as the parent's.

## Conclusions and axis status

1. **Convergence-shortfall account confirmed on the identified chain**: the stable
   head's rapid exact fit — not the residual's geometry alone — carries the body lift
   (theory §6.3's first separation cell).
2. **The O(d²) RLS head is not removable** at this horizon by an O(d) SGD head at its
   best calibrated step (theory §6.6 answered in the negative for plain SGD).
3. Do not re-probe plain-SGD-head replacements on this chain without a mechanism that
   changes within-task convergence rate (per-direction adaptive or second-order head),
   not merely a step-size retune — the bracket's entire stable range is measured far
   below the parent. Recorded as negative result #26 in
   `docs/evidence/negative-results.md`.

The incumbent stands at `..._idet003_s0031640625_r02` (0.94859); this arm is a
mechanism-identity control, not a hillclimb step, and claims no promotion.

## Validation

- **Reduction pins (new `TestRLSHeadIdentMapIdet003S0031640625R02Sgdhead`, 6 tests):**
  `rls_head_sgd=0.0` (explicit off) follows the registered r02 parent **bit-for-bit**;
  zero step is the documented frozen-head degenerate (wout stays 0, accuracy = class-0
  indicator, loss 0.5/step); the first update equals the exact decayed SGD residual
  step with the residual in the RLS gain's sign (3-ulp tolerance: the scalar head step
  enters XLA's fused multiply chain, so changing the registered constant can shift
  individual products by 1–2 ulp; the flag-off and registry pins stay bitwise);
  registry shape (parent keys + `rls_head_sgd`, `rls_head_sgd_step`); composition
  guards reject untested envelopes (non-residual body, ungated, whitened, Newton,
  negative step). `test_registry_arms` updated to enumerate the new arm.
- **Focused checks:** `ruff check` clean on both touched files; `mypy
  alberta_framework/benchmarks/ipmnist_screening.py` → no issues;
  `pytest tests/test_ipmnist_screening.py -k "RLSHead or registry_arms"` → 58 passed.
- **Environment identity:** the control and all three parent repro shards reproduce
  the pinned references bit-exactly (see Protocol).
- **Retained artifacts:**
  `outputs/ipmnist_screening/wave_identmap50_r_idet003_s0031640625_r02_sgdhead/`
  (README, 3 candidate shards schema v2 binding commit `ec23f65a`, control repro
  shard, 3 parent repro shards, `summary.json`, `worktree_clean: true`, per-shard
  logs).

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`,
`evidence_class: development_screening_diagnostic`). It claims no promotion;
confirmation would require the frozen full protocol under a maintainer-run lane.

AI provider/model: zai / glm-5.3-flash
Client / agent tooling: pi
Contribution skill revision: N/A - ordinary GitHub contribution without optional run evidence
Attribution status: self-reported
— [contributor]
<!-- slop-contribution-attribution:v1 {"provider":"zai","model":"glm-5.3-flash","client":"pi","skill_revision":"N/A - ordinary GitHub contribution without optional run evidence"} -->
