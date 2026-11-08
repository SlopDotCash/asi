# Wave: `rls_head_resid_identmap50_r_idet003_s0031640625_r02_preset003` first screen

New arm (this PR): `rls_head_resid_identmap50_r_idet003_s0031640625_r02_preset003`
— the gated `idet003_s0031640625_r02` incumbent (`rls_head_resid_identmap50_r`
schedule 50/200/2000, utility gate kept, identifier trigger `ident_reset_frac`
0.03, body step 0.0031640625, RLS head ridge 0.03 — the gated chain's best
measured screen, PR #3123) with the head P-reset detector threshold
`rls_reset_frac` lowered 0.05 → 0.03. Every other constant is verbatim the
parent.

This is the first cell of the gated chain's **reset axis**, never swept on the
identified chain: the 0.05 default was calibrated at a 2.8x margin over the
within-task detector maximum of the PRE-identmap champion, whose inner
detector watched the un-remapped stream — on the identmap chain the inner
detector watches `x[remap]`, whose statistics step-change at the initial
shift AND at each match-induced remap refinement (50/200/2000 samples), so
the calibration premise is stale. The reset is level-triggered (every step
with `mean(shifted) >= reset_frac` holds P at `I/ridge`), so the lower
threshold lengthens each hold window. 0.03 is not arbitrary: it is the
trigger-axis measured optimum for this detector family on the raw stream
(PRs #3115/#3116/#3119). At `rls_lambda=1.0` the head P is nonincreasing
outside reset windows, so no wind-up collapse (negative results #10/#20) can
recur; the P-trace cap stays disabled exactly as in the parent.

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=step`, sequential fresh processes, one
commit for code and wave (`3286e2c2`, shards bind it). Before the wave, a
3-task `upgd_w_control` seed-0 reproduction shard
(`repro_control_seed0_3task.json`) matched the pinned control shard's
per-task accuracies bit-exactly on the shared prefix (0.69019997,
0.77059996, 0.78119999), verifying this environment against the pinned
baseline.

The candidate shards are schema v2 while the pinned reference shards are v1,
and the merge tool refuses to mix schemas — so the shards live outside the
v1 `shards/*.json` glob, the wave `summary.json` merges only the candidate
shards (candidate named as its own control), and the paired differences
below are computed directly from the shards' `per_task_accuracy` vectors.
The direct parent composition `rls_head_resid_identmap50_r_idet003_
s0031640625_r02` is the retained `wave_identmap50_r_idet003_s0031640625_
r02/` shards from PR #3123's screen (committed on this branch's parent);
the pinned `upgd_w_control` shards are the third reference.

Runner-version note: 3-task `rls_head_resid_identmap50_r` shards rerun on
this commit (`repro_parent_seed{0,1,2}_3task.json`) are **value-identical to
the r02 wave's repro shards** on every seed (both commits are main plus a
registry entry and tests only — the same runner code path, now verified
rather than assumed). The parent pairing column therefore carries no
version-offset caveat.

Mechanism note: the threshold change touches only when the level-triggered
reset holds, so the trajectories are bit-identical through task 0 on every
seed (first differing task index 1 on all three — no within-task detector
event reached even 0.03 through the whole first task, a direct observation
supporting the calibration margin on the inner stream) and diverge from the
first detected boundary onward.

## Results: rejected — loses on every seed

Mean online accuracy (60 tasks):

| seed | reset 0.03 (candidate) | r02 reset 0.05 (parent) | diff | control | diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.94840 | 0.94896 | −0.00056 | 0.77748 | +0.17092 |
| 1 | 0.94823 | 0.94867 | −0.00044 | 0.77856 | +0.16967 |
| 2 | 0.94797 | 0.94814 | −0.00017 | 0.77724 | +0.17073 |
| mean | 0.94820 | 0.94859 | **−0.00039** | 0.77776 | **+0.17044** |

- **Loses to the direct parent on 3/3 seeds** (mean −0.00039, paired stderr
  0.00016, ~2.4σ) — the gated chain's first negative arm. This is a valid
  rejection at screen resolution; no retuning, no second cell in this
  direction.
- **The loss concentrates late**, the mirror image of the ridge axis's win:
  per-seed mean diffs vs the parent by task block — tasks 1–10
  +0.00012/+0.00064/+0.00010, tasks 11–30 −0.00015/−0.00029/+0.00007,
  tasks 31–60 −0.00106/−0.00089/−0.00041 (last-15 −0.00137/−0.00021/
  −0.00065). The longer holds help briefly around each disturbance (the
  r02-style early-gain mechanism) but the re-inflated `33·I` head keeps
  taking large updates on partially misaligned inputs through the recovery
  tails, and the accumulated cost lands in the steady state.
- No collapse: per-seed minima are the candidates' own task-0 values
  (0.84440/0.85260/0.85260); late-window slopes +0.000121/−0.000073/
  −0.000202 vs the parent's −0.000071/−0.000053/−0.000206 — both
  plateau.
- Per-task spread is unchanged or slightly tighter (candidate per-seed
  stderr 0.0056–0.0058 vs the parent's 0.0056–0.0059); mean plasticity is
  slightly higher (0.700–0.705 vs 0.691–0.701) — telemetry, not a resource
  cost; wall-clock per shard is unchanged (64.7–65.0 s vs 64.7–65.6 s,
  sequential).
- Screen-scale caution applies in reverse here: negative 60-task effects in
  this family can decay at 200 tasks (ledger #21/#23), but with 3/3 seeds
  negative and the mechanism matching the r02 axis-status prediction, this
  is recorded as a **valid development rejection** (`development_only`,
  `scientific_promotion_allowed: false`).

## Axis status: measured, closed at 0.05

Reset ladder on the gated chain at λ=1.0, `rls_ridge_init=0.03` (whole-run
mean diffs vs the immediate parent, 3/3 seeds each): 0.05 → 0.03
**−0.00039** (this wave). The upward direction needs no run: every real
disturbance's above-threshold window peaks far above 0.08 (the initial
shift's shifted fraction rises toward ~1 and each match-induced remap
change re-scrambles most coordinates), so raising the threshold would only
truncate the recovery tails of genuine windows — shortening the holds that
the r02 mechanism just monetized — while task 0's bit-identity through
5,000 steps shows the within-task inner-stream maximum sits below 0.03
anyway. The axis is therefore **bracketed at 0.05**: longer holds lose,
shorter holds give back the r02 win, and the pre-identmap 2.8x-margin
calibration survives on the identified chain. Do not re-probe
`rls_reset_frac` on this chain without a mechanism that changes the
post-disturbance recovery behaviour. The gated chain's incumbent stands at
`..._r02` (0.94859); with the ridge axis saturated (PR #3123), the body
step closed at its monotone boundary (PR #3121), the trigger axis peaked
at 0.03 (PRs #3116/#3119), and the match schedule closed at 50/200/2000
(ledger #22), the chain's named single-knob axes are now all measured;
new movement needs a new mechanism class.

This is a **development-grade screening measurement**
(`development_only`, `scientific_promotion_allowed: false`). It claims
no promotion; confirmation would require the frozen full protocol under
a maintainer-run lane.
