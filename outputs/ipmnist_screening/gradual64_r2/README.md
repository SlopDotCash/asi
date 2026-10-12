# Wave: second gradual-transition screen (`gradual64_r2`)

The second executed gradual wave, under the same gradual-change protocol as
`gradual64_r1` (issue #1569 / PR #3045; development-only and permanently
nonpromoting). The first wave left an explicit gap: the current step-noise
incumbent `rls_head_resid_identmap50_r_idet003_s0031640625_r02` (PR #3123)
was not in `gradual64_r1`, so its gradual behavior was unmeasured. This wave
closes that gap with one question: does the improved head incumbent keep —
and grow — the chain's gradual advantage over the pinned `upgd_w` control?

Two configs × 3 seeds = 6 schema-v3 shards at `8b08aba1` (the retained r02
wave commit; the candidate config was registered by PR #3123 at `25038b14`,
and the two commits have an identical relevant-source digest, the delta being
outputs only): the incumbent `rls_head_resid_identmap50_r_idet003_s0031640625_r02`
and the pinned `upgd_w_control`. `summary.json` merges all 6 shards (shared
gradual window 64, schema `alberta.ipmnist_screening.summary.v3`).

## Protocol

Identical to `gradual64_r1`: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=gradual`, `noise_gradual_steps=64`,
sequential fresh processes. The ramp draws no RNG keys, so gradual runs are
exactly reproducible given the seed (RUNBOOK).

Environment verification, all bitwise in this worktree:

- The 3-task `repro_control_gradualw1_seed0_3task.json` (gradual with
  `noise_gradual_steps=1`, v3 shard) is bitwise identical to
  `repro_control_step_seed0_3task.json` (abrupt step, v2 shard) on per-task
  accuracy *and* loss (0.69019997, 0.77059996, 0.78119999) — window 1
  collapses to the step protocol as pinned by `TestGradualConfirmation`,
  reproducing r1's pinned values at this commit.
- The incumbent 3-task step repros (`repro_incumbent_step_seed{0,1,2}_3task.json`)
  are bitwise identical (accuracy and loss) to the first three tasks of the
  r02 wave's 60-task step shards at `25038b14` on every seed — no
  version-offset caveat between the registration and retention commits.
- The control gradual64 shards are bitwise identical (all 60 task
  accuracies) to r1's control shards, which were run at main `415e7d22` —
  cross-commit consistency from the deterministic ramp.

## Results

Mean online accuracy (60 tasks):

| config | step reference | gradual 64 (this wave, per seed) | mean | diff vs control (paired) |
| --- | --- | --- | --- | --- |
| `rls_head_resid_identmap50_r_idet003_s0031640625_r02` | 0.94859 (PR #3123) | 0.87999 / 0.89439 / 0.89733 | 0.89057 | **+0.11100** |
| `upgd_w_control` | 0.77776 (pinned) | 0.77855 / 0.78037 / 0.77979 | 0.77957 | — |

- **The incumbent's advantage survives gradual change on 3/3 seeds and
  grows.** Paired diffs vs the control: +0.10144 / +0.11402 / +0.11753
  (mean +0.11100, paired stderr 0.00489, ~22.7σ) — vs r1's chain-head
  +0.08897. The head axis's step gain (+0.03717 over the r1 head on step)
  keeps about 59% of its value under the ramp (+0.02203 gradual), so the
  gradual advantage rises from +0.089 to +0.111.
- **Retention profile is unchanged in shape.** The candidate keeps 65% of
  its step advantage (0.11100 of 0.17083) vs r1 head's 67%; the exact-fit
  head still pays more under transitions than the control does (step→gradual
  −0.05802 vs r1 head's −0.04288; control +0.00181).
- **No late reversal anywhere.** Windowed paired diffs stay positive on
  every seed and window (tasks 1–20 +0.094 to +0.125, 21–40 +0.105 to
  +0.115, 41–60 +0.096 to +0.126); last-15 slope +0.00009/task for the
  candidate vs +0.00003 for the control.
- Mean plasticity (telemetry only): candidate 0.619–0.662, control 0.500.
  Wall-clock per shard: candidate 67.2–72.2 s, control 292.4–305.4 s — the
  same control-side utility-update asymmetry as every earlier wave.

## Status

Development screen (`development_only`,
`scientific_promotion_allowed: false`), permanently nonpromoting. The
summary's `confirmation_candidate` flag is the merge tool's protocol-generic
0.005 bar and carries no promotion weight for gradual lanes: per the
RUNBOOK, gradual shards never merge with exact shards, never pass proxy
validation, and never produce a mechanism receipt. No confirmation run, no
evaluation seeds; the wave is appended to the campaign outputs, not promoted
anywhere.
