# Wave: first gradual-transition screen (`gradual64_r1`)

The first executed gradual wave, under the gradual-change protocol added by
issue #1569 / PR #3045 and pre-declared in `RUNBOOK.md` ("any future wave is
development-only and permanently nonpromoting"). Question: does the RLS-head
chain's advantage over the pinned `upgd_w` control survive **gradual input
transitions** — a 64-step linear ramp from the previous task's permutation to
the current one — where the abrupt-step advantage was measured at +0.134
(`rls_head_resid_identmap50_r` 0.91142 vs control 0.77776 on the step
protocol)?

Five configs × 3 seeds = 15 schema-v3 shards at main `415e7d22` (the current
main merge of PR #3100; no code changes on this branch): the chain families
`rls_head_resid_identmap50_r`, `rls_head_resid_identmap200_r`,
`rls_head_resid_l1_preset005`, `sigma0_shiftnorm_d099`, and the pinned
`upgd_w_control`. `summary.json` merges all 15 shards (shared gradual window
64, schema `alberta.ipmnist_screening.summary.v3`).

## Protocol

Standard screening scale: 60 tasks × 5,000 steps, seeds 0–2,
`OMP_NUM_THREADS=1`, `noise_mode=gradual`, `noise_gradual_steps=64`,
sequential fresh processes. The ramp draws no RNG keys, so gradual runs are
exactly reproducible given the seed (RUNBOOK).

Environment verification, both in one check: the 3-task
`repro_control_gradualw1_seed0_3task.json` (gradual with
`noise_gradual_steps=1`, v3 shard) is **bitwise identical** to
`repro_control_step_seed0_3task.json` (abrupt step, v2 shard) on per-task
accuracy *and* loss (0.69019997, 0.77059996, 0.78119999), which also matches
the pinned control shard's prefix bit-exactly — window 1 collapses to the
step protocol as pinned by `TestGradualConfirmation`, and this environment
reproduces the pinned baseline.

## Results

Mean online accuracy (60 tasks):

| config | step reference (retained waves) | gradual 64 (this wave, per seed) | mean | diff vs control (paired) |
| --- | --- | --- | --- | --- |
| `rls_head_resid_identmap50_r` | 0.91142 | 0.86852 / 0.86796 / 0.86914 | 0.86854 | **+0.08897** |
| `rls_head_resid_identmap200_r` | 0.90470 | 0.86852 / 0.86784 / 0.86914 | 0.86850 | **+0.08893** |
| `rls_head_resid_l1_preset005` | 0.86939 | 0.86852 / 0.86886 / 0.86914 | 0.86884 | **+0.08927** |
| `sigma0_shiftnorm_d099` | 0.86395 | 0.86846 / 0.86817 / 0.86843 | 0.86835 | **+0.08878** |
| `upgd_w_control` | 0.77776 | 0.77855 / 0.78037 / 0.77979 | 0.77957 | — |

Step references are the retained step-protocol 60-task means
(`identmap_star_r1` / `replication_r1` summaries and the pinned control
shards) from earlier commits; the documented cross-version drift bound is
≤0.001 per task, two-plus orders below every number compared here.

- **The chain's advantage survives gradual change on 3/3 seeds for every
  family member.** Paired diffs vs the control: identmap50_r
  +0.08997/+0.08759/+0.08934 (mean +0.08897, paired stderr 0.00071),
  identmap200_r +0.08893 (0.00075), l1_preset005 +0.08927 (0.00043),
  sigma0_shiftnorm_d099 +0.08878 (0.00061) — all ≥118σ as paired screens.
- **About two-thirds of the abrupt-step advantage is retained** (chain head
  +0.134 on step → ~+0.089 gradual). The ramp also helps the control itself
  (0.77957 vs its step value 0.77776) while the chain head drops
  (0.91142 → 0.86854): continuous transitions cost the exact-fit head much
  more than they cost the control.
- **The chain members' fine distinctions collapse under the ramp.** On seeds
  0 and 2 the three RLS-head chain arms are **bit-identical across all 60
  tasks**; on seed 1 they coincide until task 46 and diverge afterwards. The
  ident-schedule and preset differences never separated on 2/3 seeds — the
  gradual protocol exercises the shared gated-recursion path, not the
  per-arm mechanisms (the reset/trigger differences apparently rarely or
  never fire mid-ramp). The wave's arm-vs-arm ordering (spread 0.00049,
  all within 1.1 paired stderr of zero) is therefore noise, not a ranking;
  the real content is chain-vs-control.
- Windowed paired diffs vs the control grow over the run for the RLS-head
  arms (identmap50_r: tasks 1–20 ~+0.081, 21–40 ~+0.091, 41–60 ~+0.095 per
  seed) and stay flat for sigma0_shiftnorm_d099 (~+0.088 throughout); no
  late reversal anywhere (last-15 slopes −0.00020 to +0.00064 per task).
- Mean plasticity (telemetry only): chain arms 0.429–0.443, sigma0 0.298,
  control 0.500 — all below their step-protocol values (chain 0.61–0.70).
  Wall-clock per shard: candidates 58.4–62.3 s, control 263.2–265.9 s, the
  same control-side utility-update asymmetry as the step waves.
- Scope: the current step-noise incumbent
  `rls_head_resid_identmap50_r_idet003_s0031640625_r02` (PR #3123) and the
  gated idet003 chain are **not** in this wave; their gradual behavior is
  unmeasured. This wave covers the four registered chain families as of
  main #3100.

## Status

Development screen (`development_only`,
`scientific_promotion_allowed: false`), permanently nonpromoting. Per the
RUNBOOK, gradual shards never merge with exact shards, never pass proxy
validation, and never produce a mechanism receipt; the wave is appended to
the campaign outputs, not promoted anywhere.
