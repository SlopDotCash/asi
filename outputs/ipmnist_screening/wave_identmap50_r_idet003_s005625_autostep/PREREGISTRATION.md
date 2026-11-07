# Preregistration: Autostep per-weight body step sizes on `rls_head_resid_identmap50_r_idet003_s005625`

Committed before any benchmark shard for this arm was run. Local development
screen; evidence tier `development_screening_diagnostic, nonpromoting`.

## Hypothesis

The best merged arm, `rls_head_resid_identmap50_r_idet003_s005625` (0.93469
mean over seeds 0-2, PR #3117), uses one global body step size (0.005625).
The step-size axis was hand-tuned down to that value on both chains (PRs
#3109-#3111 and #3117). Autostep (Mahmood, Sutton, Degris & Pilarski 2012)
adapts a per-weight step size online. Hypothesis: replacing the scalar
body step with Autostep per-weight step sizes, initialised at the parent's
0.005625, raises 60-task mean online accuracy by at least +0.002 paired
against the parent.

## Single changed variable

Only the body step-size rule changes: the gated residual body update's
scalar `step_size` becomes a per-weight `alpha` adapted by Autostep
(normaliser `v` with time constant `tau`, effective-step bound
`M = max(sum(alpha * z^2), 1)` over the four body tensors, `z` = gated body
gradient, as in the repo's `upgd_autostep_update`). The utility EMA, the
sigmoid gate, the RLS head, the identifier and every other constant stay
verbatim the parent. `alpha` starts at 0.005625 and `tau = 1e4`. Meta step
size `body_as_meta = 0` delegates at build time to the parent's factory,
pinned bit-exactly by a test.

## Arms

- `rls_head_resid_identmap50_r_idet003_s005625_as1e3` (meta 1e-3)
- `rls_head_resid_identmap50_r_idet003_s005625_as1e2` (meta 1e-2)

## Seed plan

1. Environment check: 3-task seed-0 `upgd_w_control` and parent shards must
   match the pinned per-task values (control 0.69019997, 0.77059996,
   0.78119999; parent = `repro_parent_seed0_3task.json`).
2. Tuning (seed 0 only, 3 tasks x 5000 steps): run both arms. Pick the meta
   rate with the higher 3-task mean. Ties go to 1e-3. If neither beats the
   parent's 3-task seed-0 mean, still screen the better one (60 tasks are
   needed to judge) and record the tuning result.
3. Screen: the chosen arm on seeds 0-2, 60 tasks x 5000 steps,
   `noise_mode=step`. The parent is re-run locally on seeds 0-2 and must
   match the stored shards.
4. Extension: only if the screen's mean paired difference vs parent is
   >= +0.002, run the chosen arm and the parent on seeds 3-9 (60 tasks).
   Control seeds 3-9 run as well, so all columns are measured in this
   environment.
5. Seeds 10-19 are held out and not run in this work.

## Metrics

Primary: mean online accuracy over 60 tasks per seed; paired difference
vs parent per seed; mean, sample SD and 95% t-interval of the paired
difference; count of seeds improved. Secondary: difference vs
`upgd_w_control`; wall-clock seconds per shard; summary of final per-weight
alpha (if recorded).

## Decision and stop rules

- Bar: mean paired difference vs parent >= +0.002 on seeds 0-2 to extend.
- After seeds 0-9: report as positive only if the mean paired difference
  >= +0.002 and the 95% CI lower bound > 0. Otherwise report as negative or
  inconclusive.
- Stop early with a negative result if any shard is non-finite or diverges.
- No further meta-rate tuning after this preregistration. The result is
  packaged whatever its sign.
- Nothing is announced publicly. 60-task effects have shrunk 3.5-4.7x at
  200 tasks before (ledger #21, #23), so a pass is a screen, not a
  confirmation.

## Resource plan

At most 4 workers, one thread each (`OMP_NUM_THREADS=1`,
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`). Check `free -g` and `cargo-build-sbf`
before each wave and during it.
