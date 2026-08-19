# Gate-ablation r2 (2026-08-19)

Pre-registered n=10 paired screen resolving the issue #52 ambiguous band.
Pre-registration: issue #1937. Nonpromoting development screening; not a
performance or scientific result.

## Question

Is the RLS residual-body utility gate in the current screening champion
load-bearing? Issue #52 measured `rls_head_resid_l1_preset005_nogate` vs
`rls_head_resid_l1_preset005` at n=3 (+0.001558 ± 0.000230) inside the
ambiguous band and recommended a higher-n preregistration. This run executes
that preregistration.

## Protocol (frozen before the run, issue #1937)

- Arms: `rls_head_resid_l1_preset005` (control) vs
  `rls_head_resid_l1_preset005_nogate`, both remeasured on the same runner.
- Screen seeds 0-9, n=10; evaluation seeds 10-19 untouched.
- 60 tasks x 5000 steps, step noise, default protocol.
- Decision rule:
  - win: paired mean diff > +0.002, every seed positive, diff > 3x stderr
    -> 200-task confirmation + held-out seeds.
  - ambiguous: +0.0015 ... +0.002 -> report, no escalation, no claim.
  - tie: < +0.0015 -> report.
  - loss: < -0.0015 -> report and stop.

## Result

| Arm | mean | stderr | per-shard wall (s) |
|---|---|---|---|
| `rls_head_resid_l1_preset005` (remeasured) | 0.869634 | 0.000250 | ~375 |
| `rls_head_resid_l1_preset005_nogate` | 0.871345 | 0.000250 | ~129 |

Paired diff: +0.001712 ± 0.000174 (stderr_diff). Per-seed diffs all ten
positive, range +0.001090 ... +0.002610. Ratio to stderr: 9.84x.

Verdict by the frozen rule: **ambiguous**. The diff clears the tie band and
every seed improves, but +0.001712 < +0.002, so no escalation to the 200-task
confirmation and no evaluation-seed touch. No threshold was moved after the
result.

## Sanity

Remeasured control seeds 0-2 (0.869227 / 0.869257 / 0.869027) track the
archived n=3 seeds (0.869137 / 0.869770 / 0.869227) within the run-to-run
jitter class (~0.0003), so the harness is not drifting on this runner.

## Cost

Control ~375 s/shard, nogate ~129 s/shard on CPU (8 cores, OMP_NUM_THREADS=1
per worker, 6 parallel). Gate removal is 2.9x cheaper at equal protocol.

## Recorded fact

At 60 tasks, gate removal is consistently not-worse and 2.9x cheaper, but the
effect sits below the campaign's +0.002 escalation bar. No claim either way.
Ledger entry 16.

## Artifacts

- `shards/rls_head_resid_l1_preset005_seed{0..9}.json` (10, v2, provenance-bound)
- `shards/rls_head_resid_l1_preset005_nogate_seed{0..9}.json` (10, v2)
- `summary.json` (schema `alberta.ipmnist_screening.summary.v2`, control
  `rls_head_resid_l1_preset005`, 20 shards)
- `worker_r2.sh`, `jobs_r2.txt` (wave launcher and job list; per-shard logs in
  `logs/` are transient and not checked in)
- Source: HEAD `cc877f78566675214e2f356bd797f0f3c5ec1bb0`