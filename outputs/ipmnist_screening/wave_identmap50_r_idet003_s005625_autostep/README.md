# Autostep per-weight body step sizes on `rls_head_resid_identmap50_r_idet003_s005625` (negative)

Preregistered in `PREREGISTRATION.md` (commit 5f5d8568, committed before any run).
Evidence tier: `development_screening_diagnostic`, nonpromoting.

## Arm
`rls_head_resid_identmap50_r_idet003_s005625_as1e2`: Autostep (Mahmood et al. 2012) per-weight
step sizes on the four body tensors, meta 1e-2, tau 1e4, alpha initialised at the parent step size.
At meta=0 the path is bit-exact with the parent (pinned in tests, mutation-checked).

## Tuning (seed 0, 3 tasks)
as1e3 0.82887, as1e2 0.83927 -> as1e2 selected.

## Screen (seeds 0-2, 60 tasks x 5000, step noise)
| seed | as1e2 | parent (local) | parent (stored, #3117) | control (local) | diff vs local parent |
|---|---|---|---|---|---|
| 0 | 0.92546 | 0.93429 | 0.93446 | 0.77748 | -0.00883 |
| 1 | 0.92513 | 0.93457 | 0.93467 | 0.77856 | -0.00944 |
| 2 | 0.92626 | 0.93467 | 0.93495 | 0.77724 | -0.00841 |
| mean | 0.92562 | 0.93451 | 0.93469 | 0.77776 | **-0.00889** |

Paired diff SD 0.00052; 95% t-CI (df=2) [-0.01018, -0.00761]; seeds improved 0/3.
Fails the +0.002 screen bar, so seeds 3-9 and holdout 10-19 were not run (per preregistration).

## Deviations
- Parent does not bit-reproduce in this environment (Python 3.13.5, kernel 6.12; stored shards: Python 3.14.4,
  kernel 7.0). Same offset at stored commit c3d02e3 and at main (see `diagnostics/`), so it is environmental,
  not from this patch. Local parent is -0.00018 vs stored. Pairing uses the locally measured parent; the gap
  (-0.0089) is ~50x this offset, so the conclusion is unaffected.
- Control reproduces bit-exactly (3-task and 60-task seeds 0-2: 0.77748/0.77856/0.77724).
- Interruption: the orchestrating session was interrupted while the control seeds 0-2 were running. The
  worker processes survived and completed normally (each log reports `done`, 60 tasks); no seed was re-run
  and no file was edited.

## Wall clock (4 workers, 1 thread each)
as1e2 ~490 s/seed, parent ~181-193 s/seed, control ~804-808 s/seed.
