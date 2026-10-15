# Post-collection resource-policy correction

This responds to [review 5529518230](https://github.com/SlopDotCash/asi/pull/3225#pullrequestreview-5529518230)
at frozen draft head `b19b48f1428c108ab4baf416df32027069a313e3`.
The original runner recorded resource settings without rejecting wrong values;
the original validator required cross-shard equality but did not require the
declared values. Consistently noncompliant inputs could pass.

The defect is reproduced with local copies of all six complete records: all
three thread settings changed to `8` and JAX preallocation changed to `true`
in every copy. The original validator exits 0. The new checked validator exits
2 on the same fixtures without writing a summary. These are synthetic test
fixtures, not benchmark observations, and their raw copies are not retained as
experimental evidence. [Regression evidence](regression-evidence.json).

## Correction and actual records

- [run.py](run.py) checks all four exact string values before loading the frozen
  runner or importing JAX. Missing, mistyped and wrong values fail closed with
  exit 2 before a life starts. It delegates to the unchanged, hash-pinned input
  runner; all original CLI arguments are forwarded.
- [validate.py](validate.py) checks the recorded settings of every comparison
  shard and both named excluded preflights before delegating to the unchanged,
  hash-pinned full validator. Cross-shard equality cannot substitute for
  compliance with the declared values.
- [resource_policy.py](resource_policy.py) shares the exact required settings:
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Values must be strings matching exactly.
- All six actual comparison records have those four values, as do both excluded
  one-event preflights. The checked validator accepts the six actual records and
  regenerates the retained summary byte for byte. [Audit](resource-audit.json).

The 39 [regression tests](test_resource_policy.py) cover absent/noncanonical/
non-string settings, real runner CLI rejection before dispatch/output, real
validation CLI rejection when all six fixtures or an excluded preflight share a
violation, the positive full retained-data path, and compliant runner `--help`
forwarding. No comparison or new benchmark seed was executed to test the
correction.

## Timing and scope

This check was added **after** comparison collection. It does not retroactively
make the original runner enforce the policy before execution. All 20 original
artifact files, including plan, runner, validator, results, summary and manifest,
remain byte-identical to result commit
`08b146636e8c53ab9568d955b65ea8c98f54be51`. This directory and its own manifest are
append-only supplemental evidence. The original manifest continues to cover its
original inventory. Nothing is relabeled as a fresh preregistration or run.

The new entry points check environment declarations and recorded settings, not
independently measured native thread counts, actual process/peak memory or
authenticated execution. The experiment remains permanently nonpromoting and
its decision remains `bounded_gamma_increase_inconclusive`. Consumed seeds,
frozen criteria, resources excluded from measurement and scientific status are
unchanged. New comparison work requires a separately frozen plan and fresh
seeds. An old-seed replay cannot be fresh evaluation.

## Reproduce

From the repository root, use the project venv. Set the exact four resource
environment values above before using the checked runner, for example its
`--help` or the original excluded preflight CLI. The original runner is retained
as historical input; the checked entry point is this directory's `run.py`.

```sh
python -m pytest outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15/resource-policy-audit/test_resource_policy.py -q -o addopts=""
python outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15/resource-policy-audit/validate.py --records outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15 --source-root . --preregistration outputs/reference_life_control_calibration/discounted-sarsa-horizon-2026-10-15/preregistration.json --out /new/path/summary.json --audit-report /new/path/resource-audit.json
```

The aggregate summary must match the original retained `summary.json` byte for
byte. Output paths must be new. Framework source, original artifacts and frozen
gates are unchanged; no learner performance or timing improvement is claimed.
