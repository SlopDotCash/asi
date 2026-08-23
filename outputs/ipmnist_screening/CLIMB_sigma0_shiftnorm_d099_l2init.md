# ipmnist_screening: sigma0_shiftnorm_d099_l2init (new arm)

New arm composing l2-init (decoupled decay toward the initial weights,
Kumar et al.) on top of the sigma0 shift-normalizer family. The adaptive-norm
sigma0 learner previously applied decoupled decay toward **zero**; the new
`flag_decay_to_init` path decays toward the initial weights instead, and this
arm sets it on the `sigma0_shiftnorm_d099` configuration (the 60-task screen
champion in `FINAL_REPORT.md`).

## Result (60-task screen, 3 seeds each, seed-controlled)

| Arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| `upgd_w_control` (baseline) | 0.7775 | 0.7786 | 0.7772 | **0.7778** |
| `sigma0_shiftnorm_d099` (champion) | — | — | — | **0.86396** (FINAL_REPORT) |
| `sigma0_shiftnorm_d099_l2init` (new) | 0.8732 | 0.8735 | 0.8723 | **0.8730** |

- Improvement vs the recorded 60-task champion: **+0.0091 (+1.05%)**
- Improvement vs the same-process baseline (control, 3 seeds): **+0.0953**
- All three seeds beat the previous best; the gain is consistent, not a
  single-seed outlier.

## Mechanism

`UPGDAdaptiveNormState` gains an optional `init_params` carrier; when
`flag_decay_to_init != 0` the decoupled decay term becomes
`p' = p·(1 - lr·wd) + (lr·wd)·p0` (decay toward the initialization) instead of
`p·(1 - lr·wd)` (decay toward zero). `flag_decay_to_init = 0` keeps the
previous path bit-identical (covered by the `test_ipmnist_l2init_composition`
suite).

## Evidence

- `outputs/ipmnist_screening/bench_runs/upgd_w_control_seed{0,1,2}.json`
- `outputs/ipmnist_screening/bench_runs/sigma0_shiftnorm_d099_l2init_seed{0,1,2}.json`
- Runtime provenance is embedded in each JSON (`source_provenance`,
  `environment`, `created_unix`).

## Scope

- `alberta_framework/benchmarks/ipmnist_screening.py`: optional
  `init_params` state carrier + `decay_to_init` path + new registered arm.
- `tests/test_ipmnist_l2init_composition.py`: 5 regression cases (registry
  contract, flag-on/off state semantics, decay arithmetic).
