# V6 — Is recurrence unexploited? Measuring the headroom for direction D

Pre-registration: [#1875](https://github.com/elizaOS/asi/issues/1875).
Measurement artifact: `measurement.json`. Development diagnostic, permanently
nonpromoting. Micro suite only: touches no IPMNIST lane, no pinned artifact, no
registered source, no promotion seed.

## Result: H0 is rejected

The preregistration predicted H0 (no existing arm exploits recurrence). The
paired measurement rejects it: **every one of the six `LADDER_ARMS` scores
materially better on M4 (recurrence, pool = 5) than on M1 (fresh permutation
per regime), with all three seeds positive in every arm.**

| arm | M1 mean | M4 mean | paired D (M4 - M1) | per-seed D | all seeds > 0 |
|---|---|---|---|---|---|
| gated_norm | 0.691108 | 0.738266 | **+0.047158** | +0.051644, +0.042554, +0.047276 | yes |
| sgd_norm | 0.680883 | 0.727718 | **+0.046835** | +0.051410, +0.042210, +0.046884 | yes |
| sgd_raw | 0.276952 | 0.668528 | **+0.391576** | +0.378448, +0.416548, +0.379732 | yes |
| naive_bayes | 0.605175 | 0.652579 | **+0.047404** | +0.047918, +0.043608, +0.050686 | yes |
| adamw | 0.228189 | 0.578296 | **+0.350107** | +0.340908, +0.373194, +0.336218 | yes |
| upgd_raw | 0.280117 | 0.374178 | **+0.094061** | +0.106932, +0.105964, +0.069286 | yes |

Bayes reference (micro suite): mean 0.984312 (seeds 0-2, 200k samples).
Best arm on M4: gated_norm 0.738266. M1 transfer-validation receipt:
`transfer_valid=True`, all six primary checks pass.

## Interpretation (honest, bounded)

- **Recurrence is already captured by existing mechanisms, not unexploited.**
  Even `sgd_raw` — a plain SGD baseline with no per-permutation state — gains
  +0.39 paired. The hypothesis that "none stores per-permutation state, so a
  repeated permutation is indistinguishable from a fresh one" is falsified by
  data, not argument.
- **Why a naive arm benefits is not answered by V6.** Plausible mechanisms
  (smoothing across repeated identical losses, reduced effective novelty per
  regime, or something else) are out of scope; V6 measures, it does not
  explain. A future A/B must pair the headroom hypothesis against this
  baseline before any recurrence-aware mechanism is proposed.
- **The headroom statement changes shape.** The gap between the best arm on M4
  (0.738) and the micro ceiling (0.984) is still large, but that gap can no
  longer be credited to recurrence per se. Direction D must be restated: the
  headroom is what a recurrence-aware mechanism could win *on top of* the
  recurrence benefit the ladder already extracts, which this measurement
  brackets at roughly zero to the current best arm's margin.
- **Not a promotion route, stated in advance per #1875:** best-of-arms,
  per-seed cherry-picking, `recurrence_pool` changes, extra arms, and
  mean-only reporting when the all-seeds condition fails are all excluded.
  No mechanism is proposed or implemented here.

## Reproducibility

```
.venv/bin/python -m alberta_framework.benchmarks.micro_continual ladder \
  --family input_permutation --seeds 0 1 2 \
  --out outputs/new_directions/V6_recurrence_headroom/m1_permutation
.venv/bin/python -m alberta_framework.benchmarks.micro_continual ladder \
  --family recurrence --seeds 0 1 2 \
  --out outputs/new_directions/V6_recurrence_headroom/m4_recurrence
```

Both commands run the shipped CLI with shipped defaults (arm set, stream
parameters, hidden sizes, bayes samples). Source HEAD:
`ec035f73ad8e1e0545155e94c3e689d4b8d5d82b`.
