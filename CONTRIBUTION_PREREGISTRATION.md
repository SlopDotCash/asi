# Pre-registration: RLS Ridge Optimization Screen

**Date:** 2026-08-14  
**Status:** Pre-registered, not executed

## Contribution type
**Climb** — Beat the incumbent baseline `sigma0_shiftnorm_d099` (0.86449 ± 0.00009, n=20) on the IPMNIST screening lane.

## Lane and metric
- **Lane:** `alberta_framework.benchmarks.ipmnist_screening`
- **Metric:** `average_online_accuracy` (one example per step, scored before update, fresh permutation every 5,000 steps)
- **Protocol:** ICLR-2024 Input-permuted MNIST, 200 tasks × 5,000 steps, 300×150 ReLU MLP

## Baseline measurement
- **Arm:** `sigma0_shiftnorm_d099` (shift-triggered re-conditioning with fast decay 0.9, norm_decay 0.99)
- **Value:** 0.86449 ± 0.00009 (n=20, seeds 0-19)
- **Source:** `outputs/ipmnist_screening/publication_runs/RESULTS.md`
- **Held-out subset (seeds 3-19, n=17):** 0.86447 ± 0.00009

## The one thing changing
**Candidate arm:** `rls_head_resid_l1_preset005`

This arm replaces the champion's MLP readout with a streaming recursive-least-squares readout on the penultimate (150-dim) features:
- **Base body:** sigma0_shiftnorm_d099 (champion conditioning pipeline)
- **Readout:** One-vs-all RLS on 150 penultimate features
- **Key hyperparameters:**
  - `rls_lambda=1.0` (no forgetting; exact least-squares within-task, growing-window across)
  - `rls_reset_frac=0.05` (detector-driven P-matrix reset at 5% shifted-feature fraction)
  - `head_resid=1.0` (body trained on the head's own residual error)

**Rationale:** The standing record `rls_head_resid_l1_preset005` measured at 0.87114 ± 0.00010 (n=20, seeds 0-19, development-grade, consumed seeds 0-2) suggests the RLS readout can capture the remaining 0.009 gap. This pre-registration validates whether that result holds on fresh held-out seeds and establishes reproducibility.

## Seed and splitting strategy
- **Tuning seeds:** 0, 1, 2 (reuse consumed selection seeds; standard practice for diagnostics)
- **Evaluation seeds:** 3-19 (n=17, selection-untouched, held-out)
- **Screen + confirm:** 60 tasks (paired baseline rerun) → 200 tasks (full-protocol confirmation)
- **Within-arm comparison:** Paired against `sigma0_shiftnorm_d099` on shared seeds

## Success threshold
- **Win criterion:** Held-out mean (seeds 3-19) improvement >+0.0009 (three times the incumbent's standard error), with **all 17 held-out seeds individually positive**.
- **Reporting:** Full-seed mean and held-out-only mean separately; pre/post shard comparison artifact paths.

## If it loses
- Report the held-out mean and per-seed spread
- Record the finding in `NEGATIVE_RESULTS_LEDGER.md`: "RLS readout on champion body does not sustain standing-record on held-out seeds / held-out-only mean falls below 0.86447"
- Close this direction with the measured evidence

## Commands and flags
```bash
# 60-task screen (paired against sigma0_shiftnorm_d099 shards)
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rls_head_resid_l1_preset005 --seed 0 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_rls_resid_l1p5_seed0 \
  --noise-mode step

# 200-task full protocol (confirm)
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rls_head_resid_l1_preset005 --seed 0 --n-tasks 200 \
  --out outputs/ipmnist_screening/confirm_rls_resid_l1p5_seed0 \
  --noise-mode step
```

## Deviations from this pre-registration that invalidate the result
- Tuning on evaluation seeds (3-19)
- Reusing held-out seeds for any other arm's selection
- Changing threshold after seeing numbers
- Rerunning collapsed shards instead of reporting them

## Notes
- All arms in this lane are development-grade, nonpromoting (no frozen protocol)
- This is an existence proof: if the RLS readout generalizes, the method deserves design work; if it doesn't, the result is equally informative
- Compute cost: ~30 minutes per seed on CPU, ~2 hours per seed total for 60+200 task runs
- The champion baseline shards already exist at `outputs/ipmnist_screening/shards/sigma0_shiftnorm_d099_seed*` (reuse for pairing)
