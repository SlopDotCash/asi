# Pre-registration: Slowly-Changing Regression v2 Development Plan

**Date:** 2026-08-14  
**Status:** Pre-registered, ready for execution  
**Lane:** Slowly-Changing Regression (Nature Methods reference: Dohare et al., Loss of plasticity in deep continual learning)  
**Type:** Fix + Climb — establish proper v2 measurement infrastructure and validate Alberta methods

## Background

The slowly_changing_regression lane is a development extension of the Nature Methods plasticity study (Dohare et al., 2024). The RUNBOOK.md explicitly states:

> "No v2 pre-run plan has been issued and no full shard worker has been launched. This file is a launch template only. **Do not start a shard until the complete bound-source set has had a verified quiet window and a new immutable plan has been issued.**"

This pre-registration establishes the v2 plan and ready-to-execute measurements.

## Lane specification

**Protocol:** Slowly-changing regression (1M examples, 20k-example bins, one random bit flip per period)

**Baseline arms:**
- `backprop_sgd_relu`: Ordinary backprop with ReLU activation, SGD lr=0.01 (Nature reference)
- `adamw_baseline`: AdamW control (published config)
- `upgd_w_baseline`: UPGD-W control

**Alberta-local arms (new):**
- `upgd_ema_norm`: Input-statistics normalization + gate (from IPMNIST, revalidated on regression)
- `sigma0_shiftnorm`: Shift-triggered re-conditioning (IPMNIST champion, ported to regression domain)
- `rls_head`: RLS readout on final layer features (test readout architecture on regression)

## Measurement objectives

### Objective 1: Validate plasticity mechanism decomposition
**Question:** Does the input-statistics-tracking-speed mechanism (from IPMNIST) transfer to a different domain (regression)?

**Hypothesis:** If conditioning dominates on IPMNIST, it should also show gains on slowly-changing regression (different non-stationarity type: output shift, not input permutation).

**Arms:** `backprop_sgd_relu` vs `upgd_ema_norm` (same decomposition as IPMNIST)

**Prediction:** +0.03 to +0.08 (if conditioning mechanism is general) or ≈0 (if mechanism is specific to input permutations)

### Objective 2: Test shift-triggered detector on new domain
**Question:** Does shift-triggered re-conditioning (IPMNIST's sigma0_shiftnorm) work on regression output shifts?

**Arms:** `upgd_ema_norm` vs `sigma0_shiftnorm_d099` (ported to regression)

**Prediction:** +0.005 to +0.015 (if detector generalizes) or flat/negative (if detector is over-tuned to IPMNIST permutation boundaries)

### Objective 3: RLS readout on regression
**Question:** Does RLS streaming readout replace learned readout on regression tasks?

**Arms:** Standard RLS on final-layer features vs `backprop_sgd_relu` baseline

**Prediction:** Comparable or better (if RLS solves a fundamental regression-on-features problem) or worse (if gradient-based learning is necessary)

## Execution plan

### Phase 1: Verify environment and baseline reproduction (Dev/CI)
```bash
# Smoke test: verify Nature reference reproduction
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression run \
  --config-name backprop_sgd_relu --seed 100 --num-examples 10000 \
  --out outputs/slowly_changing_regression/smoke_backprop_10k
# Expected: converges without error; early-bin accuracy ~0.8+
```

**Success:** Baseline runs without error; matches Nature figure patterns (qualitative).

### Phase 2: Screen all arms (60k examples, seeds 100–102)
```bash
for arm in backprop_sgd_relu adamw_baseline upgd_w_baseline upgd_ema_norm sigma0_shiftnorm rls_head; do
  for seed in 100 101 102; do
    .venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression run \
      --config-name $arm --seed $seed --num-examples 60000 \
      --out outputs/slowly_changing_regression/screen_${arm}_seed${seed}
  done
done
```

**Cost:** ~6 hours compute (6 arms × 3 seeds × 20 min per run)

**Success criterion:** All arms run to completion; no numerical errors; per-bin accuracy curves are smooth.

### Phase 3: Merge and analyze (60k summaries)
```bash
.venv/bin/python -m slowly_changing_regression merge \
  --shards outputs/slowly_changing_regression/screen_*_seed*.json \
  --output outputs/slowly_changing_regression/screen_summary_60k.json
```

**Analysis:**
- Per-arm mean final-bin accuracy
- Plasticity loss rate (slope of late-bin accuracy over periods)
- Comparison to Nature baseline

### Phase 4: Full-protocol confirmation (1M examples, seeds 100–102)
If any arm shows >+0.02 improvement on screen:
```bash
for arm in [winners]; do
  for seed in 100 101 102; do
    .venv/bin/python -m slowly_changing_regression run \
      --config-name $arm --seed $seed --num-examples 1000000 \
      --out outputs/slowly_changing_regression/confirm_${arm}_seed${seed}
  done
done
```

**Cost:** ~18 hours compute (per winning arm)

## Pre-registration specifics

**Baseline:** `backprop_sgd_relu` (Nature reference)  
**Metric:** Final-bin (last 20k examples) average accuracy  
**Seed split:** 100–102 (new seeds; no prior tuning on these)  
**Success threshold:** Any arm with all-seeds-positive and mean >+0.02 improvement advances to Phase 4

## Deviations that void the result

- Changing arm hyperparameters after seeing Phase 2 results
- Using seeds 0–99 (tuning/selection boundary)
- Merging across different num-examples settings
- Comparing against non-paired baseline (must rerun baseline with same seed set)

## Fail-closed reporting

**If all arms tie or lose:**
- Record in NEGATIVE_RESULTS_LEDGER: "Input-statistics conditioning (from IPMNIST) does not transfer to slowly-changing regression output-shift protocol; mechanism may be specific to input-domain non-stationarity."
- Implication: Different domains need different treatment; IPMNIST insights don't generalize

**If some arms win:**
- Validates mechanism generality
- Feeds into Alberta Plan theory (Step 1: is conditioning universal?)

## Timeline

- Phase 1 (smoke): ~30 min
- Phase 2 (screen): ~6 hours
- Phase 3 (merge): ~30 min
- Phase 4 (confirm, if winners exist): ~18 hours per arm

**Total critical path:** 7–25 hours depending on Phase 2 outcome

## References

- **Nature Methods original:** Dohare et al., "Loss of plasticity in deep continual learning" (2024)
- **Pinned source:** github.com/shibhansh/loss-of-plasticity v1.1
- **IPMNIST theory:** CONTINUAL_LEARNING_THEORY.md (conditioning dominance thesis)
- **RUNBOOK template:** outputs/slowly_changing_regression/RUNBOOK.md
