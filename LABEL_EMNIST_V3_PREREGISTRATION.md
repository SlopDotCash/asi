# Pre-registration: Label-Permuted EMNIST v3 — Gating & Protection Extensions

**Date:** 2026-08-14  
**Status:** Pre-registered, awaiting execution  
**Lane:** UPGD Label-Permuted EMNIST (Elsayed & Mahmood, ICLR 2024)  
**Type:** Climb — extend v2 conditioning arms with protection mechanisms

## Background and motivation

The UPGD Label-Permuted EMNIST v2 lane (2026-08-02) tested whether input-conditioning from IPMNIST transfers to label-permutation non-stationarity. **Pre-registered prediction: conditioning should transfer weakly because inputs are stationary** (only labels permute). Results confirmed this: `upgd_ema_norm` stayed ±0.02 of baseline `upgd_w` (0.6715).

This v3 extension tests the complementary hypothesis: **on label-shift domains, protection/consolidation (the utility gate + CBP recycling) should dominate**, not conditioning. v2 used bare conditioning; v3 adds the full protection arsenal from IPMNIST to see if it transfers to label-permutation non-stationarity.

**Reference:** IPMNIST decomposition showed gate +0.011, conditioning +0.061. On L/P EMNIST, the roles should reverse: gate significant, conditioning minimal.

## Arms to add (v3)

All arms use same v2 seeds (100–102) for pairing against v1 baseline (0.6715).

### Arm 1: `upgd_ema_norm_cbp`
- **Config:** upgd_ema_norm (EMA conditioning) + CBP dormant-unit recycling
- **Mechanism:** Test if CBP (from IPMNIST) adds protection on top of conditioning
- **Prediction:** +0.005 to +0.015 (gate should matter more on label shifts)

### Arm 2: `sgd_norm_cbp`
- **Config:** sgd_ema_norm (gate+decay, no utility UPGD gate) + CBP
- **Mechanism:** Isolate CBP contribution independent of UPGD utility gate
- **Prediction:** +0.010 to +0.025 (if CBP is load-bearing under label shift)

### Arm 3: `upgd_l2init`
- **Config:** UPGD-W with weight decay pulling toward init (not zero), no conditioning
- **Mechanism:** Alternative to gating: explicit bias toward initial weights
- **Prediction:** +0.002 to +0.008 (transient protection via init-bias)

### Arm 4: `upgd_shiftnorm_emnist` (stretch goal)
- **Config:** Port sigma0_shiftnorm_d099 to label-permutation domain
- **Mechanism:** Test shift-triggered re-conditioning on label boundaries (abstract detector)
- **Prediction:** +0.005 to +0.010 (if detector generalizes to label shifts)

## Pre-registration specifics

**Baseline:** v1 `upgd_w` (0.6715 ± ~0.02, seeds 100–102)  
**Metric:** Online accuracy averaged over 400 tasks  
**Seed split:** 100–102 (same tuning seeds as v2; v1 baseline rerun for pairing)  
**Protocol:** L/P EMNIST (400 tasks, 2,500 steps/task, labels permuted)

**Success threshold:** Any arm with all-three-seeds positive and mean >+0.010 vs baseline (3× the v1 stderr)

## Measurement plan

### Phase 1: Re-run v1 baseline on v2 seeds (pairing foundation)
```bash
for seed in 100 101 102; do
  OMP_NUM_THREADS=2 .venv/bin/python -m alberta_framework.benchmarks.upgd_label_emnist shard \
    --plan outputs/upgd_label_emnist/plan.v1.json \
    --learner-id upgd_w --seed-id $seed \
    --partial-out outputs/upgd_label_emnist/partials_v1_rerun/upgd_w_seed${seed}.json
done
```

**Cost:** ~3.25 hours (3 seeds × 65 min/seed)

### Phase 2: Run v3 arms (all 4 new arms × 3 seeds)
```bash
for arm in upgd_ema_norm_cbp sgd_norm_cbp upgd_l2init upgd_shiftnorm_emnist; do
  for seed in 100 101 102; do
    OMP_NUM_THREADS=2 .venv/bin/python -m alberta_framework.benchmarks.upgd_label_emnist shard \
      --plan outputs/upgd_label_emnist/plan.v3.json \
      --learner-id $arm --seed-id $seed \
      --partial-out outputs/upgd_label_emnist/partials_v3/${arm}_seed${seed}.json
  done
done
```

**Cost:** ~8 hours compute (4 arms × 3 seeds; costs vary by arm; upgd_shiftnorm may be slower if detector adds computation)

### Phase 3: Merge and paired analysis
```bash
.venv/bin/python -m alberta_framework.benchmarks.upgd_label_emnist merge \
  --plan outputs/upgd_label_emnist/plan.v3.json \
  --partials outputs/upgd_label_emnist/partials_v3/*.json \
  --baseline outputs/upgd_label_emnist/partials_v1_rerun/upgd_w_*.json \
  --output outputs/upgd_label_emnist/results.v3.json
```

**Analysis:** Per-arm mean, per-seed deltas vs baseline, all-seeds-improve check

## Factory implementation (if arms don't exist)

New arms may require factory functions in `ipmnist_screening.py` or inline in the EMNIST benchmark:

```python
# Pseudo-code; exact implementation depends on existing factory patterns
def _make_upgd_ema_norm_cbp_learner(...):
    """UPGD-W + EMA conditioning + CBP dormant recycling."""
    # Base: upgd_ema_norm_learner
    # Add: CBP hyperparameters (cbp_replacement_rate, cbp_maturity_threshold)
    # Integration: apply CBP in the update step after EMA norm

def _make_upgd_shiftnorm_emnist_learner(...):
    """Port sigma0_shiftnorm_d099 to EMNIST domain."""
    # Reuse: shift detector logic from ipmnist_screening._make_upgd_shiftnorm_learner
    # Adapt: shift_delta, fast_decay tuned for label-permutation schedule (every 2,500 steps)
```

**Scope:** ~50–100 lines per new factory (code reuse from existing implementations)

## Predictions and hypotheses

**H1 (Protection dominates on label shift):** `upgd_ema_norm_cbp` or `sgd_norm_cbp` shows +0.010 to +0.025

**H2 (Init-bias is viable alternative):** `upgd_l2init` shows +0.002 to +0.008

**H3 (Shiftnorm generalizes to label domain):** `upgd_shiftnorm_emnist` shows +0.005 to +0.010

**If all three hold:** Validates that both conditioning (input-shift) and protection (label/output-shift) are general mechanisms, just domain-dependent in importance

**If all fail:** Suggests IPMNIST mechanisms don't generalize beyond input-permutation; L/P EMNIST requires different treatment

## Fail-closed reporting

**If all arms ≤+0.005 mean improvement:**  
Record in NEGATIVE_RESULTS_LEDGER: "Protection mechanisms (gate, CBP, L2-init, shiftnorm) from IPMNIST do not transfer to label-permutation non-stationarity; either IPMNIST mechanisms are input-specific or L/P EMNIST label-shift requires different architecture."

**Implication:** Different domains need tailored solutions; universality of conditioning/protection mechanisms remains unproven.

## Timeline

- Phase 1 (baseline rerun): ~3.5 hours
- Phase 2 (v3 arms): ~8 hours  
- Phase 3 (merge/analysis): ~30 min
- **Total:** ~12 hours compute + ~2 hours dev (if new factories needed)

## References

- **v1 RUNBOOK:** outputs/upgd_label_emnist/RUNBOOK.md
- **v2 results:** outputs/upgd_label_emnist/results.v2.json (conditioning effect minimal as predicted)
- **v1 baseline:** outputs/upgd_label_emnist/results.v1.json (0.6715)
- **IPMNIST decomposition:** CONTINUAL_LEARNING_THEORY.md (gate +0.011, conditioning +0.061)
- **CBP reference:** IPMNIST screening arms (adamw_cbp, upgd_cbp, etc.)
