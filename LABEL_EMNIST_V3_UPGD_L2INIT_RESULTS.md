# Label-EMNIST V3 Partial Results - upgd_l2init

**Date:** 2026-08-16 00:23 UTC  
**Pre-registration:** LABEL_EMNIST_V3_PREREGISTRATION.md  
**Status:** PARTIAL - 2/3 seeds completed  
**Arm:** upgd_l2init (UPGD with L2-to-init weight decay)

---

## Results Summary

### Completed Seeds
- **Seed 101:** 0.6725 (vs baseline 0.6715)
- **Seed 102:** 0.6746 (vs baseline 0.6715)

### Missing
- **Seed 100:** Incomplete/failed to write output

### Analysis
- **Mean (2 seeds):** 0.6735
- **Baseline (upgd_w):** 0.6715
- **Improvement:** +0.0020 (+0.3%)
- **Per-seed improvements:** +0.0010, +0.0031

---

## Pre-Registration Hypothesis

**From LABEL_EMNIST_V3_PREREGISTRATION.md:**
> **Arm 3: upgd_l2init**  
> **Prediction:** +0.002 to +0.008 (transient protection via init-bias)

---

## Assessment

### Against Pre-Registration
- **Predicted range:** +0.002 to +0.008
- **Observed (2 seeds):** +0.0020
- **Status:** At lower bound of prediction

### Significance
- Small sample (n=2, missing seed 100)
- Both seeds show positive improvement
- Improvement is small but consistent with hypothesis

### Conclusion
**INCONCLUSIVE** - Results suggest small positive effect at lower bound of prediction, but:
1. Missing seed 100 (incomplete data)
2. Small effect size (+0.3%)
3. Needs full 3-seed analysis for proper assessment

---

## Comparison to Other V3 Arms

**Pre-registered predictions:**
1. `upgd_ema_norm_cbp`: +0.005 to +0.015 - **BLOCKED (CBP bug)**
2. `sgd_norm_cbp`: +0.010 to +0.025 - **BLOCKED (CBP bug)**
3. `upgd_l2init`: +0.002 to +0.008 - **PARTIAL (+0.002, 2/3 seeds)**
4. `upgd_shiftnorm`: +0.005 to +0.010 - **BLOCKED (tensorstore DLL)**

---

## Technical Details

### Hyperparameters
```python
{
  "l2_init_strength": 0.01,
  "noise_std": 0.001,
  "step_size": 0.01,
  "utility_decay": 0.9,
  "weight_decay": 0.0
}
```

### Task Performance (Seed 101)
- Early tasks (1-100): 0.13 → 0.66 (learning phase)
- Mid tasks (100-300): 0.66 → 0.71 (stabilization)
- Late tasks (300-400): 0.71 → 0.73 (continued improvement)

### Observations
1. Steady accuracy improvement throughout run
2. No catastrophic forgetting visible
3. Performance curve similar to baseline but slightly elevated

---

## Next Steps

1. **Investigate seed 100 failure** - Check logs for cause
2. **Retry seed 100** - Complete 3-seed analysis
3. **Full analysis** - Calculate statistics with all 3 seeds
4. **Comparison** - Paired analysis vs baseline seeds 100-102

---

## Files
- `outputs/upgd_label_emnist/partials_v3/upgd_l2init_seed101.json`
- `outputs/upgd_label_emnist/partials_v3/upgd_l2init_seed102.json`
- `outputs/upgd_label_emnist/partials_v3/upgd_l2init_seed100.json` (MISSING)

---

## Session Context

**Part of:** ASI mission cycle, label_emnist v3 pre-registration  
**Status:** Work continues - seed 100 needs retry, other v3 arms blocked  
**Next:** Fix blockers (tensorstore DLL, CBP bug) and complete v3 suite
