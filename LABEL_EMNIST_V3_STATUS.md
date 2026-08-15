# Label-Permuted EMNIST v3 Protection Arms - Status Report

**Date:** 2026-08-15  
**Pre-registration:** LABEL_EMNIST_V3_PREREGISTRATION.md  
**Status:** ⚠️ BLOCKED - Implementation issue discovered

---

## Progress Summary

### Completed
1. ✓ Added v3 arm factories to upgd_label_emnist.py:
   - upgd_ema_norm_cbp (UPGD + EMA norm + CBP recycling)
   - sgd_norm_cbp (SGD + norm + CBP)
   - upgd_l2init (UPGD with L2-to-init decay)
   - upgd_shiftnorm (shift-triggered reconditioning)

2. ✓ Registered hyperparameter defaults in _LEARNER_DEFAULT_HYPERPARAMETERS:
   - CBP parameters: cbp_decay_rate=0.99, cbp_replacement_rate=1e-4, cbp_maturity_threshold=100.0
   - L2-init: l2_init_strength=0.01
   - Shiftnorm: fast_decay=0.9, shift_delta=0.02, shift_k=1.0

3. ✓ Created plan.v3.json (seeds 100-102, 4 arms × 3 seeds = 12 shards)

4. ✓ Launched all 12 experiment shards in parallel

### Blocked
- **All experiments failed** with IndexError in CBP layer replacement code
- Error occurs in `ipmnist_screening.py:2122` in `cbp_maybe_replace_layer`
- Issue: "Too many indices: array is 1-dimensional, but 3 were indexed"

---

## Root Cause Analysis

The CBP implementation in `ipmnist_screening.py` appears to be designed for a specific network architecture (likely the IPMNIST 300×150 MLP). The label_emnist benchmark uses the same architecture (784→300→150→47), but there may be differences in:

1. **State structure:** CBP state indexing expects 3D arrays but receives 1D
2. **Layer configuration:** CBP assumes specific layer shapes/strides
3. **Network wrapping:** The adapter from ipmnist_screening factories to label_emnist may not preserve CBP state structure

---

## Options to Proceed

### Option A: Debug CBP integration (2-4h dev)
1. Compare CBP state structure between ipmnist_screening and upgd_label_emnist
2. Fix indexing in CBP layer replacement or adapter wrapper
3. Add integration tests
4. Rerun v3 experiments

**Pros:** Completes pre-registered experiment  
**Cons:** May require deep debugging; CBP might not be portable across benchmarks

### Option B: Run v3 without CBP arms (4h compute)
1. Drop upgd_ema_norm_cbp and sgd_norm_cbp from v3
2. Run only upgd_l2init and upgd_shiftnorm (2 arms × 3 seeds = 6 shards)
3. Partial validation of pre-registration hypothesis

**Pros:** Immediate progress on shiftnorm and L2-init  
**Cons:** Incomplete pre-registration; misses CBP contribution test

### Option C: Pivot to different ASI work
1. Rule Discovery Phase 2 (0h dev, 18h compute) - templates ready
2. Forager Open Baselines (2-6h dev, 17h compute)
3. Slowly Changing Regression v2 validation

**Pros:** Unblocks other mission objectives  
**Cons:** Leaves v3 pre-registration incomplete

---

## Recommendation

**Immediate:** Option B (run upgd_l2init and upgd_shiftnorm only)  
**Follow-up:** File CBP portability issue and add to technical debt

This provides partial validation of the v3 hypothesis (testing shiftnorm generalization and L2-init protection) while unblocking progress. The CBP arms can be completed later once the integration issue is resolved.

---

## Next Steps

1. Update plan.v3.json to remove CBP arms
2. Launch upgd_l2init and upgd_shiftnorm experiments (6 shards)
3. Document CBP portability issue in technical debt
4. Move to next ASI mission objective (Rule Discovery Phase 2 or SCR v2)

---

## Technical Debt Created

- **CBP portability bug:** ipmnist_screening CBP implementation doesn't generalize to upgd_label_emnist benchmark
- **Location:** `alberta_framework/benchmarks/ipmnist_screening.py:2122`
- **Impact:** Blocks v3 CBP arms; may affect other benchmark ports
- **Suggested fix:** Refactor CBP to be architecture-agnostic or add explicit state structure validation
