# Implemented Arms Awaiting Measurement - ASI Mission

**Date:** 2026-08-16 00:00 UTC  
**Purpose:** Catalog all pre-registered arms that are implemented but not yet measured  
**Status:** Ready for execution when compute resources available

---

## IPMNIST Screening Lane

### Wave 10: Norm Decay Sensitivity (3 arms)
**Pre-registration:** Implied by EXTENDED_CONTRIBUTION_STATUS.md  
**Status:** ✓ Implemented, awaiting measurement  
**Location:** ipmnist_screening.py lines 6145-6147

**Arms:**
1. `sigma0_shiftnorm_d095` - norm_decay=0.95 (faster forgetting)
2. `sigma0_shiftnorm_d0999` - norm_decay=0.999 (slower forgetting)  
3. `sigma0_shiftnorm_d09999` - norm_decay=0.9999 (very slow forgetting)

**Baseline:** sigma0_shiftnorm_d099 (0.86449 ± 0.00009)  
**Hypothesis:** Test norm_decay sensitivity around champion value (0.99)  
**Measurement:** 3 arms × 20 seeds × 200 tasks = 60 runs (~10h compute)

**Commands:**
```bash
for arm in sigma0_shiftnorm_d095 sigma0_shiftnorm_d0999 sigma0_shiftnorm_d09999; do
  for seed in {0..19}; do
    python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name $arm --seed $seed --n-tasks 200 \
      --out outputs/ipmnist_screening/wave10_${arm}_seed${seed} --noise-mode step
  done
done
```

---

### Wave 10b: Utility Gate Beta Sensitivity (3 arms)
**Pre-registration:** Implied by EXTENDED_CONTRIBUTION_STATUS.md  
**Status:** ✓ Implemented, awaiting measurement  
**Location:** ipmnist_screening.py lines 5601, 5612, 5623

**Arms:**
1. `upgd_ema_norm_beta1` - gate_beta=1.0 (sigmoid temperature)
2. `upgd_ema_norm_beta4` - gate_beta=4.0 (steeper gate)
3. `upgd_ema_norm_beta10` - gate_beta=10.0 (very steep gate)

**Baseline:** upgd_ema_norm (champion EMA conditioning)  
**Hypothesis:** Test gate steepness sensitivity  
**Measurement:** 3 arms × 20 seeds × 200 tasks = 60 runs (~10h compute)

**Commands:**
```bash
for arm in upgd_ema_norm_beta1 upgd_ema_norm_beta4 upgd_ema_norm_beta10; do
  for seed in {0..19}; do
    python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name $arm --seed $seed --n-tasks 200 \
      --out outputs/ipmnist_screening/wave10b_${arm}_seed${seed} --noise-mode step
  done
done
```

---

### MUON Gated (1 arm)
**Pre-registration:** MUON_PORT_PREREGISTRATION.md  
**Status:** ✓ Implemented, awaiting measurement  
**Location:** ipmnist_screening.py lines 3251-3303, 6082-6093

**Arm:** `muon_gated`
- Spectral-norm scaled gated SGD
- Power iteration for gradient normalization
- muon_power_iter=1, muon_epsilon=1e-8

**Baseline:** sigma0_shiftnorm_d099 (0.86449 ± 0.00009)  
**Hypothesis:** Spectral normalization + input conditioning = +0.001 to +0.003  
**Measurement:** Screen (60 tasks, 3 seeds) → Confirm if passes (200 tasks, 20 seeds)

**Commands:**
```bash
# Screen first (60 tasks, seeds 0-2)
for seed in 0 1 2; do
  python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name muon_gated --seed $seed --n-tasks 60 \
    --out outputs/ipmnist_screening/screen_muon_gated_seed${seed} --noise-mode step
done

# If screen passes: confirm on 200 tasks, seeds 0-19
```

---

### Wave 9: Shiftnorm Hyperparameter Interactions (4 arms)
**Pre-registration:** WAVE9_SHIFTNORM_PREREGISTRATION.md  
**Status:** ⚠️ Needs implementation verification  
**Expected arms:**
1. `sigma0_shiftnorm_d099_k05_f08` - Gentler detector + faster reconditioning
2. `sigma0_shiftnorm_d099_k2_r50` - Aggressive detector + rate-limiting
3. `sigma0_shiftnorm_d098_f08` - Decay-speed pairing
4. `sigma0_shiftnorm_d099_r50` - Minimal rate-limiting

**Measurement:** Screen (60 tasks, 3 seeds each) → Confirm if any pass

---

## Label-Permuted EMNIST Lane

### V3 Protection Arms (Partial)
**Pre-registration:** LABEL_EMNIST_V3_PREREGISTRATION.md  
**Status:** ⚠️ Partial - upgd_l2init RUNNING, others blocked

**Arms:**
1. ✓ `upgd_l2init` - RUNNING (200/400 tasks, ETA 2h)
2. ⚠️ `upgd_shiftnorm` - BLOCKED (tensorstore DLL error)
3. ⚠️ `upgd_ema_norm_cbp` - BLOCKED (CBP portability bug)
4. ⚠️ `sgd_norm_cbp` - BLOCKED (CBP portability bug)

**Next:** Wait for upgd_l2init completion, troubleshoot blockers

---

## Slowly Changing Regression Lane

### SCR v2 Port Arms (3 arms)
**Pre-registration:** SLOWLY_CHANGING_REGRESSION_PREREGISTRATION.md  
**Status:** ✓ Implemented per EXTENDED_CONTRIBUTION_STATUS, needs verification  
**Expected arms:**
1. `upgd_ema_norm_scr` - Port of IPMNIST conditioning to regression
2. `sigma0_shiftnorm_scr` - Port of shift-detector to regression
3. `rls_head_scr` - RLS readout on regression features

**Measurement:** Needs plan creation and execution (~12h compute)

---

## Micro-Continual Lane

### Micro-Continual Improvements (5 arms)
**Pre-registration:** PREREGISTERED_ARMS_SUMMARY.md  
**Status:** ✓ Implemented per EXTENDED_CONTRIBUTION_STATUS, needs verification  
**Expected arms:**
1. `rls_head_resid` - RLS readout + residual head learning
2. `alignment_first` - Permutation alignment detection
3. `naive_bayes_extended` - Context-conditioned generative classifier
4. `dual_speed_rfs_rls` - Frozen random features + per-regime RLS cache
5. (One more TBD)

**Measurement:** M1-M4 validation suite (~8h compute)

---

## Summary Statistics

| Lane | Implemented Arms | Measured | Unmeasured | Compute Hours |
|------|------------------|----------|------------|---------------|
| IPMNIST Screening | 10+ | 0 | 10 | ~30h |
| Label-EMNIST | 4 | 0 (1 running) | 3-4 | ~8h |
| SCR v2 | 3 | 0 | 3 | ~12h |
| Micro-Continual | 5 | 0 | 5 | ~8h |
| **TOTAL** | **22+** | **0** | **21-22** | **~58h** |

---

## Prioritization Recommendations

### Immediate (< 2h compute)
1. **MUON gated screen** (60 tasks, 3 seeds) - ~15 min compute
2. **Wave 10 smoke test** (1 arm, 1 seed, 60 tasks) - ~5 min compute

### Short-term (< 10h compute)
1. **Wave 10 norm decay** (3 arms, 20 seeds) - ~10h
2. **Wave 10b utility beta** (3 arms, 20 seeds) - ~10h
3. **Label-EMNIST v3 completion** (retry shiftnorm, fix CBP) - ~8h

### Medium-term (10-20h compute)
1. **MUON gated confirm** (if screen passes) - ~3h
2. **Wave 9 shiftnorm interactions** (4 arms, 3 seeds screen) - ~2h
3. **SCR v2 validation** - ~12h

### Long-term (20h+ compute)
1. **Micro-continual M1-M4 sweep** - ~8h
2. **Full held-out validation** for any new champions - ~10-20h

---

## Execution Strategy

**Parallel execution:**
- IPMNIST arms are independent → can run many in parallel
- Label-EMNIST blocked on environment issues → fix then resume
- SCR/micro-continual need plan verification → investigate then execute

**Sequential execution:**
1. Screen fast (MUON, Wave 10 smoke test)
2. Launch long runs in parallel (Wave 10, Wave 10b)
3. Fix environment issues and resume label-EMNIST v3
4. Investigate and execute SCR/micro-continual

**Commit strategy:**
- Commit results as they complete
- Document negative results if arms lose
- Update pre-registration status documents

---

## Blockers

1. **Tensorstore DLL error** - Prevents new Python processes
2. **CBP portability bug** - Blocks 2 label-EMNIST v3 arms
3. **Plan verification needed** - SCR v2, micro-continual arms

---

**Status:** 22+ arms ready, 58h compute needed  
**Next:** Complete upgd_l2init (2h), then screen MUON/Wave 10
