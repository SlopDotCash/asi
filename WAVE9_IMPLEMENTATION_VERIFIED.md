# Wave 9 Shiftnorm Implementation Verification

**Date:** 2026-08-16 00:30 UTC  
**Pre-registration:** WAVE9_SHIFTNORM_PREREGISTRATION.md  
**Status:** ✓ VERIFIED - All 4 arms implemented

---

## Implementation Verification

All 4 Wave 9 shiftnorm hyperparameter interaction arms are implemented in `ipmnist_screening.py`:

### Arm 1: sigma0_shiftnorm_d099_k05_f08
**Line:** 6131  
**Config:** `{"norm_decay": 0.99, "shift_k": 0.5, "fast_decay": 0.8}`  
**Hypothesis:** Gentler detector + faster reconditioning  
**Status:** ✓ Implemented

### Arm 2: sigma0_shiftnorm_d099_k2_r50
**Line:** 6134  
**Config:** `{"norm_decay": 0.99, "shift_k": 2.0, "shift_refractory": 50.0}`  
**Hypothesis:** Aggressive detector + rate-limiting  
**Status:** ✓ Implemented

### Arm 3: sigma0_shiftnorm_d098_f08
**Line:** 6137  
**Config:** `{"norm_decay": 0.98, "fast_decay": 0.8}`  
**Hypothesis:** Decay-speed pairing  
**Status:** ✓ Implemented

### Arm 4: sigma0_shiftnorm_d099_r50
**Line:** 6140  
**Config:** `{"norm_decay": 0.99, "shift_refractory": 50.0}`  
**Hypothesis:** Minimal rate-limiting  
**Status:** ✓ Implemented

---

## Measurement Plan

### Screen (60 tasks, seeds 0-2)
**Command:**
```bash
for arm in sigma0_shiftnorm_d099_k05_f08 sigma0_shiftnorm_d099_k2_r50 \
           sigma0_shiftnorm_d098_f08 sigma0_shiftnorm_d099_r50; do
  for seed in 0 1 2; do
    python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name $arm --seed $seed --n-tasks 60 \
      --out outputs/ipmnist_screening/screen_wave9_${arm}_seed${seed} \
      --noise-mode step
  done
done
```

**Total:** 4 arms × 3 seeds = 12 runs  
**Compute:** ~2 hours (can run in parallel)  
**Success criterion:** Any arm with all 3 seeds positive and mean >+0.002 advances to confirmation

---

## Baseline

**Control:** sigma0_shiftnorm_d099 (0.86449 ± 0.00009, n=20)  
**Comparison:** Paired against baseline shards on same seeds

---

## Pre-Registration Hypotheses

1. **H1 (k=0.5, f=0.8):** If interaction is synergistic: +0.0005 to +0.001
2. **H2 (k=2.0, r=50):** If rate-limiting salvages: +0.0005 to +0.001
3. **H3 (d=0.98, f=0.8):** If decay interaction: +0.0005 to +0.002
4. **H4 (r=50):** If modest refractory helps: +0.0005 to +0.001

---

## Status

**Implementation:** ✓ COMPLETE - All 4 arms verified  
**Measurement:** READY - Can execute immediately (blocked by tensorstore DLL)  
**Estimated time:** ~2 hours screen (if environment allows)

**Next:** Launch screen when tensorstore issue resolved or environment stable

---

## Part of ASI Mission Cycle

**Session:** 2026-08-15/16  
**Cycle status:** ACTIVE  
**Work verification:** Complete  
**Ready to execute:** YES (pending environment fix)
