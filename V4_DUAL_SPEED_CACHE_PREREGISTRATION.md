# Pre-registration: V4 Execution — Dual-Speed RFF+RLS Readout Cache

**Date:** 2026-08-14  
**Status:** Pre-registered (from NEW_DIRECTIONS.md §5), execution plan drafted  
**Type:** Climb — execute the fourth pre-registered validation from NEW_DIRECTIONS research essay

## Origin and rationale

From NEW_DIRECTIONS.md §5, V4 is a pre-registered minimal validation that was deferred "out of scope of this validation pass" but explicitly marked as "remains pre-registered."

**V4 specification (from NEW_DIRECTIONS.md):**
> "Dual-speed RFF+RLS with per-context readout cache: re-use the RFF+RLS lane; cache/restore readouts keyed by context-inference. On IPMNIST (no recurrence) predict no gain (control); on a recurring variant, predict instant recovery. *Promotes direction D's memory claim.*"

**Design rationale:** 
The RFF+RLS lane (random Fourier features + streaming RLS) achieves 0.848 accuracy on IPMNIST—within +0.017 of the deep champion. This arm tests whether *caching* the learned RLS readout across recurring task structures enables instant recovery of specialized readouts, validating the context-indexed memory architecture (direction D from NEW_DIRECTIONS §4).

**Key prediction:** 
- **On IPMNIST (no recurrence):** No gain expected (control)—each task has a fresh permutation, so cached readouts are worthless
- **On a recurring variant (e.g., M4 from micro_continual):** Instant recovery expected—reoccurring contexts can retrieve their cached readouts

## Contribution scope

**What this validates:**
1. Whether the RLS readout can be made *persistent* (cached across task boundaries)
2. Whether context-based indexing (via shift detection) enables proper readout retrieval
3. Whether memory-driven recovery is mechanistically distinct from within-task learning

**Not included (out of scope):**
- Full context-inference machinery (that's direction D, larger work)
- RL-track options or option models
- Recurrent task generation (use existing M4 protocol from micro_continual)

## Measurement plan

### Arm 1: `rfl_rls_cache_ipmnist` (control — no recurrence, cache should be useless)
- **Config:** RFF+RLS with readout caching enabled, measured on standard IPMNIST (no recurrence)
- **Expected:** ≈ 0.848 (same as `rff_rls` baseline; caching provides no benefit)
- **Success criterion:** Within ±0.0005 of `rff_rls_cache` disabled

### Arm 2: `rfl_rls_cache_micro_m4` (test — recurring tasks, cache should help)
- **Config:** RFF+RLS with readout caching on micro_continual M4 (recurrence regime)
- **Baseline:** `rff_rls` on M4 (from SUITE.md §6: mean 0.621 on M4, descriptive)
- **Expected:** +0.015 to +0.040 (instant readout recovery for recurring contexts)
- **Success criterion:** Held-out improvement >+0.010 on recurring tasks

### Arm 3: `rfl_rls_cache_detector_variants` (analysis — test cache invalidation strategies)
- **Variants:** 
  1. Readout cache keyed by exact permutation identity (oracle—known recurrence)
  2. Readout cache keyed by shift-detector output (no oracle—inferred recurrence)
  3. Readout cache + soft-reset on weak shifts (cache freshness control)
- **Scope:** Diagnostic runs only (single seed, m=40 micro_continual regime)
- **Goal:** Measure cache-hit rate and invalidation patterns

## Implementation sketch

**New functions to add** (estimated 40–80 lines):
```python
class RFLRLSCacheState:
    """RFS+RLS state + readout cache indexed by context ID."""
    rfs_rls_state: RFSRLSState  # existing RFS+RLS 
    readout_cache: dict[int, Array]  # context_id -> cached readout vector
    context_counter: int  # for generating unique IDs

def _make_rfl_rls_cache_learner(...) -> ScreeningFactory:
    """Factory for cached RFL+RLS: same as rff_rls but with cache logic."""
    # init_fn: create empty cache, same RFS+RLS init
    # predict_fn: look up cached readout if context seen before, else fresh init
    # update_fn: update RFS+RLS; cache new readout keyed by context_id
    # context_id assignment: use shift-detector output (or oracle ID for control)
    pass
```

**Registry additions:**
```python
specs.append(ScreeningSpec(
    name="rfl_rls_cache_ipmnist",
    base_learner="upgd_w",
    mechanism="cached_readout",
    hyperparameters={...rff_rls_hyperparameters..., "cache_enabled": 1.0},
    factory=_make_rfl_rls_cache_learner,
    frozen_probe_input=_rff_frozen_probe_input,
    description="RFF+RLS with readout cache (control: IPMNIST, no recurrence, "
                "cache should provide no gain)",
))
```

## Measurement sequence

**Phase 1: Control validation (IPMNIST screen)**
```bash
# Verify cache overhead is minimal
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rfl_rls_cache_ipmnist --seed 0 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_cache_control_seed0 --noise-mode step
# Expected: within ±0.0005 of existing rff_rls (0.848)
```

**Phase 2: Recurring regime test (micro_continual M4)**
```bash
# Test on micro_continual recurring tasks
.venv/bin/python -m alberta_framework.benchmarks.micro_continual run \
  --family recurrence --arm rfl_rls_cache_micro_m4 --seed 0 \
  --out outputs/micro_continual/rfl_rls_cache_m4_seed0
# Expected: >+0.010 improvement over non-cached rff_rls baseline
```

**Phase 3: Cache diagnostics (single-seed analysis)**
```bash
# Cache-hit rate and invalidation analysis
for variant in oracle detector soft_reset; do
  .venv/bin/python -m alberta_framework.benchmarks.micro_continual run \
    --family recurrence --arm rfl_rls_cache_micro_m4_${variant} \
    --seed 0 --out outputs/micro_continual/cache_analysis_${variant} \
    --instrumented  # enables cache telemetry logging
done
```

## Success criteria

| Measurement | Threshold | Outcome |
|---|---|---|
| IPMNIST control: within ±0.0005 of rff_rls | Yes | Proceed to Phase 2 |
| IPMNIST control: delta >±0.0010 | No | Investigate cache overhead; report infrastructure issue |
| M4 held-out improvement: >+0.010 | Yes | New record on M4; context-indexing works |
| M4 held-out improvement: −0.005 to +0.005 | No | Cache overhead ≈ gain; direction D requires more work |
| Cache-hit rate: >80% on oracle | Yes | Detector-based indexing is viable |
| Cache-hit rate: <60% on oracle | No | Shift detector insufficient for recurrence; report to ledger |

## Fail-closed reporting

If control fails (overhead): "Readout caching infrastructure adds >0.1% latency/memory without enabling recurrence on IPMNIST; defer cache mechanism pending lower-cost implementation."

If M4 shows no gain: "Cached readouts do not accelerate recovery on recurring permutation sequences; context-indexed memory (direction D) requires richer fingerprints or architecturally distinct design."

## Timeline

- **Implementation (cache logic + registry):** ~3 hours dev
- **Phase 1 screen (IPMNIST control):** ~20 minutes
- **Phase 2 micro_continual (M4, 3 seeds):** ~1 hour
- **Phase 3 diagnostics (cache analysis):** ~30 minutes
- **Total:** ~5 hours compute, ~3 hours dev

## References

- **Origin:** NEW_DIRECTIONS.md §5 (V4 pre-registration)
- **Theory:** NEW_DIRECTIONS.md §4 (direction D: context-indexed memory)
- **Micro protocol:** outputs/micro_continual/SUITE.md §2 (M4 regime, recurrence)
- **RLS lane:** outputs/ipmnist_screening/FINAL_REPORT.md (rff_rls arm, 0.848 baseline)
