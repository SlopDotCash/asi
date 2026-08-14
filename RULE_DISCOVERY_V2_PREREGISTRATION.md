# Pre-registration: Rule Discovery Automation v2 — Gaussian Suite Migration & Fitness Harness

**Date:** 2026-08-14  
**Status:** Pre-registered, awaiting execution  
**Lane:** Rule Discovery (automated update-rule search via micro_continual fitness)  
**Type:** Fix + Climb — migrate from provisional digits suite to canonical Gaussian suite and expand discovered-rule coverage

## Background

The rule_discovery lane (outputs/rule_discovery/) is an automated search for update rules that beat the campaign champion on the micro_continual suite. Current status:

1. **Provisional digits suite** (sklearn-digits, M1+M2+M3 fitness, seeds 0–1):
   - Used for live search fitness
   - No transfer validation
   - No analytic Bayes ceiling

2. **Canonical Gaussian suite** (analytic Bayes, validated M1 transfer):
   - Available in micro_continual.py (gauss-v1 defaults)
   - Frozen transfer-validated operating point
   - Analytic references

**SUITE.md reconciliation plan** (§7):
> "The discovery harness migrates its fitness tasks to the canonical Gaussian suite (analytic Bayes ceilings, validated M1 transfer, campaign-parity arms); the digits suite then becomes a holdout family or is retired."

This pre-registration executes that migration and expands the fitness harness.

## Migration scope

### Phase 1: Rebase fitness evaluation to Gaussian suite
**Current state:** search_v1.json uses digits suite (M1+M2+M3, seeds 0–1)

**Target:** 
- Fitness: canonical Gaussian M1 (input permutation, 40 regimes, seeds 0–1)
- Validation: digits suite M1+M2+M3 (seeds 101–103, held-out, unchanged)
- Add: analytic Bayes ceiling per fitness run for sanity-checking

**Implementation:**
```python
# In rule_discovery fitness loop, replace:
# fitness_stream = build_micro_stream(MICRO_SUITE(M1+M2+M3), seed=seed_id)
# With:
from alberta_framework.benchmarks.micro_continual import (
    MicroStreamConfig, bayes_reference, run_micro_arm
)

config = MicroStreamConfig(family='input_permutation', n_regimes=40, seed=seed_id)
fitness_stream = GaussianMicroStream(config)
fitness_ceiling = bayes_reference(config, seed=seed_id)  # ~0.97 oracle
```

**Cost:** ~20 lines code change + re-run search fitness (~4 hours compute)

### Phase 2: Expand search to new rule classes
**Current winners** (from search_v1.json, top 3):
- Discovered-rule-1: error-gated plasticity + hidden RMS norm
- Discovered-rule-2: [hidden]
- Discovered-rule-3: [hidden]

All three "dropped the utility gate and adopted the error-gated plasticity budget (surprise_budget) + hidden RMS normalization" (NEW_DIRECTIONS.md §6 comment).

**New search directions (Tier 2 exploration):**

#### Direction A: Gate variants
- Hypothesis: error-gating (surprise_budget) works; test other gating signals
- Candidates: loss-gating, gradient-norm-gating, entropy-gating
- Scope: 5–10 new rule templates in search space

#### Direction B: Normalization locations
- Current: hidden RMS (inside network)
- Test: input-side RMS, layer-wise norms, output-side norms
- Scope: 3–5 location variants × 2 norm types = 6–10 templates

#### Direction C: Adaptation speed (meta-learning)
- Hypothesis: discovered rules have implicit fast/slow timescales
- Test: explicit meta-decay / meta-step-size adaptation
- Scope: 3–5 meta-speed settings

**New search run:**
```bash
# Expanded fitness sweep (Gaussian suite)
.venv/bin/python -m rule_discovery.search \
  --fitness-config canonical-gaussian \
  --fitness-seeds 0 1 \
  --validation-seeds 101 102 103 \
  --rule-templates [A+B+C expansion] \
  --n-candidates 500 \
  --n-generations 50 \
  --out outputs/rule_discovery/search_v2_gaussian_expanded.json
```

**Cost:** ~12–20 hours compute (depending on template count)

## Pre-registration specifics

**Fitness metric:** Micro-continual M1 (Gaussian, 40 regimes, seeds 0–1)  
**Fitness ceiling:** Analytic Bayes reference (~0.97)  
**Validation (unchanged):** Digits M1+M2+M3 (seeds 101–103)  
**Campaign pairing:** Compare top-3 discovered rules to 6-arm IPMNIST ladder (sgd_raw, adamw, upgd_raw, sgd_norm, gated_norm, naive_bayes)

**Search space:** Current 3 winners + 15–25 new templates (gate/norm/meta variants)

**Success threshold:** 
- At least 1 new rule beats current micro champion (gated_norm, ~0.88 on M1)
- New rule transfers to IPMNIST: beats 0.80 baseline at 60-task screen
- Rule is interpretable (human-readable mechanism, not black-box parameters)

## Phase breakdown

### Phase 1: Migration & re-baseline (Fix)
```bash
# 1a. Verify Gaussian suite fitness equivalence
.venv/bin/python -m micro_continual ladder \
  --family input_permutation --seeds 0 1 --out outputs/micro_continual/validation_gaussian_fitness
# Expected: ordering matches IPMNIST campaign (gated_norm > sgd_norm > others)

# 1b. Rerun search_v1 candidates on Gaussian fitness
for candidate in [top-3 from search_v1]; do
  .venv/bin/python -m micro_continual run \
    --family input_permutation --arm $candidate --seed 0 \
    --out outputs/rule_discovery/revalidate_${candidate}_seed0
done
# Expected: scores similar to digits results (if mechanism generalizes)

# 1c. Merge and compare
.venv/bin/python -m rule_discovery validate \
  --old-results outputs/rule_discovery/search_v1.json \
  --new-fitness outputs/rule_discovery/search_v2_revalidate_summary.json
```

**Cost:** ~2 hours (Phase 1a + 1b)

### Phase 2: Expanded search (Climb)
```bash
# Full expanded search on Gaussian suite
.venv/bin/python -m rule_discovery search \
  --fitness-config canonical-gaussian-m1 \
  --fitness-seeds 0 1 \
  --n-candidates 500 \
  --n-generations 50 \
  --rule-templates [all A+B+C] \
  --out outputs/rule_discovery/search_v2_gaussian_expanded.json
```

**Cost:** ~16 hours compute

### Phase 3: Transfer validation (Climb)
If Phase 2 produces new winners (fitness > 0.89):

```bash
# Screen on IPMNIST (60 tasks, paired vs incumbent)
for discovered_rule in [winners]; do
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name $discovered_rule --seed 0 --n-tasks 60 \
    --out outputs/ipmnist_screening/screen_discovered_${discovered_rule}_seed0
done
```

**Cost:** ~2 hours (if 2–3 winners)

## Fail-closed reporting

**If Phase 1 migration fails (new fitness != old fitness):**  
Record: "Gaussian suite fitness does not preserve discovered-rule ordering vs. digits suite; domain-specificity of evolved rules is higher than expected."

**If Phase 2 produces no new winners:**  
Record: "Expanded search space (gates, norms, meta-speeds) yields no improvement over current best discovered rules; current winners occupy local optimum in micro_continual fitness landscape."

**If Phase 3 transfer fails (IPMNIST screen loss):**  
Record: "Discovered rules optimized on micro_continual do not transfer to IPMNIST; micro-suite is not a valid proxy for full-protocol development."

## Code deliverables

1. **Migration script** (~50 lines): Replace digits fitness with Gaussian, add Bayes ceiling logging
2. **Expanded templates** (~100 lines): 15–25 new rule classes (gate variants, norm locations, meta-speeds)
3. **Validation harness** (~50 lines): Compare old/new fitness, transfer-check pipeline

**Total dev:** ~4 hours

## Timeline

- Phase 1 (migration + re-baseline): ~2 hours
- Phase 2 (expanded search): ~16 hours
- Phase 3 (transfer validation): ~2 hours (if needed)
- **Total:** 20 hours compute + 4 hours dev

## References

- **Current search:** outputs/rule_discovery/search_v1.json
- **Provisional suite:** micro_continual.py, MICRO_SUITE, build_micro_stream
- **Canonical suite:** micro_continual.py, MicroStreamConfig (gauss-v1)
- **SUITE.md reconciliation:** outputs/micro_continual/SUITE.md §7
- **Current winners:** NEW_DIRECTIONS.md §6 (discovered-rule-1/2/3 commentary)
- **Transfer validation:** outputs/ipmnist_screening/RUNBOOK.md
