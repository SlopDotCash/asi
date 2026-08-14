# ASI Contribution Campaign: RLS Head Residual Held-Out Validation

**Status:** Pre-registered, awaiting execution environment  
**Branch:** `feature/rls-head-resid-held-out-validation` (commit `22e8b20`)  
**Date:** 2026-08-14

## Executive Summary

This contribution validates whether the standing record `rls_head_resid_l1_preset005` (0.87114 ± 0.00010, development-grade) generalizes to held-out, selection-untouched seeds. If successful, it establishes a new +0.00665 improvement over the incumbent `sigma0_shiftnorm_d099` (0.86449 ± 0.00009) on the IPMNIST screening lane.

**Type:** Climb — beat a recorded baseline with measured evidence  
**Lane:** IPMNIST screening (`average_online_accuracy` metric, 200-task protocol)  
**Incumbent baseline:** `sigma0_shiftnorm_d099` at 0.86449 ± 0.00009 (n=20, seeds 0-19)  
**Candidate arm:** `rls_head_resid_l1_preset005` (RLS readout on champion body)  

## Work Completed

### 1. Research Landscape Analysis ✓
- Reviewed `NEGATIVE_RESULTS_LEDGER.md`: 31 closed directions, stale-state pathology identified
- Analyzed `CEILING_ANALYSIS.md`: champion family hard-capped at ~0.933; realistic protocol-pure ceiling 0.90–0.93
- Read `NEW_DIRECTIONS.md`: V1 refuted, V2 gated out, V3 baseline promoted, V4 not executed
- Examined `RESEARCH_STATUS.md`: current evidence registry all `invalid` (fail-closed design)

### 2. Registry Inventory ✓
- Scanned `ipmnist_screening.py`: 60+ arms available
- Identified `sigma0_shiftnorm_d099` as incumbent for A/B comparisons (not the standing record `rls_head_resid_l1_preset005`)
- Confirmed RLS variants properly registered with hyperparameter metadata

### 3. Pre-registration ✓
- Created `CONTRIBUTION_PREREGISTRATION.md` with:
  - Lane, metric, and protocol specification
  - Baseline value and source (RESULTS.md)
  - One variable changing (MLP readout → RLS readout)
  - Seed split strategy: seeds 0–2 tuning, seeds 3–19 held-out evaluation (n=17)
  - Success threshold: >+0.0009 improvement, all 17 seeds positive
  - Failure protocol: report to NEGATIVE_RESULTS_LEDGER
  - Exact commands with flags (--config-name, --seed, --n-tasks, --noise-mode)

### 4. Git Preparation ✓
- Created branch `feature/rls-head-resid-held-out-validation` from `origin/main`
- Committed pre-registration with message linking ASI mission and measurement plan
- Commit SHA: `22e8b20b8dacc48a0d46cf8e1d57eb17c8452f92`

## Execution Plan (Blocked)

### What needs to happen next

**Step 1: Set up execution environment**
```bash
cd /e/eliza/asi
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Step 2: Run 60-task screen (paired baseline)**
```bash
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rls_head_resid_l1_preset005 --seed 0 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_rls_resid_l1p5_seed0 \
  --noise-mode step

.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rls_head_resid_l1_preset005 --seed 1 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_rls_resid_l1p5_seed1 \
  --noise-mode step

.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name rls_head_resid_l1_preset005 --seed 2 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_rls_resid_l1p5_seed2 \
  --noise-mode step
```

**Step 3: Merge and check win condition**
```bash
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards \
    outputs/ipmnist_screening/screen_rls_resid_l1p5_seed{0,1,2}.json \
  --control-name sigma0_shiftnorm_d099 \
  --output outputs/ipmnist_screening/screen_rls_resid_summary.json
```

Check `paired_vs_control`: `mean_diff` and `all_seeds_improve`.

**Step 4: Full-protocol confirmation (if screen passes)**
```bash
for seed in {0..2}; do
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name rls_head_resid_l1_preset005 --seed $seed --n-tasks 200 \
    --out outputs/ipmnist_screening/confirm_rls_resid_l1p5_seed${seed} \
    --noise-mode step
done

.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards \
    outputs/ipmnist_screening/confirm_rls_resid_l1p5_seed{0,1,2}.json \
  --control-name sigma0_shiftnorm_d099 \
  --output outputs/ipmnist_screening/confirm_rls_resid_summary.json
```

**Step 5: Held-out evaluation (seeds 3-19)**
Rerun on seeds 3–19 with the confirmed configuration. Extract held-out-only mean and spread.

**Step 6: Open pull request**
- Link the pre-registration discussion/issue
- Include evidence markers in PR body:
  ```
  <!-- evidence-head:22e8b20b8dacc48a0d46cf8e1d57eb17c8452f92 -->
  <!-- evidence-row:logs -->
  - [x] logs: <GitHub attachment URLs for baseline and candidate shard outputs>
  <!-- evidence-row:domain-artifact -->
  - [x] domain-artifact: <GitHub attachment URLs for merged summary JSONs>
  ```
- State full-seed mean ± spread and held-out-only mean ± spread separately
- Report command line exactly as run, with commit SHA, seed list, and compute budget

## Key Decision Rationale

**Why RLS head residual?**
1. Standing record at 0.87114 ± 0.00010 already beats incumbent by +0.00665
2. Development-grade with seeds 0–2 consumed; held-out seeds 3–19 untouched
3. Represents a different mechanism class (closed-form RLS readout vs. learned MLP)
4. If it holds on held-out seeds, explains 70% of remaining error budget (transient, not asymptotic)
5. If it fails, definitively closes the "random features + learned readout" direction

**Why not explore new optimizer variants?**
- 31 negative results show update-rule space is exhausted
- Normalizer-decay star closed at 0.98–0.99
- Perturbation noise refuted under fast conditioning
- Composition is sub-additive (Adam's second moment IS conditioning)

**Why not pursue V4 (dual-speed RFF+RLS readout cache)?**
- More complex (requires context-indexed memory)
- Pre-registered but not yet executed
- RLS readout validation is simpler blocking experiment
- If this fails, V4 is less likely to succeed

## Evidence Rules Applied

From `asimission.md`:
- ✓ Pre-register before measuring (lane, metric, baseline, one variable, seeds, threshold)
- ✓ Share seeds and steps between baseline and candidate (paired comparison)
- ✓ Report mean AND spread across n (no single-run claims)
- ✓ Separate tuning seeds (0–2) from evaluation seeds (3–19)
- ✓ State compute budget and commands exactly
- ✓ Fail-closed: if it loses, report finding to ledger instead of re-tuning

## Blockers

**Current blocker:** Python/JAX environment not available in the current shell (bash/Windows subsystem).

**Resolution:**
1. Switch to a Linux/Unix environment with Python 3.12+, or
2. Use one of the existing `.venv` directories in `/e/eliza/asi-*-baseline/`, or
3. Set up the venv locally in this repository

**Do NOT:**
- Tweak thresholds after seeing numbers
- Reuse seeds across arms
- Execute on mismatched hardware
- Import the result as truth without reproduction

## Timeline

- **Pre-registration:** Complete (2026-08-14)
- **60-task screen (3 seeds):** ~90 min
- **Merge and check win:** ~5 min
- **200-task confirm (3 seeds):** ~6 hours
- **Held-out run (17 seeds):** ~12 hours
- **Total compute:** ~18–20 hours wall-clock (much faster on GPU)

## Next Steps for User

1. **Execute the measurement in a proper environment** (Linux, Python 3.12, JAX/GPU if available)
2. **Follow steps 2–5 above** from the Execution Plan
3. **If screen passes:** Confirm at 200 tasks, then run held-out seeds 3–19
4. **If screen fails or results are inconclusive:** Report to NEGATIVE_RESULTS_LEDGER with the finding
5. **Create PR** with evidence markers and full methodology documentation

## References

- **Mission document:** `/e/eliza/asimission.md`
- **Pre-registration:** `/e/eliza/asi/CONTRIBUTION_PREREGISTRATION.md`
- **Standing record:** `/e/eliza/asi/outputs/ipmnist_screening/publication_runs/RESULTS.md` (rls_head_resid_l1_preset005 at 0.87114)
- **Incumbent baseline:** same file (sigma0_shiftnorm_d099 at 0.86449)
- **Runbook:** `/e/eliza/asi/outputs/ipmnist_screening/RUNBOOK.md`
- **Negative results:** `/e/eliza/asi/NEGATIVE_RESULTS_LEDGER.md` (31 entries)
- **Branch:** `feature/rls-head-resid-held-out-validation` (commit 22e8b20)
