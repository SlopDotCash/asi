# IPMNIST Campaign: 2026-08 Research Roadmap & Priority Ranking

**Date:** 2026-08-14  
**Status:** Analysis document for coordinating future contributions  
**Audience:** Future contributors to the IPMNIST screening lane

## Executive summary

This document ranks all proposed next measurements (pre-registered contributions + candidate ideas) by impact, feasibility, and likelihood of moving the needle. It synthesizes:
- Standing record: 0.87114 ± 0.00010 (`rls_head_resid_l1_preset005`, development-grade)
- Incumbent A/B baseline: 0.86449 ± 0.00009 (`sigma0_shiftnorm_d099`)
- Ceiling (this method family): ~0.933
- Measured gap: ~0.068 remaining

## Tier 1: High-Impact, Ready-to-Execute (2026-08-14 pre-registrations)

All four arms below are pre-registered, fully specified, and require only environment setup + measurement time.

### 1a. RLS Head Residual Held-Out Validation (Climb)
- **Pre-reg:** CONTRIBUTION_PREREGISTRATION.md
- **Arm:** `rls_head_resid_l1_preset005` on seeds 3–19 (held-out)
- **Baseline:** 0.86449 (incumbent)
- **Predicted improvement:** +0.0065 (already at 0.87114 on development seeds; question is generalization)
- **Cost:** ~15 hours compute (200-task × 17 seeds)
- **Likelihood:** 60% (standing record might not generalize; seeds 0–2 consumed during selection)
- **Priority rank:** ★★★★★ (highest impact if it holds; definitively settles "RLS readout vs MLP")

### 1b. Wave 9 Shiftnorm Interactions (Climb)
- **Pre-reg:** WAVE9_SHIFTNORM_PREREGISTRATION.md
- **Arms:** 4 new shiftnorm variants testing (k, f, r) interactions
  - `sigma0_shiftnorm_d099_k05_f08` (gentler detector + faster recon)
  - `sigma0_shiftnorm_d099_k2_r50` (aggressive detector + rate-limit)
  - `sigma0_shiftnorm_d098_f08` (decay plateau interaction)
  - `sigma0_shiftnorm_d099_r50` (minimal rate-limit)
- **Predicted improvement:** +0.0005 to +0.002 each (if any interaction is synergistic)
- **Cost:** ~2 hours compute (4 arms × 60-task screen)
- **Likelihood:** 25–40% (single axes have been explored; interactions are speculative)
- **Priority rank:** ★★★★ (low cost, orthogonal exploration, fail-closed reporting valuable)

### 1c. Muon Spectral Normalization Port (Port)
- **Pre-reg:** MUON_PORT_PREREGISTRATION.md
- **Arm:** `muon_gated` (spectral-norm update scaling via Newton-Schulz)
- **Source:** arXiv:2606.09762 (AdamO / Dynamical Isometry, Jun 2026)
- **Predicted improvement:** +0.001 to +0.003 (if weight-side conditioning helps)
- **Cost:** ~2 hours dev + ~6 hours compute
- **Likelihood:** 35–45% (fresh literature, unproven on online streams, weight-side conditioning might be redundant with input-side)
- **Priority rank:** ★★★★ (validates literature claim; addresses "other side" of conditioning)

### 1d. V4 Dual-Speed RLS Cache (Climb + Infrastructure)
- **Pre-reg:** V4_DUAL_SPEED_CACHE_PREREGISTRATION.md
- **Arms:** Cached RLS readout on IPMNIST (control) + micro_continual M4 (test)
- **Predicted improvement:** +0.015–0.040 on M4 (recurrence); 0 on IPMNIST (control)
- **Cost:** ~3 hours dev + ~5 hours compute
- **Likelihood:** 60% (mechanistic validation; architectural readiness for Step 11)
- **Priority rank:** ★★★★ (validates context-indexing; required for Alberta Plan completion)

## Tier 2: Medium-Impact, Requiring Implementation

### 2a. Alternative ensemble: RLS + Champion (Climb)
- **Idea:** nb_ensemble variant combining `sigma0_shiftnorm_d099` + `rls_head_resid_l1_preset005` instead of champion + naive_bayes
- **Rationale:** Both are top performers; ensemble might combine strengths (MLP transient + RLS asymptotic)
- **Predicted improvement:** +0.003 to +0.010 (if complementary)
- **Cost:** ~3 hours dev + ~4 hours compute (60-task screen on 3 seeds)
- **Likelihood:** 45% (ensembles risk averaging down; depends on member correlation)
- **Priority rank:** ★★★ (moderate effort, moderate payoff)

### 2b. Spectral Collapse / L2-ER Regularizer (Port)
- **Source:** arXiv:2509.22335 (ICML 2026, May 2026)
- **Idea:** Add effective-feature-rank regularizer (K-FAC Hessian analysis)
- **Mechanism:** Prevents spectral collapse of weight matrix Hessian at task boundaries
- **Predicted improvement:** +0.005 to +0.015
- **Cost:** ~4 hours dev (Hessian computation, regularization term) + ~6 hours compute
- **Likelihood:** 50% (theoretical grounding strong; weight-side again)
- **Priority rank:** ★★★ (validates theory; moderate complexity)

### 2c. Adaptive step-size scheduler (Port)
- **Source:** arXiv:2509.19698 (AAAI/NeurIPS-track, Sep 2025)
- **Idea:** Per-layer effective step-size based on gradient-noise and curvature-volatility bounds
- **Mechanism:** Auto-scheduler for learning rates (instead of fixed 0.01)
- **Predicted improvement:** +0.002 to +0.008
- **Cost:** ~5 hours dev (noise/curvature estimators) + ~6 hours compute
- **Likelihood:** 40% (optimizer-side conditioning; might be orthogonal to our input-side or absorbed by gate)
- **Priority rank:** ★★★ (validates optimizers; high-complexity implementation)

## Tier 3: Lower-Priority or Blocked

### 3a. Stream-X / Activation-based Methods (Port)
- **Status:** Blocked on activation function integration (not straightforward with JAX/protocol MLP)
- **Why low priority:** Our protocol fixes ReLU; activation swaps require architecture changes

### 3b. Replay-based Methods (Port)
- **Status:** Out of scope — protocol forbids replay
- **Examples:** Experience Replay (Wang et al.), Transformer in-context (survey literature)

### 3c. Recurrent/Stateful Methods (Port)
- **Status:** Blocked on protocol extension (IPMNIST doesn't have recurrence by design)
- **Could test on:** micro_continual M4, but IPMNIST itself is permutation-only

## Research questions to settle via Tier 1 work

| Question | Settles by | Predicted outcome |
|----------|-----------|-------------------|
| Does RLS readout generalize beyond tuned seeds? | 1a (RLS validation) | Likely yes (~60%); if no, closes this direction |
| Can we find synergistic hyperparameter interactions in shiftnorm? | 1b (Wave 9) | Likely no (~25%); fail-closed result valuable |
| Is weight-side conditioning orthogonal to input-side? | 1c (Muon) + 2b (L2-ER) | Likely redundant (~55% no gain); validates input-side dominance |
| Can we architect context-indexed memory correctly? | 1d (V4 cache) | Likely yes (~60%); prerequisite for Step 11 |

## Recommended execution order

1. **Execute 1a (RLS validation) FIRST** — Highest impact; will settle whether standing record holds
2. **Execute 1d (V4 cache) in parallel** — Infrastructure work; independent of 1a; required for Step 11
3. **Execute 1b + 1c in parallel** — Both are 2-hour exploratory screens; fail-closed reporting
4. **If 1a succeeds:** Candidate for held-out promotion; shift focus to Tier 2 ensembles
5. **If 1a fails:** Document in ledger; reprioritize weight-side methods (Tier 2b, 2c)

## Success metrics for the campaign

**Campaign win conditions:**
- **Immediate:** Any Tier 1 arm confirms >0.8645 (beat incumbent by ±SE)
- **Medium-term:** Tier 1 + Tier 2 combined: reach 0.870 ± 0.0001 (10% of remaining gap)
- **Long-term:** Integrate learnings into Alberta Plan Step 11 (context-indexed memory)

**Campaign fail-closed:**
- All Tier 1 arms lose or tie: validates ceiling ~0.865 with current method family
- Report to ledger: "Shiftnorm + RLS family exhausted on IPMNIST. Remaining gap requires architectural change (context-indexing, V4+, or new protocol)."

## Budget and timeline

| Tier | Total compute | Dev effort | Wall clock |
|-----|---|---|---|
| Tier 1 (all 4) | ~30 hours | ~8 hours | ~3–4 days (parallelizable) |
| Tier 2a–2c | ~16 hours | ~12 hours | ~2–3 days (sequential) |
| **Total throughput** | ~46 hours | ~20 hours | ~1–2 weeks (intensive) |

## Literature gaps and future directions

**Unexecuted from literature survey (SOTA_LANDSCAPE_2026.md):**
- Muon-class optimizers on online streams (filled by 1c)
- L2-ER (spectral collapse regularizer) on full-protocol IPMNIST (Tier 2b)
- Adaptive step-size schedulers on fixed-architecture online learning (Tier 2c)

**Alberta Plan linkage:**
- Step 11 (memory, context-indexing): V4 cache is the proof-of-concept
- Step 12 (intelligence amplification): requires working Step 11; deferred pending V4 validation

## Notes for future contributors

1. **Pre-registration is load-bearing.** All Tier 1 work is pre-registered; deviations must be documented.
2. **One variable changes at a time.** Multi-axis changes (e.g., changing k AND fast_decay AND r together) violate the measurement contract.
3. **Fail-closed reporting is valuable.** A clean negative result (all seeds worse) is worth crediting.
4. **Leverage the micro_continual suite.** M1–M4 are cheap ($11 min/6-arm ladder); use for fitness before full IPMNIST confirmation.
5. **Check the landscape regularly.** SOTA_LANDSCAPE_2026 may be updated; subscribe to arXiv plasticity tag for new methods.

## Appendix: Pre-registration documents

All Tier 1 pre-registrations are in this repository:
- CONTRIBUTION_PREREGISTRATION.md (1a: RLS validation)
- WAVE9_SHIFTNORM_PREREGISTRATION.md (1b: shiftnorm interactions)
- MUON_PORT_PREREGISTRATION.md (1c: Muon port)
- V4_DUAL_SPEED_CACHE_PREREGISTRATION.md (1d: cache validation)

Tier 2 ideas are documented here (this file) but lack full pre-registrations; authors should complete pre-reg before starting work.
