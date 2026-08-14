# Pre-registration: Muon-Inspired Spectral Normalization Port

**Date:** 2026-08-14  
**Status:** Pre-registered, awaiting implementation & execution  
**Type:** Port — implement Muon-style update rule from literature on IPMNIST screening lane

## Literature source

**"Preserving Plasticity via Dynamical Isometry / AdamO"** (Rousseau, Müller, Nowé, arXiv:2606.09762, Jun 2026)

**Core finding:** Plasticity preserved by keeping layer-wise Jacobian singular values ≈ 1 (empirical-NTK argument). Isometry regularization decoupled from gradient updates ("AdamO", analogous to AdamW). Matches/beats existing approaches on supervised+RL plasticity benchmarks.

**Related work:** Muon (Ivgi et al., arXiv:2306.05882) — spectral-norm update geometry using orthogonal projection and Newton-Schulz matrix sign. Historically evaluated on optimization speed and generalization; **not yet on online continual streams like IPMNIST.**

**SOTA_LANDSCAPE_2026.md note (§1.2):** "Muon-class optimizers have **not** yet been evaluated on IPMNIST-style online streams (open arm for us)."

## Contribution plan

### Arm specification: `muon_gated`
- **Base:** Champion conditioning (sigma0_shiftnorm_d099) + utility gate
- **Update rule:** Spectral-norm scaling via Newton-Schulz orthogonalization (cheap approximation)
- **Mechanism:** For each layer weight matrix W:
  1. Compute spectral norm σ₁(∇L) via power iteration (1–2 steps; cheap)
  2. Scale gradient: ḡ ← ∇L / σ₁(∇L)
  3. Apply champion's gate: gate = sigmoid(β * utility_ema) ∈ [0, 1]
  4. Update: W ← W - α * gate * ḡ + decay_term

**Why this arm?**
1. **Fresh in literature (Jun 2026):** AdamO is recent; Muon evaluation on online streams is open
2. **Mechanistically aligned:** Our champion uses input-statistics conditioning (input-side); Muon/AdamO target weight-side conditioning (Jacobian isometry). Together they address both sides of the ill-conditioning problem
3. **Testable hypothesis:** If spectral normalization helps on IPMNIST (a protocol with abrupt distribution shifts), it should interact positively with our fast input-statistics retracking

### Implementation scope
- **New factory function:** `_make_muon_gated_learner()` in `ipmnist_screening.py`
- **Hyperparameters:** 
  - `muon_power_iter=1` (Newton-Schulz steps; 1 is cheap, 2 more accurate)
  - `muon_epsilon=1e-8` (numerical stability)
  - `gate_beta=1.0` (reuse champion's gate temperature)
  - `step_size=0.01` (tuned empirically vs randomized sweep)
  - `weight_decay=0.01` (champion's decay)
  - Rest: champion's hyperparameters verbatim (norm_decay=0.99, fast_decay=0.9, shift_k=1.0, etc.)
- **Cost:** ~50–80 lines of code (power iteration + gating + update)
- **Tests:** Bitwise reduction pin to a control arm (e.g., `sgd_ema_norm_d099` with spectral norm disabled)

## Success criteria

| Scenario | Action |
|----------|--------|
| Screen (60 tasks): paired mean >+0.0025, all seeds improve | Promote to 200-task confirm |
| Confirm (200 tasks): mean >0.86449 (beat incumbent) + held-out seeds 3–19 all improve | New record candidate |
| Confirm (200 tasks): mean 0.86300–0.86449, inconclusive but not negative | Rescreen with n=5 seeds |
| Screen or confirm: mean <0.86200 | Report to NEGATIVE_RESULTS_LEDGER: "Spectral normalization harms online continual learning" |

## Pre-registration

**Hypothesis:** On an online permuted-MNIST stream where weight matrices undergo rapid re-convergence post-shift, layer-wise Jacobian conditioning (via spectral normalization) will reduce the convergence overshoot and transient error, complementing our input-side conditioning. Prediction: +0.0010 to +0.003 over incumbent.

**Baseline:** `sigma0_shiftnorm_d099` (0.86449 ± 0.00009)

**Tuning seeds:** 0–2 (standard)  
**Evaluation seeds:** 3–19 (held-out, n=17)

**Screen plan:**
```bash
# Implement muon_gated factory + register arm
# Run screen (60 tasks)
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name muon_gated --seed 0 --n-tasks 60 \
  --out outputs/ipmnist_screening/screen_muon_gated_seed0 --noise-mode step

# Merge and check win condition (same as shiftnorm waves)
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/screen_muon_gated_seed*.json \
  --control-name sigma0_shiftnorm_d099 \
  --output outputs/ipmnist_screening/screen_muon_summary.json
```

**Confirmation plan (if screen passes):**
```bash
# Full 200-task protocol
for seed in 0 1 2; do
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name muon_gated --seed $seed --n-tasks 200 \
    --out outputs/ipmnist_screening/confirm_muon_gated_seed${seed} --noise-mode step
done

# Then run seeds 3–19 for held-out validation
for seed in {3..19}; do
  .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name muon_gated --seed $seed --n-tasks 200 \
    --out outputs/ipmnist_screening/heldout_muon_gated_seed${seed} --noise-mode step
done
```

## Why this is a genuine Port

1. **Core mechanism is from the paper:** Newton-Schulz orthogonalization and spectral-norm scaling are verbatim from Muon/AdamO
2. **Adaptation to our protocol:** Replacing full Adam with spectral-norm gated SGD (our champion's update rule) + integration with our shift-detector
3. **Not a simple hyperparameter tune:** Requires new code path (power iteration, spectral-norm computation)
4. **Literature-stated gap:** SOTA_LANDSCAPE explicitly notes Muon "has not yet been evaluated on IPMNIST-style online streams"

## Deviations that void the result

- Using published Muon code and running it unmodified (must integrate with our gating/detector)
- Tuning power iteration order / epsilon after seeing results
- Changing from spectral-norm to another curvature approximation mid-measurement
- Comparing against non-paired baseline

## Fail-closed reporting

If `muon_gated` loses to incumbent or ties:
- Record in NEGATIVE_RESULTS_LEDGER: "Spectral normalization (Muon-style Newton-Schulz power iteration + weight update scaling) does not improve or decreases performance on IPMNIST permutation stream; likely because champion's fast input-statistics tracking already solves the Jacobian-conditioning problem on this protocol's fast transients."
- Reinforces: input-side conditioning dominates weight-side conditioning on this problem

## Timeline

- **Implementation (factory + tests):** ~2 hours dev time
- **Screen (3 seeds × 1–2 min):** ~15 minutes
- **Confirm (3 seeds × ~10 min):** ~30 minutes
- **Held-out (17 seeds × ~10 min):** ~3 hours
- **Total:** ~6 hours compute, ~1 hour if fails screen

## References

- **Primary:** arXiv:2606.09762 (AdamO / Dynamical Isometry)
- **Related:** arXiv:2306.05882 (Muon, original)
- **Landscape survey:** outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md (§1.2, Muon-OGD row)
- **Theory context:** CONTINUAL_LEARNING_THEORY.md, CEILING_ANALYSIS.md (conditioning decomposition, asymptotic floor)
