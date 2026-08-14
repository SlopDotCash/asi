# Pre-registration: Shiftnorm Hyperparameter Interaction Screen (Wave 9)

**Date:** 2026-08-14  
**Status:** Pre-registered, awaiting execution  
**Related:** Extends the shiftnorm mini-star campaign documented in FINAL_REPORT.md

## Contribution type
**Climb** — Four hypothesis-driven arms screening against incumbent `sigma0_shiftnorm_d099` (0.86449 ± 0.00009).

## Lane and metric
- **Lane:** IPMNIST screening
- **Metric:** `average_online_accuracy`
- **Protocol:** 200 tasks × 5,000 steps (full confirmation) or 60 tasks (screen only)

## Background and motivation

The shiftnorm mini-star explored individual hyperparameter axes (shift_k, fast_decay, shift_refractory) but did not systematically test *interactions* between them. The frontier-2 decay star showed d098 ≈ d099 (statistically ties); earlier waves confirmed fast_decay at 0.9 and shift_k at 1.0 (default). This wave tests four interaction hypotheses:

### Hypothesis 1: Gentler detector + faster re-conditioning
**Arm:** `sigma0_shiftnorm_d099_k05_f08`
- **Config:** norm_decay=0.99, shift_k=0.5, fast_decay=0.8
- **Rationale:** Detector sensitivity (k) controls false-positive boundary detections. When combined with very fast fast_decay (0.8), a gentler detector (k=0.5) might reduce oscillation while still re-conditioning rapidly post-permutation. Single-axis fast_decay=0.8 showed −0.0016 (worse than 0.9), but interaction with gentler detector might restore the gain.
- **Prediction:** If interaction is synergistic: +0.0005 to +0.001. If fast_decay is universally harmful: −0.002.

### Hypothesis 2: Aggressive detector + rate-limiting trade-off
**Arm:** `sigma0_shiftnorm_d099_k2_r50`
- **Config:** norm_decay=0.99, shift_k=2.0, shift_refractory=50.0
- **Rationale:** More aggressive detector (k=2.0) was single-axis tested and lost (−0.0012). Rate-limiting resets (r=50) might prevent repeated trigger-noise at the boundary without the full r=200 dampening. Hypothesizes that k=2 loses because noisy boundary detection fires repeatedly; r=50 provides moderate dampening.
- **Prediction:** If rate-limiting salvages aggressive detection: +0.0005 to +0.001. If k=2 is fundamentally wrong: −0.001.

### Hypothesis 3: Decay-speed pairing
**Arm:** `sigma0_shiftnorm_d098_f08`
- **Config:** norm_decay=0.98, fast_decay=0.8
- **Rationale:** Frontier-2 showed d098 and d099 are nearly equivalent (0.86242 vs 0.86245, δ≈0). If they represent a shallow valley in the decay space, then *both* d098 and fast_decay=0.8 (each slightly suboptimal alone) might interact positively when paired. Tests whether the 0.98–0.99 plateau emerges from interaction, not from a single parameter.
- **Prediction:** If decay interaction is real: +0.0005 to +0.002. If each axis is independent: −0.002 (both suboptimal).

### Hypothesis 4: Minimal rate-limiting
**Arm:** `sigma0_shiftnorm_d099_r50`
- **Config:** norm_decay=0.99, shift_refractory=50.0
- **Rationale:** Single r=200 was tested; r=50 (2.5x reduction) tests whether moderate refractory is sufficient. Current design uses r=0 (no refractory); r=200 showed −0.0037. Hypothesis: r=50 provides a middle ground—rate-limiting real boundaries without over-suppressing legitimate shifts.
- **Prediction:** If modest refractory helps: +0.0005 to +0.001. If any refractory hurts: −0.0010.

## Baseline measurement
- **Arm:** `sigma0_shiftnorm_d099` (0.86449 ± 0.00009, n=20)
- **Source:** `outputs/ipmnist_screening/publication_runs/RESULTS.md`

## Screen and confirm plan

**Screen (60 tasks, seeds 0–2):**
- Run all four new arms + baseline on paired seeds
- Success criterion: Any arm with **all three seeds positive** and **paired mean delta >+0.002** advances to confirmation
- If all four fail screen: report to `NEGATIVE_RESULTS_LEDGER.md` with finding

**Confirm (200 tasks, seeds 0–2):**
- Run advancing arm(s) at full protocol
- Report mean ± spread; compare against incumbent

**Held-out validation (seeds 3–19):**
- If any arm confirms, rerun on held-out seeds to estimate generalization

## Success and failure criteria

| Outcome | Action |
|---------|--------|
| Any arm: paired mean >+0.0025, all seeds improve on screen | Promote to 200-task confirm |
| Confirm mean >+0.0009 (3× incumbent stderr) | New record candidate; test on held-out seeds 3–19 |
| Confirm mean >+0.0009 on held-out seeds | Eligible for RESULTS.md promotion |
| All screen results: no arm all-seeds-positive | Close wave; report in ledger with summary |
| Inconclusive (one arm at +0.0005, mixed seeds) | Re-screen with more seeds or higher bar |

## Commands
```bash
# Screen all four new arms (+ baseline for pairing)
for arm in sigma0_shiftnorm_d099_k05_f08 sigma0_shiftnorm_d099_k2_r50 \
           sigma0_shiftnorm_d098_f08 sigma0_shiftnorm_d099_r50; do
  for seed in 0 1 2; do
    .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name $arm --seed $seed --n-tasks 60 \
      --out outputs/ipmnist_screening/screen_wave9_${arm}_seed${seed} \
      --noise-mode step
  done
done

# Merge screen results
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/screen_wave9_*_seed*.json \
  --control-name sigma0_shiftnorm_d099 \
  --output outputs/ipmnist_screening/screen_wave9_summary.json
```

## Code changes
- **File modified:** `alberta_framework/benchmarks/ipmnist_screening.py`
- **Lines added:** 5478–5490 (shiftnorm_variants tuple extended with Wave 9 arms)
- **Registry impact:** Four new arms automatically registered; CLI accepts --config-name for each
- **Tests affected:** Existing bitwise-reduction tests remain unchanged (Wave 9 arms don't reduce)

## Timeline
- Screen (4 arms × 3 seeds × ~2 min per shard): ~24 minutes
- Merge/summary: ~2 minutes
- Confirm (advancing arms × 3 seeds × ~10 min): ~30–40 minutes if all four advance
- Total compute: ~1–2 hours wall-clock

## Notes

1. **Single axis at a time is insufficient.** Most earlier waves tested one hyperparameter, holding others constant. Interaction effects (two axes varying together) are genuinely distinct hypothesis tests and can have opposite signs from single-axis projections.

2. **Why these four specifically?** They target the remaining unexplored corners of the design space:
   - (k, f) interaction: neither individual axis showed big gains; interaction might unlock them
   - (k, r) interaction: k=2 loses alone; rate-limiting as compensating mechanism
   - (d, f) interaction: both d098 and f08 are boundary points; test plateau robustness
   - r-only: full r=200 showed loss; r=50 as minimal viable rate-limiting

3. **Fail-closed reporting.** If all four arms lose on screen, this validates that the shiftnorm space is locally optimal—a meaningful negative result worth recording in the ledger, as it bounds the remaining low-hanging fruit.

4. **Deviations from pre-registration that void the result:**
   - Tuning threshold after seeing screen numbers
   - Rerunning shards instead of reporting failure
   - Changing seed assignments mid-campaign
   - Comparing against held-out results before screening

## References

- **FINAL_REPORT.md:** Shiftnorm mini-star results (frontier wave, shift-triggered normalizer wave)
- **CEILING_ANALYSIS.md:** ~0.933 family ceiling analysis
- **NEGATIVE_RESULTS_LEDGER.md:** Entry 20, waveA verdict; entry 3, perturbation noise findings
