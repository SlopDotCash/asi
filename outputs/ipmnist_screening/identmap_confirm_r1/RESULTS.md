# Results — 200-task, 20-seed confirmation of the identifier arm

Frozen bars: PREREGISTRATION in `../identmap_r1/`. Both arms run on one
runner at source commit 626b5605; merged `summary.json`. Development
screening diagnostic; permanently nonpromoting; seeds 0-19 are the lane's
consumed development seeds (same instrument as `summary_rls_head_confirm`).

## Outcome: CONFIRMED — new development standing best

| arm | avg online acc | paired | all seeds |
|---|---|---|---|
| `rls_head_resid_identmap200_r` | **0.909118 ± 0.000886** | **+0.038040 ± 0.000902** | 20/20 |
| `rls_head_resid_l1_preset005` | 0.871078 ± 0.000104 | — | — |

- The remeasured incumbent reproduces the standing 0.8711435 to 7e-5 on
  this hardware — the instrument is sound.
- Weakest of 20 seeds: +0.028669. The confirmation bar was +0.002.
- The paired effect GREW from the 60-task screen (+0.0352) to the 200-task
  horizon (+0.0380): the mechanism is per-boundary (re-earned at each of
  the 199 shifts) while its one-time cost (task-0 reference building) is
  amortized. This is the opposite behaviour of the composition arm, whose
  cumulative effect decayed 3.46x (negative result #21).
- Late-window slope -5.6e-05 (flat): no late-life drift over 200 tasks.
- Plasticity 0.00451 vs the incumbent's 0.00448: unchanged.

## Claim, stated at its exact strength

At the ICLR-2024 IPMNIST development protocol (200 tasks x 5000 steps, one
example per step, 300x150 MLP), the identifier arm crosses the 0.90
"transient-solved" line of `CEILING_ANALYSIS.md`: 0.9091 against the
champion-family stationary asymptote of 0.933 and the plateau of ~0.904.
Post-shift accuracy is recovered by identifying the permutation from V1's
online class-conditional fingerprint (~200 samples) and remapping the
input, instead of re-learning the input layer by gradient descent
(~2000-4000 samples). The remaining ~0.024 to the family asymptote is the
within-task convergence shortfall, which this mechanism does not touch.

This is a development-grade result on consumed development seeds. It is
not promotable evidence, not `reference-dev`, and no scientific protocol
has been run. The mechanism consumes labels post-prediction (the protocol
permits this and V1 preregistered it as legal).
