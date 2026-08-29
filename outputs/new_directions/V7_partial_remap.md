# V7 — Does PARTIAL permutation identification buy anything? (preregistration)

**Written 2026-08-28, BEFORE any run.** Development diagnostic, permanently
nonpromoting. Runner: `V7_partial_remap_runner.py`; numbers will be written to
`V7_partial_remap.json`.

## The gap this closes

Every gate in the identification chain is an **identification-accuracy** gate:

- V1 pre-registered ">90% of relevant pixels within <=500 samples" and measured
  0.785 at N=500, 0.840 at N=2000 -> refuted.
- V4 pairwise fingerprints missed the same gate wider (0.081 at N=500),
  `N* > 2000` for every reduction.
- V5 independently confirmed the floor at `N* = 1978.5`.

Nobody has measured the **utility** of a partially-correct answer. "Can we name
the permutation?" and "does an 80%-correct input remap help a learner that
would otherwise re-learn its input layer?" are different questions, and only
the first has been asked. V1 itself frames the negative result in terms of
matching accuracy, not downstream metric.

## Design — an ORACLE UPPER BOUND, no identifier is built

For each task `t >= 1` the learner is fed a partially corrected input layout.
The protocol runner computes `x[j] = base[permutation[j]]`, so a perfect
oracle remap is exactly `permutation = pi_0` (every task presented in task 0's
layout). A partial oracle at accuracy `p` places a random fraction `p` of
**relevant** pixels (base variance > 0.01, V1's definition, ~507/784) at their
task-0 positions and scrambles the remainder, keeping the result a valid
permutation. Irrelevant pixels are left to the scramble, exactly as V1 argues:
constant-background pixels are mutually interchangeable.

Identification is granted **free and instantly at step 0 of every task**. This
is impossible in practice — a real identifier needs N samples, during which the
ordinary transient is paid in full — so every number here is a strict upper
bound on what an identifier of that accuracy could deliver. **That is the
point: if the free instant oracle does not pay, no achievable identifier can.**

## Arms

Learner: `rls_head_resid_l1_preset005` (the standing incumbent).
60 tasks x 5000 steps, seeds 0-2, one runner.

| arm | permutations fed |
|---|---|
| `control` | the true schedule (the ordinary protocol) |
| `p000` | p = 0.00 (scramble; sanity check that it matches `control`) |
| `p062` | p = 0.62 — achievable at N=200 (V1: 0.619) |
| `p079` | p = 0.79 — achievable at N=500 (V1: 0.785) |
| `p090` | p = 0.90 — the accuracy V1's gate DEMANDED but never reached at N<=500 |
| `p100` | p = 1.00 — perfect identification; the carried-oracle anchor |

`p100` should approach the carried-oracle regime of `CEILING_ANALYSIS.md` (1b),
where the champion family plateaus near 0.933 with the permutation held fixed.
That is a built-in sanity anchor: if `p100` does not land far above the
control, the construction is wrong and every other cell is void.

## Pre-committed predictions

1. **`p000` reproduces `control`** within noise. If it does not, the
   construction is broken and the run is void.
2. **`p100` >> `control`**, approaching the carried-oracle regime.
3. **`p079` buys less than +0.010** over `control`. This is the prediction that
   matters. Reasoning: the residual transient is 0.0366, of which 0.0223 sits
   in the first 500 steps — which a real identifier has already paid before it
   can answer — so only the ~0.0143 tail is addressable, and a 21%-wrong layout
   should not capture all of it.
4. If **`p079` >= +0.015**, the identification family is worth reopening and I
   was wrong to recommend closing it.

Recording 3 and 4 in advance because the whole point of this probe is to be
able to close the family honestly, and a prediction written afterwards could
not do that.

## What this probe does NOT do

It does not attempt identification, so it says nothing about whether accuracy
`p` is reachable at any sample budget — V1/V4/V5 already answered that. It
grants the answer for free and measures only what the answer is worth. A
delayed-onset variant (remap applied from step N rather than step 0) is
deliberately NOT run here: delay strictly reduces the benefit, so the instant
version already bounds it. If and only if `p079` clears +0.015 does the delayed
variant become worth running.
# V7 — RESULTS (appended after the preregistered run)

## Run A (v7_parts/): VOID under prediction 1

The first execution pinned background pixels to their task-0 positions,
making `p000` a background-pinned stream rather than the protocol's uniform
draw. The preregistered null check caught it: `p000` measured -0.0142 vs
control, so run A is void in full, per the preregistration's own terms. Raw
artifacts retained in `v7_parts/`; the defect and fix are documented in the
runner docstring.

## Run B (v7b_parts/): VALID — null holds, prediction 3 REFUTED

60 tasks x 5000 steps, seeds 0-2, learner `rls_head_resid_l1_preset005`,
507/784 relevant pixels (matches V1).

| arm | acc | vs control | all seeds + |
|---|---|---|---|
| control | 0.869503 | — | — |
| p000 | 0.869906 | +0.000402 ± 0.000451 | (null: HOLDS) |
| p062 | 0.917826 | +0.048322 ± 0.000288 | yes |
| p079 | 0.925607 | +0.056103 ± 0.000694 | yes |
| p090 | 0.927529 | +0.058026 ± 0.000291 | yes |
| p100 | 0.931139 | +0.061636 ± 0.000153 | yes |

- **Prediction 1 (p000 == control): HELD.** The run is valid.
- **Prediction 2 (p100 anchor): HELD.** 0.9311 vs the carried-oracle 0.9324
  measured by a completely different route in CEILING_ANALYSIS (1b).
- **Prediction 3 (p079 < +0.010): REFUTED by 5x.** Measured +0.0561.
- **Prediction 4 (>= +0.015 reopens the family): TRIGGERED.**

## The finding

**V1's promotion gate was miscalibrated against the downstream objective.**
V1 demanded 90% relevant-pixel accuracy within 500 samples and measured
0.785 -> refuted. But 62% accuracy — which V1 measured as achievable at
N=200 — already lifts the incumbent to 0.9178, above the campaign's 0.90
target. The gate was ~30 points stricter than the problem requires, so the
identification direction was closed against a threshold the mechanism never
needed to meet. V1's *measurements* stand (identification accuracy vs N is
untouched); its *verdict-to-target mapping* does not.

Utility is also strongly concave in identification accuracy: the first 62%
of pixels buys +0.048, the last 38% only +0.013 more. Partial answers are
where nearly all the value lives.

## Bounds and caveats

- Oracle grants identification FREE at step 0 of each task; every number is
  a strict upper bound on a real identifier at that accuracy. The delayed
  variant (V8) measures the achievable version.
- Development diagnostic, seeds 0-2, permanently nonpromoting. No claim
  beyond the consumed seeds.

## V8 — the delayed-onset variant (honest identifier bound)

Runner `V8_delayed_remap_runner.py`, parts in `v8_parts/`. Each task runs the
TRUE permutation for its first N steps — the ordinary transient is paid in
full — then switches to the partial remap, with (N, p) pairs taken from V1's
own measurements of what is achievable at that budget.

| arm | acc | vs control | all seeds + |
|---|---|---|---|
| control | 0.869503 | — | — |
| N=200, p=0.62 | **0.899660** | +0.030157 ± 0.000112 | yes |
| N=500, p=0.79 | 0.898061 | +0.028558 ± 0.000238 | yes |
| N=2000, p=0.84 | 0.874319 | +0.004816 ± 0.000167 | yes |

Two findings:

1. **Timing dominates accuracy.** Identifying at 200 samples with 62%
   accuracy beats identifying at 500 with 79%, and identifying at 2000 with
   84% is nearly worthless — by then the gradient transient it would have
   replaced has already been paid. This inverts V1's implicit assumption
   that higher identification accuracy is the goal.
2. **An honest single-shot identifier reaches 0.8997** — 0.0003 below the
   0.90 target. This bound is conservative: the V8 oracle switches once and
   never improves, whereas a real identifier re-matches as post-shift
   samples accumulate (0.62 at 200 -> 0.79 at 500 -> 0.84 at 2000), riding
   the upper envelope of these arms.

Consequence: a real in-protocol identifier arm is now justified as the next
implementation step. Development diagnostic, seeds 0-2, permanently
nonpromoting.
