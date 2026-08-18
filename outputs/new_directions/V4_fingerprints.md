# V4 — Higher-order permutation fingerprints across a boundary

Pre-registered in [elizaOS/asi#1311](https://github.com/elizaOS/asi/issues/1311)
(development diagnostic, permanently nonpromoting). Full numbers:
`V4_fingerprints.json`; runner: `V4_fingerprints_runner.py`.

## Verdict: REFUTED — and the floor moves the wrong way

No control-valid fingerprint reaches the pre-registered bar (>90% of relevant
`var > 0.01` pixels within <=500 samples, min over the 9 seed x boundary
cells). The secondary measurement is the one that carries the F3 result:

**`N*` — smallest N where mean relevant-pixel accuracy crosses 0.90 —
is `> 2000` for every control-valid F3 fingerprint and both solvers.**

V1 measured `N* ~= 2000` for its best estimator. V4's second-order
fingerprints do not beat that floor; they do not reach it.

## F3 — pairwise correlation structure (both controls passed)

Mean / min relevant-pixel accuracy (`var > 0.01`), Hungarian, 9 cells:

| N | F3a row-sum | F3b top-k | F3c spectral |
|---|---|---|---|
| 50 | 0.0036 / 0.0000 | 0.0048 / 0.0000 | 0.0058 / 0.0000 |
| 200 | 0.0093 / 0.0041 | 0.0238 / 0.0080 | 0.0161 / 0.0040 |
| **500** | 0.0116 / 0.0081 | **0.0809 / 0.0549** | 0.0676 / 0.0120 |
| 2000 | 0.0268 / 0.0099 | 0.3647 / 0.2740 | 0.4983 / 0.0820 |

Greedy is worse than Hungarian throughout, and the gap widens with descriptor
dimensionality: -0.006 (F3a), -0.053 (F3b), -0.090 (F3c) at N=2000. This is
consistent with V1's "~0.1 worse" observation.

**The decisive comparison.** At the pre-registered operating point:

| estimator | relevant-pixel accuracy @ N=500 |
|---|---|
| V1 — marginal mean/var | 0.019 |
| **V4 — best second-order (F3b)** | **0.081** |
| V1 — class-conditional means | **0.785** |

Second-order structure beats bare marginals by ~4x and loses to V1's own
first-order class-conditional fingerprint by ~10x. The pre-registered
hypothesis was that richer descriptors would break the radial symmetry that
defeated marginal matching. They do — but the estimator cost of a 784x784
correlation matrix dwarfs the gain. At N=500 the matrix has rank <= 500 and
its entries are noise; the low-dimensional reductions concentrate faster than
the matrix, exactly as registered, and still land an order of magnitude below
the cheaper first-order fingerprint.

The oracle control licenses the sharp reading: with exact full-dataset
statistics all three F3 reductions recover **1.000**. The information is
present and the reductions are sound. **Second-order structure is unusable at
this budget** — not unusable in principle.

## F4 — model-side activation coupling: VOID, not refuted

F4a and F4b **failed the pre-registered oracle gate** (0.002 / 0.000 against a
0.95 bar). Per the pre-registration this makes them mis-implemented rather
than uninformative, so their online numbers are recorded in the artifact,
marked `void` in `control_verdict`, and have `N*` marked `void` rather than a
sample-floor result. They are **not** reported as evidence about model-side
probes.

The two controls together diagnose the fault exactly:

| control (exact statistics) | F4a | F4b |
|---|---|---|
| oracle — permutation differs | 0.002 | 0.000 |
| no-shift — identity map | **1.000** | **1.000** |

F4 recovers the identity mapping perfectly and the true permutation not at
all. That is the signature of a descriptor that encodes **position, not pixel
content**:

```
a_h = relu( sum_k x_k * w1[k,h] )   =>   corr(x_j, a_h) ~ w1[j,h] * var(x_j) + cross-terms
```

The coupling of input position `j` to hidden unit `h` is dominated by the
weight *at position j*. Both sides' descriptors therefore reduce to rows of
`w1`, and matching them recovers `j -> j` regardless of the permutation.

The pre-registered premise — "activations whose coupling to each input pixel
is a fingerprint of where that pixel used to live" — has the direction
backwards. A shared network cannot report where a pixel *used to be*, because
the pixel's influence flows through whatever weight occupies its *current*
address. Any working model-side probe must break that symmetry, e.g. by
scoring post-shift content against the *reference-side weights* rather than
correlating both sides through the same forward pass.

**Supporting diagnostic (pre-registered):** the first-layer activation
covariance has participation-ratio effective rank **6.55-6.82** across all
cells. The nominal 300 dimensions of F4a were never 300; even without the
position confound the family carried under 7 effective dimensions.

## Controls

| control | gate | F3a | F3b | F3c | F4a | F4b |
|---|---|---|---|---|---|---|
| exact-statistic oracle | >= 0.95 | 1.000 | 1.000 | 1.000 | 0.002 | 0.000 |
| no-shift | >= 0.99 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Both controls use exact full-dataset statistics. They are pipeline-correctness
checks, so running them at the online checkpoints would conflate estimator
noise with an implementation fault — an error in the first draft of the
runner, corrected before any headline number existed.

The shipped runner computed the first seed/boundary's online diagnostic rows
before computing its controls, rather than literally running the controls
first as the preregistration required. The F3 controls passed, so this ordering
does not change its deterministic measurements, but it is an execution-order
deviation. The F4 controls failed and those arms remain void regardless of
their recorded online rows.

## Consequence for the chain

- **V2 remains gated out.** Its premise requires a promoted identification
  step; none exists.
- **The F3 family closes.** Second-order structure costs more samples than it
  saves at the protocol's operating point, with `N* > 2000` measured.
- **F4 is not closed — it was never tested.** The pre-registered construction
  is confounded. Closing model-side probes would require a corrected probe and
  a fresh pre-registration; this run licenses no claim about them.
- Per V1's standard, a fingerprint only matters if it beats the ~2,000-sample
  information floor. Nothing here does, so the transient headroom quoted in
  `CEILING_ANALYSIS.md` remains unrealisable by this route.

## Reproduction

```bash
python outputs/new_directions/V4_fingerprints_runner.py \
  --data-home <openml cache> \
  --out outputs/new_directions/V4_fingerprints.json
```

360 cells (9 seed x boundary x 4 checkpoints x 5 fingerprints x 2 solvers)
plus both controls in **128.6 s**. V1 was 14.8 s; the increase is the online
training of the reference network required by F4, which V1 had no equivalent
of. Environment: Python 3.12.14, jax 0.11.0, numpy 2.5.2, CPU, network-isolated
container.

`sigma0_ndecay099` supplies the reference network, resolved on the issue
before any numbers existed: it is the arm whose plain annealed fast-EMA
(decay 0.99) is the pre-registered reference, and using the shift-triggered
champion would have trained the probe under a different input normalizer than
the reference statistics use.
