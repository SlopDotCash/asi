# V5 — Model-side identification against reference-side weights

Pre-registered in [elizaOS/asi#1870](https://github.com/elizaOS/asi/issues/1870)
(development diagnostic, permanently nonpromoting). Full numbers:
`V5_model_side.json`; runner: `V5_model_side_runner.py`.

## Verdict

Two separable outcomes:

1. **V1's information floor is independently confirmed.** The data-side control
   arm crosses 0.90 at **`N* = 1978.5`**, against V1's reported "~2,000-sample
   information floor" — measured in a different runner, with a stronger
   post-shift estimator.
2. **The model-side arms are VOID again** — for a different reason than V4:
   these reference and post-shift descriptors are not commensurable.

Ledger entry 15 therefore stays open as *untested*, not closed. This run does
not refute model-side probes; it records a second failed construction and a
constraint that future constructions must address.

## F5c — data-side control (passed both controls)

Mean / min relevant-pixel accuracy (`var > 0.01`), 9 cells:

| N | Hungarian mean / min | V1 published |
|---|---|---|
| 50 | 0.2063 / 0.1187 | 0.197 |
| 200 | 0.6527 / 0.5800 | 0.619 |
| 500 | 0.8044 / 0.7840 | 0.785 |
| 2000 | 0.9014 / 0.8604 | 0.840 |

Greedy is worse throughout (0.8160 vs 0.9014 at N=2000), consistent with V1's
"~0.1 worse" observation.

**This is V1's fingerprint, not V1 verbatim.** The pre-registration fixed the
post-shift descriptor as *batch* class-conditional means; V1 used a fresh
annealed fast-EMA. The two agree closely at N=50-500, where an EMA with a
~100-sample effective window is near the batch mean, and diverge at N=2000
(0.9014 vs 0.840), where the batch estimator keeps all 2,000 samples and the
EMA does not. So F5c is a slightly *stronger* estimator than V1's, which makes
its floor a conservative bound.

**`N* = 1978.5` (Hungarian), `> 2000` (greedy).** V1 reported ~2,000. An
independent runner, a different post estimator, and a different implementation
land on the same floor.

## F5a / F5b — model-side arms: VOID

Both failed the pre-registered oracle gate at **0.248** against 0.95.

| control (exact statistics) | F5a | F5b | F5c |
|---|---|---|---|
| oracle — permutation differs | 0.248 | 0.248 | **1.000** |
| no-shift — identity map | 0.248 | 0.248 | **1.000** |

Their online values are recorded in the artifact and marked `void`; they are
**not** evidence about model-side probes.

### Why this is not V4's failure

V4's F4 scored 1.000 on no-shift and 0.002 on the oracle — the signature of a
descriptor encoding position. V5's arms score **identically with and without a
permutation**, so position is not the confound. The descriptors simply do not
identify pixels: 0.248 of relevant pixels with *exact* statistics on both
sides, well above chance (~0.002) but far below the gate.

### Why these descriptors are not commensurable

The reference descriptor is the model's per-position class affinity — a
quantity in **weight space**, "how does input position `i` drive class `c`".
The post descriptor is class-conditional pixel intensity — a quantity in
**data space**, "how bright is position `j` on average for class `c`". For a
trained network these correlate, which is why 0.248 is not chance, but they are
not the same object and their standardized distance is not a reliable identity
score.

The reusable constraint for a follow-up is narrower than a family-level
conclusion:

> These model-side quantities are indexed by the model's **input position**,
> which is the pre-shift layout. Scoring them against post-shift evidence needs
> a commensurable post-shift quantity; the content statistic used here is not
> commensurable with either weight-space descriptor.

For this construction, improving the model-side **reference** does not change
the post-shift estimator, which still sees roughly N/10 samples per class. That
explains the registered H0 and why F5c's floor lands where V1's did, but it does
not establish that every possible model-side probe must use the same estimator.

## Controls

Computed for every arm **before any online checkpoint**, as pre-registered —
correcting V4, whose runner computed the first cell's online rows first and had
to record that as an execution-order deviation. Control cell: seed 0,
boundary 0.

The oracle control uses **exact** class-conditional means on the data side.
Using the online annealed EMA there was a mis-specification in the first draft
of this runner: an EMA with decay 0.99 has a ~100-sample effective window and
never converges to a full-dataset mean, so it reported 0.406 and would have
marked the V1 control arm void by construction. Caught and fixed before any
headline number existed; the corrected control reproduces V1's 100%
class-conditional oracle exactly.

## Consequence for the chain

- **Ledger entry 15 stays open.** Model-side probes remain untested, now across
  two independent constructions. This run identifies a commensurability
  requirement for the next construction, not a refutation.
- **V1's floor is confirmed at `N* = 1978.5`** in an independent runner with a
  stronger post estimator.
- **V2 remains gated out.** No identification step has been promoted.
- Any future model-side attempt must state, in advance, what post-shift
  quantity it scores against and why that quantity is not itself the binding
  estimator.

## Reproduction

```bash
python outputs/new_directions/V5_model_side_runner.py \
  --data-home <openml cache> \
  --out outputs/new_directions/V5_model_side.json
```

216 cells (9 seed x boundary x 4 checkpoints x 3 arms x 2 solvers) plus both
controls in **73.3 s**. Environment: Python 3.12.14, jax 0.11.0, numpy 2.5.2,
CPU, network-isolated container. Reference network: `sigma0_ndecay099`, trained
online through tasks 0..boundary.
