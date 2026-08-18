# V6 — Is recurrence unexploited? Measuring the headroom for direction D

Pre-registered in [elizaOS/asi#1875](https://github.com/elizaOS/asi/issues/1875)
(development diagnostic, permanently nonpromoting). Full numbers:
`V6_recurrence_headroom.json`; driver: `V6_recurrence_headroom_runner.py`.

## Verdict: H0 REFUTED — every arm benefits from recurrence

I registered H0 ("no existing arm exploits recurrence") and it is wrong. All
six ladder arms clear the pre-registered criterion — mean paired gap `> 0` with
all three seeds positive.

| arm | M1 `input_permutation` | M4 `recurrence` | gap | per-seed |
|---|---|---|---|---|
| `sgd_raw` | 0.2827 | 0.6696 | **+0.3869** | +0.3749, +0.4148, +0.3709 |
| `adamw` | 0.2281 | 0.5788 | **+0.3507** | +0.3409, +0.3740, +0.3371 |
| `upgd_raw` | 0.2797 | 0.3726 | +0.0929 | +0.1067, +0.0942, +0.0777 |
| `sgd_norm` | 0.6808 | 0.7270 | +0.0461 | +0.0510, +0.0423, +0.0452 |
| `gated_norm` | 0.6911 | 0.7383 | +0.0472 | +0.0516, +0.0425, +0.0475 |
| `naive_bayes` | 0.6052 | 0.6526 | +0.0474 | +0.0479, +0.0436, +0.0507 |

## Where the prediction went wrong

I reasoned: *"none of these arms has a mechanism that could index a
permutation, therefore none can exploit recurrence."* That conflates **having a
recurrence mechanism** with **benefiting from recurrence**.

No arm indexes permutations, and this result does not show that any does. But
every arm carries persistent weights, and when a mapping repeats those weights
are already partly fit for it. The benefit is implicit reuse, not recognition.

Stating it plainly because the distinction bounds every downstream claim:
**"exploits recurrence" here means "scores better when permutations repeat",
not "implements a recurrence mechanism".**

## The structure is the finding

| group | mean gap |
|---|---|
| unconditioned (`sgd_raw`, `adamw`) | **+0.3688** |
| conditioned (`sgd_norm`, `gated_norm`, `naive_bayes`) | **+0.0469** |
| ratio | **7.9x** |

Arms that are *bad* on fresh permutations gain enormously from repetition; arms
that are *good* on fresh permutations gain almost nothing. Input conditioning
already recovers most of what implicit reuse would give a learner — a
conditioned arm re-adapts fast enough that a repeat is barely cheaper than a
fresh permutation, while an unconditioned arm re-adapts slowly and a repeat
saves most of that cost.

This mirrors the IPMNIST campaign's central result — conditioning is the
load-bearing mechanism — measured here on the memory axis rather than the
input-shift axis.

## What this does and does not bound for direction D

**Measured.** With perfect recurrence structure — 5 permutations across 100
regimes, 20x reuse — a champion-like conditioned arm gains **+0.047**. That is
what implicit weight retention already captures, and it is the baseline any
explicit recurrence-indexing mechanism must beat rather than a starting point
of zero.

**Not bounded by this run.** This does *not* cap direction D. An explicit
mechanism could in principle jump straight to stored per-permutation state
rather than partially re-adapting, and would then beat implicit retention. The
distance from the best M4 arm to the Bayes ceiling is `0.9833 - 0.7383 =
0.2450`, but most of that is ordinary online estimation cost rather than
recurrence-specific headroom, so it is an upper bound only in the loosest sense.

The honest scoping statement: **the easy part of recurrence is already taken by
conditioning; a direction-D proposal needs to argue it beats a +0.047 implicit
baseline on conditioned arms, and this experiment does not tell us whether it
can.**

## Control

Run first and reported regardless, per the pre-registration:

| family | regimes | distinct permutations |
|---|---|---|
| M1 `input_permutation` | 100 | 100 — all fresh |
| M4 `recurrence` | 100 | **5** — exactly `recurrence_pool` |

`separated: True`. This mattered: had the families been secretly identical,
every gap would be zero and would have read as clean support for H0 — an
outcome indistinguishable from the experiment's own failure mode.

**Bayes ceiling is family-invariant at 0.9833** (`bayes_reference`, 200k Monte
Carlo samples, `mc_sem` 0.0003), identical for M1 and M4, as the module
documents: all transforms are bijections of the base distribution. So the gaps
are genuine learning gains, not a difference in attainable ceiling. I checked
this rather than assuming it, because a family-dependent ceiling would have
made every number above meaningless.

## Deviation

Wall clock **1911.6 s (32 min)** against the ~14 min estimated in the
pre-registration. The estimate came from timing `gated_norm`; the raw arms and
`naive_bayes` are slower. No protocol term changed — only my cost estimate was
wrong.

## Reproduction

```bash
python outputs/new_directions/V6_recurrence_headroom_runner.py \
  --out outputs/new_directions/V6_recurrence_headroom.json
```

36 runs (2 families x 6 arms x 3 seeds) plus the control, using the shipped
`micro_continual` runner with no new benchmark code. Environment: Python
3.12.14, jax 0.11.0, numpy 2.5.2, CPU, network-isolated container.
