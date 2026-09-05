# Preregistration — does the Newton direction compose with gate removal?

**Written 2026-08-28, BEFORE any 60-task run of the combined cell.**
Development screening diagnostic; permanently nonpromoting. New path;
nothing existing under `outputs/ipmnist_screening/` is rewritten.

## Why this experiment

Two independent modifications of the *same* body update on the *same*
incumbent have each measured a real effect and each missed the frozen
+0.002 win bar on its own:

| modification | paired vs incumbent | record |
|---|---|---|
| feature-space Newton error direction (`resid_whiten=1, resid_newton=1`) | +0.001672 +/-0.000299 (3/3 seeds) | `precond_r1/`, this repo |
| gate removal (`gate_scale=0`) | +0.001712 +/-0.000174 (10/10 seeds) | `gate_ablation_r2/`, issue #1937 |

They have never been run together. They are orthogonal in construction:
one changes *which direction* the body's error signal points, the other
changes *how that direction is scaled* before it is applied (utility-gated
sigma-0 SGD vs plain decayed SGD).

The hypothesis with teeth is not "the effects add". It is that the utility
gate **attenuates** the preconditioned direction: the gate multiplies each
tensor's update by `1 - sigmoid(utility / global_max)`, so a rotated
gradient that points somewhere the gate considers low-value is damped
exactly where the rotation was supposed to help. If that is what is
happening, removing the gate should let more of the Newton direction
through and the combination should be **super-additive**. If the two are
simply independent, it should be additive. If the gate was in fact
protecting the body from an over-aggressive rotation, it should be
**sub-additive** — a real possibility, since the Newton direction is the
exact minimum-norm step and rank <= n_classes.

## Design — full 2x2 factorial, seeds 0-4, 60 tasks x 5000 steps

| | gate on | gate off |
|---|---|---|
| **plain error signal** | `rls_head_resid_l1_preset005` (baseline) | `..._nogate` |
| **Newton direction** | `..._tp` | `..._tp_nogate` (new) |

All four cells are rerun here on one runner at one source commit
(negative result #8: cross-runner bitwise prefix equality is false, so
paired comparisons must stay within one runner; and the merge guard
requires a single source provenance across shards).

Seeds 0-4 rather than the usual 0-2: the quantity of interest is an
*interaction*, whose standard error combines those of both main effects,
so it needs more power than either main effect did.

## Decision bars — FROZEN

Paired mean difference in average online accuracy vs the **remeasured
incumbent**, seeds 0-4:

| outcome | bar | consequence |
|---|---|---|
| **WIN** | diff >= **+0.002** AND all 5 seeds improve | escalate that arm to n=10, then to a 200-task, 20-seed confirmation |
| **TIE** | +0.0005 <= diff < +0.002 | reported inconclusive; NOT escalated |
| **LOSS** | diff < +0.0005 | valid rejection; recorded in `docs/evidence/negative-results.md` |

Identical to the bars used in `precond_r1/` and issue #52 / #1937. They are
not adjusted for the larger seed count: more seeds buy a tighter interval,
not an easier bar.

## The interaction readout is DESCRIPTIVE, not a gate

Define `interaction = diff(tp_nogate) - diff(tp) - diff(nogate)`, all
paired against the remeasured incumbent in this same run. It will be
reported with its standard error and classified as super-additive,
additive, or sub-additive. **No promotion or escalation decision depends on
it** — only the win bar above does. This is stated in advance so that a
favourable interaction cannot be retro-fitted into a reason to escalate an
arm that missed the bar.

## Pre-committed reading of the likely outcomes

- `tp_nogate` >= +0.002 with 5/5 seeds: WIN, escalate to n=10.
- `tp_nogate` lands in +0.0005..+0.002: TIE. This is the expected outcome
  if the two effects are NOT additive, and it closes the composition
  question rather than leaving it open. It will be recorded as such and
  the arm will NOT be escalated.
- `tp_nogate` <= `nogate` alone: the Newton direction does not survive gate
  removal; recorded as a negative result about the mechanism, not about
  the gate.

## Disclosure

A 2-task seed-0 smoke test was run to confirm all four cells execute and
stay finite, before the bars were fixed: incumbent 0.7324/0.8424, nogate
0.7770/0.8654, tp 0.6994/0.8356, tp_nogate 0.7516/0.8612. It was used only
as a liveness check. `precond_r1` established that this lane's 2-3 task
orderings **invert** by 60 tasks (negative result #20), so no weight is
placed on it and no arm was added, dropped, or retuned in response.

## Plasticity

Reported for every arm, and compared only WITHIN the rls_head family — for
these arms it measures the RLS head's one-step squared-error improvement,
not the cross-entropy improvement the MLP arms report.
