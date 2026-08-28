# Preregistration — preconditioned residual signal on the RLS-head incumbent

**Written 2026-08-27, against `origin/main` (c9aba7b5), BEFORE any 60-task
run of the candidate arms.** Development screening diagnostic
(`development_screening_diagnostic`); permanently nonpromoting. New path;
nothing existing under `outputs/ipmnist_screening/` is rewritten.

## Hypothesis

`CEILING_ANALYSIS.md` splits the champion's error into a ~0.037
re-adaptation transient and a ~0.029 within-task convergence shortfall
(plateau 0.904 vs the 0.933 family asymptote). The RLS head cashed in the
**readout's** share of the shortfall — a closed-form readout converges in
~d samples rather than thousands — and that is what took the standing best
to 0.87114 (200 tasks, n=20). The **body's** share is untouched: the body
is still utility-gated SGD driven by an unconditioned feature-space error
`g = wout @ err`.

Every preconditioner previously screened against that share (IDBD,
Autostep, Adam+CBP) failed on *continual stability*, not on speed. The arms
below precondition the body's error direction using only state the
incumbent already maintains and that the shift detector already resets at
task boundaries, so they add no new cross-permutation carry.

All preconditioned arms renormalize the direction back to `||g||`, so the
step size stays on its frozen calibration (negative result #1: learning
rates do not transfer across update geometries).

## Arms (paired against the incumbent, seeds 0-2, 60 tasks x 5000 steps)

| arm | mechanism |
|---|---|
| `rls_head_resid_l1_preset005_gn` | activation whitening, `precond = p @ g`, alpha=1 |
| `rls_head_resid_l1_preset005_gn05` | activation whitening, alpha=0.5 |
| `rls_head_resid_l1_preset005_tp` | feature-space Newton, `precond = wout @ gram^-1 err`, alpha=1 |
| `rls_head_resid_l1_preset005_tp05` | feature-space Newton, alpha=0.5 |
| `rls_head_resid_l0999_pcap` | untested head 2x2 cell: forgetting 0.999 under BOTH wind-up guards, carrying the residual body signal |

`gn` is explicitly a heuristic, not a Gauss-Newton step: `p` is the
activation second moment, whereas the loss curvature with respect to `phi`
is `wout @ wout.T`. `tp` is the actual feature-space Newton /
target-propagation direction. Both are screened because the correct
curvature is rank <= n_classes and may be too aggressive.

Controls remeasured on THIS runner (negative result #8 — cross-runner
bitwise prefix equality is false; paired comparisons must stay within one
runner):

- `rls_head_resid_l1_preset005` — the incumbent, and the paired baseline.
- `sigma0_shiftnorm_d099` — the champion, anchoring to the standing numbers.

## Verified no-op (before this file was written)

The mechanism is a build-time branch inside `_make_rls_head_learner`.
Against pristine `origin/main`, 3 tasks x 5000 steps, seed 0, same runner,
the per-task accuracies of `rls_head_resid_l1_preset005`,
`sigma0_shiftnorm_d099`, and `rls_head_resid_l1_preset005_nogate` are
**bitwise identical** with and without the patch. An explicit
`resid_whiten = 0.0` spec also reproduces the incumbent bitwise.

## Decision bars — FROZEN

Paired mean difference in average online accuracy vs the **remeasured
incumbent**, seeds 0-2:

| outcome | bar | consequence |
|---|---|---|
| **WIN** | diff >= **+0.002** AND all 3 seeds improve | escalate the single best arm to a 200-task, 20-seed confirmation |
| **TIE** | +0.0005 <= diff < +0.002 | reported inconclusive; NOT escalated |
| **LOSS** | diff < +0.0005 | valid rejection; recorded in `docs/evidence/negative-results.md` |

Confirmation bar, frozen now: at 200 tasks x 20 seeds, paired mean diff vs
the remeasured incumbent >= **+0.002** with all seeds improving to claim a
new standing best.

These bars are the lane's established precedent: issue #52 used win +0.002 /
tie +0.0015 on this same 60-task, 3-seed instrument, and `gate_ablation_r2`
re-ran that arm at n=10 and still reported +0.0017117 as inconclusive rather
than moving the bar. The paired stderr on this instrument is ~0.00023, so
+0.002 is ~9 sigma — the bar is scientific, not statistical.

**A failed gate is a valid rejection and will not be retuned. Seeds 0-2 are
the screen; confirmation seeds are 0-19 as in
`summary_rls_head_confirm.json`.**

## Disclosure

A 3-task seed-0 diagnostic was run before these bars were fixed, as the
lane's standard constant-freezing practice (see the registry comments for
the existing rls_head waves). Per-task accuracies:

| arm | t1 | t2 | t3 |
|---|---|---|---|
| incumbent | 0.7324 | 0.8424 | 0.8558 |
| `_gn` | 0.7224 | 0.8398 | 0.8546 |
| `_tp` | 0.6994 | 0.8356 | 0.8572 |
| `_l0999_pcap` | 0.7774 | 0.8704 | 0.8744 |

It was NOT used to choose the bars — those are copied from the lane's
precedent — and no arm was added, dropped, or retuned in response to it.
All five arms were registered before it was run.

## Plasticity reporting

`average_plasticity_mean` is reported for every arm. It is **not comparable
across the MLP and RLS-head families**: for MLP arms it is the one-step
cross-entropy improvement from a full parameter update, while for every
`rls_head*` arm it is the one-step improvement in the head's squared error
from the RLS update alone (the factory docstring states this). The ~66x
"plasticity drop" of the RLS incumbent versus the champion is therefore
largely a metric-definition artifact, not a measured collapse. Plasticity
comparisons here are made only WITHIN the rls_head family.
