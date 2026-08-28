# Results — preconditioned residual signal on the RLS-head incumbent

60 tasks x 5000 steps, seeds 0-2, all shards on one runner
(negative result #8). Paired baseline: `rls_head_resid_l1_preset005`,
remeasured here. Bars frozen in `PREREGISTRATION.md` before any 60-task run.
Development screening diagnostic; permanently nonpromoting.
Source commit `3818d2b7`; merged summary `summary.json`.

## Outcome: NO ARM CLEARED THE WIN BAR. Nothing is escalated.

| arm | avg online acc | paired vs incumbent | all seeds improve | verdict |
|---|---|---|---|---|
| `..._tp` (feature-space Newton, a=1) | 0.871176 | **+0.001672** +/-0.000299 | yes | **TIE** |
| `rls_head_resid_l0999_pcap` | 0.870089 | +0.000586 +/-0.000490 | no | **TIE** |
| `..._tp05` (Newton, a=0.5) | 0.869514 | +0.000011 +/-0.000088 | no | **LOSS** |
| `rls_head_resid_l1_preset005` (baseline) | 0.869503 | — | — | — |
| `..._gn05` (activation whitening, a=0.5) | 0.869224 | -0.000279 +/-0.000102 | no | **LOSS** |
| `..._gn` (activation whitening, a=1) | 0.868042 | -0.001461 +/-0.000587 | no | **LOSS** |
| `sigma0_shiftnorm_d099` (champion) | 0.864129 | -0.005374 +/-0.000073 | no | — |

Sanity check on the instrument: the incumbent beats the champion by
+0.005374 here, against +0.005440 in main's own `replication_r1/` on
different hardware, and +0.006653 at the 200-task / 20-seed horizon. The
runner and the standing numbers agree.

## What was actually learned

**1. The direction matters, and the theoretically correct one is the one
that works.** The two preconditioners are monotone in the interpolation
weight `a` and have OPPOSITE signs:

| a | activation whitening (`gn`) | feature-space Newton (`tp`) |
|---|---|---|
| 0.0 | 0 (the incumbent, bitwise) | 0 (the incumbent, bitwise) |
| 0.5 | -0.000279 | +0.000011 |
| 1.0 | -0.001461 | +0.001672 |

Whitening the body's error by the head's *activation* second moment `p` is
actively harmful, and the harm grows with `a`. Rotating toward the true
feature-space curvature (`wout @ gram^-1 err`, gram = `wout.T @ wout`)
helps, and helps more at full strength. Since both arms are identical apart
from which matrix preconditions the same vector, and both reduce bitwise to
the incumbent at `a = 0`, this isolates the effect to the choice of metric.
It is a direct confirmation that `p` was the wrong quantity — the loss
curvature with respect to `phi` is `wout @ wout.T`, not the activation
second moment.

**2. `tp` is real but an order of magnitude too small.** +0.00167 with all
three seeds improving, se 0.00030. It is a genuine effect, not noise. It is
also ~6% of the 0.0289 needed to reach 0.90.

**3. Short diagnostics mislead in this lane, again.** At 3 tasks
`rls_head_resid_l0999_pcap` led the incumbent on every task
(0.7774/0.8704/0.8744 vs 0.7324/0.8424/0.8558) and `tp` trailed on the
average. At 60 tasks the ordering reverses: `pcap` collapses to +0.00059
with one seed negative, and `tp` is the best arm. The `lambda < 1` "wins
short diagnostics, then fades" pattern of negative result #10 reproduces at
the 60-task horizon even with BOTH wind-up guards active (trace cap 1e4 and
the detector-driven P reset) — the cap does prevent the overflow collapse,
it just does not make forgetting pay.

**4. Plasticity did not move.** Every rls_head arm sits at 0.0053-0.0178
against the champion's 0.2949, and late-window slopes are all ~-3e-4,
statistically indistinguishable across arms. The preconditioners rotate the
body's error direction without changing how much the head adapts per step,
which is what the norm-preserving rescaling was designed to guarantee.

## Not escalated, and the bar is not moving

`tp` at +0.00167 lands in the same ambiguous band as issue #52's gate
ablation (+0.00156 at n=3, +0.00171 at n=10 in `gate_ablation_r2/`). That
precedent is directly on point: adding seeds to a point estimate already
below the bar only tightens the interval around a value below the bar. No
arm is escalated to the 200-task confirmation, and the bars stand as frozen.
