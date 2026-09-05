# Preregistration — online permutation identification + input remap

**Written 2026-08-29, BEFORE any 60-task run of the candidate arms.**
Development screening diagnostic; permanently nonpromoting. New path.

## Mechanism and evidence chain

V7 (oracle, valid run B): a partial input remap free at step 0 is worth
+0.048 at p=0.62 to the incumbent; utility is concave in accuracy.
V8 (delayed oracle): the same remap delivered at N=200 post-shift samples
with V1's measured achievable accuracy (0.62) is worth +0.030 and reaches
0.8997; timing dominates accuracy.

The arms implement the real mechanism inside the screening protocol: V1's
class-conditional + marginal fingerprint accumulated online (labels
consumed post-prediction — protocol-legal), the champion's own shift
detector run on the raw stream (constants unchanged), Hungarian assignment
via a host callback at the preregistered sample counts, prediction and
learning on `x[remap]`.

| arm | matches at |
|---|---|
| `rls_head_resid_identmap200` | N=200 (V8's best single-shot cell) |
| `rls_head_resid_identmap200_r` | N=200, 500, 2000 (rides V1's accuracy curve) |

Control: `rls_head_resid_l1_preset005` (the incumbent), remeasured in this
screen (source commit changed; provenance-valid merges require same-source
shards, and negative result #8 requires same-runner pairing anyway).

Reduction pin (verified before this file): `ident_match_at = 0` delegates
verbatim to the incumbent factory; 3-task seed-0 per-task accuracy is
bitwise identical. Task 1 of the identmap arms is bitwise the incumbent's
task 1 (no reference frozen yet), which doubles as an in-run null.

## Decision bars — FROZEN (lane precedent, unchanged)

Paired vs the remeasured incumbent, seeds 0-2, 60 tasks x 5000 steps:
WIN >= +0.002 with all seeds improving -> escalate the better arm to the
200-task, 20-seed confirmation; TIE +0.0005..+0.002; LOSS below.

Confirmation bar (frozen now): 200 tasks, seeds 0-19, paired vs the
remeasured incumbent >= +0.002 with all seeds improving claims a new
standing best. The 0.90 line is reported descriptively at both horizons;
crossing it is NOT a gate and cannot rescue a failed bar.

## Predictions (recorded before the screen)

1. Both arms WIN by a wide margin (V8 bounds ~+0.030; the smoke's shifted
   tasks ran ~+0.035).
2. The refining arm >= the single-shot arm.
3. Horizon risk, stated honestly: precond_r2 measured a 3.46x decay of a
   different mechanism's effect from 60 to 200 tasks. This mechanism's
   effect is per-boundary (re-paid at every shift) rather than cumulative,
   so I predict much weaker decay — but the 200-task confirmation, not this
   screen, is the arbiter.
4. Failure condition: if the in-loop identifier's matching accuracy falls
   materially below V1's batch-estimated 0.62 at N=200 (EMA reference
   drift, detector timing), the gain halves or worse.

## Disclosure

A 3-task seed-0 smoke ran before these bars were fixed (liveness + pin):
incumbent 0.7324/0.8424/0.8558; identmap200 0.7324/0.8796/0.8866;
identmap200_r 0.7324/0.8818/0.8900. Bars are the lane's standing precedent
and were not chosen from it. Negative result #20 (3-task readings invert)
is why the screen, not the smoke, decides.
