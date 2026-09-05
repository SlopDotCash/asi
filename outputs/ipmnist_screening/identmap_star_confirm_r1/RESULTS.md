# Results — 200-task, 20-seed confirmation of the N=50 first-match arm

Bars frozen in `../identmap_star_r1/PREREGISTRATION.md` before the 60-task
screen. Both arms on one runner; merged `summary.json`. Development
screening diagnostic; permanently nonpromoting; consumed development seeds
0-19.

## Outcome: CONFIRMED — development best moves again

| arm | acc (n=20) | paired | all seeds |
|---|---|---|---|
| `rls_head_resid_identmap50_r` | **0.916569 ± 0.001119** | **+0.007451 ± 0.000241** | 20/20 |
| `rls_head_resid_identmap200_r` | 0.909118 ± 0.000886 | — | — |

- The control reproduces its own prior confirmation (0.909118) exactly —
  the arm is deterministic per seed and its code is untouched between the
  two commits, so identical trajectories are the expected behaviour and a
  live instrument check.
- Weakest seed +0.004904 against the +0.002 bar.
- The paired effect grew from 60 tasks (+0.006721) to 200 (+0.007451):
  the per-boundary amortization signature, second occurrence.
- Late-window slope flat; plasticity unchanged.

## Mechanism reading, at exact strength

A ~20%-accurate Hungarian match applied 50 samples after the detected
boundary, refined at 200 and 2000, strictly beats waiting 200 samples for
a ~62%-accurate first match — monotone across N in {50, 100, 200} at both
horizons. Utility concavity (V7) plus transient front-loading (V8) explain
the direction; the preregistered accuracy-floor failure mode did not
materialize at any probed accuracy. The trend has not plateaued at N=50;
matching at the detector trigger itself is the natural next probe and is
NOT run here.

Cumulative development-best movement this campaign wave:
0.871078 -> 0.909118 -> 0.916569 (total +0.0455 against the incumbent
measured on this runner; the 0.933 family asymptote is now 0.016 away).
