# Results: 200-task, 20-seed confirmation of `rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000`

Development screening diagnostic; permanently nonpromoting. Bars, seeds and arms are as preregistered in commit 7db2a016.

## Verdict: CONFIRMED
- The paired mean is +0.006576, which clears the ≥ +0.002 bar.
- All 20 of 20 seeds improved.

## Protocol
- **Runs:** 200 tasks x 5000 steps, seeds 0-19, at most 4 one-thread workers.
- **Candidate source:** commit 91da7dcc (3c38b025 + this folder); every shard records a clean worktree.
- **Parent:** the 20 shards from `idet003_s005625_confirm_r1` were reused under the preregistered conditions; see PARENT_REUSE.md.
  - The reused parent mean, 0.943181 ± 0.000209, equals that confirmation's reported parent mean.

## Summary
| | value |
|---|---|
| candidate mean ± SD | 0.949757 ± 0.000225 |
| parent mean ± SD | 0.943181 ± 0.000209 |
| paired mean (SD) | +0.006576 (0.000135) |
| 95% t-CI (df = 19) | [+0.006513, +0.006640] |
| seeds improved | 20/20 |
| weakest seed | 11 (+0.006351) |
| first 60 tasks, paired | +0.002225, CI [+0.002101, +0.002350], 20/20 |
| tasks 181-200, paired | +0.009640, CI [+0.009464, +0.009816], 20/20 |
| ratio of 200-task gain to first-60 gain | 2.96 (the gain grows with horizon; it does not shrink) |

## Per seed
| seed | candidate | parent | diff | first60 diff | 181-200 diff |
|---|---|---|---|---|---|
| 0 | 0.949542 | 0.942928 | +0.006614 | +0.002033 | +0.009570 |
| 1 | 0.949328 | 0.942792 | +0.006536 | +0.002503 | +0.009430 |
| 2 | 0.949928 | 0.943148 | +0.006780 | +0.002433 | +0.009740 |
| 3 | 0.949653 | 0.943264 | +0.006389 | +0.002067 | +0.009010 |
| 4 | 0.949883 | 0.943187 | +0.006696 | +0.002387 | +0.009940 |
| 5 | 0.949514 | 0.943076 | +0.006438 | +0.001970 | +0.009260 |
| 6 | 0.949871 | 0.943474 | +0.006397 | +0.001923 | +0.009580 |
| 7 | 0.949916 | 0.943295 | +0.006621 | +0.002630 | +0.009630 |
| 8 | 0.949930 | 0.943421 | +0.006509 | +0.002373 | +0.009110 |
| 9 | 0.949606 | 0.943116 | +0.006490 | +0.002233 | +0.009790 |
| 10 | 0.949709 | 0.942966 | +0.006743 | +0.002403 | +0.009510 |
| 11 | 0.949267 | 0.942916 | +0.006351 | +0.001840 | +0.010010 |
| 12 | 0.949891 | 0.943367 | +0.006524 | +0.001737 | +0.009280 |
| 13 | 0.949740 | 0.943119 | +0.006621 | +0.002547 | +0.009900 |
| 14 | 0.949930 | 0.943378 | +0.006552 | +0.002030 | +0.009830 |
| 15 | 0.949876 | 0.943350 | +0.006526 | +0.002243 | +0.009490 |
| 16 | 0.950138 | 0.943424 | +0.006714 | +0.002107 | +0.009450 |
| 17 | 0.949859 | 0.943015 | +0.006844 | +0.002537 | +0.010540 |
| 18 | 0.949953 | 0.943440 | +0.006513 | +0.002007 | +0.010280 |
| 19 | 0.949604 | 0.942938 | +0.006666 | +0.002500 | +0.009450 |

## Checks
- All 20 candidate shards have 200 finite per-task accuracies.
- No log contains an error. No worker failed, so no shard was requeued.
- For seeds 0-9, the first 60 per-task accuracies of the candidate shards equal the 60-task anneal-screen shards exactly. The first-60 diff here (+0.002225) is consistent with the screen's +0.00226.
- The table was computed by `analyze.py`, read directly from the raw shards' `per_task_accuracy`.

## Seeds and holdout
- Seeds 10-19 are now seen for this arm at 200 tasks.
- The 60-task holdout cells (seeds 10-19 at 60 tasks) remain unrun.
- Our reviewer confirmed this seed choice and the parent reuse.

## Interruptions
- The executor session was interrupted twice during monitoring, at about 19:05 and about 19:20 GMT.
- The detached runs were unaffected: the launch queue finished at 19:15 GMT and the last shard at about 19:25 GMT.
- Nothing was requeued.

## Upstream
- At 19:35 GMT, #3118 and #3123 were open and unmerged, and main was fd25c476.
- No re-pairing was needed.
