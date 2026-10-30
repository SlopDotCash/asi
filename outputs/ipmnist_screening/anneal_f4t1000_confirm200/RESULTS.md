# RESULTS: anneal_f4t1000 vs anneal_f2t1000 at 200 tasks, seeds 0-19

Preregistration: commit 4b2d6de0 (before any run). Setup: commit 15719be5. Run: 22:45 to about 23:44 GMT on 2026-10-30. Single-threaded workers, at most 4 in parallel; no job failures.

## Environment check: PASS (all bit-for-bit, from ENVCHECK_STATUS and envcheck.py)
- `upgd_w_control` seed 0, 60 tasks: MATCH vs 3c38b025.
- `anneal_f2t1000` seed 0, 200 tasks: MATCH vs 612491db.
- `anneal_f4t1000` seed 0, 60 tasks: MATCH vs 0240ff25. This exercises the anneal path, per the #3585 review.

Parent: the 20 stored 200-task shards from 612491db were reused verbatim (see `parent_reused/SHA256SUMS`), as the preregistration allows once check 2 passes.

## Result (`python3 analyze.py`)
| seed | parent f2t1000 | arm f4t1000 | paired Δ |
|---|---|---|---|
| 0 | 0.949542 | 0.949604 | +0.000062 |
| 1 | 0.949328 | 0.949372 | +0.000044 |
| 2 | 0.949928 | 0.950050 | +0.000122 |
| 3 | 0.949653 | 0.949647 | -0.000006 |
| 4 | 0.949883 | 0.950073 | +0.000190 |
| 5 | 0.949514 | 0.949738 | +0.000224 |
| 6 | 0.949871 | 0.950137 | +0.000266 |
| 7 | 0.949916 | 0.949959 | +0.000043 |
| 8 | 0.949930 | 0.950040 | +0.000110 |
| 9 | 0.949606 | 0.949791 | +0.000185 |
| 10 | 0.949709 | 0.949840 | +0.000131 |
| 11 | 0.949267 | 0.949546 | +0.000279 |
| 12 | 0.949891 | 0.950048 | +0.000157 |
| 13 | 0.949740 | 0.949842 | +0.000102 |
| 14 | 0.949930 | 0.949996 | +0.000066 |
| 15 | 0.949876 | 0.949992 | +0.000116 |
| 16 | 0.950138 | 0.950366 | +0.000228 |
| 17 | 0.949859 | 0.950049 | +0.000190 |
| 18 | 0.949953 | 0.950038 | +0.000085 |
| 19 | 0.949604 | 0.949864 | +0.000260 |

mean +0.000143; SE 0.000018; 95% CI [+0.000104, +0.000181]; improved 19/20
bars: mean>=+0.002 False; >=14/20 True; CI lo>0 True -> FAILS
blocks: 1-50 -0.000123; 51-100 +0.000223; 101-150 +0.000195; 151-200 +0.000275
seeds 0-9 mean +0.000124 seeds 10-19 mean +0.000161

## Verdict: FAILS (mean bar)
- The paired gain is real but small. It is positive on 19/20 seeds, and the 95% CI [+0.000104, +0.000181] excludes 0.
- The mean is +0.000143, about 14 times below the preregistered +0.002 bar.
- It also falls below the prior estimate in the preregistration (+0.0005 to +0.0015).
- The block profile matches the sweep: a cost in tasks 1-50, then a roughly flat +0.0002 to +0.0003 per block afterwards, not a gain that keeps growing.
- Conclusion: more within-task decay than f2t1000 does not pay materially over a full 200-task life. This closes the floor axis of the anneal schedule. anneal_f2t1000 remains the incumbent. No promotion.

Disclosures:
- Seeds 0-9 were seen at 60 tasks in the sweep (0240ff25). Their first 60 tasks are prefix-identical here, as the environment check for seed 0 confirms.
- The parent shards are reused, not re-run.
