# Result: source-bound replication of the RLS-residual gain (r1)

Development-grade, permanently nonpromoting. Pre-registration:
`PREREG.md` (frozen before any r1 shard ran).

## Decision: REPLICATED (directional, 60-task screen)

`rls_head_resid_l1_preset005` minus `sigma0_shiftnorm_d099`, seeds 0-2,
same runner, current source `c9aba7b5`, v2 schema (`summary.json`):

| arm | mean | stderr | per-seed |
|---|---|---|---|
| `sigma0_shiftnorm_d099` (remeasured control) | 0.86396 | 0.000204 | 0.8636 / 0.8642 / 0.8642 |
| `rls_head_resid_l1_preset005` | 0.86927 | 0.000098 | 0.8691 / 0.8694 / 0.8693 |
| paired diff | **+0.005307** | 0.000135 | +0.005553 / +0.005280 / +0.005087, all positive |

Frozen rule required mean diff > 0 with all 3 per-seed diffs > 0: met.
The remeasured control tracks the stored 60-task screen (0.86396) exactly;
the candidate tracks its stored screen (0.86938) within 0.00011.
No held-out seed touched, no threshold moved, no escalation to 200-task
from this turn (needs a separate pre-registration).

## Provenance

- 6 v2 shards in `shards/` (source tree `c55a4f2a…`, worktree clean at run
  time; runtime CPU, JAX 0.11.0, NumPy 2.5.1, Python 3.14.4).
- OOM note: a first `-P 4` launch was OOM-killed on this contended 30 GB
  box (exit 137); the 5 surviving shards were rerun serially (`-P 1`).
  The one completed parallel shard was kept (idempotent skip); all shards
  merge cleanly under one runtime binding.
- Merge required the identical derivation env (`OMP_NUM_THREADS=1`); a
  first merge without it failed closed on the v2 runtime binding, then
  succeeded with matched env. No validator, threshold, or test was
  weakened. No library source edited.
