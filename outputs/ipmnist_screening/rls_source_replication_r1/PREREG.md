# Pre-registration: source-bound replication of the RLS-residual gain (r1)

Frozen before any r1 shard runs. Development-grade, permanently nonpromoting.

- **Lane:** `alberta_framework.benchmarks.ipmnist_screening`, 60 tasks x 5000
  steps, exact per-step noise (`--noise-mode step`), one example per step.
- **Metric:** `average_online_accuracy`, paired candidate-minus-control on
  shared seeds (same runner, same source).
- **Control (baseline):** `sigma0_shiftnorm_d099`. Stored reference only (NOT
  imported as a live guarantee): 200-task confirmation mean 0.86449 (n=20,
  `summary_rls_head_confirm.json` control row); 60-task screen 0.86396
  (`FINAL_REPORT.md` shift wave). Remeasured here at current source.
- **Candidate:** `rls_head_resid_l1_preset005` (registered arm, unchanged
  config). Stored reference only: 200-task confirmation mean 0.87114 (n=20),
  paired +0.00665 +/- 0.00013, all 20 seeds improve
  (`summary_rls_head_confirm.json`).
- **One change:** none to either config. This is open theory hypothesis 1
  (source-bound replication): same two configs, current source
  `c9aba7b54dedd647f8bd5f5c7bf6780b1413b676` (`origin/main`), same runner,
  v2 shard schema, new output path. Deviation from RUNBOOK reuse advice is
  deliberate: the campaign index requires remeasuring the control in the same
  runner at current source rather than importing the stored mean.
- **Seeds:** 0, 1, 2 (consumed development screening seeds, shared across
  arms => paired). Held-out seeds 3-19 are NOT touched in r1. n=3.
- **Commands** (each with `OMP_NUM_THREADS=1`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`):
  `.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run
  --config-name <arm> --seed <seed> --n-tasks 60 --task-length 5000
  --noise-mode step --out
  outputs/ipmnist_screening/rls_source_replication_r1/shards/<arm>_seed<seed>.json`
  then `merge --shards <6 r1 shards> --control-name sigma0_shiftnorm_d099
  --output outputs/ipmnist_screening/rls_source_replication_r1/summary.json`.
- **Frozen decision rule (directional replication):** success iff
  candidate-minus-control mean diff > 0 AND all 3 per-seed diffs > 0 (same
  sign pattern as the stored +0.00665). No 200-task escalation follows from
  this turn regardless of outcome; any confirmation needs a separate
  pre-registration. Thresholds are not moved after seeing numbers.
- **Loss plan:** report exact means, spread, per-seed diffs either way. If
  diff <= 0 or signs mix, record as failed/inconclusive replication, touch no
  held-out seed, retune nothing, open no issue from this run.
- **Scope notes:** micro-suite prototype skipped with reason — the RLS head
  is IPMNIST-protocol-specific with no micro analogue; the 60-task screen IS
  the cheap tier here. No library source is edited, so no failing-test-first
  cycle applies. Closed lanes untouched (no v3 frozen lifecycle, no
  EMNIST/regression/IA, no forager matched campaigns).
