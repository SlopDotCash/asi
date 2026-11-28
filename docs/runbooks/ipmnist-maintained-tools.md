# Maintained IPMNIST campaign tools

Programs stored beside existing `outputs/` records are historical provenance,
not the maintained implementation. Preserve every existing byte: pinned
evidence artifacts are immutable, while the active IPMNIST and UPGD campaign
roots are append-only at new paths under their own protocols. The package
commands below supersede the old programs' analysis and reproduction roles.
All commands are permanently nonpromoting: they do not create scientific
evidence or update a reference channel.

Use an explicit new directory outside the repository for every expensive run.
The runner atomically publishes one self-contained `.npz` artifact per shard
and refuses to replace an existing path.

## Produce new ceiling shards

Choose a fresh run ID; do not reuse a directory from an earlier execution.

```bash
ASI_RUN_DIR=../asi-runs/ipmnist-ceiling-001

.venv/bin/asi-ipmnist-ceiling stationary \
  --output-dir "$ASI_RUN_DIR" \
  --spec sigma0_ndecay099 --seed 0

.venv/bin/asi-ipmnist-ceiling carried \
  --output-dir "$ASI_RUN_DIR" \
  --spec sigma0_ndecay099 --seed 0 --n-tasks 60

.venv/bin/asi-ipmnist-ceiling full \
  --output-dir "$ASI_RUN_DIR" \
  --spec sigma0_ndecay099 --seed 0

.venv/bin/asi-ipmnist-ceiling batch \
  --output-dir "$ASI_RUN_DIR" \
  --seed 0 --epochs 30 --batch-size 128
```

Repeat with the declared development seeds and comparator arms required
by the intended diagnostic. The command retains per-step accuracy and uses the
same screening specification, schedule construction, initialization, and RNG
chain as the IPMNIST screening implementation. These are expensive benchmark
executions and must not run inside pytest. New artifacts bind the maintained
source files, runtime and dependency versions, dataset arrays, schedule arrays,
resolved arm configuration, seed, and protocol configuration.

## Recompute a ceiling summary

The analyzer reads paths supplied by the caller and writes JSON to stdout. It
uses sample standard deviation (`ddof=1`) for across-seed spread.

```bash
.venv/bin/asi-ipmnist-campaign ceiling \
  --ceiling-dir "$ASI_RUN_DIR" \
  --confirm-dir outputs/ipmnist_screening/confirm_full \
  > "${ASI_RUN_DIR}-summary.json"
```

The full-mode cross-check freezes `rtol=0` and `atol=1e-7` for the historical
JSON decimal-encoding difference, reports the maximum observed delta, and
fails above that tolerance. A summary produced from historical raw runs is a
new development diagnostic; it does not revise the historical report.

## Rebuild the paired frontier

```bash
.venv/bin/asi-ipmnist-campaign frontier \
  --screen-dir outputs/ipmnist_screening/shards \
  --confirm-dir outputs/ipmnist_screening/confirm_full \
  > ../asi-runs/ipmnist-frontier-current.json
```

Every screen comparison requires the candidate and base to have the same
non-empty seed set. A present confirmation arm must likewise match the base's
complete confirmation seed set. Mismatched or disjoint sets fail closed rather
than being silently intersected; an arm with no confirmation shards has no
confirmation comparison.

## Rebuild the rule-discovery comparison

```bash
.venv/bin/asi-rule-discovery-summary \
  --screen-dir outputs/ipmnist_screening/shards \
  --confirm-dir outputs/ipmnist_screening/confirm_full \
  > ../asi-runs/rule-discovery-current.json
```

This command is read-only with respect to its inputs and emits JSON to stdout.
Keep generated diagnostics outside `outputs/` unless a separately reviewed,
append-only artifact protocol assigns a new destination and schema.

## Historical provenance

The following files remain useful only as provenance for the runs beside them:

- `outputs/ipmnist_screening/ceiling/ceiling_runs.py`
- `outputs/ipmnist_screening/ceiling/ceiling_analyze.py`
- `outputs/ipmnist_screening/frontier_rank.py`
- `outputs/rule_discovery/write_real_screen_summary.py`

Their machine-specific paths and historical statistical behavior are preserved
intentionally. Contributor workflows must use the maintained package commands.


## Reading between-seed uncertainty

When inspecting a partial one-seed development merge, read the per-seed
accuracy and paired difference directly. The iPMNIST screening and Micro
Continual mergers emit JSON `null` for the between-seed standard error until
there are at least two independent seeds. With at least two seeds, a numeric
zero means the observed values were equal. Earlier one-seed summaries can
contain a numeric zero for this unavailable estimate; check the seed count
when reading those archived results.
