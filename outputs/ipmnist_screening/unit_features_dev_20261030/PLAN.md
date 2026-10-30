# Unit-length RLS features: bounded development screen

Declared before collecting any candidate benchmark result on 2026-10-30.
Base: `fd25c476957a36937bd5fccfb8be57f4f33e9210`.

Before any MNIST shard, recursive artifact inspection corrected the selected
control from the older RLS parent to the strongest registered step-mode arm.
The preliminary parent-only implementation was exercised on synthetic unit
inputs only; no benchmark data, tuning result or evaluation seed was collected.

Question: does removing per-example penultimate-feature length variation improve
the confirmed residual-trained RLS control's online accuracy? The hypothesis is
that unit-length hidden features make the RLS ridge/gain less sensitive to
activation magnitude across input permutations. This is an experiment, not an
established benefit or a literature replication.

The one intervention replaces the hidden part of the RLS feature vector
`a2 / sqrt(hidden2 + 1)` with `a2 / sqrt(sum(a2**2) + 1e-8)`. The appended bias
remains exactly one. The same forward is used for the pre-update prediction,
residual body gradient, RLS update, and post-update plasticity diagnostic.
All control constants, state shapes, detector/reset behavior, data order and
initialization remain fixed. A mechanism-off reduction must be bit-exact to the
control before execution. No changes to frozen scientific or forager lanes.

- Control: `rls_head_resid_identmap50_r_idet003_s005625`, the highest-scoring
  registered arm in the retained main-branch step-mode summaries (wave summary
  under `wave_identmap50_r_idet003_s005625/`). Stronger unreviewed branches also
  exist; no comparison or best-overall claim against them is made here.
- Candidate: `rls_head_resid_identmap50_r_idet003_s005625_unitfeatures`.
- Seeds: paired development seeds `[0, 1, 2]`, n=3 for each arm. These are already
  consumed development seeds, never holdouts or promotion evidence.
- Protocol: abrupt `step`, 60 tasks x 5,000 examples, canonical 60,000-row MNIST
  train split, unchanged 784/300/150/10 network and predict-before-update metric.
- Primary metric: average online accuracy over all 300,000 examples per seed.
- Decision fixed in advance: candidate passes this development screen only if
  mean paired delta is greater than 0.005 and every seed improves; call a material
  negative only if mean delta is below -0.005 and every seed loses. All other
  results are inconclusive. Report all per-seed values, means, sample standard
  deviations and paired deltas, irrespective of outcome.
- No tuning or extra arms after inspecting results. No 200-task confirmation,
  held-out seed use, automatic promotion, or source/artifact rewriting.
- Execution: one local CPU worker, `OMP_NUM_THREADS=1`, no GPU/API/cloud spend.
  The selected control's retained three shards record 70.00–74.99 seconds each;
  budget up to two hours for the six full shards on this different CPU. If that
  budget is exceeded, report incomplete rather than changing the sample rule.
  Timing is telemetry, not a hardware-comparable performance claim.
- Run both arms from the same clean source commit, pin it in every shard, and
  use the CLI only. Run control then candidate for each seed. Use new immutable
  output paths below this directory; never overwrite a shard.

Commands after committing and checking the mechanism:

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run --config-name ARM --seed SEED --n-tasks 60 --task-length 5000 --noise-mode step --data-home outputs/upgd_ipmnist/reconstructed_openml_cache --out outputs/ipmnist_screening/unit_features_dev_20261030/shards/ARM_seedSEED.json --progress-every 10
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge --shards outputs/ipmnist_screening/unit_features_dev_20261030/shards/*.json --control-name rls_head_resid_identmap50_r_idet003_s005625 --output outputs/ipmnist_screening/unit_features_dev_20261030/summary.json
```

Required checks: Ruff, mypy, focused IPMNIST tests and the evidence registry.
Inspect actual artifacts and source/runtime/data bindings. An evidence-registry
source mismatch is not a gate to weaken. No accepted scientific claim is made.
