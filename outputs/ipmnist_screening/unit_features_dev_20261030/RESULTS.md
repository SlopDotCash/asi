# Unit-length RLS features lose this bounded development screen

The frozen decision is **material negative on the three consumed development
seeds**. This rejects the tested unit-feature configuration for this 60-task
protocol. It is not a population-level statistical claim, a rejection of all
feature normalization, or scientific/promoting evidence.

The hypothesis, single intervention, paired seeds and thresholds were published
before MNIST execution in source commit
`9884c20403ff889307f80242ee032692ad0e80ef` (tree
`397ea6471500e914b8c2c9fa64289d902c11da8c`). Both arms were freshly measured
from that clean source with identical runtime/data bindings. No older score was
substituted for the control. The control was the highest registered main-branch
step-mode summary found before the run; stronger unreviewed branches exist, so
this is not an overall-best comparison.

## Metric and paired observations

Lane: `alberta_framework.benchmarks.ipmnist_screening`, abrupt `step`, 60 tasks
x 5,000 examples, unchanged 784/300/150/10 network, canonical MNIST 60,000-row
training split. The metric is average **pre-update online accuracy** across all
300,000 examples per seed. All observations below come from the retained raw
60-task accuracy arrays; SD means sample SD, not standard error.

| Development seed | Remeasured control | Unit-feature candidate | Candidate minus control |
| --- | ---: | ---: | ---: |
| 0 | 0.934433308333 | 0.874093309000 | -0.060339999333 |
| 1 | 0.934563313833 | 0.919723311667 | -0.014840002167 |
| 2 | 0.934706645000 | 0.925043309500 | -0.009663335500 |
| Mean, n=3 | 0.934567755722 | 0.906286643389 | -0.028281112333 |
| Sample SD | 0.000136722460 | 0.028006850565 | 0.027884200656 |

The candidate loses all three pairs and its mean delta is below the frozen
-0.005 threshold. The loss is 2.828111 percentage points on this development
sample. Candidate spread is large; no confidence, generalization or significance
claim is inferred from n=3. No extra seeds, calibration, retuning, arm selection,
200-task confirmation, held-out evaluation or promotion followed the result.

## What changed and what the result resolves

Only the hidden part of the RLS feature vector changed, from
`a2 / sqrt(hidden2 + 1)` to `a2 / sqrt(sum(a2**2) + 1e-8)`. Bias stays one.
That same normalized forward supplies prediction, residual-body differentiation,
RLS update and post-update plasticity. All control constants, state shapes,
identifier schedule, reset/detector behavior, initialization and data order stay
fixed. This tests normalization with the **fixed control ridge and step size**;
it does not determine the cause of the loss or rule out a differently calibrated
method. The mechanism-off endpoint was bit-exact to the control through 120
synthetic updates, including a shift, before any benchmark execution.

The bounded negative prevents repeating this exact configuration expecting a
cheap gain. The losing arm and its tests are removed from the live tree in the
final contribution. Its exact experimental implementation and reduction tests
remain reproducible in `source.bundle` and the published source commit, rather
than adding an unused live option.

## Resources, bindings and limitations

One sequential local CPU worker, `OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu`.
Python 3.12.10, JAX/jaxlib 0.11.0, NumPy 2.5.1, macOS arm64. Dependencies restored
from unchanged `uv.lock`; no paid API, GPU or cloud compute. Spend: $0.
Six recorded shard runtimes total 413.66 seconds. The launcher including data
materialization, process startup and merge took 538.27 seconds, below the frozen
7,200-second budget. Runtime varied substantially; these are telemetry, not an
efficiency comparison. Logs retain all six runs without selection.

The official merge accepted all six source/runtime/data-matched v2 shards before
HEAD changed. Each binds clean source commit above, source SHA-256
`f0be17e23b12ae9d330cb34d73bd3515ca459173e2fa924f90ad035a212deefb`,
the unchanged lock hash, and the materialized dataset hashes. `analysis.json`
independently recomputes per-seed means, sample SD and paired deltas from the
raw arrays and checks the summary's input file hashes/sizes and bindings.
Hashes establish consistency, not authenticated independent execution.

Required prechecks at the experimental source: Ruff passed; 410 focused tests
passed in 62.45 seconds. Mypy ran and failed with the same 27 errors in nine
unmodified files on base and candidate. The evidence registry ran and exited 2
on both, with identical historical-source mismatch diagnostics for all five
claims (apart from validation timestamps). No frozen artifact, threshold,
validator or type configuration was changed. `check-comparison.json` and
`PRECHECKS.md` record these limits; full unrelated historical registry output
stays outside this artifact packet.

Deviation disclosure: before any MNIST shard, recursive artifact inspection
corrected the initially considered older RLS parent to the stronger registered
identmap control. Only synthetic tests had run then. The frozen plan already
records that correction. No deviations after the plan was published.

## Reproduction

`commands.json` contains every executed run/merge argument and the environment.
`PLAN.md` and `preregistration.json` are the original pre-run bytes. Do not run
the candidate at the final PR head: it is deliberately retired there. Restore
the exact experimental source first. In an ASI clone containing base commit
`fd25c476957a36937bd5fccfb8be57f4f33e9210`:

```sh
git bundle verify /absolute/path/to/source.bundle
git fetch /absolute/path/to/source.bundle refs/heads/codex/screen-rls-unit-features
git switch --detach 9884c20403ff889307f80242ee032692ad0e80ef
uv sync --frozen --extra dev
OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_ipmnist_screening.py tests/test_ipmnist_unit_features.py -q -o addopts=''
```

Run in a separate clean checkout. Use the exact arguments in `commands.json`;
retained shards can be copied into the untracked output paths before the merge
only. For fresh execution, use new output destinations and update the merge
input paths; never overwrite these retained artifacts. The CLI binds the actual
runtime, so another stack is a new measurement and cannot be mixed with these
shards. No benchmark execution through pytest.

Authored with OpenAI / gpt-6.1-sol / codex-desktop 26.1027.30824.
