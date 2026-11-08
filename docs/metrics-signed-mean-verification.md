# Signed metric mean verification

Base: `ea6eee9d0da3fce024028810b60edfb7fa4389b4`.

The public `compute_prequential_performance` and `compare_learners` APIs accept
signed finite reward traces. Permuting `[1e16, 1, -1e16]` on the base reports
means of `0`, `1/3`, or `0.3700743415417188`, although every ordering has the
same exactly representable sum, `1`. `compare_learners` already retains that
sum in `cumulative`, making its mean and cumulative fields disagree.

The shared mean reducer now sums the original values with compensated
summation. An exact rational fallback handles intermediate-sum overflow,
including cancellation with a tiny remaining reward. Dividing before the final
conversion also retains finite means whose sum exceeds float64. Cumulative
summaries use the same total. The domain, NaN-probe handling, public schemas and
scientific thresholds are unchanged. This is a numerical measurement fix,
not a learning improvement or a claim that any benchmark has improved.

## Reproduction

The added tests exercise all six orderings through both public APIs, the
`mean`/`cumulative` relationship, skipped NaN probes, zero traces, and a
same-sign `1e308` control whose unscaled sum would overflow. Two additional
full-range controls retain a `1e-100` residual beside cancelling `1e308` values,
with and without intermediate-sum overflow.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_continual_metrics.py \
  -q -o addopts="" -k signed_metric_mean
```

With the new tests and the base implementation: **7 failed, 2 passed**.
With the fix, the focused metric suites below: **111 passed**.
Full-tree Ruff 0.16.1: **passed**. Full-tree mypy 2.3.0: **failed**, with
the same 26 diagnostics in eight unrelated files on the clean base and
candidate (226 source files checked); the diagnostic lines were compared.
The evidence CLI exited **1**, reporting all five claims `invalid` on both
the clean base and candidate because the retained artifacts no longer bind
current registered sources. These failures were not suppressed or repaired.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_continual_metrics.py tests/test_metrics_hostile_validation.py \
  tests/test_metrics_boolean_trace_width_hang.py -q -o addopts=""
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/alberta-evidence-status
```

Environment: Linux/WSL2 CPU, Python 3.12.3, JAX/JAXlib 0.11.0, NumPy 2.5.1,
SciPy 1.18.0, pytest 9.1.1. Native Linux checkouts retained the committed LF
bytes for provenance validation. No benchmark shard, seed selection, promotion, or external
provider call was run. Pinned artifacts were not modified. The metrics source
is registered for the multiagent claim; changing it cannot preserve an old
artifact's source binding, and no such claim is made here.
