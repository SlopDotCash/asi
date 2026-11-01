# Exact integer rank comparison verification

Base: `ea6eee9d0da3fce024028810b60edfb7fa4389b4`.

Eight paired counters `2**60 + [1, ..., 8]` versus a constant `2**60` are
distinct exact int64 observations. On the base, SciPy narrows each group to
float64 before subtraction, losing all eight differences. The wrapper returns
Wilcoxon p=1 with a runtime warning instead of the exact all-positive signed-rank
p=`2 / 2**8 = 0.0078125`. This changes the 0.05 significance decision. Mixed
int64/uint64 Mann–Whitney inputs can likewise lose their ordering during
concatenation and produce an invalid p-value.

The fix preserves integer differences, signs, zeros, and tied absolute ranks
before the Wilcoxon call. Mann–Whitney receives shared ordinal encodings of the
exact combined integer observations. These encodings change no ranks or
tie groups. Float-input paths, significance thresholds and report schemas are
unchanged. Pooled Cohen's d, used by the Wilcoxon report, centers integer groups
with exact rational means before floating spread reduction, so the accompanying
effect is not falsely zero. Its existing zero-variance conventions remain.

## Verification

Linux/WSL2 CPU, Python 3.12.3, JAX/JAXlib 0.11.0, NumPy 2.5.1, SciPy 1.18.0,
pytest 9.1.1, Ruff 0.16.1, mypy 2.3.0.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_statistics_integer_rank.py -q -o addopts=""
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_statistics_validation.py tests/test_statistics_integer_rank.py \
  -q -o addopts=""
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/alberta-evidence-status
```

The new cases cover large signed and unsigned offsets, mixed integer dtypes,
zero differences and tied ranks, unchanged float-input behavior, genuine
identical-pair rejection, and exactly separated zero-variance integer groups.
On the base implementation, the new cases produce **6 failures and 1 pass**.
On the fix, the combined focused statistics suites produce **275 passes**.
The existing statistics suite remains in the validation set, including its
previously present SciPy precision-loss warnings for constant paired shifts.

Full-tree Ruff passed. Full-tree mypy reports the same 26 diagnostics in eight
unrelated files as the clean base (226 source files checked). The evidence CLI
exits 1 with all five claims `invalid` on both the clean base and candidate due
to existing registered-source drift; these failures were not suppressed.
No frozen artifact, threshold, benchmark shard or seed allocation was changed.
No learning improvement, new benchmark result, or promotion is claimed.
