# Automatic report captions remain plain experiment text

Base: `ea6eee9d0da3fce024028810b60edfb7fa4389b4`.

`save_experiment_report` accepts a filename such as `trial_1`, then inserts
that filename directly into a LaTeX caption. The actual saved table fails
to compile with `Missing $ inserted` at `\caption{trial_1 Results}`.
The same data and an alphanumeric filename compile successfully.

Escape only automatically derived caption text. Preserve safe existing
reference labels, including `tab:trial_1`; names containing TeX-active
characters or whitespace use a bounded, deterministic SHA-256 label. A
private renderer lets the already-validated, at-most-200-byte filename
expand during escaping without changing the public raw-caption 256-byte
limit. Explicit `generate_latex_table` caption/label markup remains raw,
as required by the existing interface test. Filenames and numeric data are
unchanged.

## Before and after

Five real `save_experiment_report` cases use plain, underscore, percent,
brace, and 200-byte underscore-containing names. Identical pytest tests
give **4 failed / 1 passed before**, **5 passed after**. The full export
and hostile-input suites give **5 failed / 190 passed before**,
**1 failed / 194 passed after** (the unchanged failure is below).

Tectonic `0.17.0+20260731`, `--only-cached --untrusted --keep-logs`, compiled
complete documents containing the actual saved table bytes. Before: the
plain-name and brace-name documents compile; underscore, percent, and long
underscore cases fail, so **2 PDFs / 3 failures**. After: **5 PDFs, no
compiler failures**. The brace regression additionally preserves literal
braces instead of treating them as a TeX group. Raw compiler output and
PDFs were retained locally for the PR evidence. A fontconfig warning was
emitted on this Windows compiler; it did not prevent the control or fixed
PDFs. This is export/compilation evidence, not scientific evidence.

## Checks and limits

Linux/WSL2 CPU, Python 3.12.3, NumPy 2.5.1, pytest 9.1.1, Ruff 0.16.1,
and mypy 2.3.0. This is a deterministic export repair, not a benchmark run,
scientific improvement, promotion, or model-backed result. No frozen lane
was run and no retained output artifact was modified. The export module is
not a registered source of the five retained scientific claims.

Run from the checkout with its Python 3.12 virtual environment:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_export.py tests/test_export_hostile_validation.py -q -o addopts=""
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
PYTHONPATH=. .venv/bin/alberta-evidence-status
```

Full-tree Ruff passed. Mypy returned 1 with 26 pre-existing diagnostics in
eight unrelated files; baseline and candidate logs were byte-identical.
The evidence CLI returned 2 on both stages: five retained claims were
invalid due to existing source-provenance drift and retained IA budget
validation failures. Each claim's status and error list matched exactly
between stages. These failures were not suppressed.

The export suites retain one pre-existing failure on this NumPy version:
`TestCrossConfigFinalWindow.test_equal_windows_still_export` expects the
final-window rejection, but the earlier exact summary/raw-trace comparison
rejects its literal `0.11` fixture first. No tolerance or validation rule was
changed to hide that failure.

## Integration with the other report correction

The separate automatic-caption and selected-metric-label repairs start from
the same base and repair different outputs. Their combined integration was
tested in a temporary checkout: all 15 caption, metric-label, explicit-label
and related regression cases passed. Whichever change merges second needs
a rebase. In particular, after the caption helper extraction,
`save_experiment_report` must give `_render_latex_table` the metric-aware
default (`Error` for `squared_error`, otherwise the escaped metric name),
rather than retaining its old hard-coded `Error`. Keep explicit caller
LaTeX labels/captions untouched. This semantic integration must not be
assumed from a conflict-free textual merge.
