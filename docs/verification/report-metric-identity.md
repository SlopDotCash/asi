# Saved tables identify the selected measurement

Base: `ea6eee9d0da3fce024028810b60edfb7fa4389b4`.

A real `save_experiment_report(..., metric="accuracy")` report labels its
Markdown and LaTeX measurements `Error`. The JSON correctly names
`accuracy`, so the human-readable artifacts identify a different kind of
measurement. The same failure occurs for reward and custom metric names.

The existing optional display-label argument now has a metric-aware
default: retain `Error` for the historical `squared_error` default, and
otherwise render the selected metric as plain table text. Explicit
caller-supplied labels retain their existing behavior, including raw
LaTeX markup. Validate the raw metric before escaping so a valid 256-byte
metric name remains exportable when its escaped form is longer. No
measurement values, significance decisions, file schemas, CSV columns, or
metric dictionary keys change.

## Before and after

Six actual export/interface checks cover accuracy, reward, a custom name
with underscore/pipe, a 256-byte name, the legacy default, and explicit
display labels. Identical tests give **4 failed / 2 passed before**,
**6 passed after**. The complete export/hostile suites give
**5 failed / 191 passed before**, **1 failed / 195 passed after**.
The remaining failure is unchanged and is documented below.

Five actual report pairs retained JSON means of `0.75` under the exact
selected metric keys, alongside their saved Markdown and LaTeX headers.
Before, every header says `Error`; after, the four non-default headers
identify their metric and the `squared_error` control still says `Error`.
All ten complete documents containing those saved table bytes compile with
cached Tectonic `0.17.0+20260731` in untrusted mode. This verifies usable
artifacts, not a scientific or performance result.

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
