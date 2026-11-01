# Linear batched device-completion regression

This is a reproduced runtime measurement defect, not a benchmark improvement
or scientific evidence. The linear batched wrapper stopped its uptime clock
after dispatching the vmapped scan, while actual result arrays were unfinished.
The single-seed barrier runs inside the vmap transformation and does not finish
the outer dispatched work. The MLP batched wrapper already has an outer barrier.

Base: `ea6eee9d0da3fce024028810b60edfb7fa4389b4`.
Product/test commit: `79af86835d4653e9a817136d4ade9a5ece9dd716`.
The only product change waits on batched states, metrics, step-size histories,
and normalizer histories before reading the ending clock.

## Actual before/after reproduction

Environment: WSL2 Ubuntu, Python 3.12.3, JAX/jaxlib 0.11.0, CPU backend.
The new `tests/test_linear_batch_readiness.py` runs real JAX array work: two
Threefry keys split from synthetic development seed 712831, 10,000 steps,
1,000 random-walk features, and LMS step size 0.0001. Both plain and tracked
cases probe `metrics.is_ready()` immediately after the public API returns,
before a conversion or array read can accidentally synchronize it. These are
small library regression workloads within existing protocol limits, not a
frozen benchmark, a new protocol, or seed reservation.

Copy the new test file into a clean worktree at the base revision and run:

```sh
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/test_linear_batch_readiness.py -q -o addopts=""
```

Three separate base runs each produced **2 failed, 1 passed**: both readiness
assertions observed `False`; the independent sequential-learning equivalence
control passed. With the completion barrier, the same tests pass. All 10,000
updates complete for both seeds, and the small fresh-seed 712832 control checks
batched metrics and weights against independent sequential runs.

The assertions cover scan-produced arrays. The uptime addition itself creates
a new array after the barrier and need not already be ready at API return.
No callback, mocked scan, simulated timer, or wall-clock threshold supplies the
before/after proof. A preliminary blocked-debug-callback experiment did not
reproduce early return because callbacks synchronized dispatch; it is not used
as evidence for this fix.

## Required checks and limits

Commands used the existing project virtual environment, with `PYTHONPATH`
pointing explicitly at this worktree rather than its editable-install default:

```sh
PYTHONPATH="$PWD" .venv/bin/python -m ruff check .
PYTHONPATH="$PWD" .venv/bin/python -m mypy
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/test_linear_batch_readiness.py tests/test_learners.py -q -o addopts=""
PYTHONPATH="$PWD" .venv/bin/alberta-evidence-status
```

- Ruff: passed.
- Focused plus existing learner suite: **75 passed**. This includes existing
  normalized batched tests with both histories, step-size history alone, and
  normalizer history alone.
- Mypy: base and candidate both report the same **26 errors in 8 unchanged
  files**, none in `core/learners.py`; the complete diagnostic logs match.
- Evidence registry: base and candidate both exit **2**, with all five claims
  `invalid`. Each claim's error list and registered-source hash mapping is
  identical between the two runs. Windows-worktree checks ran through WSL
  with explicit `GIT_DIR`/`GIT_WORK_TREE` so provenance resolved the real base
  HEAD; candidate checks captured its local product diff before commit. The
  existing source/artifact mismatches and historical reconstruction failures
  were retained. No pinned artifact or registered source inventory was edited.

Only CPU behavior was verified. This does not claim accurate accelerator
benchmark timings, an algorithmic gain, improved cold/warm performance, or an
evidence promotion. Existing nested tracing-time accounting is unchanged.
No language model inference or model-backed benchmark behavior was exercised.
