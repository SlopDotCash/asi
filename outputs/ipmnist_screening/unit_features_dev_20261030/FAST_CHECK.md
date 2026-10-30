# Broader fast-suite result

Executed after retirement, with the final live source/tests identical to base
`fd25c476957a36937bd5fccfb8be57f4f33e9210`:

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -m 'not slow and not package' -q -o addopts='' --strict-config --strict-markers
```

Exit **1**. **18,151 passed, 71 failed, 125 skipped, 1,291 deselected**,
31 warnings; 2,880.77 seconds (48 minutes). This gate did not pass.
`fast-failed-nodeids.txt` lists all 71 observed failures across 16 unmodified
modules. The complete local log's SHA-256 is
`9453902c4dfaa08066355600426c36c69b388d578f670336c9664307e5b59262`.

Observed failures include an unavailable `jax.experimental.enable_x64` API,
missing `previous_discounts` in option-state construction, RiverSwim checker
configuration rejection, dtype/overflow assertions and other untouched
components. This contribution does not fix, waive or assign one cause to those
failures. A separate complete fast-suite run on detached base was **not run**;
no all-green base or individual baseline reproduction is claimed. Git confirms
no net changes to `alberta_framework/`, `tests/`, `tools/`, `pyproject.toml` or
`uv.lock` versus base.

The narrower IPMNIST checks and actual six-shard workflow are separately
recorded in `PRECHECKS.md`, `FINAL_CHECKS.md` and `logs/`. Their success does not
make this broader failing gate green. The negative result stays permanently
nonpromoting; maintainer review is required.
