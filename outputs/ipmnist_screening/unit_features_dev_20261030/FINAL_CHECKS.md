# Final tree checks after retiring the losing implementation

The live benchmark source and existing screening tests are byte-for-byte back
to base `fd25c476957a36937bd5fccfb8be57f4f33e9210`. The experimental source and
three mechanism tests remain in `source.bundle` at commit
`9884c20403ff889307f80242ee032692ad0e80ef`, where all 410 focused tests passed.
The net PR adds the negative result and its evidence packet, with no live arm.

- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_ipmnist_screening.py -q -o addopts=''`: **407 passed in 61.02 seconds**.
- `.venv/bin/python -m ruff check .`: **passed**.
- `.venv/bin/python -m mypy`: **failed**, 27 errors in nine unmodified files;
  diagnostics byte-identical to base and experimental source.
- `.venv/bin/alberta-evidence-status`: **exit 2**, all five historical claims
  invalid for existing registered-source drift; report matches base except
  validation timestamps and the validation snapshot's Git HEAD metadata.
- Official six-shard merge: **exit 0**, executed at the original clean source
  commit before retirement. Raw artifacts and input bindings inspected.
- Source bundle: `git bundle verify` accepted its prerequisite-bound contents.

The two failing gates are disclosed, not bypassed or relaxed. No frozen
artifacts, registered validators, thresholds or dependency files were changed.
