# Checks before the development run

- Base RLS-focused selection: 25 passed, 382 deselected (6.31 seconds).
- New mechanism tests initially fail on the unmodified base (missing endpoint).
- Final IPMNIST screening plus unit-feature tests: 410 passed (62.45 seconds).
  The strict registry enumeration now explicitly includes the one new endpoint;
  existing arm constants and factory checks remain enforced.
- Ruff across the repository: passed.
- Mypy: exits 1, 27 errors in 9 unmodified files, checked 226 files. The base
  and candidate diagnostics are identical. No type configuration is relaxed.
- Evidence registry: exits 2 on base and candidate, with identical statuses
  and diagnostics for all five historical claims. Their pinned source hashes
  no longer match current core sources. No artifact, validator or gate changed.
- No real MNIST benchmark result had been collected when these checks and the
  decision rule were recorded. Synthetic unit tests are not benchmark evidence.
- Environment restored from the repository lock using uv sync --frozen --extra dev,
  with the existing Python 3.12.10 interpreter. JAX/jaxlib 0.11.0 and NumPy 2.5.1.
- Authoring: OpenAI / gpt-6.1-sol, codex-desktop 26.1027.30824. Local CPU
  experiment only; no paid provider, GPU, cloud compute or new monetary spend.
