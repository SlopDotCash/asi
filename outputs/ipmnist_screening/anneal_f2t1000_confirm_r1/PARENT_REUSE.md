# Parent shards: REUSED (all three preregistered conditions hold, checked 18:38 GMT 2026-10-30)
1. Parent jaxpr (SMALL config, `screening_spec(...).factory` step) at 80ee0654 (fd25c476 + confirm_r1 folder) and at 7db2a016 (3c38b025 + prereg) are byte-identical: 17543 chars, sha256 ec000d32a03e1f033100309e805482ca4f8838c67cee40f3c6951ae813c4fce8.
2. `uv pip freeze` of this worktree's venv (`uv sync --frozen --extra dev`) is identical to asi-confirm200-wt's venv.
3. For seeds 0-9, the first 60 per-task accuracies and losses in the confirm_r1 200-task parent shards equal the 60-task parent shards re-run at 3c38b025 (anneal wave) exactly.
Copies are in `parent_reused/` (unmodified; sha256s in parent_reused/SHA256SUMS). Only the 20 candidate shards are run here.
