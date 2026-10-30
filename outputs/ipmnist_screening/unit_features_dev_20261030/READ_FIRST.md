# Scope and terminology clarification

This packet records a **predeclared development screen**, never a scientific
preregistration. `CONTRIBUTING.md` reserves "preregistered" for scientific
protocols. The original `preregistration.json` filename/schema and that word in
`RESULTS.md` are retained as original bytes, but do not grant that status.
Every machine-readable shard/summary says development-only and forbids
scientific promotion. The plan used already-consumed development seeds.

Read the original report with that terminology correction. All measurements,
thresholds, commands, sample SDs and the bounded negative decision are unchanged.
No held-out, promoting or population-level claim is made. This clarification
adds no experiment, seed, arm, calibration or threshold change.

Resource detail: the two arms have identical persistent learner-state shapes
and parameter counts; the intervention adds no persistent feature state.
Both see 300,000 examples/updates per seed, with no replay, pretraining, extra
model queries, task-boundary labels or cloud compute. Peak memory and per-step
latency were not measured; no memory or speed improvement is claimed.

The final tree contains no new live mechanism. Reproduce the experimental arm
at source commit `9884c20403ff889307f80242ee032692ad0e80ef` using `source.bundle`,
not at the final PR head. `MANIFEST.sha256` covers the original packet; later
append-only additions are covered by `MANIFEST.v2.sha256`.

The broader fast suite was executed and failed: see `FAST_CHECK.md` and all
71 failed node IDs. Its failure is explicitly disclosed alongside the existing
mypy and historical evidence-registry failures.
