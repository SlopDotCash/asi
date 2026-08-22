# TeLAPA policy-archive qualification smoke

This lane exercises ASI's byte-bounded `BoundedPolicyArchive` in an executable
current `SwitchingTwoStateMDP` stream. It is a permanently nonpromoting
development smoke, not TeLAPA, a paper reproduction, a benchmark result, or a
state-of-the-art claim.

## Provenance audit

The cited paper is *Beyond Single-Model Optimization: Preserving Plasticity in
Continual Reinforcement Learning*, pinned to `arXiv:2604.15414v1` (submitted
2026-04-16). The attributed public repository is now
`lute47lillo/telapa_collas2026`, bound at commit
`a4dc16ed0ea015b1b8efb271e4d664931adccd3e` and tree
`e58072c9c87f984ec9644c7a8fb18e4ce9455286`. The catalog also binds the observed
8,621,070-byte GitHub source archive and the README and `environment.yml` bytes.

The repository README says the project is MIT-licensed and points to `LICENSE`,
but that file is absent from the complete committed tree and GitHub reports no
detected license. The catalog therefore records the attribution and immutable
source identity while leaving license review incomplete, vending no source
bytes, and categorically rejecting paper parity and external execution.

The paper's mechanism is substantially larger than this lane: per-task policy
neighborhoods, PPO, post-training MAP-Elites illumination, few-shot origin
selection, and a learned trajectory embedding maintained with anchors, replay,
alignment, and periodic archive re-embedding. It evaluates recurring MiniGrid
tasks and reports standardized time-to-threshold, success, backward transfer,
and transfer diagnostics. Its appendix also studies same-morphology MuJoCo
friction shifts. None of those protocols or metrics are implemented here.

## Development protocol

Each frozen development seed runs four arms for the same bounded number of
environment steps, observations, online table updates, action queries,
descriptor queries, and boundary disclosures:

- `diverse_archive`: distance-filtered immutable policy snapshots;
- `one_model`: the latest snapshot only;
- `fixed_snapshot`: the first boundary snapshot only;
- `mechanism_off`: bypasses the archive and retains the same fixed anchor.

The live adapter uses the repository's action-dependent two-state switching
environment. A policy is a 2x2 float32 action-value table. At each declared
phase boundary, a JIT-compatible descriptor summarizes state occupancy, action
occupancy, reward, and action switching. The descriptor has no learned state.
Task boundaries are visible only to archive maintenance; task identity is not
an input to the policy, future tasks are hidden, and prior environments cannot
be queried.

The fixed-snapshot and archive-off paths must have identical observation,
action, reward, initial-policy, and final-policy hashes. Resource receipts
separately count environment steps, observations, updates, policy/descriptor/
archive queries, disclosed boundaries, payload bytes, active policy bytes,
archive or anchor bytes, and environment-state bytes. Timing is absent and
telemetry-only. Every valid development outcome, including a tie or regression,
must be retained outside this smoke under a new nonpromoting path; this CLI does
not write `outputs/`.

Run the CI-cheap smoke or print only its catalog:

```bash
.venv/bin/asi-telapa-qualification-smoke --steps 32 --phase-length 4
.venv/bin/asi-telapa-qualification-smoke --catalog
```

## Gates still closed

- obtain and content-bind the license file declared by the official README;
- reproduce the exact dependency/runtime lock and source/config identities;
- implement the paper environments, curricula, PPO and MAP-Elites budgets;
- implement the learned embedder, normalization, anchors, replay, alignment,
  re-embedding, trajectory banks, and few-shot selection;
- match all environment, evaluation, optimizer-reset, task-information, seed,
  query, compute, peak-memory, and persistent-state contracts;
- run strong paper baselines and causal archive/descriptor/maintenance
  ablations, then an untouched preregistered scientific evaluation.

Until all gates close, the lane cannot support TeLAPA parity, performance,
transfer, continual-learning benefit, robotics readiness, or SOTA claims.
