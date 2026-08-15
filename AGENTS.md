# ASI — agent guide

ASI is elizaOS's JAX continual-learning research and hillclimbing project. Its
north star is one end-to-end agent that keeps learning through an operational
life, retains and reuses useful knowledge, adapts without whole-agent or
task-by-task reinitialization, operates within explicit compute, memory, and
latency budgets, and scales from research benchmarks to real work — especially
robotics.
State-of-the-art continual learning is the destination, not a current claim.

[The Alberta Plan](https://arxiv.org/abs/2208.11173) is a foundational
inspiration and a source of inherited mechanisms, vocabulary, and file names.
It is not ASI's specification or outer boundary. Follow the evidence wherever
it leads: improve Alberta-derived ideas, combine them with other continual-
learning and reinforcement-learning methods, or replace them when stronger
concepts win.

This tree is a **development fork** of `lalalune/alberta` (fork point
`2ac3533`), not a lightly-patched vendor copy — see `VENDORING.md` for the
divergence summary and canonical upstream URL. ASI is the project identity;
`alberta_framework`, the `alberta-framework` distribution, `alberta-*` CLIs,
and historical schema IDs remain compatibility and provenance interfaces. Do
not casually rename them. The robot track imports the continual-RL subset
in-process; keep `requires-python >= 3.12`, the `numpy >= 1.26` floor, and the
existing import surface intact.

**Current program hillclimb:** continue implementing the selected shared
reference-agent protocol in `docs/design/asi-reference-agent-protocol.md`. The
versioned L0 transaction ledger in `alberta_framework/reference_agent.py` and
its 17 retained tests cover immutable typed payloads, separate authorization,
settlement, receipt, and outcome records, explicit reset identities, and
fail-closed phase/rejection semantics. It is not a current `reference-dev`
designation: concrete Prototype and robot adapters, aggregate life state and
runner, adapter-level dispatch settlement, whole-life checkpoint/exact resume,
and a CI-cheap regression panel still have to pass the Proposed ADR's remaining
acceptance sequence. The L0 receipt is an executor acknowledgement, not proof
of physical dispatch. The current robot and Forager paths do not consume
`PrototypeAgent`, and Forager's extended-action dispatch edge remains open. In
the monorepo, use
`../robot/docs/asimov-1.md` and
`../robot/docs/ALBERTA_PRODUCTION_READINESS.md` as the existing ASIMOV-1
application interface and open-gate record; do not create a duplicate robotics
ladder or treat its smoke plumbing as performance evidence.

**Current measured subsystem campaign:** IPMNIST development screening and
development confirmation is one plasticity/conditioning lane, not the
definition of ASI. It is permanently nonpromoting. Results move; read the
`summary_*.json` files and `publication_runs/RESULTS.md` under
`outputs/ipmnist_screening/` instead of copying numbers into overview docs,
and re-measure the selected control before any A/B. The theory snapshot is
`docs/research/ipmnist-theory.md`; raw records and audits live beside the
outputs. Check `docs/evidence/negative-results.md` before retrying an idea.

## Research operating loop

- **Measure the live baseline.** Bind the current source, workload, seeds,
  resources, and pre-update metric before comparing a change.
- **Name the bottleneck and hypothesis.** State the predicted benefit, causal
  mechanism, resource cost, ablation, and failure condition before a long run.
- **Build an end-to-end slice.** Prefer changes that run through the existing
  agent and environment interfaces over isolated surfaces with no consumer.
- **Screen cheaply and honestly.** Development runs select and reject ideas;
  use paired schedules and strong baselines, and never treat screening as
  promotion.
- **Test transfer, retention, and control.** A local score improvement is
  provisional until it survives recurrence or distribution change, resource
  accounting, and a downstream agent/control check.
- **Advance development, then evaluate scientifically.** A matched development
  win may enter the explicitly nonpromoting `reference-dev` configuration
  after its regression panel passes. Freeze a separate protocol with fresh
  seeds only when a claim warrants scientific evaluation.
- **Integrate and remeasure.** Record negative results, keep resource-acceptable
  wins in the appropriate reference channel, and rerun the whole-life
  scorecard before the next hillclimb.

Prioritize work that closes an integration gap or resolves a high-value
uncertainty. More APIs, mechanisms, or tests are not progress by themselves.
The durable strategy and application ladder live in
`docs/research/asi-roadmap.md`.

## Layout

```
alberta_framework/
  core/        learners and adaptive optimizers, Horde, prediction/control,
               learned state, memory, world models, dreaming,
               options/STOMP/OaK, feature lifecycles, and PrototypeAgent
  streams/     synthetic prediction streams + gauntlet, closed_loop,
               pavlovian, recurring_multiagent
  evaluation/  strict evidence artifacts, validators, the evidence registry,
               evidence CLIs, and bounded development diagnostics
  benchmarks/  IPMNIST lanes (upgd_ipmnist, ipmnist_screening,
               upgd_label_emnist), Forager integrations
  utils/       multi-seed experiments, statistics, metrics, export
  steps/       inherited Alberta Step 1–12 kernels: smoke CLIs for Steps 1–2,
               pipeline.py consumes Steps 3–4, Steps 5–12 are
               library-surface only; this is not ASI's outer roadmap
outputs/       evidence + campaign artifacts — see immutability rules below
tests/         unit, integration, scientific, and replay checks
```

Key documents:

- Mission and hillclimb ladder: `docs/research/asi-roadmap.md`
- Proposed reference-agent protocol: `docs/design/asi-reference-agent-protocol.md`
- Implemented L0 transaction ledger: `alberta_framework/reference_agent.py`
- Status & evidence: `docs/status.md` (levels L0–L3, completion gates) ·
  `docs/evidence/methodology.md` (property-by-property map)
- Active campaign: `docs/research/ipmnist-theory.md` ·
  `outputs/ipmnist_screening/{RUNBOOK,FINAL_REPORT,AUDIT,CEILING_ANALYSIS,SOTA_LANDSCAPE_2026}.md`
- Durable records: `docs/archive/forager-comparator-audit.md` ·
  `docs/design/rtu-taylor-correction.md` ·
  `docs/evidence/negative-results.md`
- Runbook: `docs/runbooks/foragax-open-screen.md`
- Benchmarks/infra: `FORAGER_BENCHMARK.md` ·
  `docs/archive/historical-forager-reconstruction.md` · `VENDORING.md` · `CHANGELOG.md`

`README.md` is the index; if you add a root doc, link it there.

## Running things

Always use the project venv:

```bash
.venv/bin/python -m pytest tests/<file> -q                  # one file
.venv/bin/python -m pytest tests -q                         # full suite
.venv/bin/python -m pytest --collect-only -q | tail -1     # count of record
.venv/bin/python -m ruff check .                           # lint (line length 100)
.venv/bin/python -m mypy                                   # strict, py312
.venv/bin/alberta-evidence-status                          # evidence registry
```

See `[project.scripts]` in `pyproject.toml` for the current console-script
inventory. The ones you'll reach for are `alberta-evidence-status`,
`alberta-forager-benchmark`, `alberta-foragax-open-screen`, and the
`alberta-forager-matched-*` family. Benchmark executions happen through
scripts/CLIs, never inside pytest — tests must stay CI-cheap unless
explicitly registered as a scientific lane.

## Marker lanes

- `unit` — fast isolated behavior/contract tests; never scientific evidence.
- `integration` — spans components, persistence, or process/CLI boundaries.
- `scientific` — frozen promoted-evidence protocols; may be expensive and
  require preregistered seeds.
- `slow` — wall-clock heavy modules (>~30s serial); excluded from the fast
  per-PR CI lane (`-m "not slow"`).

## Evidence-promotion rules (fail-closed)

- **Never auto-promote.** Passing tests, replays, or reruns do not upgrade a
  claim. Promotion requires a frozen preregistered protocol, untouched
  held-out seeds, a versioned artifact schema, and its strict validator
  accepting the artifact.
- **Frozen seeds stay frozen.** Calibration/development seeds and consumed
  evidence seeds can never be reused for promotion. Consumed-seed replays are
  explicitly nonpromoting.
- **Pinned `outputs/` artifacts are immutable.** Never overwrite, edit, or
  delete `outputs/ftl_decision/` (sha-pinned), `outputs/continual_ia/`
  (historical chain + source snapshot), `outputs/recurring_feature/`,
  `outputs/scale_robust_feature/evidence.v2.json`,
  `outputs/continual_multiagent/`, `outputs/step2_canonical/`,
  `outputs/evidence_manifest.json`, the sealed/`QUARANTINED.md` forager
  roots, or the chmod-frozen negative-result dirs. New runs write to NEW
  paths and new schema versions. `outputs/ipmnist_screening/` and
  `outputs/upgd_ipmnist/` hold the active campaign's development artifacts —
  append, don't rewrite.
- **Registered source hashes are load-bearing.** Editing a registered source
  file invalidates persisted evidence until the frozen protocol is rerun; the
  registry reports `invalid` (exit 2) — that is working-as-designed, not a
  bug to silence. Read `alberta_framework/evaluation/evidence_manifest.py`
  for the current
  five-claim source inventory, and inspect each development lane's own source
  manifest before touching it. Counts in narrative docs are not authoritative.
- Thresholds are calibrated empirically on development data with ≥2x margins,
  then frozen. Retuning a threshold after seeing held-out results is
  disallowed (a failed gate stays a valid rejection).
- Library changes are failing-test-first; state is frozen chex dataclasses;
  RNG uses explicit `jr.key(...)` seeds.

## Evidence registry (5 claims)

`alberta-evidence-status` exits `0` (all accepted), `1` (valid rejection or
missing), `2` (invalid). Each claim's CLI is also
`.venv/bin/python -m alberta_framework.evaluation.<module>`.

Run the command for live status. A claim becomes `invalid` when its registered
source bytes no longer match the pinned artifact; unrelated dirty-worktree
changes alone are not a registered-source mismatch. The frozen outcomes
recorded in the pinned artifacts are:

| Claim | Frozen artifact outcome | Artifact | CLI |
|---|---|---|---|
| `recurring_pair_features` | accepted (narrow L2) | `outputs/recurring_feature/evidence.v1.json` | `alberta-recurring-feature-evidence` |
| `scale_robust_pair_features` | accepted (narrow L2) | `outputs/scale_robust_feature/evidence.v2.json` | `alberta-scale-robust-evidence` |
| `ftl_world_model_decision_fidelity` | accepted (historical chain) | `outputs/ftl_decision/evidence.v1.json` | `alberta-ftl-evidence` |
| `recurring_multiagent_coadaptation` | accepted (narrow L2) | `outputs/continual_multiagent/evidence.json` | `alberta-multiagent-evidence` |
| `continual_intelligence_amplification` | valid rejection (frozen 10% gate) | `outputs/continual_ia/evidence.json` | `alberta-ia-evidence` |

No accepted claim establishes an integrated ASI agent, robotics readiness,
state of the art, or Alberta Plan completion; keep README/status wording narrow
and honest.

## Files that are load-bearing outside the docs

- `FORAGER_BENCHMARK.md` is hashed into Forager run provenance
  (`forager_cli._source_tree_sha256`) — edits change benchmark receipts.
- `README.md`, `CHANGELOG.md`, and `FORAGER_BENCHMARK.md` ship in the sdist.
- The CHANGELOG version heading is asserted by `test_release_metadata.py`.
- The robot track imports `core/{actor_critic,continual_backprop,
  initializers,normalizers,optimizers,sarsa}` via `import
  alberta_framework` — `alberta_framework/__init__.py` must stay importable,
  so every module deletion is a two-file change.

## Conventions

- ruff line length 100; ESM/TS conventions do not apply here — this is a pure
  Python track.
- Use **ASI** for the current project and research program. Preserve Alberta
  names when referring to upstream history, Plan-specific mechanisms, the
  compatibility package/CLI surface, or historical evidence identifiers.
- `CLAUDE.md` and `AGENTS.md` are identical: author `CLAUDE.md`, copy to
  `AGENTS.md`.
- No git commits unless explicitly asked.
