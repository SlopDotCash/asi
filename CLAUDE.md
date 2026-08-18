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

**Current program hillclimb:** run a permanently nonpromoting matched
development scorecard across `SwitchingTwoStateMDP` and `RiverSwimMDP` with
frozen/no-learning, random, finite-horizon privileged dynamic-programming, and
strong SARSA-family controls plus explicit resource accounting. This is
development selection only; it does not populate `reference-dev` or create
performance or scientific evidence.

The scorecard implementation is in
`alberta_framework/{reference_life_controls.py,benchmarks/reference_life_scorecard.py}`
and runs through `asi-reference-life-scorecard`. Its literal-frozen plan has
12 consumed development seeds, 2 environments, and 6 arms (144 fresh-process
shards); every shard binds its current source identity, and aggregation requires
one matching current identity. The privileged normalization arm is an
environment-bound finite-horizon dynamic-programming control and is excluded
from candidate Pareto comparisons.
All agent RNG roots use explicit Threefry keys. Records enforce the fixed reward
lattice, complete counters, canonical initial/final numeric payload accounting
(including static oracle policy bytes), and permanently nonpromoting policy.
Timing remains telemetry-only until a separately qualified timing protocol
exists. The implementation and validators do not constitute a completed run or
performance result; consistency hashes are not authenticated execution proof.

The unfrozen `preview1` transaction protocol, pure reducer/CAS ledger,
primitive-only exact-dispatch Prototype bridge, and aggregate runner are in
`alberta_framework/{reference_agent,prototype_reference_adapter,reference_life}.py`.
One immutable aggregate owns agent, environment, transaction, dispatch, RNG
cursor, metrics, counters, recovery state, transcript, and generations. Its
process-local outer lock covers command issuance through execution, receipt,
outcome, learning/metrics staging, and one aggregate commit; retained tests
cover concurrency, horizon, strict execution validation, faults, and
no-redispatch pending-outcome recovery. The primitive RiverSwim slice uses a
distinct environment manifest/state discriminator and stationary metrics,
rejects configurations outside `2 <= n_states <= 12` before its exponential
oracle is constructed, passes the exact same runner-derived JAX key to
execution and validation, and replays the keyed stochastic transition during
strict validation.

The development-only checkpoint codec in
`alberta_framework/reference_life_checkpoint.py`, covered by
`tests/test_reference_life_checkpoint.py` and
`tests/test_reference_life_riverswim_checkpoint.py`, now implements the
current-schema quiescent whole-life checkpoint and exact-resume gate for the
primitive Prototype + SwitchingTwoState and Prototype + RiverSwim lives. A
successful save uses Linux atomic no-replace publication for one immutable
generation, advances commit/checkpoint generations, nests the complete
Prototype v3 checkpoint, binds the aggregate state plus current
source/runtime/dependency identities and consistency hashes, reconstructs fresh
components from the environment discriminator, validates strict
cross-component adoption, and produces exact original-versus-restored
continuation from the same persisted barrier.

This remains a `preview1` L0 mechanism, not a frozen or portable checkpoint
contract, `reference-dev`, safety conformance, robotics readiness, RiverSwim
learning or performance benefit, performance evidence, or scientific evidence.
Checkpoints support only quiescent pre-completion state for the two implemented
simulators and fail closed across current source/runtime/dependency drift; they
do not restore in-flight, halted,
pending-outcome, completed, or physical state. The hashes bind consistency but
are not authenticated execution attestation. The ordinary-`Exception`
at-most-once guard remains process-local and supplies no process-death,
`BaseException`, durable/idempotent executor, hardware-delivery, or
reconciled-resume guarantee. Independent safety/veto, options,
replacement/rebinding, boundaries, wire/durable dispatch replay,
additional/general environment support, Forager, and robot adapters remain
open. In the monorepo, use
`../robot/docs/asimov-1.md` and
`../robot/docs/ALBERTA_PRODUCTION_READINESS.md` as the existing ASIMOV-1
application interface and open-gate record; do not create a duplicate robotics
ladder or treat its smoke plumbing as performance evidence.

**Current measured subsystem campaign:** IPMNIST development screening and
development confirmation is one plasticity/conditioning lane, not the
definition of ASI. It is permanently nonpromoting. The current stored paired
200-task record is the protocol-extended `rls_head_resid_l1_preset005` arm at
0.87114 ± 0.00010 versus 0.86449 ± 0.00009 for its conditioning control over
20 consumed development seeds, all paired differences positive. This is not a
scientific or external SOTA claim. Results move; use
`docs/research/ipmnist-campaign-index.md` and the `summary_*.json` files rather
than copying rankings into overview docs, and re-measure the selected control
before any A/B. The theory snapshot is `docs/research/ipmnist-theory.md`; raw
records and audits live beside the outputs. Use
`docs/research/sota-landscape.md` for external papers, official projects,
protocol gaps, and the comparison queue. Check
`docs/evidence/negative-results.md` before retrying an idea.

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
- Implemented `preview1` transaction ledger, primitive Prototype bridge,
  aggregate Switching/RiverSwim life runner, quiescent exact-resume gates, and
  matched development-scorecard machinery:
  `alberta_framework/reference_agent.py` ·
  `alberta_framework/prototype_reference_adapter.py` ·
  `alberta_framework/reference_life.py` ·
  `alberta_framework/reference_life_checkpoint.py` ·
  `alberta_framework/reference_life_controls.py` ·
  `alberta_framework/benchmarks/reference_life_scorecard.py` ·
  `tests/test_reference_life.py` ·
  `tests/test_reference_life_checkpoint.py` ·
  `tests/test_reference_life_riverswim.py` ·
  `tests/test_reference_life_riverswim_checkpoint.py` ·
  `tests/test_reference_life_controls.py` ·
  `tests/test_reference_life_scorecard.py`
- Status & evidence: `docs/status.md` (levels L0–L3, completion gates) ·
  `docs/evidence/methodology.md` (property-by-property map)
- Contribution and experiment protocol: `CONTRIBUTING.md`
- Active campaign: `docs/research/ipmnist-campaign-index.md` ·
  `docs/research/ipmnist-theory.md` ·
  `outputs/ipmnist_screening/{RUNBOOK,FINAL_REPORT,AUDIT,CEILING_ANALYSIS,SOTA_LANDSCAPE_2026}.md`
- Current external paper/project library and comparison queue:
  `docs/research/sota-landscape.md`
- Machine-readable implementation backlog and benchmark setup:
  `docs/research/implementation-backlog.json` ·
  `docs/runbooks/continual-benchmark-suite.md` · `asi-benchmark-catalog`
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
inventory. The ones you'll reach for are `asi-reference-life-scorecard`,
`alberta-evidence-status`,
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
  per-PR CI lane.
- `package` — built-distribution and installed-entry-point contracts; isolated
  in the package CI lane.

The fast runtime selector is `-m "not slow and not package"`.

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

The current-tree audit on 2026-08-17 returns `invalid` (exit 2) for all five
claims, principally because the registered source hashes have drifted; the IA
chain also reports current canonical controller-budget payload mismatches.
Preserve the frozen outcomes above as history, but do not describe them as live
accepted claims. Do not edit pinned artifacts or weaken validators to repair
this expected fail-closed state.

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
