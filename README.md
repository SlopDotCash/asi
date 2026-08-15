# ASI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)

ASI is elizaOS's evidence-driven continual-learning research and hillclimbing project. We are
building one end-to-end agent that can keep learning throughout its operational life: adapt to
change, retain and reuse useful knowledge, build state and models, plan and act, and remain
within explicit compute, memory, and latency budgets. The intended application envelope
includes useful ongoing work and embodied systems such as robotics.

The goal is a state-of-the-art continual-learning application. That is an aspiration and a
comparative research target, not a claim about the current checkout. ASI does not yet have a
completed whole-agent L3 protocol or result, state-of-the-art application evidence, or
robotics readiness.

The repository provides a JAX package of online learners, adaptive optimizers, prediction and
control agents, learned-state and memory mechanisms, world models, planning and option
components, non-stationary streams, benchmarks, and strict evidence validators.

## Research direction

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) is a foundational
inspiration and a useful coverage lens. It is not ASI's specification, mandatory sequence, or
scope boundary. ASI can reorder, combine, revise, reject, or replace Alberta-derived mechanisms
and pursue ideas from the wider continual-learning and reinforcement-learning literature when
measured results justify the move.

The intended operating loop repeatedly remeasures the current lane-specific control, names a
falsifiable bottleneck, runs a bounded paired development screen, retains or rejects the idea,
checks transfer and system regressions, and issues a separate frozen held-out protocol only
when a scientific claim is warranted. Once a canonical reference life exists, the same loop
will remeasure its versioned whole-agent baseline. Improvements are judged across online
utility, adaptation, retention, transfer, stability, compute, memory, latency, and downstream
control — not one leaderboard number.

The durable mission, application ladder, whole-life scorecard, and current program priorities
are in the [ASI research roadmap](docs/research/asi-roadmap.md).

## Identity and compatibility

This repository is [`elizaOS/asi`](https://github.com/elizaOS/asi). It began from
[`lalalune/alberta`](https://github.com/lalalune/alberta) at fork point `2ac3533` and is now a
substantially divergent development line. [VENDORING.md](VENDORING.md) records that history.

ASI is the project identity. The Python namespace `alberta_framework`, distribution name
`alberta-framework`, `alberta-*` commands, Alberta-specific Step modules, and historical
`alberta.*` schemas remain stable compatibility or provenance names. The elizaOS robot track
imports part of that surface in-process, so this rebrand intentionally does not break it. The
package keeps Python 3.12 compatibility and a NumPy 1.26 floor. The name labels the research
program; it is not a claim about the current software's level of intelligence.

## Current status

Individual mechanisms range from contract-tested kernels to narrow historical evidence
packages. The retained `PrototypeAgent` is a candidate composition surface, not a completed
reference application. Important end-to-end, retention, control-benefit, resource-scaling,
robustness, and robotics gates remain open.

The selected architectural direction is a shared reference-agent protocol with adapters for the
retained `PrototypeAgent` and the sibling robot controller. The
[Proposed protocol ADR](docs/design/asi-reference-agent-protocol.md) specifies state ownership,
dispatch lineage, an exact-resume acceptance gate, and its ordered implementation sequence. The
[initial L0 transaction contract](alberta_framework/reference_agent.py) and its
[17 retained tests](tests/test_reference_agent_protocol.py) now cover immutable typed payloads,
separate authorization, learner settlement, dispatch receipt, and receipt-bound outcome records,
explicit reset identities, and a fail-closed functional phase ledger. This is not a concrete
adapter or whole-life conformance result. It does not populate `reference-dev`: no Prototype or
robot adapter, aggregate life state or runner, whole-life checkpoint, or exact-resume result
exists. The robot and Forager paths do not currently consume `PrototypeAgent`, and Forager still
records an unresolved extended-action dispatch edge.

The package also contains inherited surfaces related to all twelve steps of the Alberta Plan.
That crosswalk is useful for finding gaps, but completing a checklist of Plan mechanisms would
not by itself establish the ASI north star.

Keep these boundaries in mind:

- Development and screening records are permanently nonpromoting. A distinct run under a
  separately frozen protocol with untouched seeds may test a newly scoped scientific claim.
- A passing unit test, smoke run, replay, or benchmark does not promote a scientific claim.
- Registered evidence claims are narrow. Acceptance of one does not certify the package or
  establish ASI's whole-agent target, robotics readiness, state of the art, or Alberta Plan
  completion.
- Pinned artifacts are immutable historical records. Source drift makes compatibility checks
  fail closed; it is not repaired by editing the artifact or loosening its validator.
- Consumed development or evidence seeds cannot be reused as fresh promotion seeds.

See [the research status](docs/status.md) for the current capability and Alberta crosswalk and
[the evidence methodology](docs/evidence/methodology.md) for promotion rules, artifact
contracts, and validator semantics.

## Install

ASI currently requires Python 3.12 or newer. The existing `alberta-framework` project on PyPI is a different distribution and does not track this fork's version, Python floor, or dependency
extras. Install this project from its repository checkout:

```bash
git clone https://github.com/elizaOS/asi.git
cd asi
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

Optional dependency groups are available from the same checkout:

```bash
.venv/bin/python -m pip install -e '.[gymnasium]'  # Gymnasium adapters
.venv/bin/python -m pip install -e '.[forager]'    # continual-foragax testbed
.venv/bin/python -m pip install -e '.[research]'   # plots, tables, and dataset loaders
.venv/bin/python -m pip install -e '.[gpu]'        # JAX CUDA 12 build
.venv/bin/python -m pip install -e '.[dev]'        # tests, lint, typing, and research tools
```

For repository development, install the development extra and use the project virtual
environment for every command:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

## Quick start

This small example runs one online-learning primitive on a drifting synthetic stream. It is
useful for checking the library surface, not a demonstration of the target end-to-end agent.
JAX keys are explicit, and the learning loop uses `jax.lax.scan`.

```python
import jax.random as jr

from alberta_framework import (
    Autostep,
    LinearLearner,
    RandomWalkStream,
    run_learning_loop,
)

stream = RandomWalkStream(feature_dim=10, drift_rate=0.01)
learner = LinearLearner(optimizer=Autostep())

state, metrics = run_learning_loop(
    learner,
    stream,
    num_steps=10_000,
    key=jr.key(42),
)
```

The repository also exposes two inherited Alberta Step integration probes:

```bash
.venv/bin/alberta-step1-smoke --steps 256 --seed 0
.venv/bin/alberta-step2-smoke --steps 128 --seed 0
```

These commands check that the selected kernel runs and returns finite metrics. They are not
scientific experiments or evidence gates.

## Package layout

```text
alberta_framework/
  core/         online learners, optimizers, control, state, models, memory,
                planning, options, feature lifecycles, and agent composition
  streams/      synthetic prediction, closed-loop, Pavlovian, and recurring
                multi-agent streams
  evaluation/   evidence schemas, strict validators, registries, and CLIs
  benchmarks/   IPMNIST and Forager integrations and campaign runners
  utils/        experiment, metric, statistics, and export helpers
  steps/        inherited Alberta Step 1-12 mechanism kernels and smoke integration
tests/          unit, integration, scientific, and replay tests
outputs/        evidence and campaign artifacts; see the immutability rules
```

Most numerical state is represented by immutable Chex dataclasses and carried as JAX PyTrees.
Randomness is passed explicitly. Host orchestration, artifact validation, external benchmark
loading, and some bounded lifecycle operations remain Python-level by design.

The major package surfaces include:

| Area | Examples |
|---|---|
| Online prediction | `LinearLearner`, `MLPLearner`, TD learners, Horde |
| Adaptation | LMS, IDBD, Autostep, SwiftTD, UPGD, normalization, bounding |
| Control | SARSA, actor-critic, average-reward and off-policy variants |
| Continual mechanisms | learned state, feature lifecycles, memory, world models |
| Temporal abstraction | subtasks, STOMP, OaK, option models and bounded planning |
| Composition | `PrototypeAgent` candidate surface and explicit transition/decision ownership |
| Evaluation | versioned artifacts, strict validators, evidence registry |

API presence means that a mechanism is available for research. It does not imply empirical
benefit, calibrated thresholds, autonomous integration, or scientific acceptance.

## Evidence registry

From a repository checkout, inspect every registered claim with:

```bash
.venv/bin/alberta-evidence-status
```

The exit-code contract is:

| Code | Meaning |
|---:|---|
| `0` | every registered narrow claim is accepted |
| `1` | at least one artifact is missing or is a valid rejection |
| `2` | at least one artifact is invalid |

The registry validates artifact schema, protocol metadata, and registered source hashes. It is
an operational index of narrow claims, not a package-wide evidence score, hillclimb metric, or
completion certificate.

Wheels and source distributions intentionally exclude `outputs/`. Consequently, running the
status command from a normal package installation reports missing artifacts. Use a checkout
when validating the repository's stored evidence chain.

Do not overwrite, repair, or regenerate a pinned artifact in place. A new run must use a new
path and, when required by its contract, a new schema version. The full rules are in
[the evidence methodology](docs/evidence/methodology.md).

## Current measured subsystem campaign

IPMNIST development screening and development confirmation is the current measured plasticity
and conditioning lane. It generates and rejects mechanism hypotheses; it is not the end-to-end
target or the top-level reference-life hillclimb. The lane is permanently nonpromoting. Results
change as new shards are appended, so this README does not copy arm rankings, means, test
counts, or seed counts.

Use the mutable campaign index to find the current record. The output documents are
append-only chronological records, so their historical words such as "final" or "current"
may have been superseded by a later summary.

- [Current IPMNIST campaign index](docs/research/ipmnist-campaign-index.md)
- [IPMNIST theory and forward hypotheses](docs/research/ipmnist-theory.md)
- `outputs/ipmnist_screening/summary_*.json` for stored measurement records
- [Chronological campaign runbook](outputs/ipmnist_screening/RUNBOOK.md)
- [Historical accumulated report](outputs/ipmnist_screening/FINAL_REPORT.md)
- [Historical artifact and reproducibility audit](outputs/ipmnist_screening/AUDIT.md)
- [Pre-RLS publication-run record](outputs/ipmnist_screening/publication_runs/RESULTS.md)

Remeasure the intended baseline under the current development protocol before making an A/B
comparison. A survivor still needs complementary-stream, resource, and downstream-control
checks before it is an integrated improvement. Do not infer an ASI, robotics, scientific, or
state-of-the-art claim from the screening record.

Forager integration and comparator details are in
[FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md).

Before repeating a failed or bounded idea, check
[the negative-results ledger](docs/evidence/negative-results.md).

## Development and testing

Run targeted tests first, then broaden verification as appropriate:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
```

The repository uses these pytest markers:

- `unit`: fast, isolated behavior or contract tests
- `integration`: component, persistence, process, or CLI boundaries
- `scientific`: frozen promoted-evidence protocols
- `slow`: wall-clock-heavy tests excluded from the fast per-change lane

Benchmark campaigns run through their scripts or console CLIs, never as ordinary pytest work.
Keep tests CI-cheap unless the protocol is deliberately registered as scientific evidence.

Library changes should start with a failing test. Preserve immutable state, explicit
`jax.random` keys, Python 3.12 support, and the NumPy 1.26 minimum. Before editing evaluation
or benchmark sources, check whether a stored artifact registers their hashes.

Research changes should also start from a freshly measured control and a named integration
path. Prefer a small falsifiable comparison that can remove an idea over a broad new mechanism
surface with no whole-agent consumer. Use the process in the
[ASI research roadmap](docs/research/asi-roadmap.md).

Do not auto-promote results, retune a frozen threshold after seeing held-out data, reuse
consumed seeds, or modify immutable `outputs/` records. See
[the evidence methodology](docs/evidence/methodology.md) before changing any evidence lane.

## Documentation

### Mission and strategy

- [ASI research roadmap and whole-life scorecard](docs/research/asi-roadmap.md)
- [Proposed ASI reference-agent protocol](docs/design/asi-reference-agent-protocol.md)

### Status and evidence

- [Research status and completion gates](docs/status.md)
- [Evidence methodology and property map](docs/evidence/methodology.md)
- [Negative and bounded results](docs/evidence/negative-results.md)
- [Repository anti-LARP audit](docs/audits/repository-larp-audit.md)

### Runbooks

- [Foragax open development screen](docs/runbooks/foragax-open-screen.md)

### Research and historical audits

- [IPMNIST theory](docs/research/ipmnist-theory.md)
- [Current IPMNIST campaign index](docs/research/ipmnist-campaign-index.md)
- [RTU Taylor-correction derivation](docs/design/rtu-taylor-correction.md)
- [Forager comparator audit](docs/archive/forager-comparator-audit.md)
- [Historical Forager reconstruction](docs/archive/historical-forager-reconstruction.md)
- [OPMNIST closure provenance](outputs/step2_canonical/step2_opmnist_solution_800task_3seed_PROVENANCE.md)

### Repository and benchmark records

- [Forager benchmark](FORAGER_BENCHMARK.md)
- [Vendoring and fork history](VENDORING.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)

## Citation

ASI citation metadata is provided in [CITATION.cff](CITATION.cff). Cite the original papers
for algorithms and benchmarks used in a particular experiment, including the
[Alberta Plan](https://arxiv.org/abs/2208.11173).

## License

ASI is licensed under the [Apache License 2.0](LICENSE).
