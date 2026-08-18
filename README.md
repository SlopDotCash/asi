# ASI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)

ASI is elizaOS's evidence-driven continual-learning hillclimb. The objective is one agent that
keeps learning through an operational life: adapting without whole-agent or task-by-task
reinitialization, retaining and reusing useful knowledge, predicting and planning, and acting
within explicit compute, memory, and latency budgets. Robotics is the first demanding
application direction.

State-of-the-art continual learning is the destination, not a claim about this checkout. ASI
does not yet have a selected integrated `reference-dev` agent, a completed whole-agent L3
result, a protocol-comparable external SOTA result, or robotics readiness.

## The hillclimb

This repository is organized around a repeated empirical loop:

1. remeasure the current control from the current source;
2. name one bottleneck and a falsifiable mechanism hypothesis;
3. implement the smallest end-to-end intervention that can test it;
4. screen it cheaply with paired schedules and strong controls;
5. retain or reject it, including its resource cost;
6. test recurrence, transfer, and downstream control;
7. advance the permanently nonpromoting development reference only after its regression panel;
8. freeze a separate protocol with untouched seeds only when a scientific claim is warranted.

The target is not the number of modules, tests, or favorable subsystem scores. A step is uphill
only when useful lifetime behavior improves without concealing regressions in adaptation,
retention, stability, autonomy, robustness, compute, memory, or latency.

Start with the [research roadmap](docs/research/asi-roadmap.md), the
[current status map](docs/status.md), and the
[protocol-aware SOTA and research library](docs/research/sota-landscape.md). Contributors should
then follow [CONTRIBUTING.md](CONTRIBUTING.md), which turns an idea or paper into a bounded
comparison and defines what evidence is required to call it a win.

## Current scoreboard

Different protocols answer different questions. The table deliberately does not combine them
into one leaderboard.

| Lane | Best truthful statement in this checkout | Status |
|---|---|---|
| Reference life | A 144-shard matched scorecard is implemented for SwitchingTwoState and RiverSwim over Prototype, frozen, random, privileged finite-horizon DP, differential SARSA, and discounted SARSA controls. | No completed aggregate result; no `reference-dev` selection; permanently nonpromoting. |
| IPMNIST plasticity | `rls_head_resid_l1_preset005` records **0.87114 ± 0.00010** whole-stream online accuracy over 20 seeds versus **0.86449 ± 0.00009** for its paired conditioning control; all paired differences are positive. | Stored development result on a protocol-extended readout; selected/inspected seeds and no frozen promotion authority. Not a scientific or external SOTA claim. |
| Protocol-pure IPMNIST | `adamw_cbp_r3e4` records **0.80126** over 3 development seeds; the stored published-configuration UPGD-W reproduction is **0.77915 ± 0.00006** over 10 seeds. | Development only; seed counts and method classes differ. Remeasure before a new A/B. |
| Forager | Open-screen infrastructure and development records cover feed-forward, recurrent/stateful, plasticity, PPO, and RTU-family controls. | No paper-length matched-resource ASI-versus-upstream result and no promoted claim. |
| Evidence registry | Four stored artifacts have narrow accepted outcomes and one has a valid-rejection outcome under their historical contracts. | **Live status on 2026-08-17: invalid (exit 2), all five claims**, principally because registered source hashes have drifted; IA also fails current canonical payload checks. This is fail-closed behavior, not a reason to edit an artifact. |

The current IPMNIST record is in
[`summary_rls_head_confirm.json`](outputs/ipmnist_screening/summary_rls_head_confirm.json) and is
explained in the [campaign index](docs/research/ipmnist-campaign-index.md). It is valuable evidence
for choosing the next experiment, but cannot be promoted retroactively. As of the literature
audit dated 2026-08-17, no later paper was found reporting a directly comparable number on the
exact ICLR-2024 IPMNIST protocol. That is a search result, not proof of SOTA; a defensible claim
still requires an updated systematic search, external implementations under one frozen runner,
matched resources, and fresh held-out evaluation.

## What to hillclimb now

The highest-leverage work is tied to an existing bottleneck and a downstream consumer:

1. **Run the reference-life scorecard.** Complete the implemented SwitchingTwoState + RiverSwim
   development comparison, validate every shard, and decide whether any candidate qualifies.
2. **Challenge the IPMNIST leader.** Reproduce high-priority contemporary controls under the
   exact stream: L2 plus effective-rank regularization, AdamO/dynamical-isometry regularization,
   Intentional Updates adapted to supervised prediction, and bounded growing/elastic networks.
3. **Test whether the IPMNIST mechanism transfers.** Remeasure controls on recurrence,
   label/output changes, and a continual-control setting. Fast input normalization may erase
   task-relevant magnitude information; it is not assumed safe for robotics.
4. **Compare online world models.** Put the existing shallow FTL model against matched online
   recurrent/latent controls. JEPA-WM, V-JEPA 2-AC, Dreamer-CDP, JEDI, and Dreamer-style systems
   are research directions, not drop-in claims: replay, pretraining, pixel input, and compute must
   be declared and charged.
5. **Close a real control bridge.** Add a reference-life adapter for Forager or the sibling robot
   path only after action ownership, checkpoint/resume, resource, safety-authority, and metric
   gates are written down.

The research library records paper links, official code, protocol mismatches, implementation
state, and a prioritized comparison queue. Check it and the
[negative-results ledger](docs/evidence/negative-results.md) before opening a new mechanism lane.

## Research position

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) is a foundational
inspiration and a useful coverage lens. It is not ASI's specification, mandatory sequence, or
outer boundary. ASI may improve, combine, reorder, or replace Alberta-derived mechanisms when
stronger evidence wins.

The external comparison set therefore includes:

- streaming optimization and loss-of-plasticity methods such as continual backpropagation,
  UPGD, NaP, SNR, L2-ER, AdamO, Intentional Updates, and growing/elastic networks;
- replay-free and replay-based continual learning, with memory and task information charged;
- continual-control benchmarks including RiverSwim, Forager, Continual World, CORA, and COOM;
- online and latent world models including FTL Online Agent, Dreamer-family systems,
  reconstruction-free prediction, and JEPA-style physical planning; and
- standard libraries such as Avalanche and Mammoth as sources of baselines and protocol checks,
  not as interchangeable scoreboards.

Numbers from different Permuted-MNIST variants are especially easy to misuse. Task count,
examples per task, batch size, boundary access, replay, pretraining, architecture, evaluation
time, and whether accuracy is prequential, final, per-task, or last-window all change the
problem. ASI reports them separately until one runner makes them comparable.

## Implemented reference-life slice

The selected architectural direction is one reference-agent protocol shared by adapters. The
current `preview1` development slice includes:

- an immutable transaction ledger with authorization, settlement, command, applied-action
  receipt, outcome, halt/recovery, and process-local lock/CAS semantics;
- a manifest-bound primitive-action bridge for `PrototypeAgent`;
- aggregate SwitchingTwoState and RiverSwim lives that own agent, environment, transaction,
  dispatch, RNG cursor, metrics, counters, recovery state, transcript, and generations;
- exact validation of RiverSwim's keyed stochastic transition, with `2 <= n_states <= 12` bound
  before its exponential oracle is built;
- quiescent whole-life checkpoint and exact-resume gates for both simulators; and
- matched development controls and a fresh-process scorecard runner.

The relevant implementation and tests are indexed in
[the protocol ADR](docs/design/asi-reference-agent-protocol.md). These are L0 mechanisms. They do
not provide portable checkpoint migration, authenticated execution proof, process-death or
hardware-delivery guarantees, independent safety, options/rebinding/boundaries, general
environment support, a Forager or robot adapter, or learning-performance evidence.

`PrototypeAgent` remains a candidate composition surface, not the completed ASI agent. API
presence elsewhere in the package means a mechanism can be researched; it does not establish
empirical benefit or integration.

## Identity and compatibility

This repository is [`elizaOS/asi`](https://github.com/elizaOS/asi), a development fork of
[`lalalune/alberta`](https://github.com/lalalune/alberta) from commit `2ac3533`.
[VENDORING.md](VENDORING.md) records the divergence.

ASI is the project identity. The `alberta_framework` Python namespace, `alberta-framework`
distribution, `alberta-*` commands, historical schemas, and Alberta-specific Step names remain
compatibility and provenance interfaces. The robot track imports part of this surface in-process,
so Python 3.12 support, the NumPy 1.26 floor, and existing imports must remain intact.

## Install

The existing `alberta-framework` project on PyPI is a different distribution and does not track
this fork. Install from a checkout:

```bash
git clone https://github.com/elizaOS/asi.git
cd asi
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Optional extras:

```bash
.venv/bin/python -m pip install -e '.[gymnasium]'  # Gymnasium adapters
.venv/bin/python -m pip install -e '.[forager]'    # continual-foragax testbed
.venv/bin/python -m pip install -e '.[research]'   # plots, tables, dataset loaders
.venv/bin/python -m pip install -e '.[gpu]'        # JAX CUDA 12 build
```

## Quick start

This is a mechanism smoke example, not the end-to-end target:

```python
import jax.random as jr

from alberta_framework import Autostep, LinearLearner, RandomWalkStream, run_learning_loop

stream = RandomWalkStream(feature_dim=10, drift_rate=0.01)
learner = LinearLearner(optimizer=Autostep())
state, metrics = run_learning_loop(
    learner,
    stream,
    num_steps=10_000,
    key=jr.key(42),
)
```

Inherited Step probes likewise check execution, not scientific performance:

```bash
.venv/bin/alberta-step1-smoke --steps 256 --seed 0
.venv/bin/alberta-step2-smoke --steps 128 --seed 0
```

## Commands

| Commands | Purpose |
|---|---|
| `asi-reference-life-scorecard` | Build, run, aggregate, or validate the permanently nonpromoting reference-life scorecard |
| `alberta-evidence-status` | Validate the five-claim evidence registry and current source bindings |
| `alberta-step1-smoke`, `alberta-step2-smoke` | Run inherited nonpromoting mechanism probes |
| `alberta-forager-benchmark`, `alberta-historical-forager` | Run development comparisons or inspect historical Forager families |
| `alberta-foragax-open-screen`, `alberta-foragax-oci` | Operate bounded Foragax screening and qualified OCI workflows |
| `alberta-forager-matched-*` | Operate matched Forager qualification/evaluation campaign stages |
| `alberta-*-evidence` | Validate or build one versioned narrow evidence artifact under its own contract |

Every command supports `--help`. Benchmark execution belongs in scripts and CLIs, not ordinary
pytest. Read [FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md) and the
[open-screen runbook](docs/runbooks/foragax-open-screen.md) before a Forager run.

## Package map

```text
alberta_framework/
  core/         online learners, optimizers, control, state, memory, world models,
                planning, options, feature lifecycles, and PrototypeAgent
  streams/      synthetic prediction, closed-loop, Pavlovian, and multi-agent streams
  evaluation/   evidence schemas, validators, registry, and bounded diagnostics
  benchmarks/   IPMNIST, Forager, and reference-life campaign runners
  utils/        experiment, metric, statistics, and export helpers
  steps/        inherited Alberta Step 1-12 kernels and integration surfaces
tests/          unit, integration, scientific, replay, and package checks
outputs/        immutable evidence plus append-only development campaign records
```

Most numerical state is represented by immutable Chex dataclasses and JAX PyTrees. Randomness
uses explicit JAX keys. Host orchestration, strict artifact validation, external benchmark
loading, and bounded lifecycle operations remain Python-level where appropriate.

## Evidence rules

The evidence registry contains five narrow historical claim families: recurring pair features,
scale-robust pair features, FTL world-model decision fidelity, recurring multi-agent
coadaptation, and continual intelligence amplification. Their stored outcomes are four narrow
acceptances and one valid rejection. On 2026-08-17 the live current-tree validator classified all
five as `invalid` and exited `2`, principally due to registered source-hash drift; the IA chain
also reports current canonical payload mismatches. Preserve both facts: historical frozen outcome
and current validity are different fields.

```bash
.venv/bin/alberta-evidence-status
```

| Exit | Meaning |
|---:|---|
| `0` | all registered narrow claims are accepted |
| `1` | an artifact is missing or is a valid rejection |
| `2` | an artifact or source binding is invalid |

Pinned output artifacts are immutable. Never repair, overwrite, or delete one. Registered source
hash drift invalidates the corresponding persisted claim by design. Development seeds and
consumed evidence seeds can never become fresh promotion seeds. Passing tests, rerunning a
screen, or accumulating more inspected seeds does not promote a claim.

Read the [evidence methodology](docs/evidence/methodology.md) before editing evaluation sources or
running a claim-bearing protocol.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the inbound licensing terms and
maintainer review policy.

Use the project environment for every command. Run targeted tests first, then broaden
verification as appropriate:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -q
.venv/bin/python -m pytest -m "not slow and not package" tests -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/asi-benchmark-catalog doctor ipmnist-iclr2024 reference-life
```

The pytest lanes are `unit`, `integration`, `scientific`, `slow`, and `package`. A scientific
marker identifies a frozen evidence protocol; it does not make an arbitrary expensive test into
evidence. See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment proposals, comparison hygiene,
artifact rules, documentation expectations, and the pull-request checklist.

## Documentation

- [Research roadmap and whole-life scorecard](docs/research/asi-roadmap.md)
- [SOTA landscape, paper library, projects, and comparison backlog](docs/research/sota-landscape.md)
- [Continual benchmark suite setup](docs/runbooks/continual-benchmark-suite.md)
- [Current status and Alberta Plan crosswalk](docs/status.md)
- [Reference-agent protocol ADR](docs/design/asi-reference-agent-protocol.md)
- [IPMNIST campaign index](docs/research/ipmnist-campaign-index.md)
- [IPMNIST mechanistic synthesis](docs/research/ipmnist-theory.md)
- [Evidence methodology](docs/evidence/methodology.md)
- [Negative and bounded results](docs/evidence/negative-results.md)
- [Forager benchmark](FORAGER_BENCHMARK.md)
- [Documentation index](docs/README.md)

Citation metadata is in [CITATION.cff](CITATION.cff). Cite the original papers and benchmark
versions used by each experiment, including the Alberta Plan where applicable.

ASI is licensed under the [Apache License 2.0](LICENSE).
