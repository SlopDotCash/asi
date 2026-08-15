# ASI research roadmap

Status: living strategy. This document defines the direction of the project; it
does not certify current capability or scientific evidence. Current
implementation and evidence are tracked in [the status map](../status.md), and
promotion rules live in [the evidence methodology](../evidence/methodology.md).

## Mission

ASI is an evidence-driven continual-learning hillclimbing project. The goal is
an end-to-end agent that keeps learning throughout one operational life:
adapting to change, retaining and reusing useful knowledge, discovering useful
state and predictions, planning and acting, and doing so within bounded compute,
memory, and latency. The intended application envelope includes useful ongoing
work and embodied systems such as robotics.

The target is state-of-the-art continual learning in an application, not a
collection of state-of-the-art component scores. ASI does not currently claim
that target; no whole-agent L3 protocol or result has been completed, and the
project is not robotics-ready.

[The Alberta Plan](https://arxiv.org/abs/2208.11173) is a major source of ideas
and a valuable coverage lens. It is not a binding sequence, architecture, or
scope boundary. ASI may reorder, combine, revise, reject, or replace its
mechanisms and pursue ideas from continual learning, reinforcement learning,
streaming optimization, representation learning, memory, world models,
exploration, control, and other relevant research.

## What the target means

"State of the art" is comparative and time-dependent. It must be established
against strong contemporaneous baselines under a frozen, matched protocol; it
cannot be inferred from an internal record or a dated literature table. An ASI
application is successful only when all of the following are true together:

1. **Continual adaptation.** The agent improves from an ongoing stream without
   benchmark-only reinitialization. Any replay, task cue, boundary signal, or
   offline work is explicit and charged to the resource budget.
2. **Retention and reuse.** It preserves, reacquires, and transfers useful
   knowledge while remaining plastic enough to learn genuinely new behavior.
3. **End-to-end benefit.** Learned state, prediction, memory, planning, and
   control are judged by downstream lifetime behavior, not merely by isolated
   component diagnostics.
4. **Strong comparisons.** It beats or meaningfully extends competitive
   continual and non-continual baselines without hiding losses on retention,
   robustness, or cost.
5. **Bounded operation.** Compute per event, memory growth, latency, checkpoint
   behavior, and long-horizon numerical stability are measured and compatible
   with the target application.
6. **Application transfer.** Gains survive a ladder from controlled streams to
   continual control, embodied simulation, and guarded real-world workloads.
   A benchmark-only win is a subsystem result.
7. **Reproducible evidence.** Claims survive preregistered held-out evaluation,
   causal ablation, source and environment binding, and independent scrutiny.

Every SOTA statement must be scoped to a named benchmark and version, task
distribution, resource envelope, primary statistic and uncertainty/test,
multiple-comparison rule, and baseline roster with an as-of date. The broader
application target is a bundle of separately passed capability and operational
gates, not one global SOTA label.

No single metric can prove this target. A method that improves accuracy while
destroying adaptation speed, lifetime return, latency, or safety has not moved
the whole application uphill.

## One reference agent

Research should converge on one runnable reference composition rather than a
growing set of disconnected mechanisms. The retained `PrototypeAgent` and the
robot-imported continual-RL modules are candidate integration surfaces, not a
completed application. They may be simplified, extended, or replaced when an
explicit migration is justified by evidence.

The architectural direction is selected in the
[Proposed reference-agent protocol ADR](../design/asi-reference-agent-protocol.md):
one semantic lifecycle, dispatch, state-ownership, and exact-resume contract with
adapters for `PrototypeAgent` and the sibling robot controller. The
[versioned L0 transaction ledger](../../alberta_framework/reference_agent.py) and
its [17 retained tests](../../tests/test_reference_agent_protocol.py) now cover
immutable typed payloads, distinct authorization/settlement/receipt/outcome
records, explicit reset identities, and fail-closed phase and rejection
semantics. This resolves the initial host transaction slice, not completion of
the implementation or baseline selection. No concrete adapter, aggregate life
state or runner, whole-life checkpoint, or exact-resume result exists. The robot
path still does not import `PrototypeAgent`, the retained Forager agent still
excludes it because closed-loop dispatch has not demonstrated preservation of
the OaK/STOMP credited extended action, and neither adapter has passed the ADR's
acceptance sequence. None is already canonical.

Every proposed subsystem should answer four questions:

- Which observation-to-action path consumes it?
- Which owner updates and checkpoints its state?
- Which whole-life metric should improve, and what is the matched control?
- What are its incremental compute, memory, latency, and failure costs?

An implementation with no credible consumer belongs in a research note or
branch, not the reference agent. An integrated change remains provisional until
the system-level regression panel passes.

ASI separates three channels:

- **`reference-dev`** is the designation that will hold the current integrated
  development baseline. It is unpopulated until the Proposed protocol's
  conformance, exact-resume, adapter, and whole-life regression gates pass and a
  separate decision names a concrete configuration. Once established, it may
  advance after a matched development comparison and the complete regression
  panel, but it and its source data remain permanently nonpromoting.
- **A future release reference** advances only through a versioned,
  predeclared engineering acceptance and rollback policy. Passing that policy
  does not create a scientific claim.
- **Scientific claims** use separately frozen protocols, untouched held-out
  data, and their own acceptance authority. They never retroactively promote
  `reference-dev` or release-acceptance runs.

## Hillclimb loop

Each research cycle follows the same loop:

1. **Reproduce the starting point.** Measure the selected baseline with the
   current source, data, environment, seeds, and resource accounting. Historical
   means are context, not a live control.
2. **Localize a bottleneck.** Use traces, ablations, failures, and scaling curves
   to identify the limiting behavior. Write a falsifiable mechanism hypothesis
   and a stop condition.
3. **Choose the smallest coherent intervention.** Prefer a change that exercises
   the real learning and decision path. Declare expected benefit, integration
   surface, comparison arm, and cost before the expensive run.
4. **Run a bounded development screen.** Use paired schedules, multiple seeds,
   strong baselines, and enough horizon to expose recurrence or instability.
   Development data may select ideas but can never promote them.
5. **Retain or reject.** Keep an idea only when the predicted mechanism and
   practical effect survive the planned controls. Record negative, bounded, and
   superseded outcomes in the durable ledger.
6. **Test generality.** Move a surviving idea to a complementary stream and then
   a downstream control or agent setting. Measure the complete scorecard and
   reject benchmark-specific regressions unless the scope is intentionally
   narrow.
7. **Advance `reference-dev`.** A resource-acceptable development winner may
   become the next development baseline after its full regression panel passes.
   Record the decision and keep its nonpromotion status explicit.
8. **Evaluate scientifically when warranted.** Freeze a new protocol, untouched
   held-out seeds, artifact schema, source closure, thresholds, and validator
   before observing a claim-bearing result. This is not required for every
   exploratory hillclimb and does not relabel earlier runs.
9. **Rebaseline.** Rerun earlier panels against the selected reference channel
   and begin the next cycle from that measured state.

The project optimizes for information gain and end-to-end leverage, not the
number of modules, experiments, or passing tests. High-risk ideas are welcome
when the screen is cheap and the failure teaches something durable.

## Application ladder

The `R0`–`R5` stages route work toward an application. They are orthogonal to
the L0–L3 evidence-strength levels and to promotion class. For example, an R2
subsystem screen can remain development-only, while a narrow L2 scientific
claim need not advance the whole application beyond R1 or R2.

| Stage | Purpose | Typical evidence | What it cannot establish |
|---|---|---|---|
| R0. Contracts | Numerical, state, ownership, serialization, and API correctness | Unit and integration tests | Learning benefit |
| R1. Diagnostics | Isolate plasticity, retention, scale, recurrence, and causal mechanisms | Short controlled streams and ablations | Application value |
| R2. Subsystems | Compare learners and representations over meaningful continual horizons | IPMNIST, label-permutation, recurrence, and related paired campaigns | End-to-end control or robotics |
| R3. Continual control | Measure lifetime return, recovery, learned state, and planning under matched budgets | Continuing-control suites and Forager-class environments | Embodied reliability |
| R4. Embodied simulation | Exercise perception-to-action adaptation, disturbances, latency, and recovery | Reproducible robot task sequences and fault panels | Physical readiness by itself |
| R5. Guarded application | Validate useful behavior under real interfaces and operational constraints | Staged canaries with independent safety and rollback authority | Unbounded generality |

A candidate advances only when its lower-stage benefit remains visible at the
next relevant stage. R0–R1 stay cheap enough for routine development;
campaign and application execution happens through explicit runners, not by
turning pytest into a benchmark scheduler.

## Whole-life scorecard

Every integrated comparison declares the relevant metrics before execution.
At minimum, the scorecard considers:

| Dimension | Questions to answer |
|---|---|
| Online utility | What reward, accuracy, or task success is achieved before each learning update? |
| Adaptation | How much utility is lost after change, and how quickly is it recovered? |
| Retention | What remains when a condition recurs, and is reacquisition faster than first learning? |
| Transfer | Does prior learning help or obstruct new tasks, contexts, morphologies, or partners? |
| Stability | Are learning state, predictions, and actions finite and reliable over the full horizon? |
| Resources | What are per-event compute, peak and growing memory, latency, model queries, and stored experience? |
| Autonomy | Which task IDs, boundaries, labels, demonstrations, resets, or oracle features are supplied? |
| Robustness | What happens under scale shifts, observation faults, delayed feedback, and changed dynamics? |
| Application fit | Can the same agent interface checkpoint, resume, respect safety authority, and meet the target control rate? |

Exact metrics and thresholds belong to each protocol. The scorecard prevents a
single favorable mean from concealing a system-level regression.

Every comparison also freezes an acceptance policy. Safety, numerical
stability, resource ceilings, and other application-critical dimensions are
hard guardrails with declared noninferiority margins. Among candidates that
pass them, advancement requires either a preregistered primary-metric
improvement or a preregistered Pareto/utility rule over the remaining
dimensions. Post-hoc tradeoff weights cannot turn a regression into a win.

## Current program priorities

1. **Continue implementing the selected reference-life protocol.** Build the
   aggregate life configuration/state, authoritative runner, whole-life
   checkpoint, and adapter-level dispatch settlement around the completed L0
   transaction ledger. Then add the Prototype and robot adapters, exact-resume
   gate, and low-cost whole-life regression panel. Only then may a separate
   decision select one executable `reference-dev` configuration, environment
   interface, checkpoint contract, command, and rollback policy.
2. **Turn plasticity gains into agent gains.** Use the development-only IPMNIST
   campaign to generate mechanisms, then remeasure controls and test survivors
   on recurrence, a complementary stream, and continual control. Do not promote
   or directly ship an inspected campaign winner.
3. **Strengthen the benchmark ladder.** Maintain strong baselines, matched
   resource accounting, scaling curves, and separate development and held-out
   seed roles. Add a workload only when it tests a named open gate.
4. **Close the robotics path.** Adopt the sibling robot track's existing
   ASIMOV-1 sequence (`stand_up`, `walk_forward`, `walk_backward`,
   `sidestep_left`, `sidestep_right`, `turn_left`, and `turn_right`) and its
   checkpoint/bridge validation interface as the first concrete R4–R5 target,
   or document why it cannot serve. Its smoke and integration plumbing are not
   a matched continual-learning result. Full-budget comparisons, adaptation/
   retention metrics, control-rate budgets, fault/safety panels, and guarded
   hardware evidence remain required.
5. **Scale deliberately.** Increase horizon, task diversity, observation size,
   action complexity, and number of learned predictions while measuring how
   performance and resource use scale.
6. **Keep the search open.** Regularly compare the current bottleneck with ideas
   inside and outside the Alberta Plan. Prefer causal experiments and portable
   mechanisms over allegiance to an existing component.

The [IPMNIST theory](ipmnist-theory.md) owns its current mechanism hypotheses.
The [negative-results ledger](../evidence/negative-results.md) prevents closed
ideas from being silently recycled. The [status map](../status.md) records what
is actually implemented and evidenced; neither this roadmap nor a future plan
can upgrade that status by assertion.

## Naming and provenance

ASI is the current project and repository identity. The Python import namespace
`alberta_framework`, distribution name `alberta-framework`, `alberta-*` console
commands, Alberta-specific Step modules, and historical `alberta.*` artifact
schemas remain stable compatibility or provenance names. A future software
namespace migration must be explicit, versioned, and tested across robot
consumers, packaging, CLIs, validators, and stored evidence. Immutable output
records are never rewritten to apply the new brand.

Genuinely new ASI-native protocol and artifact families should use versioned
`asi.*` schema IDs and may add `asi-*` CLI aliases. A new version that extends a
frozen Alberta-family schema retains its `alberta.*` lineage and compatible
loader. New aliases are additive; historical IDs are never renamed.
