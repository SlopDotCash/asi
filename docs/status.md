# ASI research status

This document is the compact, human-readable status map for ASI. It separates
available mechanisms, stored scientific outcomes, whole-agent capabilities,
and the evidence still required for a continual-learning application.

**Verdict: in progress.** No whole-agent L3 protocol or result has been
completed, and there is no state-of-the-art application or robotics-readiness
result. The package also exposes research surfaces related to all twelve steps
of the Alberta Plan, but no Step satisfies this repository's full completion
rule. The Plan is an inspiration and crosswalk, not ASI's binding roadmap.

This page deliberately does not record a dated live evidence-registry result,
module count, test count, campaign ranking, or session chronology. Those facts
change. Run the relevant command or read the primary artifact instead.

## Sources of authority

Use each record only for the question it owns:

- The [ASI research roadmap](research/asi-roadmap.md) owns the mission,
  hillclimb loop, application ladder, and whole-life scorecard.
- This page records the stable capability gaps, evidence levels, Alberta Step
  1–12 crosswalk, and completion gates.
- [Evidence methodology](evidence/methodology.md) owns promotion rules,
  property-level evidence, artifact contracts, and scientific limitations.
- The [negative-results ledger](evidence/negative-results.md) owns rejected,
  bounded, consumed, and closed development results.
- [`evidence_manifest.py`](../alberta_framework/evaluation/evidence_manifest.py)
  owns the live five-claim registry, its exact registered source sets, and exit-code
  semantics.
- Versioned JSON artifacts own frozen scientific outcomes. Validators, not
  narrative summaries, decide whether those artifacts match current sources.
- IPMNIST summaries and reports under
  [`outputs/ipmnist_screening/`](../outputs/ipmnist_screening/) own that moving
  development campaign's measurements.

Implementation, tests, smoke runs, development experiments, and scientific
evidence are different kinds of progress. A public class or passing test proves
that a mechanism exists; it does not by itself prove benefit, retention,
resource parity, or integration.

## Evidence levels

- **L0 — mechanism:** API, shape, finite-value, serialization, ownership, or
  local-update contracts pass.
- **L1 — learning:** a component learns in a controlled toy problem.
- **L2 — comparison:** a preregistered, multi-seed, matched-resource benchmark
  establishes the frozen claim against strong baselines.
- **L3 — integration:** one uninterrupted agent life demonstrates the required
  interactions, retention, recovery, bounded resources, and causal ablations.

For this repository, a Step is complete only when its defining outcome reaches
L2 and its required links to earlier Steps are exercised at L3. Missing
promoted evidence must fail closed; it must not be treated as a skipped test or
inferred from adjacent results.

All twelve public Step modules contain mechanism or smoke surfaces. Only the
Step 1 and Step 2 smoke probes are console scripts. Smoke execution is L0 and
is structurally nonpromoting.

## ASI capability map

This map is primary for the end-to-end application. The Alberta Plan crosswalk
below supplies a second view over the inherited research program.

| Capability | Current surface | Open application gate |
|---|---|---|
| Online adaptation and plasticity | Online learners, adaptive optimizers, normalization, UPGD/CBP mechanisms, and development campaigns | No completed multi-domain comparison establishes sustained plasticity without hidden retention or resource regressions |
| Learned state and representation | Feature construction, utility, routing, lifecycle, temporal-context, and recurrent-state mechanisms | No autonomous representation lifecycle has a whole-life downstream control result |
| Prediction, memory, and world modeling | Horde/GVF learners, working and experiential memory, one-step/latent/recurrent models, and dreaming | No reference agent shows that these components causally improve retained control under one owner and budget |
| Continual control and planning | SARSA, actor-critic, average-reward/off-policy control, options, STOMP/OaK, and bounded planning | No promoted matched-resource continual-control result closes adaptation, retention, and planning-benefit gates |
| Agent composition | [`preview1` L0 reference transaction contract](../alberta_framework/reference_agent.py), versioned but not frozen v1, with immutable typed payloads, separate authorization/settlement/receipt/outcome records, explicit bootstrap/reset observation IDs, and a process-local ledger whose lock/current-object CAS rejects stale snapshots and repeated initialization; rejection preserves the unconsumed event in `HALTED`, while an accepted final uint64 event enters `EXHAUSTED`; covered by [retained contract tests](../tests/test_reference_agent_protocol.py). A [manifest-bound primitive-only Prototype bridge](../alberta_framework/prototype_reference_adapter.py) and its [retained tests](../tests/test_prototype_reference_adapter.py) implement exact dispatch and continuing-task outcome updates at L0, alongside explicit Prototype transition ownership, agent-only checkpoints, and Step/pipeline kernels | The Prototype bridge is not an environment/executor adapter, runner, whole-life checkpoint/exact resume, options/rebinding/boundary conformance, `reference-dev`, or evidence. No closed-loop Prototype or robot adapter, replacement-settlement proof, durable aggregate life state or authoritative runner, recovery protocol, wire decoder/sidecar policy, durable replay/restore, exact-resume result, selected `reference-dev`, or uninterrupted L3 life exists; reward/discount are scalar-only preview fields, the live ledger is non-picklable, the receipt is only an executor acknowledgement, and robot and Forager paths do not consume `PrototypeAgent`, whose closed-loop extended-action dispatch remains unresolved |
| Multi-agent learning | Recurring coadaptation surfaces, partner-policy fusion, and an IA intervention protocol | A frozen historical coadaptation outcome is not causal amplification; the frozen historical IA outcome is a valid rejection, while live validity must be checked separately |
| Scale and operations | JAX scan/PyTree kernels, fixed-shape state in many paths, artifact tooling, and robot-compatible imports | Long-horizon compute, memory, latency, numerical stability, checkpoint recovery, and workload scaling are not jointly established |
| Robotics and real work | The sibling robot track consumes a continual-RL subset and defines an ASIMOV-1 task sequence plus checkpoint, bridge, and validation plumbing | No ASI reference-life binding or matched full-budget adaptation/retention result closes the control-rate, fault/safety, and guarded-hardware gates |

Implementation presence in the middle column is not a benefit claim. Each open
gate needs its own matched protocol, and the integrated target needs the
whole-life dimensions to pass together.

The current Prototype bridge's fail-stop counter test does not establish the host
ledger's full uint64 horizon: Prototype may disarm at its int32 capacity first. A
future canonical life configuration must bind `max_accepted_events` at or below
every selected adapter and environment counter capacity, and runner construction must
reject an oversized life rather than halt unexpectedly during execution.

## Registered evidence

The five-claim registry is intentionally narrow. Even an `accepted` overall
registry result would mean only that every listed narrow gate passed. It would
not certify the package, complete a Step, or establish Alberta Plan completion.

The immutable stored artifacts record these frozen outcomes:

| Claim ID | Frozen scope | Artifact | Frozen outcome |
|---|---|---|---|
| `recurring_pair_features` | Retention and active-bank allocation of supplied pair-product features in a Gaussian/L2 recurring probe | [`evidence.v1.json`](../outputs/recurring_feature/evidence.v1.json) | Accepted, narrow L2 |
| `scale_robust_pair_features` | Scale-robust selection and structural retention of relevant pair products in a visibly cued regression gauntlet | [`evidence.v2.json`](../outputs/scale_robust_feature/evidence.v2.json) | Accepted, narrow L2 |
| `ftl_world_model_decision_fidelity` | Low menu regret for a sparse online transition model in one deterministic A–B–A decision-fidelity protocol | [`evidence.v1.json`](../outputs/ftl_decision/evidence.v1.json) | Accepted historical L2 |
| `recurring_multiagent_coadaptation` | Fixed-memory coadaptation and retention in one visibly cued two-agent A–B–A sanity benchmark | [`evidence.json`](../outputs/continual_multiagent/evidence.json) | Accepted, narrow L2 |
| `continual_intelligence_amplification` | Causal recommendation-channel uplift in one deterministic hidden-phase micro-MDP | [`evidence.json`](../outputs/continual_ia/evidence.json) | Valid rejection at the frozen L2 gate |

The IA artifact remains a scientifically useful rejection. Its reward uplift
and augmentation-control checks do not override the failed frozen
action-changing-intervention gate. The threshold must not be lowered after the
result.

### Live validation semantics

Run the registry from a repository checkout:

```bash
.venv/bin/alberta-evidence-status
```

The operational statuses are:

- `accepted`: a valid promoting L2/L3 artifact passed its frozen gate;
- `valid-rejection`: a valid promoting artifact failed at least one gate;
- `not-run`: a required artifact is absent;
- `invalid`: schema, provenance, source, integrity, reconstruction, or
  contract validation failed; and
- `verified-nonpromoting`: a valid unit or smoke record that cannot promote.

The command exits `0` only when all registered promoting claims are accepted,
`1` for a missing artifact, valid rejection, or nonpromoting-only result, and
`2` when any claim is invalid.

Validation hashes each claim's exact registered source paths. Those manually
enumerated sets are load-bearing, but they are not complete recursive import
closures. A clean worktree is not required. Changes outside a claim's registered
source set are recorded as operational provenance but are not independently
disqualifying. A change to a registered source is a source mismatch and normally
makes the persisted artifact invalid until a newly authorized frozen protocol
writes a new artifact that passes its strict validator.

Two historical compatibility routes are deliberately narrower than a general
source-drift waiver:

- The FTL route is eligible only for its exact v1 contract and prescribed
  builder-only drift. It reconstructs the historical acceptance and checks an
  already-consumed-seed replay against unchanged invariant sources. That replay
  is compatibility evidence, never fresh promotion evidence.
- The IA route is eligible only for its exact v1 historical-rejection chain.
  Its archived source snapshot and consumed-seed replay can preserve the
  original rejection classification, never turn it into current-source
  acceptance.

Any additional source drift, artifact mutation, schema mismatch, or failed
reconstruction remains invalid. Pinned artifacts must not be edited, repaired,
or overwritten. New work writes a new path and, when the contract requires it,
a new schema version with untouched preregistered seeds.

Normal package installations do not include `outputs/`, so registry execution
from an installed wheel or sdist normally reports artifacts as missing. Use a
checkout to inspect the stored evidence chain.

## Alberta Plan crosswalk: Steps 1–12

### Step 1 — Nonstationary prediction

**Required outcome.** Track nonstationary affine prediction online with
normalization, relevance-sensitive step sizes, robustness, and bounded work.

**Available surface.** The package includes drifting Step 1 streams, LMS,
IDBD, Autostep and comparison optimizers, online normalizers, update bounding,
and a deterministic smoke kernel.

**Open gate.** No frozen matched multi-seed Step 1 comparison and no L3 link to
the later learned-state/control agent satisfy the completion rule.

### Step 2 — Feature construction and replacement

**Required outcome.** Generate useful nonlinear features, estimate future
utility, and replace features within a fixed budget while retaining recurring
critical structure.

**Frozen historical evidence.** Pair construction, bounded feature banks,
utility and lifecycle mechanisms exist. The recurring-pair and
scale-robust-pair artifacts record two historically accepted but deliberately
narrow L2 outcomes. They certify the current tree only when live validation is
valid.

**Open gate.** The stored protocols begin from constrained pair-product
families and do not establish autonomous question discovery, general
selective retention, or an uninterrupted downstream control benefit. Separate
development failures and bounds remain nonpromoting and are recorded in the
[negative-results ledger](evidence/negative-results.md).

### Step 3 — Many continuing predictions

**Required outcome.** Learn many continuing, potentially off-policy GVFs with
history and feature finding.

**Available surface.** Horde, mixed and independent demons, TD/GTD variants,
traces, normalization, and a causal Step 2 feature handoff have mechanism and
small learning coverage.

**Open gate.** There is no promoted matched comparison showing useful learned
questions/features across recurrence, nor an L3 connection to the final
control loop.

### Step 4 — Control I

**Required outcome.** Progress from bandit and contextual control to
sequential actor-critic or action-value control with learned features.

**Available surface.** SARSA, discrete and continuous actor-critic,
average-reward and off-policy variants, bounded updates, and small online
control diagnostics are implemented.

**Open gate.** No frozen resource-matched continual-control result establishes
retention and recovery while learned state changes, and no complete L3 link to
prediction and feature lifecycles exists.

### Step 5 — Continuing prediction II

**Required outcome.** Learn differential average-reward predictions together
with conventional value and expected-duration predictions needed by options.

**Available surface.** Differential TD/GTD/Horde learners and bounded
return/duration model components have mechanism and toy-learning coverage.

**Open gate.** There is no promoted comparison or integrated option-control
result demonstrating calibrated multi-timescale predictions.

### Step 6 — Continuing control II

**Required outcome.** Demonstrate reproducible continuing average-reward
control across the intended suite, including RiverSwim, access control,
Jellybean, GARNET, and continuing conversions.

**Available surface.** Differential SARSA and actor-critic mechanisms,
closed-loop micro-environments, benchmark adapters, and Forager campaign
tooling are present.

**Open gate.** The named suite has no single promoted result, and no completed
paper-length, matched-resource Alberta-versus-baseline Forager comparison
exists.

### Step 7 — Incremental planning

**Required outcome.** Validate bounded incremental average-reward planning,
then planning with function approximation and adaptive features.

**Available surface.** World-model updates, bounded dreaming, and option
search-control are available as mechanism surfaces.

**Open gate.** Model support and uncertainty are not yet externally calibrated,
and there is no frozen matched-budget result showing reliable planning benefit
under changing dynamics and representations.

### Step 8 — Learned world and representation loop

**Required outcome.** Close the perception → world model → feature ranking →
feature replacement → model-feedback loop.

**Frozen historical evidence.** One-step, action-conditioned, sparse
fixed-shape, ensemble, recurrent, and latent model components exist. The FTL
artifact records a narrow historical L2 decision-fidelity acceptance for one
fixed-shape model and protocol; live validity is separate.

**Open gate.** That historically accepted FTL scope does not establish a
calibrated general world model, learned-target quality, retained planning
benefit, partner modeling, or the complete feedback loop under one owner and
lifetime.

### Step 9 — Exploration and search control

**Required outcome.** Improve exploration and planning order under matched
real-transition, model-query, and backup budgets without exploiting noisy or
irrelevant novelty.

**Available surface.** Surprise, priority, utility, guarded dreaming, and
bounded search-control mechanisms exist.

**Open gate.** Development diagnostics do not provide a preregistered
matched-budget benefit, calibrated causal score production, or an integrated
held-out exploration result.

### Step 10 — Subtasks, options, models, and planning

**Required outcome.** Discover reward-respecting subtasks, learn options and
option models, and consume those models in planning.

**Available surface.** STOMP supports specified subtasks, temporally extended
actions, option learning, outcome models, and bounded option-model backups.

**Open gate.** The default path does not autonomously discover and repeatedly
maintain useful subtasks under one continual owner, and no held-out matched
benefit result closes the loop.

### Step 11 — Causal utility and OaK

**Required outcome.** Track causal utility, safely replace features, subtasks,
options, and models, and compose behaviours through an option keyboard.

**Available surface.** OaK utility tracking, option curation, keyboard
mechanics, and feature-lifecycle transactions are implemented.

**Open gate.** Autonomous go/no-go authority, repeated safe replacement,
automatic keyboard consumption, and causal outcome evidence remain absent.
Mechanism-level lifecycle receipts do not grant deployment or promotion.

### Step 12 — Intelligence amplification

**Required outcome.** Measurably increase another learning agent's capability
through a closed, continuing interaction loop.

**Frozen historical evidence.** Prediction augmentation, recommendation
protocols, partner-policy fusion, generic world models, and two-agent streams
exist. The recurring-multiagent artifact records one narrow historically
accepted L2 coadaptation outcome. The frozen IA artifact records a historical
valid rejection. Neither statement substitutes for live validation.

**Open gate.** Coadaptation is not the same as causal amplification. The IA
intervention gate remains failed, and there is no L3 partner-benefit result
under changing reliability, communication cost, retained skills, and bounded
resources.

No Step currently satisfies the repository completion rule. Even completing
all twelve Step gates would be evidence about the Alberta-derived program, not
automatic proof of ASI's broader application target.

## Active development campaigns

### IPMNIST development screening and development confirmation

IPMNIST is the current measured optimization/plasticity subsystem campaign,
not ASI's top-level reference-life hillclimb. It is
**development-grade and permanently nonpromoting**. Development-screening,
development-confirmation, and publication-run records may support descriptive
development conclusions; they do not become scientific evidence through
replication, more seeds, or better performance.

Use the mutable index to identify the current record rather than copying
rankings or means here. Output documents are append-only chronological records;
their historical superlatives and run-status language may have been superseded.

- [current campaign index](research/ipmnist-campaign-index.md),
- [theory and forward hypotheses](research/ipmnist-theory.md),
- `outputs/ipmnist_screening/summary_*.json` for stored measurement records,
- [chronological campaign runbook](../outputs/ipmnist_screening/RUNBOOK.md),
- [historical accumulated report](../outputs/ipmnist_screening/FINAL_REPORT.md),
- [historical artifact and reproducibility audit](../outputs/ipmnist_screening/AUDIT.md), and
- [pre-RLS publication-run record](../outputs/ipmnist_screening/publication_runs/RESULTS.md).

Remeasure the intended baseline under the current development protocol before
any A/B comparison. Do not infer registry acceptance, a promoted
state-of-the-art claim, ASI progress at the whole-agent level, robotics
readiness, or an Alberta Step completion from this lane. Seeds used for
development or selection cannot later serve as untouched promotion seeds.
