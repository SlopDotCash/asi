# ASI reference-agent protocol

Status: **Proposed**

This ADR selects a shared reference-life protocol with implementation-specific
adapters as ASI's architectural direction. It does not select a current agent
configuration, populate `reference-dev`, establish a whole-life result, or make
a robotics-readiness or state-of-the-art claim.

## Context

ASI needs one runnable agent identity that can learn throughout an operational
life and move from controlled continual-learning environments toward robotics.
The repository currently has two relevant but incompatible composition paths:

- `alberta_framework.core.prototype_agent.PrototypeAgent` has a strict,
  functional discrete-action transition contract, decision ownership, and a
  complete agent-state checkpoint wrapper. It has no canonical environment
  runner or whole-life checkpoint, and its external closed-loop dispatch path
  has not closed the OaK/STOMP extended-action ownership edge.
- The sibling robot track has continuous-action controllers, Gymnasium/MuJoCo
  loops, checkpoint manifests, and safety/bridge plumbing. It does not consume
  `PrototypeAgent`, and its ordinary parameter snapshots are not an exact
  checkpoint of every learner, runner, environment, and in-flight state.

Making either implementation the universal interface would force the other
action domain into the wrong abstraction. Maintaining unrelated lifecycle
contracts would prevent one exact-resume gate and one whole-life scorecard from
governing both paths.

## Decision

ASI defines one semantic reference-agent protocol and will implement adapters
for concrete agents and environments. The protocol owns lifecycle, dispatch,
transition, checkpoint, and measurement semantics; it does not prescribe a
particular learner, observation encoding, or action tensor shape.

A conforming implementation requires one authoritative life runner. Agent,
environment, dispatch/safety, and metric adapters expose all mutable state to
that runner. They may use functional or object-oriented internals, but no
protocol implementation may hide mutable state that affects a future
observation, action, update, metric, or replay decision.

The selected direction is therefore the shared-protocol path. The ADR remains
Proposed until the conformance and exact-resume gates below pass. A separate
record must name the first `reference-dev` configuration after those gates; this
document cannot do so by assertion.

### Implemented `preview1` L0 transaction slice

The [host transaction module](../../alberta_framework/reference_agent.py)
and its [retained contract tests](../../tests/test_reference_agent_protocol.py)
implement the first acceptance slice:

- `asi.reference_agent.preview1`, `asi.reference_agent_manifest.preview1`, and
  `asi.reference_transaction_state.preview1` identify a versioned preview, not a
  frozen v1 contract;
- canonical configuration and manifest identities plus fixed, exact-dtype
  observation/action spaces;
- deeply immutable typed payloads and manifest-bound, lifecycle-scoped decisions;
- separate independent authorization, learner settlement, executor receipt, and
  receipt-bound outcome records;
- explicit bootstrap and post-reset observation IDs at episode boundaries;
- a process-local, single-writer live ledger whose lock-protected current-object
  identity compare-and-swap rejects stale snapshots, phase/chain forgery, and
  repeated `init()` within that ledger object;
- rejection semantics that leave the event unconsumed, retain its transaction in
  `halted`, arm no next decision, and require recovery; and
- a bounded uint64 decision index whose final accepted event is consumed before
  the ledger clears the transaction and enters `exhausted` without wrapping.

This is an L0 host transaction contract, not an agent, environment, safety, or
metrics adapter. It does not implement the canonical life configuration,
aggregate life state, authoritative runner, whole-life checkpoint, exact resume,
or `reference-dev`. The live ledger deliberately refuses pickling and supplies
no durable replay protection, restore path, wire decoder, checkpoint format, or
cross-process exact-resume claim. The preview accepts only finite scalar reward
and discount values and defines no extension sidecars.

For an authorized action replacement, the ledger checks that the manifest
declares rebinding and that the adapter asserts `rebinding_applied=True`. It
cannot prove that a concrete adapter updated every credit owner and action-bound
cache; that remains an adapter conformance gate. Its `DispatchReceipt` is a typed
executor acknowledgement, not proof of physical dispatch.

### Implemented Prototype L0 agent transaction bridge

The development-only
[Prototype reference adapter](../../alberta_framework/prototype_reference_adapter.py)
and its [retained tests](../../tests/test_prototype_reference_adapter.py) implement a
manifest-bound, primitive-only, exact-dispatch bridge from the preview records to a
sidecar-free `PrototypeAgent` configuration on continuing transactions. Its immutable state
envelope binds the underlying agent state to the manifest and configuration and owns the host
lifecycle, decision index, and observation identity. It stages functional updates, preserves
the supplied state on rejection, and advances only an exact receipt-bound transaction.

This is an L0 agent transaction bridge, not an environment or executor adapter. It neither proves
that the acknowledged action reached an environment nor supplies an authoritative runner,
aggregate life state, whole-life checkpoint, durable replay/restore, or exact resume. Options,
action replacement/rebinding, and episode-boundary semantics are deliberately unsupported; the
bridge is not closed-loop conformance, `reference-dev`, or scientific evidence.

## Protocol records

Exact Python names and packaging may be refined during implementation, but the
following semantic records are required and versioned.

### Life configuration

The canonical, JSON-compatible life configuration binds:

- protocol and configuration schema versions;
- agent adapter type and complete agent configuration;
- environment adapter type and complete environment configuration;
- observation and action codec versions;
- dispatch and independent safety-authority configuration;
- agent, environment, and adapter seed schedules;
- task/regime schedule and all exposed boundary signals;
- canonical `max_accepted_events`, bounded by the smallest non-wrapping counter
  capacity declared by every selected agent, environment, dispatch, metrics, and
  runner component;
- checkpoint cadence and transaction-boundary policy;
- metric definitions, resource ceilings, and latency deadlines; and
- source, dependency, and runtime identity required by the declared lane.

Defaults that affect behavior are materialized before hashing. Unknown fields,
noncanonical encodings, incompatible adapter combinations, and event horizons
above any selected component's capacity fail closed before initialization.

### Life state

The life runner is the sole owner of one aggregate state containing:

| State | Required owner and contents |
|---|---|
| Agent | Agent adapter; all parameters, optimizer/traces, learned state, caches, counters, and RNGs |
| Environment | Environment adapter; complete simulation state, reset state, RNGs, and task/regime position |
| Dispatch | Dispatch adapter; proposed and authorized action lineage, safety/watchdog state, and any pending receipt |
| Runner | Life runner; lifecycle ID, generation/event counters, phase, checkpoint generation, and seed derivation state |
| Metrics | Metric adapter; online accumulators and enough state to continue the scorecard exactly |
| Provenance | Canonical configuration digest plus declared source, dependency, environment, and runtime identities |

Adapter objects may retain immutable compiled functions or static configuration.
All mutable values belong in the aggregate state. There is no second informal
owner and no parameter-only checkpoint exception.

### Decision, authorization, and outcome

Every decision has a manifest identity, lifecycle-scoped non-reusable decision
ID and index, the observation identity from which it was produced, an immutable
typed proposed action carrying its codec semantic ID, and an armed/disarmed
status.

Before an action reaches an environment, an independent dispatch/safety adapter
returns an authorization record bound to that decision ID. It contains the
authorized action, whether it differs from the proposal, the authority and
policy version that made the decision, and an authorization ID. The agent
adapter then returns a distinct settlement record binding learner credit to the
authorized action. A configuration that cannot safely settle a changed action
must halt rather than learn from a counterfactual proposal. After the host
reports execution, a distinct receipt binds the settlement, effective action,
executor, and receipt ID.

An outcome is bound to the exact dispatch receipt and therefore to its
lifecycle and decision. In `preview1` it contains one finite scalar reward and
one finite scalar continuation discount, termination and truncation flags, the
final observation and identity used for bootstrapping, and the potentially
distinct post-reset observation and identity used for the next decision.
Additional learning-signal sidecars require a future protocol version.

## Lifecycle

One life follows this order:

1. Canonicalize and validate the complete life configuration.
2. Create a fresh lifecycle ID and initialize every state owner exactly once.
3. Initialize or restore the environment and obtain the first observation.
4. Start the agent once and arm one decision.
5. Authorize and settle that decision, then record the dispatch receipt.
6. Dispatch the settled action exactly once.
7. Record the matching outcome exactly once.
8. Apply one atomic agent update, advance metrics and runner counters, and arm
   the next decision.
9. At a declared quiescent boundary, optionally write a whole-life checkpoint.
10. Continue across task and episode boundaries without reinitializing learned
    agent state. Only explicitly declared episodic state may reset.

The planned first whole-life runner checkpoints only after an outcome has been
consumed and before the next action has been dispatched. In-flight physical
actions are outside its planned exact-resume gate until a later protocol defines
durable, idempotent dispatch and acknowledgement semantics.

In the preview ledger, rejecting an outcome leaves its event unconsumed, retains
the transaction in `halted`, and requires an external recovery protocol that is
not yet defined. Accepting the event at the maximum uint64 decision index consumes
that final event and moves the ledger to `exhausted`; it never wraps the counter.

## Dispatch and transition invariants

Conforming implementations must enforce all of the following:

- A decision is used at most once and only within its lifecycle.
- The observation, decision ID, authorized action, receipt, and outcome form one
  ownership chain. Stale, replayed, cross-life, out-of-order, or mismatched
  records fail closed without a partial learning update.
- Learning credits the action actually dispatched. Safety clipping, action
  substitution, skill selection, residual blending, or actuator projection may
  not be hidden behind the proposed action.
- Extended or hierarchical action identity remains available while the
  environment receives its primitive action. Rebinding a primitive must update
  the correct base or intra-option owner without erasing the extended owner.
- The final and next-decision observation identities are distinct at an
  autoreset boundary, identical on a continuing transition, and the next
  observation is absent until an explicit arm after a non-autoreset boundary.
- Termination, truncation, and discount semantics are explicit and validated.
- One accepted environment event causes one atomic learner transaction. A
  rejected event remains unconsumed and halts the runner for recovery. Concrete
  adapter conformance must prove that every agent state and RNG remains
  synchronized with that event.
- Independent safety authority may veto or replace an action and may halt the
  runner. The learner cannot weaken, train through, or relabel that authority.
- The final uint64-indexed event is consumed before counter exhaustion disarms
  the life; the state becomes `exhausted` and the counter never wraps.

## Exact-resume gate

An agent checkpoint alone is not a life checkpoint. A conforming whole-life
checkpoint binds the canonical configuration and every state in the ownership
table, including environment and runner RNGs, metric accumulators, current
decision lineage, and the quiescent transaction phase.

For a deterministic simulation fixture, exact resume must pass this comparison:

1. Run one life uninterrupted for `N + M` events.
2. Run the same life for `N` events, checkpoint at the declared quiescent
   boundary, reconstruct new adapter and runner objects, restore, and run `M`
   more events.
3. Require exact equality of the full aggregate state, every subsequent
   decision and settled action, environment observations and outcomes, metric
   state, counters, and RNG state. Canonical configuration and provenance
   digests must also match.

Floating-point tolerances do not satisfy this gate unless the protocol for that
backend explicitly proves why bitwise equality is unavailable and freezes a
stronger backend-specific equivalence rule before testing. Serialized checkpoint
directory bytes need not match; restored semantic state must.

A physical environment generally cannot be restored exactly. Hardware restart
must use a separately named reconciled-resume mode that re-establishes sensor,
safety, and actuator state under independent authority. Reconciled resume is an
important application gate, but it does not count as passing deterministic
exact resume.

## Prototype adapter

The implemented primitive-only L0 bridge maps the agent transaction subset as follows:

- `PrototypeAgent.init(..., lifecycle_id=...)` initializes agent state.
- `start` arms the initial decision; the adapter derives the primitive action
  and all host decision identity from its manifest-bound state envelope.
- The adapter constructs the explicit `PrototypeTransition` and calls
  `update_transition`; the legacy `update` path is not conforming.
- Only exact primitive dispatch is accepted; veto and replacement are rejected
  without mutating the supplied functional state.

The current bridge does not support the host ledger's full uint64 horizon:
Prototype decision/cache validity may disarm at its int32 telemetry or observation
capacity before the host reaches its final decision index. The retained fail-stop
test establishes atomic rejection only. A future runner must bind the declared
Prototype capacity into `max_accepted_events` and reject an oversized life during
construction rather than discover the mismatch mid-run.

The remaining conforming Prototype adapter must nest the existing v3 Prototype
checkpoint as the agent portion of the future whole-life bundle, not treat it as
the entire bundle. A public, decision-bound settlement path must either preserve
or atomically rebind OaK/STOMP base-versus-option credit when authorization
changes the primitive action, and it must update every action-bound cache. Direct
runner mutation of Prototype internals is forbidden.

The adapter first targets the pure continuing environments in
`alberta_framework.streams.closed_loop`. Before it may enter Forager or be called
conforming, a retained test must exercise an active option through real runner
dispatch and prove that the primitive is credited to the correct intra-option
owner while extended-action identity survives.

## Robot adapter

The robot adapter will retain the sibling track's continuous action space,
environment/task interfaces, and independent bridge safety authority. It will
map the controller's start/observe behavior into the shared decision,
authorization, outcome, and boundary records rather than coercing continuous
actuator vectors into Prototype primitives.

To conform, the adapter must externalize the controller's complete mutable
state: learned parameters, optimizer and eligibility traces, normalizer and
feature state, last observation and actual action, task routing, counters, RNGs,
dispatch/safety state, and any curriculum or environment state affecting future
behavior. A parameter-only `state_dict` is insufficient. Clipping, residual
composition, gait/skill priors, and bridge substitutions must be represented as
authorization/settlement operations so learning is bound to the delivered
action.

The first robot conformance target is deterministic embodied simulation. The
existing ASIMOV-1 task sequence and readiness documents remain the application
ladder; this ADR does not duplicate them or turn existing smoke artifacts into
performance evidence.

## Non-goals

This ADR does not:

- choose a learner, `PrototypeAgent` configuration, robot controller, task
  schedule, benchmark winner, or `reference-dev` contents;
- make discrete and continuous action policies share one learner implementation;
- require every retained ASI mechanism to coexist in one configuration;
- establish learning benefit, state of the art, scientific promotion, Alberta
  Plan completion, robotics readiness, or guarded hardware authorization;
- replace benchmark-specific frozen protocols or evidence validators;
- rename the `alberta_framework` compatibility namespace or historical schemas;
- claim exact restoration of the physical world; or
- authorize edits to immutable or append-only `outputs/` artifacts.

## Acceptance sequence

The proposal advances only in this order:

1. **Host transaction preview — implemented at L0.** The versioned-but-unfrozen
   `preview1` manifest and transaction-state schemas, immutable typed payloads,
   distinct authorization, settlement, receipt, and outcome records, explicit
   bootstrap/reset observation IDs, process-local current-object ledger, and
   retained validation/ownership tests implement the host transaction slice.
   This is structural and nonpromoting; it is not a frozen compatibility version.
2. **Whole-life conformance core — open.** Define the canonical life
   configuration and aggregate life state; implement agent, environment,
   dispatch/safety, and metric adapter contracts plus the authoritative runner;
   define extension/sidecar policy, wire decoding, halted-event recovery, and the
   whole-life checkpoint schema; and add a mock whole-life conformance fixture.
   Every mutable owner and RNG must be explicit, and the runner must provide
   durable transaction-state ownership plus restart replay/restore semantics and
   reject `max_accepted_events` above any selected component's counter capacity.
   Step 1 does not satisfy this gate.
3. **Prototype agent bridge — partial; closed-loop adapter open.** The retained
   primitive-only, exact-dispatch, continuing-task tests implement the L0 agent
   transaction bridge. Still run one canonical but explicitly development-only
   configuration through a real environment/executor path on both retained
   continuing micro-MDPs. Exercise active-option dispatch, action replacement and
   rebinding, veto, boundary semantics, stale-event rejection, and counter
   disarming. The implemented bridge does not satisfy this closed-loop gate.
4. **Prototype exact resume.** Pass the uninterrupted-versus-restored gate,
   including environment, runner, metrics, decision lineage, and every RNG.
5. **Forager bridge.** Implement the ordinary-observation policy adapter, retain
   extended-action ownership through the host runner, and declare measured
   compute, memory, and latency costs. This remains development work unless a
   separate frozen protocol says otherwise.
6. **Robot simulation adapter.** Externalize complete controller state, bind
   actual continuous dispatch after safety/residual transforms, pass deterministic
   exact resume where the simulator supports it, and pass fault/latency checks.
7. **Whole-life regression panel.** Provide one CI-cheap command covering
   ownership, uninterrupted operation, checkpoint recovery, recurrence,
   numerical stability, and resource ceilings. Benchmark execution remains
   outside pytest.
8. **Separate `reference-dev` selection.** Only after steps 1–7 pass may a new
   decision name a concrete agent adapter, configuration, runner command, and
   rollback policy as `reference-dev`. That channel is permanently
   nonpromoting.

Until the final step is recorded, ASI has a selected proposed architecture but
no canonical reference agent or `reference-dev` baseline.
