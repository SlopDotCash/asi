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
  complete agent-state checkpoint wrapper. Development-only primitive
  SwitchingTwoState and RiverSwim runners plus quiescent whole-life checkpoints
  now consume it and pass same-runtime exact-continuation gates. Broader
  environment generality and the OaK/STOMP extended-action ownership edge
  remain open.
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

The selected direction is therefore the shared-protocol path. It remains
Proposed while broader conformance is open. The current-schema quiescent
checkpoint/exact-resume gate now passes for the primitive Prototype +
SwitchingTwoState and Prototype + RiverSwim constructions; those narrow L0
results neither freeze the protocol nor establish RiverSwim learning or
performance benefit nor select `reference-dev`. A separate record must name the
first `reference-dev` configuration after the remaining gates; this document
cannot do so by assertion.

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
- separate independent authorization, learner settlement, pre-execution command,
  post-execution executor receipt, and receipt-bound outcome records;
- explicit bootstrap and post-reset observation IDs at episode boundaries;
- a process-local, single-writer live ledger whose lock-protected current-object
  identity compare-and-swap rejects stale snapshots, phase/chain forgery, and
  repeated `init()` within that ledger object;
- rejection semantics that leave the event unconsumed, retain its transaction in
  `halted`, arm no next decision, and require recovery; and
- a bounded uint64 decision index whose final accepted event is consumed before
  the ledger clears the transaction and enters `exhausted` without wrapping.

This is an L0 host transaction contract, not an agent, environment, safety, or
metrics adapter. The host transaction module alone does not implement the
canonical life configuration, aggregate life state, authoritative runner,
whole-life checkpoint, exact resume, or `reference-dev`. The live ledger
deliberately refuses pickling and supplies
no durable replay protection, restore path, wire decoder, checkpoint format, or
cross-process exact-resume claim. The preview accepts only finite scalar reward
and discount values and defines no extension sidecars.

For an authorized action replacement, the ledger checks that the manifest
declares rebinding and that the adapter asserts `rebinding_applied=True`. It
cannot prove that a concrete adapter updated every credit owner and action-bound
cache; that remains an adapter conformance gate. Its `DispatchReceipt` is a typed
executor acknowledgement, not proof of physical dispatch. The host now issues a
distinct `DispatchCommand`, bound to the settlement and the declared executor ID
and epoch, before execution. Only after execution may it record a receipt carrying
the independently encoded applied action. An applied-action mismatch retains the
receipt, halts before an outcome can be recorded, and cannot reach learning.

### Implemented Prototype L0 agent transaction bridge

The development-only
[Prototype reference adapter](../../alberta_framework/prototype_reference_adapter.py)
and its [retained tests](../../tests/test_prototype_reference_adapter.py) implement a
manifest-bound, primitive-only, exact-dispatch bridge from the preview records to a
sidecar-free `PrototypeAgent` configuration on continuing transactions. Its immutable state
envelope binds the underlying agent state to the manifest and configuration and owns the host
lifecycle, decision index, and observation identity. It stages functional updates, preserves
the supplied state on rejection, and advances only an exact receipt-bound transaction.

This is an L0 agent transaction bridge, not by itself an environment or executor adapter. It
neither proves that the acknowledged action reached an environment nor supplies a whole-life
checkpoint, durable replay/restore, or exact resume. Options, action replacement/rebinding, and
episode-boundary semantics are deliberately unsupported; the bridge is not `reference-dev` or
scientific evidence.

### Implemented development-only aggregate L0 life slices

The [aggregate reference-life module](../../alberta_framework/reference_life.py)
and its [base retained tests](../../tests/test_reference_life.py) plus
[RiverSwim tests](../../tests/test_reference_life_riverswim.py) implement the
authoritative process-local runner through the primitive `PrototypeAgent` +
`SwitchingTwoStateMDP` and `PrototypeAgent` + `RiverSwimMDP` paths:

- a canonical immutable life configuration binds the complete agent, environment,
  declared-static exact-dispatch, and metric configurations and digests, plus a
  `max_accepted_events` bounded by the smallest selected component capacity;
- one immutable aggregate state owns agent, environment, transaction, dispatch,
  environment-RNG cursor, metrics, event counters, pending outcome, halt,
  transcript, commit generation, and checkpoint generation;
- a pure transaction reducer lets the aggregate runner own the only live CAS,
  while the standalone ledger retains its process-local lock/current-object
  behavior;
- runner-derived observation IDs and one outer lock cover authority,
  settlement, command issuance, strict action validation, functional
  environment execution, post-execution receipt, outcome, agent/metric staging,
  and one aggregate commit;
- each concrete environment adapter recomputes state progression, observation,
  reward, regime, oracle reward, and continuing-boundary semantics before a
  result reaches learning;
- RiverSwim has a distinct manifest/state discriminator, requires stationary
  metrics, and rejects `n_states` outside `[2, 12]` before constructing its
  exponential exact oracle;
- RiverSwim execution and validation receive the identical runner-derived JAX
  key, and validation replays the stochastic transition exactly;
- phase/reward/oracle/regret metrics, horizon completion, transcript hashing,
  abort, known post-execution divergence, and complete-outcome recovery without
  redispatch are retained; and
- an ordinary-`Exception` post-issue guard commits a process-local emergency
  CAS halt so the original immutable snapshot becomes stale.

This runner remains a `preview1` L0 development mechanism. Its declared static
exact authority is not an independent safety policy, and its synthetic veto
test is not safety conformance. Its at-most-once property is relative to stale
snapshots within one live process; it is not a durable executor guarantee and
excludes process death, `BaseException`, hardware delivery, and reconciliation.
Options, rebinding, boundaries, wire decoding, additional/general environments,
robot and Forager adapters, `reference-dev`, learning/performance benefit, and
evidence remain open.

### Implemented quiescent checkpoints and exact resume

The development-only
[checkpoint module](../../alberta_framework/reference_life_checkpoint.py) and
its [Switching tests](../../tests/test_reference_life_checkpoint.py) and
[RiverSwim tests](../../tests/test_reference_life_riverswim_checkpoint.py)
implement a canonical current-schema bundle for both supported primitive lives.
Save is permitted only at an armed, quiescent, pre-completion boundary. On Linux
it atomically publishes one immutable no-replace generation, nests the complete
Prototype v3 checkpoint, encodes every aggregate owner, binds current selected
source/runtime/dependency identities and child consistency hashes, reconstructs
fresh components from the distinct environment implementation/state
discriminator, and requires strict cross-component adoption before
continuation.

The retained gate checkpoints after `N` events, continues the original runner
for `M` events from the committed barrier, restores that same barrier into a
fresh runner, and requires exact continuation steps, events, and aggregate
state, including exact keyed stochastic RiverSwim continuation. These are L0
simulation results, not a frozen or portable migration contract, authenticated
execution attestation, durable dispatch replay, process-crash guarantee, safety
result, RiverSwim learning or performance result, `reference-dev`, or evidence.
In-flight, recovery, completed, unimplemented-environment, and physical-state
restore are unsupported.

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
| Dispatch | Dispatch adapter; proposed and authorized action lineage, safety/watchdog state, and any pending command or receipt |
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
must halt rather than learn from a counterfactual proposal. The host next issues
a distinct command that binds the settlement to a canonical command ID and the
configured executor ID and epoch. After the executor reports execution, a
distinct receipt binds that command, receipt ID, and independently encoded
applied action. If the applied action differs from the settled command, the
ledger halts before recording an outcome or applying learning.

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
5. Authorize and settle that decision.
6. Issue one executor-bound dispatch command.
7. Dispatch the commanded action exactly once.
8. Record the executor's matching post-execution receipt and applied action.
9. Record the matching outcome exactly once.
10. Apply one atomic agent update, advance metrics and runner counters, and arm
   the next decision.
11. At a declared quiescent boundary, optionally write a whole-life checkpoint.
12. Continue across task and episode boundaries without reinitializing learned
    agent state. Only explicitly declared episodic state may reset.

The implemented checkpoint API writes only after an accepted outcome and before
the next command. A successful publication advances commit and checkpoint
generations and persists that exact barrier; restore/adoption advances neither.
In-flight and physical actions remain outside the exact-resume gate until a
later protocol defines durable idempotent dispatch and reconciliation.

In the preview ledger, rejecting an outcome leaves its event unconsumed and
retains the transaction in `halted`. The aggregate runner can retry agent and
metric staging only from a complete retained outcome, without command issuance
or environment execution; vetoes, incomplete/uncertain results, known
post-execution divergence, and stale snapshots cannot use that recovery path.
This recovery is process-local and is not durable replay/restore. Accepting the
event at the maximum uint64 decision index consumes that final event and moves
the ledger to `exhausted`; it never wraps the counter.

## Dispatch and transition invariants

Conforming implementations must enforce all of the following:

- A decision is used at most once and only within its lifecycle.
- The observation, decision ID, authorized action, command, post-execution receipt,
  applied action, and outcome form one
  ownership chain. Stale, replayed, cross-life, out-of-order, or mismatched
  records fail closed without a partial learning update.
- Learning credits the action actually dispatched. Safety clipping, action
  substitution, skill selection, residual blending, or actuator projection may
  not be hidden behind the proposed action.
- A command exists before execution and a receipt only afterward. A receipt must
  report the independently encoded applied action; a mismatch with the settled
  command halts before outcome recording or learning.
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

For the implemented current-schema simulator gates, including keyed stochastic
RiverSwim:

1. Run `N` events and atomically publish quiescent barrier `B`.
2. Continue the original runner from `B` for `M` events.
3. Reconstruct fresh components, restore the same persisted `B`, and run `M`
   events.
4. Require exact equality of every continuation step and event and the full
   aggregate state, including decisions/actions, observations/outcomes, agent
   and environment state, metrics, transcript, counters, generations, and RNGs.

A checkpoint-free control intentionally lacks the barrier's commit/checkpoint
generation increments and is not the full-state oracle. Floating tolerances do
not satisfy this gate. Serialized checkpoint directory bytes need not match;
restored semantic state must.

A physical environment generally cannot be restored exactly. Hardware restart
must use a separately named reconciled-resume mode that re-establishes sensor,
safety, and actuator state under independent authority. Reconciled resume is an
important application gate, but it does not count as passing exact resume.

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
Prototype decision/cache validity may disarm at its int32 telemetry or
observation capacity before the host reaches its final decision index. The
aggregate runner now binds
`PROTOTYPE_REFERENCE_MAX_ACCEPTED_EVENTS = 2**31 - 4` into
`max_accepted_events` and rejects an oversized life before component
initialization. This is capacity enforcement, not a long-horizon stability or
resource result.

The implemented quiescent whole-life checkpoint nests the existing v3
Prototype checkpoint as its agent portion rather than treating it as the entire
bundle. A future rebinding-capable adapter must expose a public, decision-bound
settlement path that either preserves or atomically rebinds OaK/STOMP
base-versus-option credit when authorization changes the primitive action, and
it must update every action-bound cache. Direct runner mutation of Prototype
internals is forbidden.

The concrete runner and checkpoint paths use `SwitchingTwoStateMDP` and
`RiverSwimMDP` in `alberta_framework.streams.closed_loop`. RiverSwim is bounded
to at most 12 states in this reference slice because its exact stationary oracle
enumerates `2**n` policies; this is a resource guard, not a scaling or
performance result. Before the adapter may enter Forager or be called
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
- establish RiverSwim or broader learning/performance benefit, state of the
  art, scientific promotion, Alberta Plan completion, robotics readiness, or
  guarded hardware authorization;
- replace benchmark-specific frozen protocols or evidence validators;
- rename the `alberta_framework` compatibility namespace or historical schemas;
- claim exact restoration of the physical world; or
- authorize edits to immutable or append-only `outputs/` artifacts.

## Acceptance sequence

The proposal advances only in this order:

1. **Host transaction preview — implemented at L0.** The versioned-but-unfrozen
   `preview1` manifest and transaction-state schemas, immutable typed payloads,
   distinct authorization, settlement, command, receipt, and outcome records, explicit
   bootstrap/reset observation IDs, process-local current-object ledger, and
   retained validation/ownership tests implement the host transaction slice.
   This is structural and nonpromoting; it is not a frozen compatibility version.
2. **Primitive aggregate lives — implemented at L0.** Process-local Prototype +
   SwitchingTwoState and Prototype + RiverSwim paths own the full aggregate and
   cover synchronous dispatch, learning updates, metrics, faults, and
   no-redispatch recovery. RiverSwim uses a distinct discriminator, stationary
   metrics, exact keyed stochastic validation, and a strict 12-state cap.
3. **Quiescent exact resume — implemented at L0.** The current-schema bundle,
   strict restored-state adoption, and exact barrier-fork continuation gate pass
   for both supported constructions, including keyed stochastic RiverSwim.
4. **Matched development scorecard — implemented at L0 and permanently
   nonpromoting.** The scorecard module freezes 12 consumed development seeds,
   two environments, six fresh-process arms, explicit Threefry roots, an
   environment-bound finite-horizon privileged dynamic-programming control,
   fixed reward-lattice checks, and canonical numeric-payload accounting. Its
   validator is a consistency gate, not authenticated execution attestation.
   Issuing or validating the plan does not select `reference-dev` or create a
   completed performance or scientific result.
5. **Broader conformance — open.** Exercise active-option ownership,
   replacement/rebinding, real veto authority, boundaries, extension/wire
   policy, numerical stability, and resource ceilings.
6. **Forager bridge — open.** Add the ordinary-observation adapter only after
   extended-action ownership survives real runner dispatch.
7. **Robot simulation adapter — open.** Externalize complete controller state,
   bind continuous action after safety/residual transforms, and pass
   deterministic simulation, fault, and latency gates.
8. **`reference-dev` decision — open.** Only a separate permanently
   nonpromoting decision after the full regression panel may select a concrete
   `reference-dev` configuration and rollback policy.

Until the final step is recorded, ASI has a selected proposed architecture but
no canonical reference agent or `reference-dev` baseline.
