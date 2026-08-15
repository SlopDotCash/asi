# Evidence methodology

This guide explains what counts as evidence in ASI, how evidence is promoted,
and which Alberta Plan properties have registered scientific artifacts. It is
intentionally not a session log, implementation inventory, roadmap, or live
results dashboard. The same fail-closed rules apply to Alberta-derived and
non-Alberta research.

For current project status, read [the research status](../status.md). For the
machine-readable claim contracts, registered source sets, and current
validation result, use
[`evidence_manifest.py`](../../alberta_framework/evaluation/evidence_manifest.py)
and run:

```bash
.venv/bin/alberta-evidence-status
```

The command validates existing artifacts; it does not rerun a scientific
protocol. Unless `--output` is supplied, it only prints the operational
manifest.

## Authority and interpretation

When descriptions disagree, use this order of authority:

1. the strict artifact loader and validator for the artifact schema;
2. the registered contract in `EVIDENCE_SPECS`;
3. the immutable artifact bytes and their reconstructed scientific payload;
4. the frozen protocol or runbook that produced those bytes;
5. this guide and other narrative documents.

A stored `passed` flag is never trusted by itself. Validators reconstruct the
protocol, configuration, seed schedule, thresholds, source provenance,
statistics, acceptance checks, and scientific/content digest. The registry
then checks that reconstruction against its own frozen contract.

An accepted registry entry supports only its `claim_scope`. It is not evidence
for a broader mechanism, a different source tree, ASI's whole-agent target, or
completion of the Alberta Plan. In particular:

- **mechanism** means a controlled component behavior; it may use supplied or
  oracle information;
- **integrated** means the behavior occurs inside a closed-loop or
  multi-component agent;
- **autonomous** means the learner discovers and maintains the required
  machinery without supplied target features, context labels, task IDs, or
  boundary callbacks.

These labels are not interchangeable. Integration does not imply autonomy,
and a mechanism test does not imply an integrated result.

## Evidence levels

The project uses four evidence levels:

- **L0 — mechanism:** API, shape, finite-value, serialization, transaction, or
  local-update correctness.
- **L1 — learning:** a component learns in a controlled toy problem.
- **L2 — comparison:** a preregistered, multi-seed, matched-resource benchmark
  establishes the frozen comparative claim.
- **L3 — integration:** one uninterrupted agent life demonstrates the required
  interactions, retention, recovery, bounded resources, and causal ablations.

The evolving completion gates and requirement matrix live in
[the research status](../status.md). As a general rule, a Plan step needs its
defining outcome at L2 and the required cross-step interactions at L3. L0 and
L1 are necessary engineering evidence, not substitutes for those gates.

## Evidence classes and marker lanes

Evidence level describes strength; evidence class describes promotion
authority.

| Class or marker | Proper use | Promotion authority |
|---|---|---|
| `unit` / `smoke` | Fast correctness or local behavior | None; valid records are nonpromoting |
| `integration` | A test crossing component, persistence, CLI, or process boundaries | None by marker alone |
| `development` | Calibration, debugging, screening, feasibility, ablation design, and negative-result discovery | Permanently nonpromoting |
| `scientific` | A frozen protocol with untouched held-out seeds and a versioned strict artifact | May support its exact L2/L3 claim if every gate passes |
| `slow` | Wall-clock classification | Says nothing about evidence strength |

The IPMNIST screening campaign and other development lanes remain
nonpromoting even when they use many seeds, strict schemas, confidence
intervals, or immutable outputs. Scientific promotion is a protocol property,
not a reward for scale or test coverage.

## Fail-closed promotion contract

Promotion is manual and claim-specific. A promotable run requires all of the
following before held-out evidence is inspected:

1. **Frozen claim and protocol.** The estimand, configuration, schedule,
   baselines, resource accounting, metrics, statistics, gates, and excluded
   claims are explicit.
2. **Disjoint seed roles.** Development/calibration seeds and promoted
   held-out seeds are fixed and nonoverlapping. Exposed or consumed seeds can
   never become fresh evidence for a new claim.
3. **Frozen thresholds.** Thresholds are calibrated empirically on development
   data with at least 2× margins, then frozen. A held-out failure is a valid
   rejection; the gate is not retuned.
4. **Versioned artifact.** The writer uses a new path and schema version and
   records the complete scientific payload, source provenance, environment,
   and digest. Existing pinned output is never overwritten.
5. **Strict reconstruction.** The loader rejects malformed, non-finite,
   incomplete, reordered, or extra data as required by its schema. The
   validator recomputes aggregates, intervals, comparisons, and acceptance
   from primitive records.
6. **Registered source set.** Exact bytes for every manually registered source
   path must match the artifact or an explicitly defined historical compatibility
   path. Current v1 sets are not complete recursive execution closures; a future
   protocol must version and bind that stronger identity rather than relabeling v1.
7. **Matched comparison.** Claimed resource, state, action, seed, and work
   matching must be measured and validated, not inferred from similar code.
8. **Immutable result.** Acceptance, rejection, and execution failure are all
   retained. Tests, replays, or later reruns never auto-promote a claim.

Development runs may be rigorous and reproducible, but they must say
`development_only` or its schema-equivalent and must not claim scientific
promotion. If development data influenced an arm, threshold, baseline, or
interpretation, those data are consumed for promotion purposes.

Consumed-seed replay is allowed only for compatibility or reproducibility
diagnosis. It cannot refresh held-out status, authorize retuning, strengthen a
claim, or promote changed code.

## Source hashes, dirty trees, and immutable artifacts

The registry's dirty-state policy is narrower than “git must be clean”:

- an unregistered worktree change is recorded as operational provenance but
  is neither sufficient for promotion nor independently disqualifying;
- a changed registered source hash invalidates the ordinary current-source
  validation path;
- file presence, a clean worktree, or matching artifact SHA-256 is not enough
  without strict scientific reconstruction.

SHA-256 fields establish byte identity and integrity context. They do not
establish authorship, origin authenticity, or scientific generality.

Pinned `outputs/` artifacts are immutable. Never edit, overwrite, or delete a
registered artifact, its sealed provenance, or an allowed historical replay
chain. A renewed claim requires a new output path, a new schema version,
preregistration, and untouched seeds. Consult the repository `AGENTS.md`
before touching any output because immutability extends beyond the five
artifacts listed here.

## Evidence registry semantics

The ordinary registry is an operational index over narrow claims. It is not a
scientific artifact, a signature, an ASI score, or an Alberta Plan completion
certificate.

### Per-claim status

| Status | Meaning |
|---|---|
| `accepted` | The artifact is present, valid, scientific, L2/L3, promotion-enabled, and passed every frozen gate |
| `valid-rejection` | The artifact is valid but at least one frozen scientific gate failed |
| `not-run` | The required artifact is absent |
| `invalid` | Parsing, schema, integrity, provenance, source, reconstruction, or registered-contract validation failed |
| `verified-nonpromoting` | A valid unit/smoke or otherwise nonpromoting record; it cannot support a scientific claim |

### Overall status and exit code

| Exit | Overall status | Interpretation |
|---:|---|---|
| `0` | `accepted` | Every registered promoting claim is valid and accepted |
| `1` | `valid-rejection`, `not-run`, or `nonpromoting` | Evidence is valid but rejected, absent, or has no promotion authority |
| `2` | `invalid` | At least one claim fails strict validation, or the manifest status is unrecognized |

Always run the command for live status. Do not copy a dated registry snapshot
into overview documentation. An immutable artifact can retain a historical
accepted or rejected outcome while no longer certifying the current registered
source tree.

## Registered scientific claims

The registry currently defines five exact L2 contracts. The “frozen outcome”
column describes the result stored in the immutable artifact under its frozen
protocol; it is not a claim about the live registry result.

| Claim | Exact scope | Frozen outcome | Principal exclusions |
|---|---|---|---|
| `recurring_pair_features` | Retention and deployed-bank allocation of supplied pair products in the frozen Gaussian/L2 recurring probe | **Accepted.** All 30 held-out lives retained A/B/C in the active bank and evicted obsolete D from that bank | The exhaustive candidate archive remains supplied and counted; no archive erasure, autonomous head/target discovery, general feature discovery, control, indefinite memory, or general forgetting claim |
| `scale_robust_pair_features` | Online scale-robust selection and structural active-bank retention from a counted exhaustive degree-two archive in a visibly cued nine-phase regression gauntlet | **Accepted** on the frozen 30-seed namespace-derived schedule | No open-ended discovery, uncued context inference, archive erasure, retention-caused performance claim, multi-initialization robustness, isolation of normalization alone, or control claim |
| `ftl_world_model_decision_fidelity` | A sparse fixed-shape online transition model preserves low known-reward open-loop menu regret on deterministic A-B-A visitation and beats the included untrained and raw online-ridge baselines | **Accepted** as the frozen historical result | No closed-loop acting, learned reward model, stochastic/POMDP or visual fidelity, compute/capacity-matched superiority, global FTL guarantee, or indefinite retention claim |
| `recurring_multiagent_coadaptation` | Fixed-memory online coadaptation and retention in a visibly cued two-agent A-B-A sanity benchmark | **Accepted.** The frozen 30-seed artifact passed matched-budget uplift, probe, forgetting, recovery, and stability gates | Tiny contextual bandit; no uncued inference, feature discovery, recommendation intervention, general coadaptation, IA, or general forgetting claim |
| `continual_intelligence_amplification` | Causal recommendation-channel uplift for one selected IA/partner pair in a deterministic hidden-phase micro-MDP | **Valid rejection.** Uplift and augmentation controls passed, but the action-changing intervention rate was `0.08728`, below the frozen `0.10` gate | Fixed hand-designed MDP and selected `p_accept=0.5`; no general partner/environment/population, autonomous discovery, realistic Step 12, or Alberta Plan claim |

Artifact and validator entry points:

| Claim | Immutable artifact | Strict artifact implementation |
|---|---|---|
| `recurring_pair_features` | [`evidence.v1.json`](../../outputs/recurring_feature/evidence.v1.json) | [`recurring_feature_artifact.py`](../../alberta_framework/evaluation/recurring_feature_artifact.py) |
| `scale_robust_pair_features` | [`evidence.v2.json`](../../outputs/scale_robust_feature/evidence.v2.json) | [`scale_robust_feature_artifact.py`](../../alberta_framework/evaluation/scale_robust_feature_artifact.py) |
| `ftl_world_model_decision_fidelity` | [`evidence.v1.json`](../../outputs/ftl_decision/evidence.v1.json) | [`ftl_decision_artifact.py`](../../alberta_framework/evaluation/ftl_decision_artifact.py) |
| `recurring_multiagent_coadaptation` | [`evidence.json`](../../outputs/continual_multiagent/evidence.json) | [`continual_multiagent_artifact.py`](../../alberta_framework/evaluation/continual_multiagent_artifact.py) |
| `continual_intelligence_amplification` | [`evidence.json`](../../outputs/continual_ia/evidence.json) | [`continual_ia_artifact.py`](../../alberta_framework/evaluation/continual_ia_artifact.py) |

The manifest contains the authoritative seed schedules and threshold payloads;
they are intentionally not duplicated here. In brief, each artifact separates
development/calibration seeds from a frozen 30-seed promoted schedule. The
scale-robust claim uses a namespace-derived schedule because part of the
ordinary numeric namespace had already been exposed.

### Historical compatibility chains

The FTL and IA entries have narrow historical-chain handling in the registry:

- FTL may retain the original historical acceptance only when the immutable
  artifact, invariant registered sources, reconstructed primitive results,
  attestation, and exact consumed-seed compatibility replay satisfy the
  special contract. The replay is not new evidence, and the chain does not
  claim complete historical source recoverability.
- IA may retain only the original historical valid rejection through its
  archived-source and compatibility chain. The failed `0.10` intervention
  gate remains failed.

If either special chain fails, the claim is invalid. A changed protocol or a
new current-code promotion still needs a new schema/path and untouched seeds.

## Alberta Plan evidence crosswalk

This table routes each Plan property to its strongest durable evidence class
without duplicating the changing implementation inventory. Read
[the research status](../status.md) for current mechanisms, executions, and
blockers.

| Step | Property | Durable evidence route and present ceiling |
|---:|---|---|
| 1 | Track nonstationary prediction with normalization and relevance-sensitive step sizes | L0/L1 mechanism and toy-learning coverage; no registered scientific claim |
| 2 | Generate, value, retain, and replace nonlinear features under a fixed budget | Two narrow registered L2 contracts (`recurring_pair_features`, `scale_robust_pair_features`) have frozen accepted outcomes; supplied finite pair archives and visible cues leave autonomous, open-ended selective retention open |
| 3 | Learn many continuing and off-policy predictions with history and feature finding | L0/L1 Horde, TD/GTD, trace, and state machinery; no registered matched comparison or L3 result |
| 4 | Learn contextual and sequential control with adaptive features | L0/L1 control mechanisms plus development diagnostics; no promoted matched-resource retention/control result |
| 5 | Learn differential GVFs and option value/duration predictions | L0/L1 prediction mechanisms; no promoted integrated option-control result |
| 6 | Provide reproducible continuing-control benchmark coverage | Development and replication lanes only unless a separately frozen scientific artifact is issued; no registered suite-completion claim |
| 7 | Validate average-reward planning, then adaptive-feature planning | L0 world-model, dreaming, and option-search mechanisms; no promoted closed-loop planning-benefit result |
| 8 | Close perception → model → feature ranking/replacement → model feedback | `ftl_world_model_decision_fidelity` has a narrow frozen accepted L2 outcome; it is open-loop with known rewards, while the end-to-end feedback loop remains below L3 |
| 9 | Improve exploration and planning order under matched budgets | L0 search-control mechanisms; no registered scientific claim |
| 10 | Discover, learn, model, compose, and retire reward-respecting options | L0 STOMP and option-model mechanisms; no autonomous repeated-lifecycle or matched-benefit result |
| 11 | Use causal utility to replace features, options, and models safely | L0 utility, feature-lifecycle, and option-keyboard mechanisms; no promoted causal lifecycle result |
| 12 | Measurably improve another learning agent in a closed loop | `recurring_multiagent_coadaptation` has a narrow frozen accepted L2 outcome; `continual_intelligence_amplification` is a frozen valid rejection; realistic causal partner benefit and L3 integration remain open |

## Known limitations

- No whole-agent L3 protocol or result has been completed.
- The feature claims cover finite, supplied degree-two archives and visible
  context, not autonomous open-ended feature discovery or an indefinite
  retention theorem.
- The frozen historical FTL outcome is a deterministic known-reward menu
  diagnostic, not closed-loop world-model control or compute-matched planner
  superiority; its current validity must be checked live.
- The frozen historical multi-agent outcome is a visibly cued sanity benchmark,
  not IA. The frozen IA artifact outcome is a valid rejection; neither phrase
  claims that the current source passes validation.
- Development campaigns, including IPMNIST screening, cannot promote their
  inspected seeds or selected arms.
- Publication-shaped or source-faithful replication machinery is not a
  scientific result until its full frozen execution and validation contract is
  satisfied.
- Source-compatibility replay can diagnose reproducibility but cannot create
  untouched evidence.
- Artifact and source hashes establish integrity, not authenticity or broad
  external validity.

Record bounded negative conclusions in
[`negative-results.md`](negative-results.md) so failed gates and consumed
development ideas are not silently retried or reinterpreted.

## Working procedure

Before changing evaluation code or registered sources:

1. inspect `EVIDENCE_SPECS` and the artifact's strict validator;
2. determine whether the file is in a registered source set or another
   output provenance manifest;
3. expect persisted evidence to become invalid if registered bytes change;
4. do not edit the pinned artifact or weaken its validator to restore status.

For a development run:

1. use a new output path;
2. declare development/nonpromotion status in the schema;
3. bind configuration, seed roles, source bytes, environment, and chronology;
4. retain rejections and execution failures;
5. append the bounded conclusion to the negative-results ledger when useful.

For a new scientific claim:

1. preregister the exact narrow claim and excluded claims;
2. freeze matched baselines, budgets, thresholds, statistics, and untouched
   seeds before execution;
3. create a new versioned artifact schema and strict validator;
4. execute once under the frozen protocol;
5. validate without retuning;
6. register the claim only through an explicit review—never automatically.

Useful checks:

```bash
.venv/bin/alberta-evidence-status
.venv/bin/python -m pytest tests/test_evidence_manifest.py -q
```

The command output and immutable artifacts are the evidence record. This file
is only the durable map for interpreting them.
