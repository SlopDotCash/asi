# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the execution-gated `asi-bimu-matched-development` campaign for the
  bounded five-task BiMU mechanism-on versus memory-off comparison. Its fixed
  six-shard namespace, fresh-process roster, paired outcome rule, validators,
  and resource/provenance bindings are permanently nonpromoting and not
  paper-comparable; execution remains explicitly unauthorized. Its publicly
  exposed three-seed development roster is consumed for promotion and has no
  retained matched result.
- Added the unfrozen `preview1` reference-agent transaction ledger, primitive
  Prototype adapter, aggregate SwitchingTwoState/RiverSwim life runner, and
  development-only quiescent checkpoint codec. These L0 mechanisms enforce
  exact dispatch ownership, immutable aggregate state, strict transition
  validation, and exact continuation for the documented simulator/configuration
  pairs; they are not `reference-dev`, safety, robotics, performance, or
  scientific evidence.
- Added the permanently nonpromoting reference-life development scorecard and
  `asi-reference-life-scorecard` CLI. Its literal-frozen 144-shard plan compares
  six arms across two environments and 12 consumed development seeds with
  explicit source identity, Threefry roots, reward/counter validation, and
  numeric-payload accounting. The machinery does not constitute a completed
  run or result.
- Added the code-only, permanently nonpromoting IPMNIST gate-ablation arm
  `rls_head_resid_l1_preset005_nogate`. It has no result artifact and does not authorize a
  benchmark run; execution remains gated by the issue's source receipt and owner-approved
  compute budget.
- Added the code-only, permanently nonpromoting IPMNIST arm
  `rls_head_resid_l1_preset005_l2init`, which snapshots and decays exactly `w1`, `b1`, `w2`,
  and `b2` toward initialization while leaving `w3`, `b3`, the incumbent state/configuration,
  and the RLS recursion unchanged. It has no result artifact and does not authorize a benchmark
  run; execution still requires a replacement frozen protocol, fresh eligible seeds, and
  owner-approved compute.
- Added the code-only, permanently nonpromoting IPMNIST reset ablation
  `rls_head_resid_l1_noreset`, which changes only the incumbent's detector-driven covariance
  reset threshold to its existing untriggerable endpoint. It has no result artifact and does
  not authorize a benchmark run; execution remains gated on protocol, seed, and compute
  approval.

### Changed

- Replaced the unissued activation/feature campaign's exposed seed rosters with
  untouched v2 rosters; `0`–`4`, `156610`–`156614`, `2156600`–`2156604`, and
  `2156610`–`2156614` are quarantined because they were exposed or exercised.
  All future artifacts use NEW v2 namespaces. Public shard execution
  now fails before dataset access or runner dispatch pending a separate reviewed
  authorization transition. Output transactions reserve through pinned
  no-follow directory descriptors before work, publish without replacement,
  fsync, and strictly reread through the pinned descriptor with the artifact's
  plan-bound validator. No campaign execution or output is included.
- Made new IPMNIST screening shards and summaries strict v2 artifacts bound to a clean Git
  commit/tree, the actual tracked package and `uv.lock` bytes, runtime/JAX details, and the
  canonical materialized MNIST feature/label bytes. The run CLI now captures these bindings
  before execution and verifies them again before immutable publication. Merge and proxy
  validation now bind their exact input files and live derivation environment, enforce strict
  policy, hyperparameter, metric-domain, device, and tolerance contracts, and reject changed or
  mismatched provenance while legacy v1 shards remain readable.
- Moved plotting, dataset, progress, and parallel-execution dependencies from the base
  installation into a `research` extra; the `dev` extra retains the complete contributor
  toolchain.
- Made the source-checkout installation path explicit and documented that the existing
  `alberta-framework` project on PyPI is a different distribution.
- Consolidated locked CI setup, pytest shard planning, and package validation so each concern
  has one implementation and built artifacts are tested without rebuilding them.
- Extended pre-1.0 learner and optimizer result schemas with explicit scalar or per-channel
  `update_applied` masks. Transaction-aware callers use the checked optimizer boundary while
  the legacy two-value `update_from_gradient` API remains available.
- Added explicit scalar and per-event transaction masks to working-memory checked-update,
  step, and array results while preserving the latter two legacy unpacking surfaces.
- Kept finite one-step optimizer results byte-pinned. The added JAX transaction predicates can
  lower differently in compiled multi-step scans and may change last-bit rounding without
  changing the finite-path equations or acceptance semantics.

### Fixed

- `upgd_memory._normalize_simplex` now returns a probability simplex for
  zero, negative, sub-`1e-12`, and float32-overflowing inputs.  The previous
  floored-denominator helper could drop mass (including the reachable
  zeros `previous_targets` target-trace blend) without reporting it.
- Canonicalized discounted-SARSA lifecycle timers at the exact-dispatch
  reference-control boundary. The first compiled update can no longer grow
  persistent numeric state by promoting two host floats into JAX leaves, so
  scorecard resource accounting remains fixed from initialization.
- Required `PartialObservationWrapper`'s RANDOM-mode `mask_prob` to be a
  concrete, non-bool real number in `[0, 1]` before use. The prior check
  compared the raw value directly (`0.0 <= mask_prob <= 1.0`) with no type
  gate at all, so any object supporting `__le__`/`__ge__` — including a
  `__class__`-spoofed non-real whose comparison hooks raise — reached that
  comparison and could leak an uncaught exception instead of the documented
  `ValueError`, or silently pass through as a non-canonical value.
- Required persisted micro-stream scalar fields to be concrete real values and
  canonicalized them to built-in floats, preventing numeric strings or custom
  conversion objects from crossing the strict development-artifact boundary.
- Validated frequency-mismatch bounds and Pavlovian noise in their actual
  float32 execution domain, rejecting overflowed or collapsed frequency
  intervals and overflowed noise scales before stream state can become invalid.
- Preserved non-finite sentinels on rejected recurring-multiagent transition
  payloads because the compatibility `step()` surface does not return the
  separate transaction-validity bit; invalid events cannot masquerade as
  ordinary finite zero-reward samples.
- Rejected SIGReg samples, embeddings, directions, and derived projections that are
  non-finite in float32 arithmetic before they can become silent NaN losses, including
  under compiled and vectorized JAX calls.
- Added an explicit traced validity verdict to behavior-model input gradients so finite neutral
  payloads from rejected inputs cannot be mistaken for valid state-builder updates; eager,
  compiled, and vectorized consumers can gate mutation on the same result field.
- Built action-conditioned world-model interaction features from finite internal operands while
  preserving NaN as the fail-visible public result for non-finite observations or invalid actions;
  valid eager, compiled, and vectorized predictions retain the same values.
- Required automatic feature-to-subtask counts to be non-negative integer scalars before
  ranking, while retaining NumPy/JAX integer-scalar compatibility and zero/oversized bounds.
- Aligned plotted trailing-window learning-curve means and confidence intervals with the
  time step where each complete window closes instead of backdating padded values.
- Preserved initialized weights and input-scale vectors for one complete configured segment
  before `XDistShiftStream`, `AbruptChangeStream`, and `DynamicScaleShiftStream` perform
  their first abrupt changes.
- Made RiverSwim reject malformed, non-finite, subnormal, or jointly invalid
  transition probabilities before float32 narrowing can create a negative or
  non-stochastic transition row.
- Required GVF discount and eligibility-decay parameters to be concrete finite
  float32-domain probabilities, preventing invalid or silently narrowed values from
  corrupting Horde TD targets and traces.
- Rejected malformed or float32-unsafe SIGReg kernel widths before Epps-Pulley
  arithmetic can emit NaN, while preserving valid dynamic JIT and vectorized calls;
  SIGReg configuration scalars now enforce the same finite representability contract.
- Isolated Step 9 candidate, behavior-rollout, and control-rollout RNG branches, persisted
  a reserved future master through real and dream control-update rejection, rejected dreams
  with any failed rollout transaction, and kept zero-budget updates identical to the
  real-only path.
- Kept Step 7 real and rollout control RNG streams linear through transactional rejection
  whenever planning is enabled, while retaining the exact real-only path at planning zero.
- Rejected nonfinite and out-of-range Pavlovian phase contingencies before they can
  silently behave like zero- or certain-reinforcement probabilities.
- Required nexting RMSE inputs to be equal nonempty rank-two arrays and their
  burn-in/window controls to be static built-in integers within trace bounds, preventing
  broadcasted metrics, empty outputs, and late JAX slicing errors.
- Preserved every leading batch axis when applying gauntlet EMA smoothing along the last
  axis, fixing rank-three and higher inputs while retaining rank-one and rank-two behavior.
- Made the pair-, triple-, and micro-component stream generators select exactly their
  configured fixed count under tied random scores, with stable source-index tie-breaking;
  pair/triple counts now require positive built-in integers before JAX initialization.
- Bound each new IPMNIST screening pool shard and merged summary to its effective
  `noise_pool_steps`, rejected malformed or cross-shard pool sizes, and made merge refuse
  legacy pool shards whose omitted size remains inspectable but unknowable. Legacy shards
  without a noise mode remain compatible as exact step-mode runs.
- Made strict evidence, IPMNIST-screening, micro-continual, UPGD/Label-EMNIST, and sealed
  Forager artifact loaders reject duplicate JSON object keys at every nesting depth and JSON
  numeric tokens whose float conversion overflows to infinity.
- Rejected out-of-int32-range, zero, negative, boolean, and non-integer schedule divisors in
  synthetic and closed-loop streams before JAX modulo, floor-division, or periodic arithmetic
  can silently corrupt a regime clock or raise late overflow errors; valid int32 schedules
  retain their existing trajectories.
- Required every periodic partial-observation mask to match the wrapped stream's exact feature
  shape before stacking, preventing broadcast expansion of feature vectors into matrices.
- Made experiment CSV and JSON artifacts preserve finite binary64 measurements exactly. A
  shared preflight now requires nonempty aggregates, canonical uint32 seed identities,
  aligned nonempty metric arrays and summary samples, finite measurements, and the requested
  report metric before touching a parent, destination, or report directory; complete
  serialized files publish atomically, while display-only table rounding remains unchanged.
- Protected the reserved checkpoint `_format_version` metadata key so every newly saved
  checkpoint carries the canonical on-disk format stamp even if caller metadata collides.
- Made sparse initialization reject malformed dimensions and non-finite or out-of-range
  sparsity before array allocation, while retaining the paper-specified uniform bound.
- Validated rule-discovery and UPGD run, plan, CLI, shard, and artifact seed identities before
  integer/uint32 conversion: schedules must use unique built-in integers in the JAX key domain,
  so duplicate or aliased trajectories cannot be counted or recorded as independent runs.
- `get_final_performance` now refuses a non-positive `window` and an empty time
  axis instead of silently averaging the whole trace (`window=0`, because
  `arr[:, -0:]` is a full slice) or a prefix (`window < 0`) and publishing NaN.
- Statistical helpers now reject empty Mann–Whitney groups and unequal or
  underpowered Wilcoxon pairs before calling SciPy; an empty Bonferroni family
  is a stable no-op that retains the requested family-wise alpha.
- Continual-learning metrics now require finite first and final evaluations and
  reject infinite post-exposure trajectories instead of backfilling across
  numerical divergence; intermediate NaN gaps remain valid missing probes.
- Removed broad package-import fallbacks that hid internal failures, and broke the
  learner/stream import cycle without making Gymnasium a base dependency.
- Added regression coverage for package import boundaries, release metadata, CI shard
  planning, synchronized agent guides, and local documentation links.
- Required at least two shared seeds and a positive candidate-minus-control
  difference on every aligned seed before an IPMNIST arm can authorize
  confirmation compute; one-seed paired summaries remain available for
  development inspection.
- Kept core Autostep and the screening lane's IDBD/Autostep meta-state finite when a
  non-finite correlation would otherwise turn a silent trace into a persistent NaN.
- Made optimizer, learner, Horde, actor-critic, model, option, normalizer, and utility updates
  reject non-finite transactions atomically, preserve the last valid state, expose the
  rejection, and return neutral outward diagnostics instead of partially committing a model.
- Selected exact zero bootstraps before multiplication in Horde, SARSA, nexting, and
  Gymnasium value streams, so terminal or disabled channels do not manufacture `0 * inf`
  NaNs while nonzero channels still expose invalid inputs.
- Held temporal/history context state on non-finite observations and kept valid finite
  coordinates observable without silently converting arbitrary invalid inputs into evidence.
- Made working/prototype/UPGD memory, state builders, world models, and their scan wrappers
  roll back complete invalid transitions. Rejected results expose explicit verdicts where the
  public result schema supports them and never report plausible success metrics; the Step 7
  and Step 9 scan facades retain the per-event model verdict.
- Rejected non-finite dream candidates and invalid or float32-unrepresentable resource costs;
  exact zero-weight channels remain unused, while positive-weight products must be finite.
  Discrete action/context identifiers and Exp3 credit probabilities are validated before any
  integer cast, lookup, or state update, and invalid prediction/selection queries stay visible.
- Strengthened IPMNIST and micro-continual shard merging to reject cross-seed learner,
  hyperparameter, mechanism, or runtime-environment drift, plus malformed or
  aggregate-overflowing wall-clock timings. The common runtime environment is now retained in
  derived summaries and validation receipts instead of being discarded.
- Aligned IPMNIST screening shard loading with the runner's noise-mode contract: unknown modes
  and pool attribution to arms without a noise-consuming update now fail before publication,
  while legacy shards with no mode remain compatible as exact step-mode runs.
- Made proxy receipts require strict historical reference schemas, frozen control identities
  and hyperparameters, compatible horizons, unique paired control seeds, and one common
  protocol before a validation result can be issued.
- Left the immutable evidence artifacts untouched. Registered-source hardening keeps the live
  five-claim registry fail-closed and `invalid` until separately authorized evidence renewal.

### Removed

- Removed superseded private helpers, unreachable branches, and the duplicate documentation
  index while preserving public APIs, evidence validators, and protocol boundaries.

## [0.29.0] - 2026-08-14

Changes since 0.28.0 are summarized by durable user-visible behavior. Development campaign
measurements remain in their primary output records; none promotes a registry claim.

### Added

#### Continual-learning benchmarks

- Extended the nonpromoting IPMNIST campaign with RLS-head, transient-ensemble,
  reviewer-comparison, update-rule-discovery, and input-conditioning controls. Moving
  measurements live under `outputs/ipmnist_screening/`; bounded conclusions are retained in
  `docs/evidence/negative-results.md`.
- Added the `gauss-v1` micro-continual suite for input permutation, label permutation, scale
  shift, and recurrence, with analytic references and campaign-arm reuse.
- Completed the three-seed, 800-task OPMNIST closure (48 million online updates per seed).
  Hybrid-memory candidates win the recorded online/tracking metrics while fair MLPs retain
  test-MSE/test-accuracy advantages. The strict gate remains
  `solved_opmnist_step2=false`; this is limited OPMNIST evidence, not a Step 2 solution.
- Added a fail-closed audit of the historical IPMNIST record. All stored shards parse and
  reported means reconstruct, but the proxy receipt rejects three UPGD prefix comparisons,
  the summary is incomplete, and the round-2 driver cannot reproduce its later result file.
  Those v1 shards remain hypothesis-generating observations, not current-source evidence.

### Changed

#### Reliability and compatibility

- OaK extended-action masks now exclude cold options consistently from selection, real
  bootstraps, planning selection, and planning bootstraps.
- **Pre-1.0 breaking API cleanup:** contracted the `alberta_framework` and
  `alberta_framework.core` convenience re-export namespaces. APIs owned by retired HCCL,
  embodied, prototype-expansion, and self-certifying evaluation modules were removed with
  those modules. Retained low-level schema, checkpoint-migration, resource-accounting,
  experimental-optimizer, and prototype-integration APIs now require imports from their
  defining modules instead of the package convenience namespaces. The robot track's documented
  continual-RL imports remain available.
- CI now runs the complete non-slow suite on Python 3.12 and 3.13, checks the lock and built
  distributions, exercises the exact NumPy/JAX floor, and exposes the retained slow lane by
  manual dispatch. The redundant main-only and sparse `unit`-only gates were removed.
- Raised the supported JAX/JAXlib floor from 0.4 to 0.7.1 while retaining NumPy 1.26
  compatibility for the robot track. The GPU extra uses the same JAX floor.
- The test harness preserves Chex tree-assertion semantics when exercising the NumPy 1.26
  floor, whose `numpy.testing.assert_allclose` predates the `strict` keyword Chex passes.
- Console-script loading is now safe when uv installs wheel files as hard links. The strict
  Forager source-identity checks still run when an attested command is actually executed.
- Normalizer coverage now uses a small deterministic exact-moment sequence instead of
  thousands of eager random JAX updates, so the file returns to the normal CI lane.
- Removed the unused direct Flax dependency from the base installation. The Forager extra
  still installs it transitively through Gymnax where that runtime inventory is relevant.

#### Scientific and evidence status

- No 0.29.0 result promotes an evidence-registry claim. IPMNIST, micro-suite,
  rule-discovery, partner, memory, and planning measurements remain development-only.
- The live evidence registry remains fail-closed with all five claims `invalid` because
  registered source bytes have changed since their artifacts were pinned. The immutable
  artifacts are unchanged; this release does not renew or promote those claims.
- Closed and consumed conclusions are indexed in `docs/evidence/negative-results.md`.

### Fixed

- Paired statistical comparisons now validate seed identities and metric-row counts,
  align paired t-test and Wilcoxon samples by seed, and reject unequal seed sets instead
  of silently pairing rows by position. Paired t-tests reject empty, undersized, or
  identical inputs; Wilcoxon rejects identical pairs instead of inheriting
  version-dependent SciPy outcomes; and empty bootstrap samples reject instead of
  producing NaN intervals. Valid unpaired singleton-vs-multi-value tests remain supported,
  Mann-Whitney comparisons remain unpaired, and one-seed time-series confidence intervals
  return the finite point trajectory.
- Seed-axis summaries, final-performance helpers, and per-step spread bands now use the
  sample standard deviation consistently, with zero spread for one seed. The within-run
  `compare_learners` helper remains explicitly population dispersion over recorded steps.
- Statistical helpers now reject nonpositive comparison windows, empty metric step axes,
  invalid confidence levels, unknown bootstrap statistics, and nonpositive bootstrap
  counts. Cohen's d reports a signed infinite effect for separated zero-variance groups
  instead of falsely reporting no effect.
- Multi-seed experiments now reject duplicate configuration names and duplicate explicit
  seeds before executing any learner or stream factory, preventing distinct treatments
  from being merged or deterministic trajectories from being counted twice.
- IDBD now preserves finite weight and bias step-size state when a non-finite meta-update
  reaches the clip guard, while retaining the existing bitwise finite-update trajectory.
- IPMNIST shard merging now refuses a missing named control or a candidate with no shared
  control seeds instead of ranking a comparison-less entry. Sigma-zero screening factories
  share one RNG-free noise path, keeping the reference and inert extension on identical
  lowered graphs.

### Removed

- Removed pytest-embedded lifetime, gauntlet, planning, discovery-control, and
  held-out research campaigns that calibrated pass/fail thresholds from their
  own runs without registered artifacts. Context-inference coverage now tests
  only the bounded mechanism.
- Removed the unissued Forager matched-v3 stack, the unexecuted HCCL/embodied/prototype
  expansion, and self-certifying evaluation islands that had no external consumer or
  scientific artifact.
- Removed source-presence manifests, not-assessed matrices, replay-only gates, public-export
  assertions, and permanently skipped tests that did not test runtime behavior.
- Removed checkout-mode and unavailable local-cache assertions from historical development
  receipt tests; retained records are still checked against their stored byte hashes.
- Retired the unconsumed slowly-changing-regression development harness; its historical
  partial records remain under `outputs/` and its bounded conclusion remains in the ledger.
- Retired the unissued UPGD-IPMNIST v3 governance stack, validator, runbook, and tests.
  The benchmark implementation and immutable v1/v2 development records remain unchanged.
- Removed the deprecated `alberta-evidence-gate` alias and its forwarding shim; use
  `alberta-evidence-status`.
- Removed the speculative WP0–WP9 and HCCL design documents. Current status and evidence
  authority now live in `docs/status.md`, `docs/evidence/methodology.md`, and versioned
  artifacts.

## [0.28.0] - 2026-08-01

### Added

- **Reference trust resolver** (`benchmarks/forager_matched_trust.py`, 25
  tests): stdlib HMAC-SHA256 signed-receipt resolver implementing the
  matched-campaign `TrustResolver` contract — trust-anchor document schema,
  0600 single-link key loading, detached receipt issuance, and a
  fail-closed `SignedReceiptTrustResolver` with immediate revocation and
  freshness windows. Not wired as a default; the reserved
  `content_only_unendorsed_v1` identity is refused. Asymmetric (Ed25519)
  variant deferred until a crypto dependency is pinned.
- **Hidden-partner lifecycle-world v6 runner and strict validator**
  (`evaluation/hidden_partner_lifecycle_world_v6_runner.py`,
  `..._v6_validator.py`): development machinery executing single v6 control
  bindings deterministically with content-hashed result records, plus
  fail-closed output validation. Certification and promotion flags remain
  false; reserved evidence namespaces are refused at run time.
- **Trainable-encoder latent world model** (opt-in, default-off; bitwise
  identical trajectories when disabled): encoder backprop through the latent
  prediction loss under the module's bounded-update discipline, gated by the
  anti-collapse diagnostic. Development-only (L0).
- **Kondo selection accounting** in the delightful-policy-gradient
  development lane: per-step accounting of forward-gate-selected actor logical work
  as a measured counterfactual. Actual compute gating remains unimplemented
  and `KONDO_IMPLEMENTED` remains false.
- **Development-only IPMNIST confirmations**: `adamw_cbp` completed 10 matched
  200-task seeds at mean online accuracy `0.79876` and `upgd_w_wd0005`
  completed 10 at `0.78431`, versus the repository's matched UPGD-W
  reproduction at `0.779147`. These are nonpromoting development results,
  not scientific evidence or a SOTA claim. The three-seed `upgd_ema_norm`
  result and its still-running extension, `upgd_idbd`, and subsequent
  composition screening are deferred to `[Unreleased]`.
- **STOMP checkpoint migration loader** (`core/options.py`): pre-expansion
  checkpoints (missing env-return/duration/baseline-mass fields) now restore
  with documented zero-fill semantics instead of failing on template
  mismatch.
- `alberta-forager-matched-sealed-evaluation` console script for the sealed
  held-out evaluation stage.

### Fixed

- Evidence CLI overwrite footguns: `ftl_decision_cli`, `continual_ia_cli`,
  and `continual_multiagent_cli` no longer default to writing over their
  sha-pinned canonical artifacts — pinned and pre-existing output paths are
  refused before any protocol runs, and write sites use exclusive-create.
  `scale_robust_feature_cli`'s default invocation now explains itself
  instead of exiting 2 bare.
- The legacy `alberta-evidence-gate` no longer checks vendored-out narrative
  documents or five unregenerable Step 1/2 JSON paths. It is now a deprecated
  compatibility wrapper for the strict `alberta-evidence-status` registry;
  the former `--step` selector is rejected because no current per-step claim
  contract exists.
- Construction-time validation for MLP-path optimizers
  (`supported_for_mlp()`): unsupported optimizers now fail at learner
  construction instead of raising `NotImplementedError` inside a jitted
  update.
- `steps/step7.py` vestigial zero-multiplied reward term removed from
  planning action scoring; `streams/gymnasium.py` VALUE mode no longer a
  silent stub; the unused `feature_discovery.replace_fraction` knob was
  removed while legacy serialized configs still load; duplicate
  candidate-imprint formula in `compositional_features.py` consolidated.
- Test hygiene: `test_integrated_hidden_partner.py` marked `slow`;
  Step 1/Step 2 replication suites now skip loudly with a registered
  `replication` marker and a terminal summary of skipped counts; missing
  upstream script trees now surface as visible skips rather than being hidden
  by `collect_ignore`.

### Added (continued)

- Added an opt-in bounded `PrototypeFeatureLifecycle` and its narrow
  `PrototypeAgent` composition. A fixed-width base is augmented with pair
  products and trained from one owner-bound behavior TD target before the
  linear OaK consumers are descriptor-routed. Builder gradients use an exact
  generation-and-full-descriptor-bound pullback; the same identity travels
  atomically with the enabled OaK subtree, so stale or forked consumers fail
  closed even when their observation cache collides numerically. Unsafe
  curation is deferred by proposal rollback, while safe routing is atomic.
  Allocation ceilings, exact resource
  declarations, strict config/state checks, and versioned checkpoints are
  public. The compatible lane deliberately excludes world models, Horde,
  replay, dreaming, IA, partner fusion, experiential memory, and GRU
  perception. This is L0 mechanism coverage only: it proves no benefit,
  promotion eligibility, WP7 completion, multi-consumer/deletion utility, or
  autonomous cumulant, subtask, or option discovery.
- Added an opt-in, stateless `OptionSearchControl` for Prototype's learned
  STOMP option models. It recomputes completion-supported differential
  semi-MDP targets and absolute Bellman residuals after each accepted backup,
  uses a fixed resource budget and stable tie order, and commits only the base
  value learner while preserving real traces, option/lifecycle state, action
  ownership, OaK counters, and RNG. The current action cache is deliberately
  not refreshed, so value effects begin only at a later extended-action
  selection boundary. This is L0 value-backup prioritization, not calibrated
  or combined primitive/option search and not evidence of control benefit.
- **Matched-current Forager campaign machinery** (the `forager_matched_*`
  benchmark modules): OCI runtime/executor qualification, a frozen
  21-candidate open-tuning protocol and resumable campaign runner, RNG-parity
  qualification, and the sealed stage (seal, sealed-evaluation schedule,
  final analysis, paired statistics). Registered two console scripts:
  `alberta-forager-matched-qualification` and
  `alberta-forager-matched-campaign`. Execution status at the time of
  writing: qualification completed
  (`outputs/forager/matched_current_qualification_2c3b214c_v1`); the
  open-tuning campaign is prepared and frozen with **zero executed cells**
  (`outputs/forager/matched_current_open_tuning_2c3b214c_v1` has empty
  `runs/` and `completions/`); the sealed stage is implemented and
  contract-tested with the `alberta-forager-matched-sealed-evaluation`
  console script, but has never been executed. A reference receipt resolver
  exists in-tree, but no operational external anchor/key ceremony or fresh
  authority receipts have been provisioned, so all outputs remain
  content-only, unendorsed, and nonpromoting
  (`promotion_authorized: false`).
- **Matched-current Forager qualification provenance v2**: the exact canonical
  UTF-8 qualification-manifest bytes and SHA-256 now flow into the execution
  plan/executor manifest, score evidence, execution closure, verification
  subject/request, resolver bindings, open and sealed campaign summaries, seal,
  and final-analysis bundle. Every replay boundary cross-checks the shared
  digest, qualification copies preserve their original bytes, legacy v1
  carriers fail closed, and typed plans replay, re-prepare, and freeze their exact
  protocol/candidate/source closure from the retained host assets. Direct construction
  now reparses the protocol, enforces the qualified runtime lock, reconstructs the whole
  executor manifest from live qualified inputs, and rejects forged candidate indexes,
  same-ID substitutions, altered capability receipts, cyclic mappings, and source drift
  after preparation. Qualification now takes a stable, descriptor-backed snapshot of the
  live Alberta source tree, revalidates all transitive staged sources and the bound absolute
  OCI executable/daemon/exact image before and after every probe, and accepts only the
  project tree that loaded the verifier. Fresh loader/protocol/plan replay runs under the
  same fixed non-root user in the exact networkless, read-only, capability-dropped OCI image,
  with actively capped output, both-case proxy credentials cleared, a deterministic
  OCI-readable content-tree mode contract normalized through held descriptor-relative
  handles, and all transitive project imports confined to the staged snapshot. The pinned
  upstream Git archive is likewise produced by one system-path-resolved, content-rebound Git
  executable under a minimal environment, with 4 KiB metadata caps and an exact-size streamed
  archive cap, an explicit built-in tar format/default umask, and no repository-selected
  archive command before its size and SHA-256 are accepted. Git identity and archive launch
  `OSError`s, timeouts, and bounded-output overflows are normalized to the public qualification
  error contract while the executable identity is rebound after every attempt. The complete
  closure is replayed both before fsync/rename and again under the held-inode post-publication
  validator.
  Post-rename validation or durability uncertainty preserves the occupied destination,
  exits with the distinct `PUBLISHED-UNCERTAIN` status, and forbids path reuse. The
  shared-source family is named for its RNG-isolated source contract rather than recurrent
  architecture. The live executor now drains stdout/stderr concurrently into separate,
  actively bounded sinks, preserves its 512 MiB/16 MiB asymmetric limits, never persists an
  overflow witness byte, and performs bounded kill/reap plus cidfile cleanup. Both runners
  assign collision-resistant container names and fall back to exact-name force removal if an
  interruption lands before the cidfile is materialized. CID-file disappearance, partial-read,
  and other read-race failures trigger exact-name cleanup first and then surface as a public
  qualification error carrying the confirmed cleanup state. Completed nonzero clients are
  cleaned too; an unsuccessful removal is accepted only after a separate actively bounded
  daemon query proves the exact random name absent. Absence-query launch, timeout, or overflow
  failures, together with injected runtime-inspector, OCI-probe, and fresh-replay runner
  `OSError`, timeout, and overflow failures, are normalized to their public fail-closed
  qualification errors instead of leaking private process exceptions. Both bounded runners
  tolerate exited-child races, always attempt bounded reaping, independently close their
  selector and both pipes, and normalize kill, reap, and close failures to public errors.
  Malformed nested caches retain layer-specific error contracts. The added provenance remains
  content identity—not endorsement, promotion authority, or a performance claim.
- **Matched-current claim-boundary correction**: candidate-universe schema v2 now states that
  the preregistered procedure identifies only its three named Alberta-versus-selected-external
  contrasts; it does not identify a best member of the 23-candidate registered panel or a
  winner among the six held-out arms. Statistics-result schema v4 carries the same detached
  interpretation boundary: bootstrap superiority and Holm rejection fields are mechanical
  frozen-protocol flags, the bootstrap endpoint is not a population confidence interval, the
  sign-flip calculation has no asserted sign-exchangeability model, and neither result grants
  confirmatory, ranking, or SOTA authority. Final-analysis manifest schema v3 replaces its
  obsolete v1-wording disclaimer with the v2-native contrast-specific/no-panel-ranking
  boundary. Historical artifacts remain unchanged.
- **Nonpromoting causal q-grid diagnostic v2 hardening**: the unchanged public-seed-0,
  epsilon-0.05, q=.50/.75/.90, 10,000-transition diagnostic now writes receipt schema
  `alberta.forager_causal_grid_divergence_probe.v2` under the default
  `causal_q_grid_divergence_seed0_v2` root. Its portable canonical qualification root is bound
  to exact manifest-relative source, inventory, archive, snapshot-descriptor, configuration,
  and capability-receipt paths and schemas; the canonical manifest sidecar joins the 14-file
  read-only mirror, and the receipt carries the replay-critical paths, schemas, archive size,
  and digests. OCI execution clears both-case proxy variables, and cleanup accepts a failed
  force removal only after a separately bounded exact-name absence proof. The harness and tests
  also normalize runner, kill, bounded-reap, and resource-close failures; inability to confirm
  reaping remains fail-closed while exact-name container cleanup is still required. Every final
  receipt descriptor is closed independently, and a close failure after rename receives the
  distinct published-uncertain status. The live probe remains unexecuted, permanently
  open-development, and nonpromoting; no scientific q semantics, evidence gate, or frozen
  protocol changed. A fresh qualification must still be published and its canonical root plus
  all qualification-derived hashes repinned as one reviewed set before this v2 diagnostic is
  run.
- **Beat-SOTA screening lane** (`benchmarks/ipmnist_screening.py`, 51 tests):
  30 registered mechanism-combination arms on a validated 60-task proxy (an
  exact bit-prefix of the 200-task protocol; control parity pinned bitwise),
  including UPGD×IDBD, UPGD×Autostep, UPGD+CBP, AdamW+CBP, UPGD+L2-Init,
  UPGD+EMA-input-norm, a UPGD-W hyperparameter star, weight clipping
  (Elsayed et al. RLC 2024, ±κ/√fan_in, 4 configs), per-layer gate
  normalization, FADE-style meta-learned per-parameter head decay
  (arXiv 2604.27063, sign conventions derived and unit-pinned), and a
  SwiftTD-stabilized UPGD×IDBD (overshoot bound + persistent step-size
  decay). Includes plan/run/validate-proxy/merge CLI, idempotent shard
  workers, and autonomous endgame pipelines with full-protocol pool64
  confirmation and a strict paired BEATS/TIES/BELOW verdict
  (`outputs/ipmnist_screening/`). Screening results are permanently
  nonpromoting.
- **Label-permuted EMNIST protocol-exact lane**
  (`benchmarks/upgd_label_emnist.py`, 21 tests): EMNIST balanced 47-class,
  labels permuted every 2,500 steps, 400 tasks, pinned to the audited
  upstream commit; first artifact (`outputs/upgd_label_emnist/results.v1.json`,
  3 seeds) reproduces the published qualitative separation — UPGD-W online
  accuracy rises across the 400 tasks (first-quarter mean 0.5616 →
  last-quarter 0.7284; whole-run mean 0.67151 vs the ~0.74 figure read-off,
  gap flagged) while AdamW collapses (whole-run mean 0.20081 vs the ~0.35
  read-off, gap flagged). Descriptive only; both reproduction gaps are
  recorded in the artifact.
- **Slowly-Changing Regression v2 sharded protocol** (plan/run-shard/merge/
  validate with immutable self-issued plans and exact-replay validation;
  300 shards = 3 methods × 100 seeds).
- Added `run_ipmnist(noise_mode="pool", noise_pool_steps=N)`, a screening-only
  fast mode for the UPGD Input-permuted MNIST lane that replaces the dominant
  per-step cost -- generating a fresh 282,160-element N(0, sigma^2)
  perturbation every step, ~85-90% of single-core UPGD-W step time -- with
  random contiguous slices of a per-task regenerated noise pool (OpenAI-ES
  style shared noise table). Measured single-core UPGD-W throughput rises
  from 133 steps/s (exact) to 559 steps/s (pool 64) and 951 steps/s
  (pool 16) at the protocol shape. Per-step noise marginals stay exactly
  N(0, sigma^2) but values are reused across steps, so `IPMNISTRunResult`
  now records `noise_mode` and `partial_payload` refuses pool-mode results:
  they can never enter the v2/v3 artifact lifecycle. The default
  `noise_mode="step"` inner loop is verified bitwise identical to the
  pre-change runner on fixed-seed 2x5000-step publication-shape runs for
  both learners, and existing shards/artifacts are unaffected. Full-protocol
  200-task pool-validation runs are recorded under
  `outputs/upgd_ipmnist_runner_opt/`.
- Added the strict, namespaced continual-IA v2 development-only contract and
  `CONTINUAL_IA_V2_RUNBOOK.md`. It freezes treatment `recommendation_p075`,
  exact acceptance probability 0.75, the old gates with seed start 60, and
  seeds 60–89 under immutable source/runtime-byte-bound
  plan/reservation/one-seed-shard/merge files. Persistent reservations are
  published before execution, shards retain complete primitive causal traces
  and exact replay, and merge replays once before reconstructing all paired
  intervals, budgets, credit, identity, and gates. The self-issued schema has
  no external pre-run chronology and can never set `internally_accepted=true`.
  The contract is unissued: no plan, reservation, v2 seed execution, shard, or
  artifact exists.
- Added the active, namespaced UPGD Input-permuted MNIST v3 future-execution
  contract and `UPGD_IPMNIST_V3_RUNBOOK.md`. It requires one immutable pre-run
  plan, exact selected configuration/hyperparameters and closed deviations,
  exactly 20 operator-reserved fresh seeds, data/runtime/static-import-closure/command
  bindings, one learner/seed per shard, atomic no-overwrite publication, and
  exact Cartesian merge with raw shard hashes and recomputed summaries. No v3
  plan has been issued, no v3 result exists, and no fresh v3 seed has been
  consumed. The self-issued, externally unattested schema is permanently
  nonpromoting; all sealed completed-diagnostic records remain unchanged.
- Added a fail-closed verifier-issued tuning-envelope boundary for a future
  immutable Forager OCI evaluation adapter. It binds the tuning report, raw
  evidence, source tree/archive, read-only content-addressed source mount,
  runtime profile, and environment RNG schedule to an externally authenticated
  authority. The host/snapshot runner cannot invoke the adapter or promote its
  own tuning report, and candidate OCI metadata is not attestation.
- Added a strict, permanently nonpromoting validator and reconciliation
  receipt for the completed 10-seed UPGD Input-permuted MNIST diagnostic. The
  original finalizer artifact remains byte-exact; the canonical reconciled
  artifact repairs its note and portable post-hoc cache binding without
  repairing the absent execution-time provenance or 10-vs-20-seed deviation.
- Added immutable corrections for the four-seed Forager development records:
  an RTU capture-digest addendum and a reconciled DQN comparison receipt using
  the public importer. The descriptive RTU-minus-DQN mean is `+0.3309` on four
  consumed seeds, but comparator timing, runtime, representation, resources,
  replay, and update work are unmatched, so the receipt permanently forbids
  inferential, causal, speed, SOTA, and Alberta Plan claims.
- Replaced the slowly-changing-regression v1 writer and retrospectively
  calibrated qualitative gate with a strict namespaced v2 development
  protocol. It adds a selected ReLU/Kaiming/true-MSE ordinary-BP path,
  distinguishes Alberta-local CBP/UPGD extensions, requires immutable
  one-method/seed shards under a pre-run source/run/runtime-bound plan, checks
  exact paired coverage and shared environments, and reconstructs descriptive
  results from size/SHA-bound shards. The self-recorded envelope is not
  external attestation, so every valid v2 artifact remains permanently
  nonpromoting.

### Changed

- The live evidence registry currently reports all five registered claims
  `invalid` (overall `invalid`, exit `2`): the post-0.27.0 source and
  documentation waves edited registered source files after their artifacts
  were pinned. This is the fail-closed design working as intended. The pinned
  `outputs/` artifacts themselves are unchanged; renewing a claim requires
  rerunning its frozen protocol to a new artifact path and schema version
  with untouched preregistered seeds.

## [0.27.0] - 2026-07-31

### Added

- Added an initial publication-shaped Nature-2024 slowly-changing-regression runner
  (Dohare et al.): the `m + 1`-bit stream with `f` flipping bits every
  `T` examples, the 100-LTU random sign-weight target network with
  `(m + 1) * beta - S_i` thresholds, and a fully vmapped 100-run runner over
  plain-SGD `MLPLearner`, `CBPMLPLearner`, and `UPGDLearner` with 40k-example
  bins over 3M examples. This initial path was not protocol exact: learner
  initialization, MSE scaling, RNG/numeric semantics, target-bias encoding,
  comparator identity, provenance, and post-hoc thresholds differed or were
  incomplete. No full artifact was produced; the Unreleased v2 contract
  replaces its artifact-writing path.

- Added a strict, reconstructing continual-evaluation report with
  predict-before-update traces, evaluator-only regime metadata, a candidate
  plus at least two exactly budget-matched baselines, continual-transfer and
  recovery metrics, p50/p95/p99 latency, delayed/dropped observations, memory
  and optional energy measurements, graded safety/near-miss accounting, and
  required component/plasticity diagnostics. Its bounded scalar streaming
  executor rejects prediction/probe/source-state mutation, noncanonical or
  nondeterministic checkpoint state, configuration drift, deadline and host
  memory-budget violations, and inconsistent exposure metadata. Safety and
  metric applicability distinguish unavailable measurements from measured
  zeros. A strict evaluator-bound artifact now hashes the full evaluator
  identity (including stream/probe and learner configuration) and metric core,
  cross-validates their shared protocol/budget/condition semantics, and uses
  atomic report/checkpoint writes. Reports remain explicitly `not-assessed`;
  constructing one is not a scientific pass.
- Added a strict v2 continuing-control evaluator and hardened `PrototypeAgent`
  adapter. Candidate and baselines receive independent functional environment
  states; exact observation/action/decision-ID ownership enforces
  predict-before-outcome ordering while regime identity stays evaluator-only.
  The reconstructing report pins metric direction, exposure rows, recovery and
  stability references, then derives lifetime/prequential and per-regime
  return, post-change adaptation AUC, sustained recovery, final held-out action
  scores, forgetting, backward/forward transfer, stability, and worst-window
  return with explicit applicability. Canonical config/source digests and
  atomic checkpoints fail closed on identity or metric tampering. This is
  development-only infrastructure; no completed Prototype comparison or
  scientific claim is attached.
- Added a strict development-only paired continual-control campaign. Explicit
  seed wrappers bind each environment and learner configuration; cross-seed
  invariants require identical protocol, budget, probes, condition roles and
  normalized identities. Every raw report is retained, unavailable pairs stay
  unavailable, and direction-normalized candidate-minus-baseline differences
  receive deterministic stratified paired-bootstrap intervals from a frozen
  counter-hash RNG. Config/source/report/comparison hashes and atomic I/O make
  the artifact reconstructing. Campaigns remain `not-assessed`; no threshold,
  completed Prototype campaign, or scientific claim is supplied.
- Added a strict development-only privileged continual-control reference suite
  outside the ordinary matched-condition list. It runs one independently
  initialized learner per evaluator regime identity and retains it on
  recurrence, a stationary-multitask learner trained from an exactly budgeted
  frozen extra stream, and an exact frozen counterfactual action-outcome upper
  reference. Canonical reports and checkpoints bind source/config hashes,
  decision ownership, extra-data and callback counts, resources, unavailable
  recurrence states, and privilege/comparability disclosures. These are
  descriptive `not-assessed` context bounds, not baselines or evidence.
- Added the WP4 shallow world-model reference: an action-indexed affine
  regularized-FTL/ridge learner over grounded next-observation, reward, and
  continuation targets. It predicts before recursively updating fixed
  Gram/cross sufficient statistics, solves only the selected action block,
  validates counts, normal equations, positive-semidefinite statistics and
  numeric bounds fail-closed, has strict RNG-free checkpoints and exact
  allocation accounting, and exposes a pure one-step supplied-linear-value
  action scorer. This is L0 mechanism coverage, not a paper reproduction,
  calibrated model, MPC result, or control-benefit claim.
- Added an isolated bounded recurrent latent world-model ensemble with one
  trainable GRU and grounded mean/heteroscedastic-variance heads per bootstrap
  member. Exact start/decision/transition ownership, predict-before-update NLL,
  final-target/reset-cache boundaries, one recurrent advance per accepted
  event, stopped-target representation gradients, atomic failure handling,
  fixed resource accounting, JIT/scan, and digest-bound checkpoint continuation
  have L0 tests. Added it as an opt-in fourth mutually exclusive Prototype
  model lane with exact dispatched-decision caching, transactional causal
  signal state, real-NLL-only builder/mixer/candidate-audit routing, and whole-transition
  rollback on recurrent rejection. It is not a replay, planning, calibration,
  or efficacy result.
- Added a bounded `PartnerPolicyFusion` L0 core with fixed typed
  message batches, five explicit routes, discrete score-based blending, a
  caller-owned hard action mask, one exact decision-bound feedback record, and
  contextual logistic reliability updates from realized assistance plus
  observed safety. Stale, duplicate, and misattributed feedback is an atomic
  no-op; cold-start acceptance is labeled uncalibrated development exploration.
  Checkpoint, resource, eager/JIT, and scan contracts are tested. Added an
  opt-in `PrototypeAgent` composition that binds the full lifecycle ID, applies
  prior feedback before the next fusion decision, derives real OaK base and
  keyboard proposals, rewrites the exact primitive credit owner, synchronizes
  the recurrent action cache, and rolls back the whole transition on unsafe
  base dispatch or corrupt post-state. It makes no calibration, benefit, or
  WP8 completion claim.
- Added a strict development-only `ExperientialMemory` transfer evaluator. It
  runs a fixed evaluator-owned recurring A/B/A trace from an immutable empty
  snapshot, preserves exact query-before-write ownership, and records raw
  neighbor/gate predictions, no-memory fallbacks, harmful recall, abstention
  causes, eviction provenance, first/return descriptions, loophole checks, and
  exact resources. Config, protocol, source, snapshot, canonical report, and
  checkpoint hashes reconstruct fail closed with eager/compiled parity. It has
  no threshold or transfer, retention, efficacy, promotion, WP8, or SOTA claim.
- Added a stateless `ExperientialMemoryPolicy` and opt-in `PrototypeAgent`
  composition. Retrieved action vectors are categorical score mass—not rounded
  action identifiers—and selection is the lowest-index safe positive-mass
  argmax. Prototype queries the next decision before writing a grounded
  one-hot of the primitive action that actually executed plus bootstrap
  representation and reward, then composes memory dispatch before partner
  fusion. Full lifecycle IDs, no-memory state-shape preservation, whole-event
  rollback, checkpoints, curation, eager/JIT/scan behavior, and exact resources
  are tested. The resource declaration exposes two deterministic pre-state
  queries and zero RNG; no transfer or control benefit is claimed.
- Added a strict option-keyboard policy proposal and primitive-dispatch
  boundary. It computes the deterministic chord argmax from current option
  values, verifies exact decision-observation ownership and the complete fixed
  STOMP/OaK state, applies a caller-owned hard safety mask, and rewrites either
  the base primitive-action cache or the active option's intra-option cache so
  subsequent TD credit follows the action actually executed. Unsafe proposals
  use an independently safe base; unsafe bases and corrupt inputs are exact
  fail-closed no-ops. RNG/state shape and package exports are preserved. This
  is L0 ownership/safety integration, not chord-learning or control-benefit
  evidence.
- Added a strict development-only frozen world-model snapshot evaluator. It
  retains raw ensemble-member and mean grounded predictions/targets, derives
  reconstructable disagreement/error bins, correlations, coverage-risk, and
  ID/OOD plus state/action-region summaries, and keeps the residual EMA
  explicitly separate and non-probabilistic. Exact open-loop diagnostics are
  optional and bounded; configs, probes, sources, snapshots, traces, summaries,
  resources, checkpoints, and no-overwrite reports are hash-bound. It applies
  no threshold and makes no calibration or scientific claim.
- Extended that instrumentation with a separate recurrent development adapter.
  It binds a frozen initial snapshot, scores each exact cached distribution
  before one update of an isolated copy, and retains reconstructable member
  means, heteroscedastic variances, NLLs, warm-up applicability, ID/OOD and
  evaluator-owned state/action regions, final isolated-state counters, sources,
  resources, canonical reports, and source-bound snapshot checkpoints. The
  supplied snapshot remains unchanged. The Gaussian objective is not evidence
  of calibrated likelihood, and no threshold, efficacy, or promotion claim is
  made.
- Added a strict recurrent-only retention companion that requires exact ordered
  case reuse after an evaluator-owned intervening context and recurrent resets,
  then reconstructs phase, recurrence-entry, ID/OOD, and within-occurrence NLL
  summaries from the source-bound prequential trace. It keeps the supplied
  snapshot unchanged and remains development-only `not-assessed`; it is not a
  retention, calibration, Prototype integration, or scientific claim.
- Added a strict development-only average-reward actor/critic A/B/A companion.
  Regime identities, reward tables, preferred actions, and value targets remain
  evaluator-only while exact cached target/epsilon-mixture behavior policies,
  critic error, actor margin, churn, return/recovery, plasticity, action
  activity, resources, source hashes, and checkpoints remain reconstructable.
  The core now learns from the cached decision, commits the update, and samples
  its successor from committed parameters. Raw `pi / b` is diagnostic and the
  logged `(1 - epsilon) pi / b` term is explicitly the behavior-score chain
  rule, not off-policy correction. The probe is one-seed, `not-assessed`, and
  supplies no retention, efficacy, Prototype, promotion, or SOTA claim.
- Added an isolated bounded continuous average-reward actor/critic. It uses a
  direct affine-`tanh` diagonal Gaussian without latent clipping or endpoint
  adjustment, retains exact pre-`tanh` decision ownership and stable
  transformed target/behavior log densities, and applies the analytically
  cancelling latent-density ratio to the complete actor eligibility trace.
  Actor, critic, traces, LMS states, differential reward baseline, typed RNG,
  and saturating counters are separate; one successor is sampled only after an
  atomic finite commit. Strict config/checkpoint/resources, density/score,
  saturation, rollback, causal-cache, eager/JIT/scan, and public-export tests
  pass. This is L0 mechanism coverage with no state-distribution correction,
  off-policy convergence, retention, efficacy, or SOTA claim.
- Added a strict continuous actor/critic recurrence evaluator over one fixed
  12-event, one-action-dimensional A/B/A life from an immutable source-bound
  snapshot. It reconstructs cached latent/action ownership, transformed
  target/behavior densities, the exact latent ratio, rewards, same-state
  gauge-centered critic error, actor error/churn, plasticity/activity,
  successor ownership, counters, resources, final state, and exact live replay.
  Reports and checkpoints are hash-bound, no-overwrite, and subject to absolute
  byte/scalar ceilings. A symmetric eight-float32-ULP allowance is isolated to
  transformed diagnostic-density reconstruction; the policy-defining latent
  ratio remains bit-exact and larger tampering fails. The result is
  development-only `not-assessed`, with no retention, efficacy, convergence,
  calibration, promotion, or SOTA claim.
- Added an isolated nonlinear discrete off-policy actor/critic. A shared tanh
  trunk, categorical actor, and scalar critic learn only from the exact cached
  executed-action receipt, including target and caller-declared behavior log
  probabilities/revisions and exact action identity. Clipped per-decision
  action ratios advance separate actor/critic head and trunk traces; actor,
  critic, and trunk have fixed independent plastic/frozen policies. Typed
  Threefry sampling, exact fail-stop clocks, atomic rollback, strict
  config/checkpoint construction, complete persistent-state byte accounting,
  a hand-derived two-step trace, and eager/JIT/scan parity are tested. This is
  L0 `not_assessed` discounted scalar-V machinery with no state-visitation
  correction, average-reward baseline, learned utility policy, authenticated
  external behavior owner, convergence, retention, efficacy, artifact,
  promotion, or SOTA claim.
- Added an isolated nonlinear discrete average-reward actor/critic with
  separate one-hidden-layer `tanh` actor and critic networks, head/trunk
  traces, momentum, fixed component plasticity policies, bounded utility
  telemetry, and a learned reward-rate baseline. Exact target and caller-owned
  behavior distributions, revisions, action identity, and a fixed owner digest
  flow through a pure proposal and a commit that recomputes and bit-validates
  the candidate before taking the sole successor draw. Ordinary
  epsilon-mixture behavior-score and clipped per-decision target-importance
  modes, fail-stop clocks, raw numeric bounds, softmax underflow, rollback,
  checkpoints, resources, and eager/JIT/scan parity are covered. This is L0
  `not_assessed` machinery: importance is action-only, utility has no learned
  plasticity authority, and no convergence, retention, matched control,
  safety, promotion, or SOTA claim follows.
- Added a tiny matched-stream world-model retention diagnostic comparing the
  shallow reference, plain bootstrap ensemble, and atomic model-only rehearsal
  over uninterrupted A/B/A lives with interleaved noisy-TV outcomes. Raw
  grounded errors, adaptation/recurrence descriptions, ensemble disagreement,
  typed signals, replay composition, and logical resource/event accounting
  reconstruct from fixed streams. The two-seed, 18-step protocol is
  resource-unmatched and permanently `not-assessed`; it supports no retention,
  superiority, control, or SOTA claim.
- Added identity, fixed-trace, and online trainable gated state builders under
  one causal fixed-budget contract, including checkpoint parity and a
  development-only partially observable diagnostic.
- Added a multi-objective candidate-update safety audit under the historical
  `assess_gradient_joy` API. It uses
  caller-supplied objective, retention, and safety probes under a required
  `probe_independence_attested` contract plus a trust bound, fails closed on
  unavailable, unattested, or invalid evidence, and requires explicit
  availability for all eight separately reported learning-value channels. The
  `LearningValue.delight` evidence channel is valid only when its bits exactly
  equal the finite float32 advantage-surprisal product; it remains evidence
  rather than a candidate-update verdict.
  Raw-candidate factors produce a tentative soft-weighted update; both stages
  must satisfy objective, retention, and safety magnitude gates. Numeric
  controls must survive as finite normal float32 values, and cosine alignment
  is range-clipped with an explicit float32 endpoint policy. Paper-specific
  Delightful Policy Gradient remains a distinct actor-sample experiment.
  Added `apply_gradient_joy_update` as the atomic parameter-application boundary:
  it reassesses internally, requires exact parameter/update PyTree contracts,
  re-audits the effective stored delta after dtype cast and parameter addition,
  applies atomically, and reports the formed candidate and effective-delta
  assessments separately from a parameter change actually being `applied`.
  The Prototype's opt-in learned-state lane is now a development consumer: it
  binds probe evidence to the dispatched decision and commits only an accepted
  effective delta into the advanced recurrent builder state. No realized-
  benefit result is claimed.
- Added the paper-defined `KondoGate` forward/sparse-gather boundary. Delight
  is derived internally as advantage times selected-action surprisal, and
  its historical `sparks_joy` view marks forward admission intent for the
  caller. The forward-only gate does not run autodiff; only an actor consumer
  establishes actual backward execution.
  Finite-temperature
  Bernoulli-price and deterministic fixed-rate top-k modes have typed RNG,
  bounded accounting, strict checkpoints, caller-declared force preservation,
  and an explicit flag requiring caller-managed full-shape fallback on
  overflow. When configured capacity is below batch size, the fixed-capacity
  gather gives downstream autodiff a genuinely smaller input; tests inspect
  that backward JAXPR rather than equating a masked full-batch loss with saved work.
  This is L0 mechanism coverage with no integrated consumer, measured compute
  saving, DG reproduction, or learning/safety claim.
- Added `KondoSparseActor`, the first real nonlinear categorical actor consumer
  of that boundary. It binds exact action identity, policy revision, and
  behavior-log-probability bits, gathers the fixed-capacity actor batch first,
  and only then invokes `jax.value_and_grad`. Tests witness a capacity-3
  backward JAXPR instead of the full batch of 6. Forced guardrail rows and
  Bernoulli overflow take an explicit full-shape fallback without dropping a
  selected sample. Returns and baseline predictions enter the actor only via
  detached advantage; critic and safety features stay outside its loss, while
  all protected learners remain full-batch and ungated. State, resources,
  rollback, and source-bound checkpoint integrity are tested. Host
  orchestration remains required, and there is no demonstrated compute saving,
  closed-loop on-policy or behavior-policy-corrected actor-critic integration,
  efficacy, safety, promotion, or L3 claim.
- Added a strict nonpromoting four-arm `KondoSparseActor` development evaluator.
  Ordinary-full, capacity-matched uniform-sparse, Kondo-top-k, and diagnostic-
  overflow arms share one immutable parameter snapshot and source trace, one
  update opportunity per external batch, and explicit selected-sample and
  compiled-backward shape/invocation accounting. An update-free timing section
  compiles, warms, blocks, and evaluator-interleaves the fixed kernels before
  retaining raw `perf_counter_ns` samples and independently reconstructed
  nearest-rank p50/p95. Host screen/gathering, accelerator memory, energy, and
  end-to-end latency are excluded and disclosed. Wall-clock bytes are outside
  deterministic replay; config, source/runtime, trace, snapshot, and causal
  checkpoint prefixes fail closed. All statuses are `not_assessed`, with no
  speedup, efficacy, safety, output-write, policy, promotion, or L3 claim.
- Added a strict nonpromoting Kondo actor/critic replay lane. Ordinary-full,
  capacity-matched uniform, paper top-k Kondo, and fixed-capacity top-k plus a
  minimum random reserve share one evaluator-fixed A1/B/A2 contextual-gambling
  trace and exactly one actor/protected update opportunity per source batch.
  Baseline, critic, representation, world-model, and safety/guardrail learning
  remains full-batch and independently bit-identical across arms, including
  rare failures. Current-policy delight, executed actor-backward inclusion
  masks, gather shapes, logical proxies, descriptive recurrence readouts, and causal
  checkpoint replay are retained. Because the fixed actions have no source
  behavior policy and no importance correction, actor updates are explicitly
  off-policy surrogates with no policy-gradient/DG-efficacy, compute, safety,
  output, evidence, or promotion claim; every result is `not_assessed`.
- Added a strict closed-loop on-policy Kondo development evaluator. Its four
  arms sample from their own immutable actor revisions using evaluator-owned
  typed Threefry common uniforms for exogenous randomness only; actions and
  trajectories are never shared or assumed equal. Updates occur once at each
  batch boundary, all five protected learners remain full-batch, and forced
  rare failures reach both actor and guardrail learning. Exact action,
  behavior-probability, revision, causal-chain, checkpoint, source/runtime, and
  replay contracts are tested. It writes nothing and remains `not_assessed`,
  with no efficacy, compute, safety, evidence, or promotion claim.
- Added a bounded `LearningValueRouter` for the eight separately typed
  learning-value channels. Each channel has explicit producer/object/units/
  domain metadata, independent validation, and causal pre-update Welford
  normalization. Six exact-mask routes serve the paper-DG actor, exploration,
  model memory/replay, adaptation/change, safety, and the complete evidence
  bundle for the separate candidate-update audit; there is no default sum.
  Invalid unrelated inputs cannot suppress valid safety learning, delight must
  exactly equal its float32 advantage-surprisal product, and the router performs
  neither the candidate audit nor Kondo selection. Fixed resources, counter-capacity behavior,
  strict checkpoints, and eager/JIT/scan parity are mechanism-tested. An
  opt-in owner-bound Prototype integration now routes exactly once after causal
  typed signals on an accepted real transition, keeps producer availability
  independent of representation-candidate validity, and supplies only raw
  routed values to the candidate audit. It adds a v19 enabled checkpoint while
  leaving disabled config and state PyTrees unchanged; no calibration or
  consumer-benefit claim is made.
- Added an isolated continuing categorical actor-critic for the separate
  paper-specific DG policy-gradient experiment. Ordinary and paper-specific DG
  modes share the actor, differential critic, reward-rate baseline,
  typed RNG, and action sampler; the actor trace is fixed to zero, and the
  detached paper-defined delight coefficient never gates critic/baseline
  or explicit safety/model/representation routes. Exact on-policy records, atomic failure,
  JIT/scan/checkpoint parity, gate strata, effective sample size, and logical
  resource accounting are contract-tested. A strict development runner now
  compares both modes under paired random-number schedules on contextual
  heteroskedastic gambling and uninterrupted six-state RiverSwim A/B/A, with a
  validator that reconstructs every declared trace and derived diagnostic.
  The seeds/thresholds are nonpromoting, trajectories may diverge after policy
  divergence, and no policy-quality, safety, compute-saving, or paper-
  reproduction claim follows.
- Added a predict-before-update typed learning-signal producer that keeps
  ensemble epistemic disagreement, aleatoric uncertainty, normalized residual,
  learning progress, and sustained change probability separate, with warm-up
  flags, noisy-TV and change diagnostics, bounded counters, and fixed resource
  accounting. Causal ordering is part of the caller contract.
- Added a fixed-size bootstrap world-model ensemble with distinct initialization,
  persistent mask RNG/counters, typed warm-up signals, a causal representation
  gradient, exact resource accounting, and full checkpoint parity. It is
  integrated into Prototype as a mutually exclusive development lane; its
  residual-variance proxy is not externally calibrated aleatoric uncertainty,
  and ensemble dreaming remains disabled.
- Added `DualReplayMemory`, a fixed-capacity recency-FIFO plus long-term replay
  substrate with reservoir or configured surprise/coverage/progress retention,
  explicit aleatoric control, policy/value and representation provenance,
  fixed-quota stale-aware sampling, exact resource accounting, and strict
  deterministic checkpoints. Added `ModelReplayRehearsal` to atomically compose
  the real ensemble update, signal-aware record, fixed-quota sample, and
  model-member-only rehearsal, with isolated replay RNG/masks/counters and
  strict composition checkpoints/resources. Prototype exposes it as a mutually
  exclusive model lane and sends only the commit-gated real gradient to builder
  learning and the separate candidate-update audit. Replay never trains the
  actor, critic, builder, or causal calibrator; no retention or control benefit
  is claimed.
- Added a development-only component-retention evaluator with frozen
  non-learning snapshots; seven separately applicable representation, model,
  reward/termination, value, and actor channels; mutation checks; bounded
  record/state budgets; and reconstructing reports/checkpoint resume. It is not
  a longitudinal or scientific retention result.
- Added fixed-capacity experiential memory with typed provenance/version
  metadata, query-before-write ordering, similarity/reliability/staleness
  retrieval, deterministic utility/recency eviction, controlled stale- and
  wrong-version abstention checks, exact byte accounting, and checkpoint/scan
  parity. No transfer benefit is claimed.
- Added source-profiled canonical UPGD implementations for the paper,
  official README, and official experiment equations, plus a numerically safe
  extended default. Regression tests pin their documented normalization and
  update-multiplier differences instead of silently blending variants.
- Preserved the immutable recurring- and scale-pair evidence artifacts after
  multiple registered implementation, artifact-builder, and CLI sources
  evolved. The live evidence registry now reports both claims as
  source-invalidated; consumed-seed
  compatibility reruns cannot promote, and any renewed claim needs a new
  path/schema plus untouched preregistered seeds. Their generation CLIs now
  reject the reserved canonical paths even if those files are missing, reject
  every existing destination before running, and publish new-path
  reproductions atomically without overwriting a concurrent writer.
- Added the historically accepted immutable
  `alberta.scale_robust_pair_feature_evidence.v2` package comparison on 30
  exact namespace-derived fresh seeds. The strict artifact records primitive
  phase rows, structural retention, paired intervals, fixed resources, source
  hashes, and scientific digest
  `c2fee922c04a59fe26b4b8c9cfa77ddd9198cfa2bc923f54fec14b649bd3bb2c`.
  Its scope remains narrow: visible context, an exhaustive finite pair archive,
  one fixed learner initialization, and a primary-versus-legacy comparison
  that changes scale normalization and ObGD while adding 464 bytes.
- Added L0 substrates for the next integrated kernel: an uncued recurring
  scripted-partner stream, online partner-action prediction with an input-loss
  gradient, a bounded joint-outcome model with external partner-belief
  marginalization, and fail-closed atomic feature-bank routing by descriptor
  identity. These do not yet constitute learning-partner coadaptation or L3.
- Added a bounded L0 hidden-partner kernel that composes online gated state,
  pair-feature discovery, partner prediction, all-cell joint-world planning,
  atomic descriptor routing, and explicit-action differential SARSA in one
  causal update. Shape-matched ablations isolate state learning, recurrent
  memory, feature deployment, survivor carry, active-utility retention, and
  planning. Evicted downstream columns have no dormant identity archive, and
  full-life discovery/control benefit remains development work.
- Added `PrototypeTransition`, full-Prototype-state checkpoint helpers, and
  public Prototype exports. This effective-outcome/continuation path carries
  environment
  continuation through real world-model, primitive/option control, the IA
  exo-cortex, scan, and accepted dream updates, while optional per-GVF
  continuation reaches Horde without erasing each demon's declared horizon.
  The authoritative contract now also records the raw observation, exact
  dispatched primitive action, lifecycle/generation decision token,
  terminated/truncated flags, final/bootstrap observation, and post-reset
  decision observation. Runtime-invalid transitions are transactional no-ops;
  positive-discount truncation bootstraps on the final state while resetting
  episode-local recurrence and option execution before selecting on the reset
  state. Prototype checkpoints now use the v3 rehearsal-isolation schema. A v2
  ensemble checkpoint preserves every learned member, signal statistic, and
  real bootstrap stream while deterministically initializing only the new
  replay key/mask/counters; ambiguous v1 lifecycle restoration still requires
  an explicit provenance trust override.
- Integrated the common `StateBuilder` protocol into `PrototypeAgent` with
  identity, fixed-trace, and online-gated modes, exact recurrence/event counts,
  cached-action semantics, scan parity, and config-bound checkpoints. The
  online-gated learner now rejects non-finite/overflowing gradients atomically;
  a successor opt-in Prototype mixer combines its real grounded-model gradient
  with the current base-Q or intra-option control semi-gradient. It records
  source norms, weights, clipping, cosine/conflict, and failures, excludes
  delayed option-start and replay gradients, and sends the exact mixed
  candidate to both builder learning and the candidate-update audit boundary. GVF,
  inverse, feature-utility, causal-deletion, empirical-balancing, and matched
  Forager evidence remain absent, so this is not a learned-state efficacy claim.
- Added an opt-in `sample_one_hot` Prototype dream-observation mode for wholly
  one-hot control features. It projects finite model outputs to categorical
  mass, samples on an RNG stream isolated from legacy anchor/action choices,
  and vetoes malformed, non-finite, or zero-mass projections without changing
  learner state. The legacy expectation-valued path and serialized config stay
  unchanged by default. An eight-seed consumed-development RiverSwim diagnostic
  measured `0.2475` versus `0.1930` mean lifetime reward (paired `+0.0545`,
  6/8 seeds positive); it is explicitly nonpromoting and carries no held-out
  efficacy claim.
- Added **SwiftTD** (`core/swift_td.py`) — Javed, Sharifnassab & Sutton
  (RLC 2024) step-size optimization with an overshoot bound and step-size
  decay, float32-exact against the authors' C++ reference. Follows the
  `TDOptimizer` interface, so it drives `TDLinearLearner` and the TD learning
  loops.
- Added the **stacked linear Horde** (`core/stacked_horde.py`): the GVF demon
  axis as one batched array axis with exact TD(λ) semantics, per-decision
  importance sampling, NaN cumulant masking, and a `nexting_spec` helper.
  Measured on CPU: 1,024 demons × 2,000 steps in ~0.2 s steady state
  (~0.3 s compile) versus ~140 s run + ~144 s compile for the loop-unrolled
  multi-head path, and 65,536 demons at ~4.0e7 demon-updates/s.
- Added the **Alberta Gauntlet and lifetime streams plus certification
  suites** (`streams/gauntlet.py`, `tests/test_gauntlet_certification.py`,
  `tests/test_gauntlet_discovery.py`, `tests/test_lifetime_demonstration.py`,
  `tests/test_lifetime_longevity.py`): tracking/relevance/recovery/
  scale-robustness scorecards (P1–P6), the exhaustive pair-discovery rung, the
  64k-step single-life demonstration, and the reproducible 1M-step × 8-seed
  longevity extension (zero non-finite steps over 8M updates, late-life
  savings 10–18x with no erosion, recurrence re-entry error 0.19x first
  exposure).
- Added **closed-loop control gates** (`streams/closed_loop.py`,
  `tests/test_control_learning_gates.py`): the 2-state switching MDP and
  RiverSwim with analytic optima, plus SARSA/DifferentialSARSA reward-rise
  gates against random baselines.
- Added **multi-agent streams and simulations** (`streams/opponent.py`,
  `streams/matrix_game.py`, `streams/recurring_multiagent.py`,
  `tests/test_multiagent_sim.py`): learning-opponent and adversarial-pursuit
  streams (endogenous non-stationarity; a frozen predictor is driven to 1771
  MSE while continual learners hold ≤0.12), the recurring convention game
  (instant convention recall on rule recurrence with context), and the frozen
  two-agent world behind the promoted coadaptation claim.
- Added **context inference** (`core/context_inference.py`,
  `tests/test_context_inference.py`): a bounded bank of per-(state, action)
  reward tables inferring the active hidden regime and gating control
  features by inferred slot. Development evidence: +0.519 mean paired gap
  over the no-context ablation on the tested hidden two-rule life.
- Added **TD/GVF-target discovery** (`tests/test_td_target_discovery.py`):
  pair-feature discovery against the controller's differential-SARSA target
  `r − r̄ + Q(s', a')`; at least three of four context-binding products found
  per seed and ~+0.16 recurrence advantage over the raw twin (development,
  oracle-visible context).
- Added **discovery-driven control** (`tests/test_discovery_control_life.py`):
  features discovered online from action-conditioned reward prediction feed a
  bias-free `DifferentialSARSAAgent` with Q-weight carry-over by feature
  identity across bank refreshes; the discovered bank recovers 92–94% of the
  oracle-representation advantage (development).
- Added the **integrated single-life diagnostic**
  (`tests/test_integrated_life.py`): one uninterrupted 48,000-step
  closed-loop life with 120 payoff switches and a mid-life input-noise
  stressor; the gated rung reaches ≥0.969 lifetime reward with flat
  re-coordination and a +0.60 paired memory gap versus its ablation
  (development; visible context, hand-built representation).
- Added **held-out confirmation** of the wave-3 development results
  (`tests/test_held_out_confirmation.py`): precommitted floors passed on
  first-run, then-uninspected batches for primary-only scale discovery,
  discovery-control, a hand-gated life, and longevity (nonpromoting
  robustness evidence).
- Added the **Forager benchmark family**
  (`alberta_framework/benchmarks/`: `forager.py`, `causal_map_forager.py`,
  `forager_matrix.py`, `forager_results.py`, `official_foragax.py`): the
  pinned `continual-foragax==0.55.0` integration with paper presets, a causal
  feature encoder, the `alberta_horde_ac` and `alberta_causal_map` methods,
  official-NPZ and legacy-SQLite importers, and strict paired statistics —
  plus the `alberta-forager-benchmark` console script, the `[forager]`
  optional-dependency extra, and `FORAGER_BENCHMARK.md`. Preserved the
  completed four-seed RTU-RTRL 500k development run as a byte-pinned,
  explicitly nonpromoting receipt with recomputed summary tests.
- Added publication-shaped development runners for slowly-changing regression
  and UPGD Input-permuted MNIST (`upgd_ipmnist`). The UPGD lane
  completed a 10-seed matched development diagnostic: UPGD-W/AdamW mean online
  accuracies were `0.7791470803916454`/`0.7190002817213534`, and the paired
  descriptive mean difference was `0.06014679867029188` (10/10 positive).
  Preserved the original `outputs/upgd_ipmnist/results.v1.json`, whose strict
  failure is limited to its 10-vs-20-seed note, and added the structurally valid
  `outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json` plus
  current addendum `outputs/upgd_ipmnist/nonpromoting_receipt.v2.json`, which
  preserves `nonpromoting_receipt.v1.json` byte-for-byte. They remain permanently
  nonpromoting: 10 rather than 20 published seeds, documented
  stream/logging/numeric deviations, no execution-time
  source/full-closure/command/environment/data binding, and an AdamW
  figure-read-off gap of about `+0.039`. No inferential, SOTA, or Alberta Plan
  claim follows; a fresh source-bound full-seed run is required.

### Changed

- Narrowed README and PrototypeAgent claims: the repository contains
  mechanisms across the Alberta Plan, not a completed twelve-step agent.
- Replaced the package import docstring's all-steps-Complete table with the
  evidence-level status and relabeled the legacy `alberta-evidence-gate` as an
  artifact-availability check rather than scientific validation.
- Corrected STOMP's discounted differential semi-MDP reward/baseline
  accounting, primitive-action credit, option-model planning, OaK transition
  ownership and active-option curation, and Prototype dream isolation.
- Added held-out recurring-feature, world-model decision-fidelity, and
  recurring multi-agent evaluations. Each documents its restricted claim;
  none is an end-to-end Alberta Plan result.
- Added a fail-closed held-out IA artifact. Its v1 run remains a valid
  scientific rejection: causal reward uplift and augmentation controls pass,
  but action-changing intervention prevalence is 8.73% versus the frozen 10%
  gate. A consumed-seed replay reproduced every v1 scientific field exactly at
  the time and remains explicitly nonpromoting; later `average_reward.py` drift
  now makes live current-source compatibility invalid. The old threshold was
  not retuned.
- Restored the original narrow FTL decision-fidelity claim through a
  fail-closed historical/current-source compatibility chain. The chain pins
  the original bytes and scientific digest, reconstructs acceptance from
  primitive rows, requires an exact current-source replay on consumed seeds
  30–59, and permits drift only in the artifact-builder hash. The replay is
  nonpromoting, and the absence of the exact historical builder source is
  recorded as a source-recoverability limitation.
- Reclassified the context-inference, TD-target-discovery,
  Prototype-retention, and historical wave-3 confirmation suites as
  development/nonpromoting evidence. Their inspected seeds, oracle or
  hand-built context paths, frozen feature bank, unequal resource budgets,
  and missing strict artifacts or component ablations remain explicit.
- Added separate conventional option-return and expected-duration TD heads
  plus a deterministic Step 5 renewal diagnostic. The diagnostic reaches L1:
  it shows that return/duration ranks a fast option correctly where return
  alone does not, but uses supplied options and features and is not a promoted
  semi-Markov control result.
- Added `alberta-evidence-status`, a machine-readable index that invokes each
  strict promoted-artifact validator and distinguishes accepted evidence,
  valid scientific rejections, missing runs, and invalid artifacts. The index
  records frozen commands, protocols, configurations, seeds, thresholds,
  artifact and source hashes, environment provenance, limitations, and
  validation timestamps. Its explicit dirty-state policy requires registered
  source hashes to match while recording unrelated worktree changes. Unit and
  smoke evidence is structurally barred from scientific promotion, with
  pytest lanes registered for unit, integration, scientific, and development
  work. The index explicitly is not an Alberta Plan completion certificate.

### Fixed

All fixed failing-test-first during this campaign:

- SARSA dead-λ for control heads.
- STOMP environment-reward grounding and its idle-update leak.
- OaK curation zeroing instead of re-initializing replaced options, and
  step-0 eviction.
- IA exo-cortex crediting its own actions instead of the partner's.
- Step-7 pre-warmup RNG freeze.
- Off-policy Horde ρ² trace composition and the GTD correction term's
  missing ρ.
- `reset_dormant_neurons` optimizer-pytree corruption.
- Baseline optimizers missing from the config registry.
- Prototype-basis recycled slots inheriting stale readouts.
- Compositional raw-index aliasing.
- UPGD-memory blend-logit gradient bias.
- Mann-Whitney rank-biserial sign inversion in the statistics utilities.

### Removed

- Removed the legacy root-`benchmarks` import shim from
  `alberta_framework/__init__.py`. It registered any importable top-level
  `benchmarks` package under the `alberta_framework.benchmarks` name, which
  could shadow the real packaged subpackage once one existed. The real
  subpackage is now imported eagerly and always wins
  (`tests/test_benchmarks_shim.py`).

### Packaging

- Version 0.27.0 across `pyproject.toml`, `alberta_framework.__version__`,
  and `CITATION.cff` (which had been stuck at 0.17.1).
- Registered the five evidence CLIs as console scripts:
  `alberta-ia-evidence`, `alberta-multiagent-evidence`, `alberta-ftl-evidence`,
  `alberta-recurring-feature-evidence`, and `alberta-scale-robust-evidence`.
- The sdist now includes `RESEARCH_STATUS.md` and
  `CONTINUAL_LEARNING_EVIDENCE.md`.
- ruff and mypy now target Python 3.12, matching `requires-python`.
- README rewritten: streams/testbeds, new mechanisms, the evidence registry,
  and benchmark lanes; removed dead CI/PyPI badges and the dead hosted-docs
  section; the canonical repository URL is now
  `https://github.com/lalalune/alberta` (see `VENDORING.md` for fork status).
- `VENDORING.md` rewritten to describe this tree as a development fork with a
  divergence summary; `CLAUDE.md`/`AGENTS.md` contributor guides added.

### Tests

- 2,641 passed and 55 skipped in the last full pre-release run, up from
  1,901 at 0.26.0; the suite now collects 3,284 tests across the `unit`,
  `integration`, `scientific`, and `development` marker lanes.

## [0.26.0] - 2026-05-22

### Added

- **GRU recursive perception for PrototypeAgent (Step 8a)** — fixed-weight echo-state
  GRU augments raw observations with recurrent hidden state before passing to all
  downstream components (OaK, Horde, world model). Glorot-uniform weight init,
  zero hidden init, pure-functional `_gru_step`. Controlled via
  `GRUPerceptionConfig(observation_dim, hidden_dim)` in `PrototypeAgentConfig`.
  12 new tests cover config validation, weight shapes, hidden dynamics, and
  augmented-obs routing. (`alberta_framework/core/prototype_agent.py`,
  `alberta_framework/core/__init__.py`)

- **Step 9 prioritized dreaming** — multi-step imagined rollouts with scored
  candidate selection. `score_dream_candidates` picks the most surprising/useful
  anchor state from `dream_candidate_count` random candidates; rollouts proceed
  for `dream_rollout_horizon` steps under the behavior model. BehaviorModel now
  tracked in `Step9DreamingState` and updated every real step.
  (`alberta_framework/steps/step9.py`)

- **Intelligence Amplification (Step 12) now exported from public API** —
  `ExoCerebellumAgent/Config/State`, `ExoCortexAgent`, `IAAgent/Config/State/
  UpdateResult/ArrayResult`, `RecommendationProtocolConfig/State/Result`,
  `init_recommendation_protocol_state`, and `update_recommendation_protocol`
  are now importable directly from `alberta_framework.core`.
  (`alberta_framework/core/__init__.py`)

### Fixed

- **`PrototypeAgentConfig` now validates `world_model.observation_dim`** when
  `gru_perception` is set, ensuring the world model's observation dimension
  matches `gru_perception.augmented_dim()`. Previously only `oak.observation_dim`
  was validated. (`alberta_framework/core/prototype_agent.py`)

- **`out_of_class_results.json` artifact restored** — reconstructed from the
  completed `out_of_class_SUMMARY.md` (30 seeds, 3 streams; original JSON was
  lost). Step 2 evidence gate now passes.
  (`outputs/step2_canonical/out_of_class_results.json`)

### Tests

- 1901 tests pass (up from 1900); new test `test_world_model_dim_mismatch_raises`
  covers the GRU + world-model dimension validation.

## [0.25.0] - 2026-05-21

### Added

- **CartPole FA Dyna benchmark** (Step 7) — 10-seed 5000-step comparison of
  linear-model Dyna vs real-only DifferentialSARSA on CartPole-v1 continuing.
  Result: ceiling effect — both agents achieve optimal reward=1.000 on all 10
  seeds; linear world model is stable (no degradation) but CartPole is too easy
  to reveal planning benefit. Faster benchmark (JIT-wrapped update functions,
  module-level agent objects) runs in 26 s vs the original 135+ min.
  (`benchmarks/step7_cartpole_dyna.py`, `outputs/step7_cartpole_dyna/`)

- **`NonlinearQHordeActorCriticAgent`** — action-value Horde critic with
  expected-SARSA targets, one control head per action; exported from the
  package. 10-seed variant search shows best variant ties Q at +0.0 improvement
  vs SARSA's +12.0 on catch — action-value critic substitution ruled out as
  Step 4 closure.

- **Step 4 probe suite** — adaptive-ObGD NLHAC at 500 and 1000 steps (both
  regress catch: -1.4 / -6.0 vs Q while SARSA is +6.6 / +13.4); wider (64,64)
  actor/critic NLHAC (catch -2.0 vs SARSA +9.33); rules out three more
  approaches to close the AC-vs-SARSA catch gap.

- **`rlsecd_external_audit.py`** — reproducible script checking availability
  of external `rlsecd` / `chronos-sec` sibling repos; result embedded in
  Step 3 solution gate.

### Fixed

- **Step 7 CartPole benchmark JIT regression** — original `step7_cartpole_dyna.py`
  created new agent/model Python objects per seed, forcing JAX to re-trace the
  planning scan body on every step; rewritten to use module-level agent/model
  objects and explicit `jax.jit(static_argnums=...)` wrappers.

## [0.24.0] - 2026-05-21

### Added

- **PrototypeAgent CartPole composition smoke** — a 5-seed continuing-control
  run exercised the configured composition without non-finite weights. Both
  flat DifferentialSARSAAgent and PrototypeAgent reached the reward ceiling
  (mean 1.000), so this was a wiring/stability check, not evidence that the
  composed mechanisms helped or that all twelve steps were integrated
  (`benchmarks/prototype_end_to_end.py`, results in
  `outputs/prototype_end_to_end/`)

### Added (continued)

- **`AdaptiveObGDBounding`** (Elsayed et al. 2024, Appendix B) — ObGD global
  bounding followed by per-weight RMS normalisation; registered in
  `_BOUNDER_REGISTRY` and exported from `alberta_framework`; 3 tests added to
  `tests/test_config_serialization.py`

### Fixed

- **PrototypeAgent dreaming JIT regression** — guarded dreaming scan closure
  was redefined on every `update()` call, causing JAX to retrace the XLA
  computation graph each step (~720 ms/step); extracted to
  `PrototypeAgent._run_dreams()` with `@functools.partial(jax.jit, static_argnums=(0,))`
  reducing to ~8 ms/step (94× speedup)

## [0.23.0] - 2026-05-21

### Added

- **Step 11 OaK curation benchmark** — 10-seed 6-state chain proves utility
  tracking detects and replaces counterproductive options; post-curation
  avg-reward recovers to 0.935 (8/10 seeds ≥ 0.70) from mean 0.70 pre-curation
  (`benchmarks/step11_oak_curation.py`, results in `outputs/step11_oak/`)

- **Step 12 IA augmentation benchmark** — 5-seed demonstration that
  exo-cerebellum MSE ≈ 0 (vs zero-baseline 0.167) and cortex recommendation
  accuracy 60% (>50% random) on 6-state chain
  (`benchmarks/step12_ia_augmentation.py`, results in `outputs/step12_ia/`)

- **Neuron utility tracking** — per-hidden-unit EMA of gradient L2 norm for
  dormant-neuron detection in long-running continual agents
  - `MLPLearner(track_neuron_utility=True, neuron_utility_decay=0.99)` stores
    `MLPLearnerState.neuron_utility: tuple[Array, ...] | None` (one `(h_i,)`
    array per hidden layer, None when disabled)
  - `MLPLearner.dormant_neuron_fraction(state, threshold)` returns the fraction
    of neurons below the utility threshold
  - `MLPLearner.reset_dormant_neurons(state, key, threshold)` re-initialises
    incoming weights, eligibility traces, and optimizer states for dormant
    neurons; zeroes outgoing weights from next layer to prevent signal injection
  - Config roundtrip via `to_config()` / `from_config()` includes new fields
  - 11 tests covering shapes, EMA dynamics, dormancy counts, reset, and serialisation

## [0.22.0] - 2026-05-21

### Added

- **Autostep-for-actor** — per-weight adaptive step-sizes for all actor MLPs
  - `NonlinearHordeActorCriticAgent` actor now uses `Autostep.init_for_shape` /
    `update_from_gradient`; fixed scalar `actor_step_size` removed from config
  - `AverageRewardHordeActorCriticAgent` receives the same upgrade; per-weight
    `AutostepParamState` stored in `AverageRewardHordeActorCriticState`
  - Both agents accept an optional `actor_optimizer: Autostep | None` constructor arg
    (default `Autostep(initial_step_size=0.05)`) and expose `actor_optimizer` property
  - Config roundtrip includes `actor_optimizer` serialisation

- **Nonlinear STOMP / OaK base Q-function** — replaces hard-coded linear weights
  - `STOMPState.base_learner_state: MultiHeadMLPState` replaces `base_q_weights` /
    `base_traces`; the underlying `MultiHeadMLPLearner` has `n_heads = n_total_actions`
  - `STOMPConfig.base_hidden_sizes: tuple[int, ...] = ()` enables nonlinear trunks;
    the empty-tuple default preserves previous linear behaviour exactly
  - Discounted semi-MDP differential Q target
    (`R_o^γ - r̄·Σ_{k=0}^{T_o-1}γ^k + γ^{T_o}·max Q(s')`) computed via
    NaN-masked `MultiHeadMLPLearner.update` — compatible with both linear and
    MLP paths; at `γ=1` the baseline mass reduces to the raw duration `T_o`
  - OaK curation resets the curated option's head (weights, biases, traces, optimizer
    states) inside the `MultiHeadMLPState` rather than zeroing a raw weight slice
  - `STOMPAgent.base_q_values(state, obs)` and `OaKAgent.base_q_values(state, obs)`
    expose Q-value computation through the agent API, used by `ExoCortexAgent.recommend`
    and `PrototypeAgent.act` (both now robot-ready for high-dimensional observations)
  - `feature_to_subtask_specs` in `prototype_agent.py` handles both linear (head-weight
    stack) and nonlinear (first trunk-layer proxy) feature-importance extraction

## [0.21.0] - 2026-05-21

### Added

- **PrototypeAgent** — experimental composition surface for mechanisms
  associated with Steps 1–12
  - `PrototypeAgentConfig`: minimal defaults (just `n_primitive_actions` + `observation_dim`)
  - Single `.update()` integrates world model, buffer, OaK, dreaming, Horde, and IA
  - `feature_to_subtask_specs`: automatic subtask extraction from OaK Q-weight importances
  - `run_prototype_scan` / `run_prototype_smoke`: JIT-compiled loop and validity probe
  - 50 tests covering all components and 200-step fineness

## [0.20.0] - 2026-05-21

### Added

- **Steps 11 and 12: OaK and Intelligence Amplification**
  - `OaKAgent`: extends STOMP with utility EMA, curation, and option keyboard (Barreto et al.)
  - `ExoCerebellumAgent` / `ExoCortexAgent` / `IAAgent`: paired cerebellum + cortex
    augmenting a partner agent's observations and action recommendations
  - 32 OaK tests + 30 IA tests

## [0.19.0] - 2026-05-20

### Added

- **Step 10: STOMP temporal abstraction**
  - `STOMPAgent`: subtask-defined options, intra-option differential Q, option outcome models
  - `SubtaskSpec` / `STOMPSpecArrays` / `STOMPState`: JAX-compatible option state
  - 36 tests; seeded benchmark proves STOMP options accelerate control ~10x vs flat
  - `Step10STOMPConfig` production facade

## [0.18.0] - 2026-05-19

### Added

- **Steps 5–9: Average-reward control, world models, and dreaming**
  - `DifferentialTDLearner` / `DifferentialSARSAAgent` / `DifferentialGTDLearner`:
    continuing average-reward prediction and control (Steps 5–6)
  - `AverageRewardHordeLearner` / `AverageRewardHordeActorCriticAgent`:
    nonlinear shared-trunk Horde for differential GVF prediction and actor-critic
  - `OneStepWorldModel` / `ActionConditionedWorldModel`: reward + next-obs prediction (Step 8)
  - `GuardedDreamer` + `RecentObservationBuffer`: error-gated dreaming with real-state anchors (Step 9)
  - `Step7DynaConfig`: real-transition update + fixed `planning_steps` Dyna backups
  - Seeded benchmarks: Step 7 prioritized Dyna improves final-window reward from 0.92 → 1.00;
    six-state chain Dyna wins cumulative reward 8/10 seeds (+41.7%)
  - RiverSwim benchmark (Step 6): 10/10 seeds, 97.5% right-action rate

## [0.17.1] - 2026-04-10

### Added

- **Autostep-for-GTD(λ)** — per Kearney et al. (2019)
  - `AutoTDIDBD` optimizer with per-weight step-size adaptation for TD learning
  - Eligibility traces integrate with Autostep's normalizer and overshoot prevention

### Fixed

- Flax dependency added and version pinned in `pyproject.toml`

## [0.17.0] - 2026-04-05

### Added

- **Replacing traces** — `TraceMode.REPLACING` on `MultiHeadMLPLearner`
  - Replaces stale trace magnitude on re-visit rather than accumulating
  - Configurable per-head alongside `TraceMode.ACCUMULATING`
  - Trace-bounding integration: replaced traces scaled by ObGD bounding factor

## [0.16.0] - 2026-03-15

### Added

- **SARSA agent (Step 4a)** — on-policy control via Horde architecture
  - `SARSAAgent`: wraps `HordeLearner` with epsilon-greedy action selection and SARSA target computation
  - `SARSAConfig`: configuration for n_actions, gamma, epsilon schedule
  - `SARSAState`, `SARSAUpdateResult`: immutable state and result types
  - `run_sarsa_episode`: Python loop for episodic Gymnasium environments
  - `run_sarsa_continuing`: continuing mode with pseudo-boundary handling (daemon-style)
  - `run_sarsa_from_arrays`: JIT-compiled `jax.lax.scan` for pre-collected data (security-gym)
  - Gumbel trick tie-breaking for uniform action selection among equal Q-values
  - Linear epsilon decay schedule (configurable start, end, decay steps)
  - Optional prediction demons coexist with control demons in the same Horde
  - Config serialization via `to_config()` / `from_config()` roundtrip
  - 30 new tests covering init, action selection, update logic, epsilon decay, bounding, serialization, and scan loop
  - Example: `examples/The Alberta Plan/Step4/sarsa_cartpole.py`
  - Documentation: `docs/guide/sarsa-control.md`

- **Trunk trace guard** — validation preventing `gamma * lamda > 0` on `MultiHeadMLPLearner` with hidden layers
  - VJP backward pass folds error into trunk cotangent before trace accumulation; only correct when traces reset each step
  - Linear baseline (`hidden_sizes=()`) allows any gamma/lamda
  - `HordeLearner` enforces trunk gamma=0 by design (per-head trace decay only)
  - Expanded docstrings on `MultiHeadMLPLearner` and `HordeLearner` explaining the constraint

## [0.10.0] - 2026-02-27

### Added

- **Hybrid optimizer (`head_optimizer`)** — separate optimizer for trunk vs head layers on `MLPLearner` and `MultiHeadMLPLearner`
  - `MLPLearner(head_optimizer=...)`: output layer uses `head_optimizer`, hidden layers use `optimizer`
  - `MultiHeadMLPLearner(head_optimizer=...)`: all prediction heads use `head_optimizer`, trunk uses `optimizer`
  - Enables stable LMS+ObGD for non-convex hidden layers with adaptive Autostep for the linear output head
  - Backwards compatible: `head_optimizer=None` (default) keeps all layers on the same optimizer
  - 10 new tests (6 MLPLearner, 4 MultiHeadMLPLearner)

## [0.9.0] - 2026-02-22

### Added

- **Agent lifecycle tracking** — `step_count`, `birth_timestamp`, `uptime_s` on all learner states
  - `LearnerState`, `MLPLearnerState`, `TDLearnerState`: new fields with backward-compatible defaults
  - `MultiHeadMLPState`: added `birth_timestamp` and `uptime_s` (already had `step_count`)
  - `step_count` incremented inside `update()` (JAX-traced, safe in `jax.lax.scan`)
  - `birth_timestamp` set at `init()`, immutable across updates
  - `uptime_s` accumulated after each `jax.lax.scan` completes in all learning loops
  - All learning loop functions stamp uptime: `run_learning_loop` (simple + tracking), `run_learning_loop_batched`, `run_mlp_learning_loop` (simple + tracking), `run_mlp_learning_loop_batched`, `run_td_learning_loop`, `run_multi_head_learning_loop`, `run_multi_head_learning_loop_batched`, `learn_from_trajectory`
- `agent_age_s(state)` — wall-clock seconds since agent birth
- `agent_uptime_s(state)` — cumulative active seconds inside learning loops
- 28 new lifecycle tracking tests across all learner types

## [0.8.1] - 2026-02-21

### Added

- **bsuite benchmark integration** — bridges framework to bsuite for standardized RL diagnostics
  - `ContinuingWrapper`: converts episodic envs to continuing streams (Alberta Plan Step 6)
  - `AlbertaAgent`: bridges bsuite `Agent` ABC to `MultiHeadMLPLearner` with Q-learning
  - Three agent factories: Autostep+ObGD, LMS+ObGD, Adam (haiku/optax external baseline)
  - Hyperparameter configs with standard `(64, 64)` and bottleneck `(16, 16)` variants
  - `run_single.py` / `run_sweep.py` CLIs with `--continual-sequence` and `--use-scythe` flags
  - Analysis module: result loading, comparison plots, representation analysis, summary tables
  - Representation utility logging: per-weight step-sizes, trunk trace magnitudes, per-head metrics
  - 22 tests covering wrapper, agents, factories, representation logging, and integration

### Dependencies

- Added `[bsuite]` optional dependency group (dm-env, optax, dm-haiku, plotnine)

## [0.8.0] - 2026-02-16

### Added

- **`MultiHeadMLPLearner`** — shared-trunk MLP with multiple prediction heads for multi-task continual learning
  - VJP-based gradient computation with accumulated cotangents (single backward pass through trunk)
  - NaN target masking for selective head activation (inactive heads skip gradient updates)
  - Composable: accepts any `Optimizer`, optional `Bounder`, optional `Normalizer`
  - Eligibility traces managed per-head and per-trunk-layer
- `MultiHeadMLPState`, `MultiHeadMLPUpdateResult`, `MultiHeadLearningResult`, `BatchedMultiHeadResult` types
- `run_multi_head_learning_loop()` — `jax.lax.scan` over observation/target arrays with NaN masking
- `run_multi_head_learning_loop_batched()` — `jax.vmap` over initialization keys for multi-seed parallelization
- `multi_head_metrics_to_dicts()` — convert array metrics to per-head dicts for online use

## [0.7.3] - 2026-02-09

### Added

- `MLPLearner(use_layer_norm=False)` — toggle parameterless LayerNorm for ablation studies (default `True`, backwards-compatible)

## [0.7.2] - 2026-02-08

### Fixed

- IDBD operation ordering now matches Sutton 1992 Figure 2: meta-update first, then NEW alpha for weight and trace updates

### Changed (Breaking)

- Autostep rewritten to match Mahmood et al. 2012 Table 1 exactly:
  - `v_i` now tracks meta-gradient magnitude `|δ*x*h|` (was primary gradient `|δ*x|`)
  - `v_i` uses self-regulated EMA (Eq. 4), not `max(|grad|, v*τ)`
  - Overshoot prevention via `M = max(Σ α_i*x_i², 1)` (Eq. 6-7)
  - Trace decay includes `x²`: `h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i`
  - Normalizers and traces initialized to 0 (was 1 and 0)
  - Normalization only applies to meta-update, not to weight/trace updates
- `Autostep(normalizer_decay=...)` renamed to `Autostep(tau=...)`, default changed from 0.99 to 10000.0
- `AutostepState.normalizer_decay` renamed to `AutostepState.tau`
- `AutostepParamState.normalizer_decay` renamed to `AutostepParamState.tau`

### Added

- `Autostep.update_from_gradient()` now accepts optional `error` parameter for full paper algorithm in MLP path
- `Optimizer.update_from_gradient()` base signature accepts optional `error` parameter

## [0.7.1] - 2026-02-07

### Added

- **`AGCBounding`** — Adaptive Gradient Clipping (Brock et al. 2021) as a `Bounder` ABC, per-unit clipping scaled by weight norm
- `_unitwise_norm()` helper for unit-wise L2 norm computation (1D: abs, 2D+: norm over fan-in axes)

## [0.7.0] - 2026-02-07

### Changed (Breaking)

- Removed `NormalizedLinearLearner`, `NormalizedMLPLearner` — use `LinearLearner(normalizer=...)` and `MLPLearner(normalizer=...)` instead
- Removed `run_normalized_learning_loop`, `run_normalized_learning_loop_batched`, `run_mlp_normalized_learning_loop`, `run_mlp_normalized_learning_loop_batched` — unified into `run_learning_loop` and `run_mlp_learning_loop` (detect normalization from learner)
- Removed `NormalizedLearnerState`, `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedNormalizedResult`, `BatchedMLPNormalizedResult`, `MLPObGDState` types
- `MLPLearner` no longer accepts `kappa` parameter — use `bounder=ObGDBounding(kappa=2.0)` instead

### Added

- `Bounder` ABC and `ObGDBounding` for decoupled update bounding (composable with any optimizer)
- `AutostepParamState` for per-parameter Autostep optimization (arbitrary array shapes)
- `Optimizer.init_for_shape()` and `Optimizer.update_from_gradient()` for shape-agnostic optimization (LMS, Autostep)
- `MLPLearner` now accepts composable `optimizer`, `bounder`, and `normalizer` parameters
- `LinearLearner` now accepts optional `bounder` and `normalizer` parameters
- Unified learning loops: 4 functions instead of 8 (linear + MLP, each with single + batched)

### Fixed

- mypy override errors — base class `init_for_shape`/`update_from_gradient` use `Any` since return type varies by subclass

## [0.6.1] - 2026-02-07

- Version bump only

## [0.6.0] - 2026-02-07

### Changed (Breaking)

- Replaced `OnlineNormalizer`, `NormalizerState`, `create_normalizer_state` with `Normalizer` ABC hierarchy

### Added

- `Normalizer` ABC with generic `StateT` constraint, following the `Optimizer[StateT]` pattern
- `EMANormalizer` — exponential moving average normalization (renamed from `OnlineNormalizer`, corrected docstrings)
- `WelfordNormalizer` — true Welford's algorithm with Bessel's correction for stationary distributions
- `EMANormalizerState`, `WelfordNormalizerState`, `AnyNormalizerState` types
- `NormalizedLinearLearner` now accepts any `Normalizer` subclass
- `NormalizedMLPLearner` — wraps `MLPLearner` with online normalization (EMA or Welford)
- `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedMLPNormalizedResult` types
- `run_mlp_normalized_learning_loop()` with optional `NormalizerTrackingConfig`
- `run_mlp_normalized_learning_loop_batched()` for vmap-based multi-seed normalized MLP training

## [0.5.3] - 2026-02-06

### Added

- `run_mlp_learning_loop_batched()` for vmap-based multi-seed MLP training with `BatchedMLPResult` return type

## [0.5.2] - 2026-02-06

### Fixed

- Resolved mypy type error in `MLPLearner` z_sum computation — replaced `sum()` over JAX arrays with explicit `jnp.array(0.0)` accumulator

## [0.5.0] - 2026-02-06

### Added

- **ObGD Optimizer**: Observation-bounded Gradient Descent for overshooting prevention (Elsayed et al. 2024). Dynamically bounds effective step-size based on error magnitude and trace norms. Works as a linear optimizer (`ObGD`) and within the MLP learner.
- **MLPLearner**: Multi-layer perceptron with ObGD optimizer for nonlinear function approximation in the streaming setting. Architecture: `Input -> [Dense -> LayerNorm -> LeakyReLU] x N -> Dense(1)`. Configurable depth via `hidden_sizes` tuple.
- **Sparse Initialization**: `sparse_init()` function implementing LeCun-scale initialization with per-neuron sparsity (default 90%), following Elsayed et al. 2024.
- **`run_mlp_learning_loop()`**: JIT-compiled MLP training via `jax.lax.scan`, same pattern as existing linear learning loops.
- **MLP Types**: `MLPParams`, `MLPObGDState`, `MLPLearnerState`, `MLPUpdateResult` chex dataclasses.
- **ObGD Types**: `ObGDState` chex dataclass with `create_obgd_state()` factory.
- **Step 2 Example**: `linear_vs_mlp_comparison.py` comparing LinearLearner+Autostep vs MLPLearner+ObGD on RandomWalk, AbruptChange, and DynamicScaleShift streams.

### Notes

- ObGD defaults to `gamma=0, lamda=0` for supervised learning (traces = current observation). Nonzero values enable eligibility traces for future RL use (Steps 3-4).
- MLP implementation is self-contained (no Flax/Haiku dependency). Uses `jax.grad` for backpropagation and parameterless layer normalization.
- The `Optimizer` generic constraint now includes `ObGDState`, so `ObGD` can be used with `LinearLearner` as well.

## [0.4.0] - 2026-02-04

### Added

- TD-IDBD optimizer for temporal-difference learning with per-weight adaptive step-sizes and eligibility traces (Kearney et al., 2019)
- AutoTDIDBD optimizer with AutoStep-style normalization for improved stability
- `TDLinearLearner` class for linear value function approximation in TD learning
- `run_td_learning_loop()` for JIT-compiled TD learning via `jax.lax.scan`
- TD state types: `TDIDBDState`, `AutoTDIDBDState`, `TDLearnerState`, `TDTimeStep`
- `TDStream` protocol for TD experience streams

## [0.3.2] - 2026-02-03

### Fixed

- Relaxed test tolerance in batched vs sequential comparison tests (`rtol=1e-5`) to account for floating-point differences between vmap and sequential execution paths
- Added `ignore = ["F722"]` to ruff config for jaxtyping shape annotation syntax that ruff doesn't understand
- Removed unused `PRNGKeyArray` import from `core/types.py`

## [0.3.0] - 2026-02-03

### Added

- Migrated all state types from NamedTuple to `@chex.dataclass(frozen=True)` for DeepMind-style JAX compatibility
- jaxtyping shape annotations for compile-time type safety (`Float[Array, " feature_dim"]`, `PRNGKeyArray`, etc.)
- Updated test suite to use chex assertions (`chex.assert_shape`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`)

### Dependencies

- Added `chex>=0.1.86` and `jaxtyping>=0.2.28` as required dependencies
- Added `beartype>=0.18.0` as optional dev dependency for runtime type checking

## [0.2.2] - 2026-02-02

### Fixed

- mypy type errors in `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions
- Added `typing.cast` to properly handle conditional return type unpacking in batched learning loops

## [0.1.0] - 2026-01-19

### Added

- **Core Optimizers**: LMS (baseline), IDBD (Sutton 1992), and Autostep (Mahmood et al. 2012) with per-weight adaptive step-sizes
- **Linear Learners**: `LinearLearner` and `NormalizedLinearLearner` with pluggable optimizers
- **Scan-based Learning Loops**: JIT-compiled training with `jax.lax.scan` for efficiency
- **Online Normalization**: Streaming feature normalization with exponential moving averages
- **Experience Streams**: `RandomWalkStream`, `AbruptChangeStream`, `CyclicStream`, `SuttonExperiment1Stream`
- **Gymnasium Integration**: Trajectory collection and learning from Gymnasium RL environments
- **Step-Size Tracking**: Optional per-weight step-size history recording for meta-adaptation analysis
- **Multi-Seed Experiments**: `run_multi_seed_experiment` with optional parallelization via joblib
- **Statistical Analysis**: Pairwise comparisons, confidence intervals, effect sizes (requires scipy)
- **Publication Visualization**: Learning curves, bar charts, heatmaps with matplotlib
- **Export Utilities**: CSV, JSON, LaTeX, and Markdown table generation
- **Documentation**: MkDocs-based documentation with auto-generated API reference

### Notes

- Requires Python 3.13+
- Implements Step 1 of the Alberta Plan: demonstrating that IDBD/Autostep can match or beat hand-tuned LMS
- All state uses immutable NamedTuples for JAX compatibility
- Follows temporal uniformity principle: every component updates at every time step
