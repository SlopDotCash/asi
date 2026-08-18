# Repository anti-LARP audit

> **Historical snapshot.** This audit is bound to the revision and cleanup worktree named
> below. It is not the authority for the current checkout's file counts, implementation state,
> evidence validity, or remaining work. Use [`../status.md`](../status.md), the live
> `alberta-evidence-status` command, and current campaign summaries for those facts.

**Audit date:** 2026-08-14–15 (America/Los_Angeles)
**Audited base revision:** `22739da0b5d8a06a621b6743297f1c46d8d87903`
**Cleanup state:** an uncommitted worktree observed during the audit; it is not reconstructable
from the base revision alone
**Scope:** implementation, tests, package metadata, documentation, registered evidence, and
the active IPMNIST and Forager development records

## Bottom line

The pre-cleanup repository had a very large research-theatre surface. The audit-time worktree
removed many unsupported, redundant, unissued, self-certifying, or superseded paths. Because
that cleanup state was not recorded as an immutable commit, its line-count reduction is a
historical observation rather than an independently reproducible repository metric. Calling
the removed surface “LARP” is a documented audit judgment, not a measured percentage of false
mathematical statements, fabricated experimental values, or known LLM authorship.

The strongest measurements are:

| Measure | Result | Meaning |
|---|---:|---|
| Package and test Python before cleanup | 1,264,156 lines in 1,192 files | Audited base revision |
| Package and test Python retained in the audit-time worktree | 326,768 lines in 356 files | Contemporaneous count; not the current tree and not reconstructable from the base commit alone |
| Python surface reported removed | **74.15% by physical lines; 70.13% by files** | Contemporaneous calculation, not a live or independently reproducible metric |
| Single `3d195c3` “updates” commit | 758,229 insertions in 849 files | Largest concentrated expansion |
| Audit-cleanup deletions born in that commit | 680,481 pre-cleanup lines across 643 files | Recorded provenance cohort, not every deleted file |
| Alberta Plan steps complete | **0/12** | Mechanism presence is not Step completion |
| Live registered claims valid | **0/5** | Registry exits 2; frozen outcomes are historical records only |
| IPMNIST promotion-grade claims | **0** | The campaign explicitly forbids promotion |
| Matched-current Forager performance cells executed | **0/210** | The stored campaign is prepared, not run |

The retained repository is not empty or fraudulent. It contains substantial working JAX
learners, optimizers, streams, control components, real development measurements, and strict
validators that correctly fail closed. The failure was that implementation breadth,
governance machinery, tests, and prose grew far beyond executable evidence.

## What “LARP” means in this audit

A file or claim was classified as LARP when one or more of these conditions held:

1. no production caller, console entry point, external consumer, or meaningful test exercised
   it;
2. an advertised path was a stub, raised only after an expensive protocol had started, or
   could not execute its registered configurations;
3. tests asserted symbol presence, manifests, schemas, source text, or declarations without
   exercising the claimed behavior;
4. a development experiment calibrated its own passing threshold inside pytest or replayed
   consumed evidence while looking like a fresh gate;
5. extensive issuance, trust, publication, qualification, or evidence machinery existed for
   a campaign that was never issued or executed;
6. narrative wording such as “accepted,” “final,” “SOTA,” “complete,” or “protocol exact” was
   contradicted by the live validator or primary record; or
7. a superseded or negative lane remained presented as active without a durable closure.

Normal library functions are not dead merely because this repository has no internal caller;
public extension surfaces were retained when their contracts and tests made an external use
credible. Likewise, a negative result is not LARP when it is clearly labeled and preserved.

## Historical size record

The audit recorded the following comparison between the base revision and its uncommitted
cleanup worktree:

```text
HEAD package Python:       833,938 lines / 492 files
HEAD test Python:          430,218 lines / 700 files
combined:                1,264,156 lines / 1,192 files

retained package Python:   210,330 lines / 160 files
retained test Python:      116,438 lines / 196 files
combined:                  326,768 lines / 356 files
```

Those recorded counts imply 937,388 lines (74.15%) and 836 files (70.13%) removed. The cleanup
worktree was not preserved as a commit or per-file classification manifest, so the calculation
cannot be rerun from the named base revision and must not be presented as a current statistic.
Physical removal is also not a semantic classification. The 70–80% LARP estimate was a review
judgment based on callers, tests, execution records, evidence authority, and narrative claims;
it is not a statistical confidence interval.

Git history supplies a separate provenance check. Commit `3d195c3` (2026-08-12), whose entire
subject was `updates`, added 758,229 lines: about 510,000 under `alberta_framework/` and
238,000 under `tests/`. Intersecting Python paths added by that commit with paths present at
`HEAD` and deleted by this cleanup identifies exactly 643 files containing 680,481 physical
`HEAD` lines. This concentration, repeated generated naming, and declaration-heavy tests are
consistent with LLM-amplified code generation. They cannot prove who or what authored the
text, so this audit does not assign an “LLM-written” percentage and does not call that cohort
an objective lower bound on LARP.

The recorded 74% figure also does **not** mean the retained surface was scientifically
validated. Scientific completeness is measured separately: 0/12 complete Plan steps and 0/5
live-valid registered claims.

## Confirmed LARP removed or retired

### Unissued expansion stacks

The cleanup removes the unexecuted HCCL, embodied-agent, hidden-partner, and prototype
expansions that consisted largely of transaction types, authority manifests, synthetic
governance, and isolated tests. It retains the smaller, actually integrated
`PrototypeAgent`, feature-lifecycle, partner-fusion, world-model, and continual-RL surfaces.

### Forager matched-v3 bureaucracy

The matched-v3 family had a very large chain of candidate-universe, external transport,
wheelhouse, OCI, trust, qualification, publication, and reward-bundle modules without a
completed scientific comparison. The cleanup removes that unissued v3 stack and retains the
smaller matched-current protocol that has real qualification records and an explicit zero-cell
execution state.

Preparation is still not performance evidence. The retained matched-current open campaign has
210 planned cells and no scored results. Its qualification snapshot reads no reward arrays,
uses no tuning seeds, and explicitly sets both `performance_claim` and
`promotion_authorized` false.

### Self-certifying tests and evaluation islands

Removed tests included public-export assertions, source-presence manifests, “not assessed”
matrices, replay-only gates, permanently skipped suites, and development campaigns that chose
their own thresholds. These tests inflated counts without increasing confidence in behavior.
Retained tests target numerical updates, state transitions, serialization, CLI/process
boundaries, strict validators, and bounded integration behavior.

### Superseded research lanes

The slowly-changing-regression harness, UPGD-IPMNIST v3 governance layer, speculative WP0–WP9
documents, and other concluded lanes were retired. Their immutable or append-only output
records remain in place, and reusable negative conclusions remain in
`docs/evidence/negative-results.md`.

## Scientific truth at audit time

### The audit-time registry supported zero current claims

`.venv/bin/alberta-evidence-status` exits 2 with all five registered claims invalid:

| Claim | Frozen artifact outcome | Live status | Main reason |
|---|---|---|---|
| `recurring_pair_features` | accepted, narrow L2 | invalid | registered source drift |
| `scale_robust_pair_features` | accepted, narrow L2 | invalid | registered source drift |
| `ftl_world_model_decision_fidelity` | accepted, historical L2 | invalid | compatibility chain fails; drift exceeds its waiver |
| `recurring_multiagent_coadaptation` | accepted, narrow L2 | invalid | registered source drift |
| `continual_intelligence_amplification` | valid rejection | invalid | source drift and 180 controller-budget mismatches |

These failures are committed source drift, not merely unrelated local modifications. The
artifacts remain useful historical records, but no frozen acceptance currently validates the
checked-out implementation. Editing hashes, weakening validators, or overwriting artifacts
would manufacture support and is prohibited.

The pinned `outputs/evidence_manifest.json` is itself a historical snapshot from when two
claims were reported supported; it is not the live authority and was not rewritten. The live
CLI recomputes all five entries and now refuses to replace any pre-existing manifest path.

The registry’s manually selected source lists are also not complete recursive execution
closures. Direct local dependencies are omitted in several claims. Future evidence contracts
must bind a versioned recursive local-import closure plus the dependency/runtime lock; changing
the v1 closure in place would not repair an already-invalid historical artifact.

### No Alberta Plan Step is complete

The repository has mechanisms related to Steps 1–12. Only Steps 1 and 2 have console smoke
commands, and those are L0 integration probes. The narrow historical L2 records do not close
the required retention, resource-matching, control-benefit, and uninterrupted L3 integration
gates. Completion at audit time was therefore 0/12, not “twelve implemented steps.”

## IPMNIST: real numbers, weak identity

The stored numerical record is internally consistent. Recomputing the aggregate from the 40
confirmation shards reproduces the current summary exactly (this reconstruction does not rerun
the benchmark):

```text
rls_head_resid_l1_preset005  20 seeds  mean 0.8711435286
sigma0_shiftnorm_d099        20 seeds  mean 0.8644904269
paired difference                              +0.0066531017
positive pairs                                 20 / 20
```

The LARP was in the surrounding claim and provenance layer:

- older append-only reports call `sigma0_shiftnorm_d099` the “final best” or record holder;
- `BEATS-SOTA` means only “beats this repository’s local selected-configuration UPGD-W
  development reproduction on reused development seeds”;
- the proxy’s own exactness flag is false for UPGD-W prefixes;
- v1 shards do not bind source, dataset bytes, command, RNG contract, or a content digest; and
- the same configuration name appears in legacy shards with materially different
  hyperparameters, so a name alone is not an experiment identity.

All 459 stored shards correctly say `development_only=true` and
`scientific_promotion_allowed=false`. The numbers may guide development; they cannot establish
a scientific or state-of-the-art claim. They remain v1 records: the immutable runbook's old
v1 validation/merge examples are historical and are not accepted by the current strict merge.

The implemented v2 successor binds arm fields and callback descriptors (normalized Python code
structure, defaults, closures, and canonical module origin), post-cast array bytes, a stable
package-wide Python disk snapshot plus available checkout metadata/lock documents, selected
runtime/JAX and RNG configuration, invocation fields, exact per-arm seed sets, and raw
input-shard bytes. It rejects duplicate JSON keys, non-finite values, type aliases, unexpected
fields, missing controls, source/environment drift visible as a pre/post identity mismatch, v1
strict merges, and summaries whose derived results do not reconstruct from their inputs. V1
parsing exists only behind the proxy audit's explicit quarantine flag.

That is strong consistency and merge-comparability, not authenticated evidence. V2 explicitly
records that its digest is an unkeyed self-hash, its invocation is self-declared, its source
identity does not attest the complete already-loaded module/global state, its lock document does
not attest the active environment matches that lock, and its input digest does not attest that
caller-supplied arrays semantically are MNIST. No v2 campaign output was created or promoted
during this audit.

The UPGD receipt has a related portability failure: its hash-bound local MNIST cache is ignored,
untracked, and absent, so live strict validation fails even though a stored narrative says it
passes. Future data inputs need a fetch-and-verify path or tracked immutable content-addressed
storage.

## Misreporting and metadata drift

The audit found these user-facing contradictions:

- the README’s plain `pip install alberta-framework` resolves to an independently published
  0.17.1 project requiring Python 3.13, not this 0.28.0 development fork;
- package URLs pointed at upstream while the actual remote and vendoring docs identify the
  divergent elizaOS/ASI fork;
- the package docstring claimed every component updates every step with no batch or special
  phases, despite public batched loops, replay, and periodic lifecycle work;
- current docs linked stale pre-RLS IPMNIST reports as if they were the latest campaign index;
- status text named shallow and partner-world-model modules removed by the cleanup; and
- a retired slowly-changing runbook still looked operational and referenced deleted commands.

The cleanup corrects mutable indexes and metadata. Checkout installation is now explicit,
fork/upstream URLs agree, optional research dependencies are separated from the base runtime,
Forager entry points import optional stacks lazily, and the built wheel/sdist exclude tests and
outputs. Append-only campaign records were not rewritten; a current campaign index labels
their authority and historical schema boundary.

## Genuine implementation gaps found and closed

### Recurring-IPMNIST sentinel inference

The recurring-retention runner accepted every registered screening arm but computed sentinel
performance by feeding a transformed input through the MLP logits path. The current confirmed
winner is an RLS-readout arm, so the advertised active path deterministically reached
`NotImplementedError`. Hidden-RMS, RFF/RLS, linear-RLS, naive-Bayes, naive-Bayes ensembles, and
the other RLS-head arms had the same structural mismatch.

The runner now has a per-arm frozen-logits capability. Hidden-RMS, RFF/linear-RLS,
naive-Bayes, naive-Bayes ensemble, and RLS-head arms expose their exact deployed pure forward
equations; the evaluator checks output shape, finiteness, and state immutability. Direct
equation tests cover the special paths. This closes the software defect but creates no new
scientific evidence.

### CLI contract bugs

The Step 1 smoke CLI rejected a short `--steps` value because its hidden fixed final window was
longer than the requested run. Both smoke commands now derive a bounded default window, while
an explicit invalid override still fails. The top-level public export list also contained one
duplicate; a uniqueness-and-resolution contract now catches this class of metadata inflation.

One evidence CLI returned exit 1 for invalid artifacts while the documented package convention
is exit 2. The manifest CLI could overwrite the pinned historical manifest even though the
agent guide calls it immutable. The CLI now returns exit 2 for invalid input, manifest writes
refuse every pre-existing target, and the continual-IA and multiagent public artifact writers
use exclusive creation with overwrite- and concurrent-writer tests.

Cleanup verification also caught ordinary regressions before handoff. The public
Step 10 facade now rejects an empty subtask set while the lower-level primitive-only STOMP
configuration remains valid; forced-option and near-counter-capacity fixtures now preserve the
exact ownership clocks required by the fail-closed transition checks; and SARSA config loading
again consumes and validates its serialized state schema. A class-scoped pytest fixture was
updated for pytest 10 compatibility. Finally, the installed OCI console command can render its
top-level and subcommand help from a shared dependency-light parser when an installer hard-links
wheel files, while every real operation still imports the source-attested implementation and
rejects external hard-link aliases. Each fix has a focused regression.

## What remains credible

The following parts survived because they do substantive work:

- JAX learners, normalization, optimizers, TD/control algorithms, Horde, explicit RNG, and
  immutable state transitions;
- the robot-imported continual-RL subset (`actor_critic`, `continual_backprop`, initializers,
  normalizers, optimizers, and SARSA);
- synthetic and closed-loop streams with bounded integration tests;
- retained feature, memory, world-model, option/STOMP/OaK, and Prototype mechanisms where
  executable contracts exist;
- the IPMNIST benchmark implementation and raw nonpromoting development shards;
- small versioned evidence validators that reject drift rather than silently blessing it;
- honest negative artifacts, quarantines, and the negative-results ledger; and
- the historical and matched-current Forager tools whose nonpromotion and execution state are
  explicit.

“Credible” here means implemented and appropriately classified. It does not mean every retained
mechanism has an empirical benefit result.

## Residual software risk assessed at audit time

The post-cleanup static pass found no remaining high-confidence private duplicate, stub-only
module, permanently skipped lane, orphan package module, or active `NotImplementedError` path.
The remaining base-optimizer `NotImplementedError` hooks are intentional abstract capability
fallbacks and are rejected at MLP construction sites. Optional-dependency and system-dependent
test skips are explicit. Public loop/export/compatibility APIs with no internal caller were
retained where they have a meaningful contract and plausible external consumers; absence of a
repository caller is not enough to delete a library API.

This does not justify a “0% risk” claim. A reasonable audit estimate is that **10–20% of the
retained surface remains uncertain or thinly exercised**, concentrated in the large
matched-current Forager orchestration stack, externally consumed public APIs, and mechanisms
with only bounded integration tests. That range is residual review risk, not confirmed LARP.
The matched-current campaign is especially infrastructure-heavy, but its records honestly say
zero performance cells have run, so it is unfinished science rather than a fabricated result.

Two provenance limits remain deliberately explicit instead of being papered over. Registered
evidence v1 source sets are manually enumerated rather than recursive execution closures. The
IPMNIST v2 package-Python disk snapshot, callback descriptors, and self-hash support consistency
checks but do not authenticate an execution or attest the complete loaded module/global state or
dataset semantics. Stronger claims would need fresh-process or signed external execution receipts
plus independently identified inputs; adding grander schema language would not supply those
facts.

V2 also validates exact seeds only for the arms actually supplied; it does not attest that a
larger preregistered candidate universe is complete. Summary input paths are deliberately
absolute and machine-location-bound. The merge/verification host is not recorded, and invocation
interface or progress-logging cadence is not a merge-compatibility field, so aggregated wall-clock
totals are operational diagnostics rather than a cross-setting speed comparison.

## Residual work that code alone cannot finish

Some requested completion work is experimental, not implementational. It cannot honestly be
closed by adding classes, schemas, or tests:

1. rerun or supersede the five invalid evidence claims under frozen v2 protocols with complete
   source/data/runtime closures and untouched preregistered seeds;
2. execute the 210-cell matched-current Forager open campaign, select on development data,
   freeze a sealed protocol, and run untouched evaluation seeds;
3. keep the active IPMNIST campaign permanently nonpromoting; any future scientific claim must
   be a separate frozen protocol with independently identified data and untouched seeds; and
4. close the L2/L3 comparison and integration gates listed for all twelve Alberta Plan steps.

Calling those items “implemented” before the runs exist would recreate the problem this audit
removes.

## Live follow-up commands

```bash
# Interpret evidence status from the current checkout.
.venv/bin/alberta-evidence-status

# Current inventory; these commands do not reproduce the audit-time counts.
find alberta_framework -name '*.py' -type f -print0 | xargs -0 wc -l
find tests -name '*.py' -type f -print0 | xargs -0 wc -l

# Current static and executable verification.
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest tests -q -m 'not slow'
.venv/bin/alberta-step1-smoke --steps 8 --seed 0
.venv/bin/alberta-step2-smoke --steps 8 --seed 0
```

Passing those checks establishes software consistency only. It does not promote a research
claim.
