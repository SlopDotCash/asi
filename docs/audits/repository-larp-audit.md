# Repository anti-LARP audit

**Audit date:** 2026-08-14–15 (America/Los_Angeles)
**Baseline revision:** `22739da0b5d8a06a621b6743297f1c46d8d87903`
**Production base:** `e1328798a676175cdf65cfcbf7fe6b1226d10b68` plus the audited forward-port
worktree
**Scope:** implementation, tests, package metadata, documentation, registered evidence, and
the active IPMNIST and Forager development records

## Bottom line

The pre-cleanup repository had a very large research-theatre surface. The production line
retains roughly **27% of the baseline package-and-test Python lines**, and the audit judges
roughly **70–80%** of the old surface to have been unsupported, redundant, unissued,
self-certifying, or superseded. Those are related but different statements: the physical
reduction is an exactly reproducible inventory ratio,
while calling the removed surface “LARP” is a documented audit judgment. It is not a measured
percentage of false mathematical statements, fabricated experimental values, or known LLM
authorship.

The strongest measurements are:

| Measure | Result | Meaning |
|---|---:|---|
| Package and test Python before cleanup | 1,264,156 lines in 1,192 files | Audited baseline at `22739da` |
| Package and test Python retained | 343,781 lines in 373 files | Final audited forward-port inventory |
| Python surface removed | **72.81% by physical lines; 68.71% by files** | Exact inventory ratio; LARP classification is the audit judgment |
| Single `3d195c3` “updates” commit | 758,229 insertions in 849 files | Largest concentrated expansion |
| Current-cleanup deletions born in that commit | 680,481 pre-cleanup lines across 643 files | Exact provenance cohort, not every deleted file |
| Alberta Plan steps complete | **0/12** | Mechanism presence is not Step completion |
| Live registered claims valid | **0/5** | Registry exits 2; frozen outcomes are historical records only |
| IPMNIST promotion-grade claims | **0** | The campaign explicitly forbids promotion |
| Historical matched-current Forager cells executed | **0/210** | The immutable v1 roots were prepared, never run, and are source-incompatible now |

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

## How the percentage was derived

The physical pruning ratio compares the pre-cleanup `22739da` tree with the retained
production-forward-port working tree:

```text
22739da package Python:    833,938 lines / 492 files
22739da test Python:       430,218 lines / 700 files
combined:                1,264,156 lines / 1,192 files

retained package Python:   216,669 lines / 165 files
retained test Python:      127,112 lines / 208 files
combined:                  343,781 lines / 373 files
```

That is 920,375 lines (72.81%) and 819 files (68.71%) removed. Physical removal is not
independently a semantic classification: the audit reached its 70–80% judgment by reviewing
callers, tests, execution records, evidence authority, and narrative claims while deciding
what to remove or retain. The range is a sensitivity judgment, not a statistical confidence
interval. A per-file classification manifest would be required to turn it into a separately
recomputable semantic percentage.

Git history supplies a separate provenance check. Commit `3d195c3` (2026-08-12), whose entire
subject was `updates`, added 758,229 lines: about 510,000 under `alberta_framework/` and
238,000 under `tests/`. Intersecting Python paths added by that commit with paths present at
`22739da` and absent from the retained tree identifies exactly 643 files containing 680,481
physical baseline lines. This concentration, repeated generated naming, and declaration-heavy tests are
consistent with LLM-amplified code generation. They cannot prove who or what authored the
text, so this audit does not assign an “LLM-written” percentage and does not call that cohort
an objective lower bound on LARP.

The 72.81% figure also does **not** mean the retained 27.19% is scientifically validated. Scientific
completeness is measured separately: 0/12 complete Plan steps and 0/5 live-valid registered
claims.

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
smaller matched-current implementation. Its only qualification/open records are historical v1
roots with an explicit zero-cell execution state; those immutable roots bind `2c3b214c`-era
source and are not resumable with the current builder.

Preparation is still not performance evidence. The historical open campaign has 210 planned
cells and no scored results. Its qualification snapshot reads no reward arrays, uses no tuning
seeds, and explicitly sets both `performance_claim` and `promotion_authorized` false. Any new
run needs fresh current-source qualification and a new output namespace.

### Self-certifying tests and evaluation islands

Removed tests included broad public-export presence matrices, source-presence manifests, “not assessed”
matrices, replay-only gates, permanently skipped suites, and development campaigns that chose
their own thresholds. One focused package-export uniqueness/resolution check remains as a release
metadata contract. The removed matrices inflated counts without increasing confidence in behavior.
Retained tests target numerical updates, state transitions, serialization, CLI/process
boundaries, strict validators, and bounded integration behavior.

### Superseded research lanes

The slowly-changing-regression harness, UPGD-IPMNIST v3 governance layer, speculative WP0–WP9
documents, and other concluded lanes were retired. Their immutable or append-only output
records remain in place, and reusable negative conclusions remain in
`docs/evidence/negative-results.md`.

## Current scientific truth

### The live registry supports zero current claims

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
gates. Current completion is therefore 0/12, not “twelve implemented steps.”

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
a scientific or state-of-the-art claim. They remain v1 records. The immutable runbook's old v1
examples are historical; the current implementation keeps homogeneous v1 shards readable and
mergeable into v1 summaries, but those summaries acquire no v2 provenance by being reprocessed.

The implemented v2 successor requires the CLI to start from a clean Git checkout and binds the
commit/tree, all tracked package bytes plus `pyproject.toml` and `uv.lock`, canonical materialized
MNIST bytes, and selected Python/JAX/package/device/process configuration. It checks those
bindings again before exclusive shard publication. Merge rejects mixed schemas and incompatible
source, dataset, runtime, protocol, or noise contracts; requires a present control; rejects
duplicate arm/seed shards; records the input-shard bytes and paths; and rechecks the derivation
context and inputs before publishing a v2 summary. Strict JSON, metric-domain, registered-arm,
seed, pool-mode, and dataset-shape checks fail closed.

That is strong consistency and merge-comparability, not authenticated evidence. V2 has no
signature, independently issued execution receipt, or artifact-level content signature. It does
not prove that already-loaded code or mutable process globals match disk, or that the active
environment conforms to `uv.lock`. The CLI binds the canonical MNIST source selection and bytes,
but library-created provenance mappings remain self-declared. There is no standalone strict v2
summary reload/reconstruction API, and supplied paths can be working-directory-dependent. No v2
campaign output was created or promoted during this audit.

The UPGD receipt has a related portability failure: its hash-bound local MNIST cache is ignored,
untracked, and absent, so live strict validation fails even though a stored narrative says it
passes. Future data inputs need a fetch-and-verify path or tracked immutable content-addressed
storage.

## Misreporting and metadata drift

The audit found these user-facing contradictions:

- the README’s plain `pip install alberta-framework` resolves to an independently published
  0.17.1 project requiring Python 3.13, not this 0.29.0 development fork;
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
an explicit invalid override still fails. A new uniqueness-and-resolution release contract now
guards the top-level public export list against metadata inflation.

One evidence CLI returned exit 1 for invalid artifacts while the documented package convention
is exit 2. The manifest CLI could overwrite the pinned historical manifest even though the
agent guide calls it immutable. The CLI now returns exit 2 for invalid input, manifest writes
refuse every pre-existing target, and the continual-IA and multiagent public artifact writers
use exclusive creation with overwrite- and concurrent-writer tests.

Final sharded verification also caught ordinary cleanup regressions before handoff. The public
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

## Residual software risk after cleanup

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
IPMNIST v2 clean-Git source binding and derivation manifests support consistency checks but do
not authenticate an execution or attest already-loaded code, mutable process globals, or active
lock conformance. Stronger claims would need fresh-process or signed external execution receipts;
adding grander schema language would not supply those facts.

V2 validates the arms and seeds actually supplied; it does not attest that a larger
preregistered candidate universe is complete. Supplied input paths may be relative and
working-directory-dependent. The merge/verification host and invocation interface are not a
complete execution identity, so aggregated wall-clock totals are operational diagnostics rather
than a cross-setting speed comparison.

## Residual work that code alone cannot finish

Some requested completion work is experimental, not implementational. It cannot honestly be
closed by adding classes, schemas, or tests:

1. rerun or supersede the five invalid evidence claims under new versioned frozen protocols,
   each with a new schema/path as required, complete source/data/runtime closures, and untouched
   preregistered seeds;
2. issue a fresh current-source Forager qualification in a new output namespace, execute its
   new matched-current open campaign, select on development data, freeze a sealed protocol, and
   run untouched evaluation seeds without modifying or resuming the historical v1 roots;
3. keep the active IPMNIST campaign permanently nonpromoting; any future scientific claim must
   be a separate frozen protocol with independently identified data and untouched seeds; and
4. close the L2/L3 comparison and integration gates listed for all twelve Alberta Plan steps.

Calling those items “implemented” before the runs exist would recreate the problem this audit
removes.

## Reproduction commands

```bash
# Current registered evidence; expected to exit 2 until new evidence is issued.
.venv/bin/alberta-evidence-status

# Code and test inventory.
find alberta_framework -name '*.py' -type f -print0 | xargs -0 wc -l
find tests -name '*.py' -type f -print0 | xargs -0 wc -l
git ls-tree -r --name-only HEAD -- alberta_framework tests

# Exact removed cohort introduced by 3d195c3 and present at the audited baseline.
comm -12 \
  <(git diff-tree --no-commit-id --name-only --diff-filter=A -r 3d195c3 -- \
      alberta_framework tests | rg '\.py$' | sort) \
  <(git ls-tree -r --name-only 22739da -- alberta_framework tests | \
      rg '\.py$' | sort) |
while IFS= read -r path; do
  if [ ! -f "$path" ]; then git show "22739da:$path" </dev/null | wc -l; fi
done |
awk '{lines += $1; files += 1} END {print lines, files}'

# Static and executable verification.
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest tests -q -m 'not slow'
.venv/bin/alberta-step1-smoke --steps 8 --seed 0
.venv/bin/alberta-step2-smoke --steps 8 --seed 0
```

Passing those checks establishes software consistency only. It does not promote a research
claim.
