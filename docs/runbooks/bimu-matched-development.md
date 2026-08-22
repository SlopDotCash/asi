# BiMU bounded matched development campaign

This campaign compares the current binary Bayesian learner against its exact
mechanism-off reduction. It is a five-task, 256-train/256-test, width-32
development slice. It is permanently nonpromoting, is not paper-comparable,
and cannot support a scientific, SOTA, or paper-reproduction claim.

## Official-source qualification

The read-only catalog can be inspected without importing or executing the
external implementation:

```bash
.venv/bin/python -m alberta_framework.benchmarks.bimu_external_qualification
```

It binds official commit `1b8a1a1fb892fbe89401390b3ff9611d7f3a5168`, Git tree
`cbeeb50cdd3421fc046e7a2b73e26147419227e9`, the observed source archive,
CC BY 4.0 license bytes, and the exact configuration, optimizer, model, layer,
data, training, and environment files used to audit the ASI lane. The archive
was downloaded only for this read-only audit; its code was not imported or
executed.

The official `environment.yml` pins Python and PyTorch but leaves several
dependencies and the Conda base/channel resolution mutable. It is therefore
content-bound as an input file but is not a qualified runtime lock. Official
dataset bytes and downloader behavior also remain unbound. External execution,
paper parity, and scientific promotion remain explicitly unauthorized.

The literal plan fixes OpenML `mnist_784` version 1 through the canonical
60,000-row loader, the first 256 scaled rows as training data, the last 256 as
a disjoint development test, seeds 157001–157003, and arms `memory_off` and
`bimu`. The two arm configurations differ only in `memory_window` (`None`
versus `128`). Every other configuration, schedule, initial state, counter,
and numeric resource field must match within each pair.

Seeds 157001–157003 are publicly exposed by this literal development plan and
are therefore consumed for every promotion purpose. They have not produced a
retained matched result. Any future scientific protocol needs a new path and
untouched seeds; changing the execution gate cannot make this roster eligible.

The preregistered primary outcome uses the final-model mean accuracy over the
five task permutations. It is `supported` only when all three candidate-minus-
control deltas are strictly positive, `rejected` when all three are
nonpositive, and `inconclusive` otherwise. The whole-stream online metric is
secondary and cannot change that classification. Timing is telemetry only.

## Review and execution gate

`EXECUTION_AUTHORIZED` and `AUTHORIZATION_TRANSITION_APPROVED` are frozen to
`False` in
`alberta_framework/evaluation/bimu_matched_nonpromoting.py`. The `run-shard`
command and every publisher fail before plan, data, replay, or execution work
unless both literals are exact `True`. A separate reviewed authorization change
must open both gates; both values are bound into the plan, identity, policy,
completed and failed shards, and status. That source change also
changes the plan's source identity, so the plan must then be published from
the authorized revision before any shard starts. That review must also update
the literal `FROZEN_PLAN_SHA256`; an authorization flip without the matching
reviewed plan digest fails closed.

Inspect the currently unauthorized plan without publishing an artifact:

```bash
.venv/bin/python -c 'import json; from alberta_framework.evaluation.bimu_matched_campaign import build_plan_document; print(json.dumps(build_plan_document(), sort_keys=True))'
```

Every mutating CLI command accepts only the registered repository root and
fails while either authorization gate is closed. Do not invoke `plan --root .`
until a reviewed authorization transition is ready to publish the source-bound
plan in the new immutable namespace.

If any file already exists in `development.v1`, the authorization change must
advance the namespace rather than replace that file.

After separate authorization, launch each of the six commands in its own fresh
Linux process. Substitute each frozen arm and seed exactly once:

```bash
.venv/bin/asi-bimu-matched-development run-shard \
  --root . --arm memory_off --seed 157001
```

Once all six fixed shard paths exist, summarize and revalidate:

```bash
.venv/bin/asi-bimu-matched-development summarize --root .
.venv/bin/asi-bimu-matched-development validate --root .
```

All files publish without replacement under the registered repository's exact
`outputs/bimu_matched/development.v1/` namespace. Plan and aggregate work is
reserved before validation, data loading, or strict replay; reservation cleanup
compares held and live inode identities. Keep supported, rejected, and
inconclusive aggregates. The runner attempts to publish one generic
failed-attempt receipt when plan admission, shard execution, or prepublication
strict dataset-bound reexecution raises an ordinary `Exception`. The dataset is
loaded once, and the same validated arrays feed the initial execution and
strict reexecution immediately before completed-shard publication. The
publisher does not accept caller-asserted replay evidence. A successfully
published receipt returns nonzero, retains no exception type, message, or
representation, cannot enter aggregation, and forbids retry in this namespace.
After the first learner dispatch, the runner attempts to convert its inode-owned
reservation into a `consumed-without-result` tombstone when a `BaseException`,
completed-result publication failure, or failure-receipt publication failure
escapes. A retained tombstone prevents retry in this namespace. Tombstone I/O
failure, another asynchronous interruption during that conversion, and process
death remain outside this retention boundary. A pre-dispatch failure that
escapes without a published failed-attempt receipt releases the reservation.
Source,
runtime, dependency, process, dataset, resource, and telemetry digests are
consistency bindings, not authenticated execution attestation.

The frozen plan accounts for both learner passes rather than treating strict
admission as free validation work. Each shard performs one initial execution
and one strict reexecution over the same validated array tuple. Across the six
fresh-process shards this is 12 learner dispatches, 15,360 observations, label
queries, optimizer updates, optimizer-seen steps, and environment steps, plus
122,880 model-forward queries. Each shard process loads its dataset once. The
existing per-arm result counters describe one retained result; the separate
transaction accounting binds the doubled work actually required for admission.
Replay timing is not retained and timing remains unqualified telemetry only.
