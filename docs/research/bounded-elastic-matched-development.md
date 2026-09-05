# Bounded elastic matched development

Issue #1562 compares the existing bounded structure-off, bounded growth, bounded elastic, and
fixed-capacity CBP arms under one learner-owned persistent-memory and final-size budget. This is
an ASI fixed-shape
IPMNIST adaptation of `arXiv:2608.01475v1`; the paper discloses no official code repository, and
the protocol records the architecture, task-length, input-scaling, boundary, and pruning-sample
differences. It is not a paper reproduction.

The prospective plan binds the canonical materialized OpenML MNIST training-array digests,
current source/runtime identity, exact `pyproject.toml` and `uv.lock` bytes, an exact 8-task by
5,000-example configuration, all four arms, and five repository-history-checked campaign seeds.
The catalog records the distinct test-only roster; the two literal rosters are disjoint, and the
authorization gate rejects the campaign roster even through private helpers while authorization
remains false.
Each result retains observations, updates, data and environment steps, model queries, persistent
bytes, peak budget, active final size, structural events, and telemetry-only timing. One static
256 MiB accounting envelope covers one caller-owned C-contiguous host dataset, the exact
task-length schedule, and peak learner-persistent bytes. It is checked before schedule
construction, parameter initialization, or execution; backend copies, compiler state, gradients,
and transient execution buffers are explicitly outside that arithmetic envelope. The plan also
charges the transaction's 20 initial runner dispatches and 20 strict reexecution dispatches:
1,600,000 observations, optimizer updates, and data steps plus 3,200,000 model queries in total.

The preregistered primary comparison is paired whole-stream mean online accuracy for
`bounded_growth` and `bounded_elastic` against `bounded_fixed_cbp`. A candidate is supported only
when all five paired deltas are strictly positive, rejected only when all five are nonpositive,
and otherwise inconclusive. The campaign is supported when either candidate is supported,
rejected when both are rejected, and otherwise inconclusive. This conservative sign rule is a
development-selection outcome only; it is not a significance test or scientific evidence.

Standalone execution, strict reexecution, and publication are permanently disabled. One public
run-and-publish transaction remains hard-disabled until both a separately reviewed source transition
and runtime authorization become exact `true`. The reviewed plan records both authorization fields
as literal `false`; a later transition cannot retroactively change that plan identity. The future
output path is NEW. Before any dataset load or runner dispatch, the transaction reserves the path
through per-segment no-follow directory descriptors. It strictly rereads and validates an unnamed
staged inode before linking it into the output namespace. Post-link failure removes only that exact
inode. A failure after the first runner dispatch permanently retains the inode-owned marker as a
consumed-without-result tombstone, so the five-seed roster cannot be retried; a successful linked
output is itself the immutable retention barrier before the marker is released. Publication is
create-only and fsynced. A failure before the first dispatch has no structured receipt and releases
the reservation. Process death leaves the owned reservation marker in place and also prevents an
implicit retry. Source, runtime, and dataset identities are checked again after both initial
execution and strict reexecution. Every result is development-only and permanently nonpromoting;
negative, inconclusive, supported, and rejected development outcomes remain subject to retention.

The read-only catalog is available without loading MNIST:

```bash
.venv/bin/python -m alberta_framework.evaluation.bounded_elastic_matched_runner --catalog
```

No campaign run or result artifact is included with this plan.
