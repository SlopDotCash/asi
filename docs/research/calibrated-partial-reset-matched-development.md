# Calibrated Partial Reset matched development plan

This is the prospective, permanently nonpromoting plan for issue #1563. It does not
authorize a run, contain a result, close the issue, or create scientific evidence.

## Mechanism and provenance

The hypothesis is that scaling a partial pull toward initialization by a hidden unit's
incoming-gradient utility preserves useful features while restoring low-utility capacity.
The paper is *Calibrated Partial Resets* (`arXiv:2607.24996v1`). The audited official
implementation is `LucMc/continual-learning` at
`6fc2af34783159f5dda50c6915dda32c2d443604`.

The official implementation periodically normalizes each hidden layer's mean absolute
incoming-weight gradients, maintains an exponential utility trace, pulls incoming weights
toward a newly sampled initialization, and pulls outgoing weights toward zero. This ASI
IPMNIST adaptation instead uses per-parameter rather than per-neuron utility, the run's
retained seed initialization rather than fresh reset draws, and pulls every parameter. It
uses the authoritative batch-size-one normalized-SGD CPR family already registered in
`ipmnist_screening`, and evaluates online permuted MNIST rather than the paper's
reinforcement-learning suites. It therefore tests the mechanism in ASI and does not claim
paper-protocol parity.

The five matched arms are mechanism-off normalized SGD, utility-scaled CPR, utility-free
uniform partial reset, continuous L2 pull to initialization, and thresholded hard reset.
Unit tests pin the mechanism-off trajectory and show that each named reduction changes the
end-to-end trajectory.

## Frozen matching and resources

The unexecuted replacement campaign roster is `1563260101` through `1563260105`.
Repository and Git history searches found these seeds globally absent before this plan
exposed them. Tests use the separate `301` through `305` roster and never execute campaign
seeds.

The authoritative lane measures per-parameter utility and pulls all parameters toward the
retained initialization. Its post-update clock pulls first on update 100. The official
pre-update positive-clock convention would pull one update later; this timing and reset-axis
difference is frozen rather than silently presented as parity.

Every arm shares seed-derived initial parameters and example schedule, 8 tasks of 5,000
updates, observations, current-example labels, and no task-boundary signal. Each initial
row is strictly reexecuted before publication. The 25 rows and 25 reexecutions total 50
runner dispatches, 2,000,000 observations/data steps/updates, zero environment steps, and
4,000,000 model queries.

The prospectively frozen primary question is whether utility-scaled CPR improves each
seed's mean online accuracy relative to mechanism-off normalized SGD. The report retains all five
paired `utility - off` deltas and their exact `math.fsum` mean. It advances only to another
nonpromoting development follow-up when the mean delta is positive and at least four of
five seed deltas are positive; otherwise the outcome is `do_not_advance`. The utility-free,
L2-init, and hard-reset reductions are descriptive ablations and cannot change that primary
outcome. This directional development rule has no scientific threshold or promotion force.

The static numeric envelope counts one caller-owned C-contiguous float32 dataset and int32
labels, one materialized permutation/index schedule, and the peak retained learner state:
live parameters, retained initialization target, utility traces, normalizer, and step
counter. Its 256 MiB ceiling excludes backend/compiler copies, gradients, and transient
execution buffers; those exclusions prevent this byte envelope from being presented as
total process memory. Timing is bounded telemetry only and cannot select an outcome.

The report binds exact source bytes, including `pyproject.toml` and `uv.lock`, installed
direct dependency versions, Python/platform identity, JAX backend/config/device inventory,
selected JAX environment, canonical dataset digests, and per-seed initialization and
schedule digests. Validators require exact bounded JSON containers and scalars and reject
resource, arithmetic, roster, identity, reexecution, or policy drift.

## Authorization and publication

Two separate literal flags are frozen `False`. Public execution fails before output
reservation, dataset loading, or runner dispatch. A later reviewed transition must change
both literals. The single run-and-publish transaction pins each output-directory component
with `O_DIRECTORY|O_NOFOLLOW`, acquires the canonical NEW-path reservation with `O_EXCL`
before dataset work, and publishes a strictly validated anonymous inode by no-replace link
plus directory `fsync`. After the first runner dispatch, any failure leaves an inode-owned
`consumed-without-result` tombstone, permanently preventing retry at that namespace.

All accepted and negative outcomes remain development-only and nonpromoting. Negative
outcomes are retained. No result can populate `reference-dev`, support a performance or
scientific claim, or authorize promotion without a separate protocol and untouched seeds.
