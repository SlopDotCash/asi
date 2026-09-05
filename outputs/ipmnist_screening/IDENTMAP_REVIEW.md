# Identifier integration review

This records a code and artifact consistency review of the identifier branch.
It does not rerun its long MNIST experiments or authenticate contributor
execution. The retained confirmation shards bind historical source identities;
merging the implementation does not make those identities current or promote
the records to scientific evidence, reference-dev, or whole-life performance.
Seeds 0–19 are consumed development seeds.

The two confirmation summaries in `identmap_confirm_r1/` and
`identmap_star_confirm_r1/` reconstruct from their respective 40 shards: arm,
seed, workload, dataset and source identities agree, and the per-task accuracy
curves reproduce the means, standard errors and positive paired differences.
Their existing shard and summary bytes remain unchanged.

The deployed identifier consumes raw observations and previously supplied
labels. The current prediction and learner update run using the previous
remap before the current label enters the fingerprint. Its shift detector
receives no task index or true permutation. A new assignment affects the
next observation. The V7/V8 oracle experiments remain explicitly privileged
mechanism probes and are not the deployed identifier.

Resources are not matched or qualified by these records. At the default
784-input/10-class geometry, the identifier adds 84,757 persistent numeric
payload bytes to its inner learner, measured from the initialized JAX state
leaves (excluding Python/runtime/allocator overhead). The host assignment
forms a 784-by-784 float32 cost array (2,458,624 bytes) plus fingerprint,
linear algebra and SciPy solver workspaces; this is not a peak-memory bound.
Hungarian matching adds host computation and device/host synchronization at
each match, with cubic worst-case assignment complexity in input dimension.
The detector can trigger within a task, so nominal matches per task are not
an enforced execution budget. Existing wall-clock records are telemetry and
cannot establish a resource-normalized win or a latency guarantee. A matched
resource-accounted downstream/control experiment remains open.
