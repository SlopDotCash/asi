# AdaLin matched PMNIST development lane

This lane integrates the existing AdaLin learner into one matched,
permanently nonpromoting PMNIST transaction. It compares learnable per-neuron
alpha against the exact alpha-zero ReLU reduction using consumed development
seeds `1571001`–`1571003`.

Both arms receive identical caller-supplied data, pixel and example schedules,
non-alpha parameter initialization, observations, labels, update counts, and
evaluation queries. The learner receives neither task identity nor boundary
events. The strict matched validator reloads both complete arm receipts,
requires their dataset, configuration, schedule, shared initialization, and
resource axes to agree, and derives the whole-stream pre-update accuracy delta.
There is no claim threshold and no automatic reference selection.

The arm receipt counts data steps, observations, label queries, optimizer
updates, model queries, forward calls, persistent parameter and alpha bytes,
and telemetry-only wall time. The matched receipt reports exact sums across
both arms. Alpha state remains allocated in the mechanism-off arm but is
initialized to zero, has identically zero gradients, and remains exactly zero.

This is not the paper's 400-task result, a reproduction, scientific evidence,
or a promotion protocol. The official pinned repository contains no runnable
implementation or experiment configuration. ASI uses caller-supplied data and
does not know the paper's exact MNIST sample selection, permutation seeds, or
dataloader order. Any retained campaign result requires a separately reviewed
transaction, canonical dataset identity, explicit execution authorization,
and a new append-only output path.
