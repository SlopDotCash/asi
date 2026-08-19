# Normalize-and-Project bounded comparator

This is an executable, permanently nonpromoting development comparator for
issue #1564. It integrates the existing `nap_project` primitive into the real
bounded hidden-network input-permuted-MNIST lane introduced by #1583. It is not
a reproduction, performance result, scientific evaluation, or SOTA claim.

## Primary-source audit

The authoritative paper is the final NeurIPS 2024 proceedings version of
*Normalization and effective learning rates in reinforcement learning*, paper
identity `c04d37be05ba74419d2d5705972a9d64`; its arXiv record is
`arXiv:2407.01800v1`.

NaP inserts normalization before each nonlinearity, projects each hidden weight
matrix to its initial norm, does not project the final non-scale-invariant
output, and treats learned scale/offset parameters separately. The paper says
projection every update and every 1,000 Rainbow steps gave nearly identical
results. It removes biases made redundant by normalization and discusses joint
scale/offset normalization or regularization. Those details are load-bearing.

The paper cites DQN Zoo as the Rainbow baseline implementation, but it neither
discloses nor pins a method-specific official NaP repository. The audit found no
official NaP code, so the catalog records official code as absent and gives the
DQN Zoo baseline no invented commit. A later independent MIT-licensed research
library, Plasticine, contains a secondary implementation and is pinned only as
non-official context at
`RLE-Foundation/Plasticine@aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8`.

## ASI development protocol

The lane explicitly depends on #1583 commit
`8383d6438b81c7620189c6fedba30c345994cb12`. It uses that lane's caller-supplied
MNIST validation, cumulative pixel-permutation schedule, two-hidden-layer
predict-before-update network, frozen development profiles, and per-example
updates. One material deviation is that this comparator uses unlearned
LayerNorm statistics and retains unprojected biases; it has no learned
scale/offset pair. It projects the two hidden weight matrices after every
update, never the output matrix.

Five matched arms receive the same frozen seed, scheduled examples, labels,
update count, observation count, and task-hidden learner inputs:

- `sgd_current_control`: the exact current #1583 SGD lane;
- `nap_mechanism_off`: the NaP runner with both mechanisms disabled;
- `normalization_only`: normalization before both hidden ReLUs;
- `projection_only`: hidden-weight projection without normalization;
- `nap`: normalization plus hidden-weight projection.

The current control and mechanism-off arm must match exactly in every curve,
final state digest, norm, and non-timing receipt. Task boundaries are used only
for reporting; neither boundary nor task identity enters the learner.

Receipts include data steps and bytes, observations, model queries, updates,
normalization calls/elements, projection events/tensor queries/elements,
logical forward and gradient multiply-accumulates, auxiliary logical scalar
operations, state and projection-target bytes, and elapsed nanoseconds. Logical
compute follows the declared dense-shape formulas; it is not a hardware FLOP
measurement. Timing is telemetry-only and cannot select or promote an arm.
Every valid negative, tie, and regression must be retained in a new
nonpromoting path. The CLI itself writes no `outputs/` data.

```bash
.venv/bin/asi-nap-ipmnist --catalog
.venv/bin/asi-nap-ipmnist --dataset /path/to/mnist.npz --seed 15640
```

## Paper protocol differences and closed gates

The main supervised experiment uses CIFAR-10, 20 million steps, and 200 random
target resets; the paper reports Adam with learning rate `1e-4` for that
continual experiment. The sequential ALE experiment uses the DQN Zoo Rainbow
baseline, 20 million frames per game, and a cosine schedule restarted at every
task change with a `1e-8` initial value, `0.000625` peak, 1,000-step warmup, and
`1e-6` end value. The current lane implements none of those protocols.

Paper parity remains closed until an immutable method source is available or an
independent implementation is fully specified; CIFAR/ALE data and runtimes are
qualified; architecture, bias, scale/offset, optimizer, projection, evaluation,
and schedule semantics match; compute, memory, and query budgets are matched;
strong paper controls and causal ablations run; and a separately preregistered
fresh-seed scientific evaluation is completed. This lane supports no claim of
NaP parity, benefit, sustained plasticity, robotics readiness, or SOTA.
