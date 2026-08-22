# Canonical loss-of-plasticity diagnostics

This is a short, permanently nonpromoting diagnostic for a real hidden network.
It is not a reproduction, performance claim, or SOTA result.

## Audited authority and protocol

The paper is **Maintaining Plasticity in Deep Continual Learning**,
`arXiv:2306.13812v3` (9 April 2024). The later Nature publication and the
official repository use the title **Loss of Plasticity in Deep Continual
Learning**. The official code is `shibhansh/loss-of-plasticity` at current main
commit `a6b79580d85f3025bdb601566d3627c5f489f13b`, inspected 18 August 2026.
That tip postdates the paper and includes later fixes, including an Adam/GnT
change, so code-tip parity is not automatically paper parity.

The official MNIST experiment is **input-permuted**, not random-label MNIST. The labels
remain the digits 0–9. In each of 800 paper tasks, one pixel permutation is
shared by all 60,000 training images, the images are presented one at a time in
random order for one pass, and no task-switch indication is given to the
network. The official current code applies each new pixel permutation to the
already permuted array, so the adapter freezes the equivalent cumulative
permutation. The official-paper protocol cannot be relabeled as random-label MNIST.

The paper network has three hidden ReLU layers with 2,000 units, Kaiming
initialization, batch size one, SGD/cross entropy, up to 800 tasks, and 10 runs
in the checked standard-network config. Its CBP config uses learning rate
0.003, utility decay 0.99, maturity 100, adaptable-contribution utility, and
replacement rates 1e-4, 1e-5, and 1e-6.

## Bounded ASI slice

`asi-plasticity-diagnostic` consumes a caller-supplied NPZ containing exactly
`images` (float32 `[N,784]`, values in `[0,1]`) and `labels` (int32 `[N]`,
values 0–9). It never downloads or writes data. Its development profile uses
eight tasks, 64 examples per task, two hidden ReLU layers of width 64, learning
rate 0.003, maturity 100, and replacement rate 1e-4. This is a mechanism
diagnostic, not paper-scale parity: depth, width, task count, observations, and
run count differ.

The runner also exposes `independent-random-labels`, a #1583 protocol extension.
For every task it selects a Threefry-rooted example order from the same bounded
caller array and independently draws one label in `[0,9]` for every consumed
example. Pixels are not permuted in this arm. The dataset identity still binds
the complete caller images and original labels, even though original labels are
not learner targets in this protocol. This extension uses the same frozen seeds,
arms, updates, observations, queries, diagnostics, and resource accounting as
the input-permutation lane. It explicitly sets `paper_parity_claimed=false`.

The matched roster is SGD, CBP with replacement disabled, and bounded CBP.
Every arm gets identical Threefry-rooted permutations, example orders,
initialization, updates, observations, and task-hidden learner inputs. The
mechanism-off arm must reduce exactly to SGD in curves and final numeric state.
Task boundaries are used only by the stream constructor and per-task
diagnostics, never passed into a learning step.

The bounded CBP arm uses activation times mean absolute outgoing weight as its
adaptable-contribution proxy and permits at most one replacement per layer per
example. The official implementation includes additional age correction,
accumulation, and optimizer-state handling. This slice tests the causal
replacement mechanism; it is not an exact port or parity result.

Predictions are recorded before each online update. A second batched forward
pass per task measures dead ReLU units and representation effective rank.
Receipts count examples and bytes, prediction/diagnostic model queries,
parameter updates, replacements, and numeric learner-state persistent bytes.
The runner-owned Threefry cursor is outside that learner-state byte field;
elapsed nanoseconds remain telemetry-only. Every record also binds the
full caller array bytes, current ASI module bytes, Python, JAX, NumPy, and JAX
backend identities. Negative results must be retained and no result can promote.

```bash
.venv/bin/asi-plasticity-diagnostic --catalog
.venv/bin/asi-plasticity-diagnostic \
  --dataset /approved/mnist-train.npz \
  --profile bounded-development \
  --seed 15830
.venv/bin/asi-plasticity-diagnostic \
  --dataset /approved/mnist-train.npz \
  --task-protocol independent-random-labels \
  --profile bounded-development \
  --seed 15830
```

The `contract-smoke` default is a CI-sized mechanism exercise. Neither profile
is scientific evidence. The implementation is additive to issue #1578's
native supervised suite but intentionally separate because its cumulative
online protocol and hidden CBP network differ from that suite's generic linear
IPMNIST slice.

## Costly lanes remain closed

Continual ImageNet uses 2,000 randomly selected binary tasks, 32×32 images,
1,200 train and 200 test images per task, 250 epochs, batch size 100, SGD with
momentum 0.9, three reported step sizes, and 30 runs per hyperparameter. The
official README estimates 12 A100-hours per run. It also resets the output-head
weights at task boundaries, privileged information that must be matched or
ablated. Dataset licensing, exact byte/split identity, head-reset semantics,
accelerator budget, and isolated runtime are unresolved; execution is blocked.

The RL appendix/current project covers continual PPO on MuJoCo tasks including
Slippery Ant. The README says 50 million steps and about 24 CPU-hours per run,
while the current `Ant-v3` standard config says `n_steps: 1e8`. Runtime/MuJoCo
versions, this step discrepancy, environment sequence, seeds, and budget remain
unresolved; execution is blocked.

Before any scientific comparison, reproduce the full three-layer, width-2,000,
800-task MNIST protocol in a qualified accelerator runtime; verify the official
CBP utility and replacement traces against the pinned code; add a from-scratch
per-task ceiling and L2/shrink-and-perturb controls; freeze untouched seeds;
and separate plasticity from retention. JIT parity and the bounded diagnostic
do not close those gates.
