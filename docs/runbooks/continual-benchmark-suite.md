# Continual benchmark suite setup

The machine-readable entry point is:

```bash
.venv/bin/asi-benchmark-catalog list
.venv/bin/asi-benchmark-catalog doctor
.venv/bin/asi-benchmark-catalog doctor ipmnist-iclr2024 reference-life
```

`list` describes comparison protocols and pinned upstream revisions. `doctor` performs only
read-only host checks; it never downloads a dataset, installs a simulator, accepts a license,
or executes third-party code. A nonzero `doctor` exit means at least one requested benchmark is
not currently runnable as an integrated ASI lane. Catalog and smoke success are development
infrastructure, not scientific evidence.

## Why there are three setup modes

- **native** benchmarks share ASI's Python 3.12/JAX environment and must preserve the package's
  NumPy floor and import surface;
- **optional** benchmarks use a declared project extra, such as `forager`; and
- **isolated** benchmarks get a container or separate locked environment plus an adapter artifact.

Do not install isolated suites into `.venv`. The audited upstreams currently include mutually
incompatible stacks: Continual World uses TensorFlow and legacy `mujoco-py`; CORA uses legacy Gym
and Atari; COOM's learning extras pin TensorFlow 2.11; the loss-of-plasticity repository pins
Torch 2.1, Gym 0.23, and NumPy 1.24; and DreamerV3 pins an older JAX/NumPy combination.

## Initial benchmark ladder

| Stage | Benchmarks | Purpose |
|---|---|---|
| L0 | reference-life, IPMNIST | Fast contract and current plasticity/control anchors |
| L1 | Split/Rotated MNIST, random-label MNIST | Retention, plasticity, and protocol diagnostics |
| L2 | Split CIFAR-100, CLEAR10 | Representation and real temporal-distribution shift |
| L3 | Forager, one CORA family, COOM | Partial observation and embodied continual control |
| L4 | Continual World CW20 | Robot-task transfer, forgetting, and resource scaling |
| WM | state-control pilot, DMC/Crafter/Atari | World-model utility and sample efficiency |

The order is a cost ladder, not a claim that the earlier suites are more important.

## Qualification contract for an isolated suite

Before an adapter may run a comparison, add a lock that records:

1. repository URL and full commit SHA;
2. Python, OS/container, accelerator, simulator, ROM/asset, and dependency identities;
3. dataset or asset terms and cryptographic digests where redistribution permits them;
4. task sequence, boundary/task-ID access, reset semantics, step budget, and evaluation cadence;
5. observation/action wrappers and RNG schedule;
6. primary metrics plus exact resource counters; and
7. a deterministic fixed-action or fixed-data smoke trace.

The adapter must emit those fields into every result. Missing identity fails closed. Benchmark
commands write to new paths and never run inside pytest.

## Current setup state

- **Runnable native:** reference-life; IPMNIST when research dependencies and MNIST data are
  available.
- **Runnable optional:** Forager after installing the `forager` extra and satisfying its locks.
- **Scaffolded, adapter pending:** Split/Rotated MNIST, Split CIFAR-100, loss-of-plasticity,
  Continual World, CORA, COOM, and DreamerV3.
- **Dataset qualification pending:** CLEAR10.

Use the corresponding GitHub issue before implementing an adapter. Passing a smoke trace moves a
suite from scaffolded to integrated; it does not make a result promoted evidence.
