"""Ceiling-analysis measurement runs for the IPMNIST screening lane.

Development-only diagnostic (never promotable evidence). Reuses the exact
screening machinery (specs, seed derivation, schedule, per-step RNG chain)
from alberta_framework.benchmarks.ipmnist_screening, but records PER-STEP
online accuracy so within-task transients can be decomposed.

Modes:
  stationary  -- n_tasks=1, identity permutation (unpermuted MNIST), fresh net:
                 per-task ceiling for a from-scratch online learner.
  carried     -- champion arm, SAME permutation every task (no non-stationarity):
                 late-task accuracy = ceiling with full feature reuse.
  full        -- champion arm, normal protocol (200 tasks), per-step recording:
                 error-budget decomposition. Per-task means cross-checked
                 against outputs/ipmnist_screening/confirm_full shards.
  batch       -- converged minibatch-Adam 300x150 MLP on MNIST, test accuracy:
                 the architecture reference (~0.98).

Writes JSON + uint8 .npy per-step accuracy arrays to
outputs/ipmnist_screening/ceiling/ (new, non-pinned directory).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import screening_spec
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

OUT = Path(__file__).resolve().parent


def _load_train() -> tuple[np.ndarray, np.ndarray]:
    return load_mnist_train(default_openml_data_home())


def run_arm_per_step(
    spec_name: str,
    seed: int,
    n_tasks: int,
    perm_mode: str,  # "protocol" | "same" | "identity"
    progress_every: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one arm recording per-step accuracy.

    Mirrors run_screening_config's seed derivation, schedule, init, and
    per-step RNG chain exactly (protocol perm mode reproduces the confirm_full
    shards' per-task means).

    Returns (per_step_accuracy uint8 [n_tasks, task_length],
             per_task_accuracy float64 [n_tasks], wall_seconds).
    """
    config = IPMNISTConfig(n_tasks=n_tasks)
    spec = screening_spec(spec_name)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    data_x_np, data_y_np = _load_train()
    data_x = jnp.asarray(data_x_np, dtype=jnp.float32)
    data_y = jnp.asarray(data_y_np, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    root = jr.key(jnp.uint32(seed))
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    if perm_mode == "same":
        perms = jnp.tile(schedule.permutations[0][None, :], (n_tasks, 1))
    elif perm_mode == "identity":
        perms = jnp.tile(
            jnp.arange(config.input_dim, dtype=jnp.int32)[None, :], (n_tasks, 1)
        )
    elif perm_mode == "protocol":
        perms = schedule.permutations
    else:
        raise ValueError(perm_mode)
    state = init_fn(params)

    def run_task(params, state, key, permutation, examples):
        def one_step(carry, example):
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            return (new_params, new_state, key), metrics

        (params, state, key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies

    run_task_jit = jax.jit(run_task)
    per_step = np.zeros((n_tasks, config.task_length), dtype=np.uint8)
    per_task = np.zeros(n_tasks, dtype=np.float64)
    started = time.monotonic()
    for task in range(n_tasks):
        params, state, key_noise, accuracies = run_task_jit(
            params, state, key_noise, perms[task], schedule.example_indices[task]
        )
        acc = np.asarray(accuracies)
        per_step[task] = acc.astype(np.uint8)
        per_task[task] = float(acc.mean())
        if (task + 1) % progress_every == 0:
            print(
                f"{spec_name} seed={seed} perm={perm_mode} task {task + 1}/{n_tasks} "
                f"acc={per_task[task]:.4f} elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
    return per_step, per_task, time.monotonic() - started


def save_run(tag: str, spec_name: str, seed: int, perm_mode: str, n_tasks: int) -> None:
    per_step, per_task, wall = run_arm_per_step(spec_name, seed, n_tasks, perm_mode)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{tag}_seed{seed}_per_step.npy", per_step)
    payload = {
        "schema": "alberta.ipmnist_ceiling.run.v1",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "tag": tag,
        "spec_name": spec_name,
        "seed": seed,
        "perm_mode": perm_mode,
        "n_tasks": n_tasks,
        "task_length": 5000,
        "per_task_accuracy": [round(float(v), 8) for v in per_task],
        "mean_accuracy": round(float(per_task.mean()), 8),
        "wall_clock_seconds": round(wall, 2),
        "jax": jax.__version__,
    }
    (OUT / f"{tag}_seed{seed}.json").write_text(json.dumps(payload, indent=1))
    print(f"WROTE {tag} seed={seed} mean={per_task.mean():.5f} wall={wall:.0f}s", flush=True)


def run_batch_reference(seed: int, epochs: int = 30, batch_size: int = 128) -> None:
    """Converged minibatch-Adam reference for the protocol 300x150 ReLU MLP."""
    import optax
    from sklearn.datasets import fetch_openml

    home = default_openml_data_home()
    raw = fetch_openml(
        "mnist_784", version=1, as_frame=False, data_home=str(home), n_retries=3, delay=2.0
    )
    x = np.asarray(raw.data, dtype=np.float32)
    y = np.asarray(raw.target, dtype=np.int32)
    x = (x / 255.0 - 0.5) / 0.5  # protocol scaling
    train_x, train_y = jnp.asarray(x[:60_000]), jnp.asarray(y[:60_000])
    test_x, test_y = jnp.asarray(x[60_000:]), jnp.asarray(y[60_000:])

    config = IPMNISTConfig(n_tasks=1)
    params = init_mlp_params(jr.key(jnp.uint32(seed)), config)

    def loss_fn(params, xb, yb):
        h1 = jax.nn.relu(xb @ params["w1"] + params["b1"])
        h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
        logits = h2 @ params["w3"] + params["b3"]
        return -jnp.mean(
            jnp.take_along_axis(jax.nn.log_softmax(logits), yb[:, None], axis=1)
        )

    def acc_fn(params, xb, yb):
        h1 = jax.nn.relu(xb @ params["w1"] + params["b1"])
        h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
        logits = h2 @ params["w3"] + params["b3"]
        return jnp.mean((jnp.argmax(logits, axis=1) == yb).astype(jnp.float32))

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    acc_jit = jax.jit(acc_fn)
    rng = np.random.default_rng(seed)
    n = 60_000
    history = []
    started = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = order[i : i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, train_x[idx], train_y[idx])
        test_acc = float(acc_jit(params, test_x, test_y))
        history.append(round(test_acc, 5))
        print(f"batch seed={seed} epoch {epoch + 1}/{epochs} test={test_acc:.4f}", flush=True)
    train_acc = float(acc_jit(params, train_x[:20_000], train_y[:20_000]))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alberta.ipmnist_ceiling.batch_reference.v1",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "adam(1e-3)",
        "architecture": "784-300-150-10 ReLU (protocol init)",
        "test_accuracy_final": round(history[-1], 5),
        "test_accuracy_best": round(max(history), 5),
        "train_accuracy_20k": round(train_acc, 5),
        "test_curve": history,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
    }
    (OUT / f"batch_reference_seed{seed}.json").write_text(json.dumps(payload, indent=1))
    print(f"WROTE batch_reference seed={seed} best={max(history):.5f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stationary", "carried", "full", "batch"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--spec", default="sigma0_ndecay099")
    p.add_argument("--n-tasks", type=int, default=None)
    args = p.parse_args()
    if args.mode == "stationary":
        save_run(f"stationary_{args.spec}", args.spec, args.seed, "identity", 1)
    elif args.mode == "carried":
        n = args.n_tasks or 60
        save_run(f"carried_{args.spec}", args.spec, args.seed, "same", n)
    elif args.mode == "full":
        save_run(f"full_{args.spec}", args.spec, args.seed, "protocol", 200)
    elif args.mode == "batch":
        run_batch_reference(args.seed)


if __name__ == "__main__":
    main()
