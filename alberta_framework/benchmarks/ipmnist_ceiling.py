"""Maintained IPMNIST ceiling-run producer.

The scripts preserved under ``outputs/`` describe historical executions and
must not be edited.  This module is their maintained replacement: destinations
are explicit, publication is no-replace, and every result remains permanently
development-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import alberta_framework.benchmarks.ipmnist_provenance as ipmnist_provenance_module
import alberta_framework.benchmarks.ipmnist_screening as ipmnist_screening_module
import alberta_framework.benchmarks.upgd_ipmnist as upgd_ipmnist_module
from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_provenance import (
    array_identity,
    repository_specification_identities,
    runtime_identity,
    source_identities,
)
from alberta_framework.benchmarks.ipmnist_screening import screening_spec
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

_BATCH_TRAIN_SIZE = 60_000
_MIN_BATCH_SIZE = 32
_MAX_BATCH_EPOCHS = 30
_MAX_BATCH_UPDATES = 50_000


def _load_train() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    return load_mnist_train(default_openml_data_home())


def run_arm_per_step(
    spec_name: str,
    seed: int,
    n_tasks: int,
    permutation_mode: str,
    *,
    progress_every: int = 20,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, dict[str, Any]]:
    """Run one screening arm while retaining per-step online accuracy."""
    seed = require_jax_seed(seed)
    config = IPMNISTConfig(n_tasks=n_tasks)
    spec = screening_spec(spec_name)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    data_x_numpy, data_y_numpy = _load_train()
    data_x = jnp.asarray(data_x_numpy, dtype=jnp.float32)
    data_y = jnp.asarray(data_y_numpy, dtype=jnp.int32)
    root = jr.key(jnp.uint32(seed))
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, int(data_x.shape[0]))
    if permutation_mode == "same":
        permutations = jnp.tile(schedule.permutations[0][None, :], (n_tasks, 1))
    elif permutation_mode == "identity":
        permutations = jnp.tile(
            jnp.arange(config.input_dim, dtype=jnp.int32)[None, :], (n_tasks, 1)
        )
    elif permutation_mode == "protocol":
        permutations = schedule.permutations
    else:
        raise ValueError("permutation_mode must be identity, same, or protocol")
    state = init_fn(params)

    def run_task(
        task_params: Any,
        task_state: Any,
        task_key: Any,
        permutation: Any,
        examples: Any,
    ) -> tuple[Any, Any, Any, Any]:
        def one_step(carry: tuple[Any, Any, Any], example: Any) -> tuple[Any, Any]:
            step_params, step_state, key = carry
            observation = data_x[example][permutation]
            target = data_y[example]
            key, step_key = jr.split(key)
            next_params, next_state, metrics = step_fn(
                step_params, step_state, observation, target, step_key
            )
            return (next_params, next_state, key), metrics

        (next_params, next_state, next_key), metrics = jax.lax.scan(
            one_step, (task_params, task_state, task_key), examples
        )
        accuracies, _, _ = metrics
        return next_params, next_state, next_key, accuracies

    run_task_jit = jax.jit(run_task)
    per_step = np.zeros((n_tasks, config.task_length), dtype=np.uint8)
    per_task = np.zeros(n_tasks, dtype=np.float64)
    started = time.monotonic()
    for task in range(n_tasks):
        params, state, key_noise, accuracies = run_task_jit(
            params,
            state,
            key_noise,
            permutations[task],
            schedule.example_indices[task],
        )
        accuracy_array = np.asarray(accuracies)
        per_step[task] = accuracy_array.astype(np.uint8)
        per_task[task] = float(accuracy_array.mean())
        if progress_every > 0 and (task + 1) % progress_every == 0:
            print(
                f"{spec_name} seed={seed} mode={permutation_mode} "
                f"task={task + 1}/{n_tasks} accuracy={per_task[task]:.4f}",
                flush=True,
            )
    provenance = {
        "schema": "asi.ipmnist.ceiling_run_provenance.v1",
        "runtime": runtime_identity(dependencies=("jax", "jaxlib", "numpy")),
        "sources": source_identities(
            {
                "ceiling_runner": Path(__file__),
                "ipmnist_screening": Path(ipmnist_screening_module.__file__),
                "ipmnist_provenance": Path(ipmnist_provenance_module.__file__),
                "upgd_ipmnist": Path(upgd_ipmnist_module.__file__),
            },
            repository_root=Path(__file__).resolve().parents[2],
        ),
        "environment_specifications": repository_specification_identities(
            Path(__file__).resolve().parents[2]
        ),
        "dataset": {
            "name": "mnist_784",
            "version": 1,
            "train_observations": array_identity(data_x_numpy),
            "train_labels": array_identity(data_y_numpy),
        },
        "schedule": {
            "derivation": "jr.split(root,3)[1]; per-task fold_in permutations/samples",
            "permutations": array_identity(np.asarray(permutations)),
            "example_indices": array_identity(np.asarray(schedule.example_indices)),
        },
        "inputs": {
            "spec_name": spec.name,
            "base_learner": spec.base_learner,
            "mechanism": spec.mechanism,
            "hyperparameters": dict(sorted(spec.hyperparameters.items())),
            "seed": seed,
            "permutation_mode": permutation_mode,
            "config": config.to_config(),
        },
    }
    return per_step, per_task, time.monotonic() - started, provenance


def _publish_run(
    output_dir: Path,
    *,
    tag: str,
    spec_name: str,
    seed: int,
    permutation_mode: str,
    n_tasks: int,
    per_step: np.ndarray[Any, Any],
    per_task: np.ndarray[Any, Any],
    wall_seconds: float,
    provenance: dict[str, Any],
) -> Path:
    """Atomically publish one self-contained run without replacement."""
    tag = _safe_output_tag(tag)
    spec_name = _safe_output_tag(spec_name)
    if type(permutation_mode) is not str:
        raise ValueError("permutation_mode must be an exact string")
    seed = require_jax_seed(seed)
    if type(n_tasks) is not int or n_tasks <= 0:
        raise ValueError("n_tasks must be a positive built-in integer")
    _validate_run_identity(tag, spec_name, permutation_mode, n_tasks)
    if type(wall_seconds) not in (int, float) or not np.isfinite(wall_seconds):
        raise ValueError("wall_seconds must be a finite built-in number")
    if wall_seconds < 0:
        raise ValueError("wall_seconds must be nonnegative")
    if type(per_step) is not np.ndarray or per_step.dtype != np.dtype(np.uint8):
        raise ValueError("per_step must be an exact uint8 numpy array")
    if per_step.ndim != 2 or per_step.shape != (n_tasks, 5_000):
        raise ValueError("per_step shape must be (n_tasks, 5000)")
    if not bool(np.all((per_step == 0) | (per_step == 1))):
        raise ValueError("per_step must contain only binary correctness values")
    if type(per_task) is not np.ndarray or per_task.dtype != np.dtype(np.float64):
        raise ValueError("per_task must be an exact float64 numpy array")
    if per_task.shape != (n_tasks,) or not bool(np.all(np.isfinite(per_task))):
        raise ValueError("per_task must be a finite vector with one value per task")
    if not bool(np.all((0.0 <= per_task) & (per_task <= 1.0))):
        raise ValueError("per_task values must be in [0, 1]")
    if not bool(
        np.allclose(
            per_task,
            per_step.astype(np.float64).mean(axis=1),
            rtol=0.0,
            atol=1e-7,
            equal_nan=False,
        )
    ):
        raise ValueError("per_task must match the per_step task means")
    if (
        type(provenance) is not dict
        or provenance.get("schema") != "asi.ipmnist.ceiling_run_provenance.v1"
    ):
        raise ValueError("provenance must use the maintained ceiling-run schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{tag}_seed{seed}.npz"
    if artifact_path.exists():
        raise FileExistsError(f"refusing to replace {tag} seed {seed} in {output_dir}")
    payload = {
        "schema": "asi.ipmnist_ceiling.run.v2",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "tag": tag,
        "spec_name": spec_name,
        "seed": seed,
        "perm_mode": permutation_mode,
        "n_tasks": n_tasks,
        "task_length": int(per_step.shape[1]),
        "per_task_accuracy": [round(float(value), 8) for value in per_task],
        "mean_accuracy": round(float(per_task.mean()), 8),
        "wall_clock_seconds": round(wall_seconds, 2),
        "jax": jax.__version__,
        "provenance": provenance,
    }
    metadata = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    def write_archive(stream: BinaryIO) -> None:
        np.savez_compressed(
            stream,
            metadata=np.asarray(metadata),
            per_step=per_step,
        )

    _atomic_publish(artifact_path, write_archive)
    return artifact_path


def _safe_output_tag(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("tag must be a non-empty safe exact identifier")
    alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if value[0] not in alphanumeric or not all(
        character in alphanumeric or character in "_-" for character in value
    ):
        raise ValueError("tag must be a non-empty safe exact identifier")
    return value


def _validate_run_identity(
    tag: str, spec_name: str, permutation_mode: str, n_tasks: int
) -> None:
    expected: tuple[str, int | None]
    if tag == f"stationary_{spec_name}":
        expected = ("identity", 1)
    elif tag == f"carried_{spec_name}":
        expected = ("same", None)
    elif tag == f"full_{spec_name}":
        expected = ("protocol", 200)
    else:
        raise ValueError("tag must bind its run family and spec_name")
    expected_mode, expected_tasks = expected
    if permutation_mode != expected_mode:
        raise ValueError("permutation_mode disagrees with tag family")
    if expected_tasks is not None and n_tasks != expected_tasks:
        raise ValueError("n_tasks disagrees with tag family")
    if expected_tasks is None and n_tasks < 10:
        raise ValueError("carried runs require at least ten tasks")


def _atomic_publish(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Publish one file by durable temp write plus atomic no-replace link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = True
        if os.name != "nt":
            # Windows cannot open directory handles through os.open().
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def run_and_publish(
    output_dir: Path,
    *,
    mode: str,
    spec_name: str,
    seed: int,
    n_tasks: int | None = None,
) -> Path:
    """Run one stationary, carried, or full ceiling shard."""
    if mode == "stationary":
        tasks, permutation_mode, tag = 1, "identity", f"stationary_{spec_name}"
    elif mode == "carried":
        tasks = 60 if n_tasks is None else n_tasks
        permutation_mode, tag = "same", f"carried_{spec_name}"
    elif mode == "full":
        tasks, permutation_mode, tag = 200, "protocol", f"full_{spec_name}"
    else:
        raise ValueError("mode must be stationary, carried, or full")
    if tasks <= 0:
        raise ValueError("n_tasks must be positive")
    per_step, per_task, wall_seconds, provenance = run_arm_per_step(
        spec_name, seed, tasks, permutation_mode
    )
    return _publish_run(
        output_dir,
        tag=tag,
        spec_name=spec_name,
        seed=seed,
        permutation_mode=permutation_mode,
        n_tasks=tasks,
        per_step=per_step,
        per_task=per_task,
        wall_seconds=wall_seconds,
        provenance=provenance,
    )


def run_batch_reference(
    output_dir: Path,
    *,
    seed: int,
    epochs: int = 30,
    batch_size: int = 128,
) -> Path:
    """Run the converged minibatch-Adam architecture reference."""
    seed = require_jax_seed(seed)
    if type(epochs) is not int or not 1 <= epochs <= _MAX_BATCH_EPOCHS:
        raise ValueError(
            f"epochs must be a built-in integer in [1, {_MAX_BATCH_EPOCHS}]"
        )
    if (
        type(batch_size) is not int
        or not _MIN_BATCH_SIZE <= batch_size <= _BATCH_TRAIN_SIZE
    ):
        raise ValueError(
            "batch_size must be a built-in integer in "
            f"[{_MIN_BATCH_SIZE}, {_BATCH_TRAIN_SIZE}]"
        )
    update_count = epochs * (_BATCH_TRAIN_SIZE // batch_size)
    if update_count > _MAX_BATCH_UPDATES:
        raise ValueError(
            f"batch run exceeds the {_MAX_BATCH_UPDATES} optimizer-update bound"
        )
    import importlib

    optax = importlib.import_module("optax")
    from sklearn.datasets import fetch_openml  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"batch_reference_seed{seed}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to replace {output_path}")

    raw = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=str(default_openml_data_home()),
        n_retries=3,
        delay=2.0,
    )
    observations = np.asarray(raw.data, dtype=np.float32)
    labels = np.asarray(raw.target, dtype=np.int32)
    observations = (observations / 255.0 - 0.5) / 0.5
    train_x = jnp.asarray(observations[:60_000])
    train_y = jnp.asarray(labels[:60_000])
    test_x = jnp.asarray(observations[60_000:])
    test_y = jnp.asarray(labels[60_000:])
    config = IPMNISTConfig(n_tasks=1)
    params = init_mlp_params(jr.key(jnp.uint32(seed)), config)

    def loss_fn(model: Any, batch_x: Any, batch_y: Any) -> Any:
        hidden1 = jax.nn.relu(batch_x @ model["w1"] + model["b1"])
        hidden2 = jax.nn.relu(hidden1 @ model["w2"] + model["b2"])
        logits = hidden2 @ model["w3"] + model["b3"]
        return -jnp.mean(
            jnp.take_along_axis(jax.nn.log_softmax(logits), batch_y[:, None], axis=1)
        )

    def accuracy_fn(model: Any, batch_x: Any, batch_y: Any) -> Any:
        hidden1 = jax.nn.relu(batch_x @ model["w1"] + model["b1"])
        hidden2 = jax.nn.relu(hidden1 @ model["w2"] + model["b2"])
        logits = hidden2 @ model["w3"] + model["b3"]
        return jnp.mean((jnp.argmax(logits, axis=1) == batch_y).astype(jnp.float32))

    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        model: Any, state: Any, batch_x: Any, batch_y: Any
    ) -> tuple[Any, Any]:
        _, gradients = jax.value_and_grad(loss_fn)(model, batch_x, batch_y)
        updates, next_state = optimizer.update(gradients, state)
        return optax.apply_updates(model, updates), next_state

    accuracy_jit = jax.jit(accuracy_fn)
    rng = np.random.default_rng(seed)
    schedule_digest = hashlib.sha256()
    history: list[float] = []
    started = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(_BATCH_TRAIN_SIZE)
        schedule_digest.update(memoryview(np.ascontiguousarray(order)).cast("B"))
        for start in range(
            0, _BATCH_TRAIN_SIZE - batch_size + 1, batch_size
        ):
            indices = order[start : start + batch_size]
            params, optimizer_state = train_step(
                params, optimizer_state, train_x[indices], train_y[indices]
            )
        test_accuracy = float(accuracy_jit(params, test_x, test_y))
        history.append(test_accuracy)
        print(
            f"batch seed={seed} epoch={epoch + 1}/{epochs} "
            f"test_accuracy={test_accuracy:.4f}",
            flush=True,
        )
    train_accuracy = float(accuracy_jit(params, train_x[:20_000], train_y[:20_000]))
    payload = {
        "schema": "asi.ipmnist_ceiling.batch_reference.v2",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "adam(1e-3)",
        "architecture": "784-300-150-10 ReLU (protocol init)",
        "test_accuracy_final": round(history[-1], 5),
        "test_accuracy_best": round(max(history), 5),
        "train_accuracy_20k": round(train_accuracy, 5),
        "test_curve": [round(value, 5) for value in history],
        "wall_clock_seconds": round(time.monotonic() - started, 1),
        "provenance": {
            "schema": "asi.ipmnist.ceiling_batch_provenance.v1",
            "runtime": runtime_identity(
                dependencies=("jax", "jaxlib", "numpy", "optax", "scikit-learn")
            ),
            "sources": source_identities(
                {
                    "ceiling_runner": Path(__file__),
                    "ipmnist_provenance": Path(ipmnist_provenance_module.__file__),
                    "upgd_ipmnist": Path(upgd_ipmnist_module.__file__),
                },
                repository_root=Path(__file__).resolve().parents[2],
            ),
            "environment_specifications": repository_specification_identities(
                Path(__file__).resolve().parents[2]
            ),
            "dataset": {
                "name": "mnist_784",
                "version": 1,
                "observations": array_identity(observations),
                "labels": array_identity(labels),
                "split": {"train_rows": 60_000, "test_rows": 10_000},
            },
            "schedule": {
                "kind": "numpy.default_rng epoch permutation",
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "epoch_order_sha256": schedule_digest.hexdigest(),
            },
            "inputs": {"config": config.to_config(), "optimizer": "adam(1e-3)"},
        },
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()

    def write_payload(stream: BinaryIO) -> None:
        stream.write(encoded)

    _atomic_publish(output_path, write_payload)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stationary", "carried", "full", "batch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--spec", default="sigma0_ndecay099")
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly addressed, permanently nonpromoting ceiling shard."""
    args = _parser().parse_args(argv)
    if args.mode == "batch":
        print(
            run_batch_reference(
                args.output_dir,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        )
    else:
        path = run_and_publish(
            args.output_dir,
            mode=args.mode,
            spec_name=args.spec,
            seed=args.seed,
            n_tasks=args.n_tasks,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
