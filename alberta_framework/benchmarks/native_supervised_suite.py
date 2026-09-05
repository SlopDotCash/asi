"""Deterministic, bounded supervised continual-learning development suite.

Dataset bytes are supplied by the caller.  This module never downloads data,
writes outputs, or promotes results.  It constructs matched task streams and
runs simple end-to-end online controls using predict-before-update semantics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import platform
import time
from pathlib import Path
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

SCHEMA = "asi.native_supervised_cl_development.v1"
AVALANCHE_REVISION = "ContinualAI/avalanche@eb075be393e1f458b2c352514ff6c17b5a2c0f4e"
FROZEN_SEEDS = (15780, 15781, 15782, 15783)
BENCHMARK_IDS = ("split_mnist", "rotated_mnist", "split_cifar100", "ipmnist")
ARM_IDS = ("online_sgd", "replay_sgd", "running_centroid", "frozen_no_learning")
MAX_EXAMPLES_PER_TASK = 64
MAX_INPUT_DIM = 4096
ROTATIONS = (0, 45, 90, 135, 180)


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_identity() -> tuple[str, str, str, str]:
    return (platform.python_version(), jax.__version__, np.__version__, jax.default_backend())


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    benchmark_id: str
    n_tasks: int
    n_classes: int
    boundary_available_to_learner: bool
    task_id_available_to_learner: bool
    canonical_dataset: str
    native_transform: str
    external_parity: bool

    def __post_init__(self) -> None:
        if type(self.benchmark_id) is not str or self.benchmark_id not in BENCHMARK_IDS:
            raise ValueError("unknown benchmark_id")
        _exact_int(self.n_tasks, "n_tasks", 1, 200)
        _exact_int(self.n_classes, "n_classes", 2, 100)
        for name in ("boundary_available_to_learner", "task_id_available_to_learner"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact bool")
        for name in ("canonical_dataset", "native_transform"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
                raise ValueError(f"{name} must be a bounded exact string")
        if type(self.external_parity) is not bool:
            raise ValueError("external_parity must be an exact bool")


CATALOG = (
    BenchmarkSpec(
        "split_mnist", 5, 10, False, False, "MNIST train split, labels 0..9",
        "fixed class pairs (0,1)..(8,9); no task ID; no augmentation", False,
    ),
    BenchmarkSpec(
        "rotated_mnist", 5, 10, False, False, "MNIST train split, labels 0..9",
        "fixed 0/45/90/135/180 degree nearest-neighbor rotations; no task ID", False,
    ),
    BenchmarkSpec(
        "split_cifar100", 20, 100, False, False, "CIFAR-100 train split, fine labels",
        "fixed ascending five-class experiences; no crop/flip; no task ID", False,
    ),
    BenchmarkSpec(
        "ipmnist", 200, 10, False, False, "OpenML mnist_784 v1 first 60,000 rows",
        "seeded fresh pixel permutation per 5,000-example task", False,
    ),
)


def benchmark_spec(benchmark_id: object) -> BenchmarkSpec:
    if type(benchmark_id) is not str:
        raise ValueError("benchmark_id must be an exact string")
    for spec in CATALOG:
        if spec.benchmark_id == benchmark_id:
            return spec
    raise KeyError("unknown benchmark")


@dataclasses.dataclass(frozen=True, slots=True)
class TaskBatch:
    task_index: int
    inputs: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        _exact_int(self.task_index, "task_index", 0, 199)
        if type(self.inputs) is not np.ndarray or self.inputs.dtype != np.float32:
            raise ValueError("inputs must be an exact float32 ndarray")
        if type(self.labels) is not np.ndarray or self.labels.dtype != np.int32:
            raise ValueError("labels must be an exact int32 ndarray")
        if self.inputs.ndim != 2 or self.labels.shape != (self.inputs.shape[0],):
            raise ValueError("task arrays have incompatible shapes")
        if not 1 <= self.inputs.shape[0] <= MAX_EXAMPLES_PER_TASK:
            raise ValueError("task batch is empty or unbounded")
        if not 1 <= self.inputs.shape[1] <= MAX_INPUT_DIM:
            raise ValueError("input dimension is empty or unbounded")
        if not np.all(np.isfinite(self.inputs)):
            raise ValueError("task inputs must be finite")


def _validated_arrays(
    images: object, labels: object, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    if type(images) is not np.ndarray or images.dtype != np.float32 or images.ndim < 2:
        raise ValueError("images must be an exact rank>=2 float32 ndarray")
    if type(labels) is not np.ndarray or labels.dtype != np.int32:
        raise ValueError("labels must be an exact int32 ndarray")
    if labels.shape != (images.shape[0],) or images.shape[0] == 0:
        raise ValueError("images and labels must have one non-empty shared leading axis")
    if images.size > 2**28 or not np.all(np.isfinite(images)):
        raise ValueError("image payload is nonfinite or unbounded")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels lie outside the benchmark class range")
    return images, labels


def _dataset_sha256(images: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"asi-native-supervised-caller-arrays-v1\0")
    digest.update(str(images.dtype).encode("ascii"))
    digest.update(np.asarray(images.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(images).astype("<f4", copy=False).tobytes(order="C"))
    digest.update(str(labels.dtype).encode("ascii"))
    digest.update(np.asarray(labels.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(labels).astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _schedule_sha256(tasks: tuple[TaskBatch, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"asi-native-supervised-task-schedule-v1\0")
    for task in tasks:
        digest.update(task.task_index.to_bytes(4, "little"))
        digest.update(np.asarray(task.inputs.shape, dtype="<i8").tobytes())
        digest.update(task.inputs.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(np.asarray(task.labels.shape, dtype="<i8").tobytes())
        digest.update(task.labels.astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _rotate_nearest(image: np.ndarray, degrees: int) -> np.ndarray:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("native rotated MNIST requires square rank-2 images")
    height, width = image.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    ys, xs = np.indices((height, width), dtype=np.float64)
    dx, dy = xs - center_x, ys - center_y
    source_x = np.rint(cosine * dx + sine * dy + center_x).astype(np.int64)
    source_y = np.rint(-sine * dx + cosine * dy + center_y).astype(np.int64)
    valid = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
    result = np.zeros_like(image)
    result[valid] = image[source_y[valid], source_x[valid]]
    return result


@jax.jit
def _sgd_step(
    weights: Array, bias: Array, inputs: Array, label: Array
) -> tuple[Array, Array]:
    logits = inputs @ weights + bias
    probabilities = jax.nn.softmax(logits)
    error = probabilities.at[label].add(-1.0)
    next_weights = weights - 0.01 * jnp.outer(inputs, error)
    next_bias = bias - 0.01 * error
    return next_weights, next_bias


def build_task_stream(
    benchmark_id: object,
    images: object,
    labels: object,
    *,
    seed: object,
    examples_per_task: object,
) -> tuple[TaskBatch, ...]:
    """Build a bounded deterministic task stream from caller-supplied data."""
    spec = benchmark_spec(benchmark_id)
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    count = _exact_int(
        examples_per_task, "examples_per_task", 1, MAX_EXAMPLES_PER_TASK
    )
    data, targets = _validated_arrays(images, labels, spec.n_classes)
    input_dim = int(np.prod(data.shape[1:], dtype=np.int64))
    if input_dim > MAX_INPUT_DIM:
        raise ValueError("flattened input dimension exceeds the bound")
    rng = np.random.default_rng(host_seed)
    tasks: list[TaskBatch] = []
    for task_index in range(spec.n_tasks):
        if spec.benchmark_id in ("split_mnist", "split_cifar100"):
            classes_per_task = spec.n_classes // spec.n_tasks
            low = task_index * classes_per_task
            eligible = np.flatnonzero((targets >= low) & (targets < low + classes_per_task))
            if eligible.size < count:
                raise ValueError("dataset has too few examples for a frozen class experience")
            indices = rng.permutation(eligible)[:count]
            transformed = data[indices]
            task_labels = targets[indices]
        else:
            if data.shape[0] < count:
                raise ValueError("dataset has too few examples for a frozen experience")
            indices = rng.permutation(data.shape[0])[:count]
            transformed = data[indices]
            task_labels = targets[indices]
            if spec.benchmark_id == "rotated_mnist":
                transformed = np.stack(
                    tuple(_rotate_nearest(image, ROTATIONS[task_index]) for image in transformed)
                )
            else:
                permutation = rng.permutation(input_dim)
                transformed = transformed.reshape((count, input_dim))[:, permutation]
        flattened = np.asarray(transformed.reshape((count, input_dim)), dtype=np.float32)
        tasks.append(
            TaskBatch(
                task_index,
                np.ascontiguousarray(flattened),
                np.asarray(task_labels, np.int32),
            )
        )
    return tuple(tasks)


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    data_steps: int
    data_bytes_read: int
    model_queries: int
    parameter_updates: int
    replay_inserts: int
    replay_samples: int
    logical_compute_units: int
    persistent_bytes: int
    peak_replay_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name == "timing_telemetry_only":
                continue
            _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.data_steps == 0 or self.model_queries == 0 or self.persistent_bytes == 0:
            raise ValueError("steps, queries, and persistent bytes must be positive")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    online_accuracy: float
    task_accuracies: tuple[float, ...]
    receipt: ResourceReceipt

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm_id")
        if type(self.online_accuracy) is not float or not 0.0 <= self.online_accuracy <= 1.0:
            raise ValueError("online_accuracy must be a finite exact probability")
        if type(self.task_accuracies) is not tuple or not self.task_accuracies:
            raise ValueError("task_accuracies must be a non-empty exact tuple")
        if any(
            type(value) is not float or not 0.0 <= value <= 1.0
            for value in self.task_accuracies
        ):
            raise ValueError("task accuracies must be finite exact probabilities")


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteResult:
    schema: str
    benchmark_id: str
    seed: int
    examples_per_task: int
    replay_capacity: int
    input_dim: int
    n_classes: int
    dataset_sha256: str
    schedule_sha256: str
    source_sha256: str
    runtime_identity: tuple[str, str, str, str]
    arms: tuple[ArmResult, ...]
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True
    task_information_used_by_learner: bool = False

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.seed not in FROZEN_SEEDS:
            raise ValueError("schema or frozen seed mismatch")
        spec = benchmark_spec(self.benchmark_id)
        _exact_int(self.examples_per_task, "examples_per_task", 1, MAX_EXAMPLES_PER_TASK)
        _exact_int(self.replay_capacity, "replay_capacity", 1, 64)
        _exact_int(self.input_dim, "input_dim", 1, MAX_INPUT_DIM)
        if self.n_classes != spec.n_classes:
            raise ValueError("class count differs from the catalog")
        for name in ("dataset_sha256", "schedule_sha256", "source_sha256"):
            _digest(getattr(self, name), name)
        if self.source_sha256 != _source_sha256():
            raise ValueError("current ASI source identity drift")
        if self.runtime_identity != _runtime_identity():
            raise ValueError("current runtime identity drift")
        if type(self.arms) is not tuple or any(type(arm) is not ArmResult for arm in self.arms):
            raise ValueError("arms must contain exact ArmResult values")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arms differ from the frozen roster")
        flags = (
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
            not self.task_information_used_by_learner,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("result must remain nonpromoting, retained, and task-agnostic")


def _run_arm(
    tasks: tuple[TaskBatch, ...], n_classes: int, capacity: int, arm_id: str
) -> ArmResult:
    input_dim = tasks[0].inputs.shape[1]
    weights = np.zeros((input_dim, n_classes), dtype=np.float32)
    bias = np.zeros((n_classes,), dtype=np.float32)
    sums = np.zeros((n_classes, input_dim), dtype=np.float32)
    counts = np.zeros((n_classes,), dtype=np.int32)
    replay: list[tuple[np.ndarray, int]] = []
    queries = updates = inserts = samples = correct = 0
    real_seen = 0
    per_task: list[float] = []
    start = time.perf_counter_ns()

    def linear_update(x: np.ndarray, label: int) -> None:
        nonlocal weights, bias, queries, updates
        queries += 1
        next_weights, next_bias = _sgd_step(
            jnp.asarray(weights), jnp.asarray(bias), jnp.asarray(x), jnp.asarray(label)
        )
        weights = np.asarray(next_weights, dtype=np.float32)
        bias = np.asarray(next_bias, dtype=np.float32)
        updates += 1

    for task in tasks:
        task_correct = 0
        for x, raw_label in zip(task.inputs, task.labels, strict=True):
            label = int(raw_label)
            if arm_id == "running_centroid":
                distances = np.sum((sums / np.maximum(counts[:, None], 1) - x) ** 2, axis=1)
                distances = np.where(counts > 0, distances, np.inf)
                prediction = int(np.argmin(distances)) if np.any(counts > 0) else 0
                queries += 1
            else:
                prediction = int(np.argmax(x @ weights + bias))
                queries += 1
            hit = int(prediction == label)
            correct += hit
            task_correct += hit
            if arm_id in ("online_sgd", "replay_sgd"):
                # The external prediction above is the predict-before-update query;
                # this update's logits are charged as a separate model query.
                linear_update(x, label)
                if arm_id == "replay_sgd":
                    if replay:
                        replay_x, replay_label = replay[(updates + label) % len(replay)]
                        linear_update(replay_x, replay_label)
                        samples += 1
                    replay.append((x.copy(), label))
                    if len(replay) > capacity:
                        replay.pop(0)
                    inserts += 1
                elif real_seen > 0:
                    # Match replay SGD's optimizer-call budget without exposing
                    # another observation: repeat the current real example.
                    linear_update(x, label)
            elif arm_id == "running_centroid":
                sums[label] += x
                counts[label] += 1
                updates += 1
            real_seen += 1
        per_task.append(task_correct / task.inputs.shape[0])
    elapsed = time.perf_counter_ns() - start
    steps = sum(task.inputs.shape[0] for task in tasks)
    bytes_per_example = input_dim * 4 + 4
    if arm_id == "running_centroid":
        persistent = sums.nbytes + counts.nbytes
    else:
        persistent = weights.nbytes + bias.nbytes
    peak_replay = min(capacity, steps) * bytes_per_example if arm_id == "replay_sgd" else 0
    compute = steps + queries + updates + inserts + samples
    return ArmResult(
        arm_id=arm_id,
        online_accuracy=float(correct / steps),
        task_accuracies=tuple(float(value) for value in per_task),
        receipt=ResourceReceipt(
            data_steps=steps,
            data_bytes_read=steps * bytes_per_example,
            model_queries=queries,
            parameter_updates=updates,
            replay_inserts=inserts,
            replay_samples=samples,
            logical_compute_units=compute,
            persistent_bytes=int(persistent),
            peak_replay_bytes=peak_replay,
            elapsed_ns=elapsed,
        ),
    )


def run_native_suite(
    benchmark_id: object,
    images: object,
    labels: object,
    *,
    seed: object,
    examples_per_task: object = 8,
    replay_capacity: object = 16,
) -> SuiteResult:
    """Construct and execute one bounded matched development shard."""
    spec = benchmark_spec(benchmark_id)
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    count = _exact_int(examples_per_task, "examples_per_task", 1, MAX_EXAMPLES_PER_TASK)
    capacity = _exact_int(replay_capacity, "replay_capacity", 1, 64)
    data, targets = _validated_arrays(images, labels, spec.n_classes)
    tasks = build_task_stream(
        spec.benchmark_id, data, targets, seed=host_seed, examples_per_task=count
    )
    result = SuiteResult(
        schema=SCHEMA,
        benchmark_id=spec.benchmark_id,
        seed=host_seed,
        examples_per_task=count,
        replay_capacity=capacity,
        input_dim=tasks[0].inputs.shape[1],
        n_classes=spec.n_classes,
        dataset_sha256=_dataset_sha256(data, targets),
        schedule_sha256=_schedule_sha256(tasks),
        source_sha256=_source_sha256(),
        runtime_identity=_runtime_identity(),
        arms=tuple(_run_arm(tasks, spec.n_classes, capacity, arm_id) for arm_id in ARM_IDS),
    )
    return validate_result(result)


def validate_result(value: object) -> SuiteResult:
    if type(value) is not SuiteResult:
        raise ValueError("result must be an exact SuiteResult")
    SuiteResult.__post_init__(value)
    spec = benchmark_spec(value.benchmark_id)
    steps = spec.n_tasks * value.examples_per_task
    bytes_per_example = value.input_dim * 4 + 4
    for arm in value.arms:
        ArmResult.__post_init__(arm)
        if len(arm.task_accuracies) != spec.n_tasks:
            raise ValueError("accuracy record mismatch")
        total_correct = 0
        for accuracy in arm.task_accuracies:
            correct = round(accuracy * value.examples_per_task)
            if accuracy != correct / value.examples_per_task:
                raise ValueError("accuracy record mismatch")
            total_correct += correct
        if arm.online_accuracy != total_correct / steps:
            raise ValueError("accuracy record mismatch")
        if type(arm.receipt) is not ResourceReceipt:
            raise ValueError("receipt must be an exact ResourceReceipt")
        ResourceReceipt.__post_init__(arm.receipt)
        receipt = arm.receipt
        if receipt.data_steps != steps or receipt.data_bytes_read != steps * bytes_per_example:
            raise ValueError("data receipt mismatch")
        expected_updates = 0 if arm.arm_id == "frozen_no_learning" else steps
        expected_samples = steps - 1 if arm.arm_id == "replay_sgd" else 0
        if arm.arm_id in ("online_sgd", "replay_sgd"):
            expected_updates += steps - 1
        expected_queries = (
            steps + expected_updates
            if arm.arm_id in ("online_sgd", "replay_sgd")
            else steps
        )
        expected_inserts = steps if arm.arm_id == "replay_sgd" else 0
        expected_peak = (
            min(value.replay_capacity, steps) * bytes_per_example if expected_inserts else 0
        )
        expected_persistent = (
            value.n_classes * value.input_dim * 4 + value.n_classes * 4
        )
        if receipt.persistent_bytes != expected_persistent:
            raise ValueError("persistent-byte receipt mismatch")
        expected_compute = (
            steps + expected_queries + expected_updates + expected_inserts + expected_samples
        )
        expected = (
            receipt.parameter_updates,
            receipt.model_queries,
            receipt.replay_inserts,
            receipt.replay_samples,
            receipt.peak_replay_bytes,
            receipt.logical_compute_units,
        )
        if expected != (
            expected_updates, expected_queries, expected_inserts, expected_samples,
            expected_peak, expected_compute,
        ):
            raise ValueError("resource receipt mismatch")
    return value


def catalog_payload() -> dict[str, object]:
    return {
        "schema": "asi.native_supervised_cl_catalog.v1",
        "avalanche_revision": AVALANCHE_REVISION,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "frozen_seeds": list(FROZEN_SEEDS),
        "benchmarks": [dataclasses.asdict(spec) for spec in CATALOG],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the nonpromoting native CL catalog")
    parser.add_argument("--catalog", action="store_true", help="print the frozen catalog JSON")
    args = parser.parse_args(argv)
    if not args.catalog:
        parser.error("only --catalog is available; dataset execution requires supplied arrays")
    print(json.dumps(catalog_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_IDS", "AVALANCHE_REVISION", "BENCHMARK_IDS", "CATALOG", "FROZEN_SEEDS",
    "ArmResult", "BenchmarkSpec", "ResourceReceipt", "SuiteResult", "TaskBatch",
    "benchmark_spec", "build_task_stream", "catalog_payload", "main", "run_native_suite",
    "validate_result",
]
