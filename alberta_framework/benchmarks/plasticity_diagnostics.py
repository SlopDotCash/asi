"""Bounded hidden-network diagnostics for canonical loss of plasticity.

The implemented workload is input-permuted MNIST.  Labels never change.  It
is a deliberately short development diagnostic, not a reproduction of the
800-task paper experiment and not a random-label benchmark.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

SCHEMA = "asi.loss_of_plasticity_mnist_development.v1"
PAPER_REVISION = "arXiv:2306.13812v3"
OFFICIAL_CODE_COMMIT = "a6b79580d85f3025bdb601566d3627c5f489f13b"
FROZEN_SEEDS = (15830, 15831, 15832, 15833)
ARM_IDS = ("sgd_control", "cbp_mechanism_off", "cbp_bounded")
MAX_DATASET_EXAMPLES = 60_000
INPUT_DIM = 784
N_CLASSES = 10


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
class DiagnosticProfile:
    profile_id: str
    n_tasks: int
    examples_per_task: int
    hidden_width: int
    learning_rate: float
    replacement_rate: float
    maturity_threshold: int

    def __post_init__(self) -> None:
        if self.profile_id not in ("contract-smoke", "bounded-development"):
            raise ValueError("unknown diagnostic profile")
        _exact_int(self.n_tasks, "n_tasks", 2, 16)
        _exact_int(self.examples_per_task, "examples_per_task", 2, 128)
        _exact_int(self.hidden_width, "hidden_width", 4, 128)
        _exact_int(self.maturity_threshold, "maturity_threshold", 1, 10_000)
        for name in ("learning_rate", "replacement_rate"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite exact float")
        if self.replacement_rate > 1.0:
            raise ValueError("replacement_rate must not exceed one")


PROFILES: Mapping[str, DiagnosticProfile] = MappingProxyType({
    "contract-smoke": DiagnosticProfile("contract-smoke", 2, 4, 8, 0.003, 0.25, 2),
    "bounded-development": DiagnosticProfile(
        "bounded-development", 8, 64, 64, 0.003, 0.0001, 100
    ),
})


class MLPState(NamedTuple):
    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array
    utility1: Array
    utility2: Array
    age1: Array
    age2: Array
    replacement_credit1: Array
    replacement_credit2: Array


def _init_state(key: Array, width: int) -> MLPState:
    k1, k2, k3 = jr.split(key, 3)
    w1 = jr.normal(k1, (INPUT_DIM, width), dtype=jnp.float32) * math.sqrt(2.0 / INPUT_DIM)
    w2 = jr.normal(k2, (width, width), dtype=jnp.float32) * math.sqrt(2.0 / width)
    w3 = jr.normal(k3, (width, N_CLASSES), dtype=jnp.float32) * math.sqrt(2.0 / width)
    zeros = jnp.zeros((width,), dtype=jnp.float32)
    return MLPState(
        w1,
        zeros,
        w2,
        zeros,
        w3,
        jnp.zeros((N_CLASSES,), dtype=jnp.float32),
        zeros,
        zeros,
        jnp.zeros((width,), dtype=jnp.int32),
        jnp.zeros((width,), dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _forward(state: MLPState, inputs: Array) -> tuple[Array, Array, Array]:
    hidden1 = jax.nn.relu(inputs @ state.w1 + state.b1)
    hidden2 = jax.nn.relu(hidden1 @ state.w2 + state.b2)
    return hidden2 @ state.w3 + state.b3, hidden1, hidden2


def _loss(state: MLPState, inputs: Array, label: Array) -> tuple[Array, tuple[Array, Array, Array]]:
    logits, hidden1, hidden2 = _forward(state, inputs)
    return -jax.nn.log_softmax(logits)[label], (logits, hidden1, hidden2)


def _replace_first_layer(
    state: MLPState, utility: Array, eligible: Array, key: Array
) -> tuple[MLPState, Array]:
    index = jnp.argmin(jnp.where(eligible, utility, jnp.inf))
    incoming = jr.normal(key, (INPUT_DIM,), dtype=jnp.float32) * math.sqrt(2.0 / INPUT_DIM)
    return state._replace(
        w1=state.w1.at[:, index].set(incoming),
        b1=state.b1.at[index].set(0.0),
        w2=state.w2.at[index, :].set(0.0),
        utility1=utility.at[index].set(0.0),
        age1=state.age1.at[index].set(0),
    ), jnp.asarray(1, dtype=jnp.int32)


def _replace_second_layer(
    state: MLPState, utility: Array, eligible: Array, key: Array
) -> tuple[MLPState, Array]:
    width = state.w2.shape[0]
    index = jnp.argmin(jnp.where(eligible, utility, jnp.inf))
    incoming = jr.normal(key, (width,), dtype=jnp.float32) * math.sqrt(2.0 / width)
    return state._replace(
        w2=state.w2.at[:, index].set(incoming),
        b2=state.b2.at[index].set(0.0),
        w3=state.w3.at[index, :].set(0.0),
        utility2=utility.at[index].set(0.0),
        age2=state.age2.at[index].set(0),
    ), jnp.asarray(1, dtype=jnp.int32)


@jax.jit
def _step(
    state: MLPState,
    inputs: Array,
    label: Array,
    key: Array,
    learning_rate: Array,
    replacement_rate: Array,
    maturity_threshold: Array,
) -> tuple[MLPState, Array, Array, Array]:
    (loss, (logits, hidden1, hidden2)), gradients = jax.value_and_grad(
        _loss, has_aux=True, allow_int=True
    )(state, inputs, label)
    # Utilities and ages are mechanism state, not differentiated parameters.
    trainable = tuple(
        value - learning_rate * gradient if index < 6 else value
        for index, (value, gradient) in enumerate(zip(state, gradients, strict=True))
    )
    updated = MLPState(*trainable)
    utility1 = 0.99 * state.utility1 + 0.01 * hidden1 * jnp.mean(jnp.abs(state.w2), axis=1)
    utility2 = 0.99 * state.utility2 + 0.01 * hidden2 * jnp.mean(jnp.abs(state.w3), axis=1)
    age1 = state.age1 + 1
    age2 = state.age2 + 1
    eligible1 = age1 >= maturity_threshold
    eligible2 = age2 >= maturity_threshold
    credit1 = state.replacement_credit1 + replacement_rate * jnp.sum(eligible1)
    credit2 = state.replacement_credit2 + replacement_rate * jnp.sum(eligible2)
    updated = updated._replace(
        utility1=utility1,
        utility2=utility2,
        age1=age1,
        age2=age2,
        replacement_credit1=credit1,
        replacement_credit2=credit2,
    )
    key1, key2 = jr.split(key)
    replace1 = (credit1 >= 1.0) & jnp.any(eligible1) & (replacement_rate > 0.0)
    updated, count1 = jax.lax.cond(
        replace1,
        lambda current: _replace_first_layer(current, current.utility1, eligible1, key1),
        lambda current: (current, jnp.asarray(0, dtype=jnp.int32)),
        updated,
    )
    updated = updated._replace(
        replacement_credit1=jnp.where(replace1, credit1 - 1.0, credit1)
    )
    replace2 = (credit2 >= 1.0) & jnp.any(eligible2) & (replacement_rate > 0.0)
    updated, count2 = jax.lax.cond(
        replace2,
        lambda current: _replace_second_layer(current, current.utility2, eligible2, key2),
        lambda current: (current, jnp.asarray(0, dtype=jnp.int32)),
        updated,
    )
    updated = updated._replace(
        replacement_credit2=jnp.where(replace2, credit2 - 1.0, credit2)
    )
    prediction = jnp.argmax(logits)
    return updated, loss, prediction, count1 + count2


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    data_steps: int
    data_bytes_read: int
    training_model_queries: int
    diagnostic_model_queries: int
    model_queries: int
    parameter_updates: int
    replacements: int
    logical_forward_macs: int
    logical_gradient_macs: int
    persistent_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name != "timing_telemetry_only":
                _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.data_steps == 0 or self.model_queries == 0 or self.persistent_bytes == 0:
            raise ValueError("resource receipt must describe a non-empty run")
        if self.timing_telemetry_only is not True:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    task_accuracy: tuple[float, ...]
    task_loss: tuple[float, ...]
    dead_unit_fraction: tuple[float, ...]
    effective_rank: tuple[float, ...]
    final_state_sha256: str
    receipt: ResourceReceipt

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm")
        curves = (
            self.task_accuracy,
            self.task_loss,
            self.dead_unit_fraction,
            self.effective_rank,
        )
        if any(type(curve) is not tuple for curve in curves):
            raise ValueError("diagnostic curves must be exact tuples")
        lengths = {
            len(self.task_accuracy),
            len(self.task_loss),
            len(self.dead_unit_fraction),
            len(self.effective_rank),
        }
        if (
            type(self.task_accuracy) is not tuple
            or lengths != {len(self.task_accuracy)}
            or not self.task_accuracy
        ):
            raise ValueError("diagnostic curves must be equal non-empty exact tuples")
        for value in self.task_accuracy + self.dead_unit_fraction:
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("probability diagnostics must be finite exact floats")
        for value in self.task_loss + self.effective_rank:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("loss and rank diagnostics must be finite and nonnegative")
        if len(self.final_state_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.final_state_sha256
        ):
            raise ValueError("final state identity must be a lowercase SHA-256")


@dataclasses.dataclass(frozen=True, slots=True)
class DiagnosticResult:
    schema: str
    paper_revision: str
    official_code_commit: str
    profile_id: str
    profile: DiagnosticProfile
    seed: int
    dataset_sha256: str
    source_sha256: str
    runtime_identity: tuple[str, str, str, str]
    task_protocol: str
    labels_permuted: bool
    task_boundary_available_to_learner: bool
    task_id_available_to_learner: bool
    arms: tuple[ArmResult, ...]
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not str
            or type(self.paper_revision) is not str
            or self.schema != SCHEMA
            or self.paper_revision != PAPER_REVISION
        ):
            raise ValueError("schema or paper revision drift")
        if (
            type(self.official_code_commit) is not str
            or self.official_code_commit != OFFICIAL_CODE_COMMIT
        ):
            raise ValueError("official code revision drift")
        if type(self.profile_id) is not str or self.profile_id not in PROFILES:
            raise ValueError("profile or frozen seed drift")
        if type(self.profile) is not DiagnosticProfile or self.profile != PROFILES[self.profile_id]:
            raise ValueError("profile payload differs from the immutable registry")
        if type(self.seed) is not int or self.seed not in FROZEN_SEEDS:
            raise ValueError("profile or frozen seed drift")
        if (
            type(self.dataset_sha256) is not str
            or len(self.dataset_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.dataset_sha256)
        ):
            raise ValueError("dataset identity drift")
        if self.source_sha256 != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("current ASI source identity drift")
        if self.runtime_identity != _runtime_identity():
            raise ValueError("current runtime identity drift")
        if (
            self.task_protocol != "cumulative-input-permutation"
            or self.labels_permuted is not False
        ):
            raise ValueError("the lane must not be described as random-label MNIST")
        flags = (
            self.task_boundary_available_to_learner is False,
            self.task_id_available_to_learner is False,
            self.development_only is True,
            self.scientific_promotion_allowed is False,
            self.negative_results_must_be_retained is True,
        )
        if not all(flags):
            raise ValueError("information or nonpromotion boundary drift")
        if type(self.arms) is not tuple or tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arm roster drift")


def _arrays(images: object, labels: object) -> tuple[np.ndarray, np.ndarray]:
    if type(images) is not np.ndarray or images.dtype != np.float32:
        raise ValueError("images must be an exact float32 ndarray")
    if type(labels) is not np.ndarray or labels.dtype != np.int32:
        raise ValueError("labels must be an exact int32 ndarray")
    if images.ndim != 2 or images.shape[1] != INPUT_DIM or labels.shape != (images.shape[0],):
        raise ValueError("MNIST arrays must have shapes [N,784] and [N]")
    if not 2 <= images.shape[0] <= MAX_DATASET_EXAMPLES:
        raise ValueError("MNIST dataset size is empty or unbounded")
    if not np.all(np.isfinite(images)) or np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("MNIST pixels must be finite and normalized to [0,1]")
    if np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise ValueError("MNIST labels must lie in [0,9]")
    return images, labels


def _dataset_sha(images: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"mnist-train-float32-int32-v1\0")
    digest.update(np.asarray(images.shape, dtype="<i8").tobytes())
    digest.update(images.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(labels.astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _schedule(
    images: np.ndarray, labels: np.ndarray, profile: DiagnosticProfile, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    key = jr.key(seed, impl="threefry2x32")
    permutation = np.arange(INPUT_DIM, dtype=np.int32)
    tasks: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(profile.n_tasks):
        key, pixel_key, data_key = jr.split(key, 3)
        next_permutation = np.asarray(jr.permutation(pixel_key, INPUT_DIM), dtype=np.int32)
        permutation = permutation[next_permutation]
        order = np.asarray(jr.permutation(data_key, images.shape[0]), dtype=np.int32)
        indices = order[: profile.examples_per_task]
        tasks.append(
            (np.ascontiguousarray(images[indices][:, permutation]), labels[indices].copy())
        )
    return tuple(tasks)


def _state_sha256(state: MLPState) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(state):
        value = np.asarray(leaf)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _effective_rank(features: np.ndarray) -> float:
    singular = np.linalg.svd(features, compute_uv=False)
    total = float(singular.sum())
    if total == 0.0:
        return 0.0
    probabilities = singular / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(math.exp(entropy))


def _run_arm(
    arm_id: str,
    tasks: tuple[tuple[np.ndarray, np.ndarray], ...],
    profile: DiagnosticProfile,
    seed: int,
) -> ArmResult:
    key = jr.key(seed)
    key, init_key = jr.split(key)
    state = _init_state(init_key, profile.hidden_width)
    rate = 0.0 if arm_id in ("sgd_control", "cbp_mechanism_off") else profile.replacement_rate
    accuracies: list[float] = []
    losses: list[float] = []
    dead: list[float] = []
    ranks: list[float] = []
    replacements = 0
    start = time.perf_counter_ns()
    for inputs, labels in tasks:
        correct = 0
        task_losses: list[float] = []
        for inputs_row, label in zip(inputs, labels, strict=True):
            key, step_key = jr.split(key)
            state, loss, prediction, replaced = _step(
                state,
                jnp.asarray(inputs_row),
                jnp.asarray(label),
                step_key,
                jnp.asarray(profile.learning_rate, dtype=jnp.float32),
                jnp.asarray(rate, dtype=jnp.float32),
                jnp.asarray(profile.maturity_threshold, dtype=jnp.int32),
            )
            correct += int(prediction == label)
            task_losses.append(float(loss))
            replacements += int(replaced)
        _, hidden1, hidden2 = _forward(state, jnp.asarray(inputs))
        host1, host2 = np.asarray(hidden1), np.asarray(hidden2)
        dead_count = np.sum(np.all(host1 == 0.0, axis=0)) + np.sum(
            np.all(host2 == 0.0, axis=0)
        )
        accuracies.append(float(correct / len(labels)))
        losses.append(float(np.mean(task_losses)))
        dead.append(float(dead_count / (2 * profile.hidden_width)))
        ranks.append(_effective_rank(host2))
    elapsed = time.perf_counter_ns() - start
    steps = profile.n_tasks * profile.examples_per_task
    training_queries = steps * 2
    diagnostic_queries = steps
    model_queries = training_queries + diagnostic_queries
    forward_macs = (
        INPUT_DIM * profile.hidden_width
        + profile.hidden_width * profile.hidden_width
        + profile.hidden_width * N_CLASSES
    )
    state_bytes = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(state))
    return ArmResult(
        arm_id,
        tuple(accuracies),
        tuple(losses),
        tuple(dead),
        tuple(ranks),
        _state_sha256(state),
        ResourceReceipt(
            data_steps=steps,
            data_bytes_read=steps * (INPUT_DIM * 4 + 4),
            training_model_queries=training_queries,
            diagnostic_model_queries=diagnostic_queries,
            model_queries=model_queries,
            parameter_updates=steps,
            replacements=replacements,
            logical_forward_macs=model_queries * forward_macs,
            logical_gradient_macs=training_queries * forward_macs,
            persistent_bytes=state_bytes,
            elapsed_ns=elapsed,
        ),
    )


def run_diagnostic(
    images: object, labels: object, *, seed: object, profile_id: object = "contract-smoke"
) -> DiagnosticResult:
    data, targets = _arrays(images, labels)
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    if type(profile_id) is not str or profile_id not in PROFILES:
        raise ValueError("unknown diagnostic profile")
    profile = PROFILES[profile_id]
    if data.shape[0] < profile.examples_per_task:
        raise ValueError("dataset has too few examples for the frozen profile")
    tasks = _schedule(data, targets, profile, seed)
    result = DiagnosticResult(
        SCHEMA,
        PAPER_REVISION,
        OFFICIAL_CODE_COMMIT,
        profile.profile_id,
        profile,
        seed,
        _dataset_sha(data, targets),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        _runtime_identity(),
        "cumulative-input-permutation",
        False,
        False,
        False,
        tuple(_run_arm(arm, tasks, profile, seed) for arm in ARM_IDS),
    )
    return validate_result(result)


def validate_result(value: object) -> DiagnosticResult:
    if type(value) is not DiagnosticResult:
        raise ValueError("result must be an exact DiagnosticResult")
    DiagnosticResult.__post_init__(value)
    profile = PROFILES[value.profile_id]
    expected_steps = profile.n_tasks * profile.examples_per_task
    first, mechanism_off, candidate = value.arms
    for arm in value.arms:
        if type(arm) is not ArmResult or type(arm.receipt) is not ResourceReceipt:
            raise ValueError("arms and receipts must have exact types")
        ArmResult.__post_init__(arm)
        ResourceReceipt.__post_init__(arm.receipt)
        receipt = arm.receipt
        expected_persistent = sum(
            array.nbytes for array in jax.tree.leaves(_init_state(jr.key(0), profile.hidden_width))
        )
        training_queries = expected_steps * 2
        diagnostic_queries = expected_steps
        model_queries = training_queries + diagnostic_queries
        forward_macs = (
            INPUT_DIM * profile.hidden_width
            + profile.hidden_width * profile.hidden_width
            + profile.hidden_width * N_CLASSES
        )
        if (
            len(arm.task_accuracy) != profile.n_tasks
            or receipt.data_steps != expected_steps
            or receipt.data_bytes_read != expected_steps * (INPUT_DIM * 4 + 4)
            or receipt.training_model_queries != training_queries
            or receipt.diagnostic_model_queries != diagnostic_queries
            or receipt.model_queries != model_queries
            or receipt.parameter_updates != expected_steps
            or receipt.logical_forward_macs != model_queries * forward_macs
            or receipt.logical_gradient_macs != training_queries * forward_macs
            or receipt.persistent_bytes != expected_persistent
            or not 0 <= receipt.replacements <= expected_steps * 2
        ):
            raise ValueError("diagnostic curve or exact resource receipt mismatch")
    if (
        first.task_accuracy != mechanism_off.task_accuracy
        or first.task_loss != mechanism_off.task_loss
        or first.dead_unit_fraction != mechanism_off.dead_unit_fraction
        or first.effective_rank != mechanism_off.effective_rank
        or first.final_state_sha256 != mechanism_off.final_state_sha256
        or first.receipt.replacements != 0
        or mechanism_off.receipt.replacements != 0
    ):
        raise ValueError("CBP mechanism-off does not reduce exactly to SGD")
    if candidate.receipt.replacements == 0 and profile.profile_id == "contract-smoke":
        raise ValueError("contract smoke must exercise the CBP replacement path")
    return value


def costly_lane_gates() -> Mapping[str, object]:
    return {
        "imagenet": {
            "qualified": False,
            "paper_protocol": "2000 binary tasks; 250 epochs; batch 100; 30 runs per step size",
            "minimum_known_cost": "12 A100-hours per run (official README)",
            "blockers": [
                "dataset license/checksum",
                "accelerator budget",
                "head-reset information match",
            ],
        },
        "reinforcement_learning": {
            "qualified": False,
            "paper_protocol": "continual PPO on Slippery Ant and related MuJoCo tasks",
            "minimum_known_cost": (
                "README says 50M steps/24 CPU-hours; current Ant config says 100M"
            ),
            "blockers": ["MuJoCo/runtime pin", "step-count discrepancy", "environment budget"],
        },
        "execution_authorized": False,
    }


def require_costly_lane(lane: object) -> None:
    if type(lane) is not str or lane not in ("imagenet", "reinforcement_learning"):
        raise ValueError("unknown costly lane")
    raise RuntimeError(f"{lane} execution is not qualified or authorized")


def _json_result(result: DiagnosticResult) -> str:
    return json.dumps(dataclasses.asdict(result), sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--seed", type=int, default=FROZEN_SEEDS[0])
    parser.add_argument("--profile", choices=tuple(PROFILES), default="contract-smoke")
    parser.add_argument("--catalog", action="store_true")
    args = parser.parse_args(argv)
    if args.catalog:
        print(
            json.dumps(
                {"schema": SCHEMA, "costly_lane_gates": costly_lane_gates()},
                sort_keys=True,
            )
        )
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --catalog is used")
    if (
        args.dataset.is_symlink()
        or not args.dataset.is_file()
        or args.dataset.stat().st_size > 256 * 1024 * 1024
    ):
        raise ValueError("dataset NPZ must be a bounded regular file")
    with np.load(args.dataset, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "labels"}:
            raise ValueError("dataset NPZ must contain exactly images and labels")
        result = run_diagnostic(
            payload["images"], payload["labels"], seed=args.seed, profile_id=args.profile
        )
    print(_json_result(result))
    return 0
