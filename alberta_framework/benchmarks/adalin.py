"""AdaLin equation primitives and non-comparable PMNIST protocol declaration."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

ADALIN_PAPER_URL = "https://arxiv.org/abs/2505.09486v1"
ADALIN_OFFICIAL_REPOSITORY = "https://github.com/RoozbehRazavi/AdaLin"
ADALIN_OFFICIAL_COMMIT = "011469138bc22bf82955b16d68f33e4fbd04e3f8"
ADALIN_OFFICIAL_TREE_AUDIT = "single README.md blob containing '# AdaLin'; no runnable code"
ADALIN_RESULT_SCHEMA = "asi.adalin.pmnist-development-result.v2"
ADALIN_MATCHED_RESULT_SCHEMA = "asi.adalin.pmnist-matched-development-result.v1"
ADALIN_MATCHED_DEVELOPMENT_SEEDS = (1_571_001, 1_571_002, 1_571_003)

ADALIN_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.adalin.protocol.v1",
        "paper_revision": "arXiv:2505.09486v1",
        "paper_revision_date": "2025-05-14",
        "paper_url": ADALIN_PAPER_URL,
        "official_repository": ADALIN_OFFICIAL_REPOSITORY,
        "official_commit": ADALIN_OFFICIAL_COMMIT,
        "official_commit_audit": ADALIN_OFFICIAL_TREE_AUDIT,
        "paper_pmnist_tasks": 400,
        "paper_examples_per_task": 10_000,
        "paper_batch_size": 16,
        "paper_hidden_widths": (100, 100),
        "asi_target_tasks": 200,
        "asi_examples_per_task": 5_000,
        "asi_batch_size": 1,
        "asi_hidden_widths": (300, 150),
        "learner_observes_task_boundary": False,
        "mechanism_off": "alpha_zero_exact_base_activation",
        "finite_kernel_preflight_required": True,
        "matched_axes": ("seed", "updates", "observations"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_MAX_ARRAY_ELEMENTS = 1_000_000
_MAX_DATASET_ELEMENTS = 10_000_000
_INT32_MAX = 2**31 - 1


def _trusted_float_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.size < 1
        or array.size > _MAX_ARRAY_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise ValueError(f"{name} must be a bounded floating array")
    return array


def _adalin_transaction(
    x: Array, alpha: Array, *, activation: Array, derivative: Array
) -> tuple[Array, Array]:
    try:
        output_shape = np.broadcast_shapes(x.shape, alpha.shape, activation.shape, derivative.shape)
    except ValueError as error:
        raise ValueError("x and alpha must have compatible shapes") from error
    if math.prod(output_shape) > _MAX_ARRAY_ELEMENTS:
        raise ValueError("broadcast output exceeds the 1000000-element limit")
    gate = jax.lax.stop_gradient(jnp.cos(0.5 * jnp.pi * jnp.abs(derivative)))
    candidate = activation + alpha * x * gate
    valid = (
        jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(alpha)) & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def _unwrap_transaction(result: tuple[Array, Array]) -> Array:
    safe, valid = result
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("AdaLin activation must be finite")
    return safe


def adalin_relu(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for ReLU (algebraically PReLU)."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    return _unwrap_transaction(
        _adalin_transaction(
            value,
            coefficient,
            activation=jax.nn.relu(value),
            derivative=(value > 0).astype(value.dtype),
        )
    )


def adalin_relu_transaction(x: object, alpha: object) -> tuple[Array, Array]:
    """Return a finite ReLU-AdaLin value and caller-visible validity bit."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    return _adalin_transaction(
        value,
        coefficient,
        activation=jax.nn.relu(value),
        derivative=(value > 0).astype(value.dtype),
    )


def adalin_tanh(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for tanh, whose Lipschitz constant is one."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    activation = jnp.tanh(value)
    return _unwrap_transaction(
        _adalin_transaction(
            value,
            coefficient,
            activation=activation,
            derivative=1.0 - activation**2,
        )
    )


def adalin_tanh_transaction(x: object, alpha: object) -> tuple[Array, Array]:
    """Return a finite tanh-AdaLin value and caller-visible validity bit."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    activation = jnp.tanh(value)
    return _adalin_transaction(
        value,
        coefficient,
        activation=activation,
        derivative=1.0 - activation**2,
    )


@dataclass(frozen=True)
class AdaLinConfig:
    """One PMNIST schedule; defaults reproduce the paper's declared axes."""

    tasks: int = 400
    examples_per_task: int = 10_000
    batch_size: int = 16
    hidden_widths: tuple[int, int] = (100, 100)
    learning_rate: float = 1e-2
    epochs_per_task: int = 1
    classes: int = 10

    def __post_init__(self) -> None:
        integer_fields = {
            "tasks": self.tasks,
            "examples_per_task": self.examples_per_task,
            "batch_size": self.batch_size,
            "epochs_per_task": self.epochs_per_task,
            "classes": self.classes,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value < 1 or value > _INT32_MAX:
                raise ValueError(f"{name} must be a positive int32 integer")
        if self.tasks > 400:
            raise ValueError("tasks cannot exceed the paper's 400-task horizon")
        if self.examples_per_task > 10_000:
            raise ValueError("examples_per_task cannot exceed the paper's 10000-example pool")
        if self.epochs_per_task != 1:
            raise ValueError("this PMNIST lane supports the paper's one epoch per task")
        if self.examples_per_task % self.batch_size != 0:
            raise ValueError("examples_per_task must be divisible by batch_size")
        if (
            type(self.hidden_widths) is not tuple
            or len(self.hidden_widths) != 2
            or any(
                type(width) is not int or width < 1 or width > 10_000
                for width in self.hidden_widths
            )
        ):
            raise ValueError("hidden_widths must contain exactly two positive bounded integers")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a finite positive float")
        total_updates = self.tasks * (self.examples_per_task // self.batch_size)
        if total_updates > _INT32_MAX:
            raise ValueError("configuration exceeds the int32 update horizon")


PAPER_PMNIST_CONFIG = AdaLinConfig()


@chex.dataclass(frozen=True)
class AdaLinParameters:
    """All learned MLP values, including one owned alpha per hidden neuron."""

    weight1: Array
    bias1: Array
    alpha1: Array
    weight2: Array
    bias2: Array
    alpha2: Array
    weight3: Array
    bias3: Array


@chex.dataclass(frozen=True)
class AdaLinState:
    """Persisted learner state; task changes never replace this aggregate."""

    parameters: AdaLinParameters
    updates: Array


@chex.dataclass(frozen=True)
class PMNISTSchedule:
    """Deterministic, runner-owned schedule that is never shown to the learner."""

    pixel_permutations: Array
    example_orders: Array


def _require_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0 or seed > _INT32_MAX:
        raise ValueError("seed must be an integer in [0, 2147483647]")
    return seed


def make_pmnist_schedule(config: AdaLinConfig, *, seed: int, input_dim: int) -> PMNISTSchedule:
    """Construct independent pixel and example permutations from explicit Threefry keys."""
    if type(config) is not AdaLinConfig:
        raise ValueError("config must be an exact AdaLinConfig")
    resolved_seed = _require_seed(seed)
    if type(input_dim) is not int or input_dim < 1 or input_dim > _MAX_ARRAY_ELEMENTS:
        raise ValueError("input_dim must be a positive bounded integer")
    if config.tasks * (input_dim + config.examples_per_task) > _MAX_DATASET_ELEMENTS:
        raise ValueError("schedule exceeds the 10000000-element allocation limit")
    root = jr.key(resolved_seed)
    pixel_root = jr.fold_in(root, 0x414441)
    example_root = jr.fold_in(root, 0x504D4E)
    pixels = jnp.stack(
        [jr.permutation(jr.fold_in(pixel_root, task), input_dim) for task in range(config.tasks)]
    )
    examples = jnp.stack(
        [
            jr.permutation(jr.fold_in(example_root, task), config.examples_per_task)
            for task in range(config.tasks)
        ]
    )
    return PMNISTSchedule(  # type: ignore[call-arg]
        pixel_permutations=pixels, example_orders=examples
    )


def initialize_adalin_state(
    config: AdaLinConfig,
    *,
    input_dim: int,
    classes: int,
    seed: int,
    mechanism_enabled: bool = True,
) -> AdaLinState:
    """Initialize one continuous learner using paper-style uniform alpha in ``[0, 1)``."""
    if type(config) is not AdaLinConfig:
        raise ValueError("config must be an exact AdaLinConfig")
    _require_seed(seed)
    if type(input_dim) is not int or input_dim < 1 or input_dim > _MAX_ARRAY_ELEMENTS:
        raise ValueError("input_dim must be a positive bounded integer")
    if type(classes) is not int or classes < 2 or classes != config.classes:
        raise ValueError("classes must equal config.classes and be at least two")
    if type(mechanism_enabled) is not bool:
        raise ValueError("mechanism_enabled must be a bool")
    width1, width2 = config.hidden_widths
    parameter_count = (
        input_dim * width1 + width1 * 2 + width1 * width2 + width2 * 2 + width2 * classes + classes
    )
    if parameter_count > _MAX_DATASET_ELEMENTS:
        raise ValueError("model exceeds the 10000000-element allocation limit")
    keys = jr.split(jr.fold_in(jr.key(seed), 0x494E49), 5)
    weight1 = jr.normal(keys[0], (input_dim, width1), dtype=jnp.float32) * math.sqrt(
        2.0 / input_dim
    )
    weight2 = jr.normal(keys[1], (width1, width2), dtype=jnp.float32) * math.sqrt(2.0 / width1)
    weight3 = jr.normal(keys[2], (width2, classes), dtype=jnp.float32) * math.sqrt(2.0 / width2)
    alpha1 = (
        jr.uniform(keys[3], (width1,), dtype=jnp.float32)
        if mechanism_enabled
        else jnp.zeros((width1,), jnp.float32)
    )
    alpha2 = (
        jr.uniform(keys[4], (width2,), dtype=jnp.float32)
        if mechanism_enabled
        else jnp.zeros((width2,), jnp.float32)
    )
    parameters = AdaLinParameters(  # type: ignore[call-arg]
        weight1=weight1,
        bias1=jnp.zeros((width1,), jnp.float32),
        alpha1=alpha1,
        weight2=weight2,
        bias2=jnp.zeros((width2,), jnp.float32),
        alpha2=alpha2,
        weight3=weight3,
        bias3=jnp.zeros((classes,), jnp.float32),
    )
    return AdaLinState(  # type: ignore[call-arg]
        parameters=parameters, updates=jnp.asarray(0, dtype=jnp.uint32)
    )


def _mlp_logits(parameters: AdaLinParameters, inputs: Array, mechanism_enabled: bool) -> Array:
    hidden1_input = inputs @ parameters.weight1 + parameters.bias1
    hidden1 = (
        adalin_relu(hidden1_input, parameters.alpha1)
        if mechanism_enabled
        else jax.nn.relu(hidden1_input)
    )
    hidden2_input = hidden1 @ parameters.weight2 + parameters.bias2
    hidden2 = (
        adalin_relu(hidden2_input, parameters.alpha2)
        if mechanism_enabled
        else jax.nn.relu(hidden2_input)
    )
    return hidden2 @ parameters.weight3 + parameters.bias3


def adalin_logits(
    parameters: AdaLinParameters, inputs: object, *, mechanism_enabled: bool
) -> Array:
    """Evaluate the two-hidden-layer MLP without exposing task identity."""
    if type(parameters) is not AdaLinParameters:
        raise ValueError("parameters must be an exact AdaLinParameters")
    values = _trusted_float_array(inputs, name="inputs")
    if values.ndim != 2 or values.shape[1] != parameters.weight1.shape[0]:
        raise ValueError("inputs must be a batch matching the model input width")
    if type(mechanism_enabled) is not bool:
        raise ValueError("mechanism_enabled must be a bool")
    return _mlp_logits(parameters, values, mechanism_enabled)


def _cross_entropy(
    parameters: AdaLinParameters, inputs: Array, labels: Array, enabled: bool
) -> Array:
    logits = _mlp_logits(parameters, inputs, enabled)
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], axis=1))


def adalin_gradients(
    parameters: AdaLinParameters,
    inputs: object,
    labels: object,
    *,
    mechanism_enabled: bool,
) -> AdaLinParameters:
    """Return full gradients; the off arm has identically zero alpha gradients."""
    if type(parameters) is not AdaLinParameters:
        raise ValueError("parameters must be an exact AdaLinParameters")
    values = _trusted_float_array(inputs, name="inputs")
    actual_type = type(labels)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError("labels must be an exact NumPy or JAX array")
    targets = jnp.asarray(labels)
    if (
        values.ndim != 2
        or values.shape[1] != parameters.weight1.shape[0]
        or targets.ndim != 1
        or targets.shape[0] != values.shape[0]
        or not jnp.issubdtype(targets.dtype, jnp.integer)
    ):
        raise ValueError("inputs and integer labels must form one compatible batch")
    if type(mechanism_enabled) is not bool:
        raise ValueError("mechanism_enabled must be a bool")
    return cast(
        AdaLinParameters,
        jax.grad(_cross_entropy)(parameters, values, targets, mechanism_enabled),
    )


def adalin_sgd_step(
    state: AdaLinState,
    inputs: object,
    labels: object,
    *,
    learning_rate: float,
    mechanism_enabled: bool,
) -> AdaLinState:
    """Apply one SGD update; traced invalidity is exposed through NaN parameters."""
    safe, valid = adalin_sgd_step_transaction(
        state,
        inputs,
        labels,
        learning_rate=learning_rate,
        mechanism_enabled=mechanism_enabled,
    )
    if isinstance(valid, jax.core.Tracer):
        exposed = jax.tree.map(
            lambda value: jnp.where(valid, value, jnp.full_like(value, jnp.nan)),
            safe.parameters,
        )
        return AdaLinState(  # type: ignore[call-arg]
            parameters=exposed, updates=safe.updates
        )
    if not bool(valid):
        raise ValueError("AdaLin update produced non-finite state")
    return safe


def adalin_sgd_step_transaction(
    state: AdaLinState,
    inputs: object,
    labels: object,
    *,
    learning_rate: float,
    mechanism_enabled: bool,
) -> tuple[AdaLinState, Array]:
    """Return a finite candidate and a caller-visible update-validity bit."""
    if type(state) is not AdaLinState:
        raise ValueError("state must be an exact AdaLinState")
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive float")
    gradients = adalin_gradients(
        state.parameters, inputs, labels, mechanism_enabled=mechanism_enabled
    )
    candidate = jax.tree.map(
        lambda parameter, gradient: parameter - learning_rate * gradient,
        state.parameters,
        gradients,
    )
    finite = jax.tree_util.tree_reduce(
        lambda left, value: left & jnp.all(jnp.isfinite(value)), candidate, jnp.asarray(True)
    )
    safe = jax.tree.map(lambda new, old: jnp.where(finite, new, old), candidate, state.parameters)
    updates = jnp.where(finite, state.updates + jnp.asarray(1, jnp.uint32), state.updates)
    return AdaLinState(parameters=safe, updates=updates), finite  # type: ignore[call-arg]


def _require_dataset(
    inputs: object, labels: object, *, name: str, expected_examples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if type(inputs) is not np.ndarray or type(labels) is not np.ndarray:
        raise ValueError(f"{name} inputs and labels must be exact NumPy arrays")
    if (
        inputs.ndim != 2
        or inputs.size < 1
        or inputs.size > _MAX_DATASET_ELEMENTS
        or inputs.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(inputs))
    ):
        raise ValueError(f"{name} inputs must be a bounded finite float32 matrix")
    if labels.ndim != 1 or labels.shape[0] != inputs.shape[0] or labels.dtype != np.dtype(np.int32):
        raise ValueError(f"{name} labels must be a matching int32 vector")
    if expected_examples is not None and inputs.shape[0] != expected_examples:
        raise ValueError(f"{name} must contain exactly examples_per_task rows")
    return inputs, labels


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_hash(state: AdaLinState) -> str:
    leaves = jax.tree_util.tree_leaves(state)
    return _hash_arrays(*(np.asarray(leaf) for leaf in leaves))


def _shared_parameter_hash(parameters: AdaLinParameters) -> str:
    """Hash the matched initialization surface, excluding mechanism-owned alpha."""
    return _hash_arrays(
        *(np.asarray(value) for value in (
            parameters.weight1,
            parameters.bias1,
            parameters.weight2,
            parameters.bias2,
            parameters.weight3,
            parameters.bias3,
        ))
    )


def _implementation_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _parameter_bytes(parameters: AdaLinParameters) -> int:
    return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(parameters))


def _alpha_bytes(parameters: AdaLinParameters) -> int:
    return int(np.asarray(parameters.alpha1).nbytes + np.asarray(parameters.alpha2).nbytes)


def _accuracy(logits: Array, labels: Array) -> tuple[int, int]:
    predictions = jnp.argmax(logits, axis=1)
    return int(jnp.sum(predictions == labels)), int(labels.shape[0])


def run_adalin_development(
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    config: AdaLinConfig = PAPER_PMNIST_CONFIG,
    seed: int,
    mechanism_enabled: bool = True,
) -> dict[str, object]:
    """Run one continuous PMNIST life; task boundaries affect data, never the learner."""
    if type(config) is not AdaLinConfig:
        raise ValueError("config must be an exact AdaLinConfig")
    resolved_seed = _require_seed(seed)
    if type(mechanism_enabled) is not bool:
        raise ValueError("mechanism_enabled must be a bool")
    train_x, train_y = _require_dataset(
        train_inputs, train_labels, name="train", expected_examples=config.examples_per_task
    )
    test_x, test_y = _require_dataset(test_inputs, test_labels, name="test")
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test input widths must match")
    if (
        np.any(train_y < 0)
        or np.any(train_y >= config.classes)
        or np.any(test_y < 0)
        or np.any(test_y >= config.classes)
    ):
        raise ValueError("labels must lie in [0, config.classes)")
    schedule = make_pmnist_schedule(config, seed=resolved_seed, input_dim=train_x.shape[1])
    state = initialize_adalin_state(
        config,
        input_dim=train_x.shape[1],
        classes=config.classes,
        seed=resolved_seed,
        mechanism_enabled=mechanism_enabled,
    )
    initial_hash = _state_hash(state)
    initial_shared_hash = _shared_parameter_hash(state.parameters)
    initial_bytes = _parameter_bytes(state.parameters) + int(np.asarray(state.updates).nbytes)
    task_online: list[float] = []
    task_test: list[float] = []
    online_correct = 0
    model_forward_calls = 0
    started = time.perf_counter()
    for task in range(config.tasks):
        pixels = np.asarray(schedule.pixel_permutations[task])
        order = np.asarray(schedule.example_orders[task])
        task_correct = 0
        for start in range(0, config.examples_per_task, config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_x = jnp.asarray(train_x[indices][:, pixels])
            batch_y = jnp.asarray(train_y[indices])
            correct, _ = _accuracy(
                adalin_logits(state.parameters, batch_x, mechanism_enabled=mechanism_enabled),
                batch_y,
            )
            task_correct += correct
            online_correct += correct
            model_forward_calls += 1
            state = adalin_sgd_step(
                state,
                batch_x,
                batch_y,
                learning_rate=config.learning_rate,
                mechanism_enabled=mechanism_enabled,
            )
        task_online.append(task_correct / config.examples_per_task)
        test_correct, test_total = _accuracy(
            adalin_logits(
                state.parameters,
                jnp.asarray(test_x[:, pixels]),
                mechanism_enabled=mechanism_enabled,
            ),
            jnp.asarray(test_y),
        )
        model_forward_calls += 1
        task_test.append(test_correct / test_total)
    wall_clock = time.perf_counter() - started
    observations = config.tasks * config.examples_per_task
    optimizer_updates = observations // config.batch_size
    test_queries = config.tasks * test_x.shape[0]
    final_bytes = _parameter_bytes(state.parameters) + int(np.asarray(state.updates).nbytes)
    schedule_hash = _hash_arrays(
        np.asarray(schedule.pixel_permutations), np.asarray(schedule.example_orders)
    )
    result: dict[str, object] = {
        "schema": ADALIN_RESULT_SCHEMA,
        "protocol": json.loads(json.dumps(dict(ADALIN_PROTOCOL))),
        "config": json.loads(json.dumps(asdict(config))),
        "arm": "adalin" if mechanism_enabled else "relu_alpha_zero_mechanism_off",
        "seed": resolved_seed,
        "dataset": {
            "sha256": _hash_arrays(train_x, train_y, test_x, test_y),
            "train_examples": int(train_x.shape[0]),
            "test_examples": int(test_x.shape[0]),
            "input_dim": int(train_x.shape[1]),
            "classes": config.classes,
        },
        "provenance": {
            "paper": ADALIN_PAPER_URL,
            "official_repository": ADALIN_OFFICIAL_REPOSITORY,
            "official_commit": ADALIN_OFFICIAL_COMMIT,
            "official_commit_audit": ADALIN_OFFICIAL_TREE_AUDIT,
            "implementation_sha256": _implementation_hash(),
            "schedule_sha256": schedule_hash,
            "initial_state_sha256": initial_hash,
            "initial_shared_parameters_sha256": initial_shared_hash,
            "final_state_sha256": _state_hash(state),
        },
        "metrics": {
            "asi_whole_stream_preupdate_online_accuracy": online_correct / observations,
            "paper_current_task_test_accuracy_mean": float(np.mean(task_test)),
            "task_preupdate_online_accuracy": task_online,
            "task_postupdate_test_accuracy": task_test,
        },
        "state": {
            "updates": int(state.updates),
            "initial_total_bytes": initial_bytes,
            "final_total_bytes": final_bytes,
            "parameter_bytes": _parameter_bytes(state.parameters),
            "alpha_bytes": _alpha_bytes(state.parameters),
            "optimizer_state_bytes": 0,
            "final_alpha_l2": float(
                jnp.sqrt(jnp.sum(state.parameters.alpha1**2) + jnp.sum(state.parameters.alpha2**2))
            ),
        },
        "resources": {
            "environment_data_steps": observations,
            "observations": observations,
            "label_queries": observations,
            "optimizer_updates": optimizer_updates,
            "model_queries": observations + test_queries,
            "model_forward_calls": model_forward_calls,
            "wall_clock_seconds_telemetry": wall_clock,
            "timing_qualified": False,
        },
        "comparison": {
            "paper_comparable": False,
            "residual_gaps": [
                "the pinned official commit contains no implementation or experiment config",
                "the runner input is caller-supplied and does not bind official MNIST bytes "
                "or the paper's unspecified sampled indices",
                "the paper does not specify pixel/example permutation seeds or exact "
                "dataloader order",
                "one result is one consumed development seed rather than the paper's three "
                "final evaluation seeds",
                "ASI records whole-stream pre-update accuracy separately from the paper's "
                "per-task train/test summaries",
            ],
            "schedule_difference": (
                "paper: 400x10000, batch16, one epoch; ASI campaign: 200x5000, "
                "batch1; this result uses its embedded config"
            ),
            "boundary_difference": (
                "paper and this runner do not pass task identity or boundary events to the "
                "learner; the runner uses boundaries only to transform/evaluate data"
            ),
        },
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
    }
    validate_adalin_result(result)
    return result


_RESULT_KEYS = {
    "schema",
    "protocol",
    "config",
    "arm",
    "seed",
    "dataset",
    "provenance",
    "metrics",
    "state",
    "resources",
    "comparison",
    "policy",
}


def _exact_dict(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact string-keyed mapping")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in dict.keys(result)):
        raise ValueError(f"{name} must be an exact string-keyed mapping")
    return cast(dict[str, object], result)


def _require_builtin_json(value: object, *, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("result nesting exceeds the validation limit")
    actual_type = type(value)
    if actual_type is dict:
        mapping = cast(dict[object, object], value)
        if len(mapping) > 256:
            raise ValueError("result mapping exceeds the validation limit")
        for key, item in dict.items(mapping):
            if type(key) is not str:
                raise ValueError("result keys must be exact strings")
            _require_builtin_json(item, depth=depth + 1)
        return
    if actual_type is list:
        sequence = cast(list[object], value)
        if len(sequence) > _MAX_ARRAY_ELEMENTS:
            raise ValueError("result list exceeds the validation limit")
        for item in list.__iter__(sequence):
            _require_builtin_json(item, depth=depth + 1)
        return
    if actual_type not in {str, int, float, bool, type(None)}:
        raise ValueError("result must contain only exact JSON builtin values")


def _exact_finite_number(value: object, name: str) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError(f"{name} must be an exact finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be an exact finite number")
    return result


def _require_keys(mapping: dict[str, object], keys: set[str], name: str) -> None:
    if set(dict.keys(mapping)) != keys:
        raise ValueError(f"{name} fields do not match the schema")


def validate_adalin_result(value: object) -> None:
    """Validate canonical policy, provenance, metrics, and resource relationships."""
    _require_builtin_json(value)
    result = _exact_dict(value, "result")
    _require_keys(result, _RESULT_KEYS, "result")
    if type(result["schema"]) is not str or result["schema"] != ADALIN_RESULT_SCHEMA:
        raise ValueError("unexpected AdaLin result schema")
    protocol = _exact_dict(result["protocol"], "protocol")
    if protocol != json.loads(json.dumps(dict(ADALIN_PROTOCOL))):
        raise ValueError("protocol does not match the current declaration")
    raw_config = _exact_dict(result["config"], "config")
    expected_config_keys = set(asdict(PAPER_PMNIST_CONFIG))
    _require_keys(raw_config, expected_config_keys, "config")
    try:
        config_values = dict(raw_config)
        widths = config_values.get("hidden_widths")
        if (
            type(widths) is not list
            or len(widths) != 2
            or any(type(width) is not int for width in widths)
        ):
            raise ValueError("config.hidden_widths must be an exact two-item integer list")
        config_values["hidden_widths"] = tuple(widths)
        config = AdaLinConfig(**cast(dict[str, Any], config_values))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid AdaLin config") from error
    if type(result["seed"]) is not int:
        raise ValueError("seed must be an exact integer")
    _require_seed(result["seed"])
    if type(result["arm"]) is not str or result["arm"] not in {
        "adalin",
        "relu_alpha_zero_mechanism_off",
    }:
        raise ValueError("unexpected arm")
    dataset = _exact_dict(result["dataset"], "dataset")
    _require_keys(
        dataset, {"sha256", "train_examples", "test_examples", "input_dim", "classes"}, "dataset"
    )
    for field in ("train_examples", "test_examples", "input_dim", "classes"):
        if type(dataset[field]) is not int or cast(int, dataset[field]) < 1:
            raise ValueError(f"dataset.{field} must be a positive exact integer")
    if (
        dataset["train_examples"] != config.examples_per_task
        or dataset["classes"] != config.classes
    ):
        raise ValueError("dataset metadata disagrees with config")
    provenance = _exact_dict(result["provenance"], "provenance")
    _require_keys(
        provenance,
        {
            "paper",
            "official_repository",
            "official_commit",
            "official_commit_audit",
            "implementation_sha256",
            "schedule_sha256",
            "initial_state_sha256",
            "initial_shared_parameters_sha256",
            "final_state_sha256",
        },
        "provenance",
    )
    expected_literals = {
        "paper": ADALIN_PAPER_URL,
        "official_repository": ADALIN_OFFICIAL_REPOSITORY,
        "official_commit": ADALIN_OFFICIAL_COMMIT,
        "official_commit_audit": ADALIN_OFFICIAL_TREE_AUDIT,
        "implementation_sha256": _implementation_hash(),
    }
    for field, expected in expected_literals.items():
        if type(provenance[field]) is not str or provenance[field] != expected:
            raise ValueError(f"provenance.{field} does not match current source")
    for field in (
        "schedule_sha256",
        "initial_state_sha256",
        "initial_shared_parameters_sha256",
        "final_state_sha256",
        "implementation_sha256",
    ):
        candidate = provenance[field]
        if (
            type(candidate) is not str
            or len(candidate) != 64
            or any(char not in "0123456789abcdef" for char in candidate)
        ):
            raise ValueError(f"provenance.{field} must be a lowercase SHA-256 digest")
    resources = _exact_dict(result["resources"], "resources")
    _require_keys(
        resources,
        {
            "environment_data_steps",
            "observations",
            "label_queries",
            "optimizer_updates",
            "model_queries",
            "model_forward_calls",
            "wall_clock_seconds_telemetry",
            "timing_qualified",
        },
        "resources",
    )
    observations = config.tasks * config.examples_per_task
    updates = observations // config.batch_size
    expected_resources: dict[str, object] = {
        "environment_data_steps": observations,
        "observations": observations,
        "label_queries": observations,
        "optimizer_updates": updates,
        "model_queries": observations + config.tasks * cast(int, dataset["test_examples"]),
        "model_forward_calls": updates + config.tasks,
        "timing_qualified": False,
    }
    for resource_field, resource_expected in expected_resources.items():
        if (
            type(resources[resource_field]) is not type(resource_expected)
            or resources[resource_field] != resource_expected
        ):
            raise ValueError(f"resources.{resource_field} disagrees with the canonical count")
    if _exact_finite_number(resources["wall_clock_seconds_telemetry"], "wall clock") < 0:
        raise ValueError("wall clock telemetry cannot be negative")
    state = _exact_dict(result["state"], "state")
    _require_keys(
        state,
        {
            "updates",
            "initial_total_bytes",
            "final_total_bytes",
            "parameter_bytes",
            "alpha_bytes",
            "optimizer_state_bytes",
            "final_alpha_l2",
        },
        "state",
    )
    for field in (
        "updates",
        "initial_total_bytes",
        "final_total_bytes",
        "parameter_bytes",
        "alpha_bytes",
        "optimizer_state_bytes",
    ):
        if type(state[field]) is not int or cast(int, state[field]) < 0:
            raise ValueError(f"state.{field} must be a nonnegative exact integer")
    if state["updates"] != updates or state["optimizer_state_bytes"] != 0:
        raise ValueError("state counters disagree with SGD configuration")
    width1, width2 = config.hidden_widths
    input_dim = cast(int, dataset["input_dim"])
    parameter_scalars = (
        input_dim * width1
        + width1
        + width1
        + width1 * width2
        + width2
        + width2
        + width2 * config.classes
        + config.classes
    )
    if state["parameter_bytes"] != parameter_scalars * 4:
        raise ValueError("parameter bytes disagree with the configured float32 model")
    if state["alpha_bytes"] != (width1 + width2) * 4:
        raise ValueError("alpha bytes disagree with per-neuron ownership")
    if (
        state["initial_total_bytes"] != state["final_total_bytes"]
        or state["final_total_bytes"] != state["parameter_bytes"] + 4
    ):
        raise ValueError("persistent byte counts are inconsistent")
    alpha_l2 = _exact_finite_number(state["final_alpha_l2"], "final alpha norm")
    if alpha_l2 < 0 or (result["arm"] == "relu_alpha_zero_mechanism_off" and alpha_l2 != 0):
        raise ValueError("mechanism-off alpha state must remain exactly zero")
    metrics = _exact_dict(result["metrics"], "metrics")
    _require_keys(
        metrics,
        {
            "asi_whole_stream_preupdate_online_accuracy",
            "paper_current_task_test_accuracy_mean",
            "task_preupdate_online_accuracy",
            "task_postupdate_test_accuracy",
        },
        "metrics",
    )
    online = metrics["task_preupdate_online_accuracy"]
    tested = metrics["task_postupdate_test_accuracy"]
    if (
        type(online) is not list
        or type(tested) is not list
        or len(online) != config.tasks
        or len(tested) != config.tasks
    ):
        raise ValueError("task metrics must be exact lists covering every task")
    for name, series in (("online", online), ("test", tested)):
        for item in cast(list[object], series):
            resolved = _exact_finite_number(item, f"{name} metric")
            if resolved < 0 or resolved > 1:
                raise ValueError(f"{name} metrics must lie in [0, 1]")
    if not math.isclose(
        _exact_finite_number(
            metrics["asi_whole_stream_preupdate_online_accuracy"], "whole-stream metric"
        ),
        float(np.mean(cast(list[float], online))),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("whole-stream metric disagrees with task metrics")
    if not math.isclose(
        _exact_finite_number(metrics["paper_current_task_test_accuracy_mean"], "paper metric"),
        float(np.mean(cast(list[float], tested))),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("paper test metric disagrees with task metrics")
    comparison = _exact_dict(result["comparison"], "comparison")
    _require_keys(
        comparison,
        {"paper_comparable", "residual_gaps", "schedule_difference", "boundary_difference"},
        "comparison",
    )
    if (
        comparison["paper_comparable"] is not False
        or type(comparison["residual_gaps"]) is not list
        or len(cast(list[object], comparison["residual_gaps"])) < 1
    ):
        raise ValueError("paper comparison must fail closed with explicit gaps")
    policy = _exact_dict(result["policy"], "policy")
    if policy != {
        "status": "development-only-nonpromoting",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
    }:
        raise ValueError("result policy is not permanently nonpromoting")


_MATCHED_RESULT_KEYS = {
    "schema",
    "arms",
    "matched_axes",
    "comparison",
    "resources",
    "policy",
}
_MATCHED_RESOURCE_FIELDS = (
    "environment_data_steps",
    "observations",
    "label_queries",
    "optimizer_updates",
    "model_queries",
    "model_forward_calls",
)


def run_adalin_matched_development(
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    config: AdaLinConfig = PAPER_PMNIST_CONFIG,
    seed: int,
) -> dict[str, object]:
    """Run AdaLin and its exact alpha-zero reduction on one matched PMNIST schedule."""
    if type(seed) is not int or seed not in ADALIN_MATCHED_DEVELOPMENT_SEEDS:
        raise ValueError("seed is not in the frozen AdaLin matched-development roster")
    enabled = run_adalin_development(
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        config=config,
        seed=seed,
        mechanism_enabled=True,
    )
    disabled = run_adalin_development(
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        config=config,
        seed=seed,
        mechanism_enabled=False,
    )
    enabled_metric = cast(dict[str, object], enabled["metrics"])[
        "asi_whole_stream_preupdate_online_accuracy"
    ]
    disabled_metric = cast(dict[str, object], disabled["metrics"])[
        "asi_whole_stream_preupdate_online_accuracy"
    ]
    delta = float(cast(float, enabled_metric)) - float(cast(float, disabled_metric))
    outcome = "positive" if delta > 0.0 else "negative" if delta < 0.0 else "tied"
    enabled_resources = cast(dict[str, object], enabled["resources"])
    disabled_resources = cast(dict[str, object], disabled["resources"])
    result: dict[str, object] = {
        "schema": ADALIN_MATCHED_RESULT_SCHEMA,
        "arms": [enabled, disabled],
        "matched_axes": {
            "seed": enabled["seed"],
            "config": enabled["config"],
            "dataset_sha256": cast(dict[str, object], enabled["dataset"])["sha256"],
            "schedule_sha256": cast(dict[str, object], enabled["provenance"])[
                "schedule_sha256"
            ],
            "initial_shared_parameters_sha256": cast(
                dict[str, object], enabled["provenance"]
            )["initial_shared_parameters_sha256"],
            "learner_observes_task_boundary": False,
        },
        "comparison": {
            "primary_metric": "asi_whole_stream_preupdate_online_accuracy",
            "adalin_minus_mechanism_off": delta,
            "outcome": outcome,
            "claim_threshold": None,
        },
        "resources": {
            field: cast(int, enabled_resources[field]) + cast(int, disabled_resources[field])
            for field in _MATCHED_RESOURCE_FIELDS
        }
        | {
            "wall_clock_seconds_telemetry": float(
                cast(float, enabled_resources["wall_clock_seconds_telemetry"])
            )
            + float(cast(float, disabled_resources["wall_clock_seconds_telemetry"])),
            "timing_qualified": False,
        },
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
            "automatic_reference_selection_allowed": False,
        },
    }
    validate_adalin_matched_result(result)
    return result


def validate_adalin_matched_result(value: object) -> None:
    """Strictly validate one matched AdaLin/mechanism-off development result."""
    _require_builtin_json(value)
    result = _exact_dict(value, "matched result")
    _require_keys(result, _MATCHED_RESULT_KEYS, "matched result")
    if result["schema"] != ADALIN_MATCHED_RESULT_SCHEMA:
        raise ValueError("unexpected matched AdaLin result schema")
    raw_arms = result["arms"]
    if type(raw_arms) is not list or len(raw_arms) != 2:
        raise ValueError("matched result must contain exactly two arms")
    arms = cast(list[object], raw_arms)
    for arm in arms:
        validate_adalin_result(arm)
    enabled = _exact_dict(arms[0], "enabled arm")
    disabled = _exact_dict(arms[1], "disabled arm")
    if enabled["arm"] != "adalin" or disabled["arm"] != "relu_alpha_zero_mechanism_off":
        raise ValueError("matched arms are out of order or unsupported")
    if type(enabled["seed"]) is not int or enabled["seed"] not in ADALIN_MATCHED_DEVELOPMENT_SEEDS:
        raise ValueError("seed is not in the frozen AdaLin matched-development roster")

    enabled_dataset = _exact_dict(enabled["dataset"], "enabled dataset")
    disabled_dataset = _exact_dict(disabled["dataset"], "disabled dataset")
    enabled_provenance = _exact_dict(enabled["provenance"], "enabled provenance")
    disabled_provenance = _exact_dict(disabled["provenance"], "disabled provenance")
    if (
        enabled["seed"] != disabled["seed"]
        or enabled["config"] != disabled["config"]
        or enabled_dataset != disabled_dataset
        or enabled_provenance["schedule_sha256"] != disabled_provenance["schedule_sha256"]
        or enabled_provenance["initial_shared_parameters_sha256"]
        != disabled_provenance["initial_shared_parameters_sha256"]
    ):
        raise ValueError("AdaLin arms do not share the matched execution axes")

    expected_axes = {
        "seed": enabled["seed"],
        "config": enabled["config"],
        "dataset_sha256": enabled_dataset["sha256"],
        "schedule_sha256": enabled_provenance["schedule_sha256"],
        "initial_shared_parameters_sha256": enabled_provenance[
            "initial_shared_parameters_sha256"
        ],
        "learner_observes_task_boundary": False,
    }
    axes = _exact_dict(result["matched_axes"], "matched axes")
    if axes != expected_axes:
        raise ValueError("matched axes disagree with arm receipts")

    enabled_resources = _exact_dict(enabled["resources"], "enabled resources")
    disabled_resources = _exact_dict(disabled["resources"], "disabled resources")
    for field in _MATCHED_RESOURCE_FIELDS:
        if enabled_resources[field] != disabled_resources[field]:
            raise ValueError(f"matched arm resource {field} differs")
    resources = _exact_dict(result["resources"], "matched resources")
    expected_resources: dict[str, object] = {
        field: cast(int, enabled_resources[field]) + cast(int, disabled_resources[field])
        for field in _MATCHED_RESOURCE_FIELDS
    }
    expected_resources.update(
        {
            "wall_clock_seconds_telemetry": float(
                cast(float, enabled_resources["wall_clock_seconds_telemetry"])
            )
            + float(cast(float, disabled_resources["wall_clock_seconds_telemetry"])),
            "timing_qualified": False,
        }
    )
    if resources != expected_resources:
        raise ValueError("matched resources are not the exact arm totals")

    comparison = _exact_dict(result["comparison"], "matched comparison")
    _require_keys(
        comparison,
        {"primary_metric", "adalin_minus_mechanism_off", "outcome", "claim_threshold"},
        "matched comparison",
    )
    enabled_metric = _exact_finite_number(
        _exact_dict(enabled["metrics"], "enabled metrics")[
            "asi_whole_stream_preupdate_online_accuracy"
        ],
        "enabled primary metric",
    )
    disabled_metric = _exact_finite_number(
        _exact_dict(disabled["metrics"], "disabled metrics")[
            "asi_whole_stream_preupdate_online_accuracy"
        ],
        "disabled primary metric",
    )
    delta = enabled_metric - disabled_metric
    expected_outcome = "positive" if delta > 0.0 else "negative" if delta < 0.0 else "tied"
    if comparison != {
        "primary_metric": "asi_whole_stream_preupdate_online_accuracy",
        "adalin_minus_mechanism_off": delta,
        "outcome": expected_outcome,
        "claim_threshold": None,
    }:
        raise ValueError("matched comparison is not derived from arm metrics")
    if result["policy"] != {
        "status": "development-only-nonpromoting",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
        "automatic_reference_selection_allowed": False,
    }:
        raise ValueError("matched result policy is not permanently nonpromoting")
