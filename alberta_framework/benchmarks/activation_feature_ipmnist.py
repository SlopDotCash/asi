"""Issue #1566: matched, development-only activation/feature IPMNIST lane.

The lane deliberately runs through :func:`run_screening_config`, so task and
example schedules, initialization, EMA input conditioning, online metrics and
per-step key issuance remain those of the live IPMNIST screening runner.  It
does not claim paper-protocol parity or a result; its receipts are permanently
nonpromoting and enumerate the remaining comparison gaps.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import platform
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks.ipmnist_screening import (
    EMANormState,
    ScreeningRunResult,
    ScreeningSpec,
    _make_sgd_ema_norm_learner,
    ema_normalize,
    run_screening_config,
)
from alberta_framework.benchmarks.plasticity_comparators import (
    deep_fourier_features,
    interval_dropout,
    smooth_leaky,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    _PLASTICITY_LOSS_FLOOR,
    IPMNISTConfig,
    atomic_write_new_json,
    build_schedule,
    default_openml_data_home,
    load_mnist_train,
)

RESULT_SCHEMA: Final[str] = "asi.activation_feature_ipmnist.result.v1"
CAMPAIGN_RESULT_SCHEMA: Final[str] = "asi.activation_feature_ipmnist.result.v2"
COMPARISON_ID: Final[str] = "issue-1566-low-cost-activation-feature-controls"
DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
DEVELOPMENT_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"supported", "rejected", "inconclusive"}
)
_MAX_STEPS: Final[int] = 2_000_000

ACTIVATION_FEATURE_SOURCES: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "smooth_leaky": MappingProxyType(
            {
                "paper_revision": "arXiv:2509.22562v4",
                "paper_revision_date": "2026-04-30",
                "official_code_revision": (
                    "anonymous.4open.science snapshot activations_plasticity-E431"
                ),
            }
        ),
        "aid": MappingProxyType(
            {
                "paper_revision": "arXiv:2502.01342v2",
                "paper_revision_date": "2025-06-15",
                "official_code_revision": (
                    "none disclosed; paper Algorithm 2 is the implementation source"
                ),
            }
        ),
        "deep_fourier": MappingProxyType(
            {
                "paper_revision": "arXiv:2410.20634v1",
                "paper_revision_date": "2024-10-27",
                "official_code_revision": (
                    "none disclosed; paper equation is the implementation source"
                ),
            }
        ),
    }
)


@chex.dataclass(frozen=True)
class ActivationNormState:
    """Only the live control's EMA input-normalizer state is persistent."""

    norm: EMANormState


def _init_state(params: dict[str, Array]) -> ActivationNormState:
    width = params["w1"].shape[0]
    return ActivationNormState(  # type: ignore[call-arg]
        norm=EMANormState(  # type: ignore[call-arg]
            mean=jnp.zeros(width, jnp.float32),
            var=jnp.ones(width, jnp.float32),
            count=jnp.asarray(0.0, jnp.float32),
        )
    )


def _loss(logits: Array, label: Array) -> Array:
    return -jax.nn.log_softmax(logits)[label]


def _metrics(
    params: dict[str, Array],
    x: Array,
    y: Array,
    loss: Array,
    logits: Array,
    forward: Any,
    key: Array,
) -> tuple[Array, Array, Array]:
    accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
    # ``loss`` came from the training-mode forward that produced the gradient.
    # The runner's ``_step_metrics`` applies one forward function to both sides
    # of the plasticity ratio, so the post-update loss is scored in the same
    # mode from the same per-step key: for ``aid`` and ``ordinary_dropout`` that
    # re-derives the identical activation mask, leaving the parameter update as
    # the only thing that moved.  Every mode-independent arm is unaffected.
    after = _loss(forward(params, x, key, True), y)
    plasticity = jnp.clip(1.0 - after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0)
    return accuracy, loss, plasticity


def _smooth_forward(
    params: dict[str, Array], x: Array, key: Array, training: bool, *, fixed: bool
) -> Array:
    del key, training
    z1 = x @ params["w1"] + params["b1"]
    if fixed:
        a1 = jnp.where(z1 >= 0, z1, 0.1 * z1)
    else:
        a1 = smooth_leaky(z1, alpha=0.1, power=3.0, curvature=5.0)
    z2 = a1 @ params["w2"] + params["b2"]
    if fixed:
        a2 = jnp.where(z2 >= 0, z2, 0.1 * z2)
    else:
        a2 = smooth_leaky(z2, alpha=0.1, power=3.0, curvature=5.0)
    return a2 @ params["w3"] + params["b3"]


def _aid_forward(
    params: dict[str, Array],
    x: Array,
    key: Array,
    training: bool,
    *,
    expected: bool,
    ordinary: bool,
) -> Array:
    key1, key2 = jr.split(key)
    z1 = x @ params["w1"] + params["b1"]
    z2_key = key2
    if ordinary:
        if training:
            mask = jr.bernoulli(key1, 0.9, z1.shape)
            a1 = jnp.where(mask, jax.nn.relu(z1), 0.0)
        else:
            a1 = 0.9 * jax.nn.relu(z1)
    elif expected:
        a1 = jnp.where(z1 >= 0, 0.9 * z1, 0.1 * z1)
    else:
        a1 = interval_dropout(z1, key1, relu_probability=0.9, training=training)
    z2 = a1 @ params["w2"] + params["b2"]
    if ordinary:
        if training:
            mask = jr.bernoulli(z2_key, 0.9, z2.shape)
            a2 = jnp.where(mask, jax.nn.relu(z2), 0.0)
        else:
            a2 = 0.9 * jax.nn.relu(z2)
    elif expected:
        a2 = jnp.where(z2 >= 0, 0.9 * z2, 0.1 * z2)
    else:
        a2 = interval_dropout(z2, z2_key, relu_probability=0.9, training=training)
    return a2 @ params["w3"] + params["b3"]


def _fourier_forward(
    params: dict[str, Array],
    x: Array,
    key: Array,
    training: bool,
    *,
    first_only: bool,
    sine_only: bool,
) -> Array:
    del key, training
    half1 = params["b1"].shape[0] // 2
    z1 = x @ params["w1"][:, :half1] + params["b1"][:half1]
    if sine_only:
        a1 = jnp.concatenate((jnp.sin(z1), jnp.zeros_like(z1)))
    else:
        a1 = deep_fourier_features(z1)
    if first_only:
        z2 = a1 @ params["w2"] + params["b2"]
        a2 = jax.nn.relu(z2)
    else:
        half2 = params["b2"].shape[0] // 2
        z2 = a1 @ params["w2"][:, :half2] + params["b2"][:half2]
        if sine_only:
            a2 = jnp.concatenate((jnp.sin(z2), jnp.zeros_like(z2)))
        else:
            a2 = deep_fourier_features(z2)
    return a2 @ params["w3"] + params["b3"]


def _factory(forward: Any) -> tuple[Any, Any]:
    step_size = 0.01
    decay_factor = 1.0 - step_size * 0.01

    def step(
        params: dict[str, Array], state: ActivationNormState, x: Array, y: Array, key: Array
    ) -> tuple[dict[str, Array], ActivationNormState, tuple[Array, Array, Array]]:
        x_norm, norm = ema_normalize(state.norm, x, 0.99, 1e-8)

        def objective(current: dict[str, Array]) -> tuple[Array, Array]:
            logits = forward(current, x_norm, key, True)
            return _loss(logits, y), logits

        (loss, logits), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updated = {name: params[name] * decay_factor - step_size * grads[name] for name in params}
        return (
            updated,
            ActivationNormState(norm=norm),  # type: ignore[call-arg]
            _metrics(updated, x_norm, y, loss, logits, forward, key),
        )

    return _init_state, step


def _smooth_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    if hp["enabled"] == 0.0:
        return _make_sgd_ema_norm_learner(hp)
    return _factory(lambda p, x, k, t: _smooth_forward(p, x, k, t, fixed=False))


def _fixed_leak_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    del hp
    return _factory(lambda p, x, k, t: _smooth_forward(p, x, k, t, fixed=True))


def _aid_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    if hp["enabled"] == 0.0:
        return _make_sgd_ema_norm_learner(hp)
    return _factory(lambda p, x, k, t: _aid_forward(p, x, k, t, expected=False, ordinary=False))


def _aid_expected_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    del hp
    return _factory(lambda p, x, k, t: _aid_forward(p, x, k, t, expected=True, ordinary=False))


def _dropout_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    del hp
    return _factory(lambda p, x, k, t: _aid_forward(p, x, k, t, expected=False, ordinary=True))


def _fourier_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    if hp["enabled"] == 0.0:
        return _make_sgd_ema_norm_learner(hp)
    return _factory(
        lambda p, x, k, t: _fourier_forward(p, x, k, t, first_only=False, sine_only=False)
    )


def _fourier_first_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    del hp
    return _factory(
        lambda p, x, k, t: _fourier_forward(p, x, k, t, first_only=True, sine_only=False)
    )


def _fourier_sine_factory(hp: Mapping[str, float]) -> tuple[Any, Any]:
    del hp
    return _factory(
        lambda p, x, k, t: _fourier_forward(p, x, k, t, first_only=False, sine_only=True)
    )


_BASE = {"step_size": 0.01, "weight_decay": 0.01, "norm_decay": 0.99, "norm_epsilon": 1e-8}


def _spec(name: str, mechanism: str, factory: Any, *, enabled: float = 1.0) -> ScreeningSpec:
    return ScreeningSpec(
        name=name,
        base_learner="upgd_w",
        mechanism=mechanism,
        hyperparameters={**_BASE, "enabled": enabled},
        factory=factory,
        description="Issue #1566 permanently nonpromoting matched comparator.",
    )


_SPECS = (
    _spec("smooth_leaky", "smooth_leaky", _smooth_factory),
    _spec("smooth_leaky_off", "smooth_leaky_off", _smooth_factory, enabled=0.0),
    _spec("smooth_leaky_fixed_leak", "smooth_leaky_ablation", _fixed_leak_factory),
    _spec("aid", "aid", _aid_factory),
    _spec("aid_off", "aid_off", _aid_factory, enabled=0.0),
    _spec("aid_expected", "aid_ablation", _aid_expected_factory),
    _spec("ordinary_dropout", "aid_ablation", _dropout_factory),
    _spec("deep_fourier", "deep_fourier", _fourier_factory),
    _spec("deep_fourier_off", "deep_fourier_off", _fourier_factory, enabled=0.0),
    _spec("deep_fourier_first_layer", "deep_fourier_ablation", _fourier_first_factory),
    _spec("deep_fourier_sine_only", "deep_fourier_ablation", _fourier_sine_factory),
)
ACTIVATION_FEATURE_SPECS: Final[Mapping[str, ScreeningSpec]] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)

_MAX_SCHEDULE_BYTES: Final[int] = 256 * 1024 * 1024


def activation_feature_spec(name: object) -> ScreeningSpec:
    if type(name) is not str or name not in ACTIVATION_FEATURE_SPECS:
        raise ValueError("unknown activation/feature arm")
    return ACTIVATION_FEATURE_SPECS[name]


def _preflight_activation_feature_resources(
    config: IPMNISTConfig, *, n_train: int | None = None
) -> int:
    """Reject retained schedule payloads outside the lane's bounded envelope."""
    sampled_width = config.task_length if n_train is None else n_train
    if type(sampled_width) is not int or sampled_width < 1:
        raise ValueError("activation/feature training-row count must be positive")
    # build_schedule vmaps a full n_train permutation before slicing each row.
    schedule_scalars = config.n_tasks * (config.input_dim + sampled_width)
    schedule_bytes = schedule_scalars * np.dtype(np.int32).itemsize
    if schedule_bytes > _MAX_SCHEDULE_BYTES:
        raise ValueError(
            "activation/feature schedule exceeds its 268435456-byte bound"
        )
    return schedule_bytes


def _array_bundle_sha256(data_x: np.ndarray, data_y: np.ndarray) -> str:
    digest = hashlib.sha256(b"asi-activation-feature-dataset-v1\0")
    for value in (data_x, data_y):
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _schedule_sha256(permutations: Array, example_indices: Array) -> str:
    digest = hashlib.sha256(b"asi-activation-feature-schedule-v1\0")
    for value in (permutations, example_indices):
        host = np.asarray(value, dtype=np.int32)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _runtime_identity() -> tuple[str, str, str, str]:
    return (platform.python_version(), jax.__version__, np.__version__, jax.default_backend())


def _current_source_identity() -> tuple[str, str, str]:
    names = ("activation_feature_ipmnist.py", "ipmnist_screening.py", "upgd_ipmnist.py")
    digests = tuple(
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in names
    )
    return digests[0], digests[1], digests[2]


def _host_array(value: object, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(actual_type, jax.Array):
        raise TypeError(f"{name} must be an exact NumPy or JAX array")
    return np.asarray(value, dtype=dtype)


@dataclasses.dataclass(frozen=True, slots=True)
class ActivationFeatureRunResult:
    screening: ScreeningRunResult
    dataset_sha256: str
    schedule_sha256: str
    source_identity: tuple[str, str, str]
    runtime_identity: tuple[str, str, str, str]
    n_train: int
    retained_schedule_numeric_bytes: int

    @property
    def config_name(self) -> str:
        return self.screening.config_name

    @property
    def hyperparameters(self) -> dict[str, float]:
        return self.screening.hyperparameters

    @property
    def seed(self) -> int:
        return self.screening.seed

    @property
    def config(self) -> IPMNISTConfig:
        return self.screening.config

    @property
    def per_task_accuracy(self) -> np.ndarray:
        return self.screening.per_task_accuracy

    @property
    def per_task_loss(self) -> np.ndarray:
        return self.screening.per_task_loss

    @property
    def per_task_plasticity(self) -> np.ndarray:
        return self.screening.per_task_plasticity

    @property
    def wall_clock_seconds(self) -> float:
        return self.screening.wall_clock_seconds

    @property
    def noise_mode(self) -> str:
        return self.screening.noise_mode


def run_activation_feature_arm(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    *,
    arm: str,
    seed: int,
    config: IPMNISTConfig,
) -> ActivationFeatureRunResult:
    """Execute one arm using the current screening runner and its schedule."""
    _preflight_activation_feature_resources(config)
    observations = config.n_tasks * config.task_length
    if observations > _MAX_STEPS:
        raise ValueError("activation/feature lane exceeds its 2000000-step bound")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an exact uint32 value")
    if arm.startswith("deep_fourier") and (config.hidden1 % 2 or config.hidden2 % 2):
        raise ValueError("Deep Fourier arms require even hidden widths")
    host_x = _host_array(data_x, name="data_x", dtype=np.dtype(np.float32))
    host_y = _host_array(data_y, name="data_y", dtype=np.dtype(np.int32))
    if host_x.ndim != 2 or host_y.ndim != 1 or host_x.shape[0] != host_y.shape[0]:
        raise ValueError("activation/feature data must be matched rank-2/rank-1 arrays")
    retained_schedule_bytes = _preflight_activation_feature_resources(
        config, n_train=int(host_x.shape[0])
    )
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    _, key_schedule, _ = jr.split(root, 3)
    schedule = build_schedule(key_schedule, config, int(host_x.shape[0]))
    screening = run_screening_config(
        host_x, host_y, activation_feature_spec(arm), seed, config
    )
    return ActivationFeatureRunResult(
        screening=screening,
        dataset_sha256=_array_bundle_sha256(host_x, host_y),
        schedule_sha256=_schedule_sha256(
            schedule.permutations, schedule.example_indices
        ),
        source_identity=_current_source_identity(),
        runtime_identity=_runtime_identity(),
        n_train=int(host_x.shape[0]),
        retained_schedule_numeric_bytes=retained_schedule_bytes,
    )


_TOP_FIELDS = frozenset(
    {
        "schema",
        "comparison_id",
        "arm",
        "source",
        "execution_identity",
        "seed",
        "development_seed_protocol",
        "config",
        "allowed_boundary_information",
        "allowed_task_information",
        "hyperparameters",
        "metrics",
        "resources",
        "comparability_gaps",
        "paper_metric_reported",
        "outcome",
        "outcome_retained",
        "development_only",
        "scientific_promotion_allowed",
    }
)
_RESOURCE_FIELDS = frozenset(
    {
        "data_steps",
        "environment_steps",
        "updates",
        "gradient_evaluations",
        "optimizer_scalar_updates",
        "model_queries",
        "random_bernoulli_variates",
        "activation_scalar_evaluations",
        "allocated_parameter_scalars",
        "active_parameter_scalars",
        "forward_affine_weight_applications",
        "sigmoid_evaluations",
        "trigonometric_evaluations",
        "persistent_numeric_bytes",
        "retained_schedule_numeric_bytes",
        "timing_seconds",
        "timing_is_telemetry_only",
    }
)


def _parameter_counts(config: IPMNISTConfig, arm: str) -> tuple[int, int]:
    allocated = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    if not arm.startswith("deep_fourier") or arm == "deep_fourier_off":
        return allocated, allocated
    half1, half2 = config.hidden1 // 2, config.hidden2 // 2
    active = config.input_dim * half1 + half1 + config.hidden2 * config.n_classes + config.n_classes
    active += config.hidden1 * (config.hidden2 if arm == "deep_fourier_first_layer" else half2)
    active += config.hidden2 if arm == "deep_fourier_first_layer" else half2
    return allocated, active


def _validated_seed_protocol(value: object) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or not 1 <= len(value) <= 64
        or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("development seed protocol must be a unique exact uint32 tuple")
    return cast(tuple[int, ...], value)


def _activation_feature_result_payload(
    result: ActivationFeatureRunResult,
    *,
    outcome: str,
    schema: str,
    seed_protocol: tuple[int, ...],
) -> dict[str, object]:
    """Build one versioned strict nonpromotion/resource receipt."""
    checked_seeds = _validated_seed_protocol(seed_protocol)
    if type(result) is not ActivationFeatureRunResult:
        raise ValueError("result must be an exact ActivationFeatureRunResult")
    if type(result.screening) is not ScreeningRunResult:
        raise ValueError("screening result must be exact")
    if type(result.config_name) is not str:
        raise ValueError("result config_name must be an exact string")
    spec = activation_feature_spec(result.config_name)
    if (
        type(result.hyperparameters) is not dict
        or any(
            type(key) is not str or type(value) is not float
            for key, value in result.hyperparameters.items()
        )
        or result.hyperparameters != spec.hyperparameters
        or type(result.noise_mode) is not str
        or result.noise_mode != "step"
    ):
        raise ValueError("result does not match the registered exact-step arm")
    if type(outcome) is not str or outcome not in DEVELOPMENT_OUTCOMES:
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if type(result.config) is not IPMNISTConfig:
        raise ValueError("result config must be an exact IPMNISTConfig")
    config_values = {
        name: getattr(result.config, name)
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
    }
    try:
        config = IPMNISTConfig(**config_values)
    except (TypeError, ValueError) as error:
        raise ValueError("result config is invalid") from error
    if type(result.seed) is not int or result.seed not in checked_seeds:
        raise ValueError("result seed must belong to the frozen development seed set")
    if (
        type(result.wall_clock_seconds) is not float
        or not math.isfinite(result.wall_clock_seconds)
        or result.wall_clock_seconds < 0.0
    ):
        raise ValueError("result timing must be an exact nonnegative finite float")
    for name in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
        values = getattr(result, name)
        if (
            type(values) is not np.ndarray
            or values.dtype != np.dtype(np.float64)
            or values.shape != (config.n_tasks,)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"result {name} must be one finite float64 value per task")
    if np.any((result.per_task_accuracy < 0.0) | (result.per_task_accuracy > 1.0)):
        raise ValueError("result per_task_accuracy must lie in [0, 1]")
    if np.any(result.per_task_loss < 0.0):
        raise ValueError("result per_task_loss must be nonnegative")
    if np.any((result.per_task_plasticity < 0.0) | (result.per_task_plasticity > 1.0)):
        raise ValueError("result per_task_plasticity must lie in [0, 1]")
    steps = config.n_tasks * config.task_length
    allocated, active = _parameter_counts(config, result.config_name)
    stochastic = result.config_name in {"aid", "ordinary_dropout"}
    activation_width = config.hidden1 + config.hidden2
    if result.config_name == "deep_fourier_first_layer":
        active_biases = config.hidden1 // 2 + config.hidden2 + config.n_classes
    elif result.config_name in {"deep_fourier", "deep_fourier_sine_only"}:
        active_biases = config.hidden1 // 2 + config.hidden2 // 2 + config.n_classes
    else:
        active_biases = config.hidden1 + config.hidden2 + config.n_classes
    sigmoid_evaluations = (
        2 * steps * activation_width if result.config_name == "smooth_leaky" else 0
    )
    if result.config_name == "deep_fourier_first_layer":
        trig_evaluations = 2 * steps * config.hidden1
    elif result.config_name == "deep_fourier":
        trig_evaluations = 2 * steps * activation_width
    elif result.config_name == "deep_fourier_sine_only":
        trig_evaluations = steps * activation_width
    else:
        trig_evaluations = 0
    family = (
        "smooth_leaky"
        if result.config_name.startswith("smooth_leaky")
        else "aid"
        if result.config_name.startswith("aid") or result.config_name == "ordinary_dropout"
        else "deep_fourier"
    )
    gaps = ["optimizer_and_schedule", "paper_metric_and_horizon"]
    if family == "deep_fourier" and result.config_name != "deep_fourier_off":
        gaps.append("parameterization")
    payload: dict[str, object] = {
        "schema": schema,
        "comparison_id": COMPARISON_ID,
        "arm": result.config_name,
        "source": dict(ACTIVATION_FEATURE_SOURCES[family]),
        "execution_identity": {
            "dataset_sha256": result.dataset_sha256,
            "schedule_sha256": result.schedule_sha256,
            "source_sha256": list(result.source_identity),
            "runtime": list(result.runtime_identity),
            "n_train": result.n_train,
        },
        "seed": result.seed,
        "development_seed_protocol": list(checked_seeds),
        "config": {
            "n_tasks": config.n_tasks,
            "task_length": config.task_length,
            "input_dim": config.input_dim,
            "hidden1": config.hidden1,
            "hidden2": config.hidden2,
            "n_classes": config.n_classes,
        },
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "hyperparameters": dict(spec.hyperparameters),
        "metrics": {
            "asi_whole_stream_mean_accuracy": float(np.mean(result.per_task_accuracy)),
            "asi_whole_stream_mean_loss": float(np.mean(result.per_task_loss)),
            "asi_whole_stream_mean_plasticity": float(np.mean(result.per_task_plasticity)),
        },
        "resources": {
            "data_steps": steps,
            "environment_steps": 0,
            "updates": steps,
            "gradient_evaluations": steps,
            "optimizer_scalar_updates": steps * allocated,
            "model_queries": 2 * steps,
            # Both metric forwards run in training mode, so a stochastic arm
            # draws one mask per hidden unit in each of them.
            "random_bernoulli_variates": 2 * steps * activation_width if stochastic else 0,
            "activation_scalar_evaluations": 2 * steps * activation_width,
            "allocated_parameter_scalars": allocated,
            "active_parameter_scalars": active,
            "forward_affine_weight_applications": 2 * steps * (active - active_biases),
            "sigmoid_evaluations": sigmoid_evaluations,
            "trigonometric_evaluations": trig_evaluations,
            "persistent_numeric_bytes": 4 * (allocated + 2 * config.input_dim + 1),
            "retained_schedule_numeric_bytes": result.retained_schedule_numeric_bytes,
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "comparability_gaps": gaps,
        "paper_metric_reported": False,
        "outcome": outcome,
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return _validate_activation_feature_result(
        payload, schema=schema, seed_protocol=checked_seeds
    )


def activation_feature_result_payload(
    result: ActivationFeatureRunResult, *, outcome: str
) -> dict[str, object]:
    """Build a result-v1 receipt under its fixed seed protocol."""
    return _activation_feature_result_payload(
        result,
        outcome=outcome,
        schema=RESULT_SCHEMA,
        seed_protocol=DEVELOPMENT_SEEDS,
    )


def activation_feature_campaign_result_payload(
    result: ActivationFeatureRunResult,
    *,
    outcome: str,
    development_seeds: tuple[int, ...],
) -> dict[str, object]:
    """Build a result-v2 receipt for an explicitly frozen campaign seed set."""
    return _activation_feature_result_payload(
        result,
        outcome=outcome,
        schema=CAMPAIGN_RESULT_SCHEMA,
        seed_protocol=development_seeds,
    )


def _exact_dict(value: object, fields: frozenset[str], context: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"{context} must contain exact fields {sorted(fields)}")
    return cast(dict[str, object], value)


def _validate_activation_feature_result(
    payload: object, *, schema: str, seed_protocol: tuple[int, ...]
) -> dict[str, object]:
    """Fail-closed bounded validator for one exact receipt version."""
    checked_seeds = _validated_seed_protocol(seed_protocol)
    root = _exact_dict(payload, _TOP_FIELDS, "result")
    if root["schema"] != schema or root["comparison_id"] != COMPARISON_ID:
        raise ValueError("unsupported result identity")
    arm = root["arm"]
    spec = activation_feature_spec(arm)
    if root["development_only"] is not True or root["scientific_promotion_allowed"] is not False:
        raise ValueError("result must remain permanently nonpromoting")
    if root["outcome_retained"] is not True:
        raise ValueError("all outcomes, including negative outcomes, must be retained")
    if root["paper_metric_reported"] is not False:
        raise ValueError("paper_metric_reported must remain false")
    if type(root["seed"]) is not int or root["seed"] not in checked_seeds:
        raise ValueError("seed must belong to the frozen development seed set")
    if root["development_seed_protocol"] != list(checked_seeds):
        raise ValueError("development_seed_protocol must match the frozen seed set")
    if type(root["outcome"]) is not str or root["outcome"] not in DEVELOPMENT_OUTCOMES:
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if root["allowed_boundary_information"] != []:
        raise ValueError("learner boundary information must remain empty")
    if root["allowed_task_information"] != ["current_example_label"]:
        raise ValueError("learner task information must remain the current example label")
    config = _exact_dict(
        root["config"],
        frozenset({"n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"}),
        "config",
    )
    for name, value in config.items():
        if type(value) is not int or not 1 <= value <= _MAX_STEPS:
            raise ValueError(f"config {name} must be a bounded exact integer")
    if cast(int, config["n_tasks"]) * cast(int, config["task_length"]) > _MAX_STEPS:
        raise ValueError("config step product exceeds bound")
    try:
        validated_config = IPMNISTConfig(
            n_tasks=cast(int, config["n_tasks"]),
            task_length=cast(int, config["task_length"]),
            input_dim=cast(int, config["input_dim"]),
            hidden1=cast(int, config["hidden1"]),
            hidden2=cast(int, config["hidden2"]),
            n_classes=cast(int, config["n_classes"]),
        )
    except ValueError as error:
        raise ValueError("config violates the executable IPMNIST bounds") from error
    _preflight_activation_feature_resources(validated_config)
    identity = _exact_dict(
        root["execution_identity"],
        frozenset(
            {"dataset_sha256", "schedule_sha256", "source_sha256", "runtime", "n_train"}
        ),
        "execution_identity",
    )
    for name in ("dataset_sha256", "schedule_sha256"):
        value = identity[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if identity["source_sha256"] != list(_current_source_identity()):
        raise ValueError("current runner/source identity drift")
    if identity["runtime"] != list(_runtime_identity()):
        raise ValueError("current runtime identity drift")
    n_train_value = identity["n_train"]
    if (
        type(n_train_value) is not int
        or not validated_config.task_length <= n_train_value <= _MAX_STEPS
    ):
        raise ValueError("n_train must be an exact bounded dataset row count")
    retained_schedule_bytes = _preflight_activation_feature_resources(
        validated_config, n_train=n_train_value
    )
    hp = _exact_dict(root["hyperparameters"], frozenset(spec.hyperparameters), "hyperparameters")
    for name, expected in spec.hyperparameters.items():
        value = hp[name]
        if type(value) is not float or not math.isfinite(value) or value.hex() != expected.hex():
            raise ValueError(f"hyperparameter {name} does not match registered arm")
    metrics = _exact_dict(
        root["metrics"],
        frozenset(
            {
                "asi_whole_stream_mean_accuracy",
                "asi_whole_stream_mean_loss",
                "asi_whole_stream_mean_plasticity",
            }
        ),
        "metrics",
    )
    for name, value in metrics.items():
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"metric {name} must be an exact finite float")
    for name in ("asi_whole_stream_mean_accuracy", "asi_whole_stream_mean_plasticity"):
        if not 0.0 <= cast(float, metrics[name]) <= 1.0:
            raise ValueError(f"metric {name} must lie in [0, 1]")
    if cast(float, metrics["asi_whole_stream_mean_loss"]) < 0.0:
        raise ValueError("mean loss must be nonnegative")
    resources = _exact_dict(root["resources"], _RESOURCE_FIELDS, "resources")
    for name, value in resources.items():
        if name == "timing_is_telemetry_only":
            if value is not True:
                raise ValueError("timing_is_telemetry_only must be true")
        elif name == "timing_seconds":
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError("timing_seconds must be an exact nonnegative finite float")
        elif type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError(f"{name} must be an exact bounded nonnegative integer")
    n_tasks = cast(int, config["n_tasks"])
    task_length = cast(int, config["task_length"])
    input_dim = cast(int, config["input_dim"])
    hidden1 = cast(int, config["hidden1"])
    hidden2 = cast(int, config["hidden2"])
    n_classes = cast(int, config["n_classes"])
    steps = n_tasks * task_length
    allocated = (
        input_dim * hidden1
        + hidden1
        + hidden1 * hidden2
        + hidden2
        + hidden2 * n_classes
        + n_classes
    )
    active = allocated
    active_biases = hidden1 + hidden2 + n_classes
    if cast(str, arm).startswith("deep_fourier") and arm != "deep_fourier_off":
        if hidden1 % 2 or hidden2 % 2:
            raise ValueError("Deep Fourier result hidden widths must be even")
        half1, half2 = hidden1 // 2, hidden2 // 2
        active = input_dim * half1 + half1 + hidden2 * n_classes + n_classes
        active += hidden1 * (hidden2 if arm == "deep_fourier_first_layer" else half2)
        active += hidden2 if arm == "deep_fourier_first_layer" else half2
        active_biases = (
            half1 + hidden2 + n_classes
            if arm == "deep_fourier_first_layer"
            else half1 + half2 + n_classes
        )
    stochastic = arm in {"aid", "ordinary_dropout"}
    sigmoid_evaluations = 2 * steps * (hidden1 + hidden2) if arm == "smooth_leaky" else 0
    if arm == "deep_fourier_first_layer":
        trig_evaluations = 2 * steps * hidden1
    elif arm == "deep_fourier":
        trig_evaluations = 2 * steps * (hidden1 + hidden2)
    elif arm == "deep_fourier_sine_only":
        trig_evaluations = steps * (hidden1 + hidden2)
    else:
        trig_evaluations = 0
    expected_resources = {
        "data_steps": steps,
        "environment_steps": 0,
        "updates": steps,
        "gradient_evaluations": steps,
        "optimizer_scalar_updates": steps * allocated,
        "model_queries": 2 * steps,
        "random_bernoulli_variates": 2 * steps * (hidden1 + hidden2) if stochastic else 0,
        "activation_scalar_evaluations": 2 * steps * (hidden1 + hidden2),
        "allocated_parameter_scalars": allocated,
        "active_parameter_scalars": active,
        "forward_affine_weight_applications": 2 * steps * (active - active_biases),
        "sigmoid_evaluations": sigmoid_evaluations,
        "trigonometric_evaluations": trig_evaluations,
        "persistent_numeric_bytes": 4 * (allocated + 2 * input_dim + 1),
        "retained_schedule_numeric_bytes": retained_schedule_bytes,
    }
    for name, expected in expected_resources.items():
        if resources[name] != expected:
            raise ValueError(f"{name} disagrees with the exact resource schedule")
    gaps = root["comparability_gaps"]
    if (
        type(gaps) is not list
        or not gaps
        or any(type(item) is not str or not item for item in gaps)
    ):
        raise ValueError("comparability_gaps must be a non-empty exact string list")
    expected_gaps = ["optimizer_and_schedule", "paper_metric_and_horizon"]
    if cast(str, arm).startswith("deep_fourier") and arm != "deep_fourier_off":
        expected_gaps.append("parameterization")
    if gaps != expected_gaps:
        raise ValueError("comparability_gaps must match the registered arm")
    source = root["source"]
    family = (
        "smooth_leaky"
        if cast(str, arm).startswith("smooth_leaky")
        else "aid"
        if cast(str, arm).startswith("aid") or arm == "ordinary_dropout"
        else "deep_fourier"
    )
    if type(source) is not dict or source != dict(ACTIVATION_FEATURE_SOURCES[family]):
        raise ValueError("source must exactly match the audited revision")
    return root


def validate_activation_feature_result(payload: object) -> dict[str, object]:
    """Validate the result-v1 contract and fixed seed set."""
    return _validate_activation_feature_result(
        payload, schema=RESULT_SCHEMA, seed_protocol=DEVELOPMENT_SEEDS
    )


def validate_activation_feature_campaign_result(
    payload: object, *, development_seeds: tuple[int, ...]
) -> dict[str, object]:
    """Validate a result-v2 receipt against its externally frozen seed set."""
    return _validate_activation_feature_result(
        payload,
        schema=CAMPAIGN_RESULT_SCHEMA,
        seed_protocol=development_seeds,
    )


def _validate_matched_activation_feature_results(
    payloads: object, *, schema: str, seed_protocol: tuple[int, ...]
) -> list[dict[str, object]]:
    """Validate one versioned complete, same-seed matched comparison."""
    checked_seeds = _validated_seed_protocol(seed_protocol)
    if type(payloads) is not list or len(payloads) != len(ACTIVATION_FEATURE_SPECS):
        raise ValueError("matched comparison must contain every registered arm exactly once")
    validated = [
        _validate_activation_feature_result(
            payload, schema=schema, seed_protocol=checked_seeds
        )
        for payload in payloads
    ]
    by_arm = {cast(str, payload["arm"]): payload for payload in validated}
    if len(by_arm) != len(validated) or set(by_arm) != set(ACTIVATION_FEATURE_SPECS):
        raise ValueError("matched comparison must contain every registered arm exactly once")
    first = validated[0]
    for payload in validated[1:]:
        for field in (
            "seed",
            "development_seed_protocol",
            "config",
            "execution_identity",
            "allowed_boundary_information",
            "allowed_task_information",
        ):
            if payload[field] != first[field]:
                raise ValueError(f"matched comparison differs on {field}")
        resources = cast(dict[str, object], payload["resources"])
        first_resources = cast(dict[str, object], first["resources"])
        for field in (
            "data_steps",
            "environment_steps",
            "updates",
            "gradient_evaluations",
            "model_queries",
            "allocated_parameter_scalars",
            "persistent_numeric_bytes",
        ):
            if resources[field] != first_resources[field]:
                raise ValueError(f"matched comparison differs on resource axis {field}")
    return validated


def validate_matched_activation_feature_results(payloads: object) -> list[dict[str, object]]:
    """Validate one complete result-v1 comparison."""
    return _validate_matched_activation_feature_results(
        payloads, schema=RESULT_SCHEMA, seed_protocol=DEVELOPMENT_SEEDS
    )


def validate_matched_activation_feature_campaign_results(
    payloads: object, *, development_seeds: tuple[int, ...]
) -> list[dict[str, object]]:
    """Validate one complete same-seed 11-arm result-v2 comparison."""
    return _validate_matched_activation_feature_results(
        payloads,
        schema=CAMPAIGN_RESULT_SCHEMA,
        seed_protocol=development_seeds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ACTIVATION_FEATURE_SPECS))
    parser.add_argument("--seed", required=True, type=int, choices=DEVELOPMENT_SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--task-length", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one fresh development shard; benchmark work never runs in pytest."""
    args = _parser().parse_args(argv)
    config = IPMNISTConfig(n_tasks=args.tasks, task_length=args.task_length)
    _preflight_activation_feature_resources(config)
    x, y = load_mnist_train(args.data_home)
    result = run_activation_feature_arm(x, y, arm=args.arm, seed=args.seed, config=config)
    payload = activation_feature_result_payload(result, outcome="inconclusive")
    atomic_write_new_json(args.output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via installed entry point
    raise SystemExit(main())
