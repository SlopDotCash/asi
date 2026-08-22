"""Frozen, hard-disabled, permanently nonpromoting issue #1567 campaign.

The transaction compares the four registered noise-curvature scheduler arms
with the live RLS development control in one matched 5-arm by 5-seed screen.
It binds dataset, source, runtime, schedule, initialization, counters, and
frozen paired decisions. Execution requires a separate reviewed source change;
this module creates no performance or scientific evidence by itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    _array_bundle_sha256,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    PAPER_REVISION,
    noise_curvature_persistent_bytes,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    BASE_HYPERPARAMETERS,
    LIVE_CONTROL,
    OFFICIAL_CODE_STATUS,
    PROTOCOL_DIFFERENCES,
    registered_arms,
)

PLAN_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-plan.v1"
SHARD_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-shard.v1"
AGGREGATE_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.campaign-aggregate.v1"
CONTROL_RESULT_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.live-control-result.v1"
SCHEDULER_RESULT_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.scheduler-result.v1"
PLAN_ID: Final[str] = "issue-1567.noise-curvature.cheap-screen.v1"
CONFIG: Final[IPMNISTConfig] = IPMNISTConfig(n_tasks=2, task_length=500)
ARM_ROSTER: Final[tuple[str, ...]] = (LIVE_CONTROL, *registered_arms())

# Roots 0--4 are public/test-visible. These replacements were selected before
# authorization and must never appear in tests or preauthorization executions.
CONSUMED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SEEDS: tuple[int, ...] = (
    3_186_771_201,
    3_186_771_202,
    3_186_771_203,
    3_186_771_204,
    3_186_771_205,
)
_EXECUTION_AUTHORIZED: Final[bool] = False
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "alberta_framework/benchmarks/ipmnist_screening.py",
    "alberta_framework/benchmarks/noise_curvature_ipmnist.py",
    "alberta_framework/benchmarks/plasticity_comparators.py",
    "alberta_framework/benchmarks/upgd_ipmnist.py",
    "alberta_framework/evaluation/noise_curvature_campaign.py",
    "alberta_framework/evaluation/noise_curvature_ipmnist_nonpromoting.py",
    "pyproject.toml",
    "uv.lock",
)
_CANONICAL_X_SHAPE: Final[tuple[int, int]] = (60_000, 784)
_CANONICAL_Y_SHAPE: Final[tuple[int]] = (60_000,)
_CANONICAL_X_SHA256: Final[str] = "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
_CANONICAL_Y_SHA256: Final[str] = "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
_T95_DF4: Final[float] = 2.7764451051977987
_MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_PLAN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "plan_id",
        "matrix",
        "config",
        "protocol",
        "execution_gate",
        "identity",
        "statistics",
        "resources",
        "policy",
        "output_namespace",
        "plan_sha256",
    }
)
_SHARD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "plan_id",
        "plan_sha256",
        "arm",
        "seed",
        "authorization",
        "execution_identity",
        "result",
        "shard_sha256",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"campaign source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, str]:
    return {name: _sha256_file(_REPO_ROOT / name) for name in _SOURCE_PATHS}


def _runtime_identity() -> dict[str, object]:
    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
        }
        for device in jax.devices()
    ]
    if not devices:
        raise RuntimeError("campaign runtime found no JAX devices")
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scikit-learn")
        },
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": devices,
            "config": {
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_disable_jit": bool(jax.config.jax_disable_jit),
                "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
                "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
                "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            },
        },
        "environment": {
            name: os.environ.get(name)
            for name in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES")
        },
    }


def _dataset_hashes(data_x: np.ndarray, data_y: np.ndarray) -> tuple[str, str]:
    return (
        _array_bundle_sha256("alberta.ipmnist_screening.materialized_x.v1", {"x": data_x}),
        _array_bundle_sha256("alberta.ipmnist_screening.materialized_y.v1", {"y": data_y}),
    )


def _dataset_sha256(data_x: np.ndarray, data_y: np.ndarray) -> str:
    x_digest, y_digest = _dataset_hashes(data_x, data_y)
    return hashlib.sha256(f"{x_digest}:{y_digest}".encode("ascii")).hexdigest()


def _canonical_dataset_shapes() -> tuple[tuple[int, int], tuple[int]]:
    return _CANONICAL_X_SHAPE, _CANONICAL_Y_SHAPE


def _canonical_dataset_hashes() -> tuple[str, str]:
    return _CANONICAL_X_SHA256, _CANONICAL_Y_SHA256


def _validated_arrays(data_x: object, data_y: object) -> tuple[np.ndarray, np.ndarray]:
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("dataset features must be an exact float32 NumPy array")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("dataset labels must be an exact int32 NumPy array")
    if (
        data_x.shape != _canonical_dataset_shapes()[0]
        or data_y.shape != _canonical_dataset_shapes()[1]
    ):
        raise ValueError("dataset shape differs from the canonical MNIST train split")
    if not data_x.flags.c_contiguous or not data_y.flags.c_contiguous:
        raise ValueError("dataset arrays must be C-contiguous")
    if not np.isfinite(data_x).all() or np.any(data_x < -1.0) or np.any(data_x > 1.0):
        raise ValueError("dataset features must be finite and lie in [-1,1]")
    if np.any(data_y < 0) or np.any(data_y >= CONFIG.n_classes):
        raise ValueError("dataset labels lie outside the ten-class range")
    if _dataset_hashes(data_x, data_y) != _canonical_dataset_hashes():
        raise ValueError("dataset bytes differ from the canonical materialization")
    return data_x, data_y


def _dataset_identity(data_x: np.ndarray, data_y: np.ndarray) -> dict[str, object]:
    x_digest, y_digest = _dataset_hashes(data_x, data_y)
    return {
        "provider": "OpenML",
        "dataset": "mnist_784",
        "version": 1,
        "rows": data_x.shape[0],
        "x_shape": list(data_x.shape),
        "y_shape": list(data_y.shape),
        "x_dtype": data_x.dtype.str,
        "y_dtype": data_y.dtype.str,
        "x_sha256": x_digest,
        "y_sha256": y_digest,
        "sha256": _dataset_sha256(data_x, data_y),
        "numeric_bytes": data_x.nbytes + data_y.nbytes,
    }


def _validated_dataset_identity(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("plan dataset identity must be an exact object")
    dataset = cast(dict[str, object], value)
    fields = {
        "provider",
        "dataset",
        "version",
        "rows",
        "x_shape",
        "y_shape",
        "x_dtype",
        "y_dtype",
        "x_sha256",
        "y_sha256",
        "sha256",
        "numeric_bytes",
    }
    expected_x_shape, expected_y_shape = _canonical_dataset_shapes()
    expected_x_sha256, expected_y_sha256 = _canonical_dataset_hashes()
    expected_combined = hashlib.sha256(
        f"{expected_x_sha256}:{expected_y_sha256}".encode("ascii")
    ).hexdigest()
    expected_bytes = (np.prod(expected_x_shape) + np.prod(expected_y_shape)) * 4
    if (
        set(dataset) != fields
        or dataset.get("provider") != "OpenML"
        or dataset.get("dataset") != "mnist_784"
        or dataset.get("version") != 1
        or dataset.get("rows") != expected_x_shape[0]
        or dataset.get("x_shape") != list(expected_x_shape)
        or dataset.get("y_shape") != list(expected_y_shape)
        or dataset.get("x_dtype") != np.dtype(np.float32).str
        or dataset.get("y_dtype") != np.dtype(np.int32).str
        or dataset.get("x_sha256") != expected_x_sha256
        or dataset.get("y_sha256") != expected_y_sha256
        or dataset.get("sha256") != expected_combined
        or dataset.get("numeric_bytes") != expected_bytes
    ):
        raise ValueError("plan dataset identity differs from canonical MNIST")
    return dataset


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-noise-curvature-initial-parameters-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        host = np.asarray(leaf)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _execution_identity(seed: int, n_train: int) -> dict[str, str]:
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    params = init_mlp_params(key_init, CONFIG)
    schedule = build_schedule(key_schedule, CONFIG, n_train)
    digest = hashlib.sha256(b"asi-noise-curvature-schedule-v1\0")
    for raw in (schedule.permutations, schedule.example_indices):
        value = np.asarray(raw, dtype=np.int32)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.astype("<i4", copy=False).tobytes(order="C"))
    return {
        "schedule_sha256": digest.hexdigest(),
        "initial_parameters_sha256": _array_tree_sha256(params),
        "prng_implementation": "threefry2x32",
    }


def _statistics() -> dict[str, object]:
    return {
        "primary_metric": "mean_online_accuracy",
        "method": "two_sided_paired_student_t",
        "confidence_level": 0.95,
        "degrees_of_freedom": 4,
        "critical_value": _T95_DF4,
        "mechanism": "combined_minus_fixed_adam_l2_ci_lower_gt_0",
        "causal": "combined_minus_each_single_signal_ablation_ci_lower_gt_0",
        "hillclimb": "combined_minus_live_rls_mean_ge_0.005_and_ci_lower_gt_0",
    }


def _unsigned_plan(dataset: dict[str, object]) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "matrix": {
            "arms": list(ARM_ROSTER),
            "seeds": list(SEEDS),
            "shard_count": len(ARM_ROSTER) * len(SEEDS),
            "ordering": "seed_major_then_arm_roster",
            "execution": "one_shard_per_fresh_python_process",
        },
        "config": CONFIG.to_config(),
        "protocol": {
            "paper_revision": PAPER_REVISION,
            "official_code_status": OFFICIAL_CODE_STATUS,
            "protocol_differences": list(PROTOCOL_DIFFERENCES),
            "allowed_boundary_information": [],
            "allowed_task_information": ["current_example_label"],
            "matched_axes": [
                "seed_derived_task_permutations",
                "seed_derived_example_indices",
                "initial_parameters",
                "observations",
                "updates",
                "allowed_boundary_information",
                "allowed_task_information",
            ],
            "arm_specific_resources_are_charged_not_forced_equal": True,
        },
        "execution_gate": {
            "execution_authorized": False,
            "authorization_transition": "separate_reviewed_source_change_required",
            "reservation_before_dataset_rng_or_runner_work": True,
            "first_dispatch_failure": "durable_failure_marker_blocks_retry_until_disposition",
        },
        "identity": {
            "dataset": copy.deepcopy(dataset),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_execution_attestation": True,
        },
        "statistics": _statistics(),
        "resources": {
            "data_steps_per_shard": CONFIG.n_steps,
            "environment_steps_per_shard": 0,
            "updates_per_shard": CONFIG.n_steps,
            "shard_count": len(ARM_ROSTER) * len(SEEDS),
            "timing_is_telemetry_only": True,
        },
        "policy": {
            "development_only": True,
            "permanently_nonpromoting": True,
            "scientific_promotion_allowed": False,
            "reference_dev_update_allowed": False,
            "all_completed_outcomes_retained": True,
            "negative_outcomes_retained": True,
        },
        "output_namespace": "outputs/noise_curvature_ipmnist/cheap_screen.v1",
    }


def build_plan(data_x: object, data_y: object) -> dict[str, object]:
    """Build the exact source/runtime/dataset-bound preauthorization plan."""
    x, y = _validated_arrays(data_x, data_y)
    plan = _unsigned_plan(_dataset_identity(x, y))
    plan["plan_sha256"] = _digest(plan)
    return validate_plan(plan, data_x=x, data_y=y)


def validate_plan(
    value: object, *, data_x: object | None = None, data_y: object | None = None
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("plan must be an exact object")
    plan = cast(dict[str, object], value)
    if set(plan) != _PLAN_FIELDS:
        raise ValueError("plan fields differ from the frozen schema")
    identity = plan.get("identity")
    if type(identity) is not dict or type(identity.get("dataset")) is not dict:
        raise ValueError("plan identity is invalid")
    dataset = _validated_dataset_identity(identity["dataset"])
    if (data_x is None) != (data_y is None):
        raise ValueError("dataset validation requires both arrays")
    if data_x is not None and data_y is not None:
        x, y = _validated_arrays(data_x, data_y)
        if dataset != _dataset_identity(x, y):
            raise ValueError("plan dataset identity differs from supplied dataset")
    expected = _unsigned_plan(dataset)
    claimed = plan.get("plan_sha256")
    if type(claimed) is not str or claimed != _digest(expected):
        raise ValueError("plan digest drifted")
    expected["plan_sha256"] = claimed
    if plan != expected:
        raise ValueError("plan differs from the current frozen campaign")
    return copy.deepcopy(plan)


def _metrics(result: ScreeningRunResult) -> dict[str, float]:
    return {
        "mean_online_accuracy": float(np.mean(result.per_task_accuracy)),
        "mean_loss": float(np.mean(result.per_task_loss)),
        "mean_plasticity": float(np.mean(result.per_task_plasticity)),
    }


def _common_result(result: ScreeningRunResult, schema: str) -> dict[str, object]:
    spec = screening_spec(result.config_name)
    if result.config != CONFIG or result.seed not in SEEDS or result.noise_mode != "step":
        raise ValueError("result does not belong to the frozen campaign")
    if result.hyperparameters != spec.hyperparameters:
        raise ValueError("result hyperparameters drift from the registered arm")
    return {
        "schema": schema,
        "arm": result.config_name,
        "base_learner": result.base_learner,
        "seed": result.seed,
        "config": CONFIG.to_config(),
        "hyperparameters": dict(result.hyperparameters),
        "metrics": _metrics(result),
        "resources": {
            "persistent_bytes": _persistent_bytes(result.config_name),
            "environment_steps": 0,
            "data_steps": CONFIG.n_steps,
            "updates": CONFIG.n_steps,
            "first_order_gradient_queries": _gradient_queries(result.config_name),
            "loss_only_queries": CONFIG.n_steps,
            "hessian_vector_product_queries": _hvp_queries(result.config_name),
            "controller_events": _controller_events(result.config_name),
            "rls_updates": CONFIG.n_steps if result.config_name == LIVE_CONTROL else 0,
            "model_queries": _model_queries(result.config_name),
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "outcome": "inconclusive",
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _persistent_bytes(arm: str) -> int:
    if arm != LIVE_CONTROL:
        return noise_curvature_persistent_bytes(
            parameter_count=CONFIG.parameter_count,
            input_dim=CONFIG.input_dim,
            control_interval=int(BASE_HYPERPARAMETERS["control_interval"]),
        )
    feature_dim = CONFIG.hidden2 + 1
    scalars = (
        CONFIG.parameter_count  # utility tree
        + 1  # step
        + 4 * CONFIG.input_dim  # norm mean/var/count plus fast mean
        + feature_dim * feature_dim  # RLS inverse covariance
        + feature_dim * CONFIG.n_classes  # RLS readout
    )
    return 4 * scalars


def _controller_events(arm: str) -> int:
    if arm == LIVE_CONTROL:
        return 0
    return CONFIG.n_steps // int(BASE_HYPERPARAMETERS["control_interval"])


def _hvp_queries(arm: str) -> int:
    if arm == LIVE_CONTROL:
        return 0
    return _controller_events(arm) * 3 * int(BASE_HYPERPARAMETERS["power_iterations"])


def _gradient_queries(arm: str) -> int:
    if arm == LIVE_CONTROL:
        return CONFIG.n_steps
    return CONFIG.n_steps + _controller_events(arm) * int(BASE_HYPERPARAMETERS["control_interval"])


def _model_queries(arm: str) -> int:
    if arm == LIVE_CONTROL:
        return CONFIG.n_steps * 2
    return _gradient_queries(arm) + CONFIG.n_steps + _hvp_queries(arm)


def live_control_result_payload(result: ScreeningRunResult) -> dict[str, object]:
    """Build the exact separately protocolled receipt for the live RLS arm."""
    if result.config_name != LIVE_CONTROL:
        raise ValueError("live control receipt requires the registered RLS arm")
    return validate_result(_common_result(result, CONTROL_RESULT_SCHEMA))


def scheduler_result_payload(result: ScreeningRunResult) -> dict[str, object]:
    if result.config_name not in registered_arms():
        raise ValueError("scheduler receipt requires one registered scheduler arm")
    return validate_result(_common_result(result, SCHEDULER_RESULT_SCHEMA))


def validate_result(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("campaign result must be an exact object")
    result = cast(dict[str, object], value)
    arm = result.get("arm")
    schema = CONTROL_RESULT_SCHEMA if arm == LIVE_CONTROL else SCHEDULER_RESULT_SCHEMA
    if result.get("schema") != schema or type(arm) is not str or arm not in ARM_ROSTER:
        raise ValueError("campaign result schema or arm is invalid")
    seed = result.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("campaign result seed is outside the frozen roster")
    spec = screening_spec(arm)
    if result.get("base_learner") != spec.base_learner:
        raise ValueError("campaign result base learner drifted")
    if (
        result.get("config") != CONFIG.to_config()
        or result.get("hyperparameters") != spec.hyperparameters
    ):
        raise ValueError("campaign result hyperparameters or config drifted")
    metrics = result.get("metrics")
    if type(metrics) is not dict or set(metrics) != {
        "mean_online_accuracy",
        "mean_loss",
        "mean_plasticity",
    }:
        raise ValueError("campaign result metrics are invalid")
    for name, raw in metrics.items():
        if type(raw) is not float or not math.isfinite(raw):
            raise ValueError(f"campaign metric {name} must be finite")
    if not 0.0 <= cast(float, metrics["mean_online_accuracy"]) <= 1.0:
        raise ValueError("mean online accuracy must lie in [0,1]")
    resources = result.get("resources")
    expected_resources = {
        "persistent_bytes": _persistent_bytes(arm),
        "environment_steps": 0,
        "data_steps": CONFIG.n_steps,
        "updates": CONFIG.n_steps,
        "first_order_gradient_queries": _gradient_queries(arm),
        "loss_only_queries": CONFIG.n_steps,
        "hessian_vector_product_queries": _hvp_queries(arm),
        "controller_events": _controller_events(arm),
        "rls_updates": CONFIG.n_steps if arm == LIVE_CONTROL else 0,
        "model_queries": _model_queries(arm),
        "timing_seconds": cast(dict[str, object], resources).get("timing_seconds")
        if type(resources) is dict
        else None,
        "timing_is_telemetry_only": True,
    }
    if type(resources) is not dict or resources != expected_resources:
        raise ValueError("campaign result resource accounting drifted")
    timing = resources["timing_seconds"]
    if type(timing) is not float or not math.isfinite(timing) or not 0.0 <= timing <= 604_800.0:
        raise ValueError("timing telemetry is invalid")
    fixed = {
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "outcome": "inconclusive",
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    if any(result.get(name) != expected for name, expected in fixed.items()):
        raise ValueError("campaign result policy or information boundary drifted")
    expected_fields = _result_fields()
    if set(result) != expected_fields:
        raise ValueError("campaign result fields differ from the frozen schema")
    return copy.deepcopy(result)


def _result_fields() -> set[str]:
    return set(
        (
            "schema",
            "arm",
            "base_learner",
            "seed",
            "config",
            "hyperparameters",
            "metrics",
            "resources",
            "allowed_boundary_information",
            "allowed_task_information",
            "outcome",
            "outcome_retained",
            "development_only",
            "scientific_promotion_allowed",
        )
    )


def _unsigned_shard(
    plan: dict[str, object], result: dict[str, object], execution_identity: dict[str, str]
) -> dict[str, object]:
    return {
        "schema": SHARD_SCHEMA,
        "plan_id": PLAN_ID,
        "plan_sha256": plan["plan_sha256"],
        "arm": result["arm"],
        "seed": result["seed"],
        "authorization": {
            "execution_authorized": False,
            "transition": "separate_reviewed_source_change_required",
        },
        "execution_identity": copy.deepcopy(execution_identity),
        "result": copy.deepcopy(result),
    }


def _build_shard_for_test(
    plan: dict[str, object], data_x: np.ndarray, data_y: np.ndarray, result: ScreeningRunResult
) -> dict[str, object]:
    """Construct synthetic validator input without exposing an execution capability."""
    checked = validate_plan(plan, data_x=data_x, data_y=data_y)
    receipt = (
        live_control_result_payload(result)
        if result.config_name == LIVE_CONTROL
        else scheduler_result_payload(result)
    )
    value = _unsigned_shard(checked, receipt, _execution_identity(result.seed, data_x.shape[0]))
    value["shard_sha256"] = _digest(value)
    return validate_shard(value, checked)


def validate_shard(value: object, plan: object) -> dict[str, object]:
    checked_plan = validate_plan(plan)
    if type(value) is not dict:
        raise ValueError("shard must be an exact object")
    shard = cast(dict[str, object], value)
    if set(shard) != _SHARD_FIELDS:
        raise ValueError("shard fields differ from the frozen schema")
    result = validate_result(shard.get("result"))
    arm = cast(str, result["arm"])
    seed = cast(int, result["seed"])
    expected_identity = _execution_identity(
        seed, cast(dict[str, Any], checked_plan["identity"])["dataset"]["rows"]
    )
    unsigned = _unsigned_shard(checked_plan, result, expected_identity)
    if shard.get("execution_identity") != expected_identity:
        raise ValueError("shard execution identity drifted")
    claimed = shard.get("shard_sha256")
    if type(claimed) is not str or claimed != _digest(unsigned):
        raise ValueError("shard digest drifted")
    unsigned["shard_sha256"] = claimed
    if shard != unsigned or shard.get("arm") != arm or shard.get("seed") != seed:
        raise ValueError("shard/result identity drifted")
    return copy.deepcopy(shard)


def _paired(values: list[float]) -> dict[str, object]:
    if len(values) != len(SEEDS) or any(not math.isfinite(value) for value in values):
        raise ValueError("paired comparison requires five finite deltas")
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    half = _T95_DF4 * sem
    return {"mean_delta": mean, "ci95_lower": mean - half, "ci95_upper": mean + half}


def _summary(by_key: dict[tuple[int, str], dict[str, object]]) -> dict[str, object]:
    combined = "noise_curvature_combined"

    def metric(seed: int, arm: str) -> float:
        result = cast(dict[str, Any], by_key[(seed, arm)]["result"])
        return cast(float, result["metrics"]["mean_online_accuracy"])

    def comparison(control: str) -> dict[str, object]:
        stats = _paired([metric(seed, combined) - metric(seed, control) for seed in SEEDS])
        stats["candidate"] = combined
        stats["control"] = control
        return stats

    mechanism = comparison("noise_curvature_fixed_adam_l2")
    mechanism["outcome"] = _zero_threshold_outcome(mechanism)
    causal_comparisons = [
        comparison("noise_curvature_gradient_only"),
        comparison("noise_curvature_volatility_only"),
    ]
    for item in causal_comparisons:
        item["outcome"] = _zero_threshold_outcome(item)
    causal_supported = all(item["outcome"] == "supported" for item in causal_comparisons)
    causal_rejected = any(item["outcome"] == "rejected" for item in causal_comparisons)
    causal = {
        "comparisons": causal_comparisons,
        "outcome": (
            "supported" if causal_supported else "rejected" if causal_rejected else "inconclusive"
        ),
    }
    hillclimb = comparison(LIVE_CONTROL)
    hillclimb["outcome"] = _hillclimb_outcome(hillclimb)
    outcomes = (mechanism["outcome"], causal["outcome"], hillclimb["outcome"])
    return {
        "mechanism": mechanism,
        "causal": causal,
        "hillclimb": hillclimb,
        "overall_outcome": (
            "supported"
            if outcomes == ("supported",) * 3
            else "rejected"
            if "rejected" in outcomes
            else "inconclusive"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _zero_threshold_outcome(comparison: dict[str, object]) -> str:
    if cast(float, comparison["ci95_lower"]) > 0.0:
        return "supported"
    if cast(float, comparison["ci95_upper"]) < 0.0:
        return "rejected"
    return "inconclusive"


def _hillclimb_outcome(comparison: dict[str, object]) -> str:
    if (
        cast(float, comparison["mean_delta"]) >= 0.005
        and cast(float, comparison["ci95_lower"]) > 0.0
    ):
        return "supported"
    if cast(float, comparison["ci95_upper"]) < 0.005:
        return "rejected"
    return "inconclusive"


def _unsigned_aggregate(
    plan: dict[str, object], shards: list[dict[str, object]]
) -> dict[str, object]:
    by_key = {(cast(int, item["seed"]), cast(str, item["arm"])): item for item in shards}
    resources = {
        "data_steps": sum(
            cast(dict[str, Any], item["result"])["resources"]["data_steps"] for item in shards
        ),
        "environment_steps": 0,
        "updates": sum(
            cast(dict[str, Any], item["result"])["resources"]["updates"] for item in shards
        ),
        "model_queries": sum(
            cast(dict[str, Any], item["result"])["resources"]["model_queries"] for item in shards
        ),
        "persistent_bytes_by_arm": {arm: _persistent_bytes(arm) for arm in ARM_ROSTER},
        "timing_is_telemetry_only": True,
    }
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_retained_nonpromoting_development_result",
        "plan": copy.deepcopy(plan),
        "plan_sha256": plan["plan_sha256"],
        "shards": copy.deepcopy(shards),
        "summary": _summary(by_key),
        "resources": resources,
        "policy": copy.deepcopy(plan["policy"]),
    }


def build_aggregate(plan: object, shards: object) -> dict[str, object]:
    checked_plan = validate_plan(plan)
    if type(shards) is not list or len(shards) != len(ARM_ROSTER) * len(SEEDS):
        raise ValueError("aggregate requires the complete 5-arm by 5-seed matrix")
    checked = [validate_shard(item, checked_plan) for item in shards]
    by_key = {(cast(int, item["seed"]), cast(str, item["arm"])) for item in checked}
    expected = {(seed, arm) for seed in SEEDS for arm in ARM_ROSTER}
    if by_key != expected:
        raise ValueError("aggregate requires the complete 5-arm by 5-seed matrix")
    ordered = sorted(
        checked,
        key=lambda item: (
            SEEDS.index(cast(int, item["seed"])),
            ARM_ROSTER.index(cast(str, item["arm"])),
        ),
    )
    value = _unsigned_aggregate(checked_plan, ordered)
    value["aggregate_sha256"] = _digest(value)
    return validate_aggregate(value)


def validate_aggregate(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("aggregate must be an exact object")
    aggregate = cast(dict[str, object], value)
    plan = validate_plan(aggregate.get("plan"))
    raw_shards = aggregate.get("shards")
    if type(raw_shards) is not list:
        raise ValueError("aggregate shards must be an exact list")
    checked = [validate_shard(item, plan) for item in raw_shards]
    expected_keys = [(seed, arm) for seed in SEEDS for arm in ARM_ROSTER]
    actual_keys = [(cast(int, item["seed"]), cast(str, item["arm"])) for item in checked]
    if actual_keys != expected_keys:
        raise ValueError("aggregate requires the complete 5-arm by 5-seed matrix")
    unsigned = _unsigned_aggregate(plan, checked)
    claimed = aggregate.get("aggregate_sha256")
    if type(claimed) is not str or claimed != _digest(unsigned):
        raise ValueError("aggregate digest drifted")
    unsigned["aggregate_sha256"] = claimed
    if aggregate != unsigned:
        raise ValueError("aggregate statistics, resources, or policy drifted")
    return copy.deepcopy(aggregate)


def validate_shard_against_dataset(
    shard: object, plan: object, data_x: object, data_y: object
) -> dict[str, object]:
    """Independently rebind dataset, schedule, initialization, and receipt identity."""
    checked_plan = validate_plan(plan, data_x=data_x, data_y=data_y)
    checked = validate_shard(shard, checked_plan)
    identity = _execution_identity(cast(int, checked["seed"]), cast(np.ndarray, data_x).shape[0])
    if checked["execution_identity"] != identity:
        raise ValueError("replay execution identity drifted")
    return checked


def replay_shard(shard: object, plan: object, data_x: object, data_y: object) -> dict[str, object]:
    """Rerun one authorized shard and require deterministic behavioral equality."""
    _require_execution_authorized()
    checked = validate_shard_against_dataset(shard, plan, data_x, data_y)
    replayed = run_shard(
        plan,
        data_x,
        data_y,
        arm=cast(str, checked["arm"]),
        seed=cast(int, checked["seed"]),
    )
    expected_result = copy.deepcopy(cast(dict[str, Any], checked["result"]))
    actual_result = copy.deepcopy(cast(dict[str, Any], replayed["result"]))
    expected_result["resources"]["timing_seconds"] = 0.0
    actual_result["resources"]["timing_seconds"] = 0.0
    if (
        actual_result != expected_result
        or replayed["execution_identity"] != checked["execution_identity"]
    ):
        raise ValueError("behavioral replay differs from the retained shard")
    return {
        "schema": "asi.noise-curvature-ipmnist.replay.v1",
        "plan_sha256": cast(dict[str, object], plan)["plan_sha256"],
        "shard_sha256": checked["shard_sha256"],
        "dataset_sha256": cast(dict[str, Any], cast(dict[str, object], plan)["identity"])[
            "dataset"
        ]["sha256"],
        "behavior_matched": True,
        "timing_excluded_from_deterministic_comparison": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _require_execution_authorized() -> None:
    if _EXECUTION_AUTHORIZED is not True:
        raise RuntimeError(
            "campaign execution is disabled; a separate reviewed authorization "
            "transition is required"
        )


def run_shard(
    plan: object,
    data_x: object,
    data_y: object,
    *,
    arm: str,
    seed: int,
) -> dict[str, object]:
    """Execute one shard only after the literal source authorization transition."""
    _require_execution_authorized()
    checked_plan = validate_plan(plan, data_x=data_x, data_y=data_y)
    if arm not in ARM_ROSTER or seed not in SEEDS:
        raise ValueError("shard identity is outside the frozen matrix")
    x, y = _validated_arrays(data_x, data_y)
    result = run_screening_config(x, y, screening_spec(arm), seed, CONFIG, noise_mode="step")
    return _build_shard_for_test(checked_plan, x, y, result)


def write_new_json(path: Path, value: object) -> Path:
    """Publish one create-only, fsynced, read-only JSON generation."""
    _require_execution_authorized()
    with _reserved_output(path) as target:
        return _publish_reserved_json(target, value)


@contextmanager
def _reserved_output(path: Path) -> Iterator[tuple[Path, Path]]:
    """Reserve one output before dataset, RNG, or runner work begins."""
    _require_execution_authorized()
    if type(path) is not type(Path()):
        raise TypeError("output must be an exact Path")
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace immutable output: {destination}")
    marker = destination.with_name(f".{destination.name}.reservation")
    failure = destination.with_name(f".{destination.name}.failure")
    if failure.exists() or failure.is_symlink():
        raise RuntimeError(f"prior dispatch failure requires disposition: {failure}")
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
    )
    try:
        os.write(descriptor, b"asi-noise-curvature-output-reservation-v1\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield destination, marker
    except BaseException:
        os.replace(marker, failure)
        raise
    else:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass


def _publish_reserved_json(target: tuple[Path, Path], value: object) -> Path:
    """Create, fsync, chmod, and byte-for-byte reload one reserved output."""
    _require_execution_authorized()
    path, marker = target
    if not marker.is_file() or marker.is_symlink():
        raise RuntimeError("campaign output reservation was lost")
    encoded = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    if not 0 < len(encoded) <= _MAX_JSON_BYTES:
        raise ValueError("campaign output exceeds its byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400
    )
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("short campaign publication write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    reread = json.loads(path.read_text(encoding="utf-8"))
    if _canonical(reread) != _canonical(value):
        raise ValueError("published campaign output changed during strict reload")
    return path


def _load_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError("campaign input must be one bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("campaign input must be bounded UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("campaign input root must be an exact object")
    return cast(dict[str, object], value)


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"campaign JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="publish the frozen dataset-bound plan")
    plan.add_argument("--data-home", type=Path, default=default_openml_data_home())
    plan.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="strictly validate a plan or aggregate")
    validate.add_argument("input", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Expose hard-disabled planning and read-only strict validation."""
    args = _parser().parse_args(argv)
    if args.command == "plan":
        _require_execution_authorized()
        with _reserved_output(args.output) as target:
            data_x, data_y = load_mnist_train(args.data_home)
            _publish_reserved_json(target, build_plan(data_x, data_y))
        return 0
    value = _load_json(args.input)
    if value.get("schema") == PLAN_SCHEMA:
        checked: object = validate_plan(value)
    elif value.get("schema") == AGGREGATE_SCHEMA:
        checked = validate_aggregate(value)
    else:
        raise ValueError("input is not a noise-curvature campaign plan or aggregate")
    print(json.dumps({"valid": True, "sha256": _digest(checked)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
