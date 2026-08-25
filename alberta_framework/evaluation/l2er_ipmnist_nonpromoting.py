"""Strict development-only receipts for the adapted L2-ER IPMNIST comparator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

PAPER_REVISION = "arXiv:2509.22335v3"
OFFICIAL_COMMIT = "52ae3eb0702a9e6923f252c1f7cb29340eb5b3d5"
RESULT_SCHEMA = "asi.l2er-ipmnist.development-result.v3"
COMPARISON_ID = "asi.l2er-ipmnist.current-runner.v2"

L2ER_PROTOCOL = MappingProxyType(
    {
        "paper_revision": PAPER_REVISION,
        "official_repository": "https://github.com/KevinGuo27/lop-jax",
        "official_commit": OFFICIAL_COMMIT,
        "official_files": (
            "permuted_mnist/train_permuted_mnist.py",
            "permuted_mnist/config.py",
            "permuted_mnist/scripts/hyperparams/{l2,er,l2_er}.py",
        ),
        "protocol_differences": (
            "ASI uses its current 784-300-150-10 two-hidden-layer MLP, not 3x1000",
            "ASI uses 200x5000 IPMNIST by default, not 800x60000 P-MNIST",
            "ASI inputs are scaled to [-1,1], while lop-jax uses ToTensor [0,1]",
            "ASI uses its seed-derived task/example schedule",
            "ASI retains the 100-example ER buffer as charged persistent state",
            "ASI plasticity telemetry is an additional post-update metric",
            "ASI evaluates the same entropy-rank ratio with overflow-safe scaling",
            "ASI follows the pinned official implementation by buffering raw examples and "
            "recomputing features after each block, rather than buffering contemporaneous "
            "hidden features as shown in paper Algorithm 1",
        ),
        "official_er_batch": 100,
        "official_er_steps_per_batch": 1,
        "effective_rank_epsilon": 1e-8,
        "update_accounting": (
            "supervised_updates matches observations across arms; effective_rank_updates "
            "counts arm-specific auxiliary ER steps; total_optimizer_updates is their exact sum"
        ),
        "persistent_bytes_scope": (
            "float32 parameters plus the fixed 100-example float32 ER buffer, int32 count, "
            "and sticky bool transaction-validity flag; "
            "runner RNG and the externally supplied schedule are excluded"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }
)

_ARMS = {
    "l2er_mechanism_off": (0.0, 0.0, 0.0),
    "l2er_l2_only": (1e-4, 0.0, 0.0),
    "l2er_er_only": (0.0, 1e-3, 1.0),
    "l2er_combined": (1e-4, 1e-3, 1.0),
}
_HYPERPARAMETER_KEYS = frozenset(
    {
        "step_size",
        "weight_decay",
        "er_step_size",
        "er_batch_size",
        "er_steps_per_batch",
        "er_epsilon",
        "er_enabled",
    }
)
_INT32_MAX = (1 << 31) - 1
_MAX_BYTES = 256 * 1024 * 1024
_MAX_STRINGS = 16
_MAX_STRING_BYTES = 512


def _object(value: object, keys: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    trusted = cast(dict[object, object], value)
    raw_keys = tuple(trusted.keys())
    if len(raw_keys) != len(keys) or any(type(key) is not str for key in raw_keys):
        raise ValueError(f"{context} keys must be exactly {sorted(keys)}")
    if frozenset(cast(tuple[str, ...], raw_keys)) != keys:
        raise ValueError(f"{context} keys must be exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _int(value: object, *, context: str, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value <= _INT32_MAX:
        raise ValueError(f"{context} must be a {'positive' if positive else 'nonnegative'} int")
    return value


def _float(value: object, *, context: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or len(value) > _MAX_STRINGS
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > _MAX_STRING_BYTES
            for item in value
        )
    ):
        raise ValueError(f"{context} must be a list of non-empty strings")
    resolved = tuple(value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{context} must not contain duplicates")
    return resolved


def validate_l2er_development_result(payload: object) -> dict[str, object]:
    """Return a normalized copy after strict schema and accounting validation."""
    outer = _object(
        payload,
        frozenset(
            {
                "schema",
                "comparison_id",
                "paper_revision",
                "official_commit",
                "arm",
                "seed",
                "n_tasks",
                "task_length",
                "input_dim",
                "hidden1",
                "hidden2",
                "n_classes",
                "observations",
                "supervised_updates",
                "effective_rank_updates",
                "total_optimizer_updates",
                "allowed_boundary_information",
                "allowed_task_information",
                "hyperparameters",
                "metrics",
                "resources",
                "outcome",
                "outcome_retained",
                "development_only",
                "scientific_promotion_allowed",
            }
        ),
        context="result",
    )
    for key, expected in (
        ("schema", RESULT_SCHEMA),
        ("comparison_id", COMPARISON_ID),
        ("paper_revision", PAPER_REVISION),
        ("official_commit", OFFICIAL_COMMIT),
    ):
        if type(outer[key]) is not str or outer[key] != expected:
            raise ValueError(f"{key} does not match the frozen L2-ER protocol")
    arm = outer["arm"]
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError(f"arm must be one of {sorted(_ARMS)}")
    seed = _int(outer["seed"], context="seed")
    n_tasks = _int(outer["n_tasks"], context="n_tasks", positive=True)
    task_length = _int(outer["task_length"], context="task_length", positive=True)
    input_dim = _int(outer["input_dim"], context="input_dim", positive=True)
    hidden1 = _int(outer["hidden1"], context="hidden1", positive=True)
    hidden2 = _int(outer["hidden2"], context="hidden2", positive=True)
    n_classes = _int(outer["n_classes"], context="n_classes", positive=True)
    observations = _int(outer["observations"], context="observations", positive=True)
    supervised_updates = _int(
        outer["supervised_updates"], context="supervised_updates", positive=True
    )
    effective_rank_updates = _int(
        outer["effective_rank_updates"], context="effective_rank_updates"
    )
    total_optimizer_updates = _int(
        outer["total_optimizer_updates"], context="total_optimizer_updates", positive=True
    )
    if n_tasks > _INT32_MAX // task_length:
        raise ValueError("n_tasks * task_length exceeds signed int32")
    if observations != n_tasks * task_length or supervised_updates != observations:
        raise ValueError(
            "observations and supervised_updates must equal n_tasks * task_length"
        )
    boundary = _strings(
        outer["allowed_boundary_information"], context="allowed_boundary_information"
    )
    task_info = _strings(outer["allowed_task_information"], context="allowed_task_information")
    if boundary or task_info != ("current_example_label",):
        raise ValueError("allowed boundary/task information does not match the protocol")

    hp = _object(outer["hyperparameters"], _HYPERPARAMETER_KEYS, context="hyperparameters")
    normalized_hp = {
        key: _float(hp[key], context=f"hyperparameters.{key}", nonnegative=True)
        for key in _HYPERPARAMETER_KEYS
    }
    expected_wd, expected_er_lr, expected_enabled = _ARMS[arm]
    expected_hp = {
        "step_size": 1e-3,
        "weight_decay": expected_wd,
        "er_step_size": expected_er_lr,
        "er_batch_size": 100.0,
        "er_steps_per_batch": 1.0,
        "er_epsilon": 1e-8,
        "er_enabled": expected_enabled,
    }
    if normalized_hp != expected_hp:
        raise ValueError("hyperparameters do not match the registered L2-ER arm")
    if task_length % int(normalized_hp["er_batch_size"]) != 0:
        raise ValueError("task_length must be divisible by er_batch_size")

    metrics = _object(
        outer["metrics"],
        frozenset({"mean_online_accuracy", "mean_loss", "mean_plasticity"}),
        context="metrics",
    )
    normalized_metrics = {
        key: _float(metrics[key], context=f"metrics.{key}", nonnegative=True)
        for key in metrics
    }
    if normalized_metrics["mean_online_accuracy"] > 1.0:
        raise ValueError("mean_online_accuracy must lie in [0,1]")
    correct_count = normalized_metrics["mean_online_accuracy"] * observations
    if not math.isclose(
        correct_count,
        round(correct_count),
        rel_tol=0.0,
        abs_tol=64.0 * math.ulp(correct_count),
    ):
        raise ValueError(
            "mean_online_accuracy must lie on the integer correct-count lattice"
        )
    if normalized_metrics["mean_plasticity"] > 1.0:
        raise ValueError("mean_plasticity must lie in [0,1]")

    resources = _object(
        outer["resources"],
        frozenset(
            {
                "persistent_bytes",
                "environment_steps",
                "data_steps",
                "model_queries",
                "timing_seconds",
                "timing_is_telemetry_only",
            }
        ),
        context="resources",
    )
    persistent_bytes = _int(resources["persistent_bytes"], context="persistent_bytes")
    environment_steps = _int(resources["environment_steps"], context="environment_steps")
    data_steps = _int(resources["data_steps"], context="data_steps")
    model_queries = _int(resources["model_queries"], context="model_queries")
    timing_seconds = _float(
        resources["timing_seconds"], context="timing_seconds", nonnegative=True
    )
    if persistent_bytes > _MAX_BYTES or timing_seconds > 604_800.0:
        raise ValueError("resources exceed the bounded development protocol")
    if environment_steps != 0 or data_steps != observations:
        raise ValueError("environment_steps/data_steps do not match the supervised protocol")
    er_updates = observations // 100 if expected_enabled == 1.0 else 0
    if effective_rank_updates != er_updates:
        raise ValueError("effective_rank_updates does not match the executed ER schedule")
    if total_optimizer_updates != supervised_updates + effective_rank_updates:
        raise ValueError(
            "total_optimizer_updates must equal supervised_updates plus effective_rank_updates"
        )
    if model_queries != 2 * observations + er_updates:
        raise ValueError("model_queries does not match the executed update semantics")
    if resources["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain True")
    products = (
        input_dim * hidden1,
        hidden1 * hidden2,
        hidden2 * n_classes,
        100 * input_dim,
    )
    if any(product > _INT32_MAX for product in products):
        raise ValueError("derived parameter/buffer size exceeds signed int32")
    parameter_count = sum(products[:3]) + hidden1 + hidden2 + n_classes
    if parameter_count > _INT32_MAX - products[3] - 1:
        raise ValueError("derived persistent scalar count exceeds signed int32")
    expected_persistent_bytes = 4 * (parameter_count + products[3] + 1) + 1
    if expected_persistent_bytes > _MAX_BYTES:
        raise ValueError("derived persistent bytes exceed 256 MiB")
    if persistent_bytes != expected_persistent_bytes:
        raise ValueError("persistent_bytes must exactly include parameters and ER buffer")
    if type(outer["outcome"]) is not str or outer["outcome"] not in {
        "supported",
        "rejected",
        "inconclusive",
    }:
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if outer["outcome_retained"] is not True:
        raise ValueError("outcome_retained must permanently remain True")
    if outer["development_only"] is not True:
        raise ValueError("development_only must permanently remain True")
    if outer["scientific_promotion_allowed"] is not False:
        raise ValueError("scientific_promotion_allowed must permanently remain False")

    return {
        **dict(outer),
        "seed": seed,
        "n_tasks": n_tasks,
        "task_length": task_length,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "n_classes": n_classes,
        "observations": observations,
        "supervised_updates": supervised_updates,
        "effective_rank_updates": effective_rank_updates,
        "total_optimizer_updates": total_optimizer_updates,
        "allowed_boundary_information": list(boundary),
        "allowed_task_information": list(task_info),
        "hyperparameters": normalized_hp,
        "metrics": normalized_metrics,
        "resources": {
            "persistent_bytes": persistent_bytes,
            "environment_steps": environment_steps,
            "data_steps": data_steps,
            "model_queries": model_queries,
            "timing_seconds": timing_seconds,
            "timing_is_telemetry_only": True,
        },
    }


def validate_matched_l2er_development_results(
    payloads: object,
) -> tuple[dict[str, object], ...]:
    """Validate the four-arm comparison and every required matched axis."""
    payload_type = type(payloads)
    if not (payload_type is list or payload_type is tuple):
        raise ValueError("the matched comparison must contain exactly four results")
    trusted_payloads = cast(list[object] | tuple[object, ...], payloads)
    if len(trusted_payloads) != len(_ARMS):
        raise ValueError("the matched comparison must contain exactly four results")
    results = tuple(validate_l2er_development_result(payload) for payload in trusted_payloads)
    if {result["arm"] for result in results} != set(_ARMS):
        raise ValueError("the matched comparison must contain each registered arm exactly once")
    first = results[0]
    axes = (
        "comparison_id",
        "seed",
        "n_tasks",
        "task_length",
        "input_dim",
        "hidden1",
        "hidden2",
        "n_classes",
        "observations",
        "supervised_updates",
        "allowed_boundary_information",
        "allowed_task_information",
    )
    for result in results[1:]:
        if any(result[axis] != first[axis] for axis in axes):
            raise ValueError("all required L2-ER comparison axes must match")
    return results
