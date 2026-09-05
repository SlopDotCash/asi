"""Strict development receipts for the bounded growing/elastic IPMNIST slice."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

PAPER_REVISION = "arXiv:2608.01475v1"
PAPER_SOURCE_SHA256 = "3fd7590d58d32c0315ac8e3b8d7f68241d7e2bac8d32830b462bd75b22c645c3"
RESULT_SCHEMA = "asi.bounded-elastic-ipmnist.development-result.v1"
COMPARISON_ID = "asi.bounded-elastic-ipmnist.current-runner.v1"

BOUNDED_ELASTIC_PROTOCOL = MappingProxyType(
    {
        "paper_revision": PAPER_REVISION,
        "paper_source_sha256": PAPER_SOURCE_SHA256,
        "paper_url": "https://arxiv.org/abs/2608.01475v1",
        "official_repository": None,
        "official_code_status": (
            "No official repository is linked by arXiv v1 or its source, and a targeted "
            "GitHub search on 2026-08-18 found none."
        ),
        "protocol_differences": (
            "ASI uses its fixed-shape two-hidden-layer MLP, not a dynamically deep cascade",
            "capacity is preallocated and hidden-1 activity is masked to keep JAX shapes static",
            "ASI begins with half of hidden layer 1 active and caps growth at its configured width",
            "growth and pruning occur at each known 5,000-example boundary",
            "elastic pruning uses the prior task's online activation sums, not a future 5% sample",
            "ASI elastic events prune exactly one least-active unit and grow exactly one unit; "
            "the paper removes zero or more units estimated dead at a boundary",
            "a pruned slot is freshly initialized before it is reactivated",
            "ASI uses 5,000 MNIST examples per task, not the paper's 10,000/40,000 lanes",
            "ASI inputs are [-1,1], while the paper uses [0,1] MNIST/FMNIST",
            "the fixed-capacity comparator is current-runner SGD plus CBP-style recycling",
        ),
        "persistent_bytes_scope": (
            "float32 learner parameters plus learner-owned state; runner RNG, schedule, and "
            "externally materialized examples are excluded"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcome_retention_required": True,
    }
)

_STRUCTURE_BASE = {
    "step_size": 1e-3,
    "initial_active_fraction": 0.5,
    "structure_interval": 5000.0,
}
_ARMS: dict[str, dict[str, float]] = {
    "bounded_structure_off": {
        **_STRUCTURE_BASE,
        "growth_enabled": 0.0,
        "pruning_enabled": 0.0,
    },
    "bounded_growth": {
        **_STRUCTURE_BASE,
        "growth_enabled": 1.0,
        "pruning_enabled": 0.0,
    },
    "bounded_elastic": {
        **_STRUCTURE_BASE,
        "growth_enabled": 1.0,
        "pruning_enabled": 1.0,
    },
    "bounded_fixed_cbp": {
        "step_size": 1e-3,
        "cbp_decay_rate": 0.99,
        "cbp_replacement_rate": 1e-4,
        "cbp_maturity_threshold": 100.0,
    },
}
_INT32_MAX = 2**31 - 1


def registered_bounded_elastic_hyperparameters(arm: str) -> dict[str, float]:
    """Return a mutable copy of one exact registered arm."""
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError(f"arm must be one of {sorted(_ARMS)}")
    return dict(_ARMS[arm])


def _object(value: object, keys: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    if set(value) != keys:
        raise ValueError(f"{context} keys must be exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _int(value: object, *, context: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value > _INT32_MAX:
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{context} must be a {requirement} signed-int32 integer")
    return value


def _float(value: object, *, context: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{context} must be a list of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates")
    return result


def _parameter_count(
    input_dim: int, hidden1: int, hidden2: int, n_classes: int, *, active1: int
) -> int:
    return (
        input_dim * active1
        + active1
        + active1 * hidden2
        + hidden2
        + hidden2 * n_classes
        + n_classes
    )


def bounded_elastic_resource_expectations(
    *, arm: str, n_tasks: int, input_dim: int, hidden1: int, hidden2: int, n_classes: int
) -> dict[str, int]:
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError(f"arm must be one of {sorted(_ARMS)}")
    n_tasks = _int(n_tasks, context="n_tasks", positive=True)
    input_dim = _int(input_dim, context="input_dim", positive=True)
    hidden1 = _int(hidden1, context="hidden1", positive=True)
    hidden2 = _int(hidden2, context="hidden2", positive=True)
    n_classes = _int(n_classes, context="n_classes", positive=True)
    full_parameters = _parameter_count(
        input_dim, hidden1, hidden2, n_classes, active1=hidden1
    )
    parameter_bytes = 4 * full_parameters
    structural_state_bytes = hidden1 + 4 * hidden1 + 4
    cbp_state_bytes = 8 * (hidden1 + hidden2) + 8
    persistent_budget = parameter_bytes + max(structural_state_bytes, cbp_state_bytes)
    initial_active = max(1, int(hidden1 * 0.5))
    if arm == "bounded_growth":
        peak_active = min(hidden1, initial_active + n_tasks)
        grown = peak_active - initial_active
        pruned = 0
        events = n_tasks
    elif arm == "bounded_elastic":
        peak_active = initial_active
        grown = n_tasks
        pruned = n_tasks
        events = n_tasks
    elif arm == "bounded_structure_off":
        peak_active = initial_active
        grown = pruned = events = 0
    else:
        peak_active = hidden1
        grown = pruned = events = 0
    persistent_bytes = parameter_bytes + (
        cbp_state_bytes if arm == "bounded_fixed_cbp" else structural_state_bytes
    )
    return {
        "persistent_bytes": persistent_bytes,
        "peak_persistent_bytes_budget": persistent_budget,
        "peak_parameter_bytes": parameter_bytes,
        "final_active_parameter_bytes_budget": parameter_bytes,
        "final_active_parameter_bytes": 4
        * _parameter_count(input_dim, hidden1, hidden2, n_classes, active1=peak_active),
        "final_active_hidden1_units": peak_active,
        "peak_active_hidden1_units": peak_active,
        "structure_events": events,
        "units_grown": grown,
        "units_pruned": pruned,
    }


def validate_bounded_elastic_development_result(payload: object) -> dict[str, object]:
    """Validate one exact, permanently nonpromoting bounded-structure result."""
    outer = _object(
        payload,
        frozenset(
            {
                "schema",
                "comparison_id",
                "paper_revision",
                "paper_source_sha256",
                "arm",
                "seed",
                "n_tasks",
                "task_length",
                "input_dim",
                "hidden1",
                "hidden2",
                "n_classes",
                "observations",
                "updates",
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
        ("paper_source_sha256", PAPER_SOURCE_SHA256),
    ):
        if outer[key] != expected:
            raise ValueError(f"{key} does not match the registered protocol")
    arm = outer["arm"]
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError(f"arm must be one of {sorted(_ARMS)}")
    seed = _int(outer["seed"], context="seed")
    dimensions = {
        key: _int(outer[key], context=key, positive=True)
        for key in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
    }
    observations = _int(outer["observations"], context="observations", positive=True)
    updates = _int(outer["updates"], context="updates", positive=True)
    if dimensions["task_length"] != 5000:
        raise ValueError("task_length must match the registered 5,000-example boundary")
    if observations != dimensions["n_tasks"] * dimensions["task_length"]:
        raise ValueError("observations must equal n_tasks * task_length")
    if updates != observations:
        raise ValueError("updates must equal observations")
    boundary = _strings(
        outer["allowed_boundary_information"], context="allowed_boundary_information"
    )
    task_info = _strings(outer["allowed_task_information"], context="allowed_task_information")
    if boundary != ("known_fixed_length_task_boundary",):
        raise ValueError("allowed boundary information does not match the protocol")
    if task_info != ("current_example_label",):
        raise ValueError("allowed task information does not match the protocol")

    expected_hp = _ARMS[arm]
    hp = _object(outer["hyperparameters"], frozenset(expected_hp), context="hyperparameters")
    normalized_hp = {
        key: _float(hp[key], context=f"hyperparameters.{key}", nonnegative=True)
        for key in expected_hp
    }
    if normalized_hp != expected_hp:
        raise ValueError("hyperparameters do not match the registered arm")

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
    if normalized_metrics["mean_plasticity"] > 1.0:
        raise ValueError("mean_plasticity must lie in [0,1]")

    expected_resources = bounded_elastic_resource_expectations(
        arm=arm,
        n_tasks=dimensions["n_tasks"],
        input_dim=dimensions["input_dim"],
        hidden1=dimensions["hidden1"],
        hidden2=dimensions["hidden2"],
        n_classes=dimensions["n_classes"],
    )
    resource_keys = frozenset(
        {
            *expected_resources,
            "environment_steps",
            "data_steps",
            "model_queries",
            "timing_seconds",
            "timing_is_telemetry_only",
        }
    )
    resources = _object(outer["resources"], resource_keys, context="resources")
    normalized_resources = {
        key: _int(resources[key], context=f"resources.{key}") for key in expected_resources
    }
    if normalized_resources != expected_resources:
        raise ValueError("declared structure or memory resources do not match the protocol")
    environment_steps = _int(resources["environment_steps"], context="environment_steps")
    data_steps = _int(resources["data_steps"], context="data_steps")
    model_queries = _int(resources["model_queries"], context="model_queries")
    timing_seconds = _float(
        resources["timing_seconds"], context="timing_seconds", nonnegative=True
    )
    if environment_steps != 0 or data_steps != observations or model_queries != 2 * observations:
        raise ValueError("step/query resources do not match the executed supervised protocol")
    if resources["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain True")
    if expected_resources["persistent_bytes"] > expected_resources["peak_persistent_bytes_budget"]:
        raise ValueError("the arm exceeds the registered peak persistent-memory budget")
    if (
        expected_resources["final_active_parameter_bytes"]
        > expected_resources["final_active_parameter_bytes_budget"]
    ):
        raise ValueError("the arm exceeds the registered final active-size budget")
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
        **dimensions,
        "observations": observations,
        "updates": updates,
        "allowed_boundary_information": list(boundary),
        "allowed_task_information": list(task_info),
        "hyperparameters": normalized_hp,
        "metrics": normalized_metrics,
        "resources": {
            **normalized_resources,
            "environment_steps": environment_steps,
            "data_steps": data_steps,
            "model_queries": model_queries,
            "timing_seconds": timing_seconds,
            "timing_is_telemetry_only": True,
        },
    }


def validate_matched_bounded_elastic_results(
    payloads: Sequence[object],
) -> tuple[dict[str, object], ...]:
    """Require all four arms and exact workload/information/budget matching."""
    if type(payloads) not in {list, tuple} or len(payloads) != len(_ARMS):
        raise ValueError("the matched comparison must contain exactly four results")
    results = tuple(validate_bounded_elastic_development_result(item) for item in payloads)
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
        "updates",
        "allowed_boundary_information",
        "allowed_task_information",
    )
    first_resources = cast(Mapping[str, object], first["resources"])
    for result in results[1:]:
        if any(result[axis] != first[axis] for axis in axes):
            raise ValueError("all workload and information axes must match")
        resources = cast(Mapping[str, object], result["resources"])
        if (
            resources["peak_persistent_bytes_budget"]
            != first_resources["peak_persistent_bytes_budget"]
            or resources["peak_parameter_bytes"] != first_resources["peak_parameter_bytes"]
            or resources["final_active_parameter_bytes_budget"]
            != first_resources["final_active_parameter_bytes_budget"]
        ):
            raise ValueError("all arms must share the same peak-memory and final-size budgets")
    return results
