"""Fail-closed development receipts for the adapted C-CHAIN IPMNIST lane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

from alberta_framework.benchmarks.cchain_ipmnist import (
    COEFFICIENT_DELAY,
    COEFFICIENT_WINDOW,
    MAX_NTK_EXAMPLES,
    NTK_ENERGY_THRESHOLD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    PAPER_REVISION,
    REFERENCE_CAPACITY,
    SNAPSHOT_WARMUP_UPDATES,
)
from alberta_framework.benchmarks.upgd_ipmnist import ADAMW_PROTOCOL_HYPERPARAMETERS

RESULT_SCHEMA: Final = "asi.cchain-ipmnist.development-result.v1"
COMPARISON_ID: Final = "asi.cchain-ipmnist.current-runner.v1"
ADAPTATION_ID: Final = "online-prior-example-ring-and-one-update-snapshot.v1"
DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

COMPARABILITY_GAPS: Final = (
    "IPMNIST is continual supervised learning, not the paper's primary continual-RL setting",
    "the official release contains RL implementations but no appendix supervised-learning source",
    "one-example online updates replace PPO or DoubleDQN minibatch updates",
    "a 32-example prior-stream ring replaces independently shuffled current-iteration batches",
    "the prior-update ring excludes the current update but repeated data identities can recur",
    "one-update parameter snapshots replace the official mutable Python model-history list",
    "a static 50-update loss window replaces 50 host-side outer-iteration means",
    "coefficient adaptation is evaluated per online update after ten active updates",
    "the current 784-300-150-10 ASI MLP replaces the paper's task-specific networks",
    "ASI's beta1=0 beta2=0.99 AdamW control replaces the official PPO Adam defaults",
    "the empirical NTK uses every example/class logit row rather than a policy-network output",
    "stored results are development diagnostics and cannot substantiate the paper's RL claims",
)

CCHAIN_PROTOCOL: Final[Mapping[str, object]] = MappingProxyType(
    {
        "paper_revision": PAPER_REVISION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_files": (
            "crl_gym_classic_control/control_train_aligned_c_chain.py",
            "crl_gym_classic_control/basic_utils.py",
        ),
        "adaptation_id": ADAPTATION_ID,
        "comparability_gaps": COMPARABILITY_GAPS,
        "reference_capacity": REFERENCE_CAPACITY,
        "coefficient_window": COEFFICIENT_WINDOW,
        "coefficient_delay": COEFFICIENT_DELAY,
        "snapshot_warmup_updates": SNAPSHOT_WARMUP_UPDATES,
        "ntk_diagnostic_examples": MAX_NTK_EXAMPLES,
        "ntk_energy_threshold": NTK_ENERGY_THRESHOLD,
        "persistent_bytes_scope": (
            "parameters, two Adam moment trees and their stored scalar constants, one "
            "parameter snapshot, the reference ring, loss windows, and state scalars"
        ),
        "ntk_bytes_scope": (
            "exact logical float32 envelope for the Jacobian pytree plus concatenated "
            "Jacobian; backend/compiler scratch is not qualified"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }
)

_ARMS: Final[Mapping[str, tuple[float, float, float, float]]] = MappingProxyType(
    {
        "cchain_mechanism_off": (0.0, 0.0, 0.0, 0.0),
        "cchain_full": (1.0, 1.0, 10_000.0, 0.0),
        "cchain_orthogonal_only": (1.0, 1.0, 10_000.0, 1.0),
        "cchain_projective_only": (1.0, 1.0, 10_000.0, 2.0),
    }
)
_HP_KEYS = frozenset(
    {
        *ADAMW_PROTOCOL_HYPERPARAMETERS,
        "reference_capacity",
        "coefficient_window",
        "coefficient_delay",
        "snapshot_warmup_updates",
        "churn_enabled",
        "adaptive_coefficient",
        "target_relative_loss_scale",
        "initial_coefficient",
        "gradient_component",
        "ntk_diagnostic_examples",
        "ntk_energy_threshold",
    }
)
_METRIC_KEYS = frozenset(
    {
        "mean_online_accuracy",
        "mean_loss",
        "mean_plasticity",
        "mean_probability_kl",
        "mean_logit_mse",
        "final_coefficient",
        "diagnostic_updates",
        "ntk_threshold_rank",
        "ntk_off_diagonal_abs_mean",
        "ntk_diagonal_mean",
        "ntk_examples",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "persistent_bytes",
        "ntk_jacobian_envelope_bytes",
        "environment_steps",
        "data_steps",
        "optimizer_updates",
        "task_model_queries",
        "churn_reference_updates",
        "churn_model_queries",
        "ntk_model_queries",
        "model_queries",
        "timing_seconds",
        "timing_is_telemetry_only",
    }
)
_INT32_MAX = (1 << 31) - 1
_MAX_BYTES = 256 * 1024 * 1024
_MAX_STRING_BYTES = 512
_MAX_GAPS = 32
_FLOAT32_MAX = 3.4028234663852886e38


def _object(value: object, keys: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{context} must be an exact object with keys {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _int(value: object, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{context} must be an exact integer in [{minimum}, {_INT32_MAX}]")
    return value


def _float(
    value: object,
    *,
    context: str,
    minimum: float = 0.0,
    maximum: float = _FLOAT32_MAX,
) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{context} must be an exact finite float in [{minimum}, {maximum}]"
        )
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or len(value) > _MAX_GAPS
        or any(
            type(item) is not str
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > _MAX_STRING_BYTES
            for item in value
        )
    ):
        raise ValueError(f"{context} must be a bounded list of non-empty UTF-8 strings")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates")
    return result


def _checked_product(*factors: int, context: str) -> int:
    product = 1
    for factor in factors:
        if factor < 1 or product > _INT32_MAX // factor:
            raise ValueError(f"derived {context} exceeds signed int32")
        product *= factor
    return product


def _expected_hyperparameters(arm: str) -> dict[str, float]:
    enabled, adaptive, target, component = _ARMS[arm]
    return {
        **ADAMW_PROTOCOL_HYPERPARAMETERS,
        "reference_capacity": float(REFERENCE_CAPACITY),
        "coefficient_window": float(COEFFICIENT_WINDOW),
        "coefficient_delay": float(COEFFICIENT_DELAY),
        "snapshot_warmup_updates": float(SNAPSHOT_WARMUP_UPDATES),
        "churn_enabled": enabled,
        "adaptive_coefficient": adaptive,
        "target_relative_loss_scale": target,
        "initial_coefficient": 1.0,
        "gradient_component": component,
        "ntk_diagnostic_examples": float(MAX_NTK_EXAMPLES),
        "ntk_energy_threshold": NTK_ENERGY_THRESHOLD,
    }


def validate_cchain_development_result(payload: object) -> dict[str, object]:
    """Normalize a result only after exact protocol and resource validation."""
    outer = _object(
        payload,
        frozenset(
            {
                "schema",
                "comparison_id",
                "paper_revision",
                "official_commit",
                "adaptation_id",
                "comparability_gaps",
                "arm",
                "seed",
                "development_seed_protocol",
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
    for name, expected in (
        ("schema", RESULT_SCHEMA),
        ("comparison_id", COMPARISON_ID),
        ("paper_revision", PAPER_REVISION),
        ("official_commit", OFFICIAL_COMMIT),
        ("adaptation_id", ADAPTATION_ID),
    ):
        if type(outer[name]) is not str or outer[name] != expected:
            raise ValueError(f"{name} does not match the frozen C-CHAIN adaptation")
    gaps = _strings(outer["comparability_gaps"], context="comparability_gaps")
    if gaps != COMPARABILITY_GAPS:
        raise ValueError("comparability_gaps must match the complete frozen gap list")
    arm = outer["arm"]
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError(f"arm must be one of {sorted(_ARMS)}")

    seed = _int(outer["seed"], context="seed")
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed must belong to the frozen development seed set")
    if outer["development_seed_protocol"] != list(DEVELOPMENT_SEEDS):
        raise ValueError("development_seed_protocol must match the frozen seed set")
    n_tasks = _int(outer["n_tasks"], context="n_tasks", minimum=1)
    task_length = _int(outer["task_length"], context="task_length", minimum=1)
    input_dim = _int(outer["input_dim"], context="input_dim", minimum=1)
    hidden1 = _int(outer["hidden1"], context="hidden1", minimum=1)
    hidden2 = _int(outer["hidden2"], context="hidden2", minimum=1)
    n_classes = _int(outer["n_classes"], context="n_classes", minimum=2)
    observations = _int(outer["observations"], context="observations", minimum=1)
    updates = _int(outer["updates"], context="updates", minimum=1)
    expected_observations = _checked_product(
        n_tasks, task_length, context="observation count"
    )
    if observations != expected_observations or updates != observations:
        raise ValueError("observations and updates must equal n_tasks * task_length")
    boundary = _strings(
        outer["allowed_boundary_information"], context="allowed_boundary_information"
    )
    task_info = _strings(
        outer["allowed_task_information"], context="allowed_task_information"
    )
    if boundary or task_info != ("current_example_label",):
        raise ValueError("task/boundary information does not match online IPMNIST")

    hp_object = _object(outer["hyperparameters"], _HP_KEYS, context="hyperparameters")
    hp = {
        name: _float(hp_object[name], context=f"hyperparameters.{name}")
        for name in _HP_KEYS
    }
    if hp != _expected_hyperparameters(arm):
        raise ValueError("hyperparameters do not match the registered C-CHAIN arm")

    metric_object = _object(outer["metrics"], _METRIC_KEYS, context="metrics")
    metrics = {
        name: _float(metric_object[name], context=f"metrics.{name}")
        for name in _METRIC_KEYS
    }
    if metrics["mean_online_accuracy"] > 1.0 or metrics["mean_plasticity"] > 1.0:
        raise ValueError("accuracy and plasticity metrics must lie in [0,1]")
    active_updates = max(observations - SNAPSHOT_WARMUP_UPDATES, 0)
    ntk_examples = min(observations, REFERENCE_CAPACITY, MAX_NTK_EXAMPLES)
    if metrics["diagnostic_updates"] != float(active_updates):
        raise ValueError("diagnostic_updates does not match the executed warmup")
    if metrics["ntk_examples"] != float(ntk_examples):
        raise ValueError("ntk_examples does not match the bounded final diagnostic")
    max_rank = ntk_examples * n_classes
    rank = metrics["ntk_threshold_rank"]
    if not rank.is_integer() or not 0.0 <= rank <= float(max_rank):
        raise ValueError("ntk_threshold_rank is outside the exact diagnostic domain")
    if arm == "cchain_mechanism_off" and metrics["final_coefficient"] != 1.0:
        raise ValueError("mechanism-off coefficient must remain exactly one")
    if arm != "cchain_mechanism_off" and metrics["final_coefficient"] < 1.0:
        raise ValueError("adaptive C-CHAIN coefficient must remain at least one")
    if (
        arm != "cchain_mechanism_off"
        and active_updates < COEFFICIENT_DELAY
        and metrics["final_coefficient"] != 1.0
    ):
        raise ValueError("C-CHAIN coefficient cannot adapt before its delay")

    resource_object = _object(outer["resources"], _RESOURCE_KEYS, context="resources")
    integer_resources = {
        name: _int(resource_object[name], context=f"resources.{name}")
        for name in _RESOURCE_KEYS
        if name not in {"timing_seconds", "timing_is_telemetry_only"}
    }
    timing = _float(
        resource_object["timing_seconds"],
        context="resources.timing_seconds",
        maximum=604_800.0,
    )
    if resource_object["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain true")
    if integer_resources["environment_steps"] != 0:
        raise ValueError("a supervised C-CHAIN run cannot report environment steps")
    if (
        integer_resources["data_steps"] != observations
        or integer_resources["optimizer_updates"] != updates
        or integer_resources["task_model_queries"] != 2 * observations
        or integer_resources["churn_reference_updates"] != active_updates
        or integer_resources["churn_model_queries"] != 2 * active_updates
        or integer_resources["ntk_model_queries"] != ntk_examples
    ):
        raise ValueError("resource counters do not match the C-CHAIN execution semantics")
    expected_queries = 2 * observations + 2 * active_updates + ntk_examples
    if integer_resources["model_queries"] != expected_queries:
        raise ValueError("model_queries does not equal the complete derived query count")

    products = (
        _checked_product(input_dim, hidden1, context="w1 scalars"),
        _checked_product(hidden1, hidden2, context="w2 scalars"),
        _checked_product(hidden2, n_classes, context="w3 scalars"),
    )
    parameter_count = sum(products) + hidden1 + hidden2 + n_classes
    if parameter_count > _INT32_MAX:
        raise ValueError("parameter count exceeds signed int32")
    # Parameters + two Adam moment trees + the snapshot tree + six Adam
    # states' five scalar fields + reference ring + two windows + nine state scalars.
    persistent_scalars = (
        4 * parameter_count
        + 5 * len(_PARAMETER_LEAVES)
        + REFERENCE_CAPACITY * input_dim
        + 2 * COEFFICIENT_WINDOW
        + 9
    )
    if persistent_scalars > _INT32_MAX // 4:
        raise ValueError("persistent scalar count exceeds the byte domain")
    expected_persistent_bytes = 4 * persistent_scalars
    rows = ntk_examples * n_classes
    expected_ntk_envelope_bytes = 2 * rows * parameter_count * 4
    if expected_persistent_bytes > _MAX_BYTES or expected_ntk_envelope_bytes > _MAX_BYTES:
        raise ValueError("derived C-CHAIN resources exceed the 256 MiB bounds")
    if integer_resources["persistent_bytes"] != expected_persistent_bytes:
        raise ValueError("persistent_bytes does not cover the complete resident state")
    if integer_resources["ntk_jacobian_envelope_bytes"] != expected_ntk_envelope_bytes:
        raise ValueError(
            "ntk_jacobian_envelope_bytes does not match the full-logit Jacobian envelope"
        )

    if type(outer["outcome"]) is not str or outer["outcome"] not in {
        "supported",
        "rejected",
        "inconclusive",
    }:
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if outer["outcome_retained"] is not True:
        raise ValueError("outcome_retained must permanently remain true")
    if outer["development_only"] is not True:
        raise ValueError("development_only must permanently remain true")
    if outer["scientific_promotion_allowed"] is not False:
        raise ValueError("scientific_promotion_allowed must permanently remain false")

    return {
        **dict(outer),
        "comparability_gaps": list(gaps),
        "seed": seed,
        "n_tasks": n_tasks,
        "task_length": task_length,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "n_classes": n_classes,
        "observations": observations,
        "updates": updates,
        "allowed_boundary_information": list(boundary),
        "allowed_task_information": list(task_info),
        "hyperparameters": hp,
        "metrics": metrics,
        "resources": {
            **integer_resources,
            "timing_seconds": timing,
            "timing_is_telemetry_only": True,
        },
    }


_PARAMETER_LEAVES: Final = ("w1", "b1", "w2", "b2", "w3", "b3")


def validate_matched_cchain_development_results(
    payloads: object,
) -> tuple[dict[str, object], ...]:
    """Require all four causal arms and exact matched workload axes."""
    if type(payloads) not in (list, tuple):
        raise ValueError("the matched comparison must be an exact list or tuple")
    trusted = cast(list[object] | tuple[object, ...], payloads)
    if len(trusted) != len(_ARMS):
        raise ValueError("the matched comparison must contain exactly four results")
    results = tuple(validate_cchain_development_result(item) for item in trusted)
    if {result["arm"] for result in results} != set(_ARMS):
        raise ValueError("the matched comparison must contain every C-CHAIN arm once")
    axes = (
        "comparison_id",
        "seed",
        "development_seed_protocol",
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
        "adaptation_id",
        "comparability_gaps",
    )
    first = results[0]
    for result in results[1:]:
        if any(result[name] != first[name] for name in axes):
            raise ValueError("all required C-CHAIN comparison axes must match")
    return results
