"""Strict, permanently nonpromoting reports for gradual-input IPMNIST pairs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_gradual import (
    GRADUAL_IPMNIST_PROTOCOL,
    GradualInputPairResult,
    run_gradual_input_pair,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    validated_ipmnist_data,
)

RESULT_SCHEMA = "asi.ipmnist.gradual-input.development-result.v1"
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_TEXT_BYTES = 4096
_MAX_JSON_NODES = 20_000
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_ARMS = ("abrupt", "input_interpolation")


@dataclass(frozen=True)
class GradualInputDevelopmentPlan:
    """Exact paired plan for the issue-specific input-interpolation adapter."""

    seeds: tuple[int, ...]
    config: IPMNISTConfig
    transition_steps: int
    learner_name: str = "adamw_control"

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 128:
            raise ValueError("seeds must be a bounded exact tuple")
        seeds = tuple(require_jax_seed(seed, name="seed") for seed in self.seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if type(self.config) is not IPMNISTConfig:
            raise ValueError("config must be an exact IPMNISTConfig")
        config = IPMNISTConfig(**self.config.to_config())
        if type(self.transition_steps) is not int or not (
            1 <= self.transition_steps < config.task_length
        ):
            raise ValueError("transition_steps must be in [1, task_length)")
        if type(self.learner_name) is not str or self.learner_name != "adamw_control":
            raise ValueError("learner_name must be adamw_control")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "config", config)


FROZEN_GRADUAL_INPUT_PLAN = GradualInputDevelopmentPlan(
    seeds=(1_569_001, 1_569_002, 1_569_003),
    config=IPMNISTConfig(n_tasks=10, task_length=128),
    transition_steps=32,
)


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("result must be finite canonical JSON") from error
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("result exceeds the byte ceiling")
    return encoded


def _json_preflight(value: object) -> None:
    nodes = 0
    text_bytes = 0
    stack = [value]
    while stack:
        current = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("result exceeds the exact JSON node ceiling")
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 64:
                raise ValueError("result object exceeds the field ceiling")
            keys = list(mapping.keys())
            if any(type(key) is not str for key in keys):
                raise ValueError("result must be an exact JSON tree with exact string keys")
            stack.extend(mapping.values())
            stack.extend(keys)
        elif type(current) is list:
            values = cast(list[object], current)
            if len(values) > 4096:
                raise ValueError("result list exceeds the item ceiling")
            stack.extend(values)
        elif type(current) is str:
            try:
                text_bytes += len(current.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("result text must be UTF-8") from error
            if text_bytes > 1_000_000 or len(current.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise ValueError("result text exceeds the byte ceiling")
        elif type(current) is int:
            if not -_INT64_MAX <= current <= _INT64_MAX:
                raise ValueError("result integer exceeds signed int64")
        elif type(current) is float:
            if not math.isfinite(current):
                raise ValueError("result float must be finite")
        elif type(current) not in (bool, type(None)):
            raise ValueError("result must be an exact JSON tree")


def _fields(value: object, expected: tuple[str, ...], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact object")
    result = cast(dict[str, object], value)
    if len(result) != len(expected) or set(result) != set(expected):
        raise ValueError(f"{name} fields do not match the exact schema")
    return result


def _same(actual: object, expected: object) -> bool:
    """Return type-exact equality for already-preflighted JSON subtrees."""
    return _canonical_bytes(actual) == _canonical_bytes(expected)


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path("alberta_framework/_seed_validation.py"),
        Path("alberta_framework/benchmarks/ipmnist_gradual.py"),
        Path("alberta_framework/benchmarks/upgd_ipmnist.py"),
        Path("alberta_framework/core/baseline_optimizers.py"),
        Path("alberta_framework/core/update_safety.py"),
        Path("alberta_framework/evaluation/gradual_ipmnist_nonpromoting.py"),
    )
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{name} must be bounded non-empty text")
    return value


def _runtime_identity() -> dict[str, object]:
    devices = sorted(
        [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": _text(str(device.platform), name="device platform"),
                "kind": _text(str(device.device_kind), name="device kind"),
            }
            for device in jax.devices()
        ],
        key=lambda item: (
            item["process_index"],
            item["id"],
            item["platform"],
            item["kind"],
        ),
    )
    if not 1 <= len(devices) <= 128:
        raise ValueError("runtime device inventory exceeds its bound")
    environment = {}
    for name in (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    ):
        raw = os.environ.get(name)
        environment[name] = None if raw is None else _text(raw, name=name)
    return {
        "schema": "asi.ipmnist.gradual-input.runtime.v1",
        "python_implementation": _text(platform.python_implementation(), name="python"),
        "python_version": list(sys.version_info[:3]),
        "byteorder": _text(sys.byteorder, name="byteorder"),
        "platform": _text(sys.platform, name="platform"),
        "machine": _text(platform.machine(), name="machine"),
        "packages": {
            "jax": _text(jax.__version__, name="jax"),
            "jaxlib": _text(importlib.metadata.version("jaxlib"), name="jaxlib"),
            "numpy": _text(np.__version__, name="numpy"),
        },
        "backend": _text(jax.default_backend(), name="backend"),
        "devices": devices,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": _text(str(jax.config.jax_default_prng_impl), name="default PRNG"),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": _text(
            str(jax.config.jax_numpy_dtype_promotion), name="dtype promotion"
        ),
        "jax_numpy_rank_promotion": _text(
            str(jax.config.jax_numpy_rank_promotion), name="rank promotion"
        ),
        "environment": environment,
    }


def _plan_payload(plan: GradualInputDevelopmentPlan) -> dict[str, object]:
    checked = GradualInputDevelopmentPlan(
        seeds=plan.seeds,
        config=plan.config,
        transition_steps=plan.transition_steps,
        learner_name=plan.learner_name,
    )
    horizon = checked.config.n_steps
    return {
        "seeds": list(checked.seeds),
        "config": checked.config.to_config(),
        "transition_steps": checked.transition_steps,
        "learner_name": checked.learner_name,
        "arms": list(_ARMS),
        "expected_observations": horizon,
        "expected_updates": horizon,
        "expected_data_steps": horizon,
        "expected_environment_steps": 0,
        "expected_model_queries": 2 * horizon,
        "learner_observes_transition_alpha": False,
        "learner_observes_task_boundary": False,
        "outcome_rule": "descriptive_sign_of_paired_mean_no_claim_threshold",
    }


def _dataset_identity(data_x: object, data_y: object, config: IPMNISTConfig) -> dict[str, object]:
    for name, value in (("data_x", data_x), ("data_y", data_y)):
        actual_type = type(value)
        if not (actual_type is np.ndarray or issubclass(actual_type, jax.Array)):
            raise ValueError(f"{name} must be an exact NumPy or JAX array")
    raw_x = cast(np.ndarray | jax.Array, data_x)
    raw_y = cast(np.ndarray | jax.Array, data_y)
    if len(raw_x.shape) != 2 or len(raw_y.shape) != 1 or raw_x.shape[0] != raw_y.shape[0]:
        raise ValueError("dataset metadata is invalid")
    if (math.prod(raw_x.shape) + math.prod(raw_y.shape)) * 4 > _MAX_RESOURCE_BYTES:
        raise ValueError("materialized dataset exceeds 256 MiB")
    x, y = validated_ipmnist_data(
        raw_x,
        raw_y,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.task_length,
    )
    digest = hashlib.sha256(b"asi.ipmnist.gradual-input.dataset.v1\0")
    digest.update(x.dtype.str.encode("ascii"))
    digest.update(str(x.shape).encode("ascii"))
    digest.update(x.tobytes(order="C"))
    digest.update(y.dtype.str.encode("ascii"))
    digest.update(str(y.shape).encode("ascii"))
    digest.update(y.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "rows": int(x.shape[0]), "columns": int(x.shape[1])}


def _validated_run(
    run: object, plan: GradualInputDevelopmentPlan, expected_seed: int
) -> GradualInputPairResult:
    if type(run) is not GradualInputPairResult:
        raise ValueError("run must be an exact GradualInputPairResult")
    result = run
    if (
        type(result.arm_names) is not tuple
        or any(type(arm) is not str for arm in result.arm_names)
        or result.arm_names != _ARMS
        or type(result.learner_name) is not str
        or result.learner_name != plan.learner_name
    ):
        raise ValueError("run arm or learner identity does not match the plan")
    if type(result.config) is not IPMNISTConfig:
        raise ValueError("run config does not match the plan")
    try:
        checked_result_config = IPMNISTConfig(**result.config.to_config())
    except (TypeError, ValueError) as error:
        raise ValueError("run config does not match the plan") from error
    if checked_result_config.to_config() != plan.config.to_config():
        raise ValueError("run config does not match the plan")
    if (
        type(result.seed) is not int
        or result.seed != expected_seed
        or type(result.transition_steps) is not int
        or result.transition_steps != plan.transition_steps
    ):
        raise ValueError("run seed or transition width does not match the plan")
    horizon = plan.config.n_steps
    counters = (
        result.observations,
        result.updates,
        result.data_steps,
        result.environment_steps,
        result.model_queries,
    )
    if any(type(value) is not int for value in counters) or counters != (
        horizon,
        horizon,
        horizon,
        0,
        2 * horizon,
    ):
        raise ValueError("run counter does not match the frozen plan")
    if (
        type(result.correct_counts) is not np.ndarray
        or result.correct_counts.dtype != np.int32
        or result.correct_counts.shape != (2, plan.config.n_tasks)
        or np.any(result.correct_counts < 0)
        or np.any(result.correct_counts > plan.config.task_length)
    ):
        raise ValueError("run correctness numerators are invalid")
    if (
        type(result.loss_sums) is not np.ndarray
        or result.loss_sums.dtype != np.float64
        or result.loss_sums.shape != (2, plan.config.n_tasks)
        or not np.all(np.isfinite(result.loss_sums))
        or np.any(result.loss_sums < 0.0)
    ):
        raise ValueError("run loss sums are invalid")
    for name, array, dtype, maximum in (
        (
            "persistent bytes",
            result.persistent_numeric_bytes,
            np.dtype(np.int64),
            _MAX_RESOURCE_BYTES,
        ),
        ("timing", result.timing_ns, np.dtype(np.int64), _INT64_MAX),
    ):
        if (
            type(array) is not np.ndarray
            or array.dtype != dtype
            or array.shape != (2,)
            or np.any(array < 0)
            or np.any(array > maximum)
        ):
            raise ValueError(f"run {name} receipt is invalid")
    for identity in (result.schedule_sha256, result.example_order_sha256):
        if (
            type(identity) is not str
            or len(identity) != 71
            or not identity.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in identity[7:])
        ):
            raise ValueError("run schedule identity is invalid")
    return result


def _outcome(delta: float) -> str:
    return "improved" if delta > 0.0 else "worse" if delta < 0.0 else "tied"


def _schedule_identities(seed: int, config: IPMNISTConfig, n_train: int) -> tuple[str, str]:
    _, key_schedule, _ = jr.split(jr.key(np.uint32(seed)), 3)
    schedule = build_schedule(key_schedule, config, n_train)

    def digest(domain: bytes, value: jax.Array) -> str:
        raw = np.asarray(jax.device_get(value))
        value_digest = hashlib.sha256()
        value_digest.update(domain)
        value_digest.update(raw.dtype.str.encode("ascii"))
        value_digest.update(str(raw.shape).encode("ascii"))
        value_digest.update(raw.tobytes(order="C"))
        return f"sha256:{value_digest.hexdigest()}"

    return (
        digest(b"asi.ipmnist.gradual.permutations.v1\0", schedule.permutations),
        digest(b"asi.ipmnist.gradual.example-order.v1\0", schedule.example_indices),
    )


def build_gradual_input_development_report(
    plan: GradualInputDevelopmentPlan,
    runs: tuple[GradualInputPairResult, ...],
    data_x: object,
    data_y: object,
) -> dict[str, object]:
    """Build one canonical issue-specific paired report from direct run receipts."""
    if type(plan) is not GradualInputDevelopmentPlan:
        raise ValueError("plan must be an exact GradualInputDevelopmentPlan")
    checked_plan = GradualInputDevelopmentPlan(
        seeds=plan.seeds,
        config=plan.config,
        transition_steps=plan.transition_steps,
        learner_name=plan.learner_name,
    )
    if type(runs) is not tuple or len(runs) != len(checked_plan.seeds):
        raise ValueError("runs must contain one exact receipt per planned seed")
    records = []
    deltas = []
    arm_accuracies: list[list[float]] = [[], []]
    arm_persistent_bytes: list[list[int]] = [[], []]
    arm_timing_ns: list[list[int]] = [[], []]
    horizon = checked_plan.config.n_steps
    for seed, raw_run in zip(checked_plan.seeds, runs, strict=True):
        run = _validated_run(raw_run, checked_plan, seed)
        arms = []
        accuracies = []
        for index, arm in enumerate(_ARMS):
            correct = int(run.correct_counts[index].sum())
            loss_sum = float(run.loss_sums[index].sum())
            accuracy = correct / horizon
            accuracies.append(accuracy)
            arm_accuracies[index].append(accuracy)
            arm_persistent_bytes[index].append(int(run.persistent_numeric_bytes[index]))
            arm_timing_ns[index].append(int(run.timing_ns[index]))
            arms.append(
                {
                    "arm": arm,
                    "metrics": {
                        "correct": correct,
                        "online_accuracy": accuracy,
                        "loss_sum": loss_sum,
                        "mean_loss": loss_sum / horizon,
                    },
                    "resources": {
                        "observations": run.observations,
                        "updates": run.updates,
                        "data_steps": run.data_steps,
                        "environment_steps": run.environment_steps,
                        "model_queries": run.model_queries,
                        "persistent_numeric_bytes": int(run.persistent_numeric_bytes[index]),
                        "timing_ns": int(run.timing_ns[index]),
                        "timing_qualified": False,
                    },
                }
            )
        delta = accuracies[1] - accuracies[0]
        deltas.append(delta)
        records.append(
            {
                "seed": seed,
                "schedule_sha256": run.schedule_sha256,
                "example_order_sha256": run.example_order_sha256,
                "arms": arms,
                "comparison": {
                    "metric": "online_accuracy",
                    "higher_is_better": True,
                    "candidate_minus_control": delta,
                    "outcome": _outcome(delta),
                },
            }
        )
    mean_delta = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    stderr = 0.0 if len(deltas) == 1 else float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
    arm_summaries = []
    for index, arm in enumerate(_ARMS):
        values = np.asarray(arm_accuracies[index], dtype=np.float64)
        arm_summaries.append(
            {
                "arm": arm,
                "online_accuracy_mean": float(np.mean(values)),
                "online_accuracy_stderr": (
                    0.0
                    if len(values) == 1
                    else float(np.std(values, ddof=1) / math.sqrt(len(values)))
                ),
                "total_observations": horizon * len(values),
                "total_updates": horizon * len(values),
                "total_data_steps": horizon * len(values),
                "total_environment_steps": 0,
                "total_model_queries": 2 * horizon * len(values),
                "persistent_numeric_bytes_max": max(arm_persistent_bytes[index]),
                "timing_ns_total_telemetry": sum(arm_timing_ns[index]),
                "timing_qualified": False,
            }
        )
    sources = _source_identity()
    runtime = _runtime_identity()
    plan_payload = _plan_payload(checked_plan)
    identity_unsigned = {"source_sha256": sources, "runtime": runtime, "plan": plan_payload}
    report: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "identity": {
            "source_sha256": sources,
            "runtime": runtime,
            "plan_sha256": hashlib.sha256(_canonical_bytes(identity_unsigned)).hexdigest(),
            "consistency_not_attestation": True,
        },
        "protocol": {
            "schema": GRADUAL_IPMNIST_PROTOCOL["schema"],
            "paper_revision": GRADUAL_IPMNIST_PROTOCOL["paper_revision"],
            "scope": "input_interpolation_vs_abrupt_only",
            "aggregate_working_set_bytes_claimed": False,
            "timing_is_telemetry_only": True,
        },
        "plan": plan_payload,
        "dataset": _dataset_identity(data_x, data_y, checked_plan.config),
        "records": records,
        "aggregate": {
            "paired_accuracy_delta_mean": mean_delta,
            "paired_accuracy_delta_stderr": stderr,
            "n": len(deltas),
            "outcome": _outcome(mean_delta),
            "arms": arm_summaries,
        },
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
    }
    report["result_sha256"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    validate_gradual_input_development_report(report, data_x, data_y)
    return report


def run_frozen_gradual_input_development(data_x: object, data_y: object) -> dict[str, object]:
    """Execute all frozen seeds and return one strictly validated report."""
    plan = FROZEN_GRADUAL_INPUT_PLAN
    # Bind and preflight the complete supplied materialization before the first
    # seed; every per-seed runner independently repeats its allocation gate.
    _dataset_identity(data_x, data_y, plan.config)
    runs = tuple(
        run_gradual_input_pair(
            data_x,
            data_y,
            learner_name=plan.learner_name,
            seed=seed,
            config=plan.config,
            transition_steps=plan.transition_steps,
        )
        for seed in plan.seeds
    )
    return build_gradual_input_development_report(plan, runs, data_x, data_y)


def validate_gradual_input_development_report(
    report: object, data_x: object, data_y: object
) -> None:
    """Validate exact structure, current identities, and all derived report values."""
    _json_preflight(report)
    root = _fields(
        report,
        (
            "schema",
            "identity",
            "protocol",
            "plan",
            "dataset",
            "records",
            "aggregate",
            "policy",
            "result_sha256",
        ),
        name="report",
    )
    if root["schema"] != RESULT_SCHEMA:
        raise ValueError("report schema is invalid")
    plan_raw = _fields(
        root["plan"],
        (
            "seeds",
            "config",
            "transition_steps",
            "learner_name",
            "arms",
            "expected_observations",
            "expected_updates",
            "expected_data_steps",
            "expected_environment_steps",
            "expected_model_queries",
            "learner_observes_transition_alpha",
            "learner_observes_task_boundary",
            "outcome_rule",
        ),
        name="plan",
    )
    config_raw = _fields(
        plan_raw["config"],
        ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"),
        name="plan.config",
    )
    seeds_raw = plan_raw["seeds"]
    if type(seeds_raw) is not list:
        raise ValueError("plan seeds must be a list")
    try:
        plan = GradualInputDevelopmentPlan(
            seeds=tuple(seeds_raw),
            config=IPMNISTConfig(
                n_tasks=cast(int, config_raw["n_tasks"]),
                task_length=cast(int, config_raw["task_length"]),
                input_dim=cast(int, config_raw["input_dim"]),
                hidden1=cast(int, config_raw["hidden1"]),
                hidden2=cast(int, config_raw["hidden2"]),
                n_classes=cast(int, config_raw["n_classes"]),
            ),
            transition_steps=cast(int, plan_raw["transition_steps"]),
            learner_name=cast(str, plan_raw["learner_name"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("plan is invalid") from error
    if not _same(plan_raw, _plan_payload(plan)):
        raise ValueError("plan is not canonical")
    identity = _fields(
        root["identity"],
        ("source_sha256", "runtime", "plan_sha256", "consistency_not_attestation"),
        name="identity",
    )
    expected_identity_unsigned = {
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "plan": _plan_payload(plan),
    }
    expected_plan_sha = hashlib.sha256(_canonical_bytes(expected_identity_unsigned)).hexdigest()
    if not _same(
        identity,
        {
            "source_sha256": expected_identity_unsigned["source_sha256"],
            "runtime": expected_identity_unsigned["runtime"],
            "plan_sha256": expected_plan_sha,
            "consistency_not_attestation": True,
        },
    ):
        raise ValueError("identity does not match current source/runtime/plan")
    expected_dataset = _dataset_identity(data_x, data_y, plan.config)
    if not _same(root["dataset"], expected_dataset):
        raise ValueError("dataset identity does not match supplied materialization")
    records = root["records"]
    if type(records) is not list or len(records) != len(plan.seeds):
        raise ValueError("records must contain one canonical seed record")
    deltas = []
    aggregate_accuracies: list[list[float]] = [[], []]
    aggregate_persistent_bytes: list[list[int]] = [[], []]
    aggregate_timing_ns: list[list[int]] = [[], []]
    for index, (record_raw, seed) in enumerate(zip(records, plan.seeds, strict=True)):
        record = _fields(
            record_raw,
            ("seed", "schedule_sha256", "example_order_sha256", "arms", "comparison"),
            name=f"records[{index}]",
        )
        if record["seed"] != seed:
            raise ValueError("record ordering does not match the plan")
        expected_schedule = _schedule_identities(
            seed, plan.config, cast(int, expected_dataset["rows"])
        )
        for identity_name in ("schedule_sha256", "example_order_sha256"):
            schedule_identity = record[identity_name]
            if (
                type(schedule_identity) is not str
                or len(schedule_identity) != 71
                or not schedule_identity.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in schedule_identity[7:])
            ):
                raise ValueError("record schedule identity is invalid")
        if (
            record["schedule_sha256"],
            record["example_order_sha256"],
        ) != expected_schedule:
            raise ValueError("record schedule identity does not derive from seed and dataset")
        arms = record["arms"]
        if type(arms) is not list or len(arms) != 2:
            raise ValueError("record arms are invalid")
        accuracies = []
        arm_resources: list[dict[str, object]] = []
        for arm_index, (arm_raw, arm_name) in enumerate(zip(arms, _ARMS, strict=True)):
            arm = _fields(arm_raw, ("arm", "metrics", "resources"), name="arm")
            if arm["arm"] != arm_name:
                raise ValueError("arm ordering is invalid")
            metrics = _fields(
                arm["metrics"],
                ("correct", "online_accuracy", "loss_sum", "mean_loss"),
                name="metrics",
            )
            correct = metrics["correct"]
            loss_sum = metrics["loss_sum"]
            if (
                type(correct) is not int
                or not 0 <= correct <= plan.config.n_steps
                or type(loss_sum) is not float
                or loss_sum < 0.0
            ):
                raise ValueError("metric numerators are invalid")
            expected_accuracy = correct / plan.config.n_steps
            if (
                type(metrics["online_accuracy"]) is not float
                or type(metrics["mean_loss"]) is not float
                or metrics["online_accuracy"] != expected_accuracy
                or metrics["mean_loss"] != loss_sum / plan.config.n_steps
            ):
                raise ValueError("metrics are not derived from exact numerators")
            accuracies.append(expected_accuracy)
            aggregate_accuracies[arm_index].append(expected_accuracy)
            resources = _fields(
                arm["resources"],
                (
                    "observations",
                    "updates",
                    "data_steps",
                    "environment_steps",
                    "model_queries",
                    "persistent_numeric_bytes",
                    "timing_ns",
                    "timing_qualified",
                ),
                name="resources",
            )
            expected_counters = (
                plan.config.n_steps,
                plan.config.n_steps,
                plan.config.n_steps,
                0,
                2 * plan.config.n_steps,
            )
            counter_names = (
                "observations",
                "updates",
                "data_steps",
                "environment_steps",
                "model_queries",
            )
            if (
                any(type(resources[name]) is not int for name in counter_names)
                or tuple(resources[name] for name in counter_names) != expected_counters
            ):
                raise ValueError("resource counters do not match the plan")
            if (
                type(resources["persistent_numeric_bytes"]) is not int
                or not 0 < resources["persistent_numeric_bytes"] <= _MAX_RESOURCE_BYTES
                or type(resources["timing_ns"]) is not int
                or not 0 <= resources["timing_ns"] <= _INT64_MAX
                or resources["timing_qualified"] is not False
            ):
                raise ValueError("resource receipt is invalid")
            arm_resources.append(resources)
            aggregate_persistent_bytes[arm_index].append(
                resources["persistent_numeric_bytes"]
            )
            aggregate_timing_ns[arm_index].append(resources["timing_ns"])
        for resource_name in (
            "observations",
            "updates",
            "data_steps",
            "environment_steps",
            "model_queries",
            "persistent_numeric_bytes",
        ):
            if arm_resources[0][resource_name] != arm_resources[1][resource_name]:
                raise ValueError("matched arm resource receipts are not paired")
        delta = accuracies[1] - accuracies[0]
        deltas.append(delta)
        expected_comparison = {
            "metric": "online_accuracy",
            "higher_is_better": True,
            "candidate_minus_control": delta,
            "outcome": _outcome(delta),
        }
        if not _same(record["comparison"], expected_comparison):
            raise ValueError("comparison is not derived from arm metrics")
    mean_delta = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    stderr = 0.0 if len(deltas) == 1 else float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
    expected_arm_summaries = []
    for arm_index, arm_name in enumerate(_ARMS):
        values = np.asarray(aggregate_accuracies[arm_index], dtype=np.float64)
        expected_arm_summaries.append(
            {
                "arm": arm_name,
                "online_accuracy_mean": float(np.mean(values)),
                "online_accuracy_stderr": (
                    0.0
                    if len(values) == 1
                    else float(np.std(values, ddof=1) / math.sqrt(len(values)))
                ),
                "total_observations": plan.config.n_steps * len(values),
                "total_updates": plan.config.n_steps * len(values),
                "total_data_steps": plan.config.n_steps * len(values),
                "total_environment_steps": 0,
                "total_model_queries": 2 * plan.config.n_steps * len(values),
                "persistent_numeric_bytes_max": max(aggregate_persistent_bytes[arm_index]),
                "timing_ns_total_telemetry": sum(aggregate_timing_ns[arm_index]),
                "timing_qualified": False,
            }
        )
    expected_aggregate = {
        "paired_accuracy_delta_mean": mean_delta,
        "paired_accuracy_delta_stderr": stderr,
        "n": len(deltas),
        "outcome": _outcome(mean_delta),
        "arms": expected_arm_summaries,
    }
    if not _same(root["aggregate"], expected_aggregate):
        raise ValueError("aggregate is not derived from paired records")
    expected_policy = {
        "status": "development-only-nonpromoting",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
    }
    if not _same(root["policy"], expected_policy):
        raise ValueError("policy is not permanently nonpromoting")
    protocol = _fields(
        root["protocol"],
        (
            "schema",
            "paper_revision",
            "scope",
            "aggregate_working_set_bytes_claimed",
            "timing_is_telemetry_only",
        ),
        name="protocol",
    )
    expected_protocol = {
        "schema": GRADUAL_IPMNIST_PROTOCOL["schema"],
        "paper_revision": GRADUAL_IPMNIST_PROTOCOL["paper_revision"],
        "scope": "input_interpolation_vs_abrupt_only",
        "aggregate_working_set_bytes_claimed": False,
        "timing_is_telemetry_only": True,
    }
    if not _same(protocol, expected_protocol):
        raise ValueError("protocol does not match the frozen input-only scope")
    digest = root["result_sha256"]
    if type(digest) is not str or len(digest) != 64:
        raise ValueError("result digest is invalid")
    unsigned = dict(root)
    del unsigned["result_sha256"]
    if digest != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest():
        raise ValueError("result digest does not bind canonical bytes")


def canonical_gradual_input_development_bytes(
    report: object, data_x: object, data_y: object
) -> bytes:
    """Validate and canonically encode one finite gradual-input report."""
    validate_gradual_input_development_report(report, data_x, data_y)
    return _canonical_bytes(report)


def retain_frozen_gradual_input_development_report(
    report: object,
    data_x: object,
    data_y: object,
    *,
    repository_root: Path,
) -> Path:
    """Publish one immutable frozen-plan result through no-follow directory FDs."""
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    encoded = canonical_gradual_input_development_bytes(report, data_x, data_y)
    root = cast(dict[str, object], report)
    if not _same(root["plan"], _plan_payload(FROZEN_GRADUAL_INPUT_PLAN)):
        raise ValueError("only the exact frozen gradual-input plan may be retained")
    digest = cast(str, root["result_sha256"])
    segments = ("outputs", "ipmnist_gradual", "development.v1")
    directory = repository_root.joinpath(*segments)
    destination = directory / f"result.{digest}.json"
    temporary_name = f".result.{digest}.tmp"
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(repository_root, directory_flags)
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(segment, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o444,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        read_descriptor = os.open(
            destination.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(read_descriptor, "rb", closefd=False) as stream:
                retained = stream.read(_MAX_RESULT_BYTES + 1)
        finally:
            os.close(read_descriptor)
        if retained != encoded:
            raise RuntimeError("retained gradual-input report bytes changed during publication")
        loaded = json.loads(retained)
        if canonical_gradual_input_development_bytes(loaded, data_x, data_y) != encoded:
            raise RuntimeError("retained gradual-input report failed strict reload validation")
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination
