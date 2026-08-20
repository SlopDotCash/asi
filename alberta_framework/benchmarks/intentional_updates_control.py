"""Prospective Intentional Updates TD/control matched-development lane.

This is a linear end-to-end consumer of the optimizer published at the pinned
author revision. It is development infrastructure, not a reproduction result
or scientific evidence. Campaign execution remains closed until a separate
reviewed authorization change.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.adamo_diagnostic import _load_dataset
from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    _screening_source_provenance,
    intentional_updates_development_record,
    run_screening_config,
    screening_spec,
    validate_intentional_updates_development_record,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

SCHEMA: Final[str] = "asi.intentional-updates.control-shard.v1"
REPORT_SCHEMA: Final[str] = "asi.intentional-updates.matched-development-report.v1"
PLAN_ID: Final[str] = "issue-1561-intentional-updates-matched-development-v1"
PAPER_REVISION: Final[str] = "arXiv:2604.19033v1"
OFFICIAL_CODE_REVISION: Final[str] = (
    "sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685"
)
QUARANTINED_SEEDS: Final[tuple[int, ...]] = (
    31_561_001,
    31_561_002,
    31_561_003,
    31_561_004,
)
CAMPAIGN_SEEDS: Final[tuple[int, ...]] = (
    41_562_001,
    41_562_002,
    41_562_003,
    41_562_004,
)
TEST_ONLY_SEEDS: Final[tuple[int, ...]] = (101, 102, 103, 104)
SEEDS: tuple[int, ...] = CAMPAIGN_SEEDS
CONTROL_ARMS: Final[tuple[str, ...]] = (
    "fixed_td0",
    "intentional_td0",
    "fixed_trace",
    "intentional_trace",
    "fixed_q_lambda",
    "intentional_q_lambda",
)
_OFF_ALIASES: Final[dict[str, str]] = {
    "intentional_td0_off": "fixed_td0",
    "intentional_trace_off": "fixed_trace",
    "intentional_q_lambda_off": "fixed_q_lambda",
}
_EXECUTION_AUTHORIZED: Final[bool] = False
_REVIEWED_EXECUTION_TRANSITION: Final[bool] = False
_EXECUTION_CAPABILITY: Final[object] = object()
_TEST_EXECUTION_CAPABILITY: Final[object] = object()
_MAX_HORIZON: Final[int] = 10_000
_MAX_JSON_NODES: Final[int] = 200_000
_MAX_STRING_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_TIMING_NS: Final[int] = 7 * 24 * 60 * 60 * 1_000_000_000
_MAX_REPORT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_RUNTIME_DEVICES: Final[int] = 64
_AGENT_RNG_IMPL: Final[str] = "threefry2x32"
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = (
    _REPO_ROOT / "outputs/intentional_updates_matched_development/report.v1.json"
)
BONFERRONI_T_DF3: Final[float] = 5.391949071934058
DATASET_X_SHA256: Final[str] = (
    "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
)
DATASET_Y_SHA256: Final[str] = (
    "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
)
SUPERVISED_CONFIG: Final[IPMNISTConfig] = IPMNISTConfig(
    n_tasks=8,
    task_length=64,
    input_dim=784,
    hidden1=300,
    hidden2=150,
    n_classes=10,
)


@dataclasses.dataclass(frozen=True, slots=True)
class _Reservation:
    directory_fd: int
    destination_name: str
    reservation_name: str
    reservation_device: int
    reservation_inode: int


def frozen_plan() -> dict[str, object]:
    """Return the literal protocol that must precede any retained execution."""
    return {
        "plan_id": PLAN_ID,
        "paper_revision": PAPER_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "source_identity_policy": {
            "binding": "execution-time Git/source hashes retained in every report",
            "hashed_files": [
                "alberta_framework/benchmarks/intentional_updates_control.py",
                "alberta_framework/benchmarks/ipmnist_screening.py",
                "alberta_framework/benchmarks/plasticity_comparators.py",
                "alberta_framework/benchmarks/upgd_ipmnist.py",
                "pyproject.toml",
                "uv.lock",
            ],
        },
        "runtime_identity_policy": (
            "retain exact execution Python, platform, all direct dependency versions, "
            "JAX backend/device/configuration, and relevant process environment"
        ),
        "seeds": list(CAMPAIGN_SEEDS),
        "quarantined_test_consumed_seeds": list(QUARANTINED_SEEDS),
        "protocol_families": ["supervised_ipmnist", "td_control"],
        "supervised_pair": ["intentional_updates_off", "intentional_updates_ipmnist"],
        "supervised_config": SUPERVISED_CONFIG.to_config(),
        "dataset": {
            "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
            "source": {
                "provider": "openml",
                "name": "mnist_784",
                "version": 1,
                "row_start": 0,
                "row_stop_exclusive": 60_000,
            },
            "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
            "x": {"dtype": "<f4", "shape": [60_000, 784], "sha256": DATASET_X_SHA256},
            "y": {"dtype": "<i4", "shape": [60_000], "sha256": DATASET_Y_SHA256},
        },
        "control_pairs": [
            ["fixed_td0", "intentional_td0"],
            ["fixed_trace", "intentional_trace"],
            ["fixed_q_lambda", "intentional_q_lambda"],
        ],
        "control_workload": {
            "id": "asi.recurring-two-state-continuing-mdp.v1",
            "horizon": 512,
            "phase_length": 64,
            "discount": 0.95,
            "trace_decay": 0.8,
            "behavior_policy": "seeded_uniform_random_common_within_pair",
        },
        "matched_axes": {
            "all_families": [
                "seed",
                "initial_parameters",
                "observations",
                "updates",
                "allowed_boundary_and_task_information",
            ],
            "supervised_ipmnist": ["example_schedule"],
            "td_control": [
                "transition_schedule",
                "behavior_rng_root_and_index_schedule",
            ],
        },
        "boundary_information": [],
        "task_information": [],
        "paired_metrics": {
            "supervised_ipmnist": "mean_per_task_accuracy:higher",
            "td0": "mean_squared_td_error:lower",
            "trace": "mean_squared_td_error:lower",
            "q_lambda": "mean_squared_td_error:lower",
        },
        "confidence_method": "two_sided_student_t_bonferroni_four_comparisons",
        "familywise_confidence": 0.95,
        "per_comparison_confidence": 0.9875,
        "confidence_degrees_of_freedom": 3,
        "confidence_critical": BONFERRONI_T_DF3,
        "multiple_comparison_policy": (
            "four predeclared paired questions; no cross-family ranking or aggregate winner"
        ),
        "timing_policy": "retained telemetry only; never a selection metric",
        "resource_policy": (
            "retain observations, environment/data steps, updates, model/action queries, "
            "RNG operations, optimizer solves, persistent bytes, and timing per shard"
        ),
        "output_path": "outputs/intentional_updates_matched_development/report.v1.json",
        # Runtime authorization is separate from this immutable plan identity.
        "execution_authorized": False,
        "reviewed_execution_transition": False,
        "execution_status": "blocked_pending_independent_plan_audit",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
        "execution_failure_policy": (
            "a failure after consumer dispatch leaves the immutable reservation as a "
            "consumed-without-result tombstone and forbids retry"
        ),
    }


def _bounded_json(value: object, *, context: str) -> object:
    nodes = 0
    string_bytes = 0

    def visit(item: object, *, depth: int, label: str) -> object:
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 18:
            raise ValueError(f"{context} exceeds the bounded exact JSON structure")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError(f"{label} exceeds signed-int64")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} must be finite exact JSON")
            return item
        if type(item) is str:
            encoded = item.encode("utf-8")
            if len(encoded) > 16_384 or b"\0" in encoded:
                raise ValueError(f"{label} must be a bounded exact JSON string")
            string_bytes += len(encoded)
            if string_bytes > _MAX_STRING_BYTES:
                raise ValueError(f"{context} exceeds its exact JSON string budget")
            return item
        if type(item) is list:
            if list.__len__(item) > _MAX_HORIZON + 1:
                raise ValueError(f"{label} exceeds its exact JSON list bound")
            return [
                visit(child, depth=depth + 1, label=f"{label}[{index}]")
                for index, child in enumerate(list.__iter__(item))
            ]
        if type(item) is dict:
            if dict.__len__(item) > 128:
                raise ValueError(f"{label} exceeds its exact JSON object bound")
            result: dict[str, object] = {}
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError(f"{label} keys must be exact JSON strings")
                result[key] = visit(child, depth=depth + 1, label=f"{label}.{key}")
            return result
        raise ValueError(f"{label} must contain only bounded exact JSON values")

    return visit(value, depth=0, label=context)


def _exact_object(value: object, keys: frozenset[str], *, context: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{context} keys do not match the exact contract")
    return cast(dict[str, object], value)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_identity() -> dict[str, str]:
    relative_paths = (
        "alberta_framework/benchmarks/intentional_updates_control.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/plasticity_comparators.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "pyproject.toml",
        "uv.lock",
    )
    return {
        relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _runtime_identity() -> dict[str, object]:
    environment_names = (
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET",
        "XLA_FLAGS",
    )
    devices = tuple(jax.devices())
    if len(devices) == 0 or len(devices) > _MAX_RUNTIME_DEVICES:
        raise RuntimeError("Intentional Updates runtime device inventory is out of bounds")
    return {
        "schema": "asi.intentional-updates.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            name: _version(name)
            for name in (
                "chex",
                "jax",
                "jaxlib",
                "jaxtyping",
                "numpy",
                "orbax-checkpoint",
                "scikit-learn",
                "scipy",
            )
        },
        "jax": {
            "backend": jax.default_backend(),
            "devices": [
                {
                    "id": int(device.id),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                    "process_index": int(device.process_index),
                }
                for device in devices
            ],
            "config": {
                "jax_default_matmul_precision": str(
                    jax.config.jax_default_matmul_precision
                ),
                "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
                "jax_disable_jit": bool(jax.config.jax_disable_jit),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_numpy_dtype_promotion": str(
                    jax.config.jax_numpy_dtype_promotion.value
                ),
                "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
                "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
                "jax_threefry_partitionable": bool(
                    jax.config.jax_threefry_partitionable
                ),
            },
        },
        "process_environment": {
            name: os.environ.get(name) for name in environment_names
        },
    }


def _features(state: int) -> np.ndarray:
    value = np.zeros(2, dtype=np.float32)
    value[state] = np.float32(1.0)
    return value


def _transition(state: int, action: int, step: int, phase_length: int) -> tuple[int, float]:
    successor = state if action == 0 else 1 - state
    goal = (step // phase_length) % 2
    return successor, 1.0 if successor == goal else 0.0


def _behavior_action(key: jax.Array) -> int:
    """Return one arm-independent action from the paired behavior stream."""
    return int(jr.randint(key, (), 0, 2, dtype=np.int32))


def _intentional_update(
    *,
    gradient: np.ndarray,
    trace: np.ndarray,
    second_moment: np.ndarray,
    sigma: np.float32,
    clip_ema: np.float32,
    step: int,
    delta: np.float32,
    discount: float,
    trace_decay: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.float32, np.float32, np.float32
]:
    """Pinned author optimizer equations specialized to one linear parameter array."""
    beta2 = np.float32(0.999)
    beta_clip = np.float32(0.9998)
    gamma_lambda = np.float32(discount * trace_decay)
    second = (beta2 * second_moment + (np.float32(1.0) - beta2) * gradient**2).astype(
        np.float32
    )
    correction = np.float32(1.0 - float(beta2) ** step)
    v_hat = (np.sqrt(second / correction) + np.float32(1e-8)).astype(np.float32)
    updated_trace = (gamma_lambda * trace + gradient).astype(np.float32)
    norm_gradient = np.float32(np.sum(gradient**2 / v_hat, dtype=np.float32))
    sigma_next = np.float32(
        sigma + (np.float32(1.0) - gamma_lambda) * (norm_gradient - sigma)
    )
    sigma_correction = np.float32(1.0 - float(gamma_lambda) ** step)
    sigma_hat = np.float32(sigma_next / sigma_correction)
    trace_energy = np.float32(np.sum(updated_trace**2 / v_hat, dtype=np.float32))
    normalizer = np.float32(np.sqrt(max(float(sigma_hat * trace_energy), 0.0)))
    step_size = np.float32(0.5 / max(float(normalizer), 1e-8))
    clip_next = np.float32(
        beta_clip * clip_ema + (np.float32(1.0) - beta_clip) * delta**2
    )
    clip_correction = np.float32(1.0 - float(beta_clip) ** step)
    cap = np.float32(20.0 * np.sqrt(max(float(clip_next / clip_correction), 0.0)))
    safe_delta = np.float32(math.copysign(min(abs(float(delta)), float(cap)), float(delta)))
    direction = (updated_trace / v_hat).astype(np.float32)
    return direction, updated_trace, second, sigma_next, clip_next, step_size * safe_delta


def _run(
    execution_arm: str, *, seed: int, horizon: int, phase_length: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    is_control = execution_arm.endswith("q_lambda")
    intentional = execution_arm.startswith("intentional_")
    trace_decay = 0.0 if execution_arm.endswith("td0") else 0.8
    shape = (2, 2) if is_control else (2,)
    weights = np.zeros(shape, dtype=np.float32)
    trace = np.zeros(shape, dtype=np.float32)
    second = np.zeros(shape, dtype=np.float32)
    sigma = np.float32(0.0)
    clip_ema = np.float32(0.0)
    root = jr.key(seed, impl=_AGENT_RNG_IMPL)
    state = int(jr.randint(jr.fold_in(root, _MAX_HORIZON), (), 0, 2, dtype=np.int32))
    states = [state]
    actions: list[int] = []
    rewards: list[float] = []
    errors: list[float] = []
    step_sizes: list[float] = []
    start_ns = time.perf_counter_ns()
    for index in range(horizon):
        action = _behavior_action(jr.fold_in(root, index))
        non_greedy = is_control and action != int(np.argmax(weights[state]))
        successor, reward = _transition(state, action, index, phase_length)
        if is_control:
            prediction = np.float32(weights[state, action])
            successor_prediction = np.float32(np.max(weights[successor]))
            gradient = np.zeros(shape, dtype=np.float32)
            gradient[state, action] = np.float32(1.0)
        else:
            feature = _features(state)
            next_feature = _features(successor)
            prediction = np.float32(np.dot(weights, feature))
            successor_prediction = np.float32(np.dot(weights, next_feature))
            gradient = feature
        delta = np.float32(reward + 0.95 * float(successor_prediction) - float(prediction))
        if intentional:
            direction, trace, second, sigma, clip_ema, update_scale = _intentional_update(
                gradient=gradient,
                trace=trace,
                second_moment=second,
                sigma=sigma,
                clip_ema=clip_ema,
                step=index + 1,
                delta=delta,
                discount=0.95,
                trace_decay=trace_decay,
            )
            weights = (weights + update_scale * direction).astype(np.float32)
            step_sizes.append(float(update_scale / delta) if delta != 0.0 else 0.0)
        else:
            trace = (np.float32(0.95 * trace_decay) * trace + gradient).astype(np.float32)
            weights = (weights + np.float32(0.05) * delta * trace).astype(np.float32)
            step_sizes.append(0.05)
        if is_control and non_greedy:
            trace.fill(np.float32(0.0))
        if not np.isfinite(weights).all():
            raise ValueError("Intentional Updates consumer produced non-finite state")
        states.append(successor)
        actions.append(action)
        rewards.append(float(reward))
        errors.append(float(delta))
        state = successor
    timing_ns = time.perf_counter_ns() - start_ns
    numeric_bytes = int(weights.nbytes + trace.nbytes)
    if intentional:
        numeric_bytes += int(
            second.nbytes
            + 2 * np.dtype(np.float32).itemsize
            + np.dtype(np.int32).itemsize
        )
    numeric_bytes += int(root.nbytes)
    trajectory = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "td_errors": errors,
        "step_sizes": step_sizes,
    }
    final_state = {
        "weights": weights.tolist(),
        "eligibility_trace": trace.tolist(),
        "entrywise_squared_gradient": second.tolist() if intentional else [],
        "global_normalizer": float(sigma) if intentional else 0.0,
        "clip_squared_error_ema": float(clip_ema) if intentional else 0.0,
        "optimizer_step": horizon if intentional else 0,
        "behavior_rng_root": jr.key_data(root).tolist(),
    }
    resources: dict[str, object] = {
        "observations": horizon,
        "environment_steps": horizon,
        "data_steps": 0,
        "reward_observations": horizon,
        "updates": horizon,
        "model_queries": 2 * horizon,
        "action_queries": 0,
        "rng_splits": 0,
        "rng_fold_ins": horizon + 1,
        "rng_uniform_draws": 0,
        "rng_integer_draws": horizon + 1,
        "intentional_step_size_solves": horizon if intentional else 0,
        "persistent_numeric_bytes": numeric_bytes,
        "timing_telemetry_ns": timing_ns,
        "timing_is_selection_metric": False,
    }
    return trajectory, final_state, resources


def run_control_shard(
    arm: str, *, seed: int, horizon: int = 512, phase_length: int = 64
) -> dict[str, object]:
    """Fail closed until a separate reviewed transition authorizes execution."""
    if (
        _REVIEWED_EXECUTION_TRANSITION is not True
        or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("Intentional Updates matched-development execution is not authorized")
    return _run_control_shard_authorized(
        arm,
        seed=seed,
        horizon=horizon,
        phase_length=phase_length,
        _capability=_EXECUTION_CAPABILITY,
    )


def _run_control_shard_authorized(
    arm: str,
    *,
    seed: int,
    horizon: int = 512,
    phase_length: int = 64,
    _capability: object,
) -> dict[str, object]:
    """Private bounded executor used by contract tests and a future gated campaign."""
    if _capability is _EXECUTION_CAPABILITY:
        allowed_seeds = CAMPAIGN_SEEDS
    elif _capability is _TEST_EXECUTION_CAPABILITY:
        allowed_seeds = TEST_ONLY_SEEDS
    else:
        raise RuntimeError("private Intentional execution capability is invalid")
    if type(arm) is not str or (arm not in CONTROL_ARMS and arm not in _OFF_ALIASES):
        raise ValueError("arm must name one exact registered control arm")
    if type(seed) is not int or seed not in allowed_seeds:
        raise ValueError("seed must be one exact prospectively reserved seed")
    if type(horizon) is not int or not 1 <= horizon <= _MAX_HORIZON:
        raise ValueError("horizon must be an exact integer in [1, 10000]")
    if type(phase_length) is not int or not 1 <= phase_length <= horizon:
        raise ValueError("phase_length must be an exact integer in [1, horizon]")
    execution_arm = _OFF_ALIASES.get(arm, arm)
    trajectory, final_state, resources = _run(
        execution_arm, seed=seed, horizon=horizon, phase_length=phase_length
    )
    errors = np.asarray(trajectory["td_errors"], dtype=np.float64)
    rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "arm": arm,
        "execution_arm": execution_arm,
        "seed": seed,
        "config": {
            "horizon": horizon,
            "phase_length": phase_length,
            "discount": 0.95,
            "trace_decay": 0.0 if execution_arm.endswith("td0") else 0.8,
            "fixed_step_size": 0.05,
            "intentional_fraction": 0.5,
            "behavior_policy": "seeded_uniform_random_common_within_pair",
        },
        "references": {
            "paper": PAPER_REVISION,
            "official_code": OFFICIAL_CODE_REVISION,
            "protocol_difference": (
                "linear two-state ASI consumer with fixed float32 arithmetic; not the "
                "author neural-network benchmark or publication-equivalent"
            ),
        },
        "information": {"boundary_information": [], "task_information": []},
        "identity": {
            "behavior_rng_impl": _AGENT_RNG_IMPL,
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "workload": "asi.recurring-two-state-continuing-mdp.v1",
        },
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcome_retention_required": True,
            "publication_equivalent": False,
        },
        "resources": resources,
        "trajectory": trajectory,
        "final_state": final_state,
        "metrics": {
            "mean_reward": float(np.mean(rewards)),
            "mean_squared_td_error": float(np.mean(np.square(errors))),
        },
    }
    return validate_control_shard(record)


def validate_control_shard(value: object) -> dict[str, object]:
    """Strictly reexecute all deterministic fields and admit timing as telemetry only."""
    if type(value) is not dict:
        raise ValueError("control shard must be an exact JSON object")
    record = cast(dict[str, object], _bounded_json(value, context="control shard"))
    expected_keys = frozenset(
        {
            "schema", "arm", "execution_arm", "seed", "config", "references",
            "information", "identity", "policy", "resources", "trajectory",
            "final_state", "metrics",
        }
    )
    _exact_object(record, expected_keys, context="control shard")
    if type(record["schema"]) is not str or record["schema"] != SCHEMA:
        raise ValueError("control shard schema drift")
    if (
        type(record["arm"]) is not str
        or (record["arm"] not in CONTROL_ARMS and record["arm"] not in _OFF_ALIASES)
        or type(record["seed"]) is not int
        or record["seed"] not in SEEDS
    ):
        raise ValueError("control shard arm and seed must be exact")
    config = _exact_object(
        record["config"],
        frozenset(
            {
                "horizon", "phase_length", "discount", "trace_decay", "fixed_step_size",
                "intentional_fraction", "behavior_policy",
            }
        ),
        context="control config",
    )
    if type(config["horizon"]) is not int or type(config["phase_length"]) is not int:
        raise ValueError("control config counters must be exact integers")
    arm = record["arm"]
    seed = record["seed"]
    horizon = config["horizon"]
    phase_length = config["phase_length"]
    if not 1 <= horizon <= _MAX_HORIZON:
        raise ValueError("control config horizon is outside the exact bound")
    if not 1 <= phase_length <= horizon:
        raise ValueError("control config phase_length is outside the exact bound")
    resources = _exact_object(
        record["resources"],
        frozenset(
            {
                "observations", "environment_steps", "data_steps", "reward_observations",
                "updates", "model_queries", "action_queries", "rng_splits", "rng_fold_ins",
                "rng_uniform_draws", "rng_integer_draws",
                "intentional_step_size_solves", "persistent_numeric_bytes",
                "timing_telemetry_ns", "timing_is_selection_metric",
            }
        ),
        context="control resources",
    )
    timing = resources["timing_telemetry_ns"]
    if (
        type(timing) is not int
        or not 0 <= timing <= _MAX_TIMING_NS
        or resources["timing_is_selection_metric"] is not False
    ):
        raise ValueError("timing must remain bounded telemetry only")
    # Avoid validator recursion: execute the primitive once, then construct through a
    # private clone of the public record with the observed timing substituted below.
    trajectory, final_state, expected_resources = _run(
        _OFF_ALIASES.get(arm, arm),
        seed=seed,
        horizon=horizon,
        phase_length=phase_length,
    )
    expected_resources["timing_telemetry_ns"] = timing
    errors = np.asarray(trajectory["td_errors"], dtype=np.float64)
    rewards = np.asarray(trajectory["rewards"], dtype=np.float64)
    expected_record = dict(record)
    execution_arm = _OFF_ALIASES.get(arm, arm)
    expected_record["schema"] = SCHEMA
    expected_record["arm"] = record["arm"]
    expected_record["seed"] = record["seed"]
    expected_record["execution_arm"] = execution_arm
    expected_record["config"] = {
        "horizon": config["horizon"],
        "phase_length": config["phase_length"],
        "discount": 0.95,
        "trace_decay": 0.0 if execution_arm.endswith("td0") else 0.8,
        "fixed_step_size": 0.05,
        "intentional_fraction": 0.5,
        "behavior_policy": "seeded_uniform_random_common_within_pair",
    }
    expected_record["references"] = {
        "paper": PAPER_REVISION,
        "official_code": OFFICIAL_CODE_REVISION,
        "protocol_difference": (
            "linear two-state ASI consumer with fixed float32 arithmetic; not the "
            "author neural-network benchmark or publication-equivalent"
        ),
    }
    expected_record["information"] = {"boundary_information": [], "task_information": []}
    expected_record["identity"] = {
        "behavior_rng_impl": _AGENT_RNG_IMPL,
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "workload": "asi.recurring-two-state-continuing-mdp.v1",
    }
    expected_record["policy"] = {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcome_retention_required": True,
        "publication_equivalent": False,
    }
    expected_record["resources"] = expected_resources
    expected_record["trajectory"] = trajectory
    expected_record["final_state"] = final_state
    expected_record["metrics"] = {
        "mean_reward": float(np.mean(rewards)),
        "mean_squared_td_error": float(np.mean(np.square(errors))),
    }
    if record != expected_record:
        raise ValueError("control shard differs from exact current reexecution")
    return record


def _current_source() -> dict[str, object]:
    result = dict(_screening_source_provenance(_REPO_ROOT))
    result["intentional_updates_control_source_sha256"] = _source_identity()
    return result


def _current_runtime() -> dict[str, object]:
    runtime = _bounded_json(_runtime_identity(), context="runtime identity")
    if type(runtime) is not dict:  # pragma: no cover - construction is a dict
        raise RuntimeError("runtime identity must be one exact object")
    return cast(dict[str, object], runtime)


def _digest(value: object, *, length: int, context: str) -> str:
    hexdigits = frozenset("0123456789abcdef")
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in hexdigits for character in value)
    ):
        raise ValueError(f"{context} must be one lowercase hexadecimal digest")
    return value


def _mean(values: object, *, context: str) -> float:
    if type(values) is not list or not values:
        raise ValueError(f"{context} must be one nonempty exact list")
    converted: list[float] = []
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{context} must contain exact finite floats")
        converted.append(value)
    return math.fsum(converted) / len(converted)


def _paired(deltas: list[float]) -> dict[str, object]:
    if len(deltas) != len(SEEDS) or any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired deltas must cover the exact seed roster")
    mean = math.fsum(deltas) / len(deltas)
    centered = math.fsum((value - mean) ** 2 for value in deltas)
    standard_error = math.sqrt(centered / (len(deltas) - 1)) / math.sqrt(len(deltas))
    lower = mean - BONFERRONI_T_DF3 * standard_error
    upper = mean + BONFERRONI_T_DF3 * standard_error
    outcome = "supported" if lower > 0.0 else "rejected" if upper <= 0.0 else "inconclusive"
    return {
        "deltas": deltas,
        "mean_delta": mean,
        "standard_error": standard_error,
        "ci9875_lower": lower,
        "ci9875_upper": upper,
        "outcome": outcome,
    }


def _validated_runs(
    supervised_records: object, control_records: object
) -> list[dict[str, object]]:
    if type(supervised_records) is not list or len(supervised_records) != 2 * len(SEEDS):
        raise ValueError("supervised records must contain the complete paired schedule")
    if type(control_records) is not list or len(control_records) != len(CONTROL_ARMS) * len(SEEDS):
        raise ValueError("control records must contain the complete paired schedule")
    supervised_by_key: dict[tuple[int, str], dict[str, object]] = {}
    for raw in supervised_records:
        record = validate_intentional_updates_development_record(raw)
        seed = record["seed"]
        arm = record["arm"]
        if (
            type(seed) is not int
            or seed not in SEEDS
            or type(arm) is not str
            or arm not in {"intentional_updates_off", "intentional_updates_ipmnist"}
            or record["config"] != SUPERVISED_CONFIG.to_config()
            or (seed, arm) in supervised_by_key
        ):
            raise ValueError("supervised record identity/configuration drift")
        supervised_by_key[seed, arm] = record
    control_by_key: dict[tuple[int, str], dict[str, object]] = {}
    for raw in control_records:
        record = validate_control_shard(raw)
        seed = record["seed"]
        arm = record["arm"]
        config = cast(dict[str, object], record["config"])
        if (
            type(seed) is not int
            or seed not in SEEDS
            or type(arm) is not str
            or arm not in CONTROL_ARMS
            or config["horizon"] != 512
            or config["phase_length"] != 64
            or (seed, arm) in control_by_key
        ):
            raise ValueError("control record identity/configuration drift")
        control_by_key[seed, arm] = record
    expected_supervised = {
        (seed, arm)
        for seed in SEEDS
        for arm in ("intentional_updates_off", "intentional_updates_ipmnist")
    }
    expected_control = {(seed, arm) for seed in SEEDS for arm in CONTROL_ARMS}
    if set(supervised_by_key) != expected_supervised or set(control_by_key) != expected_control:
        raise ValueError("records must contain the complete frozen seed-by-arm matrix")
    return [
        {
            "seed": seed,
            "supervised": [
                supervised_by_key[seed, "intentional_updates_off"],
                supervised_by_key[seed, "intentional_updates_ipmnist"],
            ],
            "control": [control_by_key[seed, arm] for arm in CONTROL_ARMS],
        }
        for seed in SEEDS
    ]


def _comparisons(runs: list[dict[str, object]]) -> dict[str, object]:
    supervised_deltas: list[float] = []
    td0_deltas: list[float] = []
    trace_deltas: list[float] = []
    q_deltas: list[float] = []
    for run in runs:
        supervised = cast(list[dict[str, object]], run["supervised"])
        supervised_deltas.append(
            _mean(
                cast(dict[str, object], supervised[1]["metrics"])["per_task_accuracy"],
                context="candidate supervised accuracy",
            )
            - _mean(
                cast(dict[str, object], supervised[0]["metrics"])["per_task_accuracy"],
                context="control supervised accuracy",
            )
        )
        controls = {
            cast(str, record["arm"]): record
            for record in cast(list[dict[str, object]], run["control"])
        }

        def metric(arm: str, name: str) -> float:
            value = cast(dict[str, object], controls[arm]["metrics"])[name]
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("control primary metric must be an exact finite float")
            return value

        td0_deltas.append(
            metric("fixed_td0", "mean_squared_td_error")
            - metric("intentional_td0", "mean_squared_td_error")
        )
        trace_deltas.append(
            metric("fixed_trace", "mean_squared_td_error")
            - metric("intentional_trace", "mean_squared_td_error")
        )
        q_deltas.append(
            metric("fixed_q_lambda", "mean_squared_td_error")
            - metric("intentional_q_lambda", "mean_squared_td_error")
        )
    return {
        "supervised_ipmnist": _paired(supervised_deltas),
        "td0": _paired(td0_deltas),
        "trace": _paired(trace_deltas),
        "q_lambda": _paired(q_deltas),
    }


def build_report(
    supervised_records: object,
    control_records: object,
    *,
    dataset_provenance: object,
    execution_source_commit: object,
) -> dict[str, object]:
    """Build the complete two-family development report without cross-family ranking."""
    commit = _digest(execution_source_commit, length=40, context="execution source commit")
    checked_dataset = _bounded_json(dataset_provenance, context="dataset provenance")
    if checked_dataset != frozen_plan()["dataset"]:
        raise ValueError("dataset identity does not match the frozen plan")
    runs = _validated_runs(supervised_records, control_records)
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "plan": frozen_plan(),
        "dataset_provenance": checked_dataset,
        "execution_source_commit": commit,
        "source_provenance": _current_source(),
        "runtime_environment": _current_runtime(),
        "runs": runs,
        "paired_comparisons": _comparisons(runs),
        "development_disposition": "four_separate_nonpromoting_outcomes",
        "policy": {
            "reviewed_execution_transition": _REVIEWED_EXECUTION_TRANSITION,
            "execution_authorized": _EXECUTION_AUTHORIZED,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
            "timing_is_telemetry_only": True,
            "cross_family_ranking_allowed": False,
        },
    }
    return validate_report(report, require_current_source=True)


def validate_report(value: object, *, require_current_source: bool = True) -> dict[str, object]:
    """Fail closed over identities, complete children, and exact paired arithmetic."""
    if type(require_current_source) is not bool:
        raise ValueError("require_current_source must be an exact bool")
    if type(value) is not dict:
        raise ValueError("report must be an exact object")
    report = cast(dict[str, object], _bounded_json(value, context="report"))
    _exact_object(
        report,
        frozenset(
            {
                "schema", "plan", "dataset_provenance",
                "execution_source_commit", "source_provenance", "runtime_environment",
                "runs", "paired_comparisons", "development_disposition", "policy",
            }
        ),
        context="report",
    )
    if type(report["schema"]) is not str or report["schema"] != REPORT_SCHEMA:
        raise ValueError("report schema drift")
    if json.dumps(report["plan"], sort_keys=True, separators=(",", ":")) != json.dumps(
        frozen_plan(), sort_keys=True, separators=(",", ":")
    ):
        raise ValueError("report plan differs from the literal prospective plan")
    if report["dataset_provenance"] != frozen_plan()["dataset"]:
        raise ValueError("report dataset identity drift")
    commit = _digest(report["execution_source_commit"], length=40, context="execution commit")
    source = _bounded_json(report["source_provenance"], context="source provenance")
    runtime = _bounded_json(report["runtime_environment"], context="runtime environment")
    if type(source) is not dict or source.get("git_commit") != commit:
        raise ValueError("execution commit does not match source provenance")
    if require_current_source and source != _current_source():
        raise ValueError("source provenance does not match current source")
    if require_current_source and runtime != _current_runtime():
        raise ValueError("runtime environment does not match current runtime")
    runs_raw = report["runs"]
    if type(runs_raw) is not list or len(runs_raw) != len(SEEDS):
        raise ValueError("runs must contain the complete frozen schedule")
    supervised: list[dict[str, object]] = []
    control: list[dict[str, object]] = []
    for index, raw_run in enumerate(runs_raw):
        run = _exact_object(
            raw_run, frozenset({"seed", "supervised", "control"}), context="run"
        )
        if type(run["seed"]) is not int or run["seed"] != SEEDS[index]:
            raise ValueError("runs must use deterministic seed ordering")
        if type(run["supervised"]) is not list or type(run["control"]) is not list:
            raise ValueError("run children must be exact lists")
        supervised.extend(cast(list[dict[str, object]], run["supervised"]))
        control.extend(cast(list[dict[str, object]], run["control"]))
    normalized_runs = _validated_runs(supervised, control)
    if normalized_runs != runs_raw:
        raise ValueError("runs differ from exact child validation")
    paired = _bounded_json(report["paired_comparisons"], context="paired comparisons")
    if paired != _comparisons(normalized_runs):
        raise ValueError("paired arithmetic does not match retained runs")
    policy = _exact_object(
        report["policy"],
        frozenset(
            {
                "development_only", "scientific_promotion_allowed",
                "negative_outcomes_retained", "timing_is_telemetry_only",
                "cross_family_ranking_allowed", "reviewed_execution_transition",
                "execution_authorized",
            }
        ),
        context="report policy",
    )
    if policy != {
        "reviewed_execution_transition": _REVIEWED_EXECUTION_TRANSITION,
        "execution_authorized": _EXECUTION_AUTHORIZED,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
        "timing_is_telemetry_only": True,
        "cross_family_ranking_allowed": False,
    } or any(type(item) is not bool for item in policy.values()):
        raise ValueError("report policy must remain permanently nonpromoting")
    if (
        type(report["development_disposition"]) is not str
        or report["development_disposition"] != "four_separate_nonpromoting_outcomes"
    ):
        raise ValueError("report must not manufacture a cross-family disposition")
    return report


def _open_parent(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("output path contains an unsafe directory component")
            created = False
            try:
                os.mkdir(component, 0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reserve(path: Path) -> _Reservation:
    if type(path) is not type(Path()) or path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError(f"output must be the exact reserved NEW path {OUTPUT_PATH}")
    directory_fd = _open_parent(path)
    reservation_name = f".{path.name}.reservation"
    descriptor = -1
    reservation_acquired = False
    reservation_identity: tuple[int, int] | None = None
    try:
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
        descriptor = os.open(
            reservation_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=directory_fd,
        )
        reservation_acquired = True
        marker = (
            f"reserved:{path.name}; retained as consumed-without-result after dispatch\n"
        ).encode("ascii")
        view = memoryview(marker)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable-output reservation write")
            view = view[written:]
        os.fsync(descriptor)
        marker_stat = os.fstat(descriptor)
        reservation_identity = (marker_stat.st_dev, marker_stat.st_ino)
        os.close(descriptor)
        descriptor = -1
        os.fsync(directory_fd)
        return _Reservation(
            directory_fd,
            path.name,
            reservation_name,
            marker_stat.st_dev,
            marker_stat.st_ino,
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if reservation_acquired:
            try:
                marker_stat = os.stat(
                    reservation_name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (marker_stat.st_dev, marker_stat.st_ino) == reservation_identity:
                    os.unlink(reservation_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
        raise


def _release(reservation: _Reservation) -> None:
    try:
        marker = os.stat(
            reservation.reservation_name,
            dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        if (marker.st_dev, marker.st_ino) == (
            reservation.reservation_device,
            reservation.reservation_inode,
        ):
            os.unlink(reservation.reservation_name, dir_fd=reservation.directory_fd)
            os.fsync(reservation.directory_fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(reservation.directory_fd)


def _retain_consumed_reservation(reservation: _Reservation) -> None:
    """Close a dispatched reservation without making the consumed schedule reusable."""
    try:
        marker = os.stat(
            reservation.reservation_name,
            dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(marker.st_mode)
            or marker.st_nlink != 1
            or (marker.st_dev, marker.st_ino)
            != (reservation.reservation_device, reservation.reservation_inode)
        ):
            raise RuntimeError("consumed reservation identity changed")
        os.fsync(reservation.directory_fd)
    finally:
        os.close(reservation.directory_fd)


def _strict_reread(
    reservation: _Reservation,
    expected: bytes,
    *,
    expected_identity: tuple[int, int],
) -> object:
    descriptor = os.open(
        reservation.destination_name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=reservation.directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= _MAX_REPORT_BYTES
        ):
            raise ValueError("published report must be one bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("published report ended before its pinned size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("published report grew during strict reread")
        final = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(metadata, field) != getattr(final, field) for field in stable)
            or final.st_nlink != 1
        ):
            raise ValueError("published report changed during strict reread")
    finally:
        os.close(descriptor)
    actual = b"".join(chunks)
    if actual != expected:
        raise ValueError("published bytes differ from the validated generation")
    try:
        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("published report contains a duplicate JSON key")
                result[key] = item
            return result

        return _bounded_json(
            json.loads(actual.decode("utf-8"), object_pairs_hook=exact_object),
            context="published report",
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("published report is not strict JSON") from error


def _publish_reserved(reservation: _Reservation, report: dict[str, object]) -> None:
    encoded = (
        json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("validated report exceeds its publication bound")
    temporary_name = f".{reservation.destination_name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    published_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=reservation.directory_fd,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("immutable report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        temporary = os.fstat(descriptor)
        if not stat.S_ISREG(temporary.st_mode) or temporary.st_nlink != 1:
            raise RuntimeError("temporary report must have one private regular-file link")
        published_identity = (temporary.st_dev, temporary.st_ino)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            reservation.destination_name,
            src_dir_fd=reservation.directory_fd,
            dst_dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=reservation.directory_fd)
        linked = os.stat(
            reservation.destination_name,
            dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(linked.st_mode)
            or (linked.st_dev, linked.st_ino) != published_identity
            or linked.st_nlink != 1
        ):
            raise ValueError("published report must have one unique regular-file link")
        os.fsync(reservation.directory_fd)
        reread = _strict_reread(
            reservation, encoded, expected_identity=published_identity
        )
        if validate_report(reread, require_current_source=True) != report:
            raise ValueError("published semantic payload differs from validated report")
    except BaseException:
        if published_identity is not None:
            try:
                destination = os.stat(
                    reservation.destination_name,
                    dir_fd=reservation.directory_fd,
                    follow_symlinks=False,
                )
                if (destination.st_dev, destination.st_ino) == published_identity:
                    os.unlink(reservation.destination_name, dir_fd=reservation.directory_fd)
                    os.fsync(reservation.directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=reservation.directory_fd)
            os.fsync(reservation.directory_fd)
        except FileNotFoundError:
            pass


def publish_report(path: Path, report: object) -> Path:
    """Fail closed until both reviewed and runtime authorization are true."""
    if (
        _REVIEWED_EXECUTION_TRANSITION is not True
        or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("Intentional Updates matched-development execution is not authorized")
    return _publish_report_authorized(
        path, report, _capability=_EXECUTION_CAPABILITY
    )


def _publish_report_authorized(
    path: Path, report: object, *, _capability: object
) -> Path:
    """Capability-private immutable publisher used only after an outer gate."""
    if (
        _capability is not _EXECUTION_CAPABILITY
        and _capability is not _TEST_EXECUTION_CAPABILITY
    ):
        raise RuntimeError("private Intentional publication capability is invalid")
    reservation = _reserve(path)
    try:
        validated = validate_report(report, require_current_source=True)
        _publish_reserved(reservation, validated)
    finally:
        _release(reservation)
    return path


def _execution_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _digest(completed.stdout.strip(), length=40, context="execution commit")


def run_campaign(dataset_path: Path, output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    """Run both frozen subprotocols exactly once after separate authorization."""
    if (
        _REVIEWED_EXECUTION_TRANSITION is not True
        or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("Intentional Updates matched-development execution is not authorized")
    if type(dataset_path) is not type(Path()) or type(output_path) is not type(Path()):
        raise ValueError("dataset and output must be exact pathlib.Path values")
    reservation = _reserve(output_path)
    execution_started = False
    report_published = False
    try:
        source_before = _current_source()
        runtime_before = _current_runtime()
        inputs, labels = _load_dataset(dataset_path)
        dataset_provenance = _screening_dataset_provenance(inputs, labels)
        if dataset_provenance != frozen_plan()["dataset"]:
            raise ValueError("dataset numeric payload does not match the frozen reviewed input")
        execution_started = True
        supervised_records = [
            intentional_updates_development_record(
                run_screening_config(
                    inputs,
                    labels,
                    screening_spec(arm),
                    seed,
                    SUPERVISED_CONFIG,
                )
            )
            for seed in CAMPAIGN_SEEDS
            for arm in ("intentional_updates_off", "intentional_updates_ipmnist")
        ]
        control_records = [
            _run_control_shard_authorized(
                arm, seed=seed, _capability=_EXECUTION_CAPABILITY
            )
            for seed in CAMPAIGN_SEEDS
            for arm in CONTROL_ARMS
        ]
        if source_before != _current_source():
            raise RuntimeError("source identity changed during matched execution")
        if runtime_before != _current_runtime():
            raise RuntimeError("runtime identity changed during matched execution")
        if dataset_provenance != _screening_dataset_provenance(inputs, labels):
            raise RuntimeError("dataset numeric payload changed during matched execution")
        report = build_report(
            supervised_records,
            control_records,
            dataset_provenance=dataset_provenance,
            execution_source_commit=_execution_commit(),
        )
        _publish_reserved(reservation, report)
        report_published = True
        return report
    finally:
        if report_published or not execution_started:
            _release(reservation)
        else:
            _retain_consumed_reservation(reservation)


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the frozen plan or reject execution while authorization is false."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.catalog:
        if args.dataset is not None:
            parser.error("--catalog and --dataset are mutually exclusive")
        print(json.dumps(frozen_plan(), sort_keys=True))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --catalog is used")
    run_campaign(args.dataset, args.output)
    return 0


__all__ = [
    "BONFERRONI_T_DF3",
    "CONTROL_ARMS",
    "DATASET_X_SHA256",
    "DATASET_Y_SHA256",
    "SEEDS",
    "SUPERVISED_CONFIG",
    "build_report",
    "frozen_plan",
    "main",
    "publish_report",
    "run_campaign",
    "run_control_shard",
    "validate_control_shard",
    "validate_report",
]
