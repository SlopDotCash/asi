"""Frozen, sharded, permanently nonpromoting issue #1566 campaigns.

Both stages use result-v2 with separately frozen external seed protocols. The
earlier public result-v1 and preauthorization rosters are quarantined. A plan
binds one exact dataset and runtime, every arm eventually executes in its own
CLI process, and aggregation admits only the complete 11-arm by 5-seed matrix.
Execution is disabled until a separate reviewed authorization transition. The
resulting decisions remain development diagnostics, not paper reproductions or
scientific evidence.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np

import alberta_framework.benchmarks.activation_feature_ipmnist as activation_lane
import alberta_framework.benchmarks.ipmnist_screening as screening_lane
import alberta_framework.benchmarks.plasticity_comparators as comparator_lane
import alberta_framework.benchmarks.upgd_ipmnist as ipmnist_lane
from alberta_framework.benchmarks.activation_feature_ipmnist import (
    ACTIVATION_FEATURE_SPECS,
    _array_bundle_sha256,
    activation_feature_campaign_result_payload,
    run_activation_feature_arm,
    validate_activation_feature_campaign_result,
    validate_matched_activation_feature_campaign_results,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

PLAN_SCHEMA: Final[str] = "asi.activation-feature-ipmnist.campaign-plan.v2"
SHARD_SCHEMA: Final[str] = "asi.activation-feature-ipmnist.campaign-shard.v2"
AGGREGATE_SCHEMA: Final[str] = "asi.activation-feature-ipmnist.campaign-aggregate.v2"

ARM_ROSTER: Final[tuple[str, ...]] = tuple(ACTIVATION_FEATURE_SPECS)
STAGE_ROSTER: Final[tuple[str, ...]] = ("cheap_screen", "full_confirmation")
QUARANTINED_CHEAP_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
QUARANTINED_FULL_SEEDS: Final[tuple[int, ...]] = (
    156_610,
    156_611,
    156_612,
    156_613,
    156_614,
)
QUARANTINED_REPLACEMENT_CHEAP_SEEDS: Final[tuple[int, ...]] = (
    2_156_600,
    2_156_601,
    2_156_602,
    2_156_603,
    2_156_604,
)
QUARANTINED_REPLACEMENT_FULL_SEEDS: Final[tuple[int, ...]] = (
    2_156_610,
    2_156_611,
    2_156_612,
    2_156_613,
    2_156_614,
)
CHEAP_SCREEN_SEEDS: Final[tuple[int, ...]] = (
    3_975_019_531,
    3_975_019_532,
    3_975_019_533,
    3_975_019_534,
    3_975_019_535,
)
FULL_CONFIRMATION_SEEDS: Final[tuple[int, ...]] = (
    2_924_933_221,
    2_924_933_222,
    2_924_933_223,
    2_924_933_224,
    2_924_933_225,
)
_SEEDS: Final[Mapping[str, tuple[int, ...]]] = {
    "cheap_screen": CHEAP_SCREEN_SEEDS,
    "full_confirmation": FULL_CONFIRMATION_SEEDS,
}
ALL_SEEDS: Final[tuple[int, ...]] = CHEAP_SCREEN_SEEDS + FULL_CONFIRMATION_SEEDS
CAMPAIGN_PAPER_SOURCES: Final[Mapping[str, Mapping[str, str]]] = {
    "smooth_leaky": {
        "publication": "ICLR-2026:XZf6wObHX4",
        "preprint": "arXiv:2509.22562v4",
        "official_repository": "https://github.com/lute47lillo/activations_plasticity",
        "official_commit": "bdce354782cd183d63550819550b33312506d3e3",
        "license_status": "no license file; read-only disambiguation reference",
        "implementation_source": "paper Equation 1",
    },
    "aid": {
        "publication": "ICML-2025:park25b; PMLR-v267:47991-48026",
        "preprint": "arXiv:2502.01342v2",
        "official_repository": "none located",
        "official_commit": "none",
        "license_status": "not applicable",
        "implementation_source": "paper Algorithm 2 simplified element-wise rule",
    },
    "deep_fourier": {
        "publication": "ICLR-2025:NIkfix2eDQ",
        "preprint": "arXiv:2410.20634v1",
        "official_repository": "none located",
        "official_commit": "none",
        "license_status": "not applicable",
        "implementation_source": "paper Fourier feature equation",
    },
}

_CONFIGS: Final[Mapping[str, IPMNISTConfig]] = {
    "cheap_screen": IPMNISTConfig(n_tasks=2, task_length=500),
    "full_confirmation": IPMNISTConfig(n_tasks=200, task_length=5_000),
}
_PLAN_IDS: Final[Mapping[str, str]] = {
    "cheap_screen": "issue-1566.activation-feature.cheap-screen.v2",
    "full_confirmation": "issue-1566.activation-feature.full-confirmation.v2",
}
_OUTPUT_NAMESPACES: Final[Mapping[str, str]] = {
    "cheap_screen": "outputs/activation_feature_ipmnist/cheap_screen.v2",
    "full_confirmation": "outputs/activation_feature_ipmnist/full_confirmation.v2",
}
_EXECUTION_AUTHORIZED: Final[bool] = False
_EXECUTION_CAPABILITY: Final[object] = object()
_PRIMARY_CANDIDATES: Final[tuple[str, ...]] = ("smooth_leaky", "aid", "deep_fourier")

# Eight hypotheses share one two-sided family-wise alpha.  The literal is
# scipy.stats.t.ppf(1 - 0.05 / (2 * 8), df=4), frozen before execution.
_FAMILYWISE_ALPHA: Final[float] = 0.05
_COMPARISON_ALPHA: Final[float] = 0.00625
_T_CRITICAL_DF4_BONFERRONI_8: Final[float] = 5.261057575065803

_COMPARISONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("smooth_leaky", "smooth_leaky", "smooth_leaky_off"),
    ("smooth_leaky", "smooth_leaky_fixed_leak", "smooth_leaky_off"),
    ("aid", "aid", "aid_off"),
    ("aid", "aid_expected", "aid_off"),
    ("aid", "ordinary_dropout", "aid_off"),
    ("deep_fourier", "deep_fourier", "deep_fourier_off"),
    ("deep_fourier", "deep_fourier_first_layer", "deep_fourier_off"),
    ("deep_fourier", "deep_fourier_sine_only", "deep_fourier_off"),
)

_MAX_JSON_DEPTH: Final[int] = 20
_MAX_JSON_NODES: Final[int] = 200_000
_MAX_JSON_STRING_BYTES: Final[int] = 64 * 1024
_MAX_SHARD_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_AGGREGATE_BYTES: Final[int] = 128 * 1024 * 1024
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CANONICAL_X_SHAPE: Final[tuple[int, int]] = (60_000, 784)
_CANONICAL_Y_SHAPE: Final[tuple[int]] = (60_000,)
_CANONICAL_X_SHA256: Final[str] = (
    "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
)
_CANONICAL_Y_SHA256: Final[str] = (
    "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
)
_DATASET_MATERIALIZATION: Final[str] = (
    "OpenML mnist_784 v1 rows 0:60000; float32 pixels scaled by "
    "(x / 255 - 0.5) / 0.5 and int32 labels"
)

_PLAN_FIELDS = frozenset(
    {
        "schema",
        "plan_id",
        "stage",
        "matrix",
        "config",
        "protocol",
        "execution_gate",
        "identity",
        "statistics",
        "resources",
        "paper_parity",
        "policy",
        "output_namespace",
        "plan_sha256",
    }
)
_SHARD_FIELDS = frozenset(
    {
        "schema",
        "plan_id",
        "plan_sha256",
        "stage",
        "arm",
        "seed",
        "authorization",
        "execution_identity",
        "result",
        "shard_sha256",
    }
)
_AGGREGATE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "plan",
        "plan_sha256",
        "shards",
        "prerequisite",
        "summary",
        "policy",
        "aggregate_sha256",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_object(value: object, fields: frozenset[str], context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping) or set(mapping) != fields:
        raise ValueError(f"{context} fields differ from the frozen schema")
    return cast(dict[str, object], mapping)


def _json_preflight(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("campaign JSON exceeds its structure bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("campaign JSON contains a non-finite float")
            continue
        if type(current) is str:
            try:
                encoded = current.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("campaign JSON contains invalid UTF-8") from error
            if len(encoded) > _MAX_JSON_STRING_BYTES or "\x00" in current:
                raise ValueError("campaign JSON contains an invalid string")
            continue
        if type(current) not in {dict, list}:
            raise ValueError("campaign records contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("campaign JSON contains aliased or cyclic containers")
        seen.add(identity)
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 4096 or any(type(key) is not str for key in mapping):
                raise ValueError("campaign JSON object exceeds its field bound")
            pending.extend((item, depth + 1) for item in mapping.values())
        else:
            sequence = cast(list[object], current)
            if len(sequence) > 4096:
                raise ValueError("campaign JSON list exceeds its item bound")
            pending.extend((item, depth + 1) for item in sequence)


def _json_copy(value: object) -> Any:
    return json.loads(_canonical(value))


def _source_identity() -> dict[str, str]:
    modules = (
        activation_lane,
        comparator_lane,
        screening_lane,
        ipmnist_lane,
        sys.modules[__name__],
    )
    result: dict[str, str] = {}
    for module in modules:
        path_value = getattr(module, "__file__", None)
        if type(path_value) is not str:
            raise RuntimeError(f"cannot locate source for {module.__name__}")
        path = Path(path_value).resolve(strict=True)
        try:
            name = path.relative_to(_REPO_ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError(f"source for {module.__name__} is outside the project") from error
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


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
    return {
        "schema": "asi.activation-feature-ipmnist.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn", "scipy")
        },
        "backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
        "jax_config": {
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion.value),
            "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
            "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _canonical_dataset_shapes() -> tuple[tuple[int, int], tuple[int]]:
    """Return the frozen train-split shapes (overridden only by bounded unit tests)."""

    return _CANONICAL_X_SHAPE, _CANONICAL_Y_SHAPE


def _canonical_dataset_hashes() -> tuple[str, str]:
    """Return the frozen canonical materialized-array identities."""

    return _CANONICAL_X_SHA256, _CANONICAL_Y_SHA256


def _validated_arrays(data_x: object, data_y: object) -> tuple[np.ndarray, np.ndarray]:
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("data_x must be an exact float32 NumPy array")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("data_y must be an exact int32 NumPy array")
    expected_x_shape, expected_y_shape = _canonical_dataset_shapes()
    if data_x.shape != expected_x_shape or data_y.shape != expected_y_shape:
        raise ValueError("dataset does not match the frozen OpenML MNIST train split")
    if not data_x.flags.c_contiguous or not data_y.flags.c_contiguous:
        raise ValueError("dataset arrays must use canonical C-contiguous storage")
    if data_x.nbytes + data_y.nbytes > 256 * 1024 * 1024:
        raise ValueError("dataset exceeds the 256 MiB campaign bound")
    if not np.isfinite(data_x).all():
        raise ValueError("data_x must be finite")
    if np.any(data_x < -1.0) or np.any(data_x > 1.0):
        raise ValueError("data_x must use the frozen [-1, 1] scaling")
    if np.any(data_y < 0) or np.any(data_y >= 10):
        raise ValueError("data_y lies outside the frozen ten-class label range")
    expected_x_sha256, expected_y_sha256 = _canonical_dataset_hashes()
    x_sha256 = screening_lane._array_bundle_sha256(
        "alberta.ipmnist_screening.materialized_x.v1", {"x": data_x}
    )
    y_sha256 = screening_lane._array_bundle_sha256(
        "alberta.ipmnist_screening.materialized_y.v1", {"y": data_y}
    )
    if x_sha256 != expected_x_sha256 or y_sha256 != expected_y_sha256:
        raise ValueError("dataset bytes differ from the frozen canonical OpenML materialization")
    return data_x, data_y


def _dataset_identity(data_x: np.ndarray, data_y: np.ndarray) -> dict[str, object]:
    return {
        "schema": "asi.activation-feature-ipmnist.dataset.v1",
        "provider": "OpenML",
        "dataset": "mnist_784",
        "version": 1,
        "row_start": 0,
        "row_stop_exclusive": data_x.shape[0],
        "materialization": _DATASET_MATERIALIZATION,
        "sha256": _array_bundle_sha256(data_x, data_y),
        "x_sha256": screening_lane._array_bundle_sha256(
            "alberta.ipmnist_screening.materialized_x.v1", {"x": data_x}
        ),
        "y_sha256": screening_lane._array_bundle_sha256(
            "alberta.ipmnist_screening.materialized_y.v1", {"y": data_y}
        ),
        "x_shape": list(data_x.shape),
        "y_shape": list(data_y.shape),
        "x_dtype": data_x.dtype.str,
        "y_dtype": data_y.dtype.str,
        "numeric_bytes": data_x.nbytes + data_y.nbytes,
    }


def _config_payload(config: IPMNISTConfig) -> dict[str, int]:
    return {
        name: getattr(config, name)
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
    }


def _paper_parity() -> dict[str, object]:
    return {
        "sources": {
            family: dict(source) for family, source in CAMPAIGN_PAPER_SOURCES.items()
        },
        "result_schema_semantics": {
            "cheap_screen": "result-v2 with an external frozen seed contract",
            "full_confirmation": "result-v2 with an external frozen seed contract",
            "campaign_source_binding": "exact sources in this plan's identity and registry",
        },
        "paper_metric_reported": False,
        "paper_protocol_parity_claimed": False,
        "paper_result_reproduction_claimed": False,
        "cross_method_ranking_claimed": False,
        "gaps": [
            "optimizer",
            "architecture",
            "batching",
            "task_schedule",
            "training_horizon",
            "reported_metric",
            "deep_fourier_parameterization",
        ],
        "interpretation": (
            "ASI current-runner causal development controls only; not reproduction of "
            "the papers' curves and not external state-of-the-art evidence"
        ),
    }


def _policy(stage: str) -> dict[str, object]:
    return {
        "development_only": True,
        "permanently_nonpromoting": True,
        "scientific_promotion_allowed": False,
        "reference_dev_update_allowed": False,
        "completed_shard_negative_results_retained": True,
        "execution_failure_receipts_retained": False,
        "execution_failure_note": (
            "ordinary Exception, BaseException, process death, and publication failure do not "
            "produce campaign failure receipts; the reservation marker is concurrency state, "
            "not failure evidence, so the external scheduler must retain its log before any "
            "separately authorized retry"
        ),
        "timing_is_telemetry_only": True,
        "execution_attestation": False,
        "cross_stage_seed_reuse": False,
        "cross_stage_independent_confirmation_claimed": False,
        "stage_note": (
            "full confirmation uses disjoint frozen development seeds but is selected by the "
            "cheap-screen gate, so it is not independent scientific confirmation"
            if stage == "full_confirmation"
            else "cheap development screen; it cannot select scientific evidence"
        ),
    }


def _matrix(stage: str) -> dict[str, object]:
    seeds = _SEEDS[stage]
    return {
        "arms": list(ARM_ROSTER),
        "seeds": list(seeds),
        "shard_count": len(ARM_ROSTER) * len(seeds),
        "ordering": "seed_major_then_arm_roster",
        "execution": "one_shard_per_fresh_python_process",
    }


def _statistics(stage: str) -> dict[str, object]:
    return {
        "primary_metric": "asi_whole_stream_mean_accuracy",
        "paired_direction": "candidate_minus_family_mechanism_off",
        "null_delta": 0.0,
        "method": "two_sided_paired_student_t",
        "confidence_family": "simultaneous_bonferroni",
        "familywise_alpha": _FAMILYWISE_ALPHA,
        "comparison_count": len(_COMPARISONS),
        "per_comparison_alpha": _COMPARISON_ALPHA,
        "degrees_of_freedom": len(_SEEDS[stage]) - 1,
        "critical_value": _T_CRITICAL_DF4_BONFERRONI_8,
        "comparisons": [
            {"family": family, "candidate": candidate, "control": control}
            for family, candidate, control in _COMPARISONS
        ],
        "decision_rule": {
            "supported": "simultaneous_ci_lower_gt_0",
            "rejected": "simultaneous_ci_upper_lt_0",
            "inconclusive": "otherwise",
        },
    }


def _protocol(stage: str) -> dict[str, object]:
    return {
        "runner": "alberta_framework.benchmarks.ipmnist_screening.run_screening_config",
        "shard_result_schema": activation_lane.CAMPAIGN_RESULT_SCHEMA,
        "matched_axes": [
            "seed_derived_task_permutations",
            "seed_derived_example_indices",
            "observations",
            "updates",
            "gradient_evaluations",
            "model_queries",
            "allocated_parameter_scalars",
            "persistent_numeric_bytes",
            "allowed_boundary_information",
            "allowed_task_information",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "matrix_completion": "all_55_shards_required_without_adaptive_arm_or_seed_dropping",
        "seed_provenance": (
            "globally searched replacement result-v2 roster frozen before authorization; "
            "no execution is authorized by this plan"
        ),
        "quarantined_seed_rosters": {
            "public_result_v1_and_tests": list(QUARANTINED_CHEAP_SEEDS),
            "preauthorization_pull_request_and_tests": list(QUARANTINED_FULL_SEEDS),
            "first_replacement_cheap_tests": list(QUARANTINED_REPLACEMENT_CHEAP_SEEDS),
            "first_replacement_full_tests": list(QUARANTINED_REPLACEMENT_FULL_SEEDS),
        },
        "quarantine_reason": (
            "every earlier roster was publicly exposed or used to derive schedule and "
            "initialization identities in pull-request tests, so none is represented as fresh"
        ),
        "randomness": (
            "seed and runtime PRNG implementation are source-bound; campaign records do not "
            "treat a consistency digest as authenticated execution proof"
        ),
    }


def _execution_gate(stage: str) -> dict[str, object]:
    if stage == "cheap_screen":
        return {
            "mode": "unconditional_bounded_development_screen",
            "prerequisite_stage": None,
            "required_primary_candidates": [],
            "rule": "plan_and_dataset_validation",
            "complete_matrix_if_authorized": True,
            "execution_authorized": False,
            "authorization_transition": "separate_reviewed_source_change_required",
        }
    return {
        "mode": "conditional_on_retained_cheap_screen",
        "prerequisite_stage": "cheap_screen",
        "required_primary_candidates": list(_PRIMARY_CANDIDATES),
        "rule": "at_least_one_primary_candidate_supported_by_simultaneous_ci",
        "complete_matrix_if_authorized": True,
        "execution_authorized": False,
        "authorization_transition": "separate_reviewed_source_change_required",
    }


def _resource_plan(stage: str, n_train: int) -> dict[str, object]:
    config = _CONFIGS[stage]
    steps = config.n_tasks * config.task_length
    allocated = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    schedule_bytes = 4 * config.n_tasks * (config.input_dim + n_train)
    shards = len(ARM_ROSTER) * len(_SEEDS[stage])
    return {
        "per_shard": {
            "data_steps": steps,
            "environment_steps": 0,
            "updates": steps,
            "gradient_evaluations": steps,
            "model_queries": 2 * steps,
            "allocated_parameter_scalars": allocated,
            "persistent_numeric_bytes": 4 * (allocated + 2 * config.input_dim + 1),
            "retained_schedule_numeric_bytes": schedule_bytes,
        },
        "matrix_totals": {
            "data_steps": shards * steps,
            "environment_steps": 0,
            "updates": shards * steps,
            "gradient_evaluations": shards * steps,
            "model_queries": shards * 2 * steps,
        },
        "dataset_numeric_bytes_limit": 256 * 1024 * 1024,
        "retained_schedule_numeric_bytes_limit_per_shard": 256 * 1024 * 1024,
        "shard_json_bytes_limit": _MAX_SHARD_BYTES,
        "aggregate_input_bytes_limit": _MAX_AGGREGATE_BYTES,
    }


def _plan_unsigned(stage: str, dataset: dict[str, object]) -> dict[str, object]:
    config = _CONFIGS[stage]
    x_shape = cast(list[object], dataset["x_shape"])
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": _PLAN_IDS[stage],
        "stage": stage,
        "matrix": _matrix(stage),
        "config": _config_payload(config),
        "protocol": _protocol(stage),
        "execution_gate": _execution_gate(stage),
        "identity": {
            "dataset": copy.deepcopy(dataset),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_execution_attestation": True,
        },
        "statistics": _statistics(stage),
        "resources": _resource_plan(stage, cast(int, x_shape[0])),
        "paper_parity": _paper_parity(),
        "policy": _policy(stage),
        "output_namespace": _OUTPUT_NAMESPACES[stage],
    }


def build_plan(stage: str, data_x: object, data_y: object) -> dict[str, object]:
    """Build the source-, runtime-, and dataset-bound literal stage plan."""
    if type(stage) is not str or stage not in STAGE_ROSTER:
        raise ValueError("stage is outside the frozen campaign roster")
    x, y = _validated_arrays(data_x, data_y)
    plan = _plan_unsigned(stage, _dataset_identity(x, y))
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan, data_x=x, data_y=y)
    return plan


def _validated_dataset_identity(value: object) -> dict[str, object]:
    fields = frozenset(
        {
            "schema",
            "provider",
            "dataset",
            "version",
            "row_start",
            "row_stop_exclusive",
            "materialization",
            "sha256",
            "x_sha256",
            "y_sha256",
            "x_shape",
            "y_shape",
            "x_dtype",
            "y_dtype",
            "numeric_bytes",
        }
    )
    dataset = _exact_object(value, fields, "plan.identity.dataset")
    x_shape = dataset["x_shape"]
    y_shape = dataset["y_shape"]
    if (
        dataset["schema"] != "asi.activation-feature-ipmnist.dataset.v1"
        or dataset["provider"] != "OpenML"
        or dataset["dataset"] != "mnist_784"
        or dataset["version"] != 1
        or dataset["row_start"] != 0
        or dataset["materialization"] != _DATASET_MATERIALIZATION
        or not _is_sha256(dataset["sha256"])
        or not _is_sha256(dataset["x_sha256"])
        or not _is_sha256(dataset["y_sha256"])
        or type(x_shape) is not list
        or len(x_shape) != 2
        or any(type(item) is not int for item in x_shape)
        or type(y_shape) is not list
        or len(y_shape) != 1
        or any(type(item) is not int for item in y_shape)
        or dataset["x_dtype"] != np.dtype(np.float32).str
        or dataset["y_dtype"] != np.dtype(np.int32).str
    ):
        raise ValueError("plan dataset identity differs from frozen MNIST")
    checked_x_shape = cast(list[int], x_shape)
    checked_y_shape = cast(list[int], y_shape)
    expected_x_shape, expected_y_shape = _canonical_dataset_shapes()
    expected_x_sha256, expected_y_sha256 = _canonical_dataset_hashes()
    rows = expected_x_shape[0]
    if (
        checked_x_shape != list(expected_x_shape)
        or checked_y_shape != list(expected_y_shape)
        or dataset["row_stop_exclusive"] != rows
        or dataset["x_sha256"] != expected_x_sha256
        or dataset["y_sha256"] != expected_y_sha256
    ):
        raise ValueError("plan dataset identity differs from the canonical materialization")
    numeric_bytes = dataset["numeric_bytes"]
    if type(numeric_bytes) is not int or numeric_bytes != rows * (784 + 1) * 4:
        raise ValueError("plan dataset byte accounting drifted")
    if numeric_bytes > 256 * 1024 * 1024:
        raise ValueError("plan dataset exceeds its numeric byte bound")
    return dataset


def validate_plan(
    value: object, *, data_x: object | None = None, data_y: object | None = None
) -> dict[str, object]:
    """Strictly validate one plan against current source/runtime and optional bytes."""
    _json_preflight(value)
    plan = _exact_object(value, _PLAN_FIELDS, "plan")
    if plan["schema"] != PLAN_SCHEMA or type(plan["stage"]) is not str:
        raise ValueError("unsupported activation/feature campaign plan")
    stage = plan["stage"]
    if stage not in STAGE_ROSTER:
        raise ValueError("plan stage is outside the frozen roster")
    identity = _exact_object(
        plan["identity"],
        frozenset(
            {
                "dataset",
                "source_sha256",
                "runtime",
                "consistency_not_execution_attestation",
            }
        ),
        "plan.identity",
    )
    dataset = _validated_dataset_identity(identity["dataset"])
    if (data_x is None) != (data_y is None):
        raise ValueError("dataset validation requires both data arrays")
    if data_x is not None and data_y is not None:
        x, y = _validated_arrays(data_x, data_y)
        if dataset != _dataset_identity(x, y):
            raise ValueError("plan does not bind the supplied dataset bytes")
    expected = _plan_unsigned(stage, dataset)
    claimed = plan["plan_sha256"]
    if not _is_sha256(claimed) or claimed != _digest(expected):
        raise ValueError("plan digest drifted")
    expected["plan_sha256"] = claimed
    if plan != expected:
        raise ValueError("plan differs from the current literal frozen plan")
    return plan


def _config_from_plan(plan: Mapping[str, object]) -> IPMNISTConfig:
    config = cast(dict[str, object], plan["config"])
    return IPMNISTConfig(
        n_tasks=cast(int, config["n_tasks"]),
        task_length=cast(int, config["task_length"]),
        input_dim=cast(int, config["input_dim"]),
        hidden1=cast(int, config["hidden1"]),
        hidden2=cast(int, config["hidden2"]),
        n_classes=cast(int, config["n_classes"]),
    )


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-activation-feature-initial-parameters-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        host = np.asarray(leaf)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


@functools.lru_cache(maxsize=64)
def _expected_execution_identity(stage: str, seed: int, n_train: int) -> dict[str, str]:
    config = _CONFIGS[stage]
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    parameters = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    digest = hashlib.sha256(b"asi-activation-feature-schedule-v1\0")
    for value in (schedule.permutations, schedule.example_indices):
        host = np.asarray(value, dtype=np.int32)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.astype("<i4", copy=False).tobytes(order="C"))
    return {
        "schedule_sha256": digest.hexdigest(),
        "initial_parameters_sha256": _array_tree_sha256(parameters),
        "prng_implementation": "threefry2x32",
    }


def _execution_authorization(
    plan: Mapping[str, object], prerequisite: object | None
) -> dict[str, object]:
    stage = cast(str, plan["stage"])
    if stage == "cheap_screen":
        if prerequisite is not None:
            raise ValueError("cheap screen does not accept a prerequisite aggregate")
        return {
            "mode": "unconditional_bounded_development_screen",
            "prerequisite_aggregate_sha256": None,
            "supported_primary_candidates": [],
            "rule_satisfied": True,
        }
    if prerequisite is None:
        raise ValueError("full confirmation requires the retained cheap-screen aggregate")
    cheap = validate_aggregate(prerequisite)
    cheap_plan = cast(dict[str, object], cheap["plan"])
    if cheap_plan["stage"] != "cheap_screen":
        raise ValueError("full confirmation prerequisite is not a cheap-screen aggregate")
    plan_identity = cast(dict[str, object], plan["identity"])
    cheap_identity = cast(dict[str, object], cheap_plan["identity"])
    if plan_identity != cheap_identity:
        raise ValueError("full and cheap plans must bind the same dataset, source, and runtime")
    summary = cast(dict[str, object], cheap["summary"])
    comparisons = summary["paired_comparisons"]
    if type(comparisons) is not list:
        raise ValueError("cheap-screen comparisons are malformed")
    supported = [
        cast(str, item["candidate"])
        for raw_item in comparisons
        if type(raw_item) is dict
        for item in [cast(dict[str, object], raw_item)]
        if item.get("candidate") in _PRIMARY_CANDIDATES and item.get("outcome") == "supported"
    ]
    if not supported:
        raise ValueError("cheap screen did not authorize full confirmation")
    return {
        "mode": "conditional_on_retained_cheap_screen",
        "prerequisite_aggregate_sha256": cheap["aggregate_sha256"],
        "supported_primary_candidates": supported,
        "rule_satisfied": True,
    }


def _unsigned_shard(
    plan: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    authorization: object,
    execution_identity: object,
    result: object,
) -> dict[str, object]:
    return {
        "schema": SHARD_SCHEMA,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "stage": plan["stage"],
        "arm": arm,
        "seed": seed,
        "authorization": _json_copy(authorization),
        "execution_identity": _json_copy(execution_identity),
        "result": _json_copy(result),
    }


def _require_execution_authorized() -> None:
    if _EXECUTION_AUTHORIZED is not True:
        raise RuntimeError(
            "activation/feature campaign execution is not authorized; a separate reviewed "
            "authorization transition is required"
        )


def _receipt_source_identity(plan: Mapping[str, object]) -> list[str]:
    identity = cast(dict[str, object], plan["identity"])
    sources = cast(dict[str, str], identity["source_sha256"])
    names = (
        "alberta_framework/benchmarks/activation_feature_ipmnist.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
    )
    return [sources[name] for name in names]


def _receipt_runtime_identity(plan: Mapping[str, object]) -> list[str]:
    identity = cast(dict[str, object], plan["identity"])
    runtime = cast(dict[str, object], identity["runtime"])
    python = cast(list[int], runtime["python"])
    packages = cast(dict[str, str], runtime["packages"])
    return [
        ".".join(str(part) for part in python),
        packages["jax"],
        packages["numpy"],
        cast(str, runtime["backend"]),
    ]


def build_shard(
    plan: object,
    data_x: object,
    data_y: object,
    *,
    arm: str,
    seed: int,
    prerequisite: object | None = None,
) -> dict[str, object]:
    """Fail closed until a separate reviewed transition authorizes execution."""
    _require_execution_authorized()
    return _build_shard_authorized(
        plan,
        data_x,
        data_y,
        arm=arm,
        seed=seed,
        prerequisite=prerequisite,
        _capability=_EXECUTION_CAPABILITY,
    )


def _build_shard_authorized(
    plan: object,
    data_x: object,
    data_y: object,
    *,
    arm: str,
    seed: int,
    prerequisite: object | None = None,
    _capability: object,
) -> dict[str, object]:
    """Private executor reachable only behind the reviewed public gate."""
    _require_execution_authorized()
    if _capability is not _EXECUTION_CAPABILITY:
        raise RuntimeError("private campaign execution capability is invalid")
    checked_plan = validate_plan(plan, data_x=data_x, data_y=data_y)
    if type(arm) is not str or arm not in ARM_ROSTER:
        raise ValueError("arm is outside the frozen matrix")
    stage = cast(str, checked_plan["stage"])
    seeds = _SEEDS[stage]
    authorization = _execution_authorization(checked_plan, prerequisite)
    if type(seed) is not int or seed not in seeds:
        raise ValueError("seed is outside the frozen matrix")
    x, y = _validated_arrays(data_x, data_y)
    dataset_identity = _dataset_identity(x, y)
    identity = cast(dict[str, object], checked_plan["identity"])
    planned_sources = cast(dict[str, str], identity["source_sha256"])
    planned_runtime = cast(dict[str, object], identity["runtime"])
    if _source_identity() != planned_sources or _runtime_identity() != planned_runtime:
        raise RuntimeError("source or runtime changed before shard execution")
    execution_identity = _expected_execution_identity(stage, seed, x.shape[0])
    result = run_activation_feature_arm(
        x,
        y,
        arm=arm,
        seed=seed,
        config=_config_from_plan(checked_plan),
    )
    if _source_identity() != planned_sources or _runtime_identity() != planned_runtime:
        raise RuntimeError("source or runtime changed during shard execution")
    receipt = activation_feature_campaign_result_payload(
        result, outcome="inconclusive", development_seeds=seeds
    )
    if result.dataset_sha256 != dataset_identity["sha256"]:
        raise RuntimeError("dataset identity changed during shard execution")
    shard = _unsigned_shard(
        checked_plan,
        arm=arm,
        seed=seed,
        authorization=authorization,
        execution_identity=execution_identity,
        result=receipt,
    )
    shard["shard_sha256"] = _digest(shard)
    validate_shard(shard, checked_plan, prerequisite=prerequisite)
    return shard


def validate_shard(
    value: object, plan: object, *, prerequisite: object | None = None
) -> dict[str, object]:
    """Cross-validate one versioned receipt against its exact campaign plan."""
    checked_plan = validate_plan(plan)
    expected_authorization = _execution_authorization(checked_plan, prerequisite)
    return _validate_shard_against_plan(value, checked_plan, expected_authorization)


def _validate_shard_against_plan(
    value: object,
    checked_plan: dict[str, object],
    expected_authorization: dict[str, object],
) -> dict[str, object]:
    _json_preflight(value)
    shard = _exact_object(value, _SHARD_FIELDS, "shard")
    if (
        shard["schema"] != SHARD_SCHEMA
        or shard["plan_id"] != checked_plan["plan_id"]
        or shard["plan_sha256"] != checked_plan["plan_sha256"]
        or shard["stage"] != checked_plan["stage"]
    ):
        raise ValueError("shard belongs to another plan")
    arm = shard["arm"]
    seed = shard["seed"]
    stage = cast(str, checked_plan["stage"])
    seeds = _SEEDS[stage]
    if shard["authorization"] != expected_authorization:
        raise ValueError("shard execution authorization drifted")
    if type(arm) is not str or arm not in ARM_ROSTER:
        raise ValueError("shard arm is outside the frozen matrix")
    if type(seed) is not int or seed not in seeds:
        raise ValueError("shard seed is outside the frozen matrix")
    receipt = validate_activation_feature_campaign_result(
        shard["result"], development_seeds=seeds
    )
    if receipt["arm"] != arm or receipt["seed"] != seed:
        raise ValueError("shard wrapper and result identity disagree")
    if receipt["outcome"] != "inconclusive":
        raise ValueError("individual shards cannot self-assign campaign outcomes")
    if receipt["config"] != checked_plan["config"]:
        raise ValueError("shard config silently shrinks or changes the frozen horizon")
    identity = cast(dict[str, object], checked_plan["identity"])
    dataset = cast(dict[str, object], identity["dataset"])
    execution = cast(dict[str, object], receipt["execution_identity"])
    if (
        execution["dataset_sha256"] != dataset["sha256"]
        or execution["n_train"] != cast(list[object], dataset["x_shape"])[0]
    ):
        raise ValueError("shard dataset identity disagrees with the plan")
    if execution["source_sha256"] != _receipt_source_identity(checked_plan):
        raise ValueError("shard receipt source identity disagrees with the plan")
    if execution["runtime"] != _receipt_runtime_identity(checked_plan):
        raise ValueError("shard receipt runtime identity disagrees with the plan")
    n_train = cast(int, execution["n_train"])
    expected_execution = _expected_execution_identity(stage, seed, n_train)
    if shard["execution_identity"] != expected_execution:
        raise ValueError("shard schedule, initialization, or PRNG identity drifted")
    if execution["schedule_sha256"] != expected_execution["schedule_sha256"]:
        raise ValueError("result schedule digest disagrees with the exact Threefry schedule")
    planned_resources = cast(dict[str, object], checked_plan["resources"])
    per_shard = cast(dict[str, object], planned_resources["per_shard"])
    resources = cast(dict[str, object], receipt["resources"])
    for name, expected in per_shard.items():
        if resources[name] != expected:
            raise ValueError(f"shard resource {name} disagrees with the plan")
    unsigned = dict(shard)
    claimed = unsigned.pop("shard_sha256")
    if not _is_sha256(claimed) or claimed != _digest(unsigned):
        raise ValueError("shard digest drifted")
    return shard


def _paired_summary(
    by_identity: Mapping[tuple[int, str], dict[str, object]],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for family, candidate, control in _COMPARISONS:
        deltas: list[float] = []
        candidate_values: list[float] = []
        control_values: list[float] = []
        for seed in seeds:
            candidate_metrics = cast(
                dict[str, object],
                cast(dict[str, object], by_identity[(seed, candidate)]["result"])["metrics"],
            )
            control_metrics = cast(
                dict[str, object],
                cast(dict[str, object], by_identity[(seed, control)]["result"])["metrics"],
            )
            candidate_value = cast(float, candidate_metrics["asi_whole_stream_mean_accuracy"])
            control_value = cast(float, control_metrics["asi_whole_stream_mean_accuracy"])
            candidate_values.append(candidate_value)
            control_values.append(control_value)
            deltas.append(candidate_value - control_value)
        mean = math.fsum(deltas) / len(deltas)
        centered = math.fsum((delta - mean) ** 2 for delta in deltas)
        standard_deviation = math.sqrt(centered / (len(deltas) - 1))
        standard_error = standard_deviation / math.sqrt(len(deltas))
        margin = _T_CRITICAL_DF4_BONFERRONI_8 * standard_error
        lower = mean - margin
        upper = mean + margin
        outcome = "supported" if lower > 0.0 else "rejected" if upper < 0.0 else "inconclusive"
        summaries.append(
            {
                "family": family,
                "candidate": candidate,
                "control": control,
                "paired_seed_order": list(seeds),
                "candidate_values": candidate_values,
                "control_values": control_values,
                "paired_deltas": deltas,
                "mean_delta": mean,
                "sample_standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "simultaneous_ci_lower": lower,
                "simultaneous_ci_upper": upper,
                "outcome": outcome,
            }
        )
    return summaries


def _arm_summary(
    by_identity: Mapping[tuple[int, str], dict[str, object]],
    seeds: tuple[int, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    metric_names = (
        "asi_whole_stream_mean_accuracy",
        "asi_whole_stream_mean_loss",
        "asi_whole_stream_mean_plasticity",
    )
    for arm in ARM_ROSTER:
        metrics = [
            cast(
                dict[str, object],
                cast(dict[str, object], by_identity[(seed, arm)]["result"])["metrics"],
            )
            for seed in seeds
        ]
        result[arm] = {
            name: math.fsum(cast(float, item[name]) for item in metrics) / len(metrics)
            for name in metric_names
        }
    return result


def _resource_summary(shards: Sequence[dict[str, object]]) -> dict[str, object]:
    sum_fields = (
        "data_steps",
        "environment_steps",
        "updates",
        "gradient_evaluations",
        "optimizer_scalar_updates",
        "model_queries",
        "random_bernoulli_variates",
        "activation_scalar_evaluations",
        "forward_affine_weight_applications",
        "sigmoid_evaluations",
        "trigonometric_evaluations",
    )
    resources = [
        cast(dict[str, object], cast(dict[str, object], shard["result"])["resources"])
        for shard in shards
    ]
    return {
        "totals": {
            name: sum(cast(int, item[name]) for item in resources) for name in sum_fields
        },
        "maximum_per_shard": {
            name: max(cast(int, item[name]) for item in resources)
            for name in (
                "allocated_parameter_scalars",
                "active_parameter_scalars",
                "persistent_numeric_bytes",
                "retained_schedule_numeric_bytes",
            )
        },
        "timing_seconds_total_telemetry_only": math.fsum(
            cast(float, item["timing_seconds"]) for item in resources
        ),
        "timing_is_telemetry_only": True,
    }


def _unsigned_aggregate(
    plan: dict[str, object],
    ordered: Sequence[dict[str, object]],
    prerequisite: object | None,
) -> dict[str, object]:
    seeds = _SEEDS[cast(str, plan["stage"])]
    by_identity = {
        (cast(int, shard["seed"]), cast(str, shard["arm"])): shard for shard in ordered
    }
    comparisons = _paired_summary(by_identity, seeds)
    outcomes = [cast(str, item["outcome"]) for item in comparisons]
    status = (
        "complete_with_supported_candidates"
        if "supported" in outcomes
        else "complete_with_rejections"
        if "rejected" in outcomes
        else "complete_inconclusive"
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": status,
        "plan": _json_copy(plan),
        "plan_sha256": plan["plan_sha256"],
        "shards": [_json_copy(shard) for shard in ordered],
        "prerequisite": None if prerequisite is None else _json_copy(prerequisite),
        "summary": {
            "shard_count": len(ordered),
            "arm_means": _arm_summary(by_identity, seeds),
            "paired_comparisons": comparisons,
            "resources": _resource_summary(ordered),
        },
        "policy": _json_copy(plan["policy"]),
    }


def _admit_complete_matrix(
    checked_plan: dict[str, object], shards: object, prerequisite: object | None
) -> list[dict[str, object]]:
    if type(shards) not in {list, tuple}:
        raise ValueError("aggregate shards must be an exact list or tuple")
    raw_shards = cast(Sequence[object], shards)
    stage = cast(str, checked_plan["stage"])
    seeds = _SEEDS[stage]
    expected_roster = [(seed, arm) for seed in seeds for arm in ARM_ROSTER]
    if len(raw_shards) != len(expected_roster):
        raise ValueError("aggregate requires the complete 11-arm by 5-seed matrix")
    expected_authorization = _execution_authorization(checked_plan, prerequisite)
    admitted = [
        _validate_shard_against_plan(shard, checked_plan, expected_authorization)
        for shard in raw_shards
    ]
    by_identity: dict[tuple[int, str], dict[str, object]] = {}
    for shard in admitted:
        identity = (cast(int, shard["seed"]), cast(str, shard["arm"]))
        if identity in by_identity:
            raise ValueError("aggregate contains a duplicate shard identity")
        by_identity[identity] = shard
    if set(by_identity) != set(expected_roster):
        raise ValueError("aggregate shard roster is incomplete or unexpected")
    ordered = [by_identity[identity] for identity in expected_roster]
    first_result = cast(dict[str, object], ordered[0]["result"])
    first_execution = cast(dict[str, object], first_result["execution_identity"])
    for seed in seeds:
        same_seed = [by_identity[(seed, arm)]["result"] for arm in ARM_ROSTER]
        validate_matched_activation_feature_campaign_results(
            same_seed, development_seeds=seeds
        )
    for shard in ordered[1:]:
        receipt = cast(dict[str, object], shard["result"])
        execution = cast(dict[str, object], receipt["execution_identity"])
        for name in ("dataset_sha256", "source_sha256", "runtime", "n_train"):
            if execution[name] != first_execution[name]:
                raise ValueError(f"aggregate shards disagree on execution identity {name}")
    return ordered


def build_aggregate(
    plan: object, shards: object, *, prerequisite: object | None = None
) -> dict[str, object]:
    """Build an immutable self-contained report from all 55 validated shards."""
    checked_plan = validate_plan(plan)
    _execution_authorization(checked_plan, prerequisite)
    ordered = _admit_complete_matrix(checked_plan, shards, prerequisite)
    aggregate = _unsigned_aggregate(checked_plan, ordered, prerequisite)
    aggregate["aggregate_sha256"] = _digest(aggregate)
    validate_aggregate(aggregate)
    return aggregate


def validate_aggregate(value: object) -> dict[str, object]:
    """Recompute the exact matrix, statistics, resources, and report digest."""
    _json_preflight(value)
    aggregate = _exact_object(value, _AGGREGATE_FIELDS, "aggregate")
    if aggregate["schema"] != AGGREGATE_SCHEMA:
        raise ValueError("unsupported activation/feature aggregate")
    plan = validate_plan(aggregate["plan"])
    if aggregate["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("aggregate plan digest drifted")
    raw_shards = aggregate["shards"]
    if type(raw_shards) is not list:
        raise ValueError("aggregate shards must be an exact list")
    prerequisite = aggregate["prerequisite"]
    _execution_authorization(plan, prerequisite)
    ordered = _admit_complete_matrix(plan, raw_shards, prerequisite)
    expected = _unsigned_aggregate(plan, ordered, prerequisite)
    claimed = aggregate["aggregate_sha256"]
    if not _is_sha256(claimed) or claimed != _digest(expected):
        raise ValueError("aggregate digest drifted")
    expected["aggregate_sha256"] = claimed
    if aggregate != expected:
        raise ValueError("aggregate statistics, roster, resources, or policy drifted")
    return aggregate


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_strict_with_metadata(
    path: Path, *, max_bytes: int
) -> tuple[dict[str, object], os.stat_result]:
    """Read one bounded regular file and return its descriptor-pinned metadata."""
    if type(path) is not type(Path()):
        raise TypeError("path must be an exact Path")
    if type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_AGGREGATE_BYTES:
        raise ValueError("strict JSON byte bound is invalid")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("strict campaign JSON loading requires no-follow file support")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("campaign input must be a uniquely linked regular file")
        if not 0 < opened.st_size <= max_bytes:
            raise ValueError("campaign input exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > max_bytes:
            raise ValueError("campaign input exceeds its byte bound")
        final = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            final.st_nlink != 1
            or len(encoded) != opened.st_size
            or any(getattr(opened, name) != getattr(final, name) for name in stable_fields)
        ):
            raise ValueError("campaign input changed during admission")
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("campaign input must be UTF-8 JSON") from error
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except ValueError:
        raise
    except (OSError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("campaign input is not bounded valid JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise ValueError("campaign input root must be an exact object")
    _json_preflight(value)
    return cast(dict[str, object], value), final


def load_json_strict(path: Path, *, max_bytes: int) -> dict[str, object]:
    """Read one bounded regular-file JSON object through its pinned descriptor."""
    value, _ = _load_json_strict_with_metadata(path, max_bytes=max_bytes)
    return value


def _open_output_parent(path: Path, *, create: bool) -> tuple[Path, int]:
    """Resolve an absolute parent through no-follow directory descriptors."""
    if type(path) is not type(Path()):
        raise TypeError("output must be an exact Path")
    destination = Path(os.path.abspath(os.fspath(path)))
    if destination.name in {"", ".", ".."}:
        raise ValueError("output must name a file")
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise OSError("immutable publication requires Linux no-follow directory support")
    descriptor = os.open(os.path.sep, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in destination.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("output path contains an unsafe directory component")
            if create:
                created = False
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(descriptor)
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return destination, descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _reserved_new_output(path: Path) -> Iterator[tuple[Path, int, str]]:
    """Pin and exclusively reserve one output before any expensive work."""
    _require_execution_authorized()
    destination, parent_fd = _open_output_parent(path, create=True)
    reservation_name = f".{destination.name}.reservation"
    reservation_fd: int | None = None
    reservation_acquired = False
    reservation_identity: tuple[int, int] | None = None
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to replace immutable output: {destination}")
        try:
            reservation_fd = os.open(
                reservation_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise FileExistsError(f"output is already reserved: {destination}") from error
        reservation_acquired = True
        marker = b"asi-activation-feature-campaign-output-reservation-v1\n"
        marker_view = memoryview(marker)
        marker_written = 0
        while marker_written < len(marker_view):
            count = os.write(reservation_fd, marker_view[marker_written:])
            if count <= 0:
                raise OSError("short write while reserving campaign output")
            marker_written += count
        os.fsync(reservation_fd)
        marker_stat = os.fstat(reservation_fd)
        reservation_identity = (marker_stat.st_dev, marker_stat.st_ino)
        os.close(reservation_fd)
        reservation_fd = None
        os.fsync(parent_fd)
        _probe_linkable_tmpfile(parent_fd, destination.name)
        yield destination, parent_fd, reservation_name
    finally:
        if reservation_fd is not None:
            os.close(reservation_fd)
        if reservation_acquired:
            try:
                current_marker = os.stat(
                    reservation_name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if (current_marker.st_dev, current_marker.st_ino) == reservation_identity:
                    os.unlink(reservation_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        os.close(parent_fd)


def _link_unnamed_file(file_fd: int, parent_fd: int, name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", parent_fd, os.fsencode(name), 0x1000) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), name)
        raise OSError(error, os.strerror(error), name)


def _probe_linkable_tmpfile(parent_fd: int, destination_name: str) -> None:
    """Fail before campaign work when this output filesystem cannot publish atomically."""
    if not hasattr(os, "O_TMPFILE"):
        raise OSError("immutable publication requires Linux O_TMPFILE support")
    destination_digest = hashlib.sha256(destination_name.encode("utf-8")).hexdigest()[:16]
    probe_name = f".publication-probe-{destination_digest}"
    probe_fd: int | None = None
    probe_identity: tuple[int, int] | None = None
    linked = False
    try:
        try:
            os.stat(probe_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"publication capability probe already exists: {probe_name}")
        probe_fd = os.open(
            ".",
            os.O_WRONLY | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=parent_fd,
        )
        probe_stat = os.fstat(probe_fd)
        probe_identity = (probe_stat.st_dev, probe_stat.st_ino)
        try:
            _link_unnamed_file(probe_fd, parent_fd, probe_name)
        except OSError as error:
            raise OSError(
                "campaign output filesystem cannot link O_TMPFILE; "
                "choose a filesystem with linkable unnamed-inode support"
            ) from error
        linked = True
        linked_stat = os.stat(probe_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(linked_stat.st_mode)
            or (linked_stat.st_dev, linked_stat.st_ino) != probe_identity
        ):
            raise OSError("campaign output filesystem linked the wrong probe inode")
    finally:
        if linked:
            try:
                current = os.stat(probe_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == probe_identity:
                    os.unlink(probe_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
        if probe_fd is not None:
            os.close(probe_fd)


def _load_json_strict_at(
    parent_fd: int, name: str, *, max_bytes: int
) -> tuple[dict[str, object], bytes]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= max_bytes
        ):
            raise ValueError("published campaign output is not a bounded unique regular file")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        final = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            final.st_nlink != 1
            or len(encoded) != opened.st_size
            or len(encoded) > max_bytes
            or any(getattr(opened, field) != getattr(final, field) for field in stable_fields)
        ):
            raise ValueError("published campaign output changed during strict reread")
        try:
            value = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_object_from_pairs,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("published campaign output is not bounded valid JSON") from error
        if type(value) is not dict:
            raise ValueError("published campaign output root must be an exact object")
        _json_preflight(value)
        return cast(dict[str, object], value), encoded
    finally:
        os.close(descriptor)


def _publication_validator(
    value: dict[str, object],
) -> tuple[Callable[[object], dict[str, object]], int]:
    schema = value.get("schema")
    if type(schema) is not str:
        raise ValueError("published campaign output schema must be an exact string")
    if schema == PLAN_SCHEMA:
        return validate_plan, _MAX_SHARD_BYTES
    if schema == AGGREGATE_SCHEMA:
        return validate_aggregate, _MAX_AGGREGATE_BYTES
    raise ValueError("shard publication requires its plan-bound strict validator")


def _publish_reserved_json(
    target: tuple[Path, int, str],
    value: object,
    *,
    validator: Callable[[object], dict[str, object]] | None = None,
    max_bytes: int | None = None,
) -> Path:
    _require_execution_authorized()
    destination, parent_fd, _reservation_name = target
    _json_preflight(value)
    checked_value = cast(dict[str, object], value)
    if validator is None:
        validator, inferred_max_bytes = _publication_validator(checked_value)
        max_bytes = inferred_max_bytes
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("publication requires an exact positive byte bound")
    validated_input = validator(checked_value)
    if _canonical(validated_input) != _canonical(checked_value):
        raise ValueError("publication validator changed the campaign output")
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if not 0 < len(encoded) <= max_bytes:
        raise ValueError("campaign output exceeds its publication byte bound")
    if not hasattr(os, "O_TMPFILE"):
        raise OSError("immutable publication requires Linux O_TMPFILE support")
    file_fd: int | None = None
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to replace immutable output: {destination}")
        file_fd = os.open(
            ".",
            os.O_WRONLY | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = os.write(file_fd, view[written:])
            if count <= 0:
                raise OSError("short write while publishing campaign output")
            written += count
        os.fsync(file_fd)
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        try:
            _link_unnamed_file(file_fd, parent_fd, destination.name)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace immutable output: {destination}"
            ) from error
        os.fsync(parent_fd)
        reread, reread_bytes = _load_json_strict_at(
            parent_fd, destination.name, max_bytes=max_bytes
        )
        validated = validator(reread)
        if reread_bytes != encoded or _canonical(validated) != _canonical(checked_value):
            raise ValueError("published campaign output differs after strict reread")
        return destination
    finally:
        if file_fd is not None:
            os.close(file_fd)


def write_new_json(path: Path, value: object) -> Path:
    """Publish one create-only JSON file through a descriptor-pinned parent."""
    with _reserved_new_output(path) as target:
        return _publish_reserved_json(target, value)


def summarize_shard_files(
    plan: object, paths: Sequence[Path], *, prerequisite: object | None = None
) -> dict[str, object]:
    """Admit 55 unique regular shard files and build the complete aggregate."""
    checked_plan = validate_plan(plan)
    stage = cast(str, checked_plan["stage"])
    expected_count = len(ARM_ROSTER) * len(_SEEDS[stage])
    if len(paths) != expected_count:
        raise ValueError(f"aggregate requires exactly {expected_count} shard paths")
    seen_files: set[tuple[int, int]] = set()
    total_bytes = 0
    shards: list[dict[str, object]] = []
    for raw_path in paths:
        if type(raw_path) is not type(Path()):
            raise TypeError("shard paths must be exact Paths")
        shard, metadata = _load_json_strict_with_metadata(
            raw_path, max_bytes=_MAX_SHARD_BYTES
        )
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen_files:
            raise ValueError("aggregate shard paths must name unique regular files")
        seen_files.add(identity)
        total_bytes += metadata.st_size
        if total_bytes > _MAX_AGGREGATE_BYTES:
            raise ValueError("aggregate shard inputs exceed their total byte bound")
        validate_shard(shard, checked_plan, prerequisite=prerequisite)
        shards.append(shard)
    return build_aggregate(checked_plan, shards, prerequisite=prerequisite)


def _namespace_path(stage: str, *parts: str) -> Path:
    return _REPO_ROOT / _OUTPUT_NAMESPACES[stage] / Path(*parts)


def _load_plan(path: Path, data_x: np.ndarray, data_y: np.ndarray) -> dict[str, object]:
    return validate_plan(
        load_json_strict(path, max_bytes=_MAX_SHARD_BYTES), data_x=data_x, data_y=data_y
    )


def _print_json(value: object) -> None:
    print(json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="write one exact dataset-bound plan")
    plan_parser.add_argument("--stage", choices=STAGE_ROSTER, required=True)
    plan_parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    plan_parser.add_argument("--output", type=Path)

    shard_parser = subparsers.add_parser(
        "run-shard", help="execute exactly one canonical shard in this Python process"
    )
    shard_parser.add_argument("--stage", choices=STAGE_ROSTER, required=True)
    shard_parser.add_argument("--plan", type=Path)
    shard_parser.add_argument("--arm", choices=ARM_ROSTER, required=True)
    shard_parser.add_argument("--seed", choices=ALL_SEEDS, type=int, required=True)
    shard_parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    shard_parser.add_argument("--cheap-aggregate", type=Path)
    shard_parser.add_argument("--output", type=Path)

    summarize_parser = subparsers.add_parser(
        "summarize", help="strictly combine the complete 55-shard matrix"
    )
    summarize_parser.add_argument("--stage", choices=STAGE_ROSTER, required=True)
    summarize_parser.add_argument("--plan", type=Path)
    summarize_parser.add_argument("--cheap-aggregate", type=Path)
    summarize_parser.add_argument("--output", type=Path)
    summarize_parser.add_argument("shards", type=Path, nargs="+")

    validate_parser = subparsers.add_parser("validate", help="validate a plan/shard/aggregate")
    validate_parser.add_argument("input", type=Path)
    validate_parser.add_argument("--plan", type=Path)
    validate_parser.add_argument("--cheap-aggregate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for plan publication, fresh-process shards, and atomic reports."""
    args = _parser().parse_args(argv)
    if args.command != "validate":
        _require_execution_authorized()
    if args.command == "plan":
        output = args.output or _namespace_path(args.stage, "plan.json")
        with _reserved_new_output(output) as target:
            x, y = load_mnist_train(args.data_home)
            plan = build_plan(args.stage, x, y)
            _publish_reserved_json(target, plan)
        return 0
    if args.command == "run-shard":
        output = args.output or _namespace_path(
            args.stage, "shards", f"seed-{args.seed}.{args.arm}.json"
        )
        with _reserved_new_output(output) as target:
            x, y = load_mnist_train(args.data_home)
            plan_path = args.plan or _namespace_path(args.stage, "plan.json")
            plan = _load_plan(plan_path, x, y)
            if plan["stage"] != args.stage:
                raise ValueError("CLI stage and loaded plan disagree")
            prerequisite = (
                None
                if args.cheap_aggregate is None
                else validate_aggregate(
                    load_json_strict(args.cheap_aggregate, max_bytes=_MAX_AGGREGATE_BYTES)
                )
            )
            shard = _build_shard_authorized(
                plan,
                x,
                y,
                arm=args.arm,
                seed=args.seed,
                prerequisite=prerequisite,
                _capability=_EXECUTION_CAPABILITY,
            )
            _publish_reserved_json(
                target,
                shard,
                validator=lambda value: validate_shard(
                    value, plan, prerequisite=prerequisite
                ),
                max_bytes=_MAX_SHARD_BYTES,
            )
        return 0
    if args.command == "summarize":
        output = args.output or _namespace_path(args.stage, "aggregate.json")
        with _reserved_new_output(output) as target:
            plan_path = args.plan or _namespace_path(args.stage, "plan.json")
            plan = validate_plan(load_json_strict(plan_path, max_bytes=_MAX_SHARD_BYTES))
            if plan["stage"] != args.stage:
                raise ValueError("CLI stage and loaded plan disagree")
            prerequisite = (
                None
                if args.cheap_aggregate is None
                else validate_aggregate(
                    load_json_strict(args.cheap_aggregate, max_bytes=_MAX_AGGREGATE_BYTES)
                )
            )
            aggregate = summarize_shard_files(
                plan, args.shards, prerequisite=prerequisite
            )
            _publish_reserved_json(target, aggregate)
        return 0
    if args.command == "validate":
        value = load_json_strict(args.input, max_bytes=_MAX_AGGREGATE_BYTES)
        schema = value.get("schema")
        if schema == PLAN_SCHEMA:
            validated: object = validate_plan(value)
        elif schema == SHARD_SCHEMA:
            if args.plan is None:
                raise ValueError("shard validation requires --plan")
            plan = validate_plan(load_json_strict(args.plan, max_bytes=_MAX_SHARD_BYTES))
            prerequisite = (
                None
                if args.cheap_aggregate is None
                else validate_aggregate(
                    load_json_strict(args.cheap_aggregate, max_bytes=_MAX_AGGREGATE_BYTES)
                )
            )
            validated = validate_shard(value, plan, prerequisite=prerequisite)
        elif schema == AGGREGATE_SCHEMA:
            validated = validate_aggregate(value)
        else:
            raise ValueError("input is not an activation/feature plan, shard, or aggregate")
        _print_json(
            {
                "valid": True,
                "schema": schema,
                "sha256": _digest(validated),
                "permanently_nonpromoting": True,
            }
        )
        return 0
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())
