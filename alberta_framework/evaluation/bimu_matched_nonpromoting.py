"""Prospective, permanently nonpromoting matched BiMU development plan."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jax
import numpy as np

from alberta_framework.benchmarks.bimu import BiMUConfig, _dataset_sha256

PLAN_SCHEMA: Final = "asi.bimu.matched-development-plan.v3"
OUTPUT_NAMESPACE: Final = Path("outputs/bimu_matched/development.v1")
EXECUTION_AUTHORIZED: Final = False
AUTHORIZATION_TRANSITION_APPROVED: Final = False
_DIGEST = "85c681c2f5fc5c274870b30c9accb3d2a6e9eb90a4575a2bf1ccca64f58b6227"
FROZEN_PLAN_SHA256: Final = "ab2cb84f4e93e7e3fed2c21a2e450b67ec917dce496701646d3040489f9587bd"

INVALID_PRIOR_ATTEMPT: Final[Mapping[str, object]] = MappingProxyType(
    {
        "pull_request": 1686,
        "head_commit": "86a67df39781bba77e1a2c47451f646205daee65",
        "seed": 23,
        "status": "invalid_never_merged",
        "reason": (
            "colliding RNG domains, unpinned PRNG, majority-vote inference, and an immediate-task "
            "metric mislabeled as the paper final-model late-five metric"
        ),
        "result_retained": False,
        "seed_reuse_allowed": False,
        "unmerged_result_sha256": (
            "9b11c3944379323e33ee067cf80a9f4d772a3af4080f9718cec3b6e1d1e91a23",
            "00faf161ead42d11c8daed668ba96a905ef25baf18b0c77b04bc08e4435c4fa7",
            "0f665cbddf209422456d68835f835c8302a632372e59a0e2518297a41c30a5cb",
        ),
        "unmerged_artifact_file_sha256": (
            "0e313a49c5b2e5fb3b7a4c61c6d2618815432dfc24aac30c64b16777ed1328cb",
            "7da4d6e0411546a39d431bf1d3b6c47372c7c634c372f7133f15481851132daa",
            "2d3a05db3ba2b8af50d522ba13564d82999829f340f426f0f4b1b1389607ade0",
        ),
    }
)


def _invalid_prior_attempt_payload() -> dict[str, object]:
    """Return the exact JSON form of the immutable invalid-attempt record."""
    return {
        key: list(value) if type(value) is tuple else value
        for key, value in INVALID_PRIOR_ATTEMPT.items()
    }


def _config(
    *, memory_window: int | None, input_dim: int = 784, n_classes: int = 10, examples: int = 256
) -> BiMUConfig:
    return BiMUConfig(
        input_dim=input_dim,
        hidden_units=32,
        n_classes=n_classes,
        n_tasks=5,
        train_examples_per_task=examples,
        test_examples_per_task=examples,
        train_samples=2,
        test_samples=3,
        query_samples=3,
        temperature=1.0,
        likelihood_multiplier=161.3,
        kl_multiplier=3.76,
        alpha_max=0.0023,
        memory_window=memory_window,
        gradient_scale=4.9,
        query_threshold=0.0,
    )


@dataclass(frozen=True)
class BiMUMatchedDevelopmentPlan:
    seeds: tuple[int, ...]
    arm_names: tuple[str, str]
    control_config: BiMUConfig
    candidate_config: BiMUConfig
    dataset_sha256: str
    dataset_selection: str

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or len(self.seeds) != 3:
            raise ValueError("seeds must be one exact three-seed tuple")
        if any(type(seed) is not int or not 0 <= seed <= 2**31 - 1 for seed in self.seeds):
            raise ValueError("seeds must be exact signed-int32 nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds) or 23 in self.seeds:
            raise ValueError("seeds must be distinct and must not reuse the invalid attempt")
        if type(self.arm_names) is not tuple or self.arm_names != ("memory_off", "bimu"):
            raise ValueError("arm_names must be the exact mechanism-off/candidate pair")
        if (
            type(self.control_config) is not BiMUConfig
            or type(self.candidate_config) is not BiMUConfig
        ):
            raise ValueError("arm configs must be exact BiMUConfig values")
        control = BiMUConfig(**self.control_config.__dict__)
        candidate = BiMUConfig(**self.candidate_config.__dict__)
        control_payload = control.to_protocol_payload()
        candidate_payload = candidate.to_protocol_payload()
        differences = {
            key for key in control_payload if control_payload[key] != candidate_payload[key]
        }
        if differences != {"memory_window"} or control.memory_window is not None:
            raise ValueError("matched arms may differ only by the BiMU memory window")
        if candidate.memory_window is None:
            raise ValueError("candidate memory window must be enabled")
        if type(self.dataset_sha256) is not str or len(self.dataset_sha256) != 64:
            raise ValueError("dataset_sha256 must be one lowercase SHA-256 digest")
        if any(character not in "0123456789abcdef" for character in self.dataset_sha256):
            raise ValueError("dataset_sha256 must be one lowercase SHA-256 digest")
        if type(self.dataset_selection) is not str or self.dataset_selection != (
            "OpenML mnist_784 v1 canonical train split after [-1,1] scaling; first 256 rows "
            "train and last 256 rows disjoint development test"
        ):
            raise ValueError("dataset selection must remain exact")
        object.__setattr__(self, "control_config", control)
        object.__setattr__(self, "candidate_config", candidate)


FROZEN_BIMU_MATCHED_PLAN: Final = BiMUMatchedDevelopmentPlan(
    seeds=(157001, 157002, 157003),
    arm_names=("memory_off", "bimu"),
    control_config=_config(memory_window=None),
    candidate_config=_config(memory_window=128),
    dataset_sha256=_DIGEST,
    dataset_selection=(
        "OpenML mnist_784 v1 canonical train split after [-1,1] scaling; first 256 rows "
        "train and last 256 rows disjoint development test"
    ),
)


def _test_plan(*, input_dim: int, n_classes: int, examples: int) -> BiMUMatchedDevelopmentPlan:
    data = np.arange(8 * input_dim, dtype=np.float32).reshape(8, input_dim) / 32.0
    labels = np.arange(8, dtype=np.int32) % n_classes
    return BiMUMatchedDevelopmentPlan(
        seeds=(157001, 157002, 157003),
        arm_names=("memory_off", "bimu"),
        control_config=_config(
            memory_window=None, input_dim=input_dim, n_classes=n_classes, examples=examples
        ),
        candidate_config=_config(
            memory_window=128, input_dim=input_dim, n_classes=n_classes, examples=examples
        ),
        dataset_sha256=_dataset_sha256(data[:4], labels[:4], data[4:], labels[4:]),
        dataset_selection=FROZEN_BIMU_MATCHED_PLAN.dataset_selection,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _authorization_identity() -> dict[str, bool]:
    return {
        "execution_authorized": EXECUTION_AUTHORIZED,
        "authorization_transition_approved": AUTHORIZATION_TRANSITION_APPROVED,
    }


def _plan_payload(plan: BiMUMatchedDevelopmentPlan) -> dict[str, object]:
    checked = BiMUMatchedDevelopmentPlan(**plan.__dict__)
    config = checked.candidate_config
    observations = config.n_tasks * config.train_examples_per_task
    label_queries = observations
    model_forward_queries = (
        observations * config.query_samples
        + label_queries * config.train_samples
        + 5 * config.test_examples_per_task * config.test_samples
    )
    persistent_bytes = config.trainable_scalar_count * np.dtype(np.float32).itemsize + (
        2 * np.dtype(np.uint32).itemsize
    )
    dataset_bytes = (
        config.train_examples_per_task * (config.input_dim + 1) * 4
        + config.test_examples_per_task * (config.input_dim + 1) * 4
    )
    expected_counters = {
        "environment_steps": observations,
        "observations": observations,
        "label_queries": label_queries,
        "optimizer_seen": observations,
        "model_forward_queries": model_forward_queries,
        "optimizer_updates": observations,
    }
    execution_passes_per_shard = 2
    campaign_shards = len(checked.seeds) * len(checked.arm_names)
    return {
        "schema": PLAN_SCHEMA,
        "seeds": list(checked.seeds),
        "arm_names": list(checked.arm_names),
        "control_config": checked.control_config.to_protocol_payload(),
        "candidate_config": checked.candidate_config.to_protocol_payload(),
        "dataset_sha256": checked.dataset_sha256,
        "dataset_selection": checked.dataset_selection,
        "prior_invalid_attempts": [_invalid_prior_attempt_payload()],
        "matched_axes": [
            "seed",
            "dataset",
            "schedule",
            "observations",
            "label_queries",
            "optimizer_seen",
            "model_forward_queries",
            "initial_state",
        ],
        "expected_counters_per_arm": expected_counters,
        "expected_resources_per_arm": {
            "trainable_scalar_count": config.trainable_scalar_count,
            "parameter_numeric_bytes": config.trainable_scalar_count * 4,
            "optimizer_state_numeric_bytes": 8,
            "initial_persistent_numeric_bytes": persistent_bytes,
            "final_persistent_numeric_bytes": persistent_bytes,
            "dataset_numeric_bytes": dataset_bytes,
            "timing_qualified": False,
            "aggregate_working_set_bytes_claimed": False,
            "numeric_resource_ceiling_bytes": 256 * 1024 * 1024,
        },
        "transaction_execution_accounting": {
            "campaign_shards": campaign_shards,
            "initial_execution_dispatches_per_shard": 1,
            "strict_reexecution_dispatches_per_shard": 1,
            "total_execution_dispatches_per_shard": execution_passes_per_shard,
            "total_campaign_execution_dispatches": (
                campaign_shards * execution_passes_per_shard
            ),
            "per_shard_counters_including_strict_reexecution": {
                field: value * execution_passes_per_shard
                for field, value in expected_counters.items()
            },
            "campaign_counters_including_strict_reexecution": {
                field: value * execution_passes_per_shard * campaign_shards
                for field, value in expected_counters.items()
            },
            "dataset_loads_per_shard_process": 1,
            "validated_array_tuple_reused_for_strict_reexecution": True,
            "strict_reexecution_timing_retained": False,
        },
        "comparison_scope": {
            "paper_comparable": False,
            "development_slice": "five tasks, 256 train and 256 test examples, width 32",
            "official_configuration_also_represented_by_runner": True,
            "claim": "bounded mechanism-on versus exact memory-off development comparison",
        },
        "primary_metric": "paper_late_five_test_accuracy",
        "secondary_metric": "asi_whole_stream_online_accuracy",
        "paired_outcome_rule": {
            "schema": "asi.bimu.paired-outcome-rule.v1",
            "metric": "paper_late_five_test_accuracy",
            "supported": "all_three_paired_deltas_strictly_positive",
            "rejected": "all_three_paired_deltas_nonpositive",
            "otherwise": "inconclusive",
            "ties_are_positive": False,
            "secondary_metric_affects_outcome": False,
        },
        "output_namespace": str(OUTPUT_NAMESPACE),
        "seed_status": {
            "consumed_for_promotion": True,
            "retained_matched_result_exists": False,
            "reason": "the literal development roster is publicly exposed",
        },
        "authorization": _authorization_identity(),
    }


def frozen_plan_payload() -> dict[str, object]:
    """Return the literal plan only when its preregistered digest still matches."""

    payload = _plan_payload(FROZEN_BIMU_MATCHED_PLAN)
    observed = hashlib.sha256(_canonical(payload)).hexdigest()
    if observed != FROZEN_PLAN_SHA256:
        raise RuntimeError("frozen BiMU plan payload drifted from its literal digest")
    return payload


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_identity() -> dict[str, str]:
    root = _repository_root()
    paths = (
        Path("alberta_framework/benchmarks/bimu.py"),
        Path("alberta_framework/benchmarks/upgd_ipmnist.py"),
        Path("alberta_framework/evaluation/bimu_matched_nonpromoting.py"),
        Path("alberta_framework/evaluation/bimu_matched_campaign.py"),
        Path("pyproject.toml"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _runtime_identity() -> dict[str, object]:
    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": device.id,
            "process_index": device.process_index,
        }
        for device in jax.devices()
    ]
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
        "schema": "asi.bimu.matched-runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
        },
        "backend": jax.default_backend(),
        "devices": devices,
        "jax_config": {
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_default_prng_impl": jax.config.jax_default_prng_impl,
            "jax_disable_jit": jax.config.jax_disable_jit,
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_numpy_dtype_promotion": jax.config.jax_numpy_dtype_promotion.value,
            "jax_numpy_rank_promotion": jax.config.jax_numpy_rank_promotion,
            "jax_random_seed_offset": jax.config.jax_random_seed_offset,
            "jax_threefry_partitionable": jax.config.jax_threefry_partitionable,
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _dependency_identity() -> dict[str, object]:
    root = _repository_root()
    return {
        "schema": "asi.bimu.matched-dependencies.v1",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
        },
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }
