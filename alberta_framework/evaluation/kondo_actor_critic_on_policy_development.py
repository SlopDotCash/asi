# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Strict closed-loop on-policy Kondo actor/critic development evaluator.

Four independent arms collect each environment batch under an immutable actor
revision and update only after collection closes.  Ordinary full, capacity-
matched uniform sparse, paper top-k Kondo, and top-k plus a minimum random
reserve each sample their own categorical actions from their own current
policy.  The evaluator supplies typed Threefry common-random-number uniforms;
only exogenous randomness is paired, and no trajectory equality is assumed.

Every arm performs exactly one actor update opportunity and one full-batch
protected learner update per batch.  Baseline, critic, representation,
world-model, and safety/guardrail learners retain the complete arm-specific
trajectory.  Rare failure rows are forced into every actor backward and remain
present in the full protected backward.  Actions, exact behavior log
probabilities, frozen revisions, environment parents, and common-schedule
bindings are retained in a causal hash chain.

This module is a finite L0 development diagnostic.  Reports are always
``not_assessed`` and make no efficacy, compute, safety, evidence, or promotion
claim.  It writes no files.  SHA-256 fields are unkeyed integrity/source
bindings, not authenticity.  Exact source/runtime-bound causal replay is the
authoritative validator.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.extend import backend as jax_backend

from alberta_framework.core.kondo_gate import KondoGateConfig
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorBackwardBatch,
    KondoActorBackwardSummary,
    KondoActorParameters,
    KondoActorProtectedInputs,
    KondoSparseActor,
    KondoSparseActorBatch,
    KondoSparseActorConfig,
    KondoSparseActorResult,
    KondoSparseActorState,
    kondo_actor_backward_kernel,
)
from alberta_framework.evaluation.kondo_actor_critic_replay_development import (
    ReplayProtectedBackwardResult,
    ReplayProtectedBatch,
    ReplayProtectedParameters,
    ReplayProtectedState,
    replay_protected_backward_kernel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

KONDO_ON_POLICY_CONFIG_SCHEMA = "alberta.kondo-actor-critic-on-policy.config.v2"
KONDO_ON_POLICY_PROTOCOL_SCHEMA = "alberta.kondo-actor-critic-on-policy.protocol.v2"
KONDO_ON_POLICY_REPORT_SCHEMA = "alberta.kondo-actor-critic-on-policy.report.v2"
KONDO_ON_POLICY_CHECKPOINT_SCHEMA = "alberta.kondo-actor-critic-on-policy.checkpoint.v2"

DEVELOPMENT_STATUS = "not_assessed"
ASSESSMENT_STATUS = "not_assessed"
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES = False
CHECKPOINT_HOST_ONLY = True

ArmName = Literal[
    "ordinary_full",
    "uniform_sparse",
    "kondo_top_k",
    "kondo_top_k_reserve",
]
ARM_ORDER: tuple[ArmName, ...] = (
    "ordinary_full",
    "uniform_sparse",
    "kondo_top_k",
    "kondo_top_k_reserve",
)
SPARSE_ARM_ORDER: tuple[ArmName, ...] = (
    "uniform_sparse",
    "kondo_top_k",
    "kondo_top_k_reserve",
)

_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2_147_483_647
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_BATCH_SIZE = 128
_MAX_PHASE_BATCHES = 16
_MAX_DIMENSION = 128
_MAX_REPRESENTATION_DIM = 64
_MAX_TRACE_SCALAR_SLOTS = 2_000_000
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_MAX_REPORT_BYTES = 96 * 1024 * 1024

_SOURCE_PATHS = (
    Path("alberta_framework/core/kondo_gate.py"),
    Path("alberta_framework/core/kondo_sparse_actor.py"),
    Path("alberta_framework/evaluation/kondo_actor_critic_replay_development.py"),
    Path("alberta_framework/evaluation/kondo_actor_critic_on_policy_development.py"),
)

_LIMITATIONS = (
    "development diagnostics only; every status and verdict is not_assessed",
    "closed-loop trajectories may and generally do diverge across actor arms",
    "common randomness pairs exogenous uniforms only, not realized trajectories",
    "minimum-random-reserve is an Alberta extension and is not paper Kondo",
    "logical row-slot counts are not measured FLOPs, latency, memory, or energy",
    "rare-failure coverage is descriptive and is not a physical-safety claim",
    "on-policy collection does not establish policy-gradient or DG efficacy",
    "SHA-256 provides integrity and source binding, not keyed authenticity",
    "no result grants policy, guardrail, deployment, evidence, or promotion authority",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual),
                    cast(list[object], expected),
                    strict=True,
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        upper = "unbounded" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {upper}]")
    return value


def _normal_float32(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real number")
    number = float(np.float32(float(value)))
    if not math.isfinite(number) or number < _FLOAT32_TINY:
        raise ValueError(f"{name} must be a positive normal float32")
    return number


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kondo_on_policy_source_manifest(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the complete repository source closure of this development lane."""

    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


def kondo_on_policy_runtime_identity() -> dict[str, object]:
    """Return observable non-secret runtime provenance."""

    devices = tuple(jax.devices())
    backend = jax_backend.get_backend()
    return {
        "identity_scope": (
            "observable-nonsecret-python-jax-xla-device-and-config-fields; "
            "exact-deterministic-replay-remains-authoritative"
        ),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "chex_version": version("chex"),
        "jax_version": str(jax.__version__),
        "jaxlib_version": version("jaxlib"),
        "numpy_version": str(np.__version__),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "backend": str(backend.platform),
        "backend_platform_version": str(backend.platform_version),
        "device_count": len(devices),
        "local_device_count": int(jax.local_device_count()),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [str(device.device_kind) for device in devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "jax_enable_checks": bool(jax.config.jax_enable_checks),
    }


@dataclasses.dataclass(frozen=True)
class KondoActorCriticOnPolicyConfig:
    """Finite A/B/A closed-loop protocol without assessment thresholds."""

    seed: int = 161
    batch_size: int = 8
    batches_per_phase: int = 1
    actor_feature_dim: int = 4
    hidden_dim: int = 6
    context_dim: int = 4
    critic_dim: int = 3
    safety_dim: int = 3
    representation_dim: int = 3
    action_count: int = 2
    target_rate: float = 0.25
    reserve_count: int = 1
    actor_learning_rate: float = 0.02
    protected_learning_rate: float = 0.015

    def __post_init__(self) -> None:
        _exact_int(self.seed, name="seed", maximum=_UINT32_MAX)
        _exact_int(
            self.batch_size,
            name="batch_size",
            minimum=8,
            maximum=_MAX_BATCH_SIZE,
        )
        _exact_int(
            self.batches_per_phase,
            name="batches_per_phase",
            minimum=1,
            maximum=_MAX_PHASE_BATCHES,
        )
        for name in (
            "actor_feature_dim",
            "hidden_dim",
            "context_dim",
            "critic_dim",
            "safety_dim",
        ):
            _exact_int(
                getattr(self, name),
                name=name,
                minimum=1,
                maximum=_MAX_DIMENSION,
            )
        _exact_int(
            self.representation_dim,
            name="representation_dim",
            minimum=1,
            maximum=_MAX_REPRESENTATION_DIM,
        )
        if self.actor_feature_dim < 2 or self.context_dim < 2:
            raise ValueError("actor_feature_dim and context_dim must be at least two")
        if type(self.action_count) is not int or self.action_count != 2:
            raise ValueError("action_count must be exactly two")
        target_rate = _normal_float32(self.target_rate, name="target_rate")
        if target_rate >= 1.0:
            raise ValueError("target_rate must be below one")
        object.__setattr__(self, "target_rate", target_rate)
        object.__setattr__(
            self,
            "actor_learning_rate",
            _normal_float32(self.actor_learning_rate, name="actor_learning_rate"),
        )
        object.__setattr__(
            self,
            "protected_learning_rate",
            _normal_float32(
                self.protected_learning_rate,
                name="protected_learning_rate",
            ),
        )
        _exact_int(
            self.reserve_count,
            name="reserve_count",
            minimum=1,
            maximum=self.batch_size,
        )
        if self.sparse_capacity >= self.batch_size:
            raise ValueError("sparse capacity must remain below batch_size")
        if 1 + self.reserve_count > self.sparse_capacity:
            raise ValueError("sparse capacity must fit forced failure plus reserve")
        if self.total_batches * self.batch_size > _INT32_MAX:
            raise ValueError("closed-loop accounting exceeds signed int32")
        scalar_slots = (
            self.total_batches
            * len(ARM_ORDER)
            * self.batch_size
            * (
                self.actor_feature_dim
                + 2 * self.context_dim
                + self.critic_dim
                + self.safety_dim
                + self.representation_dim
                + 10
            )
        )
        if scalar_slots > _MAX_TRACE_SCALAR_SLOTS:
            raise ValueError("closed-loop trace exceeds the scalar-slot cap")

    @property
    def total_batches(self) -> int:
        return 3 * self.batches_per_phase

    @property
    def sparse_capacity(self) -> int:
        count = int(np.rint(np.float32(self.target_rate) * np.float32(self.batch_size)))
        return max(1, min(self.batch_size, count))

    def actor_config(self, *, reserve: bool) -> KondoSparseActorConfig:
        return KondoSparseActorConfig(
            feature_dim=self.actor_feature_dim,
            hidden_dim=self.hidden_dim,
            action_count=self.action_count,
            critic_dim=self.critic_dim,
            safety_dim=self.safety_dim,
            learning_rate=self.actor_learning_rate,
            gate=KondoGateConfig(
                batch_size=self.batch_size,
                mode="top_k_rate",
                target_rate=self.target_rate,
                minimum_uniform_keep=self.reserve_count if reserve else 0,
                max_screenings=self.total_batches,
            ),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": KONDO_ON_POLICY_CONFIG_SCHEMA,
            "type": "KondoActorCriticOnPolicyConfig",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_per_phase": self.batches_per_phase,
            "total_batches": self.total_batches,
            "actor_feature_dim": self.actor_feature_dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "critic_dim": self.critic_dim,
            "safety_dim": self.safety_dim,
            "representation_dim": self.representation_dim,
            "action_count": self.action_count,
            "target_rate": self.target_rate,
            "sparse_capacity": self.sparse_capacity,
            "reserve_count": self.reserve_count,
            "actor_learning_rate": self.actor_learning_rate,
            "protected_learning_rate": self.protected_learning_rate,
            "seed_role": "development-common-exogenous-threefry-only",
            "evidence_seed": None,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> KondoActorCriticOnPolicyConfig:
        expected_fields = set(cls().to_config())
        if set(payload) != expected_fields:
            raise ValueError("Kondo on-policy config fields differ")
        fixed: dict[str, object] = {
            "schema": KONDO_ON_POLICY_CONFIG_SCHEMA,
            "type": "KondoActorCriticOnPolicyConfig",
            "seed_role": "development-common-exogenous-threefry-only",
            "evidence_seed": None,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
        }
        for name, expected in fixed.items():
            if not _strict_json_equal(payload.get(name), expected):
                raise ValueError(f"Kondo on-policy config {name} is invalid")
        integer_names = (
            "seed",
            "batch_size",
            "batches_per_phase",
            "actor_feature_dim",
            "hidden_dim",
            "context_dim",
            "critic_dim",
            "safety_dim",
            "representation_dim",
            "action_count",
            "reserve_count",
        )
        for name in integer_names:
            if type(payload[name]) is not int:
                raise ValueError(f"Kondo on-policy config {name} must be an integer")
        for name in (
            "target_rate",
            "actor_learning_rate",
            "protected_learning_rate",
        ):
            if type(payload[name]) is not float:
                raise ValueError(f"Kondo on-policy config {name} must be a float")
        result = cls(
            seed=cast(int, payload["seed"]),
            batch_size=cast(int, payload["batch_size"]),
            batches_per_phase=cast(int, payload["batches_per_phase"]),
            actor_feature_dim=cast(int, payload["actor_feature_dim"]),
            hidden_dim=cast(int, payload["hidden_dim"]),
            context_dim=cast(int, payload["context_dim"]),
            critic_dim=cast(int, payload["critic_dim"]),
            safety_dim=cast(int, payload["safety_dim"]),
            representation_dim=cast(int, payload["representation_dim"]),
            action_count=cast(int, payload["action_count"]),
            target_rate=cast(float, payload["target_rate"]),
            reserve_count=cast(int, payload["reserve_count"]),
            actor_learning_rate=cast(float, payload["actor_learning_rate"]),
            protected_learning_rate=cast(float, payload["protected_learning_rate"]),
        )
        if (
            payload.get("total_batches") != result.total_batches
            or payload.get("sparse_capacity") != result.sparse_capacity
        ):
            raise ValueError("Kondo on-policy derived config fields differ")
        if not _strict_json_equal(result.to_config(), dict(payload)):
            raise ValueError("Kondo on-policy config is noncanonical")
        return result


def kondo_on_policy_protocol(
    config: KondoActorCriticOnPolicyConfig,
) -> dict[str, object]:
    """Return the nonpromoting closed-loop mechanics contract."""

    return {
        "schema": KONDO_ON_POLICY_PROTOCOL_SCHEMA,
        "type": "KondoActorCriticOnPolicyProtocol",
        "arms": list(ARM_ORDER),
        "sparse_arms": list(SPARSE_ARM_ORDER),
        "phase_order": ["A1", "B", "A2"],
        "closed_loop_control": True,
        "on_policy": True,
        "actions_sampled_from_each_arms_own_policy": True,
        "behavior_log_probability_available": True,
        "behavior_log_probability_exact": True,
        "importance_correction_applied": False,
        "importance_correction_required_for_frozen_revision_batch": False,
        "actor_revision_immutable_within_batch": True,
        "actor_updates_only_at_batch_boundaries": True,
        "actor_update_opportunities_per_batch_per_arm": 1,
        "protected_updates_per_batch_per_arm": 1,
        "common_schedule_rng_impl": "threefry2x32",
        "common_schedule_pairs_exogenous_randomness_only": True,
        "uniform_control_allocation_randomness_paired_across_arms": False,
        "kondo_gate_randomness_is_arm_internal": True,
        "trajectory_equality_assumed": False,
        "trajectory_equality_required": False,
        "ordinary_actor_backward_shape": config.batch_size,
        "sparse_actor_backward_shape": config.sparse_capacity,
        "protected_backward_shape": config.batch_size,
        "uniform_and_kondo_backward_capacity_equal": True,
        "protected_channels": [
            "baseline",
            "critic",
            "representation",
            "world_model",
            "safety_guardrail",
        ],
        "protected_learning_full_batch": True,
        "protected_learning_shapes_equal_across_arms": True,
        "protected_learning_values_equal_across_arms_required": False,
        "rare_failure_rows_forced_into_actor_backward": True,
        "rare_failure_rows_present_in_full_protected_backward": True,
        "paper_delight": "advantage-times-on-policy-selected-action-surprisal",
        "executed_actor_backward_mask_semantics": (
            "gradient-contribution-entered-executed-actor-backward"
        ),
        "sparks_joy_scope": "KondoSparseActorResult-only",
        "manual_kernel_arms_are_kondo_sparse_actor_transactions": False,
        "ordinary_full_delight_selection_claimed": False,
        "reserve_arm_role": "nonpaper-minimum-random-reserve-within-fixed-capacity",
        "host_screen_and_gather_jittable": False,
        "collection_and_fixed_backward_kernels_jittable": True,
        "wall_clock_measured": False,
        "measured_flops": False,
        "performance_claimed": False,
        "compute_benefit_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "guardrail_authority": False,
        "output_writes": False,
        "evidence_seed": None,
        "thresholds": [],
        "assessment_status": ASSESSMENT_STATUS,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
    }


def _array_payload(array: Array | npt.NDArray[Any]) -> dict[str, object]:
    host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
    return {
        "dtype": host.dtype.str,
        "shape": list(host.shape),
        "data_hex": host.tobytes().hex(),
        "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
    }


def _array_bits_equal(left: Array, right: Array) -> bool:
    lhs = np.ascontiguousarray(np.asarray(jax.device_get(left)))
    rhs = np.ascontiguousarray(np.asarray(jax.device_get(right)))
    return lhs.dtype == rhs.dtype and lhs.shape == rhs.shape and lhs.tobytes() == rhs.tobytes()


def _tree_all_finite(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(jax.device_get(leaf)))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_l2(tree: object) -> float:
    total = sum(
        float(np.sum(np.square(np.asarray(jax.device_get(leaf), dtype=np.float64))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return float(math.sqrt(total))


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(cast(Any, leaf).shape), str(cast(Any, leaf).dtype))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_bits_equal(left: object, right: object) -> bool:
    lhs = jax.tree_util.tree_leaves(left)
    rhs = jax.tree_util.tree_leaves(right)
    return len(lhs) == len(rhs) and all(
        _array_bits_equal(cast(Array, a), cast(Array, b))
        for a, b in zip(lhs, rhs, strict=True)
    )


def _deterministic_values(size: int, *, phase: float, scale: float) -> Array:
    index: npt.NDArray[np.float32] = np.arange(size, dtype=np.float32)
    raw = np.sin(index * np.float32(0.173) + np.float32(phase)) * np.float32(scale)
    return jnp.asarray(raw.astype(np.float32))


def _initial_actor_parameters(
    config: KondoActorCriticOnPolicyConfig,
) -> KondoActorParameters:
    return KondoActorParameters(
        hidden_weight=_deterministic_values(
            config.actor_feature_dim * config.hidden_dim,
            phase=0.1,
            scale=0.055,
        ).reshape(config.actor_feature_dim, config.hidden_dim),
        hidden_bias=_deterministic_values(
            config.hidden_dim,
            phase=0.2,
            scale=0.012,
        ),
        output_weight=_deterministic_values(
            config.hidden_dim * config.action_count,
            phase=0.3,
            scale=0.06,
        ).reshape(config.hidden_dim, config.action_count),
        output_bias=_deterministic_values(
            config.action_count,
            phase=0.4,
            scale=0.018,
        ),
    )


def _initial_protected_parameters(
    config: KondoActorCriticOnPolicyConfig,
) -> ReplayProtectedParameters:
    return ReplayProtectedParameters(
        baseline_weight=_deterministic_values(
            config.context_dim,
            phase=0.5,
            scale=0.04,
        ),
        baseline_bias=jnp.asarray(0.0, dtype=jnp.float32),
        critic_weight=_deterministic_values(
            config.critic_dim * config.action_count,
            phase=0.6,
            scale=0.05,
        ).reshape(config.critic_dim, config.action_count),
        critic_bias=_deterministic_values(
            config.action_count,
            phase=0.7,
            scale=0.01,
        ),
        representation_weight=_deterministic_values(
            config.context_dim * config.representation_dim,
            phase=0.8,
            scale=0.045,
        ).reshape(config.context_dim, config.representation_dim),
        representation_bias=_deterministic_values(
            config.representation_dim,
            phase=0.9,
            scale=0.01,
        ),
        model_weight=_deterministic_values(
            config.representation_dim * config.context_dim,
            phase=1.0,
            scale=0.04,
        ).reshape(config.representation_dim, config.context_dim),
        model_bias=_deterministic_values(
            config.context_dim,
            phase=1.1,
            scale=0.01,
        ),
        safety_weight=_deterministic_values(
            config.safety_dim,
            phase=1.2,
            scale=0.035,
        ),
        safety_bias=jnp.asarray(-0.1, dtype=jnp.float32),
    )


@chex.dataclass(frozen=True)
class OnPolicyEnvironmentParameters:
    """Fixed dynamics and observation projections."""

    actor_projection: Array
    critic_projection: Array
    safety_projection: Array
    representation_projection: Array
    action_effect: Array
    regime_effect: Array


def _environment_parameters(
    config: KondoActorCriticOnPolicyConfig,
) -> OnPolicyEnvironmentParameters:
    return OnPolicyEnvironmentParameters(
        actor_projection=_deterministic_values(
            config.context_dim * config.actor_feature_dim,
            phase=1.3,
            scale=0.32,
        ).reshape(config.context_dim, config.actor_feature_dim),
        critic_projection=_deterministic_values(
            config.context_dim * config.critic_dim,
            phase=1.4,
            scale=0.28,
        ).reshape(config.context_dim, config.critic_dim),
        safety_projection=_deterministic_values(
            config.context_dim * config.safety_dim,
            phase=1.5,
            scale=0.26,
        ).reshape(config.context_dim, config.safety_dim),
        representation_projection=_deterministic_values(
            config.context_dim * config.representation_dim,
            phase=1.6,
            scale=0.3,
        ).reshape(config.context_dim, config.representation_dim),
        action_effect=jnp.stack(
            (
                _deterministic_values(
                    config.context_dim,
                    phase=1.7,
                    scale=-0.22,
                ),
                _deterministic_values(
                    config.context_dim,
                    phase=1.7,
                    scale=0.22,
                ),
            )
        ),
        regime_effect=_deterministic_values(
            config.context_dim,
            phase=1.8,
            scale=0.14,
        ),
    )


@chex.dataclass(frozen=True)
class OnPolicyEnvironmentState:
    """Arm-owned closed-loop environment state."""

    latent: Array
    last_action: Array
    step_count: Array
    cumulative_reward: Array
    failure_count: Array


@chex.dataclass(frozen=True)
class OnPolicyCommonSchedule:
    """Evaluator-owned exogenous Threefry schedule shared across arms."""

    action_uniforms: Array
    transition_uniforms: Array
    reward_uniforms: Array
    failure_mask: Array
    key_words: Array


@chex.dataclass(frozen=True)
class OnPolicyCollectedBatch:
    """One arm's exact batch collected under one immutable actor revision."""

    actor_features: Array
    context_features: Array
    critic_features: Array
    safety_features: Array
    latent_before: Array
    latent_after: Array
    actions: Array
    action_identity: Array
    policy_revision: Array
    behavior_log_probability: Array
    return_targets: Array
    representation_targets: Array
    model_targets: Array
    failure_mask: Array
    action_uniforms: Array

    def protected_batch(self) -> ReplayProtectedBatch:
        return ReplayProtectedBatch(
            context_features=self.context_features,
            critic_features=self.critic_features,
            safety_features=self.safety_features,
            actions=self.actions,
            return_targets=self.return_targets,
            representation_targets=self.representation_targets,
            model_targets=self.model_targets,
            failure_targets=self.failure_mask.astype(jnp.float32),
        )


@chex.dataclass(frozen=True)
class OnPolicyCollectionResult:
    """Pure collection-kernel result."""

    batch: OnPolicyCollectedBatch
    environment: OnPolicyEnvironmentState


@chex.dataclass(frozen=True)
class OnPolicyManualActorState:
    """Ordinary/uniform actor state with exact update accounting."""

    parameters: KondoActorParameters
    policy_revision: Array
    actor_backward_count: Array


def _selected_log_probability(
    parameters: KondoActorParameters,
    actor_features: Array,
    actions: Array,
) -> Array:
    hidden = jnp.tanh(
        actor_features @ parameters.hidden_weight + parameters.hidden_bias
    )
    logits = hidden @ parameters.output_weight + parameters.output_bias
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probabilities, actions[:, None], axis=1)[:, 0]


@functools.partial(jax.jit, static_argnums=())
def on_policy_selected_log_probability(
    parameters: KondoActorParameters,
    actor_features: Array,
    actions: Array,
) -> Array:
    """Return exact selected-action log probabilities for one frozen actor."""

    return _selected_log_probability(parameters, actor_features, actions)


@functools.partial(jax.jit, static_argnums=())
def collect_on_policy_batch_kernel(
    parameters: KondoActorParameters,
    environment: OnPolicyEnvironmentState,
    dynamics: OnPolicyEnvironmentParameters,
    schedule: OnPolicyCommonSchedule,
    regime_sign: Array,
    policy_revision: Array,
) -> OnPolicyCollectionResult:
    """Collect a full closed-loop batch without mutating the actor revision."""

    def step(
        carry: OnPolicyEnvironmentState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[OnPolicyEnvironmentState, tuple[Array, ...]]:
        action_uniform, transition_uniform, reward_uniform, failure = inputs
        centered_noise = transition_uniform * jnp.float32(2.0) - jnp.float32(1.0)
        observation = jnp.tanh(carry.latent + jnp.float32(0.12) * centered_noise)
        actor_features = jnp.tanh(observation @ dynamics.actor_projection)
        hidden = jnp.tanh(
            actor_features @ parameters.hidden_weight + parameters.hidden_bias
        )
        logits = hidden @ parameters.output_weight + parameters.output_bias
        log_probabilities = jax.nn.log_softmax(logits)
        probability_zero = jnp.exp(log_probabilities[0])
        action = jnp.where(
            action_uniform < probability_zero,
            jnp.int32(0),
            jnp.int32(1),
        )
        behavior_log_probability = log_probabilities[action]
        preferred = jnp.where(
            observation[0] + jnp.float32(0.25) * regime_sign >= 0.0,
            jnp.int32(1),
            jnp.int32(0),
        )
        centered_reward_noise = reward_uniform * jnp.float32(2.0) - jnp.float32(1.0)
        reward = (
            jnp.where(action == preferred, jnp.float32(0.8), jnp.float32(-0.35))
            + jnp.float32(0.05) * centered_reward_noise
            - failure.astype(jnp.float32) * jnp.float32(1.5)
        )
        next_latent = jnp.tanh(
            jnp.float32(0.64) * carry.latent
            + jnp.float32(0.18) * centered_noise
            + dynamics.action_effect[action]
            + regime_sign * dynamics.regime_effect
        )
        critic_features = jnp.tanh(observation @ dynamics.critic_projection)
        safety_features = jnp.tanh(
            observation @ dynamics.safety_projection
            + (action.astype(jnp.float32) * jnp.float32(2.0) - jnp.float32(1.0))
            * jnp.float32(0.08)
        )
        representation_target = jnp.tanh(
            next_latent @ dynamics.representation_projection
        )
        next_state = OnPolicyEnvironmentState(
            latent=next_latent,
            last_action=action,
            step_count=carry.step_count + jnp.int32(1),
            cumulative_reward=carry.cumulative_reward + reward,
            failure_count=carry.failure_count + failure.astype(jnp.int32),
        )
        outputs = (
            actor_features,
            observation,
            critic_features,
            safety_features,
            carry.latent,
            next_latent,
            action,
            behavior_log_probability,
            reward,
            representation_target,
            failure,
        )
        return next_state, outputs

    final, outputs = jax.lax.scan(
        step,
        environment,
        (
            schedule.action_uniforms,
            schedule.transition_uniforms,
            schedule.reward_uniforms,
            schedule.failure_mask,
        ),
    )
    batch_size = schedule.action_uniforms.shape[0]
    actions = outputs[6].astype(jnp.int32)
    batch = OnPolicyCollectedBatch(
        actor_features=outputs[0],
        context_features=outputs[1],
        critic_features=outputs[2],
        safety_features=outputs[3],
        latent_before=outputs[4],
        latent_after=outputs[5],
        actions=actions,
        action_identity=actions,
        policy_revision=jnp.full(
            (batch_size,),
            policy_revision,
            dtype=jnp.int32,
        ),
        behavior_log_probability=outputs[7],
        return_targets=outputs[8],
        representation_targets=outputs[9],
        model_targets=outputs[5],
        failure_mask=outputs[10].astype(jnp.bool_),
        action_uniforms=schedule.action_uniforms,
    )
    return OnPolicyCollectionResult(batch=batch, environment=final)


def _initial_environment(
    config: KondoActorCriticOnPolicyConfig,
) -> OnPolicyEnvironmentState:
    return OnPolicyEnvironmentState(
        latent=_deterministic_values(
            config.context_dim,
            phase=2.0,
            scale=0.18,
        ),
        last_action=jnp.asarray(0, dtype=jnp.int32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        cumulative_reward=jnp.asarray(0.0, dtype=jnp.float32),
        failure_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _parameter_payload(parameters: KondoActorParameters) -> dict[str, object]:
    return {
        name: _array_payload(getattr(parameters, name))
        for name in (
            "hidden_weight",
            "hidden_bias",
            "output_weight",
            "output_bias",
        )
    }


def _parameter_sha256(parameters: KondoActorParameters) -> str:
    return _canonical_sha256(_parameter_payload(parameters))


def _protected_parameter_payload(
    parameters: ReplayProtectedParameters,
) -> dict[str, object]:
    return {
        name: _array_payload(getattr(parameters, name))
        for name in (
            "baseline_weight",
            "baseline_bias",
            "critic_weight",
            "critic_bias",
            "representation_weight",
            "representation_bias",
            "model_weight",
            "model_bias",
            "safety_weight",
            "safety_bias",
        )
    }


def _protected_state_payload(state: ReplayProtectedState) -> dict[str, object]:
    return {
        "parameters": _protected_parameter_payload(state.parameters),
        "update_count": int(np.asarray(jax.device_get(state.update_count))),
    }


def _protected_state_sha256(state: ReplayProtectedState) -> str:
    return _canonical_sha256(_protected_state_payload(state))


def _environment_payload(state: OnPolicyEnvironmentState) -> dict[str, object]:
    return {
        "latent": _array_payload(state.latent),
        "last_action": int(np.asarray(jax.device_get(state.last_action))),
        "step_count": int(np.asarray(jax.device_get(state.step_count))),
        "cumulative_reward": float(
            np.asarray(jax.device_get(state.cumulative_reward))
        ),
        "failure_count": int(np.asarray(jax.device_get(state.failure_count))),
    }


def _manual_actor_payload(state: OnPolicyManualActorState) -> dict[str, object]:
    return {
        "parameters": _parameter_payload(state.parameters),
        "policy_revision": int(np.asarray(jax.device_get(state.policy_revision))),
        "actor_backward_count": int(
            np.asarray(jax.device_get(state.actor_backward_count))
        ),
    }


def _schedule_payload(
    schedule: OnPolicyCommonSchedule,
    *,
    event_index: int,
    phase: str,
    regime: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "event_index": event_index,
        "phase": phase,
        "regime": regime,
        "rng_impl": "threefry2x32",
        "action_uniforms": _array_payload(schedule.action_uniforms),
        "transition_uniforms": _array_payload(schedule.transition_uniforms),
        "reward_uniforms": _array_payload(schedule.reward_uniforms),
        "failure_mask": _array_payload(schedule.failure_mask),
        "key_words": _array_payload(schedule.key_words),
    }
    return {**body, "common_schedule_sha256": _canonical_sha256(body)}


def _collected_batch_payload(batch: OnPolicyCollectedBatch) -> dict[str, object]:
    return {
        name: _array_payload(getattr(batch, name))
        for name in (
            "actor_features",
            "context_features",
            "critic_features",
            "safety_features",
            "latent_before",
            "latent_after",
            "actions",
            "action_identity",
            "policy_revision",
            "behavior_log_probability",
            "return_targets",
            "representation_targets",
            "model_targets",
            "failure_mask",
            "action_uniforms",
        )
    }


def _protected_batch_payload(batch: ReplayProtectedBatch) -> dict[str, object]:
    return {
        name: _array_payload(getattr(batch, name))
        for name in (
            "context_features",
            "critic_features",
            "safety_features",
            "actions",
            "return_targets",
            "representation_targets",
            "model_targets",
            "failure_targets",
        )
    }


def _protected_prediction_sha256(result: ReplayProtectedBackwardResult) -> str:
    return _canonical_sha256(
        {
            name: _array_payload(getattr(result, name))
            for name in (
                "baseline_predictions",
                "critic_predictions",
                "representation_predictions",
                "model_predictions",
                "safety_logits",
            )
        }
    )


def _phase_for_event(
    config: KondoActorCriticOnPolicyConfig,
    event_index: int,
) -> tuple[str, str, float]:
    if event_index < config.batches_per_phase:
        return "A1", "A", 1.0
    if event_index < 2 * config.batches_per_phase:
        return "B", "B", -1.0
    return "A2", "A", 1.0


def _apply_actor_gradient(
    parameters: KondoActorParameters,
    gradient: KondoActorParameters,
    learning_rate: float,
) -> KondoActorParameters:
    updated = jax.tree.map(
        lambda parameter, grad: parameter
        - jnp.asarray(learning_rate, dtype=jnp.float32) * grad,
        parameters,
        gradient,
    )
    return cast(KondoActorParameters, updated)


def _apply_protected_gradient(
    state: ReplayProtectedState,
    gradient: ReplayProtectedParameters,
    learning_rate: float,
) -> ReplayProtectedState:
    parameters = cast(
        ReplayProtectedParameters,
        jax.tree.map(
            lambda parameter, grad: parameter
            - jnp.asarray(learning_rate, dtype=jnp.float32) * grad,
            state.parameters,
            gradient,
        ),
    )
    return ReplayProtectedState(
        parameters=parameters,
        update_count=state.update_count + jnp.int32(1),
    )


@dataclasses.dataclass(frozen=True)
class _ActorOutcome:
    parameters_before: KondoActorParameters
    parameters_after: KondoActorParameters
    policy_revision_before: int
    policy_revision_after: int
    behavior_log_probability: Array
    advantage: Array
    delight: Array
    selected_mask: Array
    selected_by_delight: Array
    selected_by_uniform_control: Array
    uniformly_reserved: Array
    force_keep_mask: Array
    gathered_indices: tuple[int, ...]
    backward: KondoActorBackwardSummary
    backward_leading_shape: int
    sparse_backward: bool
    screen_gather_order: tuple[str, ...]
    random_draw_count: int
    internal_protected_digest: str | None


def _manual_actor_outcome(
    *,
    arm: Literal["ordinary_full", "uniform_sparse"],
    state: OnPolicyManualActorState,
    batch: OnPolicyCollectedBatch,
    baseline_predictions: Array,
    uniform_scores: Array,
    config: KondoActorCriticOnPolicyConfig,
) -> tuple[_ActorOutcome, OnPolicyManualActorState]:
    behavior = on_policy_selected_log_probability(
        state.parameters,
        batch.actor_features,
        batch.actions,
    )
    if not _array_bits_equal(behavior, batch.behavior_log_probability):
        raise ValueError(f"{arm} behavior log-probability binding differs")
    if not bool(np.all(np.asarray(batch.policy_revision) == int(state.policy_revision))):
        raise ValueError(f"{arm} actor revision changed within collection")
    advantage = jax.lax.stop_gradient(batch.return_targets - baseline_predictions)
    delight = jax.lax.stop_gradient(advantage * (-behavior))
    forced = np.asarray(batch.failure_mask, dtype=np.bool_)
    if int(np.sum(forced)) != 1:
        raise ValueError("each on-policy batch must contain exactly one rare failure")
    order: tuple[str, ...]
    if arm == "ordinary_full":
        selected = np.ones((config.batch_size,), dtype=np.bool_)
        gathered_indices = tuple(range(config.batch_size))
        backward_batch = KondoActorBackwardBatch(
            actor_features=batch.actor_features,
            actions=batch.actions,
            advantage=advantage,
            sample_mask=jnp.ones((config.batch_size,), dtype=jnp.bool_),
        )
        backward_shape = config.batch_size
        sparse = False
        selected_uniform = np.zeros_like(selected)
        order = ("frozen-policy-forward", "full-batch-compiled-backward")
    else:
        scores = np.asarray(uniform_scores, dtype=np.float32)
        forced_indices = np.flatnonzero(forced).astype(np.int32).tolist()
        candidates = [
            int(index)
            for index in np.argsort(scores, kind="stable")
            if not forced[index]
        ]
        selected_indices = forced_indices + candidates[
            : config.sparse_capacity - len(forced_indices)
        ]
        gathered_indices = tuple(selected_indices)
        selected = np.zeros((config.batch_size,), dtype=np.bool_)
        selected[list(gathered_indices)] = True
        selected_uniform = selected & ~forced
        indices = jnp.asarray(gathered_indices, dtype=jnp.int32)
        backward_batch = KondoActorBackwardBatch(
            actor_features=batch.actor_features[indices],
            actions=batch.actions[indices],
            advantage=advantage[indices],
            sample_mask=jnp.ones((config.sparse_capacity,), dtype=jnp.bool_),
        )
        backward_shape = config.sparse_capacity
        sparse = True
        order = (
            "frozen-policy-forward",
            "force-rare-failure",
            "evaluator-uniform-gather",
            "compiled-backward",
        )
    backward = kondo_actor_backward_kernel(state.parameters, backward_batch)
    updated_parameters = _apply_actor_gradient(
        state.parameters,
        backward.gradient,
        config.actor_learning_rate,
    )
    if not bool(np.asarray(backward.gradient_finite)) or not _tree_all_finite(
        updated_parameters
    ):
        raise ValueError(f"{arm} actor backward/update produced nonfinite values")
    next_state = OnPolicyManualActorState(
        parameters=updated_parameters,
        policy_revision=state.policy_revision + jnp.int32(1),
        actor_backward_count=state.actor_backward_count + jnp.int32(1),
    )
    outcome = _ActorOutcome(
        parameters_before=state.parameters,
        parameters_after=updated_parameters,
        policy_revision_before=int(np.asarray(state.policy_revision)),
        policy_revision_after=int(np.asarray(next_state.policy_revision)),
        behavior_log_probability=behavior,
        advantage=advantage,
        delight=delight,
        selected_mask=jnp.asarray(selected, dtype=jnp.bool_),
        selected_by_delight=jnp.zeros((config.batch_size,), dtype=jnp.bool_),
        selected_by_uniform_control=jnp.asarray(selected_uniform, dtype=jnp.bool_),
        uniformly_reserved=jnp.zeros((config.batch_size,), dtype=jnp.bool_),
        force_keep_mask=batch.failure_mask,
        gathered_indices=gathered_indices,
        backward=backward,
        backward_leading_shape=backward_shape,
        sparse_backward=sparse,
        screen_gather_order=order,
        random_draw_count=0,
        internal_protected_digest=None,
    )
    return outcome, next_state


def _digest_words_hex(words: Array) -> str:
    host = np.asarray(jax.device_get(words), dtype=np.uint32)
    return "".join(f"{int(word):08x}" for word in host)


def _kondo_actor_outcome(
    *,
    actor: KondoSparseActor,
    state: KondoSparseActorState,
    batch: OnPolicyCollectedBatch,
    protected: KondoActorProtectedInputs,
    config: KondoActorCriticOnPolicyConfig,
    reserve: bool,
) -> tuple[_ActorOutcome, KondoSparseActorState]:
    behavior = actor.behavior_log_probability(
        state,
        batch.actor_features,
        batch.actions,
    )
    if not _array_bits_equal(behavior, batch.behavior_log_probability):
        raise ValueError("Kondo behavior log-probability binding differs")
    actor_batch = KondoSparseActorBatch(
        actor_features=batch.actor_features,
        actions=batch.actions,
        action_identity=batch.action_identity,
        policy_revision=batch.policy_revision,
        behavior_log_probability=behavior,
        valid_mask=jnp.ones((config.batch_size,), dtype=jnp.bool_),
        force_keep_mask=batch.failure_mask,
        protected=protected,
    )
    result: KondoSparseActorResult = actor.step(state, actor_batch)
    if (
        not bool(np.asarray(result.transaction_applied))
        or not bool(np.asarray(result.sparse_backward_used))
        or bool(np.asarray(result.full_shape_masked_backward_used))
    ):
        name = "reserve Kondo" if reserve else "paper Kondo"
        raise ValueError(f"{name} on-policy actor did not use sparse backward")
    if int(np.asarray(result.backward_batch_size)) != config.sparse_capacity:
        raise ValueError("Kondo actor backward capacity differs")
    if not _tree_bits_equal(result.protected, protected):
        raise ValueError("Kondo actor changed protected full-batch inputs")
    selected_indices = np.asarray(result.screen.selected_indices, dtype=np.int32)
    selected_slots = np.asarray(result.screen.selected_slot_mask, dtype=np.bool_)
    gathered_indices = tuple(int(item) for item in selected_indices[selected_slots])
    if not bool(np.all(np.asarray(batch.failure_mask) <= np.asarray(result.sparks_joy))):
        raise ValueError("Kondo actor dropped a forced rare-failure row")
    backward = KondoActorBackwardSummary(
        loss=result.actor_loss,
        gradient=result.gradient,
        selected_count=result.screen.selected_count,
        gradient_finite=result.gradient_finite,
    )
    outcome = _ActorOutcome(
        parameters_before=state.parameters,
        parameters_after=result.state.parameters,
        policy_revision_before=int(np.asarray(result.policy_revision_before)),
        policy_revision_after=int(np.asarray(result.policy_revision_after)),
        behavior_log_probability=result.current_action_log_probability,
        advantage=result.advantage,
        delight=result.screen.delight,
        selected_mask=result.sparks_joy,
        selected_by_delight=result.screen.selected_by_delight_gate,
        selected_by_uniform_control=jnp.zeros(
            (config.batch_size,), dtype=jnp.bool_
        ),
        uniformly_reserved=result.screen.uniformly_reserved,
        force_keep_mask=result.screen.force_keep_mask,
        gathered_indices=gathered_indices,
        backward=backward,
        backward_leading_shape=config.sparse_capacity,
        sparse_backward=True,
        screen_gather_order=(
            "frozen-policy-forward",
            "detached-paper-delight-screen",
            "force-rare-failure",
            "minimum-random-reserve" if reserve else "no-random-reserve",
            "audited-capacity-gather",
            "compiled-backward",
        ),
        random_draw_count=int(np.asarray(result.screen.random_draw_count)),
        internal_protected_digest=_digest_words_hex(result.protected_digest),
    )
    return outcome, result.state


def _record_for_arm(
    *,
    arm: ArmName,
    event_index: int,
    phase: str,
    regime: str,
    schedule_payload: Mapping[str, object],
    uniform_control_schedule_payload: Mapping[str, object],
    parent_sha256: str,
    environment_before: OnPolicyEnvironmentState,
    environment_after: OnPolicyEnvironmentState,
    batch: OnPolicyCollectedBatch,
    outcome: _ActorOutcome,
    protected_result: ReplayProtectedBackwardResult,
    protected_before: ReplayProtectedState,
    protected_after: ReplayProtectedState,
    config: KondoActorCriticOnPolicyConfig,
) -> dict[str, object]:
    behavior = on_policy_selected_log_probability(
        outcome.parameters_before,
        batch.actor_features,
        batch.actions,
    )
    if not _array_bits_equal(behavior, outcome.behavior_log_probability):
        raise ValueError(f"{arm} stored behavior log probabilities differ")
    expected_delight = jax.lax.stop_gradient(outcome.advantage * (-behavior))
    if not _array_bits_equal(expected_delight, outcome.delight):
        raise ValueError(f"{arm} paper delight differs")
    selected = np.asarray(outcome.selected_mask, dtype=np.bool_)
    failures = np.asarray(batch.failure_mask, dtype=np.bool_)
    selected_indices = np.flatnonzero(selected).astype(np.int32).tolist()
    selected_count = int(np.asarray(outcome.backward.selected_count))
    if selected_count != len(selected_indices) or selected_count != len(
        outcome.gathered_indices
    ):
        raise ValueError(f"{arm} selected-row accounting differs")
    if not np.all(~failures | selected):
        raise ValueError(f"{arm} failed to learn a forced rare-failure row")
    expected_shape = (
        config.batch_size if arm == "ordinary_full" else config.sparse_capacity
    )
    if outcome.backward_leading_shape != expected_shape or selected_count != expected_shape:
        raise ValueError(f"{arm} actor backward leading shape differs")
    collected_payload = _collected_batch_payload(batch)
    protected_payload = _protected_batch_payload(batch.protected_batch())
    body: dict[str, object] = {
        "arm": arm,
        "event_index": event_index,
        "phase": phase,
        "regime": regime,
        "causal_parent_sha256": parent_sha256,
        "common_schedule_sha256": schedule_payload["common_schedule_sha256"],
        "common_schedule_rng_impl": "threefry2x32",
        "common_exogenous_randomness_only": True,
        "uniform_control_schedule_sha256": (
            uniform_control_schedule_payload["uniform_control_schedule_sha256"]
            if arm == "uniform_sparse"
            else None
        ),
        "uniform_control_randomness_paired_across_arms": False,
        "trajectory_equality_assumed": False,
        "environment_before_sha256": _canonical_sha256(
            _environment_payload(environment_before)
        ),
        "environment_after_sha256": _canonical_sha256(
            _environment_payload(environment_after)
        ),
        "collected_batch": collected_payload,
        "collected_batch_sha256": _canonical_sha256(collected_payload),
        "actions": _array_payload(batch.actions),
        "action_identity": _array_payload(batch.action_identity),
        "action_sampling_uniforms": _array_payload(batch.action_uniforms),
        "behavior_log_probability": _array_payload(behavior),
        "policy_revision_rows": _array_payload(batch.policy_revision),
        "policy_revision_before": outcome.policy_revision_before,
        "policy_revision_after": outcome.policy_revision_after,
        "actor_revision_immutable_during_collection": True,
        "actor_updated_only_after_batch_collection": True,
        "actions_sampled_from_this_arms_policy": True,
        "on_policy": True,
        "behavior_log_probability_exact": True,
        "importance_correction_applied": False,
        "advantage": _array_payload(outcome.advantage),
        "selected_action_surprisal": _array_payload(-behavior),
        "paper_delight": _array_payload(outcome.delight),
        "executed_actor_backward_mask": _array_payload(outcome.selected_mask),
        "executed_actor_backward_mask_semantics": (
            "gradient-contribution-entered-executed-actor-backward"
        ),
        "selected_source_indices": selected_indices,
        "actor_backward_gather_order": list(outcome.gathered_indices),
        "selected_count": selected_count,
        "selection_mechanism": {
            "ordinary_full": "all-on-policy-rows",
            "uniform_sparse": "forced-failure-plus-evaluator-uniform-capacity",
            "kondo_top_k": "paper-delight-top-k-with-forced-failure",
            "kondo_top_k_reserve": (
                "paper-delight-top-k-with-forced-failure-and-random-reserve"
            ),
        }[arm],
        "selected_by_delight_gate": _array_payload(outcome.selected_by_delight),
        "selected_by_uniform_control": _array_payload(
            outcome.selected_by_uniform_control
        ),
        "minimum_random_reserve": _array_payload(outcome.uniformly_reserved),
        "force_keep_mask": _array_payload(outcome.force_keep_mask),
        "random_draw_count": outcome.random_draw_count,
        "actor_parameters_before_sha256": _parameter_sha256(
            outcome.parameters_before
        ),
        "actor_parameters_after_sha256": _parameter_sha256(
            outcome.parameters_after
        ),
        "actor_loss": float(np.asarray(outcome.backward.loss)),
        "actor_gradient_l2": _tree_l2(outcome.backward.gradient),
        "actor_gradient_finite": bool(np.asarray(outcome.backward.gradient_finite)),
        "actor_update_opportunities": 1,
        "actor_updates_applied": 1,
        "actor_compiled_backward_invocations": 1,
        "actor_backward_leading_shape": outcome.backward_leading_shape,
        "sparse_actor_backward": outcome.sparse_backward,
        "screen_gather_backward_order": list(outcome.screen_gather_order),
        "internal_kondo_protected_digest": outcome.internal_protected_digest,
        "protected_learning_batch_sha256": _canonical_sha256(protected_payload),
        "protected_state_before_sha256": _protected_state_sha256(protected_before),
        "protected_state_after_sha256": _protected_state_sha256(protected_after),
        "protected_predictions_sha256": _protected_prediction_sha256(
            protected_result
        ),
        "protected_total_loss": float(np.asarray(protected_result.total_loss)),
        "baseline_loss": float(np.asarray(protected_result.baseline_loss)),
        "critic_loss": float(np.asarray(protected_result.critic_loss)),
        "representation_loss": float(
            np.asarray(protected_result.representation_loss)
        ),
        "world_model_loss": float(np.asarray(protected_result.model_loss)),
        "safety_guardrail_loss": float(np.asarray(protected_result.safety_loss)),
        "protected_gradient_l2": _tree_l2(protected_result.gradient),
        "protected_gradient_finite": bool(
            np.asarray(protected_result.gradient_finite)
        ),
        "protected_channels_full_batch": True,
        "protected_rows_in_backward": config.batch_size,
        "protected_update_opportunities": 1,
        "protected_updates_applied": 1,
        "protected_compiled_backward_invocations": 1,
        "protected_backward_leading_shape": config.batch_size,
        "rare_failure_rows_collected": int(np.sum(failures)),
        "rare_failure_rows_in_actor_backward": int(np.sum(failures & selected)),
        "rare_failure_rows_in_protected_backward": int(np.sum(failures)),
        "rare_failure_guardrail_full_learning": True,
        "assessment_status": ASSESSMENT_STATUS,
        "efficacy_claimed": False,
        "compute_benefit_claimed": False,
        "safety_claimed": False,
    }
    return {**body, "record_sha256": _canonical_sha256(body)}


@functools.partial(jax.jit, static_argnums=())
def sample_on_policy_actions_from_uniforms(
    parameters: KondoActorParameters,
    actor_features: Array,
    uniforms: Array,
) -> Array:
    """Reconstruct binary categorical samples from evaluator-owned uniforms."""

    hidden = jnp.tanh(
        actor_features @ parameters.hidden_weight + parameters.hidden_bias
    )
    probabilities = jax.nn.softmax(
        hidden @ parameters.output_weight + parameters.output_bias,
        axis=-1,
    )
    return jnp.where(
        uniforms < probabilities[:, 0],
        jnp.int32(0),
        jnp.int32(1),
    )


@dataclasses.dataclass(frozen=True)
class KondoActorCriticOnPolicyRunState:
    """Integrity-sealed causal prefix for four independent closed-loop arms."""

    event_index: int
    ordinary_actor: OnPolicyManualActorState
    uniform_actor: OnPolicyManualActorState
    kondo_state: KondoSparseActorState
    reserve_state: KondoSparseActorState
    ordinary_protected: ReplayProtectedState
    uniform_protected: ReplayProtectedState
    kondo_protected: ReplayProtectedState
    reserve_protected: ReplayProtectedState
    ordinary_environment: OnPolicyEnvironmentState
    uniform_environment: OnPolicyEnvironmentState
    kondo_environment: OnPolicyEnvironmentState
    reserve_environment: OnPolicyEnvironmentState
    chain_heads: tuple[str, str, str, str]
    records_json: tuple[str, ...]
    integrity_sha256: str


class KondoActorCriticOnPolicyEvaluator:
    """Host orchestrator around JAX collection and backward kernels."""

    def __init__(self, config: KondoActorCriticOnPolicyConfig):
        if type(config) is not KondoActorCriticOnPolicyConfig:
            raise TypeError("config must be an exact KondoActorCriticOnPolicyConfig")
        self.config = config
        self.kondo_actor = KondoSparseActor(config.actor_config(reserve=False))
        self.reserve_actor = KondoSparseActor(config.actor_config(reserve=True))
        self.initial_actor_parameters = _initial_actor_parameters(config)
        self.initial_protected_parameters = _initial_protected_parameters(config)
        self.environment_parameters = _environment_parameters(config)
        self.initial_environment = _initial_environment(config)
        self._actor_signature = _tree_signature(self.initial_actor_parameters)
        self._protected_signature = _tree_signature(self.initial_protected_parameters)
        self._environment_signature = _tree_signature(self.initial_environment)
        self._root_key = jr.key(config.seed, impl="threefry2x32")

    def common_schedule(self, event_index: int) -> OnPolicyCommonSchedule:
        """Return the one evaluator-owned exogenous CRN schedule for a batch."""

        _exact_int(
            event_index,
            name="event_index",
            maximum=self.config.total_batches - 1,
        )
        event_key = jr.fold_in(self._root_key, np.uint32(10_000 + event_index))
        keys = jr.split(event_key, 3)
        cfg = self.config
        failure_index = (event_index * 3 + 1) % cfg.batch_size
        failure_mask = jnp.zeros((cfg.batch_size,), dtype=jnp.bool_).at[
            failure_index
        ].set(True)
        return OnPolicyCommonSchedule(
            action_uniforms=jr.uniform(
                keys[0],
                (cfg.batch_size,),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            ),
            transition_uniforms=jr.uniform(
                keys[1],
                (cfg.batch_size, cfg.context_dim),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            ),
            reward_uniforms=jr.uniform(
                keys[2],
                (cfg.batch_size,),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            ),
            failure_mask=failure_mask,
            key_words=jnp.stack(tuple(jr.key_data(key) for key in keys)),
        )

    def uniform_control_schedule(self, event_index: int) -> dict[str, object]:
        """Return unpaired allocation randomness used only by the uniform arm."""

        _exact_int(
            event_index,
            name="event_index",
            maximum=self.config.total_batches - 1,
        )
        key = jr.fold_in(self._root_key, np.uint32(20_000 + event_index))
        scores = jr.uniform(
            key,
            (self.config.batch_size,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        body: dict[str, object] = {
            "event_index": event_index,
            "rng_impl": "threefry2x32",
            "paired_across_arms": False,
            "consumer_arm": "uniform_sparse",
            "scores": _array_payload(scores),
            "key_words": _array_payload(jr.key_data(key)),
        }
        return {
            **body,
            "uniform_control_schedule_sha256": _canonical_sha256(body),
        }

    def _uniform_control_scores(self, event_index: int) -> Array:
        payload = self.uniform_control_schedule(event_index)
        raw = cast(Mapping[str, object], payload["scores"])
        dtype = np.dtype(cast(str, raw["dtype"]))
        shape = tuple(cast(list[int], raw["shape"]))
        return jnp.asarray(
            np.frombuffer(
                bytes.fromhex(cast(str, raw["data_hex"])),
                dtype=dtype,
            ).reshape(shape)
        )

    def _initial_chain_head(self, arm: ArmName) -> str:
        return _canonical_sha256(
            {
                "arm": arm,
                "initial_actor_parameters": _parameter_payload(
                    self.initial_actor_parameters
                ),
                "initial_protected_parameters": _protected_parameter_payload(
                    self.initial_protected_parameters
                ),
                "initial_environment": _environment_payload(
                    self.initial_environment
                ),
            }
        )

    def _state_body(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> dict[str, object]:
        return {
            "event_index": state.event_index,
            "ordinary_actor": _manual_actor_payload(state.ordinary_actor),
            "uniform_actor": _manual_actor_payload(state.uniform_actor),
            "kondo_state": self.kondo_actor.checkpoint_payload(state.kondo_state),
            "reserve_state": self.reserve_actor.checkpoint_payload(
                state.reserve_state
            ),
            "ordinary_protected": _protected_state_payload(
                state.ordinary_protected
            ),
            "uniform_protected": _protected_state_payload(state.uniform_protected),
            "kondo_protected": _protected_state_payload(state.kondo_protected),
            "reserve_protected": _protected_state_payload(state.reserve_protected),
            "ordinary_environment": _environment_payload(
                state.ordinary_environment
            ),
            "uniform_environment": _environment_payload(
                state.uniform_environment
            ),
            "kondo_environment": _environment_payload(state.kondo_environment),
            "reserve_environment": _environment_payload(state.reserve_environment),
            "chain_heads": list(state.chain_heads),
            "records": [json.loads(record) for record in state.records_json],
        }

    def _seal_state(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> KondoActorCriticOnPolicyRunState:
        body = self._state_body(state)
        return dataclasses.replace(state, integrity_sha256=_canonical_sha256(body))

    def init(self) -> KondoActorCriticOnPolicyRunState:
        """Initialize identical parameter/environment bits with independent arms."""

        zero = jnp.asarray(0, dtype=jnp.int32)
        manual = OnPolicyManualActorState(
            parameters=self.initial_actor_parameters,
            policy_revision=zero,
            actor_backward_count=zero,
        )
        protected = ReplayProtectedState(
            parameters=self.initial_protected_parameters,
            update_count=zero,
        )
        state = KondoActorCriticOnPolicyRunState(
            event_index=0,
            ordinary_actor=manual,
            uniform_actor=manual,
            kondo_state=self.kondo_actor.init(
                self.initial_actor_parameters,
                jr.fold_in(self._root_key, np.uint32(30_001)),
            ),
            reserve_state=self.reserve_actor.init(
                self.initial_actor_parameters,
                jr.fold_in(self._root_key, np.uint32(30_002)),
            ),
            ordinary_protected=protected,
            uniform_protected=protected,
            kondo_protected=protected,
            reserve_protected=protected,
            ordinary_environment=self.initial_environment,
            uniform_environment=self.initial_environment,
            kondo_environment=self.initial_environment,
            reserve_environment=self.initial_environment,
            chain_heads=cast(
                tuple[str, str, str, str],
                tuple(self._initial_chain_head(arm) for arm in ARM_ORDER),
            ),
            records_json=(),
            integrity_sha256="",
        )
        sealed = self._seal_state(state)
        if not self._valid_state_structure(sealed):
            raise ValueError("initial on-policy state is invalid")
        return sealed

    def _valid_manual_actor(
        self,
        state: OnPolicyManualActorState,
        *,
        event_index: int,
    ) -> bool:
        return (
            type(state) is OnPolicyManualActorState
            and _tree_signature(state.parameters) == self._actor_signature
            and _tree_all_finite(state.parameters)
            and getattr(state.policy_revision, "shape", None) == ()
            and getattr(state.policy_revision, "dtype", None) == jnp.dtype(jnp.int32)
            and getattr(state.actor_backward_count, "shape", None) == ()
            and getattr(state.actor_backward_count, "dtype", None)
            == jnp.dtype(jnp.int32)
            and int(np.asarray(state.policy_revision)) == event_index
            and int(np.asarray(state.actor_backward_count)) == event_index
        )

    def _valid_protected(
        self,
        state: ReplayProtectedState,
        *,
        event_index: int,
    ) -> bool:
        return (
            type(state) is ReplayProtectedState
            and _tree_signature(state.parameters) == self._protected_signature
            and _tree_all_finite(state.parameters)
            and getattr(state.update_count, "shape", None) == ()
            and getattr(state.update_count, "dtype", None) == jnp.dtype(jnp.int32)
            and int(np.asarray(state.update_count)) == event_index
        )

    def _valid_environment(
        self,
        state: OnPolicyEnvironmentState,
        *,
        event_index: int,
    ) -> bool:
        return (
            type(state) is OnPolicyEnvironmentState
            and _tree_signature(state) == self._environment_signature
            and _tree_all_finite(state)
            and int(np.asarray(state.step_count))
            == event_index * self.config.batch_size
            and 0 <= int(np.asarray(state.last_action)) < self.config.action_count
            and int(np.asarray(state.failure_count)) == event_index
        )

    def _record_chains_valid(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> bool:
        heads = {arm: self._initial_chain_head(arm) for arm in ARM_ORDER}
        try:
            parsed = [json.loads(record) for record in state.records_json]
        except json.JSONDecodeError:
            return False
        if len(parsed) != state.event_index * len(ARM_ORDER):
            return False
        for offset, item in enumerate(parsed):
            if not isinstance(item, Mapping):
                return False
            arm = ARM_ORDER[offset % len(ARM_ORDER)]
            event_index = offset // len(ARM_ORDER)
            if (
                item.get("arm") != arm
                or item.get("event_index") != event_index
                or item.get("causal_parent_sha256") != heads[arm]
            ):
                return False
            body = {name: item[name] for name in item if name != "record_sha256"}
            if item.get("record_sha256") != _canonical_sha256(body):
                return False
            heads[arm] = cast(str, item["record_sha256"])
        return tuple(heads[arm] for arm in ARM_ORDER) == state.chain_heads

    def _valid_state_structure(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> bool:
        if type(state) is not KondoActorCriticOnPolicyRunState:
            return False
        if not 0 <= state.event_index <= self.config.total_batches:
            return False
        if len(state.integrity_sha256) != 64:
            return False
        try:
            if state.integrity_sha256 != _canonical_sha256(self._state_body(state)):
                return False
            self.kondo_actor.checkpoint_payload(state.kondo_state)
            self.reserve_actor.checkpoint_payload(state.reserve_state)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not self._valid_manual_actor(
            state.ordinary_actor, event_index=state.event_index
        ) or not self._valid_manual_actor(
            state.uniform_actor, event_index=state.event_index
        ):
            return False
        if (
            int(np.asarray(state.kondo_state.policy_revision)) != state.event_index
            or int(np.asarray(state.reserve_state.policy_revision))
            != state.event_index
            or int(np.asarray(state.kondo_state.actor_backward_count))
            != state.event_index
            or int(np.asarray(state.reserve_state.actor_backward_count))
            != state.event_index
        ):
            return False
        if any(
            not self._valid_protected(item, event_index=state.event_index)
            for item in (
                state.ordinary_protected,
                state.uniform_protected,
                state.kondo_protected,
                state.reserve_protected,
            )
        ):
            return False
        if any(
            not self._valid_environment(item, event_index=state.event_index)
            for item in (
                state.ordinary_environment,
                state.uniform_environment,
                state.kondo_environment,
                state.reserve_environment,
            )
        ):
            return False
        return self._record_chains_valid(state)

    def validate_state(
        self,
        state: KondoActorCriticOnPolicyRunState,
        *,
        causal: bool = True,
    ) -> bool:
        """Validate integrity and, by default, exact causal-prefix reconstruction."""

        if not self._valid_state_structure(state):
            return False
        if not causal:
            return True
        expected = self._reconstruct(state.event_index)
        return _strict_json_equal(self._state_body(state), self._state_body(expected))

    def _validate_schedule(self, schedule: OnPolicyCommonSchedule) -> None:
        cfg = self.config
        contracts = (
            (schedule.action_uniforms, (cfg.batch_size,), jnp.float32),
            (
                schedule.transition_uniforms,
                (cfg.batch_size, cfg.context_dim),
                jnp.float32,
            ),
            (schedule.reward_uniforms, (cfg.batch_size,), jnp.float32),
            (schedule.failure_mask, (cfg.batch_size,), jnp.bool_),
            (schedule.key_words, (3, 2), jnp.uint32),
        )
        for value, shape, dtype in contracts:
            if getattr(value, "shape", None) != shape or getattr(
                value, "dtype", None
            ) != jnp.dtype(dtype):
                raise ValueError("common schedule array contract differs")
        if not _tree_all_finite(
            (
                schedule.action_uniforms,
                schedule.transition_uniforms,
                schedule.reward_uniforms,
            )
        ):
            raise ValueError("common schedule contains nonfinite values")
        if int(np.sum(np.asarray(schedule.failure_mask))) != 1:
            raise ValueError("common schedule must force exactly one rare failure")

    def _collect_one(
        self,
        *,
        parameters: KondoActorParameters,
        revision: Array,
        environment: OnPolicyEnvironmentState,
        schedule: OnPolicyCommonSchedule,
        regime_sign: float,
    ) -> OnPolicyCollectionResult:
        result = collect_on_policy_batch_kernel(
            parameters,
            environment,
            self.environment_parameters,
            schedule,
            jnp.asarray(regime_sign, dtype=jnp.float32),
            revision,
        )
        behavior = on_policy_selected_log_probability(
            parameters,
            result.batch.actor_features,
            result.batch.actions,
        )
        reconstructed_actions = sample_on_policy_actions_from_uniforms(
            parameters,
            result.batch.actor_features,
            schedule.action_uniforms,
        )
        if not _array_bits_equal(reconstructed_actions, result.batch.actions):
            raise ValueError("on-policy action sampling cannot be reconstructed")
        return OnPolicyCollectionResult(
            batch=cast(
                OnPolicyCollectedBatch,
                result.batch.replace(behavior_log_probability=behavior),
            ),
            environment=result.environment,
        )

    def _collect_all(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> tuple[
        tuple[OnPolicyCollectedBatch, ...],
        tuple[OnPolicyEnvironmentState, ...],
        OnPolicyCommonSchedule,
        Array,
        dict[str, object],
    ]:
        phase, regime, regime_sign = _phase_for_event(
            self.config,
            state.event_index,
        )
        schedule = self.common_schedule(state.event_index)
        uniform_scores = self._uniform_control_scores(state.event_index)
        self._validate_schedule(schedule)
        parameters = (
            state.ordinary_actor.parameters,
            state.uniform_actor.parameters,
            state.kondo_state.parameters,
            state.reserve_state.parameters,
        )
        revisions = (
            state.ordinary_actor.policy_revision,
            state.uniform_actor.policy_revision,
            state.kondo_state.policy_revision,
            state.reserve_state.policy_revision,
        )
        environments = (
            state.ordinary_environment,
            state.uniform_environment,
            state.kondo_environment,
            state.reserve_environment,
        )
        results = tuple(
            self._collect_one(
                parameters=actor_parameters,
                revision=revision,
                environment=environment,
                schedule=schedule,
                regime_sign=regime_sign,
            )
            for actor_parameters, revision, environment in zip(
                parameters,
                revisions,
                environments,
                strict=True,
            )
        )
        batches = list(item.batch for item in results)
        batches[2] = cast(
            OnPolicyCollectedBatch,
            batches[2].replace(
                behavior_log_probability=self.kondo_actor.behavior_log_probability(
                    state.kondo_state,
                    batches[2].actor_features,
                    batches[2].actions,
                )
            ),
        )
        batches[3] = cast(
            OnPolicyCollectedBatch,
            batches[3].replace(
                behavior_log_probability=self.reserve_actor.behavior_log_probability(
                    state.reserve_state,
                    batches[3].actor_features,
                    batches[3].actions,
                )
            ),
        )
        payload = _schedule_payload(
            schedule,
            event_index=state.event_index,
            phase=phase,
            regime=regime,
        )
        return (
            tuple(batches),
            tuple(item.environment for item in results),
            schedule,
            uniform_scores,
            payload,
        )

    def collect_current_batches(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> tuple[OnPolicyCollectedBatch, ...]:
        """Read-only exact collection for audit; no actor or learner is updated."""

        if not self.validate_state(state):
            raise ValueError("cannot collect from an invalid causal run state")
        if state.event_index >= self.config.total_batches:
            raise ValueError("on-policy run is already complete")
        batches, _, _, _, _ = self._collect_all(state)
        return batches

    def _advance_once(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> KondoActorCriticOnPolicyRunState:
        if not self._valid_state_structure(state):
            raise ValueError("on-policy run state structure is invalid")
        if state.event_index >= self.config.total_batches:
            raise ValueError("on-policy run is already complete")
        phase, regime, _ = _phase_for_event(self.config, state.event_index)
        (
            batches,
            environments_after,
            schedule,
            uniform_scores,
            schedule_payload,
        ) = self._collect_all(state)
        uniform_control_payload = self.uniform_control_schedule(state.event_index)
        protected_before = (
            state.ordinary_protected,
            state.uniform_protected,
            state.kondo_protected,
            state.reserve_protected,
        )
        protected_results = tuple(
            replay_protected_backward_kernel(before.parameters, batch.protected_batch())
            for before, batch in zip(protected_before, batches, strict=True)
        )
        if any(
            not bool(np.asarray(result.gradient_finite))
            for result in protected_results
        ):
            raise ValueError("protected full-batch backward produced nonfinite values")
        protected_after = tuple(
            _apply_protected_gradient(
                before,
                result.gradient,
                self.config.protected_learning_rate,
            )
            for before, result in zip(
                protected_before,
                protected_results,
                strict=True,
            )
        )
        if any(not _tree_all_finite(item.parameters) for item in protected_after):
            raise ValueError("protected full-batch update produced nonfinite values")
        actor_protected = tuple(
            KondoActorProtectedInputs(
                critic_features=batch.critic_features,
                baseline_predictions=result.baseline_predictions,
                return_targets=batch.return_targets,
                safety_features=batch.safety_features,
            )
            for batch, result in zip(batches, protected_results, strict=True)
        )
        ordinary_outcome, ordinary_actor = _manual_actor_outcome(
            arm="ordinary_full",
            state=state.ordinary_actor,
            batch=batches[0],
            baseline_predictions=protected_results[0].baseline_predictions,
            uniform_scores=uniform_scores,
            config=self.config,
        )
        uniform_outcome, uniform_actor = _manual_actor_outcome(
            arm="uniform_sparse",
            state=state.uniform_actor,
            batch=batches[1],
            baseline_predictions=protected_results[1].baseline_predictions,
            uniform_scores=uniform_scores,
            config=self.config,
        )
        kondo_outcome, kondo_state = _kondo_actor_outcome(
            actor=self.kondo_actor,
            state=state.kondo_state,
            batch=batches[2],
            protected=actor_protected[2],
            config=self.config,
            reserve=False,
        )
        reserve_outcome, reserve_state = _kondo_actor_outcome(
            actor=self.reserve_actor,
            state=state.reserve_state,
            batch=batches[3],
            protected=actor_protected[3],
            config=self.config,
            reserve=True,
        )
        outcomes = (
            ordinary_outcome,
            uniform_outcome,
            kondo_outcome,
            reserve_outcome,
        )
        environments_before = (
            state.ordinary_environment,
            state.uniform_environment,
            state.kondo_environment,
            state.reserve_environment,
        )
        records = tuple(
            _record_for_arm(
                arm=arm,
                event_index=state.event_index,
                phase=phase,
                regime=regime,
                schedule_payload=schedule_payload,
                uniform_control_schedule_payload=uniform_control_payload,
                parent_sha256=parent,
                environment_before=environment_before,
                environment_after=environment_after,
                batch=batch,
                outcome=outcome,
                protected_result=protected_result,
                protected_before=before,
                protected_after=after,
                config=self.config,
            )
            for (
                arm,
                parent,
                environment_before,
                environment_after,
                batch,
                outcome,
                protected_result,
                before,
                after,
            ) in zip(
                ARM_ORDER,
                state.chain_heads,
                environments_before,
                environments_after,
                batches,
                outcomes,
                protected_results,
                protected_before,
                protected_after,
                strict=True,
            )
        )
        record_json = tuple(
            _canonical_json_bytes(record).decode("utf-8") for record in records
        )
        next_state = KondoActorCriticOnPolicyRunState(
            event_index=state.event_index + 1,
            ordinary_actor=ordinary_actor,
            uniform_actor=uniform_actor,
            kondo_state=kondo_state,
            reserve_state=reserve_state,
            ordinary_protected=protected_after[0],
            uniform_protected=protected_after[1],
            kondo_protected=protected_after[2],
            reserve_protected=protected_after[3],
            ordinary_environment=environments_after[0],
            uniform_environment=environments_after[1],
            kondo_environment=environments_after[2],
            reserve_environment=environments_after[3],
            chain_heads=cast(
                tuple[str, str, str, str],
                tuple(cast(str, record["record_sha256"]) for record in records),
            ),
            records_json=state.records_json + record_json,
            integrity_sha256="",
        )
        sealed = self._seal_state(next_state)
        if not self._valid_state_structure(sealed):
            raise ValueError("on-policy next state is invalid")
        return sealed

    def _reconstruct(self, event_index: int) -> KondoActorCriticOnPolicyRunState:
        state = self.init()
        for _ in range(event_index):
            state = self._advance_once(state)
        return state

    def advance(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> KondoActorCriticOnPolicyRunState:
        """Advance only an exact source/runtime/config-derived causal prefix."""

        if not self._valid_state_structure(state):
            raise ValueError("on-policy run state is invalid")
        expected = self._reconstruct(state.event_index)
        if not _strict_json_equal(self._state_body(state), self._state_body(expected)):
            raise ValueError("on-policy run state differs from exact causal prefix")
        return self._advance_once(state)

    def run_to_end(
        self,
        state: KondoActorCriticOnPolicyRunState | None = None,
    ) -> KondoActorCriticOnPolicyRunState:
        current = self.init() if state is None else state
        if not self.validate_state(current):
            raise ValueError("cannot resume from an invalid causal run state")
        while current.event_index < self.config.total_batches:
            current = self.advance(current)
        return current

    def initial_snapshot_payload(self) -> dict[str, object]:
        return {
            "actor_parameters": _parameter_payload(self.initial_actor_parameters),
            "protected_parameters": _protected_parameter_payload(
                self.initial_protected_parameters
            ),
            "environment": _environment_payload(self.initial_environment),
            "environment_parameters": {
                name: _array_payload(getattr(self.environment_parameters, name))
                for name in (
                    "actor_projection",
                    "critic_projection",
                    "safety_projection",
                    "representation_projection",
                    "action_effect",
                    "regime_effect",
                )
            },
        }

    def checkpoint_payload(
        self,
        state: KondoActorCriticOnPolicyRunState,
    ) -> dict[str, object]:
        """Return a strict host-only causal checkpoint.

        The unkeyed digest detects corruption and binds exact reconstruction;
        it is not a MAC, signature, or authenticity claim.
        """

        if not self.validate_state(state):
            raise ValueError("cannot checkpoint an invalid causal run state")
        source = kondo_on_policy_source_manifest()
        runtime = kondo_on_policy_runtime_identity()
        protocol = kondo_on_policy_protocol(self.config)
        schedules = [
            _schedule_payload(
                self.common_schedule(index),
                event_index=index,
                phase=_phase_for_event(self.config, index)[0],
                regime=_phase_for_event(self.config, index)[1],
            )
            for index in range(self.config.total_batches)
        ]
        uniform_schedules = [
            self.uniform_control_schedule(index)
            for index in range(self.config.total_batches)
        ]
        body: dict[str, object] = {
            "schema": KONDO_ON_POLICY_CHECKPOINT_SCHEMA,
            "type": "KondoActorCriticOnPolicyCheckpoint",
            "host_only": True,
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(protocol),
            "source_manifest_sha256": _canonical_sha256(source),
            "runtime_sha256": _canonical_sha256(runtime),
            "initial_snapshot_sha256": _canonical_sha256(
                self.initial_snapshot_payload()
            ),
            "common_schedules_sha256": _canonical_sha256(schedules),
            "uniform_control_schedules_sha256": _canonical_sha256(
                uniform_schedules
            ),
            "common_schedule_prefix_sha256": _canonical_sha256(
                schedules[: state.event_index]
            ),
            "event_index": state.event_index,
            "state": self._state_body(state),
            "state_integrity_sha256": state.integrity_sha256,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
        }
        payload = {**body, "checkpoint_sha256": _canonical_sha256(body)}
        if len(_canonical_json_bytes(payload)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("on-policy checkpoint exceeds byte cap")
        return payload

    def restore_checkpoint(
        self,
        payload: object,
    ) -> KondoActorCriticOnPolicyRunState:
        """Host-restore only an exact causal prefix for this source/runtime."""

        raw = _mapping(payload, name="checkpoint")
        if len(_canonical_json_bytes(raw)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("on-policy checkpoint exceeds byte cap")
        expected_fields = {
            "schema",
            "type",
            "host_only",
            "config_sha256",
            "protocol_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
            "initial_snapshot_sha256",
            "common_schedules_sha256",
            "uniform_control_schedules_sha256",
            "common_schedule_prefix_sha256",
            "event_index",
            "state",
            "state_integrity_sha256",
            "assessment_status",
            "promotion_authority",
            "checkpoint_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError("on-policy checkpoint fields differ")
        if (
            raw.get("schema") != KONDO_ON_POLICY_CHECKPOINT_SCHEMA
            or raw.get("type") != "KondoActorCriticOnPolicyCheckpoint"
            or raw.get("host_only") is not True
        ):
            raise ValueError("on-policy checkpoint schema/type differs")
        if (
            raw.get("assessment_status") != ASSESSMENT_STATUS
            or raw.get("promotion_authority") is not False
        ):
            raise ValueError("on-policy checkpoint authority fields differ")
        body = {name: raw[name] for name in raw if name != "checkpoint_sha256"}
        if raw.get("checkpoint_sha256") != _canonical_sha256(body):
            raise ValueError("on-policy checkpoint digest integrity check failed")
        event_index = _exact_int(
            raw.get("event_index"),
            name="event_index",
            maximum=self.config.total_batches,
        )
        schedules = [
            _schedule_payload(
                self.common_schedule(index),
                event_index=index,
                phase=_phase_for_event(self.config, index)[0],
                regime=_phase_for_event(self.config, index)[1],
            )
            for index in range(self.config.total_batches)
        ]
        uniform_schedules = [
            self.uniform_control_schedule(index)
            for index in range(self.config.total_batches)
        ]
        bindings = {
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(
                kondo_on_policy_protocol(self.config)
            ),
            "source_manifest_sha256": _canonical_sha256(
                kondo_on_policy_source_manifest()
            ),
            "runtime_sha256": _canonical_sha256(
                kondo_on_policy_runtime_identity()
            ),
            "initial_snapshot_sha256": _canonical_sha256(
                self.initial_snapshot_payload()
            ),
            "common_schedules_sha256": _canonical_sha256(schedules),
            "uniform_control_schedules_sha256": _canonical_sha256(
                uniform_schedules
            ),
            "common_schedule_prefix_sha256": _canonical_sha256(
                schedules[:event_index]
            ),
        }
        if any(raw.get(name) != expected for name, expected in bindings.items()):
            raise ValueError("on-policy checkpoint source/runtime binding differs")
        expected_state = self._reconstruct(event_index)
        expected_payload = self.checkpoint_payload(expected_state)
        if not _strict_json_equal(dict(raw), expected_payload):
            raise ValueError("on-policy checkpoint differs from exact causal prefix")
        return expected_state


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty development sequence")
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise ValueError("development summary is nonfinite")
    return result


def _build_diagnostics_and_accounting(
    evaluator: KondoActorCriticOnPolicyEvaluator,
    final: KondoActorCriticOnPolicyRunState,
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    cfg = evaluator.config
    actor_states: tuple[object, ...] = (
        final.ordinary_actor,
        final.uniform_actor,
        final.kondo_state,
        final.reserve_state,
    )
    protected_states = (
        final.ordinary_protected,
        final.uniform_protected,
        final.kondo_protected,
        final.reserve_protected,
    )
    environments = (
        final.ordinary_environment,
        final.uniform_environment,
        final.kondo_environment,
        final.reserve_environment,
    )
    per_arm_diagnostics: dict[str, object] = {}
    per_arm_accounting: dict[str, object] = {}
    for arm, actor_state, protected_state, environment in zip(
        ARM_ORDER,
        actor_states,
        protected_states,
        environments,
        strict=True,
    ):
        arm_records = [item for item in records if item["arm"] == arm]
        if len(arm_records) != cfg.total_batches:
            raise ValueError("on-policy arm record count differs")
        actor_parameters = cast(Any, actor_state).parameters
        per_arm_diagnostics[arm] = {
            "mean_actor_loss": _mean(
                [cast(float, item["actor_loss"]) for item in arm_records]
            ),
            "mean_protected_loss": _mean(
                [cast(float, item["protected_total_loss"]) for item in arm_records]
            ),
            "final_actor_parameters_sha256": _parameter_sha256(actor_parameters),
            "final_protected_state_sha256": _protected_state_sha256(
                protected_state
            ),
            "final_environment_sha256": _canonical_sha256(
                _environment_payload(environment)
            ),
            "cumulative_reward": float(np.asarray(environment.cumulative_reward)),
            "rare_failures_collected": int(np.asarray(environment.failure_count)),
            "rare_failures_in_actor_backward": sum(
                cast(int, item["rare_failure_rows_in_actor_backward"])
                for item in arm_records
            ),
            "rare_failures_in_protected_backward": sum(
                cast(int, item["rare_failure_rows_in_protected_backward"])
                for item in arm_records
            ),
            "actor_policy_revision": int(
                np.asarray(cast(Any, actor_state).policy_revision)
            ),
            "protected_update_count": int(
                np.asarray(protected_state.update_count)
            ),
            "assessment_status": ASSESSMENT_STATUS,
        }
        actor_row_slots = sum(
            cast(int, item["actor_backward_leading_shape"]) for item in arm_records
        )
        per_arm_accounting[arm] = {
            "closed_loop_batches_collected": cfg.total_batches,
            "environment_rows_collected": cfg.total_batches * cfg.batch_size,
            "actor_update_opportunities": cfg.total_batches,
            "actor_updates_applied": cfg.total_batches,
            "actor_compiled_backward_invocations": cfg.total_batches,
            "actor_backward_row_slots": actor_row_slots,
            "protected_update_opportunities": cfg.total_batches,
            "protected_updates_applied": cfg.total_batches,
            "protected_compiled_backward_invocations": cfg.total_batches,
            "protected_backward_row_slots": cfg.total_batches * cfg.batch_size,
            "training_trace_replays": 1,
        }
    trajectory_audit = []
    for event_index in range(cfg.total_batches):
        event_records = [
            item for item in records if item["event_index"] == event_index
        ]
        action_digests = [
            cast(Mapping[str, object], item["actions"])["sha256"]
            for item in event_records
        ]
        environment_digests = [
            item["environment_after_sha256"] for item in event_records
        ]
        trajectory_audit.append(
            {
                "event_index": event_index,
                "unique_action_trace_count": len(set(action_digests)),
                "unique_environment_after_count": len(set(environment_digests)),
                "action_traces_identical": len(set(action_digests)) == 1,
                "environment_states_identical": len(set(environment_digests)) == 1,
                "equality_assumed_by_protocol": False,
            }
        )
    diagnostics = {
        "per_arm": per_arm_diagnostics,
        "trajectory_divergence_audit": trajectory_audit,
        "common_schedule_pairs_exogenous_randomness_only": True,
        "trajectory_equality_assumed": False,
        "protected_learning_shapes_fixed_across_arms": True,
        "protected_learning_values_equal_required": False,
        "rare_failure_coverage_is_descriptive": True,
        "assessment_thresholds": [],
        "assessment_status": ASSESSMENT_STATUS,
    }
    accounting = {
        "per_arm": per_arm_accounting,
        "common_exogenous_schedules": cfg.total_batches,
        "independent_closed_loop_arm_trajectories": len(ARM_ORDER),
        "actor_updates_only_at_batch_boundaries": True,
        "one_actor_update_opportunity_per_arm_batch": True,
        "one_full_protected_update_per_arm_batch": True,
        "training_budget_doubled": False,
        "wall_clock_measured": False,
        "measured_flops": False,
        "measured_memory": False,
        "measured_energy": False,
    }
    return diagnostics, accounting


def build_kondo_actor_critic_on_policy_report(
    config: KondoActorCriticOnPolicyConfig,
) -> dict[str, object]:
    """Execute the finite closed-loop protocol entirely in memory."""

    if type(config) is not KondoActorCriticOnPolicyConfig:
        raise TypeError("config must be an exact KondoActorCriticOnPolicyConfig")
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    final = evaluator.run_to_end()
    records = [
        cast(Mapping[str, object], json.loads(record))
        for record in final.records_json
    ]
    source = kondo_on_policy_source_manifest()
    runtime = kondo_on_policy_runtime_identity()
    protocol = kondo_on_policy_protocol(config)
    schedules = [
        _schedule_payload(
            evaluator.common_schedule(index),
            event_index=index,
            phase=_phase_for_event(config, index)[0],
            regime=_phase_for_event(config, index)[1],
        )
        for index in range(config.total_batches)
    ]
    uniform_schedules = [
        evaluator.uniform_control_schedule(index)
        for index in range(config.total_batches)
    ]
    diagnostics, accounting = _build_diagnostics_and_accounting(
        evaluator,
        final,
        records,
    )
    initial_snapshot = evaluator.initial_snapshot_payload()
    final_state = evaluator._state_body(final)
    body: dict[str, object] = {
        "schema": KONDO_ON_POLICY_REPORT_SCHEMA,
        "type": "KondoActorCriticOnPolicyReport",
        "config": config.to_config(),
        "protocol": protocol,
        "source_manifest": source,
        "runtime_identity": runtime,
        "initial_snapshot": initial_snapshot,
        "initial_snapshot_sha256": _canonical_sha256(initial_snapshot),
        "common_schedules": schedules,
        "common_schedules_sha256": _canonical_sha256(schedules),
        "uniform_control_schedules": uniform_schedules,
        "uniform_control_schedules_sha256": _canonical_sha256(
            uniform_schedules
        ),
        "arm_records": records,
        "arm_records_sha256": _canonical_sha256(records),
        "final_state": final_state,
        "final_state_sha256": _canonical_sha256(final_state),
        "diagnostics": diagnostics,
        "logical_resource_accounting": accounting,
        "closed_loop_control": True,
        "on_policy": True,
        "behavior_log_probability_available": True,
        "importance_correction_applied": False,
        "trajectory_equality_assumed": False,
        "output_written": False,
        "output_path": None,
        "performance_claimed": False,
        "compute_benefit_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "guardrail_authority": False,
        "evidence_claimed": False,
        "assessment_status": ASSESSMENT_STATUS,
        "verdict": ASSESSMENT_STATUS,
        "thresholds": [],
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "limitations": list(_LIMITATIONS),
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    if len(_canonical_json_bytes(report)) > _MAX_REPORT_BYTES:
        raise ValueError("on-policy report exceeds byte cap")
    return report


@dataclasses.dataclass(frozen=True)
class KondoActorCriticOnPolicyValidationReceipt:
    """Fail-closed validation receipt without promotion authority."""

    valid: bool
    assessment_status: str
    source_runtime_bound: bool
    causal_trace_replayed: bool
    exact_replay: bool
    output_written: bool
    promotion_authority: bool


def validate_kondo_actor_critic_on_policy_report(
    report: object,
) -> KondoActorCriticOnPolicyValidationReceipt:
    """Validate only by exact source/runtime-bound causal reconstruction."""

    raw = _mapping(report, name="report")
    if len(_canonical_json_bytes(raw)) > _MAX_REPORT_BYTES:
        raise ValueError("on-policy report exceeds byte cap")
    if (
        raw.get("schema") != KONDO_ON_POLICY_REPORT_SCHEMA
        or raw.get("type") != "KondoActorCriticOnPolicyReport"
    ):
        raise ValueError("on-policy report schema/type differs")
    body = {name: raw[name] for name in raw if name != "report_sha256"}
    if raw.get("report_sha256") != _canonical_sha256(body):
        raise ValueError("on-policy report digest integrity check failed")
    config_payload = _mapping(raw.get("config"), name="report.config")
    config = KondoActorCriticOnPolicyConfig.from_config(config_payload)
    if not _strict_json_equal(
        raw.get("source_manifest"),
        kondo_on_policy_source_manifest(),
    ):
        raise ValueError("on-policy report source manifest differs")
    if not _strict_json_equal(
        raw.get("runtime_identity"),
        kondo_on_policy_runtime_identity(),
    ):
        raise ValueError("on-policy report runtime identity differs")
    required: dict[str, object] = {
        "closed_loop_control": True,
        "on_policy": True,
        "behavior_log_probability_available": True,
        "importance_correction_applied": False,
        "trajectory_equality_assumed": False,
        "output_written": False,
        "output_path": None,
        "performance_claimed": False,
        "compute_benefit_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "guardrail_authority": False,
        "evidence_claimed": False,
        "assessment_status": ASSESSMENT_STATUS,
        "verdict": ASSESSMENT_STATUS,
        "thresholds": [],
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
    }
    for name, expected in required.items():
        if not _strict_json_equal(raw.get(name), expected):
            raise ValueError(f"on-policy report {name} differs")
    expected_report = build_kondo_actor_critic_on_policy_report(config)
    if not _strict_json_equal(dict(raw), expected_report):
        raise ValueError("on-policy report differs from exact causal reconstruction")
    return KondoActorCriticOnPolicyValidationReceipt(
        valid=True,
        assessment_status=ASSESSMENT_STATUS,
        source_runtime_bound=True,
        causal_trace_replayed=True,
        exact_replay=True,
        output_written=False,
        promotion_authority=False,
    )


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "CHECKPOINT_HOST_ONLY",
    "DEVELOPMENT_STATUS",
    "KONDO_ON_POLICY_CHECKPOINT_SCHEMA",
    "KONDO_ON_POLICY_CONFIG_SCHEMA",
    "KONDO_ON_POLICY_PROTOCOL_SCHEMA",
    "KONDO_ON_POLICY_REPORT_SCHEMA",
    "KondoActorCriticOnPolicyConfig",
    "KondoActorCriticOnPolicyEvaluator",
    "KondoActorCriticOnPolicyRunState",
    "KondoActorCriticOnPolicyValidationReceipt",
    "OUTPUT_WRITES",
    "OnPolicyCollectedBatch",
    "OnPolicyCollectionResult",
    "OnPolicyCommonSchedule",
    "OnPolicyEnvironmentParameters",
    "OnPolicyEnvironmentState",
    "OnPolicyManualActorState",
    "PROMOTION_AUTHORITY",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SPARSE_ARM_ORDER",
    "build_kondo_actor_critic_on_policy_report",
    "collect_on_policy_batch_kernel",
    "kondo_on_policy_protocol",
    "kondo_on_policy_runtime_identity",
    "kondo_on_policy_source_manifest",
    "on_policy_selected_log_probability",
    "sample_on_policy_actions_from_uniforms",
    "validate_kondo_actor_critic_on_policy_report",
]
