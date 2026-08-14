# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Strict nonpromoting Kondo actor/critic replay development lane.

One evaluator-owned, uninterrupted A/B/A contextual-gambling trace feeds four
independent actor-selection arms from one immutable initial snapshot:

* ``ordinary_full`` runs one full-batch actor backward;
* ``uniform_sparse`` gathers one evaluator-random capacity-sized microbatch;
* ``kondo_top_k`` lets paper delight select the capacity-sized microbatch; and
* ``kondo_top_k_reserve`` reserves a minimum random sample inside the same
  capacity before filling the remaining slots by delight.

All arms receive the exact same evaluator-fixed source actions, returns, and
protected learner inputs.  No source behavior policy is available: actor
updates are deliberately off-policy surrogate updates without importance
correction.  The log-probability used for Kondo screening is the exact current
actor probability of the recorded action, not a behavior-policy likelihood.
Baseline, critic, representation, world-model, and safety/guardrail gradients
always use every source row and are independently verified bit-identical across
arms.  Only actor backward rows differ.  Each arm has exactly one actor update
opportunity and one protected update per environment batch; timing reruns and
doubled training budgets do not exist.

This is offline, evaluator-supplied replay, not closed-loop control.  It writes
no files, uses no evidence seed or promotion path, and has no performance,
speedup, efficacy, safety, or policy-authority verdict.  SHA-256 fields provide
unkeyed integrity and source binding only.  Exact deterministic replay is the
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

REPO_ROOT = Path(__file__).resolve().parents[2]

KONDO_REPLAY_CONFIG_SCHEMA = "alberta.kondo-actor-critic-replay.config.v2"
KONDO_REPLAY_PROTOCOL_SCHEMA = "alberta.kondo-actor-critic-replay.protocol.v2"
KONDO_REPLAY_REPORT_SCHEMA = "alberta.kondo-actor-critic-replay.report.v2"
KONDO_REPLAY_CHECKPOINT_SCHEMA = "alberta.kondo-actor-critic-replay.checkpoint.v2"

DEVELOPMENT_STATUS = "not_assessed"
ASSESSMENT_STATUS = "not_assessed"
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False

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
)

_LIMITATIONS = (
    "development diagnostics only; every status and verdict is not_assessed",
    "offline evaluator-supplied replay is not closed-loop control",
    "selection mechanisms may yield different samples; every exact index is disclosed",
    "source actions are evaluator-fixed; no source behavior policy is available",
    "actor updates are off-policy surrogates without importance correction",
    "current-policy action surprisal is not a behavior-policy likelihood",
    "off-policy actor losses do not establish policy-gradient or DG efficacy",
    "the random-reserve arm is an extension and is not paper Kondo",
    "logical multiplication terms are shape proxies and are not measured FLOPs",
    "no wall-clock, memory, energy, or end-to-end latency measurement is made",
    "rare-failure coverage is descriptive and is not a physical-safety claim",
    "A/B/A recovery and retention readouts have no thresholds or efficacy verdict",
    "SHA-256 provides integrity and source binding, not keyed authenticity",
    "no result grants action, guardrail, deployment, evidence, or promotion authority",
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
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        upper = "unbounded" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {upper}]")
    return value


def _normal_float32(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive normal float32")
    parsed = float(np.float32(number))
    if not math.isfinite(parsed) or parsed < _FLOAT32_TINY:
        raise ValueError(f"{name} must be a positive normal float32")
    return parsed


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kondo_replay_source_manifest(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the complete repository source closure of this lane."""
    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


def kondo_replay_runtime_identity() -> dict[str, object]:
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
class KondoActorCriticReplayConfig:
    """Finite A/B/A replay protocol without assessment thresholds."""

    seed: int = 31
    batch_size: int = 8
    batches_per_phase: int = 2
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
    rare_failure_period: int = 2

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
            raise ValueError("action_count must be exactly two for the gambling protocol")
        target_rate = _normal_float32(self.target_rate, name="target_rate")
        if target_rate > 1.0:
            raise ValueError("target_rate must be at most one")
        object.__setattr__(self, "target_rate", target_rate)
        actor_rate = _normal_float32(
            self.actor_learning_rate,
            name="actor_learning_rate",
        )
        protected_rate = _normal_float32(
            self.protected_learning_rate,
            name="protected_learning_rate",
        )
        object.__setattr__(self, "actor_learning_rate", actor_rate)
        object.__setattr__(self, "protected_learning_rate", protected_rate)
        _exact_int(
            self.reserve_count,
            name="reserve_count",
            minimum=1,
            maximum=self.batch_size,
        )
        if self.reserve_count > self.sparse_capacity:
            raise ValueError("reserve_count cannot exceed sparse capacity")
        if self.sparse_capacity >= self.batch_size:
            raise ValueError("target_rate must yield a sparse capacity below batch_size")
        _exact_int(
            self.rare_failure_period,
            name="rare_failure_period",
            minimum=2,
            maximum=self.total_batches,
        )
        if self.total_batches * self.batch_size > _INT32_MAX:
            raise ValueError("replay accounting exceeds signed int32")
        scalar_slots = (
            self.total_batches
            * self.batch_size
            * (
                self.actor_feature_dim
                + self.context_dim
                + self.critic_dim
                + self.safety_dim
                + self.representation_dim
                + self.context_dim
                + 5
            )
        )
        if scalar_slots > _MAX_TRACE_SCALAR_SLOTS:
            raise ValueError("source trace exceeds the finite scalar-slot cap")

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
            "schema": KONDO_REPLAY_CONFIG_SCHEMA,
            "type": "KondoActorCriticReplayConfig",
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
            "rare_failure_period": self.rare_failure_period,
            "seed_role": "development-source-and-uniform-controls-only",
            "evidence_seed": None,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoActorCriticReplayConfig:
        expected_fields = set(cls().to_config())
        if set(payload) != expected_fields:
            raise ValueError("Kondo replay config fields differ")
        fixed: dict[str, object] = {
            "schema": KONDO_REPLAY_CONFIG_SCHEMA,
            "type": "KondoActorCriticReplayConfig",
            "seed_role": "development-source-and-uniform-controls-only",
            "evidence_seed": None,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
        }
        for name, expected in fixed.items():
            if not _strict_json_equal(payload.get(name), expected):
                raise ValueError(f"Kondo replay config {name} is invalid")
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
            "rare_failure_period",
        )
        for name in integer_names:
            if type(payload[name]) is not int:
                raise ValueError(f"Kondo replay config {name} must be an integer")
        for name in (
            "target_rate",
            "actor_learning_rate",
            "protected_learning_rate",
        ):
            if type(payload[name]) is not float:
                raise ValueError(f"Kondo replay config {name} must be a float")
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
            rare_failure_period=cast(int, payload["rare_failure_period"]),
        )
        if (
            payload.get("total_batches") != result.total_batches
            or payload.get("sparse_capacity") != result.sparse_capacity
        ):
            raise ValueError("Kondo replay derived config fields differ")
        if not _strict_json_equal(result.to_config(), dict(payload)):
            raise ValueError("Kondo replay config is noncanonical")
        return result


def kondo_replay_protocol(config: KondoActorCriticReplayConfig) -> dict[str, object]:
    """Return the frozen, nonpromoting mechanics contract."""
    return {
        "schema": KONDO_REPLAY_PROTOCOL_SCHEMA,
        "type": "KondoActorCriticReplayProtocol",
        "arms": list(ARM_ORDER),
        "sparse_arms": list(SPARSE_ARM_ORDER),
        "stream": "one-uninterrupted-A/B/A-contextual-gambling-trace",
        "phase_order": ["A1", "B", "A2"],
        "batches_per_phase": config.batches_per_phase,
        "external_source_trace_count": 1,
        "immutable_initial_snapshot_count": 1,
        "source_experience_equal_across_arms": True,
        "source_actions_equal_across_arms": True,
        "source_action_generation": "evaluator-fixed-alternating-actions",
        "source_behavior_policy_available": False,
        "on_policy": False,
        "importance_correction_applied": False,
        "actor_update_role": "off-policy-surrogate-development-diagnostic",
        "valid_policy_gradient_efficacy_claim": False,
        "selected_samples_forced_equal_across_arms": False,
        "selected_sample_overlap_disclosed": True,
        "selected_indices_disclosed": True,
        "offline_evaluator_supplied_replay": True,
        "closed_loop_control": False,
        "actor_update_opportunities_per_environment_batch_per_arm": 1,
        "protected_updates_per_environment_batch_per_arm": 1,
        "training_trace_replays_per_arm": 1,
        "training_budget_doubled": False,
        "uniform_and_kondo_backward_capacity_equal": True,
        "ordinary_actor_backward_shape": config.batch_size,
        "sparse_actor_backward_shape": config.sparse_capacity,
        "protected_backward_shape": config.batch_size,
        "protected_channels": [
            "baseline",
            "critic",
            "representation",
            "world_model",
            "safety_guardrail",
        ],
        "protected_learning_full_batch": True,
        "protected_learning_bit_identical_across_arms": True,
        "paper_delight": "advantage-times-current-policy-selected-action-surprisal",
        "delight_log_probability_role": (
            "exact-current-policy-probability-of-recorded-action-not-source-behavior"
        ),
        "executed_actor_backward_mask_semantics": (
            "gradient-contribution-entered-executed-actor-backward"
        ),
        "sparks_joy_scope": "KondoSparseActorResult-only",
        "manual_kernel_arms_are_kondo_sparse_actor_transactions": False,
        "ordinary_full_delight_selection_claimed": False,
        "reserve_arm_role": "nonpaper-minimum-random-reserve-within-fixed-capacity",
        "reserve_count": config.reserve_count,
        "rare_failure_stratum_present": True,
        "rare_failure_schedule": "phase-local-periodic-including-each-phase-start",
        "rare_failure_period": config.rare_failure_period,
        "rare_failure_actor_selection_is_not_a_safety_gate": True,
        "host_screen_and_gather_jittable": False,
        "fixed_backward_kernels_jittable_and_scannable": True,
        "wall_clock_measured": False,
        "measured_flops": False,
        "performance_claimed": False,
        "speedup_claimed": False,
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
    lhs = np.asarray(jax.device_get(left))
    rhs = np.asarray(jax.device_get(right))
    return (
        lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and np.ascontiguousarray(lhs).tobytes() == np.ascontiguousarray(rhs).tobytes()
    )


def _tree_all_finite(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(jax.device_get(leaf)))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_l2(tree: object) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        values = np.asarray(jax.device_get(leaf), dtype=np.float64)
        total += float(np.sum(values * values, dtype=np.float64))
    return math.sqrt(total)


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


def _initial_actor_parameters(config: KondoActorCriticReplayConfig) -> KondoActorParameters:
    def values(size: int, *, phase: float, scale: float) -> Array:
        index: npt.NDArray[np.float32] = np.arange(size, dtype=np.float32)
        raw = np.sin(index * np.float32(0.31) + np.float32(phase)) * np.float32(scale)
        return jnp.asarray(raw.astype(np.float32))

    return KondoActorParameters(
        hidden_weight=values(
            config.actor_feature_dim * config.hidden_dim,
            phase=0.1,
            scale=0.12,
        ).reshape(config.actor_feature_dim, config.hidden_dim),
        hidden_bias=values(config.hidden_dim, phase=0.3, scale=0.025),
        output_weight=values(
            config.hidden_dim * config.action_count,
            phase=0.5,
            scale=0.1,
        ).reshape(config.hidden_dim, config.action_count),
        output_bias=values(config.action_count, phase=0.7, scale=0.02),
    )


@chex.dataclass(frozen=True)
class ReplayProtectedParameters:
    """Parameters for learners that must never be actor-sample gated."""

    baseline_weight: Array
    baseline_bias: Array
    critic_weight: Array
    critic_bias: Array
    representation_weight: Array
    representation_bias: Array
    model_weight: Array
    model_bias: Array
    safety_weight: Array
    safety_bias: Array


@chex.dataclass(frozen=True)
class ReplayProtectedState:
    """One full-batch protected learner snapshot."""

    parameters: ReplayProtectedParameters
    update_count: Array


@chex.dataclass(frozen=True)
class ReplayProtectedBatch:
    """Fixed full-batch inputs for every protected learner."""

    context_features: Array
    critic_features: Array
    safety_features: Array
    actions: Array
    return_targets: Array
    representation_targets: Array
    model_targets: Array
    failure_targets: Array


@chex.dataclass(frozen=True)
class ReplayProtectedBackwardResult:
    """One real full-batch protected autodiff result."""

    total_loss: Array
    baseline_loss: Array
    critic_loss: Array
    representation_loss: Array
    model_loss: Array
    safety_loss: Array
    baseline_predictions: Array
    critic_predictions: Array
    representation_predictions: Array
    model_predictions: Array
    safety_logits: Array
    gradient: ReplayProtectedParameters
    gradient_finite: Array


def _initial_protected_parameters(
    config: KondoActorCriticReplayConfig,
) -> ReplayProtectedParameters:
    def values(size: int, *, phase: float, scale: float) -> Array:
        index: npt.NDArray[np.float32] = np.arange(size, dtype=np.float32)
        raw = np.cos(index * np.float32(0.19) + np.float32(phase)) * np.float32(scale)
        return jnp.asarray(raw.astype(np.float32))

    return ReplayProtectedParameters(
        baseline_weight=values(config.context_dim, phase=0.1, scale=0.04),
        baseline_bias=jnp.asarray(0.0, dtype=jnp.float32),
        critic_weight=values(
            config.critic_dim * config.action_count,
            phase=0.2,
            scale=0.05,
        ).reshape(config.critic_dim, config.action_count),
        critic_bias=values(config.action_count, phase=0.3, scale=0.01),
        representation_weight=values(
            config.context_dim * config.representation_dim,
            phase=0.4,
            scale=0.045,
        ).reshape(config.context_dim, config.representation_dim),
        representation_bias=values(
            config.representation_dim,
            phase=0.5,
            scale=0.01,
        ),
        model_weight=values(
            config.representation_dim * config.context_dim,
            phase=0.6,
            scale=0.04,
        ).reshape(config.representation_dim, config.context_dim),
        model_bias=values(config.context_dim, phase=0.7, scale=0.01),
        safety_weight=values(config.safety_dim, phase=0.8, scale=0.035),
        safety_bias=jnp.asarray(-0.1, dtype=jnp.float32),
    )


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


def _protected_predictions(
    parameters: ReplayProtectedParameters,
    batch: ReplayProtectedBatch,
) -> tuple[Array, Array, Array, Array, Array]:
    baseline_predictions = (
        batch.context_features @ parameters.baseline_weight + parameters.baseline_bias
    )
    critic_values = batch.critic_features @ parameters.critic_weight + parameters.critic_bias
    critic_predictions = jnp.take_along_axis(
        critic_values,
        batch.actions[:, None],
        axis=1,
    )[:, 0]
    representation_predictions = jnp.tanh(
        batch.context_features @ parameters.representation_weight + parameters.representation_bias
    )
    model_predictions = (
        batch.representation_targets @ parameters.model_weight + parameters.model_bias
    )
    safety_logits = batch.safety_features @ parameters.safety_weight + parameters.safety_bias
    return (
        baseline_predictions,
        critic_predictions,
        representation_predictions,
        model_predictions,
        safety_logits,
    )


def _protected_loss(
    parameters: ReplayProtectedParameters,
    batch: ReplayProtectedBatch,
) -> tuple[Array, tuple[Array, ...]]:
    (
        baseline_predictions,
        critic_predictions,
        representation_predictions,
        model_predictions,
        safety_logits,
    ) = _protected_predictions(parameters, batch)
    baseline_loss = jnp.mean(jnp.square(baseline_predictions - batch.return_targets))
    critic_loss = jnp.mean(jnp.square(critic_predictions - batch.return_targets))
    representation_loss = jnp.mean(
        jnp.square(representation_predictions - batch.representation_targets)
    )
    model_loss = jnp.mean(jnp.square(model_predictions - batch.model_targets))
    safety_loss = jnp.mean(jax.nn.softplus(safety_logits) - batch.failure_targets * safety_logits)
    total_loss = baseline_loss + critic_loss + representation_loss + model_loss + safety_loss
    auxiliary = (
        baseline_loss,
        critic_loss,
        representation_loss,
        model_loss,
        safety_loss,
        baseline_predictions,
        critic_predictions,
        representation_predictions,
        model_predictions,
        safety_logits,
    )
    return total_loss, auxiliary


@functools.partial(jax.jit, static_argnums=())
def replay_protected_backward_kernel(
    parameters: ReplayProtectedParameters,
    batch: ReplayProtectedBatch,
) -> ReplayProtectedBackwardResult:
    """Run all protected learners over the full fixed-size source batch."""
    (total_loss, auxiliary), gradient = jax.value_and_grad(
        _protected_loss,
        has_aux=True,
    )(parameters, batch)
    (
        baseline_loss,
        critic_loss,
        representation_loss,
        model_loss,
        safety_loss,
        baseline_predictions,
        critic_predictions,
        representation_predictions,
        model_predictions,
        safety_logits,
    ) = auxiliary
    gradient_finite = jnp.isfinite(total_loss)
    for leaf in jax.tree_util.tree_leaves(gradient):
        gradient_finite = gradient_finite & jnp.all(jnp.isfinite(leaf))
    return ReplayProtectedBackwardResult(
        total_loss=total_loss,
        baseline_loss=baseline_loss,
        critic_loss=critic_loss,
        representation_loss=representation_loss,
        model_loss=model_loss,
        safety_loss=safety_loss,
        baseline_predictions=baseline_predictions,
        critic_predictions=critic_predictions,
        representation_predictions=representation_predictions,
        model_predictions=model_predictions,
        safety_logits=safety_logits,
        gradient=cast(ReplayProtectedParameters, gradient),
        gradient_finite=gradient_finite,
    )


def _apply_protected_gradient(
    state: ReplayProtectedState,
    gradient: ReplayProtectedParameters,
    learning_rate: float,
) -> ReplayProtectedState:
    rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    parameters = cast(
        ReplayProtectedParameters,
        jax.tree_util.tree_map(
            lambda parameter, grad: parameter - rate * grad,
            state.parameters,
            gradient,
        ),
    )
    return ReplayProtectedState(
        parameters=parameters,
        update_count=state.update_count + jnp.asarray(1, dtype=jnp.int32),
    )


@dataclasses.dataclass(frozen=True)
class KondoReplaySourceBatch:
    """One immutable batch in the uninterrupted evaluator-owned trace."""

    event_index: int
    phase: str
    regime: str
    actor_features: Array
    context_features: Array
    critic_features: Array
    safety_features: Array
    actions: Array
    return_targets: Array
    representation_targets: Array
    model_targets: Array
    failure_mask: Array
    uniform_indices: Array

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

    def payload(self) -> dict[str, object]:
        body: dict[str, object] = {
            "event_index": self.event_index,
            "phase": self.phase,
            "regime": self.regime,
            "actor_features": _array_payload(self.actor_features),
            "context_features": _array_payload(self.context_features),
            "critic_features": _array_payload(self.critic_features),
            "safety_features": _array_payload(self.safety_features),
            "actions": _array_payload(self.actions),
            "return_targets": _array_payload(self.return_targets),
            "representation_targets": _array_payload(self.representation_targets),
            "model_targets": _array_payload(self.model_targets),
            "failure_mask": _array_payload(self.failure_mask),
            "uniform_indices": _array_payload(self.uniform_indices),
        }
        return {**body, "source_batch_sha256": _canonical_sha256(body)}


def _phase_for_event(
    config: KondoActorCriticReplayConfig,
    event_index: int,
) -> tuple[str, str]:
    if event_index < config.batches_per_phase:
        return "A1", "A"
    if event_index < 2 * config.batches_per_phase:
        return "B", "B"
    return "A2", "A"


def _expand_features(
    base: npt.NDArray[np.float32],
    *,
    dimension: int,
    event_index: int,
    phase: float,
) -> npt.NDArray[np.float32]:
    row_count = base.shape[0]
    output = np.empty((row_count, dimension), dtype=np.float32)
    output[:, 0] = base
    if dimension > 1:
        output[:, 1] = np.cos(
            np.arange(row_count, dtype=np.float32) * np.float32(0.41) + np.float32(phase)
        )
    for column in range(2, dimension):
        output[:, column] = np.sin(
            base * np.float32(0.23 * (column + 1))
            + np.float32(0.17 * event_index)
            + np.float32(0.11 * column)
        )
    return output


def build_kondo_replay_source_batch(
    config: KondoActorCriticReplayConfig,
    event_index: int,
) -> KondoReplaySourceBatch:
    """Construct one deterministic A/B/A gambling batch by exact index."""
    _exact_int(
        event_index,
        name="event_index",
        maximum=config.total_batches - 1,
    )
    phase_name, regime = _phase_for_event(config, event_index)
    row = np.arange(config.batch_size, dtype=np.float32)
    context_sign = np.where(
        (np.arange(config.batch_size) + event_index) % 2 == 0,
        np.float32(-1.0),
        np.float32(1.0),
    ).astype(np.float32)
    phase_value = np.float32(0.37 * (event_index + 1))
    actor_features = _expand_features(
        context_sign,
        dimension=config.actor_feature_dim,
        event_index=event_index,
        phase=float(phase_value),
    )
    context_features = _expand_features(
        context_sign,
        dimension=config.context_dim,
        event_index=event_index,
        phase=float(phase_value + np.float32(0.2)),
    )
    critic_features = _expand_features(
        context_sign,
        dimension=config.critic_dim,
        event_index=event_index,
        phase=float(phase_value + np.float32(0.4)),
    )
    safety_base = np.full((config.batch_size,), np.float32(-1.0), dtype=np.float32)
    failure_mask = np.zeros((config.batch_size,), dtype=np.bool_)
    phase_event_index = event_index % config.batches_per_phase
    if phase_event_index % config.rare_failure_period == 0:
        failure_index = (3 * event_index + 1) % config.batch_size
        failure_mask[failure_index] = True
        safety_base[failure_index] = np.float32(1.0)
    safety_features = _expand_features(
        safety_base,
        dimension=config.safety_dim,
        event_index=event_index,
        phase=float(phase_value + np.float32(0.6)),
    )
    actions = (np.arange(config.batch_size, dtype=np.int32) + np.int32(event_index)) % np.int32(
        config.action_count
    )
    base_optimal = (context_sign > 0.0).astype(np.int32)
    optimal = base_optimal if regime == "A" else np.int32(1) - base_optimal
    reward = np.where(actions == optimal, np.float32(1.0), np.float32(-1.0))
    reward = reward + np.sin(row * np.float32(0.29) + phase_value) * np.float32(0.05)
    reward = reward - failure_mask.astype(np.float32) * np.float32(2.0)
    representation_targets = np.empty(
        (config.batch_size, config.representation_dim),
        dtype=np.float32,
    )
    for column in range(config.representation_dim):
        direction = np.float32(1.0 if regime == "A" else -1.0)
        representation_targets[:, column] = np.tanh(
            context_sign * np.float32(0.35 * (column + 1))
            + direction * np.float32(0.18)
            + row * np.float32(0.01 * (column + 1))
        ).astype(np.float32)
    model_targets = np.roll(context_features, shift=-1, axis=0).astype(np.float32)
    uniform_key = jr.fold_in(jr.key(config.seed, impl="threefry2x32"), event_index)
    uniform_indices = jr.permutation(uniform_key, config.batch_size)[
        : config.sparse_capacity
    ].astype(jnp.int32)
    return KondoReplaySourceBatch(
        event_index=event_index,
        phase=phase_name,
        regime=regime,
        actor_features=jnp.asarray(actor_features, dtype=jnp.float32),
        context_features=jnp.asarray(context_features, dtype=jnp.float32),
        critic_features=jnp.asarray(critic_features, dtype=jnp.float32),
        safety_features=jnp.asarray(safety_features, dtype=jnp.float32),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        return_targets=jnp.asarray(reward, dtype=jnp.float32),
        representation_targets=jnp.asarray(
            representation_targets,
            dtype=jnp.float32,
        ),
        model_targets=jnp.asarray(model_targets, dtype=jnp.float32),
        failure_mask=jnp.asarray(failure_mask, dtype=jnp.bool_),
        uniform_indices=uniform_indices,
    )


def _selected_log_probability(
    parameters: KondoActorParameters,
    actor_features: Array,
    actions: Array,
) -> Array:
    hidden = jnp.tanh(actor_features @ parameters.hidden_weight + parameters.hidden_bias)
    logits = hidden @ parameters.output_weight + parameters.output_bias
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probabilities, actions[:, None], axis=1)[:, 0]


def _apply_actor_gradient(
    parameters: KondoActorParameters,
    gradient: KondoActorParameters,
    learning_rate: float,
) -> KondoActorParameters:
    rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    return cast(
        KondoActorParameters,
        jax.tree_util.tree_map(
            lambda parameter, grad: parameter - rate * grad,
            parameters,
            gradient,
        ),
    )


def _probe_actor_loss(
    parameters: KondoActorParameters,
    config: KondoActorCriticReplayConfig,
    regime: Literal["A", "B"],
) -> float:
    row = np.arange(config.batch_size, dtype=np.float32)
    context_sign = np.where(
        np.arange(config.batch_size) % 2 == 0,
        np.float32(-1.0),
        np.float32(1.0),
    ).astype(np.float32)
    features = _expand_features(
        context_sign,
        dimension=config.actor_feature_dim,
        event_index=0,
        phase=0.91 if regime == "A" else 1.27,
    )
    actions = np.arange(config.batch_size, dtype=np.int32) % np.int32(2)
    base_optimal = (context_sign > 0.0).astype(np.int32)
    optimal = base_optimal if regime == "A" else np.int32(1) - base_optimal
    returns = np.where(actions == optimal, np.float32(1.0), np.float32(-1.0))
    returns = returns + np.cos(row * np.float32(0.13)) * np.float32(0.02)
    log_probability = _selected_log_probability(
        parameters,
        jnp.asarray(features, dtype=jnp.float32),
        jnp.asarray(actions, dtype=jnp.int32),
    )
    loss = -jnp.mean(jnp.asarray(returns, dtype=jnp.float32) * log_probability)
    value = float(np.asarray(jax.device_get(loss)))
    if not math.isfinite(value):
        raise ValueError("actor probe loss is nonfinite")
    return value


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


def _actor_protected_payload(protected: KondoActorProtectedInputs) -> dict[str, object]:
    return {
        name: _array_payload(getattr(protected, name))
        for name in (
            "critic_features",
            "baseline_predictions",
            "return_targets",
            "safety_features",
        )
    }


def _tree_bits_equal(left: object, right: object) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        _array_bits_equal(cast(Array, lhs), cast(Array, rhs))
        for lhs, rhs in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(cast(Any, leaf).shape), str(cast(Any, leaf).dtype))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


@dataclasses.dataclass(frozen=True)
class _ActorArmOutcome:
    parameters_before: KondoActorParameters
    parameters_after: KondoActorParameters
    current_action_log_probability: Array
    advantage: Array
    delight: Array
    selected_mask: Array
    selected_by_delight: Array
    selected_by_uniform_control: Array
    minimum_random_reserve: Array
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
    parameters: KondoActorParameters,
    source: KondoReplaySourceBatch,
    advantage: Array,
    config: KondoActorCriticReplayConfig,
) -> _ActorArmOutcome:
    current_log_probability = _selected_log_probability(
        parameters,
        source.actor_features,
        source.actions,
    )
    delight = jax.lax.stop_gradient(advantage * (-current_log_probability))
    order: tuple[str, ...]
    if arm == "ordinary_full":
        gathered_indices = tuple(range(config.batch_size))
        selected_mask = jnp.ones((config.batch_size,), dtype=jnp.bool_)
        backward_batch = KondoActorBackwardBatch(
            actor_features=source.actor_features,
            actions=source.actions,
            advantage=advantage,
            sample_mask=selected_mask,
        )
        sparse = False
        order = ("full-forward", "full-batch-compiled-backward")
        selected_by_uniform = jnp.zeros_like(selected_mask)
    else:
        gathered_indices = tuple(int(item) for item in np.asarray(source.uniform_indices))
        selected_mask = (
            jnp.zeros((config.batch_size,), dtype=jnp.bool_).at[source.uniform_indices].set(True)
        )
        backward_batch = KondoActorBackwardBatch(
            actor_features=source.actor_features[source.uniform_indices],
            actions=source.actions[source.uniform_indices],
            advantage=advantage[source.uniform_indices],
            sample_mask=jnp.ones((config.sparse_capacity,), dtype=jnp.bool_),
        )
        sparse = True
        order = ("full-forward", "evaluator-uniform-gather", "compiled-backward")
        selected_by_uniform = selected_mask
    backward = kondo_actor_backward_kernel(parameters, backward_batch)
    parameters_after = _apply_actor_gradient(
        parameters,
        backward.gradient,
        config.actor_learning_rate,
    )
    if not bool(np.asarray(backward.gradient_finite)) or not _tree_all_finite(parameters_after):
        raise ValueError(f"{arm} actor backward/update produced nonfinite values")
    return _ActorArmOutcome(
        parameters_before=parameters,
        parameters_after=parameters_after,
        current_action_log_probability=current_log_probability,
        advantage=advantage,
        delight=delight,
        selected_mask=selected_mask,
        selected_by_delight=jnp.zeros_like(selected_mask),
        selected_by_uniform_control=selected_by_uniform,
        minimum_random_reserve=jnp.zeros_like(selected_mask),
        gathered_indices=gathered_indices,
        backward=backward,
        backward_leading_shape=(
            config.batch_size if arm == "ordinary_full" else config.sparse_capacity
        ),
        sparse_backward=sparse,
        screen_gather_order=order,
        random_draw_count=0,
        internal_protected_digest=None,
    )


def _digest_words_hex(words: Array) -> str:
    host = np.asarray(jax.device_get(words), dtype=np.uint32)
    return "".join(f"{int(word):08x}" for word in host)


def _kondo_actor_outcome(
    *,
    actor: KondoSparseActor,
    state: KondoSparseActorState,
    source: KondoReplaySourceBatch,
    protected: KondoActorProtectedInputs,
    config: KondoActorCriticReplayConfig,
    reserve: bool,
) -> tuple[_ActorArmOutcome, KondoSparseActorState]:
    current_log_probability = actor.behavior_log_probability(
        state,
        source.actor_features,
        source.actions,
    )
    batch = KondoSparseActorBatch(
        actor_features=source.actor_features,
        actions=source.actions,
        action_identity=source.actions,
        policy_revision=jnp.full(
            (config.batch_size,),
            state.policy_revision,
            dtype=jnp.int32,
        ),
        # KondoSparseActor's core field name is historical.  These are exact
        # current-policy bits for an evaluator-fixed action, not a source
        # behavior-policy likelihood.
        behavior_log_probability=current_log_probability,
        valid_mask=jnp.ones((config.batch_size,), dtype=jnp.bool_),
        force_keep_mask=jnp.zeros((config.batch_size,), dtype=jnp.bool_),
        protected=protected,
    )
    result: KondoSparseActorResult = actor.step(state, batch)
    if (
        not bool(np.asarray(result.transaction_applied))
        or not bool(np.asarray(result.sparse_backward_used))
        or bool(np.asarray(result.full_shape_masked_backward_used))
    ):
        name = "reserve Kondo" if reserve else "paper Kondo"
        raise ValueError(f"{name} actor transaction did not use sparse backward")
    if int(np.asarray(result.backward_batch_size)) != config.sparse_capacity:
        raise ValueError("Kondo actor backward capacity differs")
    if not _tree_bits_equal(result.protected, protected):
        raise ValueError("Kondo actor changed protected full-batch inputs")
    selected_indices = np.asarray(result.screen.selected_indices, dtype=np.int32)
    slot_mask = np.asarray(result.screen.selected_slot_mask, dtype=np.bool_)
    gathered_indices = tuple(int(item) for item in selected_indices[slot_mask])
    backward = KondoActorBackwardSummary(
        loss=result.actor_loss,
        gradient=result.gradient,
        selected_count=result.screen.selected_count,
        gradient_finite=result.gradient_finite,
    )
    outcome = _ActorArmOutcome(
        parameters_before=state.parameters,
        parameters_after=result.state.parameters,
        current_action_log_probability=result.current_action_log_probability,
        advantage=result.advantage,
        delight=result.screen.delight,
        selected_mask=result.sparks_joy,
        selected_by_delight=result.screen.selected_by_delight_gate,
        selected_by_uniform_control=jnp.zeros((config.batch_size,), dtype=jnp.bool_),
        minimum_random_reserve=result.screen.uniformly_reserved,
        gathered_indices=gathered_indices,
        backward=backward,
        backward_leading_shape=config.sparse_capacity,
        sparse_backward=True,
        screen_gather_order=(
            "full-forward",
            "detached-paper-delight-screen",
            "minimum-random-reserve" if reserve else "no-random-reserve",
            "audited-capacity-gather",
            "compiled-backward",
        ),
        random_draw_count=int(np.asarray(result.screen.random_draw_count)),
        internal_protected_digest=_digest_words_hex(result.protected_digest),
    )
    return outcome, result.state


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


def _arm_record(
    *,
    arm: ArmName,
    event_index: int,
    source: KondoReplaySourceBatch,
    outcome: _ActorArmOutcome,
    protected_input: KondoActorProtectedInputs,
    protected_result: ReplayProtectedBackwardResult,
    protected_before: ReplayProtectedState,
    protected_after: ReplayProtectedState,
    config: KondoActorCriticReplayConfig,
) -> dict[str, object]:
    behavior_recomputed = _selected_log_probability(
        outcome.parameters_before,
        source.actor_features,
        source.actions,
    )
    if not _array_bits_equal(behavior_recomputed, outcome.current_action_log_probability):
        raise ValueError(f"{arm} current-policy action binding differs")
    expected_delight = jax.lax.stop_gradient(
        outcome.advantage * (-outcome.current_action_log_probability)
    )
    if not _array_bits_equal(expected_delight, outcome.delight):
        raise ValueError(f"{arm} paper delight differs")
    selected_mask = np.asarray(outcome.selected_mask, dtype=np.bool_)
    selected_indices = np.flatnonzero(selected_mask).astype(np.int32).tolist()
    selected_count = int(np.asarray(outcome.backward.selected_count))
    if selected_count != len(selected_indices) or selected_count != len(outcome.gathered_indices):
        raise ValueError(f"{arm} selected-row accounting differs")
    expected_shape = config.batch_size if arm == "ordinary_full" else config.sparse_capacity
    if outcome.backward_leading_shape != expected_shape:
        raise ValueError(f"{arm} actor backward shape differs")
    failure = np.asarray(source.failure_mask, dtype=np.bool_)
    selected_failure_count = int(np.sum(selected_mask & failure))
    protected_payload = _actor_protected_payload(protected_input)
    protected_learning_payload = _protected_batch_payload(source.protected_batch())
    mechanism = {
        "ordinary_full": "all-valid-source-rows",
        "uniform_sparse": "evaluator-threefry-without-replacement",
        "kondo_top_k": "paper-delight-top-k",
        "kondo_top_k_reserve": "paper-delight-top-k-with-minimum-random-reserve",
    }[arm]
    return {
        "arm": arm,
        "event_index": event_index,
        "phase": source.phase,
        "regime": source.regime,
        "source_batch_sha256": source.payload()["source_batch_sha256"],
        "source_experience_replays_in_arm": 1,
        "policy_revision_before": event_index,
        "policy_revision_after": event_index + 1,
        "source_actions_sha256": _array_payload(source.actions)["sha256"],
        "action_identity_sha256": _array_payload(source.actions)["sha256"],
        "current_policy_selected_action_log_probability": _array_payload(
            outcome.current_action_log_probability
        ),
        "current_policy_log_probability_revision_binding_exact": True,
        "source_action_identity_exact": True,
        "source_behavior_policy_available": False,
        "on_policy": False,
        "importance_correction_applied": False,
        "actor_update_role": "off-policy-surrogate-development-diagnostic",
        "valid_policy_gradient_efficacy_claim": False,
        "advantage": _array_payload(outcome.advantage),
        "selected_action_surprisal": _array_payload(-outcome.current_action_log_probability),
        "selected_action_surprisal_semantics": (
            "current-policy-surprisal-of-evaluator-fixed-recorded-action"
        ),
        "paper_delight": _array_payload(outcome.delight),
        "executed_actor_backward_mask": _array_payload(outcome.selected_mask),
        "executed_actor_backward_mask_semantics": (
            "gradient-contribution-entered-executed-actor-backward"
        ),
        "selected_source_indices": selected_indices,
        "actor_backward_gather_order": list(outcome.gathered_indices),
        "selected_count": selected_count,
        "selected_rows_forced_equal_across_arms": False,
        "selection_mechanism": mechanism,
        "proposed_by_delight_gate": _array_payload(outcome.selected_by_delight),
        "delight_gate_proposal_semantics": (
            "pre-reserve proposal; executed_actor_backward_mask records actual "
            "backward inclusion"
        ),
        "selected_by_uniform_control": _array_payload(outcome.selected_by_uniform_control),
        "minimum_random_reserve": _array_payload(outcome.minimum_random_reserve),
        "random_draw_count": outcome.random_draw_count,
        "evaluator_uniform_permutations": int(arm == "uniform_sparse"),
        "rare_failure_rows_in_source": int(np.sum(failure)),
        "rare_failure_rows_in_actor_backward": selected_failure_count,
        "rare_failure_rows_in_protected_backward": int(np.sum(failure)),
        "actor_parameters_before_sha256": _parameter_sha256(outcome.parameters_before),
        "actor_parameters_after_sha256": _parameter_sha256(outcome.parameters_after),
        "actor_loss": float(np.asarray(outcome.backward.loss)),
        "actor_loss_semantics": ("uncorrected-off-policy-surrogate-not-policy-gradient-efficacy"),
        "actor_gradient_l2": _tree_l2(outcome.backward.gradient),
        "actor_gradient_finite": bool(np.asarray(outcome.backward.gradient_finite)),
        "actor_update_opportunities": 1,
        "actor_updates_applied": 1,
        "actor_compiled_backward_invocations": 1,
        "actor_backward_leading_shape": outcome.backward_leading_shape,
        "sparse_actor_backward": outcome.sparse_backward,
        "screen_gather_backward_order": list(outcome.screen_gather_order),
        "internal_kondo_protected_digest": outcome.internal_protected_digest,
        "actor_protected_inputs_sha256": _canonical_sha256(protected_payload),
        "protected_learning_batch_sha256": _canonical_sha256(protected_learning_payload),
        "protected_state_before_sha256": _protected_state_sha256(protected_before),
        "protected_state_after_sha256": _protected_state_sha256(protected_after),
        "protected_predictions_sha256": _protected_prediction_sha256(protected_result),
        "protected_total_loss": float(np.asarray(protected_result.total_loss)),
        "baseline_loss": float(np.asarray(protected_result.baseline_loss)),
        "critic_loss": float(np.asarray(protected_result.critic_loss)),
        "representation_loss": float(np.asarray(protected_result.representation_loss)),
        "world_model_loss": float(np.asarray(protected_result.model_loss)),
        "safety_guardrail_loss": float(np.asarray(protected_result.safety_loss)),
        "protected_gradient_l2": _tree_l2(protected_result.gradient),
        "protected_gradient_finite": bool(np.asarray(protected_result.gradient_finite)),
        "protected_channels_full_batch": True,
        "protected_rows_in_backward": config.batch_size,
        "protected_update_opportunities": 1,
        "protected_updates_applied": 1,
        "protected_compiled_backward_invocations": 1,
        "protected_backward_leading_shape": config.batch_size,
        "probe_a_actor_loss_before": _probe_actor_loss(
            outcome.parameters_before,
            config,
            "A",
        ),
        "probe_a_actor_loss_after": _probe_actor_loss(
            outcome.parameters_after,
            config,
            "A",
        ),
        "probe_b_actor_loss_before": _probe_actor_loss(
            outcome.parameters_before,
            config,
            "B",
        ),
        "probe_b_actor_loss_after": _probe_actor_loss(
            outcome.parameters_after,
            config,
            "B",
        ),
        "probe_forward_evaluations": 4,
        "probe_backward_invocations": 0,
        "assessment_status": ASSESSMENT_STATUS,
    }


@dataclasses.dataclass(frozen=True)
class KondoActorCriticReplayRunState:
    """Immutable replay prefix for exact checkpoint reconstruction."""

    event_index: int
    ordinary_parameters: KondoActorParameters
    uniform_parameters: KondoActorParameters
    kondo_state: KondoSparseActorState
    reserve_state: KondoSparseActorState
    ordinary_protected: ReplayProtectedState
    uniform_protected: ReplayProtectedState
    kondo_protected: ReplayProtectedState
    reserve_protected: ReplayProtectedState
    records_json: tuple[str, ...]


class KondoActorCriticReplayEvaluator:
    """Host-orchestrated A/B/A replay with fixed JAX backward kernels."""

    def __init__(self, config: KondoActorCriticReplayConfig):
        if not isinstance(config, KondoActorCriticReplayConfig):
            raise TypeError("config must be KondoActorCriticReplayConfig")
        self.config = config
        self.kondo_actor = KondoSparseActor(config.actor_config(reserve=False))
        self.reserve_actor = KondoSparseActor(config.actor_config(reserve=True))
        self.initial_actor_parameters = _initial_actor_parameters(config)
        self.initial_protected_parameters = _initial_protected_parameters(config)
        self._actor_signature = _tree_signature(self.initial_actor_parameters)
        self._protected_signature = _tree_signature(self.initial_protected_parameters)

    def source_batch(self, event_index: int) -> KondoReplaySourceBatch:
        return build_kondo_replay_source_batch(self.config, event_index)

    def init(self) -> KondoActorCriticReplayRunState:
        root = jr.key(self.config.seed, impl="threefry2x32")
        kondo_key = jr.fold_in(root, 30_001)
        reserve_key = jr.fold_in(root, 30_002)
        protected = ReplayProtectedState(
            parameters=self.initial_protected_parameters,
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return KondoActorCriticReplayRunState(
            event_index=0,
            ordinary_parameters=self.initial_actor_parameters,
            uniform_parameters=self.initial_actor_parameters,
            kondo_state=self.kondo_actor.init(
                self.initial_actor_parameters,
                kondo_key,
            ),
            reserve_state=self.reserve_actor.init(
                self.initial_actor_parameters,
                reserve_key,
            ),
            ordinary_protected=protected,
            uniform_protected=protected,
            kondo_protected=protected,
            reserve_protected=protected,
            records_json=(),
        )

    def _valid_protected_state(
        self,
        state: ReplayProtectedState,
        *,
        event_index: int,
    ) -> bool:
        return (
            isinstance(state, ReplayProtectedState)
            and _tree_signature(state.parameters) == self._protected_signature
            and _tree_all_finite(state.parameters)
            and getattr(state.update_count, "shape", None) == ()
            and getattr(state.update_count, "dtype", None) == jnp.dtype(jnp.int32)
            and int(np.asarray(state.update_count)) == event_index
        )

    def _valid_state(self, state: KondoActorCriticReplayRunState) -> bool:
        if not isinstance(state, KondoActorCriticReplayRunState):
            return False
        if not 0 <= state.event_index <= self.config.total_batches:
            return False
        if len(state.records_json) != state.event_index * len(ARM_ORDER):
            return False
        for parameters in (state.ordinary_parameters, state.uniform_parameters):
            if _tree_signature(parameters) != self._actor_signature or not _tree_all_finite(
                parameters
            ):
                return False
        protected_states = (
            state.ordinary_protected,
            state.uniform_protected,
            state.kondo_protected,
            state.reserve_protected,
        )
        if any(
            not self._valid_protected_state(item, event_index=state.event_index)
            for item in protected_states
        ):
            return False
        if len({_protected_state_sha256(item) for item in protected_states}) != 1:
            return False
        try:
            self.kondo_actor.checkpoint_payload(state.kondo_state)
            self.reserve_actor.checkpoint_payload(state.reserve_state)
            for record in state.records_json:
                parsed = json.loads(record)
                if not isinstance(parsed, Mapping):
                    return False
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            int(np.asarray(state.kondo_state.policy_revision)) == state.event_index
            and int(np.asarray(state.reserve_state.policy_revision)) == state.event_index
        )

    def advance(
        self,
        state: KondoActorCriticReplayRunState,
    ) -> KondoActorCriticReplayRunState:
        if not self._valid_state(state):
            raise ValueError("Kondo replay run state is invalid")
        if state.event_index >= self.config.total_batches:
            raise ValueError("Kondo replay run is already complete")
        source = self.source_batch(state.event_index)
        protected_batch = source.protected_batch()
        protected_before = (
            state.ordinary_protected,
            state.uniform_protected,
            state.kondo_protected,
            state.reserve_protected,
        )
        protected_results = tuple(
            replay_protected_backward_kernel(item.parameters, protected_batch)
            for item in protected_before
        )
        if any(not bool(np.asarray(result.gradient_finite)) for result in protected_results):
            raise ValueError("protected full-batch backward produced nonfinite values")
        reference_result = protected_results[0]
        if any(not _tree_bits_equal(reference_result, result) for result in protected_results[1:]):
            raise ValueError("protected learner results are not bit-identical across arms")
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
        if len({_protected_state_sha256(item) for item in protected_after}) != 1:
            raise ValueError("protected updates are not bit-identical across arms")
        actor_protected = tuple(
            KondoActorProtectedInputs(
                critic_features=source.critic_features,
                baseline_predictions=result.baseline_predictions,
                return_targets=source.return_targets,
                safety_features=source.safety_features,
            )
            for result in protected_results
        )
        if (
            len({_canonical_sha256(_actor_protected_payload(item)) for item in actor_protected})
            != 1
        ):
            raise ValueError("actor protected inputs differ across arms")
        advantages = tuple(
            jax.lax.stop_gradient(source.return_targets - item.baseline_predictions)
            for item in actor_protected
        )
        ordinary_outcome = _manual_actor_outcome(
            arm="ordinary_full",
            parameters=state.ordinary_parameters,
            source=source,
            advantage=advantages[0],
            config=self.config,
        )
        uniform_outcome = _manual_actor_outcome(
            arm="uniform_sparse",
            parameters=state.uniform_parameters,
            source=source,
            advantage=advantages[1],
            config=self.config,
        )
        kondo_outcome, kondo_state = _kondo_actor_outcome(
            actor=self.kondo_actor,
            state=state.kondo_state,
            source=source,
            protected=actor_protected[2],
            config=self.config,
            reserve=False,
        )
        reserve_outcome, reserve_state = _kondo_actor_outcome(
            actor=self.reserve_actor,
            state=state.reserve_state,
            source=source,
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
        records = tuple(
            _arm_record(
                arm=arm,
                event_index=state.event_index,
                source=source,
                outcome=outcome,
                protected_input=protected_input,
                protected_result=protected_result,
                protected_before=before,
                protected_after=after,
                config=self.config,
            )
            for arm, outcome, protected_input, protected_result, before, after in zip(
                ARM_ORDER,
                outcomes,
                actor_protected,
                protected_results,
                protected_before,
                protected_after,
                strict=True,
            )
        )
        next_state = KondoActorCriticReplayRunState(
            event_index=state.event_index + 1,
            ordinary_parameters=ordinary_outcome.parameters_after,
            uniform_parameters=uniform_outcome.parameters_after,
            kondo_state=kondo_state,
            reserve_state=reserve_state,
            ordinary_protected=protected_after[0],
            uniform_protected=protected_after[1],
            kondo_protected=protected_after[2],
            reserve_protected=protected_after[3],
            records_json=state.records_json
            + tuple(_canonical_json_bytes(record).decode("utf-8") for record in records),
        )
        if not self._valid_state(next_state):
            raise ValueError("Kondo replay next state is invalid")
        return next_state

    def run_to_end(
        self,
        state: KondoActorCriticReplayRunState | None = None,
    ) -> KondoActorCriticReplayRunState:
        current = self.init() if state is None else state
        while current.event_index < self.config.total_batches:
            current = self.advance(current)
        return current

    def checkpoint_payload(
        self,
        state: KondoActorCriticReplayRunState,
    ) -> dict[str, object]:
        """Serialize one source/runtime-bound prefix without granting authority."""
        if not self._valid_state(state):
            raise ValueError("cannot checkpoint invalid Kondo replay state")
        source = kondo_replay_source_manifest()
        runtime = kondo_replay_runtime_identity()
        protocol = kondo_replay_protocol(self.config)
        initial_snapshot = {
            "actor_parameters": _parameter_payload(self.initial_actor_parameters),
            "protected_parameters": _protected_parameter_payload(self.initial_protected_parameters),
        }
        source_trace = [
            self.source_batch(index).payload() for index in range(self.config.total_batches)
        ]
        source_prefix = source_trace[: state.event_index]
        body: dict[str, object] = {
            "schema": KONDO_REPLAY_CHECKPOINT_SCHEMA,
            "type": "KondoActorCriticReplayCheckpoint",
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(protocol),
            "source_manifest_sha256": _canonical_sha256(source),
            "runtime_sha256": _canonical_sha256(runtime),
            "initial_snapshot_sha256": _canonical_sha256(initial_snapshot),
            "source_trace_sha256": _canonical_sha256(source_trace),
            "source_prefix_sha256": _canonical_sha256(source_prefix),
            "event_index": state.event_index,
            "ordinary_parameters": _parameter_payload(state.ordinary_parameters),
            "uniform_parameters": _parameter_payload(state.uniform_parameters),
            "kondo_state": self.kondo_actor.checkpoint_payload(state.kondo_state),
            "reserve_state": self.reserve_actor.checkpoint_payload(state.reserve_state),
            "ordinary_protected": _protected_state_payload(state.ordinary_protected),
            "uniform_protected": _protected_state_payload(state.uniform_protected),
            "kondo_protected": _protected_state_payload(state.kondo_protected),
            "reserve_protected": _protected_state_payload(state.reserve_protected),
            "records": [json.loads(record) for record in state.records_json],
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
        }
        payload = {**body, "checkpoint_sha256": _canonical_sha256(body)}
        if len(_canonical_json_bytes(payload)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Kondo replay checkpoint exceeds byte cap")
        return payload

    def restore_checkpoint(self, payload: object) -> KondoActorCriticReplayRunState:
        """Restore only an exact causally reconstructed prefix."""
        raw = _mapping(payload, name="checkpoint")
        if len(_canonical_json_bytes(raw)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Kondo replay checkpoint exceeds byte cap")
        expected_fields = {
            "schema",
            "type",
            "config_sha256",
            "protocol_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
            "initial_snapshot_sha256",
            "source_trace_sha256",
            "source_prefix_sha256",
            "event_index",
            "ordinary_parameters",
            "uniform_parameters",
            "kondo_state",
            "reserve_state",
            "ordinary_protected",
            "uniform_protected",
            "kondo_protected",
            "reserve_protected",
            "records",
            "assessment_status",
            "promotion_authority",
            "checkpoint_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError("Kondo replay checkpoint fields differ")
        if (
            raw.get("schema") != KONDO_REPLAY_CHECKPOINT_SCHEMA
            or raw.get("type") != "KondoActorCriticReplayCheckpoint"
        ):
            raise ValueError("Kondo replay checkpoint schema/type differs")
        if (
            raw.get("assessment_status") != ASSESSMENT_STATUS
            or raw.get("promotion_authority") is not False
        ):
            raise ValueError("Kondo replay checkpoint authority fields differ")
        body = {name: raw[name] for name in raw if name != "checkpoint_sha256"}
        if raw.get("checkpoint_sha256") != _canonical_sha256(body):
            raise ValueError("Kondo replay checkpoint digest integrity check failed")
        bindings = {
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(kondo_replay_protocol(self.config)),
            "source_manifest_sha256": _canonical_sha256(kondo_replay_source_manifest()),
            "runtime_sha256": _canonical_sha256(kondo_replay_runtime_identity()),
            "initial_snapshot_sha256": _canonical_sha256(
                {
                    "actor_parameters": _parameter_payload(self.initial_actor_parameters),
                    "protected_parameters": _protected_parameter_payload(
                        self.initial_protected_parameters
                    ),
                }
            ),
            "source_trace_sha256": _canonical_sha256(
                [self.source_batch(index).payload() for index in range(self.config.total_batches)]
            ),
        }
        if any(raw.get(name) != expected for name, expected in bindings.items()):
            raise ValueError("Kondo replay checkpoint source/runtime binding differs")
        event_index = _exact_int(
            raw.get("event_index"),
            name="event_index",
            maximum=self.config.total_batches,
        )
        expected_prefix_sha256 = _canonical_sha256(
            [self.source_batch(index).payload() for index in range(event_index)]
        )
        if raw.get("source_prefix_sha256") != expected_prefix_sha256:
            raise ValueError("Kondo replay checkpoint source prefix binding differs")
        expected_state = self.init()
        for _ in range(event_index):
            expected_state = self.advance(expected_state)
        expected_payload = self.checkpoint_payload(expected_state)
        if not _strict_json_equal(dict(raw), expected_payload):
            raise ValueError("Kondo replay checkpoint differs from exact causal prefix")
        return expected_state


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty development sequence")
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise ValueError("development summary is nonfinite")
    return result


def _safety_stratum_losses(
    parameters: ReplayProtectedParameters,
    sources: Sequence[KondoReplaySourceBatch],
) -> dict[str, float]:
    logits_parts: list[npt.NDArray[np.float64]] = []
    target_parts: list[npt.NDArray[np.bool_]] = []
    for source in sources:
        logits = source.safety_features @ parameters.safety_weight + parameters.safety_bias
        logits_parts.append(np.asarray(jax.device_get(logits), dtype=np.float64))
        target_parts.append(np.asarray(jax.device_get(source.failure_mask), dtype=np.bool_))
    logits_all = np.concatenate(logits_parts)
    target_all = np.concatenate(target_parts)
    loss = np.logaddexp(0.0, logits_all) - target_all.astype(np.float64) * logits_all
    if not np.any(target_all) or not np.any(~target_all):
        raise ValueError("safety diagnostic requires both rare and common strata")
    rare_loss = float(np.mean(loss[target_all], dtype=np.float64))
    common_loss = float(np.mean(loss[~target_all], dtype=np.float64))
    if not math.isfinite(rare_loss) or not math.isfinite(common_loss):
        raise ValueError("safety stratum loss is nonfinite")
    return {
        "rare_failure_log_loss": rare_loss,
        "common_log_loss": common_loss,
    }


def _phase_loss_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for phase in ("A1", "B", "A2"):
        selected = [record for record in records if record["phase"] == phase]
        result[phase] = {
            name: _mean([cast(float, record[name]) for record in selected])
            for name in (
                "actor_loss",
                "baseline_loss",
                "critic_loss",
                "representation_loss",
                "world_model_loss",
                "safety_guardrail_loss",
            )
        }
    return result


def _recurrence_diagnostics(
    records: Sequence[Mapping[str, object]],
    config: KondoActorCriticReplayConfig,
) -> dict[str, float]:
    first = records[0]
    post_a1 = records[config.batches_per_phase - 1]
    post_b = records[2 * config.batches_per_phase - 1]
    final = records[-1]
    initial_a = cast(float, first["probe_a_actor_loss_before"])
    initial_b = cast(float, first["probe_b_actor_loss_before"])
    post_a1_a = cast(float, post_a1["probe_a_actor_loss_after"])
    post_a1_b = cast(float, post_a1["probe_b_actor_loss_after"])
    post_b_a = cast(float, post_b["probe_a_actor_loss_after"])
    post_b_b = cast(float, post_b["probe_b_actor_loss_after"])
    final_a = cast(float, final["probe_a_actor_loss_after"])
    final_b = cast(float, final["probe_b_actor_loss_after"])
    return {
        "initial_a_probe_loss": initial_a,
        "initial_b_probe_loss": initial_b,
        "post_a1_a_probe_loss": post_a1_a,
        "post_a1_b_probe_loss": post_a1_b,
        "post_b_a_probe_loss": post_b_a,
        "post_b_b_probe_loss": post_b_b,
        "final_a_probe_loss": final_a,
        "final_b_probe_loss": final_b,
        "first_a_learning_delta": initial_a - post_a1_a,
        "b_adaptation_delta": post_a1_b - post_b_b,
        "a_recovery_delta": post_b_a - final_a,
        "a_cycle_retention_delta": final_a - post_a1_a,
        "b_retention_delta": final_b - post_b_b,
    }


def _protected_parameter_count(parameters: ReplayProtectedParameters) -> int:
    return sum(int(cast(Any, leaf).size) for leaf in jax.tree_util.tree_leaves(parameters))


def _build_diagnostics_and_accounting(
    *,
    config: KondoActorCriticReplayConfig,
    evaluator: KondoActorCriticReplayEvaluator,
    final: KondoActorCriticReplayRunState,
    records: Sequence[Mapping[str, object]],
    sources: Sequence[KondoReplaySourceBatch],
) -> tuple[dict[str, object], dict[str, object]]:
    final_actor_parameters: dict[ArmName, KondoActorParameters] = {
        "ordinary_full": final.ordinary_parameters,
        "uniform_sparse": final.uniform_parameters,
        "kondo_top_k": final.kondo_state.parameters,
        "kondo_top_k_reserve": final.reserve_state.parameters,
    }
    final_protected: dict[ArmName, ReplayProtectedState] = {
        "ordinary_full": final.ordinary_protected,
        "uniform_sparse": final.uniform_protected,
        "kondo_top_k": final.kondo_protected,
        "kondo_top_k_reserve": final.reserve_protected,
    }
    initial_safety = _safety_stratum_losses(
        evaluator.initial_protected_parameters,
        sources,
    )
    total_rare = sum(
        int(np.sum(np.asarray(source.failure_mask, dtype=np.int32))) for source in sources
    )
    per_arm_diagnostics: dict[str, object] = {}
    per_arm_accounting: dict[str, object] = {}
    actor_terms_per_row = (
        config.actor_feature_dim * config.hidden_dim + config.hidden_dim * config.action_count
    )
    protected_terms_per_row = (
        config.context_dim
        + config.critic_dim * config.action_count
        + config.context_dim * config.representation_dim
        + config.representation_dim * config.context_dim
        + config.safety_dim
    )
    for arm in ARM_ORDER:
        arm_records = [record for record in records if record["arm"] == arm]
        selected_rare = sum(
            cast(int, record["rare_failure_rows_in_actor_backward"]) for record in arm_records
        )
        protected_rare = sum(
            cast(int, record["rare_failure_rows_in_protected_backward"]) for record in arm_records
        )
        actor_slots = sum(
            cast(int, record["actor_backward_leading_shape"]) for record in arm_records
        )
        protected_slots = sum(
            cast(int, record["protected_backward_leading_shape"]) for record in arm_records
        )
        per_arm_diagnostics[arm] = {
            "phase_mean_losses": _phase_loss_summary(arm_records),
            "recurrence_recovery_retention": _recurrence_diagnostics(
                arm_records,
                config,
            ),
            "rare_failure_source_rows": total_rare,
            "rare_failure_actor_selected_rows": selected_rare,
            "rare_failure_protected_learning_rows": protected_rare,
            "initial_safety_stratum_losses": initial_safety,
            "final_safety_stratum_losses": _safety_stratum_losses(
                final_protected[arm].parameters,
                sources,
            ),
            "final_actor_parameter_sha256": _parameter_sha256(final_actor_parameters[arm]),
            "final_protected_state_sha256": _protected_state_sha256(final_protected[arm]),
            "assessment_status": ASSESSMENT_STATUS,
        }
        per_arm_accounting[arm] = {
            "source_batches_consumed": len(arm_records),
            "source_trace_replays": 1,
            "actor_update_opportunities": sum(
                cast(int, record["actor_update_opportunities"]) for record in arm_records
            ),
            "actor_updates_applied": sum(
                cast(int, record["actor_updates_applied"]) for record in arm_records
            ),
            "actor_compiled_backward_invocations": sum(
                cast(int, record["actor_compiled_backward_invocations"]) for record in arm_records
            ),
            "actor_backward_row_slots": actor_slots,
            "actor_selected_samples": sum(
                cast(int, record["selected_count"]) for record in arm_records
            ),
            "protected_update_opportunities": sum(
                cast(int, record["protected_update_opportunities"]) for record in arm_records
            ),
            "protected_updates_applied": sum(
                cast(int, record["protected_updates_applied"]) for record in arm_records
            ),
            "protected_compiled_backward_invocations": sum(
                cast(int, record["protected_compiled_backward_invocations"])
                for record in arm_records
            ),
            "protected_backward_row_slots": protected_slots,
            "evaluator_uniform_permutations": sum(
                cast(int, record["evaluator_uniform_permutations"]) for record in arm_records
            ),
            "gate_random_draw_count": sum(
                cast(int, record["random_draw_count"]) for record in arm_records
            ),
            "actor_forward_multiplication_term_shape_proxy": (actor_slots * actor_terms_per_row),
            "protected_forward_multiplication_term_shape_proxy": (
                protected_slots * protected_terms_per_row
            ),
        }
    protected_digests = {
        cast(
            str,
            _mapping(per_arm_diagnostics[arm], name="diagnostic")["final_protected_state_sha256"],
        )
        for arm in ARM_ORDER
    }
    if len(protected_digests) != 1:
        raise ValueError("final protected states are not bit-identical")
    diagnostics: dict[str, object] = {
        "per_arm": per_arm_diagnostics,
        "protected_final_state_bit_identical_across_arms": True,
        "protected_final_state_sha256": next(iter(protected_digests)),
        "rare_failure_source_rows": total_rare,
        "rare_failure_total_source_rows": config.total_batches * config.batch_size,
        "rare_failure_coverage_is_descriptive": True,
        "recurrence_recovery_retention_thresholds": [],
        "assessment_status": ASSESSMENT_STATUS,
    }
    accounting: dict[str, object] = {
        "unique_environment_batches": config.total_batches,
        "deterministic_training_trace_executions": 1,
        "training_trace_replays_per_arm": 1,
        "experience_double_counted_within_arm": False,
        "updates_per_environment_batch_per_arm": {
            "actor": 1,
            "protected": 1,
        },
        "initial_actor_parameter_count": sum(
            int(cast(Any, leaf).size)
            for leaf in jax.tree_util.tree_leaves(evaluator.initial_actor_parameters)
        ),
        "initial_protected_parameter_count": _protected_parameter_count(
            evaluator.initial_protected_parameters
        ),
        "per_arm": per_arm_accounting,
        "logical_proxy_semantics": (
            "backward leading row slots times dense forward multiplication terms; "
            "excludes derivatives, compiler fusion, host work, and hardware"
        ),
        "measured_flops": False,
        "wall_clock_measured": False,
    }
    return diagnostics, accounting


def build_kondo_actor_critic_replay_report(
    config: KondoActorCriticReplayConfig | None = None,
) -> dict[str, object]:
    """Run the strict in-memory replay and return a nonpromoting report."""
    cfg = config or KondoActorCriticReplayConfig()
    evaluator = KondoActorCriticReplayEvaluator(cfg)
    final = evaluator.run_to_end()
    records = [cast(Mapping[str, object], json.loads(item)) for item in final.records_json]
    sources = [evaluator.source_batch(index) for index in range(cfg.total_batches)]
    source_trace = [source.payload() for source in sources]
    diagnostics, accounting = _build_diagnostics_and_accounting(
        config=cfg,
        evaluator=evaluator,
        final=final,
        records=records,
        sources=sources,
    )
    source_manifest = kondo_replay_source_manifest()
    runtime = kondo_replay_runtime_identity()
    protocol = kondo_replay_protocol(cfg)
    initial_snapshot = {
        "actor_parameters": _parameter_payload(evaluator.initial_actor_parameters),
        "protected_parameters": _protected_parameter_payload(
            evaluator.initial_protected_parameters
        ),
    }
    body: dict[str, object] = {
        "schema": KONDO_REPLAY_REPORT_SCHEMA,
        "type": "KondoActorCriticReplayDevelopmentReport",
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "speedup_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "guardrail_authority": False,
        "source_behavior_policy_available": False,
        "on_policy": False,
        "importance_correction_applied": False,
        "valid_policy_gradient_efficacy_claim": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "config": cfg.to_config(),
        "config_sha256": _canonical_sha256(cfg.to_config()),
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "runtime": runtime,
        "runtime_sha256": _canonical_sha256(runtime),
        "initial_snapshot": initial_snapshot,
        "initial_snapshot_sha256": _canonical_sha256(initial_snapshot),
        "source_trace": source_trace,
        "source_trace_sha256": _canonical_sha256(source_trace),
        "arm_records": [dict(record) for record in records],
        "arm_records_sha256": _canonical_sha256(records),
        "diagnostics": diagnostics,
        "logical_resource_accounting": accounting,
        "limitations": list(_LIMITATIONS),
        "evidence_seed": None,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
    }
    payload = {**body, "report_sha256": _canonical_sha256(body)}
    if len(_canonical_json_bytes(payload)) > _MAX_REPORT_BYTES:
        raise ValueError("Kondo replay report exceeds byte cap")
    return payload


@dataclasses.dataclass(frozen=True)
class KondoActorCriticReplayValidationReceipt:
    """Strict source/runtime-bound exact replay receipt."""

    valid: bool
    assessment_status: str
    source_runtime_bound: bool
    causal_trace_replayed: bool
    exact_replay: bool
    output_written: bool
    promotion_authority: bool


def validate_kondo_actor_critic_replay_report(
    report: object,
) -> KondoActorCriticReplayValidationReceipt:
    """Fail closed on structure, integrity, provenance, or exact replay drift."""
    raw = _mapping(report, name="report")
    if len(_canonical_json_bytes(raw)) > _MAX_REPORT_BYTES:
        raise ValueError("Kondo replay report exceeds byte cap")
    expected_fields = {
        "schema",
        "type",
        "development_status",
        "assessment_status",
        "performance_claimed",
        "speedup_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "guardrail_authority",
        "source_behavior_policy_available",
        "on_policy",
        "importance_correction_applied",
        "valid_policy_gradient_efficacy_claim",
        "output_writes",
        "promotion_authority",
        "scientific_promotion_allowed",
        "config",
        "config_sha256",
        "protocol",
        "protocol_sha256",
        "source_manifest",
        "source_manifest_sha256",
        "runtime",
        "runtime_sha256",
        "initial_snapshot",
        "initial_snapshot_sha256",
        "source_trace",
        "source_trace_sha256",
        "arm_records",
        "arm_records_sha256",
        "diagnostics",
        "logical_resource_accounting",
        "limitations",
        "evidence_seed",
        "thresholds",
        "verdict",
        "report_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("Kondo replay report fields differ")
    fixed: dict[str, object] = {
        "schema": KONDO_REPLAY_REPORT_SCHEMA,
        "type": "KondoActorCriticReplayDevelopmentReport",
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "speedup_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "guardrail_authority": False,
        "source_behavior_policy_available": False,
        "on_policy": False,
        "importance_correction_applied": False,
        "valid_policy_gradient_efficacy_claim": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "evidence_seed": None,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(raw.get(name), expected):
            raise ValueError(f"Kondo replay report {name} is invalid")
    body = {name: raw[name] for name in raw if name != "report_sha256"}
    if raw.get("report_sha256") != _canonical_sha256(body):
        raise ValueError("Kondo replay report digest integrity check failed")
    config_payload = _mapping(raw.get("config"), name="config")
    config = KondoActorCriticReplayConfig.from_config(config_payload)
    expected = build_kondo_actor_critic_replay_report(config)
    if not _strict_json_equal(dict(raw), expected):
        raise ValueError("Kondo replay exact deterministic reconstruction differs")
    return KondoActorCriticReplayValidationReceipt(
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
    "DEVELOPMENT_STATUS",
    "KONDO_REPLAY_CHECKPOINT_SCHEMA",
    "KONDO_REPLAY_CONFIG_SCHEMA",
    "KONDO_REPLAY_PROTOCOL_SCHEMA",
    "KONDO_REPLAY_REPORT_SCHEMA",
    "KondoActorCriticReplayConfig",
    "KondoActorCriticReplayEvaluator",
    "KondoActorCriticReplayRunState",
    "KondoActorCriticReplayValidationReceipt",
    "KondoReplaySourceBatch",
    "PROMOTION_AUTHORITY",
    "ReplayProtectedBackwardResult",
    "ReplayProtectedBatch",
    "ReplayProtectedParameters",
    "ReplayProtectedState",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SPARSE_ARM_ORDER",
    "build_kondo_actor_critic_replay_report",
    "build_kondo_replay_source_batch",
    "kondo_replay_protocol",
    "kondo_replay_runtime_identity",
    "kondo_replay_source_manifest",
    "replay_protected_backward_kernel",
    "validate_kondo_actor_critic_replay_report",
]
