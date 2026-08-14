# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Strict nonpromoting development evaluation of real Kondo actor backwards.

One evaluator-owned immutable source trace and one initial nonlinear actor
snapshot feed four arms:

``ordinary_full``
    One full-shape ordinary actor backward over every valid source row.
``uniform_sparse``
    One evaluator-random fixed-capacity gather and actor backward.  Its input
    shape exactly matches the Kondo sparse arm; its independently selected
    rows may differ and both arms disclose their exact source indices.
``kondo_top_k``
    Paper delight screening followed by the audited Kondo gather and a real
    fixed-capacity actor backward.
``kondo_overflow_diagnostic``
    A forced-sample overflow stress that must execute the explicit full-shape
    masked fallback.  It is diagnostic and is not a fair efficacy arm.

Every arm consumes each external source batch exactly once and begins from the
same parameter bits.  Experience is matched, while selected samples and
backward leading shapes are disclosed rather than described as equal.  Timing
uses compiled backwards only, after a warmup that blocks all output buffers.
Trials are interleaved in an evaluator-owned order and use
``time.perf_counter_ns`` by default.  Raw durations and nearest-rank p50/p95
are descriptive noisy observations: no threshold, verdict, speedup, efficacy,
safety, or promotion claim exists.

Host screen/gather orchestration is excluded from backward timing.  The report
does not measure accelerator memory, energy, or end-to-end learner latency.
Deterministic bytes are integrity-bound and exactly replayed.  Wall-clock
bytes are excluded from that bit-exact replay and instead receive their own
unkeyed digest bound to the deterministic report, source manifest, runtime,
and trial order.  No files or pinned outputs are written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.extend import backend as jax_backend

from alberta_framework.core.kondo_gate import KondoGate, KondoGateConfig
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorBackwardBatch,
    KondoActorBackwardResult,
    KondoActorBackwardSummary,
    KondoActorParameters,
    KondoActorProtectedInputs,
    KondoSparseActor,
    KondoSparseActorBatch,
    KondoSparseActorConfig,
    KondoSparseActorState,
    kondo_actor_backward_kernel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

KONDO_SPARSE_ACTOR_DEVELOPMENT_CONFIG_SCHEMA = (
    "alberta.kondo-sparse-actor-development.config.v2"
)
KONDO_SPARSE_ACTOR_DEVELOPMENT_PROTOCOL_SCHEMA = (
    "alberta.kondo-sparse-actor-development.protocol.v2"
)
KONDO_SPARSE_ACTOR_DEVELOPMENT_DETERMINISTIC_SCHEMA = (
    "alberta.kondo-sparse-actor-development.deterministic.v2"
)
KONDO_SPARSE_ACTOR_DEVELOPMENT_TIMING_SCHEMA = (
    "alberta.kondo-sparse-actor-development.timing.v2"
)
KONDO_SPARSE_ACTOR_DEVELOPMENT_REPORT_SCHEMA = (
    "alberta.kondo-sparse-actor-development.report.v2"
)
KONDO_SPARSE_ACTOR_DEVELOPMENT_CHECKPOINT_SCHEMA = (
    "alberta.kondo-sparse-actor-development.checkpoint.v2"
)
DEVELOPMENT_STATUS = "not_assessed"
ASSESSMENT_STATUS = "not_assessed"
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False

ArmName = Literal[
    "ordinary_full",
    "uniform_sparse",
    "kondo_top_k",
    "kondo_overflow_diagnostic",
]
ARM_ORDER: tuple[ArmName, ...] = (
    "ordinary_full",
    "uniform_sparse",
    "kondo_top_k",
    "kondo_overflow_diagnostic",
)
MATCHED_DEVELOPMENT_ARMS: tuple[ArmName, ...] = (
    "ordinary_full",
    "uniform_sparse",
    "kondo_top_k",
)
DIAGNOSTIC_ARMS: tuple[ArmName, ...] = ("kondo_overflow_diagnostic",)

_UINT32_MAX = 2**32 - 1
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_BATCH_SIZE = 512
_MAX_DIMENSION = 512
_MAX_BATCHES = 64
_MAX_TIMING_TRIALS = 128
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_MAX_DETERMINISTIC_BYTES = 96 * 1024 * 1024
_MAX_REPORT_BYTES = 128 * 1024 * 1024
_MAX_TRACE_SCALAR_SLOTS = 4_000_000
_SOURCE_PATHS = (
    Path("alberta_framework/core/kondo_gate.py"),
    Path("alberta_framework/core/kondo_sparse_actor.py"),
    Path("alberta_framework/evaluation/kondo_sparse_actor_development.py"),
)
_LIMITATIONS = (
    "development diagnostics only; every status is not_assessed",
    "the deterministic replay stream is not a closed-loop environment",
    "ordinary full and sparse arms intentionally select unequal sample counts",
    "the overflow stress is diagnostic and is not a fair efficacy comparator",
    "compiled-backward timing excludes host screening and gathering",
    "perf_counter_ns samples are noisy descriptive observations without thresholds",
    "accelerator memory, energy, and end-to-end learner latency are not measured",
    "logical multiplication-term counts are shape proxies, not measured FLOPs",
    "held-out-within-development metrics are descriptive and are not evidence seeds",
    "SHA-256 provides integrity and source binding, not keyed external authenticity",
    "no result grants policy authority, safety authority, or promotion authority",
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


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real number")
    parsed = float(np.float32(float(value)))
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}float32")
    return parsed


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kondo_sparse_actor_development_source_manifest(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Hash the complete repository source closure used by this evaluator."""
    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


def kondo_sparse_actor_development_runtime_identity() -> dict[str, object]:
    """Return observable, non-secret runtime/backend provenance.

    Exact deterministic replay remains the authoritative compatibility check;
    opaque compiler decisions and host settings without stable public APIs are
    deliberately outside this manifest.
    """
    devices = jax.devices()
    backend = jax_backend.get_backend()
    return {
        "identity_scope": (
            "observable-nonsecret-python-jax-xla-device-and-config-fields; "
            "exact-deterministic-replay-remains-authoritative"
        ),
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "chex": version("chex"),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "backend": str(backend.platform),
        "backend_platform_version": str(backend.platform_version),
        "device_count": len(devices),
        "local_device_count": int(jax.local_device_count()),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
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
class KondoSparseActorDevelopmentConfig:
    """Finite evaluator protocol without thresholds or evidence seeds."""

    seed: int = 7
    timing_order_seed: int = 19
    batch_size: int = 8
    num_batches: int = 4
    feature_dim: int = 3
    hidden_dim: int = 5
    action_count: int = 3
    critic_dim: int = 2
    safety_dim: int = 2
    target_rate: float = 0.25
    learning_rate: float = 0.02
    timing_trials: int = 7

    def __post_init__(self) -> None:
        for name in ("seed", "timing_order_seed"):
            _exact_int(getattr(self, name), name=name, maximum=_UINT32_MAX)
        _exact_int(
            self.batch_size,
            name="batch_size",
            minimum=2,
            maximum=_MAX_BATCH_SIZE,
        )
        _exact_int(
            self.num_batches,
            name="num_batches",
            minimum=1,
            maximum=_MAX_BATCHES,
        )
        for name in ("feature_dim", "hidden_dim", "action_count", "critic_dim", "safety_dim"):
            _exact_int(
                getattr(self, name),
                name=name,
                minimum=1,
                maximum=_MAX_DIMENSION,
            )
        _exact_int(
            self.timing_trials,
            name="timing_trials",
            minimum=1,
            maximum=_MAX_TIMING_TRIALS,
        )
        target_rate = _finite_float(self.target_rate, name="target_rate", positive=True)
        if target_rate > 1.0:
            raise ValueError("target_rate must be at most one")
        object.__setattr__(self, "target_rate", target_rate)
        learning_rate = _finite_float(
            self.learning_rate,
            name="learning_rate",
            positive=True,
        )
        if learning_rate < _FLOAT32_TINY:
            raise ValueError("learning_rate must be a positive normal float32")
        object.__setattr__(self, "learning_rate", learning_rate)
        capacity = max(
            1,
            min(
                self.batch_size,
                int(np.rint(np.float32(target_rate) * np.float32(self.batch_size))),
            ),
        )
        if capacity >= self.batch_size:
            raise ValueError("target_rate must yield a capacity smaller than batch_size")
        if capacity + 1 > self.batch_size:
            raise ValueError("configuration must leave one forced-overflow row")
        trace_scalar_slots = (
            (self.num_batches + 2)
            * self.batch_size
            * (
                self.feature_dim
                + self.critic_dim
                + self.safety_dim
                + 5
            )
        )
        if trace_scalar_slots > _MAX_TRACE_SCALAR_SLOTS:
            raise ValueError("source trace exceeds the finite scalar-slot cap")

    @property
    def sparse_capacity(self) -> int:
        return max(
            1,
            min(
                self.batch_size,
                int(
                    np.rint(
                        np.float32(self.target_rate) * np.float32(self.batch_size)
                    )
                ),
            ),
        )

    def actor_config(self) -> KondoSparseActorConfig:
        return KondoSparseActorConfig(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            action_count=self.action_count,
            critic_dim=self.critic_dim,
            safety_dim=self.safety_dim,
            learning_rate=self.learning_rate,
            gate=KondoGateConfig(
                batch_size=self.batch_size,
                mode="top_k_rate",
                target_rate=self.target_rate,
                max_screenings=self.num_batches,
            ),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_CONFIG_SCHEMA,
            "type": "KondoSparseActorDevelopmentConfig",
            "seed": self.seed,
            "timing_order_seed": self.timing_order_seed,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "action_count": self.action_count,
            "critic_dim": self.critic_dim,
            "safety_dim": self.safety_dim,
            "target_rate": self.target_rate,
            "learning_rate": self.learning_rate,
            "timing_trials": self.timing_trials,
            "sparse_capacity": self.sparse_capacity,
            "seed_role": "development-trace-and-uniform-control-only",
            "evidence_seed": None,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> KondoSparseActorDevelopmentConfig:
        expected = {
            "schema",
            "type",
            "seed",
            "timing_order_seed",
            "batch_size",
            "num_batches",
            "feature_dim",
            "hidden_dim",
            "action_count",
            "critic_dim",
            "safety_dim",
            "target_rate",
            "learning_rate",
            "timing_trials",
            "sparse_capacity",
            "seed_role",
            "evidence_seed",
            "development_status",
            "assessment_status",
            "promotion_authority",
            "scientific_promotion_allowed",
            "thresholds",
        }
        if set(payload) != expected:
            raise ValueError("Kondo sparse actor development config fields differ")
        fixed: dict[str, object] = {
            "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_CONFIG_SCHEMA,
            "type": "KondoSparseActorDevelopmentConfig",
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "thresholds": [],
            "seed_role": "development-trace-and-uniform-control-only",
            "evidence_seed": None,
        }
        for name, expected_value in fixed.items():
            if not _strict_json_equal(payload.get(name), expected_value):
                raise ValueError(f"Kondo sparse actor development {name} is invalid")
        integer_names = (
            "seed",
            "timing_order_seed",
            "batch_size",
            "num_batches",
            "feature_dim",
            "hidden_dim",
            "action_count",
            "critic_dim",
            "safety_dim",
            "timing_trials",
        )
        for name in integer_names:
            if type(payload[name]) is not int:
                raise ValueError(f"Kondo sparse actor development {name} must be an integer")
        for name in ("target_rate", "learning_rate"):
            if type(payload[name]) is not float:
                raise ValueError(f"Kondo sparse actor development {name} must be a float")
        result = cls(
            seed=cast(int, payload["seed"]),
            timing_order_seed=cast(int, payload["timing_order_seed"]),
            batch_size=cast(int, payload["batch_size"]),
            num_batches=cast(int, payload["num_batches"]),
            feature_dim=cast(int, payload["feature_dim"]),
            hidden_dim=cast(int, payload["hidden_dim"]),
            action_count=cast(int, payload["action_count"]),
            critic_dim=cast(int, payload["critic_dim"]),
            safety_dim=cast(int, payload["safety_dim"]),
            target_rate=cast(float, payload["target_rate"]),
            learning_rate=cast(float, payload["learning_rate"]),
            timing_trials=cast(int, payload["timing_trials"]),
        )
        if type(payload["sparse_capacity"]) is not int or (
            payload["sparse_capacity"] != result.sparse_capacity
        ):
            raise ValueError("Kondo sparse actor development sparse_capacity is invalid")
        if not _strict_json_equal(result.to_config(), dict(payload)):
            raise ValueError("Kondo sparse actor development config is noncanonical")
        return result


def kondo_sparse_actor_development_protocol(
    config: KondoSparseActorDevelopmentConfig,
) -> dict[str, object]:
    """Return the frozen nonpromoting protocol contract."""
    return {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_PROTOCOL_SCHEMA,
        "type": "KondoSparseActorDevelopmentProtocol",
        "arms": list(ARM_ORDER),
        "matched_development_arms": list(MATCHED_DEVELOPMENT_ARMS),
        "diagnostic_only_arms": list(DIAGNOSTIC_ARMS),
        "initial_parameter_snapshots": 1,
        "source_trace_replays_per_arm": 1,
        "external_source_experience_equal": True,
        "source_actions_equal": True,
        "source_protected_inputs_equal": True,
        "closed_loop_environment_experience_measured": False,
        "selected_samples_equal": False,
        "uniform_and_kondo_backward_capacity_equal": True,
        "uniform_selection": "typed-threefry-permutation-without-replacement",
        "uniform_selection_seed": config.seed,
        "kondo_selection": "top-k-paper-delight-lowest-source-index-ties",
        "delight_semantics": "advantage-times-selected-action-surprisal",
        "executed_actor_backward_inclusion_semantics": (
            "gradient-contribution-entered-executed-actor-backward"
        ),
        "sparks_joy_scope": "KondoSparseActorResult-only",
        "manual_kernel_arms_are_kondo_sparse_actor_transactions": False,
        "ordinary_full_delight_selection_claimed": False,
        "overflow_trigger": "lowest-source-indices-forced-capacity-plus-one",
        "overflow_role": "diagnostic-not-fair-efficacy-arm",
        "updates_per_source_batch_per_arm": 1,
        "training_backward_invocations_per_arm": config.num_batches,
        "heldout_backward_invocations_per_arm": 1,
        "heldout_updates_per_arm": 0,
        "timing_scope": "compiled-actor-backward-only",
        "timing_trials_per_arm": config.timing_trials,
        "timing_warmups_per_arm": 1,
        "timing_order": "interleaved-evaluator-owned-threefry-permutation",
        "timing_summary": "nearest-rank-p50-p95",
        "evidence_seed": None,
        "promotion_seed_eligible": False,
        "host_screen_gather_timed": False,
        "block_until_ready": True,
        "accelerator_memory_measured": False,
        "energy_measured": False,
        "end_to_end_latency_measured": False,
        "thresholds": [],
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "compute_saving_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
    }


def _array_payload(array: Array) -> dict[str, object]:
    host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
    return {
        "dtype": host.dtype.str,
        "shape": list(host.shape),
        "data_hex": host.tobytes().hex(),
        "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
    }


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


def _tree_l2(tree: object) -> float:
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        values = np.asarray(jax.device_get(leaf), dtype=np.float64)
        total += float(np.sum(values * values, dtype=np.float64))
    return math.sqrt(total)


def _parameter_delta_l2(
    parameters: KondoActorParameters,
    initial: KondoActorParameters,
) -> float:
    delta = jax.tree_util.tree_map(lambda left, right: left - right, parameters, initial)
    return _tree_l2(delta)


def _initial_parameters(config: KondoSparseActorDevelopmentConfig) -> KondoActorParameters:
    def values(size: int, *, phase: float, scale: float) -> Array:
        index: npt.NDArray[np.float32] = np.arange(size, dtype=np.float32)
        raw = np.sin(index * np.float32(0.37) + np.float32(phase)) * np.float32(scale)
        return jnp.asarray(raw.astype(np.float32))

    return KondoActorParameters(
        hidden_weight=values(
            config.feature_dim * config.hidden_dim,
            phase=0.1,
            scale=0.15,
        ).reshape(config.feature_dim, config.hidden_dim),
        hidden_bias=values(config.hidden_dim, phase=0.3, scale=0.03),
        output_weight=values(
            config.hidden_dim * config.action_count,
            phase=0.5,
            scale=0.12,
        ).reshape(config.hidden_dim, config.action_count),
        output_bias=values(config.action_count, phase=0.7, scale=0.02),
    )


@dataclasses.dataclass(frozen=True)
class _SourceBatch:
    actor_features: Array
    actions: Array
    baseline_predictions: Array
    return_targets: Array
    critic_features: Array
    safety_features: Array
    uniform_indices: Array

    @property
    def advantage(self) -> Array:
        return self.return_targets - self.baseline_predictions

    def payload(self, *, event_index: int, role: str) -> dict[str, object]:
        body: dict[str, object] = {
            "event_index": event_index,
            "role": role,
            "actor_features": _array_payload(self.actor_features),
            "actions": _array_payload(self.actions),
            "baseline_predictions": _array_payload(self.baseline_predictions),
            "return_targets": _array_payload(self.return_targets),
            "critic_features": _array_payload(self.critic_features),
            "safety_features": _array_payload(self.safety_features),
            "uniform_indices": _array_payload(self.uniform_indices),
        }
        return {**body, "source_batch_sha256": _canonical_sha256(body)}


def _source_batch(
    config: KondoSparseActorDevelopmentConfig,
    event_index: int,
) -> _SourceBatch:
    batch = config.batch_size
    feature_index: npt.NDArray[np.float32] = np.arange(
        batch * config.feature_dim,
        dtype=np.float32,
    ).reshape(batch, config.feature_dim)
    phase = np.float32(0.29 * (event_index + 1))
    actor_features = np.sin(feature_index * np.float32(0.17) + phase).astype(np.float32)
    actions = (
        np.arange(batch, dtype=np.int32) * np.int32(2) + np.int32(event_index)
    ) % np.int32(config.action_count)
    row: npt.NDArray[np.float32] = np.arange(batch, dtype=np.float32)
    baseline = (
        np.cos(row * np.float32(0.23) + phase) * np.float32(0.2)
    ).astype(np.float32)
    advantage = (
        np.sin(row * np.float32(0.61) + phase) * np.float32(1.1)
        + np.where((np.arange(batch) + event_index) % 3 == 0, 0.4, -0.1)
    ).astype(np.float32)
    critic_index: npt.NDArray[np.float32] = np.arange(
        batch * config.critic_dim,
        dtype=np.float32,
    ).reshape(batch, config.critic_dim)
    safety_index: npt.NDArray[np.float32] = np.arange(
        batch * config.safety_dim,
        dtype=np.float32,
    ).reshape(batch, config.safety_dim)
    critic = np.cos(critic_index * np.float32(0.11) + phase).astype(np.float32)
    safety = np.sin(safety_index * np.float32(0.07) - phase).astype(np.float32)
    key = jr.fold_in(jr.key(config.seed, impl="threefry2x32"), event_index)
    permutation = jr.permutation(key, config.batch_size).astype(jnp.int32)
    uniform_indices = permutation[: config.sparse_capacity]
    return _SourceBatch(
        actor_features=jnp.asarray(actor_features, dtype=jnp.float32),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        baseline_predictions=jnp.asarray(baseline, dtype=jnp.float32),
        return_targets=jnp.asarray(baseline + advantage, dtype=jnp.float32),
        critic_features=jnp.asarray(critic, dtype=jnp.float32),
        safety_features=jnp.asarray(safety, dtype=jnp.float32),
        uniform_indices=uniform_indices,
    )


def _selected_log_probability(
    parameters: KondoActorParameters,
    features: Array,
    actions: Array,
) -> Array:
    hidden = jnp.tanh(features @ parameters.hidden_weight + parameters.hidden_bias)
    logits = hidden @ parameters.output_weight + parameters.output_bias
    log_probability = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probability, actions[:, None], axis=1)[:, 0]


def _apply_gradient(
    parameters: KondoActorParameters,
    gradient: KondoActorParameters,
    learning_rate: float,
) -> KondoActorParameters:
    return cast(
        KondoActorParameters,
        jax.tree_util.tree_map(
            lambda parameter, grad: parameter
            - jnp.asarray(learning_rate, dtype=jnp.float32) * grad,
            parameters,
            gradient,
        ),
    )


def _backward_record(
    *,
    arm: ArmName,
    event_index: int,
    source_sha256: str,
    revision_before: int,
    behavior_log_probability: Array,
    parameters_before: KondoActorParameters,
    parameters_after: KondoActorParameters,
    backward: KondoActorBackwardResult | KondoActorBackwardSummary,
    selected_indices: Sequence[int],
    selected_count: int,
    backward_slots: int,
    sparse: bool,
    full_fallback: bool,
    screen_gather_order: Sequence[str],
) -> dict[str, object]:
    return {
        "arm": arm,
        "event_index": event_index,
        "source_batch_sha256": source_sha256,
        "policy_revision_before": revision_before,
        "policy_revision_after": revision_before + 1,
        "behavior_log_probability": _array_payload(behavior_log_probability),
        "parameters_before_sha256": _parameter_sha256(parameters_before),
        "parameters_after_sha256": _parameter_sha256(parameters_after),
        "selected_indices": list(selected_indices),
        "selected_count": selected_count,
        "backward_leading_shape": backward_slots,
        "sparse_backward": sparse,
        "full_shape_masked_fallback": full_fallback,
        "screen_gather_backward_order": list(screen_gather_order),
        "actor_loss": float(np.asarray(backward.loss)),
        "gradient_l2": _tree_l2(backward.gradient),
        "gradient_finite": bool(np.asarray(backward.gradient_finite)),
        "compiled_backward_invocations": 1,
        "training_updates": 1,
        "assessment_status": ASSESSMENT_STATUS,
    }


@dataclasses.dataclass(frozen=True)
class KondoSparseActorDevelopmentRunState:
    """Immutable prefix state for integrity-checked deterministic resume."""

    event_index: int
    ordinary_parameters: KondoActorParameters
    uniform_parameters: KondoActorParameters
    kondo_state: KondoSparseActorState
    overflow_state: KondoSparseActorState
    records_json: tuple[str, ...]


class KondoSparseActorDevelopmentEvaluator:
    """Deterministic four-arm runner with exact prefix replay."""

    def __init__(self, config: KondoSparseActorDevelopmentConfig):
        if not isinstance(config, KondoSparseActorDevelopmentConfig):
            raise TypeError("config must be KondoSparseActorDevelopmentConfig")
        self.config = config
        self.actor = KondoSparseActor(config.actor_config())
        self.initial_parameters = _initial_parameters(config)

    def init(self) -> KondoSparseActorDevelopmentRunState:
        kondo_key = jr.fold_in(jr.key(self.config.seed, impl="threefry2x32"), 10_001)
        overflow_key = jr.fold_in(jr.key(self.config.seed, impl="threefry2x32"), 10_002)
        return KondoSparseActorDevelopmentRunState(
            event_index=0,
            ordinary_parameters=self.initial_parameters,
            uniform_parameters=self.initial_parameters,
            kondo_state=self.actor.init(self.initial_parameters, kondo_key),
            overflow_state=self.actor.init(self.initial_parameters, overflow_key),
            records_json=(),
        )

    def _valid_state(self, state: KondoSparseActorDevelopmentRunState) -> bool:
        if not isinstance(state, KondoSparseActorDevelopmentRunState):
            return False
        if not 0 <= state.event_index <= self.config.num_batches:
            return False
        if len(state.records_json) != state.event_index * len(ARM_ORDER):
            return False
        if any(not math.isfinite(value) for value in (
            _tree_l2(state.ordinary_parameters),
            _tree_l2(state.uniform_parameters),
        )):
            return False
        try:
            self.actor.checkpoint_payload(state.kondo_state)
            self.actor.checkpoint_payload(state.overflow_state)
            for encoded in state.records_json:
                json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            int(np.asarray(state.kondo_state.policy_revision)) == state.event_index
            and int(np.asarray(state.overflow_state.policy_revision)) == state.event_index
        )

    def advance(
        self,
        state: KondoSparseActorDevelopmentRunState,
    ) -> KondoSparseActorDevelopmentRunState:
        if not self._valid_state(state):
            raise ValueError("Kondo sparse actor development run state is invalid")
        if state.event_index >= self.config.num_batches:
            raise ValueError("Kondo sparse actor development run is already complete")
        event_index = state.event_index
        source = _source_batch(self.config, event_index)
        source_payload = source.payload(event_index=event_index, role="training")
        source_sha256 = cast(str, source_payload["source_batch_sha256"])
        full_mask = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)

        ordinary_behavior = _selected_log_probability(
            state.ordinary_parameters,
            source.actor_features,
            source.actions,
        )
        ordinary_batch = KondoActorBackwardBatch(
            actor_features=source.actor_features,
            actions=source.actions,
            advantage=source.advantage,
            sample_mask=full_mask,
        )
        ordinary_backward = kondo_actor_backward_kernel(
            state.ordinary_parameters,
            ordinary_batch,
        )
        ordinary_after = _apply_gradient(
            state.ordinary_parameters,
            ordinary_backward.gradient,
            self.config.learning_rate,
        )
        if not bool(np.asarray(ordinary_backward.gradient_finite)) or not math.isfinite(
            _tree_l2(ordinary_after)
        ):
            raise ValueError("ordinary full backward/update produced nonfinite values")
        ordinary_record = _backward_record(
            arm="ordinary_full",
            event_index=event_index,
            source_sha256=source_sha256,
            revision_before=event_index,
            behavior_log_probability=ordinary_behavior,
            parameters_before=state.ordinary_parameters,
            parameters_after=ordinary_after,
            backward=ordinary_backward,
            selected_indices=tuple(range(self.config.batch_size)),
            selected_count=self.config.batch_size,
            backward_slots=self.config.batch_size,
            sparse=False,
            full_fallback=False,
            screen_gather_order=("full-forward", "compiled-backward"),
        )

        uniform_behavior = _selected_log_probability(
            state.uniform_parameters,
            source.actor_features,
            source.actions,
        )
        uniform_indices = source.uniform_indices
        uniform_batch = KondoActorBackwardBatch(
            actor_features=source.actor_features[uniform_indices],
            actions=source.actions[uniform_indices],
            advantage=source.advantage[uniform_indices],
            sample_mask=jnp.ones((self.config.sparse_capacity,), dtype=jnp.bool_),
        )
        uniform_backward = kondo_actor_backward_kernel(
            state.uniform_parameters,
            uniform_batch,
        )
        uniform_after = _apply_gradient(
            state.uniform_parameters,
            uniform_backward.gradient,
            self.config.learning_rate,
        )
        if not bool(np.asarray(uniform_backward.gradient_finite)) or not math.isfinite(
            _tree_l2(uniform_after)
        ):
            raise ValueError("uniform sparse backward/update produced nonfinite values")
        uniform_record = _backward_record(
            arm="uniform_sparse",
            event_index=event_index,
            source_sha256=source_sha256,
            revision_before=event_index,
            behavior_log_probability=uniform_behavior,
            parameters_before=state.uniform_parameters,
            parameters_after=uniform_after,
            backward=uniform_backward,
            selected_indices=[int(item) for item in np.asarray(uniform_indices)],
            selected_count=self.config.sparse_capacity,
            backward_slots=self.config.sparse_capacity,
            sparse=True,
            full_fallback=False,
            screen_gather_order=("full-forward", "uniform-gather", "compiled-backward"),
        )

        protected = KondoActorProtectedInputs(
            critic_features=source.critic_features,
            baseline_predictions=source.baseline_predictions,
            return_targets=source.return_targets,
            safety_features=source.safety_features,
        )
        valid = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
        no_force = jnp.zeros((self.config.batch_size,), dtype=jnp.bool_)
        kondo_behavior = self.actor.behavior_log_probability(
            state.kondo_state,
            source.actor_features,
            source.actions,
        )
        kondo_input = KondoSparseActorBatch(
            actor_features=source.actor_features,
            actions=source.actions,
            action_identity=source.actions,
            policy_revision=jnp.full(
                (self.config.batch_size,),
                state.kondo_state.policy_revision,
                dtype=jnp.int32,
            ),
            behavior_log_probability=kondo_behavior,
            valid_mask=valid,
            force_keep_mask=no_force,
            protected=protected,
        )
        kondo_result = self.actor.step(state.kondo_state, kondo_input)
        if not bool(np.asarray(kondo_result.transaction_applied)) or not bool(
            np.asarray(kondo_result.sparse_backward_used)
        ):
            raise ValueError("Kondo development sparse transaction failed")
        kondo_selected = np.flatnonzero(np.asarray(kondo_result.sparks_joy)).tolist()
        kondo_backward = KondoActorBackwardSummary(
            loss=kondo_result.actor_loss,
            gradient=kondo_result.gradient,
            selected_count=kondo_result.screen.selected_count,
            gradient_finite=kondo_result.gradient_finite,
        )
        kondo_record = _backward_record(
            arm="kondo_top_k",
            event_index=event_index,
            source_sha256=source_sha256,
            revision_before=event_index,
            behavior_log_probability=kondo_result.current_action_log_probability,
            parameters_before=state.kondo_state.parameters,
            parameters_after=kondo_result.state.parameters,
            backward=kondo_backward,
            selected_indices=kondo_selected,
            selected_count=int(np.asarray(kondo_result.screen.selected_count)),
            backward_slots=int(np.asarray(kondo_result.backward_batch_size)),
            sparse=True,
            full_fallback=False,
            screen_gather_order=(
                "full-forward",
                "detached-screen",
                "audited-gather",
                "compiled-backward",
            ),
        )
        kondo_record["delight"] = _array_payload(kondo_result.screen.delight)
        kondo_record["selected_action_surprisal"] = _array_payload(
            kondo_result.screen.action_surprisal
        )

        overflow_behavior = self.actor.behavior_log_probability(
            state.overflow_state,
            source.actor_features,
            source.actions,
        )
        force_count = self.config.sparse_capacity + 1
        force_mask = jnp.arange(self.config.batch_size, dtype=jnp.int32) < force_count
        overflow_input = dataclasses.replace(
            kondo_input,
            policy_revision=jnp.full(
                (self.config.batch_size,),
                state.overflow_state.policy_revision,
                dtype=jnp.int32,
            ),
            behavior_log_probability=overflow_behavior,
            force_keep_mask=force_mask,
        )
        overflow_result = self.actor.step(state.overflow_state, overflow_input)
        if not bool(np.asarray(overflow_result.transaction_applied)) or not bool(
            np.asarray(overflow_result.full_shape_masked_backward_used)
        ):
            raise ValueError("Kondo development overflow fallback failed")
        overflow_selected = np.flatnonzero(np.asarray(overflow_result.sparks_joy)).tolist()
        overflow_backward = KondoActorBackwardSummary(
            loss=overflow_result.actor_loss,
            gradient=overflow_result.gradient,
            selected_count=overflow_result.screen.selected_count,
            gradient_finite=overflow_result.gradient_finite,
        )
        overflow_record = _backward_record(
            arm="kondo_overflow_diagnostic",
            event_index=event_index,
            source_sha256=source_sha256,
            revision_before=event_index,
            behavior_log_probability=overflow_result.current_action_log_probability,
            parameters_before=state.overflow_state.parameters,
            parameters_after=overflow_result.state.parameters,
            backward=overflow_backward,
            selected_indices=overflow_selected,
            selected_count=int(np.asarray(overflow_result.screen.selected_count)),
            backward_slots=int(np.asarray(overflow_result.backward_batch_size)),
            sparse=False,
            full_fallback=True,
            screen_gather_order=(
                "full-forward",
                "detached-screen",
                "capacity-overflow",
                "full-shape-masked-backward",
            ),
        )
        overflow_record["forced_indices"] = list(range(force_count))
        overflow_record["diagnostic_only"] = True

        action_payload = _array_payload(source.actions)
        feature_payload = _array_payload(source.actor_features)
        protected_binding = {
            "critic_features": _array_payload(source.critic_features),
            "baseline_predictions": _array_payload(source.baseline_predictions),
            "return_targets": _array_payload(source.return_targets),
            "safety_features": _array_payload(source.safety_features),
        }
        common_source_binding = {
            "source_actions_sha256": action_payload["sha256"],
            "source_actor_features_sha256": feature_payload["sha256"],
            "source_protected_inputs_sha256": _canonical_sha256(protected_binding),
            "source_experience_replays_in_arm": 1,
        }
        for record in (
            ordinary_record,
            uniform_record,
            kondo_record,
            overflow_record,
        ):
            record.update(common_source_binding)

        encoded_records = tuple(
            _canonical_json_bytes(record).decode("utf-8")
            for record in (
                ordinary_record,
                uniform_record,
                kondo_record,
                overflow_record,
            )
        )
        next_state = KondoSparseActorDevelopmentRunState(
            event_index=event_index + 1,
            ordinary_parameters=ordinary_after,
            uniform_parameters=uniform_after,
            kondo_state=kondo_result.state,
            overflow_state=overflow_result.state,
            records_json=state.records_json + encoded_records,
        )
        if not self._valid_state(next_state):
            raise ValueError("Kondo sparse actor development next state is invalid")
        return next_state

    def run_to_end(
        self,
        state: KondoSparseActorDevelopmentRunState | None = None,
    ) -> KondoSparseActorDevelopmentRunState:
        current = self.init() if state is None else state
        while current.event_index < self.config.num_batches:
            current = self.advance(current)
        return current

    def checkpoint_payload(
        self,
        state: KondoSparseActorDevelopmentRunState,
    ) -> dict[str, object]:
        if not self._valid_state(state):
            raise ValueError("cannot checkpoint invalid Kondo development state")
        source = kondo_sparse_actor_development_source_manifest()
        runtime = kondo_sparse_actor_development_runtime_identity()
        protocol = kondo_sparse_actor_development_protocol(self.config)
        body: dict[str, object] = {
            "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_CHECKPOINT_SCHEMA,
            "type": "KondoSparseActorDevelopmentCheckpoint",
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(protocol),
            "source_manifest_sha256": _canonical_sha256(source),
            "runtime_sha256": _canonical_sha256(runtime),
            "event_index": state.event_index,
            "ordinary_parameters": _parameter_payload(state.ordinary_parameters),
            "uniform_parameters": _parameter_payload(state.uniform_parameters),
            "kondo_state": self.actor.checkpoint_payload(state.kondo_state),
            "overflow_state": self.actor.checkpoint_payload(state.overflow_state),
            "records": [json.loads(record) for record in state.records_json],
        }
        payload = {**body, "checkpoint_sha256": _canonical_sha256(body)}
        if len(_canonical_json_bytes(payload)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Kondo development checkpoint exceeds byte cap")
        return payload

    def restore_checkpoint(
        self,
        payload: object,
    ) -> KondoSparseActorDevelopmentRunState:
        raw = _mapping(payload, name="checkpoint")
        if len(_canonical_json_bytes(raw)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Kondo development checkpoint exceeds byte cap")
        expected_fields = {
            "schema",
            "type",
            "config_sha256",
            "protocol_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
            "event_index",
            "ordinary_parameters",
            "uniform_parameters",
            "kondo_state",
            "overflow_state",
            "records",
            "checkpoint_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError("Kondo development checkpoint fields differ")
        if raw.get("schema") != KONDO_SPARSE_ACTOR_DEVELOPMENT_CHECKPOINT_SCHEMA or (
            raw.get("type") != "KondoSparseActorDevelopmentCheckpoint"
        ):
            raise ValueError("Kondo development checkpoint schema/type is invalid")
        body = {name: raw[name] for name in raw if name != "checkpoint_sha256"}
        if raw.get("checkpoint_sha256") != _canonical_sha256(body):
            raise ValueError("Kondo development checkpoint digest integrity check failed")
        bindings = {
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(
                kondo_sparse_actor_development_protocol(self.config)
            ),
            "source_manifest_sha256": _canonical_sha256(
                kondo_sparse_actor_development_source_manifest()
            ),
            "runtime_sha256": _canonical_sha256(
                kondo_sparse_actor_development_runtime_identity()
            ),
        }
        if any(raw.get(name) != value for name, value in bindings.items()):
            raise ValueError("Kondo development checkpoint binding differs")
        event_index = _exact_int(
            raw.get("event_index"),
            name="event_index",
            maximum=self.config.num_batches,
        )
        expected_state = self.init()
        for _ in range(event_index):
            expected_state = self.advance(expected_state)
        expected_payload = self.checkpoint_payload(expected_state)
        if len(_canonical_json_bytes(expected_payload)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Kondo development checkpoint exceeds byte cap")
        if not _strict_json_equal(dict(raw), expected_payload):
            raise ValueError("Kondo development checkpoint differs from exact prefix replay")
        return expected_state


def _heldout_diagnostics(
    config: KondoSparseActorDevelopmentConfig,
    initial: KondoActorParameters,
    final_parameters: Mapping[ArmName, KondoActorParameters],
) -> tuple[dict[str, object], dict[str, object]]:
    heldout = _source_batch(config, config.num_batches + 1)
    heldout_payload = heldout.payload(
        event_index=config.num_batches + 1,
        role="heldout-development",
    )
    full_batch = KondoActorBackwardBatch(
        actor_features=heldout.actor_features,
        actions=heldout.actions,
        advantage=heldout.advantage,
        sample_mask=jnp.ones((config.batch_size,), dtype=jnp.bool_),
    )
    diagnostics: dict[str, object] = {}
    for arm in ARM_ORDER:
        parameters = final_parameters[arm]
        backward = kondo_actor_backward_kernel(parameters, full_batch)
        if not bool(np.asarray(backward.gradient_finite)):
            raise ValueError(f"{arm} heldout backward produced nonfinite values")
        diagnostics[arm] = {
            "heldout_actor_loss": float(np.asarray(backward.loss)),
            "heldout_gradient_l2": _tree_l2(backward.gradient),
            "parameter_change_l2": _parameter_delta_l2(parameters, initial),
            "final_parameter_sha256": _parameter_sha256(parameters),
            "heldout_backward_invocations": 1,
            "heldout_parameter_updates": 0,
            "assessment_status": ASSESSMENT_STATUS,
        }
    return heldout_payload, diagnostics


def build_kondo_sparse_actor_deterministic_payload(
    config: KondoSparseActorDevelopmentConfig | None = None,
) -> dict[str, object]:
    """Run and serialize every deterministic field, excluding wall-clock bytes."""
    cfg = config or KondoSparseActorDevelopmentConfig()
    evaluator = KondoSparseActorDevelopmentEvaluator(cfg)
    final = evaluator.run_to_end()
    records = [json.loads(record) for record in final.records_json]
    source_trace = [
        _source_batch(cfg, event_index).payload(event_index=event_index, role="training")
        for event_index in range(cfg.num_batches)
    ]
    final_parameters: dict[ArmName, KondoActorParameters] = {
        "ordinary_full": final.ordinary_parameters,
        "uniform_sparse": final.uniform_parameters,
        "kondo_top_k": final.kondo_state.parameters,
        "kondo_overflow_diagnostic": final.overflow_state.parameters,
    }
    heldout_payload, heldout_diagnostics = _heldout_diagnostics(
        cfg,
        evaluator.initial_parameters,
        final_parameters,
    )
    per_arm: dict[str, dict[str, int]] = {}
    for arm in ARM_ORDER:
        arm_records = [record for record in records if record["arm"] == arm]
        per_arm[arm] = {
            "source_batches_consumed": len(arm_records),
            "training_updates": sum(cast(int, item["training_updates"]) for item in arm_records),
            "compiled_training_backward_invocations": sum(
                cast(int, item["compiled_backward_invocations"]) for item in arm_records
            ),
            "selected_samples": sum(cast(int, item["selected_count"]) for item in arm_records),
            "backward_row_slots": sum(
                cast(int, item["backward_leading_shape"]) for item in arm_records
            ),
            "heldout_backward_invocations": 1,
            "heldout_updates": 0,
        }
    multiplication_terms_per_row = (
        cfg.feature_dim * cfg.hidden_dim + cfg.hidden_dim * cfg.action_count
    )
    for arm in ARM_ORDER:
        per_arm[arm]["forward_multiplication_term_shape_proxy"] = (
            per_arm[arm]["backward_row_slots"] * multiplication_terms_per_row
        )
    source = kondo_sparse_actor_development_source_manifest()
    runtime = kondo_sparse_actor_development_runtime_identity()
    protocol = kondo_sparse_actor_development_protocol(cfg)
    body: dict[str, object] = {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_DETERMINISTIC_SCHEMA,
        "type": "KondoSparseActorDevelopmentDeterministicPayload",
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "compute_saving_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "config": cfg.to_config(),
        "config_sha256": _canonical_sha256(cfg.to_config()),
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "source_manifest": source,
        "source_manifest_sha256": _canonical_sha256(source),
        "runtime": runtime,
        "runtime_sha256": _canonical_sha256(runtime),
        "initial_parameter_snapshot": _parameter_payload(evaluator.initial_parameters),
        "initial_parameter_sha256": _parameter_sha256(evaluator.initial_parameters),
        "source_trace": source_trace,
        "source_trace_sha256": _canonical_sha256(source_trace),
        "arm_records": records,
        "arm_records_sha256": _canonical_sha256(records),
        "heldout_development_batch": heldout_payload,
        "heldout_diagnostics": heldout_diagnostics,
        "logical_resource_accounting": {
            "unique_source_batches": cfg.num_batches,
            "deterministic_training_trace_executions": 1,
            "source_trace_replays_per_arm": 1,
            "updates_per_source_batch_per_arm": 1,
            "experience_double_counted_within_arm": False,
            "timing_invocations_apply_parameter_updates": False,
            "heldout_invocations_apply_parameter_updates": False,
            "per_arm": per_arm,
            "multiplication_term_proxy_semantics": (
                "backward-leading-row-slots times dense actor forward multiplication "
                "terms; excludes backward derivatives, fusion, host work, and hardware"
            ),
            "measured_flops": False,
        },
        "limitations": list(_LIMITATIONS),
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
    }
    payload = {**body, "deterministic_sha256": _canonical_sha256(body)}
    if len(_canonical_json_bytes(payload)) > _MAX_DETERMINISTIC_BYTES:
        raise ValueError("Kondo development deterministic payload exceeds byte cap")
    return payload


def _timing_trial_order(config: KondoSparseActorDevelopmentConfig) -> list[list[str]]:
    orders: list[list[str]] = []
    root = jr.key(config.timing_order_seed, impl="threefry2x32")
    for trial in range(config.timing_trials):
        permutation = np.asarray(jr.permutation(jr.fold_in(root, trial), len(ARM_ORDER)))
        orders.append([ARM_ORDER[int(index)] for index in permutation])
    return orders


def _block_backward(result: KondoActorBackwardResult) -> None:
    for leaf in jax.tree_util.tree_leaves(result):
        cast(Any, leaf).block_until_ready()


def _max_abs_gradient_delta(
    left: KondoActorBackwardResult,
    right: KondoActorBackwardResult,
) -> float:
    maximum = abs(float(np.asarray(left.loss)) - float(np.asarray(right.loss)))
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(left.gradient),
        jax.tree_util.tree_leaves(right.gradient),
        strict=True,
    ):
        difference = np.abs(
            np.asarray(jax.device_get(lhs), dtype=np.float64)
            - np.asarray(jax.device_get(rhs), dtype=np.float64)
        )
        maximum = max(maximum, float(np.max(difference, initial=0.0)))
    return maximum


def _nearest_rank(samples: Sequence[int], percentile: float) -> int:
    ordered = sorted(samples)
    rank = max(1, int(math.ceil(percentile * len(ordered))))
    return ordered[rank - 1]


def _prepare_timing_batches(
    config: KondoSparseActorDevelopmentConfig,
) -> tuple[KondoActorParameters, dict[ArmName, KondoActorBackwardBatch], dict[str, object]]:
    parameters = _initial_parameters(config)
    source = _source_batch(config, 0)
    advantage = source.advantage
    full_mask = jnp.ones((config.batch_size,), dtype=jnp.bool_)
    ordinary = KondoActorBackwardBatch(
        actor_features=source.actor_features,
        actions=source.actions,
        advantage=advantage,
        sample_mask=full_mask,
    )
    uniform = KondoActorBackwardBatch(
        actor_features=source.actor_features[source.uniform_indices],
        actions=source.actions[source.uniform_indices],
        advantage=advantage[source.uniform_indices],
        sample_mask=jnp.ones((config.sparse_capacity,), dtype=jnp.bool_),
    )
    gate = KondoGate(config.actor_config().gate)
    gate_state = gate.init(jr.fold_in(jr.key(config.seed, impl="threefry2x32"), 20_001))
    behavior = _selected_log_probability(parameters, source.actor_features, source.actions)
    valid = jnp.ones((config.batch_size,), dtype=jnp.bool_)
    no_force = jnp.zeros((config.batch_size,), dtype=jnp.bool_)
    kondo_screen = gate.screen(gate_state, advantage, behavior, valid, no_force)
    sparse = gate.gather_sparse(
        {
            "actor_features": source.actor_features,
            "actions": source.actions,
            "advantage": advantage,
        },
        kondo_screen,
    )
    kondo = KondoActorBackwardBatch(
        actor_features=sparse.data["actor_features"],
        actions=sparse.data["actions"],
        advantage=sparse.data["advantage"],
        sample_mask=sparse.sample_mask,
    )
    overflow_force = (
        jnp.arange(config.batch_size, dtype=jnp.int32) < config.sparse_capacity + 1
    )
    overflow_screen = gate.screen(
        gate_state,
        advantage,
        behavior,
        valid,
        overflow_force,
    )
    if not bool(np.asarray(overflow_screen.full_shape_masked_backward_required)):
        raise ValueError("timing overflow screen did not require full shape")
    overflow = KondoActorBackwardBatch(
        actor_features=source.actor_features,
        actions=source.actions,
        advantage=advantage,
        sample_mask=overflow_screen.selected_mask,
    )
    batches: dict[ArmName, KondoActorBackwardBatch] = {
        "ordinary_full": ordinary,
        "uniform_sparse": uniform,
        "kondo_top_k": kondo,
        "kondo_overflow_diagnostic": overflow,
    }
    contracts = {
        "parameter_snapshot_sha256": _parameter_sha256(parameters),
        "source_batch_sha256": source.payload(event_index=0, role="timing")[
            "source_batch_sha256"
        ],
        "ordinary_full_leading_shape": config.batch_size,
        "uniform_sparse_leading_shape": config.sparse_capacity,
        "kondo_top_k_leading_shape": config.sparse_capacity,
        "kondo_overflow_diagnostic_leading_shape": config.batch_size,
        "uniform_and_kondo_capacity_equal": True,
        "kondo_screen_detached_before_gather": True,
        "kondo_gather_before_compiled_backward": True,
        "overflow_preserves_every_forced_sample": bool(
            np.all(np.asarray(overflow_screen.selected_mask)[0 : config.sparse_capacity + 1])
        ),
        "host_screen_gather_timed": False,
    }
    return parameters, batches, contracts


def measure_kondo_sparse_actor_backward_timing(
    config: KondoSparseActorDevelopmentConfig,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    clock_name: str = "time.perf_counter_ns",
    deterministic_sha256: str | None = None,
) -> dict[str, object]:
    """Measure interleaved compiled backwards after compilation and warmup."""
    if not callable(clock_ns):
        raise TypeError("clock_ns must be callable")
    if type(clock_name) is not str or not clock_name or len(clock_name) > 128:
        raise ValueError("clock_name must be a nonempty string of at most 128 characters")
    parameters, batches, contracts = _prepare_timing_batches(config)
    compiled_full = kondo_actor_backward_kernel.lower(
        parameters,
        batches["ordinary_full"],
    ).compile()
    compiled_sparse = kondo_actor_backward_kernel.lower(
        parameters,
        batches["uniform_sparse"],
    ).compile()
    compiled_by_arm: dict[ArmName, Any] = {
        "ordinary_full": compiled_full,
        "uniform_sparse": compiled_sparse,
        "kondo_top_k": compiled_sparse,
        "kondo_overflow_diagnostic": compiled_full,
    }
    parity: dict[str, object] = {}
    for arm in ARM_ORDER:
        with jax.disable_jit():
            eager = kondo_actor_backward_kernel(parameters, batches[arm])
        compiled = compiled_by_arm[arm](parameters, batches[arm])
        _block_backward(compiled)
        maximum_delta = _max_abs_gradient_delta(eager, compiled)
        parity[arm] = {
            "eager_compiled_max_abs_delta": maximum_delta,
            "eager_compiled_numerical_parity": maximum_delta <= 1.0e-6,
            "compiled_warmup_invocations": 1,
        }
    trial_order = _timing_trial_order(config)
    raw_by_arm: dict[str, list[int]] = {arm: [] for arm in ARM_ORDER}
    events: list[dict[str, object]] = []
    previous_end: int | None = None
    for trial, order in enumerate(trial_order):
        for position, arm_string in enumerate(order):
            arm = cast(ArmName, arm_string)
            start = clock_ns()
            if (
                type(start) is not int
                or start < 0
                or (previous_end is not None and start < previous_end)
            ):
                raise ValueError(
                    "perf counter must return globally monotonic nonnegative "
                    "integer nanoseconds"
                )
            result = compiled_by_arm[arm](parameters, batches[arm])
            _block_backward(result)
            end = clock_ns()
            if type(end) is not int or end < start:
                raise ValueError(
                    "perf counter must return globally monotonic nonnegative "
                    "integer nanoseconds"
                )
            duration = end - start
            previous_end = end
            raw_by_arm[arm].append(duration)
            events.append(
                {
                    "trial": trial,
                    "position": position,
                    "arm": arm,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": duration,
                }
            )
    summaries: dict[str, object] = {}
    for arm in ARM_ORDER:
        samples = raw_by_arm[arm]
        summaries[arm] = {
            "raw_duration_ns": samples,
            "sample_count": len(samples),
            "minimum_ns": min(samples),
            "maximum_ns": max(samples),
            "p50_ns": _nearest_rank(samples, 0.50),
            "p95_ns": _nearest_rank(samples, 0.95),
            "summary_method": "nearest-rank",
            "compiled_warmup_invocations": 1,
            "compiled_timed_invocations": len(samples),
            "compiled_backward_invocations_total": len(samples) + 1,
            "assessment_status": ASSESSMENT_STATUS,
        }
    if deterministic_sha256 is None:
        deterministic_binding = cast(
            str,
            build_kondo_sparse_actor_deterministic_payload(config)[
                "deterministic_sha256"
            ],
        )
    else:
        if (
            type(deterministic_sha256) is not str
            or len(deterministic_sha256) != 64
            or any(character not in "0123456789abcdef" for character in deterministic_sha256)
        ):
            raise ValueError("deterministic_sha256 must be a lowercase SHA-256")
        deterministic_binding = deterministic_sha256
    source = kondo_sparse_actor_development_source_manifest()
    runtime = kondo_sparse_actor_development_runtime_identity()
    real_perf_counter = clock_ns is time.perf_counter_ns
    body: dict[str, object] = {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_TIMING_SCHEMA,
        "type": "KondoSparseActorDevelopmentTiming",
        "assessment_status": ASSESSMENT_STATUS,
        "timing_is_descriptive_and_noisy": True,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
        "clock_name": clock_name,
        "real_perf_counter_ns": real_perf_counter,
        "clock_resolution_ns": (
            int(math.ceil(time.get_clock_info("perf_counter").resolution * 1.0e9))
            if real_perf_counter
            else None
        ),
        "compiled_before_warmup": True,
        "warmup_before_timing": True,
        "block_until_ready": True,
        "trial_order": trial_order,
        "events": events,
        "summaries": summaries,
        "eager_compiled_parity": parity,
        "shape_and_order_contracts": contracts,
        "measurement_scope": "compiled-actor-backward-only",
        "host_screen_gather_timed": False,
        "accelerator_memory_measured": False,
        "energy_measured": False,
        "end_to_end_latency_measured": False,
        "parameter_updates_applied": False,
        "deterministic_sha256": deterministic_binding,
        "source_manifest_sha256": _canonical_sha256(source),
        "runtime_sha256": _canonical_sha256(runtime),
        "config_sha256": _canonical_sha256(config.to_config()),
        "wall_clock_bytes_in_deterministic_replay": False,
    }
    return {**body, "timing_sha256": _canonical_sha256(body)}


def build_kondo_sparse_actor_development_report(
    config: KondoSparseActorDevelopmentConfig | None = None,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    clock_name: str = "time.perf_counter_ns",
) -> dict[str, object]:
    """Build an in-memory strict report; never write or promote it."""
    cfg = config or KondoSparseActorDevelopmentConfig()
    deterministic = build_kondo_sparse_actor_deterministic_payload(cfg)
    timing = measure_kondo_sparse_actor_backward_timing(
        cfg,
        clock_ns=clock_ns,
        clock_name=clock_name,
        deterministic_sha256=cast(str, deterministic["deterministic_sha256"]),
    )
    body: dict[str, object] = {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_REPORT_SCHEMA,
        "type": "KondoSparseActorDevelopmentReport",
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "compute_saving_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "deterministic": deterministic,
        "timing": timing,
        "deterministic_replay_includes_wall_clock": False,
        "timing_separately_provenance_bound": True,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
    }
    payload = {**body, "report_sha256": _canonical_sha256(body)}
    if len(_canonical_json_bytes(payload)) > _MAX_REPORT_BYTES:
        raise ValueError("Kondo sparse actor development report exceeds byte cap")
    return payload


@dataclasses.dataclass(frozen=True)
class KondoSparseActorDevelopmentValidationReceipt:
    """Strict structural and deterministic-replay validation receipt."""

    valid: bool
    assessment_status: str
    deterministic_replay_checked: bool
    deterministic_replay_exact: bool
    timing_structure_checked: bool
    timing_provenance_bound: bool
    wall_clock_replayed: bool


def _validate_timing(
    timing: Mapping[str, object],
    config: KondoSparseActorDevelopmentConfig,
    deterministic: Mapping[str, object],
) -> None:
    expected_fields = {
        "schema",
        "type",
        "assessment_status",
        "timing_is_descriptive_and_noisy",
        "thresholds",
        "verdict",
        "clock_name",
        "real_perf_counter_ns",
        "clock_resolution_ns",
        "compiled_before_warmup",
        "warmup_before_timing",
        "block_until_ready",
        "trial_order",
        "events",
        "summaries",
        "eager_compiled_parity",
        "shape_and_order_contracts",
        "measurement_scope",
        "host_screen_gather_timed",
        "accelerator_memory_measured",
        "energy_measured",
        "end_to_end_latency_measured",
        "parameter_updates_applied",
        "deterministic_sha256",
        "source_manifest_sha256",
        "runtime_sha256",
        "config_sha256",
        "wall_clock_bytes_in_deterministic_replay",
        "timing_sha256",
    }
    if set(timing) != expected_fields:
        raise ValueError("Kondo development timing fields differ")
    body = {name: timing[name] for name in timing if name != "timing_sha256"}
    if timing.get("timing_sha256") != _canonical_sha256(body):
        raise ValueError("Kondo development timing digest is invalid")
    fixed = {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_TIMING_SCHEMA,
        "type": "KondoSparseActorDevelopmentTiming",
        "assessment_status": ASSESSMENT_STATUS,
        "timing_is_descriptive_and_noisy": True,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
        "compiled_before_warmup": True,
        "warmup_before_timing": True,
        "block_until_ready": True,
        "measurement_scope": "compiled-actor-backward-only",
        "host_screen_gather_timed": False,
        "accelerator_memory_measured": False,
        "energy_measured": False,
        "end_to_end_latency_measured": False,
        "parameter_updates_applied": False,
        "wall_clock_bytes_in_deterministic_replay": False,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(timing.get(name), expected):
            raise ValueError(f"Kondo development timing {name} is invalid")
    bindings = {
        "deterministic_sha256": deterministic.get("deterministic_sha256"),
        "source_manifest_sha256": _canonical_sha256(
            kondo_sparse_actor_development_source_manifest()
        ),
        "runtime_sha256": _canonical_sha256(
            kondo_sparse_actor_development_runtime_identity()
        ),
        "config_sha256": _canonical_sha256(config.to_config()),
    }
    for name, expected in bindings.items():
        if timing.get(name) != expected:
            raise ValueError(f"Kondo development timing {name} binding differs")
    clock_name = timing.get("clock_name")
    real_perf_counter = timing.get("real_perf_counter_ns")
    clock_resolution = timing.get("clock_resolution_ns")
    if type(clock_name) is not str or not clock_name or len(clock_name) > 128:
        raise ValueError("Kondo development timing clock_name is invalid")
    if type(real_perf_counter) is not bool:
        raise ValueError("Kondo development timing real_perf_counter_ns is invalid")
    if real_perf_counter:
        if clock_name != "time.perf_counter_ns":
            raise ValueError("real perf_counter timing must use its canonical clock name")
        _exact_int(clock_resolution, name="clock_resolution_ns", minimum=1)
    elif clock_resolution is not None:
        raise ValueError("injected timing clock resolution must be null")
    if timing.get("trial_order") != _timing_trial_order(config):
        raise ValueError("Kondo development timing trial order differs")
    events_raw = timing.get("events")
    summaries_raw = timing.get("summaries")
    if type(events_raw) is not list or not isinstance(summaries_raw, Mapping):
        raise ValueError("Kondo development timing events/summaries are invalid")
    events = cast(list[object], events_raw)
    if len(events) != config.timing_trials * len(ARM_ORDER):
        raise ValueError("Kondo development timing event count differs")
    reconstructed: dict[str, list[int]] = {arm: [] for arm in ARM_ORDER}
    previous_end: int | None = None
    for expected_index, event_raw in enumerate(events):
        event = _mapping(event_raw, name="timing event")
        if set(event) != {
            "trial",
            "position",
            "arm",
            "start_ns",
            "end_ns",
            "duration_ns",
        }:
            raise ValueError("Kondo development timing event fields differ")
        trial = expected_index // len(ARM_ORDER)
        position = expected_index % len(ARM_ORDER)
        arm = _timing_trial_order(config)[trial][position]
        if event.get("trial") != trial or event.get("position") != position or (
            event.get("arm") != arm
        ):
            raise ValueError("Kondo development timing event order differs")
        start = _exact_int(event.get("start_ns"), name="start_ns")
        end = _exact_int(event.get("end_ns"), name="end_ns")
        duration = _exact_int(event.get("duration_ns"), name="duration_ns")
        if end - start != duration:
            raise ValueError("Kondo development timing duration is inconsistent")
        if previous_end is not None and start < previous_end:
            raise ValueError("Kondo development timing clock is not globally monotonic")
        previous_end = end
        reconstructed[arm].append(duration)
    if set(summaries_raw) != set(ARM_ORDER):
        raise ValueError("Kondo development timing summary arms differ")
    for arm in ARM_ORDER:
        summary = _mapping(summaries_raw[arm], name=f"timing summary {arm}")
        samples = reconstructed[arm]
        expected_summary = {
            "raw_duration_ns": samples,
            "sample_count": config.timing_trials,
            "minimum_ns": min(samples),
            "maximum_ns": max(samples),
            "p50_ns": _nearest_rank(samples, 0.50),
            "p95_ns": _nearest_rank(samples, 0.95),
            "summary_method": "nearest-rank",
            "compiled_warmup_invocations": 1,
            "compiled_timed_invocations": config.timing_trials,
            "compiled_backward_invocations_total": config.timing_trials + 1,
            "assessment_status": ASSESSMENT_STATUS,
        }
        if not _strict_json_equal(dict(summary), expected_summary):
            raise ValueError(f"Kondo development timing summary {arm} differs")
    parity = _mapping(timing.get("eager_compiled_parity"), name="parity")
    contracts = _mapping(timing.get("shape_and_order_contracts"), name="contracts")
    if set(parity) != set(ARM_ORDER):
        raise ValueError("Kondo development eager/compiled parity arms differ")
    for arm in ARM_ORDER:
        arm_parity = _mapping(parity[arm], name=f"parity {arm}")
        if set(arm_parity) != {
            "eager_compiled_max_abs_delta",
            "eager_compiled_numerical_parity",
            "compiled_warmup_invocations",
        }:
            raise ValueError("Kondo development eager/compiled parity fields differ")
        delta = arm_parity.get("eager_compiled_max_abs_delta")
        if type(delta) is not float or not math.isfinite(delta) or not 0.0 <= delta <= 1.0e-6:
            raise ValueError("Kondo development eager/compiled delta is invalid")
        if arm_parity.get("eager_compiled_numerical_parity") is not True or (
            arm_parity.get("compiled_warmup_invocations") != 1
        ):
            raise ValueError("Kondo development eager/compiled parity is invalid")
    _, _, expected_contracts = _prepare_timing_batches(config)
    if not _strict_json_equal(dict(contracts), expected_contracts):
        raise ValueError("Kondo development timing shape/order contracts differ")


def validate_kondo_sparse_actor_development_report(
    report: object,
) -> KondoSparseActorDevelopmentValidationReceipt:
    """Fail closed on structure, source/runtime binding, replay, and timing."""
    raw = _mapping(report, name="report")
    if len(_canonical_json_bytes(raw)) > _MAX_REPORT_BYTES:
        raise ValueError("Kondo sparse actor development report exceeds byte cap")
    expected_fields = {
        "schema",
        "type",
        "development_status",
        "assessment_status",
        "performance_claimed",
        "compute_saving_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "output_writes",
        "promotion_authority",
        "scientific_promotion_allowed",
        "deterministic",
        "timing",
        "deterministic_replay_includes_wall_clock",
        "timing_separately_provenance_bound",
        "thresholds",
        "verdict",
        "report_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("Kondo sparse actor development report fields differ")
    fixed: dict[str, object] = {
        "schema": KONDO_SPARSE_ACTOR_DEVELOPMENT_REPORT_SCHEMA,
        "type": "KondoSparseActorDevelopmentReport",
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "compute_saving_claimed": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "policy_authority": False,
        "output_writes": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "deterministic_replay_includes_wall_clock": False,
        "timing_separately_provenance_bound": True,
        "thresholds": [],
        "verdict": ASSESSMENT_STATUS,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(raw.get(name), expected):
            raise ValueError(f"Kondo sparse actor development report {name} is invalid")
    body = {name: raw[name] for name in raw if name != "report_sha256"}
    if raw.get("report_sha256") != _canonical_sha256(body):
        raise ValueError("Kondo sparse actor development report digest is invalid")
    deterministic = _mapping(raw.get("deterministic"), name="deterministic")
    config_payload = _mapping(deterministic.get("config"), name="config")
    config = KondoSparseActorDevelopmentConfig.from_config(config_payload)
    deterministic_body = {
        name: deterministic[name]
        for name in deterministic
        if name != "deterministic_sha256"
    }
    if deterministic.get("deterministic_sha256") != _canonical_sha256(
        deterministic_body
    ):
        raise ValueError("Kondo development deterministic digest is invalid")
    replay = build_kondo_sparse_actor_deterministic_payload(config)
    if not _strict_json_equal(dict(deterministic), replay):
        raise ValueError("Kondo development deterministic replay differs")
    timing = _mapping(raw.get("timing"), name="timing")
    _validate_timing(timing, config, deterministic)
    return KondoSparseActorDevelopmentValidationReceipt(
        valid=True,
        assessment_status=ASSESSMENT_STATUS,
        deterministic_replay_checked=True,
        deterministic_replay_exact=True,
        timing_structure_checked=True,
        timing_provenance_bound=True,
        wall_clock_replayed=False,
    )


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_STATUS",
    "DIAGNOSTIC_ARMS",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_CHECKPOINT_SCHEMA",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_CONFIG_SCHEMA",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_DETERMINISTIC_SCHEMA",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_PROTOCOL_SCHEMA",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_REPORT_SCHEMA",
    "KONDO_SPARSE_ACTOR_DEVELOPMENT_TIMING_SCHEMA",
    "KondoSparseActorDevelopmentConfig",
    "KondoSparseActorDevelopmentEvaluator",
    "KondoSparseActorDevelopmentRunState",
    "KondoSparseActorDevelopmentValidationReceipt",
    "MATCHED_DEVELOPMENT_ARMS",
    "PROMOTION_AUTHORITY",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "build_kondo_sparse_actor_deterministic_payload",
    "build_kondo_sparse_actor_development_report",
    "kondo_sparse_actor_development_protocol",
    "kondo_sparse_actor_development_runtime_identity",
    "kondo_sparse_actor_development_source_manifest",
    "measure_kondo_sparse_actor_backward_timing",
    "validate_kondo_sparse_actor_development_report",
]
