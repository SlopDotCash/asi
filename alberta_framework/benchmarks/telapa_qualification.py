"""Bounded TeLAPA-inspired policy-archive qualification smoke.

This development lane deliberately tests ASI's archive primitive in the live
``SwitchingTwoStateMDP`` stream. It is not an implementation or reproduction
of TeLAPA: the public paper repository lacks its declared license file, and
this lane has neither PPO/MAP-Elites nor a learned embedder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.policy_archive as policy_archive_module
import alberta_framework.streams.closed_loop as closed_loop_module
from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    collect_development_identity,
    identity_from_payload,
    require_current_identity,
)
from alberta_framework.core.policy_archive import BoundedPolicyArchive, PolicyEntry
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig, SwitchingTwoStateMDP

SCHEMA = "asi.telapa_qualification_smoke.development.v2"
PAPER_REVISION = "arXiv:2604.15414v1"
PAPER_DATE = "2026-04-16"
DISCLOSED_REPOSITORY = "https://github.com/lute47lillo/telapa_collas2026"
REPOSITORY_REVISION = "a4dc16ed0ea015b1b8efb271e4d664931adccd3e"
REPOSITORY_TREE = "e58072c9c87f984ec9644c7a8fb18e4ce9455286"
SOURCE_ARCHIVE_SHA256 = "25a77241a99a83002a91e282f6a969670f2cb968d2ad685229a904e43a5a926b"
SOURCE_ARCHIVE_BYTES = 8_621_070
README_SHA256 = "c1ac75ece6ddb3eb67ead1779053c1f2c51283d4e389af75073e9ce9ec052438"
ENVIRONMENT_SHA256 = "8b2b9fe39fb7103ed090a429799ada34443b58dcd185e20b32e99913f6553a77"
_MAX_JSON_BYTES = 1_000_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_JSON_CONTAINER_ITEMS = 10_000
_MAX_JSON_STRING_BYTES = 100_000
_MAX_STEPS = 64
FROZEN_DEVELOPMENT_SEEDS = (1_586_000, 1_586_001, 1_586_002)
_POLICY_SHAPE = (2, 2)
_POLICY_BYTES = 16
_ARMS = ("diverse_archive", "one_model", "fixed_snapshot", "mechanism_off")
Arm = Literal["diverse_archive", "one_model", "fixed_snapshot", "mechanism_off"]


def _current_identity(
    config: TeLAPASmokeConfig, catalog: TeLAPACatalogEntry
) -> DevelopmentIdentity:
    return collect_development_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(policy_archive_module, closed_loop_module),
        workload_registry={"config": _config_payload(config), "arms": _ARMS},
        paper_registry=_catalog_payload(catalog),
    )


def _preflight_json_tree(value: object) -> None:
    """Bound an exact primitive JSON tree before invoking the recursive serializer."""
    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    conservative_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("JSON tree exceeds its node or depth limit")
        if item is None or type(item) is bool:
            conservative_bytes += 5
        elif type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("JSON integers must be signed 64-bit values")
            conservative_bytes += 21
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("result must be finite JSON")
            conservative_bytes += 32
        elif type(item) is str:
            if len(item) > _MAX_JSON_STRING_BYTES:
                raise ValueError("JSON string exceeds its character limit")
            encoded_length = len(item.encode("utf-8"))
            if encoded_length > _MAX_JSON_STRING_BYTES:
                raise ValueError("JSON string exceeds its byte limit")
            # ensure_ascii JSON can use two six-byte surrogate escapes per code point.
            conservative_bytes += 2 + 12 * len(item)
        elif type(item) is list:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("JSON tree cannot contain cycles or container aliases")
            seen_containers.add(identity)
            if len(item) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("JSON list exceeds its item limit")
            conservative_bytes += len(item) + 2
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("JSON tree cannot contain cycles or container aliases")
            seen_containers.add(identity)
            if len(item) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("JSON object exceeds its item limit")
            conservative_bytes += len(item) + 2
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be exact strings")
                if len(key) > _MAX_JSON_STRING_BYTES:
                    raise ValueError("JSON object key exceeds its character limit")
                key_bytes = len(key.encode("utf-8"))
                if key_bytes > _MAX_JSON_STRING_BYTES:
                    raise ValueError("JSON object key exceeds its byte limit")
                nodes += 1
                if nodes > _MAX_JSON_NODES:
                    raise ValueError("JSON tree exceeds its node limit")
                conservative_bytes += 3 + 12 * len(key)
                stack.append((child, depth + 1))
        else:
            raise ValueError("JSON tree values must use exact primitive JSON types")
        if conservative_bytes > _MAX_JSON_BYTES:
            raise ValueError("JSON tree exceeds its conservative byte limit")


def _bounded_json_bytes(value: object) -> bytes:
    _preflight_json_tree(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("result must be finite JSON with a bounded exact tree") from error
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("result exceeds the bounded JSON size")
    return raw


@dataclass(frozen=True, slots=True)
class TeLAPACatalogEntry:
    """Machine-readable provenance and adaptation-gap record."""

    issue: int = 1586
    paper_revision: str = PAPER_REVISION
    paper_revision_date: str = PAPER_DATE
    disclosed_repository: str = DISCLOSED_REPOSITORY
    repository_revision: str = REPOSITORY_REVISION
    repository_tree_digest: str = REPOSITORY_TREE
    source_archive_sha256: str = SOURCE_ARCHIVE_SHA256
    source_archive_bytes: int = SOURCE_ARCHIVE_BYTES
    readme_sha256: str = README_SHA256
    environment_sha256: str = ENVIRONMENT_SHA256
    immutable_external_source_established: bool = True
    readme_declared_license: str = "MIT"
    license_file_present: bool = False
    license_review_complete: bool = False
    source_bytes_vendored: bool = False
    paper_parity_allowed: bool = False
    paper_mechanism: str = (
        "per-task policy neighborhoods, MAP-Elites illumination, few-shot retrieval, "
        "and a learned trajectory latent maintained with anchors/replay/re-embedding"
    )
    development_adapter: str = (
        "two-state tabular policy snapshots and a fixed four-statistic rollout descriptor"
    )
    protocol_differences: tuple[str, ...] = (
        "SwitchingTwoStateMDP instead of the paper MiniGrid and MuJoCo curricula",
        "tabular reward update instead of PPO",
        "bounded deterministic insertion instead of MAP-Elites illumination",
        "fixed statistic descriptor instead of a learned aligned trajectory embedder",
        "no anchor/replay bank, re-embedding, few-shot adaptation, or transfer metric",
    )
    paper_metrics: tuple[str, ...] = (
        "standardized_time_to_threshold",
        "success_rate",
        "backward_transfer",
        "transfer_ratio",
    )
    timing_policy: str = "telemetry_only"
    promotion_policy: str = "permanently_nonpromoting_development_smoke"

    def validate(self) -> None:
        if (
            type(self.issue) is not int
            or self.issue != 1586
            or type(self.paper_revision) is not str
            or self.paper_revision != PAPER_REVISION
            or type(self.paper_revision_date) is not str
            or self.paper_revision_date != PAPER_DATE
            or type(self.disclosed_repository) is not str
            or self.disclosed_repository != DISCLOSED_REPOSITORY
        ):
            raise ValueError("TeLAPA catalog identity mismatch")
        if (
            self.repository_revision != REPOSITORY_REVISION
            or self.repository_tree_digest != REPOSITORY_TREE
            or self.source_archive_sha256 != SOURCE_ARCHIVE_SHA256
            or type(self.source_archive_bytes) is not int
            or self.source_archive_bytes != SOURCE_ARCHIVE_BYTES
            or self.readme_sha256 != README_SHA256
            or self.environment_sha256 != ENVIRONMENT_SHA256
        ):
            raise ValueError("TeLAPA public immutable identity mismatch")
        if (
            type(self.immutable_external_source_established) is not bool
            or not self.immutable_external_source_established
            or self.readme_declared_license != "MIT"
            or type(self.license_file_present) is not bool
            or self.license_file_present
            or type(self.license_review_complete) is not bool
            or self.license_review_complete
            or type(self.source_bytes_vendored) is not bool
            or self.source_bytes_vendored
            or type(self.paper_parity_allowed) is not bool
            or self.paper_parity_allowed
        ):
            raise ValueError("missing TeLAPA license file must fail closed")
        if (
            type(self.paper_mechanism) is not str
            or type(self.development_adapter) is not str
            or type(self.protocol_differences) is not tuple
            or len(self.protocol_differences) != 5
            or any(type(value) is not str for value in self.protocol_differences)
            or type(self.paper_metrics) is not tuple
            or len(self.paper_metrics) != 4
            or any(type(value) is not str for value in self.paper_metrics)
        ):
            raise ValueError("protocol differences must remain explicit")
        if type(self.timing_policy) is not str or self.timing_policy != "telemetry_only":
            raise ValueError("timing cannot be a decision axis")
        if (
            type(self.promotion_policy) is not str
            or self.promotion_policy != "permanently_nonpromoting_development_smoke"
        ):
            raise ValueError("qualification smoke must remain permanently nonpromoting")


@dataclass(frozen=True, slots=True)
class TeLAPASmokeConfig:
    seeds: tuple[int, ...] = FROZEN_DEVELOPMENT_SEEDS
    steps: int = 32
    phase_length: int = 4
    archive_byte_budget: int = 4096
    min_latent_distance: float = 0.05
    learning_rate: float = 0.125

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or self.seeds != FROZEN_DEVELOPMENT_SEEDS:
            raise ValueError("development seeds are frozen")
        if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in self.seeds):
            raise ValueError("seeds must contain exact nonnegative signed-int32 values")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if type(self.steps) is not int or not 1 <= self.steps <= _MAX_STEPS:
            raise ValueError(f"steps must be in [1, {_MAX_STEPS}]")
        if type(self.phase_length) is not int or not 1 <= self.phase_length <= 16:
            raise ValueError("phase_length must be in [1, 16]")
        if self.steps % self.phase_length != 0:
            raise ValueError("steps must contain complete phases")
        if (
            type(self.archive_byte_budget) is not int
            or not 128 <= self.archive_byte_budget <= 1024 * 1024
        ):
            raise ValueError("archive_byte_budget must be in [128, 1 MiB]")
        maximum_identity_bytes = max(
            len(f"seed-{seed}-boundary-{boundary}".encode())
            for seed in self.seeds
            for boundary in range(self.phase_length, self.steps + 1, self.phase_length)
        )
        maximum_entry_bytes = maximum_identity_bytes + _POLICY_BYTES + 4 * 8 + 8
        required_archive_bytes = maximum_entry_bytes * (self.steps // self.phase_length)
        if self.archive_byte_budget < required_archive_bytes:
            raise ValueError(
                "archive_byte_budget cannot retain the worst-case matched diverse arm"
            )
        for name, value, upper in (
            ("min_latent_distance", self.min_latent_distance, 10.0),
            ("learning_rate", self.learning_rate, 1.0),
        ):
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= upper:
                raise ValueError(f"{name} must be a bounded finite exact float")


def _catalog_payload(catalog: TeLAPACatalogEntry) -> dict[str, object]:
    payload: dict[str, object] = asdict(catalog)
    payload["protocol_differences"] = list(catalog.protocol_differences)
    payload["paper_metrics"] = list(catalog.paper_metrics)
    return payload


def _config_payload(config: TeLAPASmokeConfig) -> dict[str, object]:
    payload: dict[str, object] = asdict(config)
    payload["seeds"] = list(config.seeds)
    return payload


@jax.jit
def rollout_latent_descriptor(
    observations: Array, actions: Array, rewards: Array
) -> Array:
    """Return a fixed behavioral descriptor from one phase of live experience."""

    return jnp.stack(
        (
            jnp.mean(observations[:, 0]),
            jnp.mean(actions.astype(jnp.float32)),
            jnp.mean(rewards),
            jnp.mean(jnp.abs(jnp.diff(actions, prepend=actions[:1])).astype(jnp.float32)),
        )
    ).astype(jnp.float32)


def _policy_bytes(policy: np.ndarray) -> bytes:
    array = np.asarray(policy, dtype="<f4")
    if array.shape != _POLICY_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError("policy must be a finite 2x2 float32 table")
    payload = np.ascontiguousarray(array).tobytes()
    if len(payload) != _POLICY_BYTES:
        raise AssertionError("policy codec width drifted")
    return payload


def _decode_policy(payload: bytes) -> np.ndarray:
    if type(payload) is not bytes or len(payload) != _POLICY_BYTES:
        raise ValueError("serialized policy must be exactly 16 bytes")
    return np.frombuffer(payload, dtype="<f4").reshape(_POLICY_SHAPE).copy()


class SwitchingPolicyLifeAdapter:
    """Pure bounded policy/environment bridge for one current ASI stream step."""

    __slots__ = ("environment", "learning_rate")

    def __init__(self, *, phase_length: int, learning_rate: float) -> None:
        if (
            type(learning_rate) is not float
            or not math.isfinite(learning_rate)
            or not 0.0 <= learning_rate <= 1.0
        ):
            raise ValueError("learning_rate must be a finite exact float in [0, 1]")
        self.environment = SwitchingTwoStateMDP(
            SwitchingTwoStateConfig(phase_length=phase_length)  # type: ignore[call-arg]
        )
        self.learning_rate = learning_rate

    def init(self, key: Array) -> tuple[Any, np.ndarray]:
        environment_state = self.environment.init(jr.fold_in(key, 0))
        policy = (
            np.asarray(jr.normal(jr.fold_in(key, 1), _POLICY_SHAPE), dtype=np.float32)
            * 0.01
        )
        return environment_state, policy

    def step(
        self,
        environment_state: Any,
        policy: np.ndarray,
        *,
        key: Array,
    ) -> tuple[Any, np.ndarray, np.ndarray, int, float]:
        if (
            type(policy) is not np.ndarray
            or policy.shape != _POLICY_SHAPE
            or policy.dtype != np.dtype(np.float32)
            or not np.all(np.isfinite(policy))
        ):
            raise ValueError("live policy must be an exact finite 2x2 float32 ndarray")
        observation = np.asarray(self.environment.observe(environment_state), dtype="<f4")
        state_index = int(np.argmax(observation))
        action = int(np.argmax(policy[state_index]))
        _, reward_array, next_environment_state = self.environment.step(
            environment_state,
            jnp.asarray(action, dtype=jnp.int32),
            key,
        )
        reward = float(np.asarray(reward_array, dtype=np.float32))
        next_policy = policy.copy()
        next_policy[state_index, action] = np.float32(
            next_policy[state_index, action] + self.learning_rate * (reward - 0.5)
        )
        return next_environment_state, next_policy, observation, action, reward


def _digest(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _archive_mode(arm: Arm) -> Literal["diverse_archive", "one_model", "fixed_snapshot"]:
    if arm == "mechanism_off":
        return "fixed_snapshot"
    return arm


def _run_arm(config: TeLAPASmokeConfig, seed: int, arm: Arm) -> dict[str, Any]:
    adapter = SwitchingPolicyLifeAdapter(
        phase_length=config.phase_length,
        learning_rate=config.learning_rate,
    )
    root = jr.key(seed)
    environment_state, policy = adapter.init(root)
    initial_policy = _policy_bytes(policy)
    archive = BoundedPolicyArchive(
        byte_budget=config.archive_byte_budget,
        min_latent_distance=config.min_latent_distance,
        mode=_archive_mode(arm),
    )
    fixed_anchor: bytes | None = None
    observations: list[bytes] = []
    actions: list[bytes] = []
    rewards: list[bytes] = []
    phase_observations: list[np.ndarray] = []
    phase_actions: list[int] = []
    phase_rewards: list[float] = []
    reward_sum = 0.0
    archive_queries = 0
    descriptor_queries = 0
    boundary_disclosures = 0

    for step in range(config.steps):
        environment_state, policy, observation, action, reward = adapter.step(
            environment_state,
            policy,
            key=jr.fold_in(root, step + 2),
        )
        observations.append(np.ascontiguousarray(observation).tobytes())
        actions.append(np.asarray(action, dtype="<i4").tobytes())
        rewards.append(np.asarray(reward, dtype="<f4").tobytes())
        phase_observations.append(observation)
        phase_actions.append(action)
        phase_rewards.append(reward)
        reward_sum += reward
        if (step + 1) % config.phase_length == 0:
            descriptor = np.asarray(
                rollout_latent_descriptor(
                    jnp.asarray(np.stack(phase_observations), dtype=jnp.float32),
                    jnp.asarray(phase_actions, dtype=jnp.int32),
                    jnp.asarray(phase_rewards, dtype=jnp.float32),
                ),
                dtype=np.float32,
            )
            descriptor_queries += 1
            boundary_disclosures += 1
            score = float(np.mean(np.asarray(phase_rewards, dtype=np.float64)))
            snapshot = _policy_bytes(policy)
            if fixed_anchor is None:
                fixed_anchor = snapshot
            if arm != "mechanism_off":
                retrieved = None
                if arm == "diverse_archive":
                    query = tuple(float(value) for value in descriptor)
                    archive_queries += len(archive.entries)
                    retrieved = archive.retrieve_nearest(query)
                archive = archive.add(
                    PolicyEntry(
                        identity=f"seed-{seed}-boundary-{step + 1}",
                        policy_bytes=snapshot,
                        latent=tuple(float(value) for value in descriptor),
                        score=score,
                    )
                )
                if arm == "diverse_archive":
                    policy = _decode_policy(
                        snapshot if retrieved is None else retrieved.policy_bytes
                    )
                elif arm == "one_model":
                    archive_queries += len(archive.entries)
                    policy = _decode_policy(archive.entries[-1].policy_bytes)
                else:
                    archive_queries += len(archive.entries)
                    policy = _decode_policy(archive.entries[0].policy_bytes)
            else:
                # Exact archive-off reduction of the fixed-snapshot behavior.
                archive_queries += 1
                policy = _decode_policy(fixed_anchor)
            phase_observations.clear()
            phase_actions.clear()
            phase_rewards.clear()

    environment_bytes = 8  # state_index + step_count, both int32 scalars
    retained_archive_bytes = archive.persistent_bytes if arm != "mechanism_off" else 0
    anchor_bytes = _POLICY_BYTES if arm == "mechanism_off" else 0
    return {
        "arm": arm,
        "seed": seed,
        "observation_sha256": _digest(observations),
        "action_sha256": _digest(actions),
        "reward_sha256": _digest(rewards),
        "initial_policy_sha256": hashlib.sha256(initial_policy).hexdigest(),
        "final_policy_sha256": hashlib.sha256(_policy_bytes(policy)).hexdigest(),
        "reward_sum": float(reward_sum),
        "mean_reward": float(reward_sum / config.steps),
        "archive_entry_count": len(archive.entries) if arm != "mechanism_off" else 0,
        "resource_receipt": {
            "environment_steps": config.steps,
            "observations_consumed": config.steps,
            "policy_updates": config.steps,
            "policy_queries": config.steps,
            "descriptor_model_queries": descriptor_queries,
            "archive_entry_queries": archive_queries,
            "task_boundary_disclosures": boundary_disclosures,
            "observation_bytes": 8 * config.steps,
            "action_bytes": 4 * config.steps,
            "reward_bytes": 4 * config.steps,
            "active_policy_persistent_bytes": _POLICY_BYTES,
            "archive_persistent_bytes": retained_archive_bytes,
            "mechanism_off_anchor_bytes": anchor_bytes,
            "descriptor_model_persistent_bytes": 0,
            "environment_state_persistent_bytes": environment_bytes,
            "timing": None,
            "timing_policy": "telemetry_only",
        },
    }


def run_smoke(config: TeLAPASmokeConfig | None = None) -> dict[str, Any]:
    """Execute the bounded matched development matrix in-process."""

    config = TeLAPASmokeConfig() if config is None else config
    if type(config) is not TeLAPASmokeConfig:
        raise ValueError("config must be an exact TeLAPASmokeConfig")
    catalog = TeLAPACatalogEntry()
    catalog.validate()
    records = [_run_arm(config, seed, cast(Arm, arm)) for seed in config.seeds for arm in _ARMS]
    for seed in config.seeds:
        fixed = next(
            record
            for record in records
            if record["seed"] == seed and record["arm"] == "fixed_snapshot"
        )
        off = next(
            record
            for record in records
            if record["seed"] == seed and record["arm"] == "mechanism_off"
        )
        parity_fields = (
            "observation_sha256",
            "action_sha256",
            "reward_sha256",
            "initial_policy_sha256",
            "final_policy_sha256",
        )
        if any(fixed[field] != off[field] for field in parity_fields):
            raise AssertionError("mechanism-off reduction diverged from fixed snapshot")
    result = {
        "schema": SCHEMA,
        "catalog": _catalog_payload(catalog),
        "config": _config_payload(config),
        "identity": _current_identity(config, catalog).to_payload(),
        "matched_axes": [
            "seed",
            "environment_steps",
            "observations_consumed",
            "policy_updates",
            "policy_queries",
            "descriptor_model_queries",
            "task_boundary_disclosures",
        ],
        "allowed_information": {
            "task_boundaries_visible_to_archive": True,
            "task_identity_visible_to_policy": False,
            "future_task_information_visible": False,
            "prior_environment_access": False,
        },
        "mechanism_off_parity": "exact_behavior_and_active_policy_state_vs_fixed_snapshot",
        "records": records,
        "negative_retention": {
            "required": True,
            "policy": "retain every valid development outcome, including ties and regressions",
        },
        "classification": "bounded_synthetic_development_smoke",
        "scientific_promotion_allowed": False,
        "paper_parity_claimed": False,
        "performance_claimed": False,
    }
    validate_result(result)
    return result


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact object")
    result = cast(dict[str, Any], value)
    if any(type(key) is not str for key in result) or set(result) != expected:
        raise ValueError(f"{name} fields do not match the schema")
    return result


def validate_result(value: object) -> None:
    """Strictly validate a bounded JSON-decoded smoke result."""

    _bounded_json_bytes(value)
    root = _require_exact_keys(
        value,
        {
            "schema", "catalog", "config", "identity", "matched_axes", "allowed_information",
            "mechanism_off_parity", "records", "negative_retention", "classification",
            "scientific_promotion_allowed", "paper_parity_claimed", "performance_claimed",
        },
        "result",
    )
    if (
        type(root["schema"]) is not str
        or root["schema"] != SCHEMA
        or type(root["classification"]) is not str
        or root["classification"] != "bounded_synthetic_development_smoke"
    ):
        raise ValueError("result identity mismatch")
    for field in ("scientific_promotion_allowed", "paper_parity_claimed", "performance_claimed"):
        if root[field] is not False:
            raise ValueError(f"{field} must remain false")
    catalog_payload = dict(
        _require_exact_keys(root["catalog"], set(asdict(TeLAPACatalogEntry())), "catalog")
    )
    catalog_payload["protocol_differences"] = tuple(catalog_payload["protocol_differences"])
    catalog_payload["paper_metrics"] = tuple(catalog_payload["paper_metrics"])
    catalog = TeLAPACatalogEntry(**catalog_payload)
    catalog.validate()
    config_payload = dict(
        _require_exact_keys(root["config"], set(asdict(TeLAPASmokeConfig())), "config")
    )
    config_payload["seeds"] = tuple(config_payload["seeds"])
    config = TeLAPASmokeConfig(**config_payload)
    require_current_identity(
        identity_from_payload(root["identity"]), _current_identity(config, catalog)
    )
    expected_axes = [
        "seed", "environment_steps", "observations_consumed", "policy_updates",
        "policy_queries", "descriptor_model_queries", "task_boundary_disclosures",
    ]
    if (
        type(root["matched_axes"]) is not list
        or any(type(value) is not str for value in root["matched_axes"])
        or root["matched_axes"] != expected_axes
    ):
        raise ValueError("matched axes changed")
    allowed = _require_exact_keys(
        root["allowed_information"],
        {
            "task_boundaries_visible_to_archive",
            "task_identity_visible_to_policy",
            "future_task_information_visible",
            "prior_environment_access",
        },
        "allowed information",
    )
    if any(type(value) is not bool for value in allowed.values()) or allowed != {
        "task_boundaries_visible_to_archive": True,
        "task_identity_visible_to_policy": False,
        "future_task_information_visible": False,
        "prior_environment_access": False,
    }:
        raise ValueError("allowed information declaration changed")
    if (
        type(root["mechanism_off_parity"]) is not str
        or root["mechanism_off_parity"]
        != "exact_behavior_and_active_policy_state_vs_fixed_snapshot"
    ):
        raise ValueError("mechanism-off parity declaration changed")
    negative = _require_exact_keys(
        root["negative_retention"], {"required", "policy"}, "negative retention"
    )
    if (
        type(negative["required"]) is not bool
        or type(negative["policy"]) is not str
        or negative
        != {
        "required": True,
        "policy": "retain every valid development outcome, including ties and regressions",
        }
    ):
        raise ValueError("negative retention must remain mandatory")
    records = root["records"]
    if type(records) is not list or len(records) != len(config.seeds) * len(_ARMS):
        raise ValueError("record matrix is incomplete")
    record_keys = {
        "arm", "seed", "observation_sha256", "action_sha256", "reward_sha256",
        "initial_policy_sha256", "final_policy_sha256", "archive_entry_count",
        "reward_sum", "mean_reward", "resource_receipt",
    }
    receipt_keys = {
        "environment_steps", "observations_consumed", "policy_updates", "policy_queries",
        "descriptor_model_queries", "archive_entry_queries", "task_boundary_disclosures",
        "observation_bytes", "action_bytes", "reward_bytes",
        "active_policy_persistent_bytes", "archive_persistent_bytes",
        "mechanism_off_anchor_bytes", "descriptor_model_persistent_bytes",
        "environment_state_persistent_bytes", "timing", "timing_policy",
    }
    expected_pairs = {(seed, arm) for seed in config.seeds for arm in _ARMS}
    observed_pairs: set[tuple[int, str]] = set()
    by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for index, item in enumerate(records):
        record = _require_exact_keys(item, record_keys, f"records[{index}]")
        if type(record["seed"]) is not int or type(record["arm"]) is not str:
            raise ValueError("record seed/arm identities are invalid")
        pair = (record["seed"], record["arm"])
        if pair not in expected_pairs or pair in observed_pairs:
            raise ValueError("record matrix identity is invalid")
        observed_pairs.add(pair)
        by_pair[pair] = record
        for field in (
            "observation_sha256", "action_sha256", "reward_sha256",
            "initial_policy_sha256", "final_policy_sha256",
        ):
            digest = record[field]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        for field in ("reward_sum", "mean_reward"):
            if type(record[field]) is not float or not math.isfinite(record[field]):
                raise ValueError(f"{field} must be a finite exact float")
        if (
            type(record["archive_entry_count"]) is not int
            or not 0
            <= record["archive_entry_count"]
            <= config.steps // config.phase_length
        ):
            raise ValueError("archive entry count is invalid")
        receipt = _require_exact_keys(record["resource_receipt"], receipt_keys, "resource receipt")
        exact = {
            "environment_steps": config.steps,
            "observations_consumed": config.steps,
            "policy_updates": config.steps,
            "policy_queries": config.steps,
            "descriptor_model_queries": config.steps // config.phase_length,
            "task_boundary_disclosures": config.steps // config.phase_length,
            "observation_bytes": 8 * config.steps,
            "action_bytes": 4 * config.steps,
            "reward_bytes": 4 * config.steps,
            "active_policy_persistent_bytes": _POLICY_BYTES,
            "descriptor_model_persistent_bytes": 0,
            "environment_state_persistent_bytes": 8,
        }
        if any(
            type(receipt[name]) is not int or receipt[name] != expected
            for name, expected in exact.items()
        ):
            raise ValueError("exact resource receipt mismatch")
        for field in (
            "archive_entry_queries",
            "archive_persistent_bytes",
            "mechanism_off_anchor_bytes",
        ):
            if type(receipt[field]) is not int or receipt[field] < 0:
                raise ValueError(f"{field} must be a nonnegative exact integer")
        if receipt["archive_persistent_bytes"] > config.archive_byte_budget:
            raise ValueError("archive persistent bytes exceed the frozen budget")
        if (
            receipt["timing"] is not None
            or type(receipt["timing_policy"]) is not str
            or receipt["timing_policy"] != "telemetry_only"
        ):
            raise ValueError("timing must remain absent telemetry")
        if record["arm"] == "mechanism_off":
            if (
                record["archive_entry_count"] != 0
                or receipt["archive_persistent_bytes"] != 0
                or receipt["mechanism_off_anchor_bytes"] != _POLICY_BYTES
            ):
                raise ValueError("mechanism-off resource reduction is invalid")
        elif receipt["mechanism_off_anchor_bytes"] != 0:
            raise ValueError("archive arms cannot retain mechanism-off anchor bytes")
        expected_record = _run_arm(config, pair[0], cast(Arm, pair[1]))
        if record != expected_record:
            raise ValueError("record does not replay from the bound configuration")
    if observed_pairs != expected_pairs:
        raise ValueError("record matrix is incomplete")
    for seed in config.seeds:
        fixed = by_pair[(seed, "fixed_snapshot")]
        off = by_pair[(seed, "mechanism_off")]
        for field in (
            "observation_sha256", "action_sha256", "reward_sha256",
            "initial_policy_sha256", "final_policy_sha256",
        ):
            if fixed[field] != off[field]:
                raise ValueError("mechanism-off exact parity failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true", help="emit provenance only")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--phase-length", type=int, default=4)
    args = parser.parse_args(argv)
    payload: object
    if args.catalog:
        catalog = TeLAPACatalogEntry()
        catalog.validate()
        payload = _catalog_payload(catalog)
    else:
        payload = run_smoke(TeLAPASmokeConfig(steps=args.steps, phase_length=args.phase_length))
    sys.stdout.write(_bounded_json_bytes(payload).decode() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
