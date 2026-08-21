"""Bounded closed-loop development lane for an FTL online agent.

This is an ASI-native protocol qualification, not a reproduction of Continual
Bench and not an extension of the historical ``ftl_world_model_decision_fidelity``
claim.  It exercises the current sparse FTL model through observe, MPC, act,
and online update against a frozen mechanism-off arm and a privileged dynamics
control.  Every result is permanently nonpromoting.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import operator
import sys
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import alberta_framework.core.ftl_world_model as ftl_world_model_module
from alberta_framework._scan_resources import ScanBudget, require_scan_steps
from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    collect_development_identity,
    identity_from_payload,
    require_current_identity,
)
from alberta_framework.core.ftl_world_model import SparseFTLWorldModel, SparseFTLWorldModelConfig

SCHEMA = "asi.ftl_online_agent_development.v1"
PAPER = "PMLR 267 (2025), pp. 38397-38423; arXiv:2507.09177v1"
OFFICIAL_CODE_REVISION = "sail-sg/ContinualBench@a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad"
FROZEN_SEEDS = (1701, 1702, 1703, 1704)
FROZEN_GOALS = ((2.0, 0.0), (0.0, 2.0), (-2.0, 0.0))
ARM_IDS = ("sparse_ftl_online", "sparse_ftl_frozen", "privileged_dynamics_mpc")
ACTION_DELTAS = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=np.float32)
ACTION_DELTAS.flags.writeable = False
MAX_STEPS_PER_TASK = 16
_MPC_ENUMERATION_BUDGET = ScanBudget("FTL MPC enumeration", maximum_steps=4)
MAX_PLANNING_HORIZON = _MPC_ENUMERATION_BUDGET.maximum_steps
WORKLOAD_REGISTRY = (
    ("arm_ids", ARM_IDS),
    ("action_deltas", ((1, 0), (-1, 0), (0, 1), (0, -1))),
    ("frozen_goals", FROZEN_GOALS),
    ("frozen_seeds", FROZEN_SEEDS),
    ("max_steps_per_task", MAX_STEPS_PER_TASK),
)
PAPER_REGISTRY = MappingProxyType(
    {"paper": PAPER, "official_code_revision": OFFICIAL_CODE_REVISION}
)


def _current_identity() -> DevelopmentIdentity:
    return collect_development_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(ftl_world_model_module,),
        workload_registry=WORKLOAD_REGISTRY,
        paper_registry=PAPER_REGISTRY,
    )


def _int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _tree_nbytes(value: object) -> int:
    return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(value))


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    environment_steps: int
    model_updates: int
    model_queries: int
    planner_candidates: int
    logical_compute_units: int
    persistent_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "environment_steps", "model_updates", "model_queries", "planner_candidates",
            "logical_compute_units", "persistent_bytes", "elapsed_ns",
        ):
            _int(getattr(self, name), name, 0, 2**63 - 1)
        if self.environment_steps == 0 or self.model_queries == 0 or self.persistent_bytes == 0:
            raise ValueError("steps, queries, and persistent bytes must be positive")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    task_returns: tuple[float, ...]
    prequential_squared_errors: tuple[float, ...]
    receipt: ResourceReceipt
    candidate_eligible: bool

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm_id")
        if type(self.task_returns) is not tuple or len(self.task_returns) != len(FROZEN_GOALS):
            raise ValueError("task_returns must bind every frozen task")
        if type(self.prequential_squared_errors) is not tuple:
            raise ValueError("prequential errors must be an exact tuple")
        values = self.task_returns + self.prequential_squared_errors
        if any(type(x) is not float or not math.isfinite(x) for x in values):
            raise ValueError("metrics must be finite exact floats")
        expected_eligible = self.arm_id != "privileged_dynamics_mpc"
        if (
            type(self.candidate_eligible) is not bool
            or self.candidate_eligible != expected_eligible
        ):
            raise ValueError("privileged control must be excluded from candidate comparison")


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentResult:
    schema: str
    seed: int
    steps_per_task: int
    planning_horizon: int
    arms: tuple[ArmResult, ...]
    identity: DevelopmentIdentity
    allowed_boundary_information: tuple[str, ...] = ()
    allowed_task_information: tuple[str, ...] = ("current_goal",)
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True
    historical_ftl_claim_reused: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unsupported schema")
        if self.seed not in FROZEN_SEEDS:
            raise ValueError("seed is outside the frozen development schedule")
        _int(self.steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
        _int(self.planning_horizon, "planning_horizon", 1, MAX_PLANNING_HORIZON)
        if (
            type(self.arms) is not tuple
            or any(type(arm) is not ArmResult for arm in self.arms)
            or tuple(x.arm_id for x in self.arms) != ARM_IDS
        ):
            raise ValueError("arms differ from the frozen roster")
        if type(self.identity) is not DevelopmentIdentity:
            raise ValueError("identity must be exact")
        if (
            type(self.allowed_boundary_information) is not tuple
            or self.allowed_boundary_information
        ):
            raise ValueError("the learner and planner receive no boundary information")
        if (
            type(self.allowed_task_information) is not tuple
            or len(self.allowed_task_information) != 1
            or type(self.allowed_task_information[0]) is not str
            or self.allowed_task_information[0] != "current_goal"
        ):
            raise ValueError("the planner receives only the current goal as task information")
        flags = (
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
            not self.historical_ftl_claim_reused,
        )
        if any(type(x) is not bool or not x for x in flags):
            raise ValueError("result must remain nonpromoting, retained, and historically separate")


def _action_vector(action: int) -> jax.Array:
    return jax.nn.one_hot(jnp.asarray(action), 4, dtype=jnp.float32)


def _mpc_action(
    observation: np.ndarray,
    goal: np.ndarray,
    horizon: int,
    predict: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[int, int, int]:
    host_horizon = require_scan_steps(
        "planning_horizon", horizon, _MPC_ENUMERATION_BUDGET
    )
    candidate_count = 4**host_horizon
    best_score = -math.inf
    best_action = 0
    queries = 0
    for sequence in itertools.product(range(4), repeat=host_horizon):
        imagined = observation.copy()
        # arXiv:2507.09177v1 Eq. 2 maximizes the summed H-step reward over the
        # imagined rollout, and this lane's reported return accumulates the same
        # per-step cost at every executed step.  Scoring only the terminal
        # imagined state optimizes a different objective than the one measured,
        # which stops the privileged-dynamics control from bounding the lane.
        score = 0.0
        for action in sequence:
            imagined = np.asarray(predict(imagined, action), dtype=np.float32)
            queries += 1
            score -= float(np.sum((imagined - goal) ** 2))
        if score > best_score:
            best_score, best_action = score, sequence[0]
    return best_action, queries, candidate_count


def _run_arm(seed: int, steps: int, horizon: int, arm_id: str) -> ArmResult:
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=2, action_dim=4, projection_dim=4, bins=4)
    )
    state = model.init(jr.key(seed))
    initial_bytes = _tree_nbytes(state)
    updates = queries = candidates = 0
    errors: list[float] = []
    task_returns: list[float] = []
    start = time.perf_counter_ns()
    for task_index, goal_tuple in enumerate(FROZEN_GOALS):
        observation = np.asarray((0.0, 0.0), dtype=np.float32)
        goal = np.asarray(goal_tuple, dtype=np.float32)
        total = 0.0
        for _ in range(steps):
            if arm_id == "privileged_dynamics_mpc":
                def predict(obs: np.ndarray, action: int) -> np.ndarray:
                    return np.asarray(obs + ACTION_DELTAS[action], dtype=np.float32)
            else:
                current_state = state

                def predict(obs: np.ndarray, action: int) -> np.ndarray:
                    predicted = model.predict(
                        current_state, obs, _action_vector(action)
                    ).next_observation
                    return np.asarray(predicted, dtype=np.float32)
            action, arm_queries, arm_candidates = _mpc_action(observation, goal, horizon, predict)
            queries += arm_queries
            candidates += arm_candidates
            next_observation = observation + ACTION_DELTAS[action]
            total -= float(np.sum((next_observation - goal) ** 2))
            if arm_id == "sparse_ftl_online":
                result = model.update(state, observation, _action_vector(action), next_observation)
                errors.append(float(result.squared_error))
                state = result.state
                updates += 1
                queries += 1  # predict-before-update query inside ``update``
            elif arm_id == "sparse_ftl_frozen":
                prediction = model.predict(state, observation, _action_vector(action))
                error = next_observation - np.asarray(prediction.next_observation)
                errors.append(float(np.mean(error**2)))
                queries += 1
            observation = next_observation
        task_returns.append(total)
        # Task identity affects reward only; the world model gets no boundary signal.
        del task_index
    elapsed = time.perf_counter_ns() - start
    final_bytes = _tree_nbytes(state)
    return ArmResult(
        arm_id=arm_id,
        task_returns=tuple(task_returns),
        prequential_squared_errors=tuple(errors),
        receipt=ResourceReceipt(
            environment_steps=steps * len(FROZEN_GOALS),
            model_updates=updates,
            model_queries=queries,
            planner_candidates=candidates,
            logical_compute_units=(steps * len(FROZEN_GOALS) + queries + candidates + updates),
            persistent_bytes=(
                int(ACTION_DELTAS.nbytes)
                if arm_id == "privileged_dynamics_mpc"
                else max(initial_bytes, final_bytes)
            ),
            elapsed_ns=elapsed,
        ),
        candidate_eligible=arm_id != "privileged_dynamics_mpc",
    )


def run_development_lane(
    *, seed: object, steps_per_task: object = 4, planning_horizon: object = 2
) -> DevelopmentResult:
    """Run one bounded matched closed-loop development shard."""
    host_seed = _int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    steps = _int(steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
    horizon = _int(planning_horizon, "planning_horizon", 1, MAX_PLANNING_HORIZON)
    result = DevelopmentResult(
        schema=SCHEMA,
        seed=host_seed,
        steps_per_task=steps,
        planning_horizon=horizon,
        arms=tuple(_run_arm(host_seed, steps, horizon, arm_id) for arm_id in ARM_IDS),
        identity=_current_identity(),
    )
    validate_result(result)
    return result


def validate_result(value: object) -> DevelopmentResult:
    """Fail closed over a result or an exact dataclass-shaped mapping."""
    if type(value) is dict:
        result_fields = {field.name for field in dataclasses.fields(DevelopmentResult)}
        result_keys = tuple(dict.keys(value))
        if any(type(key) is not str for key in result_keys) or set(result_keys) != result_fields:
            raise ValueError("payload must be an exact mapping with the frozen fields")
        raw_arms = value["arms"]
        if type(raw_arms) is not list or len(raw_arms) != len(ARM_IDS):
            raise ValueError("arms must be a bounded exact list")
        raw_boundary = value["allowed_boundary_information"]
        raw_task = value["allowed_task_information"]
        if type(raw_boundary) is not list or raw_boundary != []:
            raise ValueError("allowed_boundary_information must be an empty exact list")
        if type(raw_task) is not list or raw_task != ["current_goal"]:
            raise ValueError("allowed_task_information must contain only current_goal")
        arms: list[ArmResult] = []
        for raw_arm in raw_arms:
            arm_fields = {field.name for field in dataclasses.fields(ArmResult)}
            if type(raw_arm) is not dict or set(raw_arm) != arm_fields:
                raise ValueError("arm has unexpected fields")
            raw_receipt = raw_arm["receipt"]
            if type(raw_receipt) is not dict or set(raw_receipt) != {
                field.name for field in dataclasses.fields(ResourceReceipt)
            }:
                raise ValueError("receipt has unexpected fields")
            raw_returns = raw_arm["task_returns"]
            raw_errors = raw_arm["prequential_squared_errors"]
            if type(raw_returns) is not list or type(raw_errors) is not list:
                raise ValueError("metric sequences must be exact lists")
            arms.append(
                ArmResult(
                    arm_id=raw_arm["arm_id"],
                    task_returns=tuple(raw_returns),
                    prequential_squared_errors=tuple(raw_errors),
                    receipt=ResourceReceipt(**raw_receipt),
                    candidate_eligible=raw_arm["candidate_eligible"],
                )
            )
        value = DevelopmentResult(
            schema=value["schema"], seed=value["seed"],
            steps_per_task=value["steps_per_task"], planning_horizon=value["planning_horizon"],
            arms=tuple(arms),
            identity=identity_from_payload(value["identity"]),
            allowed_boundary_information=tuple(value["allowed_boundary_information"]),
            allowed_task_information=tuple(value["allowed_task_information"]),
            development_only=value["development_only"],
            scientific_promotion_allowed=value["scientific_promotion_allowed"],
            negative_results_must_be_retained=value["negative_results_must_be_retained"],
            historical_ftl_claim_reused=value["historical_ftl_claim_reused"],
        )
    elif isinstance(value, Mapping):
        raise ValueError("payload must be an exact mapping with the frozen fields")
    if type(value) is not DevelopmentResult:
        raise ValueError("result must be an exact DevelopmentResult")
    DevelopmentResult.__post_init__(value)
    require_current_identity(value.identity, _current_identity())
    expected_steps = value.steps_per_task * len(FROZEN_GOALS)
    expected_candidates = 4**value.planning_horizon * expected_steps
    planner_queries = expected_candidates * value.planning_horizon
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(observation_dim=2, action_dim=4, projection_dim=4, bins=4)
    )
    expected_model_bytes = _tree_nbytes(model.init(jr.key(value.seed)))
    for arm in value.arms:
        expected_error_count = 0 if arm.arm_id == "privileged_dynamics_mpc" else expected_steps
        if len(arm.prequential_squared_errors) != expected_error_count:
            raise ValueError("prequential metric count disagrees with the executed arm")
        if arm.receipt.environment_steps != expected_steps:
            raise ValueError("environment-step receipt mismatch")
        expected_queries = planner_queries + (
            0 if arm.arm_id == "privileged_dynamics_mpc" else expected_steps
        )
        if arm.receipt.planner_candidates != expected_candidates:
            raise ValueError("planner-candidate receipt mismatch")
        if arm.receipt.model_queries != expected_queries:
            raise ValueError("planner receipt mismatch")
        expected_updates = expected_steps if arm.arm_id == "sparse_ftl_online" else 0
        if arm.receipt.model_updates != expected_updates:
            raise ValueError("model-update receipt mismatch")
        expected_compute = (
            expected_steps + expected_queries + expected_candidates + expected_updates
        )
        if arm.receipt.logical_compute_units != expected_compute:
            raise ValueError("logical-compute receipt mismatch")
        expected_bytes = (
            int(ACTION_DELTAS.nbytes)
            if arm.arm_id == "privileged_dynamics_mpc"
            else expected_model_bytes
        )
        if arm.receipt.persistent_bytes != expected_bytes:
            raise ValueError("persistent-byte receipt mismatch")
    return value


__all__ = [
    "ARM_IDS",
    "FROZEN_SEEDS",
    "OFFICIAL_CODE_REVISION",
    "PAPER",
    "SCHEMA",
    "ArmResult",
    "DevelopmentResult",
    "ResourceReceipt",
    "run_development_lane",
    "validate_result",
]
