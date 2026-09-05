"""Bounded end-to-end development qualification for Dreamer-family mechanisms.

This exercises real interaction, replay retention, world-model updates, guarded
imagination, and imagined value updates.  It is not DreamerV3 or
Continual-Dreamer parity and is permanently nonpromoting.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import sys
import time
from collections.abc import Mapping
from typing import SupportsIndex, cast

import jax
import jax.random as jr
import numpy as np

import alberta_framework.core.dreaming as dreaming_module
import alberta_framework.core.world_model as world_model_module
from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    collect_development_identity,
    require_current_identity,
)
from alberta_framework.core.dreaming import DreamingConfig, GuardedDreamer
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

SCHEMA = "asi.dreamer_continual_development.v1"
DREAMERV3_CODE = "danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe"
PAPERS = (
    "Hafner et al., Nature 2025, doi:10.1038/s41586-025-08744-2; arXiv:2301.04104",
    "Kessler et al., CoLLAs 2023, PMLR 232:184-204; arXiv:2211.15944",
)
FROZEN_SEEDS = (1760, 1761, 1762, 1763)
FROZEN_TASK_TARGETS = (0, 1, 0)
ARM_IDS = ("guarded_imagination", "imagination_off", "privileged_task_control")
MAX_STEPS_PER_TASK = 16
MAX_IMAGINATIONS_PER_STEP = 8
WORKLOAD_REGISTRY = (
    ("arm_ids", ARM_IDS),
    ("frozen_seeds", FROZEN_SEEDS),
    ("frozen_task_targets", FROZEN_TASK_TARGETS),
    ("max_imaginations_per_step", MAX_IMAGINATIONS_PER_STEP),
    ("max_steps_per_task", MAX_STEPS_PER_TASK),
)
PAPER_REGISTRY = (("dreamerv3_code", DREAMERV3_CODE), ("papers", PAPERS))


def _current_identity() -> DevelopmentIdentity:
    return collect_development_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(dreaming_module, world_model_module),
        workload_registry=WORKLOAD_REGISTRY,
        paper_registry=PAPER_REGISTRY,
    )


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _tree_nbytes(value: object) -> int:
    leaves = jax.tree_util.tree_leaves(value)
    if len(leaves) > 4096:
        raise ValueError("state contains too many leaves")
    return sum(int(np.asarray(leaf).nbytes) for leaf in leaves)


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    environment_steps: int
    world_model_updates: int
    world_model_queries: int
    replay_inserts: int
    replay_samples: int
    imagination_proposals: int
    imagination_accepts: int
    imagined_value_updates: int
    logical_compute_units: int
    persistent_bytes: int
    peak_replay_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name == "timing_telemetry_only":
                continue
            _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.environment_steps == 0 or self.persistent_bytes == 0:
            raise ValueError("steps and persistent bytes must be positive")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    task_returns: tuple[float, ...]
    final_action_values: tuple[float, float]
    receipt: ResourceReceipt
    candidate_eligible: bool

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm")
        if type(self.task_returns) is not tuple or len(self.task_returns) != 3:
            raise ValueError("task returns must bind the frozen task sequence")
        if type(self.final_action_values) is not tuple or len(self.final_action_values) != 2:
            raise ValueError("action values must be an exact pair")
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in self.task_returns + self.final_action_values
        ):
            raise ValueError("metrics must be finite exact floats")
        expected = self.arm_id != "privileged_task_control"
        if type(self.candidate_eligible) is not bool or self.candidate_eligible != expected:
            raise ValueError("privileged control must be excluded")


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentResult:
    schema: str
    seed: int
    steps_per_task: int
    replay_capacity: int
    imaginations_per_step: int
    arms: tuple[ArmResult, ...]
    identity: DevelopmentIdentity
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True
    dreamer_parity_claimed: bool = False

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.seed not in FROZEN_SEEDS:
            raise ValueError("schema or frozen seed mismatch")
        _exact_int(self.steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
        _exact_int(self.replay_capacity, "replay_capacity", 1, 64)
        _exact_int(
            self.imaginations_per_step,
            "imaginations_per_step",
            1,
            MAX_IMAGINATIONS_PER_STEP,
        )
        if type(self.arms) is not tuple or any(type(arm) is not ArmResult for arm in self.arms):
            raise ValueError("arms must contain exact ArmResult values")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arms differ from the frozen roster")
        if type(self.identity) is not DevelopmentIdentity:
            raise ValueError("identity must be exact")
        flags = (
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
            not self.dreamer_parity_claimed,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("result must remain retained, nonpromoting, and non-parity")


def _transition(observation: np.ndarray, action: int, target: int) -> tuple[np.ndarray, float]:
    direction = 1.0 if action == 0 else -1.0
    next_observation = np.asarray(
        (0.8 * observation[0] + direction, observation[0]), dtype=np.float32
    )
    reward = 1.0 if action == target else -1.0
    return next_observation, reward


def _run_arm(
    seed: int, steps: int, capacity: int, imagination_count: int, arm_id: str
) -> ArmResult:
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
            error_decay=0.0,
        )
    )
    state = model.init(jr.key(seed))
    initial_model_bytes = _tree_nbytes(state)
    dreamer = GuardedDreamer(
        DreamingConfig(warmup_steps=1, max_model_error_ema=1.0e6, max_uncertainty=0.0)
    )
    replay: list[np.ndarray] = []
    values = np.zeros((2,), dtype=np.float32)
    updates = queries = inserts = samples = proposals = accepts = imagined_updates = 0
    peak_replay_bytes = 0
    returns: list[float] = []
    start = time.perf_counter_ns()
    for target in FROZEN_TASK_TARGETS:
        observation = np.zeros((2,), dtype=np.float32)
        task_return = 0.0
        for _ in range(steps):
            action = target if arm_id == "privileged_task_control" else int(np.argmax(values))
            next_observation, reward = _transition(observation, action, target)
            if arm_id != "privileged_task_control":
                learned = model.update(state, observation, action, reward, 0.99, next_observation)
                state = learned.state
                updates += int(bool(learned.update_applied))
                queries += 1
                replay.append(observation.copy())
                if len(replay) > capacity:
                    replay.pop(0)
                inserts += 1
                peak_replay_bytes = max(peak_replay_bytes, sum(item.nbytes for item in replay))
                if arm_id == "guarded_imagination":
                    for index in range(imagination_count):
                        anchor = replay[(seed + index + inserts) % len(replay)]
                        imagined_action = index % 2
                        proposal = dreamer.propose(model, state, anchor, imagined_action)
                        samples += 1
                        proposals += 1
                        queries += 1
                        if bool(proposal.accepted):
                            imagined_reward = float(proposal.transition.reward)
                            values[imagined_action] += 0.1 * (
                                imagined_reward - values[imagined_action]
                            )
                            accepts += 1
                            imagined_updates += 1
            task_return += reward
            observation = next_observation
        returns.append(task_return)
    elapsed = time.perf_counter_ns() - start
    environment_steps = steps * len(FROZEN_TASK_TARGETS)
    persistent = (
        int(np.asarray(FROZEN_TASK_TARGETS, dtype=np.int32).nbytes)
        if arm_id == "privileged_task_control"
        else (
            max(initial_model_bytes, _tree_nbytes(state))
            + int(values.nbytes)
            + sum(int(item.nbytes) for item in replay)
        )
    )
    compute = (
        environment_steps + updates + queries + inserts + samples + proposals + imagined_updates
    )
    return ArmResult(
        arm_id=arm_id,
        task_returns=tuple(float(value) for value in returns),
        final_action_values=(float(values[0]), float(values[1])),
        receipt=ResourceReceipt(
            environment_steps=environment_steps,
            world_model_updates=updates,
            world_model_queries=queries,
            replay_inserts=inserts,
            replay_samples=samples,
            imagination_proposals=proposals,
            imagination_accepts=accepts,
            imagined_value_updates=imagined_updates,
            logical_compute_units=compute,
            persistent_bytes=persistent,
            peak_replay_bytes=peak_replay_bytes,
            elapsed_ns=elapsed,
        ),
        candidate_eligible=arm_id != "privileged_task_control",
    )


def run_development_lane(
    *,
    seed: object,
    steps_per_task: object = 4,
    replay_capacity: object = 8,
    imaginations_per_step: object = 2,
) -> DevelopmentResult:
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    steps = _exact_int(steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
    capacity = _exact_int(replay_capacity, "replay_capacity", 1, 64)
    imagination_count = _exact_int(
        imaginations_per_step,
        "imaginations_per_step",
        1,
        MAX_IMAGINATIONS_PER_STEP,
    )
    result = DevelopmentResult(
        schema=SCHEMA,
        seed=host_seed,
        steps_per_task=steps,
        replay_capacity=capacity,
        imaginations_per_step=imagination_count,
        arms=tuple(
            _run_arm(host_seed, steps, capacity, imagination_count, arm_id)
            for arm_id in ARM_IDS
        ),
        identity=_current_identity(),
    )
    return validate_result(result)


def validate_result(value: object) -> DevelopmentResult:
    if type(value) is not DevelopmentResult:
        if not isinstance(value, Mapping) or type(value) is not dict:
            raise ValueError("payload must be an exact result or dict")
        raise ValueError(
            "mapping decoding is intentionally unavailable until a CLI schema is frozen"
        )
    DevelopmentResult.__post_init__(value)
    require_current_identity(value.identity, _current_identity())
    steps = value.steps_per_task * len(FROZEN_TASK_TARGETS)
    for arm in value.arms:
        ArmResult.__post_init__(arm)
        receipt = arm.receipt
        if type(receipt) is not ResourceReceipt:
            raise ValueError("receipt must be an exact ResourceReceipt")
        ResourceReceipt.__post_init__(receipt)
        if receipt.environment_steps != steps:
            raise ValueError("environment receipt mismatch")
        model_arm = arm.arm_id != "privileged_task_control"
        expected_updates = steps if model_arm else 0
        if receipt.world_model_updates != expected_updates:
            raise ValueError("world-model update receipt mismatch")
        expected_inserts = steps if model_arm else 0
        if receipt.replay_inserts != expected_inserts:
            raise ValueError("replay insert receipt mismatch")
        imagined = steps * value.imaginations_per_step if arm.arm_id == "guarded_imagination" else 0
        if receipt.imagination_proposals != imagined or receipt.replay_samples != imagined:
            raise ValueError("imagination or replay-sample receipt mismatch")
        expected_queries = expected_updates + imagined
        if receipt.world_model_queries != expected_queries:
            raise ValueError("world-model query receipt mismatch")
        if receipt.imagination_accepts != receipt.imagined_value_updates:
            raise ValueError("accepted imagination must produce one value update")
        if receipt.imagination_accepts > receipt.imagination_proposals:
            raise ValueError("imagination accepts exceed proposals")
        expected_compute = steps + expected_updates + expected_queries + expected_inserts
        expected_compute += imagined + imagined + receipt.imagined_value_updates
        if receipt.logical_compute_units != expected_compute:
            raise ValueError("logical compute receipt mismatch")
        expected_peak = min(steps, value.replay_capacity) * 2 * 4 if model_arm else 0
        if receipt.peak_replay_bytes != expected_peak:
            raise ValueError("replay byte receipt mismatch")
        model = ActionConditionedWorldModel(
            ActionConditionedWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                hidden_sizes=(),
                step_size=0.05,
                sparsity=0.0,
                error_decay=0.0,
            )
        )
        expected_persistent = (
            int(np.asarray(FROZEN_TASK_TARGETS, dtype=np.int32).nbytes)
            if not model_arm
            else _tree_nbytes(model.init(jr.key(value.seed))) + 2 * 4 + expected_peak
        )
        if receipt.persistent_bytes != expected_persistent:
            raise ValueError("persistent-byte receipt mismatch")
        expected_arm = _run_arm(
            value.seed,
            value.steps_per_task,
            value.replay_capacity,
            value.imaginations_per_step,
            arm.arm_id,
        )
        observed_arm = dataclasses.replace(
            arm,
            receipt=dataclasses.replace(arm.receipt, elapsed_ns=0),
        )
        expected_arm = dataclasses.replace(
            expected_arm,
            receipt=dataclasses.replace(expected_arm.receipt, elapsed_ns=0),
        )
        if observed_arm != expected_arm:
            raise ValueError("deterministic replay mismatch")
    return value


__all__ = [
    "ARM_IDS", "DREAMERV3_CODE", "FROZEN_SEEDS", "PAPERS", "SCHEMA",
    "ArmResult", "DevelopmentResult", "ResourceReceipt", "run_development_lane",
    "validate_result",
]
