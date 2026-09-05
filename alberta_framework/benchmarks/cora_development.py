"""Bounded CORA-inspired continual-RL metric and control qualification.

The executable environment is a deterministic recurring bandit analogue.  It
qualifies task sequencing, continual evaluation, isolated forgetting/transfer,
mechanism ablation, and accounting without claiming CORA environment parity.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import time
from pathlib import Path
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

SCHEMA = "asi.cora_development.v1"
PAPER = "Powers et al., CoLLAs 2022; arXiv:2110.10067v2"
OFFICIAL_CODE = "AGI-Labs/continual_rl@f2754bb282757829765beb4703f24b87efa13ff9"
FROZEN_SEEDS = (15810, 15811, 15812, 15813)
TASK_TARGETS = (0, 1, 0)
N_CYCLES = 2
ARM_IDS = ("replay_q", "replay_off_q", "task_id_q_control", "uniform_random")
MAX_STEPS_PER_TASK = 32
# One pre-training snapshot plus one checkpoint after every task in every cycle.
_EVALUATION_MATRIX_ROWS = len(TASK_TARGETS) * N_CYCLES + 1


def _runner_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class CORAFamily:
    family_id: str
    tasks: int
    cycles: int
    train_steps_per_task: int
    external_runtime: str
    ready: bool = False

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or self.family_id not in (
            "atari", "procgen", "minihack", "chores"
        ):
            raise ValueError("unknown CORA family")
        _exact_int(self.tasks, "tasks", 1, 32)
        _exact_int(self.cycles, "cycles", 1, 8)
        _exact_int(self.train_steps_per_task, "train_steps_per_task", 1, 100_000_000)
        if type(self.external_runtime) is not str or not self.external_runtime:
            raise ValueError("external_runtime must be a non-empty exact string")
        if type(self.ready) is not bool or self.ready:
            raise ValueError("external CORA families must remain blocked")


CORA_CATALOG = (
    CORAFamily("atari", 6, 2, 50_000_000, "ALE/Atari ROMs; PyTorch IMPALA-family"),
    CORAFamily("procgen", 6, 5, 5_000_000, "procgen; 200 train levels + full eval"),
    CORAFamily("minihack", 15, 2, 10_000_000, "MiniHack + NLE"),
    CORAFamily("chores", 3, 2, 1_000_000, "AI2-THOR + crl_alfred + trajectories"),
)


@jax.jit
def _q_update(values: Array, action: Array, reward: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    error = reward - values[action]
    return values.at[action].add(jnp.asarray(0.2, jnp.float32) * error)


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    training_environment_steps: int
    evaluation_environment_steps: int
    model_queries: int
    agent_updates: int
    replay_inserts: int
    replay_samples: int
    persistent_bytes: int
    peak_replay_bytes: int
    logical_compute_units: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name == "timing_telemetry_only":
                continue
            _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.training_environment_steps == 0 or self.model_queries == 0:
            raise ValueError("training steps and model queries must be positive")
        if self.persistent_bytes == 0:
            raise ValueError("persistent bytes must be positive")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    training_return: float
    evaluation_matrix: tuple[tuple[float, ...], ...]
    continual_evaluation: float
    isolated_forgetting: float
    isolated_forward_transfer: float
    receipt: ResourceReceipt
    candidate_eligible: bool

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm_id")
        if type(self.evaluation_matrix) is not tuple or not self.evaluation_matrix:
            raise ValueError("evaluation_matrix must be a non-empty exact tuple")
        if len(self.evaluation_matrix) != _EVALUATION_MATRIX_ROWS:
            raise ValueError(
                "evaluation_matrix must contain exactly "
                f"{_EVALUATION_MATRIX_ROWS} checkpoint rows"
            )
        malformed_rows = (
            type(row) is not tuple or len(row) != len(TASK_TARGETS)
            for row in self.evaluation_matrix
        )
        if any(malformed_rows):
            raise ValueError("evaluation matrix rows must bind every task")
        scalars = (
            self.training_return,
            self.continual_evaluation,
            self.isolated_forgetting,
            self.isolated_forward_transfer,
            *(value for row in self.evaluation_matrix for value in row),
        )
        if any(type(value) is not float or not math.isfinite(value) for value in scalars):
            raise ValueError("metrics must be finite exact floats")
        if any(not 0.0 <= value <= 1.0 for row in self.evaluation_matrix for value in row):
            raise ValueError("evaluation returns must lie in [0, 1]")
        expected = self.arm_id != "task_id_q_control"
        if type(self.candidate_eligible) is not bool or self.candidate_eligible != expected:
            raise ValueError("task-ID control must be excluded from candidates")


@dataclasses.dataclass(frozen=True, slots=True)
class CORADevelopmentResult:
    schema: str
    paper_revision: str
    official_code_revision: str
    runner_source_sha256: str
    seed: int
    steps_per_task: int
    replay_capacity: int
    task_targets: tuple[int, ...]
    cycles: int
    task_boundaries_available_to_runner: bool
    task_ids_available_to_candidate: bool
    arms: tuple[ArmResult, ...]
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True
    cora_parity_claimed: bool = False

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.seed not in FROZEN_SEEDS:
            raise ValueError("schema or frozen seed mismatch")
        if self.paper_revision != PAPER or self.official_code_revision != OFFICIAL_CODE:
            raise ValueError("external source revision mismatch")
        current_source = _runner_source_sha256()
        if (
            type(self.runner_source_sha256) is not str
            or len(self.runner_source_sha256) != 64
            or self.runner_source_sha256 != current_source
        ):
            raise ValueError("runner source identity mismatch")
        _exact_int(self.steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
        _exact_int(self.replay_capacity, "replay_capacity", 1, 64)
        if self.task_targets != TASK_TARGETS or self.cycles != N_CYCLES:
            raise ValueError("task sequence differs from the frozen analogue")
        if type(self.arms) is not tuple or any(type(arm) is not ArmResult for arm in self.arms):
            raise ValueError("arms must contain exact ArmResult values")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arms differ from the frozen roster")
        flags = (
            self.task_boundaries_available_to_runner,
            not self.task_ids_available_to_candidate,
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
            not self.cora_parity_claimed,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("information and nonpromotion contract mismatch")


def _greedy(values: np.ndarray, tie: int) -> int:
    if values[0] == values[1]:
        return tie
    return int(np.argmax(values))


def _rng_key(seed: int, domain: int, index: int) -> Array:
    return jr.fold_in(jr.fold_in(jr.key(seed), domain), index)


def _evaluate(
    arm_id: str,
    shared: np.ndarray,
    task_values: np.ndarray,
    block: int,
    seed: int,
) -> tuple[float, ...]:
    returns: list[float] = []
    for task_id, target in enumerate(TASK_TARGETS):
        if arm_id == "task_id_q_control":
            action = _greedy(task_values[task_id], (block + task_id) % 2)
        elif arm_id == "uniform_random":
            query_index = block * len(TASK_TARGETS) + task_id
            action = int(jr.randint(_rng_key(seed, 1, query_index), (), 0, 2))
        else:
            action = _greedy(shared, (seed + block) % 2)
        returns.append(float(action == target))
    return tuple(returns)


def _metrics(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float]:
    if type(matrix) is not tuple or len(matrix) != _EVALUATION_MATRIX_ROWS:
        raise ValueError(
            "evaluation_matrix must contain exactly "
            f"{_EVALUATION_MATRIX_ROWS} checkpoint rows"
        )
    array = np.asarray(matrix, dtype=np.float64)
    continual = float(np.mean(array))
    seen: set[int] = set()
    forgetting: list[float] = []
    transfer: list[float] = []
    for block in range(1, array.shape[0]):
        trained_task = (block - 1) % len(TASK_TARGETS)
        before, after = array[block - 1], array[block]
        for task_id in range(len(TASK_TARGETS)):
            if task_id in seen and task_id != trained_task:
                forgetting.append(float(before[task_id] - after[task_id]))
            elif task_id not in seen and task_id != trained_task:
                transfer.append(float(after[task_id] - before[task_id]))
        seen.add(trained_task)
    return (
        continual,
        float(np.mean(forgetting)) if forgetting else 0.0,
        float(np.mean(transfer)) if transfer else 0.0,
    )


def _run_arm(seed: int, steps: int, capacity: int, arm_id: str) -> ArmResult:
    shared = np.zeros((2,), dtype=np.float32)
    task_values = np.zeros((len(TASK_TARGETS), 2), dtype=np.float32)
    replay: list[tuple[int, float]] = []
    updates = inserts = samples = queries = 0
    total_return = 0.0
    matrix: list[tuple[float, ...]] = [_evaluate(arm_id, shared, task_values, 0, seed)]
    queries += len(TASK_TARGETS)
    start = time.perf_counter_ns()
    global_step = 0
    for cycle in range(N_CYCLES):
        for task_id, target in enumerate(TASK_TARGETS):
            for local_step in range(steps):
                tie = (seed + cycle + task_id + local_step) % 2
                if arm_id == "task_id_q_control":
                    action = _greedy(task_values[task_id], tie)
                elif arm_id == "uniform_random":
                    action = int(jr.randint(_rng_key(seed, 2, global_step), (), 0, 2))
                else:
                    action = _greedy(shared, tie)
                queries += 1
                reward = float(action == target)
                total_return += reward
                if arm_id == "task_id_q_control":
                    task_values[task_id] = np.asarray(
                        _q_update(task_values[task_id], action, reward), dtype=np.float32
                    )
                    updates += 1
                elif arm_id in ("replay_q", "replay_off_q"):
                    shared = np.asarray(_q_update(shared, action, reward), dtype=np.float32)
                    updates += 1
                    if global_step > 0:
                        if arm_id == "replay_q":
                            replay_index = int(
                                jr.randint(
                                    _rng_key(seed, 3, global_step), (), 0, len(replay)
                                )
                            )
                            replay_action, replay_reward = replay[replay_index]
                            samples += 1
                        else:
                            replay_action, replay_reward = action, reward
                        shared = np.asarray(
                            _q_update(shared, replay_action, replay_reward), dtype=np.float32
                        )
                        updates += 1
                    if arm_id == "replay_q":
                        replay.append((action, reward))
                        if len(replay) > capacity:
                            replay.pop(0)
                        inserts += 1
                global_step += 1
            matrix.append(_evaluate(arm_id, shared, task_values, len(matrix), seed))
            queries += len(TASK_TARGETS)
    elapsed = time.perf_counter_ns() - start
    frozen_matrix = tuple(matrix)
    continual, forgetting, transfer = _metrics(frozen_matrix)
    train_steps = steps * len(TASK_TARGETS) * N_CYCLES
    eval_steps = len(matrix) * len(TASK_TARGETS)
    persistent = task_values.nbytes if arm_id == "task_id_q_control" else shared.nbytes
    if arm_id == "uniform_random":
        persistent = 8  # one Threefry key payload: uint32[2]
    peak_replay = min(capacity, train_steps) * 8 if arm_id == "replay_q" else 0
    compute = train_steps + eval_steps + queries + updates + inserts + samples
    return ArmResult(
        arm_id=arm_id,
        training_return=float(total_return),
        evaluation_matrix=frozen_matrix,
        continual_evaluation=continual,
        isolated_forgetting=forgetting,
        isolated_forward_transfer=transfer,
        receipt=ResourceReceipt(
            training_environment_steps=train_steps,
            evaluation_environment_steps=eval_steps,
            model_queries=queries,
            agent_updates=updates,
            replay_inserts=inserts,
            replay_samples=samples,
            persistent_bytes=int(persistent),
            peak_replay_bytes=peak_replay,
            logical_compute_units=compute,
            elapsed_ns=elapsed,
        ),
        candidate_eligible=arm_id != "task_id_q_control",
    )


def run_cora_development(
    *, seed: object, steps_per_task: object = 4, replay_capacity: object = 8
) -> CORADevelopmentResult:
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    steps = _exact_int(steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
    capacity = _exact_int(replay_capacity, "replay_capacity", 1, 64)
    result = CORADevelopmentResult(
        schema=SCHEMA,
        paper_revision=PAPER,
        official_code_revision=OFFICIAL_CODE,
        runner_source_sha256=_runner_source_sha256(),
        seed=host_seed,
        steps_per_task=steps,
        replay_capacity=capacity,
        task_targets=TASK_TARGETS,
        cycles=N_CYCLES,
        task_boundaries_available_to_runner=True,
        task_ids_available_to_candidate=False,
        arms=tuple(_run_arm(host_seed, steps, capacity, arm_id) for arm_id in ARM_IDS),
    )
    return validate_result(result)


def validate_result(value: object) -> CORADevelopmentResult:
    if type(value) is not CORADevelopmentResult:
        raise ValueError("result must be an exact CORADevelopmentResult")
    CORADevelopmentResult.__post_init__(value)
    train_steps = value.steps_per_task * len(TASK_TARGETS) * N_CYCLES
    rows = _EVALUATION_MATRIX_ROWS
    eval_steps = rows * len(TASK_TARGETS)
    for arm in value.arms:
        ArmResult.__post_init__(arm)
        if len(arm.evaluation_matrix) != rows:
            raise ValueError("evaluation matrix row count mismatch")
        if type(arm.receipt) is not ResourceReceipt:
            raise ValueError("receipt must be an exact ResourceReceipt")
        ResourceReceipt.__post_init__(arm.receipt)
        updates = 0
        if arm.arm_id == "task_id_q_control":
            updates = train_steps
        elif arm.arm_id in ("replay_q", "replay_off_q"):
            updates = 2 * train_steps - 1
        inserts = train_steps if arm.arm_id == "replay_q" else 0
        samples = train_steps - 1 if arm.arm_id == "replay_q" else 0
        peak = min(value.replay_capacity, train_steps) * 8 if inserts else 0
        persistent = 24 if arm.arm_id == "task_id_q_control" else 8
        if arm.arm_id == "uniform_random":
            persistent = 8
        queries = train_steps + eval_steps
        compute = train_steps + eval_steps + queries + updates + inserts + samples
        receipt = arm.receipt
        observed = (
            receipt.training_environment_steps,
            receipt.evaluation_environment_steps,
            receipt.model_queries,
            receipt.agent_updates,
            receipt.replay_inserts,
            receipt.replay_samples,
            receipt.persistent_bytes,
            receipt.peak_replay_bytes,
            receipt.logical_compute_units,
        )
        expected = (
            train_steps,
            eval_steps,
            queries,
            updates,
            inserts,
            samples,
            persistent,
            peak,
            compute,
        )
        if observed != expected:
            raise ValueError("resource receipt mismatch")
        expected_metrics = _metrics(arm.evaluation_matrix)
        if expected_metrics != (
            arm.continual_evaluation,
            arm.isolated_forgetting,
            arm.isolated_forward_transfer,
        ):
            raise ValueError("metric recomputation mismatch")
        expected_arm = _run_arm(
            value.seed, value.steps_per_task, value.replay_capacity, arm.arm_id
        )
        observed_arm = dataclasses.replace(
            arm, receipt=dataclasses.replace(arm.receipt, elapsed_ns=0)
        )
        expected_arm = dataclasses.replace(
            expected_arm,
            receipt=dataclasses.replace(expected_arm.receipt, elapsed_ns=0),
        )
        if observed_arm != expected_arm:
            raise ValueError("deterministic replay mismatch")
    return value


def catalog_payload() -> dict[str, object]:
    return {
        "schema": "asi.cora_qualification_catalog.v1",
        "paper": PAPER,
        "official_code": OFFICIAL_CODE,
        "external_execution_ready": False,
        "development_only": True,
        "families": [dataclasses.asdict(family) for family in CORA_CATALOG],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the blocked CORA qualification catalog")
    parser.add_argument("--catalog", action="store_true")
    args = parser.parse_args(argv)
    if not args.catalog:
        parser.error("only --catalog is available; external execution is blocked")
    print(json.dumps(catalog_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_IDS", "CORA_CATALOG", "FROZEN_SEEDS", "OFFICIAL_CODE", "PAPER", "SCHEMA",
    "ArmResult", "CORADevelopmentResult", "CORAFamily", "ResourceReceipt",
    "catalog_payload", "main", "run_cora_development", "validate_result",
]
