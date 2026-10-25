from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from alberta_framework.benchmarks.continual_world_qualification import (
    CW20_OBSERVATION_DIM,
    CW20_TASKS,
    METAWORLD_OBSERVATION_DIM,
    SCHEMA,
    ContinualWorldSmokePlan,
    IsolatedRuntimeIdentity,
    build_smoke_receipt,
    protocol_gap_record,
    validate_smoke_payload,
)

pytestmark = pytest.mark.unit


def _official_observations(plan: ContinualWorldSmokePlan) -> np.ndarray:
    horizon = len(CW20_TASKS) * plan.steps_per_task
    indices = np.repeat(
        np.arange(len(CW20_TASKS), dtype=np.int32), plan.steps_per_task
    )
    observations = np.zeros((horizon, CW20_OBSERVATION_DIM), dtype=np.float32)
    observations[np.arange(horizon), METAWORLD_OBSERVATION_DIM + indices] = 1.0
    return observations


@pytest.fixture
def runtime() -> IsolatedRuntimeIdentity:
    return IsolatedRuntimeIdentity(
        image_digest="sha256:" + "1" * 64,
        mujoco_archive_sha256="2" * 64,
        python_version="3.6.9",
        tensorflow_version="2.6.2",
        mujoco_py_version="2.0.2.13",
        gym_version="0.21.0",
        numpy_version="1.19.5",
        platform="linux-x86_64",
    )


@pytest.fixture
def receipt(runtime: IsolatedRuntimeIdentity):
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 20 * plan.steps_per_task
    return build_smoke_receipt(
        plan,
        actions=np.zeros((horizon, 4), dtype=np.float32),
        observations=_official_observations(plan),
        rewards=np.zeros((horizon,), dtype=np.float32),
        successes=np.zeros((horizon,), dtype=np.bool_),
        task_indices=np.repeat(
            np.arange(len(CW20_TASKS), dtype=np.int32), plan.steps_per_task
        ),
        persistent_environment_numeric_bytes=4096,
        timing_ns=10,
        outcome="inconclusive",
    )


def test_protocol_pins_official_sequence_sources_and_nonpromotion(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    payload = plan.payload()
    assert SCHEMA == "asi.continual-world.fixed-action-smoke.v2"
    assert len(CW20_TASKS) == 20
    assert CW20_TASKS[:10] == CW20_TASKS[10:]
    assert payload["paper_revision"] == "arXiv:2105.10919v3"
    assert payload["official_commit"] == "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
    assert payload["metaworld_commit"] == "0875192baaa91c43523708f55866d98eaf3facaf"
    assert payload["learner_boundary_information"] == [
        "observation_tail_one_hot_changes_at_task_boundary"
    ]
    assert payload["learner_task_information"] == [
        "observation_tail_one_hot_cw20_sequence_index"
    ]
    assert payload["scientific_promotion_allowed"] is False


def test_trace_builder_rejects_missing_official_task_one_hot(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = len(CW20_TASKS) * plan.steps_per_task
    with pytest.raises(ValueError, match="one-hot task tail"):
        build_smoke_receipt(
            plan,
            actions=np.zeros((horizon, 4), dtype=np.float32),
            observations=np.zeros((horizon, CW20_OBSERVATION_DIM), dtype=np.float32),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=np.repeat(
                np.arange(len(CW20_TASKS), dtype=np.int32), plan.steps_per_task
            ),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )


def test_fixed_action_receipt_is_exact_mechanism_off_and_round_trips(receipt) -> None:
    checked = validate_smoke_payload(receipt.payload())
    assert checked == receipt
    assert checked.environment_steps == 40
    assert checked.persistent_mechanism_bytes == 16
    assert checked.data_steps == checked.learner_updates == checked.model_queries == 0
    assert checked.mechanism_off is True
    assert checked.negative_outcome_retained is True


def test_trace_builder_snapshots_and_rejects_wrong_boundary_schedule(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 40
    observations = _official_observations(plan)
    task_indices = np.repeat(np.arange(len(CW20_TASKS), dtype=np.int32), 2)
    receipt = build_smoke_receipt(
        plan,
        actions=np.zeros((horizon, 4), dtype=np.float32),
        observations=observations,
        rewards=np.zeros(horizon, dtype=np.float32),
        successes=np.zeros(horizon, dtype=np.bool_),
        task_indices=task_indices,
        persistent_environment_numeric_bytes=1,
        timing_ns=0,
        outcome="rejected",
    )
    observations[0, 0] = 99.0
    checked = validate_smoke_payload(receipt.payload())
    assert receipt.observation_sha256 == checked.observation_sha256
    task_indices[0] = 1
    with pytest.raises(ValueError, match="boundary schedule"):
        build_smoke_receipt(
            plan,
            actions=np.zeros((horizon, 4), dtype=np.float32),
            observations=_official_observations(plan),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=task_indices,
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )
    hostile_actions = np.zeros((horizon, 4), dtype=np.float32)
    hostile_actions[0, 0] = 0.25
    with pytest.raises(ValueError, match="fixed-action"):
        build_smoke_receipt(
            plan,
            actions=hostile_actions,
            observations=_official_observations(plan),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=np.repeat(np.arange(len(CW20_TASKS), dtype=np.int32), 2),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )


def test_validator_rejects_hostile_expansion_resources_and_promotion(receipt) -> None:
    expanded = receipt.payload()
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        validate_smoke_payload(expanded)
    forged = copy.deepcopy(receipt.payload())
    forged["persistent_mechanism_bytes"] = 15
    with pytest.raises(ValueError, match="fixed float32 action"):
        validate_smoke_payload(forged)
    promoted = copy.deepcopy(receipt.payload())
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_smoke_payload(promoted)
    with pytest.raises(ValueError, match="plan_sha256"):
        dataclasses.replace(receipt, plan_sha256="0" * 64)


@pytest.mark.parametrize("field", ["action_sha256", "task_index_sha256"])
def test_validator_rejects_forged_frozen_trace_hashes(receipt, field: str) -> None:
    forged = receipt.payload()
    forged[field] = "0" * 64

    with pytest.raises(ValueError, match=field):
        validate_smoke_payload(forged)


def test_runtime_and_plan_fail_closed(runtime) -> None:
    with pytest.raises(ValueError, match="image_digest"):
        dataclasses.replace(runtime, image_digest="latest")
    with pytest.raises(ValueError, match="development seed"):
        ContinualWorldSmokePlan(runtime=runtime, seed=0)
    assert len(protocol_gap_record()) >= 8
