from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import alberta_framework.benchmarks.continual_world_qualification as qualification
from alberta_framework.benchmarks.continual_world_qualification import (
    CW20_TASKS,
    ContinualWorldSmokePlan,
    IsolatedRuntimeIdentity,
    build_smoke_receipt,
    protocol_gap_record,
    validate_smoke_payload,
)

pytestmark = pytest.mark.unit


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
def receipt(runtime: IsolatedRuntimeIdentity, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qualification, "LICENSE_DISPOSITION_APPROVED", True)
    monkeypatch.setattr(qualification, "EXTERNAL_EXECUTION_AUTHORIZED", True)
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 20 * plan.steps_per_task
    return build_smoke_receipt(
        plan,
        actions=np.zeros((horizon, 4), dtype=np.float32),
        observations=np.zeros((horizon, 32), dtype=np.float32),
        rewards=np.zeros((horizon,), dtype=np.float32),
        successes=np.zeros((horizon,), dtype=np.bool_),
        task_indices=np.repeat(np.arange(20, dtype=np.int32), plan.steps_per_task),
        persistent_environment_numeric_bytes=4096,
        timing_ns=10,
        outcome="inconclusive",
    )


def test_protocol_pins_official_sequence_sources_and_nonpromotion(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    payload = plan.payload()
    assert len(CW20_TASKS) == 20
    assert CW20_TASKS[:10] == CW20_TASKS[10:]
    assert payload["paper_revision"] == "arXiv:2105.10919v3"
    assert payload["official_commit"] == "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
    assert payload["metaworld_commit"] == "0875192baaa91c43523708f55866d98eaf3facaf"
    assert payload["source_audit"] == {
        "official_tree_sha256": "514f88bbf29ad64a4fbbc1bc403b6a4eca540e9f",
        "official_archive_sha256": (
            "21ed7d404975be7ca12fbb315eeece14f25cc0580dc6b074a81eac791a2d03d9"
        ),
        "official_license_status": "blocked-no-license-file",
        "metaworld_tree_sha256": "3024851b45599fe678718b92156d6e004c17039c",
        "metaworld_archive_sha256": (
            "fa1f0336719ef8110c7c22c411ace52c7936deacddce0a1f011ebde3989ec5a5"
        ),
        "metaworld_license_sha256": (
            "9d4c6640ecd8cb9e3fe55eb923517fb75a241b74949817121399260c8f549243"
        ),
        "mujoco_archive_sha256": "ba8560040f6ca47dbd89e4731bc9e06080a99eba4583cda95cdedca802389153",
        "mujoco_key_sha256": "bffe403bce6978d329239c83e874e0fd412740d149834b8c051689ba4a9adecc",
        "official_runtime_reconstructible": False,
    }
    assert payload["license_disposition_approved"] is False
    assert payload["external_execution_authorized"] is False
    assert payload["learner_boundary_information"] == []
    assert payload["scientific_promotion_allowed"] is False


def test_fixed_action_receipt_is_exact_mechanism_off_and_round_trips(receipt) -> None:
    checked = validate_smoke_payload(receipt.payload())
    assert checked == receipt
    assert checked.environment_steps == 40
    assert checked.persistent_mechanism_bytes == 16
    assert checked.data_steps == checked.learner_updates == checked.model_queries == 0
    assert checked.mechanism_off is True
    assert checked.negative_outcome_retained is True


def test_trace_builder_snapshots_and_rejects_wrong_boundary_schedule(
    runtime: IsolatedRuntimeIdentity, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "LICENSE_DISPOSITION_APPROVED", True)
    monkeypatch.setattr(qualification, "EXTERNAL_EXECUTION_AUTHORIZED", True)
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 40
    observations = np.zeros((horizon, 32), dtype=np.float32)
    task_indices = np.repeat(np.arange(20, dtype=np.int32), 2)
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
            observations=np.zeros((horizon, 32), dtype=np.float32),
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
            observations=np.zeros((horizon, 32), dtype=np.float32),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=np.repeat(np.arange(20, dtype=np.int32), 2),
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


def test_runtime_and_plan_fail_closed(runtime) -> None:
    with pytest.raises(ValueError, match="image_digest"):
        dataclasses.replace(runtime, image_digest="latest")
    with pytest.raises(ValueError, match="development seed"):
        ContinualWorldSmokePlan(runtime=runtime, seed=0)
    assert len(protocol_gap_record()) >= 8


def test_smoke_execution_fails_before_trace_access_while_license_is_blocked(
    runtime: IsolatedRuntimeIdentity,
) -> None:
    class HostileArray:
        calls = 0

        def __getattribute__(self, name: str) -> object:
            if name != "calls":
                type(self).calls += 1
                raise AssertionError("trace must not be inspected before authorization")
            return object.__getattribute__(self, name)

    hostile = HostileArray()
    with pytest.raises(PermissionError, match="license.*authorization"):
        build_smoke_receipt(
            ContinualWorldSmokePlan(runtime=runtime),
            actions=hostile,  # type: ignore[arg-type]
            observations=hostile,  # type: ignore[arg-type]
            rewards=hostile,  # type: ignore[arg-type]
            successes=hostile,  # type: ignore[arg-type]
            task_indices=hostile,  # type: ignore[arg-type]
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="inconclusive",
        )
    assert HostileArray.calls == 0
