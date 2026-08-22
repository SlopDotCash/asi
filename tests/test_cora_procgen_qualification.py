"""Fail-closed contracts for the proposed isolated CORA Procgen runtime."""

from __future__ import annotations

import dataclasses

import pytest

from alberta_framework.benchmarks.cora_development import catalog_payload
from alberta_framework.benchmarks.cora_procgen_qualification import (
    CORA_PROCGEN_TASKS,
    ProcgenCreateReceipt,
    ProcgenQualificationPlan,
    plan_sha256,
    validate_create_receipt,
)

pytestmark = pytest.mark.unit


def test_plan_binds_source_wheel_assets_runtime_and_cora_protocol() -> None:
    plan = ProcgenQualificationPlan()
    plan.validate()
    assert plan.cora_tree == "3c296057e717401053ce0acfe362adeef395aede"
    assert plan.procgen_commit == "5e1dbf341d291eff40d1f9e0c0a0d5003643aebf"
    assert plan.procgen_wheel_sha256 == (
        "4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53"
    )
    assert plan.python_version == "3.10"
    assert plan.platform_tag == "manylinux2014_x86_64"
    assert plan.tasks == CORA_PROCGEN_TASKS
    assert plan.train_levels == 200
    assert plan.eval_levels == 0
    assert plan.execution_authorized is False
    assert plan.runtime_qualified is False
    catalog = catalog_payload()
    assert catalog["schema"] == "asi.cora_qualification_catalog.v2"
    procgen = catalog["procgen_qualification"]
    assert isinstance(procgen, dict)
    assert procgen["plan_sha256"] == plan_sha256()


def test_seed_semantics_distinguish_official_randomness_from_compatibility_smoke() -> None:
    plan = ProcgenQualificationPlan()
    assert plan.official_seed_semantics == "os.urandom-derived seed_to_set"
    assert plan.compatibility_seed_semantics == "caller-injected exact uint32 seed_to_set"
    assert plan.compatibility_smoke_has_official_seed_parity is False


def test_future_success_receipt_is_create_only_and_content_bound() -> None:
    receipt = ProcgenCreateReceipt(
        plan_sha256=plan_sha256(),
        outcome="success",
        task_id=CORA_PROCGEN_TASKS[0],
        phase="train",
        seed_mode="deterministic_compatibility",
        injected_seed=1581000,
        observation_sha256="1" * 64,
        environment_instances_created=1,
        elapsed_ns=10,
    )
    assert validate_create_receipt(receipt) is receipt
    assert receipt.environment_steps == 0
    assert receipt.policy_queries == 0
    assert receipt.reward_payload is None
    assert receipt.termination_payload is None


def test_future_failure_receipt_retains_bounded_failure_without_fake_payload() -> None:
    receipt = ProcgenCreateReceipt(
        plan_sha256=plan_sha256(),
        outcome="failure",
        task_id=CORA_PROCGEN_TASKS[1],
        phase="evaluation",
        seed_mode="official_os_random",
        failure_kind="RuntimeError",
        elapsed_ns=10,
    )
    assert validate_create_receipt(receipt) is receipt
    with pytest.raises(ValueError, match="success receipt"):
        validate_create_receipt(
            dataclasses.replace(receipt, outcome="success", observation_sha256="2" * 64)
        )


def test_receipt_rejects_execution_or_identity_forgery() -> None:
    receipt = ProcgenCreateReceipt(
        plan_sha256=plan_sha256(),
        outcome="failure",
        task_id=CORA_PROCGEN_TASKS[0],
        phase="train",
        seed_mode="official_os_random",
        failure_kind="ImportError",
        elapsed_ns=0,
    )
    with pytest.raises(ValueError, match="plan identity"):
        validate_create_receipt(dataclasses.replace(receipt, plan_sha256="0" * 64))
    with pytest.raises(ValueError, match="create-only"):
        validate_create_receipt(dataclasses.replace(receipt, environment_steps=1))
    with pytest.raises(ValueError, match="seed mode"):
        validate_create_receipt(dataclasses.replace(receipt, injected_seed=7))
    with pytest.raises(ValueError, match="signed-int64"):
        validate_create_receipt(dataclasses.replace(receipt, elapsed_ns=2**80))
    with pytest.raises(ValueError, match="failure kind"):
        validate_create_receipt(dataclasses.replace(receipt, failure_kind="\ud800"))
