"""Cheap contracts for the private routed-owner operational runner."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from typing import Any, cast

import jax.numpy as jnp
import pytest

import alberta_framework.core.hccl_routed_continual_dyad_operational_runner as operational
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
)
from alberta_framework.core.hccl_routed_continual_dyad import (
    HCCLRoutedContinualDyadWork,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


def _owner_work() -> HCCLRoutedContinualDyadWork:
    return HCCLRoutedContinualDyadWork(
        hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
        world_proposal_calls=jnp.asarray(8, dtype=jnp.int32),
        attribution_proposal_calls=jnp.asarray(8, dtype=jnp.int32),
        context_steps=jnp.ones((2,), dtype=jnp.int32),
        coordinator_steps=jnp.ones((2,), dtype=jnp.int32),
        lifecycle_route_derivations=jnp.ones((2,), dtype=jnp.int32),
        memory_settlements=jnp.asarray((0, 1), dtype=jnp.int32),
        memory_steps=jnp.ones((2,), dtype=jnp.int32),
        memory_rebinds=jnp.ones((2,), dtype=jnp.int32),
        planner_behavior_updates=jnp.asarray(2, dtype=jnp.int32),
        planner_grounded_updates=jnp.asarray(2, dtype=jnp.int32),
        planner_joint_cells=jnp.asarray(8, dtype=jnp.int32),
        bmp_memory_replacements=jnp.ones((2,), dtype=jnp.int32),
        bmp_planner_replacements=jnp.ones((2,), dtype=jnp.int32),
        outer_commit_decisions=jnp.asarray(1, dtype=jnp.int32),
        output_writes=jnp.asarray(0, dtype=jnp.int32),
        rng_draws_after_event=jnp.asarray(0, dtype=jnp.int32),
    )


def test_routed_operational_schemas_and_nonclaims_are_explicit() -> None:
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL == "L0"
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA.endswith(
        "operational-work.v1"
    )
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA.endswith(
        "operational-transcript.v1"
    )
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA.endswith(
        "operational-result.v1"
    )
    limitations = operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS
    assert "full-routed-mechanism-only-no-arm-selection" in limitations
    assert "outer-audit-work-equivalence-is-explicitly-not-targeted" in limitations
    assert "awaits-reference-differential" in limitations
    assert "final-full-state-validation-is-mandatory" in limitations


def test_routed_preparation_flag_surface_is_complete_and_fixed() -> None:
    assert operational._PREPARED_SCALAR_FLAGS == (
        "source_state_valid",
        "event_valid",
        "action_bundle_valid",
        "hccl_valid",
        "credit_valid",
        "planner_valid",
        "candidate_state_valid",
        "preparation_valid",
    )
    assert operational._PREPARED_VECTOR_FLAGS == (
        "context_valid",
        "coordinator_valid",
        "lifecycle_route_valid",
        "memory_valid",
        "bmp_valid",
    )


def test_routed_operational_work_counts_public_validation_and_skipped_outer_audit() -> None:
    work = operational._make_operational_work(
        prepared_owner_work=_owner_work(),
        runner_checkpoint_state_validations=1,
    )
    assert work.public_event_preparation_calls == 1
    assert work.public_action_binding_calls == 1
    assert work.public_transaction_preparation_calls == 1
    assert work.routed_owner_full_state_validations == 4
    assert work.routed_owner_event_receipt_validations == 2
    assert work.routed_owner_action_bundle_validations == 1
    assert work.action_bundle_validation_reconstructions == 1
    assert work.hccl_action_receipt_bindings == 6
    assert work.candidate_state_seals == 1
    assert work.prepared_transaction_seals == 1
    assert work.inner_bmp_integrity_receipts == (1, 1)
    assert work.inner_bmp_adoptions == (1, 1)
    assert work.prepared_scalar_flag_fields_checked == 8
    assert work.prepared_vector_flag_fields_checked == 5
    assert work.prepared_child_flags_checked == (1, 1)
    assert work.operational_publication_decisions == 1
    assert work.runner_checkpoint_state_validations == 1
    assert work.outer_integrity_receipts == 0
    assert work.outer_prepared_semantic_reconstructions == 0
    assert work.outer_adoption_calls == 0
    assert work.outer_candidate_state_revalidations == 0
    assert work.nested_child_validation_calls_exhaustively_counted is False
    assert work.outer_audit_work_equivalence_targeted is False
    assert work.state_and_transcript_bit_equivalence_targeted is True
    assert work.prepared_owner_work is not None
    payload = work.to_config()
    assert payload["schema"] == operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA
    assert payload["prepared_owner_work"]["memory_settlements"] == [0, 1]

    with pytest.raises(TypeError, match="checkpoint"):
        operational._make_operational_work(
            prepared_owner_work=_owner_work(),
            runner_checkpoint_state_validations=cast(int, True),
        )


def test_available_full_path_mechanism_diagnostics_are_narrow_and_do_not_select_arm() -> None:
    names = tuple(
        field.name
        for field in dataclasses.fields(
            operational.HCCLRoutedContinualDyadOperationalMechanismDiagnostics
        )
    )
    assert names == (
        "schema",
        "coordinator_inner_transaction_applied",
        "coordinator_builder_learning_applied",
        "coordinator_candidate_audit_accepted",
        "context_allocation_requested",
        "context_full_bank_eviction_requested",
        "context_eviction_target_adjusted",
        "lineage_transferred",
        "rescue_incremented",
        "lifecycle_committed",
        "lifecycle_selected_active_slots",
        "lifecycle_selected_candidate_slots",
        "pair_admission_masks",
        "memory_settlements_performed",
        "memory_retrieval_categorical",
        "memory_consumed",
        "memory_proposed_actions",
        "planner_partner_belief",
        "planner_partner_belief_by_own_action",
        "planner_world_cell_valid",
        "planner_expected_net_rewards",
        "planner_proposed_actions",
        "bmp_final_actions",
    )
    assert all("arm" not in name for name in names)
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_DIAGNOSTIC_COVERAGE == (
        "coordinator-full-path-commit-flags",
        "context-and-lineage-full-path-events",
        "lifecycle-full-path-selection-and-admission",
        "memory-full-path-settlement-retrieval-and-dispatch",
        "planner-v2-full-path-beliefs-cells-rewards-and-dispatch",
        "bmp-full-path-final-actions",
    )
    assert operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_ABSENT_ARM_ALTERNATIVES == (
        "fast-state-unrouted-execution-receipt",
        "slow-context-unrouted-execution-receipt",
        "lineage-rescue-unrouted-execution-receipt",
        "random-feature-rank-execution-receipt",
        "unrouted-feature-consumer-execution-receipt",
        "base-memory-dispatch-execution-receipt",
        "uniform-partner-belief-execution-receipt",
        "memory-fallback-planner-dispatch-execution-receipt",
    )


def test_transcript_and_result_are_compact_and_evaluator_sufficient() -> None:
    assert tuple(
        field.name
        for field in dataclasses.fields(
            operational.HCCLRoutedContinualDyadOperationalTranscript
        )
    ) == (
        "schema",
        "event",
        "action_bundle",
        "source_state_content_token",
        "pp_proposal",
        "pre_transaction_words",
        "post_transaction_words",
        "mechanism_diagnostics",
    )
    assert tuple(
        field.name
        for field in dataclasses.fields(
            operational.HCCLRoutedContinualDyadOperationalEventResult
        )
    ) == (
        "schema",
        "state",
        "transcript",
        "work",
        "update_applied",
    )
    assert HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER[operational._PP_SLOT] == "PP-planner"
    assert (
        '_PP_SLOT = HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER.index("PP-planner")'
        in inspect.getsource(operational)
    )


def _call_names(function: Any) -> list[str]:
    tree = ast.parse(inspect.getsource(function))
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]


def test_routed_kernel_uses_only_the_approved_public_preparation_boundary() -> None:
    calls = _call_names(operational._execute_routed_operational_event)
    for expected in (
        "owner.prepare_event",
        "owner.bind_actions",
        "owner.prepare_transaction",
    ):
        assert calls.count(expected) == 1
    forbidden = {
        "owner.step",
        "owner.integrity_receipt",
        "owner._prepared_semantics_valid",
        "owner.adopt",
        "owner.state_valid",
    }
    assert forbidden.isdisjoint(calls)
    source = inspect.getsource(operational._execute_routed_operational_event)
    assert "prepared.event is not event" in source
    assert "prepared.action_bundle is not action_bundle" in source
    assert "prepared.source_state is not state" in source


def _raw_result(state: object) -> operational.HCCLRoutedContinualDyadOperationalEventResult:
    result = object.__new__(operational.HCCLRoutedContinualDyadOperationalEventResult)
    object.__setattr__(
        result,
        "schema",
        operational.HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA,
    )
    object.__setattr__(result, "state", state)
    object.__setattr__(result, "transcript", object())
    object.__setattr__(result, "work", object())
    object.__setattr__(result, "update_applied", True)
    return result


def _raw_executor(
    *,
    owner: object,
    state: object,
    checkpoint_interval: int | None,
) -> operational._HCCLRoutedContinualDyadOperationalExecutor:
    executor = object.__new__(operational._HCCLRoutedContinualDyadOperationalExecutor)
    object.__setattr__(executor, "_owner", owner)
    object.__setattr__(executor, "_state", state)
    object.__setattr__(executor, "_absolute_step", 0)
    object.__setattr__(executor, "_maximum_transitions", 10)
    object.__setattr__(executor, "_checkpoint_interval", checkpoint_interval)
    return executor


def test_routed_executor_does_not_publish_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = object()
    executor = _raw_executor(owner=object(), state=source, checkpoint_interval=None)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("routed preparation rejected")

    monkeypatch.setattr(operational, "_execute_routed_operational_event", fail)
    with pytest.raises(
        operational.HCCLRoutedContinualDyadOperationalError,
        match="routed preparation rejected",
    ):
        executor.step(cast(Any, object()))
    assert executor.state is source
    assert executor.absolute_step == 0


@pytest.mark.parametrize(
    ("absolute_step", "checkpoint_interval", "stage"),
    ((0, 1, "periodic-checkpoint"), (9, None, "final-checkpoint")),
)
def test_routed_executor_checks_candidate_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    absolute_step: int,
    checkpoint_interval: int | None,
    stage: str,
) -> None:
    class _RejectingOwner:
        @staticmethod
        def state_valid(_state: object) -> bool:
            return False

    source = object()
    candidate = object()
    executor = _raw_executor(
        owner=_RejectingOwner(),
        state=source,
        checkpoint_interval=checkpoint_interval,
    )
    object.__setattr__(executor, "_absolute_step", absolute_step)
    monkeypatch.setattr(
        operational,
        "_execute_routed_operational_event",
        lambda *_args, **_kwargs: _raw_result(candidate),
    )
    with pytest.raises(operational.HCCLRoutedContinualDyadOperationalError, match=stage):
        executor.step(cast(Any, object()))
    assert executor.state is source
    assert executor.absolute_step == absolute_step
