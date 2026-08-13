# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,union-attr"
"""Private compact publication path for the routed HCCL continual dyad.

The current routed owner already evaluates the complete learning transaction
inside ``prepare_transaction`` and validates its candidate.  This L0 runner
seam therefore uses the smallest safe operational boundary: public event
preparation, public B/M/P binding, and public transaction preparation.  It
requires every locally produced preparation flag, records a compact PP and
full-path mechanism transcript, then publishes the prepared candidate without
the expensive *outer* receipt, semantic reconstruction, or adoption replay.

Child BMP integrity receipts and adoptions remain part of routed preparation
and are counted truthfully.  The transcript is private and unauthenticated. It
does not select an ablation arm, execute missing arm alternatives, authorize a
benchmark, write an artifact, set a threshold, or support evidence promotion.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import numpy as np
from jax import Array

from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
)
from alberta_framework.core.hccl_routed_continual_dyad import (
    HCCLRoutedContinualDyad,
    HCCLRoutedContinualDyadActionBundle,
    HCCLRoutedContinualDyadPreparedTransaction,
    HCCLRoutedContinualDyadState,
    HCCLRoutedContinualDyadWork,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.operational-work.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_MECHANISM_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.operational-mechanism.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.operational-transcript.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.operational-result.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_STATUS = (
    "l0-development-private-routed-owner-operational-runner"
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL = "L0"
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS = (
    "private-runner-only-not-a-public-transaction-api",
    "full-routed-mechanism-only-no-arm-selection",
    "arm-alternative-execution-receipts-are-absent",
    "outer-audit-work-equivalence-is-explicitly-not-targeted",
    "outer-fault-injection-must-use-the-routed-owner-reference-api",
    "awaits-reference-differential",
    "periodic-full-state-validation-is-runner-configured",
    "final-full-state-validation-is-mandatory",
    "transcript-is-private-unauthenticated-and-nonauthorizing",
    "host-eager-only",
    "no-output-artifact-threshold-seed-benchmark-evidence-or-promotion-authority",
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_DIAGNOSTIC_COVERAGE = (
    "coordinator-full-path-commit-flags",
    "context-and-lineage-full-path-events",
    "lifecycle-full-path-selection-and-admission",
    "memory-full-path-settlement-retrieval-and-dispatch",
    "planner-v2-full-path-beliefs-cells-rewards-and-dispatch",
    "bmp-full-path-final-actions",
)
HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_ABSENT_ARM_ALTERNATIVES = (
    "fast-state-unrouted-execution-receipt",
    "slow-context-unrouted-execution-receipt",
    "lineage-rescue-unrouted-execution-receipt",
    "random-feature-rank-execution-receipt",
    "unrouted-feature-consumer-execution-receipt",
    "base-memory-dispatch-execution-receipt",
    "uniform-partner-belief-execution-receipt",
    "memory-fallback-planner-dispatch-execution-receipt",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_PAIR_SLOTS = 12
_TOKEN_NBYTES = 32
_PP_SLOT = HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER.index("PP-planner")

_PREPARED_SCALAR_FLAGS = (
    "source_state_valid",
    "event_valid",
    "action_bundle_valid",
    "hccl_valid",
    "credit_valid",
    "planner_valid",
    "candidate_state_valid",
    "preparation_valid",
)
_PREPARED_VECTOR_FLAGS = (
    "context_valid",
    "coordinator_valid",
    "lifecycle_route_valid",
    "memory_valid",
    "bmp_valid",
)


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _exact_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != _N_AGENTS:
        raise TypeError(f"{name} must be an exact two-int tuple")
    pair = cast(tuple[object, object], value)
    return (
        _exact_int(pair[0], name=f"{name}[0]"),
        _exact_int(pair[1], name=f"{name}[1]"),
    )


def _host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a concrete non-key array") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return array


def _host_true(value: object, *, name: str) -> bool:
    array = _host_array(
        value,
        name=name,
        shape=(),
        dtype=np.dtype(np.bool_),
    )
    if not bool(array):
        raise ValueError(f"{name} must be exact True")
    return True


def _host_all_true(value: object, *, name: str) -> bool:
    array = _host_array(
        value,
        name=name,
        shape=(_N_AGENTS,),
        dtype=np.dtype(np.bool_),
    )
    if not bool(np.all(array)):
        raise ValueError(f"{name} must be all true")
    return True


def _frozen_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray[Any, Any]:
    result = np.array(
        _host_array(value, name=name, shape=shape, dtype=dtype),
        copy=True,
    )
    result.setflags(write=False)
    return result


def _increment_clock_words(words: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    high = int(words[0])
    low = int(words[1])
    if high == 2**32 - 1 and low == 2**32 - 1:
        raise OverflowError("routed operational clock exhausted uint64 capacity")
    low += 1
    if low == 2**32:
        high += 1
        low = 0
    return np.asarray((high, low), dtype=np.uint32)


def _owner_work_config(work: HCCLRoutedContinualDyadWork) -> dict[str, object]:
    if type(work) is not HCCLRoutedContinualDyadWork:
        raise TypeError("prepared_owner_work must be exact routed owner work")
    vector_fields = {
        "context_steps",
        "coordinator_steps",
        "lifecycle_route_derivations",
        "memory_settlements",
        "memory_steps",
        "memory_rebinds",
        "bmp_memory_replacements",
        "bmp_planner_replacements",
    }
    result: dict[str, object] = {}
    for field in dataclasses.fields(work):
        name = field.name
        array = _host_array(
            getattr(work, name),
            name=f"prepared_owner_work.{name}",
            shape=(_N_AGENTS,) if name in vector_fields else (),
            dtype=np.dtype(np.int32),
        )
        result[name] = [int(item) for item in array] if name in vector_fields else int(array)
    return result


def _validate_owner_work(work: HCCLRoutedContinualDyadWork) -> None:
    payload = _owner_work_config(work)
    exact_scalars = {
        "hccl_stage_calls": 1,
        "world_proposal_calls": 8,
        "attribution_proposal_calls": 8,
        "planner_behavior_updates": 2,
        "planner_grounded_updates": 2,
        "planner_joint_cells": 8,
        "outer_commit_decisions": 1,
        "output_writes": 0,
        "rng_draws_after_event": 0,
    }
    for name, expected in exact_scalars.items():
        if payload[name] != expected:
            raise ValueError(f"prepared_owner_work.{name} must equal {expected}")
    for name in (
        "context_steps",
        "coordinator_steps",
        "lifecycle_route_derivations",
        "memory_steps",
        "memory_rebinds",
        "bmp_memory_replacements",
        "bmp_planner_replacements",
    ):
        if payload[name] != [1, 1]:
            raise ValueError(f"prepared_owner_work.{name} must equal [1, 1]")
    settlements = cast(list[int], payload["memory_settlements"])
    if any(value not in (0, 1) for value in settlements):
        raise ValueError("prepared_owner_work.memory_settlements must be binary")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadOperationalWork:
    """Named boundary work for one locally prepared routed event."""

    schema: str
    public_event_preparation_calls: int
    public_action_binding_calls: int
    public_transaction_preparation_calls: int
    routed_owner_full_state_validations: int
    routed_owner_event_receipt_validations: int
    routed_owner_action_bundle_validations: int
    action_bundle_validation_reconstructions: int
    hccl_action_receipt_bindings: int
    candidate_state_seals: int
    prepared_transaction_seals: int
    inner_bmp_integrity_receipts: tuple[int, int]
    inner_bmp_adoptions: tuple[int, int]
    prepared_scalar_flag_fields_checked: int
    prepared_vector_flag_fields_checked: int
    prepared_child_flags_checked: tuple[int, int]
    operational_publication_decisions: int
    runner_checkpoint_state_validations: int
    outer_integrity_receipts: int
    outer_prepared_semantic_reconstructions: int
    outer_adoption_calls: int
    outer_candidate_state_revalidations: int
    nested_child_validation_calls_exhaustively_counted: bool
    outer_audit_work_equivalence_targeted: bool
    state_and_transcript_bit_equivalence_targeted: bool
    prepared_owner_work: HCCLRoutedContinualDyadWork

    def __post_init__(self) -> None:
        if self.schema != HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA:
            raise ValueError("routed operational work schema is unsupported")
        pair_names = (
            "inner_bmp_integrity_receipts",
            "inner_bmp_adoptions",
            "prepared_child_flags_checked",
        )
        for field in dataclasses.fields(self):
            name = field.name
            if name in pair_names:
                _exact_pair(getattr(self, name), name=f"work.{name}")
            elif name not in {
                "schema",
                "nested_child_validation_calls_exhaustively_counted",
                "outer_audit_work_equivalence_targeted",
                "state_and_transcript_bit_equivalence_targeted",
                "prepared_owner_work",
            }:
                _exact_int(getattr(self, name), name=f"work.{name}")
        expected: dict[str, object] = {
            "public_event_preparation_calls": 1,
            "public_action_binding_calls": 1,
            "public_transaction_preparation_calls": 1,
            "routed_owner_full_state_validations": 4,
            "routed_owner_event_receipt_validations": 2,
            "routed_owner_action_bundle_validations": 1,
            "action_bundle_validation_reconstructions": 1,
            "hccl_action_receipt_bindings": 6,
            "candidate_state_seals": 1,
            "prepared_transaction_seals": 1,
            "inner_bmp_integrity_receipts": (1, 1),
            "inner_bmp_adoptions": (1, 1),
            "prepared_scalar_flag_fields_checked": len(_PREPARED_SCALAR_FLAGS),
            "prepared_vector_flag_fields_checked": len(_PREPARED_VECTOR_FLAGS),
            "prepared_child_flags_checked": (1, 1),
            "operational_publication_decisions": 1,
            "outer_integrity_receipts": 0,
            "outer_prepared_semantic_reconstructions": 0,
            "outer_adoption_calls": 0,
            "outer_candidate_state_revalidations": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"routed operational work.{name} must equal {value!r}")
        if self.runner_checkpoint_state_validations not in (0, 1):
            raise ValueError("routed operational checkpoint count must be zero or one")
        for name, expected_bool in (
            ("nested_child_validation_calls_exhaustively_counted", False),
            ("outer_audit_work_equivalence_targeted", False),
            ("state_and_transcript_bit_equivalence_targeted", True),
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"work.{name} must be an exact bool")
            if value is not expected_bool:
                raise ValueError(f"work.{name} must be {expected_bool}")
        _validate_owner_work(self.prepared_owner_work)

    def to_config(self) -> dict[str, object]:
        return {
            field.name: (
                _owner_work_config(self.prepared_owner_work)
                if field.name == "prepared_owner_work"
                else getattr(self, field.name)
            )
            for field in dataclasses.fields(self)
        }


def _make_operational_work(
    *,
    prepared_owner_work: HCCLRoutedContinualDyadWork,
    runner_checkpoint_state_validations: int,
) -> HCCLRoutedContinualDyadOperationalWork:
    checkpoints = _exact_int(
        runner_checkpoint_state_validations,
        name="checkpoint validation count",
    )
    return HCCLRoutedContinualDyadOperationalWork(
        schema=HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA,
        public_event_preparation_calls=1,
        public_action_binding_calls=1,
        public_transaction_preparation_calls=1,
        routed_owner_full_state_validations=4,
        routed_owner_event_receipt_validations=2,
        routed_owner_action_bundle_validations=1,
        action_bundle_validation_reconstructions=1,
        hccl_action_receipt_bindings=6,
        candidate_state_seals=1,
        prepared_transaction_seals=1,
        inner_bmp_integrity_receipts=(1, 1),
        inner_bmp_adoptions=(1, 1),
        prepared_scalar_flag_fields_checked=len(_PREPARED_SCALAR_FLAGS),
        prepared_vector_flag_fields_checked=len(_PREPARED_VECTOR_FLAGS),
        prepared_child_flags_checked=(1, 1),
        operational_publication_decisions=1,
        runner_checkpoint_state_validations=checkpoints,
        outer_integrity_receipts=0,
        outer_prepared_semantic_reconstructions=0,
        outer_adoption_calls=0,
        outer_candidate_state_revalidations=0,
        nested_child_validation_calls_exhaustively_counted=False,
        outer_audit_work_equivalence_targeted=False,
        state_and_transcript_bit_equivalence_targeted=True,
        prepared_owner_work=prepared_owner_work,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadOperationalMechanismDiagnostics:
    """Available diagnostics from the current full routed path only."""

    schema: str
    coordinator_inner_transaction_applied: np.ndarray[Any, Any]
    coordinator_builder_learning_applied: np.ndarray[Any, Any]
    coordinator_candidate_audit_accepted: np.ndarray[Any, Any]
    context_allocation_requested: np.ndarray[Any, Any]
    context_full_bank_eviction_requested: np.ndarray[Any, Any]
    context_eviction_target_adjusted: np.ndarray[Any, Any]
    lineage_transferred: np.ndarray[Any, Any]
    rescue_incremented: np.ndarray[Any, Any]
    lifecycle_committed: np.ndarray[Any, Any]
    lifecycle_selected_active_slots: np.ndarray[Any, Any]
    lifecycle_selected_candidate_slots: np.ndarray[Any, Any]
    pair_admission_masks: np.ndarray[Any, Any]
    memory_settlements_performed: np.ndarray[Any, Any]
    memory_retrieval_categorical: np.ndarray[Any, Any]
    memory_consumed: np.ndarray[Any, Any]
    memory_proposed_actions: np.ndarray[Any, Any]
    planner_partner_belief: np.ndarray[Any, Any]
    planner_partner_belief_by_own_action: np.ndarray[Any, Any]
    planner_world_cell_valid: np.ndarray[Any, Any]
    planner_expected_net_rewards: np.ndarray[Any, Any]
    planner_proposed_actions: np.ndarray[Any, Any]
    bmp_final_actions: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        if self.schema != HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_MECHANISM_SCHEMA:
            raise ValueError("routed operational mechanism schema is unsupported")
        bool_pairs = (
            "coordinator_inner_transaction_applied",
            "coordinator_builder_learning_applied",
            "coordinator_candidate_audit_accepted",
            "context_allocation_requested",
            "context_full_bank_eviction_requested",
            "context_eviction_target_adjusted",
            "lineage_transferred",
            "rescue_incremented",
            "lifecycle_committed",
            "memory_settlements_performed",
            "memory_retrieval_categorical",
            "memory_consumed",
        )
        for name in bool_pairs:
            object.__setattr__(
                self,
                name,
                _frozen_array(
                    getattr(self, name),
                    name=f"mechanism.{name}",
                    shape=(_N_AGENTS,),
                    dtype=np.dtype(np.bool_),
                ),
            )
        for name in (
            "lifecycle_selected_active_slots",
            "lifecycle_selected_candidate_slots",
            "memory_proposed_actions",
            "planner_proposed_actions",
            "bmp_final_actions",
        ):
            object.__setattr__(
                self,
                name,
                _frozen_array(
                    getattr(self, name),
                    name=f"mechanism.{name}",
                    shape=(_N_AGENTS,),
                    dtype=np.dtype(np.int32),
                ),
            )
        object.__setattr__(
            self,
            "pair_admission_masks",
            _frozen_array(
                self.pair_admission_masks,
                name="mechanism.pair_admission_masks",
                shape=(_N_AGENTS, _PAIR_SLOTS),
                dtype=np.dtype(np.bool_),
            ),
        )
        for name, shape in (
            ("planner_partner_belief", (_N_AGENTS, _N_ACTIONS)),
            (
                "planner_partner_belief_by_own_action",
                (_N_AGENTS, _N_ACTIONS, _N_ACTIONS),
            ),
            ("planner_expected_net_rewards", (_N_AGENTS, _N_ACTIONS)),
        ):
            array = _frozen_array(
                getattr(self, name),
                name=f"mechanism.{name}",
                shape=shape,
                dtype=np.dtype(np.float32),
            )
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"mechanism.{name} must be finite")
            object.__setattr__(self, name, array)
        object.__setattr__(
            self,
            "planner_world_cell_valid",
            _frozen_array(
                self.planner_world_cell_valid,
                name="mechanism.planner_world_cell_valid",
                shape=(_N_AGENTS, _N_ACTIONS, _N_ACTIONS),
                dtype=np.dtype(np.bool_),
            ),
        )
        for name in ("memory_proposed_actions", "planner_proposed_actions", "bmp_final_actions"):
            actions = cast(np.ndarray[Any, Any], getattr(self, name))
            if not bool(np.all((actions >= 0) & (actions < _N_ACTIONS))):
                raise ValueError(f"mechanism.{name} lies outside the primitive action domain")


def _mechanism_diagnostics(
    prepared: HCCLRoutedContinualDyadPreparedTransaction,
) -> HCCLRoutedContinualDyadOperationalMechanismDiagnostics:
    agents = (prepared.agent_0, prepared.agent_1)
    planner_plan = prepared.planner_result.receipt.plan
    return HCCLRoutedContinualDyadOperationalMechanismDiagnostics(
        schema=HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_MECHANISM_SCHEMA,
        coordinator_inner_transaction_applied=np.asarray(
            tuple(
                bool(_host_array(
                    agent.coordinator_result.diagnostics.inner_transaction_applied,
                    name=f"agent[{index}].inner_transaction_applied",
                    shape=(),
                    dtype=np.dtype(np.bool_),
                ))
                for index, agent in enumerate(agents)
            ),
            dtype=np.bool_,
        ),
        coordinator_builder_learning_applied=np.asarray(
            tuple(
                bool(_host_array(
                    agent.coordinator_result.diagnostics.builder_learning_applied,
                    name=f"agent[{index}].builder_learning_applied",
                    shape=(),
                    dtype=np.dtype(np.bool_),
                ))
                for index, agent in enumerate(agents)
            ),
            dtype=np.bool_,
        ),
        coordinator_candidate_audit_accepted=np.asarray(
            tuple(
                bool(_host_array(
                    agent.coordinator_result.diagnostics.candidate_audit_accepted,
                    name=f"agent[{index}].candidate_audit_accepted",
                    shape=(),
                    dtype=np.dtype(np.bool_),
                ))
                for index, agent in enumerate(agents)
            ),
            dtype=np.bool_,
        ),
        context_allocation_requested=np.asarray(
            tuple(
                bool(jax.device_get(agent.context_result.context_allocation_requested))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        context_full_bank_eviction_requested=np.asarray(
            tuple(
                bool(jax.device_get(agent.context_result.context_full_bank_eviction_requested))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        context_eviction_target_adjusted=np.asarray(
            tuple(
                bool(jax.device_get(agent.context_result.context_eviction_target_adjusted))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        lineage_transferred=np.asarray(
            tuple(
                bool(jax.device_get(agent.context_result.lineage_transferred))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        rescue_incremented=np.asarray(
            tuple(
                bool(jax.device_get(agent.context_result.rescue_incremented))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        lifecycle_committed=np.asarray(
            tuple(
                bool(jax.device_get(agent.lifecycle_proof.lifecycle_committed))
                for agent in agents
            ),
            dtype=np.bool_,
        ),
        lifecycle_selected_active_slots=np.asarray(
            tuple(
                int(jax.device_get(agent.lifecycle_proof.selected_active_slot))
                for agent in agents
            ),
            dtype=np.int32,
        ),
        lifecycle_selected_candidate_slots=np.asarray(
            tuple(
                int(jax.device_get(agent.lifecycle_proof.selected_candidate_slot))
                for agent in agents
            ),
            dtype=np.int32,
        ),
        pair_admission_masks=np.stack(
            tuple(
                np.asarray(jax.device_get(agent.lifecycle_proof.pair_admission_mask))
                for agent in agents
            )
        ).astype(np.bool_),
        memory_settlements_performed=np.asarray(
            tuple(agent.memory_settle_result is not None for agent in agents),
            dtype=np.bool_,
        ),
        memory_retrieval_categorical=np.asarray(
            tuple(bool(jax.device_get(agent.memory_retrieval_categorical)) for agent in agents),
            dtype=np.bool_,
        ),
        memory_consumed=np.asarray(
            tuple(bool(jax.device_get(agent.memory_consumed)) for agent in agents),
            dtype=np.bool_,
        ),
        memory_proposed_actions=np.asarray(
            tuple(int(jax.device_get(agent.memory_proposed_action)) for agent in agents),
            dtype=np.int32,
        ),
        planner_partner_belief=planner_plan.partner_belief,
        planner_partner_belief_by_own_action=planner_plan.partner_belief_by_own_action,
        planner_world_cell_valid=planner_plan.world_cell_valid,
        planner_expected_net_rewards=planner_plan.expected_net_rewards,
        planner_proposed_actions=planner_plan.proposed_actions,
        bmp_final_actions=np.asarray(
            tuple(
                int(jax.device_get(agent.bmp_prepared_projection.binding.final_action))
                for agent in agents
            ),
            dtype=np.int32,
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadOperationalTranscript:
    """Private event/B/M/P/PP and available full-path mechanism transcript."""

    schema: str
    event: HCCLCausalCoreEventReceipt
    action_bundle: HCCLRoutedContinualDyadActionBundle
    source_state_content_token: Array
    pp_proposal: HCCLCausalCoreProposal
    pre_transaction_words: Array
    post_transaction_words: Array
    mechanism_diagnostics: HCCLRoutedContinualDyadOperationalMechanismDiagnostics

    def __post_init__(self) -> None:
        if self.schema != HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA:
            raise ValueError("routed operational transcript schema is unsupported")
        if type(self.event) is not HCCLCausalCoreEventReceipt:
            raise TypeError("routed operational transcript event has the wrong type")
        if type(self.action_bundle) is not HCCLRoutedContinualDyadActionBundle:
            raise TypeError("routed operational transcript action bundle has the wrong type")
        if type(self.pp_proposal) is not HCCLCausalCoreProposal:
            raise TypeError("routed operational transcript PP proposal has the wrong type")
        if type(self.mechanism_diagnostics) is not (
            HCCLRoutedContinualDyadOperationalMechanismDiagnostics
        ):
            raise TypeError("routed operational transcript mechanism diagnostics are malformed")
        source_token = _host_array(
            self.source_state_content_token,
            name="transcript.source_state_content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=np.dtype(np.uint8),
        )
        bundle_source = _host_array(
            self.action_bundle.source_state_token,
            name="transcript.action_bundle.source_state_token",
            shape=(_TOKEN_NBYTES,),
            dtype=np.dtype(np.uint8),
        )
        pre = _host_array(
            self.pre_transaction_words,
            name="transcript.pre_transaction_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        post = _host_array(
            self.post_transaction_words,
            name="transcript.post_transaction_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        event_words = _host_array(
            self.event.source_step_words,
            name="transcript.event.source_step_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        proposal_words = _host_array(
            self.pp_proposal.source_step_words,
            name="transcript.pp_proposal.source_step_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        final_actions = _host_array(
            self.action_bundle.final_actions,
            name="transcript.action_bundle.final_actions",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.int32),
        )
        proposal_actions = _host_array(
            self.pp_proposal.joint_action_ids,
            name="transcript.pp_proposal.joint_action_ids",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.int32),
        )
        if not (
            np.array_equal(source_token, bundle_source)
            and np.array_equal(event_words, pre)
            and np.array_equal(proposal_words, pre)
            and np.array_equal(post, _increment_clock_words(pre))
            and np.array_equal(final_actions, proposal_actions)
        ):
            raise ValueError("routed operational event/B/M/P/PP transcript is unbound")
        if not np.array_equal(
            np.asarray(jax.device_get(self.event.content_tag_words)),
            np.asarray(jax.device_get(self.pp_proposal.event_content_tag_words)),
        ):
            raise ValueError("routed operational PP proposal names a foreign event")
        _host_true(self.pp_proposal.valid, name="transcript.pp_proposal.valid")
        regime = _host_array(
            self.pp_proposal.evaluator_regime_id,
            name="transcript.pp_proposal.evaluator_regime_id",
            shape=(),
            dtype=np.dtype(np.int32),
        )
        if not 0 <= int(regime) < 4:
            raise ValueError("routed operational PP evaluator regime is outside the domain")
        for name, value, shape in (
            ("task_score", self.pp_proposal.signals.task_score, ()),
            ("net_reward", self.pp_proposal.signals.net_reward, (_N_AGENTS,)),
            ("factors.gathering", self.pp_proposal.factors.gathering, ()),
            ("factors.velocity", self.pp_proposal.factors.velocity, ()),
            ("factors.convention_clean", self.pp_proposal.factors.convention_clean, ()),
            ("factors.convention_noisy", self.pp_proposal.factors.convention_noisy, ()),
        ):
            array = _host_array(
                value,
                name=f"transcript.pp_proposal.{name}",
                shape=shape,
                dtype=np.dtype(np.float32),
            )
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"routed operational PP {name} must be finite")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadOperationalEventResult:
    """One published routed destination and its compact private transcript."""

    schema: str
    state: HCCLRoutedContinualDyadState
    transcript: HCCLRoutedContinualDyadOperationalTranscript
    work: HCCLRoutedContinualDyadOperationalWork
    update_applied: bool

    def __post_init__(self) -> None:
        if self.schema != HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA:
            raise ValueError("routed operational result schema is unsupported")
        if type(self.state) is not HCCLRoutedContinualDyadState:
            raise TypeError("routed operational result state has the wrong type")
        if type(self.transcript) is not HCCLRoutedContinualDyadOperationalTranscript:
            raise TypeError("routed operational result transcript has the wrong type")
        if type(self.work) is not HCCLRoutedContinualDyadOperationalWork:
            raise TypeError("routed operational result work has the wrong type")
        if type(self.update_applied) is not bool:
            raise TypeError("routed operational update_applied must be an exact bool")
        if not self.update_applied:
            raise ValueError("routed operational result may expose only a committed candidate")
        destination = _host_array(
            self.state.hccl_state.world_state.step_words,
            name="routed operational destination clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        post = _host_array(
            self.transcript.post_transaction_words,
            name="routed operational transcript post clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        if not np.array_equal(destination, post):
            raise ValueError("routed operational destination and transcript clocks differ")


class HCCLRoutedContinualDyadOperationalError(RuntimeError):
    """Fail-closed routed operational error; executor retains its source."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        super().__init__(f"routed HCCL operational event failed during {stage}: {detail}")


def _proposal_at(proposals: HCCLCausalCoreProposal, index: int) -> HCCLCausalCoreProposal:
    return cast(HCCLCausalCoreProposal, jax.tree.map(lambda leaf: leaf[index], proposals))


def _require_prepared_flags(prepared: HCCLRoutedContinualDyadPreparedTransaction) -> None:
    for name in _PREPARED_SCALAR_FLAGS:
        _host_true(getattr(prepared, name), name=f"prepared.{name}")
    for name in _PREPARED_VECTOR_FLAGS:
        _host_all_true(getattr(prepared, name), name=f"prepared.{name}")
    _host_true(prepared.agent_0.child_valid, name="prepared.agent_0.child_valid")
    _host_true(prepared.agent_1.child_valid, name="prepared.agent_1.child_valid")
    _host_true(
        prepared.hccl_result.update_applied,
        name="prepared.hccl_result.update_applied",
    )
    _host_true(
        prepared.planner_result.transaction_applied,
        name="prepared.planner_result.transaction_applied",
    )
    _host_true(prepared.agent_0.bmp_result.update_applied, name="agent_0.bmp_result")
    _host_true(prepared.agent_1.bmp_result.update_applied, name="agent_1.bmp_result")


def _execute_routed_operational_event(
    owner: HCCLRoutedContinualDyad,
    state: HCCLRoutedContinualDyadState,
    next_hard_action_masks: Array | None,
) -> HCCLRoutedContinualDyadOperationalEventResult:
    """Use the routed owner's complete local preparation without outer replay."""

    if type(owner) is not HCCLRoutedContinualDyad:
        raise TypeError("owner must be an exact HCCLRoutedContinualDyad")
    if type(state) is not HCCLRoutedContinualDyadState:
        raise TypeError("state must be an exact HCCLRoutedContinualDyadState")
    if next_hard_action_masks is not None and any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree.leaves(next_hard_action_masks)
    ):
        raise TypeError("routed operational masks must be host/eager")

    event = owner.prepare_event(state)
    action_bundle = owner.bind_actions(state, event)
    masks = (
        action_bundle.hard_action_masks
        if next_hard_action_masks is None
        else next_hard_action_masks
    )
    prepared = owner.prepare_transaction(
        state,
        event,
        action_bundle,
        next_hard_action_masks=masks,
    )
    if type(prepared) is not HCCLRoutedContinualDyadPreparedTransaction:
        raise TypeError("routed owner returned a malformed prepared transaction")
    if prepared.source_state is not state:
        raise ValueError("routed preparation lost its exact local source identity")
    if prepared.event is not event:
        raise ValueError("routed preparation lost its exact local event identity")
    if prepared.action_bundle is not action_bundle:
        raise ValueError("routed preparation lost its exact local action-bundle identity")
    _require_prepared_flags(prepared)

    source_words = _host_array(
        state.hccl_state.world_state.step_words,
        name="routed operational source clock",
        shape=(2,),
        dtype=np.dtype(np.uint32),
    )
    pre_words = _host_array(
        prepared.hccl_result.pre_transaction_words,
        name="prepared.hccl_result.pre_transaction_words",
        shape=(2,),
        dtype=np.dtype(np.uint32),
    )
    post_words = _host_array(
        prepared.hccl_result.post_transaction_words,
        name="prepared.hccl_result.post_transaction_words",
        shape=(2,),
        dtype=np.dtype(np.uint32),
    )
    candidate_words = _host_array(
        prepared.candidate_state.hccl_state.world_state.step_words,
        name="prepared.candidate_state clock",
        shape=(2,),
        dtype=np.dtype(np.uint32),
    )
    if not (
        np.array_equal(pre_words, source_words)
        and np.array_equal(post_words, _increment_clock_words(source_words))
        and np.array_equal(candidate_words, post_words)
    ):
        raise ValueError("routed locally prepared source/candidate clocks are discontinuous")

    pp_proposal = _proposal_at(prepared.hccl_result.world_proposals, _PP_SLOT)
    transcript = HCCLRoutedContinualDyadOperationalTranscript(
        schema=HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA,
        event=event,
        action_bundle=action_bundle,
        source_state_content_token=state.content_token,
        pp_proposal=pp_proposal,
        pre_transaction_words=prepared.hccl_result.pre_transaction_words,
        post_transaction_words=prepared.hccl_result.post_transaction_words,
        mechanism_diagnostics=_mechanism_diagnostics(prepared),
    )
    return HCCLRoutedContinualDyadOperationalEventResult(
        schema=HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA,
        state=prepared.candidate_state,
        transcript=transcript,
        work=_make_operational_work(
            prepared_owner_work=prepared.work,
            runner_checkpoint_state_validations=0,
        ),
        update_applied=True,
    )


class _HCCLRoutedContinualDyadOperationalExecutor:
    """Own routed state and publish only source-bound checked candidates."""

    __slots__ = (
        "_owner",
        "_state",
        "_absolute_step",
        "_maximum_transitions",
        "_checkpoint_interval",
    )

    def __init__(
        self,
        owner: HCCLRoutedContinualDyad,
        state: HCCLRoutedContinualDyadState,
        *,
        checkpoint_interval: int | None,
    ) -> None:
        if type(owner) is not HCCLRoutedContinualDyad:
            raise TypeError("owner must be an exact HCCLRoutedContinualDyad")
        if type(state) is not HCCLRoutedContinualDyadState:
            raise TypeError("state must be an exact HCCLRoutedContinualDyadState")
        if checkpoint_interval is not None:
            _exact_int(checkpoint_interval, name="checkpoint_interval", minimum=1)
        try:
            _host_true(owner.state_valid(state), name="initial routed state validity")
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise HCCLRoutedContinualDyadOperationalError(
                "initial-checkpoint",
                str(error),
            ) from error
        words = _host_array(
            state.hccl_state.world_state.step_words,
            name="initial routed state clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        if int(words[0]) != 0:
            raise ValueError("fixed routed HCCL lives require a zero high clock word")
        maximum = _exact_int(
            owner.config.hccl.world_config.maximum_committed_transitions,
            name="world maximum_committed_transitions",
            minimum=1,
        )
        if int(words[1]) > maximum:
            raise ValueError("initial routed state clock exceeds the configured life")
        self._owner = owner
        self._state = state
        self._absolute_step = int(words[1])
        self._maximum_transitions = maximum
        self._checkpoint_interval = checkpoint_interval

    @property
    def state(self) -> HCCLRoutedContinualDyadState:
        return self._state

    @property
    def absolute_step(self) -> int:
        return self._absolute_step

    @property
    def checkpoint_interval(self) -> int | None:
        return self._checkpoint_interval

    def step(
        self,
        next_hard_action_masks: Array | None = None,
    ) -> HCCLRoutedContinualDyadOperationalEventResult:
        """Compute and bind one result to the owned source before publication."""

        if self._absolute_step >= self._maximum_transitions:
            raise HCCLRoutedContinualDyadOperationalError(
                "bounds",
                "configured routed life has no remaining transition capacity",
            )
        source = self._state
        source_words = _host_array(
            source.hccl_state.world_state.step_words,
            name="executor routed source clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        source_token = _host_array(
            source.content_token,
            name="executor routed source content token",
            shape=(_TOKEN_NBYTES,),
            dtype=np.dtype(np.uint8),
        )
        try:
            result = _execute_routed_operational_event(
                self._owner,
                source,
                next_hard_action_masks,
            )
        except HCCLRoutedContinualDyadOperationalError:
            raise
        except Exception as error:
            raise HCCLRoutedContinualDyadOperationalError(
                "routed-preparation",
                str(error),
            ) from error
        if type(result) is not HCCLRoutedContinualDyadOperationalEventResult:
            raise HCCLRoutedContinualDyadOperationalError(
                "result-contract",
                "kernel returned a malformed routed operational result",
            )

        next_step = self._absolute_step + 1
        transcript_pre = _host_array(
            result.transcript.pre_transaction_words,
            name="result transcript pre clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        transcript_post = _host_array(
            result.transcript.post_transaction_words,
            name="result transcript post clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        transcript_token = _host_array(
            result.transcript.source_state_content_token,
            name="result transcript source content token",
            shape=(_TOKEN_NBYTES,),
            dtype=np.dtype(np.uint8),
        )
        expected_pre = np.asarray((0, self._absolute_step), dtype=np.uint32)
        expected_post = np.asarray((0, next_step), dtype=np.uint32)
        if not (
            np.array_equal(source_words, expected_pre)
            and np.array_equal(transcript_pre, source_words)
            and np.array_equal(transcript_post, expected_post)
            and np.array_equal(transcript_token, source_token)
        ):
            raise HCCLRoutedContinualDyadOperationalError(
                "source-transcript-binding",
                "result does not name the executor's exact source token and next clock",
            )

        final_step = next_step == self._maximum_transitions
        periodic_due = (
            self._checkpoint_interval is not None
            and next_step % self._checkpoint_interval == 0
        )
        checkpoint_due = final_step or periodic_due
        if checkpoint_due:
            stage = "final-checkpoint" if final_step else "periodic-checkpoint"
            try:
                _host_true(
                    self._owner.state_valid(result.state),
                    name=f"{stage} routed candidate validity",
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise HCCLRoutedContinualDyadOperationalError(stage, str(error)) from error
            result = dataclasses.replace(
                result,
                work=dataclasses.replace(
                    result.work,
                    runner_checkpoint_state_validations=1,
                ),
            )

        self._state = result.state
        self._absolute_step = next_step
        return result

    def validate_checkpoint(self) -> HCCLRoutedContinualDyadState:
        """Run one exact full-state validation without changing owned state."""

        try:
            _host_true(
                self._owner.state_valid(self._state),
                name="explicit routed operational checkpoint",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise HCCLRoutedContinualDyadOperationalError(
                "explicit-checkpoint",
                str(error),
            ) from error
        return self._state


__all__ = (
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_ABSENT_ARM_ALTERNATIVES",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_DIAGNOSTIC_COVERAGE",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_MECHANISM_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_STATUS",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA",
    "HCCLRoutedContinualDyadOperationalError",
    "HCCLRoutedContinualDyadOperationalEventResult",
    "HCCLRoutedContinualDyadOperationalMechanismDiagnostics",
    "HCCLRoutedContinualDyadOperationalTranscript",
    "HCCLRoutedContinualDyadOperationalWork",
)
