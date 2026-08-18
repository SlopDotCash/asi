"""Hostile input, leftover identity, and type validation for protocol records."""

from __future__ import annotations

from typing import NoReturn

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    AgentRNGContract,
    DescriptiveContext,
    EnvironmentRNGContract,
    ForagerMatchedProtocolError,
    RankedSelectionGroup,
    ResolvedSelectionSlot,
    SelectionSlot,
    _validate_json_complexity,
    decode_strict_json,
    parse_forager_matched_protocol,
    parse_forager_matched_selection_result,
)


class _HookedString(str):
    calls: int

    def __new__(cls, value: str) -> _HookedString:
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def _called(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("hostile string hook must not run")

    def __bool__(self) -> bool:
        return self._called()

    def __len__(self) -> int:
        return self._called()

    def __eq__(self, other: object) -> bool:
        return self._called()

    def __hash__(self) -> int:
        return self._called()

    def encode(self, *args: object, **kwargs: object) -> bytes:
        return self._called()


def test_environment_rng_contract_validation() -> None:
    contract = EnvironmentRNGContract(
        identity="env_rng.v1",
        schedule_sha256="a" * 64,
    )
    assert contract.identity == "env_rng.v1"
    assert contract.schedule_sha256 == "a" * 64

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        EnvironmentRNGContract(
            identity="",
            schedule_sha256="a" * 64,
        )

    with pytest.raises(
        ForagerMatchedProtocolError,
        match="must be a lowercase 64-character SHA-256 digest",
    ):
        EnvironmentRNGContract(
            identity="env_rng.v1",
            schedule_sha256="invalid",
        )


def test_agent_rng_contract_validation() -> None:
    contract = AgentRNGContract(identity="agent_rng.v1", environment_key_shared=False)
    assert contract.identity == "agent_rng.v1"
    assert contract.environment_key_shared is False

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        AgentRNGContract(identity="", environment_key_shared=False)

    with pytest.raises(ForagerMatchedProtocolError, match="must be a boolean"):
        AgentRNGContract(identity="agent_rng.v1", environment_key_shared=1)  # type: ignore[arg-type]


def test_selection_slot_validation() -> None:
    slot = SelectionSlot(selection_group="group_a", rank=1)
    assert slot.selection_group == "group_a"
    assert slot.rank == 1

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        SelectionSlot(selection_group="", rank=1)

    with pytest.raises(ForagerMatchedProtocolError, match="must lie in"):
        SelectionSlot(selection_group="group_a", rank=0)


def test_resolved_selection_slot_validation() -> None:
    slot = ResolvedSelectionSlot(selection_group="group_a", rank=1, candidate_id="cand_1")
    assert slot.selection_group == "group_a"
    assert slot.rank == 1
    assert slot.candidate_id == "cand_1"

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        ResolvedSelectionSlot(selection_group="group_a", rank=1, candidate_id="")


def test_ranked_selection_group_validation() -> None:
    group = RankedSelectionGroup(
        selection_group="group_a",
        ranked_candidate_ids=("cand_1", "cand_2"),
        ranking_evidence_sha256="b" * 64,
    )
    assert group.selection_group == "group_a"
    assert len(group.ranked_candidate_ids) == 2

    with pytest.raises(ForagerMatchedProtocolError, match="must be a tuple of candidate IDs"):
        RankedSelectionGroup(
            selection_group="group_a",
            ranked_candidate_ids=["cand_1"],  # type: ignore[arg-type]
            ranking_evidence_sha256="b" * 64,
        )


def test_descriptive_context_validation() -> None:
    ctx = DescriptiveContext(
        candidate_ids=("cand_1",),
        analysis_role="descriptive_only",
        selection_eligible=False,
        pairing_eligible=False,
    )
    assert ctx.analysis_role == "descriptive_only"

    with pytest.raises(ForagerMatchedProtocolError, match="selection_eligible must be False"):
        DescriptiveContext(
            candidate_ids=("cand_1",),
            analysis_role="descriptive_only",
            selection_eligible=True,  # type: ignore[arg-type]
            pairing_eligible=False,
        )

    with pytest.raises(ForagerMatchedProtocolError, match="pairing_eligible must be False"):
        DescriptiveContext(
            candidate_ids=("cand_1",),
            analysis_role="descriptive_only",
            selection_eligible=False,
            pairing_eligible=True,  # type: ignore[arg-type]
        )


def test_protocol_string_subclasses_reject_before_hooks() -> None:
    for operation in (
        lambda value: EnvironmentRNGContract(
            identity=value,
            schedule_sha256="a" * 64,
        ),
        lambda value: RankedSelectionGroup(
            selection_group="group_a",
            ranked_candidate_ids=(value,),
            ranking_evidence_sha256="b" * 64,
        ),
        lambda value: DescriptiveContext(
            candidate_ids=(value,),
            analysis_role="descriptive_only",
            selection_eligible=False,
            pairing_eligible=False,
        ),
        lambda value: _validate_json_complexity([value]),
        lambda value: decode_strict_json(value),
        lambda value: parse_forager_matched_protocol(value),
        lambda value: parse_forager_matched_selection_result(value),
    ):
        hostile = _HookedString("candidate_a")
        with pytest.raises(ForagerMatchedProtocolError):
            operation(hostile)
        assert hostile.calls == 0
