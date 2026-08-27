"""Tests for bounded JEPA transfer feasibility."""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.benchmarks import jepa_transfer_feasibility as lane

pytestmark = pytest.mark.unit


def _tiny() -> lane.JEPATransferProtocol:
    return lane.JEPATransferProtocol(
        steps=8,
        phase_length=4,
        pretraining_steps=4,
        warmup_steps=2,
        exploration_period=2,
    )


@pytest.fixture(scope="module")
def result() -> lane.JEPATransferResult:
    return lane.run_jepa_transfer_feasibility(_tiny())


def test_end_to_end_receipt_is_matched_strict_and_nonpromoting(
    result: lane.JEPATransferResult,
) -> None:
    assert result.schema == lane.JEPA_TRANSFER_SCHEMA
    assert result.development_only is True
    assert result.scientific_promotion_allowed is False
    assert result.visual_robotics_parity_claimed is False
    assert lane.validate_jepa_transfer_payload(result.to_payload()) == result
    assert tuple(arm.arm_id for arm in result.arms[:7]) == lane.FROZEN_ARM_IDS
    assert all(arm.negative_outcome_retained for arm in result.arms)
    assert all(arm.resources.imported_pretraining_bytes == 0 for arm in result.arms)
    assert all(arm.resources.online_replay_bytes == 0 for arm in result.arms)
    assert result.arms[0].resources.pretraining_encoder_updates == 4


def test_transcripts_are_deterministic_while_timing_remains_telemetry(
    result: lane.JEPATransferResult,
) -> None:
    first = result.arms[0]
    second = lane._run_model_arm(_tiny(), first.seed, first.arm_id)
    assert first.action_sha256 == second.action_sha256
    assert first.reward_sha256 == second.reward_sha256
    assert first.return_sum == second.return_sum
    assert first.resources == second.resources
    assert first.timing.clock == "perf_counter_ns_process_local_telemetry_only"


def test_decision_off_reduces_exactly_to_mechanism_off(
    result: lane.JEPATransferResult,
) -> None:
    for offset in range(0, len(result.arms), len(lane.FROZEN_ARM_IDS)):
        decision_off, mechanism_off = result.arms[offset + 4 : offset + 6]
        assert decision_off.action_sha256 == mechanism_off.action_sha256
        assert decision_off.reward_sha256 == mechanism_off.reward_sha256
        assert decision_off.resources.pretraining_examples == 4
        assert mechanism_off.resources.pretraining_examples == 0


def test_live_transfer_selector_is_jittable() -> None:
    model = lane._model(encoder_learning=False)
    state = model.init(jr.key(1))
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    eager = lane._model_action(model, state, observation)
    compiled = jax.jit(lane._model_action, static_argnums=(0,))(model, state, observation)
    assert int(eager) in (0, 1)
    assert int(compiled) == int(eager)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"scientific_promotion_allowed": True}, "nonpromoting"),
        ({"visual_robotics_parity_claimed": True}, "nonpromoting"),
        ({"research_pins": {}}, "research pins"),
    ],
)
def test_result_rejects_promotion_parity_and_pin_drift(
    result: lane.JEPATransferResult,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(result, **changes)


def test_hostile_and_expanded_payloads_fail_closed(result: lane.JEPATransferResult) -> None:
    with pytest.raises(ValueError, match="exact dict"):
        lane.validate_jepa_transfer_payload(object())
    expanded = result.to_payload()
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        lane.validate_jepa_transfer_payload(expanded)
    forged = copy.deepcopy(result.to_payload())
    forged["arms"][0]["resources"]["imported_pretraining_bytes"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="forbids imported"):
        lane.validate_jepa_transfer_payload(forged)
    forged_timing = copy.deepcopy(result.to_payload())
    forged_timing["arms"][0]["timing"]["clock"] = "benchmark"  # type: ignore[index]
    with pytest.raises(ValueError, match="timing scope"):
        lane.validate_jepa_transfer_payload(forged_timing)
    forged_bytes = copy.deepcopy(result.to_payload())
    forged_bytes["arms"][0]["resources"]["persistent_mechanism_bytes"] += 1  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="persistent mechanism bytes"):
        lane.validate_jepa_transfer_payload(forged_bytes)
    forged_identity = copy.deepcopy(result.to_payload())
    forged_identity["identity"]["workload_registry_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        lane.validate_jepa_transfer_payload(forged_identity)


def test_validator_is_pure_on_success_and_failure(
    result: lane.JEPATransferResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = result.to_payload()
    original = copy.deepcopy(payload)

    def forbidden_execution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation must not execute pretraining or replay")

    monkeypatch.setattr(lane, "_pretrain", forbidden_execution)
    monkeypatch.setattr(lane, "_base_actions", forbidden_execution)
    monkeypatch.setattr(lane.LatentWorldModel, "update", forbidden_execution)
    lane.validate_jepa_transfer_payload(payload)
    assert payload == original

    invalid = copy.deepcopy(payload)
    invalid["arms"][0]["resources"]["persistent_environment_bytes"] += 1  # type: ignore[index,operator]
    before_failure = copy.deepcopy(invalid)
    with pytest.raises(ValueError, match="environment.*bytes"):
        lane.validate_jepa_transfer_payload(invalid)
    assert invalid == before_failure


def test_paper_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        lane.PINNED_RESEARCH["new"] = "mutable"  # type: ignore[index]


def test_result_research_pins_are_deeply_immutable(result: lane.JEPATransferResult) -> None:
    assert result.research_pins == tuple(sorted(lane.PINNED_RESEARCH.items()))
    with pytest.raises(TypeError):
        result.research_pins[0] = ("jepa_wm_paper", "forged")  # type: ignore[index]


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": True},
        {"steps": 7, "phase_length": 4},
        {"seeds": (1,)},
        {"pretraining_steps": 0},
        {"exploration_period": 1},
    ],
)
def test_protocol_rejects_hostile_or_unmatched_axes(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        lane.JEPATransferProtocol(**changes)  # type: ignore[arg-type]


def test_late_return_uses_last_complete_phase_not_trailing_window() -> None:
    """Supported unaligned protocol must not mix two payoff phases.

    ``JEPATransferProtocol`` accepts ``steps=10, phase_length=4`` (covers one
    A/B recurrence). Frozen seed 1577003 then yields a trailing 4-step window
    that straddles phase B and the next phase A, so ``rewards[-phase_length:]``
    silently reports 0.0 while the last complete phase sums to 1.0.
    """
    protocol = lane.JEPATransferProtocol(
        steps=10,
        phase_length=4,
        pretraining_steps=4,
        warmup_steps=2,
        exploration_period=2,
    )
    seed = 1_577_003
    receipt = lane._run_control(protocol, seed, sarsa=False)
    assert receipt.late_return_sum == 1.0
    assert receipt.return_sum == 3.0
    # The buggy trailing slice is 0.0 on this supported seed.
    assert receipt.late_return_sum != 0.0


def test_late_phase_return_sum_keeps_aligned_windows() -> None:
    aligned = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert lane._late_phase_return_sum(aligned, 4) == 1.0
    unaligned = aligned + (0.0, 0.0)
    assert lane._late_phase_return_sum(unaligned, 4) == 1.0
    assert sum(unaligned[-4:]) == 0.0

