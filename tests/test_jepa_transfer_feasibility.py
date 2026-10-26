"""Tests for bounded JEPA transfer feasibility."""

from __future__ import annotations

import copy
import dataclasses
from typing import cast

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


@pytest.mark.parametrize("field", ("return_sum", "late_return_sum"))
def test_payload_binds_exact_reduction_returns(
    result: lane.JEPATransferResult, field: str
) -> None:
    payload = result.to_payload()
    arms = cast(list[dict[str, object]], payload["arms"])
    decision_off = arms[4]
    mechanism_off = arms[5]
    original = decision_off[field]
    assert type(original) is float
    assert original == mechanism_off[field]
    decision_off[field] = original + 1.0

    with pytest.raises(ValueError, match="decision-off.*return"):
        lane.validate_jepa_transfer_payload(payload)


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


def test_permuted_encoder_control_is_not_a_latent_relabeling() -> None:
    """The causal control must destroy encoder structure, not relabel latents.

    The fresh predictor consumes ``concat(latent, one_hot(action),
    latent (x) one_hot)`` through i.i.d.-initialized linear heads, so any
    permutation of the latent coordinates is undone by permuting the
    predictor's weights: a column-reversed encoder reproduced the transfer
    arm's whole deployment. The control keeps the transferred weight multiset
    but must not be any column permutation of the transferred encoder.
    """
    protocol = _tiny()
    seed = lane.FROZEN_DEVELOPMENT_SEEDS[0]
    pretrained, _, _, _ = lane._pretrain(protocol, seed)
    _, transfer = lane._deployment_state("asi_encoder_transfer", seed, pretrained)
    _, permuted = lane._deployment_state("encoder_permuted", seed, pretrained)
    _, permuted_again = lane._deployment_state("encoder_permuted", seed, pretrained)
    transfer_matrix = jnp.asarray(transfer.encoder_matrix)  # type: ignore[attr-defined]
    permuted_matrix = jnp.asarray(permuted.encoder_matrix)  # type: ignore[attr-defined]
    assert permuted_matrix.shape == transfer_matrix.shape
    assert bool(jnp.array_equal(permuted_matrix, permuted_again.encoder_matrix))  # type: ignore[attr-defined]
    assert bool(
        jnp.array_equal(jnp.sort(permuted_matrix.ravel()), jnp.sort(transfer_matrix.ravel()))
    )
    transfer_columns = sorted(map(tuple, transfer_matrix.T.tolist()))
    permuted_columns = sorted(map(tuple, permuted_matrix.T.tolist()))
    assert permuted_columns != transfer_columns
    assert bool(
        jnp.array_equal(
            jnp.sort(jnp.asarray(permuted.encoder_bias)),  # type: ignore[attr-defined]
            jnp.sort(jnp.asarray(transfer.encoder_bias)),  # type: ignore[attr-defined]
        )
    )
