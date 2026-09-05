"""Tests for the nonpromoting action-conditioned latent control lane."""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.action_conditioned_latent import (
    ACTION_LATENT_SCHEMA,
    FROZEN_ARM_IDS,
    PINNED_RESEARCH,
    ActionLatentProtocol,
    run_action_conditioned_latent_lane,
    select_latent_action,
    validate_action_latent_payload,
)
from alberta_framework.core.latent_world_model import LatentWorldModel, LatentWorldModelConfig
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
    SwitchingTwoStateState,
)

pytestmark = pytest.mark.unit


def _tiny_protocol() -> ActionLatentProtocol:
    return ActionLatentProtocol(steps=8, phase_length=4, warmup_steps=2, exploration_period=2)


@pytest.fixture(scope="module")
def lane_result():
    return run_action_conditioned_latent_lane(_tiny_protocol())


def test_lane_is_deterministic_matched_nonpromoting_and_round_trips(lane_result) -> None:
    first = lane_result
    second = run_action_conditioned_latent_lane(_tiny_protocol())

    assert first == second
    assert first.schema == ACTION_LATENT_SCHEMA
    assert first.development_only is True
    assert first.scientific_promotion_allowed is False
    assert validate_action_latent_payload(first.to_payload()) == first
    assert tuple(arm.arm_id for arm in first.arms[:6]) == FROZEN_ARM_IDS
    assert all(arm.environment_steps == 8 for arm in first.arms)
    assert all(arm.negative_outcome_retained for arm in first.arms)


def test_mechanism_off_has_exact_decision_off_transcript_parity(lane_result) -> None:
    result = lane_result
    for offset in range(0, len(result.arms), len(FROZEN_ARM_IDS)):
        decision_off = result.arms[offset + 3]
        mechanism_off = result.arms[offset + 4]
        assert decision_off.action_sha256 == mechanism_off.action_sha256
        assert decision_off.reward_sha256 == mechanism_off.reward_sha256
        assert decision_off.return_sum == mechanism_off.return_sum
        assert decision_off.model_updates == 8
        assert mechanism_off.model_updates == 0


def test_live_model_action_selector_is_jittable_and_action_conditioned() -> None:
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
        )
    )
    state = model.init(jr.key(7))
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    eager = select_latent_action(model, state, observation, False)
    compiled = jax.jit(select_latent_action, static_argnums=(0, 3))(
        model, state, observation, False
    )
    assert int(eager) in (0, 1)
    assert int(compiled) == int(eager)


def test_next_observation_cannot_identify_phase_switched_immediate_reward() -> None:
    """Refute an oracle-free reward adapter for the transition-only FTL model."""
    environment = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=4))
    phase_a = SwitchingTwoStateState(
        state_index=jnp.asarray(0, dtype=jnp.int32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )
    phase_b = phase_a.replace(step_count=jnp.asarray(4, dtype=jnp.int32))
    action = jnp.asarray(0, dtype=jnp.int32)
    next_a, reward_a, _ = environment.step(phase_a, action, jr.key(0))
    next_b, reward_b, _ = environment.step(phase_b, action, jr.key(1))

    np.testing.assert_array_equal(environment.observe(phase_a), environment.observe(phase_b))
    np.testing.assert_array_equal(next_a, next_b)
    assert float(reward_a) == 0.0
    assert float(reward_b) == 1.0


@pytest.mark.parametrize(
    "replacement, message",
    [
        ({"development_only": False}, "nonpromoting"),
        ({"scientific_promotion_allowed": True}, "nonpromoting"),
        ({"research_pins": {}}, "research pins"),
    ],
)
def test_result_rejects_promotion_and_pin_drift(
    replacement: dict[str, object], message: str, lane_result
) -> None:
    result = lane_result
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(result, **replacement)


def test_validator_rejects_hostile_expanded_or_inconsistent_payloads(lane_result) -> None:
    result = lane_result
    with pytest.raises(ValueError, match="exact dict"):
        validate_action_latent_payload(object())
    expanded = result.to_payload()
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        validate_action_latent_payload(expanded)
    promoted = copy.deepcopy(result.to_payload())
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_action_latent_payload(promoted)
    aliased = copy.deepcopy(result.to_payload())
    aliased["arms"][0]["arm_id"] = "latent_action_interactions_alias"  # type: ignore[index]
    with pytest.raises(ValueError, match="unsupported arm_id"):
        validate_action_latent_payload(aliased)
    forged_bytes = copy.deepcopy(result.to_payload())
    forged_bytes["arms"][0]["persistent_mechanism_bytes"] += 1  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="mechanism bytes differ"):
        validate_action_latent_payload(forged_bytes)
    forged_identity = copy.deepcopy(result.to_payload())
    forged_identity["identity"]["dependency_versions"][0][1] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        validate_action_latent_payload(forged_identity)


def test_paper_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        PINNED_RESEARCH["new"] = "mutable"  # type: ignore[index]


def test_result_research_pins_are_deeply_immutable(lane_result) -> None:
    assert lane_result.research_pins == tuple(sorted(PINNED_RESEARCH.items()))
    with pytest.raises(TypeError):
        lane_result.research_pins[0] = ("jedi_paper", "forged")  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 7, "phase_length": 4},
        {"seeds": (1,)},
        {"exploration_period": 1},
    ],
)
def test_protocol_rejects_hostile_unbounded_or_unmatched_axes(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ActionLatentProtocol(**kwargs)  # type: ignore[arg-type]
