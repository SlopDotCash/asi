"""Causal intervention tests for the Step 12 recommendation protocol."""

from __future__ import annotations

import inspect

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.intelligence_amplification import (
    RecommendationProtocolConfig,
    RecommendationProtocolState,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)


def _update(
    state: RecommendationProtocolState,
    recommendation: jax.Array,
    partner_action: jax.Array,
    accept_recommendation: jax.Array,
) -> tuple[RecommendationProtocolState, jax.Array, jax.Array]:
    result = update_recommendation_protocol(
        RecommendationProtocolConfig(acceptance_ema_decay=0.5),
        state,
        recommendation,
        partner_action,
        accept_recommendation,
    )
    return result.state, result.effective_action, result.accepted


def test_explicit_acceptance_overrides_different_partner_fallback() -> None:
    """Acceptance must be an intervention, not post-hoc action agreement."""
    result = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        init_recommendation_protocol_state(),
        recommendation=jnp.array(2, dtype=jnp.int32),
        partner_action=jnp.array(0, dtype=jnp.int32),
        accept_recommendation=jnp.array(True),
    )

    assert bool(result.accepted)
    assert int(result.partner_action) == 0
    assert int(result.recommendation) == 2
    assert int(result.effective_action) == 2
    assert int(result.state.accepted_count) == 1
    assert int(result.state.rejected_count) == 0


def test_explicit_rejection_preserves_partner_fallback_even_on_agreement() -> None:
    """The decision is explicit; action equality must not imply acceptance."""
    result = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        init_recommendation_protocol_state(),
        recommendation=jnp.array(1, dtype=jnp.int32),
        partner_action=jnp.array(1, dtype=jnp.int32),
        accept_recommendation=jnp.array(False),
    )

    assert not bool(result.accepted)
    assert int(result.effective_action) == 1
    assert int(result.state.accepted_count) == 0
    assert int(result.state.rejected_count) == 1


def test_explicit_rejection_falls_back_when_actions_differ() -> None:
    result = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        init_recommendation_protocol_state(),
        recommendation=jnp.array(2, dtype=jnp.int32),
        partner_action=jnp.array(0, dtype=jnp.int32),
        accept_recommendation=jnp.array(False),
    )

    assert not bool(result.accepted)
    assert int(result.effective_action) == 0


def test_protocol_is_deterministic_and_jittable() -> None:
    state = init_recommendation_protocol_state()
    compiled = jax.jit(_update)
    inputs = (
        state,
        jnp.array(2, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(True),
    )

    eager = _update(*inputs)
    first = compiled(*inputs)
    second = compiled(*inputs)

    chex.assert_trees_all_equal(eager, first)
    chex.assert_trees_all_equal(first, second)
    assert int(first[1]) == 2


@pytest.mark.parametrize("invalid_candidate", ["recommendation", "partner_action"])
def test_protocol_rejects_wide_action_candidates_before_narrowing(
    invalid_candidate: str,
) -> None:
    candidates = {
        "recommendation": jnp.array(1, dtype=jnp.int32),
        "partner_action": jnp.array(0, dtype=jnp.int32),
    }
    candidates[invalid_candidate] = jnp.array(2**32 - 1, dtype=jnp.uint32)

    with pytest.raises(ValueError, match="bounded int32 action"):
        update_recommendation_protocol(
            RecommendationProtocolConfig(),
            init_recommendation_protocol_state(),
            recommendation=candidates["recommendation"],
            partner_action=candidates["partner_action"],
            accept_recommendation=jnp.array(False),
        )


def test_compiled_protocol_rolls_back_wide_action_candidate() -> None:
    state = init_recommendation_protocol_state()
    compiled = jax.jit(update_recommendation_protocol, static_argnums=0)

    result = compiled(
        RecommendationProtocolConfig(),
        state,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(2**32 - 1, dtype=jnp.uint32),
        jnp.array(False),
    )

    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.update_applied)
    assert not bool(result.action_candidates_valid)
    assert int(result.partner_action) == 0
    assert int(result.effective_action) == 0


@pytest.mark.parametrize("decision", [1, 0.5, jnp.array([True])])
def test_acceptance_gate_is_strictly_bounded_to_a_scalar_boolean(
    decision: object,
) -> None:
    with pytest.raises(TypeError, match="scalar boolean"):
        update_recommendation_protocol(
            RecommendationProtocolConfig(),
            init_recommendation_protocol_state(),
            recommendation=jnp.array(2, dtype=jnp.int32),
            partner_action=jnp.array(0, dtype=jnp.int32),
            accept_recommendation=decision,  # type: ignore[arg-type]
        )


def test_protocol_causal_api_has_no_post_transition_or_oracle_input() -> None:
    """Acceptance can depend only on values supplied before environment.step."""
    parameter_names = set(inspect.signature(update_recommendation_protocol).parameters)

    assert parameter_names == {
        "config",
        "state",
        "recommendation",
        "partner_action",
        "accept_recommendation",
    }
    assert parameter_names.isdisjoint(
        {
            "reward",
            "next_observation",
            "optimal_action",
            "oracle_action",
            "transition",
        }
    )


def test_legacy_agreement_accounting_remains_available() -> None:
    """Four-argument callers retain their original non-intervening behaviour."""
    result = update_recommendation_protocol(
        RecommendationProtocolConfig(),
        init_recommendation_protocol_state(),
        jnp.array(2, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    assert not bool(result.accepted)
    assert int(result.effective_action) == 0
