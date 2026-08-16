# mypy: disable-error-code="call-arg"
"""Contracts for bounded contextual partner-message policy fusion."""

from __future__ import annotations

import copy
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.partner_policy_fusion import (
    MECHANISM_STATUS,
    PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
    RELIABILITY_HISTORY_DEVELOPMENT_ONLINE,
    RELIABILITY_HISTORY_UNAVAILABLE,
    ROUTE_ACCEPT,
    ROUTE_BLEND,
    ROUTE_IGNORE,
    ROUTE_QUERY,
    ROUTE_REQUEST_CLARIFICATION,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SOURCE_OPTION_KEYBOARD,
    SOURCE_PARTNER,
    OptionKeyboardProposal,
    PartnerMessageBatch,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
    PartnerPolicyFusionState,
)

pytestmark = pytest.mark.unit


def _replace[T](value: T, **changes: object) -> T:
    """Use chex's immutable replacement method while retaining the type."""

    return cast(T, cast(Any, value).replace(**changes))


def _fusion(**overrides: Any) -> PartnerPolicyFusion:
    values: dict[str, Any] = {
        "max_partners": 3,
        "context_dim": 2,
        "n_actions": 4,
        "max_abs_context": 1.0,
    }
    values.update(overrides)
    return PartnerPolicyFusion(PartnerPolicyFusionConfig(**values))


def _option(
    *, available: bool = False, action: int = -1, score: float = 0.0
) -> OptionKeyboardProposal:
    return OptionKeyboardProposal(
        available=jnp.asarray(available, dtype=jnp.bool_),
        action=jnp.asarray(action, dtype=jnp.int32),
        declared_score=jnp.asarray(score, dtype=jnp.float32),
    )


def _messages(
    fusion: PartnerPolicyFusion,
    entries: list[dict[str, int | float | bool]],
    *,
    decision_id: int = 10,
    event_id: int = 20,
    observation_id: int = 30,
    context_id: int = 40,
) -> PartnerMessageBatch:
    batch = fusion.empty_messages()
    for slot, entry in enumerate(entries):
        partner_id = int(entry.get("partner_id", slot))
        batch = _replace(
            batch,
            available=batch.available.at[slot].set(bool(entry.get("available", True))),
            partner_id=batch.partner_id.at[slot].set(partner_id),
            observation_id=batch.observation_id.at[slot].set(
                int(entry.get("observation_id", observation_id))
            ),
            context_id=batch.context_id.at[slot].set(
                int(entry.get("context_id", context_id))
            ),
            suggested_action=batch.suggested_action.at[slot].set(
                int(entry.get("suggested_action", 1))
            ),
            declared_confidence=batch.declared_confidence.at[slot].set(
                float(entry.get("declared_confidence", 1.0))
            ),
            rationale_reference=batch.rationale_reference.at[slot].set(
                int(entry.get("rationale_reference", 100 + slot))
            ),
            provenance_reference=batch.provenance_reference.at[slot].set(
                int(entry.get("provenance_reference", 200 + slot))
            ),
            communication_cost=batch.communication_cost.at[slot].set(
                float(entry.get("communication_cost", 0.0))
            ),
            issued_decision_id=batch.issued_decision_id.at[slot].set(
                int(entry.get("issued_decision_id", decision_id))
            ),
            issued_event_id=batch.issued_event_id.at[slot].set(
                int(entry.get("issued_event_id", event_id))
            ),
            valid_through_event_id=batch.valid_through_event_id.at[slot].set(
                int(entry.get("valid_through_event_id", event_id + 1))
            ),
        )
    return batch


def _decide(
    fusion: PartnerPolicyFusion,
    state: PartnerPolicyFusionState,
    messages: PartnerMessageBatch | None = None,
    *,
    decision_id: int = 10,
    event_id: int = 20,
    observation_id: int = 30,
    context_id: int = 40,
    context: tuple[float, float] = (0.25, -0.5),
    base_action: int = 0,
    base_score: float = 0.0,
    mask: tuple[bool, ...] = (True, True, True, True),
    option: OptionKeyboardProposal | None = None,
) -> Any:
    return fusion.decide(
        state,
        decision_id=jnp.asarray(decision_id, dtype=jnp.int32),
        event_id=jnp.asarray(event_id, dtype=jnp.int32),
        observation_id=jnp.asarray(observation_id, dtype=jnp.int32),
        context_id=jnp.asarray(context_id, dtype=jnp.int32),
        context_features=jnp.asarray(context, dtype=jnp.float32),
        base_action=jnp.asarray(base_action, dtype=jnp.int32),
        base_declared_score=jnp.asarray(base_score, dtype=jnp.float32),
        safety_action_mask=jnp.asarray(mask, dtype=jnp.bool_),
        option_proposal=fusion.empty_option_proposal() if option is None else option,
        messages=fusion.empty_messages() if messages is None else messages,
    )


def _feedback(
    *,
    decision_id: int = 10,
    event_id: int = 20,
    action: int = 1,
    partner_id: int = 0,
    assistance: float = 10_000.0,
    safe: bool = True,
    available: bool = True,
    assistance_available: bool = True,
    safety_available: bool = True,
) -> PartnerPolicyFusionFeedback:
    return PartnerPolicyFusionFeedback(
        available=jnp.asarray(available, dtype=jnp.bool_),
        decision_id=jnp.asarray(decision_id, dtype=jnp.int32),
        executed_event_id=jnp.asarray(event_id, dtype=jnp.int32),
        executed_action=jnp.asarray(action, dtype=jnp.int32),
        partner_id=jnp.asarray(partner_id, dtype=jnp.int32),
        assistance_value_available=jnp.asarray(assistance_available, dtype=jnp.bool_),
        realized_assistance_value=jnp.asarray(assistance, dtype=jnp.float32),
        safety_outcome_available=jnp.asarray(safety_available, dtype=jnp.bool_),
        safety_outcome_ok=jnp.asarray(safe, dtype=jnp.bool_),
    )


def _assert_tree_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert str(first_tree) == str(second_tree)
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def _assert_float_leaves_finite(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            assert np.all(np.isfinite(array))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_partners", True),
        ("max_partners", 0),
        ("context_dim", 0),
        ("n_actions", 0),
        ("max_message_horizon", 0),
        ("counter_cap", 0),
        ("learning_rate", 0.0),
        ("max_abs_weight", float("nan")),
        ("max_abs_context", float("inf")),
        ("safety_target_weight", 1.1),
        ("clarification_confidence_threshold", -0.1),
        ("max_query_cost", 20_000.0),
        ("blend_net_value_threshold", 0.5),
    ],
)
def test_config_rejects_invalid_values(field: str, value: Any) -> None:
    kwargs: dict[str, Any] = {"max_partners": 2, "context_dim": 2, "n_actions": 3}
    kwargs[field] = value
    with pytest.raises(ValueError):
        PartnerPolicyFusionConfig(**kwargs)


class _SpoofedFloat:
    """Mimics ``float`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return 0.5


@pytest.mark.parametrize(
    "field",
    ["learning_rate", "max_abs_weight", "communication_cost_weight"],
)
def test_config_rejects_class_spoofed_float_fields(field: str) -> None:
    kwargs: dict[str, Any] = {
        "max_partners": 2,
        "context_dim": 2,
        "n_actions": 3,
        field: _SpoofedFloat(),
    }
    with pytest.raises(ValueError, match=field):
        PartnerPolicyFusionConfig(**kwargs)


def test_strict_config_roundtrip_and_l0_status() -> None:
    fusion = _fusion()
    restored = PartnerPolicyFusion.from_config(fusion.to_config())
    assert restored.to_config() == fusion.to_config()
    nested = fusion.to_config()["config"]
    assert isinstance(nested, dict)
    assert nested["mechanism_status"] == MECHANISM_STATUS
    assert nested["scientific_promotion_allowed"] is SCIENTIFIC_PROMOTION_ALLOWED is False
    assert "task_id" not in str(fusion.to_config())
    assert "regime_id" not in str(fusion.to_config())

    changed = copy.deepcopy(fusion.to_config())
    changed["unexpected"] = 1
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_config(changed)
    changed = copy.deepcopy(fusion.to_config())
    cast(dict[str, Any], changed["config"])["unexpected"] = 1
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_config(changed)


def test_resource_budget_matches_fixed_state_exactly() -> None:
    fusion = _fusion()
    state = fusion.init()
    fusion.validate_state(state)
    budget = fusion.resource_budget
    actual_bytes = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(state))
    assert budget.persistent_state_bytes == actual_bytes
    assert budget.trainable_float32_scalars == 3 * 4
    assert budget.max_messages_per_decision == 3
    assert budget.partner_id_pairwise_equality_comparisons_per_decision == 9
    assert budget.max_trainable_scalars_touched_per_feedback == 4
    assert budget.rng_state_bytes == 0
    assert budget.replay_capacity == 0
    assert budget.dynamic_partner_capacity == 0


def test_missing_messages_preserve_counterfactual_base_and_audit_availability() -> None:
    fusion = _fusion()
    result = _decide(fusion, fusion.init())
    decision = result.decision
    assert int(decision.route) == ROUTE_IGNORE
    assert int(decision.counterfactual_base_action) == 0
    assert int(decision.effective_action) == 0
    assert not bool(decision.partner_influenced)
    assert not bool(decision.feedback_armed)
    assert not bool(jnp.any(decision.availability.messages_declared))
    assert bool(decision.availability.base_action_available)
    assert bool(decision.availability.action_available)
    assert bool(decision.applied)
    assert int(result.state.decision_count) == 1


def test_valid_typed_message_is_accepted_and_arms_exact_feedback_record() -> None:
    fusion = _fusion()
    messages = _messages(fusion, [{"suggested_action": 2, "declared_confidence": 1.0}])
    result = _decide(fusion, fusion.init(), messages)
    decision = result.decision
    assert int(decision.route) == ROUTE_ACCEPT
    assert int(decision.selected_partner_slot) == 0
    assert int(decision.selected_partner_id) == 0
    assert int(decision.selected_partner_action) == 2
    assert int(decision.effective_action) == 2
    assert bool(decision.partner_influenced)
    assert bool(decision.feedback_armed)
    assert bool(result.state.feedback_armed)
    assert int(result.state.armed_decision_id) == 10
    assert int(result.state.armed_event_id) == 20
    assert int(result.state.armed_action) == 2
    assert int(result.state.armed_partner_id) == 0
    assert int(decision.reliability_history_status) == RELIABILITY_HISTORY_UNAVAILABLE
    assert not bool(decision.reliability_history_available)
    assert bool(decision.uncalibrated_development_exploration)
    np.testing.assert_array_equal(
        np.asarray(decision.route_one_hot),
        np.asarray([False, False, True, False, False]),
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"available": False},
        {"observation_id": 999},
        {"context_id": 999},
        {"issued_decision_id": 999},
        {"issued_event_id": 21},
        {"valid_through_event_id": 19},
        {"valid_through_event_id": 40},
        {"declared_confidence": float("nan")},
        {"declared_confidence": 1.1},
        {"communication_cost": float("inf")},
        {"rationale_reference": -1},
        {"provenance_reference": -1},
        {"suggested_action": 99},
        {"suggested_action": 3},
    ],
)
def test_invalid_expired_nonfinite_missing_and_unsafe_messages_fall_back(
    entry: dict[str, int | float | bool],
) -> None:
    fusion = _fusion()
    messages = _messages(fusion, [entry])
    mask = (True, True, True, False)
    result = _decide(fusion, fusion.init(), messages, mask=mask)
    assert int(result.decision.effective_action) == 0
    assert int(result.decision.route) == ROUTE_IGNORE
    assert not bool(result.decision.partner_influenced)
    assert not bool(result.state.feedback_armed)
    _assert_float_leaves_finite(result)


def test_duplicate_partner_identifiers_are_all_invalid() -> None:
    fusion = _fusion()
    messages = _messages(
        fusion,
        [
            {"partner_id": 1, "suggested_action": 1},
            {"partner_id": 1, "suggested_action": 2},
        ],
    )
    result = _decide(fusion, fusion.init(), messages)
    assert not bool(jnp.any(result.decision.availability.messages_unique_partner[:2]))
    assert not bool(jnp.any(result.decision.availability.messages_valid))
    assert int(result.decision.effective_action) == 0


def test_hard_mask_is_inviolable_and_unsafe_base_fails_closed_atomically() -> None:
    fusion = _fusion()
    state = fusion.init()
    messages = _messages(fusion, [{"suggested_action": 2}])
    unsafe_message = _decide(
        fusion,
        state,
        messages,
        mask=(True, True, False, True),
    )
    assert int(unsafe_message.decision.effective_action) == 0
    assert not bool(unsafe_message.decision.availability.messages_safety_valid[0])

    failed = _decide(
        fusion,
        state,
        messages,
        base_action=0,
        mask=(False, True, True, True),
    )
    assert int(failed.decision.effective_action) == -1
    assert bool(failed.decision.shield.failed_closed)
    assert not bool(failed.decision.applied)
    _assert_tree_equal(failed.state, state)


@pytest.mark.parametrize(
    ("context", "base_score", "invalid_field"),
    [
        ((float("nan"), 0.0), 0.0, "context"),
        ((2.0, 0.0), 0.0, "context"),
        ((0.0, 0.0), float("nan"), "base_score"),
        ((0.0, 0.0), 20_000.0, "base_score"),
    ],
)
def test_corrupt_context_or_base_score_uses_safe_base_without_advancing_fusion(
    context: tuple[float, float],
    base_score: float,
    invalid_field: str,
) -> None:
    fusion = _fusion()
    state = fusion.init()
    messages = _messages(fusion, [{"suggested_action": 2}])
    result = _decide(
        fusion,
        state,
        messages,
        context=context,
        base_score=base_score,
    )
    assert int(result.decision.effective_action) == 0
    assert bool(result.decision.availability.action_available)
    assert not bool(result.decision.input_valid)
    assert not bool(result.decision.applied)
    assert not bool(result.decision.availability.route_available)
    assert not bool(result.decision.shield.failed_closed)
    if invalid_field == "context":
        assert not bool(result.decision.availability.context_features_valid)
        assert bool(result.decision.availability.base_declared_score_valid)
    else:
        assert bool(result.decision.availability.context_features_valid)
        assert not bool(result.decision.availability.base_declared_score_valid)
    _assert_tree_equal(result.state, state)
    _assert_float_leaves_finite(result)


def test_validity_horizon_is_safe_at_int32_identity_limit() -> None:
    fusion = _fusion(max_message_horizon=2)
    maximum = np.iinfo(np.int32).max
    messages = _messages(
        fusion,
        [
            {
                "suggested_action": 1,
                "issued_event_id": maximum - 1,
                "valid_through_event_id": maximum,
            }
        ],
        decision_id=maximum,
        event_id=maximum,
    )
    result = _decide(
        fusion,
        fusion.init(),
        messages,
        decision_id=maximum,
        event_id=maximum,
    )
    assert bool(result.decision.availability.messages_horizon_valid[0])
    assert bool(result.decision.availability.messages_valid[0])
    assert int(result.decision.effective_action) == 1
    _assert_float_leaves_finite(result)


def _history_ready_zero_state(fusion: PartnerPolicyFusion) -> PartnerPolicyFusionState:
    cfg = fusion.config
    count = cfg.min_feedback_for_learned_routing
    base = fusion.init()
    counts = base.feedback_counts.at[0].set(count)
    return _replace(
        base,
        feedback_counts=counts,
        decision_count=jnp.asarray(count, dtype=jnp.int32),
        feedback_applied_count=jnp.asarray(count, dtype=jnp.int32),
        has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
        last_decision_id=jnp.asarray(count - 1, dtype=jnp.int32),
        last_event_id=jnp.asarray(count - 1, dtype=jnp.int32),
    )


def test_all_five_routes_have_explicit_auditable_semantics() -> None:
    fusion = _fusion()
    state = fusion.init()
    ignore = _decide(
        fusion,
        state,
        _messages(fusion, [{"declared_confidence": 0.3, "communication_cost": 0.3}]),
    )
    assert int(ignore.decision.route) == ROUTE_IGNORE

    query = _decide(
        fusion,
        state,
        _messages(fusion, [{"declared_confidence": 0.5, "communication_cost": 0.0}]),
    )
    assert int(query.decision.route) == ROUTE_QUERY
    assert int(query.decision.effective_action) == 0

    clarify = _decide(
        fusion,
        state,
        _messages(fusion, [{"declared_confidence": 0.1}]),
    )
    assert int(clarify.decision.route) == ROUTE_REQUEST_CLARIFICATION
    assert int(clarify.decision.effective_action) == 0

    accept = _decide(fusion, state, _messages(fusion, [{"declared_confidence": 1.0}]))
    assert int(accept.decision.route) == ROUTE_ACCEPT

    history_ready = _history_ready_zero_state(fusion)
    blend = _decide(
        fusion,
        history_ready,
        _messages(fusion, [{"declared_confidence": 0.5}]),
        decision_id=10,
        event_id=20,
        option=_option(available=True, action=3, score=0.3),
        base_score=0.1,
    )
    assert int(blend.decision.route) == ROUTE_BLEND
    assert int(blend.decision.scores.blend_selected_source) == SOURCE_OPTION_KEYBOARD
    assert int(blend.decision.effective_action) == 3
    assert bool(blend.decision.option_keyboard_influenced)
    assert not bool(blend.decision.partner_influenced)
    assert int(blend.decision.reliability_history_status) == (
        RELIABILITY_HISTORY_DEVELOPMENT_ONLINE
    )
    assert bool(blend.decision.reliability_history_available)
    assert not bool(blend.decision.uncalibrated_development_exploration)


def test_discrete_blend_selects_partner_by_score_without_averaging_action_ids() -> None:
    fusion = _fusion(partner_blend_weight=2.0)
    state = _history_ready_zero_state(fusion)
    result = _decide(
        fusion,
        state,
        _messages(fusion, [{"suggested_action": 3, "declared_confidence": 0.5}]),
        option=_option(available=True, action=2, score=0.3),
        base_score=0.1,
    )
    assert int(result.decision.route) == ROUTE_BLEND
    assert int(result.decision.scores.blend_selected_source) == SOURCE_PARTNER
    assert int(result.decision.effective_action) == 3
    assert int(result.decision.effective_action) in {0, 2, 3}
    assert bool(result.decision.partner_influenced)


def test_communication_cost_and_multiple_partners_change_selection() -> None:
    fusion = _fusion()
    messages = _messages(
        fusion,
        [
            {"partner_id": 0, "suggested_action": 1, "communication_cost": 0.2},
            {"partner_id": 1, "suggested_action": 2, "communication_cost": 0.0},
        ],
    )
    result = _decide(fusion, fusion.init(), messages)
    assert int(result.decision.selected_partner_id) == 1
    assert int(result.decision.effective_action) == 2
    assert float(result.decision.scores.predicted_net_value[1]) > float(
        result.decision.scores.predicted_net_value[0]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"decision_id": 11},
        {"event_id": 21},
        {"action": 2},
        {"partner_id": 1},
        {"available": False},
        {"assistance_available": False},
        {"safety_available": False},
        {"assistance": float("nan")},
    ],
)
def test_stale_duplicate_misattributed_or_incomplete_feedback_is_atomic_noop(
    changes: dict[str, int | float | bool],
) -> None:
    fusion = _fusion()
    armed = _decide(
        fusion,
        fusion.init(),
        _messages(fusion, [{"suggested_action": 1}]),
    ).state
    kwargs: dict[str, Any] = {"action": 1}
    kwargs.update(changes)
    result = fusion.apply_feedback(armed, _feedback(**kwargs))
    assert not bool(result.applied)
    _assert_tree_equal(result.state, armed)
    _assert_float_leaves_finite(result)


def test_feedback_uses_realized_value_and_safety_then_duplicate_is_noop() -> None:
    fusion = _fusion()
    armed = _decide(
        fusion,
        fusion.init(),
        _messages(fusion, [{"suggested_action": 1}]),
    ).state
    applied = fusion.apply_feedback(armed, _feedback(action=1, assistance=10_000.0, safe=True))
    assert bool(applied.applied)
    assert not bool(applied.state.feedback_armed)
    assert int(applied.state.feedback_counts[0]) == 1
    assert int(applied.state.safe_feedback_counts[0]) == 1
    assert int(applied.state.feedback_applied_count) == 1
    assert float(applied.realized_training_target) == pytest.approx(1.0)
    assert float(applied.parameter_update_l2_norm) > 0.0
    assert np.any(np.asarray(applied.state.reliability_weights[0]) != 0.0)

    duplicate = fusion.apply_feedback(
        applied.state, _feedback(action=1, assistance=10_000.0, safe=True)
    )
    assert not bool(duplicate.applied)
    _assert_tree_equal(duplicate.state, applied.state)
    assert "agreement" not in PartnerPolicyFusionFeedback.__annotations__


def test_observed_unsafe_outcome_overrides_positive_assistance_target() -> None:
    fusion = _fusion()
    armed = _decide(
        fusion,
        fusion.init(),
        _messages(fusion, [{"suggested_action": 1}]),
    ).state
    applied = fusion.apply_feedback(armed, _feedback(action=1, assistance=10_000.0, safe=False))
    assert bool(applied.applied)
    assert float(applied.realized_training_target) == 0.0
    assert int(applied.state.safe_feedback_counts[0]) == 0
    assert float(applied.state.reliability_weights[0, 0]) < 0.0


def test_pending_record_is_never_overwritten_and_old_binding_remains_usable() -> None:
    fusion = _fusion()
    first = _decide(
        fusion,
        fusion.init(),
        _messages(fusion, [{"suggested_action": 1}]),
    )
    second_messages = _messages(
        fusion,
        [{"suggested_action": 2}],
        decision_id=11,
        event_id=21,
    )
    second = _decide(
        fusion,
        first.state,
        second_messages,
        decision_id=11,
        event_id=21,
    )
    assert int(second.decision.effective_action) == 0
    assert not bool(second.decision.availability.feedback_slot_available)
    assert int(second.state.armed_decision_id) == 10
    assert int(second.state.armed_event_id) == 20
    applied = fusion.apply_feedback(second.state, _feedback(action=1))
    assert bool(applied.applied)


def test_contextual_realized_feedback_changes_partner_reliability_by_context() -> None:
    fusion = _fusion(
        max_partners=2,
        context_dim=2,
        n_actions=3,
        learning_rate=1.0,
        accept_net_value_threshold=-1.0,
        blend_net_value_threshold=-2.0,
        max_abs_context=1.0,
        min_feedback_for_learned_routing=2,
    )
    state = fusion.init()
    for index in range(4):
        positive = index % 2 == 0
        decision_id = index
        event_id = index
        context = (1.0, 0.0) if positive else (-1.0, 0.0)
        messages = _messages(
            fusion,
            [{"partner_id": 0, "suggested_action": 1}],
            decision_id=decision_id,
            event_id=event_id,
            observation_id=0,
            context_id=0,
        )
        decision = _decide(
            fusion,
            state,
            messages,
            decision_id=decision_id,
            event_id=event_id,
            observation_id=0,
            context_id=0,
            context=context,
            mask=(True, True, True),
        )
        assert bool(decision.decision.partner_influenced)
        feedback = _feedback(
            decision_id=decision_id,
            event_id=event_id,
            action=1,
            partner_id=0,
            assistance=10_000.0 if positive else -10_000.0,
            safe=positive,
        )
        state = fusion.apply_feedback(decision.state, feedback).state
        state.decision_count.block_until_ready()

    plus_messages = _messages(
        fusion,
        [{"partner_id": 0, "suggested_action": 1}],
        decision_id=100,
        event_id=100,
        observation_id=0,
        context_id=0,
    )
    plus = _decide(
        fusion,
        state,
        plus_messages,
        decision_id=100,
        event_id=100,
        observation_id=0,
        context_id=0,
        context=(1.0, 0.0),
        mask=(True, True, True),
    )
    minus_messages = _messages(
        fusion,
        [{"partner_id": 0, "suggested_action": 1}],
        decision_id=101,
        event_id=101,
        observation_id=0,
        context_id=0,
    )
    minus = _decide(
        fusion,
        state,
        minus_messages,
        decision_id=101,
        event_id=101,
        observation_id=0,
        context_id=0,
        context=(-1.0, 0.0),
        mask=(True, True, True),
    )
    assert float(plus.decision.scores.selected_predicted_reliability) > float(
        minus.decision.scores.selected_predicted_reliability
    )
    assert bool(plus.decision.reliability_history_available)


def test_jit_and_scan_match_eager_and_keep_outputs_finite() -> None:
    fusion = _fusion()
    state = fusion.init()
    kwargs: dict[str, Any] = {
        "decision_id": jnp.asarray(10, dtype=jnp.int32),
        "event_id": jnp.asarray(20, dtype=jnp.int32),
        "observation_id": jnp.asarray(30, dtype=jnp.int32),
        "context_id": jnp.asarray(40, dtype=jnp.int32),
        "context_features": jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        "base_action": jnp.asarray(0, dtype=jnp.int32),
        "base_declared_score": jnp.asarray(0.0, dtype=jnp.float32),
        "safety_action_mask": jnp.ones((4,), dtype=jnp.bool_),
        "option_proposal": fusion.empty_option_proposal(),
        "messages": _messages(fusion, [{"suggested_action": 1}]),
    }
    eager = fusion.decide(state, **kwargs)
    compiled = jax.jit(fusion.decide)(state, **kwargs)
    _assert_tree_equal(eager, compiled)
    compiled_feedback = jax.jit(fusion.apply_feedback)(
        eager.state, _feedback(action=1)
    )
    assert bool(compiled_feedback.applied)
    _assert_float_leaves_finite(compiled_feedback)

    length = 3
    empty_messages = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_messages(),
    )
    empty_options = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_option_proposal(),
    )
    final_state, decisions = jax.jit(fusion.decide_sequence)(
        state,
        decision_ids=jnp.arange(length, dtype=jnp.int32),
        event_ids=jnp.arange(length, dtype=jnp.int32),
        observation_ids=jnp.zeros((length,), dtype=jnp.int32),
        context_ids=jnp.zeros((length,), dtype=jnp.int32),
        context_features=jnp.zeros((length, 2), dtype=jnp.float32),
        base_actions=jnp.zeros((length,), dtype=jnp.int32),
        base_declared_scores=jnp.zeros((length,), dtype=jnp.float32),
        safety_action_masks=jnp.ones((length, 4), dtype=jnp.bool_),
        option_proposals=empty_options,
        messages=empty_messages,
    )
    assert int(final_state.decision_count) == length
    np.testing.assert_array_equal(np.asarray(decisions.effective_action), np.zeros(length))
    _assert_float_leaves_finite(decisions)

    repeated_feedback = jax.tree_util.tree_map(
        lambda leaf: jnp.stack((leaf, leaf)),
        _feedback(action=1),
    )
    feedback_state, feedback_results = jax.jit(fusion.feedback_sequence)(
        eager.state, repeated_feedback
    )
    np.testing.assert_array_equal(
        np.asarray(feedback_results.applied), np.asarray([True, False])
    )
    assert int(feedback_state.feedback_applied_count) == 1
    _assert_float_leaves_finite(feedback_results)


def test_saturating_counters_remain_finite_and_state_valid() -> None:
    fusion = _fusion(counter_cap=4, min_feedback_for_learned_routing=2)
    base = fusion.init()
    armed = _replace(
        base,
        feedback_counts=jnp.asarray([4, 0, 0], dtype=jnp.int32),
        safe_feedback_counts=jnp.asarray([4, 0, 0], dtype=jnp.int32),
        decision_count=jnp.asarray(4, dtype=jnp.int32),
        feedback_applied_count=jnp.asarray(4, dtype=jnp.int32),
        has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
        last_decision_id=jnp.asarray(10, dtype=jnp.int32),
        last_event_id=jnp.asarray(20, dtype=jnp.int32),
        feedback_armed=jnp.asarray(True, dtype=jnp.bool_),
        armed_decision_id=jnp.asarray(10, dtype=jnp.int32),
        armed_event_id=jnp.asarray(20, dtype=jnp.int32),
        armed_action=jnp.asarray(1, dtype=jnp.int32),
        armed_partner_id=jnp.asarray(0, dtype=jnp.int32),
        armed_route=jnp.asarray(ROUTE_ACCEPT, dtype=jnp.int32),
        armed_model_features=jnp.asarray([1.0, 0.25, -0.5, 1.0], dtype=jnp.float32),
        armed_predicted_reliability=jnp.asarray(0.5, dtype=jnp.float32),
    )
    fusion.validate_state(armed)
    result = fusion.apply_feedback(armed, _feedback(action=1))
    assert bool(result.applied)
    assert bool(result.counter_saturated)
    assert int(result.state.feedback_counts[0]) == 4
    assert int(result.state.safe_feedback_counts[0]) == 4
    assert int(result.state.feedback_applied_count) == 4
    fusion.validate_state(result.state)


def test_checkpoint_roundtrip_and_tamper_detection_are_strict() -> None:
    fusion = _fusion()
    armed = _decide(
        fusion,
        fusion.init(),
        _messages(fusion, [{"suggested_action": 1}]),
    ).state
    payload = fusion.checkpoint_payload(armed)
    assert payload["schema"] == PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA
    restored_fusion, restored_state = PartnerPolicyFusion.from_checkpoint_payload(payload)
    assert restored_fusion.to_config() == fusion.to_config()
    _assert_tree_equal(restored_state, armed)

    for field in ("config_digest", "state_digest", "resource_budget"):
        changed = copy.deepcopy(payload)
        changed[field] = "tampered"
        with pytest.raises(ValueError):
            PartnerPolicyFusion.from_checkpoint_payload(changed)

    changed = copy.deepcopy(payload)
    cast(dict[str, Any], changed["state"])["decision_count"] = -1
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_checkpoint_payload(changed)

    changed = copy.deepcopy(payload)
    cast(dict[str, Any], changed["state"])["unexpected"] = 1
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_checkpoint_payload(changed)


def test_state_tamper_and_static_shape_mismatch_fail_strict_validation() -> None:
    fusion = _fusion()
    with pytest.raises(ValueError):
        fusion.validate_state(
            _replace(
                fusion.init(),
                reliability_weights=jnp.full((3, 4), jnp.nan, dtype=jnp.float32),
            )
        )
    with pytest.raises(ValueError):
        fusion.validate_state(
            _replace(
                fusion.init(),
                feedback_counts=jnp.zeros((2,), dtype=jnp.int32),
            )
        )
