# mypy: disable-error-code="call-arg"
"""Protocol sequence ceilings for partner-policy fusion scans.

Origin scanned ``T=2048`` with no reject. The constructor already bounds
``max_partners`` at 1024; sequence T is the remaining unbounded axis.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.partner_policy_fusion import (
    _FUSION_SEQUENCE_MAX_STEPS,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
    _require_fusion_sequence_vector,
)


class _HostileVector:
    calls = 0

    @property
    def ndim(self) -> Any:
        type(self).calls += 1
        raise AssertionError("ndim hook executed")

    @property
    def shape(self) -> Any:
        type(self).calls += 1
        raise AssertionError("shape hook executed")


def _fusion() -> PartnerPolicyFusion:
    return PartnerPolicyFusion(
        PartnerPolicyFusionConfig(
            max_partners=3,
            context_dim=2,
            n_actions=4,
            max_abs_context=1.0,
        )
    )


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn: Any, init: Any, xs: Any, **kwargs: Any) -> Any:
        first = xs[0] if isinstance(xs, tuple) else xs
        length = int(getattr(first, "shape", (0,))[0])
        seen.append(length)
        raise AssertionError(f"jax.lax.scan must not run: T={length}")

    monkeypatch.setattr("alberta_framework.core.partner_policy_fusion.jax.lax.scan", spy)
    return seen


def test_protocol_ceiling_is_the_named_non_int32_constructor_bound() -> None:
    assert _FUSION_SEQUENCE_MAX_STEPS == 1_024


def test_last_fit_sequence_length_is_accepted() -> None:
    vector = jnp.arange(_FUSION_SEQUENCE_MAX_STEPS, dtype=jnp.int32)
    assert _require_fusion_sequence_vector("decision_ids", vector) == 1_024


def test_first_overflow_sequence_length_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    fusion = _fusion()
    length = _FUSION_SEQUENCE_MAX_STEPS + 1
    empty_messages = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_messages(),
    )
    empty_options = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_option_proposal(),
    )
    with pytest.raises(ValueError, match=r"decision_ids length must be an integer in \[1, 1024\]"):
        fusion.decide_sequence(
            fusion.init(),
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
    assert seen == []


def test_origin_hang_class_2048_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    fusion = _fusion()
    length = 2048
    empty_messages = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_messages(),
    )
    empty_options = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        fusion.empty_option_proposal(),
    )
    with pytest.raises(ValueError, match="decision_ids length must be an integer in"):
        fusion.decide_sequence(
            fusion.init(),
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
    assert seen == []


def test_feedback_sequence_overflow_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    fusion = _fusion()
    length = _FUSION_SEQUENCE_MAX_STEPS + 1
    feedback = PartnerPolicyFusionFeedback(
        available=jnp.asarray(True, dtype=jnp.bool_),
        decision_id=jnp.asarray(10, dtype=jnp.int32),
        executed_event_id=jnp.asarray(20, dtype=jnp.int32),
        executed_action=jnp.asarray(1, dtype=jnp.int32),
        partner_id=jnp.asarray(0, dtype=jnp.int32),
        assistance_value_available=jnp.asarray(True, dtype=jnp.bool_),
        realized_assistance_value=jnp.asarray(1.0, dtype=jnp.float32),
        safety_outcome_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_outcome_ok=jnp.asarray(True, dtype=jnp.bool_),
    )
    stacked = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (length, *leaf.shape)),
        feedback,
    )
    with pytest.raises(ValueError, match="feedback.available length must be an integer in"):
        fusion.feedback_sequence(fusion.init(), stacked)
    assert seen == []


def test_exact_gate_does_not_read_hostile_ndim_or_shape() -> None:
    _HostileVector.calls = 0
    with pytest.raises(TypeError, match="decision_ids must be a JAX array"):
        _require_fusion_sequence_vector("decision_ids", _HostileVector())
    assert _HostileVector.calls == 0
