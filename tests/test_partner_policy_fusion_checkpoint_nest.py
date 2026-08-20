"""Reject deep checkpoint mappings before json.dumps RecursionError.

Origin ``PartnerPolicyFusion.from_checkpoint_payload`` digest-binds the
caller ``fusion`` mapping with ``json.dumps`` and no nesting preflight.
A 16_000-deep object nest RecursionError's the C encoder on origin/main.
Overlay fail-closes at the shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time

import pytest

from alberta_framework.core.partner_policy_fusion import (
    _CHECKPOINT_JSON_MAX_DEPTH,
    MECHANISM_STATUS,
    PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    _canonical_json_bytes,
)

pytestmark = pytest.mark.unit


def _nest(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"x": node}
    return node


def _hostile_checkpoint(fusion: object) -> dict[str, object]:
    return {
        "schema": PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
        "mechanism_status": MECHANISM_STATUS,
        "scientific_promotion_allowed": False,
        "fusion": fusion,
        "config_digest": "0" * 64,
        "resource_budget": {},
        "state": {},
        "state_digest": "0" * 64,
    }


def test_frozen_checkpoint_json_nest_bound() -> None:
    assert _CHECKPOINT_JSON_MAX_DEPTH == 32


def test_last_fit_checkpoint_still_roundtrips() -> None:
    fusion = PartnerPolicyFusion(
        PartnerPolicyFusionConfig(
            max_partners=3,
            context_dim=2,
            n_actions=4,
            max_abs_context=1.0,
        )
    )
    payload = fusion.checkpoint_payload(fusion.init())
    restored_fusion, restored_state = PartnerPolicyFusion.from_checkpoint_payload(payload)
    assert restored_fusion.to_config() == fusion.to_config()
    assert int(restored_state.decision_count) == 0


def test_last_fit_json_chain_still_encodes() -> None:
    encoded = _canonical_json_bytes(_nest(_CHECKPOINT_JSON_MAX_DEPTH - 1))
    assert encoded.startswith(b"{")


def test_origin_recursion_class_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("json.dumps ran before the checkpoint nest gate")

    monkeypatch.setattr(json, "dumps", fail_dumps)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting depth"):
        PartnerPolicyFusion.from_checkpoint_payload(_hostile_checkpoint(_nest(16_000)))
    assert time.perf_counter() - started < 0.25
