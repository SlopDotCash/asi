"""Hostile-identity tests for partner policy fusion checkpoint consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.partner_policy_fusion import PartnerPolicyFusion


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __contains__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __contains__")

    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile keys")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")


def test_from_checkpoint_payload_rejects_hostile_top_level_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "schema": "x",
            "mechanism_status": "x",
            "scientific_promotion_allowed": False,
            "fusion": {},
            "config_digest": "x",
            "resource_budget": {},
            "state": {},
            "state_digest": "x",
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_checkpoint_payload(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_from_checkpoint_payload_rejects_hostile_nested_without_dispatch() -> None:
    hostile_inner = _HostileMapping({"type": "PartnerPolicyFusionConfig"})
    # Need a top-level exact dict but nested is hostile
    payload = {
        "schema": "alberta.partner_policy_fusion.v1",
        "mechanism_status": "development",
        "scientific_promotion_allowed": False,
        "fusion": hostile_inner,
        "config_digest": "dummy",
        "resource_budget": {},
        "state": {},
        "state_digest": "dummy",
    }
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        PartnerPolicyFusion.from_checkpoint_payload(payload)  # type: ignore[arg-type]
    # hostile must not be dispatched before exact-dict check
    # nested hostile must not be iterated via digest
    # So calls should remain 0 (hostile inner not iterated)
    assert _HostileMapping.calls == 0
