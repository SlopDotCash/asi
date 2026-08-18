"""Leftover-identity gates for security rollout-step records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.security import SecurityAction, SecurityRolloutStep


def _legal_step(**overrides: object) -> SecurityRolloutStep:
    payload: dict[str, object] = {
        "state": (0.0, 1.0),
        "action": SecurityAction.PASS,
        "reward": 0.5,
        "next_state": (0.0, 1.0),
        "terminated": False,
        "truncated": False,
    }
    payload.update(overrides)
    return SecurityRolloutStep(**payload)  # type: ignore[arg-type]


def test_security_rollout_step_rejects_leftover_identities() -> None:
    """Public rollout records must not keep leftover bool/string identities."""

    with pytest.raises(ValueError, match="reward"):
        _legal_step(reward=True)
    with pytest.raises(ValueError, match="reward"):
        _legal_step(reward=False)
    with pytest.raises(ValueError, match="terminated"):
        _legal_step(terminated=1)
    with pytest.raises(ValueError, match="truncated"):
        _legal_step(truncated=0)
    with pytest.raises(ValueError, match="action"):
        _legal_step(action=True)
    with pytest.raises(ValueError, match="action"):
        _legal_step(action="FIXED")
    with pytest.raises(ValueError, match="action"):
        _legal_step(action="BLOCK")

    legal = _legal_step()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"reward": 0.5' in dumped
    assert '"terminated": false' in dumped
    assert '"reward": true' not in dumped
    assert '"terminated": 1' not in dumped
    assert legal.action is SecurityAction.PASS
