"""Leftover-identity gates for security reward-weight records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.security import SecurityRewardWeights


def test_security_reward_weights_reject_leftover_identities() -> None:
    """Public reward-weight records must not keep leftover bool identities."""

    with pytest.raises(ValueError, match="threat_blocked"):
        SecurityRewardWeights(threat_blocked=True)
    with pytest.raises(ValueError, match="false_positive"):
        SecurityRewardWeights(false_positive=False)
    with pytest.raises(ValueError, match="recovery"):
        SecurityRewardWeights(recovery=True)

    legal = SecurityRewardWeights()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"threat_blocked": 1.0' in dumped
    assert '"recovery": 0.5' in dumped
    assert '"threat_blocked": true' not in dumped
    assert '"recovery": true' not in dumped
