"""Leftover-identity gates for security reward-weight records."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
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


@pytest.mark.parametrize("field", tuple(field.name for field in fields(SecurityRewardWeights)))
@pytest.mark.parametrize("value", ["1.0", float("nan"), float("inf")])
def test_security_reward_weights_reject_non_numeric_or_nonfinite_fields(
    field: str,
    value: object,
) -> None:
    """Invalid weights must fail at record construction, before metadata export."""

    with pytest.raises(ValueError, match=field):
        SecurityRewardWeights(**{field: value})  # type: ignore[arg-type]


def test_security_reward_weights_store_the_float32_values_used_by_reward_math() -> None:
    """Persisted weight metadata must match the values consumed by the adapter."""

    weights = SecurityRewardWeights(threat_blocked=1, latency_cost=np.float64(-0.1))

    assert type(weights.threat_blocked) is float
    assert type(weights.latency_cost) is float
    assert weights.threat_blocked == 1.0
    assert weights.latency_cost == float(np.float32(-0.1))
    assert weights.to_dict()["latency_cost"] == float(np.float32(-0.1))
