"""Leftover-identity gates for security oracle-experience records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.security import SecurityAction, SecurityOracleExperience


def _legal(**overrides: object) -> SecurityOracleExperience:
    payload: dict[str, object] = {
        "state": (0.5,),
        "action": SecurityAction.PASS,
        "reward": 0.0,
        "outcome": {"label": "safe"},
    }
    payload.update(overrides)
    return SecurityOracleExperience(**payload)  # type: ignore[arg-type]


def test_security_oracle_experience_rejects_leftover_identities() -> None:
    """Public oracle records must not keep leftover bool/string identities."""

    with pytest.raises(ValueError, match="reward"):
        _legal(reward=True)
    with pytest.raises(ValueError, match="reward"):
        _legal(reward=False)
    with pytest.raises(ValueError, match="reward"):
        _legal(reward="FIXED")
    with pytest.raises(ValueError, match="action"):
        _legal(action=0)
    with pytest.raises(ValueError, match="action"):
        _legal(action=True)
    with pytest.raises(ValueError, match="schema"):
        _legal(schema=True)
    with pytest.raises(ValueError, match="exact tuple"):
        _legal(state=[0.5])
    with pytest.raises(ValueError, match=r"state\[0\]"):
        _legal(state=(True,))

    legal = _legal()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert dumped.count('"reward": 0.0') == 1
    assert '"reward": true' not in dumped
    assert '"reward": "FIXED"' not in dumped
    assert '"schema": true' not in dumped
    assert '"state": [true]' not in dumped
    assert legal.action is SecurityAction.PASS
    assert type(legal.reward) is float
