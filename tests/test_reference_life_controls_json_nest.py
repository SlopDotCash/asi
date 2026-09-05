"""Reject deep oracle environment JSON before json.loads RecursionError.

Origin ``AnalyticOracleReferenceConfig`` loads ``environment_config_json``
with ``json.loads`` and no nesting preflight. A 16_000-deep object nest
RecursionError's the C decoder on origin/main. Overlay fail-closes at the
shared 64-deep JSON ceiling before parse.
"""

from __future__ import annotations

import json
import time

import pytest

from alberta_framework.reference_life_controls import (
    _MAX_JSON_NESTING_DEPTH,
    AnalyticOracleReferenceConfig,
    _require_oracle_config_json,
)
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig

pytestmark = pytest.mark.unit


def _nested_json(depth: int) -> str:
    return '{"k":' * depth + "0" + "}" * depth


def test_frozen_oracle_config_json_nest_bound() -> None:
    assert _MAX_JSON_NESTING_DEPTH == 64


def test_last_fit_oracle_config_still_validates() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    config = AnalyticOracleReferenceConfig.for_switching(environment, horizon=2)
    assert config.environment_kind == "switching_two_state"
    assert config.environment_config["phase_length"] == 2


def test_last_fit_json_text_still_encodes() -> None:
    _require_oracle_config_json(_nested_json(_MAX_JSON_NESTING_DEPTH))


def test_first_overflow_nest_is_value_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        _require_oracle_config_json(_nested_json(_MAX_JSON_NESTING_DEPTH + 1))


def test_origin_recursion_class_rejects_before_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("json.loads ran before the oracle-config nest gate")

    monkeypatch.setattr(json, "loads", fail_loads)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting limit"):
        AnalyticOracleReferenceConfig(
            environment_kind="switching_two_state",
            observation_dim=2,
            n_actions=2,
            horizon=2,
            policy_sha256="0" * 64,
            environment_config_json=_nested_json(16_000),
        )
    assert time.perf_counter() - started < 0.25
