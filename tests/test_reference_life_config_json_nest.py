"""Reject deep life-config JSON before json.loads RecursionError.

Origin ``ReferenceLifeConfig`` loads ``_config_json`` with ``json.loads``
and no nesting preflight. A 16_000-deep object nest RecursionErrors the
C decoder on origin/main. Overlay fail-closes at the shared 64-deep JSON
ceiling before parse.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.reference_life import (
    _MAX_JSON_NESTING_DEPTH,
    _require_config_json,
    build_prototype_switching_life,
)
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig

pytestmark = pytest.mark.unit


def _nested_json(depth: int) -> str:
    open_obj = chr(123) + chr(34) + "k" + chr(34) + ":"
    return open_obj * depth + "0" + chr(125) * depth


def _agent_config() -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=0.0,
            )
        )
    )


def _last_fit_runner():
    return build_prototype_switching_life(
        agent_config=_agent_config(),
        environment_config=SwitchingTwoStateConfig(phase_length=2),
        lifecycle_id="prototype.0000000100000002",
        seed=7,
        max_accepted_events=3,
    )


def test_frozen_life_config_json_nest_bound() -> None:
    assert _MAX_JSON_NESTING_DEPTH == 64


def test_last_fit_life_config_still_validates() -> None:
    runner = _last_fit_runner()
    assert runner.config.config["environment"]["config"]["phase_length"] == 2


def test_last_fit_json_text_still_encodes() -> None:
    _require_config_json(_nested_json(_MAX_JSON_NESTING_DEPTH), name="life configuration")


def test_first_overflow_nest_is_value_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        _require_config_json(
            _nested_json(_MAX_JSON_NESTING_DEPTH + 1),
            name="life configuration",
        )


def test_origin_recursion_class_rejects_before_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _last_fit_runner()

    def fail_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("json.loads ran before the life-config nest gate")

    monkeypatch.setattr(json, "loads", fail_loads)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting limit"):
        dataclasses.replace(runner.config, _config_json=_nested_json(16_000))
    assert time.perf_counter() - started < 0.25
