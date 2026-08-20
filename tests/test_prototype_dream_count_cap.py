"""Protocol ceiling for Prototype Dyna dream scans.

Public last-fit is n_dreams_per_step=10_000 (tests use 0-4). Origin accepted
INT32-legal counts and scanned jnp.arange(n_dreams) — hang, not leftover INT32.
"""

from __future__ import annotations

import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    _MAX_DREAMS_PER_STEP,
    PrototypeAgentConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig


def _oak() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=4,
            n_primitive_actions=2,
        )
    )


def _world_model() -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=4,
        n_actions=2,
        hidden_sizes=(),
    )


def test_documented_protocol_ceiling() -> None:
    assert _MAX_DREAMS_PER_STEP == 10_000


def test_last_fit_dream_count_is_accepted() -> None:
    cfg = PrototypeAgentConfig(
        oak=_oak(),
        world_model=_world_model(),
        n_dreams_per_step=_MAX_DREAMS_PER_STEP,
    )
    assert cfg.n_dreams_per_step == 10_000


@pytest.mark.parametrize("value", [10_001, 2**31 - 1])
def test_rejects_oversized_dream_counts(value: int) -> None:
    with pytest.raises(ValueError, match="n_dreams_per_step"):
        PrototypeAgentConfig(oak=_oak(), n_dreams_per_step=value)
