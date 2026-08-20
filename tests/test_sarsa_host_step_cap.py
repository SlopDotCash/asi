"""Protocol ceilings for SARSA Python env loops.

Public last-fit is max_steps/num_steps=10_000 (tests use 500). Origin accepted
INT32-legal counts and handed them to range() — hang, not leftover INT32 math.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alberta_framework.core.sarsa import (
    _SARSA_SEQUENCE_MAX_STEPS,
    _require_sarsa_host_steps,
    run_sarsa_continuing,
    run_sarsa_episode,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


class _BoomEnv:
    def reset(self, *args: object, **kwargs: object) -> tuple[Any, dict[str, object]]:
        raise AssertionError("env.reset must not run after an oversized step count")

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, object]]:
        raise AssertionError("env.step must not run after an oversized step count")


def test_documented_protocol_ceiling() -> None:
    assert _SARSA_SEQUENCE_MAX_STEPS == 10_000


def test_last_fit_host_steps_are_accepted() -> None:
    assert _require_sarsa_host_steps("max_steps", _SARSA_SEQUENCE_MAX_STEPS) == 10_000


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0, 10_001, 10**12, 2**31 - 1])
def test_rejects_non_exact_or_oversized_host_steps(value: object) -> None:
    with pytest.raises(ValueError, match="max_steps must be an integer in"):
        _require_sarsa_host_steps("max_steps", value)


def test_rejects_numpy_and_subclass_host_steps_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="max_steps must be an integer in"):
        _require_sarsa_host_steps("max_steps", np.int64(10))
    with pytest.raises(ValueError, match="max_steps must be an integer in"):
        _require_sarsa_host_steps("max_steps", _HostileInt(10))


def test_episode_rejects_overflow_before_env_reset() -> None:
    with pytest.raises(ValueError, match="max_steps must be an integer in"):
        run_sarsa_episode(None, None, _BoomEnv(), max_steps=10_001)  # type: ignore[arg-type]


def test_continuing_rejects_overflow_before_env_reset() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_sarsa_continuing(None, None, _BoomEnv(), num_steps=2**31 - 1)  # type: ignore[arg-type]
