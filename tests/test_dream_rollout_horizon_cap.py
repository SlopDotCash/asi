"""Protocol ceilings for public dream rollout scans.

Documented last-fit in tests is rollout_horizon=5. Origin handed large
horizons to jnp.arange with no last-fit reject — hang/OOM, not INT32 leftover.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alberta_framework.core.dreaming import (
    _DREAM_ROLLOUT_MAX_HORIZON,
    _require_dream_rollout_horizon,
    dream_rollout,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


class _HorizonConfig:
    def __init__(self, rollout_horizon: object) -> None:
        self.rollout_horizon = rollout_horizon


def _spy_arange(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        seen.append((args, kwargs))
        raise AssertionError(f"jnp.arange must not run: {args} {kwargs}")

    monkeypatch.setattr("alberta_framework.core.dreaming.jnp.arange", spy)
    return seen


def test_documented_protocol_ceiling() -> None:
    assert _DREAM_ROLLOUT_MAX_HORIZON == 10_000


def test_last_fit_protocol_horizon_is_accepted() -> None:
    assert _require_dream_rollout_horizon(_DREAM_ROLLOUT_MAX_HORIZON) == 10_000


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0, 10**12, 2**31 - 1])
def test_rejects_non_exact_or_oversized_horizons(value: object) -> None:
    with pytest.raises(ValueError, match="rollout_horizon must be an integer in"):
        _require_dream_rollout_horizon(value)


def test_rejects_numpy_and_subclass_horizons_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="rollout_horizon must be an integer in"):
        _require_dream_rollout_horizon(np.int64(10))
    with pytest.raises(ValueError, match="rollout_horizon must be an integer in"):
        _require_dream_rollout_horizon(_HostileInt(10))


def test_trillion_horizon_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="rollout_horizon must be an integer in"):
        dream_rollout(
            None,  # type: ignore[arg-type]
            None,
            None,  # type: ignore[arg-type]
            None,
            None,  # type: ignore[arg-type]
            _HorizonConfig(10**12),  # type: ignore[arg-type]
        )
    assert seen == []


def test_int32_max_horizon_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="rollout_horizon must be an integer in"):
        dream_rollout(
            None,  # type: ignore[arg-type]
            None,
            None,  # type: ignore[arg-type]
            None,
            None,  # type: ignore[arg-type]
            _HorizonConfig(2**31 - 1),  # type: ignore[arg-type]
        )
    assert seen == []
