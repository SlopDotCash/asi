"""Protocol step ceiling for public feature-discovery scans.

Documented last-fit is README / package-init ``num_steps=10_000``. Origin handed
``10**12`` and ``2**31-1`` to ``jnp.arange`` with no reject — hang/OOM, not an
INT32 leftover.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alberta_framework.core.feature_discovery import (
    _FEATURE_DISCOVERY_LOOP_MAX_STEPS,
    _require_feature_discovery_loop_steps,
    run_feature_discovery_loop,
)

pytestmark = [pytest.mark.unit]


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


def _spy_arange(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        seen.append((args, kwargs))
        raise AssertionError(f"jnp.arange must not run: {args} {kwargs}")

    monkeypatch.setattr("alberta_framework.core.feature_discovery.jnp.arange", spy)
    return seen


def test_documented_protocol_ceiling_matches_public_scan_example() -> None:
    assert _FEATURE_DISCOVERY_LOOP_MAX_STEPS == 10_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert (
        _require_feature_discovery_loop_steps(
            "num_steps", _FEATURE_DISCOVERY_LOOP_MAX_STEPS
        )
        == 10_000
    )


def test_first_overflow_protocol_step_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match=r"num_steps must be an integer in \[1, 10000\]"):
        run_feature_discovery_loop(
            object(),  # type: ignore[arg-type]
            object(),
            _FEATURE_DISCOVERY_LOOP_MAX_STEPS + 1,
            object(),  # type: ignore[arg-type]
        )
    assert seen == []


def test_trillion_steps_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_feature_discovery_loop(
            object(),  # type: ignore[arg-type]
            object(),
            10**12,
            object(),  # type: ignore[arg-type]
        )
    assert seen == []


def test_int32_max_steps_rejected_before_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_feature_discovery_loop(
            object(),  # type: ignore[arg-type]
            object(),
            2**31 - 1,
            object(),  # type: ignore[arg-type]
        )
    assert seen == []


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0])
def test_rejects_non_exact_or_non_positive_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_feature_discovery_loop_steps("num_steps", value)


def test_rejects_numpy_and_subclass_step_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_feature_discovery_loop_steps("num_steps", np.int64(10))
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_feature_discovery_loop_steps("num_steps", _HostileInt(10))
