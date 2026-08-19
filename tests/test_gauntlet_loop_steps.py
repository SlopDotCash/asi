"""Protocol step ceilings for public gauntlet scans.

Documented last-fit is the default nine-segment program (``9 * 3000``).
Origin handed ``10**12`` to ``jnp.arange`` with no reject — hang/OOM, not
an INT32 leftover.
"""

from __future__ import annotations

from typing import Any, cast

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.gauntlet import (
    _GAUNTLET_LOOP_MAX_STEPS,
    NUM_SEGMENTS,
    _require_gauntlet_loop_steps,
    run_gauntlet,
    run_gauntlet_batched,
)


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

    monkeypatch.setattr("alberta_framework.streams.gauntlet.jnp.arange", spy)
    return seen


def test_documented_protocol_ceiling_matches_default_program() -> None:
    assert NUM_SEGMENTS == 9
    assert _GAUNTLET_LOOP_MAX_STEPS == 27_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert _require_gauntlet_loop_steps("num_steps", _GAUNTLET_LOOP_MAX_STEPS) == 27_000


def test_trillion_steps_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_gauntlet(cast(Any, object()), cast(Any, object()), 10**12, jr.key(0))
    assert seen == []


def test_int32_max_steps_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_gauntlet(cast(Any, object()), cast(Any, object()), 2**31 - 1, jr.key(0))
    assert seen == []


def test_first_overflow_rejected_before_arange(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match=r"num_steps must be an integer in \[1, 27000\]"):
        run_gauntlet(
            cast(Any, object()),
            cast(Any, object()),
            _GAUNTLET_LOOP_MAX_STEPS + 1,
            jr.key(0),
        )
    assert seen == []


def test_batched_trillion_steps_rejected_before_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_arange(monkeypatch)
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        run_gauntlet_batched(
            cast(Any, object()),
            cast(Any, object()),
            10**12,
            cast(Any, object()),
        )
    assert seen == []


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 27_000.0])
def test_rejects_non_exact_or_non_positive_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_gauntlet_loop_steps("num_steps", value)


def test_rejects_numpy_and_subclass_step_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_gauntlet_loop_steps("num_steps", np.int64(10))
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_gauntlet_loop_steps("num_steps", _HostileInt(10))
