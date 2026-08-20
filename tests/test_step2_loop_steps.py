"""Protocol ceilings for public Step 2 collection and smoke scans.

Documented last-fit in tests is 128 smoke steps. Origin accepted INT32_MAX and
looped ``range(steps)`` with no last-fit reject — hang, not leftover INT32 math.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alberta_framework.steps.step2 import (
    _STEP2_LOOP_MAX_STEPS,
    _require_step2_loop_steps,
    collect_step2_arrays,
    run_step2_associative_smoke,
    run_step2_smoke,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


def _spy_range(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        seen.append((args, kwargs))
        raise AssertionError(f"range must not run: {args} {kwargs}")

    monkeypatch.setattr("builtins.range", spy)
    return seen


def test_documented_protocol_ceiling() -> None:
    assert _STEP2_LOOP_MAX_STEPS == 10_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert _require_step2_loop_steps("steps", _STEP2_LOOP_MAX_STEPS) == 10_000


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0, 10**12, 2**31 - 1])
def test_rejects_non_exact_or_oversized_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="steps must be an integer in"):
        _require_step2_loop_steps("steps", value)


def test_rejects_numpy_and_subclass_step_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="steps must be an integer in"):
        _require_step2_loop_steps("steps", np.int64(10))
    with pytest.raises(ValueError, match="steps must be an integer in"):
        _require_step2_loop_steps("steps", _HostileInt(10))


def test_trillion_steps_rejected_before_range(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_range(monkeypatch)
    with pytest.raises(ValueError, match="steps must be an integer in"):
        collect_step2_arrays(object(), steps=10**12, key=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps must be an integer in"):
        run_step2_smoke(steps=10**12)
    assert seen == []


def test_associative_smoke_rejects_oversized_steps_before_allocation() -> None:
    with pytest.raises(ValueError, match=r"steps must be an integer in \[2, 10000\]"):
        run_step2_associative_smoke(steps=10**12)
    with pytest.raises(ValueError, match=r"steps must be an integer in \[2, 10000\]"):
        run_step2_associative_smoke(steps=1)
