"""Protocol step ceiling for option-duration trajectory scans.

Documented last-fit is README / package-init ``num_steps=10_000``. Origin
scanned ``observations.shape[0]`` with no reject — hang/OOM, not constructor
256 MiB leftover.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from alberta_framework.core.option_value_duration import (
    _OPTION_DURATION_SCAN_MAX_STEPS,
    OptionValueDurationLearner,
    _require_option_duration_scan_steps,
    run_option_value_duration_from_arrays,
)

pytestmark = [pytest.mark.unit]


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn: Any, init: Any, xs: Any, **kwargs: Any) -> Any:
        first = xs[0] if isinstance(xs, tuple) else xs
        length = int(getattr(first, "shape", (0,))[0])
        seen.append(length)
        raise AssertionError(f"jax.lax.scan must not run: T={length}")

    monkeypatch.setattr(
        "alberta_framework.core.option_value_duration.jax.lax.scan", spy
    )
    return seen


def _arrays(steps: int) -> tuple[Any, Any, Any, Any, Any]:
    return (
        jnp.zeros((steps, 2), dtype=jnp.float32),
        jnp.zeros((steps,), dtype=jnp.int32),
        jnp.zeros((steps,), dtype=jnp.float32),
        jnp.zeros((steps, 2), dtype=jnp.float32),
        jnp.zeros((steps,), dtype=jnp.float32),
    )


def test_documented_protocol_ceiling_matches_public_scan_example() -> None:
    assert _OPTION_DURATION_SCAN_MAX_STEPS == 10_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert (
        _require_option_duration_scan_steps(
            "num_steps", _OPTION_DURATION_SCAN_MAX_STEPS
        )
        == 10_000
    )


def test_first_overflow_protocol_step_count_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    learner = OptionValueDurationLearner(1)
    state = learner.init(2)
    with pytest.raises(ValueError, match=r"num_steps must be an integer in \[1, 10000\]"):
        run_option_value_duration_from_arrays(
            learner, state, *_arrays(_OPTION_DURATION_SCAN_MAX_STEPS + 1)
        )
    assert seen == []


def test_int32_max_steps_rejected_by_require_helper() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_option_duration_scan_steps("num_steps", 2**31 - 1)


def test_trillion_steps_rejected_by_require_helper() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_option_duration_scan_steps("num_steps", 10**12)


def test_existing_two_step_runner_still_scans() -> None:
    learner = OptionValueDurationLearner(1)
    state = learner.init(2)
    result = run_option_value_duration_from_arrays(learner, state, *_arrays(2))
    assert result.predictions.shape[0] == 2


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0])
def test_rejects_non_exact_or_non_positive_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_option_duration_scan_steps("num_steps", value)
