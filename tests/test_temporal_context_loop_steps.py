"""Protocol step ceilings for public temporal-context scans.

Documented last-fit in tests is 3 array steps. Origin scanned the leading
observation axis with no reject — hang/OOM, not an INT32 leftover.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.temporal_context import (
    _TEMPORAL_CONTEXT_LOOP_MAX_STEPS,
    TemporalContextConfig,
    TemporalContextFeaturizer,
    _require_temporal_context_array_steps,
    _require_temporal_context_loop_steps,
    transform_temporal_context_arrays,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


class _HostileArrays:
    calls = 0

    @property
    def ndim(self) -> Any:
        type(self).calls += 1
        raise AssertionError("ndim hook executed")

    @property
    def shape(self) -> Any:
        type(self).calls += 1
        raise AssertionError("shape hook executed")


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def spy(*args: object, **kwargs: object) -> Any:
        seen.append((args, kwargs))
        raise AssertionError(f"jax.lax.scan must not run: {args} {kwargs}")

    monkeypatch.setattr("alberta_framework.core.temporal_context.jax.lax.scan", spy)
    return seen


def test_documented_protocol_ceiling() -> None:
    assert _TEMPORAL_CONTEXT_LOOP_MAX_STEPS == 10_000


def test_last_fit_protocol_step_count_is_accepted() -> None:
    assert (
        _require_temporal_context_loop_steps(
            "num_steps", _TEMPORAL_CONTEXT_LOOP_MAX_STEPS
        )
        == 10_000
    )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 10_000.0, 10**12, 2**31 - 1])
def test_rejects_non_exact_or_oversized_step_counts(value: object) -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_temporal_context_loop_steps("num_steps", value)


def test_rejects_numpy_and_subclass_step_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_temporal_context_loop_steps("num_steps", np.int64(10))
    with pytest.raises(ValueError, match="num_steps must be an integer in"):
        _require_temporal_context_loop_steps("num_steps", _HostileInt(10))


def test_array_last_fit_length_is_accepted() -> None:
    observations = jnp.zeros((_TEMPORAL_CONTEXT_LOOP_MAX_STEPS, 2), dtype=jnp.float32)
    assert _require_temporal_context_array_steps(observations) == 10_000


def test_array_first_overflow_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    featurizer = TemporalContextFeaturizer(TemporalContextConfig(input_dim=2, periods=()))
    observations = jnp.zeros((_TEMPORAL_CONTEXT_LOOP_MAX_STEPS + 1, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="observations num_steps must be an integer in"):
        transform_temporal_context_arrays(featurizer, observations)
    assert seen == []


def test_array_gate_does_not_read_hostile_shape() -> None:
    _HostileArrays.calls = 0
    with pytest.raises(TypeError, match="observations must be a JAX array"):
        _require_temporal_context_array_steps(_HostileArrays())
    assert _HostileArrays.calls == 0
