"""Protocol T ceilings for nexting reverse-scan returns.

Origin reverse-scanned T=20_000 with no reject. Documented public trajectory
last-fit is 10_000 (README / package ``__init__``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.utils.nexting import (
    _NEXTING_MAX_CHANNELS,
    _NEXTING_MAX_HORIZONS,
    _NEXTING_MAX_STEPS,
    forward_view_returns,
    multi_channel_horizon_returns,
    multi_horizon_returns,
)


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn: Any, init: Any, xs: Any, **kwargs: Any) -> Any:
        seen.append(int(xs.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={xs.shape[0]}")

    monkeypatch.setattr("alberta_framework.utils.nexting.jax.lax.scan", spy)
    return seen


def test_protocol_ceilings_match_documented_trajectory_last_fit() -> None:
    assert _NEXTING_MAX_STEPS == 10_000
    assert _NEXTING_MAX_HORIZONS == 16


def test_last_fit_series_is_accepted() -> None:
    series = jnp.ones((_NEXTING_MAX_STEPS,), dtype=jnp.float32)
    returns = forward_view_returns(series, 0.0)
    assert returns.shape == (_NEXTING_MAX_STEPS,)


def test_first_overflow_series_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    series = jnp.ones((_NEXTING_MAX_STEPS + 1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match=r"cumulants length must be an integer in \[1, 10000\]"):
        forward_view_returns(series, 0.9)
    assert seen == []


def test_origin_hang_class_20000_is_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_scan(monkeypatch)
    series = jnp.ones((20_000,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="cumulants length must be an integer in"):
        forward_view_returns(series, 0.9)
    assert seen == []


def test_horizon_overflow_is_rejected() -> None:
    series = jnp.ones((4,), dtype=jnp.float32)
    gammas = jnp.linspace(0.0, 1.0, _NEXTING_MAX_HORIZONS + 1)
    with pytest.raises(ValueError, match="gammas length must be an integer in"):
        multi_horizon_returns(series, gammas)


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (
            forward_view_returns,
            (jax.ShapeDtypeStruct((0,), jnp.float32), 0.9),
            "cumulants length must be an integer in",
        ),
        (
            forward_view_returns,
            (
                jax.ShapeDtypeStruct((_NEXTING_MAX_STEPS + 1,), jnp.float32),
                0.9,
            ),
            "cumulants length must be an integer in",
        ),
        (
            multi_horizon_returns,
            (
                jax.ShapeDtypeStruct((4,), jnp.float32),
                jax.ShapeDtypeStruct((0,), jnp.float32),
            ),
            "gammas length must be an integer in",
        ),
        (
            multi_horizon_returns,
            (
                jax.ShapeDtypeStruct((4,), jnp.float32),
                jax.ShapeDtypeStruct((_NEXTING_MAX_HORIZONS + 1,), jnp.float32),
            ),
            "gammas length must be an integer in",
        ),
        (
            multi_channel_horizon_returns,
            (
                jax.ShapeDtypeStruct((4, 0), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ),
            "cumulants channel count must be an integer in",
        ),
        (
            multi_channel_horizon_returns,
            (
                jax.ShapeDtypeStruct((4, _NEXTING_MAX_CHANNELS + 1), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ),
            "cumulants channel count must be an integer in",
        ),
    ],
)
def test_protocol_ceilings_cannot_be_bypassed_during_jit_tracing(
    function: Any,
    arguments: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        jax.eval_shape(function, *arguments)
