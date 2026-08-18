"""Overflow preflight keeps the IA update's published width in signed int32."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumAgent,
    ExoCerebellumConfig,
    IAConfig,
    _default_oak_config,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_int32_wrap_forges_a_different_published_update_width() -> None:
    cortex_obs_dim = _default_oak_config().observation_dim
    overflowing_demons = _INT32_MAX - cortex_obs_dim + 1
    published_width = cortex_obs_dim + overflowing_demons
    assert published_width == _INT32_MAX + 1
    wrapped_width = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_width == _INT32_MIN
    assert wrapped_width != published_width


def test_cerebellum_config_rejects_weight_product_overflow() -> None:
    with pytest.raises(ValueError, match="must fit signed int32"):
        ExoCerebellumConfig(n_demons=65536, obs_dim=32768)


def test_cerebellum_config_rejects_aggregate_before_allocation() -> None:
    # The weight array alone is 4 bytes below INT32_MAX, but the persistent
    # cumulant-index vector/counters and update working copies do not fit.
    with pytest.raises(ValueError, match="persistent state byte count"):
        ExoCerebellumConfig(n_demons=_INT32_MAX // 4, obs_dim=1)


def test_cerebellum_config_rejects_simultaneous_update_working_set() -> None:
    # Each individual matrix is below the byte ceiling; the live weights,
    # outer product, and candidate matrix together exceed it.
    with pytest.raises(ValueError, match="update working set byte count"):
        ExoCerebellumConfig(n_demons=100_000_000, obs_dim=1)


def test_cerebellum_working_set_includes_selected_result_weights() -> None:
    # The previous three-matrix envelope fit here. Retaining the source,
    # outer delta, candidate, and selected result matrices does not.
    with pytest.raises(ValueError, match="update working set byte count"):
        ExoCerebellumConfig(n_demons=1, obs_dim=70_000_000)


def test_ia_config_rejects_overflowing_published_update_width() -> None:
    cortex = _default_oak_config()
    overflowing_demons = _INT32_MAX - cortex.observation_dim + 1
    with pytest.raises(ValueError, match="must fit signed int32"):
        IAConfig(
            cerebellum=ExoCerebellumConfig(
                n_demons=overflowing_demons,
                obs_dim=cortex.observation_dim,
            ),
            cortex=cortex,
        )


def test_legal_cerebellum_update_identity_is_unchanged() -> None:
    cfg = ExoCerebellumConfig(n_demons=4, obs_dim=3)
    agent = ExoCerebellumAgent(cfg)
    assert int(cfg.n_demons + cfg.obs_dim) == 7
    assert list(agent._cumulant_indices) == [0, 1, 2, 0]
    result = agent.update_result(
        agent.init(),
        jnp.ones(3, dtype=jnp.float32),
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert result.state.weights.shape == (4, 3)
    assert int(result.state.step_count) == 1
