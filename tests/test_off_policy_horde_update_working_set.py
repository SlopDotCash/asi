"""Complete update working-set preflight for nonlinear shared GTD Horde."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    _preflight_nonlinear_state,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 5_000_000
_N_DEMONS = 2
_HIDDEN = 16


def _two_demon_spec() -> object:
    demons = tuple(
        GVFSpec(
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=0.8,
            lamda=0.0,
            cumulant_index=i,
        )  # type: ignore[call-arg]
        for i in range(_N_DEMONS)
    )
    return create_horde_spec(demons)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_nonlinear_horde_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    fd = _WORKING_SET_OVERFLOW
    hidden_features = _HIDDEN * fd
    persist_scalars = (
        hidden_features * (_N_DEMONS + 1)
        + _HIDDEN
        + 3 * _N_DEMONS * _HIDDEN
        + 2 * _N_DEMONS
        + 3
    )
    one_bank_bytes = 4 * hidden_features
    persistent_bytes = 4 * persist_scalars
    logical_scalars = persist_scalars
    parameter_scalars = (
        hidden_features * (_N_DEMONS + 1)
        + _HIDDEN
        + 3 * _N_DEMONS * _HIDDEN
        + 2 * _N_DEMONS
    )
    one_gradient_scalars = hidden_features + 2 * _HIDDEN + 1
    update_bytes = 4 * (
        3 * logical_scalars
        + parameter_scalars
        + 3 * one_gradient_scalars
        + 3 * fd
        + 4 * _HIDDEN
        + 24 * _N_DEMONS
        + 32
    )
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    learner = NonlinearSharedGTDHordeLearner(_two_demon_spec(), hidden_size=_HIDDEN)
    with pytest.raises(ValueError, match="update working set byte count"):
        learner.init(fd, jax.random.key(0))


def test_nonlinear_horde_persistent_byte_bound_still_fires_first() -> None:
    learner = NonlinearSharedGTDHordeLearner(_two_demon_spec(), hidden_size=_HIDDEN)
    with pytest.raises(ValueError, match="persistent state bytes"):
        learner.init(20_000_000, jax.random.key(0))


def test_legal_nonlinear_horde_update_identity_is_unchanged() -> None:
    learner = NonlinearSharedGTDHordeLearner(
        _two_demon_spec(),
        hidden_size=4,
        primary_step_size=0.002,
        secondary_step_size=1e-5,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(7))
    assert state.trunk_w.shape == (4, 2)
    assert state.secondary_trunk_w.shape == (2, 4, 2)
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array([2.0, 0.0], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )
    assert result.state.trunk_w.shape == (4, 2)
    assert result.predictions.shape == (2,)
    assert float(jnp.linalg.norm(result.state.trunk_w - state.trunk_w)) > 0.0


def test_nonlinear_horde_update_working_set_has_exact_adjacent_byte_boundary() -> None:
    # For D=H=1, the complete formula reduces exactly to 14*feature_dim+102.
    max_scalars = _INT32_MAX // 4
    last_legal = (max_scalars - 102) // 14
    first_overflowing = last_legal + 1
    learner = NonlinearSharedGTDHordeLearner(
        create_horde_spec(
            (
                GVFSpec(
                    name="demon",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.8,
                    lamda=0.0,
                    cumulant_index=0,
                ),  # type: ignore[call-arg]
            )
        ),
        hidden_size=1,
    )

    # Exercise the arithmetic-only preflight without attempting either large
    # allocation; the next scalar is the first byte-count overflow.
    _preflight_nonlinear_state(n_demons=1, hidden_size=1, feature_dim=last_legal)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_nonlinear_state(n_demons=1, hidden_size=1, feature_dim=first_overflowing)

    contributor_scalars = 13 * first_overflowing + 98
    assert 4 * contributor_scalars <= _INT32_MAX
    assert 4 * (14 * last_legal + 102) <= _INT32_MAX
    assert 4 * (14 * first_overflowing + 102) > _INT32_MAX
    assert learner.n_demons == 1
