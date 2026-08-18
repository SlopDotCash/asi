"""Leftover-identity gates for partial-observation mask modes."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.streams.partial_observation import (
    MaskMode,
    PartialObservationWrapper,
)
from alberta_framework.streams.synthetic import RandomWalkStream


def _inner() -> RandomWalkStream:
    return RandomWalkStream(feature_dim=4, drift_rate=0.0, noise_std=0.0)


def _visible_odd_mask() -> jnp.ndarray:
    return jnp.array([True, False, True, False])


@pytest.mark.parametrize(
    "leftover",
    ("FIXED", "fixed", "RANDOM", "PERIODIC", 1, True, "not-a-mode"),
)
def test_partial_observation_rejects_leftover_mode_identities(leftover: object) -> None:
    """Leftover mode identities must not skip the channel mask."""

    with pytest.raises(ValueError, match="mode"):
        PartialObservationWrapper(
            _inner(),
            mode=leftover,  # type: ignore[arg-type]
            fixed_mask=_visible_odd_mask(),
        )


def test_partial_observation_legal_fixed_mode_still_masks() -> None:
    wrapper = PartialObservationWrapper(
        _inner(),
        mode=MaskMode.FIXED,
        fixed_mask=_visible_odd_mask(),
    )
    assert wrapper.mode is MaskMode.FIXED
    timestep, _ = wrapper.step(wrapper.init(jr.key(0)), jnp.array(0))
    assert float(timestep.observation[1]) == 0.0
    assert float(timestep.observation[3]) == 0.0
