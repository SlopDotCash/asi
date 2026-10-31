"""Tiny nonzero feature_scales must not narrow silently to float32 zero."""

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.synthetic import ScaledStreamWrapper


class _StubStream:
    feature_dim = 3

    def init(self, key):
        return ()

    def step(self, state, idx):
        return TimeStep(observation=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
                        target=jnp.array([0.0], dtype=jnp.float32)), state


@pytest.mark.parametrize("tiny", [1e-46, -1e-46, 1e-300])
def test_tiny_nonzero_scale_is_rejected(tiny):
    with pytest.raises(ValueError, match="narrows to float32 zero"):
        ScaledStreamWrapper(_StubStream(), np.array([1.0, tiny, 2.0], dtype=np.float64))


def test_exact_zero_scale_stays_legal():
    wrapper = ScaledStreamWrapper(_StubStream(), np.array([1.0, 0.0, 2.0], dtype=np.float64))
    assert float(np.asarray(wrapper.feature_scales, dtype=np.float64)[1]) == 0.0
