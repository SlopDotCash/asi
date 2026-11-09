"""Phase codes must encode the configured period, not its float32 rounding."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
)


@pytest.mark.parametrize("period", [0.7, 0.3, 0.1])
@pytest.mark.parametrize("step_count", [3, 10_000_000, 30_000_000, 2**31 - 1])
def test_phase_code_uses_configured_period_not_float32_period(
    period: float, step_count: int
) -> None:
    """Phase features must track sin/cos(2*pi*t/p) for the configured ``p``.

    The digit-decomposition residues are exact host-side ratios, so the
    encoded period is the configured float64 period. Rounding the period
    itself to float32 first (0.7 -> 0.69999998807907104) makes the phase
    drift linearly with the step count: by ``t = 2**24`` the encoded phase
    is already off by a large fraction of a cycle.
    """
    config = TemporalContextConfig(
        input_dim=1, include_ema=False, include_delta=False, periods=(period,)
    )
    assert config.periods == (period,)
    featurizer = TemporalContextFeaturizer(config)
    state = TemporalContextState(
        observation_ema=jnp.zeros(1, dtype=jnp.float32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
    )

    features = np.asarray(featurizer.features(state, jnp.zeros(1, dtype=jnp.float32)))

    cycles = float((Fraction(step_count) / Fraction(period)) % 1)
    expected = np.asarray([np.sin(2.0 * np.pi * cycles), np.cos(2.0 * np.pi * cycles)])
    np.testing.assert_allclose(features[1:], expected, rtol=0.0, atol=1e-4)
