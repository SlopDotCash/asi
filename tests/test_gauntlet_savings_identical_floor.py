"""savings_ratio must not report 0 when both entry windows sit at the floor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    gauntlet_scorecard,
    savings_ratio,
    savings_ratio_steps,
)


def _nine_segment(values: float | np.ndarray, *, segment_length: int = 8, n_seeds: int = 2):
    if np.ndim(values) == 0:
        body = jnp.full((n_seeds, 9 * segment_length), float(values), dtype=jnp.float32)
    else:
        body = jnp.asarray(values, dtype=jnp.float32)
    return body, segment_length


def test_identical_zero_entry_windows_score_one_not_zero() -> None:
    sq, length = _nine_segment(0.0)
    ratio = np.asarray(savings_ratio(sq, 2, 4, length, window=4))
    steps = np.asarray(savings_ratio_steps(sq, 2, 4, length, threshold=0.05))
    np.testing.assert_allclose(ratio, [1.0, 1.0])
    np.testing.assert_allclose(steps, [1.0, 1.0])


def test_identical_underflow_entry_windows_score_one_not_floor_ratio() -> None:
    sq, length = _nine_segment(1e-12)
    ratio = np.asarray(savings_ratio(sq, 2, 4, length, window=4))
    # Origin/main reported 1e-12 / 1e-8 = 1e-4.
    np.testing.assert_allclose(ratio, [1.0, 1.0])


def test_memoryless_identical_nonzero_windows_stay_one() -> None:
    sq, length = _nine_segment(1.0)
    ratio = np.asarray(savings_ratio(sq, 2, 4, length, window=4))
    np.testing.assert_allclose(ratio, [1.0, 1.0])


def test_perfect_first_worse_revisit_still_zero() -> None:
    length = 8
    sq = jnp.ones((2, 9 * length), dtype=jnp.float32)
    sq = sq.at[..., 2 * length : 3 * length].set(0.0)
    ratio = np.asarray(savings_ratio(sq, 2, 4, length, window=4))
    np.testing.assert_allclose(ratio, [0.0, 0.0])


def test_scorecard_zero_program_reports_identity_savings() -> None:
    config = GauntletConfig(segment_length=200, relevant_dim=2, irrelevant_dim=0)
    sq = jnp.zeros((2, 9 * config.segment_length), dtype=jnp.float32)
    card = gauntlet_scorecard(sq, config)
    np.testing.assert_allclose(np.asarray(card["savings_c"]), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(card["savings_d"]), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(card["savings_c_final"]), [1.0, 1.0])
