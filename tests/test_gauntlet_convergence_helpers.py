"""Supplementary coverage for gauntlet.py convergence helpers.

Covers previously untested helpers: steps_to_criterion (EMA-smoothed first
crossing with cap; the halflife-50 EMA makes crossings only immediate or
capped for short segments), segment_mse (tail-only MSE), and
savings_ratio_steps (first vs revisit steps ratio with flooring).
"""

import jax.numpy as jnp
import pytest

from alberta_framework.streams.gauntlet import (
    savings_ratio_steps,
    segment_mse,
    steps_to_criterion,
)


def test_steps_to_criterion_immediate() -> None:
    # Already below threshold from step 0.
    sq = jnp.array([1.0, 5.0, 5.0])
    assert steps_to_criterion(sq, threshold=2.0) == 0


def test_steps_to_criterion_capped() -> None:
    # Never below threshold → capped at segment length.
    sq = jnp.array([5.0, 6.0, 7.0, 8.0])
    steps = steps_to_criterion(sq, threshold=2.0)
    assert steps == 4


def test_steps_to_criterion_boundary() -> None:
    # Exactly at threshold counts as reached immediately.
    sq = jnp.array([2.0, 2.0, 2.0])
    assert steps_to_criterion(sq, threshold=2.0) == 0


def test_steps_to_criterion_batched() -> None:
    sq = jnp.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    steps = steps_to_criterion(sq, threshold=2.0)
    assert steps.shape == (2,)
    assert steps[0] == 0
    assert steps[1] == 3


def test_steps_to_criterion_scalar_squeeze() -> None:
    # 1-D input returns a scalar (not an array).
    sq = jnp.array([1.0, 1.0])
    steps = steps_to_criterion(sq, threshold=2.0)
    assert steps.ndim == 0


def test_segment_mse_tail_only() -> None:
    # sq_errors shape (1 seed x 8 steps); segment 0 = steps 0-3, tail = last 2.
    sq = jnp.array([[1.0, 1.0, 4.0, 8.0, 9.0, 9.0, 9.0, 9.0]])
    mse = segment_mse(sq, segment=0, segment_length=4, tail_frac=0.5)
    assert mse == pytest.approx(6.0)  # mean(4, 8)


def test_segment_mse_full_tail() -> None:
    sq = jnp.array([[1.0, 3.0, 5.0, 7.0, 9.0, 9.0, 9.0, 9.0]])
    mse = segment_mse(sq, segment=0, segment_length=4, tail_frac=1.0)
    assert mse == pytest.approx(4.0)


def test_savings_ratio_steps_improvement() -> None:
    # One seed, 8 steps: segment 0 (steps 0-3) all high → capped 4;
    # segment 1 (steps 4-7) all low → immediate 0 → floored to 1 → ratio 4.
    sq = jnp.array([[10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]])
    ratio = savings_ratio_steps(
        sq, first_segment=0, revisit_segment=1, segment_length=4, threshold=2.0
    )
    assert ratio == 4.0


def test_savings_ratio_steps_no_improvement() -> None:
    # Both segments immediate → 1/1 = 1.
    sq = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    ratio = savings_ratio_steps(
        sq, first_segment=0, revisit_segment=1, segment_length=4, threshold=2.0
    )
    assert ratio == 1.0
