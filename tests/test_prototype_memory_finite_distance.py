# mypy: disable-error-code="call-arg,untyped-decorator,no-any-return"
"""Mean squared prototype distance when a component square overflows."""

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    PrototypeMemoryState,
)


def _wide_learner(*, feature_dim: int) -> PrototypeMemoryLearner:
    """Bandwidth and threshold sit on the scale of a wide squared distance."""
    return PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=feature_dim,
            n_classes=2,
            slots_per_class=1,
            update_rate=0.3,
            novelty_threshold=float(np.float32(3.0e38)),
            bandwidth=float(np.float32(1.0e20)),
        )
    )


def _occupied(
    learner: PrototypeMemoryLearner,
    means: jnp.ndarray,
) -> PrototypeMemoryState:
    return learner.init().replace(
        means=means,
        counts=jnp.ones((learner.config.n_classes, 1), dtype=jnp.int32),
    )


def _ordinary_logits(
    learner: PrototypeMemoryLearner,
    means: jnp.ndarray,
    observation: jnp.ndarray,
) -> jnp.ndarray:
    """Reference logits from a direct mean of finite squares."""
    diffs = means[:, 0, :] - observation[None, :]
    distance = jnp.mean(diffs * diffs, axis=-1)
    bandwidth = jnp.asarray(learner.config.bandwidth, dtype=jnp.float32)
    return -distance / bandwidth


@pytest.mark.parametrize("sign", [np.float32(1.0), np.float32(-1.0)])
def test_overflowing_component_square_keeps_the_nearer_class(sign: np.float32) -> None:
    """A representable mean stays finite when one squared component overflows.

    ``feature_dim`` is 2, so ``(2e19)**2 / 2`` is about ``2e38`` and
    ``(2.5e19)**2 / 2`` is about ``3.125e38``. Both raw squares exceed
    float32. The nearer prototype is the origin. Its matched update is an
    EMA step, not a rejected hold and not a novel replacement.
    """
    learner = _wide_learner(feature_dim=2)
    nearer = sign * np.float32(2.0e19)
    farther = sign * np.float32(4.5e19)
    observation = jnp.array([nearer, np.float32(0.0)], dtype=jnp.float32)
    means = jnp.zeros((2, 1, 2), dtype=jnp.float32).at[1, 0, 0].set(farther)
    state = _occupied(learner, means)
    target = jnp.array([1.0, 0.0], dtype=jnp.float32)

    logits = learner.class_logits(state, observation)
    repeated = learner.class_logits(state, observation)
    assert jnp.all(jnp.isfinite(logits))
    assert float(logits[0]) > float(logits[1])
    np.testing.assert_array_equal(np.asarray(repeated), np.asarray(logits))
    assert int(jnp.argmax(learner.predict(state, observation))) == 0

    result = learner.update(state, observation, target)
    assert bool(result.update_applied) is True
    eta = jnp.asarray(learner.config.update_rate, dtype=jnp.float32)
    matched = eta * observation
    np.testing.assert_array_equal(np.asarray(result.state.means[0, 0]), np.asarray(matched))
    np.testing.assert_array_equal(np.asarray(result.state.means[1, 0]), np.asarray(means[1, 0]))
    assert int(result.state.counts[0, 0]) == 2
    assert int(result.state.counts[1, 0]) == 1


def test_finite_squares_keep_the_direct_mean() -> None:
    """Ordinary squares, including a finite ``1e19`` square, stay bitwise."""
    learner = _wide_learner(feature_dim=2)
    observation = jnp.array([np.float32(1.0e19), np.float32(-0.25)], dtype=jnp.float32)
    means = jnp.asarray(
        [[[np.float32(0.0), np.float32(0.0)]], [[np.float32(1.0e19), np.float32(0.5)]]],
        dtype=jnp.float32,
    )
    state = _occupied(learner, means)
    logits = learner.class_logits(state, observation)
    expected = _ordinary_logits(learner, means, observation)
    assert jnp.all(jnp.isfinite(expected))
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))


def test_finite_component_mean_overflow_stays_recovered() -> None:
    """A mean that overflows only in the reduction still uses the old split."""
    learner = _wide_learner(feature_dim=8)
    square = np.float32(1.0e19) * np.float32(1.0e19)
    observation = jnp.zeros((8,), dtype=jnp.float32)
    class_mean = jnp.full((8,), np.float32(1.0e19), dtype=jnp.float32)
    means = jnp.stack(
        [class_mean, jnp.zeros((8,), dtype=jnp.float32)],
        axis=0,
    )[:, None, :]
    state = _occupied(learner, means)
    logits = learner.class_logits(state, observation)
    distance = jnp.sum((class_mean * class_mean) / np.float32(8.0))
    expected = -distance / jnp.asarray(learner.config.bandwidth, dtype=jnp.float32)
    assert not bool(jnp.isfinite(jnp.mean(jnp.full((8,), square))))
    assert bool(jnp.isfinite(expected))
    np.testing.assert_array_equal(np.asarray(logits[0]), np.asarray(expected))


def test_unrepresentable_square_still_rejects_the_update() -> None:
    """``feature_dim`` 1 cannot divide an unrepresentable square into range."""
    learner = _wide_learner(feature_dim=1)
    observation = jnp.array([np.float32(2.0e19)], dtype=jnp.float32)
    means = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    state = _occupied(learner, means)
    logits = learner.class_logits(state, observation)
    result = learner.update(state, observation, jnp.array([1.0, 0.0], dtype=jnp.float32))
    assert not bool(jnp.all(jnp.isfinite(logits)))
    assert bool(result.update_applied) is False
    np.testing.assert_array_equal(np.asarray(result.state.means), np.asarray(means))
