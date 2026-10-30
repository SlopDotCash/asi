"""Regression tests for fixed-alpha gauntlet control selection."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams import gauntlet


def test_best_fixed_alpha_compares_representable_means_without_sum_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors_by_alpha = {
        0.1: jnp.full((1, 2), 3.0e38, dtype=jnp.float32),
        0.2: jnp.full((1, 2), 2.0e38, dtype=jnp.float32),
    }

    def fake_run(
        learner: float,
        stream: Any,
        num_steps: int,
        keys: Any,
    ) -> jnp.ndarray:
        del stream, num_steps, keys
        return errors_by_alpha[learner]

    monkeypatch.setattr(gauntlet, "run_gauntlet_batched", fake_run)

    errors, alpha = gauntlet.best_fixed_alpha_errors(
        lambda candidate: candidate,
        stream=None,  # type: ignore[arg-type]
        num_steps=2,
        keys=jr.split(jr.key(0), 1),
        step_sizes=(0.1, 0.2),
    )

    assert alpha == 0.2
    np.testing.assert_array_equal(np.asarray(errors), np.asarray(errors_by_alpha[0.2]))
