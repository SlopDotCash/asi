from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_gradual import (
    GRADUAL_IPMNIST_PROTOCOL,
    GradualTransitionConfig,
    input_interpolation,
    output_interpolation,
    task_sampling_mask,
    transition_alpha,
)


def test_abrupt_mode_is_exact_new_task_reduction() -> None:
    config = GradualTransitionConfig(mode="abrupt", transition_steps=1)
    assert transition_alpha(0, config) == 1.0

    old = jnp.array([1.0, 2.0], dtype=jnp.float32)
    new = jnp.array([3.0, 4.0], dtype=jnp.float32)
    np.testing.assert_array_equal(input_interpolation(old, new, 1.0), new)


def test_input_interpolation_matches_paper_equation() -> None:
    old = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    new = jnp.array([1.0, -1.0], dtype=jnp.float32)
    np.testing.assert_array_equal(input_interpolation(old, new, 0.25), [-0.5, 0.5])


def test_output_interpolation_passes_through_uniform_distribution() -> None:
    old = output_interpolation(1, 2, 0.5, n_classes=4)
    np.testing.assert_array_equal(old, np.full(4, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(
        output_interpolation(1, 2, 0.0, n_classes=4), [0.0, 1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        output_interpolation(1, 2, 1.0, n_classes=4), [0.0, 0.0, 1.0, 0.0]
    )


def test_transition_alpha_is_deterministic_and_clamped() -> None:
    config = GradualTransitionConfig(mode="input_interpolation", transition_steps=4)
    assert [transition_alpha(step, config) for step in range(6)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_task_sampling_mask_is_matched_deterministic_and_monotone() -> None:
    first = task_sampling_mask(seed=7, transition_id=3, count=10, alpha=0.3)
    second = task_sampling_mask(seed=7, transition_id=3, count=10, alpha=0.3)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.bool_
    assert int(first.sum()) == 3
    assert int(task_sampling_mask(seed=7, transition_id=3, count=10, alpha=0.7).sum()) == 7


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), True])
def test_helpers_reject_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        input_interpolation(jnp.zeros(2), jnp.ones(2), alpha)  # type: ignore[arg-type]


def test_protocol_records_nonpromotion_and_information_allowance() -> None:
    assert GRADUAL_IPMNIST_PROTOCOL["paper_revision"] == "arXiv:2602.09234v2"
    assert GRADUAL_IPMNIST_PROTOCOL["development_only"] is True
    assert GRADUAL_IPMNIST_PROTOCOL["scientific_promotion_allowed"] is False
    assert GRADUAL_IPMNIST_PROTOCOL["learner_observes_transition_alpha"] is False
    assert GRADUAL_IPMNIST_PROTOCOL["matched_axes"] == (
        "seed",
        "updates",
        "observations",
        "example_order",
    )
