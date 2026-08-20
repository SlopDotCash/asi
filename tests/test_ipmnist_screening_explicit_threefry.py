"""The shared IPMNIST runner owns an explicit Threefry RNG root."""

from __future__ import annotations

import jax
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


def test_screening_trajectory_is_independent_of_ambient_prng_default() -> None:
    data_x = np.linspace(-1.0, 1.0, 72, dtype=np.float32).reshape(12, 6)
    data_y = np.arange(12, dtype=np.int32) % 3
    config = IPMNISTConfig(
        n_tasks=2,
        task_length=4,
        input_dim=6,
        hidden1=5,
        hidden2=4,
        n_classes=3,
    )
    spec = screening_spec("upgd_w_control")

    with jax.default_prng_impl("threefry2x32"):
        threefry_default = run_screening_config(
            data_x,
            data_y,
            spec,
            seed=156_700_901,
            config=config,
        )
    with jax.default_prng_impl("rbg"):
        rbg_default = run_screening_config(
            data_x,
            data_y,
            spec,
            seed=156_700_901,
            config=config,
        )

    np.testing.assert_array_equal(
        rbg_default.per_task_accuracy,
        threefry_default.per_task_accuracy,
    )
    np.testing.assert_array_equal(rbg_default.per_task_loss, threefry_default.per_task_loss)
    np.testing.assert_array_equal(
        rbg_default.per_task_plasticity,
        threefry_default.per_task_plasticity,
    )
