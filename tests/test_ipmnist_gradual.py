from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_gradual import (
    GRADUAL_IPMNIST_PROTOCOL,
    GradualTransitionConfig,
    input_interpolation,
    input_interpolation_transaction,
    output_interpolation,
    run_gradual_input_pair,
    task_sampling_mask,
    transition_alpha,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


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


def test_transition_width_counts_intervals_including_both_endpoints() -> None:
    config = GradualTransitionConfig(mode="input_interpolation", transition_steps=2)
    assert [transition_alpha(step, config) for step in range(4)] == [0.0, 0.5, 1.0, 1.0]


def test_transition_config_rejects_forged_and_hostile_values_without_index_hooks() -> None:
    config = GradualTransitionConfig(mode="input_interpolation", transition_steps=2)
    object.__setattr__(config, "transition_steps", 0)
    with pytest.raises(ValueError, match="transition_steps"):
        transition_alpha(0, config)

    class HostileInt(np.int64):
        calls = 0

        def __index__(self) -> int:
            self.calls += 1
            raise AssertionError("must not execute")

    hostile = HostileInt(2)
    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="input_interpolation",
            transition_steps=hostile,  # type: ignore[arg-type]
        )
    assert hostile.calls == 0

    class HostileMeta(type):
        calls = 0

        def __eq__(cls, other: object) -> bool:
            cls.calls += 1
            raise AssertionError("must not compare types")

        def __hash__(cls) -> int:
            cls.calls += 1
            raise AssertionError("must not hash types")

    class Hostile(metaclass=HostileMeta):
        pass

    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="input_interpolation",
            transition_steps=Hostile(),  # type: ignore[arg-type]
        )
    assert HostileMeta.calls == 0


@pytest.mark.parametrize("code", ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"])
def test_config_accepts_every_numpy_integer_family(code: str) -> None:
    # Every signed and unsigned C integer family numpy exposes must be admitted
    # by the exact-type gate.  On any platform exactly one signed and one
    # unsigned family has no fixed-width alias, so a hand-written fixed-width
    # allowlist orphans it; the dtype-code derivation covers all ten families.
    scalar = np.dtype(code).type(2)
    assert type(scalar) is np.dtype(code).type
    config = GradualTransitionConfig(mode="input_interpolation", transition_steps=scalar)
    assert type(config.transition_steps) is int
    assert config.transition_steps == 2


def test_config_rejects_int_subclass_bool_and_float_without_index_hooks() -> None:
    class ForgedInt(np.int64):
        def __index__(self) -> int:
            raise AssertionError("must not execute __index__")

    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="input_interpolation",
            transition_steps=ForgedInt(2),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="input_interpolation",
            transition_steps=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="integer"):
        GradualTransitionConfig(
            mode="input_interpolation",
            transition_steps=2.0,  # type: ignore[arg-type]
        )


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


def test_input_interpolation_is_outer_jit_safe() -> None:
    interpolate = jax.jit(lambda old, new: input_interpolation(old, new, 0.5))
    np.testing.assert_array_equal(interpolate(jnp.array([0.0]), jnp.array([2.0])), jnp.array([1.0]))

    transact = jax.jit(lambda old, new: input_interpolation_transaction(old, new, 0.5))
    safe, valid = transact(jnp.array([jnp.inf]), jnp.array([2.0]))
    np.testing.assert_array_equal(safe, jnp.zeros(1))
    assert not bool(valid)
    assert bool(jnp.all(jnp.isnan(interpolate(jnp.array([jnp.inf]), jnp.array([2.0])))))


def test_input_interpolation_rejects_array_protocol_objects_without_calling_them() -> None:
    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        input_interpolation(hostile, jnp.ones(1), 0.5)  # type: ignore[arg-type]
    assert hostile.calls == 0


def test_gradual_input_pair_runs_one_matched_real_learner_schedule() -> None:
    data_x = np.asarray(
        [
            [-1.0, -0.5, 0.5, 1.0],
            [1.0, 0.5, -0.5, -1.0],
            [-0.5, 1.0, -1.0, 0.5],
            [0.5, -1.0, 1.0, -0.5],
        ],
        dtype=np.float32,
    )
    data_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=3,
        task_length=4,
        input_dim=4,
        hidden1=3,
        hidden2=2,
        n_classes=2,
    )

    result = run_gradual_input_pair(
        data_x,
        data_y,
        learner_name="adamw_control",
        seed=17,
        config=config,
        transition_steps=2,
    )

    assert result.arm_names == ("abrupt", "input_interpolation")
    assert result.seed == 17
    assert result.schema == "asi.ipmnist.gradual-input-pair.result.v1"
    assert result.development_only is True
    assert result.scientific_promotion_allowed is False
    assert result.execution_attestation is False
    assert result.prng_implementation == "threefry2x32"
    assert result.learner_hyperparameters
    assert result.dataset_rows == 4
    assert result.dataset_sha256.startswith("sha256:")
    assert result.observations_per_arm == result.updates_per_arm == 12
    assert result.data_steps_per_arm == 12
    assert result.environment_steps_per_arm == 0
    assert result.model_queries_per_arm == 24
    assert result.schedule_sha256.startswith("sha256:")
    assert result.example_order_sha256.startswith("sha256:")
    assert result.correct_counts.shape == (2, 3)
    assert np.all((result.correct_counts >= 0) & (result.correct_counts <= 4))
    assert result.loss_sums.shape == (2, 3)
    assert np.all(np.isfinite(result.loss_sums))
    assert result.persistent_numeric_bytes.shape == (2,)
    # 29 params * (initial + current + Adam m + Adam v) float32 bytes, five
    # optimizer scalars per each of six leaves, dataset, and full schedule.
    assert np.all(result.persistent_numeric_bytes == 29 * 16 + 6 * 5 * 4 + 80 + 96)
    assert result.timing_ns.shape == (2,)
    assert np.all(result.timing_ns >= 0)
    assert not result.correct_counts.flags.writeable


def test_gradual_pair_rng_is_independent_of_global_default() -> None:
    data_x = np.arange(16, dtype=np.float32).reshape(4, 4)
    data_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=2, task_length=4, input_dim=4, hidden1=2, hidden2=2, n_classes=2
    )

    with jax.default_prng_impl("threefry2x32"):
        first = run_gradual_input_pair(
            data_x,
            data_y,
            learner_name="adamw_control",
            seed=9,
            config=config,
            transition_steps=2,
        )
    with jax.default_prng_impl("rbg"):
        second = run_gradual_input_pair(
            data_x,
            data_y,
            learner_name="adamw_control",
            seed=9,
            config=config,
            transition_steps=2,
        )

    assert first.dataset_sha256 == second.dataset_sha256
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.example_order_sha256 == second.example_order_sha256
    np.testing.assert_array_equal(first.correct_counts, second.correct_counts)
    np.testing.assert_array_equal(first.loss_sums, second.loss_sums)


def test_task_sampling_rng_is_independent_of_global_default() -> None:
    with jax.default_prng_impl("threefry2x32"):
        first = task_sampling_mask(seed=7, transition_id=3, count=10, alpha=0.3)
    with jax.default_prng_impl("rbg"):
        second = task_sampling_mask(seed=7, transition_id=3, count=10, alpha=0.3)
    np.testing.assert_array_equal(first, second)


def test_gradual_pair_result_snapshots_mutable_outputs() -> None:
    data_x = np.arange(8, dtype=np.float32).reshape(4, 2)
    data_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    result = run_gradual_input_pair(
        data_x,
        data_y,
        learner_name="adamw_control",
        seed=4,
        config=IPMNISTConfig(
            n_tasks=2, task_length=4, input_dim=2, hidden1=2, hidden2=2, n_classes=2
        ),
        transition_steps=2,
    )
    with pytest.raises(ValueError, match="read-only"):
        result.correct_counts[0, 0] = 99

    source = result.correct_counts.copy()
    copied = replace(result, correct_counts=source)
    source[0, 0] = 99
    assert copied.correct_counts[0, 0] != 99
    with pytest.raises(ValueError, match="model_queries_per_arm must equal 16"):
        replace(result, model_queries_per_arm=15)
    with pytest.raises(ValueError, match="persistent_numeric_bytes"):
        replace(result, persistent_numeric_bytes=np.asarray([1, 1], dtype=np.int64))


def test_gradual_pair_rejects_transition_wider_than_each_task() -> None:
    config = IPMNISTConfig(
        n_tasks=2,
        task_length=2,
        input_dim=2,
        hidden1=2,
        hidden2=2,
        n_classes=2,
    )
    with pytest.raises(ValueError, match="transition_steps must be smaller"):
        run_gradual_input_pair(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            learner_name="adamw_control",
            seed=1,
            config=config,
            transition_steps=2,
        )


def test_gradual_pair_preflights_parameter_allocation_before_initialization() -> None:
    config = IPMNISTConfig(
        n_tasks=2,
        task_length=2,
        input_dim=100_000,
        hidden1=1000,
        hidden2=2,
        n_classes=2,
    )
    with pytest.raises(ValueError, match="aggregate persistent numeric allocation"):
        run_gradual_input_pair(
            np.zeros((2, 100_000), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            learner_name="adamw_control",
            seed=1,
            config=config,
            transition_steps=1,
        )
