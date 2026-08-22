from __future__ import annotations

import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.adalin import (
    ADALIN_MATCHED_RESULT_SCHEMA,
    ADALIN_OFFICIAL_COMMIT,
    ADALIN_PROTOCOL,
    AdaLinConfig,
    adalin_gradients,
    adalin_logits,
    adalin_relu,
    adalin_relu_transaction,
    adalin_sgd_step,
    adalin_sgd_step_transaction,
    adalin_tanh,
    initialize_adalin_state,
    make_pmnist_schedule,
    run_adalin_development,
    run_adalin_matched_development,
    validate_adalin_matched_result,
    validate_adalin_result,
)


def test_relu_reduction_and_prelu_identity() -> None:
    x = jnp.array([-2.0, 3.0])
    np.testing.assert_array_equal(adalin_relu(x, jnp.zeros(2)), jax.nn.relu(x))
    np.testing.assert_array_equal(adalin_relu(x, jnp.array([0.25, 0.25])), [-0.5, 3.0])


def test_gate_is_stop_gradient() -> None:
    value, derivative = jax.value_and_grad(lambda z: adalin_tanh(z, jnp.array(0.2)))(jnp.array(2.0))
    gate = jnp.cos(0.5 * jnp.pi * jnp.abs(1.0 - jnp.tanh(2.0) ** 2))
    expected = (1.0 - jnp.tanh(2.0) ** 2) + 0.2 * gate
    assert jnp.isfinite(value)
    np.testing.assert_allclose(derivative, expected, rtol=1e-6)


def test_protocol_keeps_pmnist_difference_explicit() -> None:
    assert ADALIN_PROTOCOL["paper_revision"] == "arXiv:2505.09486v1"
    assert ADALIN_PROTOCOL["paper_pmnist_tasks"] == 400
    assert ADALIN_PROTOCOL["asi_target_tasks"] == 200
    assert ADALIN_PROTOCOL["mechanism_off"] == "alpha_zero_exact_base_activation"
    assert ADALIN_PROTOCOL["scientific_promotion_allowed"] is False


def test_adalin_is_outer_jit_safe() -> None:
    transformed = jax.jit(adalin_relu)(jnp.array([-1.0, 1.0]), jnp.array([0.2, 0.2]))
    np.testing.assert_allclose(transformed, [-0.2, 1.0])
    invalid = jax.jit(adalin_relu)(jnp.array([jnp.nan]), jnp.array([0.2]))
    assert bool(jnp.all(jnp.isnan(invalid)))
    safe, valid = jax.jit(adalin_relu_transaction)(jnp.array([jnp.nan]), jnp.array([0.2]))
    np.testing.assert_array_equal(safe, jnp.zeros(1))
    assert not bool(valid)


def test_adalin_preflights_cross_broadcast_and_hostile_array() -> None:
    with pytest.raises(ValueError, match="broadcast output"):
        adalin_relu(jnp.ones((1, 1_000_000)), jnp.ones((1_000_000, 1)))

    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        adalin_relu(hostile, jnp.ones(1))  # type: ignore[arg-type]
    assert hostile.calls == 0


def test_official_revision_and_end_to_end_surface_are_pinned() -> None:
    assert ADALIN_OFFICIAL_COMMIT == "011469138bc22bf82955b16d68f33e4fbd04e3f8"
    config = AdaLinConfig(
        tasks=2, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    schedule = make_pmnist_schedule(config, seed=7, input_dim=4)
    assert schedule.pixel_permutations.shape == (2, 4)
    state = initialize_adalin_state(config, input_dim=4, classes=2, seed=7)
    gradients = adalin_gradients(
        state.parameters,
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.int32),
        mechanism_enabled=True,
    )
    assert gradients.alpha1.shape == (3,)
    assert bool(jnp.any(gradients.alpha1 != 0))
    updated = adalin_sgd_step(
        state,
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.int32),
        learning_rate=0.01,
        mechanism_enabled=True,
    )
    assert not bool(jnp.array_equal(updated.parameters.alpha1, state.parameters.alpha1))


def test_tiny_adalin_runner_is_strict_and_nonpromoting() -> None:
    train_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], np.float32)
    train_y = np.array([0, 1, 0, 1], np.int32)
    test_x = train_x.copy()
    test_y = train_y.copy()
    config = AdaLinConfig(
        tasks=2, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_development(train_x, train_y, test_x, test_y, config=config, seed=3)
    validate_adalin_result(result)
    assert result["policy"]["scientific_promotion_allowed"] is False
    assert result["resources"]["optimizer_updates"] == 4
    assert (
        result["provenance"]["initial_state_sha256"] != result["provenance"]["final_state_sha256"]
    )


def test_runner_accepts_mnist_width_without_running_paper_horizon() -> None:
    inputs = np.zeros((2, 784), dtype=np.float32)
    inputs[0, 0] = 1.0
    inputs[1, 1] = 1.0
    labels = np.array([0, 1], dtype=np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=2, batch_size=2, hidden_widths=(2, 2), classes=2
    )
    result = run_adalin_development(inputs, labels, inputs, labels, config=config, seed=8)
    assert result["dataset"]["input_dim"] == 784
    assert result["resources"]["environment_data_steps"] == 2


def test_mechanism_off_has_exact_relu_forward_gradient_and_state_parity() -> None:
    config = AdaLinConfig(
        tasks=1, examples_per_task=2, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    state = initialize_adalin_state(config, input_dim=3, classes=2, seed=4, mechanism_enabled=False)
    inputs = jnp.array([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]])
    labels = jnp.array([0, 1], dtype=jnp.int32)
    logits = adalin_logits(state.parameters, inputs, mechanism_enabled=False)
    hidden1 = jax.nn.relu(inputs @ state.parameters.weight1 + state.parameters.bias1)
    hidden2 = jax.nn.relu(hidden1 @ state.parameters.weight2 + state.parameters.bias2)
    baseline = hidden2 @ state.parameters.weight3 + state.parameters.bias3
    np.testing.assert_array_equal(logits, baseline)
    gradients = adalin_gradients(state.parameters, inputs, labels, mechanism_enabled=False)
    np.testing.assert_array_equal(gradients.alpha1, jnp.zeros_like(gradients.alpha1))
    np.testing.assert_array_equal(gradients.alpha2, jnp.zeros_like(gradients.alpha2))
    updated = adalin_sgd_step(state, inputs, labels, learning_rate=0.01, mechanism_enabled=False)
    np.testing.assert_array_equal(updated.parameters.alpha1, state.parameters.alpha1)
    np.testing.assert_array_equal(updated.parameters.alpha2, state.parameters.alpha2)


def test_schedule_and_runner_replay_exactly_from_seed() -> None:
    inputs = np.arange(24, dtype=np.float32).reshape(6, 4) / 24
    labels = np.array([0, 1, 0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=3, examples_per_task=6, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    first_schedule = make_pmnist_schedule(config, seed=9, input_dim=4)
    second_schedule = make_pmnist_schedule(config, seed=9, input_dim=4)
    np.testing.assert_array_equal(
        first_schedule.pixel_permutations, second_schedule.pixel_permutations
    )
    first = run_adalin_development(inputs, labels, inputs, labels, config=config, seed=9)
    second = run_adalin_development(inputs, labels, inputs, labels, config=config, seed=9)
    assert first["metrics"] == second["metrics"]
    assert first["provenance"] == second["provenance"]


def test_mechanism_off_runner_keeps_alpha_zero() -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_development(
        inputs, labels, inputs, labels, config=config, seed=5, mechanism_enabled=False
    )
    assert result["arm"] == "relu_alpha_zero_mechanism_off"
    assert result["state"]["final_alpha_l2"] == 0.0
    validate_adalin_result(json.loads(json.dumps(result)))


def test_matched_runner_binds_schedule_initialization_resources_and_nonpromotion() -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=2, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_matched_development(
        inputs, labels, inputs, labels, config=config, seed=1_571_001
    )
    validate_adalin_matched_result(json.loads(json.dumps(result)))

    assert result["schema"] == ADALIN_MATCHED_RESULT_SCHEMA
    assert [arm["arm"] for arm in result["arms"]] == [
        "adalin",
        "relu_alpha_zero_mechanism_off",
    ]
    enabled, disabled = result["arms"]
    assert enabled["provenance"]["schedule_sha256"] == disabled["provenance"][
        "schedule_sha256"
    ]
    assert enabled["provenance"]["initial_shared_parameters_sha256"] == disabled[
        "provenance"
    ]["initial_shared_parameters_sha256"]
    assert enabled["resources"]["optimizer_updates"] == disabled["resources"][
        "optimizer_updates"
    ]
    assert disabled["state"]["final_alpha_l2"] == 0.0
    assert result["policy"]["scientific_promotion_allowed"] is False
    assert result["policy"]["automatic_reference_selection_allowed"] is False


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("matched_axes", "schedule_sha256", "0" * 64),
        ("matched_axes", "initial_shared_parameters_sha256", "1" * 64),
        ("comparison", "adalin_minus_mechanism_off", 1.0),
        ("resources", "optimizer_updates", 1),
        ("policy", "automatic_reference_selection_allowed", True),
    ],
)
def test_matched_validator_rejects_forged_derived_fields(
    section: str, field: str, replacement: object
) -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_matched_development(
        inputs, labels, inputs, labels, config=config, seed=1_571_002
    )
    forged = copy.deepcopy(result)
    forged[section][field] = replacement
    with pytest.raises(ValueError):
        validate_adalin_matched_result(forged)


def test_matched_validator_rejects_arm_schedule_or_shared_initialization_substitution() -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_matched_development(
        inputs, labels, inputs, labels, config=config, seed=1_571_003
    )
    for field in ("schedule_sha256", "initial_shared_parameters_sha256"):
        forged = copy.deepcopy(result)
        forged["arms"][1]["provenance"][field] = "f" * 64
        with pytest.raises(ValueError):
            validate_adalin_matched_result(forged)


def test_matched_runner_rejects_seed_outside_frozen_development_roster() -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    with pytest.raises(ValueError, match="frozen AdaLin matched-development roster"):
        run_adalin_matched_development(
            inputs, labels, inputs, labels, config=config, seed=1_571_999
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("policy", "scientific_promotion_allowed", True),
        ("resources", "optimizer_updates", 3),
        ("metrics", "asi_whole_stream_preupdate_online_accuracy", 0.0),
        ("state", "alpha_bytes", -1),
        ("provenance", "official_commit", "0" * 40),
    ],
)
def test_validator_rejects_tampering(section: str, field: str, replacement: object) -> None:
    inputs = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], np.int32)
    config = AdaLinConfig(
        tasks=1, examples_per_task=4, batch_size=2, hidden_widths=(3, 2), classes=2
    )
    result = run_adalin_development(inputs, labels, inputs, labels, config=config, seed=6)
    tampered = copy.deepcopy(result)
    tampered[section][field] = replacement
    with pytest.raises(ValueError):
        validate_adalin_result(tampered)


def test_validator_and_runner_reject_hostile_containers_without_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    hostile = HostileDict()
    with pytest.raises(ValueError, match="exact JSON builtin"):
        validate_adalin_result(hostile)
    assert hostile.calls == 0

    class HostileString(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    scalar = HostileString("unexpected")
    with pytest.raises(ValueError, match="exact JSON builtin"):
        validate_adalin_result({"schema": scalar})
    assert scalar.calls == 0

    class HostileArray:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not coerce")

    array = HostileArray()
    config = AdaLinConfig(
        tasks=1, examples_per_task=2, batch_size=1, hidden_widths=(2, 2), classes=2
    )
    with pytest.raises(ValueError, match="exact NumPy"):
        run_adalin_development(
            array,
            np.array([0, 1], np.int32),
            np.ones((2, 2), np.float32),
            np.array([0, 1], np.int32),
            config=config,
            seed=1,
        )
    assert array.calls == 0


def test_gradient_and_step_are_outer_jit_safe() -> None:
    config = AdaLinConfig(
        tasks=1, examples_per_task=2, batch_size=2, hidden_widths=(2, 2), classes=2
    )
    state = initialize_adalin_state(config, input_dim=2, classes=2, seed=2)
    inputs = jnp.eye(2, dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    gradients = jax.jit(
        lambda parameters, x, y: adalin_gradients(parameters, x, y, mechanism_enabled=True)
    )(state.parameters, inputs, labels)
    assert bool(jnp.all(jnp.isfinite(gradients.alpha1)))
    updated = jax.jit(
        lambda current, x, y: adalin_sgd_step(
            current, x, y, learning_rate=0.01, mechanism_enabled=True
        )
    )(state, inputs, labels)
    assert int(updated.updates) == 1
    invalid_inputs = inputs.at[0, 0].set(jnp.nan)
    safe, valid = jax.jit(
        lambda current, x, y: adalin_sgd_step_transaction(
            current, x, y, learning_rate=0.01, mechanism_enabled=True
        )
    )(state, invalid_inputs, labels)
    assert not bool(valid)
    assert int(safe.updates) == 0
    np.testing.assert_array_equal(safe.parameters.alpha1, state.parameters.alpha1)
    exposed = jax.jit(
        lambda current, x, y: adalin_sgd_step(
            current, x, y, learning_rate=0.01, mechanism_enabled=True
        )
    )(state, invalid_inputs, labels)
    assert bool(jnp.all(jnp.isnan(exposed.parameters.alpha1)))


def test_protocol_keeps_paper_and_asi_schedules_and_boundaries_explicit() -> None:
    assert ADALIN_PROTOCOL["paper_pmnist_tasks"] == 400
    assert ADALIN_PROTOCOL["paper_examples_per_task"] == 10_000
    assert ADALIN_PROTOCOL["paper_batch_size"] == 16
    assert ADALIN_PROTOCOL["asi_target_tasks"] == 200
    assert ADALIN_PROTOCOL["asi_examples_per_task"] == 5_000
    assert ADALIN_PROTOCOL["asi_batch_size"] == 1
    assert ADALIN_PROTOCOL["learner_observes_task_boundary"] is False
