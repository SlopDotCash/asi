from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.bimu import (
    BIMU_PAPER_CONFIG,
    BIMU_PROTOCOL,
    BiMUConfig,
    BiMUState,
    _apply_gradient,
    _state_sha256,
    bimu_update,
    bimu_update_transaction,
    build_task_schedule,
    concrete_binary_weights,
    late_window_mean,
    posterior_probability,
    posterior_probability_transaction,
    run_bimu_development,
    sample_binary_weights,
    validate_bimu_result,
)


def _tiny_config(**overrides: object) -> BiMUConfig:
    values: dict[str, object] = {
        "input_dim": 4,
        "hidden_units": 3,
        "n_classes": 2,
        "n_tasks": 5,
        "train_examples_per_task": 4,
        "test_examples_per_task": 2,
        "train_samples": 2,
        "test_samples": 3,
        "query_samples": 3,
        "temperature": 1.0,
        "likelihood_multiplier": 2.0,
        "kl_multiplier": 0.5,
        "alpha_max": 0.1,
        "memory_window": 7,
        "gradient_scale": 1.5,
        "query_threshold": 0.0,
    }
    values.update(overrides)
    return BiMUConfig(**values)


def _tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(
        [
            [2.0, 1.0, -1.0, -2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [1.5, -0.5, 0.5, -1.5],
            [-1.5, 0.5, -0.5, 1.5],
        ],
        dtype=np.float32,
    )
    train_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    test_x = np.asarray([[1.0, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 1.0]], dtype=np.float32)
    test_y = np.asarray([0, 1], dtype=np.int32)
    return train_x, train_y, test_x, test_y


def test_protocol_pins_official_source_and_paper_configuration() -> None:
    assert BIMU_PROTOCOL["paper_revision"] == "arXiv:2605.30198v1"
    assert BIMU_PROTOCOL["official_code_commit"] == ("1b8a1a1fb892fbe89401390b3ff9611d7f3a5168")
    assert BIMU_PROTOCOL["prng_implementation"] == "threefry2x32"
    assert BIMU_PAPER_CONFIG.n_tasks == 1000
    assert BIMU_PAPER_CONFIG.train_examples_per_task == 60_000
    assert BIMU_PAPER_CONFIG.hidden_units == 100
    assert BIMU_PAPER_CONFIG.train_samples == 5
    assert BIMU_PAPER_CONFIG.test_samples == 5
    assert BIMU_PAPER_CONFIG.likelihood_multiplier == pytest.approx(161.3)
    assert BIMU_PAPER_CONFIG.kl_multiplier == pytest.approx(3.76)
    assert BIMU_PAPER_CONFIG.alpha_max == pytest.approx(0.0023)
    assert BIMU_PAPER_CONFIG.memory_window == 700
    assert BIMU_PAPER_CONFIG.gradient_scale == pytest.approx(4.9)
    assert BIMU_PAPER_CONFIG.matches_paper_configuration


def test_equation_update_is_jittable_and_matches_scaled_official_rule() -> None:
    state = jnp.array([0.0, 1.0], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([2.0, -0.5], dtype=jnp.float32)
    update = jax.jit(
        lambda s, g, p: bimu_update(
            s,
            g,
            p,
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    updated = update(state, gradient, prior)
    scaled_gradient = 2.0 * gradient
    uncertainty = 3.0 * (1.0 - jnp.tanh(state) ** 2)
    reciprocal = uncertainty + 2.0 * jnp.tanh(state) * scaled_gradient + 1.0
    reciprocal += 2.0 * jnp.abs(scaled_gradient)
    expected = state - (4.0 * scaled_gradient + (state - prior) * uncertainty / 10) / reciprocal
    np.testing.assert_allclose(updated, expected, rtol=1e-6)


def test_binary_and_concrete_samples_are_explicit_and_reproducible() -> None:
    natural = jnp.asarray([-2.0, 0.0, 2.0], dtype=jnp.float32)
    key = jax.random.key(17)
    binary = sample_binary_weights(natural, key)
    np.testing.assert_array_equal(binary, sample_binary_weights(natural, key))
    assert set(np.asarray(binary).tolist()) <= {-1.0, 1.0}
    concrete = concrete_binary_weights(natural, key, temperature=0.7)
    np.testing.assert_allclose(concrete, concrete_binary_weights(natural, key, temperature=0.7))
    assert bool(jnp.all(jnp.abs(concrete) < 1.0))
    derivative = jax.grad(lambda x: concrete_binary_weights(x, key, temperature=0.7).sum())(natural)
    assert bool(jnp.all(jnp.isfinite(derivative)))


def test_mechanism_off_removes_only_controlled_forgetting() -> None:
    state = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    gradient = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    result = bimu_update(
        state,
        gradient,
        jnp.zeros_like(state),
        memory_window=None,
        alpha_max=0.5,
        likelihood_multiplier=1.0,
        kl_multiplier=1.0,
        gradient_scale=1.0,
    )
    uncertainty = 1.0 - jnp.tanh(state) ** 2
    eta = 1.0 / (uncertainty + 2.0 * jnp.tanh(state) * gradient + 2.0 + 2.0 * jnp.abs(gradient))
    np.testing.assert_allclose(result, state - eta * gradient, rtol=1e-6)


def test_equation_kernel_does_not_hide_official_zero_gradient_gate() -> None:
    state = jnp.asarray([0.5], dtype=jnp.float32)
    equation_result = bimu_update(
        state,
        jnp.zeros_like(state),
        jnp.zeros_like(state),
        memory_window=5,
        alpha_max=0.5,
    )
    assert not np.array_equal(equation_result, state)
    model_state = BiMUState(
        input_hidden=state.reshape(1, 1),
        hidden_output=state.reshape(1, 1),
    )
    gradient = BiMUState(
        input_hidden=jnp.zeros((1, 1), dtype=jnp.float32),
        hidden_output=jnp.zeros((1, 1), dtype=jnp.float32),
    )
    gated = _apply_gradient(model_state, gradient, _tiny_config())
    np.testing.assert_array_equal(gated.input_hidden, model_state.input_hidden)
    np.testing.assert_array_equal(gated.hidden_output, model_state.hidden_output)


def test_task_schedule_is_deterministic_and_task_private() -> None:
    config = _tiny_config()
    first = build_task_schedule(config, seed=9)
    second = build_task_schedule(config, seed=9)
    assert first == second
    assert first != build_task_schedule(config, seed=10)
    assert len(first) == config.n_tasks
    assert all(sorted(permutation) == list(range(config.input_dim)) for permutation in first)
    assert config.learner_observes_task_boundary is False


def test_tiny_runner_is_end_to_end_strict_and_keeps_metrics_separate() -> None:
    config = _tiny_config()
    payload = run_bimu_development(*_tiny_data(), config=config, seed=23)
    validate_bimu_result(payload)
    metrics = payload["metrics"]
    counters = payload["counters"]
    resources = payload["resources"]
    assert metrics["paper_late_five_test_accuracy"] == pytest.approx(
        np.mean(metrics["final_five_test_accuracy"])
    )
    assert len(metrics["final_five_test_accuracy"]) == 5
    assert "asi_whole_stream_online_accuracy" in metrics
    assert counters["environment_steps"] == 20
    assert counters["observations"] == 20
    assert counters["label_queries"] == 20
    assert counters["optimizer_updates"] == 20
    assert counters["optimizer_seen"] == 20
    assert counters["model_forward_queries"] == 20 * (3 + 2) + 5 * 2 * 3
    assert resources["parameter_numeric_bytes"] == (4 * 3 + 3 * 2) * 4
    assert resources["optimizer_state_numeric_bytes"] == 8
    assert resources["initial_persistent_numeric_bytes"] == (4 * 3 + 3 * 2) * 4 + 8
    assert (
        resources["final_persistent_numeric_bytes"] == resources["initial_persistent_numeric_bytes"]
    )
    assert payload["comparison"]["paper_comparable"] is False
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False


def test_runner_replays_schedule_metrics_and_state_from_same_seed() -> None:
    config = _tiny_config()
    first = run_bimu_development(*_tiny_data(), config=config, seed=4)
    second = run_bimu_development(*_tiny_data(), config=config, seed=4)
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["metrics"] == second["metrics"]
    assert first["counters"] == second["counters"]


def test_paper_metric_evaluation_occurs_only_after_the_complete_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alberta_framework.benchmarks.bimu as module

    train_x, train_y, test_x, test_y = _tiny_data()
    test_x = np.full_like(test_x, 99.0)
    phases: list[str] = []
    original_gradient = module._concrete_mean_gradient
    original_logits = module._binary_logits

    def recording_gradient(*args: object, **kwargs: object) -> BiMUState:
        phases.append("train")
        return original_gradient(*args, **kwargs)

    def recording_logits(
        state: BiMUState, features: jax.Array, key: jax.Array, *, n_samples: int
    ) -> jax.Array:
        phases.append("test" if bool(jnp.all(features == 99.0)) else "train-query")
        return original_logits(state, features, key, n_samples=n_samples)

    monkeypatch.setattr(module, "_concrete_mean_gradient", recording_gradient)
    monkeypatch.setattr(module, "_binary_logits", recording_logits)
    run_bimu_development(train_x, train_y, test_x, test_y, config=_tiny_config(), seed=29)
    first_test = phases.index("test")
    assert "train" in phases[:first_test]
    assert set(phases[first_test:]) == {"test"}


def test_state_identity_binds_live_optimizer_counters() -> None:
    state = BiMUState(
        input_hidden=jnp.zeros((2, 2), dtype=jnp.float32),
        hidden_output=jnp.zeros((2, 2), dtype=jnp.float32),
    )
    initial = _state_sha256(state, optimizer_step=0, optimizer_seen=0)
    assert _state_sha256(state, optimizer_step=1, optimizer_seen=1) != initial


def test_no_query_schedule_performs_no_updates() -> None:
    config = _tiny_config(query_threshold=1.0, memory_window=None)
    payload = run_bimu_development(*_tiny_data(), config=config, seed=5)
    validate_bimu_result(payload)
    assert payload["counters"]["label_queries"] == 0
    assert payload["counters"]["optimizer_updates"] == 0
    assert payload["counters"]["optimizer_seen"] == 20
    assert payload["resources"]["state_changed"] is True


def test_rng_domains_are_explicit_threefry_and_distinct_for_used_coordinates() -> None:
    import alberta_framework.benchmarks.bimu as module

    root = jax.random.key(23, impl="threefry2x32")
    keys = [
        module._stream_key(root, module._QUERY_DOMAIN, 102),
        module._stream_key(root, module._TRAIN_DOMAIN, 0),
        module._stream_key(root, module._TASK_PERMUTATION_DOMAIN, 96),
        module._stream_key(root, module._EXAMPLE_ORDER_DOMAIN, 0),
        module._stream_key(root, module._QUERY_DOMAIN, 200),
        module._stream_key(root, module._TEST_DOMAIN, 0),
    ]
    encoded = {bytes(np.asarray(jax.random.key_data(key), dtype=np.uint32)) for key in keys}
    assert len(encoded) == len(keys)


def test_inference_matches_official_mean_log_probability_not_majority_vote() -> None:
    import alberta_framework.benchmarks.bimu as module

    logits = jnp.asarray([[0.1, 0.0], [0.1, 0.0], [0.0, 10.0]], dtype=jnp.float32)
    prediction, _ = module._official_prediction_and_variation(logits)
    assert prediction == 1
    assert int(np.argmax(np.bincount(np.argmax(np.asarray(logits), axis=1)))) == 0


def test_receipt_derives_metrics_and_disclaims_unverifiable_digests() -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=37)
    assurance = payload["receipt_assurance"]
    assert assurance["content_digests_recomputed_by_validator"] is False
    assert assurance["metrics_recomputed_from_execution_transcript"] is False
    assert assurance["authenticated_execution_attestation"] is False

    forged_metric = deepcopy(payload)
    forged_metric["metrics"]["asi_whole_stream_online_accuracy"] = 0.123
    with pytest.raises(ValueError, match="reported count"):
        validate_bimu_result(forged_metric)

    reported_identity = deepcopy(payload)
    reported_identity["dataset_sha256"] = "0" * 64
    validate_bimu_result(reported_identity)


def test_validator_binds_zero_threshold_to_all_label_queries() -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=37)
    forged = deepcopy(payload)
    forged["counters"]["label_queries"] = 0
    forged["counters"]["optimizer_updates"] = 0
    forged["counters"]["model_forward_queries"] = 90

    with pytest.raises(ValueError, match="zero query threshold"):
        validate_bimu_result(forged)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evidence_policy", "scientific_promotion_allowed"), True),
        (("counters", "optimizer_updates"), 999),
        (("resources", "final_persistent_numeric_bytes"), 1),
        (("metrics", "paper_late_five_test_accuracy"), 0.123456),
    ],
)
def test_validator_fails_closed_on_policy_accounting_and_metric_drift(
    path: tuple[str, str], value: object
) -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=11)
    corrupted = deepcopy(payload)
    corrupted[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_bimu_result(corrupted)


def test_validator_rejects_unknown_fields() -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=11)
    payload["claim"] = "sota"
    with pytest.raises(ValueError, match="fields"):
        validate_bimu_result(payload)


def test_late_window_metric_remains_separate_from_whole_stream() -> None:
    assert late_window_mean([0.1, 0.2, 0.8, 0.9], window=2) == pytest.approx(0.85)
    np.testing.assert_allclose(posterior_probability(jnp.asarray([0.0])), [0.5])


def test_transactions_are_outer_jit_safe_and_fail_closed() -> None:
    update = jax.jit(
        lambda state, gradient, prior: bimu_update(
            state,
            gradient,
            prior,
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    assert bool(jnp.all(jnp.isfinite(update(jnp.zeros(2), jnp.ones(2), jnp.zeros(2)))))
    transact = jax.jit(
        lambda state, gradient: bimu_update_transaction(
            state,
            gradient,
            jnp.zeros(2),
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    for state, gradient in (
        (jnp.asarray([jnp.nan, 0.0]), jnp.ones(2)),
        (jnp.zeros(2), jnp.asarray([jnp.inf, 0.0])),
    ):
        safe, valid = transact(state, gradient)
        assert bool(jnp.all(jnp.isfinite(safe)))
        assert not bool(valid)


def test_primitives_reject_hostile_array_protocol_objects_without_calling_them() -> None:
    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        posterior_probability(hostile)
    assert hostile.calls == 0

    data = _tiny_data()
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        run_bimu_development(hostile, data[1], data[2], data[3], config=_tiny_config(), seed=1)
    assert hostile.calls == 0


def test_hostile_result_scalars_and_keys_are_rejected_before_hooks() -> None:
    class HostileString(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not run")

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=31)
    hostile_value = HostileString("complete")
    payload["status"] = hostile_value
    with pytest.raises(ValueError, match="identity"):
        validate_bimu_result(payload)
    assert hostile_value.calls == 0

    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=31)
    hostile_key = HostileString("claim")
    payload[hostile_key] = False
    hostile_key.calls = 0
    with pytest.raises(ValueError, match="fields"):
        validate_bimu_result(payload)
    assert hostile_key.calls == 0

    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=31)
    hostile_assurance = HostileString("true")
    payload["receipt_assurance"]["authenticated_execution_attestation"] = hostile_assurance
    with pytest.raises(ValueError, match="assurance"):
        validate_bimu_result(payload)
    assert hostile_assurance.calls == 0

    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=31)
    hostile_role = HostileString("telemetry_only")
    payload["timing"]["role"] = hostile_role
    with pytest.raises(ValueError, match="timing"):
        validate_bimu_result(payload)
    assert hostile_role.calls == 0


def test_configuration_and_concrete_scalars_are_exact_and_bounded() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        _tiny_config(input_dim=2**31)
    with pytest.raises(ValueError, match="temperature"):
        concrete_binary_weights(jnp.zeros(2), jax.random.key(1), temperature=True)


def test_float32_overflow_is_invalid_not_laundered() -> None:
    maximum = jnp.finfo(jnp.float32).max
    transact = jax.jit(
        lambda gradient: bimu_update_transaction(
            jnp.zeros(2), gradient, jnp.zeros(2), memory_window=10, alpha_max=1.0
        )
    )
    safe, valid = transact(jnp.full((2,), maximum))
    assert bool(jnp.all(jnp.isfinite(safe)))
    assert not bool(valid)
    posterior, posterior_valid = jax.jit(posterior_probability_transaction)(jnp.asarray([jnp.inf]))
    np.testing.assert_array_equal(posterior, [0.5])
    assert not bool(posterior_valid)
    finite_posterior, finite_valid = jax.jit(posterior_probability_transaction)(
        jnp.asarray([maximum])
    )
    np.testing.assert_array_equal(finite_posterior, [0.5])
    assert not bool(finite_valid)
