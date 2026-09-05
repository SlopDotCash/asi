from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.activation_feature_ipmnist import (
    ACTIVATION_FEATURE_SOURCES,
    ACTIVATION_FEATURE_SPECS,
    DEVELOPMENT_SEEDS,
    _aid_forward,
    _preflight_activation_feature_resources,
    activation_feature_result_payload,
    activation_feature_spec,
    run_activation_feature_arm,
    validate_activation_feature_result,
    validate_matched_activation_feature_results,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    ema_normalize,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import _PLASTICITY_LOSS_FLOOR, IPMNISTConfig

SMALL = IPMNISTConfig(n_tasks=2, task_length=4, input_dim=4, hidden1=4, hidden2=4, n_classes=2)


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[-1, -1, 1, 1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, 1, -1]],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    return x, y


def test_sources_pin_exact_paper_revisions_and_code_availability() -> None:
    assert ACTIVATION_FEATURE_SOURCES["smooth_leaky"]["paper_revision"] == "arXiv:2509.22562v4"
    assert ACTIVATION_FEATURE_SOURCES["smooth_leaky"]["official_code_revision"] == (
        "anonymous.4open.science snapshot activations_plasticity-E431"
    )
    assert ACTIVATION_FEATURE_SOURCES["aid"]["official_code_revision"] == (
        "none disclosed; paper Algorithm 2 is the implementation source"
    )
    assert ACTIVATION_FEATURE_SOURCES["deep_fourier"]["paper_revision"] == ("arXiv:2410.20634v1")


def test_mechanism_off_arms_are_exact_current_runner_control() -> None:
    x, y = _data()
    control = run_screening_config(x, y, screening_spec("sgd_ema_norm_d099"), 3, SMALL)
    for arm in ("smooth_leaky_off", "aid_off", "deep_fourier_off"):
        result = run_activation_feature_arm(x, y, arm=arm, seed=3, config=SMALL)
        np.testing.assert_array_equal(result.per_task_accuracy, control.per_task_accuracy)
        np.testing.assert_array_equal(result.per_task_loss, control.per_task_loss)
        np.testing.assert_array_equal(result.per_task_plasticity, control.per_task_plasticity)

    params = {
        "w1": jnp.zeros((4, 4)),
        "b1": jnp.zeros(4),
        "w2": jnp.zeros((4, 4)),
        "b2": jnp.zeros(4),
        "w3": jnp.zeros((4, 2)),
        "b3": jnp.zeros(2),
    }
    base = screening_spec("sgd_ema_norm_d099")
    base_init, base_step = base.factory(base.hyperparameters)
    for arm in ("smooth_leaky_off", "aid_off", "deep_fourier_off"):
        off = activation_feature_spec(arm)
        off_init, off_step = off.factory(off.hyperparameters)
        expected = base_step(
            params, base_init(params), jnp.ones(4), jnp.asarray(1), jax.random.key(1)
        )
        actual = off_step(params, off_init(params), jnp.ones(4), jnp.asarray(1), jax.random.key(1))
        for expected_leaf, actual_leaf in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_all_candidate_and_causal_arms_execute_end_to_end() -> None:
    x, y = _data()
    for arm in (
        "smooth_leaky",
        "smooth_leaky_fixed_leak",
        "aid",
        "aid_expected",
        "ordinary_dropout",
        "deep_fourier",
        "deep_fourier_first_layer",
        "deep_fourier_sine_only",
    ):
        result = run_activation_feature_arm(x, y, arm=arm, seed=5, config=SMALL)
        assert result.per_task_accuracy.shape == (2,)
        assert np.isfinite(result.per_task_loss).all()


def test_schedule_memory_is_bounded_before_runner_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = IPMNISTConfig(
        n_tasks=2_000_000,
        task_length=1,
        input_dim=784,
        hidden1=1,
        hidden2=1,
        n_classes=1,
    )
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.activation_feature_ipmnist.run_screening_config",
        forbidden,
    )
    with pytest.raises(ValueError, match="schedule exceeds"):
        run_activation_feature_arm(
            np.zeros((1, 784), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            arm="aid",
            seed=0,
            config=huge,
        )
    assert calls == 0


def test_schedule_memory_bound_has_exact_adjacent_boundary() -> None:
    last_fit = IPMNISTConfig(
        n_tasks=33_554_432,
        task_length=1,
        input_dim=1,
        hidden1=1,
        hidden2=1,
        n_classes=1,
    )
    first_overflow = IPMNISTConfig(
        n_tasks=33_554_433,
        task_length=1,
        input_dim=1,
        hidden1=1,
        hidden2=1,
        n_classes=1,
    )
    _preflight_activation_feature_resources(last_fit)
    with pytest.raises(ValueError, match="268435456-byte bound"):
        _preflight_activation_feature_resources(first_overflow)


def test_factories_are_jittable_and_aid_is_deterministic_per_seed() -> None:
    x, y = _data()
    first = run_activation_feature_arm(x, y, arm="aid", seed=7, config=SMALL)
    second = run_activation_feature_arm(x, y, arm="aid", seed=7, config=SMALL)
    np.testing.assert_array_equal(first.per_task_loss, second.per_task_loss)
    spec = activation_feature_spec("smooth_leaky")
    init, step = spec.factory(spec.hyperparameters)
    params = {
        "w1": jnp.zeros((4, 4)),
        "b1": jnp.zeros(4),
        "w2": jnp.zeros((4, 4)),
        "b2": jnp.zeros(4),
        "w3": jnp.zeros((4, 2)),
        "b3": jnp.zeros(2),
    }
    state = init(params)
    eager = step(params, state, jnp.ones(4), jnp.asarray(1), jax.random.key(0))
    compiled = jax.jit(step)(params, state, jnp.ones(4), jnp.asarray(1), jax.random.key(0))
    np.testing.assert_array_equal(compiled[2][1], eager[2][1])


def _stochastic_probe_params() -> dict[str, jnp.ndarray]:
    keys = jax.random.split(jax.random.key(2), 6)
    shapes = ((6, 8), (8,), (8, 8), (8,), (8, 3), (3,))
    widths = (0.9, 0.3, 0.9, 0.3, 0.9, 0.3)
    names = ("w1", "b1", "w2", "b2", "w3", "b3")
    return {
        name: jax.random.normal(key, shape) * width
        for name, key, shape, width in zip(names, keys, shapes, widths, strict=True)
    }


@pytest.mark.parametrize("arm", ["aid", "ordinary_dropout"])
def test_stochastic_arm_plasticity_holds_the_step_activation_mask_fixed(arm: str) -> None:
    """Both sides of the plasticity ratio must use one activation rule.

    The lane keeps ``run_screening_config``'s online metrics, and that runner's
    ``_step_metrics`` evaluates ``loss`` and ``loss_after`` with a single
    forward function, so plasticity isolates the parameter update.  ``aid`` and
    ``ordinary_dropout`` are the only registered arms whose forward depends on
    ``training``: their pre-update loss comes from a sampled mask, so the
    post-update loss has to be scored under that same sampled mask rather than
    under the deterministic evaluation rule.
    """
    params = _stochastic_probe_params()
    x = jnp.asarray([1.0, -2.0, 0.5, 0.25, -0.75, 1.5], dtype=jnp.float32)
    y = jnp.asarray(2, dtype=jnp.int32)
    key = jax.random.key(2)
    spec = activation_feature_spec(arm)
    init, step = spec.factory(spec.hyperparameters)
    state = init(params)
    params_after, _next_state, (_accuracy, loss, plasticity) = step(params, state, x, y, key)

    x_norm, _next_norm = ema_normalize(state.norm, x, 0.99, 1e-8)
    ordinary = arm == "ordinary_dropout"
    denominator = jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR)

    def ratio(training: bool) -> float:
        after = -jax.nn.log_softmax(
            _aid_forward(params_after, x_norm, key, training, expected=False, ordinary=ordinary)
        )[y]
        return float(jnp.clip(1.0 - after / denominator, 0.0, 1.0))

    mask_matched = ratio(True)
    evaluation_rule = ratio(False)
    # The two rules must actually disagree here, or this pins nothing.
    assert abs(mask_matched - evaluation_rule) > 0.5
    assert 0.0 < mask_matched < 1.0 and 0.0 < evaluation_rule < 1.0
    assert float(plasticity) == pytest.approx(mask_matched, abs=1e-6)


def test_activation_shard_is_independent_of_ambient_default_prng() -> None:
    x, y = _data()
    with jax.default_prng_impl("threefry2x32"):
        expected = run_activation_feature_arm(x, y, arm="aid", seed=4, config=SMALL)
    with jax.default_prng_impl("rbg"):
        actual = run_activation_feature_arm(x, y, arm="aid", seed=4, config=SMALL)
    np.testing.assert_array_equal(actual.per_task_accuracy, expected.per_task_accuracy)
    np.testing.assert_array_equal(actual.per_task_loss, expected.per_task_loss)
    np.testing.assert_array_equal(actual.per_task_plasticity, expected.per_task_plasticity)


def test_result_receipt_is_exact_bounded_and_permanently_nonpromoting() -> None:
    x, y = _data()
    result = run_activation_feature_arm(x, y, arm="aid", seed=1, config=SMALL)
    payload = activation_feature_result_payload(result, outcome="inconclusive")
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["outcome_retained"] is True
    assert payload["development_seed_protocol"] == list(DEVELOPMENT_SEEDS)
    assert payload["resources"]["data_steps"] == 8
    assert payload["resources"]["model_queries"] == 16
    # Two training-mode forwards per step, each masking hidden1 + hidden2 units.
    assert payload["resources"]["random_bernoulli_variates"] == 128
    assert validate_activation_feature_result(copy.deepcopy(payload)) == payload

    hostile = copy.deepcopy(payload)
    hostile["resources"]["model_queries"] = True
    with pytest.raises(ValueError, match="model_queries"):
        validate_activation_feature_result(hostile)
    extra = copy.deepcopy(payload)
    extra["surprise"] = 1
    with pytest.raises(ValueError, match="exact fields"):
        validate_activation_feature_result(extra)


def test_deep_fourier_receipt_discloses_active_parameter_and_comparability_gap() -> None:
    x, y = _data()
    result = run_activation_feature_arm(x, y, arm="deep_fourier", seed=2, config=SMALL)
    payload = activation_feature_result_payload(result, outcome="inconclusive")
    assert (
        payload["resources"]["active_parameter_scalars"]
        < payload["resources"]["allocated_parameter_scalars"]
    )
    assert "parameterization" in payload["comparability_gaps"]
    assert payload["paper_metric_reported"] is False


def test_matched_validator_requires_all_arms_and_exact_comparison_axes() -> None:
    x, y = _data()
    payloads = [
        activation_feature_result_payload(
            run_activation_feature_arm(x, y, arm=arm, seed=0, config=SMALL),
            outcome="inconclusive",
        )
        for arm in ACTIVATION_FEATURE_SPECS
    ]
    assert len(validate_matched_activation_feature_results(copy.deepcopy(payloads))) == 11

    missing = copy.deepcopy(payloads[:-1])
    with pytest.raises(ValueError, match="every registered arm"):
        validate_matched_activation_feature_results(missing)
    drift = copy.deepcopy(payloads)
    drift[-1]["seed"] = 1
    with pytest.raises(ValueError, match="differs on seed"):
        validate_matched_activation_feature_results(drift)
    identity_drift = copy.deepcopy(payloads)
    identity_drift[-1]["execution_identity"]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution_identity"):
        validate_matched_activation_feature_results(identity_drift)


def test_receipt_rejects_unfrozen_seed_outcome_and_negative_retention_drift() -> None:
    x, y = _data()
    result = run_activation_feature_arm(x, y, arm="aid", seed=0, config=SMALL)
    payload = activation_feature_result_payload(result, outcome="rejected")
    assert payload["outcome_retained"] is True

    bad_outcome = copy.deepcopy(payload)
    bad_outcome["outcome"] = "completed_development_run"
    with pytest.raises(ValueError, match="supported, rejected, or inconclusive"):
        validate_activation_feature_result(bad_outcome)
    dropped = copy.deepcopy(payload)
    dropped["outcome_retained"] = False
    with pytest.raises(ValueError, match="negative outcomes"):
        validate_activation_feature_result(dropped)
    unfrozen = copy.deepcopy(payload)
    unfrozen["seed"] = 99
    with pytest.raises(ValueError, match="frozen development seed"):
        validate_activation_feature_result(unfrozen)


def test_receipt_rejects_config_whose_schedule_cannot_execute() -> None:
    x, y = _data()
    payload = activation_feature_result_payload(
        run_activation_feature_arm(x, y, arm="aid", seed=0, config=SMALL),
        outcome="inconclusive",
    )
    payload["config"] = {
        "n_tasks": 2_000_000,
        "task_length": 1,
        "input_dim": 784,
        "hidden1": 300,
        "hidden2": 150,
        "n_classes": 10,
    }
    with pytest.raises(ValueError, match="schedule exceeds"):
        validate_activation_feature_result(payload)


def test_receipt_rejects_forged_identity_and_retained_schedule_bytes() -> None:
    x, y = _data()
    payload = activation_feature_result_payload(
        run_activation_feature_arm(x, y, arm="aid", seed=0, config=SMALL),
        outcome="inconclusive",
    )
    forged_runtime = copy.deepcopy(payload)
    forged_runtime["execution_identity"]["runtime"] = ["forged"] * 4
    with pytest.raises(ValueError, match="current runtime identity"):
        validate_activation_feature_result(forged_runtime)
    forged_peak = copy.deepcopy(payload)
    forged_peak["resources"]["retained_schedule_numeric_bytes"] = 1
    with pytest.raises(ValueError, match="retained_schedule_numeric_bytes"):
        validate_activation_feature_result(forged_peak)


def test_receipt_revalidates_frozen_result_without_array_protocol_dispatch() -> None:
    x, y = _data()
    result = run_activation_feature_arm(x, y, arm="aid", seed=0, config=SMALL)

    class HostileArray:
        def __array__(self) -> np.ndarray:
            raise AssertionError("must reject before array coercion")

    object.__setattr__(result.screening, "per_task_accuracy", HostileArray())
    with pytest.raises(ValueError, match="per_task_accuracy"):
        activation_feature_result_payload(result, outcome="inconclusive")
