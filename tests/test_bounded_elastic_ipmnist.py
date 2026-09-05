from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    BoundedStructureState,
    ScreeningRunResult,
    bounded_elastic_development_result_payload,
    bounded_masked_loss,
    bounded_structure_update,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    BOUNDED_ELASTIC_PROTOCOL,
    registered_bounded_elastic_hyperparameters,
    validate_bounded_elastic_development_result,
    validate_matched_bounded_elastic_results,
)


def _params() -> dict[str, jax.Array]:
    return {
        "w1": jnp.asarray([[0.4, -0.3, 0.2], [0.1, 0.5, -0.2]], dtype=jnp.float32),
        "b1": jnp.asarray([0.1, 0.2, -0.1], dtype=jnp.float32),
        "w2": jnp.asarray([[0.3, 0.2], [-0.4, 0.1], [0.2, 0.5]], dtype=jnp.float32),
        "b2": jnp.asarray([0.0, 0.1], dtype=jnp.float32),
        "w3": jnp.asarray([[0.4, -0.2], [0.1, 0.3]], dtype=jnp.float32),
        "b3": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    }


def _state(*, step: int, activation_sum: tuple[float, float, float]) -> BoundedStructureState:
    return BoundedStructureState(  # type: ignore[call-arg]
        active1=jnp.asarray([True, True, False]),
        activation_sum1=jnp.asarray(activation_sum, dtype=jnp.float32),
        step=jnp.asarray(step, dtype=jnp.int32),
    )


def test_mechanism_off_is_exact_masked_sgd_reduction() -> None:
    params = _params()
    state = _state(step=0, activation_sum=(0.0, 0.0, 0.0))
    x = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    y = jnp.asarray(1, dtype=jnp.int32)
    (_, (_, hidden1)), grads = jax.value_and_grad(bounded_masked_loss, has_aux=True)(
        params, x, y, state.active1
    )
    hp = registered_bounded_elastic_hyperparameters("bounded_structure_off")
    actual, new_state = bounded_structure_update(
        params, state, grads, hidden1, jax.random.key(4), hp
    )
    for name in params:
        np.testing.assert_array_equal(actual[name], params[name] - 1e-3 * grads[name])
    np.testing.assert_array_equal(new_state.active1, state.active1)
    assert int(new_state.step) == 1


def test_growth_activates_one_fresh_bounded_slot_and_is_jittable() -> None:
    params = _params()
    state = BoundedStructureState(  # type: ignore[call-arg]
        active1=jnp.asarray([True, False, False]),
        activation_sum1=jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
        step=jnp.asarray(4999, dtype=jnp.int32),
    )
    grads = jax.tree.map(jnp.zeros_like, params)
    hp = registered_bounded_elastic_hyperparameters("bounded_growth")
    update = jax.jit(
        lambda p, s, g, h, k: bounded_structure_update(p, s, g, h, k, hp)
    )
    actual, new_state = update(
        params,
        state,
        grads,
        jnp.asarray([0.7, 0.0, 0.0], dtype=jnp.float32),
        jax.random.key(9),
    )
    np.testing.assert_array_equal(new_state.active1, np.asarray([True, True, False]))
    assert int(new_state.step) == 5000
    assert not np.array_equal(np.asarray(actual["w1"][:, 1]), np.asarray(params["w1"][:, 1]))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in actual.values())


def test_elastic_prunes_least_active_then_freshly_grows_without_size_drift() -> None:
    params = _params()
    state = _state(step=4999, activation_sum=(10.0, 0.01, 0.0))
    grads = jax.tree.map(jnp.zeros_like, params)
    hp = registered_bounded_elastic_hyperparameters("bounded_elastic")
    actual, new_state = bounded_structure_update(
        params,
        state,
        grads,
        jnp.zeros(3, dtype=jnp.float32),
        jax.random.key(13),
        hp,
    )
    np.testing.assert_array_equal(new_state.active1, state.active1)
    np.testing.assert_array_equal(new_state.activation_sum1, jnp.zeros(3))
    assert not np.array_equal(np.asarray(actual["w1"][:, 1]), np.asarray(params["w1"][:, 1]))
    np.testing.assert_array_equal(actual["w1"][:, 0], params["w1"][:, 0])


def test_registry_contains_complete_bounded_comparison() -> None:
    mechanisms = {
        "bounded_structure_off": "bounded_structure_off",
        "bounded_growth": "bounded_growth",
        "bounded_elastic": "bounded_elastic",
        "bounded_fixed_cbp": "fixed_capacity_cbp",
    }
    for arm, mechanism in mechanisms.items():
        spec = screening_spec(arm)
        assert spec.mechanism == mechanism
        assert spec.hyperparameters == registered_bounded_elastic_hyperparameters(arm)
        assert spec.noise_update is None


def test_fixed_capacity_cbp_control_runs_one_jitted_step() -> None:
    params = _params()
    spec = screening_spec("bounded_fixed_cbp")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    update = jax.jit(step_fn)
    actual, _, metrics = update(
        params,
        state,
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jax.random.key(21),
    )
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in actual.values())
    assert all(bool(jnp.isfinite(value)) for value in metrics)


def test_runner_rejects_unregistered_boundary_length_before_execution() -> None:
    config = IPMNISTConfig(
        n_tasks=1, task_length=10, input_dim=2, hidden1=4, hidden2=2, n_classes=2
    )
    with pytest.raises(ValueError, match="task_length=5000"):
        run_screening_config(
            np.zeros((10, 2), dtype=np.float32),
            np.zeros(10, dtype=np.int32),
            screening_spec("bounded_growth"),
            seed=0,
            config=config,
        )


def test_growth_runs_end_to_end_on_current_runner_without_benchmark_outputs() -> None:
    config = IPMNISTConfig(
        n_tasks=1, task_length=5000, input_dim=2, hidden1=4, hidden2=2, n_classes=2
    )
    data_x = np.random.default_rng(7).uniform(-1.0, 1.0, size=(5000, 2)).astype(np.float32)
    data_y = np.arange(5000, dtype=np.int32) % 2
    result = run_screening_config(
        data_x,
        data_y,
        screening_spec("bounded_growth"),
        seed=5,
        config=config,
    )
    assert result.per_task_accuracy.shape == (1,)
    receipt = bounded_elastic_development_result_payload(result, outcome="inconclusive")
    resources = receipt["resources"]
    assert isinstance(resources, dict)
    assert resources["units_grown"] == 1
    assert resources["final_active_hidden1_units"] == 3


def _result(arm: str) -> ScreeningRunResult:
    config = IPMNISTConfig(
        n_tasks=1, task_length=5000, input_dim=2, hidden1=4, hidden2=2, n_classes=2
    )
    return ScreeningRunResult(
        config_name=arm,
        base_learner="upgd_w",
        hyperparameters=dict(screening_spec(arm).hyperparameters),
        seed=17,
        config=config,
        per_task_accuracy=np.asarray([0.6], dtype=np.float64),
        per_task_loss=np.asarray([0.7], dtype=np.float64),
        per_task_plasticity=np.asarray([0.1], dtype=np.float64),
        wall_clock_seconds=2.0,
    )


def test_receipt_enforces_exact_resources_and_permanent_nonpromotion() -> None:
    payload = bounded_elastic_development_result_payload(
        _result("bounded_elastic"), outcome="rejected"
    )
    assert payload["outcome_retained"] is True
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    resources = payload["resources"]
    assert isinstance(resources, dict)
    assert resources["persistent_bytes"] == 136
    assert resources["peak_persistent_bytes_budget"] == 168
    assert resources["final_active_parameter_bytes_budget"] == 112
    assert resources["final_active_parameter_bytes"] == 72
    assert resources["model_queries"] == 10_000

    hostile_resource = deepcopy(payload)
    hostile_resources = hostile_resource["resources"]
    assert isinstance(hostile_resources, dict)
    hostile_resources["peak_persistent_bytes_budget"] = 167
    with pytest.raises(ValueError, match="memory resources"):
        validate_bounded_elastic_development_result(hostile_resource)

    hostile_queries = deepcopy(payload)
    hostile_query_resources = hostile_queries["resources"]
    assert isinstance(hostile_query_resources, dict)
    hostile_query_resources["model_queries"] = 9999
    with pytest.raises(ValueError, match="step/query resources"):
        validate_bounded_elastic_development_result(hostile_queries)

    hostile_claim = deepcopy(payload)
    hostile_claim["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        validate_bounded_elastic_development_result(hostile_claim)

    hostile_schema = deepcopy(payload)
    hostile_schema["paper_parity"] = True
    with pytest.raises(ValueError, match="result keys"):
        validate_bounded_elastic_development_result(hostile_schema)

    class HostileDict(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact object"):
        validate_bounded_elastic_development_result(HostileDict(payload))

    hostile_dimension = deepcopy(payload)
    hostile_dimension["n_tasks"] = 2**31
    with pytest.raises(ValueError, match="signed-int32"):
        validate_bounded_elastic_development_result(hostile_dimension)


def test_matched_validator_requires_all_arms_workload_and_budgets() -> None:
    arms = (
        "bounded_structure_off",
        "bounded_growth",
        "bounded_elastic",
        "bounded_fixed_cbp",
    )
    payloads = [
        bounded_elastic_development_result_payload(_result(arm), outcome="inconclusive")
        for arm in arms
    ]
    assert len(validate_matched_bounded_elastic_results(payloads)) == 4
    mismatched = deepcopy(payloads)
    mismatched[0]["seed"] = 18
    with pytest.raises(ValueError, match="workload and information axes"):
        validate_matched_bounded_elastic_results(mismatched)


def test_protocol_fails_closed_on_missing_official_code_and_records_adaptation() -> None:
    assert BOUNDED_ELASTIC_PROTOCOL["paper_revision"] == "arXiv:2608.01475v1"
    assert BOUNDED_ELASTIC_PROTOCOL["official_repository"] is None
    differences = BOUNDED_ELASTIC_PROTOCOL["protocol_differences"]
    assert isinstance(differences, tuple)
    assert len(differences) == 10
    assert any("exactly one least-active unit" in difference for difference in differences)
    assert BOUNDED_ELASTIC_PROTOCOL["development_only"] is True
    assert BOUNDED_ELASTIC_PROTOCOL["scientific_promotion_allowed"] is False
