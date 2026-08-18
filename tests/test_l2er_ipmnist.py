from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    L2ERState,
    ScreeningRunResult,
    _make_l2er_learner,
    l2er_development_result_payload,
    l2er_effective_rank,
    l2er_effective_rank_loss,
    l2er_effective_rank_transaction,
    l2er_update,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.l2er_ipmnist_nonpromoting import (
    L2ER_PROTOCOL,
    validate_l2er_development_result,
    validate_matched_l2er_development_results,
)


def _params() -> dict[str, jax.Array]:
    return {
        "w1": jnp.asarray([[0.8, 0.2], [0.1, 0.6]], dtype=jnp.float32),
        "b1": jnp.asarray([0.2, 0.1], dtype=jnp.float32),
        "w2": jnp.asarray([[0.7, 0.1], [0.2, 0.9]], dtype=jnp.float32),
        "b2": jnp.asarray([0.1, 0.3], dtype=jnp.float32),
        "w3": jnp.asarray([[0.5, -0.2], [0.3, 0.4]], dtype=jnp.float32),
        "b3": jnp.asarray([0.0, 0.1], dtype=jnp.float32),
    }


def _hp(*, wd: float, er_lr: float, enabled: float) -> dict[str, float]:
    return {
        "step_size": 1e-3,
        "weight_decay": wd,
        "er_step_size": er_lr,
        "er_batch_size": 100.0,
        "er_steps_per_batch": 1.0,
        "er_epsilon": 1e-8,
        "er_enabled": enabled,
    }


def test_effective_rank_matches_official_entropy_estimator() -> None:
    features = jnp.diag(jnp.asarray([3.0, 1.0], dtype=jnp.float32))
    probabilities = np.asarray([0.75, 0.25])
    expected = np.exp(-np.sum(probabilities * np.log(probabilities + 1e-8)))
    assert float(l2er_effective_rank(features)) == pytest.approx(expected, rel=1e-6)
    assert float(l2er_effective_rank(jnp.zeros((4, 3)))) == pytest.approx(1.0)
    huge = jnp.eye(4, dtype=jnp.float32) * jnp.asarray(1e38, dtype=jnp.float32)
    assert float(l2er_effective_rank(huge)) == pytest.approx(4.0, rel=1e-5)


def test_effective_rank_preserves_invalidity_in_eager_and_traced_calls() -> None:
    invalid = jnp.asarray([[jnp.inf]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="effective rank must be finite"):
        l2er_effective_rank(invalid)
    assert bool(jnp.isnan(jax.jit(l2er_effective_rank)(invalid)))
    safe, valid = jax.jit(l2er_effective_rank_transaction)(invalid)
    assert float(safe) == 0.0
    assert not bool(valid)


def test_l2_and_mechanism_off_are_exact_reductions() -> None:
    params = _params()
    grads = jax.tree.map(jnp.ones_like, params)
    state = L2ERState(  # type: ignore[call-arg]
        example_buffer=jnp.zeros((100, 2), dtype=jnp.float32),
        buffer_count=jnp.asarray(0, dtype=jnp.int32),
        transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    example = jnp.asarray([0.4, 0.7], dtype=jnp.float32)
    off, off_state = l2er_update(params, state, grads, example, _hp(wd=0.0, er_lr=0.0, enabled=0.0))
    l2, _ = l2er_update(params, state, grads, example, _hp(wd=1e-4, er_lr=0.0, enabled=0.0))
    for name in params:
        np.testing.assert_array_equal(off[name], params[name] - 1e-3 * grads[name])
        np.testing.assert_allclose(
            l2[name], params[name] - 1e-3 * (grads[name] + 1e-4 * params[name])
        )
    assert int(off_state.buffer_count) == 1


def test_er_update_matches_official_separate_post_batch_step() -> None:
    params = _params()
    grads = jax.tree.map(lambda value: jnp.full_like(value, 0.05), params)
    buffer = jnp.linspace(0.1, 1.0, 200, dtype=jnp.float32).reshape(100, 2)
    state = L2ERState(  # type: ignore[call-arg]
        example_buffer=buffer,
        buffer_count=jnp.asarray(99, dtype=jnp.int32),
        transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    example = jnp.asarray([0.3, 0.9], dtype=jnp.float32)
    hp = _hp(wd=1e-4, er_lr=1e-3, enabled=1.0)
    actual, new_state = l2er_update(params, state, grads, example, hp)

    supervised = {
        name: value - 1e-3 * (grads[name] + 1e-4 * value)
        for name, value in params.items()
    }
    completed_batch = buffer.at[99].set(example)
    er_grads = jax.grad(l2er_effective_rank_loss)(supervised, completed_batch, 1e-8)
    expected = {
        name: value - 1e-3 * er_grads[name] for name, value in supervised.items()
    }
    for name in params:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)
        assert bool(jnp.all(jnp.isfinite(actual[name])))
    assert int(new_state.buffer_count) == 0
    np.testing.assert_array_equal(new_state.example_buffer, jnp.zeros_like(buffer))


def test_l2er_update_is_jittable() -> None:
    params = _params()
    grads = jax.tree.map(jnp.ones_like, params)
    state = L2ERState(  # type: ignore[call-arg]
        example_buffer=jnp.ones((100, 2), dtype=jnp.float32),
        buffer_count=jnp.asarray(99, dtype=jnp.int32),
        transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    update = jax.jit(
        lambda p, s, g, x: l2er_update(
            p, s, g, x, _hp(wd=0.0, er_lr=1e-3, enabled=1.0)
        )
    )
    new_params, new_state = update(params, state, grads, jnp.asarray([0.2, 0.8]))
    assert int(new_state.buffer_count) == 0
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in new_params.values())


def test_update_rejects_hostile_structure_and_is_atomic_on_runtime_invalidity() -> None:
    class HostileMapping:
        def __iter__(self) -> object:
            raise AssertionError("hostile iteration must not run")

    params = _params()
    grads = jax.tree.map(jnp.ones_like, params)
    state = L2ERState(  # type: ignore[call-arg]
        example_buffer=jnp.zeros((100, 2), dtype=jnp.float32),
        buffer_count=jnp.asarray(0, dtype=jnp.int32),
        transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    with pytest.raises(ValueError, match="hyperparameters"):
        l2er_update(  # type: ignore[arg-type]
            params, state, grads, jnp.ones(2), HostileMapping()
        )
    bad_grads = dict(grads)
    bad_grads["w2"] = jnp.ones((1, 3), dtype=jnp.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        l2er_update(
            params,
            state,
            bad_grads,
            jnp.ones(2),
            _hp(wd=0.0, er_lr=0.0, enabled=0.0),
        )

    bad_state = state.replace(buffer_count=jnp.asarray(100, dtype=jnp.int32))
    unchanged, repaired = jax.jit(
        lambda p, s, g: l2er_update(
            p, s, g, jnp.ones(2), _hp(wd=0.0, er_lr=0.0, enabled=0.0)
        )
    )(params, bad_state, grads)
    for name in params:
        np.testing.assert_array_equal(unchanged[name], params[name])
    assert int(repaired.buffer_count) == 0
    assert not bool(repaired.transaction_valid)

    nonfinite = dict(grads)
    nonfinite["w1"] = jnp.full_like(grads["w1"], jnp.inf)
    unchanged, invalid_state = l2er_update(
        params,
        state,
        nonfinite,
        jnp.ones(2),
        _hp(wd=0.0, er_lr=0.0, enabled=0.0),
    )
    for name in params:
        np.testing.assert_array_equal(unchanged[name], params[name])
    assert not bool(invalid_state.transaction_valid)
    _, still_invalid = l2er_update(
        unchanged,
        invalid_state,
        grads,
        jnp.ones(2),
        _hp(wd=0.0, er_lr=0.0, enabled=0.0),
    )
    assert not bool(still_invalid.transaction_valid)


def test_buffer_and_svd_resources_are_preflighted() -> None:
    init, _ = _make_l2er_learner(_hp(wd=0.0, er_lr=0.0, enabled=0.0))
    with pytest.raises(ValueError, match="exact protocol MLP tree"):
        init({})
    params = _params()
    params["w1"] = jnp.ones((10_001, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="buffer exceeds"):
        init(params)
    with pytest.raises(ValueError, match="bounded float array"):
        l2er_effective_rank(jnp.ones((1001, 1000), dtype=jnp.float32))


def test_registry_contains_complete_matched_ablation() -> None:
    expected = {
        "l2er_mechanism_off": (0.0, 0.0, 0.0),
        "l2er_l2_only": (1e-4, 0.0, 0.0),
        "l2er_er_only": (0.0, 1e-3, 1.0),
        "l2er_combined": (1e-4, 1e-3, 1.0),
    }
    for name, values in expected.items():
        spec = screening_spec(name)
        assert spec.mechanism == "l2_effective_rank"
        assert (
            spec.hyperparameters["weight_decay"],
            spec.hyperparameters["er_step_size"],
            spec.hyperparameters["er_enabled"],
        ) == values


def test_runner_rejects_er_batches_crossing_task_boundaries() -> None:
    config = IPMNISTConfig(
        n_tasks=1, task_length=99, input_dim=2, hidden1=2, hidden2=2, n_classes=2
    )
    with pytest.raises(ValueError, match="task_length divisible"):
        run_screening_config(
            np.zeros((99, 2), dtype=np.float32),
            np.zeros(99, dtype=np.int32),
            screening_spec("l2er_combined"),
            seed=0,
            config=config,
        )


def test_combined_arm_runs_end_to_end_on_a_synthetic_task() -> None:
    config = IPMNISTConfig(
        n_tasks=1, task_length=100, input_dim=2, hidden1=3, hidden2=2, n_classes=2
    )
    data_x = np.random.default_rng(11).normal(size=(100, 2)).astype(np.float32)
    data_y = np.arange(100, dtype=np.int32) % 2
    result = run_screening_config(
        data_x,
        data_y,
        screening_spec("l2er_combined"),
        seed=3,
        config=config,
    )
    assert result.per_task_accuracy.shape == (1,)
    assert np.isfinite(result.per_task_loss).all()
    receipt = l2er_development_result_payload(result, outcome="inconclusive")
    assert receipt["arm"] == "l2er_combined"


def _result(name: str) -> ScreeningRunResult:
    config = IPMNISTConfig(
        n_tasks=1, task_length=100, input_dim=2, hidden1=2, hidden2=2, n_classes=2
    )
    return ScreeningRunResult(
        config_name=name,
        base_learner="upgd_w",
        hyperparameters=dict(screening_spec(name).hyperparameters),
        seed=7,
        config=config,
        per_task_accuracy=np.asarray([0.6], dtype=np.float64),
        per_task_loss=np.asarray([0.7], dtype=np.float64),
        per_task_plasticity=np.asarray([0.1], dtype=np.float64),
        wall_clock_seconds=1.5,
    )


def test_result_receipt_accounts_resources_and_is_permanently_nonpromoting() -> None:
    payload = l2er_development_result_payload(_result("l2er_combined"), outcome="rejected")
    assert payload["outcome_retained"] is True
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["updates"] == 100
    assert payload["effective_rank_updates"] == 1
    resources = payload["resources"]
    assert isinstance(resources, dict)
    assert resources["data_steps"] == 100
    assert resources["environment_steps"] == 0
    assert resources["model_queries"] == 201
    assert resources["persistent_bytes"] == 877

    hostile = deepcopy(payload)
    hostile_resources = hostile["resources"]
    assert isinstance(hostile_resources, dict)
    hostile_resources["model_queries"] = 200
    with pytest.raises(ValueError, match="model_queries"):
        validate_l2er_development_result(hostile)

    hostile_er_updates = deepcopy(payload)
    hostile_er_updates["effective_rank_updates"] = 0
    with pytest.raises(ValueError, match="effective_rank_updates"):
        validate_l2er_development_result(hostile_er_updates)

    hostile_bytes = deepcopy(payload)
    hostile_byte_resources = hostile_bytes["resources"]
    assert isinstance(hostile_byte_resources, dict)
    hostile_byte_resources["persistent_bytes"] = 872
    with pytest.raises(ValueError, match="persistent_bytes"):
        validate_l2er_development_result(hostile_bytes)

    hostile_schema = deepcopy(payload)
    hostile_schema["unregistered_claim"] = True
    with pytest.raises(ValueError, match="result keys"):
        validate_l2er_development_result(hostile_schema)

    hostile_counter = deepcopy(payload)
    hostile_counter["n_tasks"] = 2**31
    with pytest.raises(ValueError, match="n_tasks"):
        validate_l2er_development_result(hostile_counter)

    class HostileObject:
        def __iter__(self) -> object:
            raise AssertionError("hostile iteration must not run")

    with pytest.raises(ValueError, match="exact object"):
        validate_l2er_development_result(HostileObject())


def test_matched_validator_requires_all_arms_and_axes() -> None:
    names = ("l2er_mechanism_off", "l2er_l2_only", "l2er_er_only", "l2er_combined")
    payloads = [
        l2er_development_result_payload(_result(name), outcome="inconclusive")
        for name in names
    ]
    assert len(validate_matched_l2er_development_results(payloads)) == 4
    mismatched = deepcopy(payloads)
    mismatched[1]["updates"] = 99
    with pytest.raises(ValueError, match="observations and updates"):
        validate_matched_l2er_development_results(mismatched)

    class HostileMeta(type):
        calls = 0

        def __hash__(cls) -> int:
            cls.calls += 1
            raise AssertionError("runtime type must not be hashed")

    class HostileContainer(metaclass=HostileMeta):
        def __len__(self) -> int:
            raise AssertionError("hostile length must not run")

    with pytest.raises(ValueError, match="exactly four"):
        validate_matched_l2er_development_results(HostileContainer())
    assert HostileMeta.calls == 0


def test_protocol_pins_sources_and_records_material_differences() -> None:
    assert L2ER_PROTOCOL["paper_revision"] == "arXiv:2509.22335v3"
    assert L2ER_PROTOCOL["official_commit"] == "52ae3eb0702a9e6923f252c1f7cb29340eb5b3d5"
    differences = L2ER_PROTOCOL["protocol_differences"]
    assert isinstance(differences, tuple)
    assert len(differences) == 8
    assert L2ER_PROTOCOL["update_accounting"] == (
        "updates equals supervised observations plus charged effective_rank_updates; "
        "effective_rank_updates is retained as the arm-specific auxiliary ER subcount"
    )
    assert L2ER_PROTOCOL["development_only"] is True
    assert L2ER_PROTOCOL["scientific_promotion_allowed"] is False
