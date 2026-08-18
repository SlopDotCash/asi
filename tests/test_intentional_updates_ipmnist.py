"""Intentional Updates' development-only IPMNIST protocol extension."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    INTENTIONAL_UPDATES_CODE_REVISION,
    INTENTIONAL_UPDATES_PAPER_REVISION,
    IPMNISTConfig,
    _make_intentional_updates_learner,
    _make_sgd_ema_norm_learner,
    intentional_updates_development_record,
    run_screening_config,
    screening_spec,
    validate_intentional_updates_development_record,
)
from alberta_framework.benchmarks.upgd_ipmnist import init_mlp_params

SMALL = IPMNISTConfig(
    n_tasks=2, task_length=4, input_dim=6, hidden1=5, hidden2=4, n_classes=3
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(72, dtype=np.float32).reshape(12, 6) / 71.0,
        np.arange(12, dtype=np.int32) % 3,
    )


def test_references_and_required_arm_set_are_pinned() -> None:
    assert INTENTIONAL_UPDATES_PAPER_REVISION == "arXiv:2604.19033v1"
    assert INTENTIONAL_UPDATES_CODE_REVISION == (
        "sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685"
    )
    expected = {
        "intentional_updates_ipmnist",
        "intentional_updates_no_diag",
        "intentional_updates_no_clip",
        "intentional_updates_head_only",
        "intentional_updates_off",
    }
    assert expected <= {name for name in expected if screening_spec(name).name == name}


def test_mechanism_off_is_bit_exact_fixed_step_control_under_jit() -> None:
    off = screening_spec("intentional_updates_off")
    init_off, step_off = off.factory(off.hyperparameters)
    control_hp = {
        "step_size": off.hyperparameters["fixed_step_size"],
        "weight_decay": off.hyperparameters["weight_decay"],
        "norm_decay": off.hyperparameters["norm_decay"],
        "norm_epsilon": off.hyperparameters["norm_epsilon"],
    }
    init_control, step_control = _make_sgd_ema_norm_learner(control_hp)
    params = init_mlp_params(jr.key(7), SMALL)
    x = jr.normal(jr.key(8), (SMALL.input_dim,))
    y = jnp.asarray(2, dtype=jnp.int32)
    result_off = jax.jit(step_off)(params, init_off(params), x, y, jr.key(9))
    result_control = jax.jit(step_control)(params, init_control(params), x, y, jr.key(99))
    for name in params:
        np.testing.assert_array_equal(result_off[0][name], result_control[0][name])
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, result_off[1], result_control[1])
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, result_off[2], result_control[2])
    )


@pytest.mark.parametrize(
    "name", [
        "intentional_updates_ipmnist",
        "intentional_updates_no_diag",
        "intentional_updates_no_clip",
        "intentional_updates_head_only",
    ]
)
def test_candidate_and_ablations_have_finite_jittable_autodiff_steps(name: str) -> None:
    spec = screening_spec(name)
    init_fn, step_fn = _make_intentional_updates_learner(spec.hyperparameters)
    params = init_mlp_params(jr.key(1), SMALL)
    x = jnp.zeros((SMALL.input_dim,), dtype=jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)
    new_params, new_state, metrics = jax.jit(step_fn)(
        params, init_fn(params), x, y, jr.key(2)
    )
    leaves = jax.tree_util.tree_leaves((new_params, new_state, metrics))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in leaves)


def test_head_only_feature_control_freezes_hidden_parameters() -> None:
    spec = screening_spec("intentional_updates_head_only")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(jr.key(3), SMALL)
    new_params, _, _ = step_fn(
        params,
        init_fn(params),
        jr.normal(jr.key(4), (SMALL.input_dim,)),
        jnp.asarray(1, dtype=jnp.int32),
        jr.key(5),
    )
    for name in ("w1", "b1", "w2", "b2"):
        np.testing.assert_array_equal(new_params[name], params[name])
    assert any(
        not np.array_equal(np.asarray(new_params[name]), np.asarray(params[name]))
        for name in ("w3", "b3")
    )


def test_zero_gradient_one_class_path_is_finite_and_stationary() -> None:
    one_class = IPMNISTConfig(
        n_tasks=1, task_length=1, input_dim=3, hidden1=2, hidden2=2, n_classes=1
    )
    spec = screening_spec("intentional_updates_ipmnist")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(jr.key(30), one_class)
    new_params, new_state, metrics = jax.jit(step_fn)(
        params,
        init_fn(params),
        jnp.zeros((one_class.input_dim,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jr.key(31),
    )
    for name in params:
        np.testing.assert_array_equal(new_params[name], params[name])
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves((new_state, metrics))
    )


def test_factory_and_state_reject_hostile_structures_without_dispatch() -> None:
    class HostileMapping:
        def __iter__(self) -> object:
            raise AssertionError("hostile mapping hook must not run")

    with pytest.raises(ValueError, match="one exact object"):
        _make_intentional_updates_learner(HostileMapping())  # type: ignore[arg-type]

    spec = screening_spec("intentional_updates_ipmnist")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(jr.key(40), SMALL)
    state = init_fn(params)
    hostile_state = state.replace(squared_gradient={})
    with pytest.raises(ValueError, match="exact protocol MLP tree"):
        step_fn(
            params,
            hostile_state,
            jnp.zeros((SMALL.input_dim,), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jr.key(41),
        )


@pytest.mark.parametrize("invalid", ["counter", "input"])
def test_jit_runtime_invalidity_is_finite_and_atomic(invalid: str) -> None:
    spec = screening_spec("intentional_updates_ipmnist")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(jr.key(42), SMALL)
    state = init_fn(params)
    x = jnp.zeros((SMALL.input_dim,), dtype=jnp.float32)
    if invalid == "counter":
        state = state.replace(
            step=jnp.asarray((1 << 31) - 1, dtype=jnp.int32),
            clip_step=jnp.asarray((1 << 31) - 1, dtype=jnp.int32),
        )
    else:
        x = jnp.full_like(x, jnp.inf)
    new_params, new_state, metrics = jax.jit(step_fn)(
        params, state, x, jnp.asarray(0, dtype=jnp.int32), jr.key(43)
    )
    for name in params:
        np.testing.assert_array_equal(new_params[name], params[name])
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves((new_state, metrics))
    )
    if invalid == "counter":
        assert int(new_state.step) == 0
        assert int(new_state.clip_step) == 0


def test_strict_development_record_binds_resources_gates_and_metrics() -> None:
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("intentional_updates_ipmnist"), seed=11, config=SMALL
    )
    record = intentional_updates_development_record(result)
    validated = validate_intentional_updates_development_record(record)
    assert validated == record
    assert record["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "publication_equivalent": False,
    }
    assert record["gates"]["backpropagation"] is True
    assert record["gates"]["feature_updates"] is True
    assert record["resources"]["observations"] == 8
    assert record["resources"]["updates"] == 8
    assert record["resources"]["backward_passes"] == 8
    assert record["resources"]["model_queries"] == 16
    assert record["resources"]["persistent_numeric_bytes"] > 0

    hostile = copy.deepcopy(record)
    hostile["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="permanently nonpromoting"):
        validate_intentional_updates_development_record(hostile)

    hostile = copy.deepcopy(record)
    hostile["resources"]["updates"] -= 1
    with pytest.raises(ValueError, match="resource counters"):
        validate_intentional_updates_development_record(hostile)

    hostile = copy.deepcopy(record)
    hostile["gates"]["backpropagation"] = False
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_intentional_updates_development_record(hostile)

    hostile = copy.deepcopy(record)
    original_accuracy = hostile["metrics"]["per_task_accuracy"][0]
    hostile["metrics"]["per_task_accuracy"][0] = (
        0.25 if original_accuracy != 0.25 else 0.5
    )
    with pytest.raises(ValueError, match="derive from per_task_correct"):
        validate_intentional_updates_development_record(hostile)

    class HostileArray:
        def __array__(self) -> object:
            raise AssertionError("hostile array hook must not run")

    hostile = copy.deepcopy(record)
    hostile["metrics"]["per_task_loss"] = HostileArray()
    with pytest.raises(ValueError, match="one exact bounded list"):
        validate_intentional_updates_development_record(hostile)

    hostile = copy.deepcopy(record)
    hostile["gates"]["head_updates"] = 1
    with pytest.raises(ValueError, match="exact boolean"):
        validate_intentional_updates_development_record(hostile)

    hostile = copy.deepcopy(record)
    hostile["references"]["official_code"] = "moving-main"
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_intentional_updates_development_record(hostile)


def test_record_revalidates_forged_host_dataclasses() -> None:
    x, y = _data()
    config = IPMNISTConfig(
        n_tasks=2, task_length=4, input_dim=6, hidden1=5, hidden2=4, n_classes=3
    )
    result = run_screening_config(
        x, y, screening_spec("intentional_updates_ipmnist"), seed=23, config=config
    )
    object.__setattr__(result.config, "n_tasks", True)
    with pytest.raises(ValueError, match="n_tasks"):
        intentional_updates_development_record(result)

def test_matched_runs_share_seed_schedule_update_and_observation_budgets() -> None:
    x, y = _data()
    records = []
    for name in ("intentional_updates_ipmnist", "intentional_updates_off"):
        result = run_screening_config(x, y, screening_spec(name), seed=19, config=SMALL)
        records.append(intentional_updates_development_record(result))
    assert records[0]["seed"] == records[1]["seed"]
    assert records[0]["config"] == records[1]["config"]
    for counter in ("observations", "updates", "backward_passes", "model_queries"):
        assert records[0]["resources"][counter] == records[1]["resources"][counter]
