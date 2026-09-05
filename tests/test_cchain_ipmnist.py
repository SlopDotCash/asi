"""Mechanism, JIT, runner, diagnostics, and receipt tests for C-CHAIN."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks import ipmnist_screening
from alberta_framework.benchmarks.cchain_ipmnist import (
    OFFICIAL_COMMIT,
    PAPER_REVISION,
    CChainState,
    cchain_gradient_component,
    cchain_hyperparameters,
    cchain_ntk_diagnostics,
    make_cchain_learner,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    cchain_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    _make_adamw_learner,
    cross_entropy_loss,
    init_mlp_params,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    CCHAIN_PROTOCOL,
    COMPARABILITY_GAPS,
    DEVELOPMENT_SEEDS,
    validate_cchain_development_result,
    validate_matched_cchain_development_results,
)


def _config() -> IPMNISTConfig:
    return IPMNISTConfig(
        n_tasks=1,
        task_length=4,
        input_dim=4,
        hidden1=3,
        hidden2=2,
        n_classes=2,
    )


def _params() -> dict[str, jax.Array]:
    return init_mlp_params(jr.key(7), _config())


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [-1.0, -0.5, 0.5, 1.0],
            [1.0, 0.5, -0.5, -1.0],
            [-0.5, 1.0, -1.0, 0.5],
            [0.5, -1.0, 1.0, -0.5],
        ],
        dtype=np.float32,
    )
    return x, np.asarray([0, 1, 0, 1], dtype=np.int32)


def _tree_allclose(left: object, right: object, *, atol: float = 0.0) -> None:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=0.0)


def test_source_audit_and_all_causal_arms_are_frozen() -> None:
    assert PAPER_REVISION == "arXiv:2506.00592v1"
    assert OFFICIAL_COMMIT == "2f8bedfefb6a0a276d7709447a5cdb75cecfdaad"
    assert CCHAIN_PROTOCOL["official_commit"] == OFFICIAL_COMMIT
    assert CCHAIN_PROTOCOL["comparability_gaps"] == COMPARABILITY_GAPS
    assert len(COMPARABILITY_GAPS) == 12
    expected = {
        "cchain_mechanism_off": (0.0, 0.0),
        "cchain_full": (1.0, 0.0),
        "cchain_orthogonal_only": (1.0, 1.0),
        "cchain_projective_only": (1.0, 2.0),
    }
    for name, (enabled, component) in expected.items():
        spec = screening_spec(name)
        assert spec.mechanism == "c_chain"
        assert spec.base_learner == "adamw"
        assert spec.hyperparameters["churn_enabled"] == enabled
        assert spec.hyperparameters["gradient_component"] == component


def test_mechanism_off_end_to_end_curves_match_current_adam_control() -> None:
    x, y = _data()
    control = run_screening_config(x, y, screening_spec("adamw_control"), 3, _config())
    off = run_screening_config(
        x, y, screening_spec("cchain_mechanism_off"), 3, _config()
    )
    assert np.array_equal(off.per_task_accuracy, control.per_task_accuracy)
    assert np.array_equal(off.per_task_loss, control.per_task_loss)
    assert np.array_equal(off.per_task_plasticity, control.per_task_plasticity)
    assert off.mechanism_diagnostics is not None
    assert off.mechanism_diagnostics["diagnostic_updates"] == 2.0
    assert off.mechanism_diagnostics["ntk_examples"] == 4.0


def test_mechanism_off_parameters_are_bit_exact_to_matched_adam() -> None:
    cchain_params = _params()
    adam_params = _params()
    spec = screening_spec("cchain_mechanism_off")
    cchain_init, cchain_step = spec.factory(spec.hyperparameters)
    adam_hp = {
        name: spec.hyperparameters[name]
        for name in ("step_size", "beta1", "beta2", "eps", "weight_decay")
    }
    adam_init, adam_step = _make_adamw_learner(adam_hp)
    cchain_state = cchain_init(cchain_params)
    adam_state = adam_init(adam_params)
    x_values, y_values = _data()
    for index, (x_value, y_value) in enumerate(
        zip(x_values, y_values, strict=True)
    ):
        x = jnp.asarray(x_value)
        y = jnp.asarray(y_value)
        (_, _), gradient = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            adam_params, x, y
        )
        cchain_params, cchain_state, _ = cchain_step(
            cchain_params, cchain_state, x, y, jr.key(index)
        )
        adam_update = adam_step(adam_params, adam_state, gradient, jr.key(index))
        adam_params, adam_state = adam_update
        for name in adam_params:
            assert jnp.array_equal(cchain_params[name], adam_params[name])
        _tree_allclose(cchain_state.optimizer_state, adam_state)


def test_gradient_decomposition_reconstructs_full_and_is_orthogonal() -> None:
    task = {name: jnp.ones_like(value) for name, value in _params().items()}
    churn = {
        name: jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape) + 1.0
        for name, value in _params().items()
    }
    projection = cchain_gradient_component(task, churn, component="projective")
    orthogonal = cchain_gradient_component(task, churn, component="orthogonal")
    full = cchain_gradient_component(task, churn, component="full")
    dot = sum(
        (jnp.vdot(task[name], orthogonal[name]).real for name in sorted(task)),
        jnp.asarray(0.0),
    )
    assert abs(float(dot)) < 2e-4
    for name in task:
        assert jnp.allclose(projection[name] + orthogonal[name], full[name], atol=1e-5)


def test_full_step_has_eager_jit_parity_and_updates_bounded_state() -> None:
    params = _params()
    spec = screening_spec("cchain_full")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    x, y = _data()
    eager = step_fn(params, state, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(1))
    compiled = jax.jit(step_fn)(
        params, state, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(1)
    )
    _tree_allclose(eager, compiled)
    assert type(eager[1]) is CChainState
    assert int(eager[1].reference_count) == 1
    assert int(eager[1].reference_cursor) == 1
    assert int(eager[1].update_count) == 1


def test_active_churn_path_has_eager_jit_parity() -> None:
    params = _params()
    spec = screening_spec("cchain_full")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    x, y = _data()
    for index in range(2):
        params, state, _ = step_fn(
            params,
            state,
            jnp.asarray(x[index]),
            jnp.asarray(y[index]),
            jr.key(index),
        )
    eager = step_fn(params, state, jnp.asarray(x[2]), jnp.asarray(y[2]), jr.key(2))
    compiled = jax.jit(step_fn)(
        params, state, jnp.asarray(x[2]), jnp.asarray(y[2]), jr.key(2)
    )
    _tree_allclose(eager, compiled, atol=1e-7)
    assert int(eager[1].diagnostic_count) == 1


def test_hostile_state_is_rejected_eager_and_cannot_commit_under_jit() -> None:
    params = _params()
    spec = screening_spec("cchain_full")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    hostile = state.replace(reference_cursor=jnp.asarray(99, dtype=jnp.int32))
    x, y = _data()
    with pytest.raises(ValueError, match="reference_cursor"):
        step_fn(params, hostile, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0))
    compiled_params, compiled_state, _ = jax.jit(step_fn)(
        params, hostile, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0)
    )
    for name in params:
        assert jnp.array_equal(compiled_params[name], params[name])
    assert int(compiled_state.reference_cursor) == 99


def test_registered_mechanism_changes_the_post_warmup_trajectory() -> None:
    params_off = _params()
    params_full = _params()
    off_spec = screening_spec("cchain_mechanism_off")
    full_spec = screening_spec("cchain_full")
    off_init, off_step = off_spec.factory(off_spec.hyperparameters)
    full_init, full_step = full_spec.factory(full_spec.hyperparameters)
    off_state = off_init(params_off)
    full_state = full_init(params_full)
    x, y = _data()
    for index in range(4):
        params_off, off_state, _ = off_step(
            params_off,
            off_state,
            jnp.asarray(x[index]),
            jnp.asarray(y[index]),
            jr.key(index),
        )
        params_full, full_state, _ = full_step(
            params_full,
            full_state,
            jnp.asarray(x[index]),
            jnp.asarray(y[index]),
            jr.key(index),
        )
    assert any(
        not jnp.array_equal(params_off[name], params_full[name]) for name in params_off
    )
    assert int(full_state.diagnostic_count) == 2


def test_full_logit_ntk_diagnostics_are_bounded_and_jittable() -> None:
    examples = jnp.asarray(_data()[0])
    eager = cchain_ntk_diagnostics(_params(), examples)
    compiled = jax.jit(cchain_ntk_diagnostics)(_params(), examples)
    _tree_allclose(eager, compiled, atol=1e-5)
    assert 0 <= int(eager["ntk_threshold_rank"]) <= 8
    assert float(eager["ntk_off_diagonal_abs_mean"]) >= 0.0
    assert float(eager["ntk_diagonal_mean"]) >= 0.0
    eager_full = cchain_ntk_diagnostics(_params(), examples, energy_threshold=1.0)
    assert 0 <= int(eager_full["ntk_threshold_rank"]) <= 8
    with pytest.raises(ValueError, match="1..4"):
        cchain_ntk_diagnostics(_params(), jnp.zeros((5, 4), dtype=jnp.float32))


def test_end_to_end_result_builds_a_strict_nonpromoting_receipt() -> None:
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("cchain_mechanism_off"), 0, _config()
    )
    receipt = cchain_development_result_payload(result, outcome="inconclusive")
    assert receipt["paper_revision"] == PAPER_REVISION
    assert receipt["comparability_gaps"] == list(COMPARABILITY_GAPS)
    assert receipt["development_only"] is True
    assert receipt["scientific_promotion_allowed"] is False
    assert receipt["development_seed_protocol"] == list(DEVELOPMENT_SEEDS)
    assert receipt["resources"]["model_queries"] == 16  # type: ignore[index]
    assert receipt["resources"]["persistent_bytes"] == 1532  # type: ignore[index]


def test_v2_shard_persists_and_revalidates_the_nested_mechanism_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Provenance validators are covered exhaustively by the existing screening
    # suite; isolate the new nested-receipt path here.
    monkeypatch.setattr(
        ipmnist_screening,
        "_validated_source_provenance",
        lambda value, *, context: dict(value),
    )
    monkeypatch.setattr(
        ipmnist_screening,
        "_validated_dataset_provenance",
        lambda value, *, context: dict(value),
    )
    monkeypatch.setattr(
        ipmnist_screening,
        "_validated_runtime_environment",
        lambda value, *, context: dict(value),
    )
    monkeypatch.setattr(
        ipmnist_screening,
        "_validate_dataset_config_binding",
        lambda dataset, config, *, context: None,
    )
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("cchain_mechanism_off"), 2, _config()
    )
    payload = ipmnist_screening.shard_payload(
        result,
        source_provenance={},
        dataset_provenance={},
        environment={},
    )
    assert payload["mechanism_receipt"]["outcome"] == "inconclusive"
    path = tmp_path / "cchain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = ipmnist_screening.load_shard(path)
    assert loaded["mechanism_receipt"]["arm"] == "cchain_mechanism_off"

    payload["mechanism_receipt"]["metrics"]["mean_online_accuracy"] += 0.01
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics drift"):
        ipmnist_screening.load_shard(drifted)


def test_receipt_validator_fails_closed_on_hostile_and_derived_drift() -> None:
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("cchain_mechanism_off"), 1, _config()
    )
    receipt = cchain_development_result_payload(result, outcome="rejected")

    class HostileInt(int):
        pass

    hostile = copy.deepcopy(receipt)
    hostile["seed"] = HostileInt(1)
    with pytest.raises(ValueError, match="seed"):
        validate_cchain_development_result(hostile)
    drifted = copy.deepcopy(receipt)
    drifted["resources"]["persistent_bytes"] += 4  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="persistent_bytes"):
        validate_cchain_development_result(drifted)
    missing_gap = copy.deepcopy(receipt)
    missing_gap["comparability_gaps"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="comparability_gaps"):
        validate_cchain_development_result(missing_gap)
    huge_metric = copy.deepcopy(receipt)
    huge_metric["metrics"]["mean_loss"] = 1e308  # type: ignore[index]
    with pytest.raises(ValueError, match="metrics.mean_loss"):
        validate_cchain_development_result(huge_metric)


def test_matched_validator_requires_each_causal_arm_and_axes() -> None:
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("cchain_mechanism_off"), 0, _config()
    )
    base = cchain_development_result_payload(result, outcome="inconclusive")
    payloads = []
    for name in (
        "cchain_mechanism_off",
        "cchain_full",
        "cchain_orthogonal_only",
        "cchain_projective_only",
    ):
        payload = copy.deepcopy(base)
        payload["arm"] = name
        payload["hyperparameters"] = dict(screening_spec(name).hyperparameters)
        payloads.append(payload)
    assert len(validate_matched_cchain_development_results(payloads)) == 4
    payloads[1]["seed"] = 1
    with pytest.raises(ValueError, match="axes"):
        validate_matched_cchain_development_results(payloads)


@pytest.mark.parametrize(
    "override",
    (
        {"churn_enabled": 2.0},
        {"adaptive_coefficient": 1.0, "target_relative_loss_scale": 0.0},
        {"gradient_component": 3.0},
    ),
)
def test_hyperparameter_gate_rejects_unregistered_semantics(
    override: dict[str, float],
) -> None:
    hp = cchain_hyperparameters(
        churn_enabled=1.0,
        adaptive_coefficient=1.0,
        target_relative_loss_scale=10_000.0,
        gradient_component=0.0,
    )
    hp.update(override)
    with pytest.raises(ValueError):
        make_cchain_learner(hp)
