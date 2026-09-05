"""Replay/frozen-feature provenance, mechanism, JIT, and receipt tests."""

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
from alberta_framework.benchmarks.ipmnist_screening import (
    replay_frozen_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.replay_frozen_ipmnist import (
    PROL_COMMIT,
    PROL_PAPER_REVISION,
    RANDUMB_COMMIT,
    RANDUMB_PAPER_REVISION,
    RANPAC_COMMIT,
    RANPAC_PAPER_REVISION,
    REPLAY_OFFICIAL_CODE,
    REPLAY_PAPER_REVISION,
    ReplayContextState,
    make_frozen_feature_learner,
    make_replay_context_learner,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    PROTOCOL_GAPS,
    validate_matched_replay_frozen_results,
    validate_replay_frozen_result,
)

_ARMS = (
    "replay_context_mechanism_off",
    "replay_gradient_only",
    "replay_context_only",
    "replay_context_full",
    "randumb_random_features",
    "ranpac_random_projection",
    "prol_prompt_mechanism_off",
    "prol_prompt_proxy",
)


def _config() -> IPMNISTConfig:
    return IPMNISTConfig(
        n_tasks=1, task_length=4, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [-1.0, -0.5, 0.5, 1.0],
                [1.0, 0.5, -0.5, -1.0],
                [-0.5, 1.0, -1.0, 0.5],
                [0.5, -1.0, 1.0, -0.5],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
    )


def _tree_allclose(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert jnp.allclose(left_leaf, right_leaf, atol=1e-6, rtol=0.0)


def test_audited_revisions_commits_and_all_registered_arms_are_frozen() -> None:
    assert REPLAY_PAPER_REVISION == "arXiv:2503.20018v1"
    assert REPLAY_OFFICIAL_CODE == "not-disclosed-as-of-2026-08-17"
    assert (RANDUMB_PAPER_REVISION, RANDUMB_COMMIT) == (
        "arXiv:2402.08823v3",
        "14a51ee0c045bff642f6ffbfe481efa4d49a3033",
    )
    assert (RANPAC_PAPER_REVISION, RANPAC_COMMIT) == (
        "arXiv:2307.02251v3",
        "cf4b301d18b0c27db030f4371b72b768005ae58a",
    )
    assert (PROL_PAPER_REVISION, PROL_COMMIT) == (
        "arXiv:2507.12305v1",
        "bfff8418a4f603a24ae578f1e108bfac89af1e18",
    )
    assert len(PROTOCOL_GAPS) == 13
    assert all(screening_spec(name).mechanism for name in _ARMS)


def test_charged_mechanism_off_is_end_to_end_exact_adam_control() -> None:
    x, y = _data()
    control = run_screening_config(x, y, screening_spec("adamw_control"), 3, _config())
    off = run_screening_config(
        x, y, screening_spec("replay_context_mechanism_off"), 3, _config()
    )
    assert np.array_equal(off.per_task_accuracy, control.per_task_accuracy)
    assert np.array_equal(off.per_task_loss, control.per_task_loss)
    assert np.array_equal(off.per_task_plasticity, control.per_task_plasticity)


def test_active_replay_step_has_eager_jit_parity() -> None:
    params = init_mlp_params(jr.key(4), _config())
    spec = screening_spec("replay_context_full")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    x, y = _data()
    params, state, _ = step_fn(params, state, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0))
    eager = step_fn(params, state, jnp.asarray(x[1]), jnp.asarray(y[1]), jr.key(1))
    compiled = jax.jit(step_fn)(
        params, state, jnp.asarray(x[1]), jnp.asarray(y[1]), jr.key(1)
    )
    _tree_allclose(eager, compiled)


@pytest.mark.parametrize(
    "arm",
    ("randumb_random_features", "ranpac_random_projection", "prol_prompt_proxy"),
)
def test_frozen_feature_steps_are_jittable_and_leave_ballast_params_frozen(
    arm: str,
) -> None:
    params = init_mlp_params(jr.key(5), _config())
    spec = screening_spec(arm)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    x, y = _data()
    eager = step_fn(params, state, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0))
    compiled = jax.jit(step_fn)(
        params, state, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0)
    )
    _tree_allclose(eager, compiled)
    for name in params:
        assert jnp.array_equal(eager[0][name], params[name])


def test_hostile_replay_state_cannot_commit_under_jit() -> None:
    params = init_mlp_params(jr.key(6), _config())
    spec = screening_spec("replay_context_full")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    assert type(state) is ReplayContextState
    hostile = state.replace(cursor=jnp.asarray(99, dtype=jnp.int32))
    x, y = _data()
    new_params, new_state, _ = jax.jit(step_fn)(
        params, hostile, jnp.asarray(x[0]), jnp.asarray(y[0]), jr.key(0)
    )
    for name in params:
        assert jnp.array_equal(new_params[name], params[name])
    assert int(new_state.cursor) == 99


def test_factories_reject_hostile_types_and_unregistered_constants() -> None:
    replay_hp = dict(screening_spec("replay_context_full").hyperparameters)
    replay_hp["step_size"] = 2e-4
    with pytest.raises(ValueError, match="Adam constants"):
        make_replay_context_learner(replay_hp)
    frozen_hp = dict(screening_spec("ranpac_random_projection").hyperparameters)
    frozen_hp["feature_dim"] = 65.0
    with pytest.raises(ValueError, match="drift"):
        make_frozen_feature_learner(frozen_hp)


def test_receipt_charges_replay_extractor_and_zero_pretraining_exactly() -> None:
    x, y = _data()
    replay = replay_frozen_development_result_payload(
        run_screening_config(
            x, y, screening_spec("replay_context_mechanism_off"), 7, _config()
        ),
        outcome="rejected",
    )
    replay_resources = replay["resources"]
    assert replay_resources["replay_bytes"] > 0  # type: ignore[index,operator]
    assert replay_resources["replay_model_queries"] == 128  # type: ignore[index]
    assert replay_resources["model_queries"] == 140  # type: ignore[index]
    frozen = replay_frozen_development_result_payload(
        run_screening_config(
            x, y, screening_spec("ranpac_random_projection"), 7, _config()
        ),
        outcome="inconclusive",
    )
    resources = frozen["resources"]
    assert resources["extractor_parameter_bytes"] > 0  # type: ignore[index,operator]
    assert resources["pretraining_bytes"] == 0  # type: ignore[index]
    assert resources["pretraining_examples"] == 0  # type: ignore[index]
    assert frozen["negative_outcome_retained"] is True
    assert frozen["scientific_promotion_allowed"] is False


def test_receipt_rejects_hostile_types_resource_drift_and_missing_gap() -> None:
    x, y = _data()
    receipt = replay_frozen_development_result_payload(
        run_screening_config(
            x, y, screening_spec("replay_context_full"), 8, _config()
        ),
        outcome="rejected",
    )

    class HostileInt(int):
        pass

    hostile = copy.deepcopy(receipt)
    hostile["seed"] = HostileInt(8)
    with pytest.raises(ValueError, match="seed"):
        validate_replay_frozen_result(hostile)
    drifted = copy.deepcopy(receipt)
    drifted["resources"]["persistent_bytes"] += 4  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="resource receipt"):
        validate_replay_frozen_result(drifted)
    missing_gap = copy.deepcopy(receipt)
    missing_gap["protocol_gaps"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="protocol gaps"):
        validate_replay_frozen_result(missing_gap)


def test_matched_validator_freezes_all_arms_and_seed_axes() -> None:
    x, y = _data()
    payloads = [
        replay_frozen_development_result_payload(
            run_screening_config(x, y, screening_spec(name), 9, _config()),
            outcome="inconclusive",
        )
        for name in _ARMS
    ]
    assert len(validate_matched_replay_frozen_results(payloads)) == 8
    payloads[1]["seed"] = 10
    with pytest.raises(ValueError, match="axes"):
        validate_matched_replay_frozen_results(payloads)


def test_v2_shard_roundtrip_revalidates_nested_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ipmnist_screening, "_validated_source_provenance", lambda value, *, context: dict(value)
    )
    monkeypatch.setattr(
        ipmnist_screening, "_validated_dataset_provenance", lambda value, *, context: dict(value)
    )
    monkeypatch.setattr(
        ipmnist_screening, "_validated_runtime_environment", lambda value, *, context: dict(value)
    )
    monkeypatch.setattr(
        ipmnist_screening,
        "_validate_dataset_config_binding",
        lambda dataset, config, *, context: None,
    )
    x, y = _data()
    result = run_screening_config(
        x, y, screening_spec("prol_prompt_proxy"), 11, _config()
    )
    payload = ipmnist_screening.shard_payload(
        result, source_provenance={}, dataset_provenance={}, environment={}
    )
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert ipmnist_screening.load_shard(path)["mechanism_receipt"]["arm"] == (
        "prol_prompt_proxy"
    )
    payload["mechanism_receipt"]["metrics"]["mean_loss"] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics drift"):
        ipmnist_screening.load_shard(path)


def test_skip_zero_scale_repairs_zero_times_inf() -> None:
    from alberta_framework.benchmarks import replay_frozen_ipmnist as mod

    value = jnp.array([jnp.inf, 2.0], dtype=jnp.float32)
    assert bool(jnp.isnan(jnp.float32(0.0) * jnp.float32(jnp.inf)))
    repaired = mod._skip_zero_scale(0.0, value)
    np.testing.assert_array_equal(repaired, jnp.zeros_like(value))
    finite = jnp.array([1.0, -2.0], dtype=jnp.float32)
    np.testing.assert_allclose(mod._skip_zero_scale(0.5, finite), 0.5 * finite)


def test_replay_weight_zero_does_not_poison_task_gradient() -> None:
    from alberta_framework.benchmarks import replay_frozen_ipmnist as mod

    task = jnp.array([1.0, -1.0], dtype=jnp.float32)
    replay = jnp.array([jnp.inf, 3.0], dtype=jnp.float32)
    combined = task + mod._skip_zero_scale(0.0, replay)
    np.testing.assert_array_equal(combined, task)


def test_mechanism_zero_feature_step_is_exact_zero_under_infinite_grad() -> None:
    from alberta_framework.benchmarks import replay_frozen_ipmnist as mod

    feature_gradient = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)
    update_scale = jnp.asarray(1.0, dtype=jnp.float32)
    step = mod._skip_zero_scale(0.0 * 1e-2, update_scale * feature_gradient)
    np.testing.assert_array_equal(step, jnp.zeros_like(feature_gradient))


def test_disabled_replay_ignores_overflow_from_finite_stored_examples() -> None:
    from alberta_framework.benchmarks.replay_frozen_ipmnist import replay_hyperparameters

    params = init_mlp_params(jr.key(4), _config())
    params = {**params, "w1": jnp.full_like(params["w1"], 2.0)}
    init, step = make_replay_context_learner(replay_hyperparameters(replay_update=0.0, context=0.0))
    state = init(params)
    state = state.replace(
        examples=jnp.full_like(state.examples, 3e38),
        count=jnp.asarray(1, dtype=jnp.int32),
    )
    _, result, _ = step(params, state, jnp.zeros(4), jnp.asarray(0), jr.key(5))
    assert int(result.update_count) == 1
