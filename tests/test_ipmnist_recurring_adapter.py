"""Focused checks for the development-only recurring IPMNIST adapter."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import alberta_framework.benchmarks.ipmnist_screening as screening
from alberta_framework.benchmarks.ipmnist_screening import (
    SCREENING_REGISTRY,
    build_recurring_ipmnist_online_indices,
    ipmnist_permutation_sha256,
    ipmnist_sentinel_set_sha256,
    run_recurring_ipmnist_retention_development,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params, mlp_logits

pytestmark = pytest.mark.unit

CONFIG = IPMNISTConfig(
    n_tasks=3,
    task_length=2,
    input_dim=4,
    hidden1=3,
    hidden2=2,
    n_classes=2,
)
DATA_X = np.asarray(
    [
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 0.5, 0.0, -0.5],
        [0.0, 1.0, -1.0, 0.5],
        [0.5, -1.0, 1.0, 0.0],
        [-0.5, 0.0, 0.5, 1.0],
        [0.25, -0.25, 0.75, -0.75],
        [-0.75, 0.75, -0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [-0.4, -0.3, -0.2, -0.1],
    ],
    dtype=np.float32,
)
DATA_Y = np.asarray([0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
PERMUTATION_A = np.asarray([0, 1, 2, 3], dtype=np.int32)
PERMUTATION_B = np.asarray([3, 1, 0, 2], dtype=np.int32)
SENTINELS = (7, 8)


def test_confirmed_rls_winner_runs_recurring_retention_with_frozen_probes() -> None:
    report = run_recurring_ipmnist_retention_development(
        DATA_X,
        DATA_Y,
        screening_spec("rls_head_resid_l1_preset005"),
        seed=19,
        config=CONFIG,
        phase_lengths=(2, 3, 2),
        permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A.copy()),
        sentinel_indices=SENTINELS,
        relearning_window=1,
    )

    assert len(report.sentinel_scores) == 5
    checkpoint_hashes = tuple(score.learner_state_sha256 for score in report.sentinel_scores)
    assert checkpoint_hashes[1] == checkpoint_hashes[2]
    assert checkpoint_hashes[3] == checkpoint_hashes[4]
    assert report.to_config()["scientific_promotion_allowed"] is False


def _hand_rls_checkpoint() -> tuple[dict[str, Array], Any]:
    spec = screening_spec("rls_head_resid_l1_preset005")
    init_fn, _ = spec.factory(spec.hyperparameters)
    params = {
        "w1": jnp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], jnp.float32),
        "b1": jnp.zeros(3, jnp.float32),
        "w2": jnp.asarray([[1, 0], [0, 1], [1, 1]], jnp.float32),
        "b2": jnp.zeros(2, jnp.float32),
        "w3": jnp.zeros((2, 2), jnp.float32),
        "b3": jnp.asarray([10, -10], jnp.float32),
    }
    state = init_fn(params)
    state = state.replace(
        norm=state.norm.replace(
            mean=jnp.asarray([1, -1, 2, 0], jnp.float32),
            var=jnp.asarray([4, 1, 1, 1], jnp.float32),
            count=jnp.full(4, 3, jnp.float32),
        ),
        wout=jnp.asarray([[2, -1], [-1, 3], [0.5, -0.25]], jnp.float32),
    )
    return params, state


def test_rls_probe_reads_the_deployed_head_and_leaves_state_and_rng_unchanged() -> None:
    spec = screening_spec("rls_head_resid_l1_preset005")
    params, state = _hand_rls_checkpoint()
    observations = jnp.asarray([[3, 1, 2, 0], [-1, -1, 4, 0]], jnp.float32)
    key = jr.key(23)
    before = screening._declared_learner_state_sha256(params, state, key)
    probe = spec.frozen_probe_logits
    assert probe is not None
    # The normalized hidden features are (1, 2) and (2, 2), respectively.
    # These hand-computed readouts disagree with the deliberately opposite MLP head.
    expected = np.asarray(
        [[0.5, 5 / math.sqrt(3) - 0.25],
         [2 / math.sqrt(3) + 0.5, 4 / math.sqrt(3) - 0.25]],
    )
    logits = probe(params, state, observations, spec.hyperparameters)
    np.testing.assert_allclose(np.asarray(logits), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(jnp.argmax(logits, axis=1)), [1, 1])
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(mlp_logits(params, observations), axis=1)), [0, 0]
    )
    np.testing.assert_array_equal(
        np.asarray(probe(params, state, observations, spec.hyperparameters)), np.asarray(logits)
    )
    assert screening._declared_learner_state_sha256(params, state, key) == before


@pytest.mark.parametrize("checkpoint", ["hand", "initialized", "trained"])
@pytest.mark.parametrize("compiled", [False, True])
def test_rls_probe_matches_actual_learner_loss_readout(
    checkpoint: str, compiled: bool,
) -> None:
    """Recover logits from the real residual learner, without copying its forward."""
    spec = screening_spec("rls_head_resid_l1_preset005")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    key = jr.key(31)
    if checkpoint == "hand":
        params, state = _hand_rls_checkpoint()
        observations = jnp.asarray([[3, 1, 2, 0], [-1, -1, 4, 0]], jnp.float32)
    else:
        params = init_mlp_params(jr.key(17), CONFIG)
        state = init_fn(params)
        if checkpoint == "trained":
            for index in range(4):
                params, state, _ = step_fn(
                    params, state, jnp.asarray(DATA_X[index]),
                    jnp.asarray(DATA_Y[index]), jr.fold_in(key, index),
                )
            assert bool(jnp.any(state.wout != 0))
        observations = jnp.asarray(DATA_X[-2:])

    def learner_logits(
        params: dict[str, Array], state: Any, x: Array, y: Array, key: Array,
    ) -> Array:
        # The returned, pre-update residual loss is 0.5 * ||logits-onehot||^2.
        # Its derivative w.r.t. the readout's unit-bias row is logits-onehot.
        # Differentiate the actual step, including its normalization and
        # residual branch; do not call a probe/helper as the reference.
        def loss(wout: Array) -> Array:
            _, _, metrics = step_fn(params, state.replace(wout=wout), x, y, key)
            return metrics[1]

        loss_gradient: Array = jax.grad(loss)(state.wout)
        return loss_gradient[-1] + jax.nn.one_hot(y, CONFIG.n_classes)

    probe = spec.frozen_probe_logits
    assert probe is not None

    def frozen_logits(params: dict[str, Array], state: Any, x: Array) -> Array:
        return probe(params, state, x, spec.hyperparameters)

    actual_forward = jax.jit(learner_logits) if compiled else learner_logits
    frozen_forward = jax.jit(frozen_logits) if compiled else frozen_logits
    learner_step = jax.jit(step_fn) if compiled else step_fn
    before = screening._declared_learner_state_sha256(params, state, key)
    for index, observation in enumerate(observations):
        label = jnp.asarray(index, jnp.int32)
        _, next_state, _ = learner_step(params, state, observation, label, key)
        # Training normalizes after advancing EMA; a probe must never do so.
        # Freeze the EMA that training actually used, retaining the SAME
        # pre-update body and readout. Both paths now see identical inputs.
        aligned_state = state.replace(norm=next_state.norm)
        aligned_before = screening._declared_learner_state_sha256(params, aligned_state, key)
        assert not np.array_equal(np.asarray(state.norm.count), np.asarray(next_state.norm.count))
        expected = actual_forward(params, state, observation, label, key)
        logits = frozen_forward(params, aligned_state, observation[None, :])
        np.testing.assert_allclose(
            np.asarray(logits[0]), np.asarray(expected), atol=1e-6, rtol=1e-6,
        )
        assert (
            screening._declared_learner_state_sha256(params, aligned_state, key) == aligned_before
        )
    assert screening._declared_learner_state_sha256(params, state, key) == before


@pytest.mark.parametrize("name", ["rls_head_resid_l1_preset005", "sigma0_shiftnorm_d099"])
def test_sentinel_probes_preserve_unprobed_training_continuation(name: str) -> None:
    """Compare the public runner with the same seven updates and no probes."""
    spec = screening_spec(name)
    report = run_recurring_ipmnist_retention_development(
        DATA_X, DATA_Y, spec, seed=19, config=CONFIG, phase_lengths=(2, 3, 2),
        permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
        sentinel_indices=SENTINELS, relearning_window=1,
    )
    init_key, _, key = jr.split(jr.key(jnp.uint32(19)), 3)
    params = init_mlp_params(init_key, CONFIG)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(params)
    orders = build_recurring_ipmnist_online_indices(
        seed=19, n_examples=len(DATA_X), phase_lengths=(2, 3, 2), sentinel_indices=SENTINELS,
    )
    data_x, data_y = jnp.asarray(DATA_X), jnp.asarray(DATA_Y)

    @jax.jit
    def unprobed_phase(
        params: dict[str, Array], state: Any, key: Array, permutation: Array, examples: Array,
    ) -> tuple[dict[str, Array], Any, Array]:
        def update(
            carry: tuple[dict[str, Array], Any, Array], example: Array,
        ) -> tuple[tuple[dict[str, Array], Any, Array], None]:
            p, s, k = carry
            k, step_key = jr.split(k)
            p, s, _ = step_fn(p, s, data_x[example][permutation], data_y[example], step_key)
            return (p, s, k), None

        carry, _ = jax.lax.scan(update, (params, state, key), examples)
        return carry

    for phase, (permutation, order) in enumerate(
        zip((PERMUTATION_A, PERMUTATION_B, PERMUTATION_A), orders, strict=True)
    ):
        params, state, key = unprobed_phase(
            params, state, key, jnp.asarray(permutation), jnp.asarray(order)
        )
        expected_hash = screening._declared_learner_state_sha256(params, state, key)
        scores = [s for s in report.sentinel_scores if s.phase_index == phase]
        assert scores
        for score in scores:
            assert score.learner_state_sha256 == expected_hash


@pytest.mark.parametrize(
    ("failure", "message"),
    [("shape", "one logit"), ("nonfinite", "non-finite logits"), ("mutation", "must not mutate")],
)
def test_runner_rejects_invalid_or_mutating_frozen_forward(
    failure: str, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_forward(
        params: dict[str, Array], state: Any, observation: Array, hp: Mapping[str, float],
    ) -> Array:
        del state, hp
        if failure == "mutation":
            params["b1"] = params["b1"] + 1
        if failure == "shape":
            return jnp.zeros((observation.shape[0], 1))
        if failure == "nonfinite":
            return jnp.full((observation.shape[0], CONFIG.n_classes), jnp.nan)
        return jnp.zeros((observation.shape[0], CONFIG.n_classes))

    spec = dataclasses.replace(
        screening_spec("rls_head_resid_l1_preset005"), frozen_probe_logits=invalid_forward,
    )
    monkeypatch.setattr(screening, "SCREENING_REGISTRY", {spec.name: spec})
    with pytest.raises(ValueError, match=message):
        run_recurring_ipmnist_retention_development(
            DATA_X, DATA_Y, spec, seed=19, config=CONFIG, phase_lengths=(2, 3, 2),
            permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
            sentinel_indices=SENTINELS, relearning_window=1,
        )


def test_other_rls_arm_still_refuses_unsupported_sentinel_path() -> None:
    with pytest.raises(NotImplementedError, match="rls_head"):
        run_recurring_ipmnist_retention_development(
            DATA_X, DATA_Y, screening_spec("rls_head_l1"), seed=19, config=CONFIG,
            phase_lengths=(2, 3, 2),
            permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
            sentinel_indices=SENTINELS, relearning_window=1,
        )


def test_adapter_returns_bound_threshold_free_report_and_frozen_probes() -> None:
    report = run_recurring_ipmnist_retention_development(
        DATA_X,
        DATA_Y,
        screening_spec("sgd_ema_norm"),
        seed=19,
        config=CONFIG,
        phase_lengths=(2, 3, 2),
        permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A.copy()),
        sentinel_indices=SENTINELS,
        relearning_window=1,
    )

    payload = report.to_config()
    assert payload["development_status"] == "development-only-not-assessed"
    assert payload["assessment_status"] == "not-assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert payload["retention_claimed"] is False
    assert payload["catastrophic_forgetting_absence_claimed"] is False

    assert tuple(summary.observation_count for summary in report.phase_summaries) == (2, 3, 2)
    assert len(report.sentinel_scores) == 5
    checkpoint_hashes = tuple(score.learner_state_sha256 for score in report.sentinel_scores)
    assert checkpoint_hashes[1] == checkpoint_hashes[2]
    assert checkpoint_hashes[3] == checkpoint_hashes[4]
    assert checkpoint_hashes[0] != checkpoint_hashes[1]

    binding_a, binding_b = report.protocol.sentinel_bindings
    assert binding_a.permutation_sha256 == ipmnist_permutation_sha256(PERMUTATION_A)
    assert binding_b.permutation_sha256 == ipmnist_permutation_sha256(PERMUTATION_B)
    assert binding_a.sentinel_set_sha256 == ipmnist_sentinel_set_sha256(
        DATA_X, DATA_Y, PERMUTATION_A, SENTINELS
    )
    assert binding_b.sentinel_set_sha256 == ipmnist_sentinel_set_sha256(
        DATA_X, DATA_Y, PERMUTATION_B, SENTINELS
    )
    assert tuple(
        (phase.permutation_id, phase.exposure_index) for phase in report.protocol.phases
    ) == (
        (binding_a.permutation_id, 0),
        (binding_b.permutation_id, 0),
        (binding_a.permutation_id, 1),
    )


def test_online_schedule_excludes_sentinels_and_exactly_matches_a_orders() -> None:
    schedule = build_recurring_ipmnist_online_indices(
        seed=19,
        n_examples=len(DATA_X),
        phase_lengths=(2, 3, 2),
        sentinel_indices=SENTINELS,
    )
    repeated = build_recurring_ipmnist_online_indices(
        seed=19,
        n_examples=len(DATA_X),
        phase_lengths=(2, 3, 2),
        sentinel_indices=SENTINELS,
    )

    assert tuple(len(phase) for phase in schedule) == (2, 3, 2)
    assert np.array_equal(schedule[0], schedule[2])
    assert not np.shares_memory(schedule[0], schedule[2])
    assert not np.array_equal(schedule[0], schedule[1][: len(schedule[0])])
    for phase, replay in zip(schedule, repeated, strict=True):
        assert np.array_equal(phase, replay)
        assert len(np.unique(phase)) == len(phase)
        assert set(int(index) for index in phase).isdisjoint(SENTINELS)


def test_adapter_rejects_cloned_custom_and_stateful_probe_specs() -> None:
    registered = screening_spec("sgd_ema_norm")
    hidden_probe_state: list[int] = []

    def stateful_probe(
        state: Any, observation: Array, hyperparameters: Mapping[str, float]
    ) -> Array:
        del state, hyperparameters
        hidden_probe_state.append(len(hidden_probe_state))
        return observation + float(len(hidden_probe_state))

    candidates = (
        dataclasses.replace(registered),
        dataclasses.replace(registered, name="custom-sgd-ema-norm"),
        dataclasses.replace(registered, frozen_probe_input=stateful_probe),
    )
    for candidate in candidates:
        with pytest.raises(ValueError, match="exact registered object"):
            run_recurring_ipmnist_retention_development(
                DATA_X,
                DATA_Y,
                candidate,
                seed=19,
                config=CONFIG,
                phase_lengths=(2, 3, 2),
                permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
                sentinel_indices=SENTINELS,
                relearning_window=1,
            )
    assert hidden_probe_state == []


def test_sentinel_digest_binds_order_labels_source_rows_and_transformed_inputs() -> None:
    original = ipmnist_sentinel_set_sha256(
        DATA_X, DATA_Y, PERMUTATION_A, SENTINELS
    )
    assert original == ipmnist_sentinel_set_sha256(
        DATA_X.copy(), DATA_Y.copy(), PERMUTATION_A.copy(), SENTINELS
    )
    assert original != ipmnist_sentinel_set_sha256(
        DATA_X, DATA_Y, PERMUTATION_A, tuple(reversed(SENTINELS))
    )

    changed_labels = DATA_Y.copy()
    changed_labels[SENTINELS[0]] = 1 - changed_labels[SENTINELS[0]]
    assert original != ipmnist_sentinel_set_sha256(
        DATA_X, changed_labels, PERMUTATION_A, SENTINELS
    )

    changed_source = DATA_X.copy()
    changed_source[SENTINELS[0], 0] += np.float32(0.125)
    assert original != ipmnist_sentinel_set_sha256(
        changed_source, DATA_Y, PERMUTATION_A, SENTINELS
    )
    assert original != ipmnist_sentinel_set_sha256(
        DATA_X, DATA_Y, PERMUTATION_B, SENTINELS
    )


@pytest.mark.parametrize(
    ("phase_lengths", "permutations", "sentinels", "message"),
    [
        ((3, 2, 3), (PERMUTATION_A, PERMUTATION_B, PERMUTATION_A), SENTINELS, "config"),
        (
            (2, 2, 2),
            (PERMUTATION_A, PERMUTATION_B, PERMUTATION_B),
            SENTINELS,
            "first and third",
        ),
        (
            (2, 2, 2),
            (PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
            (7, 7),
            "unique",
        ),
    ],
)
def test_adapter_rejects_ambiguous_or_unbound_recurrence_inputs(
    phase_lengths: tuple[int, int, int],
    permutations: tuple[np.ndarray, np.ndarray, np.ndarray],
    sentinels: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_recurring_ipmnist_retention_development(
            DATA_X,
            DATA_Y,
            screening_spec("adamw_control"),
            seed=1,
            config=CONFIG,
            phase_lengths=phase_lengths,
            permutations=permutations,
            sentinel_indices=sentinels,
            relearning_window=1,
        )


def _hidden_rms_active(spec: Any) -> bool:
    return any(
        float(spec.hyperparameters.get(key, 0.0)) != 0.0
        for key in ("hidden_rms", "flag_hidden_rms")
    )


def test_hidden_rms_active_arms_refuse_plain_mlp_sentinel_probes() -> None:
    """Arms whose forward pass RMS-normalizes hidden layers cannot be probed with mlp_logits."""
    active = sorted(name for name, spec in SCREENING_REGISTRY.items() if _hidden_rms_active(spec))
    assert {"sigma0_hidden_norm", "disc_r1", "disc_r2", "disc_r3", "disc_r1_pscale"} <= set(active)
    sentinel_inputs = jnp.zeros((2, CONFIG.input_dim), dtype=jnp.float32)
    for name in active:
        spec = SCREENING_REGISTRY[name]
        init_fn, _step_fn = spec.factory(spec.hyperparameters)
        state = init_fn(init_mlp_params(jr.key(0), CONFIG))
        with pytest.raises(NotImplementedError, match="hidden-RMS"):
            spec.frozen_probe_input(state, sentinel_inputs, spec.hyperparameters)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("bounded_structure_off", "bounded-structure"),
        ("bounded_growth", "bounded-structure"),
        ("bounded_elastic", "bounded-structure"),
        ("replay_context_only", "context-enabled replay"),
        ("replay_context_full", "context-enabled replay"),
    ],
)
def test_arms_whose_forward_is_not_the_plain_mlp_refuse_sentinel_probes(
    name: str, message: str
) -> None:
    """The probe harness scores ``mlp_logits``; arms that deploy another forward fail closed."""
    spec = screening_spec(name)
    init_fn, _step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(init_mlp_params(jr.key(0), CONFIG))
    sentinel_inputs = jnp.zeros((2, CONFIG.input_dim), dtype=jnp.float32)
    with pytest.raises(NotImplementedError, match=message):
        spec.frozen_probe_input(state, sentinel_inputs, spec.hyperparameters)


@pytest.mark.parametrize("name", ["replay_context_mechanism_off", "replay_gradient_only"])
def test_context_free_replay_arms_keep_the_raw_probe(name: str) -> None:
    spec = screening_spec(name)
    assert spec.hyperparameters["context_weight"] == 0.0
    init_fn, _step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(init_mlp_params(jr.key(0), CONFIG))
    sentinel_inputs = jnp.ones((2, CONFIG.input_dim), dtype=jnp.float32)
    probed = spec.frozen_probe_input(state, sentinel_inputs, spec.hyperparameters)
    np.testing.assert_array_equal(np.asarray(probed), np.asarray(sentinel_inputs))


def test_hidden_rms_inactive_sibling_keeps_its_input_side_probe() -> None:
    spec = screening_spec("disc_r1_pscale_norms")
    assert not _hidden_rms_active(spec)
    init_fn, _step_fn = spec.factory(spec.hyperparameters)
    state = init_fn(init_mlp_params(jr.key(0), CONFIG))
    sentinel_inputs = jnp.ones((2, CONFIG.input_dim), dtype=jnp.float32)
    probed = spec.frozen_probe_input(state, sentinel_inputs, spec.hyperparameters)
    assert probed.shape == sentinel_inputs.shape

