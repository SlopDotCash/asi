"""Focused checks for the development-only recurring IPMNIST adapter."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import alberta_framework.benchmarks.ipmnist_screening as ipmnist_screening
from alberta_framework.benchmarks.ipmnist_screening import (
    build_recurring_ipmnist_online_indices,
    ipmnist_permutation_sha256,
    ipmnist_sentinel_set_sha256,
    run_recurring_ipmnist_retention_development,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

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


@pytest.mark.parametrize(
    "config_name",
    [
        "rls_head_resid_l1_preset005",
        "rls_head_resid_l1_preset005_l2init",
    ],
)
def test_adapter_runs_rls_head_states_with_deployed_sentinel_logits(
    config_name: str,
) -> None:
    report = run_recurring_ipmnist_retention_development(
        DATA_X,
        DATA_Y,
        screening_spec(config_name),
        seed=19,
        config=CONFIG,
        phase_lengths=(2, 3, 2),
        permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A.copy()),
        sentinel_indices=SENTINELS,
        relearning_window=1,
    )

    assert len(report.sentinel_scores) == 5
    assert all(
        score.sentinel_case_count == len(SENTINELS)
        for score in report.sentinel_scores
    )


def test_protocol_id_binds_registered_probe_strategy() -> None:
    deployed = screening_spec("rls_head_resid_l1_preset005")
    legacy_input_strategy = dataclasses.replace(
        deployed,
        frozen_probe_input=ipmnist_screening._raw_frozen_probe_input,
        frozen_probe_logits=None,
    )

    def protocol_id(spec: ipmnist_screening.ScreeningSpec) -> str:
        return ipmnist_screening._recurring_protocol_id(
            spec=spec,
            seed=19,
            config=CONFIG,
            phase_lengths=(2, 3, 2),
            permutation_sha256=("1" * 64, "2" * 64, "1" * 64),
            sentinel_indices_sha256="3" * 64,
            online_indices_sha256=("4" * 64, "5" * 64, "4" * 64),
            relearning_window=1,
        )

    assert protocol_id(deployed) != protocol_id(legacy_input_strategy)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("shape", "frozen_probe_logits must return shape"),
        ("nonfinite", "non-finite logits"),
        ("mutation", "must not mutate learner state"),
    ],
)
def test_adapter_validates_custom_frozen_logits_capability(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    registered = screening_spec("adamw_control")

    def invalid_probe(
        params: dict[str, Array],
        state: Any,
        observations: Array,
        hyperparameters: Mapping[str, float],
    ) -> Array:
        del state, hyperparameters
        if failure == "shape":
            return jnp.zeros((observations.shape[0], CONFIG.n_classes + 1))
        if failure == "nonfinite":
            return jnp.full((observations.shape[0], CONFIG.n_classes), jnp.nan)
        params["b3"] = params["b3"] + 1.0
        return jnp.zeros((observations.shape[0], CONFIG.n_classes))

    custom = dataclasses.replace(
        registered,
        name=f"invalid-probe-{failure}",
        frozen_probe_logits=invalid_probe,
    )
    monkeypatch.setattr(
        ipmnist_screening,
        "SCREENING_REGISTRY",
        MappingProxyType({**ipmnist_screening.SCREENING_REGISTRY, custom.name: custom}),
    )

    with pytest.raises(ValueError, match=message):
        run_recurring_ipmnist_retention_development(
            DATA_X,
            DATA_Y,
            custom,
            seed=19,
            config=CONFIG,
            phase_lengths=(2, 3, 2),
            permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A),
            sentinel_indices=SENTINELS,
            relearning_window=1,
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
