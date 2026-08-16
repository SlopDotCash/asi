"""Tests for the narrow sparse-world-model decision-fidelity evaluation."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
)
from alberta_framework.evaluation.ftl_decision_fidelity import (
    CONDITION_NAMES,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    DecisionFidelityConfig,
    evaluate_sparse_snapshot,
    precompute_decision_probes,
    run_ftl_decision_fidelity_evaluation,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def decision_report():
    """Run the paired thirty-seed evaluation once for the module."""

    return run_ftl_decision_fidelity_evaluation()


def test_probe_precomputation_is_fixed_held_out_and_multistep() -> None:
    config = DecisionFidelityConfig()
    probes = precompute_decision_probes(config, seed=11)
    repeated = precompute_decision_probes(config, seed=11)
    probe_count = 2 * config.probes_per_domain
    menu_size = len(config.menu_amplitudes)

    assert probes.initial_observations.shape == (probe_count, 1)
    assert probes.goals.shape == (probe_count, 1)
    assert probes.action_sequences.shape == (menu_size, config.horizon, 1)
    assert probes.true_next_observations.shape == (
        probe_count,
        menu_size,
        config.horizon,
        1,
    )
    assert probes.true_rewards.shape == (probe_count, menu_size, config.horizon)
    assert probes.true_returns.shape == (probe_count, menu_size)
    np.testing.assert_array_equal(probes.true_returns, repeated.true_returns)
    np.testing.assert_allclose(
        probes.true_returns,
        probes.true_rewards.sum(axis=2),
        rtol=0.0,
        atol=1.0e-12,
    )

    # Every choice is indistinguishable at the first action and separates later.
    np.testing.assert_array_equal(
        probes.action_sequences[:, 0],
        np.zeros((menu_size, 1), dtype=np.float32),
    )
    assert np.unique(probes.action_sequences[:, 1:, 0], axis=0).shape[0] == menu_size
    assert np.all(np.ptp(probes.true_returns, axis=1) > 0.0)
    assert np.all(probes.initial_observations[probes.domain_indices == 0] < 0.0)
    assert np.all(probes.initial_observations[probes.domain_indices == 1] > 0.0)


def test_sparse_probe_branching_does_not_mutate_model_state() -> None:
    config = DecisionFidelityConfig(
        phase_steps=16,
        horizon=3,
        probes_per_domain=2,
        projection_dim=4,
        bins=4,
        bootstrap_resamples=1_000,
    )
    model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=config.projection_dim,
            bins=config.bins,
            ridge=config.ridge,
            prediction_clip=config.prediction_clip,
        )
    )
    state = model.init(jax.random.key(3))
    before = tuple(np.asarray(leaf).copy() for leaf in jax.tree.leaves(state))

    metrics = evaluate_sparse_snapshot(
        "read_only_probe",
        model,
        state,
        precompute_decision_probes(config, seed=3),
        config,
    )

    after = tuple(np.asarray(leaf) for leaf in jax.tree.leaves(state))
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)
    assert np.isfinite(dataclasses.astuple(metrics)[1:]).all()
    assert int(state.step_count) == 0


def test_thirty_seed_report_contains_paired_finite_evidence(decision_report) -> None:
    report = decision_report

    assert DEVELOPMENT_SEEDS == tuple(range(30))
    assert EVIDENCE_SEEDS == tuple(range(30, 60))
    assert set(DEVELOPMENT_SEEDS).isdisjoint(EVIDENCE_SEEDS)
    assert report.seeds == EVIDENCE_SEEDS
    assert len(report.seed_results) == 30
    assert tuple(aggregate.condition for aggregate in report.aggregates) == CONDITION_NAMES
    assert all(
        tuple(metric.condition for metric in seed_result.metrics) == CONDITION_NAMES
        for seed_result in report.seed_results
    )

    for aggregate in report.aggregates:
        for field in dataclasses.fields(aggregate):
            if field.name == "condition":
                continue
            estimate = getattr(aggregate, field.name)
            assert estimate.sample_size == 30
            assert estimate.resamples == report.config.bootstrap_resamples
            assert np.isfinite((estimate.estimate, estimate.lower, estimate.upper)).all()
        assert 0.0 <= aggregate.normalized_regret.estimate <= 1.0
        assert 0.0 <= aggregate.domain_a_normalized_regret.estimate <= 1.0
        assert 0.0 <= aggregate.domain_b_normalized_regret.estimate <= 1.0
        assert 0.0 <= aggregate.oracle_pick_rate.estimate <= 1.0
        assert aggregate.reward_mae.estimate >= 0.0
        assert aggregate.return_mae.estimate >= 0.0


def test_multi_step_sparse_model_beats_untrained_and_online_linear_baselines(
    decision_report,
) -> None:
    report = decision_report
    untrained = report.aggregate("sparse_untrained")
    learned = report.aggregate("sparse_after_b")
    linear = report.aggregate("linear_after_b")
    versus_untrained = report.comparison("sparse_after_b_vs_untrained_regret_reduction")
    versus_linear = report.comparison("sparse_after_b_vs_linear_regret_reduction")

    assert untrained.normalized_regret.lower > 0.25
    assert learned.normalized_regret.upper < 0.01
    assert versus_untrained.estimate.lower > 0.25
    assert versus_linear.estimate.lower > 0.03
    assert learned.oracle_pick_rate.estimate > 0.85
    assert learned.reward_mae.estimate < 0.2 * linear.reward_mae.estimate
    assert learned.normalized_return_mae.estimate < 0.15 * linear.normalized_return_mae.estimate


def test_recurring_visitation_retains_a_while_learning_b(decision_report) -> None:
    report = decision_report
    after_a1 = report.aggregate("sparse_after_a1")
    after_b = report.aggregate("sparse_after_b")
    after_a2 = report.aggregate("sparse_after_a2")
    interference = report.comparison("sparse_domain_a_regret_change_after_b")
    recovery = report.comparison("sparse_domain_a_recovery_after_a2")

    assert after_a1.domain_a_normalized_regret.estimate < 1.0e-3
    assert after_a1.domain_b_normalized_regret.estimate > 0.10
    assert after_b.domain_b_normalized_regret.upper < 0.01
    assert interference.estimate.upper < 0.01
    assert after_a2.domain_a_normalized_regret.estimate <= (
        after_b.domain_a_normalized_regret.estimate
    )
    assert recovery.estimate.estimate >= 0.0


def test_raw_online_ridge_baseline_is_nontrivial_and_improves_with_b(
    decision_report,
) -> None:
    after_a1 = decision_report.aggregate("linear_after_a1")
    after_b = decision_report.aggregate("linear_after_b")
    after_a2 = decision_report.aggregate("linear_after_a2")

    assert after_a1.domain_a_normalized_regret.estimate < 0.01
    assert after_b.normalized_regret.estimate < after_a1.normalized_regret.estimate
    assert after_a2.normalized_regret.estimate <= after_b.normalized_regret.estimate
    assert after_b.normalized_regret.estimate < 0.10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_steps": 15},
        {"horizon": 2},
        {"probes_per_domain": 1},
        {"menu_amplitudes": (-0.5, 0.5)},
        {"menu_amplitudes": (-0.5, 0.0, 0.0)},
        {"projection_dim": 0},
        {"bins": 1},
        {"ridge": 0.0},
        {"prediction_clip": 0.0},
        {"state_bound": 0.0},
        {"action_cost": -0.1},
        {"bootstrap_resamples": 999},
        {"confidence_level": 1.0},
        {"bootstrap_seed": -1},
    ],
)
def test_invalid_decision_configuration_is_rejected(
    kwargs: dict[str, float | int | tuple[float, ...]],
) -> None:
    with pytest.raises(ValueError):
        DecisionFidelityConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seeds",
    [(), (1, 1), (-1,), (2**31,)],
)
def test_invalid_seed_schedules_are_rejected(seeds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        run_ftl_decision_fidelity_evaluation(seeds=seeds)


def test_decision_fidelity_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="phase_steps"):
        DecisionFidelityConfig(phase_steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="phase_steps"):
        DecisionFidelityConfig(phase_steps=180.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon"):
        DecisionFidelityConfig(horizon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projection_dim"):
        DecisionFidelityConfig(projection_dim=12.5)  # type: ignore[arg-type]


def test_decision_fidelity_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = DecisionFidelityConfig(
        phase_steps=np.int32(180),
        horizon=np.int64(6),
        probes_per_domain=np.uint16(12),
        projection_dim=np.int32(12),
        bins=np.uint8(7),
        bootstrap_resamples=np.int64(5_000),
        bootstrap_seed=np.uint32(2_026_073_001),
    )
    assert type(cfg.phase_steps) is int
    assert type(cfg.horizon) is int
    assert type(cfg.probes_per_domain) is int
    assert type(cfg.projection_dim) is int
    assert type(cfg.bins) is int
    assert type(cfg.bootstrap_resamples) is int
    assert type(cfg.bootstrap_seed) is int
    assert cfg.phase_steps == 180
    assert cfg.horizon == 6
