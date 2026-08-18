"""Mechanism, reduction, resource, and protocol tests for plasticity comparators."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.plasticity_comparators import (
    PAPER_REVISIONS,
    adamo_update,
    bounded_elastic_mask,
    churn_loss,
    deep_fourier_features,
    effective_rank,
    intentional_td_step_size,
    intentional_trace_step_size,
    interval_dropout,
    isometry_gradient,
    isometry_penalty,
    l2_er_objective,
    nap_project,
    noise_curvature_critical_step_size,
    noise_curvature_step_size,
    ntk_threshold_rank,
    persistent_array_bytes,
    protocol,
    smooth_leaky,
    utility_scaled_pull,
)


def test_protocols_pin_sources_and_are_permanently_nonpromoting() -> None:
    assert len(PAPER_REVISIONS) == 11
    for name, paper in PAPER_REVISIONS.items():
        record = protocol(name, persistent_bytes=8, environment_or_data_steps=4, model_queries=2)
        assert record.paper == paper
        assert record.development_only is True
        assert record.scientific_promotion_allowed is False
        assert record.matched_axes == ("seed", "updates", "observations", "example_order")


def test_l2_er_rank_and_mechanism_off_reduction() -> None:
    identity = jnp.eye(3)
    collapsed = jnp.ones((3, 3))
    assert np.isclose(float(effective_rank(identity)), 3.0)
    assert np.isclose(float(effective_rank(collapsed)), 1.0, atol=1e-5)
    loss = jnp.asarray(2.0)
    off = l2_er_objective(loss, (identity,), (identity,), l2_strength=0, rank_strength=0)
    assert float(off) == 2.0


def test_adamo_moments_exclude_isometry_and_off_matches_adam() -> None:
    weights = jnp.asarray([[2.0, 0.0], [0.0, 1.0]])
    gradient = jnp.ones_like(weights)
    zeros = jnp.zeros_like(weights)
    updated, moment, variance = adamo_update(
        weights,
        gradient,
        zeros,
        zeros,
        step=1,
        learning_rate=0.01,
        isometry_strength=0.0,
    )
    assert jnp.allclose(moment, 0.1 * gradient)
    assert jnp.allclose(variance, 0.001 * gradient)
    assert jnp.all(updated < weights + 1e-6)
    assert jnp.allclose(isometry_gradient(jnp.eye(2)), 0)
    assert float(isometry_penalty(jnp.eye(2))) == 0


def test_intentional_update_equations_and_trace_reduction() -> None:
    gradient = jnp.asarray([3.0, 4.0])
    assert np.isclose(float(intentional_td_step_size(gradient, intended_fraction=0.5)), 0.02)
    trace = intentional_trace_step_size(
        gradient,
        jnp.ones_like(gradient),
        intended_fraction=0.5,
        discounted_gradient_energy=jnp.asarray(25.0),
    )
    assert np.isclose(float(trace), 0.02)


def test_bounded_elastic_growth_and_pruning_preserve_peak_shape() -> None:
    active = jnp.asarray([True, True, False, False])
    changed = bounded_elastic_mask(active, jnp.asarray([0.1, 0.9, 0.0, 0.0]), grow=1, prune=1)
    assert changed.shape == active.shape
    assert changed.tolist() == [False, True, True, False]
    assert bounded_elastic_mask(active, jnp.ones(4), grow=0, prune=0).tolist() == active.tolist()


def test_utility_pull_reductions() -> None:
    weights = jnp.asarray([2.0, 2.0])
    initial = jnp.asarray([0.0, 0.0])
    utility = jnp.asarray([0.0, 1.0])
    assert jnp.array_equal(utility_scaled_pull(weights, initial, utility, strength=0), weights)
    assert jnp.array_equal(
        utility_scaled_pull(weights, initial, utility, strength=0.5, mode="utility"),
        jnp.asarray([1.0, 2.0]),
    )
    assert jnp.array_equal(
        utility_scaled_pull(weights, initial, utility, strength=0.5, mode="hard_reset"),
        jnp.asarray([0.0, 2.0]),
    )
    assert jnp.array_equal(
        utility_scaled_pull(weights, initial, utility, strength=0, mode="hard_reset"),
        weights,
    )
    assert jnp.array_equal(
        utility_scaled_pull(weights, initial, utility, strength=0.5, mode="l2_init"),
        jnp.ones(2),
    )


def test_nap_projection_and_off_reduction() -> None:
    weights = jnp.asarray([3.0, 4.0])
    assert jnp.array_equal(nap_project(weights, initial_norm=2.0, enabled=False), weights)
    assert np.isclose(float(jnp.linalg.norm(nap_project(weights, initial_norm=2.0))), 2.0)


def test_c_chain_churn_loss_and_off_reduction() -> None:
    before = jnp.asarray([1.0, 2.0])
    after = jnp.asarray([2.0, 4.0])
    assert float(churn_loss(before, after, strength=0)) == 0
    assert np.isclose(float(churn_loss(before, after, strength=1)), 1.25)
    assert int(ntk_threshold_rank(jnp.eye(3), threshold=0.66)) == 2


def test_low_cost_activation_and_feature_controls() -> None:
    value = jnp.asarray([-2.0, 0.0, 2.0])
    assert jnp.allclose(smooth_leaky(value, alpha=1, power=3, curvature=5), value)
    assert jnp.array_equal(
        interval_dropout(value, jr.key(0), relu_probability=1), jnp.maximum(value, 0)
    )
    assert jnp.array_equal(
        interval_dropout(value, jr.key(0), relu_probability=0.5, training=False),
        0.5 * value,
    )
    features = deep_fourier_features(value)
    assert features.shape == (6,)
    assert jnp.array_equal(deep_fourier_features(value, enabled=False), value)


def test_noise_curvature_scheduler_and_off_reduction() -> None:
    bound = noise_curvature_critical_step_size(
        batch_size=2,
        squared_gradient_mean=4.0,
        per_sample_gradient_variance=2.0,
        normalized_curvature_variance=0.5,
        safety_margin=0.0,
    )
    assert bound == 2.0
    no_noise_bound = noise_curvature_critical_step_size(
        batch_size=1,
        squared_gradient_mean=1.0,
        per_sample_gradient_variance=0.0,
        normalized_curvature_variance=0.0,
    )
    assert no_noise_bound == float("inf")
    assert noise_curvature_step_size(
        0.1,
        effective_step_size=0.01,
        safe_bound=no_noise_bound,
        early_training=False,
    ) == 0.1
    assert noise_curvature_step_size(
        0.1,
        effective_step_size=0.2,
        safe_bound=0.15,
        early_training=False,
    ) == pytest.approx(0.099)
    assert noise_curvature_step_size(
        0.1,
        effective_step_size=0.2,
        safe_bound=0.15,
        early_training=False,
        enabled=False,
    ) == 0.1


def test_exact_resource_accounting_and_hostile_protocol_scalars() -> None:
    assert persistent_array_bytes(np.zeros((2,), np.float32), np.zeros((3,), np.int32)) == 20

    class HostileInt(int):
        pass

    with pytest.raises(ValueError, match="persistent_bytes"):
        protocol("nap", persistent_bytes=HostileInt(1))
    with pytest.raises(ValueError, match="strength"):
        utility_scaled_pull(jnp.ones(1), jnp.zeros(1), jnp.zeros(1), strength=True)
    with pytest.raises(ValueError, match="arrays"):
        persistent_array_bytes([1.0, 2.0])  # type: ignore[arg-type]
