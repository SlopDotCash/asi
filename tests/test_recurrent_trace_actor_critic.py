"""Numerical and streaming contracts for the RTU-RTRL actor-critic core."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import alberta_framework.core.recurrent_trace_actor_critic as recurrent_trace_module
from alberta_framework.core import (
    RecurrentTraceActorCriticAgent as ExportedRecurrentTraceActorCriticAgent,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
    RTUParameters,
    RTUSensitivities,
    RTUState,
    adaptive_obgd_update,
    exact_rtrl_gradient,
    initialize_rtu_network_parameters,
    obgd_update,
    parameterless_layer_norm,
    rtu_forward,
    rtu_network_encode,
    rtu_network_output,
    rtu_step,
    zero_rtu_sensitivities,
    zero_rtu_state,
)

pytestmark = pytest.mark.slow


def _assert_tree_all_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        if str(actual_leaf.dtype).startswith("key<"):
            assert jnp.array_equal(
                jr.key_data(actual_leaf),
                jr.key_data(expected_leaf),
            )
            continue
        assert jnp.allclose(
            actual_leaf,
            expected_leaf,
            atol=atol,
            rtol=rtol,
        )


def _assert_tree_finite(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _small_config(**overrides: Any) -> RecurrentTraceActorCriticConfig:
    values: dict[str, Any] = {
        "n_actions": 3,
        "hidden_size": 3,
        "encoder_width": 2,
        "output_width": 4,
        "sparsity": 0.0,
        "r_min": 0.1,
        "r_max": 0.9,
        "normalize_observations": False,
        "normalize_rewards": False,
    }
    values.update(overrides)
    return RecurrentTraceActorCriticConfig(**values)


def _unroll_rtu(
    params: RTUParameters,
    observations: Array,
    *,
    epsilon: float,
) -> RTUState:
    state = zero_rtu_state(params.nu_log.shape[0])
    for observation in observations:
        state = rtu_forward(
            params,
            state,
            observation,
            epsilon=epsilon,
        )
    return state


def _expanded_sensitivity(
    sensitivities: RTUSensitivities,
) -> RTUParameters:
    """Expand compressed diagonal sensitivities to full autodiff Jacobians."""
    hidden_size = sensitivities.nu_log.shape[1]
    input_dim = sensitivities.b_real.shape[2]
    indices = jnp.arange(hidden_size)

    def recurrent_leaf(compressed: Array) -> Array:
        expanded = jnp.zeros(
            (2 * hidden_size, hidden_size),
            dtype=compressed.dtype,
        )
        expanded = expanded.at[indices, indices].set(compressed[0])
        return expanded.at[hidden_size + indices, indices].set(compressed[1])

    def input_leaf(compressed: Array) -> Array:
        expanded = jnp.zeros(
            (2 * hidden_size, hidden_size, input_dim),
            dtype=compressed.dtype,
        )
        expanded = expanded.at[indices, indices, :].set(compressed[0])
        return expanded.at[hidden_size + indices, indices, :].set(compressed[1])

    return RTUParameters(
        nu_log=recurrent_leaf(sensitivities.nu_log),
        theta_log=recurrent_leaf(sensitivities.theta_log),
        b_real=input_leaf(sensitivities.b_real),
        b_imag=input_leaf(sensitivities.b_imag),
    )


def test_multistep_rtrl_sensitivities_match_full_unrolled_autodiff() -> None:
    """The forward trace must include history, not only the immediate Jacobian."""
    config = _small_config()
    params = initialize_rtu_network_parameters(
        jr.key(1),
        input_dim=2,
        output_dim=2,
        config=config,
    ).rtu
    observations = jnp.asarray(
        (
            (0.2, -0.4),
            (0.5, 0.1),
            (-0.3, 0.8),
            (0.7, -0.2),
        ),
        dtype=jnp.float32,
    )

    state = zero_rtu_state(config.hidden_size)
    sensitivities = zero_rtu_sensitivities(config.hidden_size, 2)
    for observation in observations:
        state, sensitivities = rtu_step(
            params,
            state,
            sensitivities,
            observation,
            epsilon=config.rtu_epsilon,
        )

    def unrolled_output(candidate: RTUParameters) -> Array:
        final_state = _unroll_rtu(
            candidate,
            observations,
            epsilon=config.rtu_epsilon,
        )
        return jnp.concatenate((final_state.real, final_state.imaginary))

    full_jacobian = jax.jacrev(unrolled_output)(params)
    _assert_tree_all_close(
        _expanded_sensitivity(sensitivities),
        full_jacobian,
        atol=2e-6,
        rtol=2e-6,
    )

    _, immediate_only = rtu_step(
        params,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, 2),
        observations[-1],
        epsilon=config.rtu_epsilon,
    )
    assert not jnp.allclose(
        sensitivities.b_real,
        immediate_only.b_real,
        atol=1e-5,
        rtol=1e-5,
    )


def test_saturated_radius_stabilization_matches_autodiff_jacobian_and_gradient() -> None:
    """The floored input norm has zero ``nu_log`` derivative in its flat branch."""
    config = _small_config(rtu_epsilon=1e-4)
    params = initialize_rtu_network_parameters(
        jr.key(19),
        input_dim=2,
        output_dim=config.n_actions,
        config=config,
    )
    params = params._replace(
        rtu=params.rtu._replace(
            nu_log=jnp.full(
                (config.hidden_size,),
                -20.0,
                dtype=jnp.float32,
            )
        )
    )
    observations = jnp.asarray(
        ((0.4, -0.7), (0.2, 0.5), (-0.6, 0.1)),
        dtype=jnp.float32,
    )

    first_state, first_sensitivities = rtu_step(
        params.rtu,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, config.encoder_width),
        observations[0],
        epsilon=config.rtu_epsilon,
    )

    def first_output(candidate: RTUParameters) -> Array:
        candidate_state = rtu_forward(
            candidate,
            zero_rtu_state(config.hidden_size),
            observations[0],
            epsilon=config.rtu_epsilon,
        )
        return jnp.concatenate((candidate_state.real, candidate_state.imaginary))

    first_jacobian = jax.jacrev(first_output)(params.rtu)
    _assert_tree_all_close(
        _expanded_sensitivity(first_sensitivities),
        first_jacobian,
        atol=1e-7,
        rtol=1e-7,
    )
    assert jnp.allclose(first_jacobian.nu_log, 0.0, atol=1e-12)

    state = first_state
    sensitivities = first_sensitivities
    for observation in observations[1:]:
        state, sensitivities = rtu_step(
            params.rtu,
            state,
            sensitivities,
            observation,
            epsilon=config.rtu_epsilon,
        )

    def objective(logits: Array) -> Array:
        return jnp.dot(
            logits,
            jnp.asarray((0.3, -0.2, 0.7), dtype=logits.dtype),
        )

    rtrl_value, rtrl_gradient = exact_rtrl_gradient(
        params,
        state,
        sensitivities,
        objective,
        rtu_epsilon=config.rtu_epsilon,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )

    def unrolled_objective(candidate: type(params)) -> Array:
        final_state = _unroll_rtu(
            candidate.rtu,
            observations,
            epsilon=config.rtu_epsilon,
        )
        return objective(
            rtu_network_output(
                candidate,
                final_state,
                layer_norm_epsilon=config.layer_norm_epsilon,
                negative_slope=config.leaky_relu_slope,
            )
        )

    autodiff_value, autodiff_gradient = jax.value_and_grad(unrolled_objective)(params)
    assert jnp.allclose(rtrl_value, autodiff_value, atol=2e-6, rtol=2e-6)
    _assert_tree_all_close(
        rtrl_gradient,
        autodiff_gradient,
        atol=3e-6,
        rtol=3e-6,
    )


def test_exact_rtrl_scalar_gradient_matches_multistep_backpropagation() -> None:
    """Explicit RTRL contraction must equal full unrolled reverse-mode AD."""
    config = _small_config()
    params = initialize_rtu_network_parameters(
        jr.key(2),
        input_dim=2,
        output_dim=config.n_actions,
        config=config,
    )
    observations = jnp.asarray(
        (
            (0.1, -0.2),
            (0.6, 0.3),
            (-0.4, 0.9),
            (0.25, -0.75),
        ),
        dtype=jnp.float32,
    )
    state = zero_rtu_state(config.hidden_size)
    sensitivities = zero_rtu_sensitivities(config.hidden_size, 2)
    for observation in observations:
        state, sensitivities = rtu_step(
            params.rtu,
            state,
            sensitivities,
            observation,
            epsilon=config.rtu_epsilon,
        )

    def objective(logits: Array) -> Array:
        log_probabilities = jax.nn.log_softmax(logits / 0.7)
        probabilities = jnp.exp(log_probabilities)
        entropy = -jnp.sum(probabilities * log_probabilities)
        return log_probabilities[1] + 0.03 * entropy

    rtrl_value, rtrl_gradient = exact_rtrl_gradient(
        params,
        state,
        sensitivities,
        objective,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )

    def unrolled_objective(candidate: type(params)) -> Array:
        final_state = _unroll_rtu(
            candidate.rtu,
            observations,
            epsilon=config.rtu_epsilon,
        )
        logits = rtu_network_output(
            candidate,
            final_state,
            layer_norm_epsilon=config.layer_norm_epsilon,
            negative_slope=config.leaky_relu_slope,
        )
        return objective(logits)

    autodiff_value, autodiff_gradient = jax.value_and_grad(unrolled_objective)(params)
    assert jnp.allclose(rtrl_value, autodiff_value, atol=2e-6, rtol=2e-6)
    _assert_tree_all_close(
        rtrl_gradient,
        autodiff_gradient,
        atol=3e-6,
        rtol=3e-6,
    )


def test_adversarial_logs_and_long_rtrl_scan_remain_finite() -> None:
    config = _small_config(hidden_size=3, encoder_width=2)
    network_params = initialize_rtu_network_parameters(
        jr.key(25),
        input_dim=2,
        output_dim=config.n_actions,
        config=config,
    )
    params = network_params.rtu._replace(
        nu_log=jnp.asarray((-100.0, 100.0, 5.0), dtype=jnp.float32),
        theta_log=jnp.asarray((-100.0, 100.0, 3.0), dtype=jnp.float32),
    )
    inputs = jnp.asarray((0.2, -0.4), dtype=jnp.float32)
    state, sensitivities = rtu_step(
        params,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, config.encoder_width),
        inputs,
        epsilon=config.rtu_epsilon,
    )

    def output(candidate: RTUParameters) -> Array:
        candidate_state = rtu_forward(
            candidate,
            zero_rtu_state(config.hidden_size),
            inputs,
            epsilon=config.rtu_epsilon,
        )
        return jnp.concatenate((candidate_state.real, candidate_state.imaginary))

    autodiff_jacobian = jax.jacrev(output)(params)
    _assert_tree_finite((state, sensitivities, autodiff_jacobian))
    _assert_tree_all_close(
        _expanded_sensitivity(sensitivities),
        autodiff_jacobian,
        atol=2e-6,
        rtol=2e-6,
    )

    def scan_step(
        carry: tuple[RTUState, RTUSensitivities],
        _: None,
    ) -> tuple[tuple[RTUState, RTUSensitivities], None]:
        next_carry = rtu_step(
            params,
            carry[0],
            carry[1],
            inputs,
            epsilon=config.rtu_epsilon,
        )
        return next_carry, None

    final_carry, _ = jax.lax.scan(
        scan_step,
        (
            zero_rtu_state(config.hidden_size),
            zero_rtu_sensitivities(
                config.hidden_size,
                config.encoder_width,
            ),
        ),
        None,
        length=4096,
    )
    _assert_tree_finite(final_carry)


def test_agent_projects_adversarial_logs_after_update() -> None:
    config = _small_config(actor_alpha=0.0, critic_alpha=0.0)
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(26))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.4, -0.2), dtype=jnp.float32),
    )

    def adversarial_network(params: Any, value: float) -> Any:
        return params._replace(
            rtu=params.rtu._replace(
                nu_log=jnp.full_like(params.rtu.nu_log, value),
                theta_log=jnp.full_like(params.rtu.theta_log, value),
            )
        )

    state = state.replace(
        actor_params=adversarial_network(state.actor_params, 100.0),
        critic_params=adversarial_network(state.critic_params, -100.0),
    )
    result = agent.update(
        state,
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.6), dtype=jnp.float32),
    )

    _assert_tree_finite(result)
    assert jnp.all(result.state.actor_params.rtu.nu_log < 10.0)
    assert jnp.all(result.state.actor_params.rtu.theta_log < 10.0)
    assert jnp.all(result.state.critic_params.rtu.nu_log > -90.0)
    assert jnp.all(result.state.critic_params.rtu.theta_log > -90.0)


def test_encoder_gradient_matches_explicit_one_step_reference() -> None:
    """The sparse encoder gets current-step credit, not an implicit fake trace."""
    config = _small_config(encoder_width=4, output_width=5)
    params = initialize_rtu_network_parameters(
        jr.key(20),
        input_dim=2,
        output_dim=config.n_actions,
        config=config,
    )
    history = jnp.asarray(
        ((0.2, -0.1), (0.4, 0.6), (-0.3, 0.5)),
        dtype=jnp.float32,
    )
    previous_state = zero_rtu_state(config.hidden_size)
    sensitivities = zero_rtu_sensitivities(
        config.hidden_size,
        config.encoder_width,
    )
    for observation in history:
        encoded = rtu_network_encode(
            params,
            observation,
            layer_norm_epsilon=config.layer_norm_epsilon,
            negative_slope=config.leaky_relu_slope,
        )
        previous_state, sensitivities = rtu_step(
            params.rtu,
            previous_state,
            sensitivities,
            encoded,
            epsilon=config.rtu_epsilon,
        )

    current_observation = jnp.asarray((0.7, -0.4), dtype=jnp.float32)
    current_encoded = rtu_network_encode(
        params,
        current_observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    current_state, current_sensitivities = rtu_step(
        params.rtu,
        previous_state,
        sensitivities,
        current_encoded,
        epsilon=config.rtu_epsilon,
    )

    def objective(logits: Array) -> Array:
        return jax.nn.log_softmax(logits)[2]

    _, gradient = exact_rtrl_gradient(
        params,
        current_state,
        current_sensitivities,
        objective,
        encoder_input=current_observation,
        rtu_epsilon=config.rtu_epsilon,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )

    def local_objective(
        encoder_weights: Array,
        encoder_bias: Array,
    ) -> Array:
        candidate = params._replace(
            encoder_weights=encoder_weights,
            encoder_bias=encoder_bias,
        )
        encoded = rtu_network_encode(
            candidate,
            current_observation,
            layer_norm_epsilon=config.layer_norm_epsilon,
            negative_slope=config.leaky_relu_slope,
        )
        one_step_state = rtu_forward(
            params.rtu,
            jax.lax.stop_gradient(previous_state),
            encoded,
            epsilon=config.rtu_epsilon,
        )
        return objective(
            rtu_network_output(
                params,
                one_step_state,
                layer_norm_epsilon=config.layer_norm_epsilon,
                negative_slope=config.leaky_relu_slope,
            )
        )

    expected_weights, expected_bias = jax.grad(
        local_objective,
        argnums=(0, 1),
    )(
        params.encoder_weights,
        params.encoder_bias,
    )
    assert not jnp.allclose(expected_weights, 0.0)
    assert jnp.allclose(
        gradient.encoder_weights,
        expected_weights,
        atol=3e-6,
        rtol=3e-6,
    )
    assert jnp.allclose(
        gradient.encoder_bias,
        expected_bias,
        atol=3e-6,
        rtol=3e-6,
    )


def test_obgd_large_error_contract_applies_signal_once() -> None:
    traces = {
        "first": jnp.asarray((2.0, -1.0), dtype=jnp.float32),
        "second": jnp.asarray((3.0,), dtype=jnp.float32),
    }
    signal = jnp.asarray(10.0, dtype=jnp.float32)
    result = obgd_update(traces, signal, alpha=0.5, kappa=2.0)

    z_sum = jnp.asarray(6.0, dtype=jnp.float32)
    expected_scale = 1.0 / (0.5 * 2.0 * signal * z_sum)
    expected_step_size = 0.5 * expected_scale
    expected_updates = jax.tree_util.tree_map(
        lambda trace: expected_step_size * signal * trace,
        traces,
    )

    assert jnp.allclose(result.scale, expected_scale)
    assert jnp.allclose(result.step_size, expected_step_size)
    _assert_tree_all_close(result.updates, expected_updates)
    update_l1 = sum(
        (jnp.sum(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(result.updates)),
        start=jnp.asarray(0.0),
    )
    # In the bounded regime, a second signal multiplication would make this 5.
    assert jnp.allclose(update_l1, 1.0 / 2.0)


def test_obgd_infinite_signal_zeros_bounded_update() -> None:
    """Inf signal zeros the ObGD scale, then scale*signal*z is 0*inf=NaN."""
    traces = {
        "first": jnp.asarray((2.0, -1.0), dtype=jnp.float32),
        "second": jnp.asarray((3.0,), dtype=jnp.float32),
    }
    result = obgd_update(traces, jnp.asarray(jnp.inf, dtype=jnp.float32), alpha=0.5, kappa=2.0)
    for leaf in jax.tree_util.tree_leaves(result.updates):
        assert bool(jnp.all(jnp.isfinite(leaf)))
        assert bool(jnp.allclose(leaf, 0.0))
    assert bool(jnp.isfinite(result.scale))
    assert bool(jnp.allclose(result.scale, 0.0))
    assert not bool(result.update_applied)

    unbounded = obgd_update(traces, jnp.asarray(jnp.inf, dtype=jnp.float32), alpha=0.5, kappa=0.0)
    for leaf in jax.tree_util.tree_leaves(unbounded.updates):
        assert bool(jnp.all(jnp.isfinite(leaf)))
        assert bool(jnp.allclose(leaf, 0.0))
    assert not bool(unbounded.update_applied)


def test_terminated_does_not_multiply_inf_discounted_return() -> None:
    """A terminal 0 * inf return tracker is NaN and would freeze reward stats."""
    config = _small_config(
        actor_alpha=0.0,
        critic_alpha=0.0,
        normalize_rewards=False,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(0))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.5, -0.25), dtype=jnp.float32),
    )
    state = state.replace(
        reward_statistics=state.reward_statistics._replace(
            discounted_return=jnp.asarray(jnp.inf, dtype=jnp.float32)
        )
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((-0.3, 0.7), dtype=jnp.float32),
        terminated=jnp.asarray(True),
    )
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.state.reward_statistics.discounted_return))
    assert float(result.state.reward_statistics.discounted_return) == 0.0


def test_adaptive_obgd_infinite_signal_keeps_finite_updates() -> None:
    traces = {
        "first": jnp.asarray((2.0, -1.0), dtype=jnp.float32),
        "second": jnp.asarray((0.5,), dtype=jnp.float32),
    }
    second_moment = {
        "first": jnp.asarray((0.25, 0.5), dtype=jnp.float32),
        "second": jnp.asarray((1.0,), dtype=jnp.float32),
    }
    result = adaptive_obgd_update(
        traces,
        second_moment,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        alpha=0.2,
        kappa=2.0,
        beta2=0.9,
        epsilon=1e-4,
        step=4,
    )
    for leaf in jax.tree_util.tree_leaves(result.updates):
        assert bool(jnp.all(jnp.isfinite(leaf)))
        assert bool(jnp.allclose(leaf, 0.0))
    assert not bool(result.update_applied)
    chex.assert_trees_all_close(result.second_moment, second_moment)
    zero_traces = jax.tree_util.tree_map(jnp.zeros_like, traces)
    zeroed = adaptive_obgd_update(
        zero_traces,
        second_moment,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        alpha=0.2,
        kappa=2.0,
        beta2=0.9,
        epsilon=1e-4,
        step=4,
    )
    for leaf in jax.tree_util.tree_leaves(zeroed.second_moment):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    for leaf in jax.tree_util.tree_leaves(zeroed.updates):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert not bool(zeroed.update_applied)
    chex.assert_trees_all_close(zeroed.second_moment, second_moment)


def test_adaptive_obgd_zero_beta2_does_not_multiply_inf_moments() -> None:
    """beta2=0 drops leftover v; 0 * inf must not freeze the update."""
    traces = {
        "first": jnp.asarray((2.0, -1.0), dtype=jnp.float32),
        "second": jnp.asarray((0.5,), dtype=jnp.float32),
    }
    second_moment = {
        "first": jnp.full((2,), jnp.inf, dtype=jnp.float32),
        "second": jnp.full((1,), jnp.inf, dtype=jnp.float32),
    }
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    signal = jnp.asarray(-3.0, dtype=jnp.float32)
    result = adaptive_obgd_update(
        traces,
        second_moment,
        signal,
        alpha=0.2,
        kappa=2.0,
        beta2=0.0,
        epsilon=1e-4,
        step=4,
    )
    assert bool(result.update_applied)
    expected = jax.tree_util.tree_map(
        lambda trace: jnp.square(signal * trace),
        traces,
    )
    chex.assert_trees_all_close(result.second_moment, expected)


def test_adaptive_obgd_matches_memorax_second_moment_formula() -> None:
    traces = {
        "first": jnp.asarray((2.0, -1.0), dtype=jnp.float32),
        "second": jnp.asarray((0.5,), dtype=jnp.float32),
    }
    second_moment = {
        "first": jnp.asarray((0.25, 0.5), dtype=jnp.float32),
        "second": jnp.asarray((1.0,), dtype=jnp.float32),
    }
    signal = jnp.asarray(-3.0, dtype=jnp.float32)
    alpha = 0.2
    kappa = 2.0
    beta2 = 0.9
    epsilon = 1e-4
    step = 4

    result = adaptive_obgd_update(
        traces,
        second_moment,
        signal,
        alpha=alpha,
        kappa=kappa,
        beta2=beta2,
        epsilon=epsilon,
        step=step,
    )
    expected_second_moment = jax.tree_util.tree_map(
        lambda previous, trace: beta2 * previous + (1.0 - beta2) * jnp.square(signal * trace),
        second_moment,
        traces,
    )
    expected_corrected = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta2**step),
        expected_second_moment,
    )
    expected_normalized = jax.tree_util.tree_map(
        lambda trace, corrected: trace / (jnp.sqrt(corrected) + epsilon),
        traces,
        expected_corrected,
    )
    expected_z_sum = sum(
        (jnp.sum(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(expected_normalized)),
        start=jnp.asarray(0.0, dtype=jnp.float32),
    )
    expected_denominator = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(signal), 1.0) * expected_z_sum * alpha * kappa,
    )
    expected_step_size = alpha / expected_denominator
    expected_updates = jax.tree_util.tree_map(
        lambda normalized: expected_step_size * signal * normalized,
        expected_normalized,
    )

    _assert_tree_all_close(result.second_moment, expected_second_moment)
    _assert_tree_all_close(result.updates, expected_updates)
    assert jnp.allclose(result.scale, 1.0 / expected_denominator)
    assert jnp.allclose(result.step_size, expected_step_size)


def test_start_and_update_keep_actor_and_critic_streams_separate() -> None:
    agent = RecurrentTraceActorCriticAgent(_small_config(actor_alpha=0.1, critic_alpha=0.2))
    state = agent.init(feature_dim=2, key=jr.key(3))
    initial_observation = jnp.asarray((1.0, -0.5), dtype=jnp.float32)
    state, action, policy = agent.start(state, initial_observation)

    assert state.actor_rtu_state is not state.critic_rtu_state
    assert state.actor_sensitivities is not state.critic_sensitivities
    assert state.actor_traces is not state.critic_traces
    assert action.shape == ()
    assert policy.shape == (agent.config.n_actions,)
    assert jnp.allclose(jnp.sum(policy), 1.0)

    actor_before = state.actor_params
    critic_before = state.critic_params
    next_observation = jnp.asarray((-0.25, 0.75), dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(1.25, dtype=jnp.float32),
        next_observation,
        jnp.asarray(False),
    )

    assert int(result.state.step_count) == 1
    assert int(result.state.last_action) in range(agent.config.n_actions)
    assert jnp.array_equal(result.state.last_observation, next_observation)
    assert not jnp.allclose(
        result.state.actor_params.head_weights,
        actor_before.head_weights,
    )
    assert not jnp.allclose(
        result.state.critic_params.head_weights,
        critic_before.head_weights,
    )
    assert 0.0 < float(result.entropy)
    assert float(result.entropy) <= float(jnp.log(agent.config.n_actions)) + 1e-5
    assert not any(
        "replay" in field_name or "buffer" in field_name for field_name in result.state._fields
    )
    _assert_tree_finite(
        (
            result.state.actor_params,
            result.state.critic_params,
            result.state.actor_rtu_state,
            result.state.critic_rtu_state,
            result.state.actor_sensitivities,
            result.state.critic_sensitivities,
            result.state.actor_traces,
            result.state.critic_traces,
            result.policy,
            result.td_error,
        )
    )


def test_update_requires_start_and_valid_transition_scalars() -> None:
    agent = RecurrentTraceActorCriticAgent(_small_config())
    state = agent.init(feature_dim=2, key=jr.key(31))
    reward = jnp.asarray(0.5, dtype=jnp.float32)
    observation = jnp.asarray((0.2, -0.1), dtype=jnp.float32)

    for invalid_observation in (
        jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        jnp.asarray((0.0, jnp.inf), dtype=jnp.float32),
    ):
        with pytest.raises(ValueError, match="observation.*finite"):
            agent.start(state, invalid_observation)
    with pytest.raises(RuntimeError, match="start must be called"):
        agent.update(state, reward, observation)

    state, _, _ = agent.start(state, observation)
    with pytest.raises(ValueError, match="reward must be finite"):
        agent.update(state, jnp.asarray(jnp.nan), observation)
    with pytest.raises(ValueError, match="observation.*finite"):
        agent.update(
            state,
            reward,
            jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="reset_observation.*finite"):
        agent.update(
            state,
            reward,
            observation,
            reset_observation=jnp.asarray((0.0, -jnp.inf), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="terminated must be scalar"):
        agent.update(
            state,
            reward,
            observation,
            terminated=jnp.asarray((False,)),
        )
    with pytest.raises(ValueError, match="terminated must have boolean"):
        agent.update(
            state,
            reward,
            observation,
            terminated=jnp.asarray(0),
        )
    for invalid_discount in (
        float("nan"),
        -0.01,
        1.01,
    ):
        with pytest.raises(ValueError, match=r"discount.*\[0, 1\]"):
            agent.update(
                state,
                reward,
                observation,
                discount=jnp.asarray(invalid_discount),
            )
    with pytest.raises(ValueError, match="discount must be scalar"):
        agent.update(
            state,
            reward,
            observation,
            discount=jnp.asarray((0.9,)),
        )
    with pytest.raises(ValueError, match="episode_boundary must have boolean"):
        agent.update(
            state,
            reward,
            observation,
            episode_boundary=jnp.asarray(1),
        )
    with pytest.raises(ValueError, match="reset_observation shape"):
        agent.update(
            state,
            reward,
            observation,
            terminated=jnp.asarray(True),
            reset_observation=jnp.asarray((1.0,)),
        )


def test_jitted_public_update_enforces_dynamic_lifecycle_and_discount_checks() -> None:
    """The complete public boundary must retain its checks under an outer JIT."""
    agent = RecurrentTraceActorCriticAgent(_small_config())
    unstarted = agent.init(feature_dim=2, key=jr.key(311))
    reward = jnp.asarray(0.5, dtype=jnp.float32)
    observation = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    compiled_update = jax.jit(agent.update)
    compiled_start = jax.jit(agent.start)

    with pytest.raises(jax.errors.JaxRuntimeError, match="observation.*finite"):
        compiled_start(
            unstarted,
            jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        )[1].block_until_ready()

    with pytest.raises(jax.errors.JaxRuntimeError, match="start must be called"):
        compiled_update(unstarted, reward, observation).td_error.block_until_ready()

    started, _, _ = agent.start(unstarted, observation)
    with pytest.raises(jax.errors.JaxRuntimeError, match="reward must be finite"):
        compiled_update(
            started,
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            observation,
        ).td_error.block_until_ready()
    with pytest.raises(jax.errors.JaxRuntimeError, match="observation.*finite"):
        compiled_update(
            started,
            reward,
            jnp.asarray((0.0, -jnp.inf), dtype=jnp.float32),
        ).td_error.block_until_ready()
    with pytest.raises(jax.errors.JaxRuntimeError, match="reset_observation.*finite"):
        compiled_update(
            started,
            reward,
            observation,
            reset_observation=jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        ).td_error.block_until_ready()
    for invalid_discount in (
        float("nan"),
        -0.01,
        1.01,
    ):
        with pytest.raises(
            jax.errors.JaxRuntimeError,
            match=r"discount.*\[0, 1\]",
        ):
            compiled_update(
                started,
                reward,
                observation,
                discount=jnp.asarray(invalid_discount, dtype=jnp.float32),
            ).td_error.block_until_ready()

    valid = compiled_update(
        started,
        reward,
        observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    _assert_tree_finite(valid)
    assert int(valid.state.step_count) == 1


def test_started_state_update_omits_dynamic_callbacks_under_proved_contracts() -> None:
    """A scan-proven lifecycle and finite inputs need no CPU callback device."""
    agent = RecurrentTraceActorCriticAgent(_small_config())
    unstarted = agent.init(feature_dim=2, key=jr.key(312))
    reward = jnp.asarray(0.5, dtype=jnp.float32)
    observation = jnp.asarray((0.2, -0.1), dtype=jnp.float32)

    with pytest.raises(RuntimeError, match="start must be called"):
        agent.update_from_started_state(unstarted, reward, observation)

    started, _, _ = agent.start(unstarted, observation)
    with pytest.raises(ValueError, match="reward must be finite"):
        agent.update_from_started_state(
            started,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
            observation,
        )
    checked = agent.update(started, reward, observation)
    scan_safe = agent.update_from_started_state(started, reward, observation)
    _assert_tree_all_close(scan_safe, checked)

    jaxpr = jax.make_jaxpr(agent.update_from_started_state)(
        started,
        reward,
        observation,
    )
    assert "debug_callback" not in str(jaxpr)

    compiled = jax.jit(agent.update_from_started_state)(
        started,
        reward,
        observation,
    )
    _assert_tree_finite(compiled)
    assert int(compiled.state.step_count) == 1


def test_running_sample_counts_remain_exact_past_float32_integer_limit() -> None:
    agent = RecurrentTraceActorCriticAgent(_small_config(actor_alpha=0.0, critic_alpha=0.0))
    state = agent.init(feature_dim=2, key=jr.key(32))
    large_count = jnp.asarray(2**24, dtype=jnp.int32)
    state = state.replace(
        observation_statistics=state.observation_statistics._replace(
            sample_count=large_count,
        ),
        reward_statistics=state.reward_statistics._replace(
            sample_count=large_count,
        ),
    )
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.1, 0.2), dtype=jnp.float32),
    )
    assert state.observation_statistics.sample_count.dtype == jnp.int32
    assert int(state.observation_statistics.sample_count) == 2**24 + 1

    result = agent.update(
        state,
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray((-0.3, 0.4), dtype=jnp.float32),
    )
    assert result.state.reward_statistics.sample_count.dtype == jnp.int32
    assert int(result.state.reward_statistics.sample_count) == 2**24 + 1
    assert int(result.state.observation_statistics.sample_count) == 2**24 + 2


def test_counters_and_welford_moments_saturate_without_wrap_eager_or_jit() -> None:
    config = _small_config(
        actor_alpha=0.0,
        critic_alpha=0.0,
        normalize_observations=True,
        normalize_rewards=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(321))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    near_limit = jnp.asarray(2**31 - 2, dtype=jnp.int32)
    state = state.replace(
        observation_statistics=state.observation_statistics._replace(
            sample_count=near_limit,
            mean=jnp.zeros((2,), dtype=jnp.float32),
            m2=jnp.ones((2,), dtype=jnp.float32),
        ),
        reward_statistics=state.reward_statistics._replace(
            sample_count=near_limit,
            mean=jnp.asarray(0.0, dtype=jnp.float32),
            m2=jnp.asarray(1.0, dtype=jnp.float32),
            discounted_return=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        step_count=near_limit,
    )
    reward = jnp.asarray(0.75, dtype=jnp.float32)
    observation = jnp.asarray((1.5, -2.0), dtype=jnp.float32)
    discount = jnp.asarray(config.gamma, dtype=jnp.float32)

    eager = agent.update(
        state,
        reward,
        observation,
        discount=discount,
    )
    compiled = jax.jit(agent.update)(
        state,
        reward,
        observation,
        discount=discount,
    )
    _assert_tree_all_close(eager, compiled)

    maximum = 2**31 - 1
    for result in (eager, compiled):
        assert int(result.state.observation_statistics.sample_count) == maximum
        assert int(result.state.reward_statistics.sample_count) == maximum
        assert int(result.state.step_count) == maximum
        assert jnp.all(result.state.observation_statistics.m2 >= state.observation_statistics.m2)
        assert result.state.reward_statistics.m2 >= state.reward_statistics.m2
        _assert_tree_finite(result)

    saturated_eager = agent.update(
        eager.state,
        jnp.asarray(-0.25, dtype=jnp.float32),
        jnp.asarray((-4.0, 3.0), dtype=jnp.float32),
        discount=discount,
    )
    saturated_compiled = jax.jit(agent.update)(
        compiled.state,
        jnp.asarray(-0.25, dtype=jnp.float32),
        jnp.asarray((-4.0, 3.0), dtype=jnp.float32),
        discount=discount,
    )
    _assert_tree_all_close(saturated_eager, saturated_compiled)

    for result, prior in (
        (saturated_eager, eager),
        (saturated_compiled, compiled),
    ):
        assert int(result.state.observation_statistics.sample_count) == maximum
        assert int(result.state.reward_statistics.sample_count) == maximum
        assert int(result.state.step_count) == maximum
        assert jnp.array_equal(
            result.state.observation_statistics.mean,
            prior.state.observation_statistics.mean,
        )
        assert jnp.array_equal(
            result.state.observation_statistics.m2,
            prior.state.observation_statistics.m2,
        )
        assert jnp.array_equal(
            result.state.reward_statistics.mean,
            prior.state.reward_statistics.mean,
        )
        assert jnp.array_equal(
            result.state.reward_statistics.m2,
            prior.state.reward_statistics.m2,
        )
        _assert_tree_finite(result)


def test_action_sampling_uses_tempered_logits_directly() -> None:
    config = _small_config(temperature=0.37)
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(33))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.2, -0.4), dtype=jnp.float32),
    )
    logits = jnp.asarray((-30.0, 0.25, 4.0), dtype=jnp.float32)
    actor_params = state.actor_params._replace(
        head_weights=jnp.zeros_like(state.actor_params.head_weights),
        head_bias=logits,
    )
    sampling_key = jr.key(34)
    state = state.replace(actor_params=actor_params, rng_key=sampling_key)
    action, next_key, probabilities = agent.select_action(state)
    expected_key, sample_key = jr.split(sampling_key)
    expected_action = jr.categorical(
        sample_key,
        logits / config.temperature,
    ).astype(jnp.int32)

    assert jnp.array_equal(action, expected_action)
    assert jnp.array_equal(jr.key_data(next_key), jr.key_data(expected_key))
    assert jnp.allclose(
        probabilities,
        jax.nn.softmax(logits / config.temperature),
    )


def test_entropy_diagnostic_uses_objective_log_softmax_path() -> None:
    config = _small_config(
        actor_alpha=0.0,
        critic_alpha=0.0,
        temperature=1.0,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(340))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.2, -0.4), dtype=jnp.float32),
    )
    logits = jnp.asarray((0.0, -20.0, -40.0), dtype=jnp.float32)
    state = state.replace(
        actor_params=state.actor_params._replace(
            head_weights=jnp.zeros_like(state.actor_params.head_weights),
            head_bias=logits,
        )
    )
    result = agent.update(
        state,
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.3), dtype=jnp.float32),
    )
    expected_log_policy = jax.nn.log_softmax(logits / config.temperature)
    expected_entropy = -jnp.sum(jnp.exp(expected_log_policy) * expected_log_policy)

    assert bool(jnp.isfinite(result.entropy))
    assert jnp.allclose(result.entropy, expected_entropy, atol=0.0, rtol=1e-6)


def test_one_transition_parameter_delta_matches_stream_ac_formula() -> None:
    config = _small_config(
        gamma=0.91,
        actor_lamda=0.73,
        critic_lamda=0.67,
        actor_alpha=1e-6,
        critic_alpha=2e-6,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(35))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.35, -0.2), dtype=jnp.float32),
    )
    reward = jnp.asarray(0.6, dtype=jnp.float32)
    observation = jnp.asarray((-0.1, 0.45), dtype=jnp.float32)

    critic_inputs = rtu_network_encode(
        state.critic_params,
        observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    bootstrap_state = rtu_forward(
        state.critic_params.rtu,
        state.critic_rtu_state,
        critic_inputs,
        epsilon=config.rtu_epsilon,
    )
    next_value = rtu_network_output(
        state.critic_params,
        bootstrap_state,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )[0]
    value, critic_gradient = agent._critic_gradient(state)
    td_error = reward + config.gamma * next_value - value
    _, actor_gradient = agent._actor_gradient(state, td_error)
    actor_traces = jax.tree_util.tree_map(
        lambda trace, gradient: config.gamma * config.actor_lamda * trace + gradient,
        state.actor_traces,
        actor_gradient,
    )
    critic_traces = jax.tree_util.tree_map(
        lambda trace, gradient: config.gamma * config.critic_lamda * trace + gradient,
        state.critic_traces,
        critic_gradient,
    )
    actor_change = obgd_update(
        actor_traces,
        td_error,
        alpha=config.actor_alpha,
        kappa=config.actor_kappa,
    ).updates
    critic_change = obgd_update(
        critic_traces,
        td_error,
        alpha=config.critic_alpha,
        kappa=config.critic_kappa,
    ).updates
    expected_actor_params = jax.tree_util.tree_map(
        lambda parameter, change: parameter + change,
        state.actor_params,
        actor_change,
    )
    expected_critic_params = jax.tree_util.tree_map(
        lambda parameter, change: parameter + change,
        state.critic_params,
        critic_change,
    )

    result = agent.update(
        state,
        reward,
        observation,
        terminated=jnp.asarray(False),
    )
    assert jnp.allclose(result.next_value, next_value, atol=2e-6, rtol=2e-6)
    assert jnp.allclose(result.td_error, td_error, atol=2e-6, rtol=2e-6)
    _assert_tree_all_close(
        result.state.actor_params,
        expected_actor_params,
        atol=2e-6,
        rtol=2e-6,
    )
    _assert_tree_all_close(
        result.state.critic_params,
        expected_critic_params,
        atol=2e-6,
        rtol=2e-6,
    )
    assert result.state.actor_second_moments is None
    assert result.state.critic_second_moments is None


def test_adaptive_moments_persist_across_restart_and_checkpoint_resume() -> None:
    config = _small_config(
        adaptive_obgd=True,
        beta2=0.9,
        epsilon=1e-6,
        actor_alpha=1e-3,
        critic_alpha=2e-3,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(350))
    assert state.actor_second_moments is not None
    assert state.critic_second_moments is not None
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.35, -0.2), dtype=jnp.float32),
    )
    first = agent.update(
        state,
        jnp.asarray(0.8, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.45), dtype=jnp.float32),
    )
    assert first.state.actor_second_moments is not None
    assert first.state.critic_second_moments is not None
    actor_moment_l1 = sum(
        (
            jnp.sum(jnp.abs(leaf))
            for leaf in jax.tree_util.tree_leaves(first.state.actor_second_moments)
        ),
        start=jnp.asarray(0.0),
    )
    critic_moment_l1 = sum(
        (
            jnp.sum(jnp.abs(leaf))
            for leaf in jax.tree_util.tree_leaves(first.state.critic_second_moments)
        ),
        start=jnp.asarray(0.0),
    )
    assert float(actor_moment_l1) > 0.0
    assert float(critic_moment_l1) > 0.0

    restarted, _, _ = agent.start(
        first.state,
        jnp.asarray((0.2, 0.3), dtype=jnp.float32),
    )
    _assert_tree_all_close(
        restarted.actor_second_moments,
        first.state.actor_second_moments,
    )
    _assert_tree_all_close(
        restarted.critic_second_moments,
        first.state.critic_second_moments,
    )

    restored = pickle.loads(pickle.dumps(first.state))
    _assert_tree_all_close(restored, first.state)
    reward = jnp.asarray(-0.35, dtype=jnp.float32)
    observation = jnp.asarray((0.6, -0.7), dtype=jnp.float32)
    uninterrupted = agent.update(first.state, reward, observation)
    resumed = agent.update(restored, reward, observation)
    _assert_tree_all_close(resumed, uninterrupted, atol=2e-6, rtol=2e-6)


def test_default_state_keeps_historical_positional_checkpoint_prefix() -> None:
    agent = RecurrentTraceActorCriticAgent(_small_config())
    state = agent.init(feature_dim=2, key=jr.key(351))
    assert state._fields[-2:] == (
        "actor_second_moments",
        "critic_second_moments",
    )

    reconstructed = type(state)(*state[:-2])
    assert reconstructed.actor_second_moments is None
    assert reconstructed.critic_second_moments is None
    _assert_tree_all_close(reconstructed, state)


def test_terminal_transition_updates_then_resets_all_temporal_traces() -> None:
    config = _small_config(actor_alpha=0.05, critic_alpha=0.05)
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(4))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.5, -0.5), dtype=jnp.float32),
    )
    state = agent.update(
        state,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.asarray((0.1, 0.9), dtype=jnp.float32),
        jnp.asarray(False),
    ).state
    actor_params_before = state.actor_params

    reset_observation = jnp.asarray((-0.7, 0.2), dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        reset_observation,
        jnp.asarray(True),
    )
    expected_actor_inputs = rtu_network_encode(
        result.state.actor_params,
        reset_observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    expected_actor_state, expected_actor_sensitivity = rtu_step(
        result.state.actor_params.rtu,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, config.encoder_width),
        expected_actor_inputs,
        epsilon=config.rtu_epsilon,
    )

    _assert_tree_all_close(result.state.actor_rtu_state, expected_actor_state)
    _assert_tree_all_close(
        result.state.actor_sensitivities,
        expected_actor_sensitivity,
    )
    _assert_tree_all_close(
        result.state.actor_traces,
        jax.tree_util.tree_map(jnp.zeros_like, result.state.actor_traces),
    )
    _assert_tree_all_close(
        result.state.critic_traces,
        jax.tree_util.tree_map(jnp.zeros_like, result.state.critic_traces),
    )
    assert not jnp.allclose(
        result.state.actor_params.head_weights,
        actor_params_before.head_weights,
    )
    assert jnp.allclose(
        result.state.reward_statistics.discounted_return,
        0.0,
    )


def test_terminal_reward_uses_prior_eligibility_before_trace_reset() -> None:
    traced_config = _small_config(
        actor_lamda=0.8,
        critic_lamda=0.7,
        actor_alpha=1e-4,
        critic_alpha=1e-4,
    )
    traced_agent = RecurrentTraceActorCriticAgent(traced_config)
    zero_trace_agent = RecurrentTraceActorCriticAgent(
        _small_config(
            actor_lamda=0.0,
            critic_lamda=0.0,
            actor_alpha=1e-4,
            critic_alpha=1e-4,
        )
    )
    state = traced_agent.init(2, jr.key(21))
    state, _, _ = traced_agent.start(
        state,
        jnp.asarray((0.4, -0.6), dtype=jnp.float32),
    )
    state = traced_agent.update(
        state,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray((0.8, 0.1), dtype=jnp.float32),
        jnp.asarray(False),
    ).state

    terminal_observation = jnp.asarray((-0.2, 0.3), dtype=jnp.float32)
    traced_result = traced_agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        terminal_observation,
        jnp.asarray(True),
    )
    no_prior_trace_result = zero_trace_agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        terminal_observation,
        jnp.asarray(True),
    )
    assert not jnp.allclose(
        traced_result.state.actor_params.head_bias,
        no_prior_trace_result.state.actor_params.head_bias,
    )
    assert not jnp.allclose(
        traced_result.state.critic_params.head_bias,
        no_prior_trace_result.state.critic_params.head_bias,
    )
    _assert_tree_all_close(
        traced_result.state.actor_traces,
        jax.tree_util.tree_map(
            jnp.zeros_like,
            traced_result.state.actor_traces,
        ),
    )


def test_truncation_bootstraps_while_resetting_temporal_state() -> None:
    """Bootstrap discount and episode reset are independent transition facts."""
    config = _small_config(
        gamma=0.9,
        actor_lamda=0.8,
        critic_lamda=0.7,
        actor_alpha=0.0,
        critic_alpha=0.0,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(22))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.6, -0.4), dtype=jnp.float32),
    )
    state = agent.update(
        state,
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray((-0.1, 0.8), dtype=jnp.float32),
        jnp.asarray(False),
    ).state

    reward = jnp.asarray(0.75, dtype=jnp.float32)
    next_observation = jnp.asarray((0.3, -0.9), dtype=jnp.float32)
    reset_observation = jnp.asarray((-0.8, -0.2), dtype=jnp.float32)
    terminated = agent.update(
        state,
        reward,
        next_observation,
        terminated=jnp.asarray(True),
        reset_observation=reset_observation,
    )
    truncated = agent.update(
        state,
        reward,
        next_observation,
        discount=jnp.asarray(config.gamma, dtype=jnp.float32),
        episode_boundary=jnp.asarray(True),
        reset_observation=reset_observation,
    )
    continuing = agent.update(
        state,
        reward,
        next_observation,
        discount=jnp.asarray(config.gamma, dtype=jnp.float32),
    )

    _assert_tree_all_close(truncated.next_value, terminated.next_value)
    _assert_tree_all_close(truncated.next_value, continuing.next_value)
    assert jnp.allclose(
        truncated.td_error,
        terminated.td_error + config.gamma * truncated.next_value,
        atol=2e-6,
        rtol=2e-6,
    )
    _assert_tree_all_close(
        truncated.state.actor_traces,
        jax.tree_util.tree_map(jnp.zeros_like, truncated.state.actor_traces),
    )
    _assert_tree_all_close(
        truncated.state.critic_traces,
        jax.tree_util.tree_map(jnp.zeros_like, truncated.state.critic_traces),
    )
    _assert_tree_all_close(
        truncated.state.actor_rtu_state,
        terminated.state.actor_rtu_state,
    )
    _assert_tree_all_close(
        truncated.state.actor_sensitivities,
        terminated.state.actor_sensitivities,
    )
    assert jnp.array_equal(
        truncated.state.last_observation,
        reset_observation,
    )
    expected_actor_inputs = rtu_network_encode(
        truncated.state.actor_params,
        reset_observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    expected_actor_state, expected_actor_sensitivities = rtu_step(
        truncated.state.actor_params.rtu,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, config.encoder_width),
        expected_actor_inputs,
        epsilon=config.rtu_epsilon,
    )
    _assert_tree_all_close(
        truncated.state.actor_rtu_state,
        expected_actor_state,
    )
    _assert_tree_all_close(
        truncated.state.actor_sensitivities,
        expected_actor_sensitivities,
    )
    _assert_tree_all_close(
        truncated.policy,
        agent.policy(truncated.state),
    )
    assert jnp.allclose(
        truncated.state.reward_statistics.discounted_return,
        0.0,
    )

    continuing_trace_l1 = sum(
        (
            jnp.sum(jnp.abs(leaf))
            for leaf in jax.tree_util.tree_leaves(continuing.state.actor_traces)
        ),
        start=jnp.asarray(0.0),
    )
    assert float(continuing_trace_l1) > 0.0
    assert not jnp.allclose(
        continuing.state.actor_sensitivities.b_real,
        truncated.state.actor_sensitivities.b_real,
    )
    assert not jnp.allclose(
        continuing.state.actor_rtu_state.real,
        truncated.state.actor_rtu_state.real,
    )
    assert not jnp.allclose(
        continuing.state.reward_statistics.discounted_return,
        0.0,
    )


def test_gamma_zero_default_and_terminated_false_remain_continuing() -> None:
    config = _small_config(
        gamma=0.0,
        actor_alpha=0.0,
        critic_alpha=0.0,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(23))
    state, _, _ = agent.start(
        state,
        jnp.asarray((0.5, -0.25), dtype=jnp.float32),
    )
    reward = jnp.asarray(0.4, dtype=jnp.float32)
    observation = jnp.asarray((-0.3, 0.7), dtype=jnp.float32)

    default = agent.update(state, reward, observation)
    explicit_continuing = agent.update(
        state,
        reward,
        observation,
        terminated=jnp.asarray(False),
    )
    legacy_explicit_zero = agent.update(
        state,
        reward,
        observation,
        discount=jnp.asarray(0.0, dtype=jnp.float32),
    )

    _assert_tree_all_close(default, explicit_continuing)
    default_trace_l1 = sum(
        (jnp.sum(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(default.state.actor_traces)),
        start=jnp.asarray(0.0),
    )
    assert float(default_trace_l1) > 0.0
    assert jnp.allclose(
        default.state.reward_statistics.discounted_return,
        reward,
    )
    _assert_tree_all_close(
        legacy_explicit_zero.state.actor_traces,
        jax.tree_util.tree_map(
            jnp.zeros_like,
            legacy_explicit_zero.state.actor_traces,
        ),
    )
    assert jnp.allclose(
        legacy_explicit_zero.state.reward_statistics.discounted_return,
        0.0,
    )


def test_boundary_reset_observation_is_second_normalization_sample() -> None:
    config = _small_config(
        actor_alpha=0.0,
        critic_alpha=0.0,
        normalize_observations=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(2, jr.key(24))
    state, _, _ = agent.start(
        state,
        jnp.asarray((1.0, -1.0), dtype=jnp.float32),
    )
    count_before = state.observation_statistics.sample_count
    final_observation = jnp.asarray((2.0, 0.5), dtype=jnp.float32)
    reset_observation = jnp.asarray((-3.0, 4.0), dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        final_observation,
        terminated=jnp.asarray(True),
        reset_observation=reset_observation,
    )

    assert jnp.array_equal(
        result.state.observation_statistics.sample_count,
        count_before + 2,
    )
    assert jnp.array_equal(
        result.state.last_observation,
        reset_observation,
    )


def test_normalization_matches_literal_memorax_wrapper_recurrences() -> None:
    """Normalization is canonical, including current-sample and return ordering."""
    config = _small_config(
        gamma=0.9,
        actor_alpha=0.0,
        critic_alpha=0.0,
        normalize_observations=True,
        normalize_rewards=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    state = agent.init(feature_dim=2, key=jr.key(5))

    first_observation = jnp.asarray((2.0, -1.0), dtype=jnp.float32)
    expected_obs_count = jnp.asarray(2.0, dtype=jnp.float32)
    expected_obs_mean = first_observation / expected_obs_count
    expected_obs_m2 = 1.0 + first_observation * (first_observation - expected_obs_mean)
    expected_normalized_observation = (first_observation - expected_obs_mean) / jnp.sqrt(
        expected_obs_m2 / expected_obs_count + config.normalization_epsilon
    )
    expected_actor_inputs = rtu_network_encode(
        state.actor_params,
        expected_normalized_observation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        negative_slope=config.leaky_relu_slope,
    )
    expected_actor_state, _ = rtu_step(
        state.actor_params.rtu,
        zero_rtu_state(config.hidden_size),
        zero_rtu_sensitivities(config.hidden_size, config.encoder_width),
        expected_actor_inputs,
        epsilon=config.rtu_epsilon,
    )

    state, _, _ = agent.start(state, first_observation)
    assert jnp.allclose(
        state.observation_statistics.sample_count,
        expected_obs_count,
    )
    assert jnp.allclose(state.observation_statistics.mean, expected_obs_mean)
    assert jnp.allclose(state.observation_statistics.m2, expected_obs_m2)
    _assert_tree_all_close(state.actor_rtu_state, expected_actor_state)

    first_reward = jnp.asarray(2.0, dtype=jnp.float32)
    expected_return = first_reward
    expected_reward_count = jnp.asarray(2.0, dtype=jnp.float32)
    expected_reward_mean = expected_return / expected_reward_count
    expected_reward_m2 = 1.0 + expected_return * (expected_return - expected_reward_mean)
    expected_scaled_reward = first_reward / jnp.sqrt(
        expected_reward_m2 / expected_reward_count + config.normalization_epsilon
    )
    first_result = agent.update(
        state,
        first_reward,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray(False),
    )
    assert jnp.allclose(first_result.normalized_reward, expected_scaled_reward)
    assert jnp.allclose(
        first_result.state.reward_statistics.discounted_return,
        expected_return,
    )

    terminal_reward = jnp.asarray(-1.0, dtype=jnp.float32)
    terminal_return = terminal_reward
    terminal_count = expected_reward_count + 1.0
    terminal_delta = terminal_return - expected_reward_mean
    terminal_mean = expected_reward_mean + terminal_delta / terminal_count
    terminal_m2 = expected_reward_m2 + terminal_delta * (terminal_return - terminal_mean)
    terminal_scaled_reward = terminal_reward / jnp.sqrt(
        terminal_m2 / terminal_count + config.normalization_epsilon
    )
    terminal_result = agent.update(
        first_result.state,
        terminal_reward,
        jnp.asarray((-0.25, 0.75), dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert jnp.allclose(
        terminal_result.normalized_reward,
        terminal_scaled_reward,
    )
    assert jnp.allclose(
        terminal_result.state.reward_statistics.discounted_return,
        0.0,
    )


def test_jit_vmap_update_is_deterministic_and_finite() -> None:
    config = _small_config(
        hidden_size=2,
        actor_alpha=0.05,
        critic_alpha=0.05,
        normalize_observations=True,
        normalize_rewards=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    keys = jr.split(jr.key(6), 2)
    initial_states = jax.vmap(lambda key: agent.init(3, key))(keys)
    initial_observations = jnp.asarray(
        ((1.0, 0.0, -0.5), (-0.25, 0.75, 0.5)),
        dtype=jnp.float32,
    )
    batched_start = jax.jit(jax.vmap(agent.start))
    started_a, actions_a, policies_a = batched_start(
        initial_states,
        initial_observations,
    )
    started_b, actions_b, policies_b = batched_start(
        initial_states,
        initial_observations,
    )
    _assert_tree_all_close(started_a, started_b)
    _assert_tree_all_close(actions_a, actions_b)
    _assert_tree_all_close(policies_a, policies_b)

    rewards = jnp.asarray((1.0, -0.5), dtype=jnp.float32)
    next_observations = jnp.asarray(
        ((0.0, 1.0, 0.25), (0.5, -0.5, 1.0)),
        dtype=jnp.float32,
    )
    terminated = jnp.asarray((False, True))
    batched_update = jax.jit(jax.vmap(agent.update, in_axes=(0, 0, 0, 0)))
    result_a = batched_update(
        started_a,
        rewards,
        next_observations,
        terminated,
    )
    result_b = batched_update(
        started_b,
        rewards,
        next_observations,
        terminated,
    )
    _assert_tree_all_close(result_a, result_b)
    _assert_tree_finite(
        (
            result_a.state.actor_params,
            result_a.state.critic_params,
            result_a.state.actor_rtu_state,
            result_a.state.critic_rtu_state,
            result_a.state.actor_sensitivities,
            result_a.state.critic_sensitivities,
            result_a.policy,
            result_a.value,
            result_a.next_value,
            result_a.td_error,
        )
    )


def test_scan_has_no_future_leakage() -> None:
    config = _small_config(
        hidden_size=2,
        actor_alpha=0.03,
        critic_alpha=0.04,
        normalize_observations=True,
        normalize_rewards=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    initial_state = agent.init(2, jr.key(7))
    initial_state, _, _ = agent.start(
        initial_state,
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    shared_observations = jnp.asarray(
        ((0.3, 0.4), (-0.5, 0.7)),
        dtype=jnp.float32,
    )
    observations_a = jnp.concatenate(
        (
            shared_observations,
            jnp.asarray(((0.9, -0.8),), dtype=jnp.float32),
        )
    )
    observations_b = jnp.concatenate(
        (
            shared_observations,
            jnp.asarray(((-9.0, 8.0),), dtype=jnp.float32),
        )
    )
    rewards_a = jnp.asarray((0.5, -0.2, 1.0), dtype=jnp.float32)
    rewards_b = jnp.asarray((0.5, -0.2, -10.0), dtype=jnp.float32)

    def run(
        observations: Array,
        rewards: Array,
    ) -> tuple[Any, tuple[Any, Array, Array, Array]]:
        def scan_step(
            state: Any,
            sample: tuple[Array, Array],
        ) -> tuple[Any, tuple[Any, Array, Array, Array]]:
            observation, reward = sample
            result = agent.update(
                state,
                reward,
                observation,
                jnp.asarray(False),
            )
            return result.state, (
                result.state,
                result.action,
                result.policy,
                result.td_error,
            )

        return jax.lax.scan(
            scan_step,
            initial_state,
            (observations, rewards),
        )

    _, outputs_a = jax.jit(run)(observations_a, rewards_a)
    _, outputs_b = jax.jit(run)(observations_b, rewards_b)
    states_a, actions_a, policies_a, errors_a = outputs_a
    states_b, actions_b, policies_b, errors_b = outputs_b
    prefix_a = jax.tree_util.tree_map(lambda value: value[:2], states_a)
    prefix_b = jax.tree_util.tree_map(lambda value: value[:2], states_b)
    _assert_tree_all_close(prefix_a, prefix_b)
    _assert_tree_all_close(actions_a[:2], actions_b[:2])
    _assert_tree_all_close(policies_a[:2], policies_b[:2])
    _assert_tree_all_close(errors_a[:2], errors_b[:2])
    assert not jnp.allclose(errors_a[2], errors_b[2])


@pytest.mark.parametrize(
    "factory",
    (
        lambda: RecurrentTraceActorCriticConfig(n_actions=0),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, hidden_size=True),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, encoder_width=1),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, output_width=1),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, gamma=1.01),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, actor_alpha=-1.0),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, critic_kappa=0.0),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, sparsity=float("nan")),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, sparsity=1.0),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, r_min=0.5, r_max=0.5),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, r_max=1.01),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, rtu_epsilon=1.0),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, beta2=-0.01),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, beta2=1.0),
        lambda: RecurrentTraceActorCriticConfig(
            n_actions=2,
            beta2=float.fromhex("0x1.fffffffffffffp-1"),
        ),
        lambda: RecurrentTraceActorCriticConfig(n_actions=2, epsilon=0.0),
        lambda: RecurrentTraceActorCriticConfig(
            n_actions=2,
            normalize_rewards=cast(bool, 1),
        ),
        lambda: RecurrentTraceActorCriticConfig(
            n_actions=2,
            adaptive_obgd=cast(bool, 1),
        ),
    ),
)
def test_config_validation(factory: Callable[[], Any]) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "field_name",
    (
        "gamma",
        "actor_lamda",
        "critic_lamda",
        "actor_alpha",
        "critic_alpha",
        "actor_kappa",
        "critic_kappa",
        "entropy_coefficient",
        "temperature",
        "sparsity",
        "r_min",
        "r_max",
        "max_phase",
        "rtu_epsilon",
        "layer_norm_epsilon",
        "leaky_relu_slope",
        "normalization_epsilon",
        "beta2",
        "epsilon",
    ),
)
def test_config_rejects_bool_for_every_numeric_float(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="numeric, not bool"):
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            **{field_name: True},
        )


def test_config_and_agent_serialization_round_trip() -> None:
    config = _small_config(
        actor_lamda=0.65,
        critic_lamda=0.75,
        entropy_coefficient=0.07,
        normalize_observations=True,
        normalize_rewards=True,
    )
    agent = RecurrentTraceActorCriticAgent(config)
    serialized = agent.to_config()
    json.dumps(serialized)

    reconstructed = RecurrentTraceActorCriticAgent.from_config(serialized)
    direct_reconstruction = RecurrentTraceActorCriticAgent.from_config(config.to_config())
    assert reconstructed.config == config
    assert direct_reconstruction.config == config
    with pytest.raises(ValueError, match="unsupported agent type"):
        RecurrentTraceActorCriticAgent.from_config(
            {"type": "DifferentAgent", "config": config.to_config()}
        )

    adaptive = _small_config(
        adaptive_obgd=True,
        beta2=0.95,
        epsilon=1e-6,
    )
    adaptive_payload = adaptive.to_config()
    assert adaptive_payload["adaptive_obgd"] is True
    assert adaptive_payload["beta2"] == 0.95
    assert adaptive_payload["epsilon"] == 1e-6
    assert RecurrentTraceActorCriticConfig.from_config(adaptive_payload) == adaptive


def test_config_rejects_hostile_numeric_and_serialized_container_hooks() -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("float hook must not run")

        def __repr__(self) -> str:
            raise AssertionError("repr hook must not run")

    class DictSubclass(dict[str, Any]):
        pass

    with pytest.raises(ValueError, match="gamma"):
        _small_config(gamma=HostileFloat(0.9))
    payload = _small_config().to_config()
    with pytest.raises(ValueError, match="exact built-in dict"):
        RecurrentTraceActorCriticConfig.from_config(DictSubclass(payload))
    envelope = RecurrentTraceActorCriticAgent(_small_config()).to_config()
    with pytest.raises(ValueError, match="exact built-in dict"):
        RecurrentTraceActorCriticAgent.from_config(DictSubclass(envelope))
    with pytest.raises(ValueError, match="fields"):
        RecurrentTraceActorCriticConfig.from_config({**payload, "unknown": 1})


def test_state_resource_budget_is_exact_and_init_preflights_before_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    assert config.state_resource_budget(2) == {
        "parameter_scalars": 124,
        "sensitivity_scalars_per_network": 36,
        "float32_state_scalars": 343,
        "state_scalars": 350,
        "state_nbytes": 1397,
    }

    def forbidden_split(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("RNG split must not run before resource validation")

    monkeypatch.setattr(recurrent_trace_module.jr, "split", forbidden_split)
    with pytest.raises(ValueError, match="derived"):
        RecurrentTraceActorCriticAgent(config).init(2**31 - 1, jr.key(0))


def test_canonical_discrete_architecture_defaults_and_core_export() -> None:
    config = RecurrentTraceActorCriticConfig(n_actions=4)
    assert config.hidden_size == 192
    assert config.encoder_width == 64
    assert config.output_width == 64
    assert config.sparsity == 0.9
    assert config.entropy_coefficient == 0.095
    assert config.actor_alpha == config.critic_alpha == 1.0
    assert config.actor_kappa == 3.0
    assert config.critic_kappa == 2.0
    assert config.adaptive_obgd is False
    assert config.beta2 == 0.999
    assert config.epsilon == 1e-8
    canonical = json.dumps(
        config.to_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        "650ab3d912d67cb184da47eb871138233cdecfe0f89235017229e9ccb980fe77"
    )
    canonical_agent = json.dumps(
        RecurrentTraceActorCriticAgent(config).to_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical_agent).hexdigest() == (
        "02dd7cb7bf29883c6f1acc9f2fa4f5c6d8348d32dff0fbfbd16977799960a8c1"
    )
    assert not ({"adaptive_obgd", "beta2", "epsilon"} & config.to_config().keys())
    explicit_defaults = RecurrentTraceActorCriticConfig(
        n_actions=4,
        adaptive_obgd=False,
        beta2=0.999,
        epsilon=1e-8,
    )
    assert explicit_defaults.to_config() == config.to_config()
    assert ExportedRecurrentTraceActorCriticAgent is RecurrentTraceActorCriticAgent


def test_reference_dense_rtu_and_live_sparse_feedforward_initialization() -> None:
    config = RecurrentTraceActorCriticConfig(
        n_actions=2,
        hidden_size=4,
        encoder_width=20,
        output_width=8,
        sparsity=0.9,
    )
    initialization_key = jr.key(8)
    params = initialize_rtu_network_parameters(
        initialization_key,
        input_dim=20,
        output_dim=2,
        config=config,
    )
    assert jnp.all(jnp.sum(params.encoder_weights == 0.0, axis=1) == 18)
    assert jnp.all(jnp.sum(params.output_weights == 0.0, axis=1) == 7)
    assert jnp.all(jnp.sum(params.head_weights == 0.0, axis=1) == 7)

    _, rtu_key, _, _ = jr.split(initialization_key, 4)
    _, _, real_key, imaginary_key = jr.split(rtu_key, 4)
    reference_initializer = jax.nn.initializers.lecun_normal()
    assert jnp.array_equal(
        params.rtu.b_real,
        reference_initializer(
            real_key,
            (config.hidden_size, config.encoder_width),
            jnp.float32,
        ),
    )
    assert jnp.array_equal(
        params.rtu.b_imag,
        reference_initializer(
            imaginary_key,
            (config.hidden_size, config.encoder_width),
            jnp.float32,
        ),
    )
    assert jnp.all(params.rtu.b_real != 0.0)
    assert jnp.all(params.rtu.b_imag != 0.0)

    one_feature_params = initialize_rtu_network_parameters(
        jr.key(81),
        input_dim=1,
        output_dim=2,
        config=config,
    )
    assert jnp.all(one_feature_params.encoder_weights != 0.0)

    inputs = jnp.asarray((1.0, 2.0, -1.0, 0.5), dtype=jnp.float32)
    normalized = parameterless_layer_norm(inputs, epsilon=1e-8)
    assert jnp.allclose(jnp.mean(normalized), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.mean(jnp.square(normalized)), 1.0, atol=1e-6)


def test_initialized_radii_respect_small_configured_interval() -> None:
    config = RecurrentTraceActorCriticConfig(
        n_actions=2,
        hidden_size=512,
        encoder_width=2,
        output_width=2,
        sparsity=0.0,
        r_min=1e-7,
        r_max=1e-6,
    )
    params = initialize_rtu_network_parameters(
        jr.key(82),
        input_dim=2,
        output_dim=2,
        config=config,
    )
    recovered_radii = jnp.exp(-jnp.exp(params.rtu.nu_log))
    assert jnp.all(recovered_radii >= config.r_min * (1.0 - 2e-5))
    assert jnp.all(recovered_radii <= config.r_max * (1.0 + 2e-5))


def test_phase_initialization_matches_upstream_and_ignores_rtu_epsilon() -> None:
    """Numerical recurrence epsilon must not truncate the phase distribution."""
    config = RecurrentTraceActorCriticConfig(
        n_actions=2,
        hidden_size=256,
        encoder_width=2,
        output_width=2,
        sparsity=0.0,
        max_phase=6.0,
        rtu_epsilon=0.9,
    )
    initialization_key = jr.key(83)
    params = initialize_rtu_network_parameters(
        initialization_key,
        input_dim=2,
        output_dim=2,
        config=config,
    )
    repeated = initialize_rtu_network_parameters(
        initialization_key,
        input_dim=2,
        output_dim=2,
        config=config,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(repeated),
        strict=True,
    ):
        assert jnp.array_equal(actual, expected)

    _, rtu_key, _, _ = jr.split(initialization_key, 4)
    _, theta_key, _, _ = jr.split(rtu_key, 4)
    upstream_uniform = jr.uniform(
        theta_key,
        (config.hidden_size,),
        dtype=jnp.float32,
    )
    float32_tiny = jnp.asarray(
        jnp.finfo(jnp.float32).tiny,
        dtype=jnp.float32,
    )
    expected_phase = jnp.maximum(
        config.max_phase * upstream_uniform,
        float32_tiny,
    )
    expected_theta_log = jnp.log(expected_phase)
    assert jnp.array_equal(params.rtu.theta_log, expected_theta_log)

    recovered_phase = jnp.exp(params.rtu.theta_log)
    assert jnp.min(recovered_phase) < config.rtu_epsilon * config.max_phase


def test_rtac_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        RecurrentTraceActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        RecurrentTraceActorCriticConfig(n_actions=3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_size"):
        RecurrentTraceActorCriticConfig(n_actions=2, hidden_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="encoder_width"):
        RecurrentTraceActorCriticConfig(n_actions=2, encoder_width=1)


def test_rtac_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = RecurrentTraceActorCriticConfig(
        n_actions=np.int32(3),
        hidden_size=np.int64(64),
        encoder_width=np.int32(32),
        output_width=np.int64(32),
    )
    assert type(cfg.n_actions) is int
    assert type(cfg.hidden_size) is int
    assert type(cfg.encoder_width) is int
    assert type(cfg.output_width) is int
    assert cfg.n_actions == 3
    assert cfg.hidden_size == 64
    assert cfg.encoder_width == 32
    assert cfg.output_width == 32


@pytest.mark.parametrize(
    "field",
    ("gamma", "actor_lamda", "critic_lamda", "sparsity", "beta2"),
)
def test_float32_endpoints_check_exact_numpy_value(field: str) -> None:
    above_one = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    with pytest.raises(ValueError, match=field):
        RecurrentTraceActorCriticConfig(n_actions=2, **{field: above_one})


@pytest.mark.parametrize(
    "field",
    (
        "gamma",
        "actor_lamda",
        "critic_lamda",
        "actor_alpha",
        "critic_alpha",
        "entropy_coefficient",
        "sparsity",
        "r_min",
        "leaky_relu_slope",
        "beta2",
    ),
)
def test_nonzero_longdouble_cannot_underflow_into_legal_zero(field: str) -> None:
    nonzero = np.nextafter(np.longdouble(0.0), np.longdouble(1.0))
    with pytest.raises(ValueError, match=rf"{field}.*finite normal float32"):
        RecurrentTraceActorCriticConfig(n_actions=2, **{field: nonzero})


@pytest.mark.parametrize("code", ("e", "f", "d", "g"))
def test_config_canonicalizes_supported_numpy_float_families(code: str) -> None:
    value = np.dtype(code).type(0.5)
    config = RecurrentTraceActorCriticConfig(n_actions=2, gamma=value)
    assert type(config.gamma) is float
    assert config.gamma == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_actions", np.int64(2)),
        ("gamma", np.float32(0.5)),
        ("normalize_rewards", np.bool_(True)),
        ("gamma", 1),
        ("n_actions", 2.0),
    ),
)
def test_from_config_requires_exact_json_scalar_types(field: str, value: object) -> None:
    payload = RecurrentTraceActorCriticConfig(n_actions=2).to_config()
    payload[field] = value
    with pytest.raises(ValueError, match="exact JSON scalar types"):
        RecurrentTraceActorCriticConfig.from_config(payload)


@pytest.mark.parametrize("code", ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))
def test_init_accepts_supported_numpy_integer_feature_dimensions(code: str) -> None:
    config = RecurrentTraceActorCriticConfig(
        n_actions=2,
        hidden_size=2,
        encoder_width=2,
        output_width=2,
    )
    state = RecurrentTraceActorCriticAgent(config).init(np.dtype(code).type(3), jr.key(1))
    assert state.last_observation.shape == (3,)
