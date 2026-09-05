"""Coverage for the permanently nonpromoting noise-curvature comparator."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    noise_curvature_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    PAPER_REVISION,
    ControllerMode,
    NoiseCurvatureConfig,
    NoiseCurvatureState,
    init_noise_curvature_state,
    noise_curvature_learning_rate,
    noise_curvature_persistent_bytes,
    noise_curvature_safe_bound,
    noise_curvature_step,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    cross_entropy_loss,
    init_mlp_params,
)
from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    DEVELOPMENT_GATES,
    DEVELOPMENT_SEEDS,
    LIVE_CONTROL,
    OFFICIAL_CODE_STATUS,
    PROTOCOL_DIFFERENCES,
    matched_arm_names,
    registered_arms,
    validate_matched_noise_curvature_results,
    validate_noise_curvature_development_result,
)


def _tiny_params() -> dict[str, jax.Array]:
    return init_mlp_params(
        jax.random.key(3),
        IPMNISTConfig(n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2),
    )


def _tree_allclose(left: object, right: object) -> bool:
    return all(
        bool(jnp.allclose(a, b, rtol=1e-6, atol=1e-7))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_protocol_pins_v3_no_official_code_and_live_control() -> None:
    assert PAPER_REVISION == "arXiv:2509.19698v3"
    assert OFFICIAL_CODE_STATUS == "none-identified-as-of-2026-08-17"
    assert len(PROTOCOL_DIFFERENCES) == 11
    assert tuple(DEVELOPMENT_GATES) == (
        "integrity",
        "mechanism",
        "causal",
        "hillclimb",
        "transfer",
        "resources",
    )
    assert matched_arm_names() == (LIVE_CONTROL, *registered_arms())
    assert screening_spec(LIVE_CONTROL).name == LIVE_CONTROL
    modes = tuple(
        screening_spec(name).hyperparameters["controller_mode"] for name in registered_arms()
    )
    assert modes == (
        0.0,
        1.0,
        2.0,
        3.0,
    )


def test_joint_and_causal_bounds_reduce_exactly() -> None:
    def bound(mode: ControllerMode) -> jax.Array:
        return noise_curvature_safe_bound(
            mode=mode,
            batch_size=4,
            squared_gradient_mean=jnp.asarray(2.0),
            per_sample_gradient_variance=jnp.asarray(4.0),
            curvature_volatility=jnp.asarray(3.0),
            volatility_inflation=1.0,
            volatility_kappa=1.0,
            safety_factor=0.8,
        )

    gradient = bound("gradient_only")
    combined = bound("combined")
    volatility = bound("volatility_only")
    assert float(gradient) == pytest.approx(1.6)
    assert float(combined) == pytest.approx(0.64)
    assert float(volatility) == pytest.approx(0.8 / 3.0)
    assert float(combined) < float(gradient)


def test_fixed_scheduler_is_inert_and_cool_warm_decisions_are_causal() -> None:
    fixed = NoiseCurvatureConfig(mode="fixed", total_steps=40)
    lr = jnp.asarray(1e-3)
    unchanged, cooled, warmed = noise_curvature_learning_rate(
        lr,
        effective_step=jnp.asarray(1.0),
        safe_bound=jnp.asarray(0.1),
        is_early=jnp.asarray(True),
        mode="fixed",
        config=fixed,
    )
    assert float(unchanged) == pytest.approx(1e-3)
    assert not bool(cooled)
    assert not bool(warmed)
    combined = NoiseCurvatureConfig(mode="combined", total_steps=40)
    cool_lr, cooled, warmed = noise_curvature_learning_rate(
        lr,
        effective_step=jnp.asarray(0.2),
        safe_bound=jnp.asarray(0.1),
        is_early=jnp.asarray(True),
        mode="combined",
        config=combined,
    )
    assert float(cool_lr) == pytest.approx(0.00099)
    assert bool(cooled) and not bool(warmed)
    warm_lr, cooled, warmed = noise_curvature_learning_rate(
        lr,
        effective_step=jnp.asarray(0.001),
        safe_bound=jnp.asarray(1.0),
        is_early=jnp.asarray(True),
        mode="combined",
        config=combined,
    )
    assert float(warm_lr) == pytest.approx(0.00101)
    assert not bool(cooled) and bool(warmed)


def test_mechanism_off_matches_fixed_adamw_updates() -> None:
    params = _tiny_params()
    config = NoiseCurvatureConfig(mode="fixed", total_steps=2, control_interval=2)
    state = init_noise_curvature_state(params, config)
    optimizer = Adam(step_size=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-3)
    reference_states = {
        name: optimizer.init_for_shape(value.shape) for name, value in params.items()
    }
    reference_params = params
    for index in range(2):
        x = jnp.asarray([0.2, -0.1, 0.4, index * 0.1], dtype=jnp.float32)
        y = jnp.asarray(index % 2, dtype=jnp.int32)
        grads = jax.grad(lambda p: cross_entropy_loss(p, x, y)[0])(params)
        params, state = noise_curvature_step(params, state, grads, x, y, cross_entropy_loss, config)
        reference_grads = jax.grad(lambda p: cross_entropy_loss(p, x, y)[0])(reference_params)
        next_reference: dict[str, jax.Array] = {}
        next_states = {}
        for name in reference_params:
            update = optimizer.update_from_gradient_checked(
                reference_states[name],
                reference_grads[name],
                param=reference_params[name],
            )
            next_reference[name] = reference_params[name] - update.step
            next_states[name] = update.new_state
        reference_params = next_reference
        reference_states = next_states
    assert all(bool(jnp.array_equal(params[name], reference_params[name])) for name in params)
    assert jnp.array_equal(state.learning_rates, jnp.full((3,), 1e-3))
    assert int(state.controller_count) == 1


def test_scheduler_step_jit_matches_eager_through_controller_event() -> None:
    params = _tiny_params()
    config = NoiseCurvatureConfig(
        mode="combined", total_steps=2, control_interval=2, power_iterations=1
    )
    state = init_noise_curvature_state(params, config)

    def step(
        p: dict[str, jax.Array], s: NoiseCurvatureState, x: jax.Array, y: jax.Array
    ) -> tuple[dict[str, jax.Array], NoiseCurvatureState]:
        grads = jax.grad(lambda candidate: cross_entropy_loss(candidate, x, y)[0])(p)
        return noise_curvature_step(p, s, grads, x, y, cross_entropy_loss, config)

    eager_params, eager_state = params, state
    jit_params, jit_state = params, state
    compiled = jax.jit(step)
    for index in range(2):
        x = jnp.asarray([0.1, -0.2, 0.3, 0.1 * index], dtype=jnp.float32)
        y = jnp.asarray(index % 2, dtype=jnp.int32)
        eager_params, eager_state = step(eager_params, eager_state, x, y)
        jit_params, jit_state = compiled(jit_params, jit_state, x, y)
    assert _tree_allclose(eager_params, jit_params)
    assert _tree_allclose(eager_state, jit_state)
    assert int(eager_state.controller_count) == 1


def test_end_to_end_registered_arm_and_receipt_accounting() -> None:
    data_x = np.linspace(-1.0, 1.0, 160, dtype=np.float32).reshape(40, 4)
    data_y = np.asarray([index % 2 for index in range(40)], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )
    result = run_screening_config(
        data_x,
        data_y,
        screening_spec("noise_curvature_combined"),
        seed=0,
        config=config,
    )
    assert isinstance(result, ScreeningRunResult)
    assert np.isfinite(result.per_task_loss).all()
    payload = noise_curvature_development_result_payload(result, outcome="inconclusive")
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["development_seed_protocol"] == list(DEVELOPMENT_SEEDS)
    resources = payload["resources"]
    assert isinstance(resources, dict)
    assert resources["controller_events"] == 1
    assert resources["first_order_gradient_queries"] == 80
    assert resources["loss_only_queries"] == 40
    assert resources["hessian_vector_product_queries"] == 3
    assert resources["model_queries"] == 123
    assert resources["persistent_bytes"] == noise_curvature_persistent_bytes(
        parameter_count=config.parameter_count,
        input_dim=4,
        control_interval=40,
    )


@pytest.mark.parametrize(
    "arm_name",
    [
        "noise_curvature_fixed_adam_l2",
        "noise_curvature_gradient_only",
        "noise_curvature_volatility_only",
    ],
)
def test_every_registered_arm_runs_end_to_end_through_the_screening_harness(
    arm_name: str,
) -> None:
    """Each registered controller-mode arm must survive a real screening run.

    ``noise_curvature_combined`` is the only arm exercised end to end through
    ``run_screening_config`` elsewhere in this file. The other three modes
    (``fixed``, ``gradient_only``, ``volatility_only``) were previously
    reachable only through ``screening_spec``/config-construction assertions
    or through the isolated ``noise_curvature_safe_bound`` unit math — never
    through the real registered-arm screening entrypoint that a live
    development run would actually invoke for these Literal-typed
    ``controller_mode`` values.
    """
    data_x = np.linspace(-1.0, 1.0, 160, dtype=np.float32).reshape(40, 4)
    data_y = np.asarray([index % 2 for index in range(40)], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )
    result = run_screening_config(
        data_x,
        data_y,
        screening_spec(arm_name),
        seed=0,
        config=config,
    )
    assert isinstance(result, ScreeningRunResult)
    assert np.isfinite(result.per_task_loss).all()
    payload = noise_curvature_development_result_payload(result, outcome="inconclusive")
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["development_seed_protocol"] == list(DEVELOPMENT_SEEDS)
    resources = payload["resources"]
    assert isinstance(resources, dict)
    # Query/byte accounting is architecture-driven and must not depend on
    # which controller mode the arm's Literal selects.
    reference = run_screening_config(
        data_x,
        data_y,
        screening_spec("noise_curvature_combined"),
        seed=0,
        config=config,
    )
    reference_resources = noise_curvature_development_result_payload(
        reference, outcome="inconclusive"
    )["resources"]
    assert isinstance(reference_resources, dict)
    # `timing_seconds` is wall-clock telemetry (CLAUDE.md: "timing remains
    # telemetry-only until a separately qualified timing protocol exists")
    # and is expected to vary run to run; every other counter is exact.
    assert resources.keys() == reference_resources.keys()
    for key, value in resources.items():
        if key == "timing_seconds":
            continue
        assert value == reference_resources[key], key


def test_receipt_validator_rejects_hostile_and_derived_counter_drift() -> None:
    result = ScreeningRunResult(
        config_name="noise_curvature_combined",
        base_learner="adamw",
        hyperparameters=dict(screening_spec("noise_curvature_combined").hyperparameters),
        seed=1,
        config=IPMNISTConfig(
            n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2
        ),
        per_task_accuracy=np.asarray([0.5], dtype=np.float64),
        per_task_loss=np.asarray([1.0], dtype=np.float64),
        per_task_plasticity=np.asarray([0.1], dtype=np.float64),
        wall_clock_seconds=1.0,
        noise_mode="step",
        noise_pool_steps=None,
    )
    payload = noise_curvature_development_result_payload(result, outcome="rejected")

    class HostileInt(int):
        pass

    hostile = copy.deepcopy(payload)
    hostile["seed"] = HostileInt(1)
    with pytest.raises(ValueError, match="seed"):
        validate_noise_curvature_development_result(hostile)
    drift = copy.deepcopy(payload)
    assert isinstance(drift["resources"], dict)
    drift["resources"]["model_queries"] += 1
    with pytest.raises(ValueError, match="resource counters"):
        validate_noise_curvature_development_result(drift)
    changed = copy.deepcopy(payload)
    assert isinstance(changed["protocol_differences"], list)
    changed["protocol_differences"][0] = "silently changed"
    with pytest.raises(ValueError, match="protocol_differences"):
        validate_noise_curvature_development_result(changed)
    invalid_utf8 = copy.deepcopy(payload)
    invalid_utf8["allowed_task_information"] = ["\ud800"]
    with pytest.raises(ValueError, match="UTF-8"):
        validate_noise_curvature_development_result(invalid_utf8)


def test_matched_scheduler_panel_binds_seed_stream_and_resource_axes() -> None:
    data_x = np.linspace(-1.0, 1.0, 160, dtype=np.float32).reshape(40, 4)
    data_y = np.asarray([index % 2 for index in range(40)], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )
    payloads = [
        noise_curvature_development_result_payload(
            run_screening_config(data_x, data_y, screening_spec(arm), seed=0, config=config),
            outcome="inconclusive",
        )
        for arm in registered_arms()
    ]
    assert len(validate_matched_noise_curvature_results(copy.deepcopy(payloads))) == 4
    with pytest.raises(ValueError, match="every arm"):
        validate_matched_noise_curvature_results(copy.deepcopy(payloads[:-1]))
    drift = copy.deepcopy(payloads)
    drift[-1]["seed"] = 1
    with pytest.raises(ValueError, match="differs on seed"):
        validate_matched_noise_curvature_results(drift)


def test_host_boundaries_reject_bool_subclasses_and_oversized_payloads() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        NoiseCurvatureConfig(mode="combined", total_steps=True)
    with pytest.raises(ValueError, match="exact finite float"):
        NoiseCurvatureConfig(mode="combined", total_steps=40, safety_factor=True)
    with pytest.raises(ValueError, match="diagnostic buffer"):
        noise_curvature_persistent_bytes(
            parameter_count=1_000_000,
            input_dim=1_000_000,
            control_interval=100,
        )
    bad_params = _tiny_params()
    bad_params["b1"] = bad_params["b1"].at[0].set(jnp.nan)
    with pytest.raises(ValueError, match="only finite"):
        init_noise_curvature_state(
            bad_params,
            NoiseCurvatureConfig(mode="combined", total_steps=40),
        )
    with pytest.raises(ValueError, match="task_length divisible"):
        run_screening_config(
            np.zeros((41, 4), dtype=np.float32),
            np.zeros((41,), dtype=np.int32),
            screening_spec("noise_curvature_combined"),
            seed=0,
            config=IPMNISTConfig(
                n_tasks=1,
                task_length=41,
                input_dim=4,
                hidden1=3,
                hidden2=2,
                n_classes=2,
            ),
        )
