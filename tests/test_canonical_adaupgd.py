"""Contracts for the explicitly Alberta-derived adaptive UPGD transform."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.canonical_upgd import (
    ALBERTA_ADAUPGD_PROFILE,
    OFFICIAL_ADAUPGD_COMMIT,
    OFFICIAL_ADAUPGD_PATH,
    OFFICIAL_ADAUPGD_PROFILE,
    AlbertaAdaUPGD,
    AlbertaAdaUPGDConfig,
    AlbertaAdaUPGDState,
    OfficialAdaUPGD,
    OfficialAdaUPGDConfig,
    OfficialAdaUPGDState,
    measure_alberta_adaupgd_state_nbytes,
    measure_official_adaupgd_state_nbytes,
)
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint


def _zero_noise(params):
    return jax.tree.map(jnp.zeros_like, params)


def _assert_key_equal(left, right) -> None:
    chex.assert_trees_all_equal(jr.key_data(left), jr.key_data(right))


def test_alberta_zero_decays_recover_unused_inf_trackers() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            utility_decay=0.0,
            second_moment_decay=0.0,
            noise_std=0.0,
        )
    )
    state = optimizer.init(params).replace(  # type: ignore[attr-defined]
        utility_ema={"w": jnp.full(2, jnp.inf, dtype=jnp.float32)},
        gradient_second_moment={
            "w": jnp.full(2, jnp.inf, dtype=jnp.float32)
        },
    )

    result = optimizer.update(
        state,
        params,
        {"w": jnp.asarray([-1.0, -0.5], dtype=jnp.float32)},
        jr.key(101),
        noise=_zero_noise(params),
    )

    assert bool(result.accepted)
    chex.assert_tree_all_finite(result.state)
    chex.assert_tree_all_finite(result.params)


def test_official_zero_decays_do_not_multiply_inf_trackers() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(
            utility_decay=0.0,
            beta1=0.0,
            beta2=0.0,
            noise_std=0.0,
        )
    )
    state = optimizer.init(params).replace(  # type: ignore[attr-defined]
        utility_ema={"w": jnp.full(2, jnp.inf, dtype=jnp.float32)},
        first_moment={"w": jnp.full(2, jnp.inf, dtype=jnp.float32)},
        second_moment={"w": jnp.full(2, jnp.inf, dtype=jnp.float32)},
    )

    result = optimizer.update(
        state,
        params,
        {"w": jnp.asarray([-1.0, -0.5], dtype=jnp.float32)},
        jr.key(102),
        noise=_zero_noise(params),
    )

    chex.assert_tree_all_finite(result.state)
    chex.assert_tree_all_finite(result.params)


def test_config_is_explicitly_derived_strict_and_roundtrips() -> None:
    config = AlbertaAdaUPGDConfig(
        step_size=0.02,
        utility_decay=0.9,
        second_moment_decay=0.8,
        noise_std=0.1,
        weight_decay=0.03,
        mode="non_protecting",
        normalization="local",
        epsilon=1e-6,
    )
    optimizer = AlbertaAdaUPGD.from_config(config.to_config())

    assert optimizer.config == config
    assert config.profile == ALBERTA_ADAUPGD_PROFILE
    assert config.official_reference_parity is False
    assert "Alberta-derived" in config.source_attribution
    assert config.to_config() == {
        "type": "AlbertaAdaUPGD",
        "profile": ALBERTA_ADAUPGD_PROFILE,
        "step_size": 0.02,
        "utility_decay": 0.9,
        "second_moment_decay": 0.8,
        "noise_std": 0.1,
        "weight_decay": 0.03,
        "mode": "non_protecting",
        "normalization": "local",
        "epsilon": 1e-6,
    }

    invalid = (
        {"step_size": 0.0},
        {"utility_decay": 1.0},
        {"second_moment_decay": 1.0},
        {"noise_std": -1.0},
        {"weight_decay": -1.0},
        {"epsilon": 0.0},
        {"epsilon": 1e-50},
        {"mode": "unknown"},
        {"normalization": "unknown"},
        {"profile": "official_adaupgd"},
    )
    for fields in invalid:
        with pytest.raises((TypeError, ValueError)):
            AlbertaAdaUPGDConfig(**fields)  # type: ignore[arg-type]
    for field in (
        "step_size",
        "utility_decay",
        "second_moment_decay",
        "noise_std",
        "weight_decay",
        "epsilon",
    ):
        with pytest.raises((TypeError, ValueError)):
            AlbertaAdaUPGDConfig(**{field: True})
        with pytest.raises(ValueError):
            AlbertaAdaUPGDConfig(**{field: float("nan")})

    missing = config.to_config()
    missing.pop("profile")
    with pytest.raises(ValueError, match="schema"):
        AlbertaAdaUPGD.from_config(missing)
    extra = config.to_config()
    extra["invented"] = 1
    with pytest.raises(ValueError, match="schema"):
        AlbertaAdaUPGD.from_config(extra)
    wrong_type = config.to_config()
    wrong_type["type"] = "CanonicalUPGD"
    with pytest.raises(ValueError, match="AlbertaAdaUPGD"):
        AlbertaAdaUPGD.from_config(wrong_type)


def test_scalar_protecting_step_matches_hand_equation_and_decoupled_decay() -> None:
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    gradients = {"w": jnp.asarray(-4.0, dtype=jnp.float32)}
    supplied_noise = {"w": jnp.asarray(0.5, dtype=jnp.float32)}
    config = AlbertaAdaUPGDConfig(
        step_size=0.1,
        utility_decay=0.5,
        second_moment_decay=0.75,
        noise_std=99.0,
        weight_decay=0.2,
        mode="protecting",
        normalization="global",
        epsilon=1e-6,
    )
    optimizer = AlbertaAdaUPGD(config)
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(1),
        noise=supplied_noise,
    )

    corrected_utility = 8.0
    gate = float(jax.nn.sigmoid(jnp.asarray(1.0)))
    second_moment = 0.25 * 16.0
    denominator = (second_moment / (1.0 - 0.75)) ** 0.5 + 1e-6
    expected = (1.0 - 0.1 * 0.2) * 2.0 - 0.1 * ((-4.0 + 0.5) / denominator) * (
        1.0 - gate
    )

    assert bool(result.accepted)
    assert float(result.corrected_utility["w"]) == pytest.approx(corrected_utility)
    assert float(result.scaled_utility["w"]) == pytest.approx(gate)
    assert float(result.state.gradient_second_moment["w"]) == pytest.approx(second_moment)
    assert float(result.adaptive_denominator["w"]) == pytest.approx(denominator)
    assert float(result.params["w"]) == pytest.approx(expected)


def test_nonprotecting_gates_only_noise_after_adaptive_normalization() -> None:
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    gradient = {"w": jnp.asarray(-4.0, dtype=jnp.float32)}
    noise = {"w": jnp.asarray(0.5, dtype=jnp.float32)}
    common = dict(
        step_size=0.1,
        utility_decay=0.0,
        second_moment_decay=0.0,
        noise_std=1.0,
        weight_decay=0.0,
        normalization="global",
        epsilon=1e-6,
    )
    protecting = AlbertaAdaUPGD(AlbertaAdaUPGDConfig(mode="protecting", **common))
    nonprotecting = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(mode="non_protecting", **common)
    )
    protected = protecting.update(
        protecting.init(params), params, gradient, jr.key(0), noise=noise
    )
    unprotected = nonprotecting.update(
        nonprotecting.init(params), params, gradient, jr.key(0), noise=noise
    )

    gate = float(jax.nn.sigmoid(jnp.asarray(1.0)))
    denominator = 4.0 + 1e-6
    protected_direction = (-4.0 + 0.5) / denominator * (1.0 - gate)
    unprotected_direction = -4.0 / denominator + 0.5 / denominator * (1.0 - gate)
    assert float(protected.params["w"]) == pytest.approx(2.0 - 0.1 * protected_direction)
    assert float(unprotected.params["w"]) == pytest.approx(
        2.0 - 0.1 * unprotected_direction
    )
    assert float(unprotected.params["w"]) > float(protected.params["w"])


def test_two_layer_global_hand_math_and_utility_order_cases() -> None:
    params = {
        "first": jnp.ones(4, dtype=jnp.float32),
        "second": {"w": jnp.asarray([2.0, -1.0], dtype=jnp.float32)},
    }
    gradients = {
        "first": jnp.asarray([-4.0, 0.0, 1.0, -4.0], dtype=jnp.float32),
        "second": {"w": jnp.asarray([-1.0, 1.0], dtype=jnp.float32)},
    }
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            utility_decay=0.0,
            second_moment_decay=0.0,
            noise_std=0.0,
            normalization="global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(2),
        noise=_zero_noise(params),
    )

    expected_corrected = {
        "first": jnp.asarray([4.0, 0.0, -1.0, 4.0], dtype=jnp.float32),
        "second": {"w": jnp.asarray([2.0, 1.0], dtype=jnp.float32)},
    }
    expected_gates = jax.tree.map(
        lambda value: jax.nn.sigmoid(value / 4.0), expected_corrected
    )
    chex.assert_trees_all_close(result.corrected_utility, expected_corrected)
    chex.assert_trees_all_close(result.scaled_utility, expected_gates)
    assert float(result.scaled_utility["first"][0]) == pytest.approx(
        float(result.scaled_utility["first"][3])
    )
    assert float(result.scaled_utility["first"][0]) > 0.5
    assert float(result.scaled_utility["first"][1]) == pytest.approx(0.5)
    assert float(result.scaled_utility["first"][2]) < 0.5
    assert float(result.metrics["global_maximum_utility"]) == pytest.approx(4.0)


def test_local_normalization_is_per_leaf_row_and_all_zero_is_finite() -> None:
    params = {
        "matrix": jnp.ones((2, 2), dtype=jnp.float32),
        "zero": jnp.ones(2, dtype=jnp.float32),
    }
    gradients = {
        "matrix": jnp.asarray([[-3.0, -4.0], [0.0, -2.0]], dtype=jnp.float32),
        "zero": jnp.zeros(2, dtype=jnp.float32),
    }
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            utility_decay=0.0,
            second_moment_decay=0.0,
            noise_std=0.0,
            normalization="local",
            epsilon=1e-6,
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(3),
        noise=_zero_noise(params),
    )

    expected_matrix = jax.nn.sigmoid(
        jnp.asarray([[3.0 / 5.0, 4.0 / 5.0], [0.0, 1.0]], dtype=jnp.float32)
    )
    chex.assert_trees_all_close(result.scaled_utility["matrix"], expected_matrix)
    chex.assert_trees_all_close(
        result.scaled_utility["zero"], jnp.full(2, 0.5, dtype=jnp.float32)
    )
    chex.assert_tree_all_finite(
        (
            result.params,
            result.state,
            result.scaled_utility,
            result.corrected_utility,
            result.adaptive_denominator,
            result.perturbation,
            result.metrics,
        )
    )


def test_bias_correction_uses_active_utility_age_and_optimizer_step_for_moment() -> None:
    params = {"w": jnp.asarray([1.0, 1.0], dtype=jnp.float32)}
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            step_size=1e-3,
            utility_decay=0.5,
            second_moment_decay=0.5,
            noise_std=0.0,
            normalization="global",
        )
    )
    first = optimizer.update(
        optimizer.init(params),
        params,
        {"w": jnp.asarray([-2.0, -8.0], dtype=jnp.float32)},
        jr.key(4),
        mask={"w": jnp.asarray([True, False])},
        noise=_zero_noise(params),
    )
    second = optimizer.update(
        first.state,
        first.params,
        {"w": jnp.asarray([-4.0, -2.0], dtype=jnp.float32)},
        first.next_key,
        mask={"w": jnp.asarray([True, True])},
        noise=_zero_noise(params),
    )

    first_inst = -float(first.params["w"][0]) * -4.0
    expected_first_corrected = (0.5 * 1.0 + 0.5 * first_inst) / 0.75
    expected_second_corrected = -float(first.params["w"][1]) * -2.0
    assert int(second.state.step) == 2
    chex.assert_trees_all_equal(
        second.state.utility_age["w"], jnp.asarray([2, 1], dtype=jnp.int32)
    )
    assert float(second.corrected_utility["w"][0]) == pytest.approx(
        expected_first_corrected
    )
    assert float(second.corrected_utility["w"][1]) == pytest.approx(
        expected_second_corrected
    )
    expected_moment = 0.5 * jnp.asarray([2.0**2, 8.0**2]) * 0.5 + 0.5 * jnp.asarray(
        [4.0**2, 2.0**2]
    )
    chex.assert_trees_all_close(second.state.gradient_second_moment["w"], expected_moment)


def test_mask_selects_upgd_protection_not_the_adaptive_base_update() -> None:
    params = {
        "trunk": jnp.asarray([2.0], dtype=jnp.float32),
        "head": jnp.asarray([2.0], dtype=jnp.float32),
    }
    gradients = jax.tree.map(lambda value: -2.0 * jnp.ones_like(value), params)
    noise = jax.tree.map(lambda value: 1.0 * jnp.ones_like(value), params)
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            step_size=0.1,
            utility_decay=0.0,
            second_moment_decay=0.0,
            noise_std=1.0,
            weight_decay=0.2,
            mode="protecting",
            normalization="global",
            epsilon=1e-6,
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(5),
        mask={"trunk": jnp.asarray(True), "head": jnp.asarray(False)},
        noise=noise,
    )

    denominator = 2.0 + 1e-6
    gate = float(jax.nn.sigmoid(jnp.asarray(1.0)))
    expected_trunk = 2.0 * 0.98 - 0.1 * ((-2.0 + 1.0) / denominator) * (1.0 - gate)
    expected_head = 2.0 * 0.98 - 0.1 * (-2.0 / denominator)
    assert float(result.params["trunk"][0]) == pytest.approx(expected_trunk)
    assert float(result.params["head"][0]) == pytest.approx(expected_head)
    chex.assert_trees_all_equal(result.state.utility_age["head"], jnp.zeros(1, jnp.int32))
    chex.assert_trees_all_equal(result.state.utility_ema["head"], jnp.zeros(1, jnp.float32))
    chex.assert_trees_all_equal(result.perturbation["head"], jnp.zeros(1, jnp.float32))
    assert float(result.state.gradient_second_moment["head"][0]) == pytest.approx(4.0)

    with pytest.raises(TypeError, match="boolean"):
        optimizer.update(
            optimizer.init(params),
            params,
            gradients,
            jr.key(0),
            mask={"trunk": jnp.asarray(1), "head": jnp.asarray(0)},
        )
    with pytest.raises(ValueError, match="broadcast"):
        optimizer.update(
            optimizer.init(params),
            params,
            gradients,
            jr.key(0),
            mask={"trunk": jnp.ones(2, dtype=jnp.bool_), "head": jnp.asarray(False)},
        )


@pytest.mark.parametrize(
    "bad_gradients",
    [
        {"a": None, "b": jnp.asarray([1.0], dtype=jnp.float32)},
        {"a": jnp.asarray([jnp.nan], dtype=jnp.float32), "b": jnp.asarray([1.0])},
        {"a": jnp.asarray([1.0], dtype=jnp.float32), "b": jnp.asarray([jnp.inf])},
    ],
)
def test_missing_or_nonfinite_gradient_rolls_back_whole_transaction(bad_gradients) -> None:
    params = {
        "a": jnp.asarray([1.0], dtype=jnp.float32),
        "b": jnp.asarray([2.0], dtype=jnp.float32),
    }
    optimizer = AlbertaAdaUPGD(AlbertaAdaUPGDConfig(noise_std=0.1))
    state = optimizer.init(params)
    key = jr.key(6)
    result = optimizer.update(
        state,
        params,
        bad_gradients,
        key,
        mask={"a": jnp.asarray(False), "b": jnp.asarray(True)},
    )

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.params, params)
    chex.assert_trees_all_equal(result.state, state)
    _assert_key_equal(result.next_key, key)
    chex.assert_trees_all_equal(result.perturbation, _zero_noise(params))


def test_nonfinite_noise_tampered_state_and_overflow_candidate_roll_back() -> None:
    params = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(step_size=10.0, noise_std=1.0, second_moment_decay=0.0)
    )
    state = optimizer.init(params)
    key = jr.key(7)

    for candidate_state, noise in (
        (
            dataclasses.replace(
                state,
                gradient_second_moment={"w": jnp.asarray([-1.0], dtype=jnp.float32)},
            ),
            {"w": jnp.zeros(1, dtype=jnp.float32)},
        ),
        (state, {"w": jnp.asarray([jnp.nan], dtype=jnp.float32)}),
        (state, {"w": jnp.asarray([jnp.finfo(jnp.float32).max], dtype=jnp.float32)}),
    ):
        result = optimizer.update(candidate_state, params, gradients, key, noise=noise)
        assert not bool(result.accepted)
        chex.assert_trees_all_equal(result.params, params)
        chex.assert_trees_all_equal(result.state, candidate_state)
        _assert_key_equal(result.next_key, key)

    exhausted = dataclasses.replace(state, step=jnp.asarray(2**31 - 1, dtype=jnp.int32))
    result = optimizer.update(exhausted, params, gradients, key, noise=_zero_noise(params))
    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, exhausted)
    _assert_key_equal(result.next_key, key)

    huge_params = {"w": jnp.asarray([jnp.finfo(jnp.float32).max], dtype=jnp.float32)}
    huge_gradients = {
        "w": jnp.asarray([jnp.finfo(jnp.float32).max], dtype=jnp.float32)
    }
    huge_state = optimizer.init(huge_params)
    result = optimizer.update(
        huge_state,
        huge_params,
        huge_gradients,
        key,
        noise=_zero_noise(huge_params),
    )
    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.params, huge_params)
    chex.assert_trees_all_equal(result.state, huge_state)
    _assert_key_equal(result.next_key, key)
    chex.assert_tree_all_finite(result.metrics)


def test_requires_typed_scalar_threefry_key_and_fixed_noise_is_key_independent() -> None:
    params = {"w": jnp.asarray([1.0, -1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.asarray([0.2, -0.3], dtype=jnp.float32)}
    fixed_noise = {"w": jnp.asarray([0.1, -0.2], dtype=jnp.float32)}
    optimizer = AlbertaAdaUPGD(AlbertaAdaUPGDConfig(noise_std=9.0))
    state = optimizer.init(params)

    with pytest.raises(TypeError, match="typed scalar threefry"):
        optimizer.update(state, params, gradients, jr.PRNGKey(0))
    with pytest.raises(TypeError, match="typed scalar threefry"):
        optimizer.update(state, params, gradients, jr.key(0, impl="rbg"))

    first = optimizer.update(state, params, gradients, jr.key(1), noise=fixed_noise)
    second = optimizer.update(state, params, gradients, jr.key(99), noise=fixed_noise)
    chex.assert_trees_all_close(first.params, second.params)
    chex.assert_trees_all_close(first.perturbation, fixed_noise)
    assert not bool(jnp.all(jr.key_data(first.next_key) == jr.key_data(second.next_key)))


def test_sigma_zero_is_deterministic_and_jit_scan_matches_loop() -> None:
    params = {"w": jnp.asarray([1.0, -1.0], dtype=jnp.float32)}
    gradients = jnp.asarray(
        [[0.2, -0.3], [-0.1, 0.4], [0.5, 0.25]], dtype=jnp.float32
    )
    optimizer = AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            step_size=0.02,
            utility_decay=0.8,
            second_moment_decay=0.7,
            noise_std=0.0,
            weight_decay=0.03,
        )
    )
    initial_state = optimizer.init(params)
    initial_key = jr.key(23)

    first = jax.jit(optimizer.update)(
        initial_state, params, {"w": gradients[0]}, initial_key
    )
    second = jax.jit(optimizer.update)(
        initial_state, params, {"w": gradients[0]}, initial_key
    )
    chex.assert_trees_all_equal(first, second)
    chex.assert_trees_all_equal(first.perturbation, _zero_noise(params))
    assert int(first.metrics["persistent_state_nbytes"]) == (
        measure_alberta_adaupgd_state_nbytes(initial_state)
    )

    loop_state, loop_params, loop_key = initial_state, params, initial_key
    loop_values = []
    for gradient in gradients:
        result = optimizer.update(loop_state, loop_params, {"w": gradient}, loop_key)
        loop_state, loop_params, loop_key = result.state, result.params, result.next_key
        loop_values.append(result.params["w"])

    def scan_step(carry, gradient):
        state, current_params, key = carry
        result = optimizer.update(state, current_params, {"w": gradient}, key)
        return (result.state, result.params, result.next_key), result.params["w"]

    (scan_state, scan_params, scan_key), scan_values = jax.jit(
        lambda: jax.lax.scan(
            scan_step, (initial_state, params, initial_key), gradients
        )
    )()
    chex.assert_trees_all_close(scan_state, loop_state)
    chex.assert_trees_all_close(scan_params, loop_params)
    chex.assert_trees_all_close(scan_values, jnp.stack(loop_values))
    _assert_key_equal(scan_key, loop_key)


def test_state_checkpoint_existing_schema_compatibility_and_resources(tmp_path) -> None:
    params = {
        "w": jnp.ones((2, 2), dtype=jnp.float32),
        "b": jnp.ones(3, dtype=jnp.float32),
    }
    optimizer = AlbertaAdaUPGD(AlbertaAdaUPGDConfig(noise_std=0.0))
    template = optimizer.init(params)
    updated = optimizer.update(
        template,
        params,
        jax.tree.map(lambda value: -jnp.ones_like(value), params),
        jr.key(8),
    ).state
    checkpoint = tmp_path / "alberta_adaupgd"
    save_checkpoint(updated, checkpoint)
    loaded, metadata = load_checkpoint(template, checkpoint)
    chex.assert_trees_all_equal(loaded, updated)
    assert metadata == {}
    assert bool(optimizer.state_valid(loaded, params))

    resources = optimizer.resource_budget(updated)
    assert resources.parameter_count == 7
    assert resources.persistent_array_count == 7
    assert resources.persistent_state_nbytes == 3 * 7 * 4 + 4
    assert resources.profile == ALBERTA_ADAUPGD_PROFILE
    assert resources.to_dict()["official_reference_parity"] is False
    assert measure_alberta_adaupgd_state_nbytes(updated) == resources.persistent_state_nbytes

    # The opt-in extension must not alter any legacy canonical config/state bytes.
    from alberta_framework.core.canonical_upgd import CanonicalUPGD, CanonicalUPGDConfig

    legacy_config = CanonicalUPGDConfig().to_config()
    assert legacy_config == {
        "type": "CanonicalUPGD",
        "step_size": 1e-3,
        "utility_decay": 0.999,
        "noise_std": 1e-3,
        "weight_decay": 0.0,
        "mode": "protecting",
        "profile": "safe_extended",
        "normalization": "global",
        "epsilon": 1e-8,
    }
    assert tuple(CanonicalUPGD().init(params)) == (
        "utility_ema",
        "utility_age",
        "step",
    )


def test_structure_dtype_and_shape_contracts_fail_before_update() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = AlbertaAdaUPGD()
    state = optimizer.init(params)
    with pytest.raises(ValueError, match="structure"):
        optimizer.update(state, params, {"x": jnp.ones(2)}, jr.key(0))
    with pytest.raises(ValueError, match="shape"):
        optimizer.update(state, params, {"w": jnp.ones(3)}, jr.key(0))
    with pytest.raises(ValueError, match="every parameter leaf"):
        optimizer.update(
            state,
            params,
            {"w": jnp.ones(2)},
            jr.key(0),
            noise={"w": None},
        )
    with pytest.raises(ValueError, match="floating"):
        optimizer.init({"w": jnp.ones(2, dtype=jnp.int32)})
    bad_state = AlbertaAdaUPGDState(
        utility_ema={"x": jnp.zeros(2)},
        utility_age=state.utility_age,
        gradient_second_moment=state.gradient_second_moment,
        step=state.step,
    )
    with pytest.raises(ValueError, match="structure"):
        optimizer.update(bad_state, params, {"w": jnp.ones(2)}, jr.key(0))


def test_official_config_binds_exact_commit_path_defaults_and_roundtrip() -> None:
    config = OfficialAdaUPGDConfig()
    assert config.profile == OFFICIAL_ADAUPGD_PROFILE
    assert config.source_commit == OFFICIAL_ADAUPGD_COMMIT
    assert config.source_path == OFFICIAL_ADAUPGD_PATH
    assert config.official_reference_parity is True
    assert config.parity_scope == (
        "finite_float32_all_active_single_group_fixed_noise_equation_parity"
    )
    assert config.to_config() == {
        "type": "OfficialAdaUPGD",
        "profile": OFFICIAL_ADAUPGD_PROFILE,
        "source_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
        "source_path": "core/run/rl/adaupgd.py",
        "step_size": 1e-5,
        "weight_decay": 0.001,
        "utility_decay": 0.999,
        "noise_std": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-5,
    }
    assert OfficialAdaUPGD.from_config(config.to_config()).config == config

    for fields in (
        {"profile": "safe_extended"},
        {"source_commit": "0" * 40},
        {"source_path": "adaupgd.py"},
        {"step_size": 0.0},
        {"weight_decay": -1.0},
        {"utility_decay": 1.0},
        {"noise_std": -1.0},
        {"beta1": 1.0},
        {"beta2": 1.0},
        {"epsilon": 0.0},
    ):
        with pytest.raises((TypeError, ValueError)):
            OfficialAdaUPGDConfig(**fields)  # type: ignore[arg-type]
    missing = config.to_config()
    missing.pop("source_path")
    with pytest.raises(ValueError, match="schema"):
        OfficialAdaUPGD.from_config(missing)


def test_official_scalar_fixed_noise_matches_pinned_source_equation() -> None:
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    gradients = {"w": jnp.asarray(-4.0, dtype=jnp.float32)}
    fixed_noise = {"w": jnp.asarray(0.5, dtype=jnp.float32)}
    config = OfficialAdaUPGDConfig(
        step_size=0.1,
        weight_decay=0.2,
        utility_decay=0.5,
        noise_std=99.0,
        beta1=0.5,
        beta2=0.75,
        epsilon=1e-6,
    )
    optimizer = OfficialAdaUPGD(config)
    result = optimizer.update(
        optimizer.init(params), params, gradients, jr.key(30), noise=fixed_noise
    )

    raw_utility = 0.5 * 8.0
    corrected_utility = raw_utility / (1.0 - 0.5)
    first_moment = 0.5 * -4.0
    corrected_first = first_moment / (1.0 - 0.5)
    second_moment = 0.25 * 16.0
    corrected_second = second_moment / (1.0 - 0.75)
    gate = float(jax.nn.sigmoid(jnp.asarray(corrected_utility / raw_utility)))
    direction = corrected_first * (1.0 - gate) / (
        corrected_second**0.5 + 1e-6
    ) + 0.5 * (1.0 - gate)
    expected = 2.0 * (1.0 - 0.1 * 0.2) - 2.0 * 0.1 * direction

    assert float(result.state.utility_ema["w"]) == pytest.approx(raw_utility)
    assert float(result.state.first_moment["w"]) == pytest.approx(first_moment)
    assert float(result.state.second_moment["w"]) == pytest.approx(second_moment)
    assert float(result.corrected_utility["w"]) == pytest.approx(corrected_utility)
    assert float(result.corrected_first_moment["w"]) == pytest.approx(corrected_first)
    assert float(result.corrected_second_moment["w"]) == pytest.approx(corrected_second)
    assert float(result.scaled_utility["w"]) == pytest.approx(gate)
    assert float(result.metrics["raw_global_maximum_utility"]) == pytest.approx(
        raw_utility
    )
    assert float(result.params["w"]) == pytest.approx(expected)


def test_official_two_leaf_raw_global_maximum_and_fixed_noise_parity() -> None:
    params = {
        "a": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "b": {"w": jnp.asarray([2.0], dtype=jnp.float32)},
    }
    gradients = {
        "a": jnp.asarray([-2.0, -1.0], dtype=jnp.float32),
        "b": {"w": jnp.asarray([-3.0], dtype=jnp.float32)},
    }
    noise = jax.tree.map(lambda value: jnp.full_like(value, 0.25), params)
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(
            step_size=0.02,
            weight_decay=0.1,
            utility_decay=0.5,
            noise_std=7.0,
            beta1=0.5,
            beta2=0.5,
            epsilon=1e-6,
        )
    )
    result = optimizer.update(
        optimizer.init(params), params, gradients, jr.key(31), noise=noise
    )

    raw_utility = {
        "a": jnp.asarray([1.0, 0.5], dtype=jnp.float32),
        "b": {"w": jnp.asarray([3.0], dtype=jnp.float32)},
    }
    corrected = jax.tree.map(lambda value: value / 0.5, raw_utility)
    expected_gate = jax.tree.map(
        lambda value: jax.nn.sigmoid(value / 3.0), corrected
    )
    chex.assert_trees_all_close(result.state.utility_ema, raw_utility)
    chex.assert_trees_all_close(result.corrected_utility, corrected)
    chex.assert_trees_all_close(result.scaled_utility, expected_gate)
    assert float(result.metrics["raw_global_maximum_utility"]) == pytest.approx(3.0)

    expected_params = jax.tree.map(
        lambda parameter, gradient, gate: parameter * (1.0 - 0.02 * 0.1)
        - 2.0
        * 0.02
        * (
            gradient * (1.0 - gate) / (jnp.abs(gradient) + 1e-6)
            + 0.25 * (1.0 - gate)
        ),
        params,
        gradients,
        expected_gate,
    )
    chex.assert_trees_all_close(result.params, expected_params)


def test_official_bias_corrected_first_second_and_utility_moments_over_time() -> None:
    params = {"w": jnp.asarray([2.0, -1.0], dtype=jnp.float32)}
    first_gradient = {"w": jnp.asarray([-2.0, 4.0], dtype=jnp.float32)}
    second_gradient = {"w": jnp.asarray([-4.0, 2.0], dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(
            step_size=1e-4,
            utility_decay=0.5,
            noise_std=0.0,
            beta1=0.5,
            beta2=0.5,
        )
    )
    first = optimizer.update(
        optimizer.init(params),
        params,
        first_gradient,
        jr.key(32),
        noise=_zero_noise(params),
    )
    second = optimizer.update(
        first.state,
        first.params,
        second_gradient,
        first.next_key,
        noise=_zero_noise(params),
    )

    expected_first_moment = 0.5 * first.state.first_moment["w"] + 0.5 * second_gradient["w"]
    expected_second_moment = 0.5 * first.state.second_moment["w"] + 0.5 * jnp.square(
        second_gradient["w"]
    )
    expected_utility = 0.5 * first.state.utility_ema["w"] + 0.5 * (
        -second_gradient["w"] * first.params["w"]
    )
    assert int(second.state.step) == 2
    chex.assert_trees_all_close(second.state.first_moment["w"], expected_first_moment)
    chex.assert_trees_all_close(second.state.second_moment["w"], expected_second_moment)
    chex.assert_trees_all_close(second.state.utility_ema["w"], expected_utility)
    chex.assert_trees_all_close(
        second.corrected_first_moment["w"], expected_first_moment / 0.75
    )
    chex.assert_trees_all_close(
        second.corrected_second_moment["w"], expected_second_moment / 0.75
    )
    chex.assert_trees_all_close(
        second.corrected_utility["w"], expected_utility / 0.75
    )


def test_official_noise_is_not_divided_by_adaptive_denominator() -> None:
    params = {"w": jnp.asarray([1.0, 1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.asarray([-1.0, -100.0], dtype=jnp.float32)}
    noise = {"w": jnp.asarray([0.5, 0.5], dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(
            step_size=0.1,
            weight_decay=0.0,
            utility_decay=0.0,
            noise_std=1.0,
            beta1=0.0,
            beta2=0.0,
            epsilon=1e-6,
        )
    )
    result = optimizer.update(
        optimizer.init(params), params, gradients, jr.key(33), noise=noise
    )
    gates = result.scaled_utility["w"]
    expected = params["w"] - 0.2 * (
        gradients["w"] * (1.0 - gates) / (jnp.abs(gradients["w"]) + 1e-6)
        + noise["w"] * (1.0 - gates)
    )
    chex.assert_trees_all_close(result.params["w"], expected)


def test_official_zero_denominator_and_nonfinite_values_preserve_source_quirks() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(noise_std=0.0, utility_decay=0.0)
    )
    zero = optimizer.update(
        optimizer.init(params),
        params,
        {"w": jnp.zeros(2, dtype=jnp.float32)},
        jr.key(34),
        noise=_zero_noise(params),
    )
    assert jnp.all(jnp.isnan(zero.scaled_utility["w"]))
    assert jnp.all(jnp.isnan(zero.params["w"]))
    assert float(zero.metrics["raw_global_maximum_utility"]) == pytest.approx(0.0)

    nonfinite = optimizer.update(
        optimizer.init(params),
        params,
        {"w": jnp.asarray([jnp.nan, 1.0], dtype=jnp.float32)},
        jr.key(35),
        noise=_zero_noise(params),
    )
    assert not bool(optimizer.state_valid(nonfinite.state, nonfinite.params))
    assert bool(jnp.any(jnp.isnan(nonfinite.params["w"])))


def test_official_rejects_unsupported_masks_missing_gradients_and_bad_keys() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD()
    state = optimizer.init(params)
    with pytest.raises(ValueError, match="does not accept masks"):
        optimizer.update(
            state,
            params,
            gradients,
            jr.key(0),
            mask={"w": jnp.asarray([True, False])},
        )
    with pytest.raises(ValueError, match="every gradient"):
        optimizer.update(state, params, {"w": None}, jr.key(0))
    with pytest.raises(TypeError, match="typed scalar threefry"):
        optimizer.update(state, params, gradients, jr.PRNGKey(0))
    with pytest.raises(ValueError, match="shape"):
        optimizer.update(state, params, {"w": jnp.ones(3)}, jr.key(0))
    with pytest.raises(ValueError, match="floating"):
        optimizer.init({"w": jnp.ones(2, dtype=jnp.int32)})
    bad_state = OfficialAdaUPGDState(
        utility_ema={"x": jnp.zeros(2, dtype=jnp.float32)},
        first_moment=state.first_moment,
        second_moment=state.second_moment,
        step=state.step,
    )
    with pytest.raises(ValueError, match="structure"):
        optimizer.update(bad_state, params, gradients, jr.key(0))


def test_official_typed_rng_jit_scan_checkpoint_and_resources(tmp_path) -> None:
    params = {
        "w": jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        "b": jnp.asarray([0.5], dtype=jnp.float32),
    }
    gradient_rows = jnp.asarray(
        [[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2], [0.5, 0.25, 0.3]],
        dtype=jnp.float32,
    )
    optimizer = OfficialAdaUPGD(
        OfficialAdaUPGDConfig(
            step_size=0.02,
            utility_decay=0.8,
            noise_std=0.1,
            beta1=0.7,
            beta2=0.6,
        )
    )
    initial_state = optimizer.init(params)
    initial_key = jr.key(36)
    first = jax.jit(optimizer.update)(
        initial_state,
        params,
        {"w": gradient_rows[0, :2], "b": gradient_rows[0, 2:]},
        initial_key,
    )
    repeated = jax.jit(optimizer.update)(
        initial_state,
        params,
        {"w": gradient_rows[0, :2], "b": gradient_rows[0, 2:]},
        initial_key,
    )
    chex.assert_trees_all_equal(first, repeated)
    split_keys = jr.split(initial_key, len(jax.tree.leaves(params)) + 1)
    expected_noise = jax.tree_util.tree_unflatten(
        jax.tree.structure(params),
        [
            jr.normal(noise_key, parameter.shape, dtype=parameter.dtype) * 0.1
            for parameter, noise_key in zip(
                jax.tree.leaves(params), split_keys[1:], strict=True
            )
        ],
    )
    chex.assert_trees_all_close(first.perturbation, expected_noise, rtol=1e-6, atol=1e-7)
    _assert_key_equal(first.next_key, split_keys[0])
    assert int(first.metrics["persistent_state_nbytes"]) == 40

    loop_state, loop_params, loop_key = initial_state, params, initial_key
    loop_values = []
    for row in gradient_rows:
        result = optimizer.update(
            loop_state,
            loop_params,
            {"w": row[:2], "b": row[2:]},
            loop_key,
        )
        loop_state, loop_params, loop_key = result.state, result.params, result.next_key
        loop_values.append(jnp.concatenate((result.params["w"], result.params["b"])))

    def scan_step(carry, row):
        state, current_params, key = carry
        result = optimizer.update(
            state, current_params, {"w": row[:2], "b": row[2:]}, key
        )
        values = jnp.concatenate((result.params["w"], result.params["b"]))
        return (result.state, result.params, result.next_key), values

    (scan_state, scan_params, scan_key), scan_values = jax.jit(
        lambda: jax.lax.scan(
            scan_step, (initial_state, params, initial_key), gradient_rows
        )
    )()
    chex.assert_trees_all_close(scan_state, loop_state)
    chex.assert_trees_all_close(scan_params, loop_params)
    chex.assert_trees_all_close(scan_values, jnp.stack(loop_values))
    _assert_key_equal(scan_key, loop_key)

    checkpoint = tmp_path / "official_adaupgd"
    save_checkpoint(loop_state, checkpoint)
    loaded, metadata = load_checkpoint(initial_state, checkpoint)
    chex.assert_trees_all_equal(loaded, loop_state)
    assert metadata == {}
    resources = optimizer.resource_budget(loop_state)
    assert resources.parameter_count == 3
    assert resources.persistent_array_count == 7
    assert resources.persistent_state_nbytes == 3 * 3 * 4 + 4
    assert resources.profile == OFFICIAL_ADAUPGD_PROFILE
    assert resources.source_commit == OFFICIAL_ADAUPGD_COMMIT
    assert resources.source_path == OFFICIAL_ADAUPGD_PATH
    assert measure_official_adaupgd_state_nbytes(loop_state) == 40


def test_saturated_protecting_gate_skips_overflowed_ada_direction() -> None:
    """gate=1 times an overflowed adaptive direction is 0*inf=NaN without a skip."""
    from alberta_framework.core.canonical_upgd import _skip_zero_scale

    gate = jnp.asarray(1.0, dtype=jnp.float32)
    adaptive = jnp.asarray(jnp.inf, dtype=jnp.float32)
    raw = (adaptive + 0.0) * (1.0 - gate)
    assert not bool(jnp.isfinite(raw))
    fixed = _skip_zero_scale(1.0 - gate, adaptive + 0.0)
    assert bool(jnp.isfinite(fixed))
    chex.assert_trees_all_close(fixed, jnp.zeros_like(adaptive))

    one_minus = jnp.asarray(0.0, dtype=jnp.float32)
    first = jnp.asarray(jnp.inf, dtype=jnp.float32)
    second = jnp.asarray(1.0, dtype=jnp.float32)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)
    pert = jnp.asarray(jnp.inf, dtype=jnp.float32)
    raw_official = first * one_minus / (jnp.sqrt(second) + eps) + pert * one_minus
    assert not bool(jnp.isfinite(raw_official))
    fixed_official = (
        _skip_zero_scale(one_minus, first) / (jnp.sqrt(second) + eps)
        + _skip_zero_scale(one_minus, pert)
    )
    assert bool(jnp.isfinite(fixed_official))
