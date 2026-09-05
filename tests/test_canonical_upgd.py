"""Equation, PyTree, and checkpoint tests for canonical first-order UPGD."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.canonical_upgd import (
    CanonicalUPGD,
    CanonicalUPGDConfig,
    _static_zero_scale,
)
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint


def _legacy_sink_decay_config(**overrides: object) -> CanonicalUPGDConfig:
    """Build the formerly accepted decay sink for defensive update-path tests."""

    config = CanonicalUPGDConfig(utility_decay=0.9, **overrides)  # type: ignore[arg-type]
    # Construction now rejects this value because it narrows to float32 1.0.
    # Simulate an already-loaded legacy/corrupt in-memory config so the atomic
    # refusal branch remains covered without reopening the public sink.
    object.__setattr__(config, "utility_decay", 0.999999999)
    return config


def test_config_validation_and_roundtrip() -> None:
    config = CanonicalUPGDConfig(
        step_size=0.02,
        utility_decay=0.9,
        noise_std=0.1,
        weight_decay=0.03,
        mode="non_protecting",
        profile="safe_extended",
        normalization="local",
        epsilon=1e-6,
    )
    restored = CanonicalUPGD.from_config(config.to_config())
    assert restored.config == config
    assert CanonicalUPGDConfig().profile == "safe_extended"
    assert CanonicalUPGDConfig().normalization == "global"

    with pytest.raises(ValueError, match="step_size"):
        CanonicalUPGDConfig(step_size=0.0)
    with pytest.raises(ValueError, match="utility_decay"):
        CanonicalUPGDConfig(utility_decay=1.0)
    with pytest.raises(ValueError, match="utility_decay"):
        CanonicalUPGDConfig(utility_decay=True)
    with pytest.raises(ValueError, match="noise_std"):
        CanonicalUPGDConfig(noise_std=-1.0)
    with pytest.raises(ValueError, match="weight_decay"):
        CanonicalUPGDConfig(weight_decay=-1.0)
    with pytest.raises(ValueError, match="mode"):
        CanonicalUPGDConfig(mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="profile"):
        CanonicalUPGDConfig(profile="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires normalization"):
        CanonicalUPGDConfig(profile="safe_extended", normalization=None)
    with pytest.raises(ValueError, match="fixes normalization"):
        CanonicalUPGDConfig(
            profile="paper_global",
            normalization="local",
        )
    with pytest.raises(ValueError, match="normalization"):
        CanonicalUPGDConfig(
            profile="safe_extended",
            normalization="other",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="only defines protecting"):
        CanonicalUPGDConfig(
            profile="official_readme_global",
            mode="non_protecting",
        )
    with pytest.raises(ValueError, match="expected CanonicalUPGD"):
        wrong_type = config.to_config()
        wrong_type["type"] = "UPGDLearner"
        CanonicalUPGD.from_config(wrong_type)
    missing_field = config.to_config()
    missing_field.pop("profile")
    with pytest.raises(ValueError, match="config fields"):
        CanonicalUPGD.from_config(missing_field)
    extra_field = config.to_config()
    extra_field["future_default"] = True
    with pytest.raises(ValueError, match="config fields"):
        CanonicalUPGD.from_config(extra_field)
    for field in ("step_size", "noise_std", "weight_decay", "epsilon"):
        with pytest.raises(ValueError, match=field):
            CanonicalUPGDConfig(**{field: float("nan")})
        with pytest.raises(ValueError, match=field):
            CanonicalUPGDConfig(**{field: float("inf")})
        with pytest.raises(ValueError, match=field):
            CanonicalUPGDConfig(**{field: True})


@pytest.mark.parametrize(
    ("profile", "normalization"),
    [
        ("paper_global", "global"),
        ("official_readme_global", "global"),
        ("official_experiment_global", "global"),
        ("official_experiment_local", "local"),
        ("paper_local_literal", "local"),
        ("safe_extended", "global"),
        ("safe_extended", "local"),
    ],
)
def test_every_profile_roundtrips_with_explicit_normalization(
    profile: str,
    normalization: str,
) -> None:
    config = CanonicalUPGDConfig(
        profile=profile,  # type: ignore[arg-type]
        normalization=normalization,  # type: ignore[arg-type]
    )
    restored = CanonicalUPGD.from_config(config.to_config())
    assert restored.config == config
    assert restored.config.resolved_normalization == normalization


def test_init_matches_nested_parameter_tree() -> None:
    params = {
        "trunk": (jnp.ones((2, 3)), jnp.ones((3,))),
        "head": jnp.ones((3, 2)),
    }
    state = CanonicalUPGD().init(params)
    chex.assert_trees_all_equal(
        state.utility_ema,
        jax.tree.map(jnp.zeros_like, params),
    )
    chex.assert_trees_all_equal(
        state.utility_age,
        jax.tree.map(
            lambda value: jnp.zeros(value.shape, dtype=jnp.int32),
            params,
        ),
    )
    assert int(state.step) == 0


def test_init_rejects_empty_and_nonfloating_parameter_trees() -> None:
    optimizer = CanonicalUPGD()
    with pytest.raises(ValueError, match="at least one"):
        optimizer.init({})
    with pytest.raises(ValueError, match="floating-point"):
        optimizer.init({"w": jnp.ones(2, dtype=jnp.int32)})


def test_beta_zero_matches_paper_global_first_order_equations() -> None:
    """A supplied perturbation pins parity independently of RNG libraries."""

    params = {"w": jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)}
    gradients = {"w": jnp.array([-2.0, 0.25, -1.0], dtype=jnp.float32)}
    noise = {"w": jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.05,
            utility_decay=0.0,
            noise_std=999.0,
            mode="protecting",
            profile="paper_global",
        )
    )

    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
        noise=noise,
    )

    utility = -gradients["w"] * params["w"]
    maximum = jnp.max(utility)
    gate = jax.nn.sigmoid(utility / maximum)
    expected = params["w"] - 0.05 * (gradients["w"] + noise["w"]) * (1.0 - gate)

    chex.assert_trees_all_close(result.corrected_utility["w"], utility)
    chex.assert_trees_all_close(result.scaled_utility["w"], gate)
    chex.assert_trees_all_close(result.perturbation["w"], noise["w"])
    chex.assert_trees_all_close(result.params["w"], expected)


@pytest.mark.parametrize(
    ("profile", "normalized", "direction_multiplier", "maximum"),
    [
        ("paper_global", [0.5, 1.0], 1.0, 2.0),
        ("official_readme_global", [1.0, 2.0], 2.0, 1.0),
        ("official_experiment_global", [1.0, 2.0], 1.0, 1.0),
    ],
)
def test_global_source_profiles_pin_denominator_coefficient_and_decay(
    profile: str,
    normalized: list[float],
    direction_multiplier: float,
    maximum: float,
) -> None:
    """Distinguish the paper, README, and experiment-code equations."""

    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.array([-1.0, -2.0], dtype=jnp.float32)}
    noise = {"w": jnp.array([0.25, -0.5], dtype=jnp.float32)}
    step_size = 0.1
    weight_decay = 0.2
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=step_size,
            utility_decay=0.5,
            noise_std=999.0,
            weight_decay=weight_decay,
            profile=profile,  # type: ignore[arg-type]
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
        noise=noise,
    )

    gate = jax.nn.sigmoid(jnp.asarray(normalized, dtype=jnp.float32))
    expected = params["w"] * (1.0 - step_size * weight_decay) - (
        direction_multiplier * step_size * (gradients["w"] + noise["w"]) * (1.0 - gate)
    )
    chex.assert_trees_all_close(
        result.corrected_utility["w"],
        jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(result.scaled_utility["w"], gate)
    chex.assert_trees_all_close(result.params["w"], expected)
    assert float(result.metrics["global_maximum_utility"]) == pytest.approx(maximum)


def test_protecting_and_nonprotecting_gate_different_causal_terms() -> None:
    params = {"w": jnp.array([1.0, 1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.array([-1.0, -2.0], dtype=jnp.float32)}
    noise = {"w": jnp.array([0.4, -0.3], dtype=jnp.float32)}
    common = dict(
        step_size=0.1,
        utility_decay=0.0,
        noise_std=0.0,
        profile="paper_global",
    )
    protecting = CanonicalUPGD(CanonicalUPGDConfig(mode="protecting", **common))
    nonprotecting = CanonicalUPGD(CanonicalUPGDConfig(mode="non_protecting", **common))

    protected = protecting.update(
        protecting.init(params),
        params,
        gradients,
        jr.key(1),
        noise=noise,
    )
    unprotected = nonprotecting.update(
        nonprotecting.init(params),
        params,
        gradients,
        jr.key(1),
        noise=noise,
    )
    gate = protected.scaled_utility["w"]

    protected_expected = params["w"] - 0.1 * (gradients["w"] + noise["w"]) * (1.0 - gate)
    unprotected_expected = params["w"] - 0.1 * (gradients["w"] + noise["w"] * (1.0 - gate))
    chex.assert_trees_all_close(protected.params["w"], protected_expected)
    chex.assert_trees_all_close(unprotected.params["w"], unprotected_expected)
    assert not jnp.allclose(protected.params["w"], unprotected.params["w"])


def test_ema_bias_correction_recovers_constant_signed_utility() -> None:
    params = {"w": jnp.array([2.0, -1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.array([-0.5, 0.25], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.9,
            noise_std=0.0,
            mode="non_protecting",
        )
    )
    state = optimizer.init(params)
    expected_utility = -gradients["w"] * params["w"]

    # Keep parameters fixed while testing the trace itself.
    for step in range(1, 5):
        result = optimizer.update(
            state,
            params,
            gradients,
            jr.key(step),
            noise={"w": jnp.zeros_like(params["w"])},
        )
        state = result.state
        chex.assert_trees_all_close(
            result.corrected_utility["w"],
            expected_utility,
            atol=1e-6,
        )
        chex.assert_trees_all_equal(
            state.utility_age["w"],
            jnp.full(params["w"].shape, step, dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    ("profile", "normalization"),
    [
        ("paper_global", "global"),
        ("official_readme_global", "global"),
        ("official_experiment_global", "global"),
        ("official_experiment_local", "local"),
        ("paper_local_literal", "local"),
        ("safe_extended", "global"),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_lifetime_counter_capacity_rejects_the_complete_update(
    profile: str,
    normalization: str,
    compiled: bool,
) -> None:
    with jax.enable_x64():
        params = {"w": jnp.array([2.0], dtype=jnp.float64)}
        gradients = {"w": jnp.array([-0.5], dtype=jnp.float64)}
        optimizer = CanonicalUPGD(
            _legacy_sink_decay_config(
                step_size=0.1,
                noise_std=0.0,
                profile=profile,  # type: ignore[arg-type]
                normalization=normalization,  # type: ignore[arg-type]
            )
        )
        maximum = jnp.iinfo(jnp.int32).max
        state = optimizer.init(params).replace(
            utility_ema={"w": jnp.array([1.0], dtype=jnp.float64)},
            utility_age={"w": jnp.array([maximum], dtype=jnp.int32)},
            step=jnp.array(maximum, dtype=jnp.int32),
        )

        def apply_update(state, params, gradients, key, noise):
            return optimizer.update(state, params, gradients, key, noise=noise)

        update = jax.jit(apply_update) if compiled else apply_update
        result = update(
            state,
            params,
            gradients,
            jr.key(0),
            {"w": jnp.zeros(1, dtype=jnp.float64)},
        )

        assert int(result.state.step) == maximum
        assert int(result.state.utility_age["w"][0]) == maximum
        chex.assert_trees_all_equal(result.params, params)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(result.next_key, jr.key(0))
        chex.assert_trees_all_equal(
            result.corrected_utility,
            jax.tree.map(jnp.zeros_like, params),
        )
        chex.assert_trees_all_equal(
            result.scaled_utility,
            jax.tree.map(jnp.zeros_like, params),
        )
        assert not bool(result.metrics["update_applied"])


@pytest.mark.parametrize("profile", ["paper_global", "safe_extended"])
def test_final_representable_lifetime_update_is_consumed_before_capacity(
    profile: str,
) -> None:
    with jax.enable_x64():
        params = {"w": jnp.array([2.0], dtype=jnp.float64)}
        gradients = {"w": jnp.array([-0.5], dtype=jnp.float64)}
        decay = float(jnp.nextafter(jnp.float32(1.0), jnp.float32(0.0)))
        optimizer = CanonicalUPGD(
            CanonicalUPGDConfig(
                step_size=0.1,
                utility_decay=decay,
                noise_std=0.0,
                profile=profile,  # type: ignore[arg-type]
                normalization="global",
            )
        )
        maximum = jnp.iinfo(jnp.int32).max
        prior_step = int(maximum) - 1
        correction = 1.0 - decay**prior_step
        state = optimizer.init(params).replace(
            utility_ema={"w": jnp.array([correction], dtype=jnp.float64)},
            utility_age={"w": jnp.array([prior_step], dtype=jnp.int32)},
            step=jnp.array(prior_step, dtype=jnp.int32),
        )

        result = optimizer.update(
            state,
            params,
            gradients,
            jr.key(1),
            noise={"w": jnp.zeros(1, dtype=jnp.float64)},
        )

        assert int(result.state.step) == maximum
        assert int(result.state.utility_age["w"][0]) == maximum
        assert float(result.corrected_utility["w"][0]) == pytest.approx(1.0)
        assert bool(result.metrics["update_applied"])


@pytest.mark.parametrize("profile", ["paper_global", "safe_extended"])
@pytest.mark.parametrize("compiled", [False, True])
def test_converged_float32_bias_clock_continues_after_counter_saturation(
    profile: str,
    compiled: bool,
) -> None:
    params = {"w": jnp.array([2.0], dtype=jnp.float32)}
    gradients = {"w": jnp.array([-0.5], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.1,
            utility_decay=0.9,
            noise_std=0.0,
            profile=profile,  # type: ignore[arg-type]
            normalization="global",
        )
    )
    maximum = jnp.iinfo(jnp.int32).max
    state = optimizer.init(params).replace(
        utility_ema={"w": jnp.array([1.0], dtype=jnp.float32)},
        utility_age={"w": jnp.array([maximum], dtype=jnp.int32)},
        step=jnp.array(maximum, dtype=jnp.int32),
    )
    key = jr.key(5)
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    result = update(
        state,
        params,
        gradients,
        key,
        noise={"w": jnp.zeros(1, dtype=jnp.float32)},
    )

    assert bool(result.metrics["update_applied"])
    assert int(result.state.step) == maximum
    assert int(result.state.utility_age["w"][0]) == maximum
    assert float(result.corrected_utility["w"][0]) == pytest.approx(1.0)
    assert float(result.params["w"][0]) != float(params["w"][0])
    assert not jnp.array_equal(jr.key_data(result.next_key), jr.key_data(key))


def test_safe_profile_global_saturation_does_not_consume_inactive_clock() -> None:
    with jax.enable_x64():
        maximum = jnp.iinfo(jnp.int32).max
        params = {"w": jnp.array([2.0, 3.0], dtype=jnp.float64)}
        gradients = {"w": jnp.array([-0.25, -0.5], dtype=jnp.float64)}
        optimizer = CanonicalUPGD(
            CanonicalUPGDConfig(
                step_size=0.1,
                utility_decay=float(jnp.nextafter(jnp.float32(1.0), jnp.float32(0.0))),
                noise_std=0.0,
                profile="safe_extended",
                normalization="global",
            )
        )
        state = optimizer.init(params).replace(
            utility_ema={"w": jnp.ones(2, dtype=jnp.float64)},
            utility_age={"w": jnp.array([maximum, 0], dtype=jnp.int32)},
            step=jnp.array(maximum, dtype=jnp.int32),
        )
        key = jr.key(23)
        result = jax.jit(optimizer.update)(
            state,
            params,
            gradients,
            key,
            mask={"w": jnp.array([False, True])},
            noise={"w": jnp.zeros(2, dtype=jnp.float64)},
        )

        assert bool(result.metrics["update_applied"])
        chex.assert_trees_all_equal(
            result.state.utility_age["w"],
            jnp.array([maximum, 1], dtype=jnp.int32),
        )
        assert float(result.params["w"][0]) != float(params["w"][0])
        assert float(result.params["w"][1]) != float(params["w"][1])
        assert not jnp.array_equal(jr.key_data(result.next_key), jr.key_data(key))


@pytest.mark.parametrize("mode", ["protecting", "non_protecting"])
def test_safe_profile_exhausted_active_clock_rejects_available_peer_atomically(
    mode: str,
) -> None:
    with jax.enable_x64():
        maximum = jnp.iinfo(jnp.int32).max
        params = {
            "available": jnp.array([2.0], dtype=jnp.float64),
            "exhausted": jnp.array([3.0], dtype=jnp.float64),
        }
        gradients = {
            "available": jnp.array([-0.25], dtype=jnp.float64),
            "exhausted": jnp.array([-0.5], dtype=jnp.float64),
        }
        optimizer = CanonicalUPGD(
            _legacy_sink_decay_config(
                step_size=0.1,
                noise_std=0.0,
                mode=mode,  # type: ignore[arg-type]
                profile="safe_extended",
                normalization="global",
            )
        )
        state = optimizer.init(params).replace(
            utility_ema=jax.tree.map(jnp.ones_like, params),
            utility_age={
                "available": jnp.array([maximum - 1], dtype=jnp.int32),
                "exhausted": jnp.array([maximum], dtype=jnp.int32),
            },
            step=jnp.array(maximum, dtype=jnp.int32),
        )
        key = jr.key(29)
        result = jax.jit(optimizer.update)(
            state,
            params,
            gradients,
            key,
            noise=jax.tree.map(jnp.zeros_like, params),
        )

        assert not bool(result.metrics["update_applied"])
        chex.assert_trees_all_equal(result.params, params)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(
            jr.key_data(result.next_key),
            jr.key_data(key),
        )
        zero_tree = jax.tree.map(jnp.zeros_like, params)
        chex.assert_trees_all_equal(result.scaled_utility, zero_tree)
        chex.assert_trees_all_equal(result.corrected_utility, zero_tree)
        chex.assert_trees_all_equal(result.perturbation, zero_tree)


@pytest.mark.parametrize("profile", ["paper_global", "safe_extended"])
@pytest.mark.parametrize("mode", ["protecting", "non_protecting"])
@pytest.mark.parametrize("compiled", [False, True])
def test_lifetime_refusal_is_reverse_mode_atomic_when_proposal_is_nonfinite(
    profile: str,
    mode: str,
    compiled: bool,
) -> None:
    with jax.enable_x64():
        maximum = jnp.iinfo(jnp.int32).max
        if profile == "paper_global":
            param = jnp.array([0.0], dtype=jnp.float64)
            gradient = jnp.array([0.0], dtype=jnp.float64)
            utility = jnp.array([0.0], dtype=jnp.float64)
        else:
            param = jnp.array([2.0], dtype=jnp.float64)
            gradient = jnp.array([-0.5], dtype=jnp.float64)
            utility = jnp.array([jnp.finfo(jnp.float64).max], dtype=jnp.float64)
        params = {"w": param}
        optimizer = CanonicalUPGD(
            _legacy_sink_decay_config(
                step_size=0.1,
                noise_std=0.0,
                mode=mode,  # type: ignore[arg-type]
                profile=profile,  # type: ignore[arg-type]
                normalization="global",
            )
        )
        state = optimizer.init(params).replace(
            utility_ema={"w": utility},
            utility_age={"w": jnp.array([maximum], dtype=jnp.int32)},
            step=jnp.array(maximum, dtype=jnp.int32),
        )
        key = jr.key(31)
        noise = {"w": jnp.zeros(1, dtype=jnp.float64)}
        result = optimizer.update(
            state,
            params,
            {"w": gradient},
            key,
            noise=noise,
        )

        assert not bool(result.metrics["update_applied"])
        chex.assert_trees_all_equal(result.params, params)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(result.next_key, key)
        zero_tree = jax.tree.map(jnp.zeros_like, params)
        chex.assert_trees_all_equal(result.scaled_utility, zero_tree)
        chex.assert_trees_all_equal(result.corrected_utility, zero_tree)
        chex.assert_trees_all_equal(result.perturbation, zero_tree)

        def rejected_identity_loss(param_leaf, utility_leaf):
            local_state = state.replace(utility_ema={"w": utility_leaf})
            update = optimizer.update(
                local_state,
                {"w": param_leaf},
                {"w": gradient},
                key,
                noise=noise,
            )
            return jnp.sum(update.params["w"]) + jnp.sum(update.state.utility_ema["w"])

        gradient_fn = jax.grad(rejected_identity_loss, argnums=(0, 1))
        if compiled:
            gradient_fn = jax.jit(gradient_fn)
        param_gradient, utility_gradient = gradient_fn(param, utility)
        chex.assert_trees_all_equal(param_gradient, jnp.ones_like(param))
        chex.assert_trees_all_equal(utility_gradient, jnp.ones_like(utility))


@pytest.mark.parametrize("profile", ["paper_global", "safe_extended"])
def test_corrupt_negative_counters_do_not_overflow_to_the_lifetime_limit(
    profile: str,
) -> None:
    params = {"w": jnp.array([2.0], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.9,
            noise_std=0.0,
            profile=profile,  # type: ignore[arg-type]
            normalization="global",
        )
    )
    state = optimizer.init(params).replace(
        utility_age={"w": jnp.array([-1], dtype=jnp.int32)},
        step=jnp.array(-1, dtype=jnp.int32),
    )
    result = jax.jit(optimizer.update)(
        state,
        params,
        {"w": jnp.array([-0.5], dtype=jnp.float32)},
        jr.key(0),
        noise={"w": jnp.zeros(1, dtype=jnp.float32)},
    )

    assert int(result.state.step) == 0
    assert int(result.state.utility_age["w"][0]) == 0


@pytest.mark.parametrize(
    ("profile", "normalized"),
    [
        (
            "official_experiment_local",
            [[0.6, 0.8], [0.0, 1.0]],
        ),
        (
            "paper_local_literal",
            [[1.2, 1.6], [0.0, 2.0]],
        ),
    ],
)
def test_local_source_profiles_distinguish_nonzero_beta_denominator(
    profile: str,
    normalized: list[list[float]],
) -> None:
    """Official code and literal Appendix E differ before EMA convergence."""

    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    gradients = {
        "w": -jnp.array(
            [
                [3.0, 4.0],
                [0.0, 5.0],
            ],
            dtype=jnp.float32,
        )
    }
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.5,
            noise_std=0.0,
            profile=profile,  # type: ignore[arg-type]
            normalization="local",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )
    chex.assert_trees_all_close(
        result.scaled_utility["w"],
        jax.nn.sigmoid(jnp.asarray(normalized, dtype=jnp.float32)),
        atol=1e-6,
    )
    assert jnp.isnan(result.metrics["global_maximum_utility"])


def test_global_normalization_spans_all_pytree_leaves() -> None:
    params = {
        "a": jnp.array([1.0], dtype=jnp.float32),
        "b": jnp.array([1.0], dtype=jnp.float32),
    }
    gradients = {
        "a": jnp.array([-1.0], dtype=jnp.float32),
        "b": jnp.array([-2.0], dtype=jnp.float32),
    }
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(utility_decay=0.0, noise_std=0.0))
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )
    chex.assert_trees_all_close(
        result.scaled_utility,
        {
            "a": jax.nn.sigmoid(jnp.array([0.5])),
            "b": jax.nn.sigmoid(jnp.array([1.0])),
        },
    )
    assert float(result.metrics["global_maximum_utility"]) == pytest.approx(2.0)


def test_zero_utility_decay_does_not_multiply_inf_ema() -> None:
    """utility_decay=0 times an infinite EMA is NaN and would be committed."""
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.array([-1.0, -0.5], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(utility_decay=0.0, noise_std=0.0))
    state = optimizer.init(params).replace(
        utility_ema={"w": jnp.full(2, jnp.inf, dtype=jnp.float32)}
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = optimizer.update(state, params, gradients, jr.key(0))
    chex.assert_tree_all_finite(result.state.utility_ema)
    chex.assert_tree_all_finite(result.params)


def test_nonzero_static_decay_preserves_legacy_hlo() -> None:
    value = jnp.ones((4,), dtype=jnp.float32)
    decay = 0.999

    legacy_hlo = (
        jax.jit(lambda leaf: decay * leaf)
        .lower(value)
        .compiler_ir(dialect="hlo")
        .as_hlo_text()
    )
    guarded_hlo = (
        jax.jit(lambda leaf: _static_zero_scale(decay, leaf))
        .lower(value)
        .compiler_ir(dialect="hlo")
        .as_hlo_text()
    )

    assert legacy_hlo.splitlines()[1:] == guarded_hlo.splitlines()[1:]


def test_paper_global_all_negative_uses_signed_maximum_and_reverses_order() -> None:
    """Pin the literal, potentially undesirable all-negative source equation."""

    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="paper_global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )

    chex.assert_trees_all_close(
        result.scaled_utility["w"],
        jax.nn.sigmoid(jnp.array([1.0, 2.0], dtype=jnp.float32)),
    )
    assert float(result.metrics["global_maximum_utility"]) == pytest.approx(-1.0)
    chex.assert_tree_all_finite(result.params)
    chex.assert_tree_all_finite(result.state)
    chex.assert_tree_all_finite(result.scaled_utility)


@pytest.mark.parametrize(
    "profile",
    [
        "paper_global",
        "official_readme_global",
        "official_experiment_global",
    ],
)
def test_source_global_all_zero_is_explicitly_undefined(profile: str) -> None:
    """The paper and released global code leave zero-over-zero unguarded."""

    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.zeros(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile=profile,  # type: ignore[arg-type]
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )

    assert jnp.all(jnp.isnan(result.scaled_utility["w"]))
    assert jnp.all(jnp.isnan(result.params["w"]))
    assert float(result.metrics["global_maximum_utility"]) == pytest.approx(0.0)


def test_safe_global_all_zero_uses_documented_finite_guard() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.zeros(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="safe_extended",
            normalization="global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )

    chex.assert_trees_all_close(
        result.scaled_utility["w"],
        jnp.full(2, 0.5, dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.params)
    chex.assert_tree_all_finite(result.state)
    chex.assert_tree_all_finite(result.scaled_utility)


def test_official_experiment_local_all_zero_uses_torch_normalize_floor() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.zeros(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="official_experiment_local",
            normalization="local",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )

    chex.assert_trees_all_close(
        result.scaled_utility["w"],
        jnp.full(2, 0.5, dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.params)


def test_sigma_zero_nonprotecting_is_exact_sgdw() -> None:
    params = {"w": jnp.array([1.5, -0.5], dtype=jnp.float32)}
    gradients = {"w": jnp.array([0.2, -0.4], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.03,
            utility_decay=0.7,
            noise_std=0.0,
            weight_decay=0.2,
            mode="non_protecting",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(2),
    )
    expected = params["w"] * (1.0 - 0.03 * 0.2) - 0.03 * gradients["w"]
    chex.assert_trees_all_close(result.params["w"], expected)


def test_parameter_mask_uses_plain_sgdw_outside_upgd_scope() -> None:
    params = {
        "trunk": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "head": jnp.array([3.0], dtype=jnp.float32),
    }
    gradients = {
        "trunk": jnp.array([-1.0, -1.0], dtype=jnp.float32),
        "head": jnp.array([2.0], dtype=jnp.float32),
    }
    noise = {
        "trunk": jnp.array([0.5, 0.5], dtype=jnp.float32),
        "head": jnp.array([100.0], dtype=jnp.float32),
    }
    mask = {
        "trunk": jnp.array([True, True]),
        "head": jnp.array([False]),
    }
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.1,
            utility_decay=0.0,
            weight_decay=0.2,
            profile="safe_extended",
            normalization="global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
        mask=mask,
        noise=noise,
    )

    expected_head = params["head"] * 0.98 - 0.1 * gradients["head"]
    chex.assert_trees_all_close(result.params["head"], expected_head)
    chex.assert_trees_all_equal(result.scaled_utility["head"], jnp.zeros(1))
    chex.assert_trees_all_equal(result.perturbation["head"], jnp.zeros(1))
    chex.assert_trees_all_equal(result.state.utility_age["head"], jnp.zeros(1))


def test_source_profiles_reject_masks_and_missing_gradient_leaves() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(profile="paper_global", noise_std=0.0))
    state = optimizer.init(params)

    with pytest.raises(ValueError, match="do not accept masks"):
        optimizer.update(
            state,
            params,
            gradients,
            jr.key(0),
            mask={"w": jnp.ones(2, dtype=jnp.bool_)},
        )
    with pytest.raises(ValueError, match="require every gradient"):
        optimizer.update(
            state,
            params,
            {"w": None},
            jr.key(0),
        )


def test_source_profile_dynamic_nonfinite_gradient_fails_closed() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.array([-2.0, jnp.nan], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="paper_global",
            normalization="global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
        noise={"w": jnp.zeros(2, dtype=jnp.float32)},
    )

    assert float(result.params["w"][1]) == pytest.approx(float(params["w"][1]))
    assert float(result.scaled_utility["w"][1]) == pytest.approx(0.0)
    assert float(result.metrics["eligible_parameter_count"]) == pytest.approx(1.0)
    assert float(result.metrics["nonfinite_or_missing_count"]) == pytest.approx(1.0)
    chex.assert_trees_all_equal(
        result.state.utility_age["w"],
        jnp.ones(2, dtype=jnp.int32),
    )


def test_safe_local_mask_excludes_stale_utility_from_row_norm_and_metrics() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="safe_extended",
            normalization="local",
        )
    )
    first = optimizer.update(
        optimizer.init(params),
        params,
        {"w": jnp.array([-1.0, -100.0], dtype=jnp.float32)},
        jr.key(0),
        mask={"w": jnp.array([True, True])},
        noise={"w": jnp.zeros(2, dtype=jnp.float32)},
    )
    second = optimizer.update(
        first.state,
        params,
        {"w": jnp.array([-2.0, -999.0], dtype=jnp.float32)},
        first.next_key,
        mask={"w": jnp.array([True, False])},
        noise={"w": jnp.zeros(2, dtype=jnp.float32)},
    )

    chex.assert_trees_all_close(
        second.state.utility_ema["w"],
        jnp.array([2.0, 100.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        second.scaled_utility["w"],
        jnp.array([jax.nn.sigmoid(1.0), 0.0], dtype=jnp.float32),
    )
    assert float(second.metrics["eligible_parameter_count"]) == pytest.approx(1.0)
    assert float(second.metrics["mean_scaled_utility"]) == pytest.approx(float(jax.nn.sigmoid(1.0)))


def test_safe_global_nonfinite_gradient_excludes_stale_global_maximum() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="safe_extended",
            normalization="global",
        )
    )
    first = optimizer.update(
        optimizer.init(params),
        params,
        {"w": jnp.array([-1.0, -100.0], dtype=jnp.float32)},
        jr.key(0),
        noise={"w": jnp.zeros(2, dtype=jnp.float32)},
    )
    second = optimizer.update(
        first.state,
        params,
        {"w": jnp.array([-2.0, jnp.nan], dtype=jnp.float32)},
        first.next_key,
        noise={"w": jnp.zeros(2, dtype=jnp.float32)},
    )

    chex.assert_trees_all_close(
        second.state.utility_ema["w"],
        jnp.array([2.0, 100.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        second.scaled_utility["w"],
        jnp.array([jax.nn.sigmoid(1.0), 0.0], dtype=jnp.float32),
    )
    assert float(second.metrics["global_maximum_utility"]) == pytest.approx(2.0)
    assert float(second.metrics["eligible_parameter_count"]) == pytest.approx(1.0)
    assert float(second.metrics["nonfinite_or_missing_count"]) == pytest.approx(1.0)
    assert float(second.params["w"][1]) == pytest.approx(float(params["w"][1]))


def test_safe_active_element_clock_pauses_and_resumes_per_element() -> None:
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    gradients = {"w": jnp.array([-2.0, -4.0], dtype=jnp.float32)}
    zero_noise = {"w": jnp.zeros(2, dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.5,
            noise_std=0.0,
            profile="safe_extended",
            normalization="global",
        )
    )
    first = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
        mask={"w": jnp.array([True, False])},
        noise=zero_noise,
    )
    second = optimizer.update(
        first.state,
        params,
        gradients,
        first.next_key,
        mask={"w": jnp.array([False, True])},
        noise=zero_noise,
    )
    third = optimizer.update(
        second.state,
        params,
        gradients,
        second.next_key,
        mask={"w": jnp.array([True, True])},
        noise=zero_noise,
    )

    chex.assert_trees_all_equal(
        first.state.utility_age["w"],
        jnp.array([1, 0], dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        second.state.utility_age["w"],
        jnp.array([1, 1], dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        third.state.utility_age["w"],
        jnp.array([2, 2], dtype=jnp.int32),
    )
    chex.assert_trees_all_close(
        third.corrected_utility["w"],
        jnp.array([2.0, 4.0], dtype=jnp.float32),
    )


def test_missing_and_nonfinite_gradients_fail_closed() -> None:
    params = {
        "missing": jnp.array([1.0], dtype=jnp.float32),
        "mixed": jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32),
    }
    gradients = {
        "missing": None,
        "mixed": jnp.array([jnp.nan, jnp.inf, 1.0], dtype=jnp.float32),
    }
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.1,
            utility_decay=0.0,
            noise_std=0.0,
            profile="safe_extended",
            normalization="global",
        )
    )
    result = optimizer.update(
        optimizer.init(params),
        params,
        gradients,
        jr.key(0),
    )

    chex.assert_trees_all_equal(result.params["missing"], params["missing"])
    chex.assert_trees_all_equal(result.params["mixed"][:2], params["mixed"][:2])
    assert float(result.params["mixed"][2]) < float(params["mixed"][2])
    chex.assert_trees_all_equal(
        result.state.utility_age["mixed"],
        jnp.array([0, 0, 1], dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        result.scaled_utility["missing"],
        jnp.zeros(1, dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.scaled_utility["mixed"][:2],
        jnp.zeros(2, dtype=jnp.float32),
    )
    assert float(result.metrics["eligible_parameter_count"]) == pytest.approx(1.0)
    assert float(result.metrics["nonfinite_or_missing_count"]) == pytest.approx(3.0)


def test_update_is_jittable_and_rng_is_deterministic() -> None:
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    gradients = {"w": jnp.array([0.2, -0.3], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(noise_std=0.2, utility_decay=0.5))
    state = optimizer.init(params)
    update = jax.jit(optimizer.update)

    first = update(state, params, gradients, jr.key(7))
    second = update(state, params, gradients, jr.key(7))
    chex.assert_trees_all_close(first.params, second.params)
    chex.assert_trees_all_close(first.state, second.state)
    chex.assert_trees_all_close(first.scaled_utility, second.scaled_utility)
    chex.assert_trees_all_close(first.corrected_utility, second.corrected_utility)
    chex.assert_trees_all_close(first.perturbation, second.perturbation)
    chex.assert_trees_all_close(first.metrics, second.metrics)
    chex.assert_trees_all_equal(
        jr.key_data(first.next_key),
        jr.key_data(second.next_key),
    )
    chex.assert_tree_all_finite(first.params)
    chex.assert_tree_all_finite(first.state)


def test_scan_matches_explicit_updates_and_prng_progression() -> None:
    """Functional state and typed keys have identical loop/scan semantics."""

    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    gradients = jnp.array(
        [
            [0.2, -0.3],
            [-0.1, 0.4],
            [0.5, 0.25],
        ],
        dtype=jnp.float32,
    )
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=0.02,
            utility_decay=0.8,
            noise_std=0.1,
            weight_decay=0.03,
        )
    )
    initial_state = optimizer.init(params)
    initial_key = jr.key(23)

    loop_state = initial_state
    loop_params = params
    loop_key = initial_key
    loop_values = []
    for gradient in gradients:
        result = optimizer.update(
            loop_state,
            loop_params,
            {"w": gradient},
            loop_key,
        )
        loop_state = result.state
        loop_params = result.params
        loop_key = result.next_key
        loop_values.append(result.params["w"])

    def scan_step(carry, gradient):
        state, current_params, key = carry
        result = optimizer.update(
            state,
            current_params,
            {"w": gradient},
            key,
        )
        return (
            result.state,
            result.params,
            result.next_key,
        ), result.params["w"]

    (scan_state, scan_params, scan_key), scan_values = jax.jit(
        lambda: jax.lax.scan(
            scan_step,
            (initial_state, params, initial_key),
            gradients,
        )
    )()

    chex.assert_trees_all_close(scan_state, loop_state)
    chex.assert_trees_all_close(scan_params, loop_params)
    chex.assert_trees_all_close(scan_values, jnp.stack(loop_values))
    chex.assert_trees_all_equal(jr.key_data(scan_key), jr.key_data(loop_key))


def test_state_roundtrips_through_repository_checkpoint(tmp_path) -> None:
    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(noise_std=0.0, utility_decay=0.8))
    template = optimizer.init(params)
    updated = optimizer.update(
        template,
        params,
        {"w": jnp.array([-1.0, 0.25], dtype=jnp.float32)},
        jr.key(0),
    ).state

    save_checkpoint(updated, tmp_path / "canonical_upgd")
    loaded, metadata = load_checkpoint(template, tmp_path / "canonical_upgd")
    chex.assert_trees_all_close(loaded, updated)
    assert metadata == {}


def test_saturated_protecting_gate_skips_overflowed_direction() -> None:
    """gate=1 times an overflowed (g+xi) is 0*inf=NaN without a zero-scale skip."""
    from alberta_framework.core.canonical_upgd import _skip_zero_scale

    gate = jnp.asarray(1.0, dtype=jnp.float32)
    overflowed = jnp.asarray(jnp.inf, dtype=jnp.float32)
    raw = (overflowed + 0.0) * (1.0 - gate)
    assert not bool(jnp.isfinite(raw))

    fixed = _skip_zero_scale(1.0 - gate, overflowed + 0.0)
    assert bool(jnp.isfinite(fixed))
    chex.assert_trees_all_close(fixed, jnp.zeros_like(overflowed))

    # Non-protecting: only the noise term is gated.
    raw_noise = overflowed * (1.0 - gate)
    assert not bool(jnp.isfinite(raw_noise))
    fixed_noise = _skip_zero_scale(1.0 - gate, overflowed)
    assert bool(jnp.isfinite(fixed_noise))


def test_saturated_gate_accepts_finite_inputs_with_overflowing_direction() -> None:
    """A protected coordinate need not form the sum of two finite huge inputs."""
    params = jnp.array([1e-20, 1e-20], dtype=jnp.float32)
    gradients = jnp.array([3e38, 1.0], dtype=jnp.float32)
    noise = jnp.array([3e38, 0.0], dtype=jnp.float32)
    optimizer = CanonicalUPGD(CanonicalUPGDConfig(utility_decay=0.0, step_size=1e-5))
    result = optimizer.update(optimizer.init(params), params, gradients, jr.key(1), noise=noise)
    assert bool(result.metrics["update_applied"])
    assert float(result.scaled_utility[0]) == 1.0
    assert float(result.params[0]) == float(params[0])
    assert float(result.params[1]) == pytest.approx(-5e-6)
    chex.assert_tree_all_finite(result.params)
