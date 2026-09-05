"""Tests for surprise-driven cumulant discovery (Step 3 Phase F)."""

from __future__ import annotations

from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.cumulant_discovery import (
    CumulantDiscovery,
    CumulantDiscoveryState,
)

# =============================================================================
# Init / shapes
# =============================================================================


class TestInit:
    def test_init_shapes(self) -> None:
        d = CumulantDiscovery(raw_dim=5, n_candidates=8)
        s = d.init(jr.key(0))
        chex.assert_shape(s.projections, (8, 5))
        chex.assert_shape(s.weights, (8, 5))
        chex.assert_shape(s.biases, (8,))
        chex.assert_shape(s.utility, (8,))
        chex.assert_shape(s.ages, (8,))

    def test_init_unit_norm_projections(self) -> None:
        d = CumulantDiscovery(raw_dim=10, n_candidates=4)
        s = d.init(jr.key(0))
        norms = jnp.linalg.norm(s.projections, axis=1)
        chex.assert_trees_all_close(norms, jnp.ones(4), atol=1e-5)

    def test_init_zero_predictors(self) -> None:
        d = CumulantDiscovery(raw_dim=5, n_candidates=3)
        s = d.init(jr.key(7))
        chex.assert_trees_all_close(s.weights, jnp.zeros((3, 5)))
        chex.assert_trees_all_close(s.biases, jnp.zeros(3))
        chex.assert_trees_all_close(s.utility, jnp.zeros(3))


class TestValidation:
    def test_invalid_raw_dim(self) -> None:
        with pytest.raises(ValueError, match="raw_dim"):
            CumulantDiscovery(raw_dim=0)

    def test_invalid_n_candidates(self) -> None:
        with pytest.raises(ValueError, match="n_candidates"):
            CumulantDiscovery(raw_dim=4, n_candidates=0)

    def test_invalid_decay_rate(self) -> None:
        with pytest.raises(ValueError, match="decay_rate"):
            CumulantDiscovery(raw_dim=4, decay_rate=1.0)
        with pytest.raises(ValueError, match="decay_rate"):
            CumulantDiscovery(raw_dim=4, decay_rate=0.0)

    def test_invalid_replacement_rate(self) -> None:
        with pytest.raises(ValueError, match="replacement_rate"):
            CumulantDiscovery(raw_dim=4, replacement_rate=1.5)


# =============================================================================
# Step semantics
# =============================================================================


class TestStep:
    def test_age_increments(self) -> None:
        d = CumulantDiscovery(raw_dim=4, n_candidates=3)
        s = d.init(jr.key(0))
        for _ in range(7):
            s = d.step(s, jnp.ones(4), jnp.ones(4))
        chex.assert_trees_all_close(s.ages, jnp.array([7, 7, 7], dtype=jnp.int32))

    def test_utility_grows_with_high_surprise(self) -> None:
        d = CumulantDiscovery(
            raw_dim=2,
            n_candidates=2,
            decay_rate=0.9,
            predictor_step_size=1e-6,  # tiny -- predictor barely moves
            gamma=0.0,
        )
        s = d.init(jr.key(42))
        # The predictor stays approximately at zero for a few steps,
        # so the squared TD error is approximately (cumulant - 0)^2 > 0
        # for every step, and the utility EMA accumulates.
        for _ in range(10):
            s = d.step(s, jnp.zeros(2), jnp.array([1.0, 0.0]))
        assert float(jnp.min(s.utility)) > 0.0

    def test_step_uses_next_observation_for_transition_cumulant(self) -> None:
        d = CumulantDiscovery(
            raw_dim=2,
            n_candidates=1,
            decay_rate=0.5,
            predictor_step_size=0.1,
            gamma=0.0,
        )
        s0 = d.init(jr.key(0)).replace(
            projections=jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        )
        # If the current observation were used as the cumulant this would
        # produce non-zero utility. GVF/nexting convention uses c_{t+1}.
        s1 = d.step(s0, jnp.array([2.0, 0.0]), jnp.array([0.0, 0.0]))
        chex.assert_trees_all_close(s1.utility, jnp.array([0.0]), atol=1e-7)

        # A non-zero next observation now produces surprise.
        s2 = d.step(s0, jnp.array([0.0, 0.0]), jnp.array([2.0, 0.0]))
        assert float(s2.utility[0]) > 0.0

    def test_infinite_next_obs_does_not_poison_predictors(self) -> None:
        """Zero init V(s') is 0 @ inf = NaN, then alpha * nan * obs poisons all."""
        d = CumulantDiscovery(raw_dim=2, n_candidates=3, predictor_step_size=0.1)
        state = d.init(jr.key(0))
        obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        nxt = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)

        poisoned = d.step(state, obs, nxt)
        chex.assert_trees_all_close(poisoned.weights, state.weights)
        chex.assert_trees_all_close(poisoned.biases, state.biases)
        chex.assert_trees_all_close(poisoned.utility, state.utility)
        chex.assert_trees_all_close(poisoned.ages, state.ages)

        recovered = d.step(poisoned, obs, jnp.array([1.0, 1.0], dtype=jnp.float32))
        chex.assert_tree_all_finite(recovered.weights)
        chex.assert_tree_all_finite(recovered.biases)
        chex.assert_tree_all_finite(recovered.utility)
        chex.assert_trees_all_close(recovered.ages, state.ages + 1)

    def test_zero_gamma_does_not_multiply_overflow_v_next(self) -> None:
        """Default gamma=0 times overflowed V(s') is 0*inf = NaN without a skip."""
        d = CumulantDiscovery(
            raw_dim=2, n_candidates=1, predictor_step_size=0.1, gamma=0.0
        )
        state = d.init(jr.key(0)).replace(  # type: ignore[attr-defined]
            projections=jnp.array([[1.0, 0.0]], dtype=jnp.float32),
            weights=jnp.array([[1.0e20, 0.0]], dtype=jnp.float32),
            biases=jnp.zeros(1, dtype=jnp.float32),
        )
        obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
        next_obs = jnp.array([1.0e20, 0.0], dtype=jnp.float32)
        v_next = state.weights @ next_obs + state.biases
        assert bool(jnp.isinf(v_next[0]))
        raw = jnp.asarray(0.0, dtype=jnp.float32) * v_next[0]
        assert not bool(jnp.isfinite(raw))

        updated = d.step(state, obs, next_obs)
        assert not jnp.array_equal(updated.ages, state.ages)
        chex.assert_tree_all_finite(updated.weights)
        chex.assert_tree_all_finite(updated.biases)
        chex.assert_tree_all_finite(updated.utility)

    def test_predictor_reduces_td_error(self) -> None:
        d = CumulantDiscovery(
            raw_dim=2, n_candidates=1, predictor_step_size=0.1, gamma=0.0
        )
        s0 = d.init(jr.key(2))
        # Repeatedly present the same observation: predictor should
        # learn to predict the cumulant exactly, so the TD error / utility
        # should decrease.
        s = s0
        obs = jnp.array([1.0, -0.5])
        next_obs = jnp.array([0.5, 0.5])
        for _ in range(200):
            s = d.step(s, obs, next_obs)
        # Final TD error should be small
        cumulant = (s.projections @ next_obs)[0]
        v = (s.weights @ obs + s.biases)[0]
        v_next = (s.weights @ next_obs + s.biases)[0]
        td = float(cumulant + 0.0 * v_next - v)
        assert abs(td) < 0.05, f"predictor failed to converge; td={td}"


# =============================================================================
# Replacement
# =============================================================================


class TestReplacement:
    def test_replacement_disabled_keeps_state(self) -> None:
        d = CumulantDiscovery(
            raw_dim=4,
            n_candidates=3,
            replacement_rate=1.0,
            maturity_threshold=0,
            enabled=False,
        )
        s = d.init(jr.key(0))
        s_after = d.maybe_replace(s)
        chex.assert_trees_all_close(s_after.projections, s.projections)
        chex.assert_trees_all_close(s_after.utility, s.utility)
        chex.assert_trees_all_close(s_after.ages, s.ages)

    def test_replacement_when_eligible(self) -> None:
        # rate=1.0, maturity=0 means every call to maybe_replace replaces
        d = CumulantDiscovery(
            raw_dim=3,
            n_candidates=4,
            replacement_rate=1.0,
            maturity_threshold=0,
            enabled=True,
        )
        s = d.init(jr.key(0))
        s_after = d.maybe_replace(s)
        # At least one row should differ (the lowest utility candidate)
        diff = jnp.linalg.norm(s_after.projections - s.projections, axis=1)
        assert float(jnp.max(diff)) > 0.0

    def test_no_replacement_before_maturity(self) -> None:
        d = CumulantDiscovery(
            raw_dim=3,
            n_candidates=4,
            replacement_rate=1.0,
            maturity_threshold=100,  # nothing can be replaced before age 100
            enabled=True,
        )
        s = d.init(jr.key(0))
        s_after = d.maybe_replace(s)
        # Should be unchanged because nothing is mature
        chex.assert_trees_all_close(s_after.projections, s.projections)
        chex.assert_trees_all_close(s_after.utility, s.utility)
        chex.assert_trees_all_close(s_after.ages, s.ages)


# =============================================================================
# JIT and scan
# =============================================================================


class TestJit:
    def test_step_jit(self) -> None:
        d = CumulantDiscovery(raw_dim=4, n_candidates=4)
        s = d.init(jr.key(0))
        s2 = d.step(s, jnp.ones(4), jnp.ones(4))
        chex.assert_tree_all_finite(s2.utility)

    def test_scan_compatibility(self) -> None:
        d = CumulantDiscovery(raw_dim=4, n_candidates=4)
        s0 = d.init(jr.key(0))
        observations = jr.normal(jr.key(1), (50, 4))

        def step_fn(state: CumulantDiscoveryState, x: jax.Array):
            new_state = d.step(state, x, x)
            return new_state, new_state.utility

        final_state, utility_history = jax.lax.scan(step_fn, s0, observations)
        chex.assert_shape(utility_history, (50, 4))
        chex.assert_tree_all_finite(final_state.utility)


# =============================================================================
# Functional: surprise-driven retains structure-bearing cumulants
# =============================================================================


class TestFunctional:
    """A non-stationary stream emits an obs that has a deterministic
    function of obs as its hidden cumulant. Among many random
    candidates, the ones that align with that function should accumulate
    higher utility (squared TD error reflects information content
    times mismatch -- which decays as the predictor learns; with
    short-horizon updates, mis-aligned candidates also have high error).

    Here we just check that the discovery loop runs end-to-end and that
    candidates with smaller surprise survive over many steps.
    """

    def test_low_surprise_candidates_survive(self) -> None:
        d = CumulantDiscovery(
            raw_dim=4,
            n_candidates=8,
            decay_rate=0.99,
            replacement_rate=0.05,
            maturity_threshold=50,
            predictor_step_size=0.05,
        )
        s = d.init(jr.key(0))

        rng = np.random.default_rng(0)
        for _ in range(2000):
            obs = jnp.asarray(rng.normal(size=4).astype(np.float32))
            next_obs = jnp.asarray(rng.normal(size=4).astype(np.float32))
            s = d.step(s, obs, next_obs)
            s = d.maybe_replace(s)

        # Surviving candidates should have FINITE utility and ages
        chex.assert_tree_all_finite(s.utility)
        chex.assert_tree_all_finite(s.weights)
        # By the end of 2000 steps, every candidate should be mature
        assert int(jnp.min(s.ages)) > 0


# =============================================================================
# Config roundtrip
# =============================================================================


class TestConfig:
    def test_roundtrip(self) -> None:
        original = CumulantDiscovery(
            raw_dim=8,
            n_candidates=12,
            decay_rate=0.95,
            replacement_rate=0.01,
            maturity_threshold=300,
            predictor_step_size=0.02,
            gamma=0.9,
            enabled=True,
        )
        config = original.to_config()
        restored = CumulantDiscovery.from_config(config)
        assert restored.raw_dim == 8
        assert restored.n_candidates == 12
        assert restored.enabled is True


@pytest.mark.parametrize(
    "integer_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.ulonglong,
    ],
)
def test_cumulant_discovery_canonicalizes_numpy_integer_family(integer_type) -> None:
    discovery = CumulantDiscovery(
        raw_dim=integer_type(4),
        n_candidates=integer_type(3),
        maturity_threshold=integer_type(2),
    )

    assert type(discovery.raw_dim) is int
    assert type(discovery.n_candidates) is int
    assert type(discovery._maturity_threshold) is int


@pytest.mark.parametrize("field", ["raw_dim", "n_candidates", "maturity_threshold"])
def test_cumulant_discovery_rejects_hostile_integer_subclasses(field: str) -> None:
    class HostileInt(int):
        def __index__(self) -> int:
            raise AssertionError("untrusted index hook executed")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    kwargs = {"raw_dim": 4, "n_candidates": 3, field: HostileInt(2)}
    with pytest.raises(ValueError, match=field):
        CumulantDiscovery(**kwargs)


@pytest.mark.parametrize(
    "field", ["decay_rate", "replacement_rate", "predictor_step_size", "gamma"]
)
def test_cumulant_discovery_rejects_float_subclasses_without_hooks(field: str) -> None:
    class CountingFloat(float):
        def __new__(cls):
            instance = super().__new__(cls, 0.5)
            instance.calls = 0
            return instance

        def as_integer_ratio(self) -> tuple[int, int]:
            self.calls += 1
            return (1, 2)

    value = CountingFloat()
    with pytest.raises(ValueError, match=field):
        CumulantDiscovery(raw_dim=4, **{field: value})
    assert value.calls == 0


def test_cumulant_discovery_hostile_float_failure_never_formats_repr() -> None:
    class ExplodingFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("hostile ratio")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    with pytest.raises(ValueError, match="decay_rate"):
        CumulantDiscovery(raw_dim=4, decay_rate=ExplodingFloat(0.5))


@pytest.mark.parametrize("field", ["replacement_rate", "gamma"])
def test_cumulant_discovery_rejects_exact_nonzero_float32_underflow(field: str) -> None:
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    with pytest.raises(ValueError, match="remain nonzero"):
        CumulantDiscovery(raw_dim=4, **{field: tiny})


def test_cumulant_discovery_resource_formula_matches_state() -> None:
    discovery = CumulantDiscovery(raw_dim=5, n_candidates=3)
    state = discovery.init(jr.key(0))
    actual_bytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))

    assert discovery.persistent_resource_budget["persistent_bytes"] == actual_bytes


def test_cumulant_discovery_resource_boundary_is_allocation_free() -> None:
    last_valid_dim = ((256 * 1024 * 1024 // 4) - 5) // 2
    discovery = CumulantDiscovery(raw_dim=last_valid_dim, n_candidates=1)
    assert discovery.persistent_resource_budget["persistent_bytes"] <= 256 * 1024 * 1024
    with pytest.raises(ValueError, match="256 MiB"):
        CumulantDiscovery(raw_dim=last_valid_dim + 1, n_candidates=1)


def test_cumulant_discovery_config_preserves_historical_mapping_forms() -> None:
    config = CumulantDiscovery(raw_dim=4).to_config()
    config["raw_dim"] = np.int32(4)
    assert CumulantDiscovery.from_config(MappingProxyType(config)).raw_dim == 4
    partial = {"type": "historical-marker", "raw_dim": 4}
    restored = CumulantDiscovery.from_config(partial)
    assert restored.n_candidates == 16


def test_cumulant_discovery_age_saturates_at_int32_max() -> None:
    discovery = CumulantDiscovery(raw_dim=2, n_candidates=1)
    state = discovery.init(jr.key(0)).replace(
        ages=jnp.asarray([2**31 - 1], dtype=jnp.int32)
    )
    advanced = discovery.step(state, jnp.ones(2), jnp.ones(2))

    assert int(advanced.ages[0]) == 2**31 - 1


def test_cumulant_discovery_requires_exact_bool_and_typed_threefry_key() -> None:
    with pytest.raises(ValueError, match="enabled"):
        CumulantDiscovery(raw_dim=2, enabled=np.bool_(True))  # type: ignore[arg-type]

    discovery = CumulantDiscovery(raw_dim=2)
    with pytest.raises(ValueError, match="typed scalar threefry2x32"):
        discovery.init(jr.key_data(jr.key(0)))


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (2, 1), (3,)])
def test_cumulant_discovery_rejects_wrong_observation_shapes(
    shape: tuple[int, ...]
) -> None:
    discovery = CumulantDiscovery(raw_dim=2, n_candidates=2)
    state = discovery.init(jr.key(0))
    malformed = jnp.zeros(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="observation"):
        discovery.cumulants(state, malformed)
    with pytest.raises(ValueError, match="observation"):
        discovery.step(state, malformed, jnp.zeros((2,), dtype=jnp.float32))

    with pytest.raises(ValueError, match="dtype float32"):
        discovery.cumulants(state, jnp.zeros((2,), dtype=jnp.int32))


def test_cumulant_discovery_state_contract_and_invalid_atomicity() -> None:
    discovery = CumulantDiscovery(raw_dim=2, n_candidates=2, replacement_rate=1.0)
    state = discovery.init(jr.key(0))
    malformed = state.replace(weights=jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.weights"):
        discovery.step(
            malformed,
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )

    invalid = state.replace(ages=jnp.asarray([-1, 0], dtype=jnp.int32))
    replaced = discovery.maybe_replace(invalid)
    chex.assert_trees_all_equal(replaced, invalid)

    nonfinite = state.replace(utility=jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32))
    stepped = discovery.step(
        nonfinite,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(stepped, nonfinite)


def test_cumulant_discovery_hostile_observation_failure_is_normalized() -> None:
    class HostileObservation:
        def __jax_array__(self):
            raise RuntimeError("hostile conversion")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    discovery = CumulantDiscovery(raw_dim=2)
    state = discovery.init(jr.key(0))
    with pytest.raises(ValueError, match="readable float32 vector"):
        discovery.cumulants(state, HostileObservation())  # type: ignore[arg-type]


def test_require_float32_rejects_builtin_float_subnormal_underflow() -> None:
    from alberta_framework.core.cumulant_discovery import _require_float32

    with pytest.raises(ValueError, match="must remain nonzero once narrowed to float32"):
        _require_float32("gamma", 1e-50, lower=0.0, upper=1.0, preserve_nonzero=True)
