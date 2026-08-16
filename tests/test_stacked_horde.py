"""Tests for the stacked linear Horde (core/stacked_horde.py).

Covers exact TD(lambda) semantics (hand-computed two-step scenario),
convergence to analytic GVF fixed points on a 3-state cycle, NaN-cumulant
masking, per-decision IS composition, nexting-style multi-timescale
prediction, and — the module's reason to exist — demon-count scaling: the
demon axis is a stacked array axis, so program size is constant in
``n_demons`` and 1024 demons run a 2000-step scan in well under a second
after compile (the loop-unrolled hordes measured ~14 steps/s = ~140 s for
the same workload, with a ~144 s compile; see the scaling notes in
docs/evidence/methodology.md).
"""

import json
import math
import time
from decimal import Decimal
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    StackedLinearHorde,
    nexting_spec,
    run_stacked_horde_scan,
)


def _simple_config(n_demons=2, feature_dim=3, **kw) -> StackedHordeConfig:
    defaults = dict(
        n_demons=n_demons,
        feature_dim=feature_dim,
        gammas=(0.9,) * n_demons,
        lamdas=(0.5,) * n_demons,
        cumulant_indices=tuple(range(n_demons)),
        step_size=0.1,
    )
    defaults.update(kw)
    return StackedHordeConfig(**defaults)


class TestConfig:
    def test_validation(self):
        with pytest.raises(ValueError, match="n_demons"):
            _simple_config(n_demons=0, gammas=(), lamdas=(), cumulant_indices=())
        with pytest.raises(ValueError, match="length"):
            _simple_config(gammas=(0.9,))
        with pytest.raises(ValueError, match="gamma"):
            _simple_config(gammas=(1.5, 0.9))
        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=0.0)

    @pytest.mark.parametrize("step_size", [math.nan, math.inf, -math.inf])
    def test_step_size_must_be_finite_and_positive(self, step_size: float) -> None:
        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=step_size)

    @pytest.mark.parametrize(
        "step_size",
        [
            1,
            np.int8(1),
            np.int64(1),
            np.uint64(1),
            np.float16(0.5),
            np.float32(0.5),
            np.float64(0.5),
            np.longdouble(0.5),
            Fraction(1, 2),
        ],
    )
    def test_step_size_supported_reals_canonicalize_for_json(self, step_size: object) -> None:
        cfg = _simple_config(step_size=step_size)
        payload = cfg.to_config()

        assert type(cfg.step_size) is float
        json.dumps(payload, allow_nan=False)
        assert StackedHordeConfig.from_config(payload) == cfg

    def test_builtin_float_step_size_preserves_compatible_payload(self) -> None:
        cfg = _simple_config(step_size=0.1)

        assert type(cfg.step_size) is float
        assert cfg.step_size == 0.1

    def test_fraction_step_size_rounds_directly_to_float32(self) -> None:
        above_half_midpoint = Fraction(1, 2) + Fraction(1, 2**25) + Fraction(1, 2**70)
        expected = float(np.nextafter(np.float32(0.5), np.float32(1.0)))

        cfg = _simple_config(step_size=above_half_midpoint)

        assert cfg.step_size == expected

    @pytest.mark.parametrize(
        "step_size",
        [
            True,
            np.bool_(True),
            Decimal("0.1"),
            jnp.asarray(0.1),
            jnp.asarray(1),
        ],
    )
    def test_step_size_rejects_non_real_or_array_scalars(self, step_size: object) -> None:
        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=step_size)

    @pytest.mark.parametrize(
        "step_size",
        [
            1.0e100,
            Fraction(1, 2**150),
            Fraction(1, 2**149),
            float(np.nextafter(np.finfo(np.float32).tiny, np.float32(0.0))),
        ],
    )
    def test_step_size_rejects_nonfinite_or_subnormal_float32_sink(self, step_size: object) -> None:
        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=step_size)

    def test_step_size_accepts_float32_normal_boundaries(self) -> None:
        minimum = float(np.finfo(np.float32).tiny)
        maximum = float(np.finfo(np.float32).max)

        assert _simple_config(step_size=minimum).step_size == minimum
        assert _simple_config(step_size=maximum).step_size == maximum

    def test_step_size_comparison_spoof_cannot_hide_negative_value(self) -> None:
        class NegativeSpoof(float):
            def __new__(cls) -> "NegativeSpoof":
                return super().__new__(cls, -0.25)

            def __le__(self, other: object) -> bool:
                return False

        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=NegativeSpoof())

    @pytest.mark.parametrize("serialized", [False, True], ids=("direct", "from-config"))
    def test_step_size_rejects_numeric_subclasses_before_conversion_hooks(
        self, serialized: bool
    ) -> None:
        calls: list[str] = []

        class RatioSpoof(float):
            def __new__(cls) -> "RatioSpoof":
                return super().__new__(cls, -0.25)

            def as_integer_ratio(self) -> tuple[int, int]:
                calls.append("ratio")
                return (1, 4)

        class IntegerSpoof(int):
            def __new__(cls) -> "IntegerSpoof":
                return super().__new__(cls, -1)

            def __int__(self) -> int:
                calls.append("int")
                return 1

        class FractionSpoof(Fraction):
            def as_integer_ratio(self) -> tuple[int, int]:
                calls.append("fraction-ratio")
                return (1, 4)

        class ExplodingRatio(float):
            def as_integer_ratio(self) -> tuple[int, int]:
                calls.append("exploding")
                raise RuntimeError("conversion hook must not run")

        for step_size in (
            RatioSpoof(),
            IntegerSpoof(),
            FractionSpoof(-1, 4),
            ExplodingRatio(0.25),
        ):
            with pytest.raises(ValueError, match="step_size"):
                if serialized:
                    payload = _simple_config().to_config()
                    payload["step_size"] = step_size
                    StackedLinearHorde.from_config(
                        {"type": "StackedLinearHorde", "config": payload}
                    )
                else:
                    _simple_config(step_size=step_size)

        assert calls == []

    @pytest.mark.parametrize("serialized", [False, True], ids=("direct", "from-config"))
    @pytest.mark.parametrize(
        "attack",
        ["equality-spoof", "raising-equality", "raising-hash"],
    )
    def test_step_size_rejects_hostile_metaclasses_without_hooks(
        self, serialized: bool, attack: str
    ) -> None:
        calls: list[str] = []

        class EqualitySpoofMeta(type):
            def __eq__(cls, other: object) -> bool:
                del cls, other
                calls.append("equality")
                return True

            def __hash__(cls) -> int:
                calls.append("hash")
                return type.__hash__(cls)

        class RaisingEqualityMeta(type):
            def __eq__(cls, other: object) -> bool:
                del cls, other
                calls.append("raising-equality")
                raise RuntimeError("metaclass equality hook must not run")

            def __hash__(cls) -> int:
                calls.append("hash")
                return type.__hash__(cls)

        class RaisingHashMeta(type):
            def __hash__(cls) -> int:
                del cls
                calls.append("raising-hash")
                raise RuntimeError("metaclass hash hook must not run")

        class EqualitySpoof(float, metaclass=EqualitySpoofMeta):
            def as_integer_ratio(self) -> tuple[int, int]:
                calls.append("ratio")
                return (1, 2)

        class RaisingEquality(float, metaclass=RaisingEqualityMeta):
            pass

        class RaisingHash(float, metaclass=RaisingHashMeta):
            pass

        step_sizes = {
            "equality-spoof": EqualitySpoof(-1.0),
            "raising-equality": RaisingEquality(0.5),
            "raising-hash": RaisingHash(0.5),
        }

        with pytest.raises(ValueError, match="step_size"):
            if serialized:
                payload = _simple_config().to_config()
                payload["step_size"] = step_sizes[attack]
                StackedLinearHorde.from_config({"type": "StackedLinearHorde", "config": payload})
            else:
                _simple_config(step_size=step_sizes[attack])

        assert calls == []

    @pytest.mark.parametrize("serialized", [False, True], ids=("direct", "from-config"))
    @pytest.mark.parametrize("missing_slot", ["numerator", "denominator"])
    def test_step_size_rejects_exact_fraction_with_missing_slot(
        self, serialized: bool, missing_slot: str
    ) -> None:
        step_size = object.__new__(Fraction)
        if missing_slot == "numerator":
            object.__setattr__(step_size, "_denominator", 2)
        else:
            object.__setattr__(step_size, "_numerator", 1)

        with pytest.raises(ValueError, match="step_size"):
            if serialized:
                payload = _simple_config().to_config()
                payload["step_size"] = step_size
                StackedLinearHorde.from_config({"type": "StackedLinearHorde", "config": payload})
            else:
                _simple_config(step_size=step_size)

    @pytest.mark.parametrize("component", ["numerator", "denominator"])
    @pytest.mark.parametrize("serialized", [False, True], ids=("direct", "from-config"))
    def test_step_size_rejects_hooks_laundered_through_an_exact_fraction(
        self, serialized: bool, component: str
    ) -> None:
        calls: list[str] = []

        class IntegerSpoof(int):
            def __int__(self) -> int:
                calls.append("int")
                return 1

        class Carrier(Fraction):
            @property
            def numerator(self) -> int:
                return IntegerSpoof(-1) if component == "numerator" else 1

            @property
            def denominator(self) -> int:
                return IntegerSpoof(-4) if component == "denominator" else 4

        step_size = Fraction(Carrier(1, 4))
        assert type(step_size) is Fraction
        assert step_size.numerator < 0 or step_size.denominator < 0

        with pytest.raises(ValueError, match="step_size"):
            if serialized:
                payload = _simple_config().to_config()
                payload["step_size"] = step_size
                StackedLinearHorde.from_config({"type": "StackedLinearHorde", "config": payload})
            else:
                _simple_config(step_size=step_size)

        assert calls == []

    @pytest.mark.parametrize("serialized", [False, True], ids=("direct", "from-config"))
    def test_step_size_rejects_exact_fraction_with_zero_denominator(self, serialized: bool) -> None:
        property_calls: list[str] = []

        class Carrier(Fraction):
            @property
            def numerator(self) -> int:
                property_calls.append("numerator")
                return 1

            @property
            def denominator(self) -> int:
                property_calls.append("denominator")
                return 0

        step_size = Fraction(Carrier(1, 1))
        assert type(step_size) is Fraction
        assert step_size.denominator == 0
        calls_after_construction = tuple(property_calls)

        with pytest.raises(ValueError, match="step_size"):
            if serialized:
                payload = _simple_config().to_config()
                payload["step_size"] = step_size
                StackedLinearHorde.from_config({"type": "StackedLinearHorde", "config": payload})
            else:
                _simple_config(step_size=step_size)

        assert tuple(property_calls) == calls_after_construction

    @pytest.mark.parametrize(
        "step_size",
        [1.0e100, Fraction(1, 2**149), Decimal("0.1"), jnp.asarray(0.1)],
    )
    def test_invalid_serialized_step_size_is_refused_before_runtime(
        self, step_size: object
    ) -> None:
        payload = _simple_config().to_config()
        payload["step_size"] = step_size

        with pytest.raises(ValueError, match="step_size"):
            StackedLinearHorde.from_config({"type": "StackedLinearHorde", "config": payload})

    def test_minimum_normal_step_size_survives_jit_execution(self) -> None:
        minimum = float(np.finfo(np.float32).tiny)
        horde = StackedLinearHorde(_simple_config(n_demons=1, step_size=minimum))
        state = horde.init()
        update = jax.jit(horde.update)

        result = update(
            state,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.zeros((3,)),
            jnp.array([1.0]),
        )

        assert bool(result.update_applied)
        assert int(result.state.step_count) == 1
        assert float(result.state.weights[0, 0]) == minimum

    def test_roundtrip(self):
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        restored = StackedLinearHorde.from_config(horde.to_config())
        assert restored.config == cfg

    def test_nexting_spec_shape(self):
        cfg = nexting_spec(feature_dim=6, cumulant_indices=(0, 2), gammas=(0.0, 0.9))
        assert cfg.n_demons == 4
        assert cfg.cumulant_indices == (0, 0, 2, 2)
        assert cfg.gammas == (0.0, 0.9, 0.0, 0.9)


@pytest.mark.unit
class TestCumulantSourceDomain:
    """Out-of-range cumulant reads must fail closed, never clip (issue #579)."""

    @staticmethod
    def _two_channel_horde() -> StackedLinearHorde:
        return StackedLinearHorde(
            StackedHordeConfig(
                n_demons=2,
                feature_dim=3,
                gammas=(0.0, 0.0),
                lamdas=(0.0, 0.0),
                cumulant_indices=(0, 2),
                step_size=0.5,
            )
        )

    @pytest.mark.parametrize(
        "bad_index",
        [-1, True, False, 1.5, 1.0, "1"],
        ids=("negative", "true", "false", "float-fraction", "float-int", "str"),
    )
    def test_config_rejects_non_builtin_or_negative_indices(self, bad_index: object) -> None:
        with pytest.raises(ValueError, match="cumulant_indices"):
            _simple_config(n_demons=1, gammas=(0.9,), lamdas=(0.5,), cumulant_indices=(bad_index,))

    def test_from_config_rejects_invalid_serialized_indices(self) -> None:
        payload = _simple_config().to_config()
        payload["cumulant_indices"] = [0, -1]
        with pytest.raises(ValueError, match="cumulant_indices"):
            StackedHordeConfig.from_config(payload)

    def test_update_rejects_source_missing_the_maximum_index(self) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="cumulant_source"):
            horde.update(state, x, jnp.zeros(3), jnp.array([1.0, 99.0]))
        # Nothing may have moved: the horde state is untouched by the rejection.
        np.testing.assert_array_equal(np.asarray(state.weights), np.zeros((2, 3)))
        assert int(state.step_count) == 0

    def test_jit_update_rejects_short_source_at_trace_time(self) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])
        jitted = jax.jit(horde.update)
        with pytest.raises(ValueError, match="cumulant_source"):
            jitted(state, x, jnp.zeros(3), jnp.array([1.0, 99.0]))

    @pytest.mark.parametrize("rank", [0, 2], ids=("rank-zero", "rank-two"))
    def test_update_rejects_non_rank_one_sources(self, rank: int) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])
        source = jnp.ones((3,) * rank, dtype=jnp.float32)
        with pytest.raises(ValueError, match="cumulant_source"):
            horde.update(state, x, jnp.zeros(3), source)

    def test_scan_rejects_non_rank_two_sources(self) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        features = jnp.zeros((3, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match="cumulant_sources"):
            run_stacked_horde_scan(horde, state, features, jnp.ones((3,), dtype=jnp.float32))

    def test_scan_rejects_sources_missing_the_maximum_index(self) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        features = jnp.zeros((3, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match="cumulant_sources"):
            run_stacked_horde_scan(horde, state, features, jnp.ones((3, 2), dtype=jnp.float32))

    def test_valid_direct_jit_and_scan_updates_are_unchanged(self) -> None:
        horde = self._two_channel_horde()
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])
        source = jnp.array([1.0, 99.0, 4.0])

        direct = horde.update(state, x, jnp.zeros(3), source)
        np.testing.assert_allclose(np.asarray(direct.td_errors), [1.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(direct.state.weights[:, 0]), [0.5, 2.0], rtol=1e-6)

        jitted = jax.jit(horde.update)(state, x, jnp.zeros(3), source)
        np.testing.assert_allclose(np.asarray(jitted.td_errors), [1.0, 4.0], rtol=1e-6)

        features = jnp.stack([x, jnp.zeros(3)])
        sources = jnp.stack([source, source])
        final_state, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        np.testing.assert_allclose(np.asarray(td_errors), [[1.0, 4.0]], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(final_state.weights[:, 0]), [0.5, 2.0], rtol=1e-6)


class TestExactSemantics:
    def test_hand_computed_two_step(self):
        """Exact TD(lambda) values for one demon over two transitions."""
        cfg = StackedHordeConfig(
            n_demons=1,
            feature_dim=2,
            gammas=(0.9,),
            lamdas=(0.5,),
            cumulant_indices=(0,),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        x0 = jnp.array([1.0, 0.0])
        x1 = jnp.array([0.0, 1.0])
        c = jnp.array([2.0])

        # Step 1: v=0, v'=0, delta = 2.0; z = 0.45*0 + x0 = [1,0];
        # w = 0.1*2.0*[1,0] = [0.2, 0].
        r1 = horde.update(state, x0, x1, c)
        np.testing.assert_allclose(np.asarray(r1.td_errors), [2.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(r1.state.weights), [[0.2, 0.0]], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(r1.state.traces), [[1.0, 0.0]], rtol=1e-6)

        # Step 2 (x1 -> x0): v = w@x1 = 0, v' = w@x0 = 0.2,
        # delta = 2.0 + 0.9*0.2 - 0 = 2.18; z = 0.45*[1,0] + [0,1] = [0.45,1];
        # w += 0.1*2.18*[0.45,1] = [0.2981, 0.218].
        r2 = horde.update(r1.state, x1, x0, c)
        np.testing.assert_allclose(np.asarray(r2.td_errors), [2.18], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(r2.state.weights), [[0.2 + 0.0981, 0.218]], rtol=1e-5)

    def test_nan_cumulant_freezes_weights_decays_trace(self):
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        x = jnp.array([1.0, 2.0, 3.0])

        # Warm up demon traces/weights.
        r = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]))
        # Demon 0 goes inactive; demon 1 stays active.
        r2 = horde.update(r.state, x, x, jnp.array([jnp.nan, 1.0, 0.0]))
        assert bool(jnp.isnan(r2.td_errors[0]))
        assert bool(jnp.isfinite(r2.td_errors[1]))
        # Demon 0 weights frozen.
        np.testing.assert_array_equal(
            np.asarray(r2.state.weights[0]), np.asarray(r.state.weights[0])
        )
        # Demon 0 trace decayed (no gradient added): z2 = gamma*lamda*z1.
        np.testing.assert_allclose(
            np.asarray(r2.state.traces[0]),
            0.9 * 0.5 * np.asarray(r.state.traces[0]),
            rtol=1e-6,
        )
        # Demon 1 weights moved.
        assert not np.array_equal(np.asarray(r2.state.weights[1]), np.asarray(r.state.weights[1]))

    def test_rho_composes_into_trace(self):
        """z = rho * (decay * z + x): rho=0 zeroes the trace and the update."""
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])

        r = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=0.0)
        np.testing.assert_array_equal(np.asarray(r.state.weights), 0.0)
        np.testing.assert_array_equal(np.asarray(r.state.traces), 0.0)

        # rho=2 doubles the trace relative to rho=1.
        r1 = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=1.0)
        r2 = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=2.0)
        np.testing.assert_allclose(
            np.asarray(r2.state.traces), 2.0 * np.asarray(r1.state.traces), rtol=1e-6
        )

    def test_infinite_rho_on_zero_feature_does_not_poison_traces(self):
        """rho * x is 0*inf = NaN on a silent feature; hold the finite state."""
        cfg = StackedHordeConfig(
            n_demons=1,
            feature_dim=2,
            gammas=(0.9,),
            lamdas=(0.0,),
            cumulant_indices=(0,),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        x = jnp.array([0.0, 1.0], dtype=jnp.float32)
        c = jnp.array([1.0], dtype=jnp.float32)

        poisoned = horde.update(state, x, x, c, rho=jnp.inf)
        np.testing.assert_array_equal(np.asarray(poisoned.state.weights), 0.0)
        np.testing.assert_array_equal(np.asarray(poisoned.state.traces), 0.0)
        assert int(poisoned.state.step_count) == 0
        assert not bool(poisoned.update_applied)
        chex.assert_trees_all_equal(
            poisoned.head_updates_applied,
            jnp.array([False]),
        )
        assert float(poisoned.td_errors[0]) == 0.0

        recovered = horde.update(poisoned.state, x, x, c, rho=1.0)
        assert bool(jnp.all(jnp.isfinite(recovered.state.weights)))
        assert bool(jnp.all(jnp.isfinite(recovered.state.traces)))
        assert int(recovered.state.step_count) == 1
        assert bool(recovered.update_applied)

    def test_zero_gamma_does_not_multiply_inf_bootstrap(self):
        """gamma=0 * inf V(s') is 0*inf = NaN and would freeze a nexting demon."""
        cfg = StackedHordeConfig(
            n_demons=1,
            feature_dim=2,
            gammas=(0.0,),
            lamdas=(0.0,),
            cumulant_indices=(0,),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        huge = jnp.float32(1e38)
        state = horde.init().replace(
            weights=jnp.asarray([[huge, 0.0]], dtype=jnp.float32),
            traces=jnp.asarray([[jnp.inf, jnp.inf]], dtype=jnp.float32),
        )
        x = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        x_next = jnp.asarray([huge, 0.0], dtype=jnp.float32)
        c = jnp.asarray([2.0], dtype=jnp.float32)
        raw = jnp.asarray(0.0, dtype=jnp.float32) * (huge * huge)
        assert not bool(jnp.isfinite(raw))

        result = horde.update(state, x, x_next, c, rho=1.0)
        chex.assert_tree_all_finite(result.state.traces)
        chex.assert_tree_all_finite(result.td_errors)
        assert bool(result.update_applied)
        assert float(result.td_errors[0]) == pytest.approx(2.0)


class TestConvergence:
    def test_three_state_cycle_analytic_fixed_point(self):
        """On a deterministic 3-state cycle with one-hot features, every
        demon's values converge to the analytic discounted fixed point
        v(s) = sum_k gamma^k c(s_{t+1+k}) — checked for gamma 0 and 0.8 at
        two different cumulant channels simultaneously."""
        gammas = (0.0, 0.8, 0.0, 0.8)
        cumulant_indices = (0, 0, 1, 1)
        cfg = StackedHordeConfig(
            n_demons=4,
            feature_dim=3,
            gammas=gammas,
            lamdas=(0.9,) * 4,
            cumulant_indices=cumulant_indices,
            step_size=0.05,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        # Cycle s0 -> s1 -> s2 -> s0; cumulant channel 0 = [1, 0, 0] by next
        # state, channel 1 = [0, 2, 0].
        eye = jnp.eye(3, dtype=jnp.float32)
        c_by_state = jnp.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        num_steps = 4000
        order = jnp.arange(num_steps) % 3
        features = eye[order]
        # Cumulant for the t -> t+1 transition is the channel value at s_{t+1}.
        next_order = (order + 1) % 3
        sources = c_by_state[next_order]

        state, _ = run_stacked_horde_scan(horde, state, features, sources)

        # Analytic: for gamma, v(s_i) = sum_{k>=0} gamma^k c(s_{i+1+k}).
        def analytic(gamma, channel):
            c = np.asarray(c_by_state[:, channel])
            v = np.zeros(3)
            for i in range(3):
                # Geometric sum over the period-3 cycle.
                per = np.array([c[(i + 1 + k) % 3] * gamma**k for k in range(3)])
                v[i] = per.sum() / (1.0 - gamma**3)
            return v

        w = np.asarray(state.weights)
        for d, (g, ch) in enumerate(zip(gammas, cumulant_indices)):
            np.testing.assert_allclose(w[d], analytic(g, ch), atol=0.02)

    def test_nexting_multi_timescale_orderings(self):
        """Nexting demons at gammas (0, 0.5, 0.9) over a recurring pulse:
        longer-timescale predictions are larger ahead of the pulse (they
        accumulate more future signal) — the qualitative nexting signature."""
        cfg = nexting_spec(
            feature_dim=8,
            cumulant_indices=(0,),
            gammas=(0.0, 0.5, 0.9),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        # Period-8 one-hot cycle; pulse fires at phase 0 (cumulant = 1 when
        # the next state is phase 0).
        num_steps = 6000
        order = jnp.arange(num_steps) % 8
        features = jnp.eye(8, dtype=jnp.float32)[order]
        pulse = (jnp.roll(order, -1) % 8 == 0).astype(jnp.float32)
        sources = pulse[:, None]

        state, _ = run_stacked_horde_scan(horde, state, features, sources)
        # At phase 7 (one step before the pulse) all timescales see it;
        # at phase 4 only the long-timescale demon still sees much of it.
        v7 = np.asarray(horde.predict(state, jnp.eye(8)[7]))
        v4 = np.asarray(horde.predict(state, jnp.eye(8)[4]))
        assert v7[0] > 0.9  # gamma=0: next-step pulse predicted ~1
        assert v4[0] < 0.1  # gamma=0: nothing next step
        assert v4[2] > v4[1] > v4[0]  # longer horizons see the coming pulse


class TestDemonAxisScaling:
    def test_1024_demons_run_fast_with_constant_program_size(self):
        """1024 demons x 2000 steps completes in seconds, not minutes.

        The loop-unrolled hordes measured ~14 steps/s at 1024 demons with a
        ~144 s compile (16.4 GB working set).  The stacked horde runs the
        same demon count as one batched update; this test bounds the whole
        thing — compile included — at 30 s and the post-compile scan at 5 s,
        both enormous (>25x) margins over measured values (~1.5 s / ~0.02 s
        on the 24-core dev box).
        """
        n_demons, feature_dim, num_steps = 1024, 32, 2000
        key = jr.key(0)
        rng = jr.split(key, 2)
        cfg = StackedHordeConfig(
            n_demons=n_demons,
            feature_dim=feature_dim,
            gammas=tuple(float(g) for g in np.linspace(0.0, 0.99, n_demons)),
            lamdas=(0.7,) * n_demons,
            cumulant_indices=tuple(int(i) for i in np.arange(n_demons) % feature_dim),
            step_size=0.01,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        features = jr.normal(rng[0], (num_steps, feature_dim), dtype=jnp.float32)
        sources = jr.normal(rng[1], (num_steps, feature_dim), dtype=jnp.float32)

        t0 = time.time()
        final_state, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        td_errors.block_until_ready()
        first_call = time.time() - t0
        assert first_call < 30.0, f"compile+run took {first_call:.1f}s"

        t1 = time.time()
        final_state, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        td_errors.block_until_ready()
        steady = time.time() - t1
        assert steady < 5.0, f"steady-state run took {steady:.1f}s"

        assert bool(jnp.all(jnp.isfinite(final_state.weights)))
        assert td_errors.shape == (num_steps - 1, n_demons)

    def test_learning_quality_survives_at_scale(self):
        """All 1024 demons actually learn: on a 3-state cycle every demon's
        prediction error shrinks between the first and last 200 steps."""
        n_demons = 1024
        cfg = StackedHordeConfig(
            n_demons=n_demons,
            feature_dim=3,
            gammas=tuple(float(g) for g in np.linspace(0.0, 0.95, n_demons)),
            lamdas=(0.8,) * n_demons,
            cumulant_indices=(0,) * n_demons,
            step_size=0.05,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        num_steps = 3000
        order = jnp.arange(num_steps) % 3
        features = jnp.eye(3, dtype=jnp.float32)[order]
        pulse = ((order + 1) % 3 == 0).astype(jnp.float32)
        sources = pulse[:, None]

        _, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        early = jnp.mean(td_errors[:200] ** 2, axis=0)  # (n_demons,)
        late = jnp.mean(td_errors[-200:] ** 2, axis=0)
        # Every single demon improved.
        assert bool(jnp.all(late < early))
        # And the late TD error is near zero for all timescales.
        assert float(jnp.max(late)) < 0.01


def test_stacked_horde_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_demons"):
        StackedHordeConfig(
            n_demons=True,  # type: ignore[arg-type]
            feature_dim=4,
            gammas=(0.9,),
            lamdas=(0.8,),
            cumulant_indices=(0,),
        )
    with pytest.raises(ValueError, match="feature_dim"):
        StackedHordeConfig(
            n_demons=1,
            feature_dim=4.5,  # type: ignore[arg-type]
            gammas=(0.9,),
            lamdas=(0.8,),
            cumulant_indices=(0,),
        )
    with pytest.raises(ValueError, match="cumulant_indices"):
        StackedHordeConfig(
            n_demons=1,
            feature_dim=4,
            gammas=(0.9,),
            lamdas=(0.8,),
            cumulant_indices=(True,),  # type: ignore[arg-type]
        )


def test_stacked_horde_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = StackedHordeConfig(
        n_demons=np.int32(2),
        feature_dim=np.int64(4),
        gammas=(0.9, 0.5),
        lamdas=(0.8, 0.7),
        cumulant_indices=(np.int32(0), np.int64(1)),
    )
    assert type(cfg.n_demons) is int
    assert type(cfg.feature_dim) is int
    assert type(cfg.cumulant_indices[0]) is int
    assert type(cfg.cumulant_indices[1]) is int
    assert cfg.n_demons == 2
    assert cfg.feature_dim == 4
    assert cfg.cumulant_indices == (0, 1)
